
'''
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
'''

import argparse
from pathlib import Path
import yaml
from munch import Munch
from tqdm import tqdm
import json
import sys
import subprocess
import zlib
import numpy as np
import imageio as iio
import lz4.block

from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list


def extract_rgb(scene):
    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {scene.iphone_video_path} -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.jpg"
    run_command(cmd, verbose=True)

def extract_masks(scene):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -start_number 0 {scene.iphone_video_mask_dir}/frame_%06d.png"
    run_command(cmd, verbose=True)

def extract_depth(scene):
    # global compression with zlib
    height, width = 192, 256
    sample_rate = 1
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(scene.iphone_depth_path, 'rb') as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(range(0, depth.shape[0], sample_rate), desc='decode_depth'):
            iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", (depth * 1000).astype(np.uint16))
    # per frame compression with lz4/zlib
    except:
        frame_id = 0
        with open(scene.iphone_depth_path, 'rb') as infile:
            while True:
                size = infile.read(4)   # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder='little')
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(data, uncompressed_size=height * width * 2)  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 6 digit frame id = 277 minute video at 60 fps
                iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", depth)
                frame_id += 1

def main(args):
    cfg = load_yaml_munch(args.config_file)

    # --- 新增代码：检查 output_root 是否在配置文件中 ---
    if not cfg.get("output_root"):
        print("Error: 'output_root' not found in the config file.")
        print("Please specify the output directory in your .yml file.")
        sys.exit(1)
    # --- 新增代码结束 ---

    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        # 从配置文件读取限制数量，如果没有则为None
        limit = cfg.get("max_scenes_per_split", None)
        for split in cfg.splits:
            # data_root 现在是你存放 splits 文件夹的目录
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scenes_from_file = read_txt_list(split_path)
            # 如果设置了 limit，则只截取前 limit 个场景
            if limit is not None and limit > 0:
                scene_ids += scenes_from_file[:limit]
            else:
                # 如果没有设置 limit，则添加所有场景
                scene_ids += scenes_from_file

    # get the options to process
    # go through each scene
    for scene_id in tqdm(scene_ids, desc='scene'):
        # 步骤 1: 使用原始的 data_root 初始化 scene 对象
        # 这可以确保所有的输入路径 (如 .mp4, .depth) 都是正确的
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / 'data')

        # --- 修改核心部分：重写输出路径 ---
        # 步骤 2: 根据 cfg.output_root 构建新的输出基路径
        # 我们将保持与原数据相似的目录结构，即 output_root/data/scene_id/
        scene_output_base = Path(cfg.output_root) / 'data' / scene.scene_id
        
        # 步骤 3: 手动覆盖 scene 对象的输出目录属性
        # ScannetppScene_Release 内部可能会将路径定义为 '.../iphone/rgb' 等
        # 我们在这里重新构建这些路径，但让它们指向新的输出基路径
        scene.iphone_rgb_dir = scene_output_base / 'iphone' / 'rgb'
        scene.iphone_video_mask_dir = scene_output_base / 'iphone' / 'video_masks'
        scene.iphone_depth_dir = scene_output_base / 'iphone' / 'depth'
        # --- 修改结束 ---

        if cfg.extract_rgb:
            extract_rgb(scene)

        if cfg.extract_masks:
            extract_masks(scene)

        if cfg.extract_depth:
            extract_depth(scene)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()

    main(args)
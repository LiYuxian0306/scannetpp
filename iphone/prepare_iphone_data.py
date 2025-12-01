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

# --- 修改 1：修改函数签名，让它们接收一个显式的 output_dir 参数 ---
def extract_rgb(scene, output_dir):
    """
    Extracts RGB frames from the iPhone video.
    :param scene: The ScannetppScene_Release object (for input paths).
    :param output_dir: The directory where RGB frames will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # 使用 scene 对象获取输入路径，使用 output_dir 作为输出路径
    cmd = f"ffmpeg -i {scene.iphone_video_path} -start_number 0 -q:v 1 {output_dir}/frame_%06d.jpg"
    run_command(cmd, verbose=True)

def extract_masks(scene, output_dir):
    """
    Extracts masks from the iPhone video mask.
    :param scene: The ScannetppScene_Release object (for input paths).
    :param output_dir: The directory where mask frames will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -start_number 0 {output_dir}/frame_%06d.png"
    run_command(cmd, verbose=True)

def extract_depth(scene, output_dir):
    """
    Extracts and decompresses depth frames.
    :param scene: The ScannetppScene_Release object (for input paths).
    :param output_dir: The directory where depth frames will be saved.
    """
    height, width = 192, 256
    sample_rate = 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(scene.iphone_depth_path, 'rb') as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(range(0, depth.shape[0], sample_rate), desc='decode_depth'):
            # 使用 output_dir
            iio.imwrite(f"{output_dir}/frame_{frame_id:06}.png", (depth * 1000).astype(np.uint16))
    except Exception:
        frame_id = 0
        with open(scene.iphone_depth_path, 'rb') as infile:
            while True:
                size = infile.read(4)
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder='little')
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                data = infile.read(size)
                try:
                    data = lz4.block.decompress(data, uncompressed_size=height * width * 2)
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 使用 output_dir
                iio.imwrite(f"{output_dir}/frame_{frame_id:06}.png", depth)
                frame_id += 1

def main(args):
    cfg = load_yaml_munch(args.config_file)

    if not cfg.get("output_root"):
        print("Error: 'output_root' not found in the config file.")
        print("Please specify the output directory in your .yml file.")
        sys.exit(1)

    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        limit = cfg.get("max_scenes_per_split", None)
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scenes_from_file = read_txt_list(split_path)
            if limit is not None and limit > 0:
                scene_ids += scenes_from_file[:limit]
            else:
                scene_ids += scenes_from_file

    for scene_id in tqdm(scene_ids, desc='scene'):
        # 步骤 1: 初始化 scene 对象，它现在只负责提供 *输入* 路径
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / 'data')

        # --- 修改 2：不再尝试修改 scene 对象，而是创建独立的输出路径变量 ---
        scene_output_base = Path(cfg.output_root) / 'data' / scene.scene_id
        
        # 为每个任务定义清晰的输出目录
        output_rgb_dir = scene_output_base / 'iphone' / 'rgb'
        output_mask_dir = scene_output_base / 'iphone' / 'video_masks'
        output_depth_dir = scene_output_base / 'iphone' / 'depth'
        
        # --- 修改 3：调用函数时，将输出路径作为参数传入 ---
        if cfg.extract_rgb:
            print(f"\nExtracting RGB frames for scene {scene_id} to {output_rgb_dir}")
            extract_rgb(scene, output_rgb_dir)

        if cfg.extract_masks:
            print(f"\nExtracting masks for scene {scene_id} to {output_mask_dir}")
            extract_masks(scene, output_mask_dir)

        if cfg.extract_depth:
            print(f"\nExtracting depth frames for scene {scene_id} to {output_depth_dir}")
            extract_depth(scene, output_depth_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()

    main(args)
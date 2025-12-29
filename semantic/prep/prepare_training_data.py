'''
Prepare training data for 3D semantic and instance segmentation tasks
by mapping raw labels to the benchmarks classes
and sampling points on the mesh to get the labels on these
'''
#config_file=semantic/configs/prepare_training_data.yml


import argparse
from pathlib import Path
from common.file_io import load_yaml_munch, read_txt_list


import torch
from tqdm import tqdm

from semantic.transforms.factory import get_transform
from semantic.datasets.scannetpp_release import ScannetPP_Release_Dataset


def main(args):
    cfg = load_yaml_munch(args.config_file) #得到config的路径

    transform = get_transform(cfg.data)
    
    class_names = read_txt_list(cfg.data.labels_path) #top100的class
    n_classes = len(class_names)
    print('Num classes in class list:', n_classes)

    ds = ScannetPP_Release_Dataset(data_root=cfg.data.data_root,
                                     list_file=cfg.data.list_path, #场景名
                                     transform=transform)
    print('Num samples in dataset:', len(ds))

    out_dir = Path(cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Saving pth data to:', out_dir)

    # save to pth files 
    for sample in tqdm(ds):
        # keep the scene ID and keys which start with vtx_ or sampled_
        keep_keys = ['scene_id'] + [k for k in sample.keys() if k.startswith('vtx_') or k.startswith('sampled_')]
        
        save_sample = {k: v for k, v in sample.items() if k in keep_keys}
        #sample.items() 会返回一个迭代器，每次循环吐出一个 (key, value) 对
        
        # NOTE: colors are in 0-1 range -> open3d format
        torch.save(save_sample, out_dir / f'{sample["scene_id"]}.pth')


if __name__ == '__main__': #只有当该文件被直接运行时，缩进块内的代码才会被执行；如果该文件是被别的模块导入（import）的，这块代码就不会执行。
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()
    main(args)
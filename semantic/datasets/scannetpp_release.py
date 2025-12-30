
from pathlib import Path
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import open3d as o3d

from common.file_io import load_json
from common.scene_release import ScannetppScene_Release

class ScannetPP_Release_Dataset(Dataset):
    def __init__(self, data_root, list_file=None, transform=None,
                no_mesh=False):
        '''
        data_root: dir containing data
        list_file: list of scenes to read
        transform: apply on each scene
        no_mesh: dont read mesh, anno only
        '''

        self.transform = transform

        data_root = Path(data_root)

        self.samples = []
        self.no_mesh = no_mesh

        if list_file is not None:
            with open(list_file) as f:
                scene_list = f.read().splitlines() #要是.txt文件有指明场景名就按照那个文件指明的来
        else:
            scene_list = os.listdir(data_root) #没有指明的话就遍历data_root下面所有的场景

        for scene_id in tqdm(scene_list, 'dataset'):
            scene = ScannetppScene_Release(scene_id, data_root=data_root)
            #遍历 scene_list，并在屏幕上打印一个标签为 dataset 的进度条。

            anno_path = scene.scan_anno_json_path #segments_anno.json
            mesh_path = scene.scan_mesh_path  #原始RGB场景 'mesh_aligned_0.05.ply'
            segs_path = scene.scan_mesh_segs_path  #其实就是点的标号 segments.json
            if anno_path.is_file() and mesh_path.is_file() and segs_path.is_file():
                self.samples.append({
                    'scene_id': scene_id,
                    'anno': anno_path,
                    'mesh': mesh_path,
                    'segs': segs_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        if self.no_mesh:
            mesh = None
        else:
            # load mesh vertices and colors
            mesh = o3d.io.read_triangle_mesh(str(sample['mesh']))
        # load segments = vertices per segment ID
        segments = load_json(sample['segs'])
        # load anno = (instance, groups of segments)
        anno = load_json(sample['anno'])

        sample = {
            'scene_id': sample['scene_id'],
            'o3d_mesh': mesh,
            'segments': segments,
            'anno': anno
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
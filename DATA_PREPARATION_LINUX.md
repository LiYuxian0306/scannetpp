# ScanNet++ 数据准备完整流程（Linux系统）

本文档提供了在Linux系统上准备ScanNet++数据的完整流程。

## 目录
1. [环境准备](#1-环境准备)
2. [数据下载](#2-数据下载)
3. [DSLR数据准备](#3-dslr数据准备)
4. [iPhone数据准备](#4-iphone数据准备)
5. [语义数据准备](#5-语义数据准备)
6. [Panocam数据准备](#6-panocam数据准备)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

### 1.1 创建Conda环境

```bash
# 创建新的conda环境
conda create -n scannetpp python=3.10
conda activate scannetpp

# 安装基础依赖
pip install -r requirements.txt
```

### 1.2 安装PyTorch（根据CUDA版本）

```bash
# 检查CUDA版本
nvidia-smi

# 根据CUDA版本安装PyTorch
# 对于CUDA 11.6（requirements.txt中指定）
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --find-links https://download.pytorch.org/whl/torch_stable.html

# 或者使用conda安装
conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.6 -c pytorch
```

### 1.3 安装额外依赖（用于深度渲染）

```bash
# 安装renderpy（用于渲染深度图）
# 从 https://github.com/liu115/renderpy 安装
pip install git+https://github.com/liu115/renderpy
```

### 1.4 验证安装

```bash
python -c "import torch; print(torch.__version__)"
python -c "import open3d; print(open3d.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

---

## 2. 数据下载

### 2.1 下载ScanNet++数据集

从[官方数据集页面](https://kaldir.vc.in.tum.de/scannetpp/)下载数据。

**数据目录结构**：
```
DATA_ROOT/
├── <scene_id>/
│   ├── dslr/
│   │   ├── images/
│   │   ├── resized_images/
│   │   ├── resized_anon_masks/
│   │   └── nerfstudio/
│   │       └── transforms.json
│   ├── iphone/
│   │   └── ...
│   ├── panocam/
│   │   └── ...
│   ├── mesh/
│   │   └── mesh.ply
│   ├── segments_anno.json
│   └── ...
```

### 2.2 设置数据根目录

```bash
# 设置环境变量（可选）
export SCANNETPP_DATA_ROOT=/path/to/scannetpp/data

# 或者在配置文件中直接指定绝对路径
```

---

## 3. DSLR数据准备

### 3.1 去畸变（Undistortion）

将鱼眼图像转换为针孔相机模型，生成去畸变图像、mask和transforms.json文件。

**步骤**：

1. **编辑配置文件** `dslr/configs/undistort.yml`：

```yaml
# folder where the data is downloaded
data_root: /path/to/scannetpp/data

splits: [nvs_sem_train, nvs_sem_val]
# 或者指定特定场景
# scene_ids: [355e5e32db]

# 降采样因子（可选）
downscale_factor: 2.0

# 输出目录（相对于DATA_ROOT/dslr/）
out_image_dir: undistorted_images
out_mask_dir: undistorted_anon_masks
out_transforms_path: nerfstudio/transforms_undistorted.json
```

2. **运行去畸变脚本**：

```bash
python -m dslr.undistort dslr/configs/undistort.yml
```

**输出**：
- `DATA_ROOT/<scene_id>/dslr/undistorted_images/` - 去畸变后的图像
- `DATA_ROOT/<scene_id>/dslr/undistorted_anon_masks/` - 去畸变后的mask
- `DATA_ROOT/<scene_id>/dslr/nerfstudio/transforms_undistorted.json` - 新的相机参数

### 3.2 降采样DSLR图像（可选）

如果需要降低内存占用，可以降采样DSLR图像。

1. **编辑配置文件** `dslr/configs/downscale.yml`：

```yaml
data_root: /path/to/scannetpp/data
splits: [nvs_sem_train, nvs_sem_val]

downscale_factor: 2.0  # 降采样因子
out_image_dir: resized_images_2
out_mask_dir: resized_anon_masks_2
out_transforms_path: nerfstudio/transforms_2.json
```

2. **运行降采样脚本**：

```bash
python -m dslr.downscale dslr/configs/downscale.yml
```

### 3.3 渲染深度图（DSLR和iPhone）

为DSLR和iPhone图像渲染深度图。

1. **编辑配置文件** `common/configs/render.yml`：

```yaml
# folder where the data is downloaded
data_root: /path/to/scannetpp/data

# 设置True以渲染对应类型的深度图
render_iphone: True
render_dslr: False  # 或True

splits: [nvs_sem_train, nvs_sem_val]
# 或指定特定场景
# scene_ids: [355e5e32db]

# 深度相机的近远平面（米）
near: 0.05
far: 20.0

# 输出目录（可选，默认保存到data_root）
# output_dir: /path/to/output
```

2. **运行渲染脚本**：

```bash
python -m common.render common/configs/render.yml
```

**输出结构**：
```
output_dir/<scene_id>/
├── dslr/
│   ├── render_rgb/
│   └── render_depth/
└── iphone/
    ├── render_rgb/
    └── render_depth/
```

**注意**：渲染的深度图是单通道uint16 PNG格式，单位为毫米，0表示无效深度。

---

## 4. iPhone数据准备

### 4.1 提取RGB帧、Mask和深度帧

从iPhone视频中提取RGB帧、匿名化mask和深度帧。

1. **创建配置文件** `iphone/configs/prepare_iphone_data.yml`（从模板复制）：

```bash
cp iphone/configs/prepare_iphone_data.yml.template iphone/configs/prepare_iphone_data.yml
```

2. **编辑配置文件**：

```yaml
# 提取选项
extract_rgb: true
extract_masks: true
extract_depth: true

# 数据根目录
data_root: /path/to/scannetpp/data

splits: [nvs_sem_train, nvs_sem_val]
# 或指定特定场景
# scene_ids: [ebff4de90b]
```

3. **运行提取脚本**：

```bash
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
```

---

## 5. 语义数据准备

### 5.1 准备3D语义训练数据

从mesh中采样点并映射标签到benchmark类别。

**重要提示**：
- Mesh顶点分布可能不均匀，不能直接作为点云使用
- 必须在mesh表面采样点，然后用于体素化等操作

1. **创建配置文件** `semantic/configs/prepare_training_data.yml`（从模板复制）：

```bash
cp semantic/configs/prepare_training_data.yml.template semantic/configs/prepare_training_data.yml
```

2. **编辑配置文件**：

```yaml
data:
  data_root: /path/to/scannetpp/data
  
  # 语义标签文件路径
  labels_path: /path/to/semantic_labels.txt
  
  # 实例分割相关
  use_instances: true
  instance_labels_path: /path/to/instance_labels.txt
  
  # 映射文件（将1.5k+原始标签映射到benchmark类别）
  # 位于 metadata/semantic_benchmark/map_benchmark.csv
  mapping_file: /path/to/metadata/semantic_benchmark/map_benchmark.csv
  
  # 场景列表文件
  list_path: /path/to/nvs_sem_train.txt
  
  ignore_label: -100
  sample_factor: 1.0  # 采样因子
  
  transforms:
    - add_mesh_vertices
    - map_label_to_index
    - get_labels_on_vertices
    - sample_points_on_mesh

# 输出目录
out_dir: /path/to/output/pth_data
```

3. **运行准备脚本**：

```bash
python -m semantic.prep.prepare_training_data semantic/configs/prepare_training_data.yml
```

**输出PTH文件字段**：
- `scene_id` - str, 场景ID
- `sampled_coords` - (n_samples, 3), 采样点的坐标
- `sampled_colors` - (n_samples, 3), RGB颜色 [0, 1]
- `sampled_labels` - (n_samples,), 语义ID 0-N
- `sampled_instance_labels` - (n_samples,), 实例ID
- `sampled_instance_anno_id` - (n_samples,), 实例ID对应segments_anno.json

### 5.2 分割PTH文件为训练块

将PTH文件分割为固定大小的块。训练时使用重叠块，验证时设置overlap为0。

1. **创建配置文件** `semantic/configs/split_pth_data_train.yml`（从模板复制）：

```bash
cp semantic/configs/split_pth_data_train.yml.template semantic/configs/split_pth_data_train.yml
```

2. **编辑配置文件**：

```yaml
prop_type: vtx_  # 或 sampled_

# 训练：只保留至少包含这些点数的块
n_points_threshold: 5000
n_labels_threshold: 5
n_instances_threshold: 3

ignore_label: -100
# 如果实例的这部分不在块内，则丢弃该实例
instance_frac_threshold: 0.8

# 原始场景列表
orig_list_path: /path/to/nvs_sem_train.txt
# 原始PTH文件目录
orig_pth_dir: /path/to/pth_files

# 输出场景列表
out_list_path: /path/to/output_scenes.txt
# 输出PTH目录
out_pth_dir: /path/to/output_pth_files

# 每个块的XY维度（米）
chunk_dims_xy: [5, 5]
# 训练：重叠块
chunk_stride_xy: [2.5, 2.5]
```

3. **运行分割脚本**：

```bash
python -m semantic.prep.split_pth_data semantic/configs/split_pth_data_train.yml
```

### 5.3 可视化训练数据（可选）

1. **创建配置文件** `semantic/configs/viz_pth_data.yml`（从模板复制）：

```bash
cp semantic/configs/viz_pth_data.yml.template semantic/configs/viz_pth_data.yml
```

2. **编辑配置文件并运行**：

```bash
python -m semantic.viz.viz_pth_data semantic/configs/viz_pth_data.yml
```

### 5.4 准备语义/实例评估Ground Truth

准备评估用的PTH文件（不进行点采样）。

1. **创建配置文件** `semantic/configs/prepare_semantic_gt.yml`（从模板复制）：

```bash
cp semantic/configs/prepare_semantic_gt.yml.template semantic/configs/prepare_semantic_gt.yml
```

2. **编辑配置文件并运行**：

```bash
python -m semantic.prep.prepare_semantic_gt semantic/configs/prepare_semantic_gt.yml
```

### 5.5 将3D Mesh栅格化到2D图像

将mesh栅格化到DSLR或iPhone图像，保存2D-3D映射（像素到面）。

**要求**：需要PyTorch3D和GPU

1. **编辑配置文件** `semantic/configs/rasterize.yaml`：

```yaml
# 配置参数
image_type: dslr  # 或 iphone
image_downsample_factor: 1  # 栅格化到降采样图像
subsample_factor: 1  # 每N张图像栅格化一次
batch_size: 1  # 栅格化批次大小
```

2. **运行栅格化脚本**：

```bash
python -m semantic.prep.rasterize
```

**注意**：此脚本使用Hydra配置，无需指定配置文件路径。

### 5.6 获取2D语义

使用栅格化结果获取2D图像上的语义和实例标注。

1. **创建配置文件** `semantic/configs/semantics_2d.yaml`（从模板复制）：

```bash
cp semantic/configs/semantics_2d.yaml.template semantic/configs/semantics_2d.yaml
```

2. **编辑配置文件**：

```yaml
image_type: dslr  # 或 iphone
subsample_factor: 1
undistort_dslr: false  # 如果为true，获取去畸变图像上的语义
```

3. **运行脚本**：

```bash
python -m semantic.prep.semantics_2d
```

**输出**：包含对象ID可视化的图像，可用于裁剪单个对象等任务。

### 5.7 选择最佳覆盖图像（可选）

使用提供的工具函数选择2D图像：

```python
from scannetpp.common.utils.anno import get_best_views_from_cache, get_visibility_from_cache

# 按最佳视角排序图像（增加场景覆盖）
get_best_views_from_cache(...)

# 查找每个对象在每个图像中的可见性
get_visibility_from_cache(...)
```

---

## 6. Panocam数据准备

### 6.1 将Panocam图像反投影到高分辨率点云

Panocam数据包含：
- RGB图像（JPG）
- 深度图（16位PNG）
- 匿名化mask（二进制PNG）
- 方位角和仰角图（球坐标，16位PNG，原始浮点值×1000）

1. **创建配置文件** `panocam/configs/backproject.yaml`（从模板复制）：

```bash
cp panocam/configs/backproject.yaml.template panocam/configs/backproject.yaml
```

2. **编辑配置文件并运行**：

```bash
python -m panocam.backproject
```

**注意**：
- 一些panocam图像宽高比为2:1（包含扫描仪）
- 一些宽高比为2.5:1（不包含扫描仪）
- 有效像素mask可通过 `depth > 0` 获得

---

## 7. 常见问题

### 7.1 路径问题

- **所有路径都应该是绝对路径**，避免相对路径导致的错误
- 确保`data_root`指向正确的数据集根目录
- 检查输出目录的写入权限

### 7.2 内存问题

- 如果内存不足，可以：
  - 降低`sample_factor`（减少采样点数）
  - 使用`downscale_factor`降采样图像
  - 分批处理场景（使用`scene_ids`而不是`splits`）

### 7.3 GPU问题

- 确保CUDA版本与PyTorch版本匹配
- 深度渲染和栅格化需要GPU
- 检查GPU内存是否足够

### 7.4 依赖问题

- 确保所有依赖都已正确安装
- 如果遇到`renderpy`问题，检查是否正确安装
- 对于栅格化，需要安装PyTorch3D

### 7.5 数据格式问题

- 确保下载的数据完整
- 检查mesh文件是否存在
- 验证`segments_anno.json`文件格式

---

## 8. 完整工作流示例

以下是一个完整的数据准备工作流示例：

```bash
# 1. 环境准备
conda create -n scannetpp python=3.10
conda activate scannetpp
pip install -r requirements.txt
pip install git+https://github.com/liu115/renderpy

# 2. DSLR去畸变
python -m dslr.undistort dslr/configs/undistort.yml

# 3. iPhone数据提取
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml

# 4. 渲染深度图
python -m common.render common/configs/render.yml

# 5. 准备语义训练数据
python -m semantic.prep.prepare_training_data semantic/configs/prepare_training_data.yml

# 6. 分割PTH文件
python -m semantic.prep.split_pth_data semantic/configs/split_pth_data_train.yml

# 7. 栅格化（如果需要2D语义）
python -m semantic.prep.rasterize

# 8. 获取2D语义
python -m semantic.prep.semantics_2d
```

---

## 9. 参考资源

- [官方数据集文档](https://kaldir.vc.in.tum.de/scannetpp/documentation)
- [提交说明](https://kaldir.vc.in.tum.de/scannetpp/benchmark/docs)
- [renderpy仓库](https://github.com/liu115/renderpy)

---

## 10. 注意事项

1. **数据大小**：预处理后的数据可能非常大，确保有足够的磁盘空间
2. **处理时间**：某些步骤（如栅格化）可能需要较长时间，建议使用GPU
3. **批次处理**：对于大量场景，建议分批处理以避免内存问题
4. **备份**：处理前建议备份原始数据

---

**最后更新**：根据ScanNet++官方README和代码库整理


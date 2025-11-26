# 自定义输出路径使用指南

## 问题背景

当您对 `input_root` 下的数据只有**读取权限**时，无法将处理结果写入到同一个目录下。本指南说明如何自定义输出路径。

## 修改内容

已修改以下文件以支持自定义输出路径：

1. **`downscale.py`** - 支持自定义输出根目录
2. **`undistort.py`** - 支持自定义输出根目录
3. **`undistort_given_intrinsics.py`** - 支持自定义输出根目录
4. **配置文件** - 所有 `.yml` 配置文件已添加 `output_root` 选项说明

## 使用方法

### 方法 1: 在配置文件中指定 `output_root`

编辑您的配置文件（例如 `dslr/configs/downscale.yml`），添加 `output_root` 选项：

```yaml
# folder where the data is downloaded
data_root: /path/to/scannetpp/data

splits: [nvs_sem_train, nvs_sem_val]

# 自定义输出根目录（必须具有写入权限）
output_root: /path/to/your/writable/directory

downscale_factor: 2.0

# 这些路径是相对于输出基目录的
out_image_dir: resized_images_2
out_mask_dir: resized_anon_masks_2
out_transforms_path: nerfstudio/transforms_2.json
```

**输出路径结构：**
- 如果不指定 `output_root`：输出到 `{data_root}/data/{scene_id}/dslr/{out_image_dir}`
- 如果指定 `output_root`：输出到 `{output_root}/data/{scene_id}/dslr/{out_image_dir}`

### 方法 2: 不指定 `output_root`（默认行为）

如果您有写入权限，可以不指定 `output_root`，代码会使用默认行为（输出到 `data_root` 下）：

```yaml
data_root: /path/to/scannetpp/data
splits: [nvs_sem_train, nvs_sem_val]
# output_root: null  # 或者直接不写这一行

downscale_factor: 2.0
out_image_dir: resized_images_2
out_mask_dir: resized_anon_masks_2
out_transforms_path: nerfstudio/transforms_2.json
```

## 具体示例

### 示例 1: 使用自定义输出路径运行 downscale

假设：
- 输入数据在：`/readonly/data/scannetpp/data/`（只读）
- 您想输出到：`/writable/output/scannetpp/`

**步骤 1：** 编辑 `dslr/configs/downscale.yml`：

```yaml
data_root: /readonly/data/scannetpp
output_root: /writable/output/scannetpp
splits: [nvs_sem_train, nvs_sem_val]
downscale_factor: 2.0
out_image_dir: resized_images_2
out_mask_dir: resized_anon_masks_2
out_transforms_path: nerfstudio/transforms_2.json
```

**步骤 2：** 运行命令：

```bash
python -m dslr.downscale dslr/configs/downscale.yml
```

**输出结果将保存在：**
- 图像：`/writable/output/scannetpp/data/{scene_id}/dslr/resized_images_2/`
- 掩码：`/writable/output/scannetpp/data/{scene_id}/dslr/resized_anon_masks_2/`
- 变换文件：`/writable/output/scannetpp/data/{scene_id}/dslr/nerfstudio/transforms_2.json`

### 示例 2: 使用相对路径

您也可以使用相对路径（相对于当前工作目录）：

```yaml
data_root: /readonly/data/scannetpp
output_root: ./output  # 相对路径
splits: [nvs_sem_train, nvs_sem_val]
downscale_factor: 2.0
out_image_dir: resized_images_2
out_mask_dir: resized_anon_masks_2
out_transforms_path: nerfstudio/transforms_2.json
```

## 不同脚本的输出路径结构

### 1. `downscale.py` 和 `undistort.py`

输出路径结构：
```
{output_root}/data/{scene_id}/dslr/{out_image_dir}/
{output_root}/data/{scene_id}/dslr/{out_mask_dir}/
{output_root}/data/{scene_id}/dslr/{out_transforms_path}
```

### 2. `undistort_given_intrinsics.py`

输出路径结构：
```
{output_root}/data/{scene_id}/dslr_undistorted_by_iphone/{out_image_dir}/
{output_root}/data/{scene_id}/dslr_undistorted_by_iphone/{out_mask_dir}/
{output_root}/data/{scene_id}/dslr_undistorted_by_iphone/{out_transforms_path}
```

## 注意事项

1. **目录权限**：确保 `output_root` 指定的目录具有写入权限
2. **目录结构**：代码会自动创建所需的子目录结构
3. **路径格式**：`output_root` 可以是绝对路径或相对路径
4. **向后兼容**：如果不指定 `output_root`，代码会使用原来的默认行为（输出到 `data_root` 下）

## 验证修改

运行前，您可以先检查输出路径是否正确：

```python
from pathlib import Path
from common.scene_release import ScannetppScene_Release

scene_id = "your_scene_id"
data_root = Path("/readonly/data/scannetpp")
output_root = Path("/writable/output/scannetpp")

scene = ScannetppScene_Release(scene_id, data_root=data_root / "data")
output_base = output_root / "data" / scene_id / "dslr"

print(f"输出图像目录: {output_base / 'resized_images_2'}")
print(f"输出掩码目录: {output_base / 'resized_anon_masks_2'}")
```

## 故障排除

### 问题 1: 权限错误

**错误信息：** `PermissionError: [Errno 13] Permission denied`

**解决方案：** 检查 `output_root` 目录的写入权限，或选择另一个有写入权限的目录。

### 问题 2: 路径不存在

**错误信息：** `FileNotFoundError`

**解决方案：** 代码会自动创建目录，但如果父目录不存在，请先创建 `output_root` 目录。

### 问题 3: 配置文件格式错误

**错误信息：** YAML 解析错误

**解决方案：** 确保 YAML 文件格式正确，注意缩进和冒号后的空格。


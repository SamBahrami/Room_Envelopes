# Room Envelopes: A Synthetic Dataset for Indoor Layout Reconstruction from Images

[![🌐 Project Page](https://img.shields.io/badge/🌐%20Project%20Page-blue)](https://sambahrami.com/room_envelopes)
[![arXiv](https://img.shields.io/badge/arXiv-2511.03970-b31b1b.svg)](https://arxiv.org/abs/2511.03970)
[![🤗 Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/hugsam/Room_Envelopes)


We introduce Room Envelopes, a synthetic dataset that provides dual depth representations for indoor scene reconstruction. This repository contains code to generate the Room Envelopes layout dataset. See our paper for information about that dataset. The pipeline consists of three main steps that require different Python environments.

For quick access to the preprocessed data, the dataset is available on [Hugging Face](https://huggingface.co/datasets/hugsam/Room_Envelopes), which includes all scenes that were successfully processed with this method.
Each tar file contains 5000 image-depth pairs and can be downloaded and used individually.
Currently, we have released **depth images and normals per view** due to file size constraints (pointmaps are not included).

We have also released the fine-tuned model we trained for layout estimation in Room Envelopes, at the same [Hugging Face](https://huggingface.co/datasets/hugsam/Room_Envelopes) repository (`room_envelopes_layout_model.pt`).

## Updates
- 2025/12/03: Released the generation code, dataset used to train the model presented in the Room Envelopes paper, and the model we trained.

## Using the Dataset and Pretrained Model

The dataset and pretrained model can be downloaded from our [Hugging Face repository](https://huggingface.co/datasets/hugsam/Room_Envelopes). 

The dataset is compatible with the [MoGe](https://github.com/microsoft/moge) project's file formats. You will need to use the `read_depth`, `write_depth`, `read_normal`, and `write_normal` functions from that library to correctly load our files. These functions are also provided in `scripts/moge_scripts.py` in this repository.

To use our pretrained model, use `moge infer` from the [MoGe](https://github.com/microsoft/moge) library with our pretrained model (`room_envelopes_layout_model.pt`). **Note:** This pretrained model is compatible with **MoGe v1** only, not v2. Use the following command to get started: 

```
moge infer -i /path/to/data/example_image.jpg -o /output/directory/ --pretrained /path/to/room_envelopes_layout_model.pt --version v1 --maps --ply --glb
```

We include an example in the `data/example_image_inference` folder using `example_image.jpg`, which is a real-world photo.

## Dataset Generation

Generating the dataset requires several steps and careful setup. If you encounter any issues with the instructions below, please let us know. The process consists of 3 main steps:

1. **Downloading Hypersim** 
2. **Point Cloud Generation** (`open3d_make_pointcloud.py`) - Creates as complete as possible downsampled point clouds from Hypersim data
3. **Layout Rendering** (`pytorch3d_render_layout.py`) - Renders depth, RGB, pointmaps, and normal maps from point clouds corresponding to the views provided in Hypersim

## Step 1: Downloading Hypersim

Clone the Hypersim project from https://github.com/apple/ml-hypersim and use the following bash commands to download all required files. Replace `/path/to/hypersim` with your own dataset path.
```
cd ml-hypersim/contrib/99991
python download.py --contains .csv --directory '/path/to/hypersim' --silent
python download.py --contains position.hdf5 --contains scene_cam_ --directory '/path/to/hypersim' --silent
python download.py --contains color.hdf5 --contains scene_cam_ --directory '/path/to/hypersim' --silent
python download.py --contains semantic.hdf5 --contains scene_cam_ --directory '/path/to/hypersim' --silent
python download.py --contains depth_meters.hdf5 --contains scene_cam_ --directory '/path/to/hypersim' --silent
python download.py --contains camera_keyframe_orientations.hdf5 --directory '/path/to/hypersim' --silent
python download.py --contains camera_keyframe_positions.hdf5 --directory '/path/to/hypersim' --silent
# We download both normal_cam (first visible surface) and normal_world (layout surface) to maintain
# the ground truth first layer from Hypersim, though normal_world alone might suffice 
python download.py --contains normal_world.hdf5 --contains scene_cam_ --directory '/path/to/hypersim' --silent
python download.py --contains normal_cam.hdf5 --contains scene_cam_ --directory '/path/to/hypersim' --silent
```

## Step 2: Generating Point Clouds
### Make an Open3D Environment
Required for `open3d_make_pointcloud.py`

```bash
# Create conda environment
conda create -n open3d_env python=3.10 -y
conda activate open3d_env

# Install dependencies
pip install open3d
pip install numpy pandas h5py pathlib tqdm

# For specific versions use 
# pip install open3d~=0.19.0
# pip install numpy~=2.2.6 pandas~=2.3.3 h5py~=3.15.1 pathlib tqdm
```

### Generate Point Clouds

Activate the Open3D environment and generate point clouds from the Hypersim dataset using the command below.
We provide a bash script (**open3d_generate_pointclouds.sh**) that can process all downloaded scenes. Edit the paths in the script and run it from the repository root to process all scenes.

```bash
python scripts/open3d_make_pointcloud.py \
    --hypersim-dir /path/to/hypersim/dataset \
    --scene ai_001_001 \
    --destination /path/to/output/pointclouds \
    --num-files 100 \
    --voxel-size 0.00125 \
    --verbose
```

**Parameters:**
- `--hypersim-dir`: Path to the root Hypersim dataset directory
- `--scene`: The scene name to process
- `--destination`: Directory where point cloud files will be saved
- `--num-files`: Number of frames to process per camera (default: 100)
- `--voxel-size`: Voxel size for point cloud downsampling (default: 0.00125)
- `--verbose`: Enable detailed output

**Output:**
- `{scene_name}.ply`: Point cloud file with RGB, normals, and semantic labels for each scene
- `{scene_name}.npz`: Semantic label data for each scene

## Step 3: Render Layout Dataset
### Create a PyTorch3D Environment
Required for `pytorch3d_render_layout.py`

```bash
# Create conda environment
conda create -n pytorch3d python=3.9 -y
conda activate pytorch3d
# Install PyTorch3D following the official instructions:
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# The commands below may not work for all systems. If they fail, follow the official installation guide.
# Note: Some users may need to compile PyTorch3D from source (I did).
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c iopath iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
# Install other dependencies
pip install -r requirements.txt
```

### Render Layout Maps

Activate the PyTorch3D environment and render the Room Envelopes dataset from the point clouds.
We also provide a bash script (**pytorch3d_generate_dataset.sh**) to process all scenes. Edit the paths at the top of the file before running.

```bash
conda activate pytorch3d

python scripts/pytorch3d_render_layout.py \
    --pointcloud-directory /path/to/output/pointclouds \
    --scene scene_name \
    --hypersim-dir /path/to/hypersim/dataset \
    --save-dir /path/to/output/renders \
    --min-frame-id 0 \
    --max-frame-id 100 \
    --keep-classes wall floor window door ceiling \
    --verbose
```

**Parameters:**
- `--pointcloud-directory`: Directory containing the `.ply` and `.npz` files from Step 1
- `--scene`: Scene to process e.g. ai_001_001/
- `--min-frame-id`: Minimum frame number to process (default: 0). Useful for re-rendering specific frame ranges from a scene
- `--max-frame-id`: Maximum frame number to process (default 100)
- `--hypersim-dir`: Path to the root Hypersim dataset directory
- `--save-dir`: Directory where rendered outputs will be saved
- `--num-files`: Number of frames to render (default: 100)
- `--keep-classes`: Semantic classes to include in rendering (defaults to wall floor window door ceiling)
- `--verbose`: More informative output in terminal

**Output for each frame id:**
- `image.png`: Original RGB image from Hypersim
- `first_depth.png`: First visible surface depth map 
- `first_normal.png`: Surface normal map (encoded for MoGe compliance)
- `layout_depth_0.png`: First layout surface depth map 
- `layout_layer_0_normal.png`: First layout surface normal map
- `final_layer_pointmap.hdf5`: Point coordinates for each pixel
- `meta.json`: Normalised camera intrinsics metadata

## Notes

- GPU support is recommended for the PyTorch3D rendering step
- Point clouds are in metric scale from the Open3D processing step
- Point cloud downsampling in Step 2 uses a configurable voxel size via the `--voxel-size` parameter (default: 0.00125) since this is in metric scale this is 1.25mm
- We comply with the CC BY-NC-SA 3.0 license provided by the original Hypersim dataset

## Citation

If you use this dataset in your research, please cite our work:

```bibtex
@inproceedings{bahrami2025roomenvelopes,
  title={Room Envelopes: A Synthetic Dataset for Indoor Layout Reconstruction from Images},
  author={Bahrami, Sam and Campbell, Dylan},
  booktitle={Australasian Joint Conference on Artificial Intelligence},
  pages={229--241},
  year={2025},
}
```


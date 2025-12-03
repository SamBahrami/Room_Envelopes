from pathlib import Path
import numpy as np  
import pandas as pd
from hdf5_helpers import (
    load_pointcloud_hdf5,
    load_rgb_image_hdf5,
    load_semantic_hdf5,
)
import torch
from tqdm import tqdm
from pytorch3d.io import IO

class BasePointCloud:
    """Base class for point clouds with semantic labeling capabilities."""
    
    def __init__(self, device: torch.device = torch.device("cuda:0")):
        self.device = device
        self.points = None
        self.features = None  # RGB + semantic labels
        self.semantic_colours = None
        self.semantic_id_map = None
        self.normals = None
    
    def scale_points(self, scale: float):
        self.points = self.points * scale

    def load_semantic_colours(self, semantic_label_description_file: Path) -> tuple:
        """Load semantic label colors and descriptions from CSV file."""
        # Read CSV and clean column names
        df = pd.read_csv(semantic_label_description_file)
        df.columns = df.columns.str.strip()

        # Clean the data by stripping whitespace and converting to integers
        colors = np.array(
            [
                [r / 255, g / 255, b / 255]
                for r, g, b in zip(
                    df["semantic_color_r"], df["semantic_color_g"], df["semantic_color_b"]
                )
            ]
        )
        # Create a mapping of semantic names to their IDs
        semantic_id_map = {
            name.strip(): int(id)
            for name, id in zip(df["semantic_name"], df["semantic_id"])
        }
        return colors, semantic_id_map
    
    def from_arrays(self, points: np.ndarray, rgbs: np.ndarray, normals: np.ndarray, semantic_labels: np.ndarray, 
                   semantic_colours_file: Path = None):
        """Initialize point cloud from numpy arrays."""
        if semantic_colours_file:
            self.semantic_colours, self.semantic_id_map = self.load_semantic_colours(semantic_colours_file)
        
        # Convert to PyTorch tensors
        self.points = torch.tensor(
            points, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, N, 3)

        normals = torch.from_numpy(normals).to(self.device).unsqueeze(0)  # (1, N, 3)
        self.normals = normals
        
        colours = torch.from_numpy(rgbs).to(self.device).unsqueeze(0)  # (1, N, 3)
        semantic_labels = (
            torch.from_numpy(semantic_labels).to(self.device).unsqueeze(0)
        )  # (1, N, 1)
        self.features = torch.cat([colours, semantic_labels], dim=2)  # (1, N, 4)

        return self
    
    def from_ply_file(self, ply_path: Path, semantic_labels_path: Path = None, 
                     semantic_colours_file: Path = None):
        """Load point cloud from PLY file and optional semantic labels using PyTorch3D."""
        # Load PLY file using PyTorch3D's IO module
        pointcloud = IO().load_pointcloud(str(ply_path))
        verts = pointcloud.points_padded()[0]
        
        # Extract points and colors (PyTorch3D returns tensors)
        points = verts.cpu().numpy()
        # Assume colors are stored as vertex features - this may need adjustment based on PLY format
        rgbs = pointcloud.features_padded()[0].cpu().numpy()  # Default gray if no colors in PLY

        # Load normals if provided
        normals = pointcloud.normals_padded()[0].cpu().numpy()
        
        # Load semantic labels if provided
        if semantic_labels_path:
            semantic_data = np.load(semantic_labels_path)
            if semantic_labels_path.suffix == '.npz':
                semantic_labels = semantic_data['semantic_labels']
            else:
                semantic_labels = semantic_data
        else:
            # Default to zeros if no semantic labels provided
            semantic_labels = np.zeros((points.shape[0], 1), dtype=np.int32)
        
        return self.from_arrays(points, rgbs, normals, semantic_labels, semantic_colours_file)

    def get_semantic_id_map(self):
        return self.semantic_id_map

    def get_semantic_colours(self):
        return self.semantic_colours
    
    def set_points(self, points: torch.Tensor):
        self.points = points

    def get_points(self):
        return self.points

    def get_rgbs(self):
        return self.features[:, :, :3]

    def get_semantic_labels(self):
        return self.features[:, :, 3]
    
    def get_features(self):
        return self.features
    
    def get_normals(self):
        return self.normals
    
    def filter_points_by_semantic(self, keep_classes: list[str], keep_unlabeled: bool = False):
        """Filter points to keep only those with specified semantic classes.
        
        Parameters:
            keep_classes: List of semantic class names to keep
            keep_unlabeled: If True, also keep points with semantic ID <= 0 (unlabeled/unknown)
        """
        if self.semantic_id_map is None:
            raise ValueError("No semantic ID map loaded. Cannot filter by semantic classes.")
            
        # Create mask for points to keep
        keep_mask = torch.zeros(self.points.shape[1], dtype=torch.bool).to(
            self.features.device
        )

        # Use semantic IDs from the CSV file
        for class_name in keep_classes:
            if class_name not in self.semantic_id_map:
                print(f"Warning: Class '{class_name}' not found in semantic ID map")
                continue
            class_id = self.semantic_id_map[class_name]
            class_mask = self.get_semantic_labels() == class_id
            keep_mask = keep_mask | class_mask.squeeze(-1).squeeze(0)

        # Optionally include unlabeled points (semantic ID <= 0)
        if keep_unlabeled:
            semantic_labels = self.get_semantic_labels().squeeze(-1).squeeze(0)
            unlabeled_mask = semantic_labels <= 0
            keep_mask = keep_mask | unlabeled_mask
            num_unlabeled = unlabeled_mask.sum().item()
            print(f"Keeping {num_unlabeled} unlabeled points (semantic ID <= 0)")

        # Apply mask to points and features
        filtered_points = self.points[:, keep_mask, :]
        filtered_features = self.features[:, keep_mask, :]
        filtered_normals = self.normals[:, keep_mask, :]
        self.points = filtered_points
        self.features = filtered_features
        self.normals = filtered_normals
        
    def to_device(self, device: torch.device):
        """Move point cloud to specified device."""
        self.device = device
        if self.points is not None:
            self.points = self.points.to(device)
        if self.features is not None:
            self.features = self.features.to(device)
        return self
    
    def save_to_ply(self, ply_path: Path, semantic_labels_path: Path = None):
        """Save point cloud to PLY file and optionally save semantic labels separately."""
        # Convert to numpy
        points_np = self.points.squeeze(0).cpu().numpy()
        rgbs_np = self.get_rgbs().squeeze(0).cpu().numpy()
        
        # Simple PLY writer (basic implementation)
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_np)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write vertex data
            for i in range(len(points_np)):
                x, y, z = points_np[i]
                r, g, b = (rgbs_np[i] * 255).astype(int)
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
        
        # Save semantic labels if requested
        if semantic_labels_path:
            semantic_labels_np = self.get_semantic_labels().squeeze(0).cpu().numpy()
            if semantic_labels_path.suffix == '.npz':
                np.savez(semantic_labels_path, semantic_labels=semantic_labels_np)
            else:
                np.save(semantic_labels_path, semantic_labels_np)


class HypersimPointCloud(BasePointCloud):
    """Point cloud class specifically for loading Hypersim dataset."""
    
    def __init__(self, base_dir: Path, scene_name: str, num_files: int = 100, cam: str = "cam_00",
                 device: torch.device = torch.device("cuda:0"), single_file: bool = False):
        super().__init__(device)
        
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.scene_details_dir = Path(base_dir / f"{scene_name}/_detail/{cam}")
        self.images_dir = Path(base_dir / f"{scene_name}/images/scene_{cam}_final_hdf5")
        self.geometry_dir = Path(base_dir / f"{scene_name}/images/scene_{cam}_geometry_hdf5/")
        
        # Load semantic colors and ID mapping
        self.semantic_colours, self.semantic_id_map = self.load_semantic_colours(
            Path("data/semantic_label_descs.csv")
        )
        
        # Load scene data
        if single_file:
            pointcloud, rgbs, semantic_labels, normals = self.load_single_file_scene_data(num_files)
        else:
            pointcloud, rgbs, semantic_labels, normals = self.load_scene_data(num_files)
        
        # Initialize using parent class method
        self.from_arrays(pointcloud, rgbs, normals, semantic_labels)

    def load_scene_data(self, num_files: int):
        """Load pointcloud, RGB and semantic data from HDF5 files."""
        pointcloud_filelist = []
        for file in self.geometry_dir.iterdir():
            if file.name.endswith("position.hdf5"):
                pointcloud_filelist.append(file)
        pointcloud_filelist.sort()
        
        color_filelist = []
        for file in self.images_dir.iterdir():
            if file.name.endswith("color.hdf5"):
                color_filelist.append(file)
        color_filelist.sort()
        
        semantic_filelist = []
        for file in self.geometry_dir.iterdir():
            if file.name.endswith("semantic.hdf5"):
                semantic_filelist.append(file)
        semantic_filelist.sort()

        normals_filelist = []
        for file in self.geometry_dir.iterdir():
            if file.name.endswith("normal_world.hdf5"):
                normals_filelist.append(file)
        normals_filelist.sort()

        pointcloud = self.load_and_merge_hdf5_files(
            pointcloud_filelist[:num_files], self.geometry_dir, load_pointcloud_hdf5
        )
        rgbs = self.load_and_merge_hdf5_files(
            color_filelist[:num_files], self.images_dir, load_rgb_image_hdf5
        )
        semantic_labels = self.load_and_merge_hdf5_files(
            semantic_filelist[:num_files],
            self.geometry_dir,
            load_semantic_hdf5,
            data_dimension=1,
        )
        return pointcloud, rgbs, semantic_labels

    def load_single_file_scene_data(self, file_number: int):
        """Load pointmap, RGB and semantic data from HDF5 files."""
        pointcloud_filelist = []
        for file in self.geometry_dir.iterdir():
            if file.name.endswith(f"0{file_number}.position.hdf5"):
                pointcloud_filelist.append(file)
        pointcloud_filelist.sort()
        
        color_filelist = []
        for file in self.images_dir.iterdir():
            if file.name.endswith(f"0{file_number}.color.hdf5"):
                color_filelist.append(file)
        color_filelist.sort()
        
        semantic_filelist = []
        for file in self.geometry_dir.iterdir():
            if file.name.endswith(f"0{file_number}.semantic.hdf5"):
                semantic_filelist.append(file)
        semantic_filelist.sort()

        normals_filelist = []
        for file in self.geometry_dir.iterdir():
            if file.name.endswith(f"0{file_number}.normal_world.hdf5"):
                normals_filelist.append(file)
        normals_filelist.sort()

        pointcloud = self.load_and_merge_hdf5_files(
            pointcloud_filelist, self.geometry_dir, load_pointcloud_hdf5
        )
        rgbs = self.load_and_merge_hdf5_files(
            color_filelist, self.images_dir, load_rgb_image_hdf5
        )
        semantic_labels = self.load_and_merge_hdf5_files(
            semantic_filelist,
            self.geometry_dir,
            load_semantic_hdf5,
            data_dimension=1,
        )
        normals = self.load_and_merge_hdf5_files(
            normals_filelist,
            self.geometry_dir,
            load_pointcloud_hdf5,
            data_dimension=3,
        )
        return pointcloud, rgbs, semantic_labels, normals

    def load_and_merge_hdf5_files(
        self, file_names: list, base_dir: Path, load_func: callable, data_dimension: int = 3
    ) -> np.ndarray:
        """Load data from multiple files and concatenate them along axis 0."""
        merged = np.empty((0, data_dimension), dtype=np.float32)
        for fname in tqdm(file_names, desc="Loading HDF5 files"):
            merged = np.concatenate((merged, load_func(base_dir / fname)), axis=0)
        return merged
    
    
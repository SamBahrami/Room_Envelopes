import h5py
import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

def load_hdf5(filepath: Path) -> np.ndarray:
    """Load and return the contents of an HDF5 file from the 'dataset' group."""
    with h5py.File(filepath, "r") as f:
        data = np.array(f["dataset"])
    return data

def load_semantic_hdf5(filepath: Path) -> np.ndarray:
    """Load semantic data from an HDF5 file"""
    data = load_hdf5(filepath)
    return data.reshape(-1, 1)

def load_pointcloud_hdf5(filepath: Path) -> np.ndarray:
    """Load point cloud data from an HDF5 file and reshape it to (-1, 3)."""
    data = load_hdf5(filepath).astype(np.float32)
    return data.reshape(-1, 3)


def load_rgb_image_hdf5(filepath: Path) -> np.ndarray:
    """Load RGB image data from an HDF5 file and reshape it to (-1, 3)."""
    data = load_hdf5(filepath)
    return data.reshape(-1, 3)


def get_hypersim_filelist(prefix: str, postfix: str, number_of_files: int = 100) -> list:
    """Generate a list of filenames with a given prefix and postfix."""
    return [f"{prefix}{i:04d}{postfix}" for i in range(number_of_files)]


def load_and_merge_hdf5_files(file_names: list, load_func: callable, data_dimension: int = 3, verbose: bool = True) -> np.ndarray:
    """
    Load data from multiple files and concatenate them along axis 0.
    
    Parameters:
        file_names : List of filenames.
        load_func  : Function to load a single file.
        data_dimension : Dimension of the data (default: 3).
        verbose : Whether to show progress bar.
        
    Returns:
        Merged numpy array.
    """
    # First pass: count total size and collect valid files
    valid_files = []
    total_size = 0
    
    # Use tqdm for progress bar if verbose and multiple files
    if verbose and len(file_names) > 1:
        iterator = tqdm(file_names, desc="Scanning files")
    else:
        iterator = file_names
    
    for fname in iterator:
        if not Path(fname).exists():
            if verbose:
                print(f"File {fname} not found")
            continue
        valid_files.append(fname)
        # Load file to get size (we'll reload it later, but this is faster than concatenate)
        data = load_func(fname)
        total_size += data.shape[0]
    
    if total_size == 0:
        return np.empty((0, data_dimension), dtype=np.float32)
    
    # Pre-allocate the final array
    merged = np.empty((total_size, data_dimension), dtype=np.float32)
    
    # Second pass: load data into pre-allocated array
    current_idx = 0
    if verbose and len(valid_files) > 1:
        iterator = tqdm(valid_files, desc="Loading files")
    else:
        iterator = valid_files
    
    for fname in iterator:
        data = load_func(fname)
        data_size = data.shape[0]
        merged[current_idx:current_idx + data_size] = data
        current_idx += data_size
    
    return merged


def calculate_scene_extent(pointcloud: np.ndarray, verbose: bool = True) -> tuple:
    """
    Calculate the extent and characteristic dimensions of a pointcloud.
    
    Args:
        pointcloud: numpy array of shape (N, 3) with XYZ coordinates
        verbose: whether to print statistics
        
    Returns:
        tuple: (extent_xyz, max_extent, diagonal_length, centroid)
    """
    # Filter out NaN values (from invalid Hypersim pixels)
    # NaN values occur in position.hdf5 for pixels with no valid 3D position
    valid_mask = ~np.isnan(pointcloud).any(axis=1)
    valid_pointcloud = pointcloud[valid_mask]
    
    if len(valid_pointcloud) == 0:
        raise ValueError("Pointcloud contains only NaN values!")
    
    min_coords = np.min(valid_pointcloud, axis=0)
    max_coords = np.max(valid_pointcloud, axis=0)
    extent_xyz = max_coords - min_coords
    max_extent = np.max(extent_xyz)
    diagonal_length = np.linalg.norm(extent_xyz)
    centroid = np.mean(valid_pointcloud, axis=0)
    
    if verbose:
        num_invalid = len(pointcloud) - len(valid_pointcloud)
        percent_valid = 100 * len(valid_pointcloud) / len(pointcloud)
        print(f"Scene extent analysis:")
        print(f"  Total points: {len(pointcloud)}")
        print(f"  Valid points: {len(valid_pointcloud)} ({percent_valid:.1f}%)")
        print(f"  NaN points filtered: {num_invalid}")
        print(f"  Min coordinates: [{min_coords[0]:.3f}, {min_coords[1]:.3f}, {min_coords[2]:.3f}]")
        print(f"  Max coordinates: [{max_coords[0]:.3f}, {max_coords[1]:.3f}, {max_coords[2]:.3f}]")
        print(f"  Extent (X,Y,Z): [{extent_xyz[0]:.3f}, {extent_xyz[1]:.3f}, {extent_xyz[2]:.3f}]")
        print(f"  Maximum extent: {max_extent:.3f}")
        print(f"  Diagonal length: {diagonal_length:.3f}")
        print(f"  Scene centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    
    return extent_xyz, max_extent, diagonal_length, centroid

def calculate_adaptive_voxel_size(pointcloud: np.ndarray, reference_voxel_size: float = 0.0015, 
                                reference_max_extent: float = 20.0, verbose: bool = True) -> float:
    """
    Calculate adaptive voxel size based on scene scale.
    
    Logic:
    - If scene extent <= reference_max_extent: use reference_voxel_size (high detail for small scenes)
    - If scene extent > reference_max_extent: scale up voxel size proportionally (reduce detail for large scenes)
    
    Args:
        pointcloud: numpy array of shape (N, 3) with XYZ coordinates  
        reference_voxel_size: voxel size for small scenes (default: 0.00125)
        reference_max_extent: extent threshold below which to use reference voxel size (default: 20.0)
        verbose: whether to print calculations
        
    Returns:
        float: recommended voxel size for this scene
    """
    extent_xyz, max_extent, diagonal_length, _ = calculate_scene_extent(pointcloud, verbose=False)
    
    if max_extent <= reference_max_extent:
        # Small scene: use high detail (provided voxel size)
        adaptive_voxel_size = reference_voxel_size
        scale_factor = 1.0
        scaling_reason = f"scene extent ({max_extent:.3f}m) <= threshold ({reference_max_extent:.3f}m)"
    else:
        # Large scene: scale up voxel size proportionally
        scale_factor = max_extent / reference_max_extent
        adaptive_voxel_size = reference_voxel_size * scale_factor
        scaling_reason = f"scene extent ({max_extent:.3f}m) > threshold ({reference_max_extent:.3f}m)"
    
    # Apply reasonable bounds to avoid extreme values
    min_voxel_size = 0.00125  # 1.25mm minimum
    max_voxel_size = 0.01     # 1cm maximum
    adaptive_voxel_size = np.clip(adaptive_voxel_size, min_voxel_size, max_voxel_size)
    
    if verbose:
        print(f"Adaptive voxel size calculation:")
        print(f"  Scene max extent: {max_extent:.3f}m")
        print(f"  Extent threshold: {reference_max_extent:.3f}m")
        print(f"  Scaling reason: {scaling_reason}")
        print(f"  Scale factor: {scale_factor:.3f}")
        print(f"  Base voxel size: {reference_voxel_size:.5f}")
        print(f"  Calculated voxel size: {reference_voxel_size * scale_factor:.5f}")
        print(f"  Final voxel size (after bounds): {adaptive_voxel_size:.5f}")
    
    return adaptive_voxel_size

def downsample_pointcloud(pointcloud: o3d.t.geometry.PointCloud, voxel_size: float = 0.01, verbose: bool = True) -> o3d.t.geometry.PointCloud:
    """Downsample a point cloud using a voxel grid."""
    if verbose:
        print(f"Downsampling point cloud with voxel size {voxel_size}.")
        points_before_downsampling = pointcloud.point['positions'].shape[0]
        print(f"Point cloud before downsampling: {points_before_downsampling} points.")
    pointcloud = pointcloud.voxel_down_sample(voxel_size)
    if verbose:
        points_after_downsampling = pointcloud.point['positions'].shape[0]
        print(f"Point cloud after downsampling: {points_after_downsampling} points.")
        print(f"Percent of points kept: {points_after_downsampling / points_before_downsampling * 100:.2f}%")
    return pointcloud


def load_and_downsample_pointcloud(base_dir: Path, scene_name: str, destination_dir: Path, num_files: int = 1, voxel_size: float = 0.0015, verbose: bool = True, scene_scale: float = 1.0, adaptive_voxel: bool = True, reference_extent: float = 10.0):
    """
    Load camera parameters, point cloud, and RGB data from Hypersim,
    then set up and run an Open3D visualizer with the correct camera view.
    
    Parameters:
        base_dir          : Base directory containing the Hypersim data.
        scene_name        : Name of the scene.
        destination_dir   : Directory to save the output files.
        num_files         : Number of files to load (default is 1).
        voxel_size        : Voxel size for small scenes (default is 0.0015), scales up for large scenes.
        verbose           : Whether to print verbose output.
    """
    # Check if output files already exist
    ply_file = destination_dir / f"{scene_name}.ply"
    npz_file = destination_dir / f"{scene_name}.npz"
    
    if ply_file.exists() and npz_file.exists():
        if verbose:
            print(f"Output files already exist for scene {scene_name}, skipping processing:")
            print(f"  {ply_file}")
            print(f"  {npz_file}")
        return
    
    # Get the paths to the images and scene info directories
    images_dir = base_dir / scene_name / "images"
    scene_info_dir = base_dir / scene_name / "_detail"
    # Find the camera paths generated for this scene
    cams = [f for f in scene_info_dir.glob("cam_*") if f.is_dir()]
    cam_names = [f.name for f in cams]
    cam_names = sorted(cam_names)
    cam_names = cam_names[:5] # Only look at the first 5 cameras, due to memory constraints.

    normal_filepaths, semantic_filepaths, point_cloud_filepaths, image_filepaths = [], [], [], []
    for cam_name in cam_names:
        final_dir = images_dir / f"scene_{cam_name}_final_hdf5"
        geometry_dir = images_dir / f"scene_{cam_name}_geometry_hdf5"

        # Load point cloud and RGB files.
        normal_filepaths = normal_filepaths + get_hypersim_filelist(f"{geometry_dir}/frame.", ".normal_world.hdf5", num_files)
        semantic_filepaths = semantic_filepaths + get_hypersim_filelist(f"{geometry_dir}/frame.", ".semantic.hdf5", num_files)
        point_cloud_filepaths = point_cloud_filepaths + get_hypersim_filelist(f"{geometry_dir}/frame.", ".position.hdf5", num_files)
        image_filepaths = image_filepaths + get_hypersim_filelist(f"{final_dir}/frame.", ".color.hdf5", num_files)
    if verbose:
        print(f"Loading normal labels from {len(normal_filepaths)} files.")
    normal_labels = load_and_merge_hdf5_files(normal_filepaths, load_pointcloud_hdf5, data_dimension=3, verbose=verbose)
    if verbose:
        print(f"Loading semantic labels from {len(semantic_filepaths)} files.")
    semantic_labels = load_and_merge_hdf5_files(semantic_filepaths, load_semantic_hdf5, data_dimension=1, verbose=verbose)
    if verbose:
        print(f"Loading pointcloud from {len(point_cloud_filepaths)} files.")
    pointcloud = load_and_merge_hdf5_files(point_cloud_filepaths, load_pointcloud_hdf5, verbose=verbose)
    if verbose:
        print(f"Loading RGB images from {len(image_filepaths)} files.")
    rgbs = load_and_merge_hdf5_files(image_filepaths, load_rgb_image_hdf5, verbose=verbose)

    assert pointcloud.shape[0] == rgbs.shape[0] == semantic_labels.shape[0] == normal_labels.shape[0]
    # Create the Open3D point cloud.
    device = o3d.core.Device("CPU:0")

    # Scale points by scalar value to make them metric scale
    pointcloud = pointcloud * scene_scale
    
    # Analyze scene and potentially use adaptive voxel size
    if verbose:
        print(f"\n=== Scene Analysis ===")
    extent_xyz, max_extent, diagonal_length, centroid = calculate_scene_extent(pointcloud, verbose=verbose)
    
    if adaptive_voxel:
        # Calculate adaptive voxel size
        final_voxel_size = calculate_adaptive_voxel_size(
            pointcloud, 
            reference_voxel_size=voxel_size,
            reference_max_extent=reference_extent,
            verbose=verbose
        )
        if verbose:
            print(f"\nUsing adaptive voxel size: {final_voxel_size:.5f} (original: {voxel_size:.5f})")
    else:
        final_voxel_size = voxel_size
        if verbose:
            print(f"\nUsing fixed voxel size: {final_voxel_size:.5f}")
    
    # Filter out NaN points before creating Open3D point cloud
    # NaN values come from invalid pixels in Hypersim (sky, far background, etc.)
    valid_mask = ~np.isnan(pointcloud).any(axis=1)
    num_before = len(pointcloud)
    pointcloud = pointcloud[valid_mask]
    rgbs = rgbs[valid_mask]
    normal_labels = normal_labels[valid_mask]
    semantic_labels = semantic_labels[valid_mask]
    num_after = len(pointcloud)
    
    if verbose:
        print(f"\n=== NaN Filtering ===")
        print(f"  Points before filtering: {num_before}")
        print(f"  Points after filtering: {num_after}")
        print(f"  NaN points removed: {num_before - num_after} ({100*(num_before - num_after)/num_before:.1f}%)")
        
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(pointcloud, device=device)
    pcd.point.colors = o3d.core.Tensor(rgbs, device=device)
    pcd.point.normals = o3d.core.Tensor(normal_labels, device=device)
    
    # Ensure semantic labels and normal labels are properly shaped and typed
    semantic_labels_1d = semantic_labels.flatten().astype(np.int32)
    if verbose:
        print(f"Semantic labels shape: {semantic_labels_1d.shape}")
        print(f"Pointcloud shape: {pointcloud.shape}")
        print(f"RGB shape: {rgbs.shape}")
    
    pcd.point["semantic_label"] = o3d.core.Tensor(semantic_labels_1d.reshape(-1, 1), dtype=o3d.core.int32, device=device)
    # Downsample point cloud.
    # send to CPU for downsampling
    cpu_device = o3d.core.Device("CPU:0")
    pcd = pcd.to(cpu_device)
    pcd = downsample_pointcloud(pcd, final_voxel_size, verbose=verbose)
    pcd = pcd.to(device)
    
    # Create destination directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    # Save semantic label npz file.
    # convert to numpy array
    semantic_labels_np = pcd.point["semantic_label"].cpu().numpy()
    np.savez(destination_dir / f"{scene_name}.npz", semantic_labels=semantic_labels_np)
    legacy_pcd = pcd.to_legacy()

    # Save point cloud to file.
    o3d.io.write_point_cloud(str(destination_dir / f"{scene_name}.ply"), legacy_pcd)
    if verbose:
        print(f"Point cloud saved to {destination_dir / f'{scene_name}.ply'}")

def main():
    parser = argparse.ArgumentParser(description="Generate point clouds from Hypersim dataset")
    parser.add_argument("--hypersim-dir", type=str, required=True,
                        help="Path to the Hypersim dataset directory")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name to process")
    parser.add_argument("--destination", type=str, required=True,
                        help="Destination directory to save output files")
    parser.add_argument("--num-files", type=int, default=100,
                        help="Number of files to load per modality (default: 100)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose output")
    parser.add_argument("--voxel-size", type=float, default=0.0015,
                        help="Voxel size for small scenes (default: 0.0015, scales up for large scenes)")
    parser.add_argument("--adaptive-voxel", action="store_true", default=True,
                        help="Use adaptive voxel sizing based on scene scale (default: True)")
    parser.add_argument("--no-adaptive-voxel", dest="adaptive_voxel", action="store_false",
                        help="Disable adaptive voxel sizing, use fixed voxel size")
    parser.add_argument("--reference-extent", type=float, default=20.0,
                        help="Reference scene max extent for adaptive voxel sizing (default: 10.0)")
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    hypersim_dir = Path(args.hypersim_dir)
    destination_dir = Path(args.destination)

    metadata_csv = Path("data/metadata_camera_parameters.csv")
    df_camera_parameters = pd.read_csv(metadata_csv, index_col="scene_name")
    scene_metric_scale = df_camera_parameters.loc[args.scene, "settings_units_info_meters_scale"]
    
    # Validate that the hypersim directory exists
    if not hypersim_dir.exists():
        print(f"Error: Hypersim directory '{hypersim_dir}' does not exist")
        return 1
    
    verbose = args.verbose
    if verbose:
        print(f"Hypersim directory: {hypersim_dir}")
        print(f"Destination directory: {destination_dir}")
    
    scene = args.scene
    try:
        scene_name = scene.strip()
        # remove trailing slash if present
        if scene_name.endswith('/'):
            scene_name = scene_name[:-1]

        # Validate that the scene directory exists
        scene_dir = hypersim_dir / scene_name
        if not scene_dir.exists():
            print(f"Error: Scene directory '{scene_dir}' does not exist", file=sys.stderr)
            return 1
    
        # Process the scene
        if verbose:
            print(f"Processing scene {scene_name}")
        load_and_downsample_pointcloud(
            base_dir=hypersim_dir,
            scene_name=scene_name,
            destination_dir=destination_dir,
            num_files=args.num_files,
            voxel_size=args.voxel_size,
            verbose=args.verbose,
            scene_scale=scene_metric_scale,
            adaptive_voxel=args.adaptive_voxel,
            reference_extent=args.reference_extent
        )
        if verbose:
            print(f"Processing complete for scene {scene_name}")
    except Exception as e:
        print(f"Error processing scene {scene_name}: {e}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    exit(main())

import numpy as np
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
import scipy.linalg
try:
    from scripts.moge_scripts import write_normal, write_depth, get_metadata_camera_parameters
    from scripts.hdf5_helpers import load_hdf5, write_hdf5_from_numpy_array
    from scripts.hypersim_pointcloud import BasePointCloud
except ModuleNotFoundError:
    # Fallback for running as a script from repository root
    from moge_scripts import write_normal, write_depth, get_metadata_camera_parameters
    from hdf5_helpers import load_hdf5, write_hdf5_from_numpy_array
    from hypersim_pointcloud import BasePointCloud
import argparse
import json
import cv2

# Create a class to store scene info
class SceneInfo:
    def __init__(self, scene_name, pointcloud_directory, cam_number, hypersim_directory):
        self.scene_name = scene_name
        self.pointcloud_directory = pointcloud_directory
        self.cam_number = cam_number
        self.hypersim_directory = hypersim_directory
        self.scene_details_dir = Path(f"{hypersim_directory}/_detail/{cam_number}")
        self.images_dir = Path(f"{hypersim_directory}/images/scene_{cam_number}_final_hdf5")
        self.geometry_dir = Path(f"{hypersim_directory}/images/scene_{cam_number}_geometry_hdf5/")

def get_scene_extrinsics(scene_details_dir: Path):
    """
    Load camera extrinsics (rotation and translation) from HDF5 files located in the scene details directory.
    Returns:
        R_array: Rotation matrices.
        T_array: Translation vectors.
    """
    orientations_path = scene_details_dir / "camera_keyframe_orientations.hdf5"
    positions_path = scene_details_dir / "camera_keyframe_positions.hdf5"
    camera_orientations = load_hdf5(orientations_path)
    camera_positions = load_hdf5(positions_path)
    return camera_orientations, camera_positions


def get_frame_camera(camera_orientations, camera_positions, frame_id: int = 0):
    """Get the camera matrix in world2cam convention for a specific frame.

    Parameters:
        camera_orientations : List of camera orientations.
        camera_positions    : List of camera positions.
        frame_id            : Frame index.

    Returns:
        Camera matrix 4x4
    """
    try:
        camera_position_world = camera_positions[frame_id]
        R_world_from_cam = camera_orientations[frame_id]
        t_world_from_cam = np.matrix(camera_position_world).T
        R_cam_from_world = np.matrix(R_world_from_cam).T
        t_cam_from_world = -R_cam_from_world * t_world_from_cam
        return np.matrix(
            np.block([[R_cam_from_world, t_cam_from_world], [np.matrix(np.zeros(3)), 1.0]])
        )
    except IndexError:
        return None


def get_cam_intrinsics(df_camera_parameters, scene_name, scale: float = 1.0):
    df_ = df_camera_parameters.loc[scene_name]

    # Optionally scale image width and height to speed up rendering
    width_pixels = int(df_["settings_output_img_width"] * scale)
    height_pixels = int(df_["settings_output_img_height"] * scale)
    if df_["use_camera_physical"]:
        fov_x = df_["camera_physical_fov"]
    else:
        fov_x = df_["settings_camera_fov"]

    fov_y = 2.0 * np.arctan(height_pixels * np.tan(fov_x / 2.0) / width_pixels)

    M_cam_from_uv = np.matrix(
        [
            [df_["M_cam_from_uv_00"], df_["M_cam_from_uv_01"], df_["M_cam_from_uv_02"]],
            [df_["M_cam_from_uv_10"], df_["M_cam_from_uv_11"], df_["M_cam_from_uv_12"]],
            [df_["M_cam_from_uv_20"], df_["M_cam_from_uv_21"], df_["M_cam_from_uv_22"]],
        ]
    )

    M_cam_from_uv_canonical = np.matrix(
        [
            [np.tan(fov_x / 2.0), 0.0, 0.0],
            [0.0, np.tan(fov_y / 2.0), 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    # PyTorch3D seems to have problems with the non-standard perspective projection matrices
    # found in Hypersim, so we construct a matrix to transform a camera-space point from its
    # original position to a warped position, such that the warped position can be projected
    # with a standard perspective projection matrix. This matrix completely accounts for the
    # non-standard Hypersim camera parameters.
    M_warp_cam_pts_ = M_cam_from_uv_canonical * M_cam_from_uv.I
    M_warp_cam_pts = scipy.linalg.block_diag(M_warp_cam_pts_, 1)
    return M_warp_cam_pts, fov_y, width_pixels, height_pixels


def create_seen_mask(
    scene_info: SceneInfo,
    frame_id: int,
    keep_classes: list[str],
    semantic_id_map: dict,
    frame_dir: Path,
    keep_unlabeled: bool = False
):
    """Create and save a seen mask based on semantic filtering.
    
    The seen mask indicates which pixels contain valid depth information
    for the specified semantic classes (e.g., wall, floor, ceiling).
    
    Parameters:
        scene_info: SceneInfo object containing scene paths
        frame_id: Frame number to process
        keep_classes: List of semantic class names to keep
        semantic_id_map: Mapping of semantic names to IDs
        frame_dir: Directory to save the seen mask
    """
    # Construct paths to semantic and depth files
    frame_prefix = f"frame.{frame_id:04d}"
    semantic_filename = f"{frame_prefix}.semantic.hdf5"
    depth_filename = f"{frame_prefix}.depth_meters.hdf5"
    
    semantic_path = scene_info.geometry_dir / semantic_filename
    depth_path = scene_info.geometry_dir / depth_filename
    
    # Check if files exist
    if not semantic_path.exists() or not depth_path.exists():
        print(f"Warning: Semantic or depth file not found for frame {frame_id}")
        return
    
    # Load semantic labels and depth image
    semantic_image = load_hdf5(semantic_path)
    depth_image = load_hdf5(depth_path)
    
    # Create semantic filter for specified classes
    semantic_image_filter = np.zeros_like(semantic_image)
    for class_name in keep_classes:
        if class_name in semantic_id_map:
            semantic_image_filter[semantic_image == semantic_id_map[class_name]] = 1
    
    # Optionally include unlabeled pixels (semantic ID <= 0)
    if keep_unlabeled:
        semantic_image_filter[semantic_image <= 0] = 1
    
    # Get image dimensions
    height, width = depth_image.shape[:2]
    
    # Reshape semantic filter to match image dimensions
    semantic_filter_2d = semantic_image_filter.reshape(height, width)
    
    # Apply semantic filter to depth image
    filtered_depth_image = depth_image * semantic_filter_2d
    
    # Set filtered out pixels to NaN
    filtered_depth_image = np.where(semantic_filter_2d == 1, filtered_depth_image, np.nan)
    
    # Create seen mask (True where depth is valid after filtering)
    seen_mask = ~np.isnan(filtered_depth_image)
    
    # Convert to binary image (255 for seen, 0 for unseen)
    seen_mask_binary = (seen_mask * 255).astype(np.uint8)
    
    # Save the seen mask
    seen_mask_filename = frame_dir / "seen_mask.png"
    cv2.imwrite(str(seen_mask_filename), seen_mask_binary)


def get_closest_points(min_depth_map, dist_map, z_buffer, idx_map, depth_tolerance=0.01):
    """Get the closest points on the ray for pixels, considering only points near the minimum depth.
    
    Parameters:
        min_depth_map: The minimum depth values for each pixel (H, W)
        dist_map: Distance from ray to point for each pixel 
        z_buffer: Z-buffer for each pixel
        idx_map: Index map for each pixel
        depth_tolerance: Tolerance for considering points "close" to minimum depth
        
    Returns:
        closest_points: Single closest point index per pixel (H, W)
        valid_points_mask: Boolean mask indicating all valid points within tolerance (H, W, points_per_pixel)
                          Use with idx_map to get actual point indices: idx_map[valid_points_mask]
    """
    # Convert inputs to tensors if they're numpy arrays
    if isinstance(min_depth_map, np.ndarray):
        min_depth_map = torch.from_numpy(min_depth_map)
    if isinstance(z_buffer, np.ndarray):
        z_buffer = torch.from_numpy(z_buffer)
    
    # Ensure all tensors are on the same device
    device = dist_map.device
    min_depth_map = min_depth_map.to(device)
    z_buffer = z_buffer.to(device)
    
    # Initialize output tensor as long (required for indexing)
    closest_points = torch.full(
        (min_depth_map.shape[0], min_depth_map.shape[1]), 
        -1, 
        dtype=torch.long,
        device=device
    )
    
    # Create mask for valid pixels (non-zero, non-NaN)
    valid_mask = (min_depth_map != 0) & (~torch.isnan(min_depth_map))
    
    if not valid_mask.any():
        return closest_points
    
    # Create mask for points that are close to the minimum depth
    # Only consider points within depth_tolerance of the minimum depth
    min_depth_expanded = min_depth_map.unsqueeze(-1)  # (H, W, 1)
    depth_close_mask = torch.abs(z_buffer - min_depth_expanded) <= depth_tolerance
    
    # Also ensure points are positive (valid)
    valid_depth_mask = z_buffer > 0
    
    # Combine all masks
    combined_mask = valid_mask.unsqueeze(-1) & depth_close_mask & valid_depth_mask
    
    # Set distances to infinity where mask is False
    masked_dist_map = torch.where(combined_mask, dist_map, torch.inf)
    
    # Find argmin along the last dimension (points_per_pixel)
    closest_indices = torch.argmin(masked_dist_map, dim=-1)
    
    # Get the actual point indices using advanced indexing
    h_indices, w_indices = torch.meshgrid(
        torch.arange(min_depth_map.shape[0], device=device),
        torch.arange(min_depth_map.shape[1], device=device),
        indexing='ij'
    )
    
    # Only update pixels that have valid points near the minimum depth
    has_valid_points = (masked_dist_map != torch.inf).any(dim=-1)
    valid_pixels_mask = valid_mask & has_valid_points
    
    # Extract the point indices for valid pixels
    selected_indices = idx_map[
        h_indices[valid_pixels_mask], 
        w_indices[valid_pixels_mask], 
        closest_indices[valid_pixels_mask]
    ]
    closest_points[valid_pixels_mask] = selected_indices.long()
    
    
    return closest_points, combined_mask


def remove_layer_from_pointcloud(valid_points_mask, idx_map, points, features, normals_cam):
    """Remove points within depth tolerance from pointcloud for depth peeling.
    
    Parameters:
        valid_points_mask: Boolean mask (H, W, points_per_pixel) from get_closest_points
        idx_map: Point index map (H, W, points_per_pixel)
        points: Point coordinates (1, N, 3)
        features: Point features (1, N, F)
        normals_cam: Normal vectors in camera space (1, N, 3)
        
    Returns:
        points_filtered: Filtered point coordinates (1, M, 3) where M < N
        features_filtered: Filtered features (1, M, F)
        normals_filtered: Filtered normals (1, M, 3)
        removed_indices: Indices of removed points for reference
    """
    # Get indices of points to remove (points within depth tolerance)
    points_to_remove = idx_map[valid_points_mask]
    # Filter out invalid indices (-1)
    points_to_remove = points_to_remove[points_to_remove >= 0]
    # Get unique indices (same point might appear in multiple pixels)
    points_to_remove = torch.unique(points_to_remove)
    # Convert to long tensor for indexing
    points_to_remove = points_to_remove.long()
    
    # Create mask to keep points (inverse of points to remove)
    total_points = points.shape[1]
    keep_mask = torch.ones(total_points, dtype=torch.bool, device=points.device)
    keep_mask[points_to_remove] = False
    
    # Filter the pointcloud
    points_filtered = points[:, keep_mask, :]
    features_filtered = features[:, keep_mask, :]  
    normals_filtered = normals_cam[:, keep_mask, :]
    
    return points_filtered, features_filtered, normals_filtered, points_to_remove


def iterative_depth_peeling(renderer, points, features, normals_cam, final_layer_z, 
                          depth_tolerance=0.05, max_iterations=10, convergence_threshold=0.8):
    """
    Iteratively remove points from the current depth layer until we reach the next layer.
    
    Parameters:
        renderer: PyTorch3D renderer
        points: Point coordinates (1, N, 3)
        features: Point features (1, N, F)
        normals_cam: Normal vectors in camera space (1, N, 3)
        final_layer_z: Current layer depth map (H, W) 
        depth_tolerance: Tolerance for considering points at "same depth"
        max_iterations: Maximum number of iterations to prevent infinite loops
        convergence_threshold: Fraction of pixels that must move to next layer (0.0-1.0)
        
    Returns:
        points_filtered: Filtered point coordinates
        features_filtered: Filtered features
        normals_filtered: Filtered normals
        total_removed: Total number of points removed
    """
    current_points = points
    current_features = features
    current_normals = normals_cam
    total_removed = 0
    
    # Convert final_layer_z to tensor for consistent operations
    final_layer_z_tensor = torch.from_numpy(final_layer_z).to(points.device)
    
    for iteration in range(max_iterations):
        print(f"Depth peeling iteration {iteration + 1}/{max_iterations}")
        
        # Create point cloud and rasterize
        point_cloud = Pointclouds(points=current_points, features=current_features)
        fragments = renderer.rasterizer(point_cloud)
        z_buffer = fragments.zbuf.squeeze()
        idx_map = fragments.idx.squeeze()
        
        # Get new minimum depths
        height, width = final_layer_z.shape
        z_buffer_reshaped = z_buffer.view(height, width, -1)
        
        # Find minimum positive depth for each pixel
        valid_mask = z_buffer_reshaped > 0
        z_buffer_masked = torch.where(valid_mask, z_buffer_reshaped, torch.tensor(float('inf'), device=z_buffer.device))
        new_min_depths = torch.min(z_buffer_masked, dim=2)[0]
        
        # Set pixels with no valid depths to 0
        new_min_depths = torch.where(new_min_depths == float('inf'), 0.0, new_min_depths)
        
        # Check convergence: how many pixels have moved to a significantly different depth
        depth_diff = torch.abs(new_min_depths - final_layer_z_tensor)
        
        # Only consider pixels that had valid depths in both original and new layers
        valid_original = (~torch.isnan(final_layer_z_tensor)) & (final_layer_z_tensor > 0)
        valid_new = (new_min_depths > 0)
        valid_comparison = valid_original & valid_new
        
        if not valid_comparison.any():
            print(f"No valid pixels left for comparison after {iteration + 1} iterations")
            break
            
        # Check if pixels have moved to a significantly different depth
        pixels_moved = depth_diff[valid_comparison] > depth_tolerance
        percent_moved = float(torch.mean(pixels_moved.float())) * 100
        
        print(f"  {percent_moved:.1f}% of pixels moved to next layer")
        
        # If enough pixels have moved to the next layer, we're done
        if percent_moved > (convergence_threshold * 100):
            print(f"Convergence reached: {percent_moved:.1f}% > {convergence_threshold * 100}%")
            break
            
        # Find points that are still at the current layer depth
        # Create mask for points close to the original layer depths
        final_layer_z_expanded = final_layer_z_tensor.unsqueeze(-1)  # (H, W, 1)
        depth_close_mask = torch.abs(z_buffer_reshaped - final_layer_z_expanded) <= depth_tolerance
        valid_depth_mask = z_buffer_reshaped > 0
        
        # Combine masks to find points to remove
        points_to_remove_mask = depth_close_mask & valid_depth_mask
        
        # Check if we have any points to remove
        if not points_to_remove_mask.any():
            print("No more points to remove at current depth tolerance")
            break
            
        # Remove these points
        current_points, current_features, current_normals, removed_indices = remove_layer_from_pointcloud(
            points_to_remove_mask, idx_map, current_points, current_features, current_normals
        )
        
        removed_count = len(removed_indices)
        total_removed += removed_count
        remaining_points = current_points.shape[1]
        
        print(f"  Removed {removed_count} points (total removed: {total_removed}, remaining: {remaining_points})")
        
        # Check if we have enough points left
        if remaining_points < 1000:  # Minimum threshold
            print("Warning: Very few points remaining, stopping depth peeling")
            break
            
        # If we didn't remove many points, we might be stuck
        if removed_count < 100:
            print("Warning: Removing very few points, may be converging slowly")
    
    print(f"Depth peeling completed {iteration + 1} iterations, removed {total_removed} points")
    
    return current_points, current_features, current_normals, total_removed

def fov_to_camera_matrix(fov_x: float, fov_y: float, width: int, height: int) -> np.ndarray:
    """Convert FOV parameters to standard camera matrix format.
    
    Parameters:
        fov_x: Horizontal field of view in radians
        fov_y: Vertical field of view in radians
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Camera matrix in the format:
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    """
    # Calculate focal lengths
    fx = width / (2 * np.tan(fov_x / 2))
    fy = height / (2 * np.tan(fov_y / 2))
    
    # Principal point is at the center of the image
    cx = width / 2
    cy = height / 2
    
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])


def load_pointcloud(scene_info: SceneInfo):
    hypersim_pointcloud = BasePointCloud()
    hypersim_pointcloud.from_ply_file(
        Path(f"{scene_info.pointcloud_directory}/{scene_info.scene_name}.ply"), 
        semantic_labels_path=Path(f"{scene_info.pointcloud_directory}/{scene_info.scene_name}.npz"),
        semantic_colours_file=Path("data/semantic_label_descs.csv")
    )
    return hypersim_pointcloud


def get_final_depth_layer(
    df_camera_parameters: pd.DataFrame,
    hypersim_pointcloud: BasePointCloud,
    scene_info: SceneInfo,
    min_frame_id: int = 0,
    max_frame_id: int = 100,
    keep_classes: list[str] = ["wall", "floor", "window", "door", "ceiling"],
    device: torch.device = torch.device("cuda"),
    save_dir: Path = Path("temp"),
    resolution_scale: float = 1.0,
    first_layer_depth: bool = False,
    point_radius: float = 0.002,
    verbose: bool = False
):
    """
    Render the scene using depth peeling to extract multiple layers.

    Parameters:
        df_camera_parameters: Camera metadata DataFrame.
        hypersim_pointcloud: BasePointCloud object with scene geometry.
        scene_info: SceneInfo object.
        min_frame_id: Minimum frame ID to render.
        max_frame_id: Maximum frame ID to render.
        keep_classes: List of semantic classes to keep for filtering.
        device: PyTorch device to use for rendering.
        save_dir: Directory to save output files.
        resolution_scale: Scale factor for image resolution.
        first_layer_depth: If True, only render first depth layer without filtering.
        point_radius: Radius for point rasterization.
        
    Returns:
        bool: True if all frames processed successfully, False if any failed.
    """
    # Track whether all frames processed successfully
    all_frames_successful = True
    
    # Load point cloud from .ply file in pytorch3d
    semantic_id_map = hypersim_pointcloud.get_semantic_id_map()

    # Get scene metric scale from df_camera_parameters. Need to get the column settings_units_info_meters_scale for the row scene name
    scene_metric_scale = df_camera_parameters.loc[scene_info.scene_name, "settings_units_info_meters_scale"]

    # Filter points to keep only specified semantic classes (skip if getting first layer depth)
    if not first_layer_depth:
        hypersim_pointcloud.filter_points_by_semantic(keep_classes, keep_unlabeled=True)

    # Transform points to camera-space
    pointcloud = hypersim_pointcloud.get_points().squeeze(0).cpu().numpy()
    num_points = pointcloud.shape[0]
    P_world = np.matrix(np.c_[pointcloud, np.ones(num_points)]).T

    # Load per-scene Hypersim camera intrinsics (scaled)
    M_warp_cam_pts, fov_y, width_pixels, height_pixels = get_cam_intrinsics(
        df_camera_parameters, scene_info.scene_name, scale=resolution_scale
    )

    # Convert Hypersim camera to PyTorch3D convention
    M_hypersimCam_to_pytorch3DCam = np.matrix(np.identity(4))
    M_hypersimCam_to_pytorch3DCam[0, 0] = -1
    M_hypersimCam_to_pytorch3DCam[2, 2] = -1

    # Properly invert fov_y to get fov_x: fov_y = 2*arctan(height*tan(fov_x/2)/width)
    # Therefore: fov_x = 2*arctan(width*tan(fov_y/2)/height)
    fov_x = 2.0 * np.arctan(width_pixels * np.tan(fov_y / 2.0) / height_pixels)

    # Get camera intrinsics and scale them to match the (possibly) reduced resolution
    # Compute intrinsics from FOVs and the scaled width/height to ensure consistency
    camera_intrinsics_scaled = fov_to_camera_matrix(
        fov_x=fov_x,
        fov_y=fov_y,
        width=width_pixels,
        height=height_pixels,
    )
    # Normalize intrinsics: divide fx, cx by width; divide fy, cy by height
    moge_meta_data_normalised_intrinsics = camera_intrinsics_scaled.copy()
    moge_meta_data_normalised_intrinsics[0, 0] /= width_pixels  # fx
    moge_meta_data_normalised_intrinsics[0, 2] /= width_pixels  # cx
    moge_meta_data_normalised_intrinsics[1, 1] /= height_pixels  # fy
    moge_meta_data_normalised_intrinsics[1, 2] /= height_pixels  # cy
    # Row 2 stays [0, 0, 1]
    moge_meta_data = {"intrinsics": moge_meta_data_normalised_intrinsics.tolist()}

    for frame_id in range(min_frame_id, max_frame_id):
        try:
            if verbose:
                print(f"========== Processing frame {frame_id} ==========")
            # Create new camera and rasterizer for each frame
            cameras = FoVPerspectiveCameras(
                device=device, fov=fov_y, degrees=False, aspect_ratio=1.0, znear=1.0, zfar=400.0
            )
            raster_settings = PointsRasterizationSettings(
                image_size=(height_pixels, width_pixels),
                radius=point_radius * resolution_scale,
                points_per_pixel=150,
            )
            renderer = PointsRenderer(
                rasterizer=PointsRasterizer(
                    cameras=cameras, raster_settings=raster_settings
                ),
                compositor=AlphaCompositor(),
            )
            # Create frame-specific directory
            frame_dir = save_dir / f"{scene_info.scene_name}-{scene_info.cam_number}-{frame_id}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if frame has already been processed
            required_files = [
                "image.png",
                "first_normal.png", 
                "first_depth.png",
                "meta.json",
                "seen_mask.png",
            ]
            
            # Add layer-specific files
            if first_layer_depth:
                required_files.append("first_depth.png")
            else:
                required_files.extend(["layout_depth_0.png", "layout_layer_0_normal.png"])

            
            # Check if ground truth files need to be saved (first_normal and first_depth from HDF5)
            if not (frame_dir / "image.png").exists() or not (frame_dir / "first_normal.png").exists() or not (frame_dir / "first_depth.png").exists():
                hdf5_color_image_path = scene_info.images_dir / f"frame.{frame_id:04d}.color.hdf5"
                hdf5_normal_image_path = scene_info.geometry_dir / f"frame.{frame_id:04d}.normal_cam.hdf5"
                hdf5_first_depth = scene_info.geometry_dir / f"frame.{frame_id:04d}.depth_meters.hdf5"
                if hdf5_first_depth.exists():
                    # Load Hypersim depth (which is ray distance, not Z-depth, and convert it)
                    ray_depth_gt = load_hdf5(hdf5_first_depth)
                    
                    # Convert ray depth to Z-depth for consistency
                    # Hypersim stores ray distance, but we want Z-depth (standard)
                    z_depth_gt = ray_depth_gt.copy()
                    valid_mask = ray_depth_gt > 0
                    if valid_mask.any():
                        # Create pixel coordinate grids
                        v_coords, u_coords = np.mgrid[0:height_pixels, 0:width_pixels]
                        
                        # Get camera intrinsics
                        fx = camera_intrinsics_scaled[0, 0]
                        fy = camera_intrinsics_scaled[1, 1]
                        cx = camera_intrinsics_scaled[0, 2]
                        cy = camera_intrinsics_scaled[1, 2]
                        
                        # Compute ray direction components
                        x_dir = (u_coords - cx) / fx
                        y_dir = (v_coords - cy) / fy
                        
                        # Ray direction magnitude: sqrt(x_dir^2 + y_dir^2 + 1)
                        ray_dir_magnitude = np.sqrt(x_dir**2 + y_dir**2 + 1.0)
                        
                        # Convert ray depth to Z-depth: z = ray_depth / ray_dir_magnitude
                        z_depth_gt[valid_mask] = ray_depth_gt[valid_mask] / ray_dir_magnitude[valid_mask]
                    
                    write_depth(frame_dir / "first_depth.png", z_depth_gt)
                if hdf5_color_image_path.exists():
                    color_image = load_hdf5(hdf5_color_image_path)
                    color_image = np.clip(color_image, 0.0, 1.0)
                    plt.imsave(frame_dir / "image.png", color_image)
                if hdf5_normal_image_path.exists():
                    normal_image = load_hdf5(hdf5_normal_image_path)
                    write_normal(frame_dir / "first_normal.png", normal_image)
                create_seen_mask(
                    scene_info=scene_info,
                    frame_id=frame_id,
                    keep_classes=keep_classes,
                    semantic_id_map=semantic_id_map,
                    frame_dir=frame_dir,
                    keep_unlabeled=True
                )

            # Check if all required files exist
            all_files_exist = all((frame_dir / fname).exists() for fname in required_files)
            
            if all_files_exist:
                print(f"  Frame {frame_id} already processed, skipping...")
                continue

            # Get extrinsics
            camera_orientations, camera_positions = get_scene_extrinsics(scene_info.scene_details_dir)
            M_world2cam = get_frame_camera(camera_orientations, camera_positions, frame_id)
            if M_world2cam is None:
                print(f"Frame {frame_id} does not exist in camera data")
                continue
            # NOTE: Camera translation should match point cloud scale (already in meters)
            M_world2cam[0:3, 3] = M_world2cam[0:3, 3] * scene_metric_scale

            P_p3dcam = (
                M_hypersimCam_to_pytorch3DCam * M_warp_cam_pts * M_world2cam * P_world
            )
            # Convert point cloud to PyTorch tensors
            hypersim_pointcloud.set_points(torch.tensor(
                P_p3dcam.T[..., :3], dtype=torch.float32, device=device
            ).unsqueeze(0))  # (1, N, 3))
            points, features, normals = hypersim_pointcloud.get_points(), hypersim_pointcloud.get_features(), hypersim_pointcloud.get_normals()

            # Transform normals from world space camera space
            R_cam_from_world = np.array(M_world2cam[:3, :3])
            R_total = (R_cam_from_world).astype(np.float32)
            R_total_torch = torch.tensor(R_total, dtype=torch.float32, device=device)
            normals_cam = torch.matmul(normals, R_total_torch.T)
            normals_cam = normals_cam / (torch.norm(normals_cam, dim=2, keepdim=True) + 1e-8)

            # For first layer depth, only process one layer; otherwise do 3 layers
            num_layers = 1 if first_layer_depth else 1
            for layer in range(0, num_layers):
                # Create a Pointclouds object for rasterisation
                point_cloud = Pointclouds(points=points, features=features)

                # Rasterize to get depth info
                fragments = renderer.rasterizer(point_cloud)
                z_buffer = fragments.zbuf.squeeze().cpu().numpy()
                idx_map = fragments.idx.squeeze()
                dist_map = fragments.dists.squeeze()
                    
                # Save meta.json 
                with open(frame_dir / "meta.json", "w") as f:
                    json.dump(moge_meta_data, f)

                # Extract the closest valid depth for each pixel directly from z_buffer
                # z_buffer shape: (height, width, points_per_pixel)
                z_buffer_reshaped = z_buffer.reshape(height_pixels, width_pixels, -1)
                
                # Find the minimum positive depth for each pixel (closest valid surface)
                # Mask out invalid depths (<=0)
                valid_mask = z_buffer_reshaped > 0
                z_buffer_masked = np.where(valid_mask, z_buffer_reshaped, np.inf)
                
                # Find minimum depth for each pixel
                final_layer_z = np.min(z_buffer_masked, axis=2)
                
                # Set pixels with no valid depths to 0
                final_layer_z = np.where(final_layer_z == np.inf, 0.0, final_layer_z).astype(np.float32)

                # replace depth 0 with NaN 
                final_layer_z[final_layer_z == 0.0] = np.nan

                # Find closest points
                minimum_distance_point_on_ray, valid_points_mask = get_closest_points(
                    final_layer_z, dist_map, z_buffer, idx_map, depth_tolerance=0.015
                )

                # Need to ignore indexes which are -1
                valid_mask = minimum_distance_point_on_ray != -1
                
                # Create pointmap and normalmap with proper spatial structure
                # Initialize with NaN for invalid pixels
                height, width = minimum_distance_point_on_ray.shape
                pointmap = torch.full((height, width, 3), float('nan'), dtype=torch.float32, device=device)
                normalmap = torch.full((height, width, 3), float('nan'), dtype=torch.float32, device=device)
                
                # Fill valid pixels with actual point coordinates and normals
                valid_indices = minimum_distance_point_on_ray[valid_mask]
                pointmap[valid_mask] = points[0, valid_indices]

                normalmap[valid_mask] = normals_cam[0, valid_indices]

                # Render normal map (handles NaN values properly)
                if first_layer_depth:
                    write_normal(frame_dir / "first_normal.png", normalmap.cpu().numpy())
                else:
                    write_normal(frame_dir / f"layout_layer_{layer}_normal.png", normalmap.cpu().numpy())
                # Save the pointmap (spatial structure preserved)
                if first_layer_depth:
                    write_hdf5_from_numpy_array(frame_dir / "first_pointmap.hdf5", pointmap.cpu().numpy())
                else:
                    write_hdf5_from_numpy_array(frame_dir / f"layout_layer_{layer}_pointmap.hdf5", pointmap.cpu().numpy())

                # Save the pointmap as a depth image just taking the z-coordinate
                depth_from_pointmap = pointmap.cpu().numpy()[:, :, 2]
                if first_layer_depth:
                    write_depth(frame_dir / "first_depth.png", depth_from_pointmap)
                else:
                    write_depth(frame_dir / f"layout_depth_{layer}.png", depth_from_pointmap)

                # Use iterative depth peeling to ensure we fully remove the current layer
                if not first_layer_depth and layer != num_layers - 1:
                    points, features, normals_cam, total_removed = iterative_depth_peeling(
                        renderer, points, features, normals_cam, final_layer_z,
                        depth_tolerance=0.02, max_iterations=10, convergence_threshold=1.0
                    )
            # Clear GPU memory and old z buffers and stuff
            del z_buffer, idx_map, dist_map, final_layer_z, pointmap, normalmap, points, features, normals_cam
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR processing frame {frame_id}: {type(e).__name__}: {e}")
            print("  Skipping to next frame...")
            all_frames_successful = False
            continue
        finally:
            try:
                del z_buffer, idx_map, dist_map, final_layer_z, pointmap, normalmap, points, features, normals_cam
            except NameError:
                pass
            torch.cuda.empty_cache()
    
    return all_frames_successful

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process Hypersim point clouds to extract depth layers")
    parser.add_argument("--pointcloud-directory", required=True, type=str, 
                       help="Directory containing point cloud files")
    parser.add_argument("--scene", required=True, type=str,
                       help="Scene name to process")
    parser.add_argument("--cam", required=False, type=str,
                       help="Camera number (e.g., 00)", default="00")
    parser.add_argument("--hypersim-dir", required=True, type=str,
                       help="Root directory of Hypersim dataset")
    parser.add_argument("--save-dir", required=True, type=str,
                       help="Directory to save output files")
    parser.add_argument("--keep-classes", nargs="+", 
                       default=["wall", "floor", "window", "door", "ceiling"],
                       help="Semantic classes to keep (default: wall floor window door ceiling)")
    parser.add_argument("--device", default="cuda", type=str,
                       help="Device to use for computation (default: cuda)")
    parser.add_argument("--scale", default=1.0, type=float,
                       help="Uniform resolution scale factor (e.g., 0.5 for half-res)")
    parser.add_argument("--first-layer-depth", action="store_true",
                       help="Extract only first depth layer without semantic filtering")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output", default=False)
    parser.add_argument("--min-frame-id", default=0, type=int,
                       help="Minimum frame ID to render")
    parser.add_argument("--max-frame-id", default=100, type=int,
                       help="Maximum frame ID to render")
    parser.add_argument("--point-radius", default=0.002, type=float,
                       help="Point rasterization radius (larger = better coverage, more blur). Default: 0.0035")
    parser.add_argument("--delete-pointcloud", action="store_true",
                       help="Delete pointcloud .ply and .npz files after processing to save disk space")
    
    args = parser.parse_args()
    
    verbose = args.verbose
    # Convert string paths to Path objects
    pointcloud_directory = Path(args.pointcloud_directory)
    hypersim_directory = Path(args.hypersim_dir)
    save_dir = Path(args.save_dir)
    scene = args.scene
    
    # Define metadata CSV path (you may want to make this configurable too)
    metadata_csv = Path("data/metadata_camera_parameters.csv")
    
    # Set up device
    device = torch.device(args.device)
    
    # Create scene info
    scene_name = scene.strip()
    # remove trailing slash if present
    if scene_name.endswith('/'):
        scene_name = scene_name[:-1]

    scene_directory = Path(f"{hypersim_directory}/{scene_name}")
    _detail_directory = scene_directory / "_detail"
    # Go in scene_directory and find all the cam_* directories in _detail folder
    cam_directories = [d for d in _detail_directory.glob("cam_*") if d.is_dir()]
    cam_directories = sorted(cam_directories)
    cam_directories = cam_directories[:5] # Only look at the first 5 cameras, due to memory constraints in pointcloud generation
    all_cameras_successful = True
    for cam_directory in cam_directories:
        if verbose:
            print(f"Processing scene {scene_name} with camera {cam_directory.name}")
        scene_info = SceneInfo(scene_name, pointcloud_directory, cam_directory.name, scene_directory)
        
        # Load camera parameters
        df_camera_parameters = get_metadata_camera_parameters(metadata_csv)
        
        hypersim_pointcloud = load_pointcloud(scene_info)

        camera_success = get_final_depth_layer(
            df_camera_parameters,
            hypersim_pointcloud,
            scene_info,
            min_frame_id=args.min_frame_id,
            max_frame_id=args.max_frame_id,
            device=device,
            keep_classes=args.keep_classes,
            save_dir=save_dir,
            resolution_scale=args.scale,
            first_layer_depth=args.first_layer_depth,
            point_radius=args.point_radius,
            verbose=verbose
        )
        
        if not camera_success:
            all_cameras_successful = False
    
    # Delete pointcloud files if requested and all frames processed successfully
    if args.delete_pointcloud:
        if all_cameras_successful:
            ply_file = pointcloud_directory / f"{scene_name}.ply"
            npz_file = pointcloud_directory / f"{scene_name}.npz"
            
            if ply_file.exists():
                ply_file.unlink()
                print(f"Deleted pointcloud file: {ply_file}")
            
            if npz_file.exists():
                npz_file.unlink()
                print(f"Deleted semantic labels file: {npz_file}")
        else:
            print("Skipping pointcloud deletion: Some frames failed to process")
    
    if all_cameras_successful:
        print(f"Processing complete. All frames successfully rendered. Results saved to {save_dir}")
    else:
        print(f"Processing complete with errors. Some frames failed. Results saved to {save_dir}")

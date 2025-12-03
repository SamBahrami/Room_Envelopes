import h5py
import numpy as np
from pathlib import Path

def write_hdf5_from_numpy_array(filepath: Path, array: np.ndarray):
    """Write a numpy array to an HDF5 file with compression.
    
    The compression_opts parameter controls the compression level:
    - 0 = no compression
    - 1 = fastest compression
    - 9 = best compression (slower)
    """
    with h5py.File(filepath, "w") as f:
        f.create_dataset("dataset", data=array, compression="gzip", compression_opts=9)

def load_hdf5(filepath: Path) -> np.ndarray:
    """Load and return the contents of an HDF5 file from the 'dataset' group."""
    with h5py.File(filepath, "r") as f:
        data = np.array(f["dataset"])
    return data

def load_pointcloud_hdf5(filepath: Path) -> np.ndarray:
    """Load point cloud data from an HDF5 file and reshape it to (-1, 3)."""
    data = load_hdf5(filepath).astype(np.float32)
    return data.reshape(-1, 3)

def load_rgb_image_hdf5(filepath: Path) -> np.ndarray:
    """Load RGB image data from an HDF5 file and reshape it to (-1, 3)."""
    data = load_hdf5(filepath)
    return data.reshape(-1, 3)

def load_semantic_hdf5(filepath: Path) -> np.ndarray:
    """Load semantic data from an HDF5 file"""
    data = load_hdf5(filepath)
    return data.reshape(-1, 1)
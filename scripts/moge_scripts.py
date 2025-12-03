"""The functions in this file come from the MoGe project https://github.com/microsoft/moge
We output files that are compatible with the MoGe project.
"""
import os
import numpy as np
from PIL import Image, PngImagePlugin
from typing import Union, IO
from pathlib import Path
import pandas as pd
import json
import io

# Force OpenCV to run in headless mode (fixes libGL.so.1 issue)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2

def get_metadata_camera_parameters(metadata_csv: Path) -> pd.DataFrame:
    """Load camera parameters metadata from a CSV file."""
    return pd.read_csv(metadata_csv, index_col="scene_name")


def write_normal(path: Union[str, os.PathLike, IO], normal: np.ndarray, compression_level: int = 7) -> np.ndarray:
    """
    Write a normal image, input float32 normal array of shape (H, W, 3).
    """
    mask_nan = np.isnan(normal).any(axis=-1)
    normal = ((normal * [0.5, 0.5, 0.5] + 0.5).clip(0, 1) * 65535).astype(np.uint16)
    normal[mask_nan] = 0
    data = cv2.imencode('.png', cv2.cvtColor(normal, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, compression_level])[1].tobytes()
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_bytes(data)
    else:
        path.write(data)

def read_normal(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a normal image, return float32 normal array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    normal = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    mask_nan = np.all(normal == 0, axis=-1)
    normal = (normal.astype(np.float32) / 65535 - 0.5) * [2.0, -2.0, -2.0]
    norm = np.sqrt(np.square(normal[..., 0]) + np.square(normal[..., 1]) + np.square(normal[..., 2])) + 1e-12
    normal = normal / norm[..., None]  # Add dimension for proper broadcasting
    normal[mask_nan] = np.nan
    return normal

def read_depth(path: Union[str, os.PathLike, IO]) -> tuple[np.ndarray, float]:
    """
    Read a depth image, return float32 depth array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    unit = float(pil_image.info.get('unit')) if 'unit' in pil_image.info else None
    depth = np.array(pil_image)
    mask_nan, mask_inf = depth == 0, depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth, unit


def write_depth(
    path: Union[str, os.PathLike, IO], 
    depth: np.ndarray, 
    unit: float = None,
    max_range: float = 1e5,
    compression_level: int = 7,
):
    """
    Encode and write a depth image as 16-bit PNG format.
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `depth: np.ndarray`
        The depth array, float32 array of shape (H, W). 
        May contain `NaN` for invalid values and `Inf` for infinite values.
    - `unit: float = None`
        The unit of the depth values.
    
    Depth values are encoded as follows:
    - 0: unknown
    - 1 ~ 65534: depth values in logarithmic
    - 65535: infinity
    
    metadata is stored in the PNG file as text fields:
    - `near`: the minimum depth value
    - `far`: the maximum depth value
    - `unit`: the unit of the depth values (optional)
    """
    mask_values, mask_nan, mask_inf = np.isfinite(depth), np.isnan(depth), np.isinf(depth)

    depth = depth.astype(np.float32)
    mask_finite = depth
    near = max(depth[mask_values].min(), 1e-5)
    far = max(near * 1.1, min(depth[mask_values].max(), near * max_range))
    depth = 1 + np.round((np.log(np.nan_to_num(depth, nan=0).clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    depth[mask_nan] = 0
    depth[mask_inf] = 65535

    pil_image = Image.fromarray(depth)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('near', str(near))
    pnginfo.add_text('far', str(far))
    if unit is not None:
        pnginfo.add_text('unit', str(unit))
    pil_image.save(path, pnginfo=pnginfo, compress_level=compression_level)

def read_meta(path: Union[str, os.PathLike, IO]) -> dict[str, any]:
    return json.loads(Path(path).read_text())
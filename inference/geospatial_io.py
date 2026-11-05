"""
inference/geospatial_io.py
--------------------------
Handles I/O for geospatial GeoTIFFs and standard PNG/JPEG images.
Preserves CRS, geotransform, nodata, and band metadata when present.
Creates calibrated RGB previews for interactive display.
"""

import os
import io
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from typing import Dict, Any, Tuple, Optional
import rasterio
from rasterio.transform import Affine

from model_registry import CLASS_MAPPING

class ImageInputBundle:
    def __init__(
        self,
        raw_array: np.ndarray,
        rgb_preview: np.ndarray,
        channels: int,
        height: int,
        width: int,
        dtype: str,
        file_format: str,
        is_georeferenced: bool = False,
        crs: Optional[str] = None,
        transform: Optional[Affine] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        nodata: Optional[float] = None,
        filename: str = "uploaded_scene"
    ):
        self.raw_array = raw_array # Shape: (C, H, W)
        self.rgb_preview = rgb_preview # Shape: (H, W, 3), dtype uint8
        self.channels = channels
        self.height = height
        self.width = width
        self.dtype = dtype
        self.file_format = file_format
        self.is_georeferenced = is_georeferenced
        self.crs = crs
        self.transform = transform
        self.bounds = bounds
        self.resolution = resolution
        self.nodata = nodata
        self.filename = filename

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "format": self.file_format.upper(),
            "dimensions": f"{self.width} × {self.height} px",
            "channels": self.channels,
            "data_type": str(self.dtype),
            "is_georeferenced": self.is_georeferenced,
            "crs": str(self.crs) if self.crs else "Non-georeferenced (Cartesian)",
            "resolution": f"{abs(self.resolution[0]):.2f} m × {abs(self.resolution[1]):.2f} m" if self.resolution else "Standard Image Grid",
            "bounds": [round(b, 5) for b in self.bounds] if self.bounds else None
        }

def stretch_8bit(band: np.ndarray, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
    """Robust 2%-98% percentile linear stretch to 8-bit [0, 255]."""
    valid = band[band > 0] if np.any(band > 0) else band
    if len(valid) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    vmin = float(np.percentile(valid, p_min))
    vmax = float(np.percentile(valid, p_max))
    if vmax <= vmin:
        vmax = vmin + 1.0
    stretched = np.clip((band - vmin) / (vmax - vmin) * 255.0, 0, 255)
    return stretched.astype(np.uint8)

def load_image_from_bytes(file_bytes: bytes, filename: str) -> ImageInputBundle:
    """
    Ingest image from raw bytes (uploaded file).
    Supports GeoTIFF/TIFF, PNG, JPEG, WebP, BMP, JP2, and ZIP archives containing imagery.
    """
    import zipfile
    ext = os.path.splitext(filename)[1].lower()

    # 1. Handle ZIP archive uploads
    if ext == ".zip":
        with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as z:
            names = z.namelist()
            # Prioritize TIFF inside zip, then PNG, JPG
            candidate = None
            for n in names:
                if n.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    candidate = n
                    break
            if candidate:
                inner_bytes = z.read(candidate)
                return load_image_from_bytes(inner_bytes, candidate)
            else:
                raise ValueError(f"Zip archive '{filename}' does not contain a supported image file (.tif, .png, .jpg).")

    is_tiff = ext in [".tif", ".tiff"]

    if is_tiff:
        try:
            with rasterio.open(io.BytesIO(file_bytes)) as ds:
                raw_data = ds.read() # Shape: (C, H, W)
                channels, height, width = raw_data.shape
                crs = str(ds.crs) if ds.crs else None
                transform = ds.transform
                bounds = ds.bounds
                res = ds.res
                nodata = ds.nodata
                is_geo = ds.crs is not None

                # Generate accurate True-Color RGB Preview based on sensor bands
                if channels == 8:
                    # WorldView-3: Band 5 is Red, Band 3 is Green, Band 2 is Blue
                    r_preview = stretch_8bit(raw_data[4])
                    g_preview = stretch_8bit(raw_data[2])
                    b_preview = stretch_8bit(raw_data[1])
                    rgb_preview = np.stack([r_preview, g_preview, b_preview], axis=-1)
                elif channels == 4:
                    # Sentinel-2: Band 3 is Red, Band 2 is Green, Band 1 is Blue
                    r_preview = stretch_8bit(raw_data[2])
                    g_preview = stretch_8bit(raw_data[1])
                    b_preview = stretch_8bit(raw_data[0])
                    rgb_preview = np.stack([r_preview, g_preview, b_preview], axis=-1)
                elif channels == 3:
                    # Standard 3-channel RGB GeoTIFF
                    r_preview = stretch_8bit(raw_data[0])
                    g_preview = stretch_8bit(raw_data[1])
                    b_preview = stretch_8bit(raw_data[2])
                    rgb_preview = np.stack([r_preview, g_preview, b_preview], axis=-1)
                elif channels == 1:
                    # Panchromatic 1-channel GeoTIFF
                    gray = stretch_8bit(raw_data[0])
                    rgb_preview = np.stack([gray, gray, gray], axis=-1)
                else:
                    # Arbitrary multi-band fallback
                    ch0 = stretch_8bit(raw_data[0])
                    rgb_preview = np.stack([ch0, ch0, ch0], axis=-1)

                return ImageInputBundle(
                    raw_array=raw_data.astype(np.float32),
                    rgb_preview=rgb_preview,
                    channels=channels,
                    height=height,
                    width=width,
                    dtype=str(raw_data.dtype),
                    file_format="geotiff",
                    is_georeferenced=is_geo,
                    crs=crs,
                    transform=transform,
                    bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top) if bounds else None,
                    resolution=res if is_geo else None,
                    nodata=nodata,
                    filename=filename
                )
        except Exception as e:
            # Fallback if rasterio encounters memory stream error
            pass

    # Standard PIL reader for PNG, JPEG, or uncompressed TIFF
    pil_img = Image.open(io.BytesIO(file_bytes))
    mode = pil_img.mode

    if mode == "L":
        # Grayscale
        gray_arr = np.array(pil_img, dtype=np.uint8)
        h, w = gray_arr.shape
        raw_arr = np.expand_dims(gray_arr.astype(np.float32), axis=0) # (1, H, W)
        rgb_preview = np.stack([gray_arr, gray_arr, gray_arr], axis=-1)
        channels = 1
    elif mode == "RGBA":
        # RGBA with transparency
        rgba_arr = np.array(pil_img, dtype=np.uint8)
        h, w, _ = rgba_arr.shape
        # Composite transparent background onto neutral dark slate background for display
        alpha = rgba_arr[:, :, 3].astype(np.float32) / 255.0
        r = rgba_arr[:, :, 0].astype(np.float32) * alpha + 18.0 * (1.0 - alpha)
        g = rgba_arr[:, :, 1].astype(np.float32) * alpha + 24.0 * (1.0 - alpha)
        b = rgba_arr[:, :, 2].astype(np.float32) * alpha + 38.0 * (1.0 - alpha)
        rgb_preview = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
        # Store as 4-channel raw array (R, G, B, A)
        raw_arr = np.transpose(rgba_arr.astype(np.float32), (2, 0, 1))
        channels = 4
    elif mode == "RGB":
        rgb_arr = np.array(pil_img, dtype=np.uint8)
        h, w, _ = rgb_arr.shape
        rgb_preview = rgb_arr
        raw_arr = np.transpose(rgb_arr.astype(np.float32), (2, 0, 1)) # (3, H, W)
        channels = 3
    else:
        # Convert any other exotic mode (P, CMYK, etc.) to RGB
        conv = pil_img.convert("RGB")
        rgb_arr = np.array(conv, dtype=np.uint8)
        h, w, _ = rgb_arr.shape
        rgb_preview = rgb_arr
        raw_arr = np.transpose(rgb_arr.astype(np.float32), (2, 0, 1))
        channels = 3

    return ImageInputBundle(
        raw_array=raw_arr,
        rgb_preview=rgb_preview,
        channels=channels,
        height=h,
        width=w,
        dtype="uint8",
        file_format="png" if ext == ".png" else "jpeg",
        is_georeferenced=False,
        crs=None,
        transform=None,
        bounds=None,
        resolution=None,
        nodata=None,
        filename=filename
    )

def create_color_mask(pred_mask: np.ndarray) -> np.ndarray:
    """
    Converts 2D integer class mask {0, 1, 2, 3, 4, 255} to 3D RGB uint8 image
    using the verified project color scheme.
    """
    h, w = pred_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, info in CLASS_MAPPING.items():
        match_px = (pred_mask == class_id)
        if np.any(match_px):
            colored[match_px] = info["rgb"]

    # Any unmapped remaining pixel defaults to ignore color
    unmapped = ~np.isin(pred_mask, list(CLASS_MAPPING.keys()))
    if np.any(unmapped):
        colored[unmapped] = CLASS_MAPPING[255]["rgb"]

    return colored

def create_blended_overlay(rgb_image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Alpha-blends the colored segmentation mask over the RGB image.
    """
    alpha = max(0.0, min(1.0, alpha))
    blended = (1.0 - alpha) * rgb_image.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)

def export_geotiff(pred_mask: np.ndarray, out_path: str, crs: str, transform: Affine, nodata: int = 255):
    """
    Exports the prediction mask as a properly georeferenced GeoTIFF on disk.
    """
    h, w = pred_mask.shape
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate"
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(pred_mask.astype(np.uint8), 1)

def export_prediction_geotiff(pred_mask: np.ndarray, transform: Affine, crs: str, nodata: int = 255) -> bytes:
    """
    Exports the prediction mask as in-memory GeoTIFF bytes for web download.
    """
    h, w = pred_mask.shape
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate"
    }
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**meta) as dst:
            dst.write(pred_mask.astype(np.uint8), 1)
        return memfile.read()

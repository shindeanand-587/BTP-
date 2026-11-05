"""
inference/geospatial_io.py
--------------------------
Handles I/O for geospatial GeoTIFFs and standard PNG/JPEG images.
Preserves CRS, geotransform, nodata, and band metadata when present.
Creates calibrated RGB previews for interactive display.
"""

import os
import io
import json
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from typing import Dict, Any, Tuple, Optional
import rasterio
from rasterio.transform import Affine

try:
    import shapefile
except ImportError:
    shapefile = None

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

def load_shapefile_bundle(file_bytes: bytes, filename: str) -> ImageInputBundle:
    """
    Ingests ESRI Shapefiles (.shp, .zip containing shapefile, or .geojson),
    rendering vector geometry (roads, building parcels, water bodies) into a georeferenced canvas.
    """
    if shapefile is None:
        raise ImportError("The 'pyshp' package is required to read Shapefiles. Please run: pip install pyshp")

    import zipfile
    ext = os.path.splitext(filename)[1].lower()

    reader_kwargs = {}
    crs_str = "WGS 84 / Projected Vector CRS"

    if ext == ".zip":
        with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as z:
            shp_names = [n for n in z.namelist() if n.lower().endswith('.shp')]
            if not shp_names:
                raise ValueError("Zip archive does not contain any .shp file.")
            target_shp = shp_names[0]
            base_name = os.path.splitext(target_shp)[0]
            reader_kwargs['shp'] = io.BytesIO(z.read(target_shp))
            
            for item in z.namelist():
                ilow = item.lower()
                if ilow.endswith('.shx') and (base_name.lower() in ilow or len(shp_names) == 1):
                    reader_kwargs['shx'] = io.BytesIO(z.read(item))
                elif ilow.endswith('.dbf') and (base_name.lower() in ilow or len(shp_names) == 1):
                    reader_kwargs['dbf'] = io.BytesIO(z.read(item))
                elif ilow.endswith('.prj') and (base_name.lower() in ilow or len(shp_names) == 1):
                    try:
                        crs_str = z.read(item).decode('utf-8', errors='ignore').strip()
                    except Exception:
                        pass
    elif ext in [".geojson", ".json"]:
        return load_geojson_bundle(file_bytes, filename)
    else:
        reader_kwargs['shp'] = io.BytesIO(file_bytes)

    r = shapefile.Reader(**reader_kwargs)
    shapes = r.shapes()
    if not shapes:
        raise ValueError(f"Shapefile '{filename}' contains 0 geometric features.")

    xmin, ymin, xmax, ymax = r.bbox
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)

    w_px = 1024
    h_px = int(np.clip(round(w_px * (dy / dx)), 256, 1536))

    img = Image.new("RGB", (w_px, h_px), (15, 23, 42))
    draw = ImageDraw.Draw(img)

    vector_mask_img = Image.new("L", (w_px, h_px), 2) # Base surface: Class 2 (Barren Land)
    draw_mask = ImageDraw.Draw(vector_mask_img)

    centerline_img = Image.new("1", (w_px, h_px), 0)
    draw_skel = ImageDraw.Draw(centerline_img)

    fn_low = filename.lower()
    is_road = any(w in fn_low for w in ["road", "street", "highway", "avenue", "traffic", "lane", "network"])
    is_built = any(w in fn_low for w in ["built", "structure", "building", "parcel", "house", "footprint"])
    is_water = any(w in fn_low for w in ["water", "creek", "river", "bay", "ocean", "sea", "lake", "drain"])

    import math
    tot_len_m = 0.0
    vector_polylines = []

    for s in shapes:
        pts = s.points
        if not pts:
            continue
        pix_pts = [
            (int(round((x - xmin) / dx * (w_px - 1))),
             int(round((ymax - y) / dy * (h_px - 1))))
            for x, y in pts
        ]
        is_poly = s.shapeType in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM)
        
        if is_road:
            if is_poly:
                draw_mask.polygon(pix_pts, fill=4)
                draw.polygon(pix_pts, fill=(250, 204, 21), outline=(255, 255, 255))
            else:
                draw_mask.line(pix_pts, fill=4, width=6)
                draw_skel.line(pix_pts, fill=1, width=1)
                draw.line(pix_pts, fill=(255, 255, 255), width=6)
                draw.line(pix_pts, fill=(250, 204, 21), width=4)
                vector_polylines.append(pix_pts)
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i+1]
                if dx > 1000:
                    tot_len_m += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                else:
                    mid_lat = math.radians((p1[1] + p2[1]) / 2.0)
                    deg_x = (p2[0] - p1[0]) * 111_000.0 * math.cos(mid_lat)
                    deg_y = (p2[1] - p1[1]) * 111_000.0
                    tot_len_m += math.hypot(deg_x, deg_y)
        elif is_built:
            if is_poly:
                draw_mask.polygon(pix_pts, fill=3)
                draw.polygon(pix_pts, fill=(225, 29, 72), outline=(255, 255, 255))
            else:
                draw_mask.line(pix_pts, fill=3, width=5)
                draw.line(pix_pts, fill=(225, 29, 72), width=3)
        elif is_water:
            if is_poly:
                draw_mask.polygon(pix_pts, fill=0)
                draw.polygon(pix_pts, fill=(8, 102, 255), outline=(147, 197, 253))
            else:
                draw_mask.line(pix_pts, fill=0, width=5)
                draw.line(pix_pts, fill=(8, 102, 255), width=5)
        else:
            if is_poly:
                draw_mask.polygon(pix_pts, fill=4)
                draw.polygon(pix_pts, fill=(6, 182, 212), outline=(255, 255, 255))
            else:
                draw_mask.line(pix_pts, fill=4, width=5)
                draw_skel.line(pix_pts, fill=1, width=1)
                draw.line(pix_pts, fill=(250, 204, 21), width=3)
                vector_polylines.append(pix_pts)
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i+1]
                if dx > 1000:
                    tot_len_m += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                else:
                    mid_lat = math.radians((p1[1] + p2[1]) / 2.0)
                    deg_x = (p2[0] - p1[0]) * 111_000.0 * math.cos(mid_lat)
                    deg_y = (p2[1] - p1[1]) * 111_000.0
                    tot_len_m += math.hypot(deg_x, deg_y)

    rgb_preview = np.array(img, dtype=np.uint8)
    raw_array = np.transpose(rgb_preview.astype(np.float32), (2, 0, 1))

    if dx > 1000:
        res_x = dx / w_px
        res_y = dy / h_px
    else:
        res_x = (dx * 111_000.0) / w_px
        res_y = (dy * 111_000.0) / h_px

    bundle = ImageInputBundle(
        raw_array=raw_array,
        rgb_preview=rgb_preview,
        channels=3,
        height=h_px,
        width=w_px,
        dtype="uint8",
        file_format="ESRI Shapefile Vector (.shp)",
        is_georeferenced=True,
        crs=crs_str[:120] if crs_str else "Project Vector CRS",
        bounds=(xmin, ymin, xmax, ymax),
        resolution=(float(res_x), float(res_y)),
        filename=filename
    )
    bundle.is_vector = True
    bundle.vector_mask = np.array(vector_mask_img, dtype=np.uint8)
    bundle.vector_centerlines = np.array(centerline_img, dtype=bool)
    bundle.vector_polylines = vector_polylines
    bundle.total_vector_length_km = tot_len_m / 1000.0
    bundle.vector_feature_count = len(shapes)
    bundle.vector_shape_type = getattr(r, "shapeTypeName", "Vector")
    bundle.vector_domain = "roads" if is_road else ("builtup" if is_built else ("water" if is_water else "mixed"))
    return bundle

def load_geojson_bundle(file_bytes: bytes, filename: str) -> ImageInputBundle:
    """Ingests GeoJSON vector files and renders them onto a georeferenced canvas."""
    data = json.loads(file_bytes.decode('utf-8'))
    features = data.get('features', [data] if 'geometry' in data else [])
    if not features:
        raise ValueError("GeoJSON contains 0 features.")

    all_coords = []
    for f in features:
        geom = f.get('geometry', {})
        coords = geom.get('coordinates', [])
        def flatten(c):
            if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
                all_coords.append((float(c[0]), float(c[1])))
            elif isinstance(c, (list, tuple)):
                for sub in c: flatten(sub)
        flatten(coords)

    if not all_coords:
        raise ValueError("GeoJSON contains no valid coordinates.")

    xs = [pt[0] for pt in all_coords]
    ys = [pt[1] for pt in all_coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)

    w_px = 1024
    h_px = int(np.clip(round(w_px * (dy / dx)), 256, 1536))

    img = Image.new("RGB", (w_px, h_px), (15, 23, 42))
    draw = ImageDraw.Draw(img)

    vector_mask_img = Image.new("L", (w_px, h_px), 2)
    draw_mask = ImageDraw.Draw(vector_mask_img)

    centerline_img = Image.new("1", (w_px, h_px), 0)
    draw_skel = ImageDraw.Draw(centerline_img)

    import math
    tot_len_m = 0.0
    vector_polylines = []

    fn_low = filename.lower()
    is_road = any(w in fn_low for w in ["road", "street", "highway", "avenue", "traffic", "lane", "network"])
    is_built = any(w in fn_low for w in ["built", "structure", "building", "parcel", "house", "footprint"])

    for f in features:
        geom = f.get('geometry', {})
        gtype = geom.get('type', '')
        coords = geom.get('coordinates', [])
        
        def to_pixels(pts_list):
            return [(int(round((x - xmin) / dx * (w_px - 1))),
                     int(round((ymax - y) / dy * (h_px - 1)))) for x, y in pts_list]

        if gtype == 'Polygon':
            c_id = 3 if is_built else (4 if is_road else 3)
            for ring in coords:
                pix_ring = to_pixels(ring)
                draw_mask.polygon(pix_ring, fill=c_id)
                draw.polygon(pix_ring, fill=(250, 204, 21) if is_road else (225, 29, 72), outline=(255, 255, 255))
        elif gtype == 'MultiPolygon':
            c_id = 3 if is_built else (4 if is_road else 3)
            for poly in coords:
                for ring in poly:
                    pix_ring = to_pixels(ring)
                    draw_mask.polygon(pix_ring, fill=c_id)
                    draw.polygon(pix_ring, fill=(250, 204, 21) if is_road else (225, 29, 72), outline=(255, 255, 255))
        elif gtype in ('LineString', 'MultiLineString'):
            lines = coords if gtype == 'MultiLineString' else [coords]
            for line in lines:
                pix_line = to_pixels(line)
                draw_mask.line(pix_line, fill=4, width=6)
                draw_skel.line(pix_line, fill=1, width=1)
                draw.line(pix_line, fill=(255, 255, 255), width=6)
                draw.line(pix_line, fill=(250, 204, 21), width=4)
                vector_polylines.append(pix_line)
                for i in range(len(line) - 1):
                    p1, p2 = line[i], line[i+1]
                    if dx > 1000:
                        tot_len_m += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    else:
                        mid_lat = math.radians((p1[1] + p2[1]) / 2.0)
                        deg_x = (p2[0] - p1[0]) * 111_000.0 * math.cos(mid_lat)
                        deg_y = (p2[1] - p1[1]) * 111_000.0
                        tot_len_m += math.hypot(deg_x, deg_y)
        elif gtype == 'Point':
            pt = to_pixels([coords])[0]
            draw.ellipse([(pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4)], fill=(37, 99, 235), outline=(255, 255, 255))

    rgb_preview = np.array(img, dtype=np.uint8)
    raw_array = np.transpose(rgb_preview.astype(np.float32), (2, 0, 1))

    res_linear = (dx * 111_000.0) / w_px if dx < 1000 else (dx / w_px)

    bundle = ImageInputBundle(
        raw_array=raw_array,
        rgb_preview=rgb_preview,
        channels=3,
        height=h_px,
        width=w_px,
        dtype="uint8",
        file_format="GeoJSON Vector (.geojson)",
        is_georeferenced=True,
        crs="EPSG:4326 (WGS 84)",
        bounds=(xmin, ymin, xmax, ymax),
        resolution=(float(res_linear), float(res_linear)),
        filename=filename
    )
    bundle.is_vector = True
    bundle.vector_mask = np.array(vector_mask_img, dtype=np.uint8)
    bundle.vector_centerlines = np.array(centerline_img, dtype=bool)
    bundle.vector_polylines = vector_polylines
    bundle.total_vector_length_km = tot_len_m / 1000.0
    bundle.vector_feature_count = len(features)
    bundle.vector_domain = "builtup" if is_built else "roads"
    return bundle

def load_image_from_bytes(file_bytes: bytes, filename: str) -> ImageInputBundle:
    """
    Ingest image or vector shapefile from raw bytes (uploaded file).
    Supports GeoTIFF/TIFF, PNG, JPEG, WebP, BMP, ESRI Shapefiles (.shp, .zip), and GeoJSON.
    """
    import zipfile
    ext = os.path.splitext(filename)[1].lower()

    # 1. Handle Shapefile and GeoJSON uploads directly
    if ext in [".shp", ".dbf", ".geojson", ".json"]:
        return load_shapefile_bundle(file_bytes, filename)

    # 2. Handle ZIP archive uploads (supports both zipped shapefiles and zipped imagery)
    if ext == ".zip":
        with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as z:
            names = z.namelist()
            # If zip contains a shapefile, load it as vector bundle
            if any(n.lower().endswith('.shp') for n in names):
                return load_shapefile_bundle(file_bytes, filename)
            # Otherwise search for imagery
            candidate = None
            for n in names:
                if n.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    candidate = n
                    break
            if candidate:
                inner_bytes = z.read(candidate)
                return load_image_from_bytes(inner_bytes, candidate)
            else:
                raise ValueError(f"Zip archive '{filename}' does not contain a supported image or shapefile.")

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

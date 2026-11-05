"""
inference/postprocessing.py
---------------------------
Postprocessing routines for class distribution calculation,
physical area estimation, and confidence map visualization.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from typing import Dict, Any, Optional, Tuple
from model_registry import CLASS_MAPPING

def compute_class_distribution(
    pred_mask: np.ndarray,
    resolution: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Computes class-wise pixel distribution and optional physical area estimates.
    Distinguishes pixel counts from physical area.
    """
    total_pixels = pred_mask.size
    nodata_count = int(np.sum(pred_mask == 255))
    valid_pixels = total_pixels - nodata_count
    
    pixel_area_m2 = None
    if resolution is not None:
        rx, ry = abs(resolution[0]), abs(resolution[1])
        # Only compute area if resolution appears in meters (typically >= 0.1 and <= 100)
        if 0.1 <= rx <= 100.0 and 0.1 <= ry <= 100.0:
            pixel_area_m2 = rx * ry

    classes_stats = []
    
    for class_id in range(5):
        info = CLASS_MAPPING[class_id]
        count = int(np.sum(pred_mask == class_id))
        pct_valid = (count / valid_pixels * 100.0) if valid_pixels > 0 else 0.0
        pct_total = (count / total_pixels * 100.0) if total_pixels > 0 else 0.0

        item = {
            "class_id": class_id,
            "name": info["name"],
            "color": info["color"],
            "rgb": info["rgb"],
            "pixel_count": count,
            "pct_valid": round(pct_valid, 2),
            "pct_total": round(pct_total, 2)
        }

        if pixel_area_m2 is not None:
            area_m2 = count * pixel_area_m2
            item["area_km2"] = round(area_m2 / 1_000_000.0, 4)
            item["area_ha"] = round(area_m2 / 10_000.0, 2)
            item["area_hectares"] = item["area_ha"]
        else:
            item["area_km2"] = None
            item["area_ha"] = None
            item["area_hectares"] = None

        classes_stats.append(item)

    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nodata_pixels": nodata_count,
        "nodata_pct": round((nodata_count / total_pixels * 100.0) if total_pixels > 0 else 0.0, 2),
        "has_physical_area": pixel_area_m2 is not None,
        "pixel_area_m2": pixel_area_m2,
        "classes": classes_stats
    }

def render_confidence_map(conf_map: np.ndarray) -> np.ndarray:
    """
    Renders 2D float array in [0, 1] as an RGB heatmap (Turbo colormap).
    Compatible with modern Matplotlib (3.8 - 3.11+) and legacy versions.
    """
    conf_norm = np.clip(conf_map, 0.0, 1.0)
    try:
        cmap = mpl.colormaps["turbo"]
    except (AttributeError, KeyError):
        try:
            cmap = cm.get_cmap("turbo")
        except AttributeError:
            cmap = getattr(cm, "turbo", None)
    colored = cmap(conf_norm)[:, :, :3] # Take RGB, drop alpha
    return (colored * 255.0).astype(np.uint8)

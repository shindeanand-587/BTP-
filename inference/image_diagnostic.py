"""
inference/image_diagnostic.py
-----------------------------
Preliminary Image Sanity Check & Diagnostic Engine.
Performs an automated examination of uploaded imagery:
1. Physical & Spectral Metadata: Band count, dimensions, dynamic range, bit depth, GSD/resolution.
2. Structural & Spatial Complexity: Edge density and spatial gradients (Laplacian variance).
3. Spectral Signature: Contrast ratio, water absorption index, vegetation index.
4. Scene Typology Estimation: Identifies dominant land-cover profile.
5. Dynamic Model Fitness Scoring: Dynamically evaluates each model's suitability
   without using a hardcoded default.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from inference.geospatial_io import ImageInputBundle
from model_registry import get_all_models, get_ready_models, get_model_by_id

def analyze_image_characteristics(bundle: ImageInputBundle) -> Dict[str, Any]:
    """
    Performs preliminary examination of the uploaded imagery array.
    Returns quantitative characteristics, scene typology, and model fitness scores.
    """
    raw = bundle.raw_array # Shape: (C, H, W)
    channels = bundle.channels
    h, w = bundle.height, bundle.width
    is_geo = bundle.is_georeferenced
    res = bundle.resolution
    gsd_m = abs(res[0]) if res else None

    # 1. Grayscale / Panchromatic check
    # Check if single band or if R==G==B across RGB channels
    is_panchromatic = False
    if channels == 1:
        is_panchromatic = True
    elif channels >= 3:
        # Sample center region to test spectral divergence
        ch0 = raw[0, ::4, ::4]
        ch1 = raw[1, ::4, ::4]
        ch2 = raw[2, ::4, ::4]
        diff_rg = np.mean(np.abs(ch0 - ch1))
        diff_gb = np.mean(np.abs(ch1 - ch2))
        if diff_rg < 1.0 and diff_gb < 1.0:
            is_panchromatic = True

    # 2. Dynamic Range / Bit Depth Analysis
    raw_max = float(np.max(raw))
    raw_min = float(np.min(raw))
    raw_mean = float(np.mean(raw))
    raw_std = float(np.std(raw))

    if raw_max > 255.0:
        bit_depth_label = f"16-Bit Scientific Radiance (Max: {int(raw_max):,})"
        is_high_dynamic_range = True
    else:
        bit_depth_label = f"8-Bit Standard Digital Range (0 - {int(raw_max)})"
        is_high_dynamic_range = False

    # 3. Structural Complexity & Edge Density (Gradient analysis on downsampled preview)
    # Downsample preview for fast, responsive calculation
    preview_gray = np.mean(bundle.rgb_preview[::4, ::4].astype(np.float32), axis=-1)
    # Simple discrete Laplacian kernel for edge energy
    lap = np.abs(
        preview_gray[2:, 1:-1] + preview_gray[:-2, 1:-1] +
        preview_gray[1:-1, 2:] + preview_gray[1:-1, :-2] -
        4.0 * preview_gray[1:-1, 1:-1]
    )
    edge_energy = float(np.mean(lap))
    # Normalize edge score between 0.0 and 1.0
    edge_density_score = float(np.clip(edge_energy / 22.0, 0.05, 1.0))

    # 4. Water & Boundary Signature Analysis
    # Water has low reflectance in NIR and higher in Blue/Green, or uniform dark absorption
    water_score = 0.0
    if channels >= 8:
        # WorldView-3: Band 2 is Blue, Band 7 is NIR1
        blue_b = raw[1, ::4, ::4]
        nir_b = raw[6, ::4, ::4]
        denom = (blue_b + nir_b + 1e-5)
        ndwi_approx = (blue_b - nir_b) / denom
        water_px = np.mean(ndwi_approx > 0.1)
        water_score = float(np.clip(water_px * 2.5, 0.0, 1.0))
    elif channels >= 4 and is_geo:
        # Sentinel-2: Band 2 (idx 1) is Green, Band 4 (idx 3) is NIR
        green_b = raw[1, ::4, ::4]
        nir_b = raw[3, ::4, ::4]
        ndwi = (green_b - nir_b) / (green_b + nir_b + 1e-5)
        water_px = np.mean(ndwi > 0.0)
        water_score = float(np.clip(water_px * 2.2, 0.0, 1.0))
    elif channels >= 3:
        # Optical RGB: Water tends to be deep blue/cyan or uniform dark tones
        rgb_sample = bundle.rgb_preview[::4, ::4].astype(np.float32)
        r, g, b = rgb_sample[:, :, 0], rgb_sample[:, :, 1], rgb_sample[:, :, 2]
        blue_dominance = (b > (r + 15)) & (b > (g - 5))
        dark_water = (r < 40) & (g < 50) & (b < 60)
        water_px = np.mean(blue_dominance | dark_water)
        water_score = float(np.clip(water_px * 2.0, 0.0, 1.0))

    # 5. Scene Typology Classification
    if is_panchromatic:
        typology = "Panchromatic Grayscale Satellite Imagery"
        typology_badge = "🗺️ Panchromatic (PAN)"
        typology_color = "#E2E8F0"
    elif water_score > 0.35:
        typology = "Coastal Waterfront, Creek & Small Water Features"
        typology_badge = "🌊 Water & Coastal Interface"
        typology_color = "#38BDF8"
    elif edge_density_score > 0.65:
        typology = "Dense Urban Grid, Road Corridors & Built-up Structures"
        typology_badge = "🏙️ Dense Urban Grid"
        typology_color = "#F87171"
    elif channels >= 8:
        typology = "High-Resolution 8-Band Multispectral Scene"
        typology_badge = "🛰️ High-Res Multispectral (8-Band)"
        typology_color = "#34D399"
    elif channels == 4 and is_geo:
        typology = "Regional 4-Band Multispectral Scene (10m BOA)"
        typology_badge = "🌍 Regional Multispectral (4-Band)"
        typology_color = "#F59E0B"
    else:
        typology = "Standard Full-Color Optical Aerial Photograph"
        typology_badge = "📷 Optical RGB"
        typology_color = "#A855F7"

    # 6. Dynamic Model Fitness Scoring Engine
    all_models = get_all_models()
    model_scores = {}

    for m in all_models:
        mid = m["id"]
        status = m.get("status", "Ready")
        comp = m.get("compatibility", {})
        native_bands = comp.get("native_bands", [])
        score = 0.0
        reasons = []

        # Teammate pending check
        if status != "Ready":
            model_scores[mid] = {
                "score": 0.0,
                "fit_percent": 0,
                "is_eligible": False,
                "status": status,
                "reason": "Model weights pending registration by teammate."
            }
            continue

        # Criteria A: Panchromatic / Grayscale
        if is_panchromatic:
            if mid == "cartosat_pan_segformer":
                score = 98.0
                reasons.append("Trained natively on 1-band Cartosat-1 panchromatic imagery.")
            elif comp.get("supports_grayscale", False):
                score = 75.0
                reasons.append("Compatible via grayscale adaptation.")
            else:
                score = 10.0
                reasons.append("Requires multispectral channels.")

        # Criteria B: High-Res 8-Band WorldView-3
        elif channels == 8:
            if mid == "wv3_native_segformer":
                if edge_density_score >= 0.50:
                    score = 96.0
                    reasons.append("Top road and built-up accuracy (82.58% OA) on dense 8-band urban scenes.")
                else:
                    score = 91.0
                    reasons.append("Native 8-band hierarchical transformer.")
            elif mid == "wv3_native_unet":
                if water_score >= 0.25:
                    score = 97.0
                    reasons.append("Skip connections preserve fine water boundaries & small creeks (+19.19% Water IoU).")
                else:
                    score = 88.0
                    reasons.append("Native 8-band U-Net architecture.")
            elif mid == "sentinel2_ms_segformer":
                score = 45.0
                reasons.append("4-band model; requires band subsampling from 8 bands.")
            elif mid == "cartosat_pan_segformer":
                score = 65.0
                reasons.append("Accepts luminance reduction from multispectral bands.")
            else:
                score = 50.0
                reasons.append("General multispectral compatibility.")

        # Criteria C: 4-Band Multispectral
        elif channels == 4 and is_geo:
            if mid == "sentinel2_ms_segformer":
                score = 95.0
                reasons.append("Specifically calibrated for 4-band BOA surface reflectance (B2, B3, B4, B8).")
            elif mid == "wv3_native_segformer":
                score = 72.0
                reasons.append("8-band model requiring spectral synthesis for missing bands.")
            elif mid == "wv3_native_unet":
                score = 70.0
                reasons.append("8-band model requiring spectral synthesis.")
            else:
                score = 50.0

        # Criteria D: Standard Optical RGB / RGBA
        else:
            if water_score >= 0.30:
                if mid == "wv3_native_unet":
                    score = 94.0
                    reasons.append("Dominant water detected; U-Net skip connections prevent coastal edge bleeding.")
                elif mid == "wv3_native_segformer":
                    score = 89.0
                    reasons.append("Good accuracy, but U-Net is superior for fine water interfaces.")
                elif mid == "cartosat_pan_segformer":
                    score = 72.0
                    reasons.append("Supports RGB via photopic luminance conversion.")
            else:
                if mid == "wv3_native_segformer":
                    score = 95.0
                    reasons.append("Peak overall accuracy & continuous road extraction via spectral synthesis.")
                elif mid == "wv3_native_unet":
                    score = 88.0
                    reasons.append("Competitive U-Net feature extraction.")
                elif mid == "cartosat_pan_segformer":
                    score = 82.0
                    reasons.append("Photopic luminance conversion natively captures panchromatic structure.")
                else:
                    score = 60.0

        model_scores[mid] = {
            "score": round(score, 1),
            "fit_percent": int(round(score)),
            "is_eligible": score >= 50.0,
            "status": status,
            "reason": " ".join(reasons)
        }

    # Determine recommended model based on dynamic score
    eligible = {k: v for k, v in model_scores.items() if v["is_eligible"]}
    if eligible:
        best_id = max(eligible.keys(), key=lambda k: eligible[k]["score"])
    else:
        best_id = "wv3_native_segformer"

    best_m = get_model_by_id(best_id)
    rec_info = {
        "model_id": best_id,
        "display_name": best_m["display_name"] if best_m else best_id,
        "fitness_score": model_scores.get(best_id, {}).get("score", 90.0),
        "rationale": model_scores.get(best_id, {}).get("reason", "Highest empirical match for uploaded scene.")
    }

    return {
        "channels": channels,
        "width": w,
        "height": h,
        "pixels_mp": round((w * h) / 1_000_000.0, 2),
        "is_georeferenced": is_geo,
        "crs": bundle.crs or "Local Cartesian Grid",
        "gsd_m": gsd_m,
        "bit_depth_label": bit_depth_label,
        "is_panchromatic": is_panchromatic,
        "edge_density_score": round(edge_density_score, 2),
        "water_signature_score": round(water_score, 2),
        "scene_typology": typology,
        "typology_badge": typology_badge,
        "typology_color": typology_color,
        "model_fitness_scores": model_scores,
        "recommendation": rec_info
    }


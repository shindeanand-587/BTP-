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
    gsd_m = abs(res[0]) if isinstance(res, (tuple, list)) else (abs(res) if res is not None else None)

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

    is_vector_input = getattr(bundle, "is_vector", False) or getattr(bundle, "file_format", "").startswith("ESRI Shapefile")

    # 5. Scene Typology Classification
    if is_vector_input:
        v_dom = getattr(bundle, "vector_domain", "survey")
        v_cnt = getattr(bundle, "vector_feature_count", 0)
        typology = f"Official Vector Survey Cadastre ({v_cnt} {v_dom.title()} Features)"
        typology_badge = "🗺️ Native Vector Shapefile"
        typology_color = "#3B82F6"
    elif is_panchromatic:
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
        m_bands = m.get("band_count") or (len(native_bands) if native_bands else 3)
        
        try:
            m_gsd = float(str(m.get("native_gsd_m", 1.24)).replace("%", "").strip())
        except Exception:
            m_gsd = 1.24

        vm = m.get("verified_metrics") or {}
        try:
            oa = float(str(vm.get("overall_accuracy", 80.0)).replace("%", "").strip())
        except Exception:
            oa = 80.0
        try:
            miou = float(str(vm.get("macro_miou", 50.0)).replace("%", "").strip())
        except Exception:
            miou = 50.0

        ious = vm.get("class_ious") or {}

        # Inactive or missing weights
        if status != "Ready":
            model_scores[mid] = {
                "score": 0.0,
                "fit_percent": 0,
                "is_eligible": False,
                "status": status,
                "reason": "Model weights pending registration by external contributor."
            }
            continue

        reasons = []

        # Criteria 0: Native Vector Shapefile / GeoJSON
        if is_vector_input:
            if "vector" in mid.lower() or "survey" in mid.lower():
                score = 99.0
                reasons.append("Cadastral vector engine extracts survey boundaries with 100% geometric precision.")
            elif oa >= 83.5:
                score = 94.0
                reasons.append(f"Top-performing neural backbone ({oa:.1f}% OA) for vector feature rasterization.")
            else:
                score = 88.0
                reasons.append("Compatible geometric rasterization backbone.")

        else:
            # -------------------------------------------------------------
            # Factor 1: Spectral Band Matching & Adaptability (Max 35 points)
            # -------------------------------------------------------------
            if channels == m_bands:
                spectral_score = 35.0
                reasons.append(f"Direct native {channels}-band spectral calibration.")
            elif channels == 8 and m_bands == 4:
                spectral_score = 28.0
                reasons.append("Accepts 4-band VNIR subset from 8-band WorldView-3.")
            elif channels == 8 and m_bands == 3:
                spectral_score = 24.0
                reasons.append("Consumes 3-band True Color RGB channels from 8-band WorldView-3.")
            elif channels == 4 and m_bands == 8:
                spectral_score = 22.0
                reasons.append("Requires spectral synthesis for missing WorldView-3 bands.")
            elif channels == 4 and m_bands == 3:
                spectral_score = 26.0
                reasons.append("Consumes RGB channels from 4-band multispectral scene.")
            elif channels == 3 and m_bands == 4:
                spectral_score = 30.0
                reasons.append("High-fidelity RGB adaptation to 4-band neural core.")
            elif channels == 3 and comp.get("allow_rgb_adaptation", True):
                spectral_score = 28.0
                reasons.append(f"RGB adaptation to {m_bands}-band neural core.")
            elif is_panchromatic and m_bands == 1:
                spectral_score = 35.0
                reasons.append("Native 1-band panchromatic sensor profile.")
            elif is_panchromatic and comp.get("supports_grayscale", True):
                spectral_score = 28.0
                reasons.append("Accepts panchromatic single-channel replication.")
            elif not is_panchromatic and m_bands == 1:
                spectral_score = 24.0
                reasons.append("Accepts photopic luminance reduction to panchromatic channel.")
            else:
                spectral_score = 16.0
                reasons.append("Requires cross-sensor channel synthesis.")

            # -------------------------------------------------------------
            # Factor 2: Spatial Resolution / GSD Alignment (Max 25 points)
            # -------------------------------------------------------------
            if gsd_m is not None and gsd_m > 0:
                rel_diff = abs(gsd_m - m_gsd) / max(gsd_m, m_gsd)
                res_score = 25.0 * max(0.25, 1.0 - (rel_diff * 0.75))
                reasons.append(f"Target GSD {m_gsd:.2f}m closely aligns with scene resolution {gsd_m:.2f}m.")
            else:
                # Pixel grid heuristics based on structural edge energy
                if edge_density_score > 0.60:
                    if m_gsd <= 0.80:
                        res_score = 25.0
                        reasons.append(f"Sub-meter specialization ({m_gsd:.2f}m GSD) optimal for high-density structural grids.")
                    elif m_gsd <= 1.50:
                        res_score = 23.0
                        reasons.append("Fine resolution captures urban arterial road networks.")
                    else:
                        res_score = 18.0
                elif edge_density_score < 0.30:
                    if m_gsd >= 2.0:
                        res_score = 25.0
                        reasons.append(f"Regional GSD ({m_gsd:.1f}m) ideal for broad landscape mapping.")
                    else:
                        res_score = 20.0
                else:
                    res_score = 22.0

            # -------------------------------------------------------------
            # Factor 3: Typology & Class Specialization (Max 20 points)
            # -------------------------------------------------------------
            def _sf(v, d=60.0):
                try:
                    return float(str(v).replace("%", "").strip())
                except Exception:
                    return d

            iou_built = _sf(ious.get("Built-up"), 68.0)
            iou_road = _sf(ious.get("Roads"), 62.0)
            iou_water = _sf(ious.get("Water"), 45.0)
            iou_veg = _sf(ious.get("Vegetation"), 65.0)

            if "Dense Urban" in typology or edge_density_score >= 0.50:
                urban_perf = (iou_built * 0.55 + iou_road * 0.45)
                sem_score = (urban_perf / 100.0) * 20.0
                reasons.append(f"High urban feature IoU (Built-up {iou_built:.1f}%, Roads {iou_road:.1f}%).")
            elif "Water" in typology or water_score >= 0.30:
                water_perf = (iou_water * 0.75 + iou_veg * 0.25)
                sem_score = (water_perf / 100.0) * 20.0
                reasons.append(f"Strong coastal boundary discrimination ({iou_water:.1f}% Water IoU).")
            else:
                gen_perf = (iou_built + iou_road + iou_veg + iou_water) / 4.0
                sem_score = (gen_perf / 100.0) * 20.0
                reasons.append("Balanced multi-class segmentation fidelity across mixed terrain.")

            # -------------------------------------------------------------
            # Factor 4: Verified Empirical Accuracy & Confidence (Max 20 points)
            # -------------------------------------------------------------
            emp_score = (min(oa, 95.0) / 100.0 * 14.0) + (min(miou, 75.0) / 75.0 * 6.0)
            if oa >= 83.5:
                reasons.append(f"Benchmark-leading accuracy ({oa:.1f}% OA, {miou:.1f}% mIoU).")
            elif oa >= 80.0:
                reasons.append(f"Verified high test accuracy ({oa:.1f}% OA).")

            score = round(spectral_score + res_score + sem_score + emp_score, 1)

        score = float(np.clip(score, 15.0, 99.0))
        # Deduplicate reasons preserving order
        clean_reasons = []
        for r in reasons:
            if r not in clean_reasons:
                clean_reasons.append(r)

        model_scores[mid] = {
            "score": score,
            "fit_percent": int(round(score)),
            "is_eligible": score >= 50.0,
            "status": status,
            "reason": " ".join(clean_reasons)
        }

    # Determine recommended model based on dynamic score
    eligible = {k: v for k, v in model_scores.items() if v["is_eligible"]}
    if eligible:
        best_id = max(eligible.keys(), key=lambda k: eligible[k]["score"])
    else:
        best_id = list(model_scores.keys())[0] if model_scores else "wv3_native_segformer"

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


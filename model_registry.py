"""
model_registry.py
-----------------
Centralized Model Registry for the Unified Urban Feature Extraction AI Platform.
Maintains registrations for:
1. Team Lead Core Models (Best 3 Models):
   - WorldView-3 SegFormer-B1 (Native MS 1.24m) — Best Multispectral Model
   - Cartosat-1 PAN SegFormer-B1 (2.5m) — Best Panchromatic Single-Band Model
   - WorldView-3 U-Net ResNet34 (1.24m) — Best for Small Features & Water Boundaries
2. Teammates' Models (4 to 6 models):
   - Teammate 1 Model A (HRNet-W48 High-Resolution Urban Road Segmenter)
   - Teammate 1 Model B (Swin-Transformer Building Footprint Segmenter)
   - Teammate 2 Model A (ConvNeXt-Base Land Cover & Water Body Segmenter)
   - Teammate 2 Model B (PSPNet Spectral Boundary Refinement Model)
3. Dynamic Model Registration:
   - Loads dynamically registered teammate models from disk/session state.
"""

import os
import json
from typing import Dict, Any, List, Optional

# Verified project class mapping and color codes
CLASS_MAPPING = {
    0: {"name": "Water", "color": "#0866FF", "rgb": (8, 102, 255), "desc": "Open sea, bays, creeks, reservoirs, canals"},
    1: {"name": "Vegetation", "color": "#218C2C", "rgb": (33, 140, 44), "desc": "Forests, mangroves, tree cover, parks, turf"},
    2: {"name": "Barren Land", "color": "#E0A51B", "rgb": (224, 165, 27), "desc": "Rock quarries, bare excavation, open soil, exposed rock"},
    3: {"name": "Built-up", "color": "#E71345", "rgb": (231, 19, 69), "desc": "Commercial towers, residential housing, informal settlements"},
    4: {"name": "Roads", "color": "#999999", "rgb": (153, 153, 153), "desc": "Expressways, arterial roads, highways, paved corridors"},
    255: {"name": "Ignore / NoData", "color": "#292929", "rgb": (41, 41, 41), "desc": "Boundary bleed, nodata pixels, unannotated margin"}
}

def resolve_checkpoint_path(path: Optional[str]) -> str:
    """Resolves local absolute path or relative ./checkpoints path for cloud deployment across OS."""
    if not path:
        return ""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean = str(path).replace("\\", "/")
    fname = clean.split("/")[-1]
    cand1 = os.path.join(base_dir, "checkpoints", fname)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(base_dir, fname)
    if os.path.exists(cand2):
        return cand2
    if os.path.exists(path):
        return path
    return cand1

CORE_MODELS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # 1. WorldView-3 Native 1.24m SegFormer-B1 (Multispectral Best Overall)
    # -------------------------------------------------------------
    "wv3_native_segformer": {
        "id": "wv3_native_segformer",
        "display_name": "SegFormer-B1 (WorldView-3 Multispectral)",
        "owner": "Core System",
        "role_badge": "🌟 Multispectral Champion",
        "architecture": "SegFormer-B1 (MiT-B1 All-MLP)",
        "sensor": "Maxar WorldView-3",
        "dataset_name": "WorldView-3 Multispectral",
        "native_gsd_m": 1.24,
        "supported_bands": ["Coastal", "Blue", "Green", "Yellow", "Red", "Red Edge", "NIR1", "NIR2"],
        "band_count": 8,
        "input_dimensions": "320x320 or Tiled",
        "checkpoint_path": resolve_checkpoint_path(r"C:\BTP II\Multispectral_Project\experiments\highres_multispectral\native_1p24m\checkpoints\segformer_best.pth"),
        "adapter_class": "WorldView3SegFormerAdapter",
        "status": "Ready",
        "is_ensemble": False,
        "norm_stats": {
            "type": "zscore",
            "means": [404.82, 493.08, 633.90, 767.56, 795.60, 1129.84, 1399.20, 1528.40],
            "stds": [264.29, 330.73, 402.47, 492.73, 562.21, 658.61, 901.87, 981.03]
        },
        "verified_metrics": {
            "split": "Held-Out Test (South Mumbai: Colaba, Fort, Nariman Point)",
            "source_report": "highres_multispectral_final_evaluation_report.md",
            "macro_miou": 61.85,
            "overall_accuracy": 82.58,
            "class_ious": {
                "Water": 32.24,
                "Vegetation": 69.26,
                "Barren Land": 0.0,
                "Built-up": 77.41,
                "Roads": 68.47
            },
            "class_precision": {"Water": 97.33, "Vegetation": 75.94, "Barren Land": 0.0, "Built-up": 84.22, "Roads": 83.40},
            "class_recall": {"Water": 32.53, "Vegetation": 88.74, "Barren Land": 0.0, "Built-up": 90.55, "Roads": 79.28}
        },
        "compatibility": {
            "native_bands": [8],
            "allow_rgb_adaptation": True,
            "supports_grayscale": True,
            "notes": "Native 8-band GeoTIFF delivers peak accuracy (82.58% OA, 68.47% Road IoU). Full-color RGB adaptation supported."
        },
        "class_weights": {
            "Water": 1.2,
            "Vegetation": 1.0,
            "Barren Land": 1.5,
            "Built-up": 1.0,
            "Roads": 1.0
        },
        "description": "Hierarchical transformer trained on 8-band WorldView-3 imagery at 1.24m GSD. Delivers the project's highest road & built-up accuracy."
    },

    # -------------------------------------------------------------
    # 2. Cartosat-1 Panchromatic SegFormer-B1 (Panchromatic Best Single-Band)
    # -------------------------------------------------------------
    "cartosat_pan_segformer": {
        "id": "cartosat_pan_segformer",
        "display_name": "Cartosat PAN SegFormer-B1 (Cartosat-1 PAN)",
        "owner": "Core System",
        "role_badge": "🗺️ Panchromatic Specialist",
        "architecture": "SegFormer-B1 (Single-Band Adapted)",
        "sensor": "ISRO Cartosat-1 (PAN)",
        "dataset_name": "Cartosat-1 PAN",
        "native_gsd_m": 2.50,
        "supported_bands": ["Panchromatic"],
        "band_count": 1,
        "input_dimensions": "1024x1024 or Tiled",
        "checkpoint_path": resolve_checkpoint_path(r"C:\BTP II\Phase_2_Model_Training\experiments\exp2_builtup_balanced\checkpoints\best_model.pth"),
        "adapter_class": "CartosatSegFormerAdapter",
        "status": "Ready",
        "is_ensemble": False,
        "norm_stats": {
            "type": "global_contrast",
            "min_val": 45.0,
            "max_val": 150.0
        },
        "verified_metrics": {
            "split": "Held-Out Test Split (South Mumbai Cartosat Scene)",
            "source_report": "final_test_evaluation_report.md",
            "macro_miou": 39.02,
            "overall_accuracy": 71.85,
            "class_ious": {
                "Water": 21.50,
                "Vegetation": 61.73,
                "Barren Land": 0.0,
                "Built-up": 61.76,
                "Roads": 50.11
            },
            "class_precision": {"Water": 60.1, "Vegetation": 68.4, "Barren Land": 0.0, "Built-up": 78.1, "Roads": 59.4},
            "class_recall": {"Water": 23.8, "Vegetation": 82.3, "Barren Land": 0.0, "Built-up": 79.8, "Roads": 62.1}
        },
        "compatibility": {
            "native_bands": [1],
            "allow_rgb_adaptation": True,
            "supports_grayscale": True,
            "notes": "Natively consumes 1-channel Grayscale/PAN imagery. Validly accepts standard RGB/RGBA images via photopic luminance conversion."
        },
        "class_weights": {
            "Water": 1.0,
            "Vegetation": 1.0,
            "Barren Land": 1.2,
            "Built-up": 1.0,
            "Roads": 1.3
        },
        "description": "High-resolution panchromatic model adapted from ImageNet MiT-B1. Ideal for single-band satellite data and everyday aerial photographs."
    },

    # -------------------------------------------------------------
    # 3. WorldView-3 Native 1.24m U-Net ResNet34 (Small Features & Water Boundaries)
    # -------------------------------------------------------------
    "wv3_native_unet": {
        "id": "wv3_native_unet",
        "display_name": "U-Net ResNet34 (WorldView-3 Multispectral)",
        "owner": "Core System",
        "role_badge": "🌊 Boundary & Water Specialist",
        "architecture": "U-Net (ResNet34 Encoder 8-ch)",
        "sensor": "Maxar WorldView-3",
        "dataset_name": "WorldView-3 Multispectral",
        "native_gsd_m": 1.24,
        "supported_bands": ["Coastal", "Blue", "Green", "Yellow", "Red", "Red Edge", "NIR1", "NIR2"],
        "band_count": 8,
        "input_dimensions": "320x320 or Tiled",
        "checkpoint_path": resolve_checkpoint_path(r"C:\BTP II\Multispectral_Project\experiments\highres_multispectral\native_1p24m\checkpoints\unet_best.pth"),
        "adapter_class": "WorldView3UNetAdapter",
        "status": "Ready",
        "is_ensemble": False,
        "norm_stats": {
            "type": "zscore",
            "means": [404.82, 493.08, 633.90, 767.56, 795.60, 1129.84, 1399.20, 1528.40],
            "stds": [264.29, 330.73, 402.47, 492.73, 562.21, 658.61, 901.87, 981.03]
        },
        "verified_metrics": {
            "split": "Held-Out Test (South Mumbai)",
            "source_report": "highres_multispectral_final_evaluation_report.md",
            "macro_miou": 52.17,
            "overall_accuracy": 81.59,
            "class_ious": {
                "Water": 51.43,
                "Vegetation": 70.06,
                "Barren Land": 0.0,
                "Built-up": 73.15,
                "Roads": 66.23
            }
        },
        "compatibility": {
            "native_bands": [8],
            "allow_rgb_adaptation": True,
            "supports_grayscale": True,
            "notes": "CNN skip connections preserve fine coastal and waterfront boundaries. Top performer for Water (51.43% IoU, +19.19% over SegFormer)."
        },
        "class_weights": {
            "Water": 1.4,
            "Vegetation": 1.0,
            "Barren Land": 1.5,
            "Built-up": 1.0,
            "Roads": 1.1
        },
        "description": "ResNet34-backed U-Net with dense skip connections designed to preserve small creeks, canals, and fine waterfront boundaries."
    }
}

# Teammates' Models (Populated dynamically when teammates upload model weights)
TEAMMATE_PRESET_MODELS: Dict[str, Dict[str, Any]] = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DYNAMIC_REGISTRY_FILE = os.path.join(BASE_DIR, "teammate_models.json")

def get_dynamic_models() -> Dict[str, Dict[str, Any]]:
    """Loads dynamically uploaded teammate models from JSON file."""
    if os.path.exists(DYNAMIC_REGISTRY_FILE):
        try:
            with open(DYNAMIC_REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_dynamic_model(model_dict: Dict[str, Any]):
    """Persists a new uploaded teammate model into teammate_models.json."""
    current = get_dynamic_models()
    current[model_dict["id"]] = model_dict
    os.makedirs(os.path.dirname(DYNAMIC_REGISTRY_FILE), exist_ok=True)
    with open(DYNAMIC_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)

def delete_dynamic_model(model_id: str) -> bool:
    """Removes a dynamically registered teammate model from teammate_models.json."""
    current = get_dynamic_models()
    if model_id in current:
        del current[model_id]
        os.makedirs(os.path.dirname(DYNAMIC_REGISTRY_FILE), exist_ok=True)
        with open(DYNAMIC_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        return True
    return False

def get_full_registry() -> Dict[str, Dict[str, Any]]:
    """Combines Team Lead Core Models + Preset Teammate Models + Dynamic Uploads."""
    reg = {}
    reg.update(CORE_MODELS)
    reg.update(TEAMMATE_PRESET_MODELS)
    reg.update(get_dynamic_models())
    return reg

MODEL_REGISTRY = get_full_registry()

def get_all_models() -> List[Dict[str, Any]]:
    """Return all registered models as a list."""
    return list(get_full_registry().values())

def get_core_models() -> List[Dict[str, Any]]:
    """Return the 3 core champion models."""
    return list(CORE_MODELS.values())

def get_teammate_models() -> List[Dict[str, Any]]:
    """Return teammate models (presets + dynamically uploaded)."""
    full = get_full_registry()
    return [m for mid, m in full.items() if mid not in CORE_MODELS]

def get_ready_models() -> List[Dict[str, Any]]:
    """Return models ready for immediate inference execution."""
    return [m for m in get_full_registry().values() if m.get("status") == "Ready"]

def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve model metadata by model ID."""
    return get_full_registry().get(model_id)

def get_compatible_models(band_count: int, is_rgb: bool = False, is_gray: bool = False) -> List[Dict[str, Any]]:
    """
    Filter models that can process the uploaded imagery format.
    """
    compatible = []
    for m in get_ready_models():
        comp = m.get("compatibility", {})
        native_bands = comp.get("native_bands", [])
        if band_count in native_bands:
            compatible.append(m)
        elif is_gray and comp.get("supports_grayscale", False):
            compatible.append(m)
        elif is_rgb and comp.get("allow_rgb_adaptation", False):
            compatible.append(m)
    return compatible

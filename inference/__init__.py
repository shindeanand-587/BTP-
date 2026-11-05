"""
inference package
-----------------
Factory for model adapters with lazy loading and in-memory caching.
Supports:
1. WorldView-3 SegFormer (Native 1.24m)
2. Cartosat-1 PAN SegFormer (2.5m)
3. WorldView-3 U-Net (ResNet34 1.24m)
4. Teammate models (HRNet, Swin, ConvNeXt, PSPNet, Custom Uploads)
"""

from typing import Dict, Optional
from inference.base import BaseModelAdapter
from model_registry import get_model_by_id

_ADAPTER_CACHE: Dict[str, BaseModelAdapter] = {}

def get_adapter(model_id: str) -> BaseModelAdapter:
    """Retrieves or instantiates the cached adapter for a model ID with lazy imports."""
    if model_id in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[model_id]

    model_info = get_model_by_id(model_id)
    if not model_info:
        raise ValueError(f"Unknown model ID: {model_id}")

    # Lazy-load specific adapter module on demand
    if model_id == "cartosat_pan_segformer":
        from inference.cartosat import CartosatSegFormerAdapter
        adapter = CartosatSegFormerAdapter(model_info)
    elif model_id == "wv3_native_segformer":
        from inference.worldview3 import WorldView3SegFormerAdapter
        adapter = WorldView3SegFormerAdapter(model_info)
    elif model_id == "wv3_native_unet":
        from inference.worldview3 import WorldView3UNetAdapter
        adapter = WorldView3UNetAdapter(model_info)
    elif model_id == "sentinel2_ms_segformer":
        from inference.sentinel2 import Sentinel2SegFormerAdapter
        adapter = Sentinel2SegFormerAdapter(model_info)
    else:
        # Generic PyTorch adapter for teammate models & custom uploads
        from inference.generic import GenericPyTorchAdapter
        adapter = GenericPyTorchAdapter(model_info)

    _ADAPTER_CACHE[model_id] = adapter
    return adapter

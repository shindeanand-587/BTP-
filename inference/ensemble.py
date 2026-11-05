"""
inference/ensemble.py
---------------------
Probability-level ensemble fusion for WorldView-3 high-resolution multispectral models.
Combines:
- 0.341 * SegFormer-B1 (Hierarchical Attention)
- 0.334 * U-Net ResNet34 (Dense Skip Boundaries)
- 0.325 * DeepLabV3+ (Multi-scale ASPP Context)
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List

from inference.base import BaseModelAdapter
from inference.geospatial_io import ImageInputBundle
from inference.preprocessing import prepare_worldview3_input
from inference.worldview3 import WorldView3SegFormerAdapter, WorldView3UNetAdapter, WorldView3DeepLabAdapter
from model_registry import get_model_by_id

class WorldView3EnsembleAdapter(BaseModelAdapter):
    def __init__(self, model_info: Dict[str, Any]):
        super().__init__(model_info)
        self.segformer_adapter = WorldView3SegFormerAdapter(get_model_by_id("wv3_native_segformer"))
        self.unet_adapter = WorldView3UNetAdapter(get_model_by_id("wv3_native_unet"))
        self.deeplab_adapter = WorldView3DeepLabAdapter(get_model_by_id("wv3_native_deeplabv3p"))

        self.w_seg = 0.341
        self.w_unet = 0.334
        self.w_dlv3 = 0.325

    def load(self):
        self.segformer_adapter.load()
        self.unet_adapter.load()
        self.deeplab_adapter.load()

    def run_inference(
        self,
        bundle: ImageInputBundle,
        allow_rgb_adaptation: bool = False,
        tile_size: int = 512,
        overlap: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
        self.load()
        t0 = time.time()

        tensor, warnings = prepare_worldview3_input(
            bundle.raw_array,
            bundle.channels,
            allow_rgb_adaptation=allow_rgb_adaptation
        )

        _, c, h, w = tensor.shape
        num_classes = 5

        # Helper to compute softmax probabilities across tiles or single shot
        def get_model_probs(model):
            if h <= tile_size and w <= tile_size:
                with torch.no_grad():
                    inp = tensor.to(self.device)
                    logits = model(inp)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    if logits.shape[2:] != (h, w):
                        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                    return F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Tiled sliding window
            stride = tile_size - overlap
            prob_accum = np.zeros((num_classes, h, w), dtype=np.float32)
            weight_accum = np.zeros((h, w), dtype=np.float32)

            y_win = np.hanning(tile_size)
            x_win = np.hanning(tile_size)
            window_2d = np.maximum(np.outer(y_win, x_win), 0.05)

            y_steps = list(range(0, h - tile_size + 1, stride))
            if not y_steps or y_steps[-1] + tile_size < h:
                y_steps.append(max(0, h - tile_size))

            x_steps = list(range(0, w - tile_size + 1, stride))
            if not x_steps or x_steps[-1] + tile_size < w:
                x_steps.append(max(0, w - tile_size))

            for y0 in y_steps:
                for x0 in x_steps:
                    tile = tensor[:, :, y0:y0+tile_size, x0:x0+tile_size].to(self.device)
                    with torch.no_grad():
                        logits = model(tile)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        if logits.shape[2:] != (tile_size, tile_size):
                            logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    for cls_idx in range(num_classes):
                        prob_accum[cls_idx, y0:y0+tile_size, x0:x0+tile_size] += probs[cls_idx] * window_2d
                    weight_accum[y0:y0+tile_size, x0:x0+tile_size] += window_2d

            weight_accum = np.maximum(weight_accum, 1e-6)
            for cls_idx in range(num_classes):
                prob_accum[cls_idx] /= weight_accum
            return prob_accum

        p_seg = get_model_probs(self.segformer_adapter.model)
        p_unet = get_model_probs(self.unet_adapter.model)
        p_dlv3 = get_model_probs(self.deeplab_adapter.model)

        # Fused probability distribution
        fused_probs = (self.w_seg * p_seg) + (self.w_unet * p_unet) + (self.w_dlv3 * p_dlv3)

        pred_mask = np.argmax(fused_probs, axis=0).astype(np.uint8)
        conf_map = np.max(fused_probs, axis=0)

        duration = time.time() - t0
        warnings.append(
            f"Multi-Model Probability Fusion: Blended predictions from SegFormer-B1 ({self.w_seg:.1%}), "
            f"U-Net ResNet34 ({self.w_unet:.1%}), and DeepLabV3+ ({self.w_dlv3:.1%})."
        )
        return pred_mask, conf_map, duration, warnings

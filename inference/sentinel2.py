"""
inference/sentinel2.py
----------------------
Adapter for ESA Sentinel-2A 4-Band Multispectral SegFormer-B1 model.
Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) at 10m GSD.
"""

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from inference.base import BaseModelAdapter
from inference.geospatial_io import ImageInputBundle
from inference.preprocessing import prepare_sentinel2_input, run_tiled_inference

class SegFormerMS4Channel(nn.Module):
    """
    SegFormer-B1 architecture adapted for 4-channel Sentinel-2A imagery.
    Channels: B02 Blue, B03 Green, B04 Red, B08 NIR.
    """
    def __init__(self, num_classes=5, pretrained_backbone="nvidia/mit-b1", dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        config = SegformerConfig.from_pretrained(
            pretrained_backbone,
            num_labels=num_classes,
            classifier_dropout_prob=dropout,
            semantic_loss_ignore_index=255,
            reshape_last_stage=True
        )

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_backbone,
            config=config,
            ignore_mismatched_sizes=True
        )

        orig_proj = self.segformer.segformer.stages[0].patch_embeddings.proj
        out_channels = orig_proj.out_channels
        k_size = orig_proj.kernel_size
        stride = orig_proj.stride
        padding = orig_proj.padding

        new_proj = nn.Conv2d(4, out_channels, kernel_size=k_size, stride=stride, padding=padding)
        with torch.no_grad():
            orig_w = orig_proj.weight # (64, 3, 7, 7) - R=0, G=1, B=2
            new_proj.weight[:, 0, :, :] = orig_w[:, 2, :, :] # B02 Blue
            new_proj.weight[:, 1, :, :] = orig_w[:, 1, :, :] # B03 Green
            new_proj.weight[:, 2, :, :] = orig_w[:, 0, :, :] # B04 Red
            new_proj.weight[:, 3, :, :] = orig_w.mean(dim=1)  # B08 NIR
            if orig_proj.bias is not None:
                new_proj.bias.copy_(orig_proj.bias)

        self.segformer.segformer.stages[0].patch_embeddings.proj = new_proj

    def forward(self, pixel_values):
        input_h, input_w = pixel_values.shape[2], pixel_values.shape[3]
        outputs = self.segformer(pixel_values=pixel_values)
        raw_logits = outputs.logits

        logits = F.interpolate(
            raw_logits,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False
        )
        return logits

class Sentinel2SegFormerAdapter(BaseModelAdapter):
    def load(self):
        if self.model is not None:
            return

        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"[Sentinel2Adapter] Loading weights from {self.checkpoint_path}...")
        model = SegFormerMS4Channel(num_classes=5)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"[Sentinel2Adapter] Model loaded on {self.device}!")

    def run_inference(
        self,
        bundle: ImageInputBundle,
        allow_rgb_adaptation: bool = False,
        tile_size: int = 512,
        overlap: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
        self.load()
        t0 = time.time()

        tensor, warnings = prepare_sentinel2_input(
            bundle.raw_array,
            bundle.channels,
            allow_rgb_adaptation=allow_rgb_adaptation
        )

        pred_mask, conf_map = run_tiled_inference(
            self.model,
            tensor,
            self.device,
            tile_size=tile_size,
            overlap=overlap
        )

        duration = time.time() - t0
        return pred_mask, conf_map, duration, warnings

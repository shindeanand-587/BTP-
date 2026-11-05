"""
inference/cartosat.py
---------------------
Adapter for ISRO Cartosat-1 Panchromatic SegFormer-B1 model.
Supports 1-channel panchromatic GeoTIFFs, grayscale images,
and photopic luminance adaptation for standard 3-channel RGB/RGBA images.
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
from inference.preprocessing import prepare_cartosat_input, run_tiled_inference

class SegFormerCartosatModel(nn.Module):
    """
    SegFormer-B1 architecture adapted for Cartosat-1 panchromatic feature extraction.
    Backbone: MiT-B1.
    Decoder: All-MLP Decoder for 5 urban classes.
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

    def forward(self, pixel_values):
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

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

class CartosatSegFormerAdapter(BaseModelAdapter):
    def load(self):
        if self.model is not None:
            return

        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"[CartosatAdapter] Loading weights from {self.checkpoint_path}...")
        model = SegFormerCartosatModel(num_classes=5)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        state_dict = ckpt["model_state"] if "model_state" in ckpt else (ckpt.get("model_state_dict") or ckpt)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"[CartosatAdapter] Model loaded successfully on {self.device}!")

    def run_inference(
        self,
        bundle: ImageInputBundle,
        allow_rgb_adaptation: bool = False,
        tile_size: int = 512,
        overlap: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
        self.load()
        t0 = time.time()

        # Preprocess input array
        tensor, warnings = prepare_cartosat_input(bundle.raw_array, bundle.channels)

        # Run tiled or direct inference
        pred_mask, conf_map, probs = run_tiled_inference(
            self.model,
            tensor,
            self.device,
            tile_size=tile_size,
            overlap=overlap,
            return_probs=True
        )
        self.last_prob_map = probs

        duration = time.time() - t0
        return pred_mask, conf_map, duration, warnings

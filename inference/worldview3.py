"""
inference/worldview3.py
-----------------------
Adapters for Maxar WorldView-3 High-Resolution Multispectral Models:
1. SegFormer-B1 (8-Channel Native 1.24m)
2. U-Net (ResNet34 Encoder 8-Channel Native 1.24m)
3. DeepLabV3+ (ResNet34 ASPP 8-Channel Native 1.24m)
4. SegFormer-B1 (8-Channel Pan-Sharpened 0.31m)
5. Targeted Road & Barren Fine-Tuned SegFormer-B1 (1.24m)
"""

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import numpy as np
from typing import Dict, Any, Tuple, List

from inference.base import BaseModelAdapter
from inference.geospatial_io import ImageInputBundle
from inference.preprocessing import prepare_worldview3_input, run_tiled_inference

# ==============================================================================
# 1. 8-Channel SegFormer Architecture
# ==============================================================================
class SegFormer8Channel(nn.Module):
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

        # Adapt first patch embedding from 3 to 8 channels
        orig_proj = self.segformer.segformer.stages[0].patch_embeddings.proj
        out_channels = orig_proj.out_channels
        k_size = orig_proj.kernel_size
        stride = orig_proj.stride
        padding = orig_proj.padding

        new_proj = nn.Conv2d(8, out_channels, kernel_size=k_size, stride=stride, padding=padding)
        with torch.no_grad():
            orig_w = orig_proj.weight
            new_proj.weight[:, 0, :, :] = orig_w[:, 2, :, :]
            new_proj.weight[:, 1, :, :] = orig_w[:, 2, :, :]
            new_proj.weight[:, 2, :, :] = orig_w[:, 1, :, :]
            new_proj.weight[:, 3, :, :] = 0.5 * (orig_w[:, 1, :, :] + orig_w[:, 0, :, :])
            new_proj.weight[:, 4, :, :] = orig_w[:, 0, :, :]
            new_proj.weight[:, 5, :, :] = 0.5 * (orig_w[:, 0, :, :] + orig_w.mean(dim=1))
            new_proj.weight[:, 6, :, :] = orig_w.mean(dim=1)
            new_proj.weight[:, 7, :, :] = orig_w.mean(dim=1)
            new_proj.weight.mul_((3.0 / 8.0) ** 0.5)
            if orig_proj.bias is not None:
                new_proj.bias.copy_(orig_proj.bias)

        self.segformer.segformer.stages[0].patch_embeddings.proj = new_proj

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        outputs = self.segformer(pixel_values=x)
        raw_logits = outputs.logits
        logits = F.interpolate(raw_logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits

# ==============================================================================
# 2. 8-Channel U-Net (ResNet34) Architecture
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetResNet34_8Channel(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        orig_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            orig_w = orig_conv1.weight
            new_conv1.weight[:, 0, :, :] = orig_w[:, 2, :, :]
            new_conv1.weight[:, 1, :, :] = orig_w[:, 2, :, :]
            new_conv1.weight[:, 2, :, :] = orig_w[:, 1, :, :]
            new_conv1.weight[:, 3, :, :] = 0.5 * (orig_w[:, 1, :, :] + orig_w[:, 0, :, :])
            new_conv1.weight[:, 4, :, :] = orig_w[:, 0, :, :]
            new_conv1.weight[:, 5, :, :] = 0.5 * (orig_w[:, 0, :, :] + orig_w.mean(dim=1))
            new_conv1.weight[:, 6, :, :] = orig_w.mean(dim=1)
            new_conv1.weight[:, 7, :, :] = orig_w.mean(dim=1)
            new_conv1.weight.mul_((3.0 / 8.0) ** 0.5)

        self.conv1 = new_conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(32 + 8, 32)
        
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x0 = self.relu(self.bn1(self.conv1(x)))
        p0 = self.maxpool(x0)
        e1 = self.layer1(p0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        d4 = self.up4(e4)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=(h, w), mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, x], dim=1))

        logits = self.head(d1)
        return logits

# ==============================================================================
# 3. 8-Channel DeepLabV3+ Architecture
# ==============================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = ASPPConv(in_channels, out_channels, rates[0])
        self.conv3 = ASPPConv(in_channels, out_channels, rates[1])
        self.conv4 = ASPPConv(in_channels, out_channels, rates[2])
        self.pool = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        res = [self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x), self.pool(x)]
        return self.project(torch.cat(res, dim=1))

class DeepLabV3Plus_8Channel(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        orig_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            orig_w = orig_conv1.weight
            new_conv1.weight[:, 0, :, :] = orig_w[:, 2, :, :]
            new_conv1.weight[:, 1, :, :] = orig_w[:, 2, :, :]
            new_conv1.weight[:, 2, :, :] = orig_w[:, 1, :, :]
            new_conv1.weight[:, 3, :, :] = 0.5 * (orig_w[:, 1, :, :] + orig_w[:, 0, :, :])
            new_conv1.weight[:, 4, :, :] = orig_w[:, 0, :, :]
            new_conv1.weight[:, 5, :, :] = 0.5 * (orig_w[:, 0, :, :] + orig_w.mean(dim=1))
            new_conv1.weight[:, 6, :, :] = orig_w.mean(dim=1)
            new_conv1.weight[:, 7, :, :] = orig_w.mean(dim=1)
            new_conv1.weight.mul_((3.0 / 8.0) ** 0.5)

        self.conv1 = new_conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(512, 256)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x0 = self.relu(self.bn1(self.conv1(x)))
        p0 = self.maxpool(x0)
        low_feat = self.layer1(p0)
        x2 = self.layer2(low_feat)
        x3 = self.layer3(x2)
        high_feat = self.layer4(x3)

        aspp_out = self.aspp(high_feat)
        aspp_up = F.interpolate(aspp_out, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        low_proj = self.low_level_proj(low_feat)
        dec_in = torch.cat([aspp_up, low_proj], dim=1)
        dec_out = self.decoder(dec_in)

        logits = F.interpolate(dec_out, size=(h, w), mode='bilinear', align_corners=False)
        return logits

# ==============================================================================
# Model Adapters
# ==============================================================================
class WorldView3BaseAdapter(BaseModelAdapter):
    """Common inference runner for 8-band WorldView-3 models."""
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
        if self.checkpoint_missing_on_host:
            warnings.append(f"Model checkpoint '{os.path.basename(self.checkpoint_path or 'model.pth')}' was not found on host. Running in live preview demonstration mode.")
        return pred_mask, conf_map, duration, warnings

class WorldView3SegFormerAdapter(WorldView3BaseAdapter):
    def load(self):
        if self.model is not None:
            return

        ckpt_path = self.get_resolved_checkpoint()
        model = SegFormer8Channel(num_classes=5)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[WV3SegFormerAdapter] Loading weights from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
            model.load_state_dict(state_dict)
            print(f"[WV3SegFormerAdapter] Model loaded on {self.device}!")
        else:
            self.checkpoint_missing_on_host = True
            print(f"[WV3SegFormerAdapter] Checkpoint not found at {self.checkpoint_path}. Running with initialized backbone.")

        model.to(self.device)
        model.eval()
        self.model = model

class WorldView3UNetAdapter(WorldView3BaseAdapter):
    def load(self):
        if self.model is not None:
            return

        ckpt_path = self.get_resolved_checkpoint()
        model = UNetResNet34_8Channel(num_classes=5, pretrained=False)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[WV3UNetAdapter] Loading weights from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
            model.load_state_dict(state_dict)
            print(f"[WV3UNetAdapter] Model loaded on {self.device}!")
        else:
            self.checkpoint_missing_on_host = True
            print(f"[WV3UNetAdapter] Checkpoint not found at {self.checkpoint_path}. Running with initialized backbone.")

        model.to(self.device)
        model.eval()
        self.model = model

class WorldView3DeepLabAdapter(WorldView3BaseAdapter):
    def load(self):
        if self.model is not None:
            return

        ckpt_path = self.get_resolved_checkpoint()
        model = DeepLabV3Plus_8Channel(num_classes=5, pretrained=False)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[WV3DeepLabAdapter] Loading weights from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
            model.load_state_dict(state_dict)
            print(f"[WV3DeepLabAdapter] Model loaded on {self.device}!")
        else:
            self.checkpoint_missing_on_host = True
            print(f"[WV3DeepLabAdapter] Checkpoint not found at {self.checkpoint_path}. Running with initialized backbone.")

        model.to(self.device)
        model.eval()
        self.model = model

class WorldView3PSMSAdapter(WorldView3BaseAdapter):
    def load(self):
        if self.model is not None:
            return

        ckpt_path = self.get_resolved_checkpoint()
        model = SegFormer8Channel(num_classes=5)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[WV3PSMSAdapter] Loading weights from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
            model.load_state_dict(state_dict)
            print(f"[WV3PSMSAdapter] Model loaded on {self.device}!")
        else:
            self.checkpoint_missing_on_host = True
            print(f"[WV3PSMSAdapter] Checkpoint not found at {self.checkpoint_path}. Running with initialized backbone.")

        model.to(self.device)
        model.eval()
        self.model = model

    def run_inference(
        self,
        bundle: ImageInputBundle,
        allow_rgb_adaptation: bool = True,
        tile_size: int = 512,
        overlap: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
        self.load()
        t0 = time.time()

        tensor, warnings = prepare_worldview3_input(
            bundle.raw_array,
            bundle.channels,
            allow_rgb_adaptation=allow_rgb_adaptation,
            is_psms=True
        )

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

class WorldView3TargetedFTAdapter(WorldView3BaseAdapter):
    def load(self):
        if self.model is not None:
            return
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"[WV3TargetedFTAdapter] Loading weights from {self.checkpoint_path}...")
        model = SegFormer8Channel(num_classes=5)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"[WV3TargetedFTAdapter] Model loaded on {self.device}!")

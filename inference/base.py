"""
inference/base.py
-----------------
Abstract base adapter for all segmentation models.
Provides unified life-cycle methods: device selection, loading, inference, and memory management.
"""

import os
import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

from inference.geospatial_io import ImageInputBundle

def resolve_checkpoint_file(checkpoint_path: Optional[str]) -> Optional[str]:
    """Resolves local absolute, relative, or cloud ./checkpoints/ paths across Windows and Linux."""
    if not checkpoint_path:
        return None
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Extract filename handling Windows backslashes even when running on Linux
    clean = str(checkpoint_path).replace("\\", "/")
    fname = clean.split("/")[-1]
    cand1 = os.path.join(base_dir, "checkpoints", fname)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(base_dir, fname)
    if os.path.exists(cand2):
        return cand2
    return None

class BaseModelAdapter(ABC):
    def __init__(self, model_info: Dict[str, Any]):
        self.model_info = model_info
        self.model_id = model_info["id"]
        self.display_name = model_info["display_name"]
        self.checkpoint_path = model_info.get("checkpoint_path")
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_missing_on_host = False

    def get_resolved_checkpoint(self) -> Optional[str]:
        return resolve_checkpoint_file(self.checkpoint_path)

    def get_device(self) -> str:
        if self.device.type == "cuda":
            return f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"

    @abstractmethod
    def load(self):
        """Loads model weights into memory on the appropriate device."""
        pass

    @abstractmethod
    def run_inference(
        self,
        bundle: ImageInputBundle,
        allow_rgb_adaptation: bool = False,
        tile_size: int = 512,
        overlap: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
        """
        Executes end-to-end prediction.
        Returns:
            pred_mask: 2D uint8 numpy array with class IDs
            conf_map: 2D float32 numpy array with prediction confidence
            duration_sec: elapsed time in seconds
            warnings: list of informative string warnings
        """
        pass

    def unload(self):
        """Safely cleans up memory."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

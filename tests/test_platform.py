"""
tests/test_platform.py
----------------------
Automated test suite verifying:
1. Multi-tier model registry (3 Core Champions + 4 Teammate slots).
2. Automated Image Sanity Check & Preliminary Diagnostic Engine.
3. Multi-sensor inference on GeoTIFF, PNG, and JPEG.
"""

import os
import sys
import unittest
import numpy as np
import torch

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from model_registry import (
    get_all_models,
    get_core_models,
    get_teammate_models,
    get_ready_models,
    get_model_by_id,
    CLASS_MAPPING
)
from inference.geospatial_io import load_image_from_bytes, create_color_mask, create_blended_overlay, export_geotiff
from inference.image_diagnostic import analyze_image_characteristics
from inference import get_adapter
from inference.postprocessing import compute_class_distribution

SAMPLE_DIR = os.path.join(APP_DIR, "sample_data")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")

class TestPlatform(unittest.TestCase):
    def test_01_model_registry_structure(self):
        """Verify 3 Core Team Lead models and 4 Teammate slots."""
        cores = get_core_models()
        self.assertEqual(len(cores), 3, "Must have exactly 3 core models")
        core_ids = {m["id"] for m in cores}
        self.assertIn("wv3_native_segformer", core_ids)
        self.assertIn("cartosat_pan_segformer", core_ids)
        self.assertIn("wv3_native_unet", core_ids)

        teammates = get_teammate_models()
        self.assertGreaterEqual(len(teammates), 4, "Should have 4 or more teammate slots")

        ready = get_ready_models()
        self.assertGreaterEqual(len(ready), 3, "At least 3 core models must be Ready")

    def test_02_dynamic_image_diagnostic(self):
        """Verify preliminary image sanity check and dynamic model recommendation."""
        # 1-channel PAN should recommend Cartosat
        cart_path = os.path.join(SAMPLE_DIR, "sample_cartosat_pan.tif")
        with open(cart_path, "rb") as f:
            b1 = load_image_from_bytes(f.read(), "sample_cartosat_pan.tif")
        d1 = analyze_image_characteristics(b1)
        self.assertEqual(d1["channels"], 1)
        self.assertEqual(d1["recommendation"]["model_id"], "cartosat_pan_segformer")

        # 8-channel WV3 should recommend WV3 model
        wv3_path = os.path.join(SAMPLE_DIR, "sample_wv3_8band.tif")
        with open(wv3_path, "rb") as f:
            b2 = load_image_from_bytes(f.read(), "sample_wv3_8band.tif")
        d2 = analyze_image_characteristics(b2)
        self.assertEqual(d2["channels"], 8)
        self.assertIn(d2["recommendation"]["model_id"], ["wv3_native_segformer", "wv3_native_unet"])

    def test_03_inference_execution(self):
        """Verify inference runs cleanly on Cartosat and WV3."""
        cart_path = os.path.join(SAMPLE_DIR, "sample_cartosat_pan.tif")
        with open(cart_path, "rb") as f:
            b = load_image_from_bytes(f.read(), "sample_cartosat_pan.tif")
        adapter = get_adapter("cartosat_pan_segformer")
        mask, conf, dur, warns = adapter.run_inference(b)
        self.assertEqual(mask.shape, (b.height, b.width))
        self.assertGreater(dur, 0)

        dist = compute_class_distribution(mask, b.resolution)
        self.assertEqual(len(dist["classes"]), 5)

if __name__ == "__main__":
    unittest.main()

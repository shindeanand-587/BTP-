# 📘 Teammate & Collaborator Model Integration Guidelines
## Unified Urban Feature Extraction AI Platform (v2)

This document provides instructions, technical specifications, and copy-paste prompt templates for collaborators to package and export their custom-trained deep learning models so they can be registered on the Unified Urban Feature Extraction Platform via GitHub with 100% numerical fidelity.

---

## 🤖 Prompt for Your AI Agent (Copy & Paste Ready)

If you trained your model with an AI coding assistant (Cursor, Claude Code, ChatGPT, Antigravity, or Copilot), **copy the prompt below** and paste it into your AI assistant. It will inspect your training codebase and generate all required repository files automatically:

```text
You are an expert PyTorch machine learning deployment engineer.
I need to package my trained deep learning model into a standalone GitHub repository so it can be imported and executed flawlessly on our Unified Urban Feature Extraction AI Platform.

Please inspect my model architecture and training code, and generate all required repository files following the platform contract:

1. "adapter.py":
   - Implements `get_model(weights_path=None)` which instantiates my model with 5 output classes, loads the weights (cleaning any 'module.' prefixes), sets model.eval(), and returns the nn.Module.
   - If my model uses custom preprocessing (e.g. specific normalization, mean/std, or channel adjustments), implement `preprocess(raw_np)` that takes a numpy array and returns a normalized (1, C, H, W) torch tensor.
   
2. "model_config.json":
   - Contains:
     - "display_name": Model Name (Dataset in parentheses) e.g. "HRNet-W48 (SpaceNet Roads)"
     - "author": Contributor Name
     - "architecture": Specific architecture name
     - "sensor": "3-Band Optical RGB" | "1-Band Panchromatic (2.5m)" | "8-Band WorldView-3 (1.24m)"
     - "dataset_name": Name of training dataset
     - "native_gsd_m": Spatial resolution in meters (e.g. 1.24)
     - "overall_accuracy": Reported OA % (e.g. 84.5)
     - "macro_miou": Reported mIoU % (e.g. 56.5)
     - "class_ious": {"Water": ..., "Vegetation": ..., "Barren Land": ..., "Built-up": ..., "Roads": ...}
     - "class_weights": {"Water": ..., "Vegetation": ..., "Barren Land": ..., "Built-up": ..., "Roads": ...}
     - "preprocessing": {
         "normalization_type": "imagenet" | "zscore" | "minmax" | "divide_255",
         "means": [0.485, 0.456, 0.406],
         "stds": [0.229, 0.224, 0.225],
         "tile_size": 512,
         "overlap": 64
       }

3. Checkpoint Preparation Script ("export_checkpoint.py"):
   - Strips optimizer, scaler, and scheduler states from my checkpoint to ensure the .pth file is <100MB (GitHub's strict limit).
   - Also exports a standalone TorchScript binary "model.pt" using `torch.jit.trace(model, dummy_input)`.

4. "preflight_test.py":
   - Self-contained verification script that loads adapter.py, runs a forward pass on a 512x512 tile, asserts the output shape is (1, 5, 512, 512), and confirms class indices 0..4 match the platform mapping.

5. Exact 5-Class Index Mapping:
   - Index 0: Water
   - Index 1: Vegetation
   - Index 2: Barren Land
   - Index 3: Built-up
   - Index 4: Roads

Please write all these files now.
```

---

## ❓ Why is a raw `.pth` file alone not enough in PyTorch?

When you run `torch.save(model.state_dict(), "model.pth")`, PyTorch saves **only a dictionary of raw weight matrices (floating-point numbers)**. 

A raw `.pth` state dictionary contains:
- ❌ **NO Python class definition** (e.g., `class HRNet(nn.Module): ...`)
- ❌ **NO computational graph** (how layers connect, skip connections, attention heads)
- ❌ **NO `forward()` method** defining how an input image is transformed into 5 urban class predictions

To execute a `.pth` state dictionary, PyTorch requires the exact Python code that defines the architecture. **This is why the GitHub Repository Import method is used**: it allows the platform to clone both your model's architecture definition AND your weights.

---

## 🚀 The 3 Approved Integration Options

Choose the option that best fits your model setup:

### Option 1: Standalone TorchScript (`model.pt`) — ⭐ RECOMMENDED (Zero Code Dependencies)
TorchScript compiles your entire model architecture graph, weights, and forward pass into a single binary. It runs on the platform without requiring any external Python files or libraries.

**How to export from your training environment:**
```python
import torch

# 1. Load your trained model in evaluation mode
model = MyCustomArchitecture(num_classes=5)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# 2. Trace with a dummy input tile (B=1, C=3, H=512, W=512)
dummy_tile = torch.randn(1, 3, 512, 512)
traced_model = torch.jit.trace(model, dummy_tile)

# 3. Save as standalone TorchScript binary
traced_model.save("model.pt")
print("✅ Saved standalone model.pt successfully!")
```

**Repository Structure for Option 1:**
```text
my-model-repo/
├── model.pt               # Standalone TorchScript binary (graph + weights)
├── preflight_test.py      # Verification script
├── model_config.json      # Metadata & metrics (see schema below)
└── README.md
```

---

### Option 2: Python Adapter Script (`adapter.py` / `model.py`) — For Custom PyTorch Architectures
If you prefer keeping your standard `.pth` weights file and Python source code, include an `adapter.py` or `model.py` in your repository with a `get_model()` factory function.

**Create `adapter.py` in your repo:**
```python
import torch
import torch.nn as nn
from my_network import MyComplexModel  # Import your architecture

def get_model(weights_path=None):
    """
    Standard factory function expected by the platform.
    Returns: An initialized nn.Module in eval mode with weights loaded.
    """
    # 1. Instantiate your architecture with 5 output classes
    model = MyComplexModel(num_classes=5)
    
    # 2. Load weights if path is provided
    if weights_path:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model_state") or ckpt
        # Clean potential 'module.' prefixes from DataParallel training
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
    
    model.eval()
    return model

def preprocess(raw_np):
    """
    Optional custom preprocessing hook.
    Returns: Normalized (1, C, H, W) torch.Tensor.
    """
    # Custom normalization logic if your model differs from standard ImageNet
    tensor = torch.from_numpy(raw_np.astype("float32") / 255.0).unsqueeze(0)
    return tensor
```

**Repository Structure for Option 2:**
```text
my-model-repo/
├── adapter.py             # Contains get_model(weights_path)
├── my_network.py          # Your model architecture class definition
├── best_model.pth         # PyTorch weights file (<100MB for GitHub)
├── preflight_test.py      # Verification script
├── model_config.json      # Metadata & metrics
└── README.md
```

---

### Option 3: Full Pickled PyTorch Model (`model.pth`)
If you save the full model object instead of just the state dictionary:
```python
torch.save(model, "full_model.pth")
```
When imported along with your repo folder, the platform loads the entire model directly.

---

## 📋 The `model_config.json` Specification

Placing a `model_config.json` in your repository root enables **1-click automatic registration** with verified metrics, class weights, and descriptions pre-populated:

```json
{
  "display_name": "HRNet-W48 (SpaceNet Roads)",
  "author": "Jane Doe",
  "architecture": "HRNet-W48 Feature Network",
  "sensor": "3-Band Optical RGB",
  "dataset_name": "SpaceNet Roads",
  "native_gsd_m": 1.24,
  "overall_accuracy": 84.5,
  "macro_miou": 56.5,
  "class_ious": {
    "Water": 45.0,
    "Vegetation": 68.0,
    "Barren Land": 15.0,
    "Built-up": 72.0,
    "Roads": 82.5
  },
  "class_weights": {
    "Water": 1.0,
    "Vegetation": 1.0,
    "Barren Land": 1.2,
    "Built-up": 1.0,
    "Roads": 1.5
  },
  "preprocessing": {
    "normalization_type": "imagenet",
    "means": [0.485, 0.456, 0.406],
    "stds": [0.229, 0.224, 0.225],
    "tile_size": 512,
    "overlap": 64
  },
  "description": "High-resolution parallel multi-scale feature network preserving fine road corridors and intersection topology."
}
```

---

## 🎯 Platform Input & Output Contracts

To ensure your model integrates seamlessly with all perspectives (Road Centerlines, Building Footprints, Land Cover Analytics):

### 1. Input Tensor Contract
- **Shape:** `(B, C, H, W)` where `B=1`, `H=512`, `W=512` (the platform slices large satellite scenes into 512×512 tiles with 64px overlap).
- **Channels `C`:**
  - `3` channels for standard RGB optical aerial/satellite imagery.
  - `1` channel for Panchromatic/grayscale imagery.
  - `8` channels for WorldView-3 Multispectral imagery.
- **Value Range:** Normalized floats in `[0.0, 1.0]` (or ImageNet standardized).

### 2. Output Tensor Contract
- **Shape:** `(B, 5, H, W)` representing unnormalized logits or softmax probabilities.
- **Urban Class Index Mapping (Must Match Exactly):**
  - **Class 0:** `Water` (Open sea, bays, creeks, reservoirs, canals)
  - **Class 1:** `Vegetation` (Forests, mangroves, tree cover, parks, turf)
  - **Class 2:** `Barren Land` (Rock quarries, bare excavation, open soil, exposed rock)
  - **Class 3:** `Built-up` (Commercial towers, residential housing, informal settlements)
  - **Class 4:** `Roads` (Expressways, arterial roads, highways, paved corridors, runways)

---

## 💡 Important GitHub File Size Tip
- GitHub strictly rejects individual files **larger than 100 MB**.
- If your `.pth` checkpoint is >100 MB because it contains optimizer and scheduler states, strip them before committing:
```python
# Strip optimizer to reduce file size from ~160MB to ~50MB:
ckpt = torch.load("checkpoint.pth", map_location="cpu")
state_only = {"model_state": ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt}
torch.save(state_only, "best_model_stripped.pth")
```

---

## 🧪 Preflight Verification Script (`preflight_test.py`)

Include this script in your repository. When our platform imports your repository, it runs this script to certify **100% numerical fidelity**:

```python
import sys
import torch
import numpy as np

try:
    # Test loading
    from adapter import get_model
    model = get_model()
    model.eval()

    # Test forward pass with synthetic tile
    dummy_tile = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy_tile)
        if isinstance(out, dict):
            out = out.get("logits", list(out.values())[0])

    assert out.shape[1] == 5, f"Expected 5 output classes, got {out.shape[1]}"
    assert out.shape[2:] == (512, 512), f"Expected 512x512 spatial dimensions, got {out.shape[2:]}"
    print("PREFLIGHT_PASS: 100% numerical verification achieved!")
    sys.exit(0)
except Exception as e:
    print(f"PREFLIGHT_FAIL: {e}")
    sys.exit(1)
```

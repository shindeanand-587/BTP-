# 📘 Model Developer & Integration Guidelines
## Unified Urban Feature Extraction AI Platform (v2)
### Standardized Model Registration & Automated AI Packaging Protocol

This document establishes the technical specifications, repository architecture, and automated AI packaging instructions for deploying deep learning models onto the Unified Urban Feature Extraction Platform via GitHub or local upload.

---

## 🤖 Automated Packaging Specification for AI Coding Agents

When packaging custom-trained models using an AI coding assistant (**Cursor, Claude Code, Antigravity, ChatGPT, or GitHub Copilot**), copy the specification prompt below and paste it directly into your AI assistant. It will inspect your codebase and generate all required repository files automatically:

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
     - "author": Contributor Name / Institution
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

## ❓ Architectural Foundation: Why a raw `.pth` file alone is insufficient in PyTorch

When saving weights via `torch.save(model.state_dict(), "model.pth")`, PyTorch serializes **only an ordered dictionary of floating-point numerical weight tensors**.

A raw `.pth` state dictionary contains:
- ❌ **No Python class definitions** (e.g., `class HRNet(nn.Module): ...`)
- ❌ **No computational graph** (layer connectivity, skip paths, attention mechanisms)
- ❌ **No `forward()` method** specifying execution flow from input tensors to output logits

To execute an arbitrary deep neural network in production, PyTorch requires the architecture's structural graph. **The GitHub Repository Integration method resolves this** by importing both the model's architecture definition and the trained weights simultaneously.

---

## 🚀 The 3 Approved Deployment Architecture Options

Choose the deployment architecture that best fits your model implementation:

### Option 1: Standalone TorchScript (`model.pt`) — ⭐ RECOMMENDED (Zero Code Dependencies)
TorchScript compiles the entire neural network computational graph, forward pass, and weights into a single binary. It executes on the platform with `torch.jit.load()` without requiring external Python scripts or library dependencies.

**Export Procedure (3 lines of Python):**
```python
import torch

# 1. Load trained model in evaluation mode
model = MyCustomArchitecture(num_classes=5)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# 2. Trace with a canonical input tile (B=1, C=3, H=512, W=512)
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
├── preflight_test.py      # Automated verification test script
├── model_config.json      # Metadata, preprocessing, & class weights
└── README.md
```

---

### Option 2: Python Adapter Module (`adapter.py` / `model.py`) — For Custom PyTorch Architectures
For custom research architectures retaining standard `.pth` weights and Python source definitions, provide an `adapter.py` with a standardized `get_model()` factory function.

**Create `adapter.py` in your repository root:**
```python
import torch
import torch.nn as nn
from my_network import MyComplexModel  # Import your architecture definition

def get_model(weights_path=None):
    """
    Standard factory function invoked dynamically by the platform.
    Returns: An initialized nn.Module in eval mode with weights loaded.
    """
    model = MyComplexModel(num_classes=5)
    
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
    Optional custom preprocessing hook for non-standard sensor pipelines.
    Returns: Normalized (1, C, H, W) torch.Tensor.
    """
    tensor = torch.from_numpy(raw_np.astype("float32") / 255.0).unsqueeze(0)
    return tensor
```

**Repository Structure for Option 2:**
```text
my-model-repo/
├── adapter.py             # Contains get_model(weights_path)
├── my_network.py          # PyTorch nn.Module architecture classes
├── best_model.pth         # PyTorch weights file (<100MB for GitHub)
├── preflight_test.py      # Automated verification test script
├── model_config.json      # Metadata, preprocessing, & class weights
└── README.md
```

---

### Option 3: Full Pickled PyTorch Object (`model.pth`)
If your model is serialized as a complete object:
```python
torch.save(model, "full_model.pth")
```
When imported alongside your repository codebase, the platform dynamically restores the full `nn.Module` instance.

---

## 📋 The `model_config.json` Specification

Placing a `model_config.json` in your repository root enables **automated 1-click registration** with verified benchmark metrics, class weights, and preprocessing pre-populated:

```json
{
  "display_name": "HRNet-W48 (SpaceNet Roads)",
  "author": "Research Contributor",
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
  "description": "High-resolution multi-scale representation network preserving fine arterial corridors and intersection topology."
}
```

---

## 🎯 Platform Input & Output Contracts

### 1. Input Tensor Contract
- **Shape:** `(B, C, H, W)` where `B=1`, `H=512`, `W=512` (the platform slices large satellite scenes into 512×512 tiles with 64px overlap).
- **Channels `C`:**
  - `3` channels for standard RGB optical aerial/satellite imagery.
  - `1` channel for Panchromatic/grayscale imagery.
  - `8` channels for WorldView-3 Multispectral imagery.
- **Value Range:** Normalized floats in `[0.0, 1.0]` or ImageNet standardized.

### 2. Output Tensor Contract
- **Shape:** `(B, 5, H, W)` representing unnormalized logits or softmax probabilities.
- **Urban Class Index Mapping (Exact Alignment Required):**
  - **Class 0:** `Water` (Open sea, bays, creeks, reservoirs, canals)
  - **Class 1:** `Vegetation` (Forests, mangroves, tree cover, parks, turf)
  - **Class 2:** `Barren Land` (Rock quarries, bare excavation, open soil, exposed rock)
  - **Class 3:** `Built-up` (Commercial towers, residential housing, informal settlements)
  - **Class 4:** `Roads` (Expressways, arterial roads, highways, paved corridors, runways)

---

## 💡 GitHub File Size Optimization (<100MB)
- GitHub strictly rejects individual files **larger than 100 MB**.
- Checkpoints containing optimizer, scaler, or scheduler states can be stripped to reduce size from ~160MB to ~50MB:
```python
ckpt = torch.load("checkpoint.pth", map_location="cpu")
state_only = {"model_state": ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt}
torch.save(state_only, "best_model_stripped.pth")
```

---

## 🧪 Preflight Verification Script (`preflight_test.py`)

Include this script in your repository. When our platform imports the repository, it executes this test to certify **100% numerical training-inference parity**:

```python
import sys
import torch
import numpy as np

try:
    from adapter import get_model
    model = get_model()
    model.eval()

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

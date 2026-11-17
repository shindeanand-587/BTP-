# 📘 Teammate & Collaborator Model Integration Guidelines
## Unified Urban Feature Extraction AI Platform (v2)

This guide provides instructions and code templates for teammates and collaborators to successfully package and integrate their custom-trained deep learning models into the Unified Urban Feature Extraction AI Platform via GitHub or local upload.

---

## ❓ Why is a `.pth` file alone not enough in PyTorch?

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
├── model.pt               # Standalone TorchScript binary
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
```

**Repository Structure for Option 2:**
```text
my-model-repo/
├── adapter.py             # Contains get_model(weights_path)
├── my_network.py          # Your model architecture class definition
├── best_model.pth         # PyTorch weights file (<100MB for GitHub)
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
- **Value Range:** Normalized floats in `[0.0, 1.0]`.

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

## 🧪 Testing Your Model Before Committing

You can test that your model meets the platform contract locally with this 5-line test:
```python
import torch

# Test loading
# Option A: from adapter import get_model; model = get_model("best_model.pth")
# Option B: model = torch.jit.load("model.pt")

model.eval()
test_tile = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    out = model(test_tile)
    if isinstance(out, dict): out = out.get("logits", list(out.values())[0])

assert out.shape[1] == 5, f"Expected 5 output classes, got {out.shape[1]}"
assert out.shape[2:] == (512, 512), f"Expected 512x512 spatial dimensions, got {out.shape[2:]}"
print("🎉 Model conforms 100% to platform specifications!")
```

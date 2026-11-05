# Unified Urban Feature Extraction AI Platform

**B.Tech Project II — Multi-Sensor Satellite Semantic Segmentation Platform**  
**Study Area:** Mumbai Metropolitan Region (MMR), Maharashtra, India  
**Collaborative Team:** Anand Shinde (Lead), Teammate 1 (Friend 1), Teammate 2 (Friend 2)  
**Hardware Target:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4) with CPU fallback  

---

## 1. Quick Launch Instructions

### Local Launch (One-Click)
Double-click `run_app.bat` in this folder:
```cmd
C:\BTP II\Urban_Extraction_Platform\run_app.bat
```

### Manual Command-Line Launch
Open PowerShell or Command Prompt:
```powershell
# 1. Activate the project virtual environment
& "C:\BTP II\.venv\Scripts\Activate.ps1"

# 2. Navigate to the application folder
cd "C:\BTP II\Urban_Extraction_Platform"

# 3. Launch Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

- **Local Web Address:** `http://localhost:8501`
- **Network / Wi-Fi Address:** `http://<YOUR_LOCAL_IP>:8501` (accessible from tablets or laptops on the same Wi-Fi network)

---

## 2. Integrated Model Inventory (12 Models)

All models are unified into a centralized registry (`model_registry.py`). Models load on demand into GPU VRAM.

| Model ID | Display Name | Owner | Architecture | Sensor / GSD | Held-Out OA | Road IoU | Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `wv3_native_segformer` | WorldView-3 SegFormer-B1 | Team Lead | MiT-B1 All-MLP (8-ch) | WV-3 (1.24m) | **82.58%** | **68.47%** | ✅ Ready |
| `wv3_native_unet` | WorldView-3 U-Net | Team Lead | ResNet34 U-Net (8-ch) | WV-3 (1.24m) | **81.59%** | **66.23%** | ✅ Ready |
| `wv3_native_deeplabv3p` | WorldView-3 DeepLabV3+ | Team Lead | ResNet34 ASPP (8-ch) | WV-3 (1.24m) | **77.04%** | **57.74%** | ✅ Ready |
| `wv3_ensemble` | WorldView-3 Weighted Ensemble | Team Lead | Multi-Model Probability Fusion | WV-3 (1.24m) | **82.10%** | **67.97%** | ✅ Ready |
| `wv3_targeted_ft` | WorldView-3 Road & Barren FT | Team Lead | SegFormer Tversky Loss | WV-3 (1.24m) | **83.13%** | **70.79%** | ✅ Ready |
| `wv3_psms_segformer` | WorldView-3 Pan-Sharpened | Team Lead | SegFormer-B1 (8-ch) | WV-3 PS-MS (0.31m) | **74.20%** | **63.30%** | ✅ Ready |
| `cartosat_pan_segformer`| Cartosat-1 PAN SegFormer | Team Lead | SegFormer-B1 (1-ch) | Cartosat-1 (2.5m) | **71.85%** | **50.11%** | ✅ Ready |
| `sentinel2_ms_segformer`| Sentinel-2A SegFormer-B1 | Team Lead | SegFormer-B1 (4-ch) | Sentinel-2 (10m) | **68.42%** | **36.51%** | ✅ Ready |
| `friend1_model_a` | HRNet Road Segmenter | Teammate 1 | HRNet-W48 | High-Res Optical | — | — | ⏳ Pending |
| `friend1_model_b` | Swin Building Segmenter | Teammate 1 | Swin-T FPN | Optical Multispectral | — | — | ⏳ Pending |
| `friend2_model_a` | ConvNeXt Land Cover | Teammate 2 | ConvNeXt-Base FPN | Multi-Sensor | — | — | ⏳ Pending |
| `friend2_model_b` | PSPNet Boundary Refiner | Teammate 2 | PSPNet + Edge Loss | Cartosat + S2 Hybrid | — | — | ⏳ Pending |

---

## 3. Verified Class Color Scheme

The 5 semantic urban classes use verified project hex codes:
- **Water (Class 0):** `#0866FF` (Deep Blue)
- **Vegetation (Class 1):** `#218C2C` (Green)
- **Barren Land (Class 2):** `#E0A51B` (Amber Gold)
- **Built-up (Class 3):** `#E71345` (Crimson Red)
- **Roads (Class 4):** `#999999` (Asphalt Grey)
- **Ignore / NoData (Class 255):** `#292929` (Dark Charcoal)

---

## 4. Ordinary PNG/JPEG Image Prediction Support

The platform allows professors or evaluators to upload ordinary images without requiring GIS software:
- **Ordinary RGB PNG / JPEG:** Natively processed by Cartosat-1 PAN model via standard photopic luminance conversion ($0.299R + 0.587G + 0.114B$).
- **Panchromatic Grayscale (1-channel PNG):** Directly normalized and predicted.
- **Transparent RGBA (4-channel PNG):** Alpha channel is composited cleanly and preserved.
- **Multiband GeoTIFF (8-band WV3 or 4-band S2):** Ingests native multispectral bands, preserving spatial coordinates and resolution.
- **Experimental RGB Adaptation:** Optional toggle allows running 8-band WorldView-3 models on RGB photos with prominent transparency disclaimers.

---

## 5. How to Demonstrate to Your Professor (5-Minute Walkthrough)

1. **Step 1 — Dashboard Overview (1 minute):**
   - Open `http://localhost:8501`.
   - Point out the **12 registered models** across 3 team members.
   - Show the **Verified Benchmark Comparison Table** with held-out test metrics (82.58% OA on WorldView-3 vs 68.42% on Sentinel-2).
   - Explain how 1.24m resolution solved the road-bloat problem observed at 10m.

2. **Step 2 — Ordinary Image Prediction (1.5 minutes):**
   - Click **🛰️ Upload Imagery**.
   - Select the preset: `Ordinary Satellite RGB PNG` (or upload any JPEG/PNG from your computer).
   - Switch to **🤖 Model Selection** → Notice **AUTO mode** automatically selects the Cartosat-1 PAN model via photopic luminance conversion.
   - Switch to **⚡ Run Inference** → Click **🚀 Run Prediction**.
   - Adjust the **Mask Overlay Opacity slider** to show building and road alignment.
   - Click **📥 Colored Mask (PNG)** or **Class-ID Mask** to demonstrate downloads.

3. **Step 3 — Native 8-Band Multispectral Inference (1.5 minutes):**
   - Go to **🛰️ Upload Imagery** → Select `WorldView-3 Native 8-Band GeoTIFF`.
   - Show the metadata panel: EPSG:4326 CRS, 8 bands, 1.24m GSD.
   - In **Model Selection**, AUTO mode selects the top-ranked **WorldView-3 SegFormer-B1** (82.58% test OA).
   - In **Run Inference**, click **Run Prediction**.
   - Point out the sharp road corridors (Grey `#999999`) and building separation (Red `#E71345`).
   - Download the **🗺️ Georeferenced GeoTIFF** and explain that spatial coordinates and EPSG are preserved for QGIS/ArcGIS!

4. **Step 4 — Multi-Model Comparison & Ensembling (1 minute):**
   - Under **Model Selection**, switch to **3️⃣ COMPARE Mode**.
   - Select `WorldView-3 SegFormer-B1` and `WorldView-3 U-Net (ResNet34)`.
   - Click **Run Prediction** → View side-by-side masks and compare inference latency (~0.12s on RTX 4050).

5. **Step 5 — Land Cover & Spatial Area Analysis:**
   - Click **📊 Class Area Analysis**.
   - Show the **hectare and km² spatial distribution** computed from pixel counts and ground sampling distance.

---

## 6. Remote / Production Deployment Guide

To deploy this platform onto a shared laboratory server or cloud VM (Ubuntu/Debian with NVIDIA GPU):

1. **Environment Setup:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r Urban_Extraction_Platform/requirements.txt
   ```

2. **Transfer Model Weights:**
   Ensure checkpoint directories match `model_registry.py` paths, or configure environment variable `MODEL_CKPT_ROOT`.

3. **Production Service (systemd or Docker):**
   ```bash
   streamlit run Urban_Extraction_Platform/app.py \
       --server.port 8501 \
       --server.address 0.0.0.0 \
       --server.headless true \
       --browser.serverAddress your-domain-or-ip.com
   ```

4. **Security & Authentication:**
   For public exposure, wrap Streamlit behind an NGINX reverse proxy with SSL/TLS (`certbot`) and HTTP basic authentication (`htpasswd`).

---

## 7. Technical Integrity Guidelines

- **No Fake Metrics:** Live uploads do not invent accuracy or IoU figures; pixel distributions are strictly labeled as predicted proportions.
- **Zero Training Interference:** This deployment runs in inference-only mode (`torch.no_grad()`). No training loops, fine-tuning scripts, or weight updates are executed.
- **Model Isolation:** Teammate models are integrated cleanly via the centralized registry without touching working adapters.

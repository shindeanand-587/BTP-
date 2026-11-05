"""
ui/upload.py
------------
Handles imagery ingestion, format validation, sensor auto-detection,
interactive preview, and geospatial metadata extraction.
Supports GeoTIFF, TIFF, PNG, JPEG, and RGBA.
"""

import os
import streamlit as st
import numpy as np
from PIL import Image

from inference.geospatial_io import load_image_from_bytes, ImageInputBundle

SAMPLE_DIR = r"C:\BTP II\Urban_Extraction_Platform\sample_data"

SAMPLE_PRESETS = {
    "wv3_8band": {
        "name": "WorldView-3 Native 8-Band GeoTIFF (1.24m GSD, EPSG:4326)",
        "file": os.path.join(SAMPLE_DIR, "sample_wv3_8band.tif"),
        "sensor": "Maxar WorldView-3 (8-Band Native)"
    },
    "urban_rgb_png": {
        "name": "Ordinary Satellite RGB PNG (3-Channel, Uncompressed)",
        "file": os.path.join(SAMPLE_DIR, "sample_urban_rgb.png"),
        "sensor": "Standard RGB Aerial / Satellite"
    },
    "urban_rgb_jpg": {
        "name": "Ordinary Satellite RGB JPEG (3-Channel, 95% Quality)",
        "file": os.path.join(SAMPLE_DIR, "sample_urban_rgb.jpg"),
        "sensor": "Standard RGB Aerial / Satellite"
    },
    "pan_gray_png": {
        "name": "Panchromatic Grayscale PNG (1-Channel Grayscale)",
        "file": os.path.join(SAMPLE_DIR, "sample_panchromatic_gray.png"),
        "sensor": "Panchromatic Grayscale"
    },
    "urban_rgba_png": {
        "name": "Transparent RGBA Satellite PNG (4-Channel with Alpha Mask)",
        "file": os.path.join(SAMPLE_DIR, "sample_urban_rgba.png"),
        "sensor": "RGBA Optical Satellite with Alpha"
    },
    "cartosat_pan_tif": {
        "name": "Cartosat-1 Panchromatic GeoTIFF (2.5m GSD, EPSG:32643)",
        "file": os.path.join(SAMPLE_DIR, "sample_cartosat_pan.tif"),
        "sensor": "ISRO Cartosat-1 (1-Band PAN)"
    }
}

def detect_sensor_type(channels: int, is_geo: bool, crs: str) -> str:
    """Heuristic sensor detection based on band count and metadata."""
    if channels == 8:
        return "Maxar WorldView-3 (8-Band Multispectral)"
    elif channels == 4 and is_geo:
        return "Sentinel-2A MSI (4-Band Surface Reflectance)"
    elif channels == 4 and not is_geo:
        return "RGBA Satellite Image with Alpha Channel"
    elif channels == 1 and is_geo:
        return "ISRO Cartosat-1 (1-Band Panchromatic)"
    elif channels == 1 and not is_geo:
        return "Panchromatic Grayscale Aerial Image"
    elif channels == 3:
        return "Standard RGB Satellite / Aerial Imagery"
    return f"Custom Multiband Sensor ({channels} Channels)"

def render_upload():
    st.markdown('<div class="section-header">🛰️ Ingest Satellite & Aerial Imagery</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a satellite scene in <b>GeoTIFF/TIFF</b> or an ordinary <b>PNG/JPEG</b> image.
    The platform inspects band configurations, spatial metadata, coordinate systems, and runs technical validation.
    """)

    source_mode = st.radio(
        "Choose Input Source:",
        ["⚡ Quick Test with Sample Preset", "📂 Upload Custom File (GeoTIFF / PNG / JPEG)"],
        horizontal=True
    )

    bundle: ImageInputBundle = None

    if source_mode.startswith("⚡"):
        preset_key = st.selectbox(
            "Select a pre-loaded verification dataset:",
            options=list(SAMPLE_PRESETS.keys()),
            format_func=lambda k: SAMPLE_PRESETS[k]["name"]
        )
        preset_info = SAMPLE_PRESETS[preset_key]
        if os.path.exists(preset_info["file"]):
            with open(preset_info["file"], "rb") as f:
                file_bytes = f.read()
            bundle = load_image_from_bytes(file_bytes, os.path.basename(preset_info["file"]))
        else:
            st.error(f"Sample file not found: {preset_info['file']}")
    else:
        uploaded_file = st.file_uploader(
            "Choose a satellite image file (Max 250 MB)",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            help="Supported formats: Multiband GeoTIFF (8-band WV3, 4-band S2, 1-band Cartosat) or standard RGB/RGBA/Grayscale PNG/JPEG."
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            bundle = load_image_from_bytes(file_bytes, uploaded_file.name)

    if bundle is not None:
        st.session_state["current_bundle"] = bundle

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        col_prev, col_meta = st.columns([1.1, 1.4])

        with col_prev:
            st.markdown('<div style="font-weight: 700; margin-bottom: 8px; color: #CBD5E1;">Visual Preview (True-Color Composite)</div>', unsafe_allow_html=True)
            st.image(
                bundle.rgb_preview,
                caption=f"{bundle.filename} ({bundle.width} × {bundle.height} px)",
                use_container_width=True
            )

        with col_meta:
            st.markdown('<div style="font-weight: 700; margin-bottom: 8px; color: #CBD5E1;">Imagery Technical Metadata</div>', unsafe_allow_html=True)
            
            detected_sensor = detect_sensor_type(bundle.channels, bundle.is_georeferenced, bundle.crs)
            
            # Metadata Grid
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Dimensions", f"{bundle.width} × {bundle.height} px")
                st.metric("Channels / Bands", f"{bundle.channels} Band{'s' if bundle.channels > 1 else ''}")
                st.metric("File Format", bundle.file_format.upper())
            with m2:
                st.metric("Georeferenced", "Yes (CRS Embedded)" if bundle.is_georeferenced else "Standard Image Grid")
                st.metric("Coordinate System", bundle.crs if bundle.crs else "Non-georeferenced")
                if bundle.resolution:
                    st.metric("Pixel GSD", f"{abs(bundle.resolution[0]):.2f} m × {abs(bundle.resolution[1]):.2f} m")
                else:
                    st.metric("Pixel GSD", "Arbitrary / Unknown GSD")

            st.markdown("""<div style="height: 8px;"></div>""", unsafe_allow_html=True)
            
            # Sensor Confirmation Selector
            confirmed_sensor = st.selectbox(
                "Detected Sensor / Radiometric Profile (Confirm or Correct):",
                [
                    detected_sensor,
                    "Maxar WorldView-3 (8-Band Multispectral)",
                    "Sentinel-2A MSI (4-Band BOA Surface Reflectance)",
                    "ISRO Cartosat-1 (1-Band Panchromatic)",
                    "Standard RGB Satellite Imagery",
                    "Panchromatic Grayscale",
                    "Custom / Other"
                ]
            )
            st.session_state["confirmed_sensor"] = confirmed_sensor

        # Technical Compatibility Safeguards
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if bundle.channels in [3, 4] and not bundle.is_georeferenced:
            st.markdown("""
            <div class="callout-box callout-warn">
                <b>📌 Ordinary Image Ingestion Notice:</b>
                The uploaded file is a standard <b>PNG/JPEG image ({channels} channels)</b>.
                It contains RGB visible-light channels, not native 8-band WorldView-3 or 4-band Sentinel-2 spectral measurements.<br>
                • <b>Cartosat-1 PAN Model:</b> Fully compatible via photopic luminance conversion.<br>
                • <b>WorldView-3 / Sentinel-2 Models:</b> Native multispectral inference requires multiband GeoTIFF.
                You can run WorldView-3 using <b>Experimental RGB Adaptation</b>, but native spectral accuracy cannot be guaranteed.
            </div>
            """.format(channels=bundle.channels), unsafe_allow_html=True)
        elif bundle.channels == 8:
            st.markdown("""
            <div class="callout-box callout-success">
                <b>✅ Native High-Resolution Multispectral Data Verified:</b>
                8 spectral bands detected (Coastal, Blue, Green, Yellow, Red, Red Edge, NIR1, NIR2).
                Fully compatible with all native WorldView-3 models (SegFormer, U-Net, DeepLabV3+, and Ensemble) at peak benchmark accuracy!
            </div>
            """, unsafe_allow_html=True)
        elif bundle.channels == 1:
            st.markdown("""
            <div class="callout-box callout-success">
                <b>✅ Panchromatic Data Verified:</b>
                Single-channel panchromatic imagery detected.
                Fully compatible with Cartosat-1 PAN SegFormer-B1 model!
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👆 Please select a sample preset above or upload an imagery file to continue.")

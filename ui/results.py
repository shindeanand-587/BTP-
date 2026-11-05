"""
ui/results.py
-------------
Renders inference execution, prediction visualizations, interactive overlay sliders,
class area distributions, multi-model side-by-side comparisons, and GIS downloads.
"""

import os
import io
import json
import time
import streamlit as st
import numpy as np
from PIL import Image

from inference.geospatial_io import (
    ImageInputBundle,
    create_color_mask,
    create_blended_overlay,
    export_geotiff
)
from inference import get_adapter
from inference.postprocessing import compute_class_distribution, render_confidence_map
from model_registry import CLASS_MAPPING

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def render_results():
    st.markdown('<div class="section-header">⚡ Run Inference & Inspect Predictions</div>', unsafe_allow_html=True)

    if "current_bundle" not in st.session_state:
        st.warning("⚠️ No image loaded yet. Please visit **Upload Imagery** first.")
        return

    bundle: ImageInputBundle = st.session_state["current_bundle"]
    selected_models = st.session_state.get("selected_models_to_run", [])
    allow_rgb_adaptation = st.session_state.get("allow_rgb_adaptation", False)

    if not selected_models:
        st.info("👆 Please select at least one compatible model in **Model Selection**.")
        return

    # Inference Execution Controls
    c_btn, c_info = st.columns([1.2, 2.5])
    with c_btn:
        run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
    with c_info:
        model_names = ", ".join([m["display_name"] for m in selected_models])
        st.markdown(f"""
        <div style="font-size: 0.88rem; color: #94A3B8; padding-top: 6px;">
            Target: <b>{model_names}</b> ({len(selected_models)} model{'s' if len(selected_models) > 1 else ''})
        </div>
        """, unsafe_allow_html=True)

    # Execute inference when button is pressed
    if run_btn:
        results = []
        progress_bar = st.progress(0, text="Initializing inference engine...")

        for idx, model_info in enumerate(selected_models):
            progress_val = int((idx / len(selected_models)) * 100)
            progress_bar.progress(progress_val, text=f"Loading & running {model_info['display_name']} on GPU...")

            try:
                adapter = get_adapter(model_info["id"])
                pred_mask, conf_map, dur, warns = adapter.run_inference(
                    bundle,
                    allow_rgb_adaptation=allow_rgb_adaptation,
                    tile_size=512,
                    overlap=64
                )

                color_mask = create_color_mask(pred_mask)
                dist = compute_class_distribution(pred_mask, bundle.resolution)
                conf_rgb = render_confidence_map(conf_map) if conf_map is not None else None

                results.append({
                    "model_info": model_info,
                    "pred_mask": pred_mask,
                    "color_mask": color_mask,
                    "conf_map": conf_map,
                    "conf_rgb": conf_rgb,
                    "distribution": dist,
                    "duration_sec": dur,
                    "warnings": warns,
                    "device": adapter.get_device()
                })
            except Exception as e:
                st.error(f"Error running inference for {model_info['display_name']}: {str(e)}")

        progress_bar.progress(100, text="Inference complete!")
        time.sleep(0.3)
        progress_bar.empty()
        st.session_state["last_inference_results"] = results

    # Display stored inference results
    if "last_inference_results" in st.session_state and st.session_state["last_inference_results"]:
        results = st.session_state["last_inference_results"]

        # If multiple models were run, show comparison tabs
        if len(results) > 1:
            st.markdown(f'<div class="section-header" style="margin-top: 20px;">📊 Multi-Model Comparison ({len(results)} Models)</div>', unsafe_allow_html=True)
            tabs = st.tabs([res["model_info"]["display_name"] for res in results] + ["🔍 Side-by-Side View"])
            for idx, res in enumerate(results):
                with tabs[idx]:
                    render_single_result(res, bundle)
            with tabs[-1]:
                render_side_by_side(results, bundle)
        else:
            render_single_result(results[0], bundle)

def render_single_result(res: dict, bundle: ImageInputBundle):
    model_info = res["model_info"]
    color_mask = res["color_mask"]
    pred_mask = res["pred_mask"]
    dist = res["distribution"]
    dur = res["duration_sec"]
    warns = res["warnings"]
    device = res["device"]

    # Model Execution Header Badge
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(30, 41, 59, 0.5); padding: 10px 16px; border-radius: 10px; margin-bottom: 14px;">
        <div>
            <span style="font-weight: 700; color: #F8FAFC; font-size: 1.05rem;">{model_info['display_name']}</span>
            <span style="color: #94A3B8; font-size: 0.85rem; margin-left: 10px;">({model_info['architecture']})</span>
        </div>
        <div>
            <span class="badge badge-gpu">⚡ {device}</span>
            <span class="badge badge-ready" style="margin-left: 6px;">⏱️ {dur*1000.0:.1f} ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Technical Warnings Callout
    if warns:
        for w in warns:
            st.markdown(f"""
            <div class="callout-box callout-warn" style="margin-top: 4px; margin-bottom: 10px;">
                {w}
            </div>
            """, unsafe_allow_html=True)

    # Interactive Alpha Slider for Blended Overlay
    alpha = st.slider(
        f"Mask Overlay Opacity ({model_info['id']}):",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05
    )

    overlay = create_blended_overlay(bundle.rgb_preview, color_mask, alpha)

    # Visual Panels Grid: Original, Mask, Overlay, Confidence Heatmap
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div style="font-weight: 600; color: #CBD5E1; margin-bottom: 6px;">1. Original Scene</div>', unsafe_allow_html=True)
        st.image(bundle.rgb_preview, use_container_width=True)
    with c2:
        st.markdown('<div style="font-weight: 600; color: #CBD5E1; margin-bottom: 6px;">2. Segmentation Mask</div>', unsafe_allow_html=True)
        st.image(color_mask, use_container_width=True)
    with c3:
        st.markdown(f'<div style="font-weight: 600; color: #CBD5E1; margin-bottom: 6px;">3. Blended Overlay ({int(alpha*100)}%)</div>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)

    # Interactive Class Legend & Distribution
    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 700; color: #F1F5F9; margin-bottom: 10px;">🏷️ Class Distribution & Pixel Proportions</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for idx, c_stat in enumerate(dist["classes"]):
        with cols[idx]:
            color = c_stat["color"]
            st.markdown(f"""
            <div style="background: rgba(18, 24, 38, 0.7); border-top: 4px solid {color}; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-weight: 700; font-size: 0.9rem; color: #F8FAFC;">{c_stat['name']}</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: {color}; margin: 4px 0; font-family: monospace;">
                    {c_stat['pct_valid']}%
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8;">{c_stat['pixel_count']:,} px</div>
                {f"<div style='font-size: 0.75rem; color: #38BDF8; margin-top: 2px;'>{c_stat['area_ha']} ha ({c_stat['area_km2']} km²)</div>" if c_stat.get('area_ha') is not None else ""}
            </div>
            """, unsafe_allow_html=True)

    # GIS & Image Downloads Section
    st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 700; color: #F1F5F9; margin-bottom: 8px;">💾 Export Predictions & GIS Layers</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)

    # 1. Colored Mask PNG
    colored_pil = Image.fromarray(color_mask)
    buf_colored = io.BytesIO()
    colored_pil.save(buf_colored, format="PNG")
    with d1:
        st.download_button(
            "📥 Colored Mask (PNG)",
            data=buf_colored.getvalue(),
            file_name=f"pred_colored_{bundle.filename.split('.')[0]}_{model_info['id']}.png",
            mime="image/png",
            use_container_width=True
        )

    # 2. Class ID Mask PNG
    mask_pil = Image.fromarray(pred_mask)
    buf_mask = io.BytesIO()
    mask_pil.save(buf_mask, format="PNG")
    with d2:
        st.download_button(
            "📥 Class-ID Mask (PNG)",
            data=buf_mask.getvalue(),
            file_name=f"pred_class_id_{bundle.filename.split('.')[0]}_{model_info['id']}.png",
            mime="image/png",
            use_container_width=True,
            help="Raw 8-bit integer class indices (0=Water, 1=Veg, 2=Barren, 3=Built-up, 4=Roads, 255=Ignore)"
        )

    # 3. Georeferenced GeoTIFF
    with d3:
        if bundle.is_georeferenced and bundle.crs and bundle.transform:
            tif_out_path = os.path.join(OUTPUTS_DIR, f"geotiff_{bundle.filename.split('.')[0]}_{model_info['id']}.tif")
            export_geotiff(pred_mask, tif_out_path, bundle.crs, bundle.transform, nodata=255)
            with open(tif_out_path, "rb") as f_tif:
                tif_bytes = f_tif.read()
            st.download_button(
                "🗺️ GeoTIFF (EPSG Preserved)",
                data=tif_bytes,
                file_name=f"pred_georeferenced_{bundle.filename.split('.')[0]}_{model_info['id']}.tif",
                mime="image/tiff",
                use_container_width=True,
                help=f"Includes valid CRS ({bundle.crs}) and affine geotransform for QGIS/ArcGIS."
            )
        else:
            st.button(
                "🗺️ GeoTIFF (Not Georeferenced)",
                disabled=True,
                use_container_width=True,
                help="GeoTIFF export with coordinates is disabled because uploaded image is an ordinary PNG/JPEG without spatial CRS."
            )

    # 4. JSON Summary
    summary_data = {
        "source_file": bundle.filename,
        "model_id": model_info["id"],
        "model_name": model_info["display_name"],
        "architecture": model_info["architecture"],
        "inference_duration_sec": round(dur, 4),
        "device": device,
        "is_georeferenced": bundle.is_georeferenced,
        "crs": bundle.crs,
        "dimensions": f"{bundle.width}x{bundle.height}",
        "class_distribution": dist["classes"],
        "total_pixels": dist["total_pixels"],
        "valid_pixels": dist["valid_pixels"]
    }
    json_bytes = json.dumps(summary_data, indent=2).encode("utf-8")
    with d4:
        st.download_button(
            "📄 Prediction Summary (JSON)",
            data=json_bytes,
            file_name=f"summary_{bundle.filename.split('.')[0]}_{model_info['id']}.json",
            mime="application/json",
            use_container_width=True
        )

def render_side_by_side(results: list, bundle: ImageInputBundle):
    st.markdown("""
    <div style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 14px;">
        Comparing predictions across selected models with aligned visual palettes.
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(results))
    for idx, res in enumerate(results):
        with cols[idx]:
            m = res["model_info"]
            st.markdown(f"**{m['display_name']}**")
            st.image(res["color_mask"], caption=f"Latency: {res['duration_sec']*1000.0:.1f} ms", use_container_width=True)

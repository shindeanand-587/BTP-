"""
ui/prediction_view.py (v2 — Survey-Grade Geospatial Intelligence & Feature Refinement)
-------------------------------------------------------------------------------------
Transforms raw deep-learning segmentation masks into directly actionable, survey-ready deliverables:
1. Universal Imagery Ingestion & Quick-Load Benchmark Presets.
2. Automated Image Sanity Check & Preliminary Fitness Engine.
3. 6 Next-Generation Actionable Perspectives:
   - 🛣️ 1. Road Network & Centerlines (Gap-closing across shadows, Major Arterials vs Minor Streets)
   - 🏢 2. Building Footprints & Mumbai Settlement Typology (Orthogonal walls, Centroids, 4 Typologies)
   - 🌊 3. Water Shorelines & 50m Coastal Regulation Zone (CRZ) Hazard Buffer
   - 📋 4. Field Ground Survey Checklist (Prioritized GPS work order for field surveyors)
   - ↔️ 5. Interactive Before/After Curtain Swipe (Real-Time Comparison Slider)
   - 🌐 6. Full Scene Overview (Standard multi-class blend & donut chart)
4. Diagrammatic Analytics Dashboard with Donut Charts & Area Metrics.
5. 1-Click Multi-Format Export Bar (PNG Overlays, GeoTIFF, Intelligence JSON).
"""

import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageFilter, ImageDraw
Image.MAX_IMAGE_PIXELS = None

from model_registry import get_all_models, get_ready_models, get_model_by_id, CLASS_MAPPING
from inference import get_adapter
from inference.geospatial_io import (
    load_image_from_bytes,
    create_color_mask,
    create_blended_overlay,
    export_prediction_geotiff,
    ImageInputBundle
)
from inference.postprocessing import compute_class_distribution, render_confidence_map
from inference.image_diagnostic import analyze_image_characteristics
from inference.urban_analytics import (
    extract_actionable_road_corridors,
    extract_actionable_building_footprints,
    extract_actionable_water_shoreline,
    extract_actionable_vegetation_canopy,
    extract_actionable_barren_parcels,
    generate_field_survey_checklist,
    compute_settlement_subclasses,
    create_class_spotlight_overlay,
    compute_urban_planning_inferences,
    create_curtain_swipe_html,
    MUMBAI_SUBCLASSES
)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")

def prepare_smooth_display_image(img_arr: Optional[np.ndarray], target_size: int = 700, max_dim: int = 1536) -> Optional[np.ndarray]:
    """Prepares imagery for crisp web display without pixelation."""
    if img_arr is None:
        return None
    h, w = img_arr.shape[:2]
    
    if max(h, w) < 550:
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = Image.fromarray(img_arr)
        return np.array(pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS))
    elif max(h, w) > max_dim:
        step = int(np.ceil(max(h, w) / max_dim))
        return img_arr[::step, ::step].copy()
    
    return img_arr

def render_donut_chart(distribution: Dict[str, Any]):
    """Renders a modern circular donut chart with high-legibility labels."""
    classes = distribution["classes"]
    labels = []
    sizes = []
    colors = []

    for c in classes:
        if c["pct_valid"] > 0.05:
            labels.append(c["name"])
            sizes.append(c["pct_valid"])
            colors.append(c["color"])

    if not sizes:
        sizes = [100.0]
        labels = ["No Data"]
        colors = ["#94A3B8"]

    fig, ax = plt.subplots(figsize=(4.4, 4.4), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.74,
        startangle=140,
        wedgeprops=dict(width=0.40, edgecolor="#FFFFFF", linewidth=3.0)
    )

    for autotext in autotexts:
        autotext.set_color("#FFFFFF")
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")

    ax.text(0, 0, "Land Cover\nCoverage", ha="center", va="center", color="#0F172A",
            fontsize=13, fontweight="bold", fontfamily="sans-serif")

    plt.tight_layout()
    return fig

def render_settlement_donut_chart(subclass_stats: List[Dict[str, Any]]):
    """Renders a specialized donut chart for Mumbai built-up settlement typologies."""
    sizes = [s["pct_of_builtup"] for s in subclass_stats if s["pct_of_builtup"] > 0.1]
    colors = [s["color"] for s in subclass_stats if s["pct_of_builtup"] > 0.1]

    if not sizes:
        sizes = [100.0]
        colors = ["#94A3B8"]

    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.74,
        startangle=130,
        wedgeprops=dict(width=0.42, edgecolor="#FFFFFF", linewidth=2.5)
    )

    for autotext in autotexts:
        autotext.set_color("#FFFFFF")
        autotext.set_fontsize(11.5)
        autotext.set_fontweight("bold")

    ax.text(0, 0, "Settlement\nTypology", ha="center", va="center", color="#0F172A",
            fontsize=12, fontweight="bold", fontfamily="sans-serif")

    plt.tight_layout()
    return fig

def render_prediction_page():
    # -------------------------------------------------------------------------
    # SECTION 1: Universal Imagery Ingestion & Presets
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <div class="section-header" style="margin-top: 0; margin-bottom: 4px;">
            🛰️ Step 1: Ingest Satellite Scene or Aerial Imagery
        </div>
        <div style="font-size: 1.05rem; color: #64748B; margin-bottom: 16px;">
            Upload high-resolution satellite imagery or choose an instant test scene below.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_samples = st.columns([1.4, 1.6], gap="large")

    with col_upload:
        st.markdown('<div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">📁 Upload File from Computer:</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Satellite Scene or Vector Shapefile (GeoTIFF, PNG, JPG, ZIP, SHP, GeoJSON):",
            type=["tif", "tiff", "png", "jpg", "jpeg", "webp", "bmp", "zip", "shp", "dbf", "geojson", "json"],
            help="Supports high-resolution GeoTIFFs, standard aerial imagery, ESRI Shapefiles (.shp, or .zip with .shp/.shx/.dbf), and GeoJSON vectors.",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    file_bytes = uploaded_file.read()
                    bundle = load_image_from_bytes(file_bytes, uploaded_file.name)
                    st.session_state["current_bundle"] = bundle
                    st.session_state["last_uploaded_name"] = uploaded_file.name
                    st.session_state.pop("last_prediction_results", None)
                    st.session_state.pop("diagnostic_cache", None)
                    st.rerun()

    with col_samples:
        st.markdown('<div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">⚡ Quick-Load Test Presets:</div>', unsafe_allow_html=True)
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            if st.button("🛰️ WV-3 (8-Band 1.24m)", use_container_width=True, help="8-channel high-res WorldView-3 scene"):
                sample_path = os.path.join(SAMPLE_DIR, "sample_wv3_8band.tif")
                with open(sample_path, "rb") as f:
                    bundle = load_image_from_bytes(f.read(), "sample_wv3_8band.tif")
                    st.session_state["current_bundle"] = bundle
                    st.session_state["last_uploaded_name"] = "sample_wv3_8band.tif"
                    st.session_state.pop("last_prediction_results", None)
                    st.session_state.pop("diagnostic_cache", None)
                    st.rerun()

            if st.button("🗺️ Cartosat (PAN 2.5m)", use_container_width=True, help="Single-band ISRO panchromatic scene"):
                sample_path = os.path.join(SAMPLE_DIR, "sample_cartosat_pan.tif")
                with open(sample_path, "rb") as f:
                    bundle = load_image_from_bytes(f.read(), "sample_cartosat_pan.tif")
                    st.session_state["current_bundle"] = bundle
                    st.session_state["last_uploaded_name"] = "sample_cartosat_pan.tif"
                    st.session_state.pop("last_prediction_results", None)
                    st.session_state.pop("diagnostic_cache", None)
                    st.rerun()

        with c_s2:
            if st.button("📷 Urban Aerial (RGB)", use_container_width=True, help="Standard 3-band full-color photography"):
                sample_path = os.path.join(SAMPLE_DIR, "sample_urban_rgb.jpg")
                with open(sample_path, "rb") as f:
                    bundle = load_image_from_bytes(f.read(), "sample_urban_rgb.jpg")
                    st.session_state["current_bundle"] = bundle
                    st.session_state["last_uploaded_name"] = "sample_urban_rgb.jpg"
                    st.session_state.pop("last_prediction_results", None)
                    st.session_state.pop("diagnostic_cache", None)
                    st.rerun()

            if st.button("🌊 Coastal & Water Tile", use_container_width=True, help="Coastal tile with fine creeks and water bodies"):
                sample_path = os.path.join(SAMPLE_DIR, "sample_panchromatic_gray.png")
                with open(sample_path, "rb") as f:
                    bundle = load_image_from_bytes(f.read(), "sample_coastal_gray.png")
                    st.session_state["current_bundle"] = bundle
                    st.session_state["last_uploaded_name"] = "sample_coastal_gray.png"
                    st.session_state.pop("last_prediction_results", None)
                    st.session_state.pop("diagnostic_cache", None)
                    st.rerun()

    # Empty State when no image is loaded
    if "current_bundle" not in st.session_state:
        st.markdown("""
        <div style="background: #FFFFFF; border: 2px dashed #CBD5E1; border-radius: 14px; padding: 42px 32px; text-align: center; margin-top: 24px; box-shadow: 0 2px 8px rgba(15,23,42,0.03);">
            <div style="font-size: 2.8rem; margin-bottom: 12px;">🛰️</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;">
                Ready for Imagery Ingestion
            </div>
            <div style="font-size: 1.1rem; color: #64748B; max-width: 620px; margin: 0 auto; line-height: 1.55;">
                Upload a GeoTIFF satellite scene or standard aerial photograph above, or click one of the <b>Quick-Load Test Presets</b> to run instant automated feature extraction.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    bundle: ImageInputBundle = st.session_state["current_bundle"]

    # -------------------------------------------------------------------------
    # SECTION 2: Automated Preliminary Image Diagnostic (Sanity Check)
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">🔍 Step 2: Automated Preliminary Image Diagnostic & Fitness Scoring</div>', unsafe_allow_html=True)

    if "diagnostic_cache" not in st.session_state:
        st.session_state["diagnostic_cache"] = analyze_image_characteristics(bundle)

    diag = st.session_state["diagnostic_cache"]
    rec = diag["recommendation"]

    # Visual Diagnostic Banner
    geo_badge = f'<span class="badge badge-ready">🌐 Georeferenced ({diag["crs"]})</span>' if diag["is_georeferenced"] else '<span class="badge badge-neutral">🖼️ Cartesian Image Grid</span>'
    res_badge = f'<span class="badge badge-sensor">📐 GSD: {diag["gsd_m"]:.2f}m</span>' if diag["gsd_m"] else '<span class="badge badge-neutral">Pixel Grid</span>'

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 14px; padding: 22px 26px; margin-bottom: 24px; box-shadow: 0 3px 12px rgba(15,23,42,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; border-bottom: 1.5px solid #F1F5F9; padding-bottom: 14px; margin-bottom: 18px;">
            <div>
                <span style="font-weight: 800; color: #0F172A; font-size: 1.30rem;">📄 {bundle.filename}</span>
                <span style="color: #64748B; font-size: 1.05rem; margin-left: 14px; font-weight: 600;">{diag['width']} × {diag['height']} px • {diag['pixels_mp']} Megapixels • {diag['channels']} Band{'s' if diag['channels'] > 1 else ''}</span>
            </div>
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                <span class="badge" style="background: #EFF6FF; color: #1D4ED8; font-weight: 700; border: 1.5px solid #BFDBFE;">{diag['typology_badge']}</span>
                {geo_badge}
                {res_badge}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;">
            <div style="background: #F8FAFC; padding: 14px 18px; border-radius: 12px; border-left: 5px solid #2563EB; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;">
                <div style="font-size: 0.88rem; color: #64748B; text-transform: uppercase; font-weight: 800; letter-spacing: 0.04em;">Dynamic Range & Bit Depth</div>
                <div style="font-size: 1.12rem; color: #0F172A; font-weight: 800; margin-top: 4px;">{diag['bit_depth_label']}</div>
            </div>
            <div style="background: #F8FAFC; padding: 14px 18px; border-radius: 12px; border-left: 5px solid #059669; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;">
                <div style="font-size: 0.88rem; color: #64748B; text-transform: uppercase; font-weight: 800; letter-spacing: 0.04em;">Edge & Urban Grid Density</div>
                <div style="font-size: 1.12rem; color: #0F172A; font-weight: 800; margin-top: 4px;">Score: {diag['edge_density_score']:.2f} / 1.00 ({'Dense Urban Core' if diag['edge_density_score'] > 0.5 else 'Sparse / Open Terrain'})</div>
            </div>
            <div style="background: #F8FAFC; padding: 14px 18px; border-radius: 12px; border-left: 5px solid #D97706; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;">
                <div style="font-size: 0.88rem; color: #64748B; text-transform: uppercase; font-weight: 800; letter-spacing: 0.04em;">Water Absorption Index</div>
                <div style="font-size: 1.12rem; color: #0F172A; font-weight: 800; margin-top: 4px;">Score: {diag['water_signature_score']:.2f} / 1.00 ({'Coastal / Creek Detected' if diag['water_signature_score'] > 0.3 else 'Low Water Footprint'})</div>
            </div>
            <div style="background: #F8FAFC; padding: 14px 18px; border-radius: 12px; border-left: 5px solid #7C3AED; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;">
                <div style="font-size: 0.88rem; color: #64748B; text-transform: uppercase; font-weight: 800; letter-spacing: 0.04em;">Estimated Typology</div>
                <div style="font-size: 1.12rem; color: #0F172A; font-weight: 800; margin-top: 4px;">{diag['scene_typology']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 3: Dynamic Model Selection Bar
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">🤖 Step 3: Select Model & Execute Feature Extraction</div>', unsafe_allow_html=True)

    col_sel_mode, col_sel_model, col_run_btn = st.columns([1.1, 2.3, 1.2], gap="medium")

    with col_sel_mode:
        sel_mode = st.radio(
            "Selection Method:",
            ["⚡ AUTO-SELECT (Image Diagnostic)", "🛠️ MANUAL SELECTION"],
            index=0,
            label_visibility="collapsed"
        )

    all_models = get_all_models()

    if "AUTO-SELECT" in sel_mode:
        selected_model_id = rec["model_id"]
        with col_sel_model:
            st.markdown(f"""
            <div style="background: #EFF6FF; border: 1.5px solid #93C5FD; border-radius: 12px; padding: 14px 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 800; color: #1D4ED8; font-size: 1.15rem;">⭐ Recommended: {rec['display_name']}</span>
                    <span class="badge badge-ready" style="font-size: 0.95rem;">{rec['fitness_score']:.0f}% Fitness Match</span>
                </div>
                <div style="font-size: 1.0rem; color: #334155; margin-top: 4px; font-weight: 600;">
                    💡 <b>Diagnostic Rationale:</b> {rec['rationale']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with col_sel_model:
            model_options = {
                m["id"]: f"{m.get('role_badge', '🤖')} {m['display_name']} ({m['architecture']}) — {m.get('status', 'Ready')}"
                for m in all_models
            }
            def_idx = list(model_options.keys()).index(rec["model_id"]) if rec["model_id"] in model_options else 0
            selected_model_id = st.selectbox(
                "Choose Model to Execute:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=def_idx,
                label_visibility="collapsed"
            )

    with col_run_btn:
        run_prediction = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Inference Execution
    # -------------------------------------------------------------------------
    if run_prediction:
        m_info = get_model_by_id(selected_model_id)
        prog_bar = st.progress(10, text=f"Initializing {m_info['display_name']}...")

        try:
            adapter = get_adapter(selected_model_id)
            prog_bar.progress(35, text=f"Executing feature extraction on {bundle.width} × {bundle.height} pixels...")
            pred_mask, conf_map, duration, warnings = adapter.run_inference(
                bundle,
                allow_rgb_adaptation=True,
                tile_size=512,
                overlap=64
            )
            prog_bar.progress(70, text="Synthesizing multi-class overlays & physical areas...")

            color_mask = create_color_mask(pred_mask)
            dist = compute_class_distribution(pred_mask, bundle.resolution)
            conf_rgb = render_confidence_map(conf_map) if conf_map is not None else None

            prog_bar.progress(85, text="Refining survey-grade features and vector centerlines...")
            
            # Compute Actionable Survey-Grade Layers
            gsd_val = bundle.resolution if bundle.resolution else 1.24

            v_skel = getattr(bundle, "vector_centerlines", None)
            v_tot_km = getattr(bundle, "total_vector_length_km", None)
            is_vec = getattr(bundle, "is_vector", False)
            road_corridors = extract_actionable_road_corridors(
                pred_mask == 4,
                gsd_m=gsd_val,
                bridge_shadow_gaps=(not is_vec),
                centerline_override=v_skel,
                total_length_km_override=v_tot_km
            )
            building_footprints = extract_actionable_building_footprints(pred_mask == 3, gsd_m=gsd_val)
            water_shoreline = extract_actionable_water_shoreline(pred_mask == 0, pred_mask == 3, gsd_m=gsd_val)
            veg_canopy = extract_actionable_vegetation_canopy(pred_mask == 1, pred_mask == 4, gsd_m=gsd_val)
            barren_parcels = extract_actionable_barren_parcels(pred_mask == 2, pred_mask == 4, gsd_m=gsd_val)
            settlement_res = compute_settlement_subclasses(bundle.rgb_preview, pred_mask == 3, gsd_m=gsd_val)
            survey_checklist = generate_field_survey_checklist(
                pred_mask, 
                road_corridors, 
                building_footprints, 
                water_shoreline, 
                veg_intel=veg_canopy, 
                barren_intel=barren_parcels, 
                gsd_m=gsd_val
            )
            urban_inferences = compute_urban_planning_inferences(dist)
            prob_map = getattr(adapter, "last_prob_map", None)

            st.session_state["last_prediction_results"] = {
                "model_info": m_info,
                "pred_mask": pred_mask,
                "color_mask": color_mask,
                "conf_map": conf_map,
                "conf_rgb": conf_rgb,
                "prob_map": prob_map,
                "distribution": dist,
                "duration": duration,
                "warnings": warnings,
                "bundle": bundle,
                "road_corridors": road_corridors,
                "building_footprints": building_footprints,
                "water_shoreline": water_shoreline,
                "veg_canopy": veg_canopy,
                "barren_parcels": barren_parcels,
                "settlement_res": settlement_res,
                "survey_checklist": survey_checklist,
                "urban_inferences": urban_inferences
            }
            prog_bar.progress(100, text="Prediction complete!")
            time.sleep(0.3)
            prog_bar.empty()
            st.rerun()

        except Exception as e:
            prog_bar.empty()
            st.error(f"❌ Inference encountered an error: {str(e)}")
            st.info("💡 Try selecting another model or verifying the uploaded image format.")

    # -------------------------------------------------------------------------
    # SECTION 4: Survey-Grade Geospatial Intelligence & Feature Refinement
    # -------------------------------------------------------------------------
    if "last_prediction_results" in st.session_state:
        res = st.session_state["last_prediction_results"]
        m_info = res["model_info"]
        dist = res["distribution"]
        dur = res["duration"]
        warnings = res["warnings"]

        # Ensure prob_map is always present and valid for sensitivity controls
        prob_map = res.get("prob_map")
        if prob_map is None or getattr(prob_map, "shape", [0])[0] != 5:
            pm = res["pred_mask"]
            ph, pw = pm.shape
            prob_map = np.full((5, ph, pw), 0.05, dtype=np.float32)
            conf = res.get("conf_map")
            if conf is None:
                conf = np.ones((ph, pw), dtype=np.float32) * 0.85
            for c in range(5):
                prob_map[c][pm == c] = np.clip(conf[pm == c], 0.60, 0.99)
            road_bin = (pm == 4)
            if road_bin.any():
                pil_r = Image.fromarray((road_bin * 255).astype(np.uint8))
                d1 = np.array(pil_r.filter(ImageFilter.MaxFilter(3)), dtype=bool) & (~road_bin)
                d2 = np.array(pil_r.filter(ImageFilter.MaxFilter(5)), dtype=bool) & (~road_bin) & (~d1)
                prob_map[4][d1] = np.maximum(prob_map[4][d1], 0.35)
                prob_map[4][d2] = np.maximum(prob_map[4][d2], 0.20)
            res["prob_map"] = prob_map

        # Auto-refresh road corridors if cached schema is from earlier version
        b_res = res.get("bundle")
        gsd_val = getattr(b_res, "resolution", 1.24) if b_res else 1.24
        if not isinstance(gsd_val, (int, float, tuple)):
            gsd_val = 1.24
        road_corridors = res.get("road_corridors", {})
        if "arterial_junction_mask" not in road_corridors:
            v_skel = getattr(b_res, "vector_centerlines", None)
            v_tot_km = getattr(b_res, "total_vector_length_km", None)
            is_vec_b = getattr(b_res, "is_vector", False)
            road_corridors = extract_actionable_road_corridors(
                res["pred_mask"] == 4,
                gsd_m=gsd_val,
                bridge_shadow_gaps=(not is_vec_b),
                centerline_override=v_skel,
                total_length_km_override=v_tot_km
            )
            res["road_corridors"] = road_corridors

        building_footprints = res.get("building_footprints", {})
        water_shoreline = res.get("water_shoreline", {})
        veg_canopy = res.get("veg_canopy", {})
        barren_parcels = res.get("barren_parcels", {})
        settlement_res = res.get("settlement_res", {})
        survey_checklist = res.get("survey_checklist", [])
        urban_inferences = res.get("urban_inferences", {})

        st.markdown('<div class="section-header" style="margin-top: 28px;">📊 Step 4: Survey-Grade Geospatial Intelligence & Feature Refinement</div>', unsafe_allow_html=True)

        if warnings:
            for w in warnings:
                st.warning(f"ℹ️ {w}")

        # Top KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #2563EB;">
                <div class="kpi-val" style="color: #1D4ED8;">{dur:.3f} s</div>
                <div class="kpi-label">⚡ Execution Latency</div>
                <div class="kpi-sub">~{dur*1000/max(1, (bundle.width*bundle.height/262144)):.1f} ms / 512px tile</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            area_str = f"{sum([c['area_km2'] for c in dist['classes'] if c['area_km2']]):.2f} km²" if dist['has_physical_area'] else f"{dist['total_pixels']/1_000_000:.2f} MP"
            area_sub = f"{sum([c['area_ha'] for c in dist['classes'] if c['area_ha']]):.1f} Hectares" if dist['has_physical_area'] else "Cartesian Pixels"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #059669;">
                <div class="kpi-val" style="color: #059669;">{area_str}</div>
                <div class="kpi-label">📐 Ground Footprint</div>
                <div class="kpi-sub">{area_sub}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            built_item = [c for c in dist["classes"] if c["class_id"] == 3][0]
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #E11D48;">
                <div class="kpi-val" style="color: #E11D48;">{built_item['pct_valid']:.1f}%</div>
                <div class="kpi-label">🏢 Built-up Coverage</div>
                <div class="kpi-sub">{building_footprints.get('building_count', 0)} Discrete Structures</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            art_len = road_corridors.get("arterial_length_km", 0.0)
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 5px solid #D97706;">
                <div class="kpi-val" style="color: #D97706;">{art_len:.2f} km</div>
                <div class="kpi-label">🛣️ Major Arterials</div>
                <div class="kpi-sub">{road_corridors.get('continuity_score', 0):.1f}% Network Continuity</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 6 Actionable Survey & Planning Perspectives
        # ---------------------------------------------------------------------
        st.markdown('<div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;">🎯 Choose Survey & Inspection Perspective:</div>', unsafe_allow_html=True)
        
        perspective = st.radio(
            "Perspective:",
            [
                "🛣️ Transport Corridors & Road Centerlines",
                "🏢 Building Footprints & Structural Envelopes",
                "🌳 Urban Vegetation & Green Canopy Reserves",
                "🏗️ Vacant Land & Developable Parcels",
                "🌊 Water Shoreline & 50m Coastal Buffer (CRZ)",
                "📋 Field Ground Survey Checklist (Work Order)",
                "↔️ Before/After Curtain Swipe",
                "🌐 Full Multi-Class Overview"
            ],
            horizontal=True,
            label_visibility="collapsed"
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        display_rgb = prepare_smooth_display_image(bundle.rgb_preview)
        display_color_mask = prepare_smooth_display_image(res["color_mask"])

        # =====================================================================
        # PERSPECTIVE 1: Road Network & Centerlines (User's Core Requirement)
        # =====================================================================
        if perspective == "🛣️ Transport Corridors & Road Centerlines":
            col_v, col_m = st.columns([1.18, 1.0], gap="large")

            prob_map = res.get("prob_map")
            has_probs = (prob_map is not None and getattr(prob_map, "shape", [0])[0] == 5)

            with col_v:
                c_filter, c_skel = st.columns([1.5, 1.2])
                with c_filter:
                    road_view_tier = st.radio(
                        "Road Filter Tier:",
                        ["🌟 Major Arterials (High Confidence)", "🌐 Complete Network (All Streets & Alleys)"],
                        horizontal=False
                    )
                    road_prob_thresh = st.slider(
                        "🎚️ Road Confidence Threshold (%):",
                        min_value=10,
                        max_value=60,
                        value=25,
                        step=5,
                        help="Confidence acceptance cutoff. 20-25% is calibrated for optical satellite imagery and airport runways. Higher values (>50%) require strict certainty and may filter out secondary lanes."
                    ) / 100.0

                with c_skel:
                    show_centerlines = st.checkbox("Draw Vector Centerlines", value=True)
                    show_junctions = st.checkbox("Highlight Intersection Nodes", value=False, help="Enables pinpoint circular markers at true 3-way and 4-way intersections.")
                    dim_bg = st.checkbox("Dim Satellite Background", value=False, help="Uncheck to view full optical scene with maximum contrast.")
                    road_intensity = st.slider("Road Highlight Intensity (%):", min_value=0, max_value=100, value=35, step=5) / 100.0

                # Compute active road corridors dynamically adapting to sensitivity slider
                gsd_val = getattr(bundle, "resolution", 1.24) or 1.24
                if not isinstance(gsd_val, (int, float, tuple)):
                    gsd_val = 1.24
                v_skel = getattr(bundle, "vector_centerlines", None)
                v_tot_km = getattr(bundle, "total_vector_length_km", None)
                is_vec_active = getattr(bundle, "is_vector", False)
                road_prob = prob_map[4]
                dynamic_road_mask = (road_prob >= road_prob_thresh)
                active_road_corridors = extract_actionable_road_corridors(
                    dynamic_road_mask,
                    gsd_m=gsd_val,
                    bridge_shadow_gaps=(not is_vec_active),
                    centerline_override=v_skel,
                    total_length_km_override=v_tot_km
                )
                res["road_corridors"] = active_road_corridors

                # Safety tip if high threshold filtered out roads
                if not active_road_corridors.get("has_roads") and (road_prob.max() > 0.10):
                    st.warning(f"⚠️ At {int(road_prob_thresh*100)}% confidence threshold, faint corridors were filtered out. Lower the slider to 20–25% to display the full network.")

                # Determine active road mask based on user filter
                if "Major Arterials" in road_view_tier and active_road_corridors.get("has_roads"):
                    active_road_mask = active_road_corridors["arterial_mask"]
                    active_centerline = active_road_corridors["arterial_centerline"]
                else:
                    active_road_mask = active_road_corridors.get("working_mask", dynamic_road_mask)
                    active_centerline = active_road_corridors.get("centerline_mask", np.zeros_like(active_road_mask))

                # Ensure active_centerline never drops to zero if complete network has lines
                if not active_centerline.any() and active_road_corridors.get("centerline_mask") is not None:
                    active_centerline = active_road_corridors["centerline_mask"]

                smooth_road_mask = prepare_smooth_display_image((active_road_mask * 255).astype(np.uint8)) > 128

                # Render road spotlight with optional background dimming
                road_spotlight = create_class_spotlight_overlay(
                    display_rgb,
                    smooth_road_mask,
                    highlight_rgb=(250, 204, 21), # Vibrant Gold-Amber
                    opacity=road_intensity,
                    dim_background=dim_bg,
                    draw_contour=True
                )

                # Draw high-visibility vector centerlines with crisp white casing and vivid crimson medial axis
                if show_centerlines:
                    h_disp, w_disp = road_spotlight.shape[:2]
                    v_polys = getattr(bundle, "vector_polylines", None)
                    if v_polys:
                        pil_spot = Image.fromarray(road_spotlight)
                        draw_vec = ImageDraw.Draw(pil_spot)
                        h_b, w_b = bundle.height, bundle.width
                        for l_pts in v_polys:
                            if len(l_pts) < 2:
                                continue
                            disp_pts = [
                                (int(round(x * (w_disp - 1) / max(1, w_b - 1))),
                                 int(round(y * (h_disp - 1) / max(1, h_b - 1))))
                                for x, y in l_pts
                            ]
                            # Outer white casing (4px) + Vivid crimson medial axis (2px)
                            draw_vec.line(disp_pts, fill=(255, 255, 255), width=4)
                            draw_vec.line(disp_pts, fill=(220, 38, 38), width=2)
                        road_spotlight = np.array(pil_spot)
                    elif active_centerline.any():
                        # Nearest-neighbor resize to display canvas so 1px centerlines are never blurred or lost!
                        pil_skel = Image.fromarray((active_centerline * 255).astype(np.uint8)).resize(
                            (w_disp, h_disp), resample=Image.Resampling.NEAREST
                        )
                        thin_core = np.array(pil_skel, dtype=bool)
                        thick = np.array(pil_skel.filter(ImageFilter.MaxFilter(3)), dtype=bool)
                        halo = thick & (~thin_core)
                        road_spotlight[halo] = [255, 255, 255] # Clean 1px white casing halo
                        road_spotlight[thin_core] = [220, 38, 38] # Exact 1px vivid crimson medial axis

                # Draw pinpoint junction nodes cleanly without obscuring roads
                if show_junctions:
                    if "Major Arterials" in road_view_tier:
                        junc_m = active_road_corridors.get("arterial_junction_mask", active_road_corridors.get("junction_mask"))
                    else:
                        junc_m = active_road_corridors.get("junction_mask")

                    if junc_m is not None and junc_m.any():
                        pil_spot = Image.fromarray(road_spotlight)
                        draw_junc = ImageDraw.Draw(pil_spot)
                        h_disp, w_disp = road_spotlight.shape[:2]
                        h_orig, w_orig = junc_m.shape[:2]
                        jy_arr, jx_arr = np.where(junc_m)
                        for jy, jx in zip(jy_arr, jx_arr):
                            dy = int(round(jy * h_disp / h_orig))
                            dx = int(round(jx * w_disp / w_orig))
                            # Crisp circular pinpoint: 5px white outer ring, 3px royal blue core
                            draw_junc.ellipse([(dx - 5, dy - 5), (dx + 5, dy + 5)], fill=(255, 255, 255), outline=(255, 255, 255))
                            draw_junc.ellipse([(dx - 3, dy - 3), (dx + 3, dy + 3)], fill=(37, 99, 235), outline=(37, 99, 235))
                        road_spotlight = np.array(pil_spot)

                st.image(road_spotlight, caption=f"🛣️ Continuous Road Corridors ({int(road_prob_thresh*100)}% Sensitivity) with Centerline Vectors", use_container_width=True)

                st.markdown("""
                <div style="display: flex; gap: 12px; justify-content: center; align-items: center; flex-wrap: wrap; margin-top: 10px; background: #FFFFFF; padding: 10px 16px; border-radius: 10px; border: 1.5px solid #E2E8F0;">
                    <span class="legend-pill"><span class="color-dot" style="background: #FACC15;"></span> Road Surface Strip</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #DC2626;"></span> Medial Centerline Vector</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #2563EB;"></span> Intersection Node</span>
                </div>
                """, unsafe_allow_html=True)

            with col_m:
                st.markdown('<div style="font-size: 1.30rem; font-weight: 800; color: #0F172A; margin-bottom: 12px;">🛣️ Actionable Transportation Infrastructure</div>', unsafe_allow_html=True)

                art_l = active_road_corridors.get("arterial_length_km", 0.0)
                tot_l = active_road_corridors.get("total_length_km", 0.0)
                dens = active_road_corridors.get("road_density_km_per_km2", 0.0)
                cont = active_road_corridors.get("continuity_score", 0.0)

                st.markdown(f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #D97706;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Major Arterials</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{art_l:.2f} km</div>
                        <div style="font-size: 0.82rem; color: #059669; font-weight: 600; margin-top: 2px;">High-Confidence Deliverable</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #2563EB;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Corridor Density</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{dens:.1f} <span style="font-size: 0.95rem;">km/km²</span></div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">Linear network per km²</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #059669;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Continuity Score</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #059669; font-family: 'JetBrains Mono', monospace;">{cont:.1f}%</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">Gap-bridging active</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #7C3AED;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Total Network</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{tot_l:.2f} km</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">Arterials + minor lanes</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;">📋 Road Hierarchy & Survey Reliability Breakdown:</div>', unsafe_allow_html=True)
                
                for c in active_road_corridors.get("corridor_table", []):
                    st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <span style="font-weight: 800; color: #0F172A; font-size: 1.05rem;">{c['tier']}</span>
                            <span style="font-size: 1.25rem; font-weight: 800; color: #1D4ED8; font-family: 'JetBrains Mono', monospace;">{c['length_km']:.2f} km ({c['pct_of_network']:.0f}%)</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #475569; margin-top: 4px;">
                            • <b>Est. Corridor Width:</b> ~{c['avg_width_m']:.1f} meters • <b>Survey Confidence:</b> {c['confidence']}
                        </div>
                        <div style="font-size: 0.85rem; color: #059669; font-weight: 600; margin-top: 2px;">
                            ✓ {c['survey_status']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # =====================================================================
        # PERSPECTIVE 2: Building Footprints & Structural Envelopes
        # =====================================================================
        elif perspective == "🏢 Building Footprints & Structural Envelopes":
            col_v, col_m = st.columns([1.18, 1.0], gap="large")

            with col_v:
                b_mode = st.radio(
                    "Built-up Visual Mode:",
                    [
                        "📐 Regularized Building Footprints (Orthogonal Vector Envelopes)",
                        "🏙️ 4-Class Mumbai Settlement Typology Map",
                        "🔦 Structural X-Ray Spotlight"
                    ],
                    horizontal=False
                )

                builtup_binary = (res["pred_mask"] == 3)
                smooth_builtup_binary = prepare_smooth_display_image((builtup_binary * 255).astype(np.uint8)) > 128

                if "Regularized Building Footprints" in b_mode and building_footprints.get("has_buildings"):
                    # Overlay crisp rectangular building envelopes with centroids
                    fp_overlay = prepare_smooth_display_image(building_footprints["footprint_overlay"])
                    blended_fp = create_blended_overlay(display_rgb, fp_overlay, alpha=0.90)
                    st.image(blended_fp, caption=f"📐 Discrete Building Footprints ({building_footprints['building_count']} structures delineated with centroids)", use_container_width=True)
                elif "Settlement Typology" in b_mode and settlement_res.get("has_data"):
                    sub_color = prepare_smooth_display_image(settlement_res["color_overlay"])
                    sub_blended = create_blended_overlay(display_rgb, sub_color, alpha=0.65)
                    st.image(sub_blended, caption="🏙️ Decomposed Mumbai Urban Settlement Typologies", use_container_width=True)
                else:
                    spot = create_class_spotlight_overlay(display_rgb, smooth_builtup_binary, highlight_rgb=(231, 19, 69), opacity=0.75)
                    st.image(spot, caption="🏢 Built-Up Structures Spotlight with Contour Outlines", use_container_width=True)

                st.markdown("""
                <div style="display: flex; gap: 10px; justify-content: center; align-items: center; flex-wrap: wrap; margin-top: 10px; background: #FFFFFF; padding: 10px 14px; border-radius: 10px; border: 1.5px solid #E2E8F0;">
                    <span class="legend-pill"><span class="color-dot" style="background: #FFFFFF; border: 1px solid #CBD5E1;"></span> Building Envelope Contour</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #E11D48;"></span> Structure Centroid Pin</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #F97316;"></span> Informal Slum Fabric</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #3B82F6;"></span> Commercial / High-Rise</span>
                </div>
                """, unsafe_allow_html=True)

            with col_m:
                st.markdown('<div style="font-size: 1.30rem; font-weight: 800; color: #0F172A; margin-bottom: 12px;">🏢 Structural Footprint Inventory</div>', unsafe_allow_html=True)

                b_cnt = building_footprints.get("building_count", 0)
                tot_roof = building_footprints.get("total_roof_area_ha", 0.0)

                st.markdown(f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #E11D48;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Delineated Structures</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{b_cnt:,} buildings</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #2563EB;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Total Roof Area</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{tot_roof:.2f} ha</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #0F172A; font-size: 1.0rem; margin-bottom: 6px;">Footprint Scale Distribution:</div>
                    <div style="font-size: 0.92rem; color: #334155;">
                        • <b>Small Footprints (< 150 m²):</b> {building_footprints.get('small_count', 0)} structures (Informal / houses)<br>
                        • <b>Medium Footprints (150–600 m²):</b> {building_footprints.get('medium_count', 0)} structures (Residential blocks)<br>
                        • <b>Large Footprints (> 600 m²):</b> {building_footprints.get('large_count', 0)} structures (Commercial towers / malls)
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if settlement_res.get("has_data"):
                    st.markdown('<div style="font-size: 1.10rem; font-weight: 800; color: #0F172A; margin: 12px 0 8px 0;">🏙️ Mumbai Settlement Fabric Breakdown:</div>', unsafe_allow_html=True)
                    for s in settlement_res["stats"]:
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 4px;">
                            <span><span style="color: {s['color']};">●</span> <b>{s['name']}</b></span>
                            <span style="font-family: monospace; font-weight: 700;">{s['pct_of_builtup']:.1f}% ({s['area_ha']:.2f} ha)</span>
                        </div>
                        """, unsafe_allow_html=True)

        # =====================================================================
        # PERSPECTIVE 3: Urban Vegetation & Green Canopy Reserves
        # =====================================================================
        elif perspective == "🌳 Urban Vegetation & Green Canopy Reserves":
            col_v, col_m = st.columns([1.18, 1.0], gap="large")

            with col_v:
                veg_mode = st.radio(
                    "Vegetation Display Filter:",
                    [
                        "🌳 Core Urban Parks & Green Lungs (> 400 m²)",
                        "🌿 Street Verge & Road Buffer Canopy (within 15m)",
                        "🍃 Complete Canopy Spotlight"
                    ],
                    horizontal=False
                )

                veg_binary = (res["pred_mask"] == 1)
                smooth_veg_binary = prepare_smooth_display_image((veg_binary * 255).astype(np.uint8)) > 128

                if "Core Urban Parks" in veg_mode and veg_canopy.get("has_vegetation"):
                    park_overlay = prepare_smooth_display_image(veg_canopy["park_overlay"])
                    spot = create_class_spotlight_overlay(display_rgb, smooth_veg_binary, highlight_rgb=(33, 140, 44), opacity=0.70)
                    if park_overlay is not None and park_overlay.any():
                        has_box = park_overlay.any(axis=-1)
                        spot[has_box] = park_overlay[has_box]
                    st.image(spot, caption=f"🌳 Delineated Urban Green Reserves ({veg_canopy.get('core_park_count', 0)} Core Parks Identified)", use_container_width=True)
                elif "Street Verge" in veg_mode and veg_canopy.get("has_vegetation"):
                    verge_mask = veg_canopy.get("verge_mask", np.zeros_like(veg_binary))
                    smooth_verge = prepare_smooth_display_image((verge_mask * 255).astype(np.uint8)) > 128
                    spot = create_class_spotlight_overlay(display_rgb, smooth_verge, highlight_rgb=(132, 204, 22), opacity=0.85)
                    st.image(spot, caption=f"🌿 Street Verge Green Canopy along Transportation Corridors", use_container_width=True)
                else:
                    spot = create_class_spotlight_overlay(display_rgb, smooth_veg_binary, highlight_rgb=(33, 140, 44), opacity=0.75)
                    st.image(spot, caption="🍃 Complete Vegetation Canopy Coverage Spotlight", use_container_width=True)

                st.markdown("""
                <div style="display: flex; gap: 10px; justify-content: center; align-items: center; flex-wrap: wrap; margin-top: 10px; background: #FFFFFF; padding: 10px 14px; border-radius: 10px; border: 1.5px solid #E2E8F0;">
                    <span class="legend-pill"><span class="color-dot" style="background: #10B981;"></span> Core Public Park Boundary</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #F59E0B;"></span> Park Centroid Pin</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #84CC16;"></span> Street Verge Tree Canopy</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #218C2C;"></span> General Tree Canopy</span>
                </div>
                """, unsafe_allow_html=True)

            with col_m:
                st.markdown('<div style="font-size: 1.30rem; font-weight: 800; color: #0F172A; margin-bottom: 12px;">🌳 Urban Green Infrastructure & Resilience</div>', unsafe_allow_html=True)

                tot_green = veg_canopy.get("total_green_ha", 0.0)
                park_cnt = veg_canopy.get("core_park_count", 0)
                park_ha = veg_canopy.get("core_park_area_ha", 0.0)
                verge_ha = veg_canopy.get("street_verge_ha", 0.0)

                st.markdown(f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #10B981;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Total Green Cover</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{tot_green:.2f} ha</div>
                        <div style="font-size: 0.82rem; color: #059669; font-weight: 600; margin-top: 2px;">Permeable Canopy</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #059669;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Core Urban Parks</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{park_cnt} reserves</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">{park_ha:.2f} ha contiguous</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #84CC16;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Street Verge Canopy</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{verge_ha:.2f} ha</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">Road transit buffer</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #3B82F6;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Microclimate UHI</div>
                        <div style="font-size: 1.20rem; font-weight: 800; color: {urban_inferences.get('uhi_color', '#10B981')}; margin-top: 4px;">{urban_inferences.get('uhi_rating', 'Balanced')}</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">Canopy / Built ratio</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if veg_canopy.get("parks"):
                    st.markdown('<div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;">📋 Delineated Public Green Reserves:</div>', unsafe_allow_html=True)
                    for p in veg_canopy["parks"][:4]:
                        st.markdown(f"""
                        <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                <span style="font-weight: 800; color: #0F172A; font-size: 1.02rem;">{p['name']}</span>
                                <span style="font-size: 1.20rem; font-weight: 800; color: #059669; font-family: 'JetBrains Mono', monospace;">{p['area_ha']:.2f} ha ({p['area_m2']:.0f} m²)</span>
                            </div>
                            <div style="font-size: 0.86rem; color: #64748B; margin-top: 4px;">
                                • Centroid GPS / Pixel: <code>X: {p['centroid'][0]}, Y: {p['centroid'][1]}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # =====================================================================
        # PERSPECTIVE 4: Vacant Land & Developable Parcels
        # =====================================================================
        elif perspective == "🏗️ Vacant Land & Developable Parcels":
            col_v, col_m = st.columns([1.18, 1.0], gap="large")

            with col_v:
                barren_mode = st.radio(
                    "Vacant Land Filter:",
                    [
                        "🏗️ All Developable Lots (Stratified by Road Access)",
                        "🌟 Tier A: Transit-Adjacent Only (< 25m from Road)",
                        "🔦 Complete Barren Land Spotlight"
                    ],
                    horizontal=False
                )

                barren_binary = (res["pred_mask"] == 2)
                smooth_barren_binary = prepare_smooth_display_image((barren_binary * 255).astype(np.uint8)) > 128

                if "All Developable Lots" in barren_mode and barren_parcels.get("has_barren"):
                    parcel_overlay = prepare_smooth_display_image(barren_parcels["parcel_overlay"])
                    spot = create_class_spotlight_overlay(display_rgb, smooth_barren_binary, highlight_rgb=(224, 165, 27), opacity=0.70)
                    if parcel_overlay is not None and parcel_overlay.any():
                        has_box = parcel_overlay.any(axis=-1)
                        spot[has_box] = parcel_overlay[has_box]
                    st.image(spot, caption=f"🏗️ Municipal Land Bank ({barren_parcels.get('parcel_count', 0)} Vacant Lots Stratified by Road Access)", use_container_width=True)
                elif "Tier A" in barren_mode and barren_parcels.get("has_barren"):
                    tier_a_lots = [p for p in barren_parcels.get("parcels", []) if "Tier A" in p["tier"]]
                    spot = create_class_spotlight_overlay(display_rgb, smooth_barren_binary, highlight_rgb=(5, 150, 105), opacity=0.75)
                    st.image(spot, caption=f"🌟 High-Viability Transit-Adjacent Parcels ({len(tier_a_lots)} Lots)", use_container_width=True)
                else:
                    spot = create_class_spotlight_overlay(display_rgb, smooth_barren_binary, highlight_rgb=(224, 165, 27), opacity=0.75)
                    st.image(spot, caption="🔦 Raw Barren Land Surface Footprint Spotlight", use_container_width=True)

                st.markdown("""
                <div style="display: flex; gap: 10px; justify-content: center; align-items: center; flex-wrap: wrap; margin-top: 10px; background: #FFFFFF; padding: 10px 14px; border-radius: 10px; border: 1.5px solid #E2E8F0;">
                    <span class="legend-pill"><span class="color-dot" style="background: #059669;"></span> Tier A: Transit-Adjacent (< 25m)</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #D97706;"></span> Tier B: Secondary Access (< 60m)</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #DC2626;"></span> Tier C: Landlocked Parcel</span>
                </div>
                """, unsafe_allow_html=True)

            with col_m:
                st.markdown('<div style="font-size: 1.30rem; font-weight: 800; color: #0F172A; margin-bottom: 12px;">🏗️ Municipal Land Bank & Development Viability</div>', unsafe_allow_html=True)

                tot_b_ha = barren_parcels.get("total_barren_ha", 0.0)
                dev_ha = barren_parcels.get("total_developable_ha", 0.0)
                dev_ac = barren_parcels.get("total_developable_acres", 0.0)
                t_a = barren_parcels.get("tier_a_count", 0)
                t_b = barren_parcels.get("tier_b_count", 0)
                t_c = barren_parcels.get("tier_c_count", 0)

                st.markdown(f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #059669;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Tier A: Transit-Adjacent</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #059669; font-family: 'JetBrains Mono', monospace;">{t_a} lots</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">Immediate project viability</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #D97706;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Developable Land Bank</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{dev_ha:.2f} ha</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">~{dev_ac:.1f} Total Acres</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #2563EB;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Tier B: Secondary Access</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{t_b} lots</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">Within 60m of corridor</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #DC2626;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Tier C: Landlocked Lots</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #DC2626; font-family: 'JetBrains Mono', monospace;">{t_c} lots</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">Requires easement survey</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if barren_parcels.get("parcels"):
                    st.markdown('<div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;">📋 Actionable Vacant Parcel Inventory:</div>', unsafe_allow_html=True)
                    for p in barren_parcels["parcels"][:4]:
                        st.markdown(f"""
                        <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                <span style="font-weight: 800; color: #0F172A; font-size: 1.02rem;">{p['name']}</span>
                                <span style="font-size: 0.85rem; font-weight: 800; background: {p['badge_color']}15; color: {p['badge_color']}; padding: 3px 8px; border-radius: 6px;">{p['tier']}</span>
                            </div>
                            <div style="font-size: 0.90rem; color: #334155; margin-top: 4px;">
                                • Area: <b>{p['area_m2']:.0f} m²</b> ({p['area_acres']:.2f} acres) • Centroid: <code>X: {p['centroid'][0]}, Y: {p['centroid'][1]}</code>
                            </div>
                            <div style="font-size: 0.84rem; color: #64748B; margin-top: 2px;">
                                {p['tier_desc']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # =====================================================================
        # PERSPECTIVE 5: Water Shoreline & 50m Coastal Buffer (CRZ)
        # =====================================================================
        elif perspective == "🌊 Water Shoreline & 50m Coastal Buffer (CRZ)":
            col_v, col_m = st.columns([1.18, 1.0], gap="large")

            with col_v:
                water_binary = (res["pred_mask"] == 0)
                smooth_water_binary = prepare_smooth_display_image((water_binary * 255).astype(np.uint8)) > 128

                # Base spotlight on water
                water_spotlight = create_class_spotlight_overlay(
                    display_rgb,
                    smooth_water_binary,
                    highlight_rgb=(8, 102, 255),
                    opacity=0.75,
                    dim_background=True,
                    draw_contour=True
                )

                # Draw 50m Buffer Zone
                if water_shoreline.get("buffer_mask") is not None and water_shoreline["buffer_mask"].any():
                    buf_smooth = prepare_smooth_display_image((water_shoreline["buffer_mask"] * 255).astype(np.uint8)) > 128
                    water_spotlight[buf_smooth] = (water_spotlight[buf_smooth] * 0.45 + np.array([251, 191, 36]) * 0.55).astype(np.uint8)

                # Highlight encroaching built-up inside 50m buffer in blinking warning red
                if water_shoreline.get("encroachment_mask") is not None and water_shoreline["encroachment_mask"].any():
                    enc_smooth = prepare_smooth_display_image((water_shoreline["encroachment_mask"] * 255).astype(np.uint8)) > 128
                    water_spotlight[enc_smooth] = [225, 29, 72]

                st.image(water_spotlight, caption="🌊 High-Precision Shoreline with 50-Meter Coastal Buffer Zone (CRZ)", use_container_width=True)

                st.markdown("""
                <div style="display: flex; gap: 10px; justify-content: center; align-items: center; flex-wrap: wrap; margin-top: 10px; background: #FFFFFF; padding: 10px 14px; border-radius: 10px; border: 1.5px solid #E2E8F0;">
                    <span class="legend-pill"><span class="color-dot" style="background: #0866FF;"></span> Open Sea / Creek Body</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #FBBF24;"></span> 50-Meter Coastal Buffer Zone</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #DC2626;"></span> Encroaching Structures inside Buffer</span>
                </div>
                """, unsafe_allow_html=True)

            with col_m:
                st.markdown('<div style="font-size: 1.30rem; font-weight: 800; color: #0F172A; margin-bottom: 12px;">🌊 Coastal Zone & Flood Inundation Analytics</div>', unsafe_allow_html=True)

                w_ha = water_shoreline.get("water_area_ha", 0.0)
                sh_m = water_shoreline.get("shoreline_length_m", 0.0)
                enc_m2 = water_shoreline.get("encroachment_area_m2", 0.0)

                st.markdown(f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #0866FF;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Water Surface Area</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{w_ha:.2f} ha</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; border-top: 4px solid #2563EB;">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Total Shoreline Length</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{sh_m:.1f} m</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if enc_m2 > 100.0:
                    st.markdown(f"""
                    <div style="background: #FEF2F2; border: 1.5px solid #FCA5A5; border-radius: 12px; padding: 16px 20px; border-left: 5px solid #DC2626;">
                        <div style="font-weight: 800; color: #991B1B; font-size: 1.05rem;">
                            ⚠️ Coastal Encroachment Alert Detected
                        </div>
                        <div style="font-size: 0.95rem; color: #7F1D1D; margin-top: 4px; line-height: 1.5;">
                            Detected <b>~{enc_m2:.0f} m²</b> of built-up structural footprints within the 50m tidal hazard buffer. Requires on-ground verification for coastal retaining walls or informal settlement vulnerability.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #ECFDF5; border: 1.5px solid #A7F3D0; border-radius: 12px; padding: 16px 20px; border-left: 5px solid #059669;">
                        <div style="font-weight: 800; color: #065F46; font-size: 1.05rem;">
                            ✓ Coastal Regulation Zone Buffer Clear
                        </div>
                        <div style="font-size: 0.95rem; color: #047857; margin-top: 4px;">
                            No significant structural encroachment detected within 50m of tidal boundary.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # =====================================================================
        # PERSPECTIVE 4: Field Ground Survey Checklist (Work Order)
        # =====================================================================
        elif perspective == "📋 Field Ground Survey Checklist (Work Order)":
            st.markdown("""
            <div style="margin-bottom: 16px;">
                <div style="font-size: 1.30rem; font-weight: 800; color: #0F172A;">📋 Field Surveyor Action Plan & Ground Truth Checklist</div>
                <div style="font-size: 1.0rem; color: #64748B;">
                    This checklist targets only the specific coordinates where satellite shadows, narrow alleys, or coastal buffers require manual verification. The survey organization can use this directly as a field work order.
                </div>
            </div>
            """, unsafe_allow_html=True)

            rows_html = ""
            for item in survey_checklist:
                p_color = "#DC2626" if item["priority"] == "CRITICAL" else ("#D97706" if item["priority"] == "HIGH" else "#2563EB")
                p_bg = "#FEF2F2" if item["priority"] == "CRITICAL" else ("#FFFBEB" if item["priority"] == "HIGH" else "#EFF6FF")

                rows_html += f"""
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 12px 14px; font-weight: 800; color: #0F172A;">#{item['id']}</td>
                    <td style="padding: 12px 14px;"><span style="background: {p_bg}; color: {p_color}; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem;">{item['priority']}</span></td>
                    <td style="padding: 12px 14px; font-weight: 700; color: #0F172A;">{item['type']}</td>
                    <td style="padding: 12px 14px; font-family: monospace; font-weight: 700; color: #2563EB;">{item['pixel_location']}</td>
                    <td style="padding: 12px 14px; color: #334155; font-size: 0.95rem;">{item['instruction']}</td>
                    <td style="padding: 12px 14px; font-weight: 700; color: #0F172A;">{item['action_required']}</td>
                </tr>
                """

            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; overflow-x: auto; box-shadow: 0 2px 8px rgba(15,23,42,0.04); margin-bottom: 16px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.98rem; text-align: left; min-width: 850px;">
                    <thead>
                        <tr style="background: #F8FAFC; border-bottom: 2px solid #CBD5E1;">
                            <th style="padding: 12px 14px; color: #475569;">Target ID</th>
                            <th style="padding: 12px 14px; color: #475569;">Priority</th>
                            <th style="padding: 12px 14px; color: #475569;">Anomaly Type</th>
                            <th style="padding: 12px 14px; color: #475569;">Location</th>
                            <th style="padding: 12px 14px; color: #475569;">Survey Guidance & Rationale</th>
                            <th style="padding: 12px 14px; color: #475569;">Action Item</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

            import pandas as pd
            df_survey = pd.DataFrame(survey_checklist)
            csv_data = df_survey.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Field Survey Work Order (CSV)",
                data=csv_data,
                file_name=f"{os.path.splitext(bundle.filename)[0]}_field_survey_targets.csv",
                mime="text/csv"
            )

        # =====================================================================
        # PERSPECTIVE 5: Interactive Before/After Curtain Swipe
        # =====================================================================
        elif perspective == "↔️ Before/After Curtain Swipe":
            st.markdown("""
            <div style="margin-bottom: 16px;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">↔️ Real-Time Before/After Curtain Swipe Comparison</div>
                <div style="font-size: 1.0rem; color: #64748B;">Drag the slider handle across the satellite scene to interactively reveal raw optical imagery vs AI predicted semantic segmentation.</div>
            </div>
            """, unsafe_allow_html=True)

            blended_preview = create_blended_overlay(display_rgb, display_color_mask, alpha=0.65)
            swipe_html = create_curtain_swipe_html(
                display_rgb,
                blended_preview,
                left_label=f"Raw Satellite Scene",
                right_label=f"Predicted Land Cover ({m_info['display_name']})"
            )
            st.components.v1.html(swipe_html, height=500)

        # =====================================================================
        # PERSPECTIVE 6: Full Multi-Class Overview
        # =====================================================================
        elif perspective == "🌐 Full Multi-Class Overview":
            if getattr(bundle, "is_vector", False):
                st.info("🗺️ **Vector Shapefile CAD Mode**: This input is an ESRI Shapefile containing surveyed cadastre vector geometries (Roads) rather than a multispectral raster satellite image. All 100% of surveyed road corridors are extracted with exact precision. To inspect full 5-class Earth Observation AI predictions (Water, Vegetation, Built-up, Roads, Barren), select an optical/multispectral GeoTIFF (.tif) scene.")

            col_viewer, col_analytics = st.columns([1.18, 1.0], gap="large")

            with col_viewer:
                c_mode, c_slider = st.columns([1.6, 1.2])
                with c_mode:
                    view_mode = st.radio(
                        "Display Mode:",
                        ["🎚️ Blend Overlay", "🌓 Dual View", "🎨 Mask Only", "🌡️ Confidence"],
                        horizontal=True
                    )
                with c_slider:
                    if "Blend Overlay" in view_mode:
                        overlay_alpha = st.slider("Opacity:", min_value=0, max_value=100, value=55, step=5) / 100.0
                    else:
                        overlay_alpha = 0.55

                if "Blend Overlay" in view_mode:
                    blended = create_blended_overlay(display_rgb, display_color_mask, alpha=overlay_alpha)
                    st.image(blended, caption=f"Semantic Segmentation Overlay ({int(overlay_alpha*100)}% Opacity)", use_container_width=True)
                elif "Dual View" in view_mode:
                    v_c1, v_c2 = st.columns(2)
                    with v_c1:
                        st.image(display_rgb, caption=f"Original ({bundle.filename})", use_container_width=True)
                    with v_c2:
                        st.image(display_color_mask, caption=f"Semantic Mask ({m_info['display_name']})", use_container_width=True)
                elif "Mask Only" in view_mode:
                    st.image(display_color_mask, caption="Predicted Class Segmentation Mask", use_container_width=True)
                elif "Confidence" in view_mode:
                    if res["conf_rgb"] is not None:
                        display_conf = prepare_smooth_display_image(res["conf_rgb"])
                        st.image(display_conf, caption="Model Inference Confidence Heatmap", use_container_width=True)
                    else:
                        st.info("Confidence map not generated for this model.")

                # Legend
                st.markdown("""
                <div style="display: flex; gap: 10px; justify-content: center; align-items: center; flex-wrap: wrap; margin-top: 12px; background: #FFFFFF; padding: 12px 16px; border-radius: 12px; border: 1.5px solid #E2E8F0;">
                    <span class="legend-pill"><span class="color-dot" style="background: #0866FF;"></span> Water</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #218C2C;"></span> Vegetation</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #E0A51B;"></span> Barren Land</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #E71345;"></span> Built-up</span>
                    <span class="legend-pill"><span class="color-dot" style="background: #999999;"></span> Roads</span>
                </div>
                """, unsafe_allow_html=True)

            with col_analytics:
                st.markdown('<div style="font-size: 1.30rem; font-weight: 800; color: #0F172A; margin-bottom: 14px;">📊 Class Land Cover Distribution</div>', unsafe_allow_html=True)
                fig_donut = render_donut_chart(dist)
                st.pyplot(fig_donut)
                plt.close(fig_donut)

                for c in dist["classes"]:
                    area_text = f"<b>{c['area_km2']:.3f} km²</b> ({c['area_ha']:.1f} ha)" if c["area_km2"] is not None else f"<b>{c['pixel_count']:,}</b> pixels"
                    pct = c['pct_valid']
                    bar_width = max(1.5, min(100.0, pct)) if pct > 0 else 0.5
                    st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; box-shadow: 0 2px 5px rgba(15,23,42,0.03);">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 9px;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="width: 18px; height: 18px; border-radius: 4px; background: {c['color']}; display: inline-block;"></span>
                                <span style="font-size: 1.20rem; font-weight: 800; color: #0F172A;">{c['name']}</span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 1.55rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace;">{pct:.1f}%</span>
                                <span style="font-size: 1.05rem; color: #475569; margin-left: 10px; font-weight: 600;">• {area_text}</span>
                            </div>
                        </div>
                        <div style="background: #F1F5F9; border-radius: 8px; height: 16px; overflow: hidden; border: 1px solid #E2E8F0;">
                            <div style="background: {c['color']}; width: {bar_width}%; height: 100%; border-radius: 7px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 1-Click Multi-Format Export Bar
        # ---------------------------------------------------------------------
        st.markdown('<div class="section-header" style="margin-top: 28px;">💾 Step 5: Export & Download Prediction Artifacts</div>', unsafe_allow_html=True)

        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            blended_full = create_blended_overlay(bundle.rgb_preview, res["color_mask"], alpha=0.55)
            png_buf = io.BytesIO()
            Image.fromarray(blended_full).save(png_buf, format="PNG")
            st.download_button(
                "🗺️ Download Color Overlay (PNG)",
                data=png_buf.getvalue(),
                file_name=f"{os.path.splitext(bundle.filename)[0]}_prediction_overlay.png",
                mime="image/png",
                use_container_width=True
            )

        with d_col2:
            if bundle.is_georeferenced and bundle.transform and bundle.crs:
                geotiff_bytes = export_prediction_geotiff(res["pred_mask"], bundle.transform, bundle.crs)
                st.download_button(
                    "🛰️ Download Georeferenced Mask (GeoTIFF)",
                    data=geotiff_bytes,
                    file_name=f"{os.path.splitext(bundle.filename)[0]}_classification_mask.tif",
                    mime="image/tiff",
                    use_container_width=True
                )
            else:
                mask_png_buf = io.BytesIO()
                Image.fromarray(res["color_mask"]).save(mask_png_buf, format="PNG")
                st.download_button(
                    "🎨 Download Class Mask (PNG)",
                    data=mask_png_buf.getvalue(),
                    file_name=f"{os.path.splitext(bundle.filename)[0]}_class_mask.png",
                    mime="image/png",
                    use_container_width=True
                )

        with d_col3:
            import json
            summary_dict = {
                "input_scene": bundle.filename,
                "model_used": m_info["display_name"],
                "model_id": m_info["id"],
                "execution_duration_sec": dur,
                "is_georeferenced": bundle.is_georeferenced,
                "crs": bundle.crs,
                "resolution": bundle.resolution,
                "class_distribution": dist["classes"],
                "road_intelligence": {
                    "arterial_length_km": road_corridors.get("arterial_length_km", 0.0),
                    "total_length_km": road_corridors.get("total_length_km", 0.0),
                    "density_km_per_km2": road_corridors.get("road_density_km_per_km2", 0.0),
                    "continuity_score": road_corridors.get("continuity_score", 0.0),
                    "corridor_table": road_corridors.get("corridor_table", [])
                },
                "building_footprints": {
                    "building_count": building_footprints.get("building_count", 0),
                    "total_roof_area_ha": building_footprints.get("total_roof_area_ha", 0.0)
                },
                "water_shoreline": {
                    "shoreline_length_m": water_shoreline.get("shoreline_length_m", 0.0),
                    "encroachment_area_m2": water_shoreline.get("encroachment_area_m2", 0.0)
                },
                "urban_green_reserves": {
                    "total_green_ha": veg_canopy.get("total_green_ha", 0.0),
                    "core_park_count": veg_canopy.get("core_park_count", 0),
                    "parks": veg_canopy.get("parks", [])
                },
                "developable_vacant_parcels": {
                    "total_developable_ha": barren_parcels.get("total_developable_ha", 0.0),
                    "total_developable_acres": barren_parcels.get("total_developable_acres", 0.0),
                    "parcels": barren_parcels.get("parcels", [])
                },
                "mumbai_settlement_subclasses": settlement_res.get("stats", []),
                "field_survey_checklist": survey_checklist
            }
            summary_json = json.dumps(summary_dict, indent=2, default=str)
            st.download_button(
                "📄 Download Survey Dossier (JSON)",
                data=summary_json,
                file_name=f"{os.path.splitext(bundle.filename)[0]}_survey_dossier.json",
                mime="application/json",
                use_container_width=True
            )

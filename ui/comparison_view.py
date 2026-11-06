"""
ui/comparison_view.py
---------------------
Multi-Model Side-by-Side Comparison Studio.
Enables side-by-side execution and visual comparison of up to 3 models simultaneously on the same scene:
1. Side-by-side synchronized prediction maps.
2. Diagrammatic class distribution comparison chart.
3. Comparative architectural trade-off table (Speed vs Road continuity vs Water boundary sharpness).
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from PIL import Image

from model_registry import get_all_models, get_ready_models, get_model_by_id
from inference import get_adapter
from inference.geospatial_io import create_color_mask, create_blended_overlay, ImageInputBundle
from inference.postprocessing import compute_class_distribution
from ui.prediction_view import prepare_smooth_display_image

def render_comparison_page():
    st.markdown('<div class="section-header" style="margin-top: 0;">🔬 Multi-Model Side-by-Side Comparison Studio</div>', unsafe_allow_html=True)

    if "current_bundle" not in st.session_state:
        st.warning("⚠️ No image loaded yet. Please upload an image or pick a test preset in the **Feature Extraction** tab first.")
        return

    bundle: ImageInputBundle = st.session_state["current_bundle"]
    all_models = get_all_models()

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);">
        <span style="font-size: 0.95rem; color: #64748B;">Active Scene:</span> 
        <b style="color: #0F172A; font-size: 1.05rem;">{bundle.filename}</b> 
        <span style="color: #64748B; font-size: 0.90rem; margin-left: 8px;">({bundle.width} × {bundle.height} px • {bundle.channels} bands)</span>
    </div>
    """, unsafe_allow_html=True)

    # Model selectors (Pick 2 or 3 models)
    st.markdown('<div style="font-weight: 700; color: #0F172A; font-size: 1.05rem; margin-bottom: 8px;">Select Models to Compare Side-by-Side:</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    model_options = {m["id"]: m["display_name"] for m in all_models}
    keys = list(model_options.keys())

    with c1:
        m1_id = st.selectbox("Model 1 (Primary):", options=keys, format_func=lambda x: model_options.get(x, x), index=0 if len(keys) > 0 else 0)
    with c2:
        m2_id = st.selectbox("Model 2 (Comparison):", options=keys, format_func=lambda x: model_options.get(x, x), index=1 if len(keys) > 1 else 0)
    with c3:
        m3_id = st.selectbox("Model 3 (Optional):", options=["None"] + keys, format_func=lambda x: model_options.get(x, "None") if x != "None" else "None", index=2 if len(keys) > 2 else 0)

    selected_ids = [m1_id, m2_id]
    if m3_id != "None" and m3_id not in selected_ids:
        selected_ids.append(m3_id)

    run_comp = st.button("⚡ Execute Side-by-Side Comparison", type="primary", use_container_width=True)

    if run_comp:
        comp_results = []
        p_bar = st.progress(0, text="Initializing comparative inference...")

        for idx, mid in enumerate(selected_ids):
            m_info = get_model_by_id(mid)
            p_bar.progress(int((idx + 1) / len(selected_ids) * 90), text=f"Running {m_info['display_name']}...")
            try:
                adapter = get_adapter(mid)
                mask, conf, dur, warns = adapter.run_inference(bundle, allow_rgb_adaptation=True)
                color_mask = create_color_mask(mask)
                dist = compute_class_distribution(mask, bundle.resolution)
                comp_results.append({
                    "model_info": m_info,
                    "pred_mask": mask,
                    "color_mask": color_mask,
                    "distribution": dist,
                    "duration": dur
                })
            except Exception as e:
                st.error(f"Error running {m_info['display_name']}: {str(e)}")

        p_bar.progress(100, text="Comparison complete!")
        p_bar.empty()
        st.session_state["comp_results"] = comp_results

    if "comp_results" in st.session_state:
        comp_results = st.session_state["comp_results"]

        st.markdown('<div class="section-header" style="margin-top: 20px;">🖼️ 1. Visual Prediction Maps Comparison</div>', unsafe_allow_html=True)

        cols = st.columns(len(comp_results))
        for idx, res in enumerate(comp_results):
            m = res["model_info"]
            with cols[idx]:
                st.markdown(f"""
                <div style="background: #FFFFFF; padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; border-top: 4px solid #2563EB; border-left: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(15,23,42,0.04);">
                    <div style="font-weight: 700; color: #0F172A; font-size: 1.0rem;">{m['display_name']}</div>
                    <div style="color: #64748B; font-size: 0.85rem; margin-top: 2px;">{m['architecture']} • <b>{res['duration']:.2f}s latency</b></div>
                </div>
                """, unsafe_allow_html=True)
                
                disp_rgb = prepare_smooth_display_image(bundle.rgb_preview)
                disp_mask = prepare_smooth_display_image(res["color_mask"])
                blended = create_blended_overlay(disp_rgb, disp_mask, alpha=0.55)
                st.image(blended, use_container_width=True)

        # ---------------------------------------------------------------------
        # 2. Diagrammatic Class Coverage Delta Chart (Light Theme)
        # ---------------------------------------------------------------------
        st.markdown('<div class="section-header" style="margin-top: 24px;">📊 2. Class Distribution & Spatial Delta Chart</div>', unsafe_allow_html=True)

        class_names = ["Water", "Vegetation", "Barren Land", "Built-up", "Roads"]
        fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(colors="#0F172A", labelsize=11)
        ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

        x = np.arange(len(class_names))
        w = 0.8 / len(comp_results)
        palette = ["#2563EB", "#059669", "#D97706"]

        for idx, res in enumerate(comp_results):
            m = res["model_info"]
            vals = [c["pct_valid"] for c in res["distribution"]["classes"]]
            offset = (idx - (len(comp_results) - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w, label=m["display_name"], color=palette[idx % len(palette)], edgecolor="white", lw=1.2)
            for bar in bars:
                h = bar.get_height()
                if h > 1.5:
                    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                                textcoords="offset points", ha="center", va="bottom", fontsize=10, color="#0F172A", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, color="#0F172A", fontsize=11.5, fontweight="bold")
        ax.set_ylabel("Class Proportion (%)", color="#334155", fontsize=12, fontweight="bold")
        ax.set_title("Direct Class Area Distribution Comparison", color="#0F172A", fontsize=13.5, fontweight="bold", pad=14)
        ax.legend(facecolor="#F8FAFC", edgecolor="#CBD5E1", labelcolor="#0F172A", fontsize=10.5)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ---------------------------------------------------------------------
        # 3. Architectural Trade-off Summary Table (Light Theme)
        # ---------------------------------------------------------------------
        st.markdown('<div class="section-header" style="margin-top: 24px;">⚖️ 3. Comparative Architecture & IoU Trade-off Summary</div>', unsafe_allow_html=True)

        rows = ""
        for res in comp_results:
            m = res["model_info"]
            dist = res["distribution"]
            metrics = m.get("verified_metrics", {})
            oa_str = f"{metrics.get('overall_accuracy', 80.0):.2f}%" if metrics.get('overall_accuracy') else "N/A"
            miou_str = f"{metrics.get('macro_miou', 50.0):.2f}%" if metrics.get('macro_miou') else "N/A"

            water_area = [c["area_km2"] for c in dist["classes"] if c["class_id"] == 0][0]
            road_area = [c["area_km2"] for c in dist["classes"] if c["class_id"] == 4][0]
            built_area = [c["area_km2"] for c in dist["classes"] if c["class_id"] == 3][0]

            water_str = f"{water_area:.3f} km²" if water_area is not None else f"{dist['classes'][0]['pct_valid']:.1f}%"
            road_str = f"{road_area:.3f} km²" if road_area is not None else f"{dist['classes'][4]['pct_valid']:.1f}%"
            built_str = f"{built_area:.3f} km²" if built_area is not None else f"{dist['classes'][3]['pct_valid']:.1f}%"

            rows += f"""
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 12px 14px; font-weight: 700; color: #0F172A;">{m['display_name']}<br><span style="font-size: 0.82rem; color: #64748B; font-weight: 500;">{m['architecture']}</span></td>
                <td style="padding: 12px 12px; color: #1D4ED8; font-weight: 700;">{oa_str}</td>
                <td style="padding: 12px 12px; color: #059669; font-weight: 700;">{miou_str}</td>
                <td style="padding: 12px 12px; color: #0284C7; font-weight: 600;">{water_str}</td>
                <td style="padding: 12px 12px; color: #D97706; font-weight: 600;">{road_str}</td>
                <td style="padding: 12px 12px; color: #DC2626; font-weight: 600;">{built_str}</td>
                <td style="padding: 12px 12px; color: #475569; font-weight: 600;">{res['duration']:.3f}s</td>
            </tr>
            """

        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 12px; overflow-x: auto; box-shadow: 0 2px 8px rgba(15,23,42,0.04);">
            <table style="width: 100%; border-collapse: collapse; font-size: 1.05rem; text-align: left;">
                <thead>
                    <tr style="background: #F8FAFC; border-bottom: 2px solid #CBD5E1;">
                        <th style="padding: 14px 16px; color: #0F172A; font-weight: 800;">Model</th>
                        <th style="padding: 14px 14px; color: #1D4ED8; font-weight: 800;">Overall Acc.</th>
                        <th style="padding: 14px 14px; color: #059669; font-weight: 800;">Mean IoU</th>
                        <th style="padding: 14px 14px; color: #0284C7; font-weight: 800;">Water</th>
                        <th style="padding: 14px 14px; color: #D97706; font-weight: 800;">Roads</th>
                        <th style="padding: 14px 14px; color: #DC2626; font-weight: 800;">Built-up</th>
                        <th style="padding: 14px 14px; color: #475569; font-weight: 800;">Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

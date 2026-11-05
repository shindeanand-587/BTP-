"""
ui/analysis.py
--------------
Class Area Analysis component.
Computes predicted pixel-area distribution and physical spatial estimates (hectares and km²).
Uses pure HTML/CSS and Matplotlib (zero pandas/altair dependency).
"""

import streamlit as st
import matplotlib.pyplot as plt

from inference.geospatial_io import ImageInputBundle
from model_registry import CLASS_MAPPING

def render_analysis():
    st.markdown('<div class="section-header">📊 Urban Land Cover & Spatial Area Analysis</div>', unsafe_allow_html=True)

    if "last_inference_results" not in st.session_state or not st.session_state["last_inference_results"]:
        st.warning("⚠️ No prediction results available. Please run an inference job in **Run Inference** first.")
        return

    bundle: ImageInputBundle = st.session_state["current_bundle"]
    results = st.session_state["last_inference_results"]

    if len(results) > 1:
        chosen_res_idx = st.selectbox(
            "Select Model Run to Analyze:",
            options=list(range(len(results))),
            format_func=lambda i: results[i]["model_info"]["display_name"]
        )
        res = results[chosen_res_idx]
    else:
        res = results[0]

    model_info = res["model_info"]
    dist = res["distribution"]

    st.markdown(f"""
    <div style="font-size: 0.95rem; color: #CBD5E1; margin-bottom: 14px;">
        Analyzing scene: <b>{bundle.filename}</b> using model: <b>{model_info['display_name']}</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-box callout-info">
        <b>💡 Methodological Distinction:</b>
        The table and charts below quantify the <b>predicted spatial footprint</b> of each urban class.
        These proportions reflect model inferences and should not be confused with ground-truth cadastral measurements.
        NoData and boundary pixels (Class 255) are isolated from land cover proportions.
    </div>
    """, unsafe_allow_html=True)

    # Summary KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Pixels", f"{dist['total_pixels']:,}")
    with k2:
        st.metric("Valid Classified Pixels", f"{dist['valid_pixels']:,}")
    with k3:
        st.metric("NoData / Margin Pixels", f"{dist['nodata_pixels']:,} ({dist['nodata_pct']}%)")
    with k4:
        if dist["has_physical_area"]:
            total_ha = sum([c["area_ha"] for c in dist["classes"] if c["area_ha"] is not None])
            st.metric("Total Physical Area", f"{total_ha:.2f} ha")
        else:
            st.metric("Physical Area", "Non-georeferenced")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Detailed Class Table (HTML/CSS)
    st.markdown('<div style="font-weight: 700; color: #F1F5F9; margin-bottom: 8px;">📋 Detailed Class Breakdown</div>', unsafe_allow_html=True)
    
    rows_html = ""
    for c in dist["classes"]:
        ha_val = f"{c['area_ha']} ha" if c.get("area_ha") is not None else "N/A"
        km2_val = f"{c['area_km2']} km²" if c.get("area_km2") is not None else "N/A"
        rows_html += f"""
        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
            <td style="padding: 10px 14px; font-weight: 700; color: #94A3B8;">{c['class_id']}</td>
            <td style="padding: 10px 14px; font-weight: 600; color: {c['color']};">
                <span style="display:inline-block; width:10px; height:10px; background:{c['color']}; border-radius:2px; margin-right:6px;"></span>
                {c['name']}
            </td>
            <td style="padding: 10px 14px; font-family: monospace; color: #F8FAFC;">{c['pixel_count']:,} px</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #38BDF8; font-weight: 600;">{c['pct_valid']}%</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #94A3B8;">{c['pct_total']}%</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #34D399;">{ha_val}</td>
            <td style="padding: 10px 14px; font-family: monospace; color: #34D399;">{km2_val}</td>
        </tr>
        """

    st.markdown(f"""
    <div class="glass-panel" style="padding: 0; overflow-x: auto; margin-bottom: 18px;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem;">
            <thead>
                <tr style="background: rgba(30, 41, 59, 0.8); border-bottom: 2px solid rgba(56, 189, 248, 0.3);">
                    <th style="padding: 12px 14px; color: #94A3B8;">ID</th>
                    <th style="padding: 12px 14px; color: #F8FAFC;">Class Name</th>
                    <th style="padding: 12px 14px; color: #F8FAFC;">Pixel Count</th>
                    <th style="padding: 12px 14px; color: #38BDF8;">Valid Share</th>
                    <th style="padding: 12px 14px; color: #94A3B8;">Total Share</th>
                    <th style="padding: 12px 14px; color: #34D399;">Physical Area (ha)</th>
                    <th style="padding: 12px 14px; color: #34D399;">Physical Area (km²)</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Donut Chart for Class Proportions via Matplotlib
    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 700; color: #F1F5F9; margin-bottom: 8px;">🍩 Relative Land Cover Composition</div>', unsafe_allow_html=True)

    active_classes = [c for c in dist["classes"] if c["pixel_count"] > 0]
    if active_classes:
        labels = [c["name"] for c in active_classes]
        sizes = [c["pct_valid"] for c in active_classes]
        colors = [c["color"] for c in active_classes]

        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0B1120")
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="#0B1120", linewidth=2),
            textprops=dict(color="#E2E8F0", fontsize=9.5)
        )
        for at in autotexts:
            at.set_color("#FFFFFF")
            at.set_fontweight("bold")

        ax.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

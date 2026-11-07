"""
ui/leaderboard_view.py
----------------------
Comprehensive Model Specifications & Empirical Benchmarks.
Displays:
1. Architectural Specification Profiles across Core and Contributed Models.
2. Interactive Class-wise IoU and Accuracy Benchmark Visualizer.
3. Complete Model Specification Table with trained class weights and verified test-set metrics.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from model_registry import get_all_models, get_core_models, get_teammate_models

def render_leaderboard_page():
    st.markdown('<div class="section-header" style="margin-top: 0;">🏆 Model Specifications & Empirical Benchmarks</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 1.0rem; color: #64748B; margin-bottom: 20px; font-weight: 500;">
        Empirical evaluation evidence, class-loss training weights, and architectural specifications across all registered models evaluated on 5 urban classes.
    </div>
    """, unsafe_allow_html=True)

    all_models = get_all_models()

    # -------------------------------------------------------------------------
    # 1. Model Architectural Specification Profiles (Clean Light Theme)
    # -------------------------------------------------------------------------
    podium_cols = st.columns(3)
    core_models = get_core_models()

    card_themes = [
        {"badge": "🌟 Multispectral Benchmark", "color": "#1D4ED8", "border": "#BFDBFE", "bg": "#EFF6FF"},
        {"badge": "🗺️ Panchromatic Benchmark", "color": "#0369A1", "border": "#BAE6FD", "bg": "#F0F9FF"},
        {"badge": "🌊 Boundary & Water Specialist", "color": "#0F766E", "border": "#99F6E4", "bg": "#F0FDFA"}
    ]

    for idx, m in enumerate(core_models[:3]):
        thm = card_themes[idx % len(card_themes)]
        metrics = m["verified_metrics"]
        cw = m.get("class_weights", {"Water": 1.0, "Vegetation": 1.0, "Barren Land": 1.0, "Built-up": 1.0, "Roads": 1.0})
        with podium_cols[idx]:
            st.markdown(f"""
            <div style="background: {thm['bg']}; border: 1.5px solid {thm['border']}; border-radius: 12px; padding: 20px 18px; text-align: center; margin-bottom: 22px; box-shadow: 0 2px 6px rgba(15,23,42,0.04);">
                <div style="font-size: 1.05rem; font-weight: 800; color: {thm['color']}; margin-bottom: 6px;">{thm['badge']}</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A;">{m['display_name']}</div>
                <div style="font-size: 0.88rem; color: #64748B; margin-bottom: 12px; font-weight: 600;">{m['architecture']} • {m['sensor']}</div>
                
                <div style="display: flex; justify-content: space-around; background: #FFFFFF; border: 1px solid {thm['border']}; padding: 10px 14px; border-radius: 10px;">
                    <div>
                        <div style="font-size: 0.76rem; color: #64748B; font-weight: 700;">ACCURACY</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #1D4ED8; font-family: 'JetBrains Mono', monospace;">{metrics['overall_accuracy']:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.76rem; color: #64748B; font-weight: 700;">MEAN IOU</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #059669; font-family: 'JetBrains Mono', monospace;">{metrics['macro_miou']:.1f}%</div>
                    </div>
                </div>

                <div style="background: #FFFFFF; border: 1px dashed {thm['border']}; border-radius: 8px; padding: 10px 12px; margin-top: 12px; text-align: left;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #334155; margin-bottom: 3px;">⚖️ Trained Class Weights:</div>
                    <div style="font-size: 0.80rem; color: #475569; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                        💧 W: <b>{cw.get('Water', 1.0):.1f}×</b> &nbsp;|&nbsp; 
                        🌿 V: <b>{cw.get('Vegetation', 1.0):.1f}×</b> &nbsp;|&nbsp; 
                        🏜️ B: <b>{cw.get('Barren Land', 1.0):.1f}×</b><br>
                        🏢 U: <b>{cw.get('Built-up', 1.0):.1f}×</b> &nbsp;|&nbsp; 
                        🛣️ R: <b>{cw.get('Roads', 1.0):.1f}×</b>
                    </div>
                </div>

                <div style="font-size: 0.85rem; color: #334155; margin-top: 10px; text-align: left; line-height: 1.45; font-weight: 500;">
                    • <b>Specialization:</b> {m['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Interactive Class-wise IoU Bar Chart (Light Theme)
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">📊 Class-Wise Metric Benchmark Across Models</div>', unsafe_allow_html=True)

    c_select, _ = st.columns([1.8, 2.2])
    with c_select:
        selected_metric_name = st.selectbox(
            "Select Evaluation Dimension to Visualize:",
            ["Overall Accuracy (OA)", "Macro Mean IoU (mIoU)", "Roads IoU", "Built-up IoU", "Water IoU", "Vegetation IoU"]
        )

    metric_lookup = {
        "Overall Accuracy (OA)": lambda m: m["verified_metrics"].get("overall_accuracy", 0.0),
        "Macro Mean IoU (mIoU)": lambda m: m["verified_metrics"].get("macro_miou", 0.0),
        "Roads IoU": lambda m: m["verified_metrics"].get("class_ious", {}).get("Roads", 0.0),
        "Built-up IoU": lambda m: m["verified_metrics"].get("class_ious", {}).get("Built-up", 0.0),
        "Water IoU": lambda m: m["verified_metrics"].get("class_ious", {}).get("Water", 0.0),
        "Vegetation IoU": lambda m: m["verified_metrics"].get("class_ious", {}).get("Vegetation", 0.0)
    }

    names = [m["display_name"].split("(")[0].strip() for m in all_models]
    scores = [metric_lookup[selected_metric_name](m) for m in all_models]

    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.tick_params(colors="#0F172A", labelsize=10.5)
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")

    bar_colors = ["#2563EB" if i < 3 else "#7C3AED" for i in range(len(names))]
    bars = ax.bar(names, scores, color=bar_colors, edgecolor="white", lw=1.2, width=0.48)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10, color="#0F172A", fontweight="bold")

    ax.set_ylabel(f"{selected_metric_name} (%)", color="#334155", fontsize=12, fontweight="bold")
    ax.set_title(f"Comparative Benchmark: {selected_metric_name} (Blue = Core Models, Purple = Contributed)",
                 color="#0F172A", fontsize=13.5, fontweight="bold", pad=14)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=15, ha="right", color="#0F172A", fontweight="700", fontsize=10.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Comprehensive Specifications Table (Light Theme)
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header" style="margin-top: 24px;">📋 Detailed Model Specification & Verification Table</div>', unsafe_allow_html=True)

    rows = ""
    for idx, m in enumerate(all_models):
        metrics = m.get("verified_metrics", {})
        ious = metrics.get("class_ious", {})
        oa = metrics.get("overall_accuracy", 0.0)
        miou = metrics.get("macro_miou", 0.0)
        cw = m.get("class_weights", {"Water": 1.0, "Vegetation": 1.0, "Barren Land": 1.0, "Built-up": 1.0, "Roads": 1.0})
        cw_str = f"[{cw.get('Water', 1.0):.1f}, {cw.get('Vegetation', 1.0):.1f}, {cw.get('Barren Land', 1.0):.1f}, {cw.get('Built-up', 1.0):.1f}, {cw.get('Roads', 1.0):.1f}]"

        badge = '<span class="badge badge-ready">🌟 Core</span>' if idx < 3 else '<span class="badge" style="background: #F5F3FF; color: #7C3AED; border: 1px solid #DDD6FE;">👥 Custom</span>'

        rows += f"""
        <tr style="border-bottom: 1px solid #E2E8F0;">
            <td style="padding: 12px 10px;">{badge}</td>
            <td style="padding: 12px 12px; font-weight: 700; color: #0F172A;">
                {m['display_name']}<br><span style="font-size: 0.82rem; color: #64748B; font-weight: 500;">{m['architecture']}</span>
            </td>
            <td style="padding: 12px 10px; color: #334155; font-size: 0.88rem; font-weight: 600;">{m['sensor']}<br><span style="color: #2563EB;">{m.get('native_gsd_m', 1.0)}m GSD</span></td>
            <td style="padding: 12px 10px; color: #1D4ED8; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{oa:.1f}%</td>
            <td style="padding: 12px 10px; color: #059669; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{miou:.1f}%</td>
            <td style="padding: 12px 10px; color: #6D28D9; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem;">{cw_str}</td>
            <td style="padding: 12px 8px; color: #0284C7; font-weight: 600;">{ious.get('Water', 0.0):.1f}%</td>
            <td style="padding: 12px 8px; color: #16A34A; font-weight: 600;">{ious.get('Vegetation', 0.0):.1f}%</td>
            <td style="padding: 12px 8px; color: #DC2626; font-weight: 600;">{ious.get('Built-up', 0.0):.1f}%</td>
            <td style="padding: 12px 8px; color: #D97706; font-weight: 600;">{ious.get('Roads', 0.0):.1f}%</td>
            <td style="padding: 12px 10px; font-size: 0.85rem; color: #475569; font-weight: 500;">{m['description']}</td>
        </tr>
        """

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 14px; overflow-x: auto; box-shadow: 0 2px 8px rgba(15,23,42,0.04);">
        <table style="width: 100%; border-collapse: collapse; font-size: 1.02rem; text-align: left; min-width: 1040px;">
            <thead>
                <tr style="background: #F8FAFC; border-bottom: 2px solid #CBD5E1;">
                    <th style="padding: 14px 12px; color: #475569; font-weight: 800;">Category</th>
                    <th style="padding: 14px 14px; color: #0F172A; font-weight: 800;">Model & Architecture</th>
                    <th style="padding: 14px 12px; color: #475569; font-weight: 800;">Sensor / GSD</th>
                    <th style="padding: 14px 12px; color: #1D4ED8; font-weight: 800;">Overall Acc.</th>
                    <th style="padding: 14px 12px; color: #059669; font-weight: 800;">Mean IoU</th>
                    <th style="padding: 14px 12px; color: #6D28D9; font-weight: 800;">Trained Weights<br><span style="font-size: 0.72rem; color: #475569;">[W, V, B, U, R]</span></th>
                    <th style="padding: 14px 10px; color: #0284C7; font-weight: 800;">Water</th>
                    <th style="padding: 14px 10px; color: #16A34A; font-weight: 800;">Vegetation</th>
                    <th style="padding: 14px 10px; color: #DC2626; font-weight: 800;">Built-up</th>
                    <th style="padding: 14px 10px; color: #D97706; font-weight: 800;">Roads</th>
                    <th style="padding: 14px 12px; color: #475569; font-weight: 800;">Specialization Note</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

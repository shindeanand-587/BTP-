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

def safe_num(val: Any, default: float = 0.0) -> float:
    """Safely extracts a float value from any primitive, handling None, strings with %, and corrupt values."""
    if val is None:
        return default
    try:
        if isinstance(val, str):
            val = val.replace("%", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return default

def render_leaderboard_page():
    st.markdown('<div class="section-header" style="margin-top: 0;">🏆 Model Specifications & Empirical Benchmarks</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 1.0rem; color: #64748B; margin-bottom: 20px; font-weight: 500;">
        Empirical evaluation evidence, class-loss training weights, and architectural specifications across all registered models evaluated on 5 urban classes.
    </div>
    """, unsafe_allow_html=True)

    all_models = get_all_models()

    # Sort all models empirically by Overall Accuracy descending, then macro mIoU descending
    ranked_models = sorted(
        all_models,
        key=lambda m: (
            safe_num((m.get("verified_metrics") or {}).get("overall_accuracy"), 0.0),
            safe_num((m.get("verified_metrics") or {}).get("macro_miou"), 0.0)
        ),
        reverse=True
    )

    # -------------------------------------------------------------------------
    # 1. Model Architectural Specification Profiles (Top-3 Empirical Podium)
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header" style="margin-top: 0;">🏆 Top Performing Neural Backbones (Empirical Benchmark Podium)</div>', unsafe_allow_html=True)

    podium_cols = st.columns(3)
    card_themes = [
        {"rank_label": "🥇 Rank 1 Benchmark Champion", "color": "#B45309", "border": "#FDE68A", "bg": "#FFFBEB", "glow": "0 4px 14px rgba(217,119,6,0.12)"},
        {"rank_label": "🥈 Rank 2 High-Accuracy Performer", "color": "#334155", "border": "#CBD5E1", "bg": "#F8FAFC", "glow": "0 3px 10px rgba(100,116,139,0.08)"},
        {"rank_label": "🥉 Rank 3 High-Fidelity Backbone", "color": "#1D4ED8", "border": "#BFDBFE", "bg": "#EFF6FF", "glow": "0 3px 10px rgba(37,99,235,0.08)"}
    ]

    for idx in range(min(3, len(ranked_models))):
        m = ranked_models[idx]
        thm = card_themes[idx]
        metrics = m.get("verified_metrics") or {}
        oa_val = safe_num(metrics.get("overall_accuracy"), 80.0)
        miou_val = safe_num(metrics.get("macro_miou"), 50.0)
        cw = m.get("class_weights") or {}
        cw_w = safe_num(cw.get("Water"), 1.0)
        cw_v = safe_num(cw.get("Vegetation"), 1.0)
        cw_b = safe_num(cw.get("Barren Land"), 1.0)
        cw_u = safe_num(cw.get("Built-up"), 1.0)
        cw_r = safe_num(cw.get("Roads"), 1.0)

        owner_str = m.get("owner", "System")
        role_badge = m.get("role_badge", "")
        role_label = role_badge if role_badge else ("🌟 Core System" if idx >= 2 else f"👥 Contributed ({owner_str})")
        role_pill = f'<span style="display: inline-block; background: #FFFFFF; border: 1px solid {thm["border"]}; border-radius: 6px; padding: 2px 8px; font-size: 0.76rem; font-weight: 700; color: {thm["color"]}; margin-top: 4px;">{role_label}</span>'

        with podium_cols[idx]:
            st.markdown(f"""
            <div style="background: {thm['bg']}; border: 1.5px solid {thm['border']}; border-radius: 14px; padding: 20px 18px; text-align: center; margin-bottom: 22px; box-shadow: {thm['glow']};">
                <div style="font-size: 1.05rem; font-weight: 800; color: {thm['color']}; margin-bottom: 4px;">{thm['rank_label']}</div>
                {role_pill}
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-top: 8px;">{m['display_name']}</div>
                <div style="font-size: 0.88rem; color: #64748B; margin-bottom: 12px; font-weight: 600;">{m.get('architecture', 'Custom Architecture')} • {m.get('sensor', 'Optical')}</div>
                
                <div style="display: flex; justify-content: space-around; background: #FFFFFF; border: 1px solid {thm['border']}; padding: 10px 14px; border-radius: 10px;">
                    <div>
                        <div style="font-size: 0.76rem; color: #64748B; font-weight: 700;">ACCURACY</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: {thm['color']}; font-family: 'JetBrains Mono', monospace;">{oa_val:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.76rem; color: #64748B; font-weight: 700;">MEAN IOU</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #059669; font-family: 'JetBrains Mono', monospace;">{miou_val:.1f}%</div>
                    </div>
                </div>

                <div style="background: #FFFFFF; border: 1px dashed {thm['border']}; border-radius: 8px; padding: 10px 12px; margin-top: 12px; text-align: left;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #334155; margin-bottom: 3px;">⚖️ Trained Class Weights:</div>
                    <div style="font-size: 0.80rem; color: #475569; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                        💧 W: <b>{cw_w:.1f}×</b> &nbsp;|&nbsp; 
                        🌿 V: <b>{cw_v:.1f}×</b> &nbsp;|&nbsp; 
                        🏜️ B: <b>{cw_b:.1f}×</b><br>
                        🏢 U: <b>{cw_u:.1f}×</b> &nbsp;|&nbsp; 
                        🛣️ R: <b>{cw_r:.1f}×</b>
                    </div>
                </div>

                <div style="font-size: 0.85rem; color: #334155; margin-top: 10px; text-align: left; line-height: 1.45; font-weight: 500;">
                    • <b>Specialization:</b> {m.get('description', '')}
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
        "Overall Accuracy (OA)": lambda m: safe_num((m.get("verified_metrics") or {}).get("overall_accuracy"), 0.0),
        "Macro Mean IoU (mIoU)": lambda m: safe_num((m.get("verified_metrics") or {}).get("macro_miou"), 0.0),
        "Roads IoU": lambda m: safe_num(((m.get("verified_metrics") or {}).get("class_ious") or {}).get("Roads"), 0.0),
        "Built-up IoU": lambda m: safe_num(((m.get("verified_metrics") or {}).get("class_ious") or {}).get("Built-up"), 0.0),
        "Water IoU": lambda m: safe_num(((m.get("verified_metrics") or {}).get("class_ious") or {}).get("Water"), 0.0),
        "Vegetation IoU": lambda m: safe_num(((m.get("verified_metrics") or {}).get("class_ious") or {}).get("Vegetation"), 0.0)
    }

    # Sort models by selected metric descending for ranked visual comparison
    sorted_chart_models = sorted(all_models, key=lambda m: metric_lookup[selected_metric_name](m), reverse=True)
    names = [m["display_name"].split("(")[0].strip() for m in sorted_chart_models]
    scores = [metric_lookup[selected_metric_name](m) for m in sorted_chart_models]

    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.tick_params(colors="#0F172A", labelsize=10.5)
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")

    bar_colors = []
    for i, m in enumerate(sorted_chart_models):
        if i == 0:
            bar_colors.append("#D97706") # Gold Leader
        elif i == 1:
            bar_colors.append("#64748B") # Silver
        elif i == 2:
            bar_colors.append("#2563EB") # Bronze
        elif m.get("id", "").startswith("github_") or m.get("id", "").startswith("custom_"):
            bar_colors.append("#7C3AED") # Contributed Purple
        else:
            bar_colors.append("#3B82F6") # Core Blue

    bars = ax.bar(names, scores, color=bar_colors, edgecolor="white", lw=1.2, width=0.48)

    for bar in bars:
        h = safe_num(bar.get_height(), 0.0)
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10, color="#0F172A", fontweight="bold")

    ax.set_ylabel(f"{selected_metric_name} (%)", color="#334155", fontsize=12, fontweight="bold")
    ax.set_title(f"Ranked Leaderboard Benchmark: {selected_metric_name} (Gold = #1, Silver = #2, Bronze = #3)",
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
    for rank_idx, m in enumerate(ranked_models, start=1):
        metrics = m.get("verified_metrics") or {}
        ious = metrics.get("class_ious") or {}
        oa = safe_num(metrics.get("overall_accuracy"), 0.0)
        miou = safe_num(metrics.get("macro_miou"), 0.0)
        cw = m.get("class_weights") or {}
        cw_w = safe_num(cw.get("Water"), 1.0)
        cw_v = safe_num(cw.get("Vegetation"), 1.0)
        cw_b = safe_num(cw.get("Barren Land"), 1.0)
        cw_u = safe_num(cw.get("Built-up"), 1.0)
        cw_r = safe_num(cw.get("Roads"), 1.0)
        cw_str = f"[{cw_w:.1f}, {cw_v:.1f}, {cw_b:.1f}, {cw_u:.1f}, {cw_r:.1f}]"

        iou_w = safe_num(ious.get("Water"), 0.0)
        iou_v = safe_num(ious.get("Vegetation"), 0.0)
        iou_u = safe_num(ious.get("Built-up"), 0.0)
        iou_r = safe_num(ious.get("Roads"), 0.0)

        if rank_idx == 1:
            rank_badge = '<span class="badge" style="background: #FEF3C7; color: #B45309; border: 1.5px solid #FCD34D; font-weight: 800;">🥇 #1</span>'
        elif rank_idx == 2:
            rank_badge = '<span class="badge" style="background: #F1F5F9; color: #334155; border: 1.5px solid #CBD5E1; font-weight: 800;">🥈 #2</span>'
        elif rank_idx == 3:
            rank_badge = '<span class="badge" style="background: #EFF6FF; color: #1D4ED8; border: 1.5px solid #BFDBFE; font-weight: 800;">🥉 #3</span>'
        else:
            rank_badge = f'<span class="badge" style="background: #F8FAFC; color: #64748B; border: 1px solid #E2E8F0; font-weight: 700;">#{rank_idx}</span>'

        is_core = m.get("id") in ["wv3_native_segformer", "cartosat_pan_segformer", "wv3_native_unet"]
        cat_badge = '<span class="badge badge-ready">🌟 Core</span>' if is_core else '<span class="badge" style="background: #F5F3FF; color: #7C3AED; border: 1px solid #DDD6FE;">👥 Contributed</span>'
        m_gsd = safe_num(m.get('native_gsd_m'), 1.0)
        m_desc = m.get('description') or "Empirical model checkpoint."
        m_arch = m.get('architecture') or "Deep Neural Network"
        m_sensor = m.get('sensor') or "Optical"
        m_dname = m.get('display_name') or "Contributed Model"

        rows += f"""
        <tr style="border-bottom: 1px solid #E2E8F0;">
            <td style="padding: 12px 8px; text-align: center;">{rank_badge}</td>
            <td style="padding: 12px 10px;">{cat_badge}</td>
            <td style="padding: 12px 12px; font-weight: 700; color: #0F172A;">
                {m_dname}<br><span style="font-size: 0.82rem; color: #64748B; font-weight: 500;">{m_arch}</span>
            </td>
            <td style="padding: 12px 10px; color: #334155; font-size: 0.88rem; font-weight: 600;">{m_sensor}<br><span style="color: #2563EB;">{m_gsd:.2f}m GSD</span></td>
            <td style="padding: 12px 10px; color: #1D4ED8; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{oa:.1f}%</td>
            <td style="padding: 12px 10px; color: #059669; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{miou:.1f}%</td>
            <td style="padding: 12px 10px; color: #6D28D9; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem;">{cw_str}</td>
            <td style="padding: 12px 8px; color: #0284C7; font-weight: 600;">{iou_w:.1f}%</td>
            <td style="padding: 12px 8px; color: #16A34A; font-weight: 600;">{iou_v:.1f}%</td>
            <td style="padding: 12px 8px; color: #DC2626; font-weight: 600;">{iou_u:.1f}%</td>
            <td style="padding: 12px 8px; color: #D97706; font-weight: 600;">{iou_r:.1f}%</td>
            <td style="padding: 12px 10px; font-size: 0.85rem; color: #475569; font-weight: 500;">{m_desc}</td>
        </tr>
        """

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 14px; overflow-x: auto; box-shadow: 0 2px 8px rgba(15,23,42,0.04);">
        <table style="width: 100%; border-collapse: collapse; font-size: 1.02rem; text-align: left; min-width: 1080px;">
            <thead>
                <tr style="background: #F8FAFC; border-bottom: 2px solid #CBD5E1;">
                    <th style="padding: 14px 8px; color: #475569; font-weight: 800; text-align: center;">Rank</th>
                    <th style="padding: 14px 10px; color: #475569; font-weight: 800;">Source</th>
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

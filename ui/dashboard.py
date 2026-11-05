"""
ui/dashboard.py
---------------
Home / Dashboard component for the Urban Feature Extraction AI Platform.
Presents the executive overview, dynamic KPIs, multi-model registry, and benchmark comparison.
Uses pure HTML/CSS tables to avoid Windows Application Control / pandas binary issues.
"""

import streamlit as st
from model_registry import get_all_models, get_ready_models, get_pending_models, CLASS_MAPPING

def render_dashboard():
    # Hero Section
    st.markdown("""
    <div class="platform-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
            <div>
                <div class="platform-title">Urban Feature Extraction AI Platform</div>
                <div class="platform-subtitle">
                    Unified Geospatial Semantic Segmentation for High-Resolution & Multispectral Earth Observation
                </div>
            </div>
            <div>
                <span class="badge badge-gpu" style="font-size: 0.85rem; padding: 6px 14px;">
                    ⚡ GPU ACCELERATED • NVIDIA RTX 4050
                </span>
            </div>
        </div>
        <div style="margin-top: 16px; color: #CBD5E1; font-size: 0.92rem; line-height: 1.6; max-width: 950px;">
            This platform integrates satellite semantic segmentation models developed independently by our 3-member project team.
            It provides standardized evaluation, multi-sensor band adaptation, sliding-window tiled inference, and GIS vectorization
            across <b>5 core urban classes</b>: Water, Vegetation, Barren Land, Built-up Structures, and Road Corridors.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic KPI Metric Cards
    all_models = get_all_models()
    ready_models = get_ready_models()
    pending_models = get_pending_models()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{len(all_models)}</div>
            <div class="kpi-label">Registered Models</div>
            <div class="kpi-sub">Across 3 team members</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color: #34D399;">{len(ready_models)}</div>
            <div class="kpi-label">Loadable & Ready</div>
            <div class="kpi-sub">Verified checkpoints loaded</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color: #FBBF24;">{len(pending_models)}</div>
            <div class="kpi-label">Pending Registration</div>
            <div class="kpi-sub">Teammate integration slots</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val" style="color: #A5B4FC;">4 Sensors</div>
            <div class="kpi-label">Supported Modalities</div>
            <div class="kpi-sub">WV-3 (0.31m/1.24m), Carto (2.5m), S2 (10m)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Target Class Legend Banner
    st.markdown("""
    <div class="glass-panel" style="padding: 16px 20px; margin-bottom: 22px;">
        <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 10px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">
            Standard 5-Class Project Schema & Color Coding
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <div class="legend-pill"><span class="color-dot" style="background:#0866FF;"></span><b>Water (0)</b></div>
            <div class="legend-pill"><span class="color-dot" style="background:#218C2C;"></span><b>Vegetation (1)</b></div>
            <div class="legend-pill"><span class="color-dot" style="background:#E0A51B;"></span><b>Barren Land (2)</b></div>
            <div class="legend-pill"><span class="color-dot" style="background:#E71345;"></span><b>Built-up (3)</b></div>
            <div class="legend-pill"><span class="color-dot" style="background:#999999;"></span><b>Roads (4)</b></div>
            <div class="legend-pill"><span class="color-dot" style="background:#292929;"></span><b>Ignore / Boundary (255)</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Multi-Model Registry Comparison Table (Pure HTML/CSS — No Pandas Dependency)
    st.markdown('<div class="section-header">📁 Central Model Inventory & Benchmark Evidence</div>', unsafe_allow_html=True)
    
    table_rows_html = ""
    for m in all_models:
        metrics = m.get("verified_metrics", {})
        oa = f"<b>{metrics.get('overall_accuracy'):.2f}%</b>" if metrics.get('overall_accuracy') is not None else "<span style='color:#64748B;'>Pending</span>"
        miou = f"{metrics.get('macro_miou'):.2f}%" if metrics.get('macro_miou') is not None else "<span style='color:#64748B;'>Pending</span>"
        
        road_iou = "—"
        bld_iou = "—"
        if "class_ious" in metrics and metrics["class_ious"]:
            if "Roads" in metrics["class_ious"]:
                road_iou = f"<span style='color:#38BDF8; font-weight:600;'>{metrics['class_ious']['Roads']:.1f}%</span>"
            if "Built-up" in metrics["class_ious"]:
                bld_iou = f"<span style='color:#F43F5E; font-weight:600;'>{metrics['class_ious']['Built-up']:.1f}%</span>"

        badge_html = "<span class='badge badge-ready'>Ready</span>" if m["status"] == "Ready" else "<span class='badge badge-pending'>Pending</span>"

        table_rows_html += f"""
        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); transition: background 0.2s;">
            <td style="padding: 10px 14px; font-weight: 600; color: #F8FAFC;">{m['display_name']}</td>
            <td style="padding: 10px 14px; color: #94A3B8; font-size: 0.82rem;">{m['owner']}</td>
            <td style="padding: 10px 14px; color: #CBD5E1; font-size: 0.82rem;">{m['architecture']}</td>
            <td style="padding: 10px 14px; color: #38BDF8; font-size: 0.82rem;">{m['sensor']} ({m['native_gsd_m']}m)</td>
            <td style="padding: 10px 14px; color: #34D399; font-family: monospace;">{oa}</td>
            <td style="padding: 10px 14px; font-family: monospace;">{miou}</td>
            <td style="padding: 10px 14px; font-family: monospace;">{road_iou}</td>
            <td style="padding: 10px 14px; font-family: monospace;">{bld_iou}</td>
            <td style="padding: 10px 14px;">{badge_html}</td>
        </tr>
        """

    st.markdown(f"""
    <div class="glass-panel" style="padding: 0; overflow-x: auto; margin-bottom: 18px;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.86rem;">
            <thead>
                <tr style="background: rgba(30, 41, 59, 0.75); border-bottom: 2px solid rgba(56, 189, 248, 0.3);">
                    <th style="padding: 12px 14px; color: #F1F5F9;">Display Name</th>
                    <th style="padding: 12px 14px; color: #94A3B8;">Owner</th>
                    <th style="padding: 12px 14px; color: #94A3B8;">Architecture</th>
                    <th style="padding: 12px 14px; color: #94A3B8;">Sensor / GSD</th>
                    <th style="padding: 12px 14px; color: #34D399;">Held-Out OA</th>
                    <th style="padding: 12px 14px; color: #CBD5E1;">Held-Out mIoU</th>
                    <th style="padding: 12px 14px; color: #38BDF8;">Road IoU</th>
                    <th style="padding: 12px 14px; color: #F43F5E;">Built-up IoU</th>
                    <th style="padding: 12px 14px; color: #CBD5E1;">Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Scientific Rigor Notice
    st.markdown("""
    <div class="callout-box callout-info">
        <b>💡 Benchmark Protocol & Scientific Integrity:</b>
        The metrics displayed above represent rigorous <b>strictly held-out test splits</b> (e.g. South Mumbai for WorldView-3 & Cartosat-1) evaluated once after training.
        Performance across different sensors (0.31m vs 1.24m vs 2.5m vs 10m) is influenced by spatial resolution and native radiometric bands;
        consequently, metrics from different test splits are not directly interchangeable.
        When you run inference on new imagery, accuracy/IoU metrics cannot be computed without verified ground-truth labels.
    </div>
    """, unsafe_allow_html=True)

    # Team Attribution Breakdown
    st.markdown('<div class="section-header" style="margin-top: 25px;">👥 Multi-Member Collaborative Architecture</div>', unsafe_allow_html=True)
    
    col_lead, col_f1, col_f2 = st.columns(3)
    with col_lead:
        st.markdown("""
        <div class="glass-panel" style="height: 100%;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #38BDF8; margin-bottom: 8px;">
                Me / Team Lead (Anand)
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                • <b>Cartosat-1 PAN:</b> SegFormer-B1 (2.5m)<br>
                • <b>WorldView-3 MS:</b> SegFormer-B1, U-Net ResNet34, DeepLabV3+ (1.24m)<br>
                • <b>Pan-Sharpened:</b> SegFormer-B1 (0.31m)<br>
                • <b>Specialized:</b> Targeted FT & Probability Ensemble<br>
                • <b>Regional Baseline:</b> Sentinel-2A 4-Band (10m)<br>
            </div>
            <div style="margin-top: 12px;">
                <span class="badge badge-ready">8 Models Verified & Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_f1:
        st.markdown("""
        <div class="glass-panel" style="height: 100%;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #FBBF24; margin-bottom: 8px;">
                Teammate 1 (Friend 1)
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                • <b>Model A:</b> HRNet-W48 High-Resolution Road Segmenter<br>
                • <b>Model B:</b> Swin-Transformer Building Footprint Extractor<br>
                • <i>Pending:</i> Checkpoint weights file & inference adapter submission.<br>
            </div>
            <div style="margin-top: 24px;">
                <span class="badge badge-pending">2 Slots Reserved</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_f2:
        st.markdown("""
        <div class="glass-panel" style="height: 100%;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #FBBF24; margin-bottom: 8px;">
                Teammate 2 (Friend 2)
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5;">
                • <b>Model A:</b> ConvNeXt-Base Land Cover & Water Classifier<br>
                • <b>Model B:</b> PSPNet Edge-Loss Boundary Refiner<br>
                • <i>Pending:</i> Checkpoint weights file & inference adapter submission.<br>
            </div>
            <div style="margin-top: 24px;">
                <span class="badge badge-pending">2 Slots Reserved</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

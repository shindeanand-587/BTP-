"""
app.py
------
Unified Urban Feature Extraction Platform.
AI-Powered Multi-Sensor Earth Observation & Urban Semantic Segmentation.
"""

import os
import sys
import types
import importlib

# Ensure Urban_Extraction_Platform directory is in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Configure PIL for Massive Satellite Imagery (>650MB / >300 Megapixels)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Windows Application Control / WDAC Safety Guard
try:
    import pandas as pd
except Exception:
    mock_pd = types.ModuleType("pandas")
    mock_pd.__version__ = "stub"
    mock_pd.DataFrame = list
    mock_pd.Series = list
    sys.modules["pandas"] = mock_pd

# PyTorch 2.6 Guard for HuggingFace Transformers
import torch
try:
    import torch._dynamo
    import torch._dynamo.config
except Exception:
    pass

import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Urban Feature Extraction Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Custom Modern Light Stylesheet
CSS_PATH = os.path.join(APP_DIR, "assets", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        custom_css = f.read()
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# Import view modules with reload protection
import ui.prediction_view
import ui.comparison_view
import ui.leaderboard_view
import ui.teammate_hub

importlib.reload(ui.prediction_view)
importlib.reload(ui.comparison_view)
importlib.reload(ui.leaderboard_view)
importlib.reload(ui.teammate_hub)

from ui.prediction_view import render_prediction_page
from ui.comparison_view import render_comparison_page
from ui.leaderboard_view import render_leaderboard_page
from ui.teammate_hub import render_teammate_hub_page

# Clean, Professional Product Header with v2 Intelligence Badge
st.markdown("""
<div class="platform-hero">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div class="platform-title">
                🛰️ Urban Feature Extraction Platform
            </div>
            <div class="platform-subtitle">
                Automated High-Resolution Satellite & Aerial Imagery Semantic Segmentation
            </div>
        </div>
        <div>
            <span class="badge badge-sensor" style="font-size: 0.95rem; padding: 6px 16px; font-weight: 700;">
                🔬 Advanced Intelligence Studio (v2)
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Streamlined 4-Tab Navigation with Professional Labels
tab_pred, tab_comp, tab_leaderboard, tab_teammates = st.tabs([
    "🎯 Feature Extraction",
    "🔬 Multi-Model Comparison",
    "🏆 Model Leaderboard",
    "👥 Model Registry"
])

with tab_pred:
    render_prediction_page()

with tab_comp:
    render_comparison_page()

with tab_leaderboard:
    render_leaderboard_page()

with tab_teammates:
    render_teammate_hub_page()

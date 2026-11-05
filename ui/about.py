"""
ui/about.py
-----------
About / Technical Details component.
Documents project objectives, sensor parameters, pseudo-label generation,
team attribution, and system limitations.
"""

import streamlit as st

def render_about():
    st.markdown('<div class="section-header">ℹ️ About the Platform & Technical Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Overview
    This platform was built as part of <b>B.Tech Project II: Unified Urban Feature Extraction from Satellite Imagery</b>,
    focused on the Mumbai Metropolitan Region (MMR), Maharashtra, India.
    
    The primary research objective is to resolve urban feature mixing, road-bloat, and boundary confusion by transitioning from
    coarse optical satellite sensors (10m Sentinel-2A) to high-resolution Panchromatic (2.5m Cartosat-1) and
    very high-resolution 8-band Multispectral imagery (1.24m & 0.31m Maxar WorldView-3).
    """)

    # 5 Target Classes Card
    st.markdown("""
    <div class="glass-panel" style="margin-top: 15px;">
        <div style="font-weight: 700; font-size: 1.1rem; color: #38BDF8; margin-bottom: 10px;">
            The 5 Target Semantic Classes
        </div>
        <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.7;">
            • <b style="color: #0866FF;">Class 0 — Water:</b> Open sea, coastal bays, tidal creeks, reservoirs, inland drainage canals.<br>
            • <b style="color: #218C2C;">Class 1 — Vegetation:</b> Mangrove buffers, forested hillsides, urban tree canopies, parks, turf.<br>
            • <b style="color: #E0A51B;">Class 2 — Barren Land:</b> Active rock quarries, dry excavation pits, bare exposed soil.<br>
            • <b style="color: #E71345;">Class 3 — Built-up:</b> High-rise towers, commercial complexes, dense residential buildings, informal settlements.<br>
            • <b style="color: #999999;">Class 4 — Roads:</b> Arterial highways, coastal expressways, bridges, planned street corridors.<br>
            • <b style="color: #64748B;">Class 255 — Ignore / Margin:</b> Boundary buffer zones, mixed boundary transitions, satellite nodata.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sensor Specifications
    st.markdown('<div class="section-header" style="margin-top: 25px;">🛰️ Sensor Modalities & Radiometric Specifications</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="glass-panel" style="height: 100%;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 1rem; margin-bottom: 6px;">
                Maxar WorldView-3
            </div>
            <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.5;">
                • <b>Native MS Resolution:</b> 1.24 m GSD<br>
                • <b>Pan-Sharpened:</b> 0.31 m GSD<br>
                • <b>Bands (8):</b> Coastal, Blue, Green, Yellow, Red, Red Edge, NIR1, NIR2<br>
                • <b>Key Advantage:</b> Resolves 7m urban roads sharply; high spectral separation between asphalt and concrete rooftops.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="glass-panel" style="height: 100%;">
            <div style="font-weight: 700; color: #34D399; font-size: 1rem; margin-bottom: 6px;">
                ISRO Cartosat-1
            </div>
            <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.5;">
                • <b>Resolution:</b> 2.50 m GSD<br>
                • <b>Bands (1):</b> Panchromatic (0.50–0.85 µm)<br>
                • <b>Swath / Extent:</b> 27 km full scenes (18,318 × 14,206 px)<br>
                • <b>Key Advantage:</b> Broad spatial coverage; sharp geometry for high-contrast building edges.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="glass-panel" style="height: 100%;">
            <div style="font-weight: 700; color: #A5B4FC; font-size: 1rem; margin-bottom: 6px;">
                ESA Sentinel-2A
            </div>
            <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.5;">
                • <b>Resolution:</b> 10.0 m GSD<br>
                • <b>Bands (4):</b> B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)<br>
                • <b>Surface Radiometry:</b> Level-2A Bottom-of-Atmosphere (BOA)<br>
                • <b>Key Advantage:</b> Free regional synoptic view; strong spectral vegetation & water detection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Known Limitations & Integrity
    st.markdown('<div class="section-header" style="margin-top: 25px;">⚠️ Known Technical Limitations & Constraints</div>', unsafe_allow_html=True)
    st.markdown("""
    1. <b>Cloud & Cloud Shadow Sensitivity:</b> Optical models do not have a dedicated cloud class. In scenes containing localized cumulus clouds, cloud tops may be predicted as built-up/road and shadows as water/vegetation. Sentinel-2 Scene Classification Layer (SCL) masking is recommended for cloudy acquisitions.
    2. <b>Barren Land Under-representation:</b> Rock quarries account for only 0.87% of total pixels across the MMR study area. In the held-out South Mumbai split, 0 barren pixels were present, yielding 0.0% IoU on that specific test split despite active detection in quarry regions (Parsik Hill).
    3. <b>Experimental RGB Adaptation:</b> When standard 3-channel RGB imagery is submitted to 8-band or 4-band models, spectral bands (such as NIR and Red Edge) are approximated. While this produces visually informative predictions, it should not be treated as equivalent to native multispectral radiometry.
    4. <b>Independent Split Autonomy:</b> All models adhere to strict spatial split policies. Buffer zones between train, validation, and test sets prevent geographic overlap and data leakage.
    """)

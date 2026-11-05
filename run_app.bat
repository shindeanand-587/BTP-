@echo off
title Urban Feature Extraction Platform v2 (Experimental Insights)
echo =========================================================================
echo Launching Urban Feature Extraction Platform v2 on http://localhost:8502
echo =========================================================================
cd /d "C:\BTP II\Urban_Extraction_Platform_v2"
"C:\BTP II\.venv\Scripts\streamlit.exe" run app.py --server.port 8502 --server.headless true
pause

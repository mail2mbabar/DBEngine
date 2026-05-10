@echo off
cd /d "%~dp0"
python -m streamlit run web\app.py --server.headless true

@echo off
REM Launch the BraTS web viewer (Frontend/) with live-reload.
REM Runs from the repo root so config.yaml and watch paths resolve.
cd /d "%~dp0"
".venv\Scripts\python.exe" Frontend\serve.py

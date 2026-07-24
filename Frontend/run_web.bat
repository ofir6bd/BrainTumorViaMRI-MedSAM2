@echo off
REM Launch the BraTS web viewer with live-reload.
REM This bat lives in Frontend\ but runs from the repo root (one level up)
REM so config.yaml and the watch paths resolve correctly.
cd /d "%~dp0.."
".venv\Scripts\python.exe" Frontend\serve.py

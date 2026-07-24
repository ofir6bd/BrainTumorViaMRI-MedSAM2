@echo off
REM Launch the BraTS web viewer with live-reload.
REM This bat lives in Frontend\ but runs from the repo root (one level up)
REM so config.yaml and the watch paths resolve correctly.
cd /d "%~dp0.."
set "URL=http://localhost:5000"

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if %ERRORLEVEL%==0 (
	echo Viewer is already running on %URL%.
	start "" "%URL%"
	exit /b 0
)

start "" "%URL%"
".venv\Scripts\python.exe" Frontend\serve.py

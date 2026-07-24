@echo off
REM Launch the BraTS web viewer (Frontend/) with live-reload.
REM Runs from the repo root so config.yaml and watch paths resolve.
cd /d "%~dp0"
set "URL=http://localhost:5000"

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if %ERRORLEVEL%==0 (
	echo Viewer is already running on %URL%.
	start "" "%URL%"
	exit /b 0
)

start "" "%URL%"
".venv\Scripts\python.exe" Frontend\serve.py

@echo off
setlocal

REM Run this script from its own folder
cd /d "%~dp0"

echo ============================================
echo Group 8 YOLOv10 - Inference App Launcher
echo ============================================

REM 1) Find Python launcher
set "PY_CMD="
where py >nul 2>&1
if %errorlevel%==0 set "PY_CMD=py -3"

if not defined PY_CMD (
    where python >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo [ERROR] Python not found in PATH.
    echo Install Python 3.10+ from https://python.org and try again.
    pause
    exit /b 1
)

REM 2) Create local virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating local virtual environment...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set "VENV_PY=.venv\Scripts\python.exe"

REM 3) Install dependencies only if imports are missing
"%VENV_PY%" -c "import torch,cv2,PIL,ultralytics" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies from requirements.txt...
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

REM 4) Run app
echo [INFO] Starting inference app...
"%VENV_PY%" inference_app.py

if errorlevel 1 (
    echo.
    echo [ERROR] App exited with an error.
    pause
    exit /b 1
)

echo.
echo [INFO] App closed.
exit /b 0

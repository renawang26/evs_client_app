@echo off
REM EVS Navigation System - Windows Setup Script (Conda)
REM This script creates conda environment and installs dependencies

setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   EVS Navigation System - Setup Script [Conda]
echo ============================================================
echo.

REM ============================================================
REM Check Conda Installation
REM ============================================================
echo [1/4] Checking Conda installation...

where conda >nul 2>&1
if !errorLevel! equ 0 (
    echo       Conda found.
) else (
    echo       Conda not found.
    echo.
    echo [ERROR] Please install Anaconda or Miniconda first:
    echo         https://www.anaconda.com/download
    echo         https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

REM ============================================================
REM Check FFmpeg Installation
REM ============================================================
echo.
echo [2/4] Checking FFmpeg installation...

where ffmpeg >nul 2>&1
if !errorLevel! equ 0 (
    echo       FFmpeg found.
) else (
    echo       FFmpeg not found.
    echo.
    echo [INFO] FFmpeg is required for audio processing.
    echo        Install with conda: conda install -c conda-forge ffmpeg
    echo        Or download from: https://ffmpeg.org/download.html
    echo.
)

REM ============================================================
REM Create/Update Conda Environment
REM ============================================================
echo.
echo [3/4] Setting up Conda environment [cw_evs_app]...

REM Check if environment exists
conda env list | findstr /C:"cw_evs_app" >nul 2>&1
if !errorLevel! equ 0 (
    echo       Environment 'cw_evs_app' already exists.
    set /p RECREATE="Recreate environment? [y/n]: "
    if /i "!RECREATE!" equ "y" (
        echo       Removing old environment...
        call conda deactivate 2>nul
        conda env remove -n cw_evs_app -y
        echo       Creating new environment...
        conda create -n cw_evs_app python -y
    )
) else (
    echo       Creating new environment 'cw_evs_app'...
    conda create -n cw_evs_app python -y
)

REM Activate environment
echo       Activating environment...
call conda activate cw_evs_app

REM Install FFmpeg via conda
echo       Installing FFmpeg via conda...
conda install -c conda-forge ffmpeg -y

REM Install PyTorch with CUDA support
echo       Installing PyTorch [this may take a while]...
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y 2>nul
if !errorLevel! neq 0 (
    echo       CUDA 12.4 not available, trying CUDA 12.1...
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y 2>nul
    if !errorLevel! neq 0 (
        echo       CUDA not available, installing CPU version...
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    )
)

REM Install pip dependencies
echo.
echo       Installing pip dependencies...
pip install -r requirements.txt

if !errorLevel! neq 0 (
    echo [ERROR] Failed to install some requirements.
    echo        You may need to install them manually.
)

REM Verify key packages are installed
echo.
echo       Verifying installation...
python -c "import streamlit; import transformers; import funasr; import plotly; import numpy; import scipy; print('All key packages verified.')"
if !errorLevel! neq 0 (
    echo [WARNING] Some packages failed to import. Check errors above.
)

REM ============================================================
REM Initialize Database
REM ============================================================
echo.
echo [4/4] Initializing database...

if exist "data\evs_repository.db" (
    echo       Database already exists.
) else (
    python init_database.py
    if !errorLevel! neq 0 (
        echo [WARNING] Database initialization had issues.
    ) else (
        echo       Database initialized successfully.
    )
)

REM ============================================================
REM Setup Complete
REM ============================================================
echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo ASR Providers:
echo   English:  CrisperWhisper (nyrahealth/CrisperWhisper)
echo   Chinese:  FunASR Paraformer-ZH (Alibaba)
echo.
echo NOTE: The CrisperWhisper model (~3GB) will be downloaded
echo       automatically on first use. Progress is shown in the GUI.
echo.
echo To start the application:
echo   1. Run start.bat
echo   2. Or manually:
echo      conda activate cw_evs_app
echo      python -m streamlit run app.py
echo.
echo The application will be available at:
echo   http://localhost:8501
echo.
pause

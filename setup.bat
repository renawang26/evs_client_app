@echo off
REM EVS Navigation System - Windows Setup Script (Conda)
REM Usage: setup.bat              - Setup (skip if env exists)
REM        setup.bat --reinstall  - Remove env and redo full setup

setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Parse arguments
set "REINSTALL=0"
if /i "%~1"=="--reinstall" set "REINSTALL=1"
if /i "%~1"=="-r" set "REINSTALL=1"

echo ============================================================
echo   EVS Navigation System - Setup Script [Conda]
echo ============================================================
if !REINSTALL! equ 1 (
    echo   Mode: Force Reinstall
)
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
REM Create Conda Environment
REM ============================================================
echo.
echo [3/4] Setting up Conda environment [cw_evs_app]...

call conda env list | findstr /C:"cw_evs_app" >nul 2>&1
if !errorLevel! equ 0 (
    if !REINSTALL! equ 1 (
        echo       Removing existing environment...
        call conda deactivate
        call conda env remove -n cw_evs_app -y
        echo       Creating new environment [Python 3.11]...
        call conda create -n cw_evs_app python=3.11 -y
    ) else (
        echo       Environment 'cw_evs_app' already exists. Skipping.
        echo       Use 'setup.bat --reinstall' to recreate.
        goto :activate
    )
) else (
    echo       Creating new environment [Python 3.11]...
    call conda create -n cw_evs_app python=3.11 -y
)

:activate
REM Activate environment
echo       Activating environment...
call conda activate cw_evs_app

REM Verify activation succeeded
if /i "!CONDA_DEFAULT_ENV!" neq "cw_evs_app" (
    echo [ERROR] Failed to activate conda environment 'cw_evs_app'.
    echo        Try running: conda init cmd.exe
    echo        Then reopen this terminal and run setup.bat again.
    pause
    exit /b 1
)
echo       Environment activated: !CONDA_DEFAULT_ENV!

REM Install FFmpeg via conda
echo       Installing FFmpeg via conda...
call conda install -c conda-forge ffmpeg -y

REM Install PyTorch via conda (before pip to avoid CPU-only wheels from pip mirrors)
echo.
echo       Detecting GPU...
nvidia-smi >nul 2>&1
if !errorLevel! equ 0 (
    echo       NVIDIA GPU detected. Installing PyTorch with CUDA via conda...
    call conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
    if !errorLevel! neq 0 (
        echo       CUDA 12.4 failed, trying CUDA 12.1...
        call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    )
) else (
    echo       No NVIDIA GPU detected. Installing CPU-only PyTorch via conda...
    call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
)

REM Install pip dependencies (torch/torchaudio already installed by conda, not in requirements.txt)
echo.
echo       Installing pip dependencies...
pip install -r requirements.txt

if !errorLevel! neq 0 (
    echo [ERROR] Failed to install some requirements.
    echo        Try: setup.bat --reinstall
)

REM Verify key packages are installed
echo.
echo       Verifying installation...
python -c "import torch; gpu='CUDA '+torch.version.cuda if torch.cuda.is_available() else 'CPU only'; print(f'PyTorch {torch.__version__} ({gpu})')"
python -c "import streamlit; import transformers; import funasr; import plotly; import numpy; import scipy; print('All key packages verified.')"
if !errorLevel! neq 0 (
    echo [WARNING] Some packages failed to import. Check errors above.
)

REM ============================================================
REM Initialize Database
REM ============================================================
echo.
echo [4/4] Initializing database...

if !REINSTALL! equ 1 (
    echo       Reinitializing database...
    if exist "data\evs_repository.db" del "data\evs_repository.db"
    python init_database.py
    if !errorLevel! neq 0 (
        echo [WARNING] Database reinitialization had issues.
    ) else (
        echo       Database reinitialized successfully.
    )
) else if exist "data\evs_repository.db" (
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
echo To start the application:
echo   1. Run start.bat
echo   2. Or manually:
echo      conda activate cw_evs_app
echo      python -m streamlit run app.py
echo.
echo If packages are missing, run:
echo   setup.bat --reinstall
echo.
echo The application will be available at:
echo   http://localhost:8501
echo.
pause

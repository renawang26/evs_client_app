@echo off
REM EVS Navigation System - Windows Setup Script
REM Usage: setup.bat              - Setup (skip if env exists)
REM        setup.bat --reinstall  - Remove env and redo full setup
REM
REM Strategy: conda creates the Python env ONLY. All packages installed via pip.
REM FFmpeg is a system binary (winget), NOT a conda package.

setlocal
cd /d "%~dp0"

REM Parse arguments
set "REINSTALL=0"
if /i "%~1"=="--reinstall" set "REINSTALL=1"
if /i "%~1"=="-r" set "REINSTALL=1"

echo ============================================================
echo   EVS Navigation System - Setup Script
echo ============================================================
if %REINSTALL% equ 1 (
    echo   Mode: Force Reinstall
)
echo.

REM ============================================================
REM Check Conda Installation
REM ============================================================
echo [1/4] Checking Conda installation...

where conda >nul 2>&1
if not errorlevel 1 (
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
REM Check FFmpeg Installation (system binary)
REM ============================================================
echo.
echo [2/4] Checking FFmpeg installation...

where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo       FFmpeg found on system PATH.
    goto :ffmpeg_done
)

echo       FFmpeg not found. Attempting install via winget...
where winget >nul 2>&1
if errorlevel 1 (
    echo       winget not available.
    goto :ffmpeg_manual
)

winget install --id=Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements >nul 2>&1
if errorlevel 1 (
    echo       winget install failed.
    goto :ffmpeg_manual
)

REM winget installs may not update the current session PATH, so check again
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo       FFmpeg installed successfully via winget.
    goto :ffmpeg_done
)

echo       FFmpeg installed but not yet on PATH.
echo       You may need to reopen this terminal for PATH changes to take effect.
goto :ffmpeg_done

:ffmpeg_manual
echo.
echo [WARNING] FFmpeg could not be installed automatically.
echo           Audio processing will not work until FFmpeg is installed.
echo.
echo           Option 1 - Winget (recommended):
echo             winget install --id=Gyan.FFmpeg -e
echo.
echo           Option 2 - Chocolatey:
echo             choco install ffmpeg
echo.
echo           Option 3 - Manual download:
echo             1. Go to https://www.gyan.dev/ffmpeg/builds/
echo             2. Download "ffmpeg-release-essentials.zip"
echo             3. Extract to C:\ffmpeg
echo             4. Add C:\ffmpeg\bin to your system PATH
echo             5. Reopen this terminal
echo.
:ffmpeg_done

REM ============================================================
REM Create Conda Environment (Python only — no conda packages)
REM ============================================================
echo.
echo [3/4] Setting up Conda environment [cw_evs_app]...

cmd /c "conda env list | findstr /C:cw_evs_app" >nul 2>&1
if not errorlevel 1 (
    if %REINSTALL% equ 1 (
        echo       Removing existing environment...
        call conda deactivate
        cmd /c conda env remove -n cw_evs_app -y
        echo       Creating new environment [Python 3.11]...
        cmd /c conda create -n cw_evs_app python=3.11 -y
    ) else (
        echo       Environment 'cw_evs_app' already exists. Skipping.
        echo       Use 'setup.bat --reinstall' to recreate.
        goto :activate
    )
) else (
    echo       Creating new environment [Python 3.11]...
    cmd /c conda create -n cw_evs_app python=3.11 -y
)

:activate
REM Activate environment
echo       Activating environment...
call conda activate cw_evs_app

REM Verify activation succeeded
if /i "%CONDA_DEFAULT_ENV%" neq "cw_evs_app" (
    echo [ERROR] Failed to activate conda environment 'cw_evs_app'.
    echo        Try running: conda init cmd.exe
    echo        Then reopen this terminal and run setup.bat again.
    pause
    exit /b 1
)
echo       Environment activated: %CONDA_DEFAULT_ENV%

REM ============================================================
REM Install PyTorch via pip (GPU or CPU)
REM ============================================================
echo.
echo       Detecting GPU...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo       NVIDIA GPU detected. Installing PyTorch with CUDA 12.4 via pip...
    pip install "torch>=2.0.0,<2.9.0" "torchaudio>=2.0.0,<2.9.0" --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo       CUDA 12.4 wheels failed, trying CUDA 12.1...
        pip install "torch>=2.0.0,<2.9.0" "torchaudio>=2.0.0,<2.9.0" --index-url https://download.pytorch.org/whl/cu121
    )
) else (
    echo       No NVIDIA GPU detected. Installing CPU-only PyTorch via pip...
    pip install "torch>=2.0.0,<2.9.0" "torchaudio>=2.0.0,<2.9.0"
)

REM ============================================================
REM Install pip dependencies
REM ============================================================
echo.
echo       Installing pip dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install some requirements.
    echo        Try: setup.bat --reinstall
)

REM Verify key packages are installed
echo.
echo       Verifying installation...
python -c "import torch; gpu='CUDA '+torch.version.cuda if torch.cuda.is_available() else 'CPU only'; print(f'PyTorch {torch.__version__} ({gpu})')"
python -c "import streamlit; import transformers; import funasr; import plotly; import numpy; import scipy; print('All key packages verified.')"
if errorlevel 1 (
    echo [WARNING] Some packages failed to import. Check errors above.
)

REM ============================================================
REM Initialize Database
REM ============================================================
echo.
echo [4/4] Initializing database...

if %REINSTALL% equ 1 (
    echo       Reinitializing database...
    if exist "data\evs_repository.db" del "data\evs_repository.db"
    python init_database.py
    if errorlevel 1 (
        echo [WARNING] Database reinitialization had issues.
    ) else (
        echo       Database reinitialized successfully.
    )
) else if exist "data\evs_repository.db" (
    echo       Database already exists.
) else (
    python init_database.py
    if errorlevel 1 (
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

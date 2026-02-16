@echo off
REM EVS Navigation System - Start Script (Conda)
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   EVS Navigation System - Starting Application
echo ============================================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Conda not found. Please install Anaconda/Miniconda.
    pause
    exit /b 1
)

REM Activate the cw_evs_app environment
set "CONDA_ENV=cw_evs_app"

REM Check if the environment exists
conda env list | findstr /C:"cw_evs_app" >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Conda environment 'cw_evs_app' not found.
    echo        Please run setup.bat first.
    pause
    exit /b 1
)

echo Activating conda environment '%CONDA_ENV%'...
call conda activate %CONDA_ENV%

REM Find an available port starting from 8501
set "PORT=8501"
:find_port
netstat -aon | findstr /R ":%PORT% " | findstr "LISTENING" >nul 2>&1
if !errorLevel! equ 0 (
    echo [INFO] Port !PORT! is in use, trying next port...
    set /a PORT+=1
    if !PORT! gtr 8510 (
        echo [ERROR] No available ports found in range 8501-8510.
        pause
        exit /b 1
    )
    goto :find_port
)

echo Starting EVS Navigation System on port !PORT!...
echo.
echo The application will open in your browser at:
echo   http://localhost:!PORT!
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Start the application
python -m streamlit run app.py --server.address localhost --server.port !PORT!

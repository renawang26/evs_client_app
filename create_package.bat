@echo off
REM EVS Navigation System - Create Deployment Package
REM Creates a folder ready to copy to another computer

cd /d "%~dp0"

echo ============================================================
echo   EVS Navigation System - Create Deployment Package
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python not found. Please install Python first.
    pause
    exit /b 1
)

echo This will create a deployment package in 'deploy_package' folder.
echo.

set /p INCLUDE_DATA="Include existing database and data? (y/n): "

if /i "%INCLUDE_DATA%" equ "y" (
    python scripts\deploy.py --target deploy_package --include-data
) else (
    python scripts\deploy.py --target deploy_package
)

echo.
echo ============================================================
echo   Package created: deploy_package
echo ============================================================
echo.
echo To deploy:
echo   1. Copy 'deploy_package' folder to the target computer
echo   2. On the target computer, run:
echo      - install_python.bat (if Python not installed)
echo      - setup.bat
echo      - start.bat
echo.
pause

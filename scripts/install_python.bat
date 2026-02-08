@echo off
REM EVS Navigation System - Python Auto-Installer for Windows
REM Downloads and installs Python 3.10 if not present

setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   EVS Navigation System - Python Installer
echo ============================================================
echo.

REM Check if Python is already installed
python --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
    echo Python is already installed: !PYVER!
    echo.
    set /p CONTINUE="Do you want to continue with setup? (y/n): "
    if /i "!CONTINUE!" equ "y" (
        goto :setup
    )
    exit /b 0
)

echo Python is not installed. Starting installation...
echo.

REM Check for winget
winget --version >nul 2>&1
if %errorLevel% equ 0 (
    echo Installing Python 3.10 using winget...
    echo.
    winget install Python.Python.3.10 --silent --accept-source-agreements --accept-package-agreements

    if %errorLevel% equ 0 (
        echo.
        echo Python installed successfully!
        echo Please close and reopen this window, then run setup.bat
        pause
        exit /b 0
    ) else (
        echo winget installation failed. Trying alternative method...
    )
)

REM Alternative: Download Python installer directly
echo.
echo Downloading Python 3.10 installer...
echo.

set PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
set PYTHON_INSTALLER=%TEMP%\python-installer.exe

REM Use PowerShell to download
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"

if not exist "%PYTHON_INSTALLER%" (
    echo.
    echo [ERROR] Failed to download Python installer.
    echo Please download manually from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo.
echo Starting Python installer...
echo.
echo IMPORTANT: When the installer opens:
echo   1. Check "Add Python to PATH" at the bottom
echo   2. Click "Install Now"
echo.
pause

"%PYTHON_INSTALLER%"

echo.
echo Python installation complete.
echo Please close this window and run setup.bat
echo.
pause
exit /b 0

:setup
echo.
echo Running setup.bat...
call setup.bat
exit /b 0

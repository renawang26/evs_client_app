@echo off
REM EVS Navigation System - Start Script for Windows
cd /d "%~dp0"

echo ============================================================
echo   EVS Navigation System - Starting Application
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo        Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Streamlit not installed.
    echo        Please run setup.bat first.
    pause
    exit /b 1
)

echo Starting EVS Navigation System...
echo.
echo The application will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Start the application
streamlit run app.py --server.address localhost --server.port 8501

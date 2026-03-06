@echo off
setlocal

echo === Face Search App — Setup ===
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check Python version >= 3.10
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%

:: Create virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists, skipping.
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install dependencies
echo Installing dependencies (this may take a few minutes)...
echo Note: dlib installs a pre-built wheel, no C++ compiler needed.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed.
    echo If dlib failed, make sure you are using Python 3.10 or 3.11.
    pause
    exit /b 1
)

:: Create data directory
if not exist "data" mkdir data

echo.
echo === Setup complete! ===
echo Run the app with: run.bat
echo.
pause

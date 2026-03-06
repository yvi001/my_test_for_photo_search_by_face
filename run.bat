@echo off
setlocal

:: Check venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

:: Optional: set data directory (default is .\data)
:: set APP_DATA_DIR=D:\my_photos_data

set FLASK_APP=app.web
set FLASK_ENV=production

echo Starting Face Search App at http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.

python -m flask run --host=127.0.0.1 --port=5000
pause

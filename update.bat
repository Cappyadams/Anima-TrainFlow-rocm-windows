@echo off
set "PYTHON_EXE=python_embeded\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Error: python_embeded not found!
    pause
    exit /b
)

echo Starting update process...
"%PYTHON_EXE%" update_app.py

echo.
echo Done!
pause
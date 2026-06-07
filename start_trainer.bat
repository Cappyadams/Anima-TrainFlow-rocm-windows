@echo off
setlocal
cd /d %~dp0

set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

set "PY_EXE=%~dp0python_embeded\python.exe"

if not exist "%PY_EXE%" (
    echo [ERROR] Portable Python not found at:
    echo "%PY_EXE%"
    pause
    exit
)

echo Starting Anima TrainFlow...
echo.

"%PY_EXE%" app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Script crashed. Check the error message above.
    pause
)
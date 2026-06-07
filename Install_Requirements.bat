@echo off
setlocal
cd /d %~dp0

echo WARNING!: This installs ROCM for GFX110x by default. if you have a different GPU, please edit this file!
pause

if not exist "python_embeded\python.exe" (
    echo [ERROR] python_embeded folder not found!
    pause
    exit
)

set "PY_EXE=%~dp0python_embeded\python.exe"

echo [1/3] Installing PyTorch (ROCM)...
"%PY_EXE%" -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch torchvision

echo [2/3] Installing Dependencies from sd-scripts...
cd /d "%~dp0training\sd-scripts"
"%PY_EXE%" -m pip install -r requirements.txt

echo [3/3] Linking sd-scripts library...
"%PY_EXE%" -m pip install -e .

cd /d %~dp0

echo.
echo Installation Complete!
echo You can now run the trainer via start_trainer.bat
pause
@echo off
setlocal EnableExtensions

REM Place this file in the same folder as spectral_tool_app_v18_fixed.py
cd /d "%~dp0"

if not exist "spectral_tool_app_v18_fixed.py" (
    echo ERROR: spectral_tool_app_v18_fixed.py was not found in:
    echo %cd%
    pause
    exit /b 1
)

if not exist "spectral_tool_app_v18_fixed.spec" (
    echo ERROR: spectral_tool_app_v18_fixed.spec was not found in:
    echo %cd%
    pause
    exit /b 1
)

where py >nul 2>nul
if errorlevel 1 goto no_py

echo Installing or updating build dependencies...
py -m pip install --upgrade pip
if errorlevel 1 goto pip_fail
py -m pip install --upgrade pyinstaller numpy pillow matplotlib tifffile rawpy
if errorlevel 1 goto pip_fail

echo Cleaning old build folders...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building with PyInstaller...
py -m PyInstaller --noconfirm --clean "%~dp0spectral_tool_app_v18_fixed.spec"
if errorlevel 1 goto build_fail

echo.
echo Build complete.
echo Output folder:
echo %cd%\dist\SpectralBandSplitter
pause
exit /b 0

:no_py
echo ERROR: Python launcher py was not found.
echo Install Python 3.10 or newer and make sure the launcher is available.
pause
exit /b 1

:pip_fail
echo.
echo Dependency installation failed.
pause
exit /b 1

:build_fail
echo.
echo Build failed.
pause
exit /b 1

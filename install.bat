@echo off
setlocal
cd /d "%~dp0"
title Forensic DGP - Install
set "SETUP_EXIT=1"
if /i "%~1"=="--no-pause" set "IMPLEMENTATION_NO_PAUSE=1"

echo ===================================================
echo Forensic DGP - Windows installation
echo ===================================================
echo Uses Python 3.12 or 3.13, 64-bit, and CPU PyTorch.
echo.

if exist "venv\Scripts\python.exe" goto check_venv

set "SETUP_PYTHON="
py -3.13 -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "SETUP_PYTHON=py -3.13"
if defined SETUP_PYTHON goto create_venv
py -3.12 -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "SETUP_PYTHON=py -3.12"
if defined SETUP_PYTHON goto create_venv
python -c "import struct,sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "SETUP_PYTHON=python"
if defined SETUP_PYTHON goto create_venv

echo [ERROR] Python 3.12 or 3.13 64-bit was not found.
echo Install it from https://www.python.org/downloads/windows/
echo Enable the Python launcher or Add Python to PATH, then run install.bat again.
goto finish

:create_venv
echo [1/4] Creating the project virtual environment...
%SETUP_PYTHON% -m venv "venv"
if errorlevel 1 goto failed

:check_venv
"venv\Scripts\python.exe" -c "import struct,sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 goto invalid_venv
echo [2/4] Installing CPU PyTorch and application dependencies...
"venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto failed
"venv\Scripts\python.exe" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0"
if errorlevel 1 goto failed
"venv\Scripts\python.exe" -m pip install --disable-pip-version-check --index-url https://pypi.org/simple -r requirements.txt
if errorlevel 1 goto failed
echo [3/4] Downloading or verifying the frozen model files...
"venv\Scripts\python.exe" -m implementation.setup_models
if errorlevel 1 goto failed
echo [4/4] Installation complete. Run start.bat to open the application.
set "SETUP_EXIT=0"
goto finish

:invalid_venv
echo [ERROR] The existing venv must use Python 3.12 or 3.13 64-bit.
echo Move the venv directory aside and run install.bat again.
goto finish

:failed
echo.
echo [ERROR] Installation stopped. Review the error above, then run install.bat again.
echo Existing model files have been preserved.

:finish
if not defined IMPLEMENTATION_NO_PAUSE pause
exit /b %SETUP_EXIT%

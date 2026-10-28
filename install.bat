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

set "SETUP_PYTHON="
py -3.13 -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "SETUP_PYTHON=py -3.13"
if defined SETUP_PYTHON goto install_environment
py -3.12 -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "SETUP_PYTHON=py -3.12"
if defined SETUP_PYTHON goto install_environment
python -c "import struct,sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "SETUP_PYTHON=python"
if defined SETUP_PYTHON goto install_environment

echo [ERROR] Python 3.12 or 3.13 64-bit was not found.
echo Install it from https://www.python.org/downloads/windows/
echo Enable the Python launcher or Add Python to PATH, then run install.bat again.
goto finish

:install_environment
%SETUP_PYTHON% -B -m implementation.environment install
if errorlevel 1 goto failed
echo.
echo Installing separate R1 completion and reviewed covering-proposal assets...
%SETUP_PYTHON% -B -m implementation.completion_install
if errorlevel 1 goto failed
set "SETUP_EXIT=0"
goto finish

:failed
echo.
echo [ERROR] Installation stopped. Review the error above, then run install.bat again.
echo Existing model files have been preserved.
echo Existing project venv directories have been preserved.

:finish
if not defined IMPLEMENTATION_NO_PAUSE pause
exit /b %SETUP_EXIT%

@echo off
setlocal
cd /d "%~dp0"
title Forensic DGP - Optimized Install
set "SETUP_EXIT=1"
if /i "%~1"=="--no-pause" set "IMPLEMENTATION_NO_PAUSE=1"

echo ===================================================
echo Forensic DGP - IMPLEMENTATION optimized installer
echo ===================================================
echo Current entry point: install.bat
echo Uses Python 3.12 or 3.13, 64-bit, CPU PyTorch, and cached uv installs.
echo Dependencies and both model families install in parallel.
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

if not defined LOCALAPPDATA (
  echo [ERROR] LOCALAPPDATA is not set. Cannot install a private Python runtime.
  goto failed
)
if exist "%LOCALAPPDATA%\DGP\python-3.13.9-amd64\python.exe" goto check_private_python
echo No supported Python found. Downloading a private Python runtime for this user...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0implementation\bootstrap_python.ps1"
if errorlevel 1 goto failed
:check_private_python
"%LOCALAPPDATA%\DGP\python-3.13.9-amd64\python.exe" -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Private Python is missing or invalid. Existing files were preserved.
  goto failed
)
set SETUP_PYTHON="%LOCALAPPDATA%\DGP\python-3.13.9-amd64\python.exe"
goto install_environment

:install_environment
%SETUP_PYTHON% -B -m implementation.environment install
if errorlevel 1 goto failed
echo.
echo Installation complete. Run start.bat to open the application.
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

@echo off
setlocal
cd /d "%~dp0"
title Forensic DGP - Implementation
if /i "%~1"=="--no-pause" set "IMPLEMENTATION_NO_PAUSE=1"

set "START_PYTHON="
py -3.13 -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "START_PYTHON=py -3.13"
if defined START_PYTHON goto launch_environment
py -3.12 -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "START_PYTHON=py -3.12"
if defined START_PYTHON goto launch_environment
python -c "import struct,sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 set "START_PYTHON=python"
if defined START_PYTHON goto launch_environment
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\DGP\python-3.13.9-amd64\python.exe" (
  "%LOCALAPPDATA%\DGP\python-3.13.9-amd64\python.exe" -c "import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
  if not errorlevel 1 goto private_python
)
echo [ERROR] A supported Python runtime was not found. Run install.bat first.
if not defined IMPLEMENTATION_NO_PAUSE pause
exit /b 1

:private_python
set START_PYTHON="%LOCALAPPDATA%\DGP\python-3.13.9-amd64\python.exe"

:launch_environment
%START_PYTHON% -B -m implementation.environment launch %*
set "START_EXIT=%ERRORLEVEL%"
if not "%START_EXIT%"=="0" if not defined IMPLEMENTATION_NO_PAUSE pause
exit /b %START_EXIT%

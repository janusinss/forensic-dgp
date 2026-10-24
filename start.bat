@echo off
setlocal
cd /d "%~dp0"
title Forensic DGP - Implementation
if /i "%~1"=="--no-pause" set "IMPLEMENTATION_NO_PAUSE=1"

if not exist "venv\Scripts\python.exe" goto missing_venv
"venv\Scripts\python.exe" -m implementation.launch %*
set "START_EXIT=%ERRORLEVEL%"
if not "%START_EXIT%"=="0" if not defined IMPLEMENTATION_NO_PAUSE pause
exit /b %START_EXIT%

:missing_venv
echo [ERROR] The project virtual environment is missing. Run install.bat first.
if not defined IMPLEMENTATION_NO_PAUSE pause
exit /b 1

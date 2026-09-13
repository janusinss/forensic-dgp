@echo off
setlocal
title Forensic DGP Web UI

cd /d "%~dp0"

echo ===================================================
echo [FORENSIC DGP] Initializing Air-Gapped Web Server
echo ===================================================

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found in venv\
    echo Please run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: 1. Clear any stale process bound to port 8000
python -c "import subprocess, os; [os.system(f'taskkill /F /PID {line.strip().split()[-1]} >nul 2>&1') for line in subprocess.check_output('netstat -ano', shell=True).decode().splitlines() if ':8000' in line and 'LISTENING' in line]"

:: 2. Launch browser in background once server is ready
start "" powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"

:: 3. Launch FastAPI server with explicit host and port
echo Starting server on http://127.0.0.1:8000 ...
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

pause

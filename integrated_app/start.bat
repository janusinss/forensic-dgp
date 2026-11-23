@echo off
setlocal
if /i "%~1"=="--source" goto source
if not defined LOCALAPPDATA goto missing
set "DGP_APP_ROOT=%LOCALAPPDATA%\DGP\app"
if not exist "%DGP_APP_ROOT%\current.json" goto missing
set "DGP_NO_PAUSE="
set "DGP_NO_BROWSER="
set "DGP_PORT="
:parse
if "%~1"=="" goto launch
if /i "%~1"=="--no-pause" (
  set "DGP_NO_PAUSE=1"
  shift
  goto parse
)
if /i "%~1"=="--no-browser" (
  set "DGP_NO_BROWSER=1"
  shift
  goto parse
)
if /i "%~1"=="--port" goto port
echo [ERROR] Unknown option: %~1
exit /b 1
:port
if "%~2"=="" exit /b 1
set "DGP_PORT=%~2"
shift
shift
goto parse
:launch
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$p=Get-Content -LiteralPath (Join-Path $env:DGP_APP_ROOT 'current.json') -Raw | ConvertFrom-Json;if($p.commit -cnotmatch '^[0-9a-f]{40}$'){throw 'Invalid managed revision'};$helper=Join-Path $env:DGP_APP_ROOT ('versions\'+$p.commit+'\implementation\distribution.ps1');$r=Get-Content -LiteralPath (Join-Path $env:DGP_APP_ROOT ('receipts\'+$p.commit+'.json')) -Raw | ConvertFrom-Json;$s=[IO.File]::OpenRead($helper);$h=[Security.Cryptography.SHA256]::Create();try{$hash=([BitConverter]::ToString($h.ComputeHash($s))).Replace('-','').ToLowerInvariant()}finally{$h.Dispose();$s.Dispose()};if($r.commit -cne $p.commit -or $hash -cne $r.files.'implementation/distribution.ps1'){throw 'Installer helper differs from its receipt'};$a=@();if($env:DGP_PORT){if($env:DGP_PORT -notmatch '^\d{1,5}$' -or [int]$env:DGP_PORT -gt 65535){throw 'Invalid port'};$a+=@('--port',$env:DGP_PORT)};if($env:DGP_NO_BROWSER){$a+='--no-browser'};if($env:DGP_NO_PAUSE){$a+='--no-pause'}; & $helper -Action Launch -ManagedRoot $env:DGP_APP_ROOT -LaunchArgs $a"
set "DGP_EXIT=%ERRORLEVEL%"
if not defined DGP_NO_PAUSE if not "%DGP_EXIT%"=="0" pause
exit /b %DGP_EXIT%
:source
call "%~dp0implementation\start_source.bat" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%
:missing
echo [ERROR] No managed installation found. Run install.bat first.
if /i not "%~1"=="--no-pause" pause
exit /b 1

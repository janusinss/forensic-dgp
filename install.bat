@echo off
setlocal
if /i "%~1"=="--source" goto source
set "DGP_BOOTSTRAP_FILE=%~f0"
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText($env:DGP_BOOTSTRAP_FILE);$p=$s.IndexOf([char]10+'#__DGP_PAYLOAD__'+[char]13+[char]10);if($p -lt 0){$p=$s.IndexOf([char]10+'#__DGP_PAYLOAD__'+[char]10);$n=18}else{$n=19};if($p -lt 0){throw 'Installer payload missing'}; & ([scriptblock]::Create($s.Substring($p+$n)))"
set "DGP_EXIT=%ERRORLEVEL%"
if /i not "%~1"=="--no-pause" pause
exit /b %DGP_EXIT%
:source
call "%~dp0implementation\install_source.bat" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%
exit /b 1
#__DGP_PAYLOAD__
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if (!$env:LOCALAPPDATA -or ![IO.Path]::IsPathRooted($env:LOCALAPPDATA)) { throw 'LOCALAPPDATA must be an absolute path.' }
$root = Join-Path $env:LOCALAPPDATA 'DGP\app'
$headers = @{ 'User-Agent'='Forensic-DGP-Installer'; 'Accept'='application/vnd.github+json' }
$revision = Invoke-RestMethod -Uri 'https://api.github.com/repos/janusinss/forensic-dgp/commits/IMPLEMENTATION' -Headers $headers -TimeoutSec 30
$commit = [string]$revision.sha
if ($commit -cnotmatch '^[0-9a-f]{40}$') { throw 'Unable to resolve IMPLEMENTATION to a commit.' }
$tree = Invoke-RestMethod -Uri "https://api.github.com/repos/janusinss/forensic-dgp/git/trees/$($commit)?recursive=1" -Headers $headers -TimeoutSec 30
if ($tree.truncated -or !$tree.tree) { throw 'Unable to read the complete application tree.' }
$entry = @($tree.tree | Where-Object { $_.path -ceq 'implementation/distribution.ps1' -and $_.type -eq 'blob' -and $_.mode -eq '100644' })
if ($entry.Count -ne 1 -or $entry[0].sha -cnotmatch '^[0-9a-f]{40}$' -or $entry[0].size -gt 100000) { throw 'Invalid installer helper metadata.' }
$request = [Net.HttpWebRequest]::Create("https://raw.githubusercontent.com/janusinss/forensic-dgp/$commit/implementation/distribution.ps1")
$request.UserAgent = 'Forensic-DGP-Installer'; $request.Timeout = 30000; $request.ReadWriteTimeout = 30000; $request.AllowAutoRedirect = $false
$response = $request.GetResponse()
try {
  if ([int]$response.StatusCode -ne 200 -or $response.ContentLength -gt 100000) { throw 'Invalid installer helper response.' }
  $input = $response.GetResponseStream(); $memory = New-Object IO.MemoryStream
  try {
    $buffer = New-Object byte[] 8192
    while (($count = $input.Read($buffer,0,$buffer.Length)) -gt 0) {
      if ($memory.Length + $count -gt 100000) { throw 'Installer helper exceeds its size limit.' }
      $memory.Write($buffer,0,$count)
    }
    $bytes = $memory.ToArray()
  } finally { $memory.Dispose(); $input.Dispose() }
} finally { $response.Dispose() }
if ($bytes.Length -ne $entry[0].size -or $bytes.Length -gt 100000) { throw 'Installer helper size differs.' }
$hash = [Security.Cryptography.SHA1]::Create()
try {
  $prefix = [Text.Encoding]::ASCII.GetBytes("blob $($bytes.Length)`0")
  $hash.TransformBlock($prefix,0,$prefix.Length,$prefix,0) | Out-Null
  $hash.TransformFinalBlock($bytes,0,$bytes.Length) | Out-Null
  $actual = ([BitConverter]::ToString($hash.Hash)).Replace('-','').ToLowerInvariant()
} finally { $hash.Dispose() }
if ($actual -cne $entry[0].sha) { throw 'Installer helper does not match the resolved commit.' }
Write-Host "Resolved IMPLEMENTATION revision $commit"
$script = [Text.Encoding]::UTF8.GetString($bytes)
& ([scriptblock]::Create($script)) -Action Install -ManagedRoot $root -Commit $commit

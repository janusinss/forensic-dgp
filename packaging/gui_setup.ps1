param(
  [Parameter(Mandatory=$true)][string]$Stage,
  [Parameter(Mandatory=$true)][string]$InstallRoot,
  [Parameter(Mandatory=$true)][string]$EventsFile,
  [Parameter(Mandatory=$true)][string]$CancelFile,
  [Parameter(Mandatory=$true)][string]$LogPath,
  [Parameter(Mandatory=$true)][string]$ExpectedManifestHash
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
function Hash-File([string]$path) {
  $h=[Security.Cryptography.SHA256]::Create(); $s=[IO.File]::OpenRead($path)
  try { return ([BitConverter]::ToString($h.ComputeHash($s))).Replace('-','').ToLowerInvariant() }
  finally { $s.Dispose(); $h.Dispose() }
}
try {
  $manifestPath=Join-Path $Stage 'release.json'
  if ($ExpectedManifestHash -cnotmatch '^[0-9a-f]{64}$' -or (Hash-File $manifestPath) -cne $ExpectedManifestHash) { throw 'Embedded release manifest differs.' }
  $release=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($release.schema -ne 1 -or $release.commit -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid embedded release identity.' }
  foreach ($item in $release.files.psobject.Properties) {
    if ($item.Name -notin @('app-source.zip','app-source.json','distribution.ps1','uninstall.ps1','gui_setup.ps1','ForensicDGP.exe','install_location.ps1')) { throw 'Unexpected release input.' }
    if ((Hash-File (Join-Path $Stage $item.Name)) -cne $item.Value) { throw "Embedded release input differs: $($item.Name)" }
  }
  if (@($release.files.psobject.Properties).Count -ne 7) { throw 'Incomplete release inputs.' }
  $env:DGP_SETUP_EVENTS=$EventsFile
  $env:DGP_SETUP_CANCEL_FILE=$CancelFile
  $env:IMPLEMENTATION_NO_PAUSE='1'
  Remove-Item Env:\DGP_BOOTSTRAP_FILE -ErrorAction SilentlyContinue
  $locationBody = & (Join-Path $Stage 'install_location.ps1') -Action Prepare -InstallRoot $InstallRoot
  if ($LASTEXITCODE -ne 0) { throw 'The installation folder could not be prepared.' }
  $location = $locationBody | ConvertFrom-Json
  $env:DGP_INSTALL_ROOT = $location.install_root
  & (Join-Path $Stage 'distribution.ps1') -Action Install -ManagedRoot $location.managed_root -Commit $release.commit -BundlePath (Join-Path $Stage 'app-source.zip') -BundleManifest (Join-Path $Stage 'app-source.json')
  if ($LASTEXITCODE -ne 0) { throw "Application preparation failed. See $LogPath" }
  $verifiedBody = & (Join-Path $Stage 'install_location.ps1') -Action Validate -InstallRoot $location.install_root
  if ($LASTEXITCODE -ne 0) { throw 'The installation folder changed during preparation. Configuration was not published.' }
  $verified = $verifiedBody | ConvertFrom-Json
  if ($verified.install_root -cne $location.install_root -or $verified.managed_root -cne $location.managed_root) { throw 'The installation folder binding changed.' }
  $configuration = @{schema=1;install_root=$location.install_root;managed_root=$location.managed_root} | ConvertTo-Json -Compress
  $configPath = Join-Path $location.install_root 'install-location.json'
  $temporary = $configPath + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
  [IO.File]::WriteAllText($temporary,$configuration,[Text.UTF8Encoding]::new($false))
  if ([IO.File]::Exists($configPath)) { [IO.File]::Replace($temporary,$configPath,($configPath+'.previous.'+[guid]::NewGuid().ToString('N'))) }
  else { [IO.File]::Move($temporary,$configPath) }
  exit 0
} catch {
  $_ | Out-String | Add-Content -LiteralPath $LogPath -Encoding UTF8
  Write-Error $_
  exit 1
}

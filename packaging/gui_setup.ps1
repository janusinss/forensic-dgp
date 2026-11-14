param(
  [Parameter(Mandatory=$true)][string]$Stage,
  [Parameter(Mandatory=$true)][string]$InstallRoot,
  [Parameter(Mandatory=$true)][string]$EventsFile,
  [Parameter(Mandatory=$true)][string]$CancelFile,
  [Parameter(Mandatory=$true)][string]$LogPath,
  [Parameter(Mandatory=$true)][string]$ExpectedManifestHash,
  [string]$PreviousRoot
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
function Hash-File([string]$path) {
  $entry=Get-Item -LiteralPath $path -Force
  if ($entry.PSIsContainer -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'An embedded release file is not an ordinary file.' }
  $h=[Security.Cryptography.SHA256]::Create(); $s=[IO.File]::OpenRead($path)
  try { return ([BitConverter]::ToString($h.ComputeHash($s))).Replace('-','').ToLowerInvariant() }
  finally { $s.Dispose(); $h.Dispose() }
}
function Assert-PathBudget([string]$root,$release) {
  $source=Get-Content -LiteralPath (Join-Path $Stage 'app-source.json') -Raw -Encoding UTF8 | ConvertFrom-Json
  $sourcePrefix=$root+'\runtime\staging\'+('0'*32)+'\tree\'
  foreach ($entry in $source.files.psobject.Properties) {
    $full=$sourcePrefix+$entry.Name.Replace('/','\')
    if ($full.Length -ge 260 -or [IO.Path]::GetDirectoryName($full).Length -ge 248) { throw 'This folder is too long for the bundled application files. Choose a shorter application folder.' }
  }

}
function Verify-Payload([string]$root,[string]$expectedHash) {
  $manifestFile=Join-Path $root 'manifest.json'
  if ((Hash-File $manifestFile) -cne $expectedHash) { throw 'The offline payload manifest differs.' }
  $manifest=Get-Content -LiteralPath $manifestFile -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($manifest.schema -isnot [int] -or $manifest.schema -ne 1 -or $manifest.platform -cne 'win_amd64' -or @($manifest.files).Count -lt 1 -or @($manifest.files).Count -gt 100000) { throw 'Invalid offline payload inventory.' }
  $seen=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
  foreach ($file in $manifest.files) {
    if ($file.path -isnot [string] -or !$file.path -or $file.path.Contains('\') -or $file.path.Contains(':') -or $file.path.StartsWith('/') -or $file.sha256 -isnot [string] -or $file.sha256 -cnotmatch '^[0-9a-f]{64}$' -or ($file.bytes -isnot [int] -and $file.bytes -isnot [long]) -or $file.bytes -lt 0) { throw 'Invalid offline payload entry.' }
    foreach ($part in $file.path.Split('/')) { if (!$part -or $part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $part -match '[\x00-\x1f<>"|?*]') { throw 'Unsafe offline payload entry.' } }
    if (!$seen.Add($file.path)) { throw 'Duplicate offline payload entry.' }
    $path=Join-Path $root $file.path.Replace('/','\')
    $cursor=$path
    while ($cursor -and $cursor.Length -ge $root.Length) {
      $entry=Get-Item -LiteralPath $cursor -Force
      if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Offline payload paths cannot contain links.' }
      if ($cursor -eq $root) { break }
      $cursor=[IO.Path]::GetDirectoryName($cursor)
    }
    $item=Get-Item -LiteralPath $path -Force
    if ($item.PSIsContainer -or $item.Length -ne $file.bytes -or (Hash-File $path) -cne $file.sha256) { throw ('Offline payload file differs: '+$file.path) }
  }
  # Windows can expand an 8.3 stage path during enumeration. Carry the manifest
  # prefix explicitly instead of slicing FullName by a differently spelled root.
  $queue=New-Object 'System.Collections.Generic.Queue[object]'
  $queue.Enqueue([pscustomobject]@{path=$root;relative=''})
  $count=0
  while ($queue.Count) {
    $directory=$queue.Dequeue()
    foreach ($entry in Get-ChildItem -LiteralPath $directory.path -Force) {
      if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Offline payload paths cannot contain links.' }
      $relative=$(if ($directory.relative) { $directory.relative+'/'+$entry.Name } else { $entry.Name })
      if ($entry.PSIsContainer) { $queue.Enqueue([pscustomobject]@{path=$entry.FullName;relative=$relative}); continue }
      if ($relative -ceq 'manifest.json') { continue }
      if (!$seen.Contains($relative)) { throw ('Unexpected offline payload file: '+$relative) }
      $count++
    }
  }
  if ($count -ne $seen.Count) { throw 'Incomplete offline payload inventory.' }
}
try {
  $manifestPath=Join-Path $Stage 'release.json'
  if ($ExpectedManifestHash -cnotmatch '^[0-9a-f]{64}$' -or (Hash-File $manifestPath) -cne $ExpectedManifestHash) { throw 'Embedded release manifest differs.' }
  $release=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($release.schema -ne 1 -or $release.commit -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid embedded release identity.' }
  $required=@('app-source.zip','app-source.json','distribution.ps1','uninstall.ps1','gui_setup.ps1','ForensicDGP.exe','install_location.ps1','install_state.ps1')
  $allowed=$required+@('offline-payload-manifest.json')
  foreach ($item in $release.files.psobject.Properties) {
    if ($item.Name -notin $allowed -or $item.Value -cnotmatch '^[0-9a-f]{64}$') { throw 'Unexpected release input.' }
    $relative=$(if ($item.Name -eq 'offline-payload-manifest.json') {'payload\manifest.json'} else {$item.Name})
    if ((Hash-File (Join-Path $Stage $relative)) -cne $item.Value) { throw "Embedded release input differs: $($item.Name)" }
  }
  foreach ($name in $required) {
    if (!$release.files.psobject.Properties[$name]) { throw 'Incomplete release inputs.' }
  }
  $offline=[bool]($release.psobject.Properties['payload'] -and $release.payload -eq 'offline-bundled')
  $inputCount=$required.Count+[int]$offline
  if (@($release.files.psobject.Properties).Count -ne $inputCount) { throw 'Release input inventory differs.' }
  if ($offline -and !$release.files.psobject.Properties['offline-payload-manifest.json']) { throw 'Offline release inputs are incomplete.' }
  if ($offline) { Verify-Payload (Join-Path $Stage 'payload') $release.files.'offline-payload-manifest.json' }
  $env:DGP_SETUP_EVENTS=$EventsFile
  $env:DGP_SETUP_CANCEL_FILE=$CancelFile
  $env:IMPLEMENTATION_NO_PAUSE='1'
  foreach ($name in @('DGP_BOOTSTRAP_FILE','DGP_SETUP_GENERATION','DGP_OFFLINE_PAYLOAD','DGP_OFFLINE_PAYLOAD_SHA256')) { Remove-Item -LiteralPath ('Env:\'+$name) -ErrorAction SilentlyContinue }
  $required=512MB
  if ($offline) {
    if ($release.payload_expanded_bytes -isnot [long] -and $release.payload_expanded_bytes -isnot [int]) { throw 'Invalid offline disk-space inventory.' }
    $required+=[long]$release.payload_expanded_bytes
    $env:DGP_OFFLINE_PAYLOAD=Join-Path $Stage 'payload'
    $env:DGP_OFFLINE_PAYLOAD_SHA256=$release.files.'offline-payload-manifest.json'
  }
  Assert-PathBudget $InstallRoot $release
  $preflight=& (Join-Path $Stage 'install_location.ps1') -Action Preflight -InstallRoot $InstallRoot -RequiredBytes $required -CheckPrerequisites
  if ($LASTEXITCODE -ne 0) { throw 'Installation prerequisites or destination checks failed. Existing files were preserved.' }
  $location=$preflight|ConvertFrom-Json
  $mode='fresh'
  $statePath=Join-Path $Stage 'install_state.ps1'
  $lifecycle=[bool]$release.files.psobject.Properties['install_state.ps1']
  if ($lifecycle) {
    $stateArgs=@{Action='Inspect';InstallRoot=$location.install_root;ReleaseVersion=$release.version;ReleaseCommit=$release.commit;SourceManifest=(Join-Path $Stage 'app-source.json')}
    if ($PreviousRoot) { $stateArgs.RegisteredRoot=$PreviousRoot }
    $stateBody=& $statePath @stateArgs
    if ($LASTEXITCODE -ne 0) { throw 'The existing installation could not be checked.' }
    $state=$stateBody|ConvertFrom-Json; $mode=$state.mode
    if ($mode -eq 'newer') { throw 'A newer application version is already installed. Open that version or download a newer installer.' }
    if ($state.previous_root) { $PreviousRoot=$state.previous_root }
  }
  if ($mode -eq 'move') {
    $moveArgs=@{Action='StageMove';InstallRoot=$location.install_root;RegisteredRoot=$PreviousRoot;ReleaseVersion=$release.version;ReleaseCommit=$release.commit;SourceManifest=(Join-Path $Stage 'app-source.json')}
    if ($offline) { $moveArgs.PayloadManifest=Join-Path $Stage 'payload\manifest.json' }
    $moved=& $statePath @moveArgs
    if ($LASTEXITCODE -ne 0) { throw 'Saved records could not be staged for relocation. Previous files were preserved.' }
  } else {
    $prepared=& (Join-Path $Stage 'install_location.ps1') -Action Prepare -InstallRoot $location.install_root
    if ($LASTEXITCODE -ne 0) { throw 'The installation folder could not be prepared.' }
  }
  $env:DGP_INSTALL_ROOT=$location.install_root
  $distributionArgs=@{Action='Install';ManagedRoot=$location.managed_root;Commit=$release.commit;BundlePath=(Join-Path $Stage 'app-source.zip');BundleManifest=(Join-Path $Stage 'app-source.json')}
  if ($mode -in @('current','repair')) { $distributionArgs.Repair=$true }
  & (Join-Path $Stage 'distribution.ps1') @distributionArgs
  if ($LASTEXITCODE -ne 0) { throw "Application preparation failed. See $LogPath" }
  $verifiedBody=& (Join-Path $Stage 'install_location.ps1') -Action Validate -InstallRoot $location.install_root
  if ($LASTEXITCODE -ne 0) { throw 'The installation folder changed during preparation. Configuration was not published.' }
  $verified=$verifiedBody|ConvertFrom-Json
  if ($verified.install_root -cne $location.install_root -or $verified.managed_root -cne $location.managed_root) { throw 'The installation folder binding changed.' }
  $configuration=@{schema=1;install_root=$location.install_root;managed_root=$location.managed_root}|ConvertTo-Json -Compress
  $configPath=Join-Path $location.install_root 'install-location.json'
  $temporary=$configPath+'.'+[guid]::NewGuid().ToString('N')+'.tmp'
  [IO.File]::WriteAllText($temporary,$configuration,[Text.UTF8Encoding]::new($false))
  if ([IO.File]::Exists($configPath)) { [IO.File]::Replace($temporary,$configPath,($configPath+'.previous.'+[guid]::NewGuid().ToString('N'))) }
  else { [IO.File]::Move($temporary,$configPath) }
  if ($mode -eq 'move') {
    # The candidate backend is verified before these already-pinned shell files are prepared.
    # Inno still owns their final installation, shortcuts and registration.
    foreach ($name in @('ForensicDGP.exe','release.json')) {
      $target=Join-Path $location.install_root $name
      $entry=Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
      if ($entry -and ($entry.PSIsContainer -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint))) { throw 'A candidate shell file changed into an unexpected entry.' }
      Copy-Item -LiteralPath (Join-Path $Stage $name) -Destination $target -Force
      if ((Hash-File $target) -cne (Hash-File (Join-Path $Stage $name))) { throw 'A candidate shell file changed while copying.' }
    }
    & $statePath -Action CommitMove -InstallRoot $location.install_root -RegisteredRoot $PreviousRoot -ReleaseVersion $release.version -ReleaseCommit $release.commit
    if ($LASTEXITCODE -ne 0) { throw 'The relocation candidate could not be verified. The previous Windows registration remains unchanged.' }
  }
  [ordered]@{schema=1;mode=$mode;install_root=$location.install_root;previous_root=$PreviousRoot}|ConvertTo-Json -Compress|Write-Output
  exit 0
} catch {
  $_|Out-String|Add-Content -LiteralPath $LogPath -Encoding UTF8
  Write-Error $_
  exit 1
}

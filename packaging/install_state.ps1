# Read-only discovery and receipt-bound relocation for the desktop installer.
param(
  [ValidateSet('Inspect','StageMove','CommitMove','CompleteMove','Cleanup')][string]$Action='Inspect',
  [Parameter(Mandatory=$true)][string]$InstallRoot,
  [string]$RegisteredRoot,
  [string]$ReleaseVersion='0.0.0',
  [string]$ReleaseCommit,
  [string]$PayloadManifest,
  [string]$SourceManifest,
  [switch]$LibraryOnly
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2
$locationHelper=Join-Path $PSScriptRoot 'install_location.ps1'
function Same-Root([string]$a,[string]$b) { return [string]::Equals([IO.Path]::GetFullPath($a).TrimEnd('\'),[IO.Path]::GetFullPath($b).TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase) }
function Ordinary([string]$path) {
  $cursor=[IO.Path]::GetFullPath($path)
  $first=$true
  while ($cursor) {
    $item=$null
    try { $item=Get-Item -LiteralPath $cursor -Force -ErrorAction Stop } catch [System.Management.Automation.ItemNotFoundException] {}
    if ($item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or (!$first -and !$item.PSIsContainer))) { throw 'A linked or ambiguous installation path was preserved.' }
    $next=[IO.Path]::GetDirectoryName($cursor)
    if (!$next -or $next -eq $cursor) { break }
    $cursor=$next; $first=$false
  }
}
function Read-Record([string]$path,[long]$limit=8192) {
  Ordinary $path
  $entry=Get-Item -LiteralPath $path -Force
  if ($entry.PSIsContainer -or $entry.Length -gt $limit) { throw 'Invalid bounded installation record.' }
  return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Hash([string]$path) {
  Ordinary $path
  $h=[Security.Cryptography.SHA256]::Create(); $s=[IO.File]::OpenRead($path)
  try { return ([BitConverter]::ToString($h.ComputeHash($s))).Replace('-','').ToLowerInvariant() } finally { $s.Dispose(); $h.Dispose() }
}
function Relative([string]$name) {
  if (!$name -or $name.Contains(':') -or $name.StartsWith('\') -or $name.StartsWith('/')) { throw 'Unsafe relative relocation path.' }
  foreach ($part in $name.Replace('/','\').Split('\')) {
    if (!$part -or $part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $part -match '[\x00-\x1f<>"|?*]' -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Unsafe relative relocation path.' }
  }
  return $name.Replace('/','\')
}
function Below([string]$root,[string]$relative) {
  $name=Relative $relative
  $target=[IO.Path]::GetFullPath((Join-Path $root $name))
  if (!$target.StartsWith($root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Relocation path escaped its owned directory.' }
  Ordinary $target
  return $target
}
function Validate-Root([string]$path) {
  $body=& $locationHelper -Action Validate -InstallRoot $path 2>&1
  if ($LASTEXITCODE -ne 0) { throw ('Installation folder is not safe: '+($body -join ' ')) }
  return [IO.Path]::GetFullPath($path).TrimEnd('\')
}
function Tree-Files([string]$root) {
  $found=New-Object 'System.Collections.Generic.List[object]'
  if (!(Test-Path -LiteralPath $root)) { return ,$found.ToArray() }
  $queue=New-Object 'System.Collections.Generic.Queue[string]'; $queue.Enqueue($root)
  while ($queue.Count) {
    $dir=$queue.Dequeue(); Ordinary $dir
    foreach ($entry in Get-ChildItem -LiteralPath $dir -Force) {
      if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'A linked relocation entry was preserved.' }
      if ($entry.PSIsContainer) { $queue.Enqueue($entry.FullName) } else { $found.Add($entry) }
      if ($found.Count+$queue.Count -gt 150000) { throw 'Relocation inventory exceeded its bound.' }
    }
  }
  return ,$found.ToArray()
}
function Code-Files([string]$root,[string]$commit) {
  if ($commit -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid installed source identity.' }
  $receiptPath=Join-Path $root ('runtime\receipts\'+$commit+'.json')
  $receipt=Read-Record $receiptPath 16MB
  if ($receipt.schema -isnot [int] -or $receipt.schema -ne 1 -or $receipt.commit -cne $commit -or !$receipt.files -or @($receipt.files.psobject.Properties).Count -lt 1) { throw 'Installed source receipt differs.' }
  $codeRoot=Join-Path $root ('runtime\versions\'+$commit)
  $files=New-Object 'System.Collections.Generic.List[object]'
  foreach ($file in $receipt.files.psobject.Properties) {
    $name=Relative $file.Name
    if ($file.Value -isnot [string] -or $file.Value -cnotmatch '^[0-9a-f]{64}$') { throw 'Installed source hash differs.' }
    $path=Below $codeRoot $name
    if (!(Test-Path -LiteralPath $path -PathType Leaf) -or (Hash $path) -cne $file.Value) { throw 'Installed source file differs.' }
    $files.Add([pscustomobject]@{path=('runtime\versions\'+$commit+'\'+$name);sha256=$file.Value})
  }
  if (@(Tree-Files $codeRoot).Count -ne $files.Count) { throw 'Unexpected installed source file.' }
  $files.Add([pscustomobject]@{path=('runtime\receipts\'+$commit+'.json');sha256=(Hash $receiptPath)})
  return ,$files.ToArray()
}
function Registered-Location {
  $key='HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{FEE649E9-195D-4576-8F30-DAE06A5099C8}_is1'
  if (Test-Path -LiteralPath $key) { return (Get-ItemProperty -LiteralPath $key).InstallLocation }
  return ''
}
function Probe-Runtime([string]$root,[string]$commit) {
  $python=Join-Path $root 'runtime\python\python-3.13.9-amd64\python.exe'
  $record=Join-Path $root ('runtime\data\installations\'+$commit+'.json')
  Ordinary $python
  if (!(Test-Path -LiteralPath $python -PathType Leaf) -or !(Test-Path -LiteralPath $record -PathType Leaf)) { throw 'The bound private runtime is incomplete; choose Repair.' }
  $configuration=Read-Record $record
  $codeRoot=Join-Path $root ('runtime\versions\'+$commit)
  if ($configuration.schema_version -notin @(1,2) -or !(Same-Root $configuration.checkout_root $codeRoot)) { throw 'The installed runtime binding differs; choose Repair.' }
  $probeCode="import os,sys; sys.path.insert(0,os.environ['DGP_STATE_CODE_ROOT']); from implementation.environment import verify_installation; sys.exit(verify_installation())"
  $start=New-Object Diagnostics.ProcessStartInfo
  $start.FileName=$python; $start.Arguments='-I -B -c "'+$probeCode+'"'; $start.WorkingDirectory=$codeRoot
  $start.UseShellExecute=$false; $start.CreateNoWindow=$true; $start.RedirectStandardOutput=$true; $start.RedirectStandardError=$true
  $start.EnvironmentVariables['DGP_INSTALL_ROOT']=$root
  $start.EnvironmentVariables['DGP_MANAGED_ROOT']=(Join-Path $root 'runtime')
  $start.EnvironmentVariables['DGP_STATE_CODE_ROOT']=$codeRoot
  foreach ($name in @('DGP_SETUP_GENERATION','DGP_OFFLINE_PAYLOAD','DGP_OFFLINE_PAYLOAD_SHA256','DGP_SETUP_CANCEL_FILE','DGP_SETUP_EVENTS')) { $start.EnvironmentVariables.Remove($name) }
  $process=New-Object Diagnostics.Process
  $process.StartInfo=$start
  try {
    if (!$process.Start()) { throw 'The installed runtime probe could not start; choose Repair.' }
    $stdout=$process.StandardOutput.ReadToEndAsync(); $stderr=$process.StandardError.ReadToEndAsync()
    if (!$process.WaitForExit(90000)) { $process.Kill(); throw 'Installed runtime verification timed out; choose Repair.' }
    if ($process.ExitCode -ne 0) { throw 'The installed packages or models failed verification; choose Repair.' }
  } finally { $process.Dispose() }
}
function Info([string]$path) {
  $result=[ordered]@{root=$path;exists=$false;owned=$false;healthy=$false;launcher_verified=$false;version='';commit='';reason='not installed';known=@()}
  if (!$path -or !(Test-Path -LiteralPath $path -PathType Container)) { return [pscustomobject]$result }
  $result.exists=$true
  try {
    $canonical=Validate-Root $path; $result.root=$canonical
    $owner=Read-Record (Join-Path $canonical 'runtime\owner.json')
    if ($owner.schema_version -isnot [int] -or $owner.schema_version -ne 2 -or $owner.repository -cne 'janusinss/forensic-dgp' -or $owner.layout -cne 'desktop-runtime-v1' -or !(Same-Root $owner.install_root $canonical) -or @($owner.psobject.Properties).Count -ne 4) { throw 'Runtime ownership differs.' }
    $result.owned=$true; $result.reason='incomplete installation'
    $config=Read-Record (Join-Path $canonical 'install-location.json')
    if ($config.schema -isnot [int] -or $config.schema -ne 1 -or !(Same-Root $config.install_root $canonical) -or !(Same-Root $config.managed_root (Join-Path $canonical 'runtime')) -or @($config.psobject.Properties).Count -ne 3) { throw 'Installed location binding differs.' }
    $release=Read-Record (Join-Path $canonical 'release.json') 65536
    if ($release.schema -isnot [int] -or $release.schema -ne 1 -or $release.version -cnotmatch '^\d{1,4}\.\d{1,4}\.\d{1,4}$' -or $release.commit -cnotmatch '^[0-9a-f]{40}$') { throw 'Installed release identity differs.' }
    $result.version=$release.version; $result.commit=$release.commit
    $launcher=Join-Path $canonical 'ForensicDGP.exe'
    if ($release.files.'ForensicDGP.exe' -cnotmatch '^[0-9a-f]{64}$' -or !(Test-Path -LiteralPath $launcher -PathType Leaf) -or (Hash $launcher) -cne $release.files.'ForensicDGP.exe') { throw 'Installed launcher differs.' }
    $result.launcher_verified=$true
    $active=Read-Record (Join-Path $canonical 'runtime\current.json')
    if ($active.schema -isnot [int] -or $active.schema -ne 1 -or $active.commit -cne $release.commit) { throw 'Installed active version differs.' }
    $known=New-Object 'System.Collections.Generic.List[object]'
    foreach ($file in (Code-Files $canonical $release.commit)) { $known.Add($file) }
    foreach ($name in @('ForensicDGP.exe','release.json','runtime\current.json')) {
      $file=Join-Path $canonical $name
      if (Test-Path -LiteralPath $file -PathType Leaf) { $known.Add([pscustomobject]@{path=$name;sha256=(Hash $file)}) }
    }
    foreach ($name in @('uninstall.ps1','install_state.ps1')) {
      $value=$release.files.psobject.Properties[$name]
      $file=Join-Path $canonical $name
      if ($value -and (Test-Path -LiteralPath $file -PathType Leaf) -and (Hash $file) -ceq $value.Value) { $known.Add([pscustomobject]@{path=$name;sha256=$value.Value}) }
    }
    $result.known=$known.ToArray()
    Probe-Runtime $canonical $release.commit
    $result.healthy=$true; $result.reason='verified source, packages and models'
  } catch { $result.reason=$_.Exception.Message }
  return [pscustomobject]$result
}
function Save-Record([string]$path,$value) {
  Ordinary $path
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path)) | Out-Null
  $temporary=$path+'.'+[guid]::NewGuid().ToString('N')+'.tmp'
  [IO.File]::WriteAllText($temporary,($value|ConvertTo-Json -Depth 8 -Compress),[Text.UTF8Encoding]::new($false))
  if (Test-Path -LiteralPath $path) { [IO.File]::Replace($temporary,$path,($path+'.previous.'+[guid]::NewGuid().ToString('N'))) } else { [IO.File]::Move($temporary,$path) }
}
$script:LifecycleMutex=$null
$script:LifecycleMutexHeld=$false
function Check-Idle {
  # Hold the same app mutex for this entire action, including when no app owns it yet.
  $script:LifecycleMutex=New-Object Threading.Mutex($false,'Local\ForensicDGP-Desktop')
  try { $script:LifecycleMutexHeld=$script:LifecycleMutex.WaitOne(0) }
  catch [Threading.AbandonedMutexException] { $script:LifecycleMutexHeld=$true }
  if (!$script:LifecycleMutexHeld) { throw 'Close Forensic DGP before continuing, then retry setup. Running work was preserved.' }
}
function Verify-SourceRecords([string]$source,$receipt) {
  $expected=@{}
  foreach ($file in @($receipt.copied_files)) {
    if ($file.path -like 'runtime\data\*' -and $file.path -notlike 'runtime\data\models\*') {
      $name=Relative $file.path
      if ($expected.ContainsKey($name) -or $file.sha256 -cnotmatch '^[0-9a-f]{64}$') { throw 'Invalid relocated-record inventory.' }
      $expected[$name]=$file.sha256
    }
  }
  $data=Join-Path $source 'runtime\data'
  $count=0
  foreach ($entry in (Tree-Files $data)) {
    $relative=$entry.FullName.Substring($data.Length+1)
    if ($relative -match '^(?i:installations|desktop|models)(\\|$)') { continue }
    $name='runtime\data\'+(Relative $relative)
    if (!$expected.ContainsKey($name) -or (Hash $entry.FullName) -cne $expected[$name]) {
      throw 'Saved records in the previous installation changed during relocation. Both folders were preserved; retry the move before removing any old files.'
    }
    $count++
  }
  if ($count -ne $expected.Count) { throw 'The previous saved-record inventory changed during relocation. Both folders were preserved.' }
}
function Copy-Verified([string]$source,[string]$destination,[string]$expected) {
  Ordinary $source; Ordinary $destination
  if ((Hash $source) -cne $expected) { throw 'A relocation source changed before copying.' }
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination)) | Out-Null
  Ordinary $destination
  if (Test-Path -LiteralPath $destination) {
    if (!(Test-Path -LiteralPath $destination -PathType Leaf) -or (Hash $destination) -cne $expected) { throw 'A destination record already exists with different content. Both copies were preserved.' }
  } else {
    $temporary=$destination+'.dgp-copy-'+[guid]::NewGuid().ToString('N')
    $input=[IO.File]::OpenRead($source); $output=$null
    try {
      $output=[IO.File]::Open($temporary,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
      $input.CopyTo($output); $output.Dispose(); $output=$null
      if ((Hash $temporary) -cne $expected -or (Hash $source) -cne $expected) { throw 'A relocation file changed during copying.' }
      [IO.File]::Move($temporary,$destination)
    } finally {
      if ($output) { $output.Dispose() }; $input.Dispose()
      if (Test-Path -LiteralPath $temporary -PathType Leaf) { Remove-Item -LiteralPath $temporary -Force }
    }
  }
  if ((Hash $source) -cne $expected -or (Hash $destination) -cne $expected) { throw 'A relocation file changed during copying.' }
}
function Invoke-Lifecycle {
try {
  if (!$RegisteredRoot) { $RegisteredRoot=Registered-Location }
  if ($ReleaseVersion -cnotmatch '^\d{1,4}\.\d{1,4}\.\d{1,4}$') { throw 'Invalid candidate version.' }
  $target=Validate-Root $InstallRoot
  $old=Info $RegisteredRoot; $selected=Info $target
  if ($old.owned -and !(Same-Root $old.root $target) -and
      ($target.StartsWith($old.root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase) -or
       $old.root.StartsWith($target.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase))) {
    throw 'The old and new installation folders cannot contain each other. Choose a separate dedicated folder.'
  }
  $pending=Join-Path $target 'runtime\relocation.pending.json'
  $boundMove=$false
  if (Test-Path -LiteralPath $pending) {
    $binding=Read-Record $pending 32MB
    $boundMove=($binding.schema -eq 1 -and $old.owned -and (Same-Root $binding.source $old.root) -and (Same-Root $binding.destination $target) -and $binding.commit -ceq $ReleaseCommit -and $binding.status -in @('staged','verified','completed','cleaned'))
  }
  $stale=[bool]($RegisteredRoot -and !$old.owned)
  $mode='fresh'; $current=$selected
  if ($old.owned -and !(Same-Root $old.root $target) -and $old.version -and [version]$old.version -gt [version]$ReleaseVersion) { throw 'A newer version is installed. This older installer cannot update or relocate it.' }
  if ($SourceManifest -and $old.owned -and $old.commit -and $old.commit -cne $ReleaseCommit) {
    $sourceIdentity=Read-Record $SourceManifest 16MB
    if ($sourceIdentity.schema -isnot [int] -or $sourceIdentity.schema -ne 1 -or $sourceIdentity.commit -cne $ReleaseCommit -or $old.commit -cnotin @($sourceIdentity.ancestors)) { throw 'The candidate is older than or unrelated to the installed source. Existing files were preserved.' }
  }
  if ($old.owned -and !(Same-Root $old.root $target)) {
    if ($selected.owned -and !$boundMove) { throw 'The destination already contains a different installation. Choose its own repair flow or an empty folder.' }
    $mode='move'; $current=$old
  } elseif ($selected.owned) {
    if (!$selected.healthy) { $mode='repair' }
    elseif ([version]$selected.version -gt [version]$ReleaseVersion) { $mode='newer' }
    elseif ($selected.version -eq $ReleaseVersion -and (!$ReleaseCommit -or $selected.commit -ceq $ReleaseCommit)) { $mode='current' }
    else { $mode='update' }
  }
  if ($Action -eq 'Inspect') {
    [ordered]@{schema=1;mode=$mode;healthy=$current.healthy;launcher_verified=$(if($current.launcher_verified){1}else{0});installed_version=$current.version;previous_root=$(if ($old.owned) {$old.root} else {''});registration_stale=$stale;reason=$current.reason} | ConvertTo-Json -Compress | Write-Output
    exit 0
  }
  Check-Idle
  if ($ReleaseCommit -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid candidate commit.' }
  $pending=Join-Path $target 'runtime\relocation.pending.json'
  if ($Action -eq 'StageMove') {
    if (!$SourceManifest) { throw 'Relocation requires the verified bundled source history.' }
    if (!$old.owned -or (Same-Root $old.root $target)) { throw 'A different, owned previous installation is required for relocation.' }
    # A retry has a prepared target owner but no published shell. Its bound receipt is the only adoption path.
    if (Test-Path -LiteralPath $pending) {
      $prior=Read-Record $pending 32MB
      if ($prior.schema -ne 1 -or !(Same-Root $prior.source $old.root) -or !(Same-Root $prior.destination $target) -or $prior.commit -cne $ReleaseCommit -or $prior.status -ne 'staged') { throw 'Existing relocation attempt differs. Files were preserved.' }
    }
    [long]$recordBytes=0
    $sourceData=Join-Path $old.root 'runtime\data'
    foreach ($entry in (Tree-Files $sourceData)) {
      $relative=$entry.FullName.Substring($sourceData.Length+1)
      if ($relative -notmatch '^(?i:installations|desktop|models)(\\|$)') { $recordBytes+=$entry.Length }
    }
    $destinationDrive=New-Object IO.DriveInfo([IO.Path]::GetPathRoot($target))
    if ($destinationDrive.AvailableFreeSpace -lt ($recordBytes+67108864)) { throw 'Not enough free disk space for the saved records being relocated. Previous files were preserved.' }
    $prepared=& $locationHelper -Action Prepare -InstallRoot $target 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'The relocation destination could not be prepared.' }
    $runtime=Join-Path $target 'runtime'; [IO.Directory]::CreateDirectory($runtime)|Out-Null; Ordinary $runtime
    $ownerPath=Join-Path $runtime 'owner.json'
    $newOwner=@{schema_version=2;repository='janusinss/forensic-dgp';layout='desktop-runtime-v1';install_root=$target}
    if (Test-Path -LiteralPath $ownerPath) {
      $existing=Read-Record $ownerPath
      if ($existing.schema_version -ne 2 -or $existing.repository -cne $newOwner.repository -or $existing.layout -cne $newOwner.layout -or !(Same-Root $existing.install_root $target)) { throw 'Relocation target ownership differs.' }
    } else { Save-Record $ownerPath $newOwner }
    if (!(Test-Path -LiteralPath $pending)) {
      Save-Record $pending ([ordered]@{schema=1;source=$old.root;destination=$target;old_root=$old.root;new_root=$target;commit=$ReleaseCommit;version=$ReleaseVersion;status='staged';copied_files=@();cleanup_files=$old.known})
    }
    $copies=New-Object 'System.Collections.Generic.List[object]'
    $data=Join-Path $old.root 'runtime\data'
    foreach ($entry in (Tree-Files $data)) {
      $relative=$entry.FullName.Substring($data.Length+1)
      if ($relative -match '^(?i:installations|desktop|models)(\\|$)') { continue }
      $name='runtime\data\'+(Relative $relative)
      $expected=Hash $entry.FullName
      Copy-Verified $entry.FullName (Below $target $name) $expected
      $copies.Add([pscustomobject]@{path=$name;sha256=$expected})
    }
    if ($PayloadManifest) {
      $payload=Read-Record $PayloadManifest 16MB
      foreach ($model in $payload.models) {
        $group=$(if ($model.group -eq 'restoration') {'implementation'} elseif ($model.group -eq 'completion') {'completion'} else { throw 'Invalid model group.' })
        $filename=Relative $model.filename
        if ($filename.Contains('\')) { throw 'A model filename cannot contain directories.' }
        $payloadFile=@($payload.files|Where-Object {$_.path -ceq $model.path})
        if ($payloadFile.Count -ne 1) { throw 'A relocation model is absent from the pinned payload.' }
        $name='runtime\data\models\'+$group+'\'+$filename
        $source=Below $old.root $name
        if ((Test-Path -LiteralPath $source -PathType Leaf) -and (Hash $source) -ceq $payloadFile[0].sha256) {
          Copy-Verified $source (Below $target $name) $payloadFile[0].sha256
          $copies.Add([pscustomobject]@{path=$name;sha256=$payloadFile[0].sha256})
        }
      }
    }
    $receipt=[ordered]@{schema=1;source=$old.root;destination=$target;old_root=$old.root;new_root=$target;commit=$ReleaseCommit;version=$ReleaseVersion;status='staged';copied_files=$copies.ToArray();cleanup_files=$old.known}
    Save-Record $pending $receipt
    [ordered]@{schema=1;mode='move-staged';copied_files=$copies.Count;receipt=$pending} | ConvertTo-Json -Compress | Write-Output
    exit 0
  }
  $receipt=Read-Record $pending 32MB
  if ($receipt.schema -isnot [int] -or $receipt.schema -ne 1 -or !(Same-Root $receipt.destination $target) -or !$old.owned -or !(Same-Root $receipt.source $old.root) -or $receipt.commit -cne $ReleaseCommit) { throw 'Relocation receipt binding differs.' }
  $new=Info $target
  if (!$new.healthy -or $new.commit -cne $ReleaseCommit -or $new.version -cne $ReleaseVersion) { throw 'The replacement installation has not passed verification. Previous files were retained.' }
  foreach ($file in $receipt.copied_files) {
    $destination=Below $target $file.path
    if (!(Test-Path -LiteralPath $destination -PathType Leaf) -or (Hash $destination) -cne $file.sha256) { throw 'A copied record changed before relocation verification.' }
  }
  if ($Action -in @('CommitMove','CompleteMove')) { Verify-SourceRecords $old.root $receipt }
  if ($Action -eq 'CommitMove') {
    if ($receipt.status -notin @('staged','verified')) { throw 'Invalid relocation state.' }
    $receipt.status='verified'
    Save-Record $pending $receipt
    [ordered]@{schema=1;mode='move-verified';receipt=$pending} | ConvertTo-Json -Compress | Write-Output
    exit 0
  }
  $registeredNow=Registered-Location
  if (!$registeredNow -or !(Same-Root $registeredNow $target)) { throw 'Windows registration has not switched to the verified replacement. Previous files were retained.' }
  if ($Action -eq 'CompleteMove') {
    if ($receipt.status -notin @('verified','completed')) { throw 'Relocation must be verified before completion.' }
    $receipt.status='completed'
    Save-Record $pending $receipt
    $history=Join-Path $target ('runtime\data\relocations\'+$ReleaseCommit+'.json')
    Save-Record $history $receipt
    [ordered]@{schema=1;mode='move-completed';receipt=$pending} | ConvertTo-Json -Compress | Write-Output
    exit 0
  }
  if ($receipt.status -ne 'completed') { throw 'Relocation must finish Windows integration before old files can be removed.' }
  $removed=0; $retained=0
  foreach ($file in @($receipt.cleanup_files)+@($receipt.copied_files)) {
    $source=Below $old.root $file.path
    if (!(Test-Path -LiteralPath $source)) { continue }
    if (!(Test-Path -LiteralPath $source -PathType Leaf) -or (Hash $source) -cne $file.sha256) { $retained++; continue }
    Remove-Item -LiteralPath $source -Force -ErrorAction Stop; $removed++
  }
  # Never recursively remove the previous root: unknown files and unproven legacy runtimes remain.
  $receipt.status='cleaned'; Save-Record $pending $receipt
  [ordered]@{schema=1;mode='move-cleaned';removed_files=$removed;retained_files=$retained;previous_root=$old.root} | ConvertTo-Json -Compress | Write-Output
  exit 0
} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }
finally {
  if ($script:LifecycleMutexHeld -and $script:LifecycleMutex) { $script:LifecycleMutex.ReleaseMutex(); $script:LifecycleMutexHeld=$false }
  if ($script:LifecycleMutex) { $script:LifecycleMutex.Dispose(); $script:LifecycleMutex=$null }
}
}
if (!$LibraryOnly) { Invoke-Lifecycle }

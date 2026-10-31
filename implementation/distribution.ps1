param(
  [ValidateSet('Install','Launch')][string]$Action,
  [Parameter(Mandatory=$true)][string]$ManagedRoot,
  [string]$Commit,
  [string]$BundlePath,
  [string]$BundleManifest,
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$LaunchArgs
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repo = 'janusinss/forensic-dgp'
$root = [IO.Path]::GetFullPath($ManagedRoot).TrimEnd('\')
$expected = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'DGP\app')).TrimEnd('\')
if (-not [string]::Equals($root,$expected,[StringComparison]::OrdinalIgnoreCase)) { throw 'Managed root must be the expected user-local DGP app directory.' }
function Check-Path([string]$path) {
  $probe = [IO.Path]::GetFullPath($path)
  while ($probe) {
    $item = Get-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
    if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Managed paths cannot contain links or reparse points.' }
    $next = [IO.Path]::GetDirectoryName($probe)
    if (!$next -or $next -eq $probe) { break }
    $probe = $next
  }
}
function Setup-Event([string]$stage,[string]$status,[string]$message,[long]$current=-1,[long]$total=-1) {
  if (!$env:DGP_SETUP_EVENTS) { return }
  if (![IO.Path]::IsPathRooted($env:DGP_SETUP_EVENTS)) { throw 'Setup event path must be absolute.' }
  Check-Path $env:DGP_SETUP_EVENTS
  $event = [ordered]@{schema=1;stage=$stage;status=$status;message=$message}
  if ($current -ge 0) { $event.current=$current }
  if ($total -gt 0) { $event.total=$total }
  $json = $event | ConvertTo-Json -Compress
  Write-Host ('DGP_EVENT ' + $json)
  $line = $json + "`n"
  $bytes = [Text.UTF8Encoding]::new($false).GetBytes($line)
  $stream = [IO.File]::Open($env:DGP_SETUP_EVENTS,[IO.FileMode]::Append,[IO.FileAccess]::Write,[IO.FileShare]::ReadWrite)
  try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
}
function Check-Cancel {
  if (!$env:DGP_SETUP_CANCEL_FILE) { return }
  if (![IO.Path]::IsPathRooted($env:DGP_SETUP_CANCEL_FILE)) { throw 'Setup cancel path must be absolute.' }
  Check-Path $env:DGP_SETUP_CANCEL_FILE
  if (Test-Path -LiteralPath $env:DGP_SETUP_CANCEL_FILE) { throw 'Installation cancelled at a safe boundary; previous active version retained.' }
}
function Load-Bundle([string]$path,[string]$manifestPath,[string]$commit) {
  if (!$path -or !$manifestPath -or ![IO.Path]::IsPathRooted($path) -or ![IO.Path]::IsPathRooted($manifestPath)) { throw 'Both bundle paths must be absolute.' }
  Check-Path $path; Check-Path $manifestPath
  if (!(Test-Path -LiteralPath $path -PathType Leaf) -or !(Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'Bundled payload is missing.' }
  if ((Get-Item -LiteralPath $manifestPath).Length -gt 4MB) { throw 'Bundle manifest is oversized.' }
  $bundle = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  if ($bundle.schema -isnot [int] -or $bundle.schema -ne 1 -or $bundle.commit -cne $commit) { throw 'Bundle identity differs.' }
  if ($bundle.source_zip_sha256 -cnotmatch '^[0-9a-f]{64}$' -or $bundle.source_zip_size -le 0 -or $bundle.source_zip_size -gt 100MB) { throw 'Bundle archive declaration differs.' }
  if ((Get-Item -LiteralPath $path).Length -ne $bundle.source_zip_size -or (File-Sha $path) -cne $bundle.source_zip_sha256) { throw 'Bundled source failed its size/SHA-256 check.' }
  $entries = @($bundle.files.PSObject.Properties)
  if ($entries.Count -lt 1 -or $entries.Count -gt 5000) { throw 'Invalid bundle file inventory.' }
  $mapped = @{}
  foreach ($entry in $entries) {
    if (!(Valid-Relative $entry.Name) -or $entry.Value -cnotmatch '^[0-9a-f]{40}$' -or $mapped.ContainsKey($entry.Name)) { throw 'Invalid bundle file declaration.' }
    $mapped[$entry.Name] = $entry.Value
  }
  if (@($bundle.ancestors).Count -lt 1 -or @($bundle.ancestors).Count -gt 50000) { throw 'Invalid release ancestry inventory.' }
  foreach ($ancestor in $bundle.ancestors) { if ($ancestor -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid release ancestor.' } }
  if ($commit -cnotin @($bundle.ancestors)) { throw 'Bundle ancestry omits its own revision.' }
  return @{files=$mapped;ancestors=@($bundle.ancestors)}
}
function Valid-Relative([string]$path) {
  if (!$path -or $path.Contains('\') -or $path.StartsWith('/') -or $path.Contains(':')) { return $false }
  foreach ($part in $path.Split('/')) {
    if (!$part -or $part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $part -match '[\x00-\x1f<>"|?*]' -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { return $false }
  }
  return $true
}
function Git-BlobBytes([byte[]]$bytes) {
  $header = [Text.Encoding]::ASCII.GetBytes("blob $($bytes.Length)`0")
  $sha = [Security.Cryptography.SHA1]::Create()
  try { $sha.TransformBlock($header,0,$header.Length,$header,0) | Out-Null; $sha.TransformFinalBlock($bytes,0,$bytes.Length) | Out-Null; return ([BitConverter]::ToString($sha.Hash)).Replace('-','').ToLowerInvariant() } finally { $sha.Dispose() }
}
function Git-Blob([string]$path) { return Git-BlobBytes ([IO.File]::ReadAllBytes($path)) }
function Verify-SourceBlob([string]$path,[string]$name,[string]$expected) {
  $raw = [IO.File]::ReadAllBytes($path)
  if ((Git-BlobBytes $raw) -ceq $expected) { return }
  # GitHub archives can apply .gitattributes eol=crlf to tracked batch files.
  # Only accept the exact canonical Git blob after reversing CRLF to LF.
  if (!$name.EndsWith('.bat',[StringComparison]::OrdinalIgnoreCase)) { throw "Git blob mismatch: $name" }
  $canonical = New-Object IO.MemoryStream
  try {
    for ($i=0; $i -lt $raw.Length; $i++) {
      if ($raw[$i] -eq 13 -and $i+1 -lt $raw.Length -and $raw[$i+1] -eq 10) { continue }
      $canonical.WriteByte($raw[$i])
    }
    $bytes = $canonical.ToArray()
  } finally { $canonical.Dispose() }
  if ((Git-BlobBytes $bytes) -cne $expected) { throw "Git blob mismatch: $name" }
  [IO.File]::WriteAllBytes($path,$bytes)
}
function File-Sha([string]$path) {
  $stream = [IO.File]::OpenRead($path); $sha = [Security.Cryptography.SHA256]::Create()
  try { return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-','').ToLowerInvariant() } finally { $sha.Dispose(); $stream.Dispose() }
}
function Ordinary-Files([string]$directory) {
  Check-Path $directory
  foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Application files cannot include links or reparse points.' }
    if ($item.PSIsContainer) { Ordinary-Files $item.FullName } else { $item }
  }
}
function Load-Owner {
  $marker = Join-Path $root 'owner.json'
  if (!(Test-Path -LiteralPath $marker -PathType Leaf)) { throw 'Managed installation marker is missing.' }
  Check-Path $marker
  $m = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
  if ($m.schema_version -isnot [int] -or $m.schema_version -ne 1 -or $m.repository -cne $repo -or @($m.PSObject.Properties).Count -ne 2) { throw 'Managed installation marker differs.' }
}
function Verify-Version([string]$sha) {
  if ($sha -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid application revision.' }
  $dir = Join-Path $root ('versions\' + $sha)
  $receipt = Join-Path $root ('receipts\' + $sha + '.json')
  Check-Path $dir; Check-Path $receipt
  if (!(Test-Path -LiteralPath $receipt -PathType Leaf)) { throw 'Application receipt is missing.' }
  $r = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
  if ($r.schema -ne 1 -or $r.commit -cne $sha) { throw 'Application receipt differs.' }
  $expectedFiles = @{}
  foreach ($p in $r.files.PSObject.Properties) { $expectedFiles[$p.Name] = $p.Value }
  $files = @(Ordinary-Files $dir)
  if ($files.Count -ne $expectedFiles.Count) { throw 'Application files changed. Existing files were preserved.' }
  foreach ($f in $files) {
    Check-Path $f.FullName
    $relative = $f.FullName.Substring($dir.Length+1).Replace('\','/')
    if (!$expectedFiles.ContainsKey($relative) -or (File-Sha $f.FullName) -cne $expectedFiles[$relative]) { throw "Application file changed: $relative. Existing files were preserved." }
  }
  return $dir
}
function Api([string]$url) { return Invoke-RestMethod -Uri $url -Headers @{ 'User-Agent'='Forensic-DGP-Installer'; 'Accept'='application/vnd.github+json' } -TimeoutSec 30 }
function Test-Archive($archive, [hashtable]$manifest, [string]$commit) {
  $seen = @{}
  if ($archive.Entries.Count -gt 5000) { throw 'Source archive contains too many members.' }
  $total = [long]0
  foreach ($entry in $archive.Entries) {
    $total += $entry.Length
    if ($total -gt 250MB) { throw 'Expanded source archive exceeds the limit.' }
    $mode = ($entry.ExternalAttributes -shr 16) -band 0xF000
    if ($mode -eq 0xA000 -or (($entry.ExternalAttributes -band 0x400) -ne 0)) { throw 'Archive links and reparse entries are not allowed.' }
    $parts = $entry.FullName.Split('/',2)
    if ($parts[0] -cne ('forensic-dgp-' + $commit)) { throw 'Archive root does not match the resolved revision.' }
    if ($parts.Count -lt 2 -or !$parts[1]) { continue }
    if ($entry.FullName.EndsWith('/')) { if (!(Valid-Relative $parts[1].TrimEnd('/'))) { throw 'Unsafe source archive directory.' }; continue }
    $name = $parts[1]
    if (!(Valid-Relative $name)) { throw 'Unsafe source archive path.' }
    if (!$manifest.ContainsKey($name) -or $seen.ContainsKey($name) -or $entry.Length -gt 30MB) { throw 'Unexpected or oversized archive member.' }
    $seen[$name] = $true
  }
  if ($seen.Count -ne $manifest.Count) { throw 'Source archive omits tracked files.' }
}
function Download-Limited([string]$url,[string]$path,[long]$limit) {
  $uri = [uri]$url; $response = $null
  for ($redirects = 0; $redirects -le 5; $redirects++) {
    if ($uri.Scheme -cne 'https' -or $uri.UserInfo) { throw 'Only HTTPS download redirects are permitted.' }
    $request = [Net.HttpWebRequest]::Create($uri); $request.UserAgent = 'Forensic-DGP-Installer'; $request.Timeout = 30000; $request.ReadWriteTimeout = 30000; $request.AllowAutoRedirect = $false
    $response = $request.GetResponse()
    if ([int]$response.StatusCode -notin @(301,302,303,307,308)) { break }
    $location = $response.Headers['Location']; $response.Dispose(); $response = $null
    if (!$location) { throw 'Archive redirect has no destination.' }
    $uri = New-Object uri($uri,$location)
  }
  if (!$response -or [int]$response.StatusCode -ne 200) { throw 'Archive download did not return a successful response.' }
  try {
    if ($response.ResponseUri.Scheme -cne 'https' -or ($response.ContentLength -gt $limit)) { throw 'Invalid or oversized archive response.' }
    $input = $response.GetResponseStream(); $output = [IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {
      $buffer = New-Object byte[] 1048576; $total = [long]0; $timer = [Diagnostics.Stopwatch]::StartNew()
      while (($count = $input.Read($buffer,0,$buffer.Length)) -gt 0) {
        $total += $count
        if ($total -gt $limit -or $timer.Elapsed.TotalSeconds -gt 180) { throw 'Archive download exceeded its size or time limit.' }
        Check-Cancel
        $output.Write($buffer,0,$count)
        Setup-Event 'source' 'progress' 'Downloading pinned application source' $total $response.ContentLength
      }
      $output.Flush()
    } finally { $output.Dispose(); $input.Dispose() }
  } finally { $response.Dispose() }
}
Check-Path $root
if (!(Test-Path -LiteralPath $root)) { New-Item -ItemType Directory -Path $root -Force | Out-Null }
Check-Path $root
$digest = [Security.Cryptography.SHA256]::Create()
try { $mutexId = ([BitConverter]::ToString($digest.ComputeHash([Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant())))).Replace('-','').Substring(0,24) } finally { $digest.Dispose() }
$mutex = New-Object Threading.Mutex($false, ('Local\DGP-app-' + $mutexId))
$held = $false
try {
  try { $held = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $held = $true }
  if (!$held) { throw 'The app or installer is already running. Close it before continuing.' }
  $marker = Join-Path $root 'owner.json'
  if ($Action -eq 'Install' -and !(Test-Path -LiteralPath $marker)) {
    if (@(Get-ChildItem -LiteralPath $root -Force).Count -ne 0) { throw 'Existing unrecognized app directory was preserved.' }
    $markerFile = [IO.File]::Open($marker,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try { $content = [Text.Encoding]::UTF8.GetBytes('{"schema_version":1,"repository":"janusinss/forensic-dgp"}'); $markerFile.Write($content,0,$content.Length) } finally { $markerFile.Dispose() }
  }
  Load-Owner
  $env:DGP_MANAGED_ROOT = $root
  if ($Action -eq 'Launch') {
    $pointer = Join-Path $root 'current.json'; Check-Path $pointer
    $active = Get-Content -LiteralPath $pointer -Raw | ConvertFrom-Json
    if ($active.schema -isnot [int] -or $active.schema -ne 1) { throw 'Invalid application pointer.' }
    $dir = Verify-Version $active.commit
    Write-Host "Managed application revision: $($active.commit)"
    if ($LaunchArgs -contains '--no-pause') { $env:IMPLEMENTATION_NO_PAUSE = '1' }
    & (Join-Path $dir 'start.bat') --source @LaunchArgs
    exit $LASTEXITCODE
  }
  if ($Commit -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid resolved revision.' }
  Check-Cancel
  Setup-Event 'source' 'running' 'Verifying pinned application source'
  if ($BundlePath -or $BundleManifest) {
    $bundle = Load-Bundle $BundlePath $BundleManifest $Commit
    $manifest = $bundle.files
    $existingPointer = Join-Path $root 'current.json'
    if (Test-Path -LiteralPath $existingPointer) {
      Check-Path $existingPointer
      $prior = Get-Content -LiteralPath $existingPointer -Raw | ConvertFrom-Json
      if ($prior.schema -isnot [int] -or $prior.schema -ne 1 -or $prior.commit -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid previous active version pointer.' }
      if ($prior.commit -cne $Commit -and $prior.commit -cnotin $bundle.ancestors) { throw 'This setup is older than or unrelated to the installed version. Existing installation was preserved.' }
    }
  } else {
  $tree = Api "https://api.github.com/repos/$repo/git/trees/$($Commit)?recursive=1"
  if ($tree.truncated -or !$tree.tree) { throw 'GitHub tree is missing or truncated.' }
  $manifest = @{}
  foreach ($entry in $tree.tree) {
    if ($entry.type -eq 'tree') { continue }
    if ($entry.type -ne 'blob' -or $entry.mode -notin @('100644','100755') -or !(Valid-Relative $entry.path) -or $entry.sha -cnotmatch '^[0-9a-f]{40}$') { throw 'Unsupported or unsafe Git tree entry.' }
    if ($manifest.ContainsKey($entry.path)) { throw 'Duplicate Git tree entry.' }
    $manifest[$entry.path] = $entry.sha
  }
  }
  $versionDir = Join-Path $root ('versions\' + $Commit)
  if (Test-Path -LiteralPath $versionDir) { $versionDir = Verify-Version $Commit }
  else {
    $stage = Join-Path $root ('staging\' + [guid]::NewGuid().ToString('N'))
    Check-Path $stage
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Check-Path $stage
    $zip = Join-Path $stage 'source.zip'
    if ($BundlePath) {
      Copy-Item -LiteralPath $BundlePath -Destination $zip
      if ((File-Sha $zip) -cne (File-Sha $BundlePath)) { throw 'Bundled source changed while copying.' }
    } else {
    Download-Limited "https://github.com/$repo/archive/$Commit.zip" $zip 100MB
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($zip)
    try {
      Test-Archive $archive $manifest $Commit
      foreach ($entry in $archive.Entries) {
        $parts = $entry.FullName.Split('/',2)
        if ($parts.Count -lt 2 -or !$parts[1] -or $entry.FullName.EndsWith('/')) { continue }
        $name = $parts[1]
        $output = Join-Path $stage ('tree\' + $name.Replace('/','\'))
        $parent = Split-Path -Parent $output
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        $input = $entry.Open(); $file = [IO.File]::Open($output,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $input.CopyTo($file) } finally { $file.Dispose(); $input.Dispose() }
        Verify-SourceBlob $output $name $manifest[$name]
      }
    } finally { $archive.Dispose() }
    Check-Path (Join-Path $root 'versions'); Check-Path (Join-Path $root 'receipts')
    New-Item -ItemType Directory -Path (Join-Path $root 'versions'),(Join-Path $root 'receipts') -Force | Out-Null
    Check-Path (Join-Path $root 'versions'); Check-Path (Join-Path $root 'receipts')
    Move-Item -LiteralPath (Join-Path $stage 'tree') -Destination $versionDir
    $files = [ordered]@{}
    foreach ($name in ($manifest.Keys | Sort-Object)) { $files[$name] = File-Sha (Join-Path $versionDir ($name.Replace('/','\'))) }
    $receipt = @{schema=1;commit=$Commit;files=$files} | ConvertTo-Json -Depth 5 -Compress
    [IO.File]::WriteAllText((Join-Path $root ('receipts\' + $Commit + '.json')),$receipt,[Text.Encoding]::UTF8)
  }
  Check-Cancel
  Setup-Event 'environment' 'running' 'Preparing dependencies and verified model assets'
  Write-Host "Preparing application revision $Commit at $versionDir"
  & (Join-Path $versionDir 'implementation\install_source.bat') --no-pause
  if ($LASTEXITCODE -ne 0) { throw 'Setup failed; the previous active version was retained.' }
  Check-Cancel
  Setup-Event 'verification' 'running' 'Verifying installed application'
  $python = & (Join-Path $versionDir 'start.bat') --source --verify-only
  if ($LASTEXITCODE -ne 0) { throw 'Verification failed; the previous active version was retained.' }
  $null = Verify-Version $Commit
  Check-Cancel
  Setup-Event 'activation' 'running' 'Activating verified application'
  $pointer = Join-Path $root 'current.json'
  Check-Path $pointer
  $tmp = Join-Path $root ('current.' + [guid]::NewGuid().ToString('N') + '.tmp')
  [IO.File]::WriteAllText($tmp, (@{schema=1;commit=$Commit} | ConvertTo-Json -Compress), [Text.Encoding]::UTF8)
  if (Test-Path -LiteralPath $pointer) {
    $backup = Join-Path $root ('previous.' + [guid]::NewGuid().ToString('N') + '.json')
    [IO.File]::Replace($tmp,$pointer,$backup)
  } else { [IO.File]::Move($tmp,$pointer) }
  $starter = Join-Path $versionDir 'start.bat'
  foreach ($dest in @((Join-Path $root 'start.bat'),$(if ($env:DGP_BOOTSTRAP_FILE) { Join-Path (Split-Path -Parent $env:DGP_BOOTSTRAP_FILE) 'start.bat' }))) {
    if (!$dest) { continue }
    if (!(Test-Path -LiteralPath $dest)) {
      try { $bytes = [IO.File]::ReadAllBytes($starter); $out = [IO.File]::Open($dest,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None); try { $out.Write($bytes,0,$bytes.Length) } finally { $out.Dispose() } }
      catch { Write-Warning "Could not create a launcher at $dest. Existing files were preserved." }
    } else {
      try { if ((File-Sha $dest) -cne (File-Sha $starter)) { Write-Warning "Existing launcher at $dest was preserved." } }
      catch { Write-Warning "Existing launcher at $dest could not be checked and was preserved." }
    }
  }
  Write-Host "Installed revision $Commit. Verified launcher: $starter"
  Setup-Event 'complete' 'complete' "Installed revision $Commit"
} catch {
  Setup-Event 'error' 'error' $_.Exception.Message
  throw
} finally { if ($held) { $mutex.ReleaseMutex() }; $mutex.Dispose() }

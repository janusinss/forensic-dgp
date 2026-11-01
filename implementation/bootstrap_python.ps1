param([string]$ManagedRoot,[switch]$ReuseExisting,[switch]$VerifyOnly)
# Install a private, pinned CPython distribution without changing PATH or existing Python installations.
# This script is intentionally only called when the private target does not already exist.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$downloadUrl = 'https://www.python.org/ftp/python/3.13.9/python-3.13.9-amd64.zip'
$expectedSha256 = 'b87eb2b5d7a31220b56f3ab19b1817065af3dd57997aff752132bc13e7832915'
$expectedBytes = 33724497
$base = $null
$work = $null
$archive = $null
$replaceExisting = $false

function Assert-OrdinaryDirectory([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not $item.PSIsContainer -or (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "Expected an ordinary directory without a link or reparse point: $Path"
    }
}

function Python-Progress([string]$status,[string]$message,[long]$current=-1,[long]$total=-1) {
    if (!$env:DGP_SETUP_EVENTS) { return }
    $record = [ordered]@{schema=1;stage='python-runtime';status=$status;message=$message}
    if ($current -ge 0) { $record.current = $current }
    if ($total -gt 0) { $record.total = $total }
    Write-Host ('DGP_EVENT ' + ($record | ConvertTo-Json -Compress))
}
function Python-Cancel {
    if ($env:DGP_SETUP_CANCEL_FILE -and (Test-Path -LiteralPath $env:DGP_SETUP_CANCEL_FILE)) { throw 'Python setup cancelled at a safe boundary; previous active version retained.' }
}
function Publish-PythonRuntime([string]$Staged,[string]$Target,[string]$Base,[bool]$ReplaceExisting) {
    $basePath=[IO.Path]::GetFullPath($Base).TrimEnd('\')
    $stagedPath=[IO.Path]::GetFullPath($Staged)
    $targetPath=[IO.Path]::GetFullPath($Target)
    if (![string]::Equals($targetPath,(Join-Path $basePath 'python-3.13.9-amd64'),[StringComparison]::OrdinalIgnoreCase) -or
        !$stagedPath.StartsWith($basePath+'\',[StringComparison]::OrdinalIgnoreCase) -or
        [string]::Equals($stagedPath,$targetPath,[StringComparison]::OrdinalIgnoreCase)) { throw 'Private Python publication escaped its dedicated runtime folder.' }
    $probe=$stagedPath
    while ($probe) {
        Assert-OrdinaryDirectory $probe
        $parent=[IO.Path]::GetDirectoryName($probe)
        if (!$parent -or $parent -eq $probe) { break }
        $probe=$parent
    }
    $backup=$null
    if ([IO.Directory]::Exists($targetPath)) {
        if (!$ReplaceExisting) { throw 'Private Python target already exists and was preserved.' }
        Assert-OrdinaryDirectory $targetPath
        $backup=$targetPath+'.previous-'+[guid]::NewGuid().ToString('N')
        if (!$backup.StartsWith($basePath+'\',[StringComparison]::OrdinalIgnoreCase) -or [IO.Directory]::Exists($backup) -or [IO.File]::Exists($backup)) { throw 'Private Python backup path is not a new owned child.' }
        [IO.Directory]::Move($targetPath,$backup)
    } elseif ([IO.File]::Exists($targetPath)) { throw 'Private Python target is an unexpected file; it was preserved.' }
    try { [IO.Directory]::Move($stagedPath,$targetPath) }
    catch {
        if ($backup -and ![IO.Directory]::Exists($targetPath) -and ![IO.File]::Exists($targetPath)) { [IO.Directory]::Move($backup,$targetPath) }
        throw
    }
    if ($backup) { Write-Host ('[Python] Previous private runtime preserved at: '+$backup) }
}

function Offline-PythonHash([string]$Path) {
    $algorithm=[Security.Cryptography.SHA256]::Create()
    $reader=[IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($reader)).Replace('-','').ToLowerInvariant() }
    finally { $reader.Dispose(); $algorithm.Dispose() }
}
function Offline-PythonArchive {
    if (!$env:DGP_OFFLINE_PAYLOAD -and !$env:DGP_OFFLINE_PAYLOAD_SHA256) { return $null }
    if (!$env:DGP_OFFLINE_PAYLOAD -or !$env:DGP_OFFLINE_PAYLOAD_SHA256 -or
        $env:DGP_OFFLINE_PAYLOAD_SHA256 -cnotmatch '^[0-9a-f]{64}$' -or
        ![IO.Path]::IsPathRooted($env:DGP_OFFLINE_PAYLOAD)) {
        throw 'Offline Python requires its payload root and pinned manifest SHA-256. No online fallback is permitted.'
    }
    $payloadRoot=[IO.Path]::GetFullPath($env:DGP_OFFLINE_PAYLOAD)
    $probe=$payloadRoot
    while ($probe) {
        Assert-OrdinaryDirectory $probe
        $parent=[IO.Path]::GetDirectoryName($probe)
        if (!$parent -or $parent -eq $probe) { break }
        $probe=$parent
    }
    $manifestPath=Join-Path $payloadRoot 'manifest.json'
    $manifestItem=Get-Item -LiteralPath $manifestPath -Force
    if ($manifestItem.PSIsContainer -or ($manifestItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $manifestItem.Length -gt 2MB) { throw 'Offline manifest is not an ordinary bounded file.' }
    $hash=(Offline-PythonHash $manifestPath)
    if ($hash -cne $env:DGP_OFFLINE_PAYLOAD_SHA256) { throw 'Offline Python manifest failed its pinned SHA-256 check.' }
    $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schema -ne 1 -or $manifest.platform -cne 'win_amd64' -or
        $manifest.python.version -cne '3.13.9' -or $manifest.python.path -cne 'python/python-3.13.9-amd64.zip' -or
        $manifest.python.bytes -ne $expectedBytes -or $manifest.python.sha256 -cne $expectedSha256) {
        throw 'Offline Python metadata differs from the fixed private runtime.'
    }
    $folder=Join-Path $payloadRoot 'python'
    Assert-OrdinaryDirectory $folder
    $result=Join-Path $folder 'python-3.13.9-amd64.zip'
    $item=Get-Item -LiteralPath $result -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $item.Length -ne $expectedBytes -or
        (Offline-PythonHash $result) -cne $expectedSha256) {
        throw 'Offline Python archive failed its fixed size/SHA-256 check.'
    }
    return $result
}

function Download-Python([string]$url,[string]$path) {
    Python-Cancel
    $request = [Net.HttpWebRequest]::Create($url)
    $request.Timeout = 30000; $request.ReadWriteTimeout = 30000; $request.AllowAutoRedirect = $false
    $response = $request.GetResponse()
    try {
        if ([int]$response.StatusCode -ne 200 -or $response.ResponseUri.Scheme -cne 'https' -or ($response.ContentLength -ge 0 -and $response.ContentLength -ne $expectedBytes)) { throw 'Unexpected Python download response.' }
        $input = $response.GetResponseStream()
        $output = [IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try {
            $buffer = New-Object byte[] 1048576
            $downloaded = [long]0; $reported = [long]0; $timer = [Diagnostics.Stopwatch]::StartNew()
            Python-Progress 'downloading' 'Downloading private Python runtime' 0 $expectedBytes
            while (($count = $input.Read($buffer,0,$buffer.Length)) -gt 0) {
                Python-Cancel
                $downloaded += $count
                if ($downloaded -gt $expectedBytes -or $timer.Elapsed.TotalSeconds -gt 900) { throw 'Python download exceeded its size or time limit.' }
                $output.Write($buffer,0,$count)
                if (($downloaded - $reported) -ge 1MB -or $downloaded -eq $expectedBytes) {
                    Python-Progress 'downloading' 'Downloading private Python runtime' $downloaded $expectedBytes
                    $reported = $downloaded
                }
            }
            $output.Flush()
        } finally { $output.Dispose(); $input.Dispose() }
    } finally { $response.Dispose() }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'A 64-bit Windows computer is required.' }
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA) -or -not [IO.Path]::IsPathRooted($env:LOCALAPPDATA)) {
        throw 'LOCALAPPDATA must name an absolute user directory.'
    }
    if ($ManagedRoot) {
        . (Join-Path $PSScriptRoot 'install_paths.ps1')
        $runtime = Assert-DGPCustomRuntime $ManagedRoot
        $base = Join-Path $runtime 'python'
    } else {
        if ($env:DGP_INSTALL_ROOT) { throw 'Custom Python requires its explicit managed root.' }
        $local = [IO.Path]::GetFullPath($env:LOCALAPPDATA)
        Assert-OrdinaryDirectory $local
        $base = [IO.Path]::Combine($local, 'DGP')
    }
    if (-not [IO.Directory]::Exists($base)) {
        if ($VerifyOnly) { throw 'Private Python is missing; run setup first.' }
        [void][IO.Directory]::CreateDirectory($base)
    }
    Assert-OrdinaryDirectory $base
    $target = [IO.Path]::Combine($base, 'python-3.13.9-amd64')
    $codeRoot=[IO.Path]::GetDirectoryName($PSScriptRoot)
    $offlineArchive=$null
    if ($env:DGP_OFFLINE_PAYLOAD -or $env:DGP_OFFLINE_PAYLOAD_SHA256) { $offlineArchive=Offline-PythonArchive }
    if ([IO.File]::Exists($target) -or [IO.Directory]::Exists($target)) {
        if (!$ManagedRoot -or (!$ReuseExisting -and !$VerifyOnly)) { throw "The private Python destination already exists and was not changed: $target" }
        Assert-OrdinaryDirectory $target
        $existing = Join-Path $target 'python.exe'
        $info=$null
        try { $info = Get-Item -LiteralPath $existing -Force -ErrorAction Stop }
        catch [System.Management.Automation.ItemNotFoundException] { }
        if ($info -and ($info.PSIsContainer -or ($info.Attributes -band [IO.FileAttributes]::ReparsePoint))) { throw 'Private Python executable is not an ordinary file.' }
        $existingValid=$false
        try {
            if (!$info) { throw 'Private Python executable is missing.' }
            & $existing -I -B -c "import sys,struct,venv,ensurepip;sys.exit(0 if sys.version_info[:3]==(3,13,9) and struct.calcsize('P')==8 else 1)"
            $existingValid=($LASTEXITCODE -eq 0)
            if ($existingValid -and $offlineArchive) {
                & $existing -I -B -c "import sys;sys.path.insert(0,sys.argv[1]);from implementation.offline_payload import verify_private_python;verify_private_python(sys.argv[2])" $codeRoot $target
                $existingValid=($LASTEXITCODE -eq 0)
            }
        } catch { $existingValid=$false }
        if ($existingValid) {
            Python-Progress 'complete' 'Private Python runtime verified without downloads'
            Write-Host '[Python] Existing private runtime verified without downloads.'
            exit 0
        }
        if ($VerifyOnly -or !$env:DGP_OFFLINE_PAYLOAD) { throw 'Existing custom Python failed verification; its files were preserved.' }
        $replaceExisting=$true
        Write-Host '[Python] Existing private runtime differs. Preparing a separate verified replacement; old files will be retained.'
    }
    if ($VerifyOnly) { throw 'Private Python is missing; run setup first.' }

    $work = [IO.Path]::Combine($base, ('python-bootstrap-' + [guid]::NewGuid().ToString('N')))
    [void][IO.Directory]::CreateDirectory($work)
    Assert-OrdinaryDirectory $work
    $zipPath = [IO.Path]::Combine($work, 'python.zip')
    $staged = [IO.Path]::Combine($work, 'python')
    [void][IO.Directory]::CreateDirectory($staged)

    if ($offlineArchive) {
        $zipPath=$offlineArchive
        Python-Progress 'running' 'Preparing bundled private Python runtime'
        Write-Host '[Python] Using verified bundled runtime; no download required.'
    } else {
        Write-Host '[Python] Downloading the official 64-bit Python runtime...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Download-Python $downloadUrl $zipPath
    }
    $download = Get-Item -LiteralPath $zipPath -Force
    if ($download.Length -ne $expectedBytes) { throw 'The Python download has an unexpected size.' }
    $hasher = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($zipPath)
    try { $hash = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $hasher.Dispose() }
    if ($hash -ne $expectedSha256) { throw 'The Python download failed its SHA-256 check.' }
    Python-Cancel
    Python-Progress 'running' 'Verifying and extracting private Python runtime'
    Write-Host '[Python] Download verified. Preparing the private runtime...'

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($zipPath)
    $root = [IO.Path]::GetFullPath($staged).TrimEnd([char]'\') + '\'
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    [long]$total = 0
    if ($archive.Entries.Count -gt 100000) { throw 'The Python archive contains too many entries.' }
    foreach ($entry in $archive.Entries) {
        $name = $entry.FullName.Replace('/', '\')
        $directory = $name.EndsWith('\')
        $trimmed = $name.TrimEnd([char]'\')
        if ([string]::IsNullOrEmpty($trimmed) -or $name.StartsWith('\') -or [IO.Path]::IsPathRooted($name) -or $name.Contains(':')) {
            throw "Unsafe archive entry: $($entry.FullName)"
        }
        foreach ($part in $trimmed.Split([char]'\')) {
            if ([string]::IsNullOrEmpty($part) -or $part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $part -match '[<>"|?*]' -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$') {
                throw "Unsafe archive entry: $($entry.FullName)"
            }
        }
        if (-not $seen.Add($trimmed)) { throw "Duplicate archive entry: $($entry.FullName)" }
        $attributes = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$entry.ExternalAttributes), 0)
        if ((($attributes -shr 16) -band 0xF000) -eq 0xA000) { throw "Archive links are not supported: $($entry.FullName)" }
        # Components above are already validated: avoid legacy .NET path-length
        # normalization before applying the extended local-drive IO spelling.
        $destination = [IO.Path]::Combine($staged, $trimmed)
        if (-not $destination.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw "Archive entry escapes its staging directory: $($entry.FullName)" }
        $destinationIO='\\?\'+$destination
        $total += $entry.Length
        if ($total -gt 2GB) { throw 'The Python archive expands beyond its size limit.' }
        if ($directory) {
            [void][IO.Directory]::CreateDirectory($destinationIO)
        } else {
            [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destinationIO))
            $inputStream = $entry.Open()
            try {
                $outputStream = [IO.File]::Open($destinationIO, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
                try { $inputStream.CopyTo($outputStream) } finally { $outputStream.Dispose() }
            } finally { $inputStream.Dispose() }
        }
    }
    $archive.Dispose()
    $archive = $null

    $python = [IO.Path]::Combine($staged, 'python.exe')
    if (-not [IO.File]::Exists($python)) { throw 'The Python archive did not contain python.exe.' }
    & $python -I -c "import sys, struct, venv, ensurepip; sys.exit(0 if sys.version_info[:3] == (3,13,9) and struct.calcsize('P') == 8 else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'The downloaded Python runtime failed validation.' }
    if ($offlineArchive) {
        Python-Progress 'running' 'Preparing verified private pip and ensurepip'
        & $python -I -B -c "import sys;sys.path.insert(0,sys.argv[1]);from implementation.offline_payload import prepare_private_python;prepare_private_python(sys.argv[2])" $codeRoot $staged
        if ($LASTEXITCODE -ne 0) { throw 'The derived offline pip runtime failed verification. Previous Python was retained.' }
    }
    Assert-OrdinaryDirectory $base
    Python-Cancel
    Publish-PythonRuntime $staged $target $base $replaceExisting
    Python-Progress 'complete' 'Private Python runtime installed and verified'
    Write-Host "[Python] Private runtime installed: $target"
    Write-Host '[Python] Continuing application setup...'
    exit 0
} catch {
    if ($null -ne $archive) { $archive.Dispose() }
    Write-Error "Python setup failed: $($_.Exception.Message)" -ErrorAction Continue
    if ($null -ne $work) { Write-Host "Download and staging files were retained for inspection: $work" }
    exit 1
}

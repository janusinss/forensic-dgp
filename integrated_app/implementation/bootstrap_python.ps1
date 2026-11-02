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

function Assert-OrdinaryDirectory([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not $item.PSIsContainer -or (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "Expected an ordinary directory without a link or reparse point: $Path"
    }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'A 64-bit Windows computer is required.' }
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA) -or -not [IO.Path]::IsPathRooted($env:LOCALAPPDATA)) {
        throw 'LOCALAPPDATA must name an absolute user directory.'
    }
    $local = [IO.Path]::GetFullPath($env:LOCALAPPDATA)
    Assert-OrdinaryDirectory $local
    $base = [IO.Path]::Combine($local, 'DGP')
    if (-not [IO.Directory]::Exists($base)) { [void][IO.Directory]::CreateDirectory($base) }
    Assert-OrdinaryDirectory $base
    $target = [IO.Path]::Combine($base, 'python-3.13.9-amd64')
    if ([IO.File]::Exists($target) -or [IO.Directory]::Exists($target)) {
        throw "The private Python destination already exists and was not changed: $target"
    }

    $work = [IO.Path]::Combine($base, ('python-bootstrap-' + [guid]::NewGuid().ToString('N')))
    [void][IO.Directory]::CreateDirectory($work)
    Assert-OrdinaryDirectory $work
    $zipPath = [IO.Path]::Combine($work, 'python.zip')
    $staged = [IO.Path]::Combine($work, 'python')
    [void][IO.Directory]::CreateDirectory($staged)

    Write-Host '[Python] Downloading the official 64-bit Python runtime...'
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl -OutFile $zipPath -TimeoutSec 900
    $download = Get-Item -LiteralPath $zipPath -Force
    if ($download.Length -ne $expectedBytes) { throw 'The Python download has an unexpected size.' }
    $hasher = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($zipPath)
    try { $hash = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $hasher.Dispose() }
    if ($hash -ne $expectedSha256) { throw 'The Python download failed its SHA-256 check.' }
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
        $destination = [IO.Path]::GetFullPath([IO.Path]::Combine($staged, $trimmed))
        if (-not $destination.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw "Archive entry escapes its staging directory: $($entry.FullName)" }
        $total += $entry.Length
        if ($total -gt 2GB) { throw 'The Python archive expands beyond its size limit.' }
        if ($directory) {
            [void][IO.Directory]::CreateDirectory($destination)
        } else {
            [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))
            $inputStream = $entry.Open()
            try {
                $outputStream = [IO.File]::Open($destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
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
    Assert-OrdinaryDirectory $base
    [IO.Directory]::Move($staged, $target)
    Write-Host "[Python] Private runtime installed: $target"
    Write-Host '[Python] Continuing application setup...'
    exit 0
} catch {
    if ($null -ne $archive) { $archive.Dispose() }
    Write-Error "Python setup failed: $($_.Exception.Message)" -ErrorAction Continue
    if ($null -ne $work) { Write-Host "Download and staging files were retained for inspection: $work" }
    exit 1
}

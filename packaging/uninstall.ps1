# Inno uninstalls its shell separately. Backend versions, environments, models,
# and records may also be used by the batch workflow and are retained here.
param(
    [ValidateSet('Check','Uninstall')][string]$Action = 'Check',
    [Parameter(Mandatory=$true)][string]$ManagedRoot,
    [string]$InstallRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

function Assert-OrdinaryParents([string]$Path) {
    $probe = [IO.Path]::GetFullPath($Path)
    while ($probe) {
        $item = $null
        try { $item = Get-Item -LiteralPath $probe -Force -ErrorAction Stop }
        catch [System.Management.Automation.ItemNotFoundException] { }
        if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Uninstall paths cannot contain links or reparse points. All backend files were retained.'
        }
        $parent = [IO.Path]::GetDirectoryName($probe)
        if (!$parent -or $parent -eq $probe) { break }
        $probe = $parent
    }
}

$mutex = $null
$held = $false
try {
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA) -or
        -not [IO.Path]::IsPathRooted($env:LOCALAPPDATA) -or
        -not [IO.Path]::IsPathRooted($ManagedRoot)) {
        throw 'Uninstall requires absolute LOCALAPPDATA and managed-root paths.'
    }
    $root = [IO.Path]::GetFullPath($ManagedRoot).TrimEnd('\')
    $legacyRoot = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'DGP\app')).TrimEnd('\')
    $legacyShell = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Programs\Forensic DGP')).TrimEnd('\')
    $custom = $false
    if (!$InstallRoot -and $env:DGP_INSTALL_ROOT) { $InstallRoot = $env:DGP_INSTALL_ROOT }
    if ($InstallRoot) {
        if ($InstallRoot -cnotmatch '^[A-Za-z]:\\' -or $InstallRoot.Contains('/') -or $InstallRoot.Substring(2).Contains(':')) { throw 'An absolute local installation folder is required.' }
        foreach ($part in $InstallRoot.Substring(3).TrimEnd('\').Split('\')) {
            if (!$part -or $part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $part -match '[\x00-\x1f<>"|?*]' -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Installation folder is ambiguous.' }
        }
        $shell = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
        if ($shell.Length -gt 120) { throw 'Installation folder exceeds the supported 120-character limit.' }
        $drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($shell))
        if ($drive.DriveType -ne [IO.DriveType]::Fixed -or !$drive.IsReady -or $shell -eq $drive.Name.TrimEnd('\')) { throw 'Installation folder must be a dedicated local fixed-drive folder.' }
        foreach ($base in @($env:SystemRoot,$env:ProgramFiles,${env:ProgramFiles(x86)},$env:ProgramData)) {
            if (!$base) { continue }
            $protected = [IO.Path]::GetFullPath($base).TrimEnd('\')
            if ([string]::Equals($shell,$protected,[StringComparison]::OrdinalIgnoreCase) -or $shell.StartsWith($protected+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Protected installation directory refused.' }
        }
        $personal = @($env:USERPROFILE,$env:LOCALAPPDATA,$env:APPDATA)
        foreach ($folder in @('Desktop','MyDocuments','MyPictures','MyMusic','MyVideos')) { $personal += [Environment]::GetFolderPath($folder) }
        if ($env:USERPROFILE) { $personal += Join-Path $env:USERPROFILE 'Downloads' }
        foreach ($base in $personal) { if ($base -and [string]::Equals($shell,[IO.Path]::GetFullPath($base).TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase)) { throw 'Personal directory root refused.' } }
        Assert-OrdinaryParents $shell
        $shellMarker = Join-Path $shell 'shell-owner.json'
        Assert-OrdinaryParents $shellMarker
        if (!(Test-Path -LiteralPath $shellMarker -PathType Leaf) -or (Get-Item -LiteralPath $shellMarker).Length -gt 8192) { throw 'Desktop shell ownership marker is missing or invalid.' }
        $shellOwner = Get-Content -LiteralPath $shellMarker -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($null -eq $shellOwner -or @($shellOwner.PSObject.Properties).Count -ne 2 -or $shellOwner.schema -isnot [int] -or $shellOwner.schema -ne 1 -or $shellOwner.application -cne 'Forensic DGP') { throw 'Desktop shell ownership marker differs.' }
        $customRoot = Join-Path $shell 'runtime'
        $custom = [string]::Equals($root,$customRoot,[StringComparison]::OrdinalIgnoreCase)
        if (!$custom -and !([string]::Equals($shell,$legacyShell,[StringComparison]::OrdinalIgnoreCase) -and [string]::Equals($root,$legacyRoot,[StringComparison]::OrdinalIgnoreCase))) { throw 'Managed root does not match the selected installation folder.' }
    } elseif (![string]::Equals($root,$legacyRoot,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Uninstall only accepts the expected user-local DGP app root.'
    }
    Assert-OrdinaryParents $root

    # Same lock identity as implementation/distribution.ps1. A caller performing
    # shell removal should acquire this reported mutex itself after Check and
    # retain its handle through uninstall (another-process Check would contend).
    $digest = [Security.Cryptography.SHA256]::Create()
    try {
        $identity = ([BitConverter]::ToString($digest.ComputeHash(
            [Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant())))).Replace('-','').Substring(0,24)
    } finally { $digest.Dispose() }
    $mutexName = 'Local\DGP-app-' + $identity
    $mutex = New-Object Threading.Mutex($false, $mutexName)
    try { $held = $mutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $held = $true }
    if (!$held) {
        throw 'The application or installer is running. Close it before uninstalling the shell.'
    }
    Assert-OrdinaryParents $root
    $exists = Test-Path -LiteralPath $root
    if ($exists) {
        $directory = Get-Item -LiteralPath $root -Force
        if (!$directory.PSIsContainer) { throw 'Managed root is not a directory; backend files were retained.' }
        $markerPath = Join-Path $root 'owner.json'
        Assert-OrdinaryParents $markerPath
        if (!(Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw 'Managed installation marker is missing. All backend files were retained.'
        }
        $markerInfo = Get-Item -LiteralPath $markerPath -Force
        if ($markerInfo.Length -gt 8192) { throw 'Managed installation marker is oversized.' }
        try { $marker = Get-Content -LiteralPath $markerPath -Raw -Encoding UTF8 | ConvertFrom-Json }
        catch { throw 'Managed installation marker could not be read.' }
        $expectedSchema = 1
        $expectedProperties = 2
        if ($custom) { $expectedSchema=2; $expectedProperties=4 }
        if ($null -eq $marker -or $marker -is [Array] -or
            @($marker.PSObject.Properties).Count -ne $expectedProperties -or
            @($marker.PSObject.Properties.Name) -notcontains 'schema_version' -or
            @($marker.PSObject.Properties.Name) -notcontains 'repository' -or
            $marker.schema_version -isnot [int] -or $marker.schema_version -ne $expectedSchema -or
            $marker.repository -cne 'janusinss/forensic-dgp') {
            throw 'Managed installation marker differs. All backend files were retained.'
        }
        if ($custom -and ($marker.layout -cne 'desktop-runtime-v1' -or $marker.install_root -cne $shell)) {
            throw 'Managed installation marker binding differs. All backend files were retained.'
        }
    }
    # There is deliberately no backend deletion action. Shared package caches,
    # model files and Python environments have no exclusive Inno ownership proof.
    # The Inno [Files] entries own only the shell, shortcuts and registration.
    [ordered]@{
        schema = 1
        action = $Action
        policy = 'retain_shared_backend'
        managed_root = $root
        managed_exists = [bool]$exists
        mutex_name = $mutexName
        retained_data = (Join-Path $root 'data')
        message = 'The desktop shell can be removed. Shared backend files and saved evidence are retained.'
    } | ConvertTo-Json -Compress | Write-Output
    exit 0
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    if ($held) { $mutex.ReleaseMutex() }
    if ($mutex) { $mutex.Dispose() }
}
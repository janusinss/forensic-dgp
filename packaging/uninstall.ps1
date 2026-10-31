# Inno uninstalls its shell separately. Backend versions, environments, models,
# and records may also be used by the batch workflow and are retained here.
param(
    [ValidateSet('Check','Uninstall')][string]$Action = 'Check',
    [Parameter(Mandatory=$true)][string]$ManagedRoot
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
    $expected = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'DGP\app')).TrimEnd('\')
    if (-not [string]::Equals($root, $expected, [StringComparison]::OrdinalIgnoreCase)) {
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
        try { $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json }
        catch { throw 'Managed installation marker could not be read.' }
        if ($null -eq $marker -or $marker -is [Array] -or
            @($marker.PSObject.Properties).Count -ne 2 -or
            @($marker.PSObject.Properties.Name) -notcontains 'schema_version' -or
            @($marker.PSObject.Properties.Name) -notcontains 'repository' -or
            $marker.schema_version -isnot [int] -or $marker.schema_version -ne 1 -or
            $marker.repository -cne 'janusinss/forensic-dgp') {
            throw 'Managed installation marker differs. All backend files were retained.'
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
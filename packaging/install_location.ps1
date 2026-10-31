# Validate a dedicated per-user full application folder before any runtime setup.
param(
    [ValidateSet('Validate','Prepare')][string]$Action = 'Validate',
    [Parameter(Mandatory=$true)][string]$InstallRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

function Ordinary-Parents([string]$Path) {
    $probe = [IO.Path]::GetFullPath($Path)
    while ($probe) {
        $item = $null
        try { $item = Get-Item -LiteralPath $probe -Force -ErrorAction Stop }
        catch [System.Management.Automation.ItemNotFoundException] { }
        if ($item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or !$item.PSIsContainer)) {
            throw 'Installation directories must be ordinary folders without links or reparse points.'
        }
        $parent = [IO.Path]::GetDirectoryName($probe)
        if (!$parent -or $parent -eq $probe) { break }
        $probe = $parent
    }
}
function Bounded-Json([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $item.Length -gt 8192) { throw 'Installation ownership metadata is not an ordinary bounded file.' }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Retained-DesktopBinding([string]$Shell) {
    # Old Inno uninstall logs can delete a marker even after an upgrade adds
    # uninsneveruninstall. Recover only from two independently bound retained
    # metadata records; directory contents alone never establish ownership.
    $runtime = Join-Path $Shell 'runtime'
    Ordinary-Parents $runtime
    if (!(Test-Path -LiteralPath $runtime -PathType Container)) { throw 'Retained runtime ownership is missing.' }
    $configurationPath = Join-Path $Shell 'install-location.json'
    $ownerPath = Join-Path $runtime 'owner.json'
    if (!(Test-Path -LiteralPath $configurationPath -PathType Leaf) -or !(Test-Path -LiteralPath $ownerPath -PathType Leaf)) { throw 'Retained install location and runtime ownership records are both required.' }
    $configuration = Bounded-Json $configurationPath
    $owner = Bounded-Json $ownerPath
    if ($configuration -isnot [pscustomobject] -or @($configuration.PSObject.Properties).Count -ne 3 -or
        @($configuration.PSObject.Properties.Name) -notcontains 'schema' -or
        @($configuration.PSObject.Properties.Name) -notcontains 'install_root' -or
        @($configuration.PSObject.Properties.Name) -notcontains 'managed_root' -or
        $configuration.schema -isnot [int] -or $configuration.schema -ne 1 -or
        $configuration.install_root -isnot [string] -or $configuration.install_root -cne $Shell -or
        $configuration.managed_root -isnot [string] -or $configuration.managed_root -cne $runtime) {
        throw 'Retained install location binding differs; files were preserved.'
    }
    if ($owner -isnot [pscustomobject] -or @($owner.PSObject.Properties).Count -ne 4 -or
        @($owner.PSObject.Properties.Name) -notcontains 'schema_version' -or
        @($owner.PSObject.Properties.Name) -notcontains 'repository' -or
        @($owner.PSObject.Properties.Name) -notcontains 'layout' -or
        @($owner.PSObject.Properties.Name) -notcontains 'install_root' -or
        $owner.schema_version -isnot [int] -or $owner.schema_version -ne 2 -or
        $owner.repository -isnot [string] -or $owner.repository -cne 'janusinss/forensic-dgp' -or
        $owner.layout -isnot [string] -or $owner.layout -cne 'desktop-runtime-v1' -or
        $owner.install_root -isnot [string] -or $owner.install_root -cne $Shell) {
        throw 'Retained runtime ownership binding differs; files were preserved.'
    }
}
function Same-Path([string]$Left,[string]$Right) {
    return [string]::Equals([IO.Path]::GetFullPath($Left).TrimEnd('\'),[IO.Path]::GetFullPath($Right).TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase)
}
try {
    if ($InstallRoot -cnotmatch '^[A-Za-z]:\\' -or $InstallRoot.Contains('/') -or $InstallRoot.Substring(2).Contains(':')) { throw 'Choose an absolute local drive folder; network, device and relative paths are unsupported.' }
    foreach ($part in $InstallRoot.Substring(3).TrimEnd('\').Split('\')) {
        if (!$part -or $part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $part -match '[\x00-\x1f<>"|?*]' -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Choose an unambiguous dedicated application folder.' }
    }
    $shell = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
    if ($shell.Length -gt 120) { throw 'Choose an application folder of at most 120 characters so Python packages fit Windows path limits.' }
    $drive = [IO.Path]::GetPathRoot($shell)
    if ([string]::Equals($shell.TrimEnd('\'),$drive.TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase)) { throw 'A drive root cannot be the application folder.' }
    $info = New-Object IO.DriveInfo($drive)
    if ($info.DriveType -ne [IO.DriveType]::Fixed -or !$info.IsReady) { throw 'Choose a ready local fixed drive.' }
    $protected = @($env:SystemRoot,$env:ProgramFiles,${env:ProgramFiles(x86)},$env:ProgramData)
    foreach ($base in $protected) {
        if (!$base) { continue }
        $fixed = [IO.Path]::GetFullPath($base).TrimEnd('\')
        if ((Same-Path $shell $fixed) -or $shell.StartsWith($fixed+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Windows and shared program directories are protected; choose a user-writable application folder.' }
    }
    $bases = @($env:USERPROFILE,$env:LOCALAPPDATA,$env:APPDATA)
    foreach ($folder in @('Desktop','MyDocuments','MyPictures','MyMusic','MyVideos')) {
        $bases += [Environment]::GetFolderPath($folder)
    }
    if ($env:USERPROFILE) { $bases += Join-Path $env:USERPROFILE 'Downloads' }
    foreach ($base in $bases) { if ($base -and (Same-Path $shell $base)) { throw 'Choose a dedicated child folder, not a profile or personal-folder root.' } }
    Ordinary-Parents $shell
    if (Test-Path -LiteralPath $shell) {
        $children = @(Get-ChildItem -LiteralPath $shell -Force)
        if ($children.Count -gt 0) {
            $markerPath = Join-Path $shell 'shell-owner.json'
            if (!(Test-Path -LiteralPath $markerPath -PathType Leaf)) {
                $markerEntry = Get-Item -LiteralPath $markerPath -Force -ErrorAction SilentlyContinue
                if ($markerEntry) { throw 'Installation ownership marker is not an ordinary file.' }
                Retained-DesktopBinding $shell
            } else {
                $owner = Bounded-Json $markerPath
                if ($null -eq $owner -or @($owner.PSObject.Properties).Count -ne 2 -or $owner.schema -isnot [int] -or $owner.schema -ne 1 -or $owner.application -cne 'Forensic DGP') { throw 'Installation ownership marker differs; existing files were preserved.' }
            }
        }
        $configPath = Join-Path $shell 'install-location.json'
        if (Test-Path -LiteralPath $configPath) {
            $config = Bounded-Json $configPath
            if ($null -eq $config -or @($config.PSObject.Properties).Count -ne 3 -or $config.schema -isnot [int] -or $config.schema -ne 1 -or !(Same-Path $config.install_root $shell) -or !(Same-Path $config.managed_root (Join-Path $shell 'runtime'))) { throw 'Existing install location binding differs; files were preserved.' }
        }
    }
    if ($Action -eq 'Prepare') {
        [IO.Directory]::CreateDirectory($shell) | Out-Null
        Ordinary-Parents $shell
        $markerPath = Join-Path $shell 'shell-owner.json'
        if (!(Test-Path -LiteralPath $markerPath)) {
            $stream = [IO.File]::Open($markerPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
            try { $bytes = [Text.UTF8Encoding]::new($false).GetBytes('{"schema":1,"application":"Forensic DGP"}');$stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
        }
    }
    [ordered]@{schema=1;install_root=$shell;managed_root=(Join-Path $shell 'runtime')} | ConvertTo-Json -Compress | Write-Output
    exit 0
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
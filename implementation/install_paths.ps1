# Shared validation for the chosen desktop runtime's private Python bootstrap.
function Assert-DGPCustomRuntime([string]$ManagedRoot) {
    if (!$env:DGP_INSTALL_ROOT -or $env:DGP_INSTALL_ROOT -cnotmatch '^[A-Za-z]:\\' -or $ManagedRoot -cnotmatch '^[A-Za-z]:\\') {
        throw 'Custom Python requires explicit absolute installation and runtime paths.'
    }
    $shell = [IO.Path]::GetFullPath($env:DGP_INSTALL_ROOT).TrimEnd('\')
    $runtime = [IO.Path]::GetFullPath($ManagedRoot).TrimEnd('\')
    if (![string]::Equals($runtime,(Join-Path $shell 'runtime'),[StringComparison]::OrdinalIgnoreCase)) { throw 'Custom Python root must be the chosen runtime child.' }
    $driveRoot = [IO.Path]::GetPathRoot($shell)
    $drive = New-Object IO.DriveInfo($driveRoot)
    if ($drive.DriveType -ne [IO.DriveType]::Fixed -or $shell -eq $driveRoot.TrimEnd('\') -or $shell.Length -gt 120) { throw 'Custom Python requires a dedicated folder on a fixed local drive.' }
    foreach ($part in $shell.Substring($driveRoot.Length).Split('\')) {
        if (!$part -or $part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $part -match '[\x00-\x1f<>"|?*:]' -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Custom Python location contains an unsafe Windows component.' }
    }
    foreach ($name in @('SystemRoot','ProgramFiles','ProgramFiles(x86)','ProgramData')) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) {
            $blocked = [IO.Path]::GetFullPath($value).TrimEnd('\')
            if ([string]::Equals($shell,$blocked,[StringComparison]::OrdinalIgnoreCase) -or $shell.StartsWith($blocked+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Custom Python cannot use system directories.' }
        }
    }
    $bases = @('USERPROFILE','LOCALAPPDATA','APPDATA','TEMP','TMP') | ForEach-Object { [Environment]::GetEnvironmentVariable($_) } | Where-Object { $_ }
    if ($env:USERPROFILE) { $bases += @('Desktop','Documents','Downloads','Pictures','Videos','Music') | ForEach-Object { Join-Path $env:USERPROFILE $_ } }
    foreach ($base in $bases) { if ([string]::Equals($shell,[IO.Path]::GetFullPath($base).TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase)) { throw 'Choose a dedicated child folder for private Python.' } }
    $probe = $runtime
    while ($probe) {
        Assert-OrdinaryDirectory $probe
        $parent = [IO.Path]::GetDirectoryName($probe)
        if (!$parent -or $parent -eq $probe) { break }
        $probe = $parent
    }
    foreach ($name in @('shell-owner.json','runtime\owner.json')) {
        $info = Get-Item -LiteralPath (Join-Path $shell $name) -Force
        if ($info.PSIsContainer -or ($info.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $info.Length -gt 8192) { throw 'Custom Python ownership marker is invalid.' }
    }
    $shellOwner = Get-Content -LiteralPath (Join-Path $shell 'shell-owner.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($shellOwner.schema -isnot [int] -or $shellOwner.schema -ne 1 -or $shellOwner.application -cne 'Forensic DGP' -or @($shellOwner.PSObject.Properties).Count -ne 2) { throw 'Desktop ownership marker differs.' }
    $owner = Get-Content -LiteralPath (Join-Path $runtime 'owner.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($owner.schema_version -isnot [int] -or $owner.schema_version -ne 2 -or $owner.repository -cne 'janusinss/forensic-dgp' -or $owner.layout -cne 'desktop-runtime-v1' -or ![string]::Equals($owner.install_root,$shell,[StringComparison]::OrdinalIgnoreCase) -or @($owner.PSObject.Properties).Count -ne 4) { throw 'Custom runtime ownership marker differs.' }
    $code = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
    if ([IO.Path]::GetFileName($code) -cnotmatch '^[0-9a-f]{40}$' -or ![string]::Equals([IO.Path]::GetDirectoryName($code),(Join-Path $runtime 'versions'),[StringComparison]::OrdinalIgnoreCase)) { throw 'Private Python bootstrap must belong to the selected installed version.' }
    return $runtime
}

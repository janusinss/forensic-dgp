[CmdletBinding()]
param(
    [ValidateSet('Validate', 'Authenticate', 'Check')]
    [string]$Mode = 'Validate',
    [string]$CredentialsJson = '',
    [string]$SharedDriveId = ''
)

# Connection setup only. This script never uploads, moves or deletes research files.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$uploaderExe = Join-Path $projectRoot 'scratch\drive_setup_20261012_v1\rclone-v1.75.2-windows-amd64\rclone.exe'
$uploaderSHA256 = '41eab3e118d264920b62c3327966c5ece0dc4dd715ee617d9e30450d8ead95a4'
$remoteName = 'thesis_archive'
if (-not $env:APPDATA) { throw 'APPDATA is unavailable; do not create credentials in the project.' }
$privateConfig = Join-Path $env:APPDATA 'rclone\forensic-dgp-drive.conf'
if (-not (Test-Path -LiteralPath $uploaderExe -PathType Leaf)) { throw 'Verified portable uploader is missing.' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $uploaderExe).Hash.ToLowerInvariant() -ne $uploaderSHA256) {
    throw 'Uploader bytes changed. Preserve the file and stop.'
}
$versionLines = @(& $uploaderExe version)
if ($LASTEXITCODE -ne 0 -or $versionLines[0] -ne 'rclone v1.75.2') { throw 'Uploader version check failed.' }

if ($Mode -eq 'Validate') {
    [pscustomobject]@{
        Prepared = $true
        Version = $versionLines[0]
        PrivateConfigExists = (Test-Path -LiteralPath $privateConfig -PathType Leaf)
        CloudConnectionChecked = $false
        ResearchFilesMoved = 0
        ResearchFilesDeleted = 0
        CloudFilesUploaded = 0
    } | ConvertTo-Json
    return
}

# Environment overrides could silently replace the selected account or scope.
$overrideNames = @(Get-ChildItem Env: | Where-Object {
    $_.Name -match '^RCLONE_CONFIG_THESIS_ARCHIVE_' -or
    $_.Name -match '^RCLONE_DRIVE_(CLIENT_ID|CLIENT_SECRET|SCOPE|TEAM_DRIVE|ROOT_FOLDER_ID)$'
} | Select-Object -ExpandProperty Name)
if ($overrideNames.Count) { throw ('Conflicting rclone setting names: ' + ($overrideNames -join ', ')) }

if ($Mode -eq 'Authenticate' -and -not (Test-Path -LiteralPath $privateConfig)) {
    if (-not $CredentialsJson -or -not (Test-Path -LiteralPath $CredentialsJson -PathType Leaf)) {
        throw 'Supply the downloaded OAuth Desktop-app JSON using -CredentialsJson. Do not paste its contents into chat.'
    }
    $credentialDocument = Get-Content -LiteralPath $CredentialsJson -Raw | ConvertFrom-Json
    $desktopProperty = $credentialDocument.PSObject.Properties['installed']
    if (-not $desktopProperty) { throw 'Expected an OAuth Desktop-app JSON with an installed section.' }
    $desktopClient = $desktopProperty.Value
    $idProperty = $desktopClient.PSObject.Properties['client_id']
    $secretProperty = $desktopClient.PSObject.Properties['client_secret']
    if (-not $idProperty -or -not $secretProperty) { throw 'Desktop OAuth credentials are incomplete.' }
    $clientIdValue = [string]$idProperty.Value
    $clientSecretValue = [string]$secretProperty.Value
    if ($clientIdValue -notmatch '^[A-Za-z0-9._-]+\.apps\.googleusercontent\.com$' -or
        [string]::IsNullOrWhiteSpace($clientSecretValue) -or $clientSecretValue -match '[\r\n]') {
        throw 'Invalid credential format; no private configuration was written.'
    }
    if ($SharedDriveId -and $SharedDriveId -notmatch '^[A-Za-z0-9_-]{10,128}$') { throw 'Invalid Shared-drive ID format.' }
    $configParent = Split-Path -Parent $privateConfig
    [System.IO.Directory]::CreateDirectory($configParent) | Out-Null
    $configurationText = @(
        '[' + $remoteName + ']'
        'type = drive'
        'scope = drive.file'
        'client_id = ' + $clientIdValue
        'client_secret = ' + $clientSecretValue
        'root_folder_id ='
        'team_drive = ' + $SharedDriveId
    ) -join "`n"
    $configBytes = (New-Object System.Text.UTF8Encoding($false)).GetBytes($configurationText + "`n")
    $configStream = New-Object System.IO.FileStream($privateConfig, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try { $configStream.Write($configBytes, 0, $configBytes.Length) } finally { $configStream.Dispose() }
    $credentialDocument = $null
    $desktopClient = $null
    $clientSecretValue = $null
    $configurationText = $null
    $configBytes = $null
}

if (-not (Test-Path -LiteralPath $privateConfig -PathType Leaf)) { throw 'Complete Authenticate mode first.' }
$existingConfiguration = [System.IO.File]::ReadAllText($privateConfig)
if ($existingConfiguration -notmatch '(?m)^\[thesis_archive\]\s*$' -or
    $existingConfiguration -notmatch '(?m)^scope\s*=\s*drive\.file\s*$') {
    throw 'Private configuration does not match the prepared limited-access profile. Do not overwrite it.'
}
if ($SharedDriveId -and $existingConfiguration -notmatch ('(?m)^team_drive\s*=\s*' + [regex]::Escape($SharedDriveId) + '\s*$')) {
    throw 'Existing destination differs. Preserve it and resolve the target before transfer.'
}
$existingConfiguration = $null

if ($Mode -eq 'Authenticate') {
    Write-Host 'Authorize the account containing your 5 TB storage in the Google browser window.'
    Write-Host 'If asked whether to use a browser, choose Y. This step transfers no research files.'
    & $uploaderExe --config $privateConfig config reconnect ($remoteName + ':')
    if ($LASTEXITCODE -ne 0) { throw 'Google authorization did not complete. The private configuration is preserved for a retry.' }
}

# Report quota only; never display client secrets, refresh tokens or config contents.
$quotaLines = @(& $uploaderExe --config $privateConfig about ($remoteName + ':') --json --timeout 30s --contimeout 10s --retries 1 --low-level-retries 1)
if ($LASTEXITCODE -ne 0) { throw 'Drive connection verification failed. No research files were transferred.' }
$quotaObject = ($quotaLines -join "`n") | ConvertFrom-Json
$freeBytes = $null
$totalBytes = $null
if ($quotaObject.PSObject.Properties['free']) { $freeBytes = $quotaObject.free }
if ($quotaObject.PSObject.Properties['total']) { $totalBytes = $quotaObject.total }
[pscustomobject]@{
    Connected = $true
    Remote = $remoteName
    Scope = 'drive.file'
    TotalBytes = $totalBytes
    FreeBytes = $freeBytes
    PrivateConfigOutsideProject = $true
    ResearchFilesMoved = 0
    ResearchFilesDeleted = 0
    CloudFilesUploaded = 0
} | ConvertTo-Json

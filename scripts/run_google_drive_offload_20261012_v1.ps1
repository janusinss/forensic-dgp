[CmdletBinding()]
param(
    [ValidateSet('Validate', 'Upload', 'Verify')]
    [string]$Mode = 'Validate',
    [ValidateSet('image', 'cache')]
    [string]$Batch = 'image',
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{64}$')]
    [string]$ExpectedPlanSHA
)

# Manual Drive transfer only. No move, sync, deletion, model or VM operations.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$receiptRoot = Join-Path $projectRoot 'outputs\google_drive_offload_20261012_v1'
$planPath = Join-Path $receiptRoot 'plan.json'
function Get-SHA([string]$FilePath) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $FilePath).Hash.ToLowerInvariant()
}
if ((Get-SHA $planPath) -ne $ExpectedPlanSHA) { throw 'Offload plan differs; preserve it and stop.' }
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($plan.version -ne 'exact-cold-backups-v1' -or $plan.root -ne $projectRoot -or
    $plan.destination -ne 'thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261012_v1') {
    throw 'Unexpected project, plan version or Drive destination.'
}
if ((Get-SHA $PSCommandPath) -ne $plan.driver_sha256) { throw 'Transfer helper differs.' }
$activePath = Join-Path $receiptRoot 'protected_active_bindings.json'
if ((Get-SHA $activePath) -ne $plan.protected_manifest_sha256) { throw 'Active dependency manifest differs.' }
$active = Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json
$batchPlan = $plan.batches.PSObject.Properties[$Batch].Value
$filesPath = Join-Path $receiptRoot ($Batch + '_files.txt')
$sumPath = Join-Path $receiptRoot ($Batch + '_sha256.txt')
if ((Get-SHA $filesPath) -ne $batchPlan.file_list_sha256 -or
    (Get-SHA $sumPath) -ne $batchPlan.sha256_list_sha256) { throw 'Frozen transfer list differs.' }
$archiveExe = Join-Path $env:LOCALAPPDATA 'Programs\rclone\rclone.exe'
if ((Get-SHA $archiveExe) -ne '41eab3e118d264920b62c3327966c5ece0dc4dd715ee617d9e30450d8ead95a4') {
    throw 'Expected verified rclone 1.75.2 executable.'
}
$privateProfile = Join-Path $env:APPDATA 'rclone\forensic-dgp-drive-manual.conf'
$privateBody = [IO.File]::ReadAllText($privateProfile)
$section = [regex]::Match($privateBody, '(?ms)^\[thesis_archive\]\s*\r?\n(.*?)(?=^\[|\z)').Groups[1].Value
function Read-Setting([string]$Name) {
    return [regex]::Match($section, '(?m)^' + $Name + '\s*=\s*(.*?)\s*$').Groups[1].Value
}
$cryptoHash = [Security.Cryptography.SHA256]::Create()
try { $currentClientHash = ([BitConverter]::ToString($cryptoHash.ComputeHash([Text.Encoding]::UTF8.GetBytes((Read-Setting 'client_id'))))).Replace('-', '').ToLowerInvariant() }
finally { $cryptoHash.Dispose() }
if ((Read-Setting 'type') -ne 'drive' -or (Read-Setting 'scope') -ne 'drive.file' -or
    (Read-Setting 'team_drive') -or (Read-Setting 'root_folder_id') -or
    $currentClientHash -ne $plan.client_id_sha256 -or -not (Read-Setting 'token')) {
    throw 'Drive application, scope or My Drive destination differs; stop before upload.'
}
$privateBody = $null
$section = $null
$settingOverrides = @(Get-ChildItem Env: | Where-Object {
    $_.Name -match '^RCLONE_CONFIG_THESIS_ARCHIVE_' -or $_.Name -match '^RCLONE_DRIVE_'
} | Select-Object -ExpandProperty Name)
if ($settingOverrides.Count) { throw ('Conflicting setting names: ' + ($settingOverrides -join ', ')) }
Write-Host ('Checking ' + $plan.protected_active_bindings + ' current local bindings before ' + $Mode + '.')
foreach ($binding in $active.PSObject.Properties) {
    $candidate = Join-Path $projectRoot $binding.Name
    if ((Get-SHA $candidate) -ne $binding.Value) { throw ('Active dependency changed: ' + $binding.Name) }
}
if ($Mode -ne 'Verify') {
    Write-Host ('Checking frozen local bytes for the ' + $Batch + ' batch; this starts no upload.')
    foreach ($item in $batchPlan.files) {
        $candidate = [IO.Path]::GetFullPath((Join-Path $projectRoot $item.path))
        if (-not $candidate.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source escaped the project.' }
        $localFile = Get-Item -LiteralPath $candidate
        if ($localFile.PSIsContainer -or ($localFile.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            $localFile.Length -ne $item.bytes -or (Get-SHA $candidate) -ne $item.sha256) {
            throw ('Frozen source differs: ' + $item.path)
        }
    }
}
if ($Mode -eq 'Validate') {
    [pscustomobject]@{ Prepared = $true; Batch = $Batch; Files = $batchPlan.file_count;
        UploadBytes = $batchPlan.upload_bytes; CandidateRemovalBytes = $batchPlan.removal_candidate_bytes;
        ActiveBindingsVerified = $plan.protected_active_bindings; FilesUploaded = 0; FilesRemoved = 0 } | ConvertTo-Json
    return
}
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$operationRoot = Join-Path $receiptRoot ($Mode.ToLowerInvariant() + '_' + $Batch + '_' + $stamp)
New-Item -ItemType Directory -Path $operationRoot | Out-Null
$destination = $plan.destination + '/' + $Batch
$logPath = Join-Path $operationRoot 'rclone.log'
if ($Mode -eq 'Upload') {
    & $archiveExe --config $privateProfile copy $projectRoot $destination --files-from-raw $filesPath --checksum --immutable --transfers 2 --checkers 4 --tpslimit 6 --drive-chunk-size 32M --retries 3 --low-level-retries 5 --timeout 2m --contimeout 20s --max-duration 8h --cutoff-mode hard --progress --log-level INFO --log-file $logPath
} else {
    # Compare downloaded bytes with the frozen SHA256 list, without a second disk copy.
    $matchPath = Join-Path $operationRoot 'combined.txt'
    Write-Host ('Streaming frozen SHA256 verification for ' + $Batch + '; log: ' + $logPath)
    & $archiveExe --config $privateProfile check $sumPath $destination --checkfile SHA256 --download --one-way --combined $matchPath --checkers 2 --tpslimit 6 --retries 1 --low-level-retries 3 --timeout 2m --contimeout 20s --max-duration 8h --cutoff-mode hard --progress --stats 15s --log-level INFO --log-file $logPath
}
$operationExitCode = $LASTEXITCODE
$complete = $operationExitCode -eq 0
if ($complete -and $Mode -eq 'Verify') {
    $comparisonLines = @(Get-Content -LiteralPath $matchPath)
    $expectedNames = @($batchPlan.files | ForEach-Object { '= ' + $_.path } | Sort-Object)
    $actualNames = @($comparisonLines | Sort-Object)
    if ($comparisonLines.Count -ne $batchPlan.file_count -or
        ($comparisonLines | Where-Object { -not $_.StartsWith('= ') }) -or
        @(Compare-Object $expectedNames $actualNames).Count -ne 0) { $complete = $false }
}
$receipt = [pscustomobject]@{
    Complete = $complete; Mode = $Mode; Batch = $Batch; PlanSHA256 = $ExpectedPlanSHA;
    RcloneExitCode = $operationExitCode; Destination = $destination;
    ExpectedFiles = $batchPlan.file_count; ExpectedBytes = $batchPlan.upload_bytes;
    DownloadedSHA256Verification = ($Mode -eq 'Verify' -and $complete);
    FilesRemoved = 0; TrainingCalls = 0; ReceiptDirectory = $operationRoot
}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $operationRoot 'receipt.json') -Encoding UTF8
$receipt | ConvertTo-Json
if (-not $complete) { throw 'Transfer or verification stopped; local originals and failure evidence are preserved.' }

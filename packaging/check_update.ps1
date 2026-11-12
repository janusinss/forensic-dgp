param([string]$LocalVersion, [string]$InstalledVersion)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version 2

function Read-ReleaseManifest {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $request = [Net.HttpWebRequest]::Create('https://forensic-dgp.pages.dev/release.json')
    $request.UserAgent = 'Forensic-DGP-Setup'
    $request.AllowAutoRedirect = $false
    $request.Timeout = 5000
    $request.ReadWriteTimeout = 3000
    $request.CachePolicy = New-Object Net.Cache.RequestCachePolicy ([Net.Cache.RequestCacheLevel]::NoCacheNoStore)
    $response = $request.GetResponse()
    try {
        if ([int]$response.StatusCode -ne 200 -or $response.ContentLength -gt 16384) {
            throw 'Unexpected release response.'
        }
        $stream = $response.GetResponseStream()
        $memory = New-Object IO.MemoryStream
        try {
            $buffer = New-Object byte[] 4096
            $started = [Diagnostics.Stopwatch]::StartNew()
            while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                if ($memory.Length + $read -gt 16384 -or $started.ElapsedMilliseconds -gt 5000) {
                    throw 'Release response exceeded its limits.'
                }
                $memory.Write($buffer, 0, $read)
            }
            return [Text.Encoding]::UTF8.GetString($memory.ToArray())
        } finally {
            $memory.Dispose()
            $stream.Dispose()
        }
    } finally {
        $response.Dispose()
    }
}

function Get-NewerRelease([string]$Json, [string]$Current) {
    $pattern = '\A(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})\z'
    $release = ConvertFrom-Json -InputObject $Json
    if ($Current -cnotmatch $pattern -or $release.schema -isnot [int] -or $release.schema -ne 1 -or
        $release.version -isnot [string] -or $release.version -cnotmatch $pattern) {
        throw 'Invalid release version.'
    }
    $version = $release.version
    $filename = "ForensicDGP_Setup-$version.exe"
    $download = "https://github.com/janusinss/forensic-dgp-downloads/releases/download/v$version/$filename"
    if ($release.filename -cne $filename -or $release.downloadUrl -cne $download -or
        $release.checksumUrl -cne "/downloads/$filename.sha256" -or
        $release.sha256 -cnotmatch '\A[0-9a-f]{64}\z' -or
        ($release.bytes -isnot [long] -and $release.bytes -isnot [int]) -or
        $release.bytes -lt 1 -or $release.bytes -ge 2GB) {
        throw 'Invalid release download metadata.'
    }
    if ([version]$version -gt [version]$Current) { return $version }
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        $newer = Get-NewerRelease (Read-ReleaseManifest) $LocalVersion
        if ($newer -and $InstalledVersion -match '\A[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}\z' -and
            [version]$InstalledVersion -ge [version]$newer) { $newer = $null }
        if ($newer) { "DGP_UPDATE $newer" } else { 'DGP_CURRENT' }
    } catch {
        'DGP_UNAVAILABLE'
    }
}

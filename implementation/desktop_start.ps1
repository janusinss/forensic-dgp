# Receipt-verified desktop startup. Paths are data, not interpolated shell code.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
if (!$env:DGP_DESKTOP_HELPER -or !$env:DGP_DESKTOP_MANAGED_ROOT) {
    throw 'Desktop startup binding is missing.'
}
& $env:DGP_DESKTOP_HELPER -Action Launch -ManagedRoot $env:DGP_DESKTOP_MANAGED_ROOT -LaunchArgs @('--port','0','--no-browser','--no-pause','--desktop')
exit $LASTEXITCODE

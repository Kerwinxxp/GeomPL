[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'Config.psd1'),
    [switch]$IncludeSam3
)

$steps = @(
    '01-Audit-Prerequisites.ps1',
    '02-Sync-Repository.ps1',
    '03-Setup-Environments.ps1',
    '04-Fetch-Models.ps1',
    '05-Warm-LaMa.ps1',
    '06-Smoke-Test.ps1'
)
foreach ($step in $steps) {
    Write-Host "`n=== $step ===" -ForegroundColor Cyan
    $args = @{ Config = $Config }
    if ($IncludeSam3 -and $step -in @('04-Fetch-Models.ps1','06-Smoke-Test.ps1')) { $args.IncludeSam3 = $true }
    & (Join-Path $PSScriptRoot $step) @args
    if (-not $?) { throw "$step failed." }
}

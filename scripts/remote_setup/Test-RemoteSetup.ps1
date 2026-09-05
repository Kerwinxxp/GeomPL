[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Failures = [System.Collections.Generic.List[string]]::new()

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { $Failures.Add($Message) }
}

$required = @(
    '.gitignore',
    'Config.Template.psd1',
    'Pins.psd1',
    '01-Audit-Prerequisites.ps1',
    '02-Sync-Repository.ps1',
    '03-Setup-Environments.ps1',
    '04-Fetch-Models.ps1',
    '05-Warm-LaMa.ps1',
    '06-Smoke-Test.ps1',
    'Strict-GeoRanker-Smoke.py',
    'Strict-LaMa-Smoke.py',
    'Strict-Sam3-Smoke.py',
    'Prefetch-Pinned-Model.py',
    'Test-SyncRepository.ps1',
    'Run-All.ps1',
    'README.md'
)

foreach ($name in $required) {
    Assert-True (Test-Path -LiteralPath (Join-Path $Here $name)) "Missing $name"
}

$scripts = Get-ChildItem -LiteralPath $Here -Filter '*.ps1' -File
foreach ($script in $scripts) {
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $script.FullName, [ref]$tokens, [ref]$errors
    )
    Assert-True ($errors.Count -eq 0) "PowerShell parse errors in $($script.Name): $errors"
}

$productionAssets = Get-ChildItem -LiteralPath $Here -File | Where-Object Name -notin @('Test-RemoteSetup.ps1','Test-SyncRepository.ps1','VALIDATION.md')
$allText = ($productionAssets | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw }) -join "`n"
$forbiddenLocalPath = 'C:\\Users\\' + 'phdwf'
Assert-True (-not ($allText -match $forbiddenLocalPath)) 'A local developer path is hardcoded.'
Assert-True (-not ($allText -match '(sk-proj-|hf_[A-Za-z0-9]{20,}|tvly-)')) 'Credential-like text found.'
Assert-True ($allText -match '2100e9e7c4b95000e16434c7d1fd4e6a4b8424d6') 'Pinned GeoRanker commit missing.'
Assert-True ($allText -match '537BC19E6A13495BD7CE9A3163DA67B274C95E0A25F3EF1C86A3EDDB995D5385') 'LoRA SHA-256 missing.'
Assert-True ($allText -match 'Qwen/Qwen2-VL-7B-Instruct') 'Qwen model ID missing.'
Assert-True ($allText -match 'facebook/sam3') 'SAM3 model ID missing.'
Assert-True ($allText -match 'eed13092ef92e448dd6875b2a00151bd3f7db0ac') 'Pinned Qwen revision missing.'
Assert-True ($allText -match '3c879f39826c281e95690f02c7821c4de09afae7') 'Pinned SAM3 revision missing.'
Assert-True ($allText -match '48910F66A263A4E5146E98F0E14ACF5421454BCA01C9A3580C0E423D69A56565') 'adapter_config SHA-256 missing.'
Assert-True ($allText -match 'symbolic-ref') 'Attached HEAD validation missing.'
Assert-True ($allText -match "'remote','get-url','origin'") 'Origin URL validation missing.'
Assert-True ($allText -match 'pip.+freeze') 'pip freeze lock snapshot missing.'
Assert-True ($allText -match 'struct\.calcsize') '64-bit Python assertion missing.'
Assert-True ($allText -match 'os\.replace') 'Atomic Hugging Face revision update missing.'
Assert-True ($allText -match 'GeoRankerConfigSha256') 'adapter_config checksum enforcement missing.'
Assert-True ($allText -match 'Assert-LockSnapshot') 'Installed environment lock verification missing.'

$strictGeo = Join-Path $Here 'Strict-GeoRanker-Smoke.py'
if (Test-Path $strictGeo) {
    $text = Get-Content -LiteralPath $strictGeo -Raw
    Assert-True ($text -match 'r1\[0\].+r1\[1\]') 'GeoRanker direction assertion missing.'
    Assert-True ($text -match '1e-3') 'GeoRanker determinism threshold missing.'
}
$strictLama = Join-Path $Here 'Strict-LaMa-Smoke.py'
if (Test-Path $strictLama) {
    $text = Get-Content -LiteralPath $strictLama -Raw
    Assert-True ($text -match 'SimpleLama') 'LaMa instantiation missing from strict smoke.'
    Assert-True ($text -match 'cuda') 'LaMa CUDA execution missing from strict smoke.'
}
$strictSam = Join-Path $Here 'Strict-Sam3-Smoke.py'
if (Test-Path $strictSam) {
    $text = Get-Content -LiteralPath $strictSam -Raw
    Assert-True ($text -match 'post_process_instance_segmentation') 'SAM3 segmentation post-processing missing.'
    Assert-True ($text -match 'Japanese banner') 'SAM3 image/text query missing.'
}

$configPath = Join-Path $Here 'Config.Template.psd1'
if (Test-Path -LiteralPath $configPath) {
    $config = Import-PowerShellDataFile -LiteralPath $configPath
    Assert-True ($config.ProjectRoot -eq 'C:\Users\xx0073.UNT\GeoBayes') 'Template project path is wrong.'
    Assert-True ($config.NumShards -eq 1) 'Template NumShards must be 1.'
    Assert-True ($config.ShardIndex -eq 0) 'Remote template ShardIndex must be 0.'
}

$ignorePath = Join-Path $Here '.gitignore'
if (Test-Path -LiteralPath $ignorePath) {
    Assert-True ((Get-Content -LiteralPath $ignorePath) -contains 'Config.psd1') 'Config.psd1 must be locally ignored.'
}

if ($Failures.Count) {
    $Failures | ForEach-Object { Write-Error $_ -ErrorAction Continue }
    throw "$($Failures.Count) remote-setup validation(s) failed."
}

Write-Host "Remote setup validation passed: $($scripts.Count) PowerShell scripts parsed." -ForegroundColor Green

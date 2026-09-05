[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'Config.psd1'),
    [switch]$IncludeSam3
)

. (Join-Path $PSScriptRoot 'Common.ps1')
$cfg = Get-SetupConfig $Config
$pins = Get-ModelPins
Assert-ProjectRoot $cfg.ProjectRoot
$geoPython = Get-PythonPath $cfg.ProjectRoot GeoRanker
$samPython = Get-PythonPath $cfg.ProjectRoot Sam3
if (-not (Test-Path $geoPython)) { throw 'GeoRanker environment is missing. Run 03-Setup-Environments.ps1.' }

$sourceCommit = $pins.GeoRankerSourceCommit
$expectedSha = $pins.GeoRankerAdapterSha256
$checkpointDir = Join-Path $cfg.ProjectRoot 'belief_elicit\georanker_ckpt'
$adapter = Join-Path $checkpointDir 'adapter_model.safetensors'
New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null

$needsAdapter = $true
if (Test-Path -LiteralPath $adapter) {
    $needsAdapter = (Get-FileHash -LiteralPath $adapter -Algorithm SHA256).Hash -ne $expectedSha
}
if ($needsAdapter) {
    $tmp = "$adapter.download"
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    $url = "https://raw.githubusercontent.com/Applied-Machine-Learning-Lab/GeoRanker/$sourceCommit/checkpoints/adapter_model.safetensors"
    Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
    $actual = (Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash
    if ($actual -ne $expectedSha) {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        throw "GeoRanker LoRA SHA-256 mismatch. Expected $expectedSha, received $actual from pinned commit $sourceCommit."
    }
    Move-Item -LiteralPath $tmp -Destination $adapter -Force
}
Set-Content -LiteralPath (Join-Path $checkpointDir 'SOURCE_COMMIT.txt') -Encoding ascii -Value @($sourceCommit,'repo: https://github.com/Applied-Machine-Learning-Lab/GeoRanker')

$adapterConfigUrl = "https://raw.githubusercontent.com/Applied-Machine-Learning-Lab/GeoRanker/$sourceCommit/checkpoints/adapter_config.json"
$adapterConfig = Join-Path $checkpointDir 'adapter_config.json'
$configTmp = "$adapterConfig.download"
Remove-Item -LiteralPath $configTmp -Force -ErrorAction SilentlyContinue
Invoke-WebRequest -Uri $adapterConfigUrl -OutFile $configTmp -UseBasicParsing
$configHash = (Get-FileHash -LiteralPath $configTmp -Algorithm SHA256).Hash
if ($configHash -ne $pins.GeoRankerConfigSha256) {
    Remove-Item -LiteralPath $configTmp -Force -ErrorAction SilentlyContinue
    throw "GeoRanker adapter_config SHA-256 mismatch. Expected $($pins.GeoRankerConfigSha256), received $configHash."
}
Move-Item -LiteralPath $configTmp -Destination $adapterConfig -Force
Write-Host "GeoRanker LoRA verified: $expectedSha"

Invoke-Checked $geoPython @((Join-Path $PSScriptRoot 'Prefetch-Pinned-Model.py'),$pins.QwenModel,$pins.QwenRevision)
Write-Host 'Public Qwen2-VL-7B-Instruct is cached.' -ForegroundColor Green

if ($IncludeSam3) {
    if (-not (Test-Path $samPython)) { throw 'SAM3 environment is missing. Run 03-Setup-Environments.ps1.' }
    try { Invoke-Checked $samPython @((Join-Path $PSScriptRoot 'Prefetch-Pinned-Model.py'),$pins.Sam3Model,$pins.Sam3Revision) }
    catch {
        throw "SAM3 prefetch failed. It is gated: accept its Hugging Face license, run 'hf auth login' on this remote machine, then rerun with -IncludeSam3. Never place a token in Config.psd1."
    }
    Write-Host 'Gated facebook/sam3 is cached.' -ForegroundColor Green
}

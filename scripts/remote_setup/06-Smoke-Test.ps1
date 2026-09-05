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

Push-Location $cfg.ProjectRoot
try {
    $adapter = Join-Path $cfg.ProjectRoot 'belief_elicit\georanker_ckpt\adapter_model.safetensors'
    $hash = (Get-FileHash -LiteralPath $adapter -Algorithm SHA256).Hash
    if ($hash -ne $pins.GeoRankerAdapterSha256) { throw 'GeoRanker LoRA checksum changed.' }
    $configHash = (Get-FileHash -LiteralPath (Join-Path $cfg.ProjectRoot 'belief_elicit\georanker_ckpt\adapter_config.json') -Algorithm SHA256).Hash
    if ($configHash -ne $pins.GeoRankerConfigSha256) { throw 'GeoRanker adapter_config checksum changed.' }

    $environments = Import-PowerShellDataFile (Join-Path $PSScriptRoot 'Environments.psd1')
    foreach ($name in @('GeoRanker','Sam3','LaMa')) {
        $python = Get-PythonPath $cfg.ProjectRoot $name
        $spec = $environments[$name]
        Assert-Python64 $python
        Invoke-Checked $python @('-m','pip','check')
        Assert-PackageVersions $python (Get-ExpectedPackages (Join-Path $PSScriptRoot $spec.Requirements))
        Assert-LockSnapshot $python (Join-Path $PSScriptRoot "locks\$($spec.Freeze)")
        Invoke-Checked $python @('-c','import importlib,sys; [importlib.import_module(x) for x in sys.argv[1].split(",")]',$spec.Imports)
    }
    Invoke-Checked $geoPython @('-c','import torch,transformers,peft,bitsandbytes; assert torch.cuda.is_available(); assert "4090" in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0), torch.__version__, transformers.__version__, peft.__version__, bitsandbytes.__version__)')
    Invoke-Checked $geoPython @((Join-Path $PSScriptRoot 'Strict-GeoRanker-Smoke.py'),$pins.QwenModel,$pins.QwenRevision)

    Invoke-Checked (Get-PythonPath $cfg.ProjectRoot LaMa) @((Join-Path $PSScriptRoot 'Strict-LaMa-Smoke.py'))

    if ($IncludeSam3) {
        Invoke-Checked $samPython @((Join-Path $PSScriptRoot 'Strict-Sam3-Smoke.py'),$pins.Sam3Model,$pins.Sam3Revision)
    }

} finally {
    Pop-Location
}

Write-Host 'Remote smoke tests passed.' -ForegroundColor Green

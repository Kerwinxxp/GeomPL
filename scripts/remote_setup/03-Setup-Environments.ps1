[CmdletBinding()]
param([string]$Config = (Join-Path $PSScriptRoot 'Config.psd1'))
. (Join-Path $PSScriptRoot 'Common.ps1')
$cfg = Get-SetupConfig $Config
Assert-ProjectRoot $cfg.ProjectRoot
# Fail before creating or changing any environment if a required interpreter is absent.
Assert-LauncherPython $cfg.PythonVersion
Assert-LauncherPython '3.11'
$environments = Import-PowerShellDataFile (Join-Path $PSScriptRoot 'Environments.psd1')
$lockDir = Join-Path $PSScriptRoot 'locks'
New-Item -ItemType Directory -Path $lockDir -Force | Out-Null
Push-Location $cfg.ProjectRoot
try {
    foreach ($name in @('GeoRanker','Sam3','LaMa')) {
        $spec = $environments[$name]
        $python = Get-PythonPath $cfg.ProjectRoot $name
        $version = if ($name -eq 'LaMa') { '3.11' } else { $cfg.PythonVersion }
        if (-not (Test-Path -LiteralPath $python)) {
            Invoke-Checked py @("-$version",'-m','venv',(Split-Path -Parent (Split-Path -Parent $python)))
        }
        Assert-Python64 $python
        Invoke-Checked $python @('-c','import sys; assert "%d.%d" % sys.version_info[:2] == sys.argv[1], "Wrong Python version; recreate this remote environment explicitly"',$version)
        $requirements = Join-Path $PSScriptRoot $spec.Requirements
        Invoke-Checked $python @('-m','pip','install','--upgrade','pip')
        Invoke-Checked $python @('-m','pip','install','--only-binary=:all:','--no-binary=fire','--extra-index-url','https://download.pytorch.org/whl/cu128','-r',$requirements)
        Invoke-Checked $python @('-m','pip','check')
        Assert-PackageVersions $python (Get-ExpectedPackages $requirements)
        Invoke-Checked $python @('-c','import importlib,sys; [importlib.import_module(x) for x in sys.argv[1].split(",")]',$spec.Imports)
        $snapshot = @(Invoke-Checked $python @('-m','pip','freeze','--all'))
        if (-not $snapshot.Count) { throw "Empty freeze for $name" }
        $snapshot | Set-Content -LiteralPath (Join-Path $lockDir $spec.Freeze) -Encoding utf8
    }
} finally { Pop-Location }
Write-Host 'GeoRanker, SAM3 and isolated LaMa environments validated.' -ForegroundColor Green

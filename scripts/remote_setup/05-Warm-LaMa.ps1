[CmdletBinding()]
param([string]$Config = (Join-Path $PSScriptRoot 'Config.psd1'))

. (Join-Path $PSScriptRoot 'Common.ps1')
$cfg = Get-SetupConfig $Config
Assert-ProjectRoot $cfg.ProjectRoot
$python = Get-PythonPath $cfg.ProjectRoot LaMa
if (-not (Test-Path $python)) { throw 'LaMa environment is missing. Run 03-Setup-Environments.ps1.' }

Invoke-Checked $python @((Join-Path $PSScriptRoot 'Strict-LaMa-Smoke.py'))

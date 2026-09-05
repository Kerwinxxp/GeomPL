$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
Write-Host "Engine: $($PSVersionTable.PSVersion) / $($PSVersionTable.PSEdition)"
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('geobayes-native-' + [guid]::NewGuid().ToString('N') + '.cmd')
Set-Content -LiteralPath $fixture -Encoding ascii -Value @('@echo off', 'echo expected-stdout', 'echo expected-stderr 1>&2', 'exit /b %1')
$actual = @(Invoke-Checked $env:ComSpec @('/d','/c',$fixture,'0') 2>$null)
if ($actual.Count -ne 1 -or $actual[0] -ne 'expected-stdout') { throw "stderr contaminated stdout: $actual" }
if ($ErrorActionPreference -ne 'Stop') { throw 'Error preference leaked' }
$caught = $false
try { Invoke-Checked $env:ComSpec @('/d','/c',$fixture,'7') | Out-Null }
catch { if ($_.Exception.Message -notmatch 'Command failed \(7\)') { throw }; $caught = $true }
if (-not $caught) { throw 'Nonzero exit did not fail' }
if ($ErrorActionPreference -ne 'Stop') { throw 'Error preference leaked after failure' }
$caught = $false
try { Invoke-Checked 'geobayes-nonexistent-command-12345' }
catch { $caught = $true }
if (-not $caught) { throw 'Missing executable did not fail' }
Write-Host 'PASS native stderr: success, stdout isolation, exit 7, missing executable, preference restoration'

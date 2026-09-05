$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
function Expect-Failure([scriptblock]$Action, [string]$Pattern) {
    try { & $Action } catch { if ($_.Exception.Message -notmatch $Pattern) { throw }; return }
    throw "Expected failure: $Pattern"
}
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('geobayes-lock-' + [guid]::NewGuid().ToString('N'))
Set-Content $fixture @('a==1','b==2')
function Fake-Python { 'b==2'; 'a==1'; $global:LASTEXITCODE = 0 }
Assert-LockSnapshot Fake-Python $fixture
function Fake-Python { 'a==2'; 'b==2'; $global:LASTEXITCODE = 0 }
Expect-Failure { Assert-LockSnapshot Fake-Python $fixture } 'differs'
function Fake-Python { $global:LASTEXITCODE = 7 }
Expect-Failure { Assert-LockSnapshot Fake-Python $fixture } 'pip freeze failed'
Expect-Failure { Invoke-Checked Fake-Python } 'Command failed'
Expect-Failure { Assert-LockSnapshot Fake-Python "$fixture-missing" } 'Missing'
function py { $global:LASTEXITCODE = 103 }
Expect-Failure { Assert-LauncherPython '3.12' } 'Install/register 64-bit Python 3.12'
Write-Host 'PASS common helpers: equal/drift/missing/failed freeze, command failure, missing Python'

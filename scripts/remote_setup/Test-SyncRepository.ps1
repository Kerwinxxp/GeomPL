[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot
. (Join-Path $Here 'Common.ps1')
$Root = Join-Path ([System.IO.Path]::GetTempPath()) ('geobayes-sync-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Root | Out-Null

function New-TestOrigin {
    param([string]$Name)
    $seed = Join-Path $Root "$Name-seed"
    $bare = Join-Path $Root "$Name.git"
    Invoke-Checked git @('init','--initial-branch=main',$seed) | Out-Null
    Invoke-Checked git @('-C',$seed,'config','user.email','remote-setup-test@example.invalid')
    Invoke-Checked git @('-C',$seed,'config','user.name','Remote Setup Test')
    Set-Content -LiteralPath (Join-Path $seed 'README.md') -Value $Name
    Invoke-Checked git @('-C',$seed,'add','README.md')
    Invoke-Checked git @('-C',$seed,'commit','-m','seed') | Out-Null
    Invoke-Checked git @('clone','--bare',$seed,$bare) | Out-Null
    return $bare
}

function New-TestConfig {
    param([string]$Name, [string]$ProjectRoot, [string]$RepoUrl)
    $path = Join-Path $Root "$Name.psd1"
    $escapedRoot = $ProjectRoot.Replace("'", "''")
    $escapedRepo = $RepoUrl.Replace("'", "''")
    Set-Content -LiteralPath $path -Value "@{ MachineRole='test'; WorkerName='test'; ShardIndex=0; NumShards=2; ProjectRoot='$escapedRoot'; RepoUrl='$escapedRepo'; Branch='main'; PythonVersion='3.12'; RequiredGpuPattern='RTX 4090'; MinimumFreeDiskGB=1 }"
    return $path
}

function Assert-ThrowsLike {
    param([scriptblock]$Action, [string]$Pattern)
    try { & $Action }
    catch { if ($_.Exception.Message -notmatch $Pattern) { throw }; return }
    throw "Expected failure matching: $Pattern"
}

$origin1 = New-TestOrigin 'origin-one'
$origin2 = New-TestOrigin 'origin-two'

# Scripts staged outside an absent ProjectRoot.
$outsideRoot = Join-Path $Root 'outside-target'
$outsideCfg = New-TestConfig 'outside' $outsideRoot $origin1
& (Join-Path $Here '02-Sync-Repository.ps1') -Config $outsideCfg
if (-not (Test-Path (Join-Path $outsideRoot '.git'))) { throw 'Outside bootstrap clone failed.' }

# ProjectRoot containing only bootstrap assets.
$insideRoot = Join-Path $Root 'inside-target'
$insideSetup = Join-Path $insideRoot 'scripts\remote_setup'
New-Item -ItemType Directory -Path $insideSetup -Force | Out-Null
Set-Content -LiteralPath (Join-Path $insideSetup 'bootstrap.marker') -Value 'preserve me'
$insideCfg = New-TestConfig 'inside' $insideRoot $origin1
& (Join-Path $Here '02-Sync-Repository.ps1') -Config $insideCfg
if (-not (Test-Path (Join-Path $insideRoot '.git'))) { throw 'Inside bootstrap clone failed.' }
if (-not (Test-Path (Join-Path $insideRoot 'scripts\remote_setup\bootstrap.marker'))) { throw 'Bootstrap assets were not restored.' }

# Existing checkout must reject a different origin.
$wrongCfg = New-TestConfig 'wrong-origin' $outsideRoot $origin2
Assert-ThrowsLike { & (Join-Path $Here '02-Sync-Repository.ps1') -Config $wrongCfg } 'Origin mismatch'

# Existing checkout must reject detached HEAD.
Invoke-Checked git @('-C',$outsideRoot,'checkout','--detach')
Assert-ThrowsLike { & (Join-Path $Here '02-Sync-Repository.ps1') -Config $outsideCfg } 'HEAD is detached'

# Existing checkout must reject an attached non-main branch.
Invoke-Checked git @('-C',$outsideRoot,'checkout','-b','feature-test')
Assert-ThrowsLike { & (Join-Path $Here '02-Sync-Repository.ps1') -Config $outsideCfg } "Expected attached branch 'main'"



# Empty bootstrap directory must also clone successfully.
$emptyRoot = Join-Path $Root 'empty-target'
New-Item -ItemType Directory -Path $emptyRoot | Out-Null
$emptyCfg = New-TestConfig 'empty' $emptyRoot $origin1
& (Join-Path $Here '02-Sync-Repository.ps1') -Config $emptyCfg
if (-not (Test-Path (Join-Path $emptyRoot '.git'))) { throw 'Empty bootstrap failed.' }

# A non-Git user directory must never be repurposed.
$userRoot = Join-Path $Root 'user-target'
New-Item -ItemType Directory -Path $userRoot | Out-Null
Set-Content (Join-Path $userRoot 'keep.txt') 'keep'
$userCfg = New-TestConfig 'user' $userRoot $origin1
Assert-ThrowsLike { & (Join-Path $Here '02-Sync-Repository.ps1') -Config $userCfg } 'contains files beyond'
if ((Get-Content (Join-Path $userRoot 'keep.txt')) -ne 'keep') { throw 'User data modified' }
# Dirty checkout is rejected; an explicit override preserves its file.
Set-Content (Join-Path $emptyRoot 'untracked.txt') 'keep'
Assert-ThrowsLike { & (Join-Path $Here '02-Sync-Repository.ps1') -Config $emptyCfg } 'local changes'
& (Join-Path $Here '02-Sync-Repository.ps1') -Config $emptyCfg -AllowDirty
if ((Get-Content (Join-Path $emptyRoot 'untracked.txt')) -ne 'keep') { throw 'Dirty data modified' }


# A failed clone must preserve staged bootstrap content.
$failedRoot = Join-Path $Root 'failed-target'
New-Item -ItemType Directory -Path (Join-Path $failedRoot 'scripts/remote_setup') -Force | Out-Null
Set-Content (Join-Path $failedRoot 'scripts/remote_setup/keep.txt') 'keep'
$failedCfg = New-TestConfig 'failed' $failedRoot (Join-Path $Root 'missing.git')
Assert-ThrowsLike { & (Join-Path $Here '02-Sync-Repository.ps1') -Config $failedCfg } 'Command failed'
if ((Get-Content (Join-Path $failedRoot 'scripts/remote_setup/keep.txt')) -ne 'keep') { throw 'Bootstrap lost after clone failure' }
# A genuine upstream advance is fast-forwarded without deleting local data.
$seed = Join-Path $Root 'origin-one-seed'
Set-Content (Join-Path $seed 'new.txt') 'upstream'
Invoke-Checked git @('-C',$seed,'add','new.txt')
Invoke-Checked git @('-C',$seed,'commit','-m','advance') | Out-Null
Invoke-Checked git @('-C',$seed,'push',$origin1,'main')
& (Join-Path $Here '02-Sync-Repository.ps1') -Config $emptyCfg -AllowDirty
if ((Get-Content (Join-Path $emptyRoot 'new.txt')) -ne 'upstream') { throw 'Fast-forward failed' }
if ((Get-Content (Join-Path $emptyRoot 'untracked.txt')) -ne 'keep') { throw 'Local data lost during fast-forward' }
Write-Host "PASS repository integration tests: $Root" -ForegroundColor Green

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-SetupConfig {
    param([Parameter(Mandatory)][string]$ConfigPath)
    $resolved = (Resolve-Path -LiteralPath $ConfigPath).Path
    $cfg = Import-PowerShellDataFile -LiteralPath $resolved
    foreach ($key in @('ProjectRoot','RepoUrl','Branch','PythonVersion','WorkerName','NumShards','ShardIndex')) {
        if (-not $cfg.ContainsKey($key)) { throw "Config is missing '$key': $resolved" }
    }
    if ($cfg.ShardIndex -lt 0 -or $cfg.ShardIndex -ge $cfg.NumShards) {
        throw 'ShardIndex must be in [0, NumShards).'
    }
    return $cfg
}

function Get-ModelPins {
    return Import-PowerShellDataFile -LiteralPath (Join-Path $PSScriptRoot 'Pins.psd1')
}

function ConvertTo-NormalizedGitUrl {
    param([Parameter(Mandatory)][string]$Url)
    $value = $Url.Trim().Replace('\','/').TrimEnd('/')
    if ($value.EndsWith('.git', [System.StringComparison]::OrdinalIgnoreCase)) {
        $value = $value.Substring(0, $value.Length - 4)
    }
    return $value.ToLowerInvariant()
}

function Assert-RepositoryIdentity {
    param([Parameter(Mandatory)][hashtable]$Config)
    $root = $Config.ProjectRoot
    try { $origin = Invoke-Checked git @('-C',$root,'remote','get-url','origin') }
    catch { throw 'Git remote origin is missing.' }
    if (-not $origin) { throw 'Git remote origin is missing.' }
    if ((ConvertTo-NormalizedGitUrl $origin) -ne (ConvertTo-NormalizedGitUrl $Config.RepoUrl)) {
        throw "Origin mismatch. Expected '$($Config.RepoUrl)', found '$origin'."
    }
    try { $branch = Invoke-Checked git @('-C',$root,'symbolic-ref','--quiet','--short','HEAD') }
    catch { throw 'HEAD is detached; an attached main branch is required.' }
    if (-not $branch) { throw 'HEAD is detached; an attached main branch is required.' }
    if ($Config.Branch -ne 'main' -or $branch.Trim() -ne 'main') {
        throw "Expected attached branch 'main'; config='$($Config.Branch)', checkout='$branch'."
    }
}

function Assert-Python64 {
    param([Parameter(Mandatory)][string]$Python)
    Invoke-Checked $Python @('-c','import struct,sys; assert struct.calcsize("P") * 8 == 64, "64-bit Python required"; print(sys.version.split()[0], "64-bit")')
}

function Assert-PackageVersions {
    param([Parameter(Mandatory)][string]$Python, [Parameter(Mandatory)][hashtable]$Expected)
    $pairs = ($Expected.GetEnumerator() | Sort-Object Key | ForEach-Object { "$($_.Key)==$($_.Value)" }) -join ';'
    $code = 'import importlib.metadata as m,sys; expected=dict(x.split("==",1) for x in sys.argv[1].split(";")); actual={k:m.version(k) for k in expected}; bad={k:(expected[k],actual[k]) for k in expected if expected[k]!=actual[k]}; print(actual); assert not bad, f"version mismatch: {bad}"'
    Invoke-Checked $Python @('-c',$code,$pairs)
}

function Assert-LockSnapshot {
    param([Parameter(Mandatory)][string]$Python, [Parameter(Mandatory)][string]$LockPath)
    if (-not (Test-Path -LiteralPath $LockPath)) { throw "Missing pip freeze lock snapshot: $LockPath" }
    try { $current = @(Invoke-Checked $Python @('-m','pip','freeze','--all')) }
    catch { throw "pip freeze failed for $Python : $($_.Exception.Message)" }
    $locked = @(Get-Content -LiteralPath $LockPath)
    if (@(Compare-Object $locked $current).Count -ne 0) { throw "Environment differs from lock snapshot: $LockPath" }
}

function Assert-ProjectRoot {
    param([Parameter(Mandatory)][string]$ProjectRoot)
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.git'))) {
        throw "GeoBayes Git checkout not found at $ProjectRoot. Run 02-Sync-Repository.ps1 first."
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$ArgumentList = @()
    )
    # PS5.1 turns redirected native stderr into ErrorRecord objects. Treat those
    # as diagnostics, not failure; only the native exit code decides success.
    # Keep stdout clean for callers capturing hashes, versions and freeze lines.
    Get-Command $FilePath -ErrorAction Stop | Out-Null
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $PSNativeCommandUseErrorActionPreference = $false
        & $FilePath @ArgumentList 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                Write-Host $_.ToString()
            } else { $_ }
        }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "Command failed ($exitCode): $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-PythonPath {
    param([Parameter(Mandatory)][string]$ProjectRoot, [Parameter(Mandatory)][ValidateSet('GeoRanker','Sam3','LaMa')][string]$Environment)
    $relative = if ($Environment -eq 'GeoRanker') { 'belief_elicit\.venv_gr\Scripts\python.exe' } elseif ($Environment -eq 'Sam3') { 'cue_extract\.venv\Scripts\python.exe' } else { 'belief_elicit\.venv_lama\Scripts\python.exe' }
    return Join-Path $ProjectRoot $relative
}

function Assert-LauncherPython {
    param([string]$Version)
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Install 64-bit Python $Version with the Windows py launcher, then rerun the audit." }
    try { Invoke-Checked py @("-$Version",'-c','import struct,sys; sys.exit(0 if struct.calcsize("P")==8 else 9)') }
    catch { throw "Install/register 64-bit Python $Version with py.exe (check py --list), then rerun the audit. Existing virtual environments will not be repaired automatically." }
}

function Get-ExpectedPackages {
    param([string]$Requirements)
    $expected = @{}
    foreach ($line in Get-Content -LiteralPath $Requirements) {
        if ($line -match '^([^=\[]+)(?:\[[^]]+\])?==(.+)$') { $expected[$Matches[1]] = $Matches[2] }
    }
    return $expected
}

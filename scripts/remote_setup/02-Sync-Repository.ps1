[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'Config.psd1'),
    [switch]$AllowDirty
)

. (Join-Path $PSScriptRoot 'Common.ps1')
$cfg = Get-SetupConfig $Config
$parent = Split-Path -Parent $cfg.ProjectRoot
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$restoredBootstrap = $false

function Test-BootstrapOnlyDirectory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $entries = @(Get-ChildItem -LiteralPath $Path -Force)
    if ($entries.Count -eq 0) { return $true }
    if ($entries.Count -ne 1 -or $entries[0].Name -ne 'scripts' -or -not $entries[0].PSIsContainer) { return $false }
    $scriptEntries = @(Get-ChildItem -LiteralPath $entries[0].FullName -Force)
    return ($scriptEntries.Count -eq 1 -and $scriptEntries[0].Name -eq 'remote_setup' -and $scriptEntries[0].PSIsContainer)
}

if (-not (Test-Path -LiteralPath $cfg.ProjectRoot)) {
    Invoke-Checked git @('clone','--branch',$cfg.Branch,'--single-branch',$cfg.RepoUrl,$cfg.ProjectRoot)
} elseif (-not (Test-Path -LiteralPath (Join-Path $cfg.ProjectRoot '.git'))) {
    if (-not (Test-BootstrapOnlyDirectory $cfg.ProjectRoot)) {
        throw "ProjectRoot exists, is not Git, and contains files beyond scripts/remote_setup: $($cfg.ProjectRoot)"
    }
    # Resolve and reject volume roots before moving the bootstrap tree.
    $projectFull = [IO.Path]::GetFullPath($cfg.ProjectRoot).TrimEnd('\','/')
    if ($projectFull -eq [IO.Path]::GetPathRoot($projectFull).TrimEnd('\','/')) { throw 'ProjectRoot cannot be a volume root.' }
    $parent = Split-Path -Parent $projectFull
    $backup = Join-Path $parent ('.geobayes-bootstrap-' + [guid]::NewGuid().ToString('N'))
    Move-Item -LiteralPath $projectFull -Destination $backup
    try {
        Invoke-Checked git @('clone','--branch',$cfg.Branch,'--single-branch',$cfg.RepoUrl,$cfg.ProjectRoot)
        $sourceSetup = Join-Path $backup 'scripts\remote_setup'
        $targetScripts = Join-Path $cfg.ProjectRoot 'scripts'
        New-Item -ItemType Directory -Path $targetScripts -Force | Out-Null
        if (Test-Path -LiteralPath $sourceSetup) {
            Copy-Item -LiteralPath $sourceSetup -Destination $targetScripts -Recurse -Force
        }
        Remove-Item -LiteralPath $backup -Recurse -Force
        $restoredBootstrap = $true
    } catch {
        if (-not (Test-Path -LiteralPath $cfg.ProjectRoot)) { Move-Item -LiteralPath $backup -Destination $cfg.ProjectRoot }
        throw
    }
}

Assert-RepositoryIdentity $cfg

$dirty = Invoke-Checked git @('-C',$cfg.ProjectRoot,'status','--porcelain')
if ($dirty -and -not $AllowDirty -and -not $restoredBootstrap) {
    throw 'Remote checkout has local changes. Commit/stash them or explicitly use -AllowDirty; no reset was performed.'
}

Invoke-Checked git @('-C',$cfg.ProjectRoot,'fetch','--prune','origin',$cfg.Branch)
$local = (Invoke-Checked git @('-C',$cfg.ProjectRoot,'rev-parse','HEAD')).Trim()
$remote = (Invoke-Checked git @('-C',$cfg.ProjectRoot,'rev-parse',"origin/$($cfg.Branch)")).Trim()
$base = (Invoke-Checked git @('-C',$cfg.ProjectRoot,'merge-base',$local,$remote)).Trim()
if ($local -ne $remote) {
    if ($base -ne $local) { throw "Remote checkout is not a fast-forward of origin/$($cfg.Branch); refusing to rewrite history." }
    Invoke-Checked git @('-C',$cfg.ProjectRoot,'merge','--ff-only',"origin/$($cfg.Branch)")
}

Assert-RepositoryIdentity $cfg

Write-Host "Repository ready at $($cfg.ProjectRoot)"
Write-Host "Commit: $(Invoke-Checked git @('-C',$cfg.ProjectRoot,'rev-parse','HEAD'))"

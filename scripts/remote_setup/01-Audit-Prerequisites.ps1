[CmdletBinding()]
param([string]$Config = (Join-Path $PSScriptRoot 'Config.psd1'))

. (Join-Path $PSScriptRoot 'Common.ps1')
$cfg = Get-SetupConfig $Config
$failures = [System.Collections.Generic.List[string]]::new()

foreach ($command in @('git','py','nvidia-smi')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        $failures.Add("Required command is unavailable: $command")
    }
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "Git: $(Invoke-Checked git @('--version'))"
}

foreach ($version in @($cfg.PythonVersion, '3.11') | Select-Object -Unique) {
    try { Assert-LauncherPython $version } catch { $failures.Add($_.Exception.Message) }
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = $null
    try { $gpu = Invoke-Checked nvidia-smi @('--query-gpu=name,driver_version,memory.total','--format=csv,noheader') | Select-Object -First 1 }
    catch { $failures.Add("nvidia-smi failed: $($_.Exception.Message)") }
    Write-Host "GPU: $gpu"
    if (-not $gpu -or $gpu -notmatch [regex]::Escape($cfg.RequiredGpuPattern)) {
        $failures.Add("Expected GPU matching '$($cfg.RequiredGpuPattern)'; detected '$gpu'.")
    }
}

$root = [System.IO.Path]::GetPathRoot($cfg.ProjectRoot)
$drive = Get-PSDrive -Name $root.Substring(0,1) -ErrorAction SilentlyContinue
if (-not $drive) {
    $failures.Add("Cannot inspect drive for $($cfg.ProjectRoot).")
} else {
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Write-Host "Free disk: $freeGB GB on $root"
    if ($freeGB -lt $cfg.MinimumFreeDiskGB) { $failures.Add("At least $($cfg.MinimumFreeDiskGB) GB free is required.") }
}

if ($failures.Count) {
    $failures | ForEach-Object { Write-Error $_ -ErrorAction Continue }
    throw "Prerequisite audit failed with $($failures.Count) issue(s)."
}
Write-Host 'Prerequisite audit passed.' -ForegroundColor Green

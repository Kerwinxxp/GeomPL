# GeoBayes remote Windows setup

These scripts build a reproducible GeoBayes worker at `C:\Users\xx0073.UNT\GeoBayes`. They never copy virtual environments, model caches, or credentials from another machine.

## One-time bootstrap

Place this directory at `C:\Users\xx0073.UNT\GeoBayes\scripts\remote_setup`. Open PowerShell at the project root:

```powershell
Set-Location C:\Users\xx0073.UNT\GeoBayes
Copy-Item .\scripts\remote_setup\Config.Template.psd1 .\scripts\remote_setup\Config.psd1
notepad .\scripts\remote_setup\Config.psd1
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\remote_setup\01-Audit-Prerequisites.ps1
.\scripts\remote_setup\02-Sync-Repository.ps1
.\scripts\remote_setup\03-Setup-Environments.ps1
.\scripts\remote_setup\04-Fetch-Models.ps1
.\scripts\remote_setup\05-Warm-LaMa.ps1
.\scripts\remote_setup\06-Smoke-Test.ps1
```

`04-Fetch-Models.ps1` downloads Qwen revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac` and verifies the official GeoRanker LoRA and adapter configuration against a pinned source commit and SHA-256. It also pins the local Hugging Face `main` reference so existing offline loaders resolve to that exact snapshot. To cache SAM3 revision `3c879f39826c281e95690f02c7821c4de09afae7`, first accept the license for `facebook/sam3`, run `hf auth login` on the remote machine, and then run:

```powershell
.\scripts\remote_setup\04-Fetch-Models.ps1 -IncludeSam3
.\scripts\remote_setup\06-Smoke-Test.ps1 -IncludeSam3
```

`scripts\remote_setup\Run-All.ps1` performs all steps. Repository synchronization refuses destructive Git updates. Installation resolves dependencies again and refreshes snapshots only after validation; freeze files record an installed environment, not a portable hashed lock. For a brand-new checkout, the setup directory may be staged elsewhere temporarily. It also supports `ProjectRoot` containing only `scripts\remote_setup`; those bootstrap assets are preserved across the clone. Existing checkouts must have the normalized configured origin URL, attached `main` branch, and clean worktree unless `-AllowDirty` is explicitly supplied.

The environment installer writes complete machine-specific snapshots to `scripts\remote_setup\locks\georanker.freeze.txt` and `sam3.freeze.txt` and `lama.freeze.txt`. Smoke tests verify key package versions and reject drift from these snapshots.

## Remote-only execution (default)

Default: `NumShards = 1`, `ShardIndex = 0`, `WorkerName = 'remote4090'`. All images run remotely. Local compute is optional: explicitly use two shards on both machines, local index 0 and remote index 1, with matching inputs and model signatures. Keep each image and all its cue subsets on one worker.

After the distributed runner is committed and available on both computers, the remote command is:

```powershell
$cfg = Import-PowerShellDataFile .\scripts\remote_setup\Config.psd1
Set-Location $cfg.ProjectRoot
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
belief_elicit\.venv_gr\Scripts\python.exe -m belief_elicit.run_distributed_georanker `
  --num-shards $cfg.NumShards --shard-index $cfg.ShardIndex `
  --worker-name $cfg.WorkerName --part control --ops inpaint
```

Write each worker to its own result shard. Merge only after the compatibility and completeness checks pass. Do not point a worker at an active canonical result file.

## Revalidation

```powershell
.\scripts\remote_setup\Test-RemoteSetup.ps1
```

The test parses every PowerShell file and checks that pinned model identifiers, the expected LoRA hash, the remote path, and credential safeguards remain present.

## Python and dependency prerequisites

Install/register **64-bit Python 3.12 and 3.11** with the Windows `py` launcher on the remote machine (`py --list`). Audit and installation fail before changing environments if either interpreter is missing. A broken existing virtual environment or wrong interpreter must be recreated explicitly on the remote machine; these scripts do not repair running local environments.

Three environments avoid incompatible upstream requirements:

| Environment | Python | Path | Purpose |
|---|---|---|---|
| GeoRanker | 3.12 | `belief_elicit\.venv_gr` | Scoring and distributed runner |
| SAM3 | 3.12 | `cue_extract\.venv` | SAM3 segmentation, NumPy 2.5.1 / Pillow 12.3.0 |
| LaMa | 3.11 | `belief_elicit\.venv_lama` | Inpainting, NumPy 1.26.4 / Pillow 9.5.0 |

LaMa 0.1.2 requires NumPy <2 and Pillow <10; SAM3's Transformers vision dependency requires Pillow >=10.0.1. Pillow 9.5 has a Windows CPython 3.11 wheel, not 3.12. The small pure-Python `fire==0.5.0` dependency is built from its source distribution; other packages require wheels. No dependency constraints are bypassed. Each environment runs `pip check`, pinned-version checks and application imports before its full freeze snapshot is saved.

For cache generation use the **LaMa environment**, replacing the older precompute module's cue_extract environment example:

```powershell
belief_elicit\.venv_lama\Scripts\python.exe -m belief_elicit.precompute_inpaint --help
```

GeoRanker's scoring import chain uses torch, NumPy, Pillow, PEFT, Transformers and qwen-vl-utils plus their resolved dependencies; its current inpaint/distributed runner chain does not import OpenAI. OpenAI remains installed in the SAM3 environment for optional GPT cue extraction.

CPU-only regression checks (no model downloads or GPU initialization):

```powershell
.\scripts\remote_setup\Test-Common.ps1
.\scripts\remote_setup\Test-SyncRepository.ps1
python .\scripts\remote_setup\test_smoke_mocked.py
```

Actual deployment still requires remote Python registrations, working NVIDIA driver/CUDA wheels, repository access, the experiment code and input/cache files, sufficient disk, model downloads, and Hugging Face SAM3 license acceptance/authentication. Run `06-Smoke-Test.ps1 -IncludeSam3` remotely for real model validation. Omitting the switch does not validate SAM3 inference. GeoRanker smoke loads the exact pinned snapshot offline and validates adapter hashes before loading.

## Validation host and test dependencies

PowerShell regression scripts are validated with `powershell.exe -NoProfile -File` (Windows PowerShell 5.1), not only pwsh. `Test-NativeCommands.ps1` checks native stderr with exit 0, stdout isolation, exit 7, missing executable, and error-preference restoration. Production native calls use this checked wrapper so redirected progress output does not become a terminating PS5.1 error.

The Python mock tests are not dependency-free. They use the standard-library unittest runner, real Pillow and NumPy, and fake model/network boundaries; pytest is unnecessary. The verified interpreter was `miniconda3\python.exe` on the validation host (exact path in VALIDATION.md), Python 3.13.9, Pillow 12.1.1, NumPy 2.4.2. A clean Python 3.14 without Pillow/NumPy will not run these tests.

To prepare a separate test environment (commands for the user; not executed against local environments here):

```powershell
py -3.13 -m venv "$env:TEMP\geobayes-smoke-tests"
& "$env:TEMP\geobayes-smoke-tests\Scripts\python.exe" -m pip install Pillow==12.1.1 numpy==2.4.2
& "$env:TEMP\geobayes-smoke-tests\Scripts\python.exe" .\scripts\remote_setup\test_smoke_mocked.py
```

This optional test environment is separate from the remote model environments. Python 3.14 and a fresh test environment were not validated in this pass.

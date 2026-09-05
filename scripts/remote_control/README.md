# Local controller for Windows experiment workers

This standard-library Python CLI controls the existing
`belief_elicit.run_distributed_georanker` module. It does not provision a host,
install SSH, synchronize code, generate caches, schedule a queue, or merge results.

**Current setup is blocked pending remote host approval. No SSH alias is known.**
The example config intentionally has empty values. SSH availability, credentials,
connectivity, and remote readiness have not been established. Live `doctor` is
blocked until you supply an SSH host alias and absolute Windows repository root.
Configuration alone does not establish host approval or readiness.

## Prerequisites and configuration

- Local Python 3.10+; `py -3.14` is available for this workspace's tests.
  The default Miniconda `python` does not have pytest installed.
- For remote operations: user-configured OpenSSH host alias, authentication,
  pre-approved known-host key, and SSH access to the approved Windows host.
  Batch mode and strict host-key checking are enabled; the controller never
  invents an address, supplies credentials, or accepts a new host key.
- OpenSSH SCP **9.0+ with its default SFTP mode** and a remote SFTP subsystem.
  Local OpenSSH 9.5 has been verified separately; remote SFTP is still unverified.
  Older legacy SCP shell mode is unsupported. No `-O`, wildcard copies, or
  filename-check disabling are used. Paths containing spaces are passed as a
  single literal SFTP source argument, not shell-quoted paths.
- Remote Windows PowerShell 5.1 (`powershell.exe`), synchronized repository,
  prepared Python environment, models, input data, sweep, and compatible caches.
  The default executable is `belief_elicit/.venv_gr/Scripts/python.exe` under
  the configured repository root. Optional `python` must be an executable path,
  not a command containing arguments.

Copy `config.example.json` to `scripts/remote_control/config.local.json` and fill `ssh_host` with your
existing SSH config alias and `remote_root` with your actual absolute drive path
(forward slashes are convenient in JSON). Host aliases allow letters, digits,
underscores and hyphens; they must start with a letter or digit. Root paths allow
spaces and apostrophes but exclude traversal, UNC paths and shell metacharacters.
Keep secrets out of this file; authentication belongs in your SSH configuration.
JSON config files in this controller directory are ignored except the blank
example; its `staging/` directory is also ignored. The root `.gitignore` excludes
`.remote_control/`, including remote job state, results and logs once that ignore
change is synchronized. Custom config/staging paths elsewhere need their own
ignore rules.

Commit synchronization and cache generation can be separately staged after
approval, using the repository's existing setup workflow. This controller neither
performs nor verifies those stages. Synchronize the intended code and inputs,
generate/transfer caches, and complete the existing readiness checks before
scoring. No commits or result synchronization are implied by a controller command.

## Commands

Run from the repository root. The config path below is a local file you create;
there is no default host.

```powershell
py -3.14 scripts/remote_control/controller.py doctor
py -3.14 scripts/remote_control/controller.py --config scripts/remote_control/config.local.json doctor
# Only after host approval and configuration; this opens a live SSH connection:
py -3.14 scripts/remote_control/controller.py --config scripts/remote_control/config.local.json doctor --live
py -3.14 scripts/remote_control/controller.py --config scripts/remote_control/config.local.json start experiment-001
py -3.14 scripts/remote_control/controller.py --config scripts/remote_control/config.local.json status experiment-001
py -3.14 scripts/remote_control/controller.py --config scripts/remote_control/config.local.json fetch experiment-001
# Explicit resume, only after the prior process is proven stopped:
py -3.14 scripts/remote_control/controller.py --config scripts/remote_control/config.local.json resume experiment-001
```

Offline `doctor` checks configuration and local executable discovery only; it
never connects or starts compute. Exit 2 indicates missing/invalid configuration.
Live `doctor` checks remote repository, runner and executable paths only. It
does not load models or certify GPU/cache readiness. Transport/configuration
errors return exit 2. `status` itself returns 0 when the query succeeds, including
when its JSON reports a failed experiment; inspect the JSON `state`/`exit_code`.

`start` defaults to **the full main/inpaint workload on the remote**: `--part main --ops inpaint
--num-shards 1 --shard-index 0 --worker-name remote4090 --limit 0`.
Controls require explicit switches such as `--part control --ops inpaint` or
`--part all --ops both`; they are not included by default.
It accepts those same options plus `--cache`, `--sweep` and `--prepare-only`.
Cache/sweep paths refer to the worker filesystem. `--prepare-only` is the
runner's metadata preparation mode; it does not generate image caches.
No arbitrary command passthrough or output override is exposed.
The child sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`: required model
files must already be in the prepared cache. Missing cached files fail instead
of triggering Hugging Face downloads. These flags do not validate cache identity
or pin code themselves; staging the intended pinned cache remains a prerequisite.

Local execution is opt-in on Windows: set `local_root` in the config and supply
both global flags `--local --allow-local-compute`. Optional `python` then refers
to the local executable. The controller never falls back to local compute on
SSH failure and never automatically assigns a local GPU. For a deliberately
split run, explicitly select matching shard counts, distinct shard indices and
worker names yourself.

## Process and artifact behavior

Each task gets `<remote_root>/.remote_control/jobs/<task>/`. Task IDs are limited
to 1-64 ASCII letters/digits/underscores/hyphens, start with a letter/digit, and
exclude Windows device names. Windows treats case variants as the same ID.

Starts reserve `launch.lock` with atomic `CreateNew` before spawning a hidden
detached PowerShell wrapper via `Start-Process`. Both the SSH command and child
wrapper use UTF-16LE `-EncodedCommand`; argument values use PowerShell literals.
`process.json` records the wrapper PID and start time, `stdout.log`/`stderr.log`
capture logs, and `exit.json` records the runner's exit code. The runner's child
process is waited for by the wrapper.

The initial reservation is permanent: repeated `start` calls cannot reuse an ID.
This prevents duplicate same-ID starts after timeouts or simultaneous requests.
It does not deduplicate different task IDs with the same experiment settings.
`launch.json` saves the exact argv, executable, working directory and encoded
child command on first start; the controller never rewrites it on resume.

Explicit `resume TASK` reuses that saved command and output directory with no
argument overrides. An exclusive file guard serializes its check and launch.
It requires a valid saved PID/start time, that original process identity to be
absent (a reused PID must have a different start time), and a parseable exit
marker. A running process, missing identity, access/query error, pending launch,
or stopped wrapper without an exit marker blocks resume. The last case may hide
orphaned runner children, so forced termination cannot always be resumed safely.
The runner performs its existing metadata compatibility checks and reuses its
partial output; result files are never deleted by the controller. Prior logs
and process/exit records move to a unique remote `attempt-*` directory before
resume; fetch retrieves the current attempt's logs and results only.

On an ambiguous launch, query status and inspect logs; do not submit a new ID
until you establish whether the original is still running. There is no automatic
retry, lock cleanup or cancellation. A failed spawn may leave `launch.pending`
and require manual investigation; the controller does not clear uncertain state.

`status` is a one-shot poll of PID **and start time**, exit state, JSON record and
variant counts, and unreadable result files. Repeat it manually as needed.
`launch-uncertain` or `stopped-unknown` requires investigation. `exited-zero`
means exit code zero, not validated result completeness (especially with
`--prepare-only`). Counts are observational, not strict scientific validation.
Forced termination may prevent exit-state writing. Detached survival across SSH
disconnects depends on the target's SSH/service policies and has not been live
verified; test that behavior on the approved host before relying on it.

`fetch` copies only allowlisted result JSON, metadata, process/exit state and logs
into a fresh uniquely named directory under `scripts/remote_control/staging/`
(or `--staging`). It never merges or replaces canonical experiment outputs.
Repository-root and `belief_elicit` staging destinations are rejected. Failed
copies retain `FETCH_INCOMPLETE`; successful copies rename it `FETCH_COMPLETE`.
That marker means **transfer only**, not that results are complete or validated.
By default, fetch refuses live or uncertain states. `fetch TASK --partial`
explicitly allows a partial snapshot for inspection. Copies are not atomic across
files, and the pre-copy status check is not a lock against a concurrent resume;
avoid concurrent resume/fetch. Poll until stopped and fetch again for final
review. Strict merge/validation remains a separate step.

## Verification

```powershell
py -3.14 -m pytest tests/test_remote_control.py -q
```

Tests mock SSH/SCP transport, including failures and quoting. Local PowerShell
checks parse generated scripts, exercise task reservation, resume with a stubbed
launcher, and count temporary fixture results. They make no live remote connections
and execute no experiment or GPU computation. Live transport and remote
prerequisites remain unverified.

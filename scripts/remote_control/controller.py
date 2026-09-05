"""Small Windows execution controller. No network or compute on import."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
import tempfile


def task_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('task ID must be 1-64 ASCII letters, digits, underscores or hyphens; start with a letter/digit')
    if value.upper() in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(10)), *('LPT'+str(i) for i in range(10))}:
        raise ValueError('reserved Windows task ID')
    return value


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def encoded(script):
    return base64.b64encode(script.encode('utf-16le')).decode('ascii')


def root_path(config):
    value = config.get('remote_root', '')
    # No SCP metacharacters, traversal, UNC paths or control characters.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]:[/\\][A-Za-z0-9 _.'()/\\-]+", value):
        raise ValueError('configure remote_root as an absolute Windows drive path')
    path = PureWindowsPath(value)
    if '..' in path.parts or any(p.endswith((' ', '.')) for p in path.parts[1:]):
        raise ValueError('unsafe remote_root')
    return path.as_posix().rstrip('/')


def job_path(config, task):
    return root_path(config) + '/.remote_control/jobs/' + task_id(task)


def prelude(config, task):
    return "$ErrorActionPreference = 'Stop'\n$job = " + quote(job_path(config, task)) + "\n"


def start_script(config, args):
    task_id(args.task)
    task_id(args.worker_name)
    if not 1 <= args.num_shards <= 999 or not 0 <= args.shard_index < args.num_shards or args.limit < 0:
        raise ValueError('invalid shard range or limit')
    root = root_path(config)
    job = job_path(config, args.task)
    python = config.get('python', root + '/belief_elicit/.venv_gr/Scripts/python.exe')
    argv = ['-m', 'belief_elicit.run_distributed_georanker', '--num-shards', str(args.num_shards),
            '--shard-index', str(args.shard_index), '--worker-name', args.worker_name.lower(),
            '--part', args.part, '--ops', args.ops, '--out-dir', job,
            '--limit', str(args.limit)]
    for option in ('cache', 'sweep'):
        if getattr(args, option):
            argv.extend(['--' + option, getattr(args, option)])
    if args.prepare_only:
        argv.append('--prepare-only')
    child = prelude(config, args.task) + f"""
$code = 1
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
try {{
    Set-Location -LiteralPath {quote(root)}
    $runnerArgs = @({', '.join(map(quote, argv))})
    & {quote(python)} @runnerArgs
    $code = $LASTEXITCODE
}} catch {{ Write-Output ($_ | Out-String) }}
finally {{
    @{{exit_code=$code; finished_utc=[DateTime]::UtcNow.ToString('o')}} | ConvertTo-Json -Compress | Set-Content -LiteralPath "$job/exit.json.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$job/exit.json.tmp" -Destination "$job/exit.json"
}}
exit $code
"""
    return prelude(config, args.task) + f"""
if (!(Test-Path -LiteralPath {quote(root + '/belief_elicit/run_distributed_georanker.py')})) {{ throw 'Runner missing; synchronize repository first' }}
if (!(Test-Path -LiteralPath {quote(python)} -PathType Leaf)) {{ throw 'Configured Python executable missing' }}
[IO.Directory]::CreateDirectory($job) | Out-Null
# Atomic, permanent reservation: ambiguous starts and completed IDs cannot relaunch.
$lock = [IO.File]::Open("$job/launch.lock", [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
$lock.Dispose()
$childEncoded = {quote(encoded(child))}
$saved = @{{child_encoded=$childEncoded; root={quote(root)}; argv=@({', '.join(map(quote, argv))}); python={quote(python)}; prepare_only=${str(args.prepare_only).lower()}}}
$saved | ConvertTo-Json -Depth 4 -Compress | Set-Content -LiteralPath "$job/launch.json" -Encoding UTF8
""" + spawn_script()


def spawn_script():
    return r'''
# A failed/uncertain spawn leaves this marker and cannot be automatically resumed.
$pending = [IO.File]::Open("$job/launch.pending", [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
$pending.Dispose()
$p = Start-Process -FilePath "$PSHOME/powershell.exe" -ArgumentList @('-NoLogo','-NoProfile','-NonInteractive','-EncodedCommand',$saved.child_encoded) -WorkingDirectory $saved.root -WindowStyle Hidden -RedirectStandardOutput "$job/stdout.log" -RedirectStandardError "$job/stderr.log" -PassThru
@{pid=$p.Id; started_ticks=$p.StartTime.ToUniversalTime().Ticks.ToString(); prepare_only=$saved.prepare_only} | ConvertTo-Json -Compress | Set-Content -LiteralPath "$job/process.json.tmp" -Encoding UTF8
Move-Item -LiteralPath "$job/process.json.tmp" -Destination "$job/process.json" -Force
Remove-Item -LiteralPath "$job/launch.pending"
@{state='launched'; pid=$p.Id; job=$job} | ConvertTo-Json -Compress
'''


def resume_script(config, task):
    return prelude(config, task) + r'''
if (!(Test-Path -LiteralPath $job -PathType Container)) { throw 'Unknown task' }
# Serialize resume checks and spawning; a concurrent caller fails closed.
$guard = [IO.File]::Open("$job/resume.guard", [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    if (Test-Path -LiteralPath "$job/launch.pending") { throw 'Ambiguous launch; resume refused' }
    $identity = Get-Content -LiteralPath "$job/process.json" -Raw | ConvertFrom-Json
    if ([int]$identity.pid -le 0 -or [long]$identity.started_ticks -le 0) { throw 'Invalid process identity' }
    $p = $null
    try { $p = [Diagnostics.Process]::GetProcessById([int]$identity.pid) }
    catch [ArgumentException] { } # Only a missing PID is treated as absent.
    if ($null -ne $p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq $identity.started_ticks) {
        throw 'Task is still running; resume refused'
    }
    # A stopped wrapper without an exit marker might have orphaned runner children.
    if (!(Test-Path -LiteralPath "$job/exit.json")) { throw 'Stopped but ambiguous; missing exit marker' }
    $previous = Get-Content -LiteralPath "$job/exit.json" -Raw | ConvertFrom-Json
    if ($null -eq $previous.exit_code) { throw 'Invalid exit marker; resume refused' }
    $saved = Get-Content -LiteralPath "$job/launch.json" -Raw | ConvertFrom-Json
    if (!$saved.child_encoded -or !$saved.root -or !$saved.argv -or !$saved.python) { throw 'Invalid saved launch' }
    # Preserve prior control records/logs. Experiment outputs and saved argv stay untouched.
    $history = "$job/attempt-" + [Guid]::NewGuid().ToString('N')
    [IO.Directory]::CreateDirectory($history) | Out-Null
    foreach ($name in @('exit.json','stdout.log','stderr.log','process.json')) {
        if (Test-Path -LiteralPath "$job/$name") { Move-Item -LiteralPath "$job/$name" -Destination "$history/$name" }
    }
''' + spawn_script() + r'''
} finally { $guard.Dispose() }
'''


def status_script(config, task):
    return prelude(config, task) + r'''
if (!(Test-Path -LiteralPath $job)) { @{state='missing'; files=@()} | ConvertTo-Json -Compress; exit 0 }
$running = $false
$identity = $null
$state = 'launch-uncertain'
if (Test-Path -LiteralPath "$job/process.json") {
    $identity = Get-Content -LiteralPath "$job/process.json" -Raw | ConvertFrom-Json
    $p = Get-Process -Id $identity.pid -ErrorAction SilentlyContinue
    $running = ($null -ne $p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq $identity.started_ticks)
    if ($running) { $state = 'running' } else { $state = 'stopped-unknown' }
}
$exitCode = $null
if (Test-Path -LiteralPath "$job/exit.json") {
    $exitCode = (Get-Content -LiteralPath "$job/exit.json" -Raw | ConvertFrom-Json).exit_code
    if (!$running) {
        if ($exitCode -eq 0) { $state = 'exited-zero' } else { $state = 'failed' }
    }
}
$records = 0; $variants = 0; $unreadable = @(); $files = @()
Get-ChildItem -LiteralPath $job -File | ForEach-Object {
    if ($_.Name -match '^(stdout\.log|stderr\.log|process\.json|exit\.json|georanker_inpaint\.(main|control|all)\.[a-z0-9_-]+\.shard-\d{3}-of-\d{3}\.json(\.meta\.json)?)$') {
        if ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing linked artifact' }
        $files += $_.Name
        if ($_.Name -match '^georanker_inpaint\..*\.json$' -and $_.Name -notmatch '\.meta\.json$') {
            try {
                $data = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                if ($null -eq $data) { $data = @() }
                if ($data -isnot [Array]) { throw 'Expected result array' }
                $records += $data.Count
                foreach ($row in $data) { if ($null -ne $row.variants) { $variants += @($row.variants).Count } }
            } catch { $unreadable += $_.Name }
        }
    }
}
if (Test-Path -LiteralPath "$job/launch.pending") { $state = 'launch-uncertain' }
@{state=$state; running=$running; process=$identity; exit_code=$exitCode; records=$records; variants=$variants; unreadable=$unreadable; files=$files} | ConvertTo-Json -Depth 5 -Compress
'''


class Transport:
    def __init__(self, config, *, local=False, allow_local_compute=False):
        self.config = dict(config)
        self.local = local
        if local:
            if not allow_local_compute:
                raise ValueError('local execution requires --allow-local-compute')
            if os.name != 'nt':
                raise ValueError('local execution requires Windows')
            self.config['remote_root'] = config.get('local_root', '')
        else:
            host = config.get('ssh_host', '')
            if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', host):
                raise ValueError('configure an explicit user-provided ssh_host alias; live doctor is blocked')
            self.host = host
        root_path(self.config)

    def run(self, argv):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'transport failed (launch may be uncertain; use status): {exc}') from exc
        if result.returncode:
            raise RuntimeError(f'transport exit {result.returncode}: {result.stderr.strip()}')
        return result.stdout

    def powershell(self, script):
        argv = ['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded(script)]
        if not self.local:
            argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes', self.host] + argv
        try:
            return json.loads(self.run(argv).lstrip('\ufeff'))
        except json.JSONDecodeError as exc:
            raise RuntimeError('transport returned invalid JSON; use status before retrying start') from exc

    def copy(self, task, name, destination):
        source = job_path(self.config, task) + '/' + name
        if self.local:
            if Path(source).is_symlink():
                raise ValueError('refusing linked artifact')
            shutil.copyfile(source, destination / name)
        else:
            # Modern OpenSSH SCP defaults to SFTP: one literal remote path argument.
            self.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
                      self.host + ':' + source, str(destination / name)])


def fetch(transport, task, staging, *, partial=False):
    task_id(task)
    state = transport.powershell(status_script(transport.config, task))
    files = state.get('files', [])
    if not isinstance(files, list) or not files:
        raise ValueError('no artifacts available')
    for name in files:
        if not isinstance(name, str) or not re.fullmatch(r'(stdout\.log|stderr\.log|process\.json|exit\.json|georanker_inpaint\.(main|control|all)\.[a-z0-9_-]+\.shard-\d{3}-of-\d{3}\.json(\.meta\.json)?)', name):
            raise ValueError('unsafe artifact filename')
    if not partial and (state.get('running') is not False or state.get('state') not in ('exited-zero', 'failed')):
        raise ValueError('job is live or uncertain; use --partial for an explicit partial transfer')
    staging = Path(staging).resolve()
    repo = Path(__file__).resolve().parents[2]
    if staging == repo or staging.is_relative_to(repo / 'belief_elicit') or staging == Path(root_path(transport.config)).resolve():
        raise ValueError('staging must be distinct from repository root and experiment results')
    staging.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=task + '-', dir=staging))
    (destination / 'FETCH_INCOMPLETE').write_text(json.dumps(state, indent=2), encoding='utf-8')
    for name in files:
        transport.copy(task, name, destination)
    (destination / 'FETCH_INCOMPLETE').rename(destination / 'FETCH_COMPLETE')
    return destination


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--local', action='store_true')
    p.add_argument('--allow-local-compute', action='store_true')
    sub = p.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor')
    doctor.add_argument('--live', action='store_true', help='explicitly connect and check paths, without compute')
    start = sub.add_parser('start')
    start.add_argument('task')
    start.add_argument('--num-shards', type=int, default=1)
    start.add_argument('--shard-index', type=int, default=0)
    start.add_argument('--worker-name', default='remote4090')
    start.add_argument('--part', choices=['main', 'control', 'all'], default='main')
    start.add_argument('--ops', choices=['inpaint', 'gray', 'both'], default='inpaint')
    start.add_argument('--limit', type=int, default=0)
    start.add_argument('--cache')
    start.add_argument('--sweep')
    start.add_argument('--prepare-only', action='store_true')
    sub.add_parser('status').add_argument('task')
    sub.add_parser('resume', help='resume a proven stopped job using its saved command').add_argument('task')
    fetch_parser = sub.add_parser('fetch')
    fetch_parser.add_argument('task')
    fetch_parser.add_argument('--partial', action='store_true')
    fetch_parser.add_argument('--staging', type=Path, default=Path(__file__).resolve().parent / 'staging')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError('config must be a JSON object')
        if args.command == 'doctor' and not args.live:
            report = {'ssh_client': shutil.which('ssh'), 'scp_client': shutil.which('scp'),
                      'remote_verified': False, 'note': 'Offline only; remote host approval and connectivity are not verified.'}
            try:
                Transport(config, local=args.local, allow_local_compute=args.allow_local_compute)
                report['configuration'] = 'valid; live checks not performed'
            except ValueError as exc:
                report['configuration'] = str(exc)
                print(json.dumps(report, indent=2))
                return 2
            print(json.dumps(report, indent=2))
            return 0
        transport = Transport(config, local=args.local, allow_local_compute=args.allow_local_compute)
        if args.command == 'doctor':
            root = root_path(transport.config)
            python = transport.config.get('python', root + '/belief_elicit/.venv_gr/Scripts/python.exe')
            script = "$ErrorActionPreference='Stop'\n" + f"@{{root=(Test-Path -LiteralPath {quote(root)} -PathType Container); python=(Test-Path -LiteralPath {quote(python)} -PathType Leaf); runner=(Test-Path -LiteralPath {quote(root + '/belief_elicit/run_distributed_georanker.py')} -PathType Leaf)}} | ConvertTo-Json -Compress"
            result = transport.powershell(script)
            print(json.dumps(result, indent=2))
            return 0 if all(result.get(k) is True for k in ('root', 'python', 'runner')) else 2
        if args.command == 'start':
            result = transport.powershell(start_script(transport.config, args))
        elif args.command == 'resume':
            result = transport.powershell(resume_script(transport.config, args.task))
        elif args.command == 'status':
            result = transport.powershell(status_script(transport.config, args.task))
        else:
            result = {'staging': str(fetch(transport, args.task, args.staging, partial=args.partial))}
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

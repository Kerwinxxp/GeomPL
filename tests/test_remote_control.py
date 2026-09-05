import base64
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/remote_control/controller.py'


def module():
    assert SOURCE.exists(), 'controller must exist'
    spec = importlib.util.spec_from_file_location('remote_control', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config():
    return {'ssh_host': 'my-alias', 'remote_root': "C:/Owner's Work/GeoBayes"}


def test_defaults_and_encoded_quoting():
    m = module()
    args = m.parser().parse_args(['start', 'job-1'])
    script = m.start_script(config(), args)
    assert "Owner''s Work" in script
    inner = base64.b64decode(script.split('$childEncoded = ')[1].split("'")[1]).decode('utf-16le')
    assert "'--num-shards', '1', '--shard-index', '0'" in inner
    assert "'remote4090'" in inner and "'--part', 'main'" in inner
    assert "'--ops', 'inpaint'" in inner
    assert '$env:HF_HUB_OFFLINE' in inner and '$env:TRANSFORMERS_OFFLINE' in inner
    assert '-WindowStyle Hidden' in script
    assert 'CreateNew' in script and 'exit.json' in inner


@pytest.mark.parametrize('task', ['../x', '-option', 'a/b', 'a\\b', 'x;whoami', '.', 'a b'])
def test_bad_task_ids(task):
    m = module()
    with pytest.raises(ValueError):
        m.task_id(task)


def test_missing_config_never_launches():
    m = module()
    with patch.object(m.subprocess, 'run') as run:
        with pytest.raises(ValueError):
            m.Transport({})
        run.assert_not_called()


def test_transport_is_argv_encoded_and_propagates_failure():
    m = module()
    t = m.Transport(config())
    with patch.object(m.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '{}', '')) as run:
        assert t.powershell("Write-Output 'a; $x'") == {}
        argv = run.call_args.args[0]
        assert argv[:3] == ['ssh', '-o', 'BatchMode=yes']
        assert base64.b64decode(argv[-1]).decode('utf-16le') == "Write-Output 'a; $x'"
        assert not run.call_args.kwargs.get('shell', False)
    with patch.object(m.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ssh', 60)):
        with pytest.raises(RuntimeError, match='transport'):
            t.powershell('anything')


def test_status_polls_identity_and_counts():
    m = module()
    script = m.status_script(config(), 'job')
    assert 'Get-Process' in script and 'StartTime' in script
    assert 'records' in script and 'variants' in script
    assert 'unreadable' in script


def test_fetch_fresh_staging_and_failure(tmp_path):
    m = module()
    t = m.Transport(config())
    state = {'state': 'exited-zero', 'running': False, 'files': ['stdout.log', 'exit.json']}
    with patch.object(t, 'powershell', return_value=state), patch.object(t, 'copy') as copy:
        first = m.fetch(t, 'job', tmp_path)
        second = m.fetch(t, 'job', tmp_path)
        assert first != second
        assert copy.call_count == 4
        assert (first / 'FETCH_COMPLETE').exists()
    with patch.object(t, 'powershell', return_value=state), patch.object(t, 'copy', side_effect=RuntimeError('failed')):
        with pytest.raises(RuntimeError):
            m.fetch(t, 'job', tmp_path)
    assert len(list(tmp_path.glob('*/FETCH_COMPLETE'))) == 2


def test_fetch_rejects_remote_traversal(tmp_path):
    m = module()
    t = m.Transport(config())
    with patch.object(t, 'powershell', return_value={'files': ['../../secret']}), patch.object(t, 'copy') as copy:
        with pytest.raises(ValueError):
            m.fetch(t, 'job', tmp_path)
        copy.assert_not_called()


def test_scp_literal_path_and_errors(tmp_path):
    m = module()
    t = m.Transport(config())
    with patch.object(m.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'denied')) as run:
        with pytest.raises(RuntimeError, match='denied'):
            t.copy('job', 'stdout.log', tmp_path)
        argv = run.call_args.args[0]
        assert '-O' not in argv and '-T' not in argv
        assert "my-alias:C:/Owner's Work/GeoBayes/.remote_control/jobs/job/stdout.log" in argv


def test_local_requires_opt_in():
    m = module()
    with pytest.raises(ValueError, match='allow-local-compute'):
        m.Transport({'local_root': 'C:/work'}, local=True)


@pytest.mark.parametrize('host', ['', '-oProxyCommand=bad', 'user@host', 'host;echo', 'host name'])
def test_rejects_non_alias_hosts(host):
    m = module()
    with pytest.raises(ValueError):
        m.Transport({**config(), 'ssh_host': host})


def test_doctor_offline_and_live_blocked_without_configuration(capsys):
    m = module()
    with patch.object(m.subprocess, 'run') as run:
        assert m.main(['doctor']) == 2
        assert 'remote_verified' in capsys.readouterr().out
        assert m.main(['doctor', '--live']) == 2
        assert 'blocked' in capsys.readouterr().err
        run.assert_not_called()


def test_invalid_json_and_connection_failure():
    m = module()
    t = m.Transport(config())
    for code, output, error in [(0, 'banner only', ''), (255, '', 'connection refused')]:
        with patch.object(m.subprocess, 'run', return_value=subprocess.CompletedProcess([], code, output, error)):
            with pytest.raises(RuntimeError):
                t.powershell('anything')


def ps(script):
    import shutil
    executable = shutil.which('powershell.exe')
    if executable is None:
        pytest.skip('Windows PowerShell unavailable')
    return subprocess.run([executable, '-NoProfile', '-NonInteractive', '-EncodedCommand',
                           base64.b64encode(script.encode('utf-16le')).decode('ascii')],
                          capture_output=True, text=True, timeout=15)


def test_generated_powershell_parses_without_execution():
    m = module()
    args = m.parser().parse_args(['start', 'job'])
    start = m.start_script(config(), args)
    child = base64.b64decode(start.split('$childEncoded = ')[1].split("'")[1]).decode('utf-16le')
    for script in (start, child, m.status_script(config(), 'job'), m.resume_script(config(), 'job')):
        result = ps('$tokens=$null; $errors=$null; '
                    '[System.Management.Automation.Language.Parser]::ParseInput('
                    + m.quote(script) + ',[ref]$tokens,[ref]$errors) | Out-Null; '
                    'if ($errors.Count) { $errors | Out-String | Write-Output; exit 1 }')
        assert result.returncode == 0, result.stdout + result.stderr


def test_atomic_reservation_rejects_second_attempt_without_launch(tmp_path):
    m = module()
    root = tmp_path / "Owner's Work"
    (root / 'belief_elicit').mkdir(parents=True)
    (root / 'belief_elicit/run_distributed_georanker.py').touch()
    python = root / 'placeholder.exe'
    python.touch()
    cfg = {'remote_root': str(root), 'python': str(python)}
    args = m.parser().parse_args(['start', 'job'])
    # Execute reservation only, never Start-Process or any runner.
    reservation = m.start_script(cfg, args).split('$childEncoded =')[0]
    first = ps(reservation)
    second = ps(reservation)
    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert not (root / '.remote_control/jobs/job/process.json').exists()


def test_status_counts_fixture_and_does_not_claim_completion(tmp_path):
    m = module()
    cfg = {'remote_root': str(tmp_path)}
    job = tmp_path / '.remote_control/jobs/job'
    job.mkdir(parents=True)
    (job / 'georanker_inpaint.all.remote4090.shard-000-of-001.json').write_text(
        json.dumps([{'variants': [{}, {}]}, {'variants': [{}]}]))
    (job / 'exit.json').write_text('{"exit_code": 0}')
    result = ps(m.status_script(cfg, 'job'))
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status['records'] == 2 and status['variants'] == 3
    assert status['state'] == 'exited-zero' and status['running'] is False


def test_invalid_shard_rejected():
    m = module()
    args = m.parser().parse_args(['start', 'job', '--shard-index', '1'])
    with pytest.raises(ValueError):
        m.start_script(config(), args)


def test_fetch_running_requires_explicit_partial(tmp_path):
    m = module()
    t = m.Transport(config())
    with patch.object(t, 'powershell', return_value={'state': 'running', 'running': True, 'files': ['stdout.log']}), patch.object(t, 'copy') as copy:
        with pytest.raises(ValueError, match='partial'):
            m.fetch(t, 'job', tmp_path)
        copy.assert_not_called()
        destination = m.fetch(t, 'job', tmp_path, partial=True)
        assert (destination / 'FETCH_COMPLETE').exists()


def test_resume_uses_saved_command_and_refuses_ambiguous_state(tmp_path):
    m = module()
    cfg = {'remote_root': str(tmp_path)}
    script = m.resume_script(cfg, 'job')
    assert 'launch.json' in script and '$saved.child_encoded' in script
    assert 'GetProcessById' in script and 'started_ticks' in script
    assert 'exit.json' in script
    job = tmp_path / '.remote_control/jobs/job'
    job.mkdir(parents=True)
    (job / 'launch.json').write_text('{"child_encoded": "not executable"}')
    # No process identity: fail before launching anything.
    result = ps(script)
    assert result.returncode != 0
    assert not (job / 'stdout.log').exists()


def test_resume_rejects_current_process_even_with_old_exit_file(tmp_path):
    m = module()
    cfg = {'remote_root': str(tmp_path)}
    job = tmp_path / '.remote_control/jobs/job'
    job.mkdir(parents=True)
    (job / 'launch.json').write_text('{"child_encoded": "not executable"}')
    (job / 'exit.json').write_text('{"exit_code": 1}')
    setup = '$p=Get-Process -Id $PID; @{pid=$PID; started_ticks=$p.StartTime.ToUniversalTime().Ticks.ToString()} | ConvertTo-Json | Set-Content -LiteralPath ' + m.quote(str(job / 'process.json')) + ';\n'
    result = ps(setup + m.resume_script(cfg, 'job'))
    assert result.returncode != 0
    assert 'running' in result.stderr.lower()
    assert (job / 'exit.json').exists()


def test_resume_saved_arguments_preserves_results_and_serializes(tmp_path):
    m = module()
    cfg = {'remote_root': str(tmp_path)}
    job = tmp_path / '.remote_control/jobs/job'
    job.mkdir(parents=True)
    saved = {'child_encoded': m.encoded("throw 'must never execute'"), 'root': str(tmp_path),
             'python': 'original-python', 'argv': ['--part', 'main']}
    saved_text = json.dumps(saved)
    (job / 'launch.json').write_text(saved_text)
    (job / 'exit.json').write_text('{"exit_code": 1}')
    output = job / 'georanker_inpaint.main.remote4090.shard-000-of-001.json'
    output.write_text('[{"variants": []}]')
    setup = '$p=Get-Process -Id $PID; @{pid=$PID; started_ticks="1"} | ConvertTo-Json | Set-Content -LiteralPath ' + m.quote(str(job / 'process.json')) + ';\n'
    # Reused PID with different start time proves the original identity stopped.
    # Stub only process launch; execute the real resume transaction and file operations.
    stub = r'''
function Start-Process {
    if ($saved.argv[1] -ne 'main' -or $saved.python -ne 'original-python') { throw 'Changed saved argv' }
    $acquired = $false
    try { $other = [IO.File]::Open("$job/resume.guard", 'OpenOrCreate', 'Write', 'None'); $acquired = $true; $other.Dispose() } catch [IO.IOException] { }
    if ($acquired) { throw 'Resume guard was not held' }
    return (Get-Process -Id $PID)
}
'''
    result = ps(setup + stub + m.resume_script(cfg, 'job'))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['state'] == 'launched'
    assert output.read_text() == '[{"variants": []}]'
    assert (job / 'launch.json').read_text() == saved_text
    assert len(list(job.glob('attempt-*/exit.json'))) == 1
    assert not (job / 'launch.pending').exists()


@pytest.mark.parametrize('content,records,unreadable', [('[]', 0, False), ('[{"variants": [{}]}]', 1, False), ('{', 0, True)])
def test_status_empty_single_and_partial_json(tmp_path, content, records, unreadable):
    m = module()
    job = tmp_path / '.remote_control/jobs/job'
    job.mkdir(parents=True)
    (job / 'georanker_inpaint.main.remote4090.shard-000-of-001.json').write_text(content)
    result = ps(m.status_script({'remote_root': str(tmp_path)}, 'job'))
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state['records'] == records
    assert bool(state['unreadable']) == unreadable


@pytest.mark.parametrize('pending', [False, True])
def test_resume_refuses_missing_exit_or_pending_launch(tmp_path, pending):
    m = module()
    job = tmp_path / '.remote_control/jobs/job'
    job.mkdir(parents=True)
    (job / 'process.json').write_text('{"pid": 2147483647, "started_ticks": "1"}')
    if pending:
        (job / 'launch.pending').touch()
        (job / 'exit.json').write_text('{"exit_code": 1}')
    result = ps(m.resume_script({'remote_root': str(tmp_path)}, 'job'))
    assert result.returncode != 0
    assert 'ambiguous' in result.stderr.lower()


def test_failed_resume_spawn_keeps_uncertainty_marker_and_results(tmp_path):
    m = module()
    job = tmp_path / '.remote_control/jobs/job'
    job.mkdir(parents=True)
    (job / 'process.json').write_text('{"pid": 2147483647, "started_ticks": "1"}')
    (job / 'exit.json').write_text('{"exit_code": 1}')
    (job / 'launch.json').write_text(json.dumps({'child_encoded': 'test', 'root': str(tmp_path),
                                                'python': 'test', 'argv': ['test']}))
    result_file = job / 'georanker_inpaint.main.remote4090.shard-000-of-001.json'
    result_file.write_text('[]')
    result = ps("function Start-Process { throw 'simulated launch failure' }\n" +
                m.resume_script({'remote_root': str(tmp_path)}, 'job'))
    assert result.returncode != 0
    assert (job / 'launch.pending').exists()
    assert result_file.read_text() == '[]'
    assert list(job.glob('attempt-*/exit.json'))

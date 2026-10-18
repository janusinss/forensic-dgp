"""Human-launched finite AdaIN-gradient-isolation training supervisor; import never launches work."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import selectors
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1 import worker as w

PACKAGE = w.PACKAGE_DIR
PARENT_PROTOCOL_SHA = w.SOURCE_PROTOCOL_FP32
read_json, require, sha, write_json = w.read_json, w.require, w.sha256, w.atomic_json
LOG_BYTE_CAP = 2 * 1024**2


def no_links(path):
    path = Path(path).absolute()
    for entry in (path, *path.parents):
        if entry.exists():
            info = entry.lstat()
            require(not entry.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400,
                    'Symlink/reparse point forbidden in manual supervisor path')
    return path


MODULE = 'research.codeformer_adain_gradient_isolation_v1.worker'


def manual_root(root, *, system=None, home=None, environment=None):
    system = platform.system() if system is None else system
    environment = os.environ if environment is None else environment
    require(system == 'Linux', 'Training is manual on the Linux L4 VM')
    require(bool(environment.get('TMUX')), 'Run inside the human-opened tmux session')
    home = Path.home() if home is None else Path(home)
    root = no_links(root).resolve(strict=True)
    require(root == no_links(home / 'forensic-dgp' / PACKAGE).resolve(strict=True),
            'Exact new comparison directory under ~/forensic-dgp required')
    return root


def competing_commands(rows, own_pid):
    conflicts = []
    for pid, command in rows:
        if pid == own_pid:
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            require('python' not in command.lower(), 'Unclassifiable possible Python workload')
            continue
        if not words or not Path(words[0]).name.startswith(('python', 'pypy')):
            continue
        low = command.lower()
        scripts = [Path(word).name.lower() for word in words if word.endswith('.py')]
        research = '--run' in words and ('codeformer' in low or 'cctv_dgp' in low)
        training = any(name.startswith(('train_cctv_dgp_', 'train_codeformer_')) for name in scripts)
        supervisor = 'supervise_codeformer' in low or 'supervise_cctv_dgp' in low
        if research or training or supervisor:
            conflicts.append({'pid': pid, 'command': command})
    return conflicts


def idle():
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    require(not gpu.stdout.strip(), 'GPU occupied; preserve the running workload')
    ps = subprocess.run(['ps', '-eo', 'pid=,args='], check=True, capture_output=True, text=True, timeout=15)
    rows = []
    for line in ps.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            rows.append((int(parts[0]), parts[1]))
    require(not competing_commands(rows, os.getpid()), 'Competing research worker/supervisor; leave it untouched')
    return {'GPU_compute_processes': [], 'competing_research_commands': [],
            'idle_tmux_sessions_are_not_workloads': True}


def capped_log_chunk(stream, chunk, count, cap=LOG_BYTE_CAP):
    require(isinstance(chunk, bytes) and type(count) is int and type(cap) is int and
            0 <= count <= cap, 'Invalid bounded log writer state')
    accepted = min(len(chunk), cap - count)
    if accepted:
        stream.write(chunk[:accepted])
    return count + accepted, len(chunk) - accepted


def stop_owned_stage(process, grace):
    """Signal only this newly created session/process group, never other work."""
    if process.poll() is not None:
        return process.returncode
    try:
        require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
                'Refuse to signal a stage that does not own its process group/session')
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait(timeout=5)
    try:
        return process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=5)


def stage(root, protocol_sha, flag, seconds, grace, log):
    require(flag in {'--verify-transfer', '--run', '--export'} and type(seconds) is int and
            type(grace) is int and seconds > 0 and grace > 0, 'Explicit finite known stage required')
    command = ['timeout', '--signal=TERM', '--kill-after=' + str(grace), str(seconds),
               sys.executable, '-B', '-u', '-m', MODULE, '--root', str(root),
               '--protocol-sha', protocol_sha, flag]
    started, written, dropped, stop_reason = time.monotonic(), 0, 0, None
    process = None
    with no_links(log).open('xb') as output:
        try:
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
            print(json.dumps({'child_pid': process.pid, 'stage': flag, 'log': str(log),
                'external_cap_seconds': seconds, 'grace_seconds': grace,
                'streaming_log_byte_cap': LOG_BYTE_CAP}), flush=True)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() - started > seconds + grace:
                        stop_reason = 'External stage/session wall-time guard exceeded'
                        stop_owned_stage(process, 1)
                        break
                    for key, _ in selector.select(timeout=0.5):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        written, omitted = capped_log_chunk(output, chunk, written)
                        dropped += omitted
                        if omitted:
                            stop_reason = 'Strict2MiB streaming log byte cap reached'
                            remaining = max(1, int(started + seconds + grace - time.monotonic()))
                            stop_owned_stage(process, min(grace, remaining))
                            break
                    if stop_reason:
                        break
            code = process.wait(timeout=5)
            if stop_reason and code == 0:
                code = 125
        except BaseException:
            if process is not None:
                remaining = max(1, int(started + seconds + grace - time.monotonic()))
                stop_owned_stage(process, min(grace, remaining))
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            output.flush(); os.fsync(output.fileno())
    require(log.stat().st_size == written <= LOG_BYTE_CAP, 'Strict log writer accounting differs')
    with log.open('rb') as source:
        source.seek(max(0, log.stat().st_size - 16000))
        sys.stdout.write(source.read().decode('utf-8', errors='replace'))
    result = {'stage': flag, 'exit_code': code, 'seconds': time.monotonic() - started,
              'external_cap_seconds': seconds, 'grace_seconds': grace,
              'supervisor_absolute_guard_seconds': seconds + grace + 10,
              'log': log.relative_to(root).as_posix(), 'log_bytes': written,
              'log_byte_cap': LOG_BYTE_CAP, 'dropped_chunk_bytes_at_stop': dropped,
              'finite_stop_reason': stop_reason}
    print(json.dumps(result), flush=True)
    return result


def supervise(root, protocol_sha):
    root = manual_root(root)
    require(re.fullmatch('[a-f0-9]{64}', protocol_sha) and sha(root / 'protocol.json') == protocol_sha,
            'Exact sealed protocol hash required')
    require(shutil.which('timeout'), 'GNU timeout must be installed')
    p = read_json(root / 'protocol.json')
    require(p['format'] == 'codeformer-adain-gradient-isolation-finite-manual-v1' and
            p['parent_protocol_sha256'] == PARENT_PROTOCOL_SHA and
            p['training']['max_updates'] == 50 and p['training']['schedule_horizon_updates'] == 600 and
            p['training']['autocast'] == 'disabled-FP32', 'Finite AdaIN isolation comparison identity differs')
    installed = read_json(root / 'dependency_install_receipt.json')
    require(installed['complete'] is True and installed['protocol_sha256'] == protocol_sha,
            'Complete verified dependency install required before supervision')
    require(not (root / 'outputs').exists(), 'Existing training evidence: preserve it; no relaunch/resume')
    idle_before = idle()
    supervision = root / 'outputs/supervision'; supervision.mkdir(parents=True)
    stages = []
    try:
        preflight = stage(root, protocol_sha, '--verify-transfer', 300, 30, supervision / 'preflight.log')
        stages.append(preflight)
        require(preflight['exit_code'] == 0, 'Transfer verification failed; no training call')
        idle()
        trained = stage(root, protocol_sha, '--run', 3000, 180, supervision / 'trainer.log')
        stages.append(trained)
        write_json(supervision / 'run_receipt.json', {
            'protocol_sha256': protocol_sha, 'manual_tmux': True, 'initial_idle_check': idle_before,
            'stages': stages, 'execution_max_updates': 50, 'schedule_horizon_updates': 600,
            'training_process_exit_code': trained['exit_code'], 'no_automatic_retry_or_resume': True,
            'training_success_not_implied': True, 'model_qualified': False, 'automatic_app_promotion': False})
        exported = stage(root, protocol_sha, '--export', 600, 30, supervision / 'export.log')
        stages.append(exported)
        write_json(supervision / 'finished.json', {'complete': True, 'stages': stages,
            'training_process_exit_code': trained['exit_code'], 'export_exit_code': exported['exit_code'],
            'completed_archive_not_model_acceptance': True, 'model_qualified': False})
        return 0 if trained['exit_code'] == exported['exit_code'] == 0 else 1
    except BaseException as exc:
        failure = supervision / 'supervisor_failure.json'
        if not failure.exists():
            write_json(failure, {'complete': False, 'protocol_sha256': protocol_sha, 'stages': stages,
                'error_type': type(exc).__name__, 'message': str(exc), 'partial_evidence_retained': True,
                'automatic_retry_or_resume': False, 'model_qualified': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    return supervise(args.root, args.protocol_sha)


if __name__ == '__main__':
    raise SystemExit(main())

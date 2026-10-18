"""Manual finite tmux supervisor for one zero-optimizer CodeFormer measurement."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_bounded_fusion_feasibility_v1 import contract as c
from research.codeformer_bounded_fusion_feasibility_v1.worker import manual_root

MODULE = 'research.codeformer_bounded_fusion_feasibility_v1.worker'


def competing_commands(rows, own_pid):
    conflicts = []
    for pid, command in rows:
        if pid == own_pid:
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            c.require('python' not in command.lower(), 'Cannot classify potential Python workload')
            continue
        if not words or not Path(words[0]).name.startswith(('python', 'pypy')):
            continue
        lowered = command.lower()
        scripts = [Path(word).name.lower() for word in words if word.endswith('.py')]
        research_run = '--run' in words and ('codeformer' in lowered or 'cctv_dgp' in lowered)
        training_script = any(name.startswith(('train_cctv_dgp_', 'train_codeformer_')) for name in scripts)
        supervisor = 'supervise_codeformer' in lowered or 'supervise_cctv_dgp' in lowered
        if research_run or training_script or supervisor:
            conflicts.append({'pid': pid, 'command': command})
    return conflicts


def idle():
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    c.require(not gpu.stdout.strip(), 'GPU occupied; leave the current workload untouched')
    ps = subprocess.run(['ps', '-eo', 'pid=,args='], check=True, capture_output=True, text=True, timeout=15)
    rows = []
    for line in ps.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            rows.append((int(parts[0]), parts[1]))
    c.require(not competing_commands(rows, os.getpid()),
              'Competing research worker/supervisor; leave it untouched')
    return {'GPU_compute_processes': [], 'competing_research_commands': [],
            'idle_tmux_sessions_are_not_workloads': True}


def stage(root, protocol_sha, flag, seconds, grace, log):
    command = ['timeout', '--signal=TERM', '--kill-after=' + str(grace), str(seconds),
               sys.executable, '-B', '-u', '-m', MODULE, '--root', str(root),
               '--protocol-sha', protocol_sha, flag]
    started = time.monotonic()
    # A separate timeout group owns only this new child and its descendants.
    with log.open('xb') as output:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        print(json.dumps({'child_pid': process.pid, 'stage': flag, 'log': str(log),
                          'external_cap_seconds': seconds}), flush=True)
        written, log_limit, log_capped = 0, 2 * 1024**2, False
        for payload in iter(lambda: process.stdout.read(65536), b''):
            room = log_limit - written
            output.write(payload[:room]); written += min(room, len(payload))
            if len(payload) > room:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
                log_capped = True
                break
        code = process.wait()
        if log_capped:
            code = 125
        output.flush(); os.fsync(output.fileno())
    # Bounded diagnostic logs are useful when returning stopped evidence.
    c.require(log.stat().st_size <= 2 * 1024**2, 'Unexpected log size; preserve it for review')
    with log.open('rb') as source:
        source.seek(max(0, log.stat().st_size - 16000))
        sys.stdout.write(source.read().decode('utf-8', errors='replace'))
    print(json.dumps({'stage': flag, 'exit_code': code, 'seconds': time.monotonic() - started}), flush=True)
    return {'stage': flag, 'exit_code': code, 'seconds': time.monotonic() - started,
            'external_cap_seconds': seconds, 'grace_seconds': grace,
            'log': log.relative_to(root).as_posix(), 'log_capped': log_capped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    root = manual_root(args.root)
    c.require(re.fullmatch('[a-f0-9]{64}', args.protocol_sha), 'Exact frozen protocol hash required')
    c.require(shutil.which('timeout'), 'GNU timeout must be installed')
    c.require(not (root / 'outputs').exists(), 'Preserve existing execution evidence; no relaunch')
    idle_before = idle()
    supervision = root / 'outputs/supervision'
    supervision.mkdir(parents=True)
    preflight = stage(root, args.protocol_sha, '--verify-transfer', 300, 30, supervision / 'preflight.log')
    c.require(preflight['exit_code'] == 0, 'Transfer or parent dependency check failed; no neural call')
    idle()
    measured = stage(root, args.protocol_sha, '--run', 900, 60, supervision / 'diagnostic.log')
    c.write_new_json(supervision / 'run_receipt.json', {
        'protocol_sha256': args.protocol_sha, 'manual_tmux': True,
        'initial_idle_check': idle_before, 'stages': [preflight, measured],
        'optimizer_updates': 0, 'neural_measurement_complete': measured['exit_code'] == 0,
        'no_automatic_retry': True, 'training_success_not_implied': True,
        'model_qualified': False, 'automatic_app_promotion': False})
    exported = stage(root, args.protocol_sha, '--export', 300, 30, supervision / 'export.log')
    c.write_new_json(supervision / 'finished.json', {'complete': True,
        'stages': [preflight, measured, exported], 'optimizer_updates': 0,
        'measurement_or_export_completion_is_not_model_acceptance': True})
    return 0 if measured['exit_code'] == exported['exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Manual Linux/tmux finite V2 supervisor. Importing this file never launches.

Only a human-invoked main() can run the guarded worker. No retries, remote calls,
package installation, process killing of unrelated tasks or app changes exist.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time


MODULE = 'research.codeformer_adaptation_v2.worker'
HEX = re.compile(r'^[a-f0-9]{64}$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def competing_python_commands(process_rows, own_pid):
    """An idle tmux session is not a workload. Inspect actual Python commands."""
    conflicts = []
    for pid, command in process_rows:
        if pid == own_pid:
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            if 'python' in command.lower() and ('--run' in command or 'train_' in command):
                conflicts.append({'pid': pid, 'command': command, 'reason': 'unparseable potential training process'})
            continue
        if not words or not Path(words[0]).name.startswith(('python', 'pypy')):
            continue
        scripts = [Path(word).name for word in words[1:] if word.endswith('.py')]
        is_worker_run = '--run' in words and any('codeformer_adaptation_' in word or 'cctv_dgp_' in word for word in words)
        is_training_script = any(name.startswith(('train_cctv_dgp_', 'train_codeformer_')) for name in scripts)
        if is_worker_run or is_training_script:
            conflicts.append({'pid': pid, 'command': command, 'reason': 'competing training command'})
    return conflicts


def guard_environment(root, protocol_sha, *, environment=None, system=None, home=None):
    environment = os.environ if environment is None else environment
    system = platform.system() if system is None else system
    home = Path.home() if home is None else Path(home)
    require(system == 'Linux', 'V2 launch is manual-only on the Linux NVIDIA L4 VM')
    require(HEX.fullmatch(protocol_sha) is not None, 'Pass the exact frozen V2 protocol SHA256')
    root = Path(root).resolve(strict=True)
    base = (home / 'forensic-dgp').resolve(strict=True)
    require(root != base and root.is_relative_to(base), 'Packet must be below ~/forensic-dgp')
    require(bool(environment.get('TMUX')), 'Enter the manual tmux session before launch')
    for name in ('trainer.log', 'export.log', 'preflight.log', 'supervisor_receipt.json', 'outputs/adaptation_v2'):
        require(not (root / name).exists(), 'Retain existing evidence; do not repeat this packet: ' + name)
    return root


def idle_receipt():
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    require(not gpu.stdout.strip(), 'Competing CUDA process; do not launch')
    result = subprocess.run(['ps', '-eo', 'pid=,args='], check=True, capture_output=True, text=True, timeout=15)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2:
            rows.append((int(fields[0]), fields[1]))
    conflicts = competing_python_commands(rows, os.getpid())
    require(not conflicts, 'Competing Python training command: ' + json.dumps(conflicts))
    return {'GPU_compute_processes': [], 'competing_training_commands': [],
            'tmux_presence_alone_is_not_a_conflict': True}


def stage_command(root, protocol_sha, mode, seconds, grace):
    require(mode in ('--verify-transfer', '--run', '--export'), 'Unknown finite stage')
    return ['timeout', '--signal=TERM', '--kill-after=' + str(grace), str(seconds),
            sys.executable, '-B', '-u', '-m', MODULE, '--root', str(root),
            '--protocol-sha', protocol_sha, mode]


def run_stage(root, protocol_sha, mode, logfile, seconds, grace):
    started = time.monotonic()
    path = root / logfile
    command = stage_command(root, protocol_sha, mode, seconds, grace)
    # GNU timeout owns only its child process group. Unrelated tasks are untouched.
    with path.open('xb') as output:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for block in iter(lambda: process.stdout.readline(), b''):
            output.write(block); output.flush()
            sys.stdout.buffer.write(block); sys.stdout.buffer.flush()
        status = process.wait()
        output.flush(); os.fsync(output.fileno())
    return {'mode': mode, 'exit_code': status, 'seconds': time.monotonic() - started,
            'external_seconds': seconds, 'termination_grace_seconds': grace, 'log': logfile}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    root = guard_environment(args.root, args.protocol_sha)
    require(shutil.which('timeout') is not None, 'GNU timeout is required for bounded manual execution')
    idle = idle_receipt()
    preflight = run_stage(root, args.protocol_sha, '--verify-transfer', 'preflight.log', 600, 30)
    require(preflight['exit_code'] == 0, 'Static transfer verification failed; no training launched')
    # Repeat fresh idle checks after hashing the packet; no old idle receipt suffices.
    idle_receipt()
    stages = [preflight]
    stages.append(run_stage(root, args.protocol_sha, '--run', 'trainer.log', 7200, 180))
    print('Trainer exit code:', stages[-1]['exit_code'], flush=True)
    # A trainer quality/resource stop still exports exactly its retained evidence.
    stages.append(run_stage(root, args.protocol_sha, '--export', 'export.log', 600, 30))
    print('Export exit code:', stages[-1]['exit_code'], flush=True)
    receipt = {'complete': True, 'protocol_sha256': args.protocol_sha, 'stages': stages,
        'initial_idle_check': idle, 'manual_tmux': True, 'training_success_not_implied': True,
        'model_qualified': False, 'automatic_resume': False, 'automatic_app_promotion': False}
    with (root / 'supervisor_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return 0 if all(stage['exit_code'] == 0 for stage in stages) else 1


if __name__ == '__main__':
    raise SystemExit(main())

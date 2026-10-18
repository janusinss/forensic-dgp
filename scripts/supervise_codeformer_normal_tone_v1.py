"""Manual finite supervisor; disposable Adam steps are real, no training commits."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_v1 import contract as c

MODULE = 'research.codeformer_normal_tone_v1.worker'
LOG_CAP = 2 * 1024**2
PREFIX = 'cctv-codeformer-normal-tone-v1'


def final_companion(root, stages):
    """Close external final-log boundary in one exclusive bounded HOME record."""
    deadline=time.monotonic()+30
    def check():
        c.require(time.monotonic()<=deadline,'Final supervisor companion30second cap exceeded')
    home=c.no_links(Path.home())
    export_path=c.no_links(home/(PREFIX+'-export.json'))
    exported=c.read_json(export_path)
    archive=c.no_links(home/(PREFIX+'-results.tar.gz'))
    checksum=c.no_links(home/(PREFIX+'-results.tar.gz.sha256'))
    c.require(exported['complete'] is True and archive.stat().st_size==exported['bytes'] and
              c.sha256(archive,budget=check)==exported['archive_sha256'],'Final archive identity differs')
    c.require(checksum.read_bytes()==(exported['archive_sha256']+'  '+archive.name+'\n').encode('ascii'),
              'Final checksum must be exact portable ASCII/LF')
    log=c.safe_path(root,'outputs/supervision/export.log')
    c.require(log.stat().st_size<=LOG_CAP,'Final export log cap exceeded')
    payload=log.read_bytes();finished=c.safe_path(root,'outputs/supervision/finished.json')
    value={'format':'codeformer-normal-tone-supervisor-final-v1','complete':True,
        'protocol_sha256':exported['protocol_sha256'],'stages':stages,
        'archive_sha256':exported['archive_sha256'],'archive_bytes':archive.stat().st_size,
        'export_receipt_sha256':c.sha256(export_path),'export_receipt_bytes':export_path.stat().st_size,
        'checksum_sha256':c.sha256(checksum),'finished_json':c.read_json(finished),
        'finished_json_sha256':c.sha256(finished),'export_log_sha256':c.sha256(log),
        'export_log_bytes':len(payload),'export_log_encoding':'base64 exact bytes',
        'export_log_base64':base64.b64encode(payload).decode('ascii'),
        'closure_boundary':'This external final record has no self-hash and is not inside the archive it closes',
        'actual_disposable_Adam_steps':exported.get('actual_disposable_Adam_steps'),
        'committed_training_updates':0,'checkpoint_exported':False,
        'training_success_not_implied':True,'model_qualified':False}
    encoded=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')
    c.require(len(encoded)<=3*1024**2 and shutil.disk_usage(home).free>=4*1024**3+len(encoded)+65536,
              'Final companion size/reserve bound exceeded')
    check();destination=home/(PREFIX+'-supervisor-final.json')
    c.write_new_json(destination,value);check()
    return {'path':str(destination),'bytes':destination.stat().st_size,'seconds_cap':30}


def competing_commands(rows, own_pid):
    conflicts = []
    for pid, command in rows:
        if pid == own_pid:
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            c.require('python' not in command.lower() and 'pypy' not in command.lower(),
                      'Cannot classify full Python process command')
            continue
        if not words:
            continue
        if Path(words[0]).name.lower() == 'tmux':
            continue  # A real Python child is independently visible in ps.
        # Full command lines include timeout/env wrappers as well as direct Python.
        python_positions = [i for i, word in enumerate(words)
                            if Path(word).name.lower().startswith(('python', 'pypy'))]
        if not python_positions:
            continue
        lowered = command.lower()
        scripts = [Path(word).name.lower() for word in words if word.lower().endswith('.py')]
        research = '--run' in words and ('codeformer' in lowered or 'cctv_dgp' in lowered or 'forensic' in lowered)
        trainer = any(name.startswith(('train_cctv_dgp_', 'train_codeformer_', 'train_forensic_')) for name in scripts)
        supervisor = any(name.startswith(('supervise_codeformer', 'supervise_cctv_dgp', 'supervise_forensic'))
                         for name in scripts)
        generic_train = any(Path(word).name.lower() in ('train.py', 'training.py') for word in words)
        if research or trainer or supervisor or generic_train:
            conflicts.append({'pid': pid, 'command': command})
    return conflicts


def idle():
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    c.require(not gpu.stdout.strip(), 'GPU occupied; leave every existing workload untouched')
    ps = subprocess.run(['ps', '-eo', 'pid=,args='], check=True, capture_output=True, text=True, timeout=20)
    rows = []
    for line in ps.stdout.splitlines():
        fields = line.strip().split(None, 1)
        c.require(len(fields) == 2 and fields[0].isdigit(), 'Cannot classify full process inventory')
        rows.append((int(fields[0]), fields[1]))
    conflicts = competing_commands(rows, os.getpid())
    c.require(not conflicts, 'Competing research worker/supervisor; leave it untouched: ' + json.dumps(conflicts))
    return {'GPU_compute_processes': [], 'competing_research_commands': [],
            'full_process_inventory': [{'pid': pid, 'command': command} for pid, command in rows],
            'idle_tmux_sessions_are_not_workloads': True, 'other_processes_killed': False}


def _group_alive(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        c.require(False, 'Cannot establish own child process group is terminal')
    return True


def _stop_own_group(process, grace):
    # This function receives only the new Popen(start_new_session=True) child.
    if _group_alive(process.pid):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while _group_alive(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if _group_alive(process.pid):
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        c.require(False, 'Own stage child failed to become terminal')
    deadline = time.monotonic() + 10
    while _group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    c.require(not _group_alive(process.pid), 'Own child descendants remain live; export forbidden')


def stage(root, protocol_sha, flag, seconds, grace, log):
    command = ['timeout', '--signal=TERM', '--kill-after=' + str(grace), str(seconds),
               sys.executable, '-B', '-u', '-m', MODULE, '--root', str(root),
               '--protocol-sha', protocol_sha, flag]
    started, written, capped, timed_out = time.monotonic(), 0, False, False
    with log.open('xb') as output:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        print(json.dumps({'stage': flag, 'child_pid': process.pid, 'external_cap_seconds': seconds,
                          'grace_seconds': grace, 'log': log.relative_to(root).as_posix()}), flush=True)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() - started > seconds + grace + 5:
                        timed_out = True; _stop_own_group(process, 0); break
                    for key, _ in selector.select(timeout=0.25):
                        payload = os.read(key.fileobj.fileno(), 65536)
                        if not payload:
                            selector.unregister(key.fileobj); continue
                        room = max(0, LOG_CAP - written)
                        output.write(payload[:room]); written += min(room, len(payload))
                        if len(payload) > room:
                            capped = True; _stop_own_group(process, grace); break
                    if capped:
                        break
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                timed_out = True; _stop_own_group(process, 0); code = process.returncode
            if _group_alive(process.pid):
                _stop_own_group(process, 0)
        except BaseException:
            _stop_own_group(process, grace)
            raise
        if capped:
            code = 125
        elif timed_out:
            code = 124
        output.flush(); os.fsync(output.fileno())
    c.require(log.stat().st_size <= LOG_CAP and not _group_alive(process.pid), 'Stage terminal/log invariant failed')
    with log.open('rb') as source:
        source.seek(max(0, log.stat().st_size - 16000))
        sys.stdout.write(source.read().decode('utf-8', errors='replace'))
    receipt = {'stage': flag, 'exit_code': code, 'seconds': time.monotonic() - started,
               'external_cap_seconds': seconds, 'grace_seconds': grace, 'child_pid': process.pid,
               'log': log.relative_to(root).as_posix(), 'log_bytes': written, 'log_capped': capped,
               'outer_watchdog_expired': timed_out, 'worker_terminal': True,
               'child_process_group_terminal': True}
    print(json.dumps(receipt), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args(argv)
    root = c.manual_root(args.root)  # Strict Linux/home/tmux path guard before any worker import.
    c.require(bool(os.environ.get('TMUX')), 'Human-opened tmux session required')
    c.require(re.fullmatch('[a-f0-9]{64}', args.protocol_sha), 'Exact frozen protocol hash required')
    c.require(shutil.which('timeout'), 'GNU timeout must be installed')
    c.require(not (root / 'outputs').exists() and not (root / 'normal_tone_v1.lock').exists(),
              'Preserve existing execution/lock evidence; no relaunch')
    initial_idle = idle()
    supervision = root / 'outputs/supervision'
    supervision.mkdir(parents=True)
    preflight = stage(root, args.protocol_sha, '--verify-transfer', 300, 30, supervision / 'preflight.log')
    c.require(preflight['exit_code'] == 0, 'Transfer/dependency preflight failed; no neural run')
    second_idle = idle()
    measured = stage(root, args.protocol_sha, '--run', 1800, 60, supervision / 'diagnostic.log')
    c.write_new_json(supervision / 'run_receipt.json', {'protocol_sha256': args.protocol_sha,
        'manual_tmux': True, 'initial_idle_check': initial_idle, 'pre_run_idle_check': second_idle,
        'stages': [preflight, measured], 'worker_terminal': True,
        'child_process_group_terminal': True, 'committed_training_updates': 0,
        'actual_disposable_Adam_steps': None, 'actual_step_count_source': 'worker completion/failure evidence',
        'maximum_disposable_Adam_steps': 8, 'neural_measurement_complete': measured['exit_code'] == 0,
        'no_automatic_retry': True, 'training_success_not_implied': True, 'model_qualified': False})
    c.write_new_json(supervision / 'export_invocation.json', {'protocol_sha256': args.protocol_sha,
        'worker_terminal': True, 'export_failure_evidence': measured['exit_code'] != 0,
        'external_cap_seconds': 600, 'grace_seconds': 30,
        'closure_boundary': 'Final export.log/finished.json and home export receipt are retained separately'})
    exported = stage(root, args.protocol_sha, '--export', 600, 30, supervision / 'export.log')
    c.write_new_json(supervision / 'finished.json', {'complete': True,
        'stages': [preflight, measured, exported], 'worker_terminal': True,
        'committed_training_updates': 0, 'actual_disposable_Adam_steps': None,
        'actual_step_count_source': 'worker completion/failure evidence',
        'measurement_or_export_completion_is_not_model_acceptance': True})
    if exported['exit_code']==0:
        print(json.dumps(final_companion(root,[preflight,measured,exported])),flush=True)
    return 0 if measured['exit_code'] == exported['exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

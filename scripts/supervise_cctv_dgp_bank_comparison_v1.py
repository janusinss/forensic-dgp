"""Manual Linux/tmux execution and export with independent process deadlines."""
import argparse
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cctv_dgp_bank_comparison_v1_contract import NAME, BUDGETS, read, write, verify


def child(root, pin, exporting=False):
    stamp = time.monotonic(); offset = 0; timeout = False
    log_name = 'export.log' if exporting else 'trainer.log'
    seconds = BUDGETS['export_external_seconds' if exporting else 'external_seconds']
    cmd = [sys.executable, '-B', '-u', str(root/'scripts/cctv_dgp_bank_comparison_v1_vm.py'),
           '--root', str(root), '--protocol-sha', pin]
    if exporting:
        cmd.append('--export')
    path = root/log_name
    with path.open('xb') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print({'child_pid': proc.pid, 'log': log_name, 'external_cap_seconds': seconds}, flush=True)
        try:
            while proc.poll() is None:
                if time.monotonic()-stamp >= seconds:
                    timeout = True; os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=BUDGETS['kill_grace_seconds'])
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
                    break
                time.sleep(1)
                with path.open('rb') as incoming:
                    incoming.seek(offset); chunk = incoming.read(); offset = incoming.tell()
                if chunk:
                    print(chunk.decode('utf-8', errors='replace'), end='', flush=True)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=BUDGETS['kill_grace_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
            raise
    with path.open('rb') as incoming:
        incoming.seek(offset); chunk = incoming.read()
    if chunk:
        print(chunk.decode('utf-8', errors='replace'), end='', flush=True)
    code = 124 if timeout else proc.returncode
    print({'log': log_name, 'exit_code': code, 'timeout': timeout}, flush=True)
    return code, time.monotonic()-stamp


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    a = parser.parse_args(); root = a.root.resolve()
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root == (Path.home()/'forensic-dgp'/NAME).resolve() and os.environ.get('TMUX')
    assert Path(sys.executable).absolute() == Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python'
    for name in ['outputs', 'trainer.log', 'trainer_exit_code.txt', 'supervisor_receipt.json',
                 'export_manifest.json', 'export.log', 'export_exit_code.txt']:
        assert not (root/name).exists(), 'Immutable run; preserve prior/partial files: ' + name
    verify(root, a.protocol_sha)
    code, seconds = child(root, a.protocol_sha)
    (root/'trainer_exit_code.txt').write_text(str(code)+'\n', encoding='ascii')
    out = root/'outputs/bank_comparison_v1'
    evidence, failure = out/'results.json', out/'failure.json'
    if not evidence.exists() and not failure.exists():
        out.mkdir(parents=True, exist_ok=True)
        write(failure, {'type': 'supervisor_retained_child_exit', 'trainer_exit_code': code,
            'exact_completed_updates_unknown': True, 'progress': {'counts': {'optimizer_updates': None}},
            'training_success_not_implied': True})
    updates = read(evidence)['counts']['optimizer_updates'] if evidence.exists() else read(failure)['progress']['counts']['optimizer_updates']
    write(root/'supervisor_receipt.json', {'complete': True, 'protocol_sha256': a.protocol_sha,
        'seconds': seconds, 'external_cap_seconds': BUDGETS['external_seconds'],
        'kill_grace_seconds': BUDGETS['kill_grace_seconds'], 'trainer_exit_code': code,
        'optimizer_updates': updates, 'quality_pass_not_implied': True})
    export_code, _ = child(root, a.protocol_sha, exporting=True)
    (root/'export_exit_code.txt').write_text(str(export_code)+'\n', encoding='ascii')
    raise SystemExit(export_code if export_code else code)


if __name__ == '__main__':
    main()

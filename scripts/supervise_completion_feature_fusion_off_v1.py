"""One bounded local frozen comparison; no automatic rerun or VM work."""
from pathlib import Path
import subprocess
import sys
import time
from completion_feature_fusion_off_v1_common import ROOT, OUT, sha, read, write


def main():
    assert Path(sys.executable).resolve() == (ROOT/'venv/Scripts/python.exe').resolve()
    p = read(OUT/'protocol.json'); assert not (OUT/'external_receipt.json').exists() and not (OUT/'inference.log').exists()
    assert p['cap_seconds'] == 600 and p['external_timeout_seconds'] == 630
    started = time.monotonic(); timed_out = False
    with (OUT/'inference.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT/'scripts/run_completion_feature_fusion_off_v1.py')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(OUT/'process.json', {'child_pid': child.pid, 'parent_pid': __import__('os').getpid(), 'protocol_sha256': sha(OUT/'protocol.json'), 'external_timeout_seconds': 630})
        print({'child_pid': child.pid, 'log': str(OUT/'inference.log')}, flush=True)
        try: code = child.wait(timeout=630)
        except subprocess.TimeoutExpired:
            timed_out = True; child.kill(); code = child.wait(timeout=30)
    result = {'complete': code == 0 and not timed_out, 'worker_exit_code': code, 'timeout': timed_out, 'external_seconds': time.monotonic()-started,
              'protocol_sha256': sha(OUT/'protocol.json'), 'supervisor_sha256': sha(Path(__file__)), 'log_sha256': sha(OUT/'inference.log'),
              'automatic_retry_permitted': False, 'VM_calls': 0, 'training': False}
    write(OUT/'external_receipt.json', result); print(result, flush=True)
    if not result['complete']: raise SystemExit(1)


if __name__ == '__main__': main()

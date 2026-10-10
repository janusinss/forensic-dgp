"""One bounded local CPU inference process; no retries, training or VM work."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_conditioning_union_v1'


def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert not (OUT / 'external_receipt.json').exists() and not (OUT / 'inference.log').exists()
    protocol = (OUT / 'protocol.json').read_bytes()
    p = json.loads(protocol)
    assert p['cap_seconds'] == 600 and p['external_timeout_seconds'] == 630
    assert p['sources_sha256']['scripts/supervise_completion_conditioning_union_v1.py'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started, timed_out = time.monotonic(), False
    with (OUT / 'inference.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / 'scripts/run_completion_conditioning_union_v1.py')],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / 'process.json', {'child_pid': child.pid, 'parent_pid': __import__('os').getpid(),
              'worker': 'scripts/run_completion_conditioning_union_v1.py',
              'protocol_sha256': hashlib.sha256(protocol).hexdigest(), 'external_cap_seconds': 630})
        print(json.dumps({'child_pid': child.pid, 'log': str(OUT / 'inference.log')}), flush=True)
        try:
            code = child.wait(timeout=630)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait(timeout=30)
    receipt = {'complete': code == 0 and not timed_out, 'worker_exit_code': code, 'timeout': timed_out,
               'external_seconds': time.monotonic() - started, 'external_cap_seconds': 630,
               'protocol_sha256': hashlib.sha256(protocol).hexdigest(),
               'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'log_sha256': hashlib.sha256((OUT / 'inference.log').read_bytes()).hexdigest(),
               'training': False, 'VM_writes': 0, 'automatic_retry_permitted': False}
    write(OUT / 'external_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
    if not receipt['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

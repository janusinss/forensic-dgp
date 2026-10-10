"""One frozen return-audit execution; preserve logs and all failure evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_return_audit'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    assert not OUT.exists(), 'Retain the previous return-audit attempt'
    OUT.mkdir()
    start = time.monotonic()
    archive = ROOT / 'outputs/cctv-dgp-multiscale-calibration-v1-results.tar.gz'
    export = ROOT / 'outputs/cctv-dgp-multiscale-calibration-v1-export.json'
    checker = ROOT / 'scripts/audit_cctv_dgp_multiscale_calibration_return_v1.py'
    receipt = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit.json'
    assert not receipt.exists()
    e = read(export)
    digest = sha(archive)
    assert e['complete'] and e['archive_sha256'] == digest and e['bytes'] == archive.stat().st_size
    assert Path(str(archive)+'.sha256').read_text().split() == [digest, archive.name]
    protected = read(ROOT / 'outputs/cctv_dgp_multiscale_archive_cleanup_v1/local_protected_sha256.json')
    assert sha(checker) == protected[checker.relative_to(ROOT).as_posix()]
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, name
    write(OUT / 'protected_before.json', protected)
    write(OUT / 'plan.json', dict(complete=True, archive_sha256=digest, bytes=e['bytes'], checker_sha256=sha(checker),
          protocol_sha256=sha(ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1/protocol.json'),
          export_sha256=sha(export), protected_bindings=len(protected), checker_unchanged=True,
          driver_sha256=sha(Path(__file__)), local_gradient_queries=0, local_optimizer_updates=0,
          timeout_seconds=1260, model_qualification=False, goal_complete=False))
    expired = False
    with (OUT / 'stdout.log').open('xb') as stdout, (OUT / 'stderr.log').open('xb') as stderr:
        try:
            result = subprocess.run([sys.executable, '-B', '-u', str(checker), '--archive', str(archive),
                '--expected-sha', digest, '--export', str(export), '--receipt', str(receipt)],
                cwd=ROOT, stdout=stdout, stderr=stderr, timeout=1260)
            code = result.returncode
        except subprocess.TimeoutExpired:
            expired = True
            code = None
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, 'Protected binding changed: ' + name
    complete = code == 0 and receipt.is_file() and read(receipt)['complete']
    write(OUT / 'execution.json', dict(complete=complete, exit_code=code, timeout=expired,
          stdout_sha256=sha(OUT / 'stdout.log'), stderr_sha256=sha(OUT / 'stderr.log'),
          protected_bindings_unchanged=len(protected), checker_sha256=sha(checker),
          receipt_sha256=sha(receipt) if receipt.exists() else None, seconds=time.monotonic()-start,
          local_gradient_queries=0, local_optimizer_updates=0, model_qualification=False, goal_complete=False))
    print(dict(complete=complete, exit_code=code, timeout=expired, local_gradient_queries=0, local_optimizer_updates=0), flush=True)
    assert complete, 'Preserved audit failure; do not change returned files or quality gates'


if __name__ == '__main__':
    main()

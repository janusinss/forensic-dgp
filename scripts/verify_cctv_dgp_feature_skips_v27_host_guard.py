"""Actual Windows transfer checks and rejected neural/install execution."""
import json
from pathlib import Path
import subprocess
import sys
import time

from verify_cctv_dgp_feature_skips_v27 import ROOT, NEW, OUT, PIN, sha


def main():
    assert sys.platform == 'win32', 'This audit records the actual Windows guard'
    destination = OUT / 'windows_execution_guard.json'
    assert not destination.exists()
    before = {path.relative_to(NEW).as_posix(): sha(path) for path in NEW.rglob('*') if path.is_file()}
    started = time.monotonic(); results = []
    jobs = [
        ('installer_transfer', 'scripts/install_v27.py', '--verify-transfer', 0),
        ('full_asset_transfer', 'scripts/cctv_dgp_feature_skips_v27_vm.py', '--verify-transfer', 0),
        ('training_rejected', 'scripts/cctv_dgp_feature_skips_v27_vm.py', '--run', 1),
        ('installation_rejected', 'scripts/install_v27.py', '--install', 1),
    ]
    for name, worker, mode, status in jobs:
        result = subprocess.run([sys.executable, '-B', str(NEW / worker), '--root', str(NEW), '--protocol-sha', PIN, mode],
            capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == status, (name, result.stdout, result.stderr)
        if status:
            assert 'Existing Linux VM only' in result.stderr
        else:
            assert json.loads(result.stdout)['complete']
        results.append({'name': name, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    after = {path.relative_to(NEW).as_posix(): sha(path) for path in NEW.rglob('*') if path.is_file()}
    assert before == after
    assert not any((NEW / name).exists() for name in ['outputs', 'trainer.log', 'installation_receipt.json', 'installation_failure.json'])
    record = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'protocol_sha256': PIN, 'platform': sys.platform, 'commands': results,
        'bundle_files_unchanged': len(before), 'no_outputs_or_install_records_created': True,
        'neural_imports_and_work_rejected_by_host_guard': True,
        'local_model_or_gradient_calls': 0, 'optimizer_updates': 0,
        'VM_actions': False, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - started}
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'commands'}, indent=2))


if __name__ == '__main__':
    main()

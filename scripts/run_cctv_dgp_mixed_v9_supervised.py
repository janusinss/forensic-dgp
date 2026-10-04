"""VM-only bounded V9 trainer, independent audit and exact return export."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def run(root, expected_protocol_sha):
    sys.path.insert(0, str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)
    from cctv_dgp_mixed_v9 import read, sha, verify, write
    p = verify(root, expected_protocol_sha)
    import torch, torchvision
    assert (root.parent / 'cctv_dgp_vm_bundle/cuda_runtime_before.txt').read_text().splitlines() == [torch.__version__, torchvision.__version__]
    assert not (root / 'supervisor_execution_v9.json').exists(), 'Preserve previous execution'
    started = time.monotonic(); deadline = started + 3600
    write(root / 'supervisor_execution_v9.json', {'protocol_sha256': expected_protocol_sha,
        'source_sha256': sha(Path(__file__)), 'auditor_sha256': sha(root / 'scripts/audit_cctv_dgp_mixed_v9.py'),
        'pid': os.getpid(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'started_unix_seconds': time.time(), 'trainer_cap_seconds': 2400,
        'supervisor_cap_seconds': 3600, 'expected_updates': 7820, 'production_promoted': False})

    def call(args, name):
        with (root / name).open('x', encoding='utf-8') as log:
            child = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'child_pid': child.pid, 'command': args, 'log': name}), flush=True)
            try:
                code = child.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise TimeoutError('V9 supervisor exceeded3600 seconds')
        if code:
            raise subprocess.CalledProcessError(code, args)
        print((root / name).read_text()[-2200:], flush=True)

    def export(name, success):
        archive = root / name
        paths = ['mixed_protocol_v9.json', 'mixed_protocol_v9.sha256', 'cctv_dgp_mixed_v9.py', 'scripts',
            'outputs/cctv_dgp_mixed_v9', 'mixed_training_v9.log', 'mixed_audit_v9.log',
            'supervisor_execution_v9.json', 'supervisor_launch_v9.json', 'training_audit_completion_v9.json', 'supervisor_failure_v9.json']
        with tarfile.open(archive, 'x:gz', compresslevel=5) as stream:
            for name in paths:
                if success and time.monotonic() > deadline:
                    raise TimeoutError('V9 export exceeded3600-second execution cap')
                path = root / name
                if path.exists():
                    stream.add(path, arcname=name)
        Path(str(archive) + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
        return archive

    try:
        call([sys.executable, '-u', str(root / 'scripts/train_cctv_dgp_mixed_v9.py'), '--root', str(root),
            '--expected-protocol-sha', expected_protocol_sha], 'mixed_training_v9.log')
        out = root / 'outputs/cctv_dgp_mixed_v9'
        call([sys.executable, '-u', str(root / 'scripts/audit_cctv_dgp_mixed_v9.py'), '--root', str(root),
            '--expected-protocol-sha', expected_protocol_sha, '--results', str(out),
            '--receipt', str(out / 'independent_audit_vm.json')], 'mixed_audit_v9.log')
        result = read(out / 'results.json')
        write(root / 'training_audit_completion_v9.json', {'complete': True, 'optimizer_updates': result['total_optimizer_updates'],
            'selected_epoch': result['selected_epoch'], 'seconds': time.monotonic() - started, 'production_promoted': False})
        archive = export('cctv-dgp-mixed-v9-results.tar.gz', True)
        assert time.monotonic() <= deadline
        write(root / 'supervisor_completion_v9.json', {'complete': True, 'protocol_sha256': expected_protocol_sha,
            'archive_sha256': sha(archive), 'bytes': archive.stat().st_size, 'seconds': time.monotonic() - started,
            'optimizer_updates': 7820, 'selected_epoch': result['selected_epoch'], 'production_promoted': False})
        print((root / 'supervisor_completion_v9.json').read_text(), flush=True)
    except Exception as error:
        if not (root / 'supervisor_failure_v9.json').exists():
            write(root / 'supervisor_failure_v9.json', {'complete': False, 'error_type': type(error).__name__,
                'error': str(error), 'seconds': time.monotonic() - started, 'resume_permitted': False})
        try:
            archive = export('cctv-dgp-mixed-v9-failure.tar.gz', False)
            write(root / 'failure_export_v9.json', {'complete': True, 'archive_sha256': sha(archive), 'bytes': archive.stat().st_size})
        except Exception as export_error:
            write(root / 'failure_export_error_v9.json', {'error': str(export_error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha', required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.expected_protocol_sha)

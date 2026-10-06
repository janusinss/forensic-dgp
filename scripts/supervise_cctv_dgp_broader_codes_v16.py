"""Finite L4 train/audit/export supervision; preserve partial evidence on failure."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def supervise(root, parent, mixed, baseline, pin):
    sys.path.insert(0, str(parent)); sys.path.insert(0, str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)
    import cctv_dgp_broader_codes_v16 as v
    import torch, torchvision
    start = time.monotonic(); p = v.verify(root, parent, mixed, baseline, pin)
    deadline = start + p['design']['supervisor_cap_seconds']
    v.require((root.parent / 'cctv_dgp_vm_bundle/cuda_runtime_before.txt').read_text().splitlines() ==
              [torch.__version__, torchvision.__version__], 'Existing CUDA runtime changed')
    v.require(not (root / 'supervisor_execution.json').exists(), 'No repeat/resume')
    v.write(root / 'supervisor_execution.json', {'complete': True, 'protocol_sha256': pin, 'pid': os.getpid(),
        'started_unix': time.time(), 'budget_seconds': 2400, 'torch': torch.__version__, 'torchvision': torchvision.__version__})

    def call(script, extra, logfile, cap):
        args = [sys.executable, '-u', str(root / 'scripts' / script), '--root', str(root),
            '--parent', str(parent), '--mixed', str(mixed), '--baseline', str(baseline), '--expected-sha', pin] + extra
        v.require(time.monotonic() < deadline, 'Overall cap exhausted before child')
        with (root / logfile).open('x') as log:
            child = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print({'child_pid': child.pid, 'log': logfile}, flush=True)
            try:
                code = child.wait(timeout=min(cap, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGINT)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise TimeoutError('Finite V16 child/overall cap exceeded')
        print((root / logfile).read_text()[-1600:], flush=True)
        if code: raise subprocess.CalledProcessError(code, args)

    def export(success):
        archive = root / ('cctv-dgp-broader-codes-v16-results.tar.gz' if success else 'cctv-dgp-broader-codes-v16-failure.tar.gz')
        names = sorted(p['assets_sha256']) + [v.PLAN, 'protocol.sha256', 'trainer.log', 'audit.log',
            'supervisor_launch.json', 'supervisor_execution.json', 'supervisor_failure.json']
        result = root / 'outputs/broader_codes_v16'
        if result.exists():
            names += [str(path.relative_to(root)).replace('\\', '/') for path in result.rglob('*')
                if path.is_file() and 'cache' not in path.relative_to(result).parts]
        with tarfile.open(archive, 'x:gz', compresslevel=3) as stream:
            for name in sorted(set(names)):
                if success: v.require(time.monotonic() <= deadline, 'V16 export cap exceeded')
                path = v.safe(root, name)
                if path.is_file(): stream.add(path, arcname=name, recursive=False)
        checksum = v.sha(archive)
        Path(str(archive) + '.sha256').write_text(checksum + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
        return archive, checksum

    try:
        call('train_cctv_dgp_broader_codes_v16.py', [], 'trainer.log', 2130)
        out = root / 'outputs/broader_codes_v16'
        call('audit_cctv_dgp_broader_codes_v16.py', ['--results', str(out), '--receipt', str(out / 'independent_audit_vm.json')],
             'audit.log', p['design']['audit_cap_seconds'])
        archive, checksum = export(True)
        r = v.read(out / 'results.json')
        v.require(time.monotonic() <= deadline, 'V16 overall cap exceeded')
        v.write(root / 'supervisor_completion.json', {'complete': True, 'protocol_sha256': pin,
            'results_sha256': v.sha(out / 'results.json'), 'archive_sha256': checksum, 'bytes': archive.stat().st_size,
            'seconds': time.monotonic() - start, 'optimizer_updates': r['optimizer_updates'],
            'backward_calls': r['backward_calls'], 'production_promoted': False})
        print((root / 'supervisor_completion.json').read_text(), flush=True)
    except BaseException as error:
        v.write(root / 'supervisor_failure.json', {'complete': False, 'protocol_sha256': pin,
            'error_type': type(error).__name__, 'error': str(error), 'seconds': time.monotonic() - start, 'resume_permitted': False})
        archive, checksum = export(False)
        v.write(root / 'failure_export.json', {'complete': True, 'archive_sha256': checksum, 'bytes': archive.stat().st_size})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'parent', 'mixed', 'baseline']: parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True); a = parser.parse_args()
    supervise(a.root.resolve(), a.parent.resolve(), a.mixed.resolve(), a.baseline.resolve(), a.expected_sha)

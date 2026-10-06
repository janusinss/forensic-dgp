"""Bound inference/audit/export; retain partial evidence and prohibit automatic resume."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def supervise(root, parent, r2, mixed, baseline, pin):
    sys.path.insert(0, str(parent))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)
    sys.path.insert(0, str(root))
    import cctv_dgp_input_selection_v19 as q
    start = time.monotonic()
    p, _, _ = q.verify(root, parent, r2, mixed, baseline, pin)
    v = q
    import torch, torchvision
    v.require((root.parent / 'cctv_dgp_vm_bundle/cuda_runtime_before.txt').read_text().splitlines()
              == [torch.__version__, torchvision.__version__], 'Existing VM runtime changed; do not reinstall')
    deadline = start + p['design']['supervisor_cap_seconds']
    v.require(not (root / 'supervisor_execution.json').exists(), 'No V19 repeat/resume')
    v.write(root / 'supervisor_execution.json', {'complete': True, 'protocol_sha256': pin,
        'pid': os.getpid(), 'started_unix': time.time(), 'budget_seconds': 1800,
        'torch': torch.__version__, 'torchvision': torchvision.__version__})

    def call(script, extra, log_name, cap):
        args = [sys.executable, '-B', '-u', str(root / 'scripts' / script), '--root', str(root),
            '--parent', str(parent), '--r2', str(r2), '--mixed', str(mixed), '--baseline', str(baseline),
            '--expected-sha', pin] + extra
        v.require(time.monotonic() < deadline, 'Overall cap exhausted before child')
        with (root / log_name).open('x', encoding='utf-8') as log:
            child = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print({'child_pid': child.pid, 'log': log_name}, flush=True)
            try:
                code = child.wait(timeout=min(cap, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGINT)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                raise TimeoutError('Finite V19 child/overall limit exceeded')
        print((root / log_name).read_text()[-2000:], flush=True)
        if code: raise subprocess.CalledProcessError(code, args)

    def export(success):
        cap = min(deadline, time.monotonic() + 180)
        archive = root / ('cctv-dgp-input-selection-v19-results.tar.gz' if success else 'cctv-dgp-input-selection-v19-failure.tar.gz')
        names = set(p['assets_sha256']) | {q.PLAN, 'protocol.sha256', 'inference.log', 'audit.log',
            'supervisor_launch.json', 'bootstrap_preflight.json', 'supervisor_execution.json', 'supervisor_failure.json'}
        out = root / 'outputs/input_selection_v19'
        if out.exists(): names |= {path.relative_to(root).as_posix() for path in out.rglob('*') if path.is_file()}
        with tarfile.open(archive, 'x:gz', compresslevel=3) as stream:
            for name in sorted(names):
                v.require(time.monotonic() <= cap, 'Finite180s export/overall cap exceeded; preserve partial archive')
                path = v.safe(root, name)
                if path.is_file(): stream.add(path, arcname=name, recursive=False)
        checksum = v.sha(archive)
        v.require(time.monotonic() <= cap, 'Finite export/hash cap exceeded; preserve archive')
        with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
            stream.write(checksum + '  ' + archive.name + '\n')
        return archive, checksum

    try:
        call('run_cctv_dgp_input_selection_v19.py', [], 'inference.log', 1200)
        out = root / 'outputs/input_selection_v19'
        call('audit_cctv_dgp_input_selection_v19.py', ['--results', str(out), '--receipt', str(out / 'independent_audit_vm.json')], 'audit.log', 300)
        archive, checksum = export(True)
        v.require(time.monotonic() <= deadline, 'Overall1800s cap exceeded')
        v.write(root / 'supervisor_completion.json', {'complete': True, 'protocol_sha256': pin,
            'results_sha256': v.sha(out / 'results.json'), 'archive_sha256': checksum,
            'bytes': archive.stat().st_size, 'seconds': time.monotonic() - start,
            'optimizer_updates': 0, 'backward_calls': 0, 'validation_cases': 520,
            'scientific_guard_passed': v.read(out / 'results.json')['preservation']['automatic_v19']['qualified_for_separate_generalization_protocol'],
            'production_promoted': False})
        print((root / 'supervisor_completion.json').read_text(), flush=True)
    except BaseException as error:
        v.write(root / 'supervisor_failure.json', {'complete': False, 'protocol_sha256': pin,
            'error_type': type(error).__name__, 'error': str(error), 'seconds': time.monotonic() - start,
            'resume_permitted': False})
        try:
            archive, checksum = export(False)
            v.write(root / 'failure_export.json', {'complete': True, 'archive_sha256': checksum,
                'bytes': archive.stat().st_size, 'protocol_sha256': pin})
            print((root / 'failure_export.json').read_text(), flush=True)
        except BaseException as export_error:
            v.write(root / 'failure_export_incomplete.json', {'complete': False, 'error_type': type(export_error).__name__,
                'error': str(export_error), 'preserve_partial_archive': True, 'resume_permitted': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'parent', 'r2', 'mixed', 'baseline']: parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    a = parser.parse_args()
    supervise(a.root.resolve(), a.parent.resolve(), a.r2.resolve(), a.mixed.resolve(), a.baseline.resolve(), a.expected_sha)

"""Safely verify/extract a fresh return, preserving failures and running no neural code."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
sys.path.insert(0, str(BUNDLE))
import cctv_dgp_broader_codes_v16 as v


def collect(archive, completion, destination):
    v.require(archive.resolve().is_relative_to(ROOT / 'outputs') and
              destination.resolve().is_relative_to(ROOT / 'outputs') and not destination.exists(),
              'Preserve previous/partial return; require outputs paths')
    bundle = BUNDLE; pin = v.sha(bundle / v.PLAN)
    terminal = v.read(completion)
    v.require(terminal['complete'] and terminal.get('protocol_sha256', pin) == pin and
              v.sha(archive) == terminal['archive_sha256'] and archive.stat().st_size == terminal['bytes'],
              'Return terminal/hash/size differs')
    v.require(Path(str(archive) + '.sha256').read_text().split() == [terminal['archive_sha256'], archive.name],
              'Return checksum sidecar differs')
    with tarfile.open(archive, 'r:gz') as stream:
        members = v.archive_members(stream, destination, 4 * 1024**3)
        destination.mkdir(); stream.extractall(destination, members=members, filter='data')
    v.require(v.sha(destination / v.PLAN) == pin, 'Returned protocol differs')
    for name, expected in v.read(bundle / v.PLAN)['assets_sha256'].items():
        v.require(v.sha(v.safe(destination, name)) == expected, 'Returned source/evidence differs:' + name)
    out = destination / 'outputs/broader_codes_v16_r2'
    if 'results_sha256' not in terminal:
        spec = importlib.util.spec_from_file_location('r2_failure_audit', BUNDLE / 'scripts/audit_cctv_dgp_broader_codes_v16.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        audit_failure = module.audit_failure
        audit_failure(bundle, ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2',
            ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2',
            ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15', pin, out,
            destination / 'supervisor_failure.json', destination / 'local_failure_import.json')
        print({'failure_preserved': True, 'resume_permitted': False}); return
    v.require(terminal['protocol_sha256'] == pin and terminal['optimizer_updates'] == 3128 and
              terminal['backward_calls'] == 3129 and v.sha(out / 'results.json') == terminal['results_sha256'],
              'Successful execution count/results differs')
    subprocess.run([sys.executable, '-X', 'utf8', '-u', str(BUNDLE / 'scripts/audit_cctv_dgp_broader_codes_v16.py'),
        '--root', str(bundle), '--parent', str(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'),
        '--mixed', str(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'), '--baseline',
        str(ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'),
        '--expected-sha', pin, '--results', str(out), '--receipt', str(destination / 'local_independent_audit.json')],
        cwd=ROOT, check=True, timeout=240)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['archive', 'completion', 'extract-to']: parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args(); collect(a.archive.resolve(), a.completion.resolve(), a.extract_to.resolve())

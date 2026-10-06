"""Verify user-downloaded return, preserve original success/failure and audit locally."""
import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_structure_vm_v18'


def load_q():
    sys.path.insert(0, str(BUNDLE))
    import cctv_dgp_structure_v18 as q
    v = q.environment(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2', ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2')
    return q, v


def collect(archive, completion, destination):
    q, v = load_q()
    start = time.monotonic()
    pin = v.sha(BUNDLE / q.PLAN)
    terminal = v.read(completion)
    p = v.read(BUNDLE / q.PLAN)
    v.require(not destination.exists() and destination.resolve().is_relative_to((ROOT / 'outputs').resolve()),
              'Fresh return within workspace outputs required; preserve existing/partial return')
    v.require(terminal['complete'] and v.sha(archive) == terminal['archive_sha256']
              and archive.stat().st_size == terminal['bytes']
              and Path(str(archive) + '.sha256').read_text().split() == [terminal['archive_sha256'], archive.name],
              'Downloaded hash/size/sidecar/receipt differ')
    v.require(terminal.get('protocol_sha256') == pin, 'Return receipt protocol differs')
    v.require(archive.name in ['cctv-dgp-structure-v18-results.tar.gz', 'cctv-dgp-structure-v18-failure.tar.gz'],
              'Expected exact V18 success/failure archive name')
    original_success = archive.name == 'cctv-dgp-structure-v18-results.tar.gz'
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        names = set()
        v.require(sum(item.size for item in members) <= 2 * 1024**3, 'Return exceeds2GiB uncompressed')
        for member in members:
            name, part = member.name, PurePosixPath(member.name)
            v.require(member.isfile() and bool(name) and not part.is_absolute() and '..' not in part.parts
                and ':' not in name and '\\' not in name and str(part) == name and name not in names
                and (destination / name).resolve().is_relative_to(destination.resolve()), 'Unsafe/duplicate returned member')
            names.add(name)
        destination.mkdir(parents=True)
        stream.extractall(destination, members=members)
    for name, digest in p['assets_sha256'].items():
        v.require(v.sha(v.safe(destination, name)) == digest == v.sha(v.safe(BUNDLE, name)), 'Frozen returned source differs:' + name)
    v.require(v.sha(destination / q.PLAN) == pin, 'Returned protocol differs')
    v.write(destination / 'downloaded_export_receipt.json', terminal)
    out = destination / 'outputs/structure_v18'
    full = (out / 'results.json').is_file() and v.read(out / 'results.json').get('complete') is True
    if original_success:
        v.require(full and terminal['optimizer_updates'] == 600
                  and terminal['results_sha256'] == v.sha(out / 'results.json')
                  and not (destination / 'supervisor_failure.json').exists(), 'Success receipt/results differ')
    else:
        v.require((destination / 'supervisor_failure.json').is_file(), 'Original supervisor failure required')
    if full:
        args = [sys.executable, '-B', '-u', str(BUNDLE / 'scripts/audit_cctv_dgp_structure_v18.py'),
            '--root', str(BUNDLE), '--parent', str(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'),
            '--r2', str(ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'),
            '--mixed', str(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'),
            '--baseline', str(ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'),
            '--expected-sha', pin, '--results', str(out), '--receipt', str(destination / 'local_independent_audit.json'), '--replay-head']
        with (destination / 'local_audit.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=330)
        print((destination / 'local_audit.log').read_text()[-2400:], flush=True)
        v.require(result.returncode == 0, 'Returned full audit failed; preserve files/log and do not retrain')
    else:
        q.verify(BUNDLE, ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2', ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2',
                 ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2', ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15', pin)
        records = [json.loads(line) for line in (out / 'trace.jsonl').read_text().splitlines()] if (out / 'trace.jsonl').is_file() else []
        scope = q.verify_trace(p, records, complete=False)
        v.require(not original_success and (destination / 'supervisor_failure.json').is_file(), 'Failure provenance required')
        v.write(destination / 'local_partial_failure_audit.json', {'complete': True, 'success': False,
            'protocol_sha256': pin, 'scope': scope, 'failure': v.read(destination / 'supervisor_failure.json'),
            'neural_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
            'limitation': 'Partial trace/source/transfer audit only; trained outputs and CUDA gradients are not independently verified.'})
    v.write(destination / 'local_import_and_audit.json', {'complete': True, 'original_supervisor_success': original_success,
        'full_output_audit': full, 'protocol_sha256': pin, 'archive_sha256': terminal['archive_sha256'],
        'archive_bytes': terminal['bytes'], 'source_assets': len(p['assets_sha256']),
        'seconds': time.monotonic() - start, 'training_restarted': False, 'production_promoted': False})
    print({'complete': True, 'original_supervisor_success': original_success, 'full_output_audit': full,
           'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['archive', 'completion', 'extract-to']: parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    collect(a.archive.resolve(), a.completion.resolve(), a.extract_to.resolve())

"""Import user-downloaded V19 success/failure; finite independent CPU audit."""
import argparse
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19_r2'


def collect(archive, completion, destination):
    sys.path.insert(0, str(BUNDLE))
    import cctv_dgp_input_selection_v19_r2 as q
    start = time.monotonic()
    terminal = q.read(completion); p = q.read(BUNDLE / q.PLAN); pin = q.sha(BUNDLE / q.PLAN)
    q.require(not destination.exists() and destination.resolve().is_relative_to((ROOT / 'outputs').resolve()),
              'Fresh workspace return required; preserve existing/partial return')
    q.require(terminal['complete'] and terminal['protocol_sha256'] == pin
              and q.sha(archive) == terminal['archive_sha256'] and archive.stat().st_size == terminal['bytes']
              and Path(str(archive) + '.sha256').read_text().split() == [terminal['archive_sha256'], archive.name], 'Transfer/receipt differs')
    q.require(archive.name in ['cctv-dgp-input-selection-v19-r2-results.tar.gz', 'cctv-dgp-input-selection-v19-r2-failure.tar.gz'], 'Exact V19 archive required')
    success = archive.name == 'cctv-dgp-input-selection-v19-r2-results.tar.gz'
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers(); names = set()
        q.require(sum(member.size for member in members) <= 2 * 1024**3, 'Return exceeds2GiB uncompressed')
        for member in members:
            part = PurePosixPath(member.name)
            q.require(member.isfile() and not part.is_absolute() and '..' not in part.parts
                      and str(part) == member.name and member.name not in names, 'Unsafe/duplicate returned member')
            q.safe(destination, member.name); names.add(member.name)
        destination.mkdir(parents=True); stream.extractall(destination, members=members, filter='data')
    for name, digest in p['assets_sha256'].items():
        q.require(q.sha(q.safe(destination, name)) == digest == q.sha(q.safe(BUNDLE, name)), 'Original/returned source differs:' + name)
    q.require(q.sha(destination / q.PLAN) == pin, 'Returned protocol differs')
    q.write(destination / 'downloaded_export_receipt.json', terminal)
    out = destination / 'outputs/input_selection_v19_r2'
    full = (out / 'results.json').is_file() and q.read(out / 'results.json').get('complete') is True
    if success:
        q.require(full and terminal['optimizer_updates'] == terminal['backward_calls'] == 0
                  and terminal['validation_cases'] == 520 and terminal['results_sha256'] == q.sha(out / 'results.json')
                  and not (destination / 'supervisor_failure.json').exists(), 'Success receipt/result differs')
    else: q.require((destination / 'supervisor_failure.json').is_file(), 'Original execution/audit failure required')
    if full:
        args = [sys.executable, '-B', '-X', 'utf8', '-u', str(BUNDLE / 'scripts/audit_cctv_dgp_input_selection_v19_r2.py'),
            '--root', str(BUNDLE), '--parent', str(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'),
            '--r2', str(ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'), '--mixed', str(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'),
            '--baseline', str(ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'),
            '--expected-sha', pin, '--results', str(out), '--receipt', str(destination / 'local_full_audit.json'), '--replay-head']
        with (destination / 'local_audit.log').open('x', encoding='utf-8') as stream:
            result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=330)
        print((destination / 'local_audit.log').read_text()[-2500:], flush=True)
        q.require(result.returncode == 0, 'Full local audit failed; preserve original evidence and do not rerun VM')
    else:
        q.verify(BUNDLE, ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2', ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2',
                 ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2', ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15', pin)
        q.write(destination / 'partial_failure_audit.json', {'complete': True, 'full_output_audit': False,
            'failure': q.read(destination / 'supervisor_failure.json'), 'scope': 'Source/protocol/transfer only; no full outputs or replay proof',
            'neural_forwards': 0, 'optimizer_updates': 0, 'backward_calls': 0})
    q.write(destination / 'local_import_and_audit.json', {'complete': True, 'original_supervisor_success': success,
        'full_output_audit': full, 'archive_sha256': terminal['archive_sha256'], 'bytes': terminal['bytes'],
        'protocol_sha256': pin, 'seconds': time.monotonic() - start, 'production_promoted': False,
        'scientific_quality_requires_review': True, 'VM_repeated': False})
    print({'complete': True, 'full_output_audit': full, 'production_promoted': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['archive', 'completion', 'extract-to']: parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args(); collect(a.archive.resolve(), a.completion.resolve(), a.extract_to.resolve())

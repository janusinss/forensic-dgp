"""Preserve the failed audit; align only its archive prefix with the pinned exporter."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_audit_correction_r2'
ORIGINAL_SHA = '9f95deaae65af44ee4afef6c1fa1623318611d9e6cfc989b8412008d8e134121'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    assert not OUT.exists()
    old_path = ROOT / 'scripts/audit_cctv_dgp_group_guard_probe_v35_r1_return.py'
    assert sha(old_path) == ORIGINAL_SHA
    run = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_audit_run_v1'
    failure = read(run / 'external_receipt.json')
    assert not failure['complete'] and failure['worker_exit_code'] == 1 and not failure['timeout']
    assert not (ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_return').exists()
    bundle = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_vm'
    p = read(bundle / 'protocol.json')
    worker = bundle / 'scripts/cctv_dgp_group_guard_probe_v35_r1_vm.py'
    assert sha(worker) == p['assets_sha256'][worker.relative_to(bundle).as_posix()]
    text = worker.read_text(encoding='utf-8')
    assert "arcname='cctv_dgp_group_guard_probe_v35_return/'" in text
    old = old_path.read_text(encoding='utf-8')
    replacements = {
        "PREFIX='cctv_dgp_group_guard_probe_v35_r1_return/'":
            "PREFIX='cctv_dgp_group_guard_probe_v35_return/'",
        "ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_independent_audit.json'":
            "ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_independent_audit_r2.json'",
    }
    revised = old
    for before, after in replacements.items():
        assert revised.count(before) == 1
        revised = revised.replace(before, after)
    ast.parse(revised, feature_version=(3, 10))
    restored = revised
    for before, after in replacements.items():
        restored = restored.replace(after, before)
    assert restored == old, 'Only archive prefix and distinct receipt path may change'
    destination = ROOT / 'scripts/audit_cctv_dgp_group_guard_probe_v35_r1_return_r2.py'
    assert not destination.exists()
    OUT.mkdir()
    (OUT / 'failed_checker.py').write_bytes(old_path.read_bytes())
    for name in ['execution.json', 'process.json', 'audit.log', 'external_receipt.json']:
        (OUT / name).write_bytes((run / name).read_bytes())
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(revised)
    spec = importlib.util.spec_from_file_location('V35_prefix_only_R2', destination)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    archive = ROOT / 'outputs/cctv-dgp-group-guard-probe-v35-r1-results.tar.gz'
    with tarfile.open(archive, 'r:gz') as stream:
        actual = stream.getmembers()
        safe, total = checker.safe_members(actual, p)
    assert len(safe) == 2135 and {m.name.split('/')[0] for m in actual} == {'cctv_dgp_group_guard_probe_v35_return'}
    invalid = []
    for name in [checker.PREFIX + '../protocol.json', checker.PREFIX + '/protocol.json',
                 checker.PREFIX + 'unknown.json', checker.PREFIX + 'outputs\\results.json',
                 checker.PREFIX + 'outputs/C:results.json', 'cctv_dgp_group_guard_probe_v35_r1_return/protocol.json']:
        item = tarfile.TarInfo(name)
        invalid.append([item])
    link = tarfile.TarInfo(checker.PREFIX + 'protocol.json')
    link.type = tarfile.SYMTYPE
    invalid.append([link])
    duplicate = tarfile.TarInfo(checker.PREFIX + 'protocol.json')
    invalid.append([duplicate, duplicate])
    over = tarfile.TarInfo(checker.PREFIX + 'protocol.json')
    over.size = 8 * 1024**2 + 1
    invalid.append([over])
    rejected = 0
    for members in invalid:
        try:
            checker.safe_members(members, p)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Unsafe archive accepted')
    assert rejected == 9 and sha(old_path) == ORIGINAL_SHA
    hashes = {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(OUT.iterdir()) if f.is_file()}
    receipt = {
        'complete': True, 'cause': 'R1 checker prefix differed from the hash-pinned VM exporter',
        'invalid_assumption': 'Renaming the packet root also renamed the exporter archive prefix',
        'failure_detected_before_extraction_or_model_calls': True,
        'failed_checker_sha256': ORIGINAL_SHA, 'corrected_checker_sha256': sha(destination),
        'pinned_worker_sha256': sha(worker), 'protocol_sha256': sha(bundle / 'protocol.json'),
        'source_changes_only': ['PREFIX constant', 'distinct R2 audit receipt path'],
        'quality_gates_changed': False, 'safe_catalog_and_resource_caps_changed': False,
        'expected_members': len(safe), 'uncompressed_bytes': total, 'unsafe_member_rejections': rejected,
        'preserved_files_sha256': hashes, 'archive_sha256': sha(archive),
        'returned_code_executed': False, 'neural_or_gradient_calls': 0, 'VM_calls': 0,
        'seconds': time.monotonic() - started}
    with (OUT / 'failure_and_correction.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'corrected_checker_sha256': receipt['corrected_checker_sha256'],
                      'valid_members': len(safe), 'unsafe_rejections': rejected}))


if __name__ == '__main__':
    main()

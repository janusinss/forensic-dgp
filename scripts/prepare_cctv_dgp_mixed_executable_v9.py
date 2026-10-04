"""Freeze executable atop independently checked V9 assets; no model calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    assert not (BASE / 'mixed_protocol_v9.json').exists(), 'Preserve frozen executable'
    assert sha(BASE / 'cohort_assets_manifest_v9.json') == 'ec25bf9dc37b32b07fb426cf7875477e70181bf55398f6fdfe420bd4ca550f5d'
    derivation = ROOT / 'outputs/cctv_dgp_mixed_assets_v9_local_audit.json'
    assert sha(derivation) == 'e4a1987964d83c955b9b27275568c77fa7730cf4ba119ba090de42b910981656'
    assert read(derivation)['complete']
    manifest = read(BASE / 'cohort_assets_manifest_v9.json')
    pins = dict(manifest['assets_sha256']); pins.update(manifest['cache_assets'])
    copies = {
        'cctv_dgp_mixed_v9.py': ROOT / 'cctv_dgp_mixed_v9.py',
        'scripts/train_cctv_dgp_mixed_v9.py': ROOT / 'scripts/train_cctv_dgp_mixed_v9.py',
        'scripts/audit_cctv_dgp_mixed_v9.py': ROOT / 'scripts/audit_cctv_dgp_mixed_v9.py',
        'scripts/run_cctv_dgp_mixed_v9_supervised.py': ROOT / 'scripts/run_cctv_dgp_mixed_v9_supervised.py',
        'scripts/prepare_cctv_dgp_mixed_executable_v9.py': Path(__file__).resolve(),
        'scripts/audit_cctv_dgp_mixed_assets_v9.py': ROOT / 'scripts/audit_cctv_dgp_mixed_assets_v9.py',
        'lineage/mixed_assets_local_audit.json': derivation,
        'lineage/mixed_runtime_test_results.json': ROOT / 'outputs/cctv_dgp_mixed_runtime_test_results.json',
    }
    assert read(copies['lineage/mixed_runtime_test_results.json'])['complete']
    for name, source in copies.items():
        if name.endswith('.py'):
            ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 10))
        destination = BASE / name; destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists(), name
        shutil.copyfile(source, destination); pins[name] = sha(destination)
    for name in ('cohort_assets_manifest_v9.json', 'training_schedule_v9.json'):
        pins[name] = sha(BASE / name)
    design = read(BASE / 'lineage/mixed_design_v9.json')
    protocol = {k: manifest[k] for k in ('references', 'training_cases', 'validation_cases', 'weights', 'cache_assets',
        'validation_preview_case_ids', 'training_preview_case_ids', 'canonical_target_rgb_sha256')}
    protocol.update({'version': 9, 'format': 'cctv-dgp-mixed-replay-v9', 'date': '2026-10-04', 'design': design,
        'starting_state_hash': design['starting_state_hash'], 'assets_sha256': pins,
        'training_location': 'forensic-dgp-thesis Linux NVIDIA L4 VM', 'native_used': False,
        'native_reserved_used': False, 'production_promotion_permitted': False,
        'exact_png_input_policy': 'Copy and verify exact prepared PNGs; never regenerate camera inputs on VM',
        'executable_prepared': True, 'training_started': False,
        'source_transcription_note': 'Initial review typos corrected with original-page/native evidence; all previous observations/failures are preserved.',
        'legacy_metadata_note': manifest['legacy_metadata_note']})
    write(BASE / 'mixed_protocol_v9.json', protocol)
    protocol_sha = sha(BASE / 'mixed_protocol_v9.json')
    (BASE / 'mixed_protocol_v9.sha256').write_text(protocol_sha + '\n', encoding='ascii', newline='\n')
    spec = importlib.util.spec_from_file_location('pinned_mixed_prepare', BASE / 'cctv_dgp_mixed_v9.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.verify(BASE, protocol_sha, weights=False)
    archive = ROOT / 'outputs/cctv-dgp-mixed-v9.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=5) as stream:
        for path in sorted(BASE.rglob('*')):
            assert time.monotonic() - started < 600, 'Executable packaging exceeds600 seconds'
            if path.is_file() and path.relative_to(BASE).as_posix() in set(pins) | {'mixed_protocol_v9.json', 'mixed_protocol_v9.sha256'}:
                stream.add(path, arcname=BASE.name + '/' + path.relative_to(BASE).as_posix())
    Path(str(archive) + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
    with tarfile.open(archive, 'r:gz') as stream:
        actual = stream.getmembers()
        wanted = {BASE.name + '/' + n for n in pins if n not in manifest['cache_assets']} | {BASE.name + '/mixed_protocol_v9.json', BASE.name + '/mixed_protocol_v9.sha256'}
        assert all(m.isfile() for m in actual) and {m.name for m in actual} == wanted and len(actual) == len(wanted)
    receipt = {'complete': True, 'version': 9, 'protocol_sha256': protocol_sha,
        'archive_sha256': sha(archive), 'bytes': archive.stat().st_size, 'archive_members': len(actual),
        'training_references': 781, 'training_cases': 3905, 'validation_references': 104, 'validation_cases': 520,
        'expected_updates': 7820, 'trainer_cap_seconds': 2400, 'supervisor_cap_seconds': 3600,
        'thin_cached_weights_sha256': manifest['cache_assets'], 'local_model_forwards': 0,
        'local_backward_calls': 0, 'local_optimizer_updates': 0, 'native_reserved_used': False,
        'execution_prepared': True, 'training_started': False, 'seconds': time.monotonic() - started,
        'execution_source_sha256': sha(Path(__file__))}
    write(ROOT / 'outputs/cctv_dgp_mixed_v9_executable_preparation.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()

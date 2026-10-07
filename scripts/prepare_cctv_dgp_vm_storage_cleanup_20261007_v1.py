"""Prepare exact backed archive/duplicate-recognizer cleanup; never train."""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v1'
HOME = '/home/janusdominic0'
REPO = HOME + '/forensic-dgp'
RECOGNIZER_SHA = '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'
INACTIVE = ('cctv_dgp_detail_prior_vm_v22_r1', 'cctv_dgp_detail_skip_vm_v23',
            'cctv_dgp_degraded_detail_vm_v24', 'cctv_dgp_spatial_features_vm_v25')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy_declared(source, destination, changes):
    original = source.read_text(encoding='utf-8')
    body = original
    for before, after, count in changes:
        assert body.count(before) == count, (source.name, before, body.count(before), count)
        body = body.replace(before, after)
    recovered = body
    for before, after, _ in reversed(changes):
        recovered = recovered.replace(after, before)
    assert ast.dump(ast.parse(recovered)) == ast.dump(ast.parse(original)), 'Undeclared maintenance source change'
    ast.parse(body, feature_version=(3, 10))
    with destination.open('x', encoding='utf-8', newline='\n') as f:
        f.write(body)


def make_sources():
    source = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261006_v2.py'
    insertion = '''        elif row['kind'] == 'backed_up_duplicate_recognizer':
            allowed = {REPO / folder / 'weights/w600k_r50.onnx' for folder in
                ('cctv_dgp_detail_prior_vm_v22_r1', 'cctv_dgp_detail_skip_vm_v23',
                 'cctv_dgp_degraded_detail_vm_v24', 'cctv_dgp_spatial_features_vm_v25')}
            if path not in allowed or row['sha256'] != '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43':
                raise ValueError('Only four exact backed inactive comparison-recognizer copies')
            retained = str(REPO / 'cctv_dgp_feature_skips_vm_v27/weights/w600k_r50.onnx')
            if protected['sha256'].get(retained) != row['sha256']:
                raise ValueError('Current retained VM recognizer differs')
'''
    changes = [
        ('20261006_v2', '20261007_v1', 2), ('20261006-v2', '20261007-v1', 1),
        ("'maintenance_storage_20261006_v1', 'maintenance_storage_20261007_v1'", "'maintenance_storage_20261006_v1', 'maintenance_storage_20261006_v2', 'maintenance_storage_20261007_v1'", 1),
        ("('feature-skips-v27',)", "('broader-mean-v30',)", 1),
        ('current_V27_inputs_weights_outputs_archives_preserved', 'current_V30_dependencies_and_all_trained_DGP_checkpoints_preserved', 1),
        ("'.npy', '.safetensors'}", "'.npy', '.safetensors', '.md', '.txt'}", 1),
        ("if path.suffix in metadata_suffixes or name == 'cuda_runtime_before.txt':", "if str(path) not in cleanup_targets and (path.suffix in metadata_suffixes or name == 'cuda_runtime_before.txt'):", 1),
        ("        elif row['kind'] == 'backed_up_inactive_feature_cache':", insertion + "        elif row['kind'] == 'backed_up_inactive_feature_cache':", 1),
        ("'pip_cache_bytes': sum(row['bytes'] for row in cache),", "'pip_cache_bytes': sum(row['bytes'] for row in cache),\n        'duplicate_recognizer_bytes': sum(row['bytes'] for row in checked if row['kind'] == 'backed_up_duplicate_recognizer'),", 1),
        ("'inactive_cache_logical_bytes_removed': receipt['inactive_cache_bytes'],", "'inactive_cache_logical_bytes_removed': receipt['inactive_cache_bytes'],\n        'duplicate_recognizer_files_removed': sum(row['kind'] == 'backed_up_duplicate_recognizer' for row in receipt['archives']),\n        'duplicate_recognizer_logical_bytes_removed': receipt['duplicate_recognizer_bytes'],", 1),
    ]
    copy_declared(source, ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261007_v1.py', changes)
    source = ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261006_v2.py'
    changes = [
        ('20261006_v2', '20261007_v1', 3),
        ("'Per-family count differs')", "'Per-family count differs')\n    require(cleanup['duplicate_recognizer_files_removed'] == sum(row['kind'] == 'backed_up_duplicate_recognizer' for row in verify['archives']) and cleanup['duplicate_recognizer_logical_bytes_removed'] == sum(row['bytes'] for row in verify['archives'] if row['kind'] == 'backed_up_duplicate_recognizer'), 'Duplicate recognizer count/bytes differ')", 1),
    ]
    copy_declared(source, ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261007_v1.py', changes)
    source = ROOT / 'scripts/verify_cctv_dgp_vm_runtime_after_storage_cleanup_20261006_v1.py'
    copy_declared(source, ROOT / 'scripts/verify_cctv_dgp_vm_runtime_after_storage_cleanup_20261007_v1.py', [('20261006_v1', '20261007_v1', 1)])


def main():
    start = time.monotonic()
    inventory = read(OUT / 'inventory_before.json')
    stat_report = read(OUT / 'inactive_large_file_stat_diagnostic.log')
    assert inventory['complete'] and inventory['hostname'] == 'forensic-dgp-thesis'
    assert inventory['GPU']['returncode'] == 0 and not inventory['GPU']['stdout'].strip()
    assert inventory['related_processes']['returncode'] == 0 and not inventory['related_processes']['stdout'].strip()
    assert inventory['tmux']['returncode'] in (0, 1)
    for line in inventory['tmux']['stdout'].splitlines():
        _, command, dead = line.split('\t')
        assert dead == '1' or command in {'bash', 'sh', 'zsh', 'fish', 'tail', 'less', 'watch', 'top', 'htop', 'tmux'}
    assert stat_report['complete'] and not stat_report['V30_installed']
    v30 = ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30'
    pin = sha(v30 / 'protocol.json')
    assert pin == 'b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1'
    p = read(v30 / 'protocol.json')
    bindings = {}

    def bind(folder, local, assets):
        for name, digest in assets.items():
            path = (local / name).resolve()
            assert path.is_relative_to(local.resolve()) and sha(path) == digest, str(path)
            remote = REPO + '/' + folder + '/' + name
            assert remote not in bindings or bindings[remote] == digest
            bindings[remote] = digest

    folder = 'cctv_dgp_feature_skips_vm_v27'
    local = ROOT / 'outputs' / folder
    assert sha(local / 'protocol.json') == p['closed_V27_protocol_sha256']
    v27 = read(local / 'protocol.json')
    bind(folder, local, {'protocol.json': p['closed_V27_protocol_sha256'], **v27['assets_sha256']})
    bind(folder, ROOT / 'outputs/cctv_dgp_feature_skips_v27_return', p['closed_V27_evidence_sha256'])
    for folder, returned, key in (
        ('cctv_dgp_original_decoder_gradient_v1_r2_vm', 'cctv_dgp_original_decoder_gradient_v1_r2_return', 'closed_R2_evidence_sha256'),
        ('cctv_dgp_active_original_decoder_vm_v28', 'cctv_dgp_active_original_decoder_v28_return', 'closed_V28_evidence_sha256'),
        ('cctv_dgp_v28_preservation_diagnostic_v1_vm', 'cctv_dgp_v28_preservation_diagnostic_v1_return', 'closed_diagnostic_evidence_sha256'),
        ('cctv_dgp_mean_centered_decoder_vm_v29', 'cctv_dgp_mean_centered_decoder_v29_return', 'closed_V29_evidence_sha256'),
    ):
        bind(folder, ROOT / 'outputs' / returned, p[key])
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    bind('cctv_dgp_mixed_vm_v9_r2', mixed, {'mixed_protocol_v9.json': p['mixed_data_protocol_sha256'], **p['mixed_TRAIN_assets_sha256']})
    assert len(p['mixed_TRAIN_assets_sha256']) == 5467
    files, skipped = [], []
    for row in inventory['archives']:
        remote = row['path']; name = remote.rsplit('/', 1)[-1]
        if remote != HOME + '/' + name or not name.startswith('cctv-dgp-') or not name.endswith('.tar.gz') or 'broader-mean-v30' in name:
            skipped.append({'path': remote, 'reason': 'Outside exact historical top-level transfer archive scope'})
            continue
        local = (ROOT / 'outputs' / name).resolve()
        if not local.is_relative_to((ROOT / 'outputs').resolve()) or not local.is_file() or local.stat().st_size != row['bytes']:
            skipped.append({'path': remote, 'reason': 'Complete same-size existing Windows archive unavailable'})
            continue
        digest = sha(local)
        if name.endswith('-results.tar.gz'):
            export = read(local.with_name(name.replace('-results.tar.gz', '-export.json')))
            assert export['complete'] and export['archive_sha256'] == digest and export['bytes'] == row['bytes']
            assert local.with_name(name + '.sha256').read_text().split()[0] == digest
        files.append({**row, 'kind': 'duplicate_archive', 'sha256': digest, 'local_backup': str(local), 'local_backup_verified': True})
    stat_map = {row['path']: row for row in stat_report['large_files']}
    for folder in INACTIVE:
        remote = REPO + '/' + folder + '/weights/w600k_r50.onnx'
        row = stat_map[remote]
        assert row['uid'] == stat_report['current_uid'] and row['nlink'] == 1
        local = ROOT / 'outputs' / folder / 'weights/w600k_r50.onnx'
        protocol = read(ROOT / 'outputs' / folder / 'protocol.json')
        assert protocol['assets_sha256']['weights/w600k_r50.onnx'] == RECOGNIZER_SHA
        assert local.stat().st_size == row['bytes'] and sha(local) == RECOGNIZER_SHA
        files.append({**row, 'kind': 'backed_up_duplicate_recognizer', 'sha256': RECOGNIZER_SHA,
                      'local_backup': str(local), 'local_backup_verified': True,
                      'retained_VM_copy': REPO + '/cctv_dgp_feature_skips_vm_v27/weights/w600k_r50.onnx'})
    assert files and not (set(bindings) & {row['path'] for row in files})
    planned = sum(row['bytes'] for row in files)
    assert planned >= 1.3 * 1024**3 and inventory['disk']['free_bytes'] + planned > 6 * 1024**3
    plan = {'format': 'user-authorized-verified-backed-VM-storage-cleanup-20261007-v1',
        'maintenance_id': 'archive-duplicates', 'home': HOME, 'repo': REPO,
        'pip_download_cache': HOME + '/.cache/pip', 'clear_pip_download_cache': True,
        'maintenance_cap_seconds': 900, 'training_launch_authorized': False,
        'original_checkpoints_splits_logs_preserved': True, 'files': files,
        'protected_assets_sha256': bindings, 'skipped_archives': skipped,
        'backup_policy': 'Complete Windows SHA256 copies plus retained active recognizer; all VM file hashes/stamps rechecked before exact unlinks',
        'scientific_cache_removal_authorized': False, 'current_V30_not_uploaded_or_installed': True,
        'current_V30_dependencies_retained': True, 'V30_protocol_sha256': pin,
        'all_trained_DGP_checkpoints_retained': True, 'local_backups_retained': True,
        'shared_V26_recognizer_retained': True, 'time': time.time()}
    make_sources()
    path = OUT / 'archive-duplicates_plan_20261007_v1.json'
    write(path, plan)
    receipt = {'complete': True, 'plan_sha256': sha(path),
        'backend_sha256': sha(ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261007_v1.py'),
        'auditor_sha256': sha(ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261007_v1.py'),
        'duplicate_archives': sum(row['kind'] == 'duplicate_archive' for row in files),
        'duplicate_recognizers': sum(row['kind'] == 'backed_up_duplicate_recognizer' for row in files),
        'backed_file_bytes': planned, 'predicted_free_bytes': inventory['disk']['free_bytes'] + planned,
        'V30_dependency_bindings': len(bindings), 'TRAIN_assets': 5467,
        'full_original_ASTs_preserved_except_declared_scope_changes': True,
        'scientific_cache_removals_planned': 0, 'trained_DGP_checkpoint_removals_planned': 0,
        'first_recognizer_inventory_failure_retained': True,
        'diagnosis': 'V26 recognizer has two hard links; keep it. Only four singly linked copies reclaim storage.',
        'files_removed': 0, 'training_started': False, 'seconds': time.monotonic() - start}
    write(OUT / 'preparation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

"""Prepare an exact archive-only cleanup from a fresh VM inventory.

Retain the current V27 transfer/return, all unpacked research and all scientific
caches. Every candidate requires an existing complete Windows recovery copy.
"""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2'
HOME = '/home/janusdominic0'
REPO = HOME + '/forensic-dgp'
V27_PIN = '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'
RETURN_SHA = '1bd4aec2bdc10cd6aca4455f24875b006bf4bca2d1a4f1cae8f4932400b386ab'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy_backend():
    source = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261006_v1.py'
    original = source.read_text(encoding='utf-8')
    body = original.replace('20261006_v1', '20261006_v2').replace('20261006-v1', '20261006-v2')
    # Current-archive protection moves from historical V22/V23 to V27. Their
    # unpacked checkpoints, sources, splits, logs and outputs remain protected.
    body = body.replace("('detail-prior-v22', 'detail-skip-v23')", "('feature-skips-v27',)")
    body = body.replace('current_V22_V23_inputs_weights_outputs_archives_preserved',
                        'current_V27_inputs_weights_outputs_archives_preserved')
    # Bind research code, inputs and cached .npy features as well as the original
    # model/metadata hashes. Large historical .bin/.npz caches retain original
    # stamp checks, with no proposed deletion or partial-backup dependency.
    body = body.replace("'.sha256', '.log'}", "'.sha256', '.log', '.py', '.sh', '.png', '.jpg', '.jpeg', '.npy', '.safetensors'}")
    body = body.replace("'maintenance_storage_20261006_v2'}", "'maintenance_storage_20261006_v1', 'maintenance_storage_20261006_v2'}")
    recovered = body.replace('20261006_v2', '20261006_v1').replace('20261006-v2', '20261006-v1')
    recovered = recovered.replace("('feature-skips-v27',)", "('detail-prior-v22', 'detail-skip-v23')")
    recovered = recovered.replace('current_V27_inputs_weights_outputs_archives_preserved',
                                  'current_V22_V23_inputs_weights_outputs_archives_preserved')
    recovered = recovered.replace("'.sha256', '.log', '.py', '.sh', '.png', '.jpg', '.jpeg', '.npy', '.safetensors'}", "'.sha256', '.log'}")
    recovered = recovered.replace("'maintenance_storage_20261006_v1', 'maintenance_storage_20261006_v1'}", "'maintenance_storage_20261006_v1'}")
    assert ast.dump(ast.parse(recovered)) == ast.dump(ast.parse(original)), 'Undeclared backend changes'
    path = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261006_v2.py'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(body)
    ast.parse(body, feature_version=(3, 10))
    source = ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261006_v1.py'
    original_auditor = source.read_text(encoding='utf-8')
    auditor = original_auditor.replace('20261006_v1', '20261006_v2')
    auditor = auditor.replace("(args.phase + '_plan.json')", "(args.phase + '_plan_20261006_v2.json')")
    auditor = auditor.replace('V22/V23', 'current research').replace('explicit_V22_V23_bindings_live_verified', 'explicit_current_research_bindings_live_verified')
    recovered = auditor.replace("(args.phase + '_plan_20261006_v2.json')", "(args.phase + '_plan.json')").replace('20261006_v2', '20261006_v1')
    recovered = recovered.replace('current research', 'V22/V23').replace('explicit_current_research_bindings_live_verified', 'explicit_V22_V23_bindings_live_verified')
    assert ast.dump(ast.parse(recovered)) == ast.dump(ast.parse(original_auditor)), 'Undeclared independent auditor changes'
    path = ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261006_v2.py'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(auditor)
    ast.parse(auditor, feature_version=(3, 10))


def main():
    started = time.monotonic()
    inventory = json.loads((OUT / 'inventory_before.json').read_text())
    assert inventory['complete'] and inventory['hostname'] == 'forensic-dgp-thesis'
    assert inventory['GPU']['returncode'] == 0 and not inventory['GPU']['stdout'].strip()
    assert not inventory['related_processes']['stdout'].strip()
    assert inventory['tmux']['returncode'] in (0, 1)
    for line in inventory['tmux']['stdout'].splitlines():
        _, command, dead = line.split('\t')
        assert dead == '1' or command in {'bash', 'sh', 'zsh', 'fish', 'tail', 'less', 'watch', 'top', 'htop', 'tmux'}
    bindings = {}
    retained_archives = []
    for folder, pin in (
        ('cctv_dgp_feature_skips_vm_v27', V27_PIN),
        ('cctv_dgp_batchmatched_identity_vm_v26', 'f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994'),
    ):
        local = ROOT / 'outputs' / folder
        assert sha(local / 'protocol.json') == pin
        protocol = json.loads((local / 'protocol.json').read_text())
        bindings[REPO + '/' + folder + '/protocol.json'] = pin
        for name, digest in protocol['assets_sha256'].items():
            path = (local / name).resolve()
            assert path.is_relative_to(local.resolve()) and sha(path) == digest
            bindings[REPO + '/' + folder + '/' + name] = digest
    for filename in ('cctv-dgp-feature-skips-v27-execution.tar.gz', 'cctv-dgp-feature-skips-v27-results.tar.gz'):
        local = ROOT / 'outputs' / filename
        digest = sha(local)
        if filename.endswith('results.tar.gz'):
            assert digest == RETURN_SHA and local.stat().st_size == 325777145
            export = json.loads((ROOT / 'outputs/cctv-dgp-feature-skips-v27-export.json').read_text())
            assert export['complete'] and export['archive_sha256'] == digest and export['bytes'] == local.stat().st_size
            assert local.with_name(filename + '.sha256').read_text().split()[0] == digest
        bindings[HOME + '/' + filename] = digest
        retained_archives.append(filename)
    files = []
    skipped = []
    for row in inventory['archives']:
        remote = row['path']
        name = remote.rsplit('/', 1)[-1]
        if name in retained_archives:
            skipped.append({'path': remote, 'reason': 'Current V27 transfer/return retained until independent result review'})
            continue
        # Deliberately avoid research-tree archives and any inferred recursive path.
        if remote != HOME + '/' + name or not name.startswith('cctv-dgp-') or not name.endswith('.tar.gz'):
            skipped.append({'path': remote, 'reason': 'Outside exact top-level CCTV transfer-archive scope'})
            continue
        local = (ROOT / 'outputs' / name).resolve()
        if not local.is_relative_to((ROOT / 'outputs').resolve()) or not local.is_file() or local.stat().st_size != row['bytes']:
            skipped.append({'path': remote, 'reason': 'No complete existing same-size Windows recovery archive'})
            continue
        digest = sha(local)
        files.append({**row, 'kind': 'duplicate_archive', 'sha256': digest,
                      'local_backup': str(local), 'local_backup_verified': True})
    assert files and not (set(bindings) & {row['path'] for row in files})
    plan = {'format': 'user-authorized-verified-backed-VM-storage-cleanup-20261006-v2',
            'maintenance_id': 'archive-duplicates', 'home': HOME, 'repo': REPO,
            'pip_download_cache': HOME + '/.cache/pip', 'clear_pip_download_cache': False,
            'maintenance_cap_seconds': 900, 'training_launch_authorized': False,
            'original_checkpoints_splits_logs_preserved': True, 'files': files,
            'protected_assets_sha256': bindings, 'skipped_archives': skipped,
            'backup_policy': 'Existing complete Windows recovery archives; VM rehash/stamp/protected-before-after checks; exact regular-file unlink only',
            'scientific_cache_removal_authorized': False, 'current_V27_archives_retained': True,
            'local_backups_retained': True, 'time': time.time()}
    copy_backend()
    path = OUT / 'archive-duplicates_plan_20261006_v2.json'
    write(path, plan)
    receipt = {'complete': True, 'plan_sha256': sha(path),
               'backend_sha256': sha(ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261006_v2.py'),
               'auditor_sha256': sha(ROOT / 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261006_v2.py'),
               'archives': len(files), 'bytes': sum(row['bytes'] for row in files),
               'protected_current_research_bindings': len(bindings),
               'original_backend_and_auditor_full_ASTs_preserved_except_declared_scope_changes': True,
               'scientific_cache_removals_planned': 0, 'files_removed': 0, 'training_started': False,
               'seconds': time.monotonic() - started}
    write(OUT / 'preparation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

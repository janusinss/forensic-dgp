"""Create fresh exact backed-file plans, protecting both V22 and V23 evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from import_cctv_dgp_detail_prior_v22_r1 import read, relative_name, require, sha, write

OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v1'
HOME = '/home/janusdominic0'
REPO = HOME + '/forensic-dgp'


def protected():
    bindings = {}
    archives = {}
    for folder, pin, prefix, execution, result in (
        ('cctv_dgp_detail_prior_vm_v22_r1', 'c431af07b8e0bd2310f67fdc1b9ccfac7118adc08dc6cddd22131af6a5a4ca96',
         'cctv_dgp_detail_prior_v22_r1_return', 'cctv-dgp-detail-prior-v22-r1-execution.tar.gz', 'cctv-dgp-detail-prior-v22-r1-results.tar.gz'),
        ('cctv_dgp_detail_skip_vm_v23', '4cddb98fb5e6a215cb84f98132cfa5a2146ee157c1cb939919ffbf8c5740e863',
         'cctv_dgp_detail_skip_v23_return', 'cctv-dgp-detail-skip-v23-execution.tar.gz', 'cctv-dgp-detail-skip-v23-results.tar.gz'),
    ):
        local = ROOT / 'outputs' / folder
        require(sha(local / 'protocol.json') == pin, 'Original protocol differs')
        p = read(local / 'protocol.json')
        remote = REPO + '/' + folder
        bindings[remote + '/protocol.json'] = pin
        for name, digest in p['assets_sha256'].items():
            relative_name(name)
            require(sha(local / name) == digest, 'Original local asset differs: ' + name)
            bindings[remote + '/' + name] = digest
        archive = ROOT / 'outputs' / result
        manifest = None
        members = {}
        with tarfile.open(archive, 'r:gz') as stream:
            for member in stream:
                part = relative_name(member.name)
                require(member.isfile() and not member.issparse() and len(part.parts) > 1 and part.parts[0] == prefix, 'Unsafe return member')
                name = '/'.join(part.parts[1:])
                require(name.casefold() not in {n.casefold() for n in members} and 0 <= member.size <= 32 * 1024**2 and len(members) < 1024, 'Return member bounds differ')
                payload = stream.extractfile(member).read()
                require(len(payload) == member.size, 'Truncated return member')
                members[name] = hashlib.sha256(payload).hexdigest()
                if name == 'export_manifest.json':
                    require(member.size <= 1024**2, 'Oversized manifest')
                    manifest = json.loads(payload)
        require(manifest is not None and manifest['complete'] and manifest['protocol_sha256'] == pin and
                set(manifest['files_sha256']) == set(members) - {'export_manifest.json'}, 'Return manifest differs')
        for name, digest in manifest['files_sha256'].items():
            require(members[name] == digest, 'Returned file hash differs')
            bindings[remote + '/' + name] = digest
        bindings[remote + '/export_manifest.json'] = members['export_manifest.json']
        require(members['protocol.json'] == pin and
                members['schedule.json'] == p['assets_sha256']['schedule.json'], 'Return source differs')
        for name in ('execution', 'result'):
            filename = execution if name == 'execution' else result
            digest = sha(ROOT / 'outputs' / filename)
            bindings[HOME + '/' + filename] = digest
            archives[filename] = digest
        sidecar = ROOT / 'outputs' / (result + '.sha256')
        require(sidecar.read_text(encoding='ascii').strip() == archives[result] + '  ' + result, 'Return sidecar differs')
    require(archives['cctv-dgp-detail-skip-v23-results.tar.gz'] == 'b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9', 'Current V23 return differs')
    return bindings


def prepare(phase):
    started = time.monotonic()
    path = OUT / (phase + '_plan.json')
    require(not path.exists(), 'Preserve existing plan')
    inventory = read(OUT / 'inventory_before.json')
    require(inventory['complete'] and inventory['hostname'] == 'forensic-dgp-thesis' and
            inventory['GPU']['returncode'] == 0 and not inventory['GPU']['stdout'].strip(), 'Fresh named idle inventory required')
    files = []
    if phase == 'archive-duplicates':
        old = read(ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261005_r2/archive-duplicates_plan.json')
        targets = {row['path']: row for row in old['files']}
        require(len(targets) == 7, 'Original archive scope differs')
        for row in inventory['archives']:
            if row['path'] not in targets:
                continue
            original = targets[row['path']]
            local = Path(original['local_backup'])
            require(local.resolve().is_relative_to(ROOT / 'outputs') and local.is_file() and
                    local.stat().st_size == row['bytes'] and sha(local) == original['sha256'], 'Exact retained archive backup differs')
            files.append({**row, 'kind': 'duplicate_archive', 'sha256': original['sha256'],
                'local_backup': str(local), 'local_backup_verified': True})
        require(len(files) == 7, 'Fresh seven-archive scope changed')
    else:
        backup = read(OUT / 'inactive_cache_Windows_backup_receipt.json')
        require(backup['complete'] and backup['manifest_sha256'] == sha(OUT / 'inactive_cache_manifest.json'), 'Incomplete/changed backup')
        for row in backup['files']:
            local = Path(row['local_backup']).resolve()
            require(local.is_relative_to(OUT / 'cache_backups') and row['local_backup_verified'] and
                    local.stat().st_size == row['bytes'] and sha(local) == row['sha256'], 'Independent cache backup read-back differs')
            if row['cleanup_candidate']:
                files.append({**row, 'kind': 'backed_up_inactive_feature_cache'})
        require(len(files) == 4429 and sum(Path(r['path']).suffix == '.bin' for r in files) == 4 and
                sum(Path(r['path']).suffix == '.npz' for r in files) == 4425, 'Four binary/4425 NPZ scope differs')
    bindings = protected()
    plan = {'format': 'user-authorized-verified-backed-VM-storage-cleanup-20261006-v1',
        'maintenance_id': phase, 'home': HOME, 'repo': REPO, 'pip_download_cache': HOME + '/.cache/pip',
        'clear_pip_download_cache': phase == 'archive-duplicates', 'maintenance_cap_seconds': 900 if phase == 'archive-duplicates' else 1800,
        'training_launch_authorized': False, 'original_checkpoints_splits_logs_preserved': True,
        'files': files, 'protected_assets_sha256': bindings,
        'backup_policy': 'Exact backed-up archive/cache files only; per-file VM rehash/stamp checks and protected before/after snapshots; no recursive deletion',
        'current_V22_V23_pixels_weights_outputs_archives_protected': True, 'local_backups_retained': True,
        'time': time.time(), 'seconds': time.monotonic() - started}
    write(path, plan)
    script = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261006_v1.py'
    ast.parse(script.read_text(encoding='utf-8'), feature_version=(3, 10))
    receipt = {'complete': True, 'phase': phase, 'plan_sha256': sha(path), 'backend_sha256': sha(script),
               'files': len(files), 'bytes': sum(r['bytes'] for r in files), 'protected_current_bindings': len(bindings),
               'local_backup_sha256_checked': True, 'seconds': time.monotonic() - started,
               'files_removed': 0, 'training_started': False}
    write(OUT / (phase + '_preparation.json'), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('archive-duplicates', 'inactive-feature-caches'))
    prepare(parser.parse_args().phase)

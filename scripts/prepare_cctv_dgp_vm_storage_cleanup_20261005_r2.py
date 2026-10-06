"""Freeze exact maintenance plans from fresh inventory and verified Windows copies."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261005_r2'
HOME = '/home/janusdominic0'
REPO = HOME + '/forensic-dgp'
BUNDLE = ROOT / 'outputs/cctv_dgp_detail_prior_vm_v22_r1'
PIN = 'c431af07b8e0bd2310f67fdc1b9ccfac7118adc08dc6cddd22131af6a5a4ca96'
RETURN_SHA = '4a734450e9ea8fca8fb240a5e06b55f99a28316a770869938b2bd2aa05ad9cad'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def protected_assets():
    if sha(BUNDLE / 'protocol.json') != PIN:
        raise ValueError('Current V22 original protocol changed')
    protocol = read(BUNDLE / 'protocol.json')
    result = {REPO + '/cctv_dgp_detail_prior_vm_v22_r1/protocol.json': PIN}
    for name, digest in protocol['assets_sha256'].items():
        if sha(BUNDLE / name) != digest:
            raise ValueError('Current V22 original asset differs: ' + name)
        result[REPO + '/cctv_dgp_detail_prior_vm_v22_r1/' + name] = digest
    archive = ROOT / 'outputs/cctv-dgp-detail-prior-v22-r1-results.tar.gz'
    if sha(archive) != RETURN_SHA:
        raise ValueError('Current V22 user-return archive differs')
    with tarfile.open(archive, 'r:gz') as stream:
        manifest = json.load(stream.extractfile('cctv_dgp_detail_prior_v22_r1_return/export_manifest.json'))
    if not manifest['complete'] or manifest['protocol_sha256'] != PIN:
        raise ValueError('Current V22 return binding differs')
    for name, digest in manifest['files_sha256'].items():
        result[REPO + '/cctv_dgp_detail_prior_vm_v22_r1/' + name] = digest
    result[HOME + '/cctv-dgp-detail-prior-v22-r1-results.tar.gz'] = RETURN_SHA
    result[HOME + '/cctv-dgp-detail-prior-v22-r1-execution.tar.gz'] = '6978e423c23909caebff65c7299267ce1a6803d15e2818e38bac6b1f685fdaba'
    return result


def prepare(phase):
    started = time.monotonic()
    target = OUT / (phase + '_plan.json')
    if target.exists():
        raise RuntimeError('Preserve frozen maintenance plan; no overwrite')
    files = []
    if phase == 'archive-duplicates':
        inventory = read(OUT / 'inventory_before_trusted.json')
        if not inventory['complete'] or inventory['hostname'] != 'forensic-dgp-thesis' or inventory['GPU']['stdout'].strip():
            raise ValueError('Require the fresh idle named VM inventory')
        for row in inventory['archives']:
            name = Path(row['path']).name
            if 'detail-prior-v22' in name or name == 'real-camera-review.tar.gz':
                continue
            local = ROOT / 'outputs' / name
            if not local.is_file() or local.stat().st_size != row['bytes']:
                raise ValueError('Archive backup missing/size differs: ' + name)
            files.append({**row, 'kind': 'duplicate_archive', 'sha256': sha(local),
                          'local_backup': str(local), 'local_backup_verified': True})
        if len(files) != 7:
            raise ValueError('Fresh exact7 older archive targets required')
    elif phase == 'inactive-feature-caches':
        backup = read(OUT / 'inactive_cache_Windows_backup_receipt.json')
        if not backup['complete'] or sha(OUT / 'inactive_cache_manifest.json') != backup['manifest_sha256']:
            raise ValueError('Incomplete/changed scientific cache backup manifest')
        for row in backup['files']:
            local = Path(row['local_backup'])
            if not local.is_relative_to(OUT / 'cache_backups') or not row['local_backup_verified'] or local.stat().st_size != row['bytes']:
                raise ValueError('Cache local backup scope/size differs')
            # The completed copy receipt already independently hashed every byte.
            # Recheck its size/manifest here; deletion still rehashes each VM file.
            if row['cleanup_candidate']:
                files.append({**row, 'kind': 'backed_up_inactive_feature_cache'})
        if len(files) != 4429 or any(Path(row['path']).suffix not in {'.bin', '.npz'} for row in files):
            raise ValueError('Exact4 binary and4425 NPZ cache targets required')
    else:
        raise ValueError('Unknown maintenance phase')
    plan = {'format': 'user-authorized-verified-backed-VM-storage-cleanup-20261005-r2',
        'maintenance_id': phase, 'home': HOME, 'repo': REPO, 'pip_download_cache': HOME + '/.cache/pip',
        'clear_pip_download_cache': phase == 'archive-duplicates', 'maintenance_cap_seconds': 900 if phase == 'archive-duplicates' else 1800,
        'training_launch_authorized': False, 'original_checkpoints_splits_logs_preserved': True,
        'files': files, 'protected_assets_sha256': protected_assets(),
        'backup_policy': 'Remove exact idle VM duplicates only after verified Windows backup; keep all original sources, checkpoints, data, splits, logs, failure gates and current V22 run.',
        'inactive_cache_usage': 'Historical training caches only; current V22 is self-contained and does not read these directories.',
        'all_local_backups_retained': True, 'seconds': time.monotonic() - started}
    write(target, plan)
    receipt = {'complete': True, 'phase': phase, 'plan_sha256': sha(target), 'script_sha256': sha(ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261005_r2.py'),
        'files': len(files), 'bytes': sum(row['bytes'] for row in files),
        'protected_explicit_assets': len(plan['protected_assets_sha256']), 'seconds': time.monotonic() - started,
        'files_removed': 0, 'training_started': False}
    write(OUT / (phase + '_preparation.json'), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['archive-duplicates', 'inactive-feature-caches'])
    prepare(parser.parse_args().phase)

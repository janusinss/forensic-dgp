"""Copy named inactive caches with gcloud; hash every Windows backup. No deletion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v1'
GCLOUD = Path(os.environ['LOCALAPPDATA']) / 'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
TRUST = ROOT / 'scratch/gcloud_windows_trust.pem'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def backup():
    started = time.monotonic()
    inventory_path = OUT / 'inactive_cache_manifest.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8-sig'))
    if not inventory['complete'] or inventory['format'] != 'inactive-feature-cache-backup-manifest-r2':
        raise ValueError('Require the fresh complete read-only VM manifest')
    expected_roots = ['/home/janusdominic0/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/anatomical_cache', '/home/janusdominic0/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/fixed_cache', '/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/outputs/broader_codes_v16_r2/cache']
    if inventory['roots'] != expected_roots:
        raise ValueError('Only the three named historical cache roots may be copied')
    cache_root = OUT / 'cache_backups'
    if cache_root.exists():
        raise RuntimeError('Preserve existing/partial backups; no automatic repeat')
    if shutil.disk_usage(ROOT).free < inventory['all_backup_bytes'] + 10 * 1024**3:
        raise RuntimeError('Windows needs exact backup bytes plus10GiB free')
    cache_root.mkdir()
    environment = dict(os.environ)
    environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = str(TRUST)
    backed = []
    for index, remote_root in enumerate(inventory['roots']):
        destination = cache_root / ('group' + str(index))
        destination.mkdir()
        command = [str(GCLOUD), 'compute', 'scp', '--project=forensic-dgp-thesis', '--zone=us-central1-a',
                   '--quiet', '--recurse', 'janusdominic0@forensic-dgp-thesis:' + remote_root, str(destination)]
        print(json.dumps({'stage': 'copy', 'group': index, 'remote': remote_root}), flush=True)
        with (OUT / ('cache_copy_' + str(index) + '.log')).open('xb') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        local = destination / Path(remote_root).name
        selected = [row for row in inventory['files'] if row['root'] == remote_root]
        actual_names = {p.relative_to(local).as_posix() for p in local.rglob('*') if p.is_file()}
        if actual_names != {row['relative'] for row in selected}:
            raise ValueError('Copied cache file set differs')
        for row in selected:
            path = local / row['relative']
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('Windows cache backup hash differs: ' + str(path))
            backed.append({**row, 'local_backup': str(path), 'local_backup_verified': True})
        print(json.dumps({'stage': 'backup_verified', 'group': index, 'files': len(selected),
                          'bytes': sum(row['bytes'] for row in selected), 'seconds': time.monotonic() - started}), flush=True)
        write(OUT / ('cache_backup_group' + str(index) + '_receipt.json'),
              {'complete': True, 'manifest_sha256': sha(inventory_path), 'files': backed[-len(selected):],
               'files_removed': 0, 'seconds': time.monotonic() - started})
    receipt = {'complete': True, 'manifest_sha256': sha(inventory_path), 'files': backed,
        'backup_bytes': sum(row['bytes'] for row in backed), 'seconds': time.monotonic() - started,
        'files_removed': 0, 'training_started': False, 'VM_stopped': False}
    write(OUT / 'inactive_cache_Windows_backup_receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('complete', 'backup_bytes', 'seconds', 'files_removed')}, indent=2))


if __name__ == '__main__':
    backup()

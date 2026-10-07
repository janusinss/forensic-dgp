"""Read two original VM export sidecars; no pilot execution or remote writes."""
import base64
import hashlib
import json
from pathlib import Path
import shlex

import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v31_return_metadata_v1'
PIN = 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'
ARCHIVE_SHA = 'cbb89bb5fb85ca7e8b77203188164ab2472342f8d7f3b3ecd7a07f6b53c6f49b'
ARCHIVE_BYTES = 1252002625
NAMES = ('cctv-dgp-profile-batches-v31-results.tar.gz.sha256',
         'cctv-dgp-profile-batches-v31-export.json')
KEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'

REMOTE = r'''
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
assert os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
assert str(Path.home()) == '/home/janusdominic0'
root = Path.home() / 'forensic-dgp/cctv_dgp_profile_batches_vm_v31'
assert hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest() == 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'
rows = []
for name in ('cctv-dgp-profile-batches-v31-results.tar.gz.sha256', 'cctv-dgp-profile-batches-v31-export.json'):
    path = Path.home() / name
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 16000
    data = path.read_bytes()
    rows.append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                 'content_base64': base64.b64encode(data).decode('ascii')})
archive = Path.home() / 'cctv-dgp-profile-batches-v31-results.tar.gz'
assert archive.is_file() and not archive.is_symlink()
print(json.dumps({'complete': True, 'read_only': True, 'VM_writes': 0,
                  'training_started_by_agent': False, 'snapshot_utc': datetime.now(timezone.utc).isoformat(),
                  'archive_bytes': archive.stat().st_size, 'files': rows}, indent=2))
'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    transport.OUT = OUT
    source = 'import base64;exec(base64.b64decode(' + repr(base64.b64encode(REMOTE.encode()).decode()) + '))'
    transport.run(['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + transport.BASE +
                  ['--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag=' + KEY,
                   '--command=python3 -B -c ' + shlex.quote(source)], 'original_return_metadata', 60)
    data = json.loads((OUT / 'original_return_metadata.log').read_text(encoding='utf-8-sig'))
    assert data['complete'] and data['read_only'] and data['VM_writes'] == 0
    assert not data['training_started_by_agent'] and data['archive_bytes'] == ARCHIVE_BYTES
    assert [r['name'] for r in data['files']] == list(NAMES)
    decoded = {}
    for row in data['files']:
        raw = base64.b64decode(row['content_base64'], validate=True)
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        decoded[row['name']] = raw
    assert decoded[NAMES[0]].decode().strip().split() == [ARCHIVE_SHA, 'cctv-dgp-profile-batches-v31-results.tar.gz']
    exported = json.loads(decoded[NAMES[1]])
    assert exported['complete'] and exported['archive_sha256'] == ARCHIVE_SHA and exported['bytes'] == ARCHIVE_BYTES
    assert exported['training_success_not_implied'] and exported['optimizer_updates'] == 50
    assert exported['failure_present'] and not exported['run_results_present']
    archive = ROOT / 'outputs/cctv-dgp-profile-batches-v31-results.tar.gz'
    assert archive.stat().st_size == ARCHIVE_BYTES and sha(archive) == ARCHIVE_SHA
    installed = []
    for name, raw in decoded.items():
        destination = ROOT / 'outputs' / name
        existed = destination.exists()
        if existed:
            assert destination.read_bytes() == raw, 'Preserve an existing different download: ' + name
        else:
            with destination.open('xb') as stream:
                stream.write(raw)
        installed.append({'name': name, 'sha256': sha(destination), 'existing_identical_download': existed})
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)),
               'remote_transport_sha256': sha(OUT / 'original_return_metadata_transport.json'),
               'remote_metadata_sha256': sha(OUT / 'original_return_metadata.log'),
               'original_VM_sidecars_retained_exactly': installed, 'archive_sha256': ARCHIVE_SHA,
               'archive_bytes': ARCHIVE_BYTES, 'VM_writes': 0, 'training_started_by_agent': False,
               'neural_or_gradient_calls': 0, 'quality_acceptance_not_implied': True, 'goal_complete': False}
    with (OUT / 'verification.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

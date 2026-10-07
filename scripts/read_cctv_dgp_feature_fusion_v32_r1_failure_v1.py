"""Read original R1 failure evidence using authorized maintenance SSH; no VM writes."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_failure_v1'
PIN = 'e25e5721a5474d767fe48710336105a1eaa843ed2e22d6cb03caebd8e3bd10da'
EXPECTED = '599d84dd2b84fdc05e55b774d115951d251e4013ae777d7f023791bdb60f59e2'
KEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
REMOTE = r'''
import base64
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
assert os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
assert str(Path.home()) == '/home/janusdominic0'
root = Path.home() / 'forensic-dgp/cctv_dgp_feature_fusion_vm_v32_r1'
def sha(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
        return h.hexdigest()
assert sha(root/'protocol.json') == 'e25e5721a5474d767fe48710336105a1eaa843ed2e22d6cb03caebd8e3bd10da'
p = json.loads((root/'protocol.json').read_text())
for name, digest in p['assets_sha256'].items():
    path = (root/name).resolve()
    assert path.is_relative_to(root) and path.is_file() and not path.is_symlink() and sha(path) == digest
names = ['outputs/failure.json', 'outputs/gradient_preflight.json', 'trainer.log',
         'trainer_exit_code.txt', 'supervisor_receipt.json', 'export_manifest.json']
files = []
for name in names:
    path = root/name
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 200000
    raw = path.read_bytes()
    files.append({'relative_path':name, 'sha256':hashlib.sha256(raw).hexdigest(),
                  'bytes':len(raw), 'content_base64':base64.b64encode(raw).decode('ascii')})
stem = 'cctv-dgp-feature-fusion-v32-r1'
for suffix in ['-results.tar.gz.sha256', '-export.json']:
    path = Path.home()/(stem+suffix)
    raw = path.read_bytes()
    assert not path.is_symlink() and len(raw) < 16000
    files.append({'relative_path':'sidecars/'+path.name,'sha256':hashlib.sha256(raw).hexdigest(),
                  'bytes':len(raw),'content_base64':base64.b64encode(raw).decode('ascii')})
archive = Path.home()/(stem+'-results.tar.gz')
assert archive.is_file() and not archive.is_symlink() and archive.stat().st_size == 169708697
assert sha(archive) == '599d84dd2b84fdc05e55b774d115951d251e4013ae777d7f023791bdb60f59e2'
print(json.dumps({'complete':True,'snapshot_UTC':datetime.now(timezone.utc).isoformat(),
 'files':files,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
 'results_present':(root/'outputs/results.json').exists(),'VM_writes':0,
 'neural_or_gradient_calls':0,'training_started_by_agent':False}))
'''


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    transport.OUT = OUT
    source = 'import base64;exec(base64.b64decode(' + repr(base64.b64encode(REMOTE.encode()).decode()) + '))'
    transport.run(['compute','ssh','janusdominic0@forensic-dgp-thesis'] + transport.BASE +
                  ['--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+KEY,
                   '--command=python3 -B -c '+shlex.quote(source)], 'original_failure', 60)
    data = json.loads((OUT/'original_failure.log').read_text(encoding='utf-8-sig'))
    assert data['complete'] and data['VM_writes'] == data['neural_or_gradient_calls'] == 0
    assert not data['training_started_by_agent'] and not data['results_present']
    assert data['archive_sha256'] == EXPECTED and data['archive_bytes'] == 169708697
    for row in data['files']:
        raw = base64.b64decode(row['content_base64'], validate=True)
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        path = OUT/'original_VM_files'/row['relative_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(raw)
    originals = OUT/'original_VM_files'
    failure = json.loads((originals/'outputs/failure.json').read_text())
    proof = json.loads((originals/'outputs/gradient_preflight.json').read_text())
    assert failure['protocol_sha256'] == PIN and failure['optimizer_updates'] == 0
    assert not failure['optimizer_constructed'] and not failure['new_checkpoint_created'] and not failure['resume_permitted']
    assert 'cctv_dgp_feature_fusion_v32_cache.py' in failure['traceback'] and 'AssertionError' in failure['traceback']
    assert proof['complete'] and proof['optimizer_updates'] == 0 and not proof['optimizer_constructed']
    exported = json.loads((originals/'sidecars/cctv-dgp-feature-fusion-v32-r1-export.json').read_text())
    assert exported['archive_sha256'] == EXPECTED and exported['failure_present'] and not exported['run_results_present']
    manifest = json.loads((originals/'export_manifest.json').read_text())
    assert manifest['complete'] and manifest['protocol_sha256'] == PIN
    for row in data['files']:
        if not row['relative_path'].startswith('sidecars/') and row['relative_path'] != 'export_manifest.json':
            assert manifest['files_sha256'][row['relative_path']] == row['sha256']
    report = {'complete':True, 'reader_sha256':sha(Path(__file__)), 'protocol_sha256':PIN,
              'archive_sha256':EXPECTED,'archive_bytes':169708697,'original_failure_verified':True,
              'optimizer_updates':0,'optimizer_constructed':False,'new_checkpoint_created':False,
              'raw_gradient_values_independently_audited':False,'full_return_download_pending':True,
              'files_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'VM_writes':0,'neural_or_gradient_calls':0,'training_started_by_agent':False,'goal_complete':False}
    with (OUT/'verification.json').open('x',encoding='utf-8',newline='\n') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'files_sha256'},indent=2))


if __name__ == '__main__':
    main()

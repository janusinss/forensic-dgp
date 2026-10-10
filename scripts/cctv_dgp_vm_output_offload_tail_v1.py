"""Finish a stopped maintenance audit only; never repeat file removal."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time

import cctv_dgp_bank_archive_cleanup_v1 as t

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_vm_output_offload_20261010_v1_r2'
OUT = PARENT / 'tail_recovery_v1'
WORKER = ROOT / 'scripts/cctv_dgp_vm_output_offload_tail_v1_vm.py'
REMOTE = '/home/janusdominic0/dgp_output_offload_tail_v1'
PIN = 'cea47392679e05d0de4e1d41d25618aa48d7e89e307213779e5839be82b30f52'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bind():
    t.transport.OUT = OUT
    t.OUT = OUT


def inspect():
    assert not OUT.exists()
    assert sha(PARENT / 'plan.json') == PIN
    failed = read(PARENT / 'transport/apply_transport.json')
    assert not failed['complete'] and failed['exit_code'] == 1
    stderr = (PARENT / 'transport/apply_stderr.log').read_text()
    assert 'TimeoutError: Maintenance900s stop' in stderr and 'after = protected(extra_array_paths)' in stderr
    OUT.mkdir(); (OUT / 'transport').mkdir(); bind()
    write(OUT / 'original_stop.json', dict(complete=True, original_apply_complete=False,
        stopped_in='post-removal retained-file hashing', original_stop_seconds=900,
        transport_sha256=sha(PARENT / 'transport/apply_transport.json'),
        stdout_sha256=sha(PARENT / 'transport/apply_stdout.log'), stderr_sha256=sha(PARENT / 'transport/apply_stderr.log'),
        deletions_will_not_be_repeated=True, model_gradient_or_training_calls=0))
    write(OUT / 'instance_inspect.json', t.api('api_inspect'))
    source = '''import json,hashlib,shutil,socket,os,stat
from pathlib import Path
import urllib.request
home=Path('/home/janusdominic0');root=home/'forensic-dgp'
assert Path.home().resolve()==home and os.getuid()==1001 and socket.gethostname().split('.')[0]=='forensic-dgp-thesis'
request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'})
assert urllib.request.urlopen(request,timeout=5).read().decode()=='4410777042005672095'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
planpath=home/'dgp_output_offload_20261010_v1_plan.json'
assert sha(planpath)=='cea47392679e05d0de4e1d41d25618aa48d7e89e307213779e5839be82b30f52'
p=json.loads(planpath.read_text());folder=home/'dgp_output_offload_20261010_v1_receipts'
ledger=folder/'deletion_ledger.jsonl';s=ledger.lstat()
assert stat.S_ISREG(s.st_mode) and not ledger.is_symlink() and s.st_uid==1001 and s.st_nlink==1
rows=[json.loads(line) for line in ledger.read_text().splitlines()]
expected={r['relative_path'] for r in p['candidates']}
gone=sum(not (root/name).exists() for name in expected)
assert len(rows)==len(expected)==76973 and {r['relative_path'] for r in rows}==expected and gone==76973
assert not (folder/'cleanup_receipt.json').exists() and not (folder/'protected_after.json.gz').exists()
print(json.dumps(dict(complete=True,candidate_count=len(expected),absent_candidates=gone,ledger_rows=len(rows),ledger_sha256=sha(ledger),ledger_bytes=s.st_size,allocated_removed_bytes=sum(r['allocated_bytes'] for r in rows),free_bytes=shutil.disk_usage(root).free,receipt_files=[dict(name=f.name,bytes=f.stat().st_size) for f in sorted(folder.iterdir())],files_removed_this_inspection=0,model_gradient_or_training_calls=0)))
'''
    (OUT / 'inspection_source.py').write_text(source, encoding='utf-8', newline='\n')
    value = read(t.ssh_source(source, 'inspect', 180))
    write(OUT / 'inspection.json', value)
    print(dict(complete=True, removed_copies_confirmed=value['absent_candidates'],
               free_GiB=value['free_bytes']/1024**3, files_removed_this_step=0), flush=True)


def close():
    started = time.monotonic(); bind()
    assert not (OUT / 'closure_execution.json').exists()
    inspection = read(OUT / 'inspection.json')
    plan = read(PARENT / 'plan.json')
    assert inspection['complete'] and inspection['absent_candidates'] == inspection['ledger_rows'] == 76973
    assert sha(PARENT / 'plan.json') == PIN and sha(ROOT / 'scripts/cctv_dgp_vm_output_offload_v1_vm.py') == plan['remote_script_sha256']
    for path, expected in read(ROOT / plan['local_protected_manifest']).items():
        assert sha(ROOT / path) == expected, path
    assert sha(PARENT / 'dgp-image-output-recovery-20261010-v1.tar') == plan['portable_backup_sha256']
    previous = ROOT / 'outputs/cctv_dgp_bank_archive_cleanup_v1/inventory_after.json'
    recovery = dict(complete=True, phase='post-removal audit and evidence export only', additional_deletions_permitted=False,
        original_plan_sha256=PIN, original_worker_sha256=plan['remote_script_sha256'],
        recovery_worker_sha256=sha(WORKER), original_apply_stop_sha256=sha(OUT / 'original_stop.json'),
        original_apply_transport_sha256=sha(PARENT / 'transport/apply_transport.json'),
        expected_ledger_sha256=inspection['ledger_sha256'], candidate_count=76973,
        previous_guest_inventory_sha256=sha(previous), previous_guest_free_bytes=read(previous)['disk']['free_bytes'],
        previous_inventory_UTC=read(previous)['UTC'], cap_seconds=900, model_gradient_or_training_calls=0)
    write(OUT / 'tail_plan.json', recovery)
    write(OUT / 'instance_close.json', t.api('api_close'))
    remote_script = '/home/janusdominic0/cctv_dgp_vm_output_offload_tail_v1_vm.py'
    remote_plan = '/home/janusdominic0/dgp_output_offload_tail_v1_plan.json'
    for file, dest, stem in [(WORKER, remote_script, 'worker_upload'), (OUT / 'tail_plan.json', remote_plan, 'plan_upload')]:
        t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE + ['--scp-flag=' + f for f in t.FLAGS] +
                 [str(file), 'janusdominic0@forensic-dgp-thesis:' + dest], stem, 180, False)
    pin = sha(OUT / 'tail_plan.json')
    command = 'python3 -B ' + shlex.quote(remote_script) + ' --plan ' + shlex.quote(remote_plan) + ' --plan-sha ' + pin
    t.invoke([str(t.CLOUD), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + t.BASE +
             ['--ssh-flag=' + f for f in t.FLAGS] + ['--command=' + command], 'close', 960, False)
    for name in ['dgp-output-offload-tail-v1-receipts.tar.gz', 'dgp-output-offload-tail-v1-export.json']:
        t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE + ['--scp-flag=' + f for f in t.FLAGS] +
                 ['janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + name, str(OUT / name)],
                 'download_' + name.replace('.', '_'), 240, False)
    write(OUT / 'closure_execution.json', dict(complete=True, driver_sha256=sha(Path(__file__)),
        tail_plan_sha256=pin, original_plan_sha256=PIN, original_apply_complete=False,
        additional_files_removed=0, model_gradient_or_training_calls=0, seconds=time.monotonic()-started))
    print(dict(complete=True, post_removal_audit_exported=True, additional_files_removed=0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['inspect', 'close'], required=True)
    a = parser.parse_args()
    inspect() if a.phase == 'inspect' else close()

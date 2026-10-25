"""Export-only Python3.10-compatible recovery. Never imports a neural framework.

Preserves the sealed training packet, saved states, original export manifest and
failed archive. Writes a separately named archive, checksum and recovery receipt.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
import sys
import tarfile
import time

PROTOCOL_SHA='3fe24e684a835268ac805ca56bfafefff429ef5f1aa13a90ec11faa644380dd9'
VM_NAME='cctv_codeformer_spatial_objectives_v1_vm'
OLD_PREFIX='cctv-codeformer-spatial-objectives-v1'
NEW_ARCHIVE=OLD_PREFIX+'-results-recovery-r1.tar.gz'
NEW_RECEIPT=OLD_PREFIX+'-export-recovery-r1.json'
CAP_SECONDS=600
MiB=1024**2


def require(value,message):
    if not value:raise ValueError(message)


class Limited:
    def __init__(self,stream,cap):self.stream=stream;self.cap=cap;self.count=0
    def write(self,payload):
        require(self.count+len(payload)<=self.cap,'Compressed archive cap exceeded')
        count=self.stream.write(payload)
        require(count==len(payload),'Short archive write')
        self.count+=count;return count
    def flush(self):return self.stream.flush()


def write_tar(stream,rows,opener,cap):
    """Explicit gzip level; no compresslevel keyword passed to tarfile stream."""
    bounded=Limited(stream,cap)
    with gzip.GzipFile(filename='',mode='wb',fileobj=bounded,compresslevel=1,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as archive:
            for row in rows:
                info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with opener(row) as source:archive.addfile(info,source)


def self_test():
    payloads={'a.json':b'{"example":1}\n','folder/data.bin':bytes(range(256))*32}
    rows=[{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in sorted(payloads.items())]
    destination=io.BytesIO();write_tar(destination,rows,lambda r:io.BytesIO(payloads[r['path']]),1024*1024)
    found={}
    with tarfile.open(fileobj=io.BytesIO(destination.getvalue()),mode='r|gz') as archive:
        for member in archive:
            require(member.isfile() and member.name not in found,'Self-test archive structure')
            found[member.name]=archive.extractfile(member).read()
    require(found==payloads,'Recovered format round-trip differs')
    try:Limited(io.BytesIO(),2).write(b'123')
    except ValueError:pass
    else:raise ValueError('Archive bound did not reject excessive write')
    require('torch' not in sys.modules,'Self test must not import Torch')
    return {'complete':True,'fixtures':['explicit gzip stream round-trip','archive bound'],
            'neural_optimizer_training_VM_calls':0}


def active_run_processes(root,child_pid):
    """Read-only project-specific inventory. Never signal another process."""
    require(type(child_pid) is int and child_pid>0,'Recorded worker PID required')
    own=os.getpid();found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==own:continue
        try:
            args=(entry/'cmdline').read_bytes().split(b'\0')
            words=[b.decode('utf-8','replace') for b in args if b]
            if not words:continue
            leaves=[Path(w).name for w in words]
            is_run=any(w in ('run_codeformer_spatial_objectives_v1.py',
                'supervise_codeformer_spatial_objectives_v1.py') for w in leaves)
            is_export=any(w=='recover_codeformer_spatial_objectives_export_v1_r1.py' for w in leaves)
            if is_run or is_export:
                found.append({'pid':int(entry.name),'arguments':words})
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    require(not found,'Another matching worker/export is live; preserve it and wait: '+str(found))
    # Recheck only the recorded group identity; a reused live group is uncertain
    # and stops export without sending any signal.
    try:os.killpg(child_pid,0)
    except ProcessLookupError:pass
    else:raise ValueError('Recorded worker process group is live; no recovery export')
    return {'matching_live_processes':[],'recorded_worker_group_absent':True,'nothing_killed':True}


def recover(expected_sha):
    require(platform.system()=='Linux','Run this export-only helper on the existing Linux VM')
    require(re.fullmatch('[0-9a-f]{64}',expected_sha or ''),'Expected helper SHA256 required')
    source=Path(__file__).absolute()
    require(hashlib.sha256(source.read_bytes()).hexdigest()==expected_sha,'Recovery helper SHA256 differs')
    home=Path.home();root=home/'forensic-dgp'/VM_NAME
    # Import only source-pinned inert IO helpers from the original sealed root.
    for path in (root,*root.parents):
        info=path.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),'Physical VM directory required')
    protocol_bytes=(root/'protocol.json').read_bytes()
    require(hashlib.sha256(protocol_bytes).hexdigest()==PROTOCOL_SHA,'Original protocol bytes differ')
    raw_protocol=json.loads(protocol_bytes)
    for name in ('research/__init__.py','research/codeformer_spatial_objectives_v1/__init__.py',
                 'research/codeformer_spatial_objectives_v1/packet.py'):
        path=root/name;info=path.lstat();bound=raw_protocol['files'][name]
        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and not stat.S_ISLNK(info.st_mode) and
                info.st_size==bound['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==bound['sha256'],
                'Pinned inert import source changed: '+name)
    sys.dont_write_bytecode=True;sys.path.insert(0,str(root))
    from research.codeformer_spatial_objectives_v1 import packet as p
    require(Path(p.__file__).resolve()==(root/'research/codeformer_spatial_objectives_v1/packet.py').resolve(),
            'IO module source must belong to this original packet')
    protocol=p.verify(root,PROTOCOL_SHA)
    require(protocol['vm_name']==VM_NAME,'Wrong experiment')
    supervisor=p.read_json(root/'supervisor.json')
    require(supervisor.get('complete') is True and supervisor.get('worker_terminal') is True and
            supervisor.get('owned_child_group_terminal') is True and supervisor.get('worker_exit')==0,
            'This recovery requires the successful, terminal worker receipt; preserve other failures for review')
    idle=active_run_processes(root,supervisor['child_pid'])
    terminal=p.read_json(root/'outputs/spatial_objectives_v1/result.json')
    require(terminal.get('complete') is True and terminal.get('protocol_sha256')==PROTOCOL_SHA and
            terminal.get('pending')=={} and terminal.get('counts')==terminal.get('returns') and
            [r.get('arm') for r in terminal.get('branches',[])]==['U','S'],
            'Completed worker/result operation receipt required')
    require(all(r.get('terminal_status') in ('complete','guard_stop') and
                type(r.get('completed_updates')) is int and 0<=r['completed_updates']<=24
                for r in terminal['branches']),'Unexpected branch state; no training recovery is allowed')
    manifest=p.read_json(root/'export.json')
    require(manifest.get('complete') is True and manifest.get('protocol_sha256')==PROTOCOL_SHA and
            manifest.get('worker_exit')==0 and manifest.get('worker_terminal') is True,
            'Original pre-archive export manifest required')
    existing=p.inventory(root)
    all_original_pins={name:p.pin(root,path) for name,path in existing.items()}
    expected=set(protocol['return_static'])|{'protocol.json'}|{
        n for n in existing if n in protocol['allowed_outputs'] and not n.startswith('cache/') and n!='export.json'}
    rows=manifest.get('members');require(type(rows) is list and all(type(r) is dict for r in rows),'Original members required')
    require(len(rows)==len({r['path'] for r in rows}) and {r['path'] for r in rows}==expected,
            'Original export member namespace differs; never drop evidence to make export pass')
    for row in rows:require(all_original_pins.get(row['path'])==row,'Original manifest member changed: '+row['path'])
    manifest_pin=p.pin(root,root/'export.json');rows=sorted(rows+[manifest_pin],key=lambda r:r['path'])
    require(sum(r['bytes'] for r in rows)<=protocol['limits']['return_bytes'],'Sealed return payload cap exceeded')
    archive=home/NEW_ARCHIVE;partial=home/(NEW_ARCHIVE+'.partial')
    checksum=home/(NEW_ARCHIVE+'.sha256');receipt=home/NEW_RECEIPT
    require(all(not f.exists() and not f.is_symlink() for f in (archive,partial,checksum,receipt)),
            'Preserve prior recovery output or partial; no overwrite')
    require(shutil.disk_usage(home).free>=protocol['limits']['disk_reserve_bytes']+protocol['limits']['archive_bytes'],
            'Need4GiB reserve plus bounded archive space before recovery')
    preserved=[]
    for name in (OLD_PREFIX+'-results.tar.gz.partial',OLD_PREFIX+'-results.tar.gz',
                 OLD_PREFIX+'-results.tar.gz.sha256',OLD_PREFIX+'-export.json'):
        path=home/name
        if path.exists():preserved.append({'path':name,'bytes':p.regular(path).stat().st_size,'sha256':p.sha256(path)})
    original_pins={r['path']:r for r in rows};start=time.monotonic()
    def opener(row):
        path=p.regular(root/row['path'])
        require(p.pin(root,path)==row,'Member changed immediately before archiving')
        return path.open('rb')
    with partial.open('xb') as destination:
        write_tar(destination,rows,opener,protocol['limits']['archive_bytes'])
        destination.flush();os.fsync(destination.fileno())
    verified={}
    with tarfile.open(partial,mode='r|gz') as saved:
        for member in saved:
            require(member.isfile() and not member.issparse() and member.name in original_pins and
                    member.name not in verified,'Recovery archive member/type/namespace differs')
            row=original_pins[member.name];require(member.size==row['bytes'],'Recovery member size differs')
            content=saved.extractfile(member);digest=hashlib.sha256();count=0
            for b in iter(lambda:content.read(MiB),b''):digest.update(b);count+=len(b)
            require(count==row['bytes'] and digest.hexdigest()==row['sha256'],'Recovery member bytes differ')
            verified[member.name]=row
    require(set(verified)==set(original_pins),'Recovery archive missing members')
    after_inventory=p.inventory(root)
    require(set(after_inventory)==set(existing),'Original run namespace changed during recovery')
    for name,path in after_inventory.items():
        require(p.pin(root,path)==all_original_pins[name],'Original run file changed during recovery: '+name)
    require(p.pin(root,root/'export.json')==manifest_pin,'Original export manifest changed')
    active_run_processes(root,supervisor['child_pid'])
    for row in preserved:
        require((home/row['path']).stat().st_size==row['bytes'] and p.sha256(home/row['path'])==row['sha256'],
                'Original failed export evidence changed')
    os.link(partial,archive);partial.unlink()
    archive_sha=p.sha256(archive)
    with checksum.open('xb') as f:f.write((archive_sha+'  '+archive.name+'\n').encode('ascii'))
    result={'format':'codeformer-spatial-objectives-export-only-recovery-r1','complete':True,
        'archive':archive.name,'archive_sha256':archive_sha,'bytes':archive.stat().st_size,
        'protocol_sha256':PROTOCOL_SHA,'helper_sha256':expected_sha,'seconds':time.monotonic()-start,
        'whole_recovery_cap_seconds':CAP_SECONDS,'members':len(rows),'member_payload_bytes':sum(r['bytes'] for r in rows),
        'original_export_manifest':manifest_pin,'original_failed_export_files_preserved':preserved,
        'inventory':idle,'all_members_streamed_and_hash_verified':True,'original_run_files_unchanged':True,
        'all_original_root_files_hash_checked':len(all_original_pins),
        'original_root_inventory_sha256':hashlib.sha256(p.json_bytes(all_original_pins)).hexdigest(),
        'original_worker_exit':0,'branches':[{k:r[k] for k in ('arm','completed_updates','terminal_status')} for r in terminal['branches']],
        'new_neural_gradient_optimizer_training_calls':0,'training_restarted':False,'training_success_not_implied':True,
        'model_qualified':False}
    with receipt.open('xb') as f:f.write(p.json_bytes(result))
    require('torch' not in sys.modules,'Recovery must never import Torch')
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True);actions.add_argument('--self-test',action='store_true')
    actions.add_argument('--expected-sha');a=parser.parse_args()
    if a.self_test:print(json.dumps(self_test()));return
    require(platform.system()=='Linux','Manual VM export-only recovery required')
    def expired(sig,frame):raise TimeoutError('Recovery600s cap exceeded; preserve partial, do not rerun blindly')
    old=signal.signal(signal.SIGALRM,expired);signal.alarm(CAP_SECONDS)
    try:recover(a.expected_sha)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':main()

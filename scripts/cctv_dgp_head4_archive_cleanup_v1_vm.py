"""Exact home-archive removal; research tree, active work and backups are guarded."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time

HOME=Path('/home/janusdominic0');ROOT=HOME/'forensic-dgp';OUT=HOME/'dgp_head4_archive_cleanup_v1_receipts'
EXTENSIONS={'.py','.md','.json','.jsonl','.pth','.pt','.onnx','.sha256','.log','.csv'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,x):
    with Path(path).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def idle(targets):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    assert result.returncode==0 and not result.stdout.strip(),'Active GPU workload; retain archives'
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:
            command=(folder/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=(folder/'cwd').resolve();comm=(folder/'comm').read_text().strip()
            assert not (str(ROOT)+'/' in command and comm not in {'sshd','bash','zsh'}),'Active research command: '+folder.name
            assert not (cwd.is_relative_to(ROOT) and comm in {'python','python3','python3.10','tar','gzip','rsync','cp','mv'}),'Active research workload: '+folder.name
            for fd in (folder/'fd').iterdir():
                try:assert str(fd.resolve()) not in targets,'Archive has an open reader: '+folder.name
                except (FileNotFoundError,PermissionError,OSError):pass
        except (FileNotFoundError,PermissionError,OSError):pass


def archive_check(row):
    path=Path(row['path']);assert path.parent==HOME and path.resolve()==path
    assert path.name.startswith('cctv-dgp-') and path.name.endswith('.tar.gz') and 'capacity' not in path.name
    s=path.lstat();assert stat.S_ISREG(s.st_mode) and not path.is_symlink() and s.st_uid==os.getuid() and s.st_nlink==1
    for key,value in [('bytes',s.st_size),('allocated_bytes',s.st_blocks*512),('uid',s.st_uid),('nlink',s.st_nlink),('inode',s.st_ino),('mtime_ns',s.st_mtime_ns)]:
        assert row[key]==value,(str(path),key)
    assert sha(path)==row['sha256'],'Remote/local backup hash differs: '+path.name


def protected():
    metadata=[];hashes={};cache={};count=0
    for folder,dirs,files in os.walk(ROOT,followlinks=False):
        dirs.sort();files.sort()
        for name in dirs+files:
            path=Path(folder)/name;s=path.lstat();relative=path.relative_to(ROOT).as_posix()
            metadata.append([relative,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_nlink,s.st_uid,
                os.readlink(path) if stat.S_ISLNK(s.st_mode) else None])
            if stat.S_ISREG(s.st_mode) and '.venv' not in path.parts and path.suffix in EXTENSIONS:
                key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
                if key not in cache:cache[key]=sha(path)
                hashes[relative]=cache[key]
            count+=1
            if count%50000==0:print({'protected_metadata_entries':count,'protected_byte_hashes':len(hashes)},flush=True)
    return {'metadata':sorted(metadata),'critical_and_evidence_hashes':hashes,
        'cache_bytes_not_all_rehashed':True,'home_archives_have_no_shared_links_and_are_outside_research_root':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--phase',choices=['verify','apply'],required=True);a=parser.parse_args();start=time.monotonic()
    assert Path.home().resolve()==HOME and socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and os.getuid()==1001
    assert sha(a.plan)==a.plan_sha;p=read(a.plan);assert sha(Path(__file__))==p['remote_script_sha256']
    assert p['research_root']==str(ROOT) and p['authorization_scope']=='inventory-first hash-bound inactive archive copies only'
    assert len(p['candidates'])==15 and len({r['path'] for r in p['candidates']})==15
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Maintenance900s stop')));signal.alarm(900)
    targets={r['path'] for r in p['candidates']};idle(targets)
    for r in p['candidates']:archive_check(r)
    if a.phase=='verify':
        assert not OUT.exists();OUT.mkdir();before=protected()
        with gzip.open(OUT/'protected_before.json.gz','xb',compresslevel=1) as f:f.write(json.dumps(before,sort_keys=True).encode())
        write(OUT/'verify_receipt.json',{'complete':True,'plan_sha256':a.plan_sha,'candidate_count':15,
            'protected_canonical_sha256':canonical(before),'protected_file_sha256':sha(OUT/'protected_before.json.gz'),
            'protected_metadata_entries':len(before['metadata']),'protected_byte_hashes':len(before['critical_and_evidence_hashes']),
            'cache_bytes_not_all_rehashed':True,'estimated_reclaim_bytes':sum(r['allocated_bytes'] for r in p['candidates']),
            'files_removed':0,'training_launched':False,'seconds':time.monotonic()-start})
        print({'complete':True,'verified_archives':15,'files_removed':0,'phase':'verify'},flush=True);return
    assert OUT.is_dir() and not (OUT/'cleanup_receipt.json').exists() and not (OUT/'deletion_ledger.jsonl').exists()
    receipt=read(OUT/'verify_receipt.json');assert receipt['complete'] and receipt['plan_sha256']==a.plan_sha
    assert sha(OUT/'protected_before.json.gz')==receipt['protected_file_sha256']
    with gzip.open(OUT/'protected_before.json.gz','rb') as f:before=json.load(f)
    assert canonical(before)==receipt['protected_canonical_sha256']
    assert protected()==before,'Protected state changed after inventory; stop before deletion'
    idle(targets)
    for r in p['candidates']:archive_check(r)
    disk_before=__import__('shutil').disk_usage(ROOT).free;removed=[]
    with (OUT/'deletion_ledger.jsonl').open('x') as ledger:
        for r in p['candidates']:
            archive_check(r);Path(r['path']).unlink();removed.append(r['path'])
            ledger.write(json.dumps({'path':r['path'],'sha256':r['sha256'],'backup_verified':True,'allocated_bytes':r['allocated_bytes']})+'\n');ledger.flush();os.fsync(ledger.fileno())
    after=protected();assert after==before,'Research metadata/protected hashes differ; preserve ledger'
    with gzip.open(OUT/'protected_after.json.gz','xb',compresslevel=1) as f:f.write(json.dumps(after,sort_keys=True).encode())
    disk_after=__import__('shutil').disk_usage(ROOT).free
    result={'complete':True,'plan_sha256':a.plan_sha,'files_removed':len(removed),'removed_paths':removed,
        'protected_canonical_before':canonical(before),'protected_canonical_after':canonical(after),
        'protected_metadata_entries':len(after['metadata']),'protected_byte_hashes':len(after['critical_and_evidence_hashes']),
        'cache_bytes_not_all_rehashed':True,'disjoint_nlink1_home_only_deletion':True,
        'free_before_bytes':disk_before,'free_after_bytes':disk_after,'recovered_bytes':disk_after-disk_before,
        'allocated_archive_bytes':sum(r['allocated_bytes'] for r in p['candidates']),
        'unpacked_research_assets_and_failures_unchanged':True,'VM_started':False,'training_launched':False,
        'seconds':time.monotonic()-start,'goal_complete':False}
    write(OUT/'cleanup_receipt.json',result)
    files=[f for f in OUT.iterdir() if f.is_file()]
    write(OUT/'manifest.json',{'complete':True,'plan_sha256':a.plan_sha,'files_sha256':{f.name:sha(f) for f in files}})
    import tarfile
    archive=HOME/'dgp-head4-archive-cleanup-v1-receipts.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for f in sorted(OUT.iterdir()):tar.add(f,arcname='receipts/'+f.name,recursive=False)
    write(HOME/'dgp-head4-archive-cleanup-v1-export.json',{'complete':True,'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'training_launched':False})
    print({'complete':True,'removed_archives':len(removed),'free_GiB':disk_after/1024**3,'phase':'apply'},flush=True)


if __name__=='__main__':main()

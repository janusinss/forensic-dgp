"""Inventory-first offload of 929 backed-up RGB outputs and one result archive.

Exact regular singly linked paths only. No imports or execution of model code.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import time
import urllib.request

HOME=Path('/home/janusdominic0')
ROOT=HOME/'forensic-dgp'
BANK=ROOT/'cctv_dgp_bank_comparison_v1_vm'
OUT=HOME/'dgp_bank_split_offload_v1_receipts'
ARCHIVE=HOME/'cctv-dgp-bank-comparison-v1-results.tar.gz'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def idle(targets):
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    assert gpu.returncode==0 and not gpu.stdout.strip(),'GPU workload active; retain files'
    tmux=subprocess.run(['tmux','list-panes','-a','-F','#S|#P|#{pane_current_command}|#{pane_pid}|#{pane_current_path}|#{pane_dead}'],capture_output=True,text=True,timeout=15)
    assert tmux.returncode==0 or (tmux.returncode==1 and 'No such file or directory' in tmux.stderr),'tmux inventory failed'
    for line in tmux.stdout.splitlines():
        p=line.split('|');assert len(p)==6 and p[2] in {'bash','zsh','sh','fish'} and p[5]=='0','Active tmux task'
    active=[];opened=[]
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:
            comm=(folder/'comm').read_text().strip()
            command=(folder/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=(folder/'cwd').resolve()
            if (str(ROOT)+'/' in command and comm not in {'sshd','bash','sh','zsh'}) or (cwd.is_relative_to(ROOT) and comm in {'python','python3','python3.10','tar','gzip','rsync','cp','mv'}):active.append(int(folder.name))
            for fd in (folder/'fd').iterdir():
                try:
                    if str(fd.resolve()) in targets:opened.append([int(folder.name),str(fd.resolve())])
                except (OSError,PermissionError):pass
        except (OSError,PermissionError):pass
    assert not active and not opened,'Active research process or open output file'
    return dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_compute_idle=True,tmux=tmux.stdout,active_research_processes=active,open_target_readers=opened)

def allowed(path):
    assert path.resolve()==path and path!=ROOT
    if path==ARCHIVE:return
    assert path.is_relative_to(BANK/'outputs/bank_comparison_v1/baseline')
    rel=path.relative_to(BANK/'outputs/bank_comparison_v1/baseline')
    assert len(rel.parts)==2 and ((rel.parts[0]=='packs' and path.suffix=='.npz') or (rel.parts[0]=='previews' and path.suffix=='.png') or (rel.parts[0]=='native' and path.suffix in {'.npy','.png'}))
    assert not any(p.is_symlink() for p in path.parents)

def checked(row,metadata=False):
    p=Path(row['path']);allowed(p);s=p.lstat()
    assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_uid==1001 and s.st_nlink==1
    assert s.st_size==row['bytes'] and sha(p)==row['sha256'],'VM bytes differ from full local backup'
    info=dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,uid=s.st_uid,nlink=s.st_nlink,inode=s.st_ino,mtime_ns=s.st_mtime_ns)
    if metadata:
        for k,v in info.items():assert row[k]==v,('Candidate changed',str(p),k)
    return {**row,**info}

def protected(targets):
    # All research-tree metadata except exact target files and their three parents.
    # Hash every retained regular file in the bank packet (including all assets,
    # model states, metrics, provenance, logs, stops and export manifest).
    parent_dirs={str(Path(p).parent) for p in targets if Path(p)!=ARCHIVE}
    rows=[];hashes={}
    for folder,dirs,files in os.walk(ROOT,followlinks=False):
        dirs.sort();files.sort()
        for n in dirs+files:
            p=Path(folder)/n
            if str(p) in targets or str(p) in parent_dirs:continue
            s=p.lstat();rel=p.relative_to(ROOT).as_posix()
            rows.append([rel,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_nlink,s.st_uid,os.readlink(p) if stat.S_ISLNK(s.st_mode) else None])
            if p.is_relative_to(BANK) and stat.S_ISREG(s.st_mode):hashes[rel]=sha(p)
    return dict(metadata=sorted(rows),bank_retained_bytes_sha256=hashes,all_scientific_cache_bytes_rehashed=False)

def gzwrite(name,value):
    with gzip.open(OUT/name,'xb',compresslevel=1) as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False).encode())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--phase',choices=['inventory','verify','apply'],required=True);a=parser.parse_args();start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite900s maintenance stop; retain evidence')));signal.alarm(900)
    assert Path.home().resolve()==HOME and os.getuid()==1001 and socket.gethostname().split('.')[0]=='forensic-dgp-thesis'
    req=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'})
    ident=urllib.request.urlopen(req,timeout=5).read().decode()
    assert ident=='4410777042005672095' and sha(a.plan)==a.plan_sha
    p=read(a.plan);assert p['complete'] and p['instance_id']==ident and p['worker_sha256']==sha(__file__)
    assert p['authorization_scope']=='inactive RGB output packs plus one archive; full verified local backup'
    assert len(p['candidates'])==930 and len({r['path'] for r in p['candidates']})==930
    assert sum(Path(r['path'])==ARCHIVE for r in p['candidates'])==1
    targets={r['path'] for r in p['candidates']};runtime=idle(targets)
    rows=[checked(r,metadata=a.phase!='inventory') for r in p['candidates']]
    if a.phase=='inventory':
        print(json.dumps(dict(complete=True,candidates=rows,free_bytes=shutil.disk_usage(ROOT).free,runtime=runtime,files_removed=0,model_gradient_or_training_calls=0)));return
    if a.phase=='verify':
        assert not OUT.exists(),'Do not overwrite a prior maintenance phase';OUT.mkdir()
        before=protected(targets);gzwrite('protected_before.json.gz',before)
        write(OUT/'verify_receipt.json',dict(complete=True,plan_sha256=a.plan_sha,worker_sha256=sha(__file__),candidate_count=len(rows),protected_sha256=sha(OUT/'protected_before.json.gz'),protected_canonical_sha256=canonical(before),protected_metadata_entries=len(before['metadata']),retained_bank_hashes=len(before['bank_retained_bytes_sha256']),runtime=runtime,files_removed=0,seconds=time.monotonic()-start))
        print(json.dumps(read(OUT/'verify_receipt.json')));return
    assert OUT.is_dir() and not (OUT/'cleanup_receipt.json').exists() and not (OUT/'deletion_ledger.jsonl').exists()
    verify=read(OUT/'verify_receipt.json');assert verify['complete'] and verify['plan_sha256']==a.plan_sha and sha(OUT/'protected_before.json.gz')==verify['protected_sha256']
    with gzip.open(OUT/'protected_before.json.gz','rb') as f:before=json.load(f)
    assert canonical(before)==verify['protected_canonical_sha256'] and protected(targets)==before,'Protected files changed; retain candidates'
    runtime=idle(targets);free_before=shutil.disk_usage(ROOT).free;removed=[]
    with (OUT/'deletion_ledger.jsonl').open('x') as ledger:
        for i,row in enumerate(rows):
            if i%50==0:idle(targets)
            checked(row,metadata=True);Path(row['path']).unlink();removed.append(row['path'])
            ledger.write(json.dumps(dict(path=row['path'],sha256=row['sha256'],local_backup=row['local_backup'],inode=row['inode'],nlink=row['nlink'],allocated_bytes=row['allocated_bytes'],backup_verified=True))+'\n');ledger.flush();os.fsync(ledger.fileno())
    after=protected(targets);gzwrite('protected_after.json.gz',after)
    assert after==before,'Retained research state differs; preserve deletion ledger'
    free_after=shutil.disk_usage(ROOT).free
    write(OUT/'cleanup_receipt.json',dict(complete=True,plan_sha256=a.plan_sha,worker_sha256=sha(__file__),files_removed=len(removed),removed_paths=removed,protected_metadata_entries=len(after['metadata']),retained_bank_hashes=len(after['bank_retained_bytes_sha256']),protected_canonical_before=canonical(before),protected_canonical_after=canonical(after),free_before_bytes=free_before,free_after_bytes=free_after,recovered_bytes=free_after-free_before,allocated_reclaim_bytes=sum(r['allocated_bytes'] for r in rows),all_scientific_cache_bytes_rehashed=False,checkpoints_inputs_splits_logs_metrics_provenance_and_failures_retained=True,runtime_before_removal=runtime,VM_started=False,model_gradient_or_training_calls=0,goal_complete=False,seconds=time.monotonic()-start))
    write(OUT/'manifest.json',dict(complete=True,plan_sha256=a.plan_sha,files_sha256={f.name:sha(f) for f in sorted(OUT.iterdir()) if f.is_file()}))
    import tarfile
    archive=HOME/'dgp-bank-split-offload-v1-receipts.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for f in sorted(OUT.iterdir()):tar.add(f,arcname='receipts/'+f.name,recursive=False)
    write(HOME/'dgp-bank-split-offload-v1-export.json',dict(complete=True,archive_sha256=sha(archive),bytes=archive.stat().st_size,model_gradient_or_training_calls=0))
    print(json.dumps(dict(complete=True,files_removed=len(removed),free_GiB=free_after/1024**3,seconds=time.monotonic()-start)))

if __name__=='__main__':main()

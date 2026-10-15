"""Inventory-first hash-bound offload of completed B4 RGB outputs/home archives.

Only exact backed-up regular singly linked files in the signed plan may be
removed. Research metadata and all retained B4 bytes are protected. No model
imports/training, cache deletion, directory removal or VM power change.
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

HOME = Path('/home/janusdominic0')
ROOT = HOME/'forensic-dgp'
BANK = ROOT/'cctv_dgp_bank_comparison_v1_r4_B_vm'
OUT = HOME/'dgp_bank_B4_output_cleanup_v2_receipts'
INSTANCE = '4410777042005672095'
PIN = '8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85'
ARCHIVES = {HOME/'cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz',
            HOME/'cctv-dgp-bank-comparison-v1-r4-B-execution.tar.gz'}
LABELS = {'baseline','B50','B50_bank_disabled'}
LEGACY = ROOT/'cctv_dgp_residual_epochs_vm_v42'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(1024**2),b''): h.update(value)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def identity():
    assert Path.home().resolve() == HOME and os.getuid() == 1001
    assert socket.gethostname().split('.')[0] == 'forensic-dgp-thesis'
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'})
    assert urllib.request.urlopen(request,timeout=5).read().decode() == INSTANCE
    assert sha(BANK/'protocol.json') == PIN
    assert sha(LEGACY/'protocol.json') == 'c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d'
    assert (LEGACY/'outputs/failure.json').is_file() and (LEGACY/'outputs/capacity_update50.json').is_file()
    stopped = read(BANK/'outputs/bank_comparison_v1/quality_stop_B.json')
    assert stopped['type'] == 'quality_gate_failure' and stopped['outcome']['updates'] == 50
    assert stopped['outcome']['capacity_pass'] is False and stopped['failed_recipe_resume_allowed'] is False
    supervisor = read(BANK/'supervisor_receipt.json')
    assert supervisor['complete'] and supervisor['trainer_exit_code'] == 2 and supervisor['optimizer_updates'] == 50


def allowed(path):
    assert path.is_absolute() and path.resolve() == path
    assert not any(parent.is_symlink() for parent in path.parents)
    if path in ARCHIVES: return
    if path.is_relative_to(LEGACY):
        relative = path.relative_to(LEGACY/'outputs')
        assert len(relative.parts) == 2 and relative.parts[0] in {'update0','update50'} and path.suffix == '.png'
        return
    relative = path.relative_to(BANK/'outputs/bank_comparison_v1')
    assert len(relative.parts) == 3 and relative.parts[0] in LABELS
    assert relative.parts[1] in {'packs','png','native'}
    assert path.suffix == ('.npz' if relative.parts[1] == 'packs' else '.png') or (
        relative.parts[1] == 'native' and path.suffix == '.npy')


def idle(targets):
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    assert gpu.returncode == 0 and not gpu.stdout.strip(), 'GPU task active; retain source'
    tmux = subprocess.run(['tmux','list-panes','-a','-F','#S|#P|#{pane_current_command}|#{pane_pid}|#{pane_current_path}|#{pane_dead}'],capture_output=True,text=True,timeout=15)
    assert tmux.returncode == 0 or (tmux.returncode == 1 and ('No such file or directory' in tmux.stderr or 'no server running' in tmux.stderr)), 'tmux inventory failed'
    for line in tmux.stdout.splitlines():
        fields = line.split('|'); assert len(fields) == 6 and fields[2] in {'bash','zsh','sh','fish'} and fields[5] == '0', 'Active tmux work'
    active = []; readers = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid(): continue
        try:
            comm = (folder/'comm').read_text().strip()
            command = (folder/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd = (folder/'cwd').resolve()
            if (str(ROOT)+'/' in command and comm not in {'sshd','bash','sh','zsh'}) or (cwd.is_relative_to(ROOT) and comm in {'python','python3','python3.10','tar','gzip','rsync','cp','mv'}): active.append(int(folder.name))
            for fd in (folder/'fd').iterdir():
                try:
                    if str(fd.resolve()) in targets: readers.append([int(folder.name),str(fd.resolve())])
                except (OSError,PermissionError): pass
        except (OSError,PermissionError): pass
    assert not active and not readers, 'Active research process or target reader'
    return dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_compute_idle=True,tmux=tmux.stdout,
                active_research_processes=active,open_target_readers=readers)


def info(path):
    allowed(path); s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and not path.is_symlink() and s.st_uid == 1001
    return dict(path=str(path),bytes=s.st_size,allocated_bytes=s.st_blocks*512,uid=s.st_uid,
                nlink=s.st_nlink,inode=s.st_ino,mtime_ns=s.st_mtime_ns)


def checked(row, metadata=False):
    path = Path(row['path']); actual = info(path)
    assert actual['nlink'] == 1 and actual['bytes'] == row['bytes']
    assert sha(path) == row['sha256'], 'Bytes differ from complete verified local backup'
    if metadata:
        assert all(row[key] == value for key,value in actual.items()), 'Candidate metadata changed'
    return {**row,**actual}


def protected(targets):
    changed_dirs = {str(Path(path).parent) for path in targets if Path(path) not in ARCHIVES}
    rows = []; hashes = {}
    for folder,dirs,files in os.walk(ROOT,followlinks=False):
        dirs.sort(); files.sort()
        for name in dirs+files:
            path = Path(folder)/name
            if str(path) in targets or str(path) in changed_dirs: continue
            s = path.lstat(); relative = path.relative_to(ROOT).as_posix()
            rows.append([relative,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_nlink,s.st_uid,
                         os.readlink(path) if stat.S_ISLNK(s.st_mode) else None])
            if (path.is_relative_to(BANK) or path.is_relative_to(LEGACY)) and stat.S_ISREG(s.st_mode): hashes[relative] = sha(path)
    return dict(metadata=sorted(rows),bank_retained_bytes_sha256=hashes,all_scientific_cache_bytes_rehashed=False)


def gzwrite(name, value):
    with gzip.open(OUT/name,'xb',compresslevel=1) as stream:
        stream.write(json.dumps(value,sort_keys=True,allow_nan=False).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['discover','inventory','verify','apply'],required=True)
    parser.add_argument('--plan',type=Path); parser.add_argument('--plan-sha')
    a = parser.parse_args(); start = time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite900s maintenance stop; retain ledger')))
    signal.alarm(900); identity()
    if a.phase == 'discover':
        paths = [path for label in sorted(LABELS) for folder in ['packs','png','native']
                 for path in sorted((BANK/'outputs/bank_comparison_v1'/label/folder).glob('*')) if path.is_file()]
        paths += [path for folder in ['update0','update50'] for path in sorted((LEGACY/'outputs'/folder).glob('*.png')) if path.is_file()]
        paths += [path for path in sorted(ARCHIVES) if path.exists()]
        runtime = idle({str(path) for path in paths})
        print(json.dumps(dict(complete=True,instance_id=INSTANCE,protocol_sha256=PIN,
            free_bytes=shutil.disk_usage(ROOT).free,prospective_candidates=[info(path) for path in paths],
            runtime=runtime,files_removed=0,model_gradient_or_training_calls=0,worker_sha256=sha(__file__))),flush=True)
        return
    assert a.plan and a.plan_sha and sha(a.plan) == a.plan_sha
    p = read(a.plan)
    assert p['complete'] and p['instance_id'] == INSTANCE and p['worker_sha256'] == sha(__file__)
    assert p['scope'] == 'completed B4 RGB output packs including archived measurement fields, backed-up home archives, and completed V42 update0/update50 PNG copies; all checkpoints, caches, raw V42 arrays, splits, instructions, logs and stop records retained'
    assert len(p['candidates']) == len({row['path'] for row in p['candidates']}) and len(p['candidates']) > 0
    targets = {row['path'] for row in p['candidates']}; runtime = idle(targets)
    rows = [checked(row,metadata=a.phase != 'inventory') for row in p['candidates']]
    if a.phase == 'inventory':
        print(json.dumps(dict(complete=True,candidates=rows,free_bytes=shutil.disk_usage(ROOT).free,
            runtime=runtime,files_removed=0,model_gradient_or_training_calls=0)),flush=True); return
    if a.phase == 'verify':
        assert not OUT.exists(), 'Retain prior maintenance evidence'; OUT.mkdir()
        before = protected(targets); gzwrite('protected_before.json.gz',before)
        write(OUT/'verify_receipt.json',dict(complete=True,plan_sha256=a.plan_sha,worker_sha256=sha(__file__),
            candidate_count=len(rows),protected_sha256=sha(OUT/'protected_before.json.gz'),
            protected_canonical_sha256=canonical(before),protected_metadata_entries=len(before['metadata']),
            retained_bank_hashes=len(before['bank_retained_bytes_sha256']),runtime=runtime,files_removed=0,
            seconds=time.monotonic()-start))
        print(json.dumps(read(OUT/'verify_receipt.json')),flush=True); return
    assert OUT.is_dir() and not (OUT/'cleanup_receipt.json').exists() and not (OUT/'deletion_ledger.jsonl').exists()
    receipt = read(OUT/'verify_receipt.json')
    assert receipt['complete'] and receipt['plan_sha256'] == a.plan_sha and sha(OUT/'protected_before.json.gz') == receipt['protected_sha256']
    with gzip.open(OUT/'protected_before.json.gz','rb') as stream: before = json.load(stream)
    assert canonical(before) == receipt['protected_canonical_sha256'] and protected(targets) == before, 'Protected assets changed'
    runtime = idle(targets); free_before = shutil.disk_usage(ROOT).free; removed = []
    with (OUT/'deletion_ledger.jsonl').open('x',encoding='utf-8',newline='\n') as ledger:
        for index,row in enumerate(rows):
            if index % 50 == 0: idle(targets)
            checked(row,metadata=True); Path(row['path']).unlink(); removed.append(row['path'])
            ledger.write(json.dumps(dict(path=row['path'],sha256=row['sha256'],local_backup=row['local_backup'],
                inode=row['inode'],nlink=row['nlink'],allocated_bytes=row['allocated_bytes'],backup_verified=True))+'\n')
            ledger.flush(); os.fsync(ledger.fileno())
    after = protected(targets); gzwrite('protected_after.json.gz',after)
    assert after == before, 'Retained research metadata or B4 hashes differ; preserve ledger'
    free_after = shutil.disk_usage(ROOT).free
    write(OUT/'cleanup_receipt.json',dict(complete=True,plan_sha256=a.plan_sha,worker_sha256=sha(__file__),
        files_removed=len(removed),removed_paths=removed,protected_metadata_entries=len(after['metadata']),
        retained_bank_hashes=len(after['bank_retained_bytes_sha256']),protected_canonical_before=canonical(before),
        protected_canonical_after=canonical(after),free_before_bytes=free_before,free_after_bytes=free_after,
        recovered_bytes=free_after-free_before,allocated_reclaim_bytes=sum(row['allocated_bytes'] for row in rows),
        all_scientific_cache_bytes_rehashed=False,checkpoints_inputs_splits_logs_metrics_provenance_and_failures_retained=True,
        runtime_before_removal=runtime,model_gradient_or_training_calls=0,VM_started=False,goal_complete=False,
        seconds=time.monotonic()-start))
    write(OUT/'manifest.json',dict(complete=True,plan_sha256=a.plan_sha,
        files_sha256={file.name:sha(file) for file in sorted(OUT.iterdir()) if file.is_file()}))
    import tarfile
    archive = HOME/'dgp-bank-B4-output-cleanup-v2-receipts.tar.gz'; assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in sorted(OUT.iterdir()): tar.add(file,arcname='receipts/'+file.name,recursive=False)
    write(HOME/'dgp-bank-B4-output-cleanup-v2-export.json',dict(complete=True,archive_sha256=sha(archive),
        bytes=archive.stat().st_size,model_gradient_or_training_calls=0))
    print(json.dumps(dict(complete=True,files_removed=len(removed),free_GiB=free_after/1024**3,seconds=time.monotonic()-start)),flush=True)


if __name__ == '__main__': main()

"""Finite read-only protected-state hashing for the explicitly scoped offload."""
from pathlib import Path
import gzip
import hashlib
import json
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/storage_offload_20261018_v1'


def require(value, message):
    if not value: raise ValueError(message)


def pin(path, budget):
    path = path.absolute()
    require(path.is_relative_to(ROOT), 'Protected path escaped Testing')
    for item in (path, *path.parents):
        if item == ROOT.parent: break
        s = item.lstat()
        require(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
                'Protected source has linked/reparse ancestor')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'Protected source must be regular')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024**2), b''):
            budget(); h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Protected file changed during read')
    return {'bytes': before.st_size, 'sha256': h.hexdigest(),
            'mtime_ns': before.st_mtime_ns, 'inode': before.st_ino,
            'nlink': before.st_nlink, 'dev': before.st_dev}


def main():
    started = time.monotonic()
    def budget(): require(time.monotonic()-started <= 900, 'Finite15minute protected hash cap')
    destination = OUT/'protected_before_v1.json'
    require(not destination.exists(), 'Preserve prior protected-state receipt')
    with gzip.open(OUT/'files_metadata_v1.json.gz', 'rt', encoding='utf-8') as f:
        metadata = json.load(f)
    proposed = json.loads((OUT/'cold_candidates_provisional_v1.json').read_bytes())
    names = {r['path'] for r in metadata if r.get('model_file')}
    names.update(r['path'] for r in proposed['protected_metadata_entries'])
    for candidate in proposed['candidate_archives']:
        base = ROOT/candidate['retained_extracted_group']['path']
        for folder in base.rglob('*'):
            if folder.is_file() and folder.suffix.lower() in ('.json','.md','.log','.txt','.py','.csv','.jsonl'):
                names.add(folder.relative_to(ROOT).as_posix())
    for directory in ('checkpoints','weights','models'):
        base = ROOT/directory
        if base.is_dir():
            names.update(p.relative_to(ROOT).as_posix() for p in base.rglob('*') if p.is_file())
    names.update(p.name for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in ('.md','.py','.json','.ps1','.bat'))
    candidates = {r['archive']['path'] for r in proposed['candidate_archives']}
    require(not names & candidates, 'Removal candidate overlaps protected assets')
    rows = {}
    journal = OUT/'protected_hash_progress_v1.jsonl'
    with journal.open('x',encoding='utf-8',newline='\n') as log:
        for n,name in enumerate(sorted(names),1):
            budget(); rows[name] = pin(ROOT/name,budget)
            log.write(json.dumps({'path':name,**rows[name]},separators=(',',':'))+'\n')
            if n%100==0:
                log.flush(); print({'protected_hashed':n,'of':len(names),'seconds':round(time.monotonic()-started,1)},flush=True)
    returned = 'outputs/cctv-codeformer-filter-fusion-training-v1-results.tar.gz'
    require(rows[returned]['sha256'] == '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16',
            'Latest return archive hash differs; preserve and investigate')
    result = {'complete':True,'protected':rows,'files':len(rows),
        'logical_bytes_hashed':sum(r['bytes'] for r in rows.values()),
        'all_inventory1022_model_files_included':True,
        'latest_return_full_hash_verified':True,'whole_cache_content_hash_claim':False,
        'all_other_retained_output_names_sizes_mtimes_require_after_inventory':True,
        'neural_gradient_optimizer_VM_or_delete_calls':0,'seconds':time.monotonic()-started}
    with destination.open('x',encoding='utf-8',newline='\n') as f: json.dump(result,f,indent=2);f.write('\n')
    print({k:v for k,v in result.items() if k!='protected'},flush=True)


if __name__=='__main__': main()

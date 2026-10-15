"""Read-only completion of directory listings blocked by the initial sandbox."""
from collections import defaultdict
from datetime import datetime, timezone
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local_storage_cleanup_20261011_v1'


def main():
    start = time.monotonic()
    target = OUT / 'complete_inventory_before.json'
    assert not target.exists()
    old = json.loads((OUT / 'inventory_before.json').read_text())
    with gzip.open(OUT / 'file_inventory_before.json.gz', 'rt', encoding='utf-8') as f:
        rows = json.load(f)
    names = {r['path'] for r in rows}
    blocked = []
    for message in old['errors']:
        match = re.search(r"'(.+)'$", message)
        assert match and 'Access is denied' in message, message
        path = Path(match[1].replace('\\\\', '\\')).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path != ROOT
        blocked.append(path)
    roots = [p for p in sorted(set(blocked)) if not any(q != p and p.is_relative_to(q) for q in blocked)]
    errors = []
    skipped = set(old['skipped_reparse_paths'])
    get_size = ctypes.windll.kernel32.GetCompressedFileSizeW
    get_size.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    get_size.restype = ctypes.c_ulong
    added = 0
    for root in roots:
        assert root.is_dir() and not root.is_symlink()
        for folder, dirs, files in os.walk(root, followlinks=False, onerror=lambda e:errors.append(str(e))):
            kept = []
            for name in dirs:
                path = Path(folder)/name
                if path.is_symlink() or getattr(path.lstat(),'st_file_attributes',0)&0x400:
                    skipped.add(path.relative_to(ROOT).as_posix())
                else:
                    kept.append(name)
            dirs[:] = kept
            for name in files:
                p = Path(folder)/name
                rel = p.relative_to(ROOT).as_posix()
                if rel in names:
                    continue
                s = p.lstat()
                if p.is_symlink() or getattr(s,'st_file_attributes',0)&0x400:
                    skipped.add(rel)
                    continue
                high = ctypes.c_ulong()
                low = get_size(str(p),ctypes.byref(high))
                assert low != 0xFFFFFFFF, rel
                rows.append(dict(path=rel, bytes=s.st_size, allocated_bytes=(high.value<<32)|low,
                    dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,
                    attributes=getattr(s,'st_file_attributes',0)))
                names.add(rel)
                added += 1
                assert time.monotonic()-start < 600
    totals = defaultdict(int)
    level2 = defaultdict(int)
    inodes = {}
    for row in rows:
        parts = Path(row['path']).parts
        totals[parts[0] if len(parts)>1 else '(root files)'] += row['bytes']
        level2['/'.join(parts[:2]) if len(parts)>1 else '(root files)'] += row['bytes']
        inodes[(row['dev'],row['inode'])] = row['allocated_bytes']
    with gzip.open(OUT/'complete_file_inventory_before.json.gz','xb',compresslevel=1) as f:
        f.write(json.dumps(rows).encode('utf-8'))
    record = dict(complete=not errors, UTC=datetime.now(timezone.utc).isoformat(),
        root=str(ROOT), initial_access_blocks=len(old['errors']), additional_files=added,
        original_partial_inventory_retained=True, file_count=len(rows), logical_bytes=sum(totals.values()),
        unique_allocated_bytes=sum(inodes.values()),disk_free_bytes=shutil.disk_usage(ROOT).free,
        directories_logical_bytes=dict(sorted(totals.items(),key=lambda r:-r[1])),
        two_level_logical_bytes=dict(sorted(level2.items(),key=lambda r:-r[1])),
        skipped_reparse_paths=sorted(skipped),errors=errors,files_removed=0,
        seconds=time.monotonic()-start)
    with target.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(record,indent=2)+'\n')
    print(dict(complete=record['complete'], additional_files=added,
        physical_GiB=record['unique_allocated_bytes']/1024**3, logical_GiB=record['logical_bytes']/1024**3,
        free_GiB=record['disk_free_bytes']/1024**3, files_removed=0),flush=True)


if __name__=='__main__':
    main()

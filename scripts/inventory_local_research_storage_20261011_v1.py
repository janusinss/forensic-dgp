"""Read-only local storage inventory. Never follows junctions or symlinks."""
from collections import defaultdict
from datetime import datetime, timezone
import ctypes
import gzip
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local_storage_cleanup_20261011_v1'


def main():
    assert not OUT.exists(), 'Keep any previous inventory unchanged'
    OUT.mkdir()
    started = time.monotonic()
    files = []
    errors = []
    reparse = []
    inode_bytes = {}
    logical = defaultdict(int)
    physical = defaultdict(int)
    level2 = defaultdict(int)
    get_size = ctypes.windll.kernel32.GetCompressedFileSizeW
    get_size.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    get_size.restype = ctypes.c_ulong
    for folder, dirs, names in os.walk(ROOT, followlinks=False, onerror=lambda e: errors.append(str(e))):
        kept = []
        for name in sorted(dirs):
            path = Path(folder) / name
            try:
                s = path.lstat()
                if getattr(s, 'st_file_attributes', 0) & 0x400 or path.is_symlink():
                    reparse.append(path.relative_to(ROOT).as_posix())
                else:
                    kept.append(name)
            except OSError as e:
                errors.append(str(e))
        dirs[:] = kept
        for name in sorted(names):
            path = Path(folder) / name
            try:
                s = path.lstat()
                rel = path.relative_to(ROOT).as_posix()
                if getattr(s, 'st_file_attributes', 0) & 0x400 or path.is_symlink():
                    reparse.append(rel)
                    continue
                parts = Path(rel).parts
                group = parts[0] if len(parts) > 1 else '(root files)'
                inode = (s.st_dev, s.st_ino)
                allocated = inode_bytes.get(inode)
                if allocated is None:
                    high = ctypes.c_ulong()
                    low = get_size(str(path), ctypes.byref(high))
                    allocated = (high.value << 32) | low
                    if allocated == 0xFFFFFFFF:
                        raise OSError('Allocated-size query failed: ' + rel)
                    inode_bytes[inode] = allocated
                    physical[group] += allocated
                logical[group] += s.st_size
                level2['/'.join(parts[:2]) if len(parts) > 1 else group] += s.st_size
                files.append(dict(path=rel, bytes=s.st_size, allocated_bytes=allocated,
                                  dev=s.st_dev, inode=s.st_ino, nlink=s.st_nlink,
                                  mtime_ns=s.st_mtime_ns, attributes=getattr(s, 'st_file_attributes', 0)))
                if len(files) % 50000 == 0:
                    print(dict(inventoried_files=len(files), seconds=round(time.monotonic()-started, 1)), flush=True)
                assert time.monotonic()-started < 600, 'Finite inventory time limit'
            except OSError as e:
                errors.append(str(e))
    result = dict(complete=not errors, UTC=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
                  disk_free_bytes=shutil.disk_usage(ROOT).free,
                  logical_bytes=sum(logical.values()), unique_allocated_bytes=sum(inode_bytes.values()),
                  file_count=len(files), unique_inode_count=len(inode_bytes),
                  directories_logical_bytes=dict(sorted(logical.items(), key=lambda r:-r[1])),
                  directories_unique_allocated_bytes=dict(sorted(physical.items(), key=lambda r:-r[1])),
                  two_level_logical_bytes=dict(sorted(level2.items(), key=lambda r:-r[1])),
                  largest_files=sorted(files, key=lambda r:-r['bytes'])[:80],
                  skipped_reparse_paths=reparse, errors=errors, files_removed=0,
                  VM_connection_or_training_calls=0, seconds=time.monotonic()-started)
    with (OUT / 'inventory_before.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    with gzip.open(OUT / 'file_inventory_before.json.gz', 'xb', compresslevel=1) as f:
        f.write(json.dumps(files).encode('utf-8'))
    print(dict(complete=result['complete'], logical_GiB=result['logical_bytes']/1024**3,
               unique_allocated_GiB=result['unique_allocated_bytes']/1024**3,
               free_GiB=result['disk_free_bytes']/1024**3, file_count=len(files),
               largest_directories_GiB={k:round(v/1024**3, 3) for k,v in list(logical.items())[:5]},
               errors=len(errors), seconds=result['seconds']), flush=True)


if __name__ == '__main__':
    main()

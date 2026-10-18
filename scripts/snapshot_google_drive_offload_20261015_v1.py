"""Capture regular-file namespace and metadata without following reparse points."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/google_drive_offload_20261015_v1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=['namespace_before', 'namespace_before_r1', 'namespace_after'], required=True)
    args = parser.parse_args()
    target = OUT / (args.name + '.json.gz')
    assert not target.exists()
    start = time.monotonic()
    rows, skipped, errors = {}, [], []
    pending = [ROOT]
    while pending:
        assert time.monotonic() - start < 600, 'Finite namespace limit'
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if path == OUT:
                        continue
                    try:
                        # Windows DirEntry.stat() omits inode/dev/nlink. Use a
                        # full lstat for the retained identity metadata.
                        info = path.lstat()
                        rel = path.relative_to(ROOT).as_posix()
                        if info.st_file_attributes & 0x400 or stat.S_ISLNK(info.st_mode):
                            skipped.append(rel)
                        elif stat.S_ISDIR(info.st_mode):
                            pending.append(path)
                        else:
                            assert stat.S_ISREG(info.st_mode), rel
                            rows[rel] = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                                             inode=info.st_ino, dev=info.st_dev, nlink=info.st_nlink,
                                             mode=info.st_mode, attributes=info.st_file_attributes)
                            if len(rows) % 100000 == 0:
                                print({'snapshot': args.name, 'files': len(rows),
                                       'seconds': round(time.monotonic() - start, 1)}, flush=True)
                    except OSError as error:
                        errors.append(str(error))
        except OSError as error:
            errors.append(str(error))
    with gzip.open(target, 'xb', compresslevel=1) as stream:
        stream.write(json.dumps(rows, separators=(',', ':')).encode())
    receipt = dict(complete=not errors, file_count=len(rows), errors=errors,
                   skipped_reparse_paths=skipped, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   disk_free_bytes=shutil.disk_usage(ROOT).free, seconds=time.monotonic()-start)
    (OUT / (args.name + '_receipt.json')).write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)
    assert not errors, 'Preserve partial scan; do not delete with an incomplete namespace'

if __name__ == '__main__':
    main()

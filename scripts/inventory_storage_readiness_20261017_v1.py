"""Bounded read-only Testing inventory; never transfers, deletes or runs models."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/storage_readiness_20261017_v1'


def put(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=False)
    free_before = shutil.disk_usage(ROOT).free
    folders, output_folders = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    extensions = Counter()
    errors, links, largest = [], [], []
    file_count = dir_count = total = 0
    stack = [ROOT]
    metadata = OUT / 'current_file_metadata.jsonl.gz'
    with metadata.open('xb') as binary:
        with gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0) as stream:
            while stack:
                if time.monotonic() - start > 300 or file_count > 2_000_000:
                    raise RuntimeError('Finite read-only inventory cap reached; retain partial evidence')
                folder = stack.pop()
                try:
                    with os.scandir(folder) as entries:
                        for entry in entries:
                            path = Path(entry.path)
                            if path == OUT:
                                continue
                            rel = path.relative_to(ROOT).as_posix()
                            try:
                                info = entry.stat(follow_symlinks=False)
                                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                                    links.append(rel)
                                    continue
                                if stat.S_ISDIR(info.st_mode):
                                    dir_count += 1
                                    stack.append(path)
                                elif stat.S_ISREG(info.st_mode):
                                    file_count += 1
                                    total += info.st_size
                                    parts = path.relative_to(ROOT).parts
                                    key = parts[0] if len(parts) > 1 else '<root files>'
                                    folders[key][0] += 1
                                    folders[key][1] += info.st_size
                                    if parts[0] == 'outputs' and len(parts) > 2:
                                        output_folders[parts[1]][0] += 1
                                        output_folders[parts[1]][1] += info.st_size
                                    extensions[path.suffix.lower()] += 1
                                    row = {'path': rel, 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                                           'inode': info.st_ino, 'nlink': info.st_nlink}
                                    stream.write((json.dumps(row, separators=(',', ':')) + '\n').encode())
                                    if info.st_size >= 256 * 1024**2:
                                        largest.append({'path': rel, 'bytes': info.st_size})
                            except OSError as exc:
                                errors.append({'path': rel, 'error': type(exc).__name__})
                except OSError as exc:
                    errors.append({'path': str(folder.relative_to(ROOT)), 'error': type(exc).__name__})
    def groups(values):
        return [{'directory': key, 'files': pair[0], 'logical_bytes': pair[1]}
                for key, pair in sorted(values.items(), key=lambda row: row[1][1], reverse=True)]
    report = {'complete': not errors, 'UTC': datetime.now(timezone.utc).isoformat(),
        'scope': str(ROOT), 'read_only_inventory': True,
        'self_evidence_directory_excluded': str(OUT.relative_to(ROOT)),
        'files': file_count, 'directories': dir_count, 'logical_file_bytes': total,
        'directory_groups': groups(folders), 'output_directory_groups': groups(output_folders),
        'extensions': dict(extensions), 'files_at_least256MiB': sorted(largest, key=lambda row: row['bytes'], reverse=True),
        'reparse_or_symlink_entries_not_followed': links, 'errors': errors,
        'file_metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
        'file_metadata_bytes': metadata.stat().st_size,
        'free_bytes_before': free_before, 'free_bytes_after': shutil.disk_usage(ROOT).free,
        'no_free_space_change_claim_from_read_only_inventory': True,
        'newly_transferred_files': 0, 'removed_files': 0, 'reclaimed_bytes': 0,
        'full_project_removal_eligibility_not_established': True,
        'local_neural_gradient_optimizer_VM_calls': 0,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seconds': time.monotonic() - start}
    put(OUT / 'inventory.json', report)
    print(json.dumps({key: report[key] for key in ('complete', 'files', 'directories', 'logical_file_bytes',
        'file_metadata_bytes', 'free_bytes_after', 'removed_files', 'seconds')}))


if __name__ == '__main__':
    main()

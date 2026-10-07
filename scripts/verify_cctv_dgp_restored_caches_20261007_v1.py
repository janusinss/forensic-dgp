"""Verify restored cache bytes without loading a model or changing research files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import time


def sha(path, deadline):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            if time.monotonic() > deadline:
                raise TimeoutError('Finite restored-cache audit stop')
            digest.update(block)
    return digest.hexdigest()


def verify(manifest, expected_sha, root, receipt, max_seconds):
    started = time.monotonic()
    deadline = started + max_seconds
    assert 1 <= max_seconds <= 3600
    assert sha(manifest, deadline) == expected_sha, 'Manifest SHA256 differs'
    data = json.loads(manifest.read_text(encoding='utf-8-sig'))
    assert data['complete'] and data['file_count'] == 4431
    rows = data['files']
    assert len(rows) == 4431 and len({row['path'] for row in rows}) == 4431
    assert sum(row['bytes'] for row in rows) == 39448585279
    root = root.resolve(strict=True)
    assert root.is_dir() and not receipt.exists(), 'Preserve prior receipts'
    for index, row in enumerate(rows, 1):
        original = PurePosixPath(row['path'])
        assert original.is_absolute() and '..' not in original.parts
        assert str(original).startswith('/home/janusdominic0/forensic-dgp/')
        candidate = root.joinpath(*original.relative_to('/').parts)
        assert not candidate.is_symlink()
        resolved = candidate.resolve(strict=True)
        assert os.path.commonpath([str(root), str(resolved)]) == str(root)
        before = resolved.stat()
        assert resolved.is_file() and before.st_size == row['bytes'], (
            'Missing or incomplete restored file: ' + row['path'])
        assert sha(resolved, deadline) == row['sha256'], (
            'Restored file SHA256 differs: ' + row['path'])
        after = resolved.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        if index == 1 or index % 500 == 0 or index == len(rows):
            print(json.dumps({'verified_files': index, 'total_files': len(rows),
                              'seconds': time.monotonic() - started}), flush=True)
    result = {'complete': True, 'hostname': socket.gethostname(),
              'root': str(root), 'verified_cache_files': len(rows),
              'verified_cache_bytes': data['bytes'], 'manifest_sha256': expected_sha,
              'seconds': time.monotonic() - started, 'training_started': False,
              'research_files_modified': 0}
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    verify(args.manifest, args.manifest_sha, args.root, args.receipt, args.max_seconds)

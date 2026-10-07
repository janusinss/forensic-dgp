"""Read-only framed cache transfer. No temporary tensor copies on the source VM."""
import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import sys
import time

MAGIC = b'DGPCBK01'
TOTAL = 39448585279
CHUNK = 256 * 1024 ** 2


def frame(data):
    encoded = json.dumps(data, separators=(',', ':')).encode()
    sys.stdout.buffer.write(MAGIC + struct.pack('!I', len(encoded)) + encoded)
    sys.stdout.buffer.flush()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def stable(row):
    path = Path(row['path'])
    assert str(path).startswith('/home/janusdominic0/forensic-dgp/')
    assert not path.is_symlink() and path.resolve() == path and path.is_file()
    stamp = path.stat()
    assert (stamp.st_size, stamp.st_ino, stamp.st_mtime_ns) == (
        row['bytes'], row['inode'], row['mtime_ns']), 'Source cache changed'
    return path


def serve(manifest, manifest_sha, seconds):
    assert os.uname().nodename == 'forensic-dgp-thesis'
    assert 1 <= seconds <= 14400
    signal.alarm(seconds)
    assert digest(manifest) == manifest_sha
    data = json.loads(manifest.read_text())
    rows = data['files']
    assert data['complete'] and len(rows) == 4431
    assert sum(row['bytes'] for row in rows) == TOTAL
    roots = data['roots']
    assert all(any(row['path'].startswith(root + '/') for root in roots) for row in rows)
    offsets = [0]
    for row in rows:
        stable(row)
        offsets.append(offsets[-1] + row['bytes'])
    frame({'op': 'hello', 'manifest_sha256': manifest_sha, 'bytes': TOTAL, 'files': len(rows)})
    for unused in range(200):
        raw = sys.stdin.buffer.readline(4096)
        if not raw:
            return
        if not raw.strip():
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            print('Unexpected transfer control line: ' + repr(raw[:100]), file=sys.stderr, flush=True)
            raise
        if request['op'] == 'close':
            frame({'op': 'closed'})
            return
        if request['op'] == 'prefix':
            count = request['bytes']
            assert 0 < count <= rows[0]['bytes'] and count <= 4 * 1024 ** 3
            assert Path(rows[0]['path']).name == 'features.bin'
            h = hashlib.sha256()
            path = stable(rows[0])
            with path.open('rb') as stream:
                remaining = count
                while remaining:
                    block = stream.read(min(1024 ** 2, remaining))
                    assert block
                    h.update(block)
                    remaining -= len(block)
            stable(rows[0])
            frame({'op': 'prefix', 'bytes': count, 'sha256': h.hexdigest()})
            continue
        assert request['op'] == 'chunk'
        start, count = request['start'], request['bytes']
        assert isinstance(start, int) and isinstance(count, int)
        assert 0 <= start < TOTAL and 0 < count <= CHUNK and start + count <= TOTAL
        frame({'op': 'chunk', 'start': start, 'bytes': count})
        h = hashlib.sha256()
        position = start
        stop = start + count
        index = bisect.bisect_right(offsets, position) - 1
        while position < stop:
            row = rows[index]
            path = stable(row)
            amount = min(stop - position, offsets[index + 1] - position)
            with path.open('rb') as stream:
                stream.seek(position - offsets[index])
                remaining = amount
                while remaining:
                    block = stream.read(min(1024 ** 2, remaining))
                    assert block, 'Unexpected source EOF'
                    h.update(block)
                    sys.stdout.buffer.write(block)
                    remaining -= len(block)
                    position += len(block)
            stable(row)
            index += 1
        sys.stdout.buffer.flush()
        frame({'op': 'receipt', 'start': start, 'bytes': count, 'sha256': h.hexdigest()})
    raise RuntimeError('Finite transfer request limit reached')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--seconds', type=int, default=14400)
    args = parser.parse_args()
    serve(args.manifest, args.manifest_sha, args.seconds)

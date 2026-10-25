"""Manual exclusive copy of the exact complete U96 initializer, no framework."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        expected = stat.S_ISDIR(info.st_mode) if item != path or directory else stat.S_ISREG(info.st_mode)
        if not expected or stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('Ordinary nonlinked install ancestry required')
        if item == path and not directory and (info.st_nlink != 1 or not 0 < info.st_size < 2 * 1024**3):
            raise ValueError('Bounded single-link install file required')
    return path.resolve(strict=True)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate install JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON install constant')
    result = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(value):
        if type(value) is float and not math.isfinite(value):
            raise ValueError('Finite install JSON values required')
        if type(value) is dict:
            for child in value.values():
                finite(child)
        if type(value) is list:
            for child in value:
                finite(child)
    finite(result)
    return result


def _install(root, source, expected_sha):
    start = time.monotonic()
    root = Path(root).absolute()
    if platform.system() != 'Linux':
        raise ValueError('Install manually in VM SSH; no local model loading')
    ordinary(root, directory=True)
    if root != Path.home() / 'forensic-dgp/cctv_codeformer_u96_decoder_readout_v1_vm':
        raise ValueError('Exact fresh manual install stage required')
    raw = ordinary(root / 'protocol.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Exact supplied protocol SHA required')
    protocol = strict_json(raw)
    active = ordinary(__file__)
    if active != root / 'research/codeformer_u96_decoder_readout_v1/install_weights.py':
        raise ValueError('Installer must execute its own staged source')
    own_pin = protocol['files'][active.relative_to(root).as_posix()]
    if own_pin != {'bytes': active.stat().st_size, 'sha256': digest(active)}:
        raise ValueError('Sealed active installer differs')
    helper = ordinary(root / 'research/codeformer_u96_decoder_readout_v1/packet.py')
    pin = protocol['files'][helper.relative_to(root).as_posix()]
    if helper.stat().st_size != pin['bytes'] or digest(helper) != pin['sha256']:
        raise ValueError('Sealed packet helper differs')
    spec = importlib.util.spec_from_file_location('verified_readout_packet', helper)
    packet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packet)
    root = packet.directory(root)
    packet.need(root == packet.directory(Path.home() / 'forensic-dgp' / packet.STAGE_NAME),
                'Exact new manual stage required')
    p = packet.verify_static(root, expected_sha, missing_weights_for_install=True)
    packet.need(not any((root / name).exists() for name in
                ('install.json', 'install_failure.json', 'outputs', 'run.log', 'supervisor.json')),
                'Existing install/run immutable; preserve partial files')
    packet.need(all(not (root / row['path']).exists() for row in p['model'].values()),
                'Never overwrite existing initializer files')
    source = packet.directory(source)
    parent = packet.directory(Path.home() / 'forensic-dgp')
    packet.need(source.is_relative_to(parent) and source != root,
                'Initializer source must be another retained forensic-dgp stage')
    sources = {key: packet.regular(source / row['path']) for key, row in p['model'].items()}
    for key, path in sources.items():
        row = p['model'][key]
        packet.need(path.stat().st_size == row['bytes'] and packet.sha256(path) == row['sha256'],
                    'Retained initializer content differs: ' + key)
    required = sum(row['bytes'] for row in p['model'].values()) + p['limits']['initial_free_disk_bytes']
    packet.need(shutil.disk_usage(root).free >= required, 'Leave8GiB free after exclusive model copies')
    copies = []
    for key, src in sources.items():
        row = p['model'][key]
        target = root / row['path']
        target.parent.mkdir(exist_ok=True)
        packet.directory(target.parent)
        with src.open('rb') as original, target.open('xb') as output:
            before = os.fstat(original.fileno())
            remaining = row['bytes']
            while remaining:
                packet.need(time.monotonic() - start < 180, 'Finite180s installation cap')
                packet.need(shutil.disk_usage(root).free >= p['limits']['initial_free_disk_bytes'],
                            'Keep8GiB free throughout install')
                chunk = original.read(min(1024**2, remaining))
                packet.need(bool(chunk), 'Initializer truncated while copying')
                output.write(chunk)
                remaining -= len(chunk)
            packet.need(original.read(1) == b'', 'Initializer grew; refuse suffix')
            after = os.fstat(original.fileno())
            packet.need((before.st_ino, before.st_mtime_ns, before.st_size) ==
                        (after.st_ino, after.st_mtime_ns, after.st_size), 'Source changed during copy')
            output.flush()
            os.fsync(output.fileno())
        packet.need(packet.pin(root, target) == row and packet.sha256(src) == row['sha256'],
                    'Exclusive copy/source readback differs')
        copies.append({'key': key, 'source': str(src), 'installed': row})
    packet.verify_static(root, expected_sha)
    result = {'complete': True, 'protocol_sha256': expected_sha, 'copies': copies,
              'seconds': time.monotonic() - start, 'neural_gradient_optimizer_training_calls': 0}
    with (root / 'install.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


def install(root, source, expected_sha):
    if platform.system() != 'Linux':
        raise ValueError('Install manually in VM SSH')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError('Refuse replacing an active installation timer')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('Absolute180s install stop includes bootstrap, copy and readback')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 180)
    try:
        return _install(root, source, expected_sha)
    except Exception as exc:
        path = Path(root).absolute()
        if path == Path.home() / 'forensic-dgp/cctv_codeformer_u96_decoder_readout_v1_vm':
            ordinary(path, directory=True)
            failure = path / 'install_failure.json'
            if not failure.exists():
                with failure.open('x', encoding='utf-8', newline='\n') as stream:
                    json.dump({'complete': False, 'error_type': type(exc).__name__, 'error': str(exc)[:4000],
                               'protocol_sha256': expected_sha, 'partial_files_retained': True,
                               'neural_optimizer_training_calls': 0}, stream, indent=2, allow_nan=False)
                    stream.write('\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    install(args.root, args.source_root, args.protocol_sha)

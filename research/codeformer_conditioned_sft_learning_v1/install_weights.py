"""Manual exclusive byte copy of complete U96 initializers; no framework load."""
import argparse
import hashlib
import importlib
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
STAGE = 'cctv_codeformer_conditioned_sft_learning_v1_vm'
PACKAGE = 'research/codeformer_conditioned_sft_learning_v1'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        expected = stat.S_ISDIR(info.st_mode) if item != path or directory else stat.S_ISREG(info.st_mode)
        if not expected or stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('Ordinary nonlinked install ancestry required')
        if item == path and not directory and (info.st_nlink != 1 or not 0 < info.st_size <= 2 * 1024**3):
            raise ValueError('Bounded single-link install file required')
    return path.resolve(strict=True)


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate install JSON key')
            value[key] = item
        return value
    def invalid(_):
        raise ValueError('Nonfinite install JSON constant')
    result = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(item):
        if type(item) is float and not math.isfinite(item):
            raise ValueError('Nonfinite install JSON number')
        if type(item) is dict:
            for child in item.values():
                finite(child)
        elif type(item) is list:
            for child in item:
                finite(child)
    finite(result)
    return result


def copy_verified(source, destination, binding, guard):
    """Exclusive miniature-testable streaming copy; retain any partial failure."""
    source = ordinary(source)
    ordinary(destination.parent, directory=True)
    if source.stat().st_size != binding['bytes'] or digest(source) != binding['sha256']:
        raise ValueError('Retained initializer differs before copy')
    with source.open('rb') as original, destination.open('xb') as output:
        before = os.fstat(original.fileno())
        remaining = binding['bytes']
        while remaining:
            guard()
            chunk = original.read(min(1024**2, remaining))
            if not chunk:
                raise ValueError('Initializer truncated during copy')
            output.write(chunk)
            remaining -= len(chunk)
        if original.read(1):
            raise ValueError('Initializer grew during copy')
        after = os.fstat(original.fileno())
        if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
            raise ValueError('Initializer changed during copy')
        output.flush()
        os.fsync(output.fileno())
    guard()
    if ordinary(destination).stat().st_size != binding['bytes'] or digest(destination) != binding['sha256']:
        raise ValueError('Installed initializer content differs')
    if digest(source) != binding['sha256']:
        raise ValueError('Retained initializer changed after copy')


def authenticate_bootstrap(root, protocol):
    """Bind every package initializer before the first staged package import."""
    for relative in ('research/__init__.py', PACKAGE + '/__init__.py',
                     PACKAGE + '/packet.py', PACKAGE + '/install_weights.py'):
        path = ordinary(root / relative)
        if protocol['files'].get(relative) != {'bytes': path.stat().st_size, 'sha256': digest(path)}:
            raise ValueError('Sealed bootstrap source differs: ' + relative)


def _install(root, source, expected_sha):
    start = time.monotonic()
    if platform.system() != 'Linux':
        raise ValueError('Install manually in VM SSH')
    root = ordinary(root, directory=True)
    if root != Path.home() / 'forensic-dgp' / STAGE:
        raise ValueError('Exact fresh manual install stage required')
    raw = ordinary(root / 'protocol.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Exact supplied protocol SHA required')
    p = strict_json(raw)
    active = ordinary(__file__)
    if active != root / PACKAGE / 'install_weights.py':
        raise ValueError('Installer must use its own staged source')
    authenticate_bootstrap(root, p)
    sys.path.insert(0, str(root))
    packet = importlib.import_module('research.codeformer_conditioned_sft_learning_v1.packet')
    if Path(packet.__file__).resolve() != root / PACKAGE / 'packet.py':
        raise ValueError('Active packet helper belongs to another stage')
    p = packet.verify_static(root, expected_sha, missing_weights_for_install=True)
    packet.need(not any((root / name).exists() for name in
                ('install.json', 'install_failure.json', 'outputs', 'run.log', 'supervisor.start.json', 'supervisor.json')),
                'Prior install/run immutable; preserve partial files')
    packet.need(all(not (root / value['path']).exists() for value in p['model'].values()),
                'Never overwrite existing initializer files')
    source = ordinary(source, directory=True)
    parent = ordinary(Path.home() / 'forensic-dgp', directory=True)
    packet.need(source.is_relative_to(parent) and source != root,
                'Initializer source must be another retained forensic-dgp stage')
    sources = {key: ordinary(source / value['path']) for key, value in p['model'].items()}
    required = sum(value['bytes'] for value in p['model'].values()) + p['limits']['initial_free_disk_bytes']
    packet.need(shutil.disk_usage(root).free >= required, 'Leave32GiB free after initializer copies')
    def guard():
        packet.need(time.monotonic() - start < 180, 'Finite180s installation cap')
        packet.need(shutil.disk_usage(root).free >= p['limits']['initial_free_disk_bytes'],
                    'Keep32GiB free during install')
    copied = []
    for key, source_file in sources.items():
        value = p['model'][key]
        target = root / value['path']
        target.parent.mkdir(exist_ok=True)
        packet.directory(target.parent)
        copy_verified(source_file, target, value, guard)
        packet.need(packet.pin(root, target) == value, 'Installed initializer pin differs')
        copied.append({'key': key, 'source': str(source_file), 'installed': value})
    packet.verify_static(root, expected_sha)
    result = {'complete': True, 'protocol_sha256': expected_sha, 'copies': copied,
              'seconds': time.monotonic() - start, 'checkpoint_deserializations': 0,
              'neural_gradient_optimizer_training_calls': 0, 'model_qualified': False}
    with (root / 'install.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(result), flush=True)
    return result


def install(root, source, expected_sha):
    if platform.system() != 'Linux':
        raise ValueError('Install manually in VM SSH')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError('Refuse replacing an active installation timer')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('Absolute180s install stop includes bootstrap/copy/readback')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 180)
    try:
        return _install(root, source, expected_sha)
    except Exception as exc:
        path = Path(root).absolute()
        if path == Path.home() / 'forensic-dgp' / STAGE:
            ordinary(path, directory=True)
            failure = path / 'install_failure.json'
            if not failure.exists():
                with failure.open('x', encoding='utf-8', newline='\n') as stream:
                    json.dump({'complete': False, 'error_type': type(exc).__name__, 'error': str(exc)[:4000],
                               'protocol_sha256': expected_sha, 'partial_files_retained': True,
                               'checkpoint_deserializations': 0, 'neural_optimizer_training_calls': 0},
                              stream, indent=2, allow_nan=False)
                    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path,
                        default=Path.home() / 'forensic-dgp/cctv_codeformer_matched_feature_learning_v1_vm')
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    install(args.root, args.source_root, args.protocol_sha)


if __name__ == '__main__':
    main()

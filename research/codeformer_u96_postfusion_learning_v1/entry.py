"""Inert transfer CLI and bounded manual worker/export supervision."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from . import packet

PREFIX = 'cctv-codeformer-u96-postfusion-learning-v1'
EXPORT_SECONDS = 330
EXPORT_EXTERNAL_SECONDS = 345
EXPORT_BYTES = 1536 * 1024**2


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def status(root):
    path = root / packet.OUTPUT_RELATIVE / 'status.json'
    pending = (root / packet.OUTPUT_RELATIVE / 'status.pending.json').exists()
    if not path.exists():
        return {'status_receipt_present': False, 'optimizer_updates': None,
                'confirmed_optimizer_updates': None, 'optimizer_update_attempts': None,
                'optimizer_outcome_unknown': True, 'interrupted_status_write_present': pending,
                'last_committed_checkpoint': None, 'active_update': None, 'active_slot': None}
    value = packet.strict_json(packet.regular(path).read_bytes())
    count = value['counters']['optimizer_updates']
    attempts = value['counters']['optimizer_update_attempts']
    packet.need(type(count) is int and 0 <= count <= 64, 'Durable committed-update receipt required')
    packet.need(type(attempts) is int and count <= attempts <= 64, 'Durable optimizer-attempt receipt required')
    return {'status_receipt_present': True, 'optimizer_updates': count,
            'confirmed_optimizer_updates': count, 'optimizer_update_attempts': attempts,
            'optimizer_outcome_unknown': bool(value['optimizer_outcome_unknown'] or pending or attempts > count),
            'interrupted_status_write_present': pending,
            'active_update': value.get('active_update'), 'active_slot': value.get('active_slot'),
            'last_committed_checkpoint': value.get('last_committed_checkpoint'),
            'status_receipt': packet.pin(root, path)}


def export(root, p, expected_sha):
    packet.verify(root, expected_sha)
    packet.need(platform.system() == 'Linux', 'Export runs manually on VM')
    archive = Path.home() / (PREFIX + '-results.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    receipt = Path.home() / (PREFIX + '-export.json')
    packet.need(not any(path.exists() for path in (archive, checksum, receipt, root / 'export.json')),
                'Existing exports are immutable; preserve any partial archive')
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_dir():
            packet.directory(path)
            continue
        path = packet.regular(path)
        if path.relative_to(root).as_posix().startswith('weights/'):
            continue
        files.append(path)
    total = sum(path.stat().st_size for path in files)
    packet.need(total <= EXPORT_BYTES, 'Return payload exceeds1536MiB')
    start = time.monotonic()
    def guard():
        packet.need(time.monotonic() - start < EXPORT_SECONDS, 'Finite330s export stop')
        packet.need(shutil.disk_usage(root).free >= p['limits']['continuing_disk_reserve'],
                    'Continuing4GiB export reserve required')
    guard()
    class Limited:
        def __init__(self, stream):
            self.stream = stream
            self.count = 0
        def write(self, data):
            guard()
            self.count += len(data)
            packet.need(self.count <= EXPORT_BYTES, 'Return archive exceeds1536MiB')
            return self.stream.write(data)
        def flush(self):
            return self.stream.flush()
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=Limited(raw), mode='wb', compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for path in files:
                    guard()
                    tar.add(path, arcname=p['stage_name'] + '/' + path.relative_to(root).as_posix(), recursive=False)
    guard()
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            guard()
            digest.update(block)
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest.hexdigest() + '  ' + archive.name + '\n')
    failure_path = root / packet.OUTPUT_RELATIVE / 'failure.json'
    failure = packet.strict_json(packet.regular(failure_path).read_bytes()) if failure_path.exists() else None
    result = {'complete': True, 'archive_sha256': digest.hexdigest(),
              'bytes': archive.stat().st_size, 'payload_files': len(files), 'payload_bytes': total,
              'seconds': time.monotonic() - start, 'protocol_sha256': expected_sha,
              **status(root),
              'result_present': (root / packet.OUTPUT_RELATIVE / 'result.json').is_file(),
              'failure_present': failure is not None,
              'quality_stop_recorded': failure is not None and failure.get('status') == 'QUALITY_STOP',
              'training_success_not_implied': True, 'model_qualified': False}
    write_new(root / 'export.json', result)
    write_new(receipt, result)
    print(json.dumps(result), flush=True)


def supervise(root, p, expected_sha):
    packet.require_vm(root, p)
    packet.need(not any((root / name).exists() for name in ('run.log', 'supervisor.json')),
                'Prior launch is immutable')
    command = [sys.executable, '-B', '-u', str(root / 'scripts/run_u96_postfusion_learning_v1.py'),
               '--root', str(root), '--protocol-sha', expected_sha, '--run']
    start = time.monotonic()
    timeout = False
    with (root / 'run.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        print(json.dumps({'child_pid': child.pid, 'log': str(root / 'run.log'),
                          'external_cap_seconds': p['limits']['external_seconds']}), flush=True)
        try:
            code = child.wait(timeout=p['limits']['external_seconds'])
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            code = 124
    write_new(root / 'supervisor.json', {'complete': True, 'worker_exit': code, 'timeout': timeout,
              'seconds': time.monotonic() - start, 'protocol_sha256': expected_sha, **status(root),
              'training_success_not_implied': True, 'model_qualified': False})
    command[-1] = '--export'
    try:
        export_exit = subprocess.run(command, cwd=str(root), timeout=EXPORT_EXTERNAL_SECONDS,
                                     check=False).returncode
    except subprocess.TimeoutExpired:
        export_exit = 124
    print(json.dumps({'worker_exit': code, 'export_exit': export_exit, **status(root),
                      'model_qualified': False}), flush=True)
    return 0 if code == export_exit == 0 else (code or export_exit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('verify-transfer', 'run', 'supervise', 'export'):
        group.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    root = packet.directory(args.root)
    p = packet.verify(root, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps({'complete': True, 'static_files': len(p['files']), 'cases': 96,
                          'legacy_guards': 24, 'neural_gradient_optimizer_training_calls': 0,
                          'actual_L4_training_and_quality_verified': False}), flush=True)
        return 0
    if args.run:
        from . import worker
        worker.run(root, p, args.protocol_sha)
        return 0
    if args.supervise:
        return supervise(root, p, args.protocol_sha)
    export(root, p, args.protocol_sha)
    return 0

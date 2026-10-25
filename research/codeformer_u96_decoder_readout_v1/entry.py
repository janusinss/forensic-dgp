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

PREFIX = 'cctv-codeformer-u96-decoder-readout-v1'
EXPORT_SECONDS = 300
EXPORT_EXTERNAL_SECONDS = 330
EXPORT_BYTES = 2 * 1024**3


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _checked_status(root, expected_sha):
    path = root / packet.OUTPUT_RELATIVE / 'status.json'
    pending = (root / packet.OUTPUT_RELATIVE / 'status.pending.json').exists()
    if not path.exists():
        return {'status_receipt_present': False, 'optimizer_updates': None,
                'confirmed_optimizer_updates': None, 'optimizer_update_attempts': None,
                'optimizer_outcome_unknown': True, 'interrupted_status_write_present': pending,
                'last_committed_checkpoint': None, 'active_update': None, 'active_slot': None}
    value = packet.strict_json(packet.regular(path).read_bytes())
    packet.need(value['protocol_sha256'] == expected_sha, 'Status protocol differs')
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


def status(root, expected_sha):
    """A corrupt receipt remains evidence; never invent zero committed updates."""
    try:
        answer = _checked_status(root, expected_sha)
        answer['status_schema_and_protocol_verified'] = answer['status_receipt_present']
        checkpoint = answer['last_committed_checkpoint']
        if checkpoint is not None:
            packet.relative(checkpoint['path'])
            packet.need(checkpoint['path'].startswith(packet.OUTPUT_RELATIVE + '/checkpoints/')
                        and checkpoint == packet.pin(root, root / checkpoint['path']),
                        'Last checkpoint binding differs')
            update = packet.strict_json((root / checkpoint['path']).read_bytes())['update']
            packet.need(type(update) is int and 0 <= update <= answer['confirmed_optimizer_updates'],
                        'Checkpoint cannot exceed committed updates')
            answer['last_checkpoint_update'] = update
        return answer
    except Exception as exc:
        answer = {'status_receipt_present': (root / packet.OUTPUT_RELATIVE / 'status.json').exists(),
                  'confirmed_optimizer_updates': None, 'optimizer_updates': None,
                  'optimizer_update_attempts': None, 'optimizer_outcome_unknown': True,
                  'status_schema_and_protocol_verified': False,
                  'error_type': type(exc).__name__, 'status_read_error': str(exc)[:2000],
                  'interrupted_status_write_present': (root / packet.OUTPUT_RELATIVE / 'status.pending.json').exists(),
                  'last_committed_checkpoint': None}
        try:
            answer['status_receipt'] = packet.pin(root, root / packet.OUTPUT_RELATIVE / 'status.json')
        except (OSError, ValueError):
            pass
        return answer


def _export(root, expected_sha, start):
    def guard():
        packet.need(time.monotonic() - start < EXPORT_SECONDS, 'Finite300s export stop')
        packet.need(shutil.disk_usage(root).free >= packet.LIMITS['continuing_disk_reserve'],
                    'Continuing4GiB export reserve required')
    guard()
    p = packet.verify_static(root, expected_sha)
    guard()
    packet.need(platform.system() == 'Linux', 'Export runs manually on VM')
    archive = Path.home() / (PREFIX + '-results.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    receipt = Path.home() / (PREFIX + '-export.json')
    packet.need(not any(path.exists() for path in (archive, checksum, receipt, root / 'export.json')),
                'Existing exports are immutable; preserve any partial archive')
    files = []
    for path in sorted(root.rglob('*')):
        guard()
        if path.is_dir():
            packet.directory(path)
            continue
        path = packet.regular(path)
        if path.relative_to(root).as_posix().startswith('weights/'):
            continue
        files.append(packet.pin(root, path))
    total = sum(row['bytes'] for row in files)
    packet.need(total <= EXPORT_BYTES, 'Return payload exceeds2GiB')
    guard()
    payload_bytes_read = 0
    class BoundSource:
        def __init__(self, stream):
            self.stream = stream
            self.digest = hashlib.sha256()
            self.count = 0
        def read(self, size=-1):
            nonlocal payload_bytes_read
            guard()
            data = self.stream.read(size)
            self.count += len(data)
            payload_bytes_read += len(data)
            packet.need(payload_bytes_read <= EXPORT_BYTES, 'Read payload exceeds2GiB')
            self.digest.update(data)
            return data
    class Limited:
        def __init__(self, stream):
            self.stream = stream
            self.count = 0
        def write(self, data):
            guard()
            self.count += len(data)
            packet.need(self.count <= EXPORT_BYTES, 'Return archive exceeds2GiB')
            return self.stream.write(data)
        def flush(self):
            return self.stream.flush()
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=Limited(raw), mode='wb', compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for row in files:
                    guard()
                    path = packet.regular(root / row['path'])
                    info = tar.gettarinfo(str(path), arcname=p['stage_name'] + '/' + row['path'])
                    packet.need(info.isfile() and info.size == row['bytes'], 'Payload changed before read')
                    with path.open('rb') as original:
                        source = BoundSource(original)
                        tar.addfile(info, source)
                    packet.need(source.count == row['bytes'] and source.digest.hexdigest() == row['sha256'],
                                'Payload changed during read; retain partial archive')
                    packet.need(packet.pin(root, path) == row, 'Payload changed after read')
    guard()
    packet.need(payload_bytes_read == total, 'Payload read count differs from inventory')
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            guard()
            digest.update(block)
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest.hexdigest() + '  ' + archive.name + '\n')
    failure_path = root / packet.OUTPUT_RELATIVE / 'failure.json'
    failure = None
    failure_read_error = None
    if failure_path.exists():
        try:
            failure = packet.strict_json(packet.regular(failure_path).read_bytes())
            packet.need(type(failure) is dict, 'Failure receipt object required')
        except Exception as exc:
            failure = None
            failure_read_error = str(exc)[:2000]
    result = {'complete': True, 'archive_sha256': digest.hexdigest(),
              'bytes': archive.stat().st_size, 'payload_files': len(files), 'payload_bytes': total,
              'seconds': time.monotonic() - start, 'protocol_sha256': expected_sha,
              **status(root, expected_sha),
              'result_present': (root / packet.OUTPUT_RELATIVE / 'result.json').is_file(),
              'failure_present': failure_path.exists(), 'failure_read_error': failure_read_error,
              'quality_stop_recorded': failure is not None and failure.get('status') == 'QUALITY_STOP',
              'training_success_not_implied': True, 'model_qualified': False}
    write_new(root / 'export.json', result)
    write_new(receipt, result)
    print(json.dumps(result), flush=True)


def export(root, expected_sha):
    start = time.monotonic()
    packet.need(platform.system() == 'Linux', 'Export manually on VM')
    failure = root / 'export_failure.json'
    packet.need(not failure.exists(), 'Prior failed export immutable; do not retry')
    packet.need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'Refuse replacing active export timer')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('Absolute300s export stop includes verification and inventory')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, EXPORT_SECONDS)
    try:
        return _export(root, expected_sha, start)
    except Exception as exc:
        write_new(failure, {'complete': False, 'protocol_sha256': expected_sha,
                  'error_type': type(exc).__name__, 'error': str(exc)[:4000],
                  'seconds': time.monotonic() - start, **status(root, expected_sha),
                  'training_success_not_implied': True, 'model_qualified': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def supervise(root, p, expected_sha):
    packet.require_vm(root)
    packet.need(not any((root / name).exists() for name in ('run.log', 'supervisor.json')),
                'Prior launch is immutable')
    command = [sys.executable, '-B', '-u', str(root / 'scripts/run_u96_decoder_readout_v1.py'),
               '--root', str(root), '--protocol-sha', expected_sha, '--run']
    start = time.monotonic()
    timeout = False
    with (root / 'run.log').open('xb') as log:
        child_env = dict(os.environ, READOUT_SUPERVISED_PROTOCOL=expected_sha,
                         READOUT_SUPERVISOR_PID=str(os.getpid()))
        child = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=child_env)
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
              'seconds': time.monotonic() - start, 'protocol_sha256': expected_sha, **status(root, expected_sha),
              'training_success_not_implied': True, 'model_qualified': False})
    command[-1] = '--export'
    try:
        export_exit = subprocess.run(command, cwd=str(root), timeout=EXPORT_EXTERNAL_SECONDS,
                                     check=False).returncode
    except subprocess.TimeoutExpired:
        export_exit = 124
    print(json.dumps({'worker_exit': code, 'export_exit': export_exit, **status(root, expected_sha),
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
    packet.need(Path(__file__).resolve() == root / 'research/codeformer_u96_decoder_readout_v1/entry.py',
                'CLI must import its own staged entry source')
    if args.export:
        export(root, args.protocol_sha)
        return 0
    p = packet.verify(root, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps({'complete': True, 'static_files': len(p['files']), 'cases': 96,
                          'legacy_guards': 24, 'neural_gradient_optimizer_training_calls': 0,
                          'actual_L4_training_and_quality_verified': False}), flush=True)
        return 0
    if args.run:
        packet.need(os.environ.get('READOUT_SUPERVISED_PROTOCOL') == args.protocol_sha
                    and os.environ.get('READOUT_SUPERVISOR_PID') == str(os.getppid()),
                    'Worker must be manually launched through the finite supervisor')
        from . import worker
        worker.run(root, p, args.protocol_sha)
        return 0
    if args.supervise:
        return supervise(root, p, args.protocol_sha)
    export(root, args.protocol_sha)
    return 0

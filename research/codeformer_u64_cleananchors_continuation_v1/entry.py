"""Static CLI, finite manual worker supervision and preserved bounded exports."""
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

PREFIX = 'cctv-codeformer-u64-cleananchors-continuation-v1'
EXPORT_SECONDS = 300
EXPORT_EXTERNAL_SECONDS = 330
EXPORT_BYTES = 2 * 1024**3
ENV_SUPERVISED = 'FORENSIC_U64_CLEANANCHORS_SUPERVISED'
ENV_PROTOCOL = 'FORENSIC_U64_CLEANANCHORS_PROTOCOL'
ENV_PARENT = 'FORENSIC_U64_CLEANANCHORS_SUPERVISOR_PID'
COUNTER_CAPS = {'full_forward_attempts': 292, 'full_forward_returns': 292, 'backward_attempts': 64, 'backward_returns': 64, 'optimizer_constructions': 1, 'optimizer_construction_returns': 1, 'optimizer_update_attempts': 16, 'optimizer_updates': 16, 'initial_quality_bank_forwards': 56, 'deterministic_repeat_forwards': 4, 'candidate_snapshot_forwards_after_initializer': 168, 'candidate_training_forwards': 64}


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def journal_snapshot_readback(root, counters):
    """A stale status snapshot cannot hide a newer durable reservation.

    This is bounded last-event consistency, not an independent full replay.
    Missing, partial, oversized or concurrently changed evidence stays unknown.
    """
    path = root / packet.OUTPUT_RELATIVE / 'events.jsonl'
    try:
        path = packet.regular(path)
        before = path.stat()
        packet.need(0 < before.st_size <= 32 * 1024**2, 'Bounded nonempty durable journal required')
        with path.open('rb') as stream:
            stream.seek(max(0, before.st_size - 65536))
            tail = stream.read(65536)
        after = path.stat()
        packet.need((before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_ino, after.st_size, after.st_mtime_ns), 'Journal changed during readback')
        packet.need(tail.endswith(b'\n'), 'Interrupted journal line remains unknown')
        last = packet.strict_json(tail.splitlines()[-1])
        packet.need(type(last) is dict and last.get('counters') == counters,
                    'Status snapshot differs from latest durable journal counters')
        return {'journal_tail_reconciled': True, 'journal_bytes_observed': before.st_size,
                'journal_tail_kind': last.get('kind'), 'journal_independent_full_replay': False}
    except TimeoutError:
        raise
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return {'journal_tail_reconciled': False, 'journal_readback_error': str(exc)[:2000],
                'journal_independent_full_replay': False}


def _checked_status(root, expected_sha):
    path = root / packet.OUTPUT_RELATIVE / 'status.json'
    pending = (root / packet.OUTPUT_RELATIVE / 'status.pending.json').exists()
    if not path.exists():
        return {'status_receipt_present': False, 'status_schema_and_protocol_verified': False,
                'confirmed_counters': None, 'operation_outcome_unknown': True,
                'interrupted_status_write_present': pending}
    value = packet.strict_json(packet.regular(path).read_bytes())
    packet.need(value['protocol_sha256'] == expected_sha, 'Status protocol differs')
    counters = value['counters']
    packet.need(type(counters) is dict, 'Durable counter object required')
    for name, cap in COUNTER_CAPS.items():
        count = counters[name]
        packet.need(type(count) is int and 0 <= count <= cap, 'Finite durable counter required: ' + name)
    pairs = [('full_forward_attempts', 'full_forward_returns'),
             ('backward_attempts', 'backward_returns'),
             ('optimizer_constructions', 'optimizer_construction_returns'),
             ('optimizer_update_attempts', 'optimizer_updates')]
    packet.need(all(counters[a] >= counters[b] for a, b in pairs), 'Returns exceed attempted operations')
    unknown = pending or any(counters[a] > counters[b] for a, b in pairs)
    unknown = unknown or value.get('operation_outcome_unknown', False) is not False
    journal = journal_snapshot_readback(root, counters)
    unknown = unknown or not journal['journal_tail_reconciled']
    return {'status_receipt_present': True, 'status_schema_and_protocol_verified': True,
            'confirmed_counters': {name: counters[name] for name in COUNTER_CAPS},
            'confirmed_counter_scope': 'Durable status snapshot; current journal-tail consistency is separate.',
            **journal,
            'operation_outcome_unknown': bool(unknown), 'interrupted_status_write_present': pending,
            'recorded_phase': value.get('phase'), 'recorded_status': value.get('status'),
            'status_receipt': packet.pin(root, path)}


def status(root, expected_sha):
    """Corrupt/missing/pending receipts never become fabricated zero counts."""
    try:
        return _checked_status(root, expected_sha)
    except TimeoutError:
        raise
    except Exception as exc:
        answer = {'status_receipt_present': (root / packet.OUTPUT_RELATIVE / 'status.json').exists(),
                  'status_schema_and_protocol_verified': False, 'confirmed_counters': None,
                  'operation_outcome_unknown': True, 'error_type': type(exc).__name__,
                  'status_read_error': str(exc)[:2000],
                  'interrupted_status_write_present': (root / packet.OUTPUT_RELATIVE / 'status.pending.json').exists()}
        try:
            answer['status_receipt'] = packet.pin(root, root / packet.OUTPUT_RELATIVE / 'status.json')
        except TimeoutError:
            raise
        except (OSError, ValueError):
            pass
        return answer


def _export(root, expected_sha, start):
    def guard():
        packet.need(time.monotonic() - start < EXPORT_SECONDS, 'Finite300s export stop')
        packet.need(shutil.disk_usage(root).free >= packet.LIMITS['continuing_free_disk_bytes'],
                    'Keep4GiB export disk reserve')
    guard()
    p = packet.verify_static(root, expected_sha)
    guard()
    packet.need(platform.system() == 'Linux', 'Export manually on VM')
    worker_evidence = ('supervisor.start.json', 'supervisor.worker.started.json',
                       'supervisor.worker.json', 'run.log', packet.OUTPUT_RELATIVE)
    if any((root / name).exists() for name in worker_evidence):
        worker_path = root / 'supervisor.worker.json'
        packet.need(worker_path.is_file(), 'Worker outcome unknown; do not export a potentially live writer')
        worker = packet.strict_json(packet.regular(worker_path).read_bytes())
        cleanup = worker.get('cleanup')
        packet.need(worker.get('complete') is True and worker.get('spawn_returned') is True
                    and type(cleanup) is dict and cleanup.get('child_reaped') is True
                    and cleanup.get('process_group_absent') is True
                    and worker.get('orphan_cleanup_unknown', False) is False,
                    'Known writer cleanup required before export')
    archive = Path.home() / (PREFIX + '-results.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    receipt = Path.home() / (PREFIX + '-export.json')
    packet.need(not any(x.exists() for x in (archive, checksum, receipt, root / 'export.json')),
                'Prior exports immutable; preserve partial archive')
    inventory = []
    for path in sorted(root.rglob('*')):
        guard()
        if path.is_dir():
            packet.directory(path)
            continue
        path = packet.regular(path)
        if path.relative_to(root).as_posix().startswith('weights/'):
            continue
        inventory.append(packet.pin(root, path))
    payload = sum(x['bytes'] for x in inventory)
    packet.need(payload <= EXPORT_BYTES, 'Return payload exceeds2GiB')
    read_bytes = 0
    class BoundSource:
        def __init__(self, stream):
            self.stream = stream; self.count = 0; self.digest = hashlib.sha256()
        def read(self, size=-1):
            nonlocal read_bytes
            guard()
            block = self.stream.read(size)
            self.count += len(block); read_bytes += len(block); self.digest.update(block)
            packet.need(read_bytes <= EXPORT_BYTES, 'Read payload exceeds2GiB')
            return block
    class Limited:
        def __init__(self, stream):
            self.stream = stream; self.count = 0
        def write(self, block):
            guard(); self.count += len(block)
            packet.need(self.count <= EXPORT_BYTES, 'Return archive exceeds2GiB')
            return self.stream.write(block)
        def flush(self):
            return self.stream.flush()
    # gzip wrapper + plain tar streaming is Python3.10-compatible; no tar compresslevel argument.
    with archive.open('xb') as output:
        with gzip.GzipFile(filename='', fileobj=Limited(output), mode='wb', compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for row in inventory:
                    guard(); source_path = packet.regular(root / row['path'])
                    info = tar.gettarinfo(str(source_path), arcname=p['stage_name'] + '/' + row['path'])
                    packet.need(info.isfile() and info.size == row['bytes'], 'Payload changed before read')
                    with source_path.open('rb') as original:
                        source = BoundSource(original); tar.addfile(info, source)
                    packet.need(source.count == row['bytes'] and source.digest.hexdigest() == row['sha256'],
                                'Payload changed during read; retain partial archive')
                    packet.need(packet.pin(root, source_path) == row, 'Payload changed after read')
        output.flush(); os.fsync(output.fileno())
    guard()
    packet.need(read_bytes == payload, 'Archive payload count differs from sealed inventory')
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            guard(); digest.update(block)
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest.hexdigest() + '  ' + archive.name + '\n')
        stream.flush(); os.fsync(stream.fileno())
    failure_path = root / packet.OUTPUT_RELATIVE / 'failure.json'
    result_path = root / packet.OUTPUT_RELATIVE / 'result.json'
    failure = None; failure_read_error = None
    if failure_path.exists():
        try:
            failure = packet.strict_json(packet.regular(failure_path).read_bytes())
            packet.need(type(failure) is dict, 'Failure object required')
        except TimeoutError:
            raise
        except Exception as exc:
            failure_read_error = str(exc)[:2000]
    worker_exit = None
    worker_receipt_error = None
    worker_receipt = root / 'supervisor.worker.json'
    if worker_receipt.exists():
        try:
            observed = packet.strict_json(packet.regular(worker_receipt).read_bytes())
            packet.need(observed['complete'] is True and type(observed['exit_code']) is int,
                        'Known supervisor worker-exit receipt required')
            worker_exit = observed['exit_code']
        except TimeoutError:
            raise
        except Exception as exc:
            worker_receipt_error = str(exc)[:2000]
    result = {'complete': True, 'archive_sha256': digest.hexdigest(), 'bytes': archive.stat().st_size,
              'payload_files': len(inventory), 'payload_bytes': payload,
              'seconds': time.monotonic() - start, 'protocol_sha256': expected_sha,
              **status(root, expected_sha), 'result_present': result_path.is_file(),
              'failure_present': failure_path.exists(), 'failure_read_error': failure_read_error,
              'worker_exit': worker_exit, 'worker_exit_receipt_error': worker_receipt_error,
              'quality_stop_exit2': worker_exit == 2,
              'quality_stop_recorded': type(failure) is dict and
                                       (failure.get('quality_stop') is True or failure.get('status') == 'QUALITY_STOP'),
              'training_success_not_implied': True, 'model_qualified': False}
    write_new(root / 'export.json', result); write_new(receipt, result)
    print(json.dumps(result), flush=True)
    return result


def export(root, expected_sha):
    start = time.monotonic()
    packet.need(platform.system() == 'Linux', 'Export manually on VM')
    failure = root / 'export_failure.json'
    packet.need(not failure.exists(), 'Prior failed export immutable; no automatic retry')
    packet.need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'Do not replace an active timer')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('Absolute300s export stop includes all verification/inventory')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, EXPORT_SECONDS)
    try:
        return _export(root, expected_sha, start)
    except BaseException as exc:
        write_new(failure, {'complete': False, 'protocol_sha256': expected_sha,
                  'error_type': type(exc).__name__, 'error': str(exc)[:4000],
                  'seconds': time.monotonic() - start, **status(root, expected_sha),
                  'partial_archive_retained': True, 'training_success_not_implied': True, 'model_qualified': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)


def _group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_group(child, grace):
    """Stop only the process group created for this exact child; record uncertainty."""
    start = time.monotonic(); actions = []
    if _group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGTERM); actions.append('SIGTERM')
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=max(.01, min(10., grace - (time.monotonic() - start))))
    except subprocess.TimeoutExpired:
        pass
    if _group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL); actions.append('SIGKILL')
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=max(.01, grace - (time.monotonic() - start)))
    except subprocess.TimeoutExpired:
        pass
    return {'actions': actions, 'child_reaped': child.poll() is not None,
            'process_group_absent': not _group_alive(child.pid), 'seconds': time.monotonic() - start}


def _child(root, command, *, phase, timeout, log_name, env):
    """Exclusive finite child with immediate PID evidence and cleanup on interruption."""
    start = time.monotonic(); child = None; timed_out = False; interrupted = None; cleanup = None
    error = None; code = None
    started_path = root / ('supervisor.' + phase + '.started.json')
    with (root / log_name).open('xb') as log:
        try:
            child = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
            write_new(started_path, {'complete': False, 'phase': phase, 'child_pid': child.pid,
                      'process_group': child.pid, 'external_cap_seconds': timeout,
                      'command': command, 'spawn_returned': True, 'model_qualified': False})
            print(json.dumps({'phase': phase, 'child_pid': child.pid, 'log': log_name,
                              'external_cap_seconds': timeout}), flush=True)
            code = child.wait(timeout=max(.01, timeout - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            timed_out = True; code = 124
        except (KeyboardInterrupt, SystemExit) as exc:
            interrupted = type(exc).__name__; code = 130
        except TimeoutError:
            raise
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)[:2000]}; code = 1
        finally:
            if child is not None:
                try:
                    cleanup = _stop_group(child, packet.LIMITS['termination_grace_seconds'])
                except BaseException as exc:
                    cleanup = {'actions': [], 'child_reaped': False, 'process_group_absent': False,
                               'error_type': type(exc).__name__, 'error': str(exc)[:2000]}
    result = {'complete': True, 'phase': phase, 'exit_code': code,
              'timeout': timed_out, 'interrupted': interrupted, 'spawn_returned': child is not None,
              'child_pid': None if child is None else child.pid, 'cleanup': cleanup,
              'error': error, 'seconds': time.monotonic() - start,
              'training_success_not_implied': True, 'model_qualified': False}
    if cleanup is not None and (not cleanup['child_reaped'] or not cleanup['process_group_absent']):
        result['exit_code'] = code or 1; result['orphan_cleanup_unknown'] = True
    return result


def supervise(root, p, expected_sha):
    packet.require_vm(root)
    packet.need(not any((root / name).exists() for name in packet.RUNTIME_FILES - {'install.json'})
                and not (root / packet.OUTPUT_RELATIVE).exists(),
                'Prior launch immutable; no automatic retry')
    command = [sys.executable, '-B', '-u', str(root / 'scripts/run_u64_cleananchors_continuation_v1.py'),
               '--root', str(root), '--protocol-sha', expected_sha, '--run']
    write_new(root / 'supervisor.start.json', {'complete': False, 'protocol_sha256': expected_sha,
              'command': command, 'worker_spawn_reserved': True, 'model_qualified': False})
    environment = dict(os.environ, **{ENV_SUPERVISED: '1', ENV_PROTOCOL: expected_sha,
                                       ENV_PARENT: str(os.getpid()), 'PYTHONDONTWRITEBYTECODE': '1'})
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor interrupted; stop owned child group and retain evidence')
    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    try:
        worker = _child(root, command, phase='worker', timeout=p['limits']['external_seconds'],
                        log_name='run.log', env=environment)
        write_new(root / 'supervisor.worker.json', worker)
        cleanup = worker.get('cleanup')
        writer_unknown = (worker.get('spawn_returned') is not True or type(cleanup) is not dict
                          or cleanup.get('child_reaped') is not True
                          or cleanup.get('process_group_absent') is not True
                          or worker.get('orphan_cleanup_unknown', False) is not False)
        if writer_unknown:
            exported = {'complete': False, 'phase': 'export', 'skipped': True, 'exit_code': None,
                        'timeout': False, 'active_writer_outcome_unknown': True,
                        'cause': 'Worker spawn/cleanup not fully confirmed; no new export process.',
                        'partial_evidence_retained': True, 'model_qualified': False}
        else:
            export_command = command[:-1] + ['--export']
            # Known quality/error stops still export only after writer cleanup is confirmed.
            exported = _child(root, export_command, phase='export', timeout=EXPORT_EXTERNAL_SECONDS,
                              log_name='export.log', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        write_new(root / 'supervisor.export.json', exported)
        result = {'complete': True, 'protocol_sha256': expected_sha,
                  'worker_exit': worker['exit_code'], 'worker_timeout': worker['timeout'],
                  'quality_stop_exit2': worker['exit_code'] == 2,
                  'export_exit': exported['exit_code'], 'export_timeout': exported['timeout'],
                  'export_attempted': not writer_unknown, 'export_skipped': writer_unknown,
                  'active_writer_outcome_unknown': writer_unknown,
                  'seconds': time.monotonic() - start, **status(root, expected_sha),
                  'training_success_not_implied': True, 'model_qualified': False}
        write_new(root / 'supervisor.json', result)
        print(json.dumps(result), flush=True)
        return 0 if result['worker_exit'] == result['export_exit'] == 0 else (result['worker_exit'] or result['export_exit'] or 1)
    except BaseException as exc:
        target = root / 'supervisor_failure.json'
        if not target.exists():
            write_new(target, {'complete': False, 'protocol_sha256': expected_sha,
                      'error_type': type(exc).__name__, 'error': str(exc)[:2000],
                      'partial_evidence_retained': True, **status(root, expected_sha), 'model_qualified': False})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('verify-transfer', 'status', 'run', 'supervise', 'export'):
        group.add_argument('--' + name, action='store_true')
    args = parser.parse_args(); root = packet.directory(args.root)
    packet.need(Path(__file__).resolve() == root / packet.PACKAGE / 'entry.py', 'Use own staged CLI source')
    if args.export:
        export(root, args.protocol_sha); return 0
    p = packet.verify_static(root, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps({'complete': True, 'static_files': len(p['files']), 'retained_cases': 96,
                          'optimization_rolecases': 64, 'adaptive_monitors': 32, 'legacy_guards': 24,
                          'cached_U96_controls': 56, 'saved_terminal_u64_parity_cases': 4,
                          'maximum_full_forwards': 292, 'maximum_new_updates': 16,
                          'neural_gradient_optimizer_training_calls': 0,
                          'actual_L4_parity_or_quality_verified': False}), flush=True)
        return 0
    if args.status:
        print(json.dumps(status(root, args.protocol_sha)), flush=True); return 0
    if args.run:
        packet.need(os.environ.get(ENV_SUPERVISED) == '1'
                    and os.environ.get(ENV_PROTOCOL) == args.protocol_sha
                    and os.environ.get(ENV_PARENT) == str(os.getppid()),
                    'Worker must be launched by its finite manual supervisor')
        packet.require_vm(root)
        from . import worker
        result = worker.run(root, p, args.protocol_sha)
        return result if type(result) is int else 0
    return supervise(root, p, args.protocol_sha)


if __name__ == '__main__':
    raise SystemExit(main())

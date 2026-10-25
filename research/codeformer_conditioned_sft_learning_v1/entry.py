"""Static CLI, finite manual worker supervision and preserved bounded exports."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from . import packet

PREFIX = 'cctv-codeformer-conditioned-sft-learning-v1'
EXPORT_SECONDS = 600
EXPORT_EXTERNAL_SECONDS = 660
EXPORT_BYTES = 12 * 1024**3
ENV_SUPERVISED = 'FORENSIC_CONDITIONED_SFT_SUPERVISED'
ENV_PROTOCOL = 'FORENSIC_CONDITIONED_SFT_PROTOCOL'
ENV_PARENT = 'FORENSIC_CONDITIONED_SFT_SUPERVISOR_PID'
COUNTER_CAPS = {'full_forward_attempts':1736,'full_forward_returns':1736,
 'backward_attempts':480,'backward_returns':480,'gradient_queries':0,
 'optimizer_construction_attempts':2,'optimizer_constructions':2,
 'optimizer_update_attempts':96,'optimizer_updates':96,
 'training_forwards':480,'initializer_forwards':432,'deterministic_repeat_forwards':8,
 'later_quality_forwards':576,'terminal_optimization_readout_forwards':240}


def _limits():
    from .contract import LIMITS
    return LIMITS


def _export_context(root):
    root = packet.directory(root)
    packet.need(platform.system() == 'Linux' and
                root == packet.directory(Path.home() / 'forensic-dgp' / packet.STAGE_NAME) and
                packet.directory(Path.cwd()) == root and
                Path(__file__).resolve() == root / packet.PACKAGE / 'entry.py' and
                packet.directory(sys.prefix) == packet.directory(
                    Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv') and
                bool(os.environ.get('TMUX', '').strip()),
                'Exact manual Linux conditioned-sft stage, retained venv, cwd and tmux required for export')


def _checked_status(root, expected_sha):
    """Reconcile a stable saved status with the exact complete journal replay."""
    raw = packet.regular(root / 'protocol.json').read_bytes()
    packet.need(hashlib.sha256(raw).hexdigest() == expected_sha, 'Exact protocol required for status')
    protocol = packet.strict_json(raw)
    execution = packet._module(root, protocol['files'], __package__ + '.execution')
    contract = packet._module(root, protocol['files'], __package__ + '.contract')
    for field in ('admission', 'candidate_data', 'parent_protocol'):
        packet._binding(root, protocol['files'], protocol[field])
    selection = contract.select_cases((root / protocol['admission']['path']).read_bytes(),
                                     (root / protocol['candidate_data']['path']).read_bytes(),
                                     (root / protocol['parent_protocol']['path']).read_bytes())
    path = packet.regular(root / packet.OUTPUT_RELATIVE / 'status.json')
    packet.need(not (path.parent / 'status.pending.json').exists(), 'Interrupted status remains unknown')
    before = packet.pin(root, path)
    value = packet.strict_json(path.read_bytes())
    packet.need(type(value) is dict and value.get('protocol_sha256') == expected_sha and
                value.get('model_qualified') is False and type(value.get('complete')) is bool and
                type(value.get('journal_records')) is int and value['journal_records'] > 0 and
                type(value.get('journal_bytes')) is int and value['journal_bytes'] > 0 and
                type(value.get('journal_sha256')) is str and
                re.fullmatch('[0-9a-f]{64}', value['journal_sha256']) is not None,
                'Protocol-bound unqualified status required')
    counters = value['counters']
    packet.need(type(counters) is dict and set(counters) == set(COUNTER_CAPS) and
                all(type(counters[key]) is int and 0 <= counters[key] <= maximum
                    for key, maximum in COUNTER_CAPS.items()), 'Exact finite status counters required')
    journal = packet.regular(path.parent / 'operations.jsonl')
    journal_before = packet.pin(root, journal)
    packet.need(0 < journal_before['bytes'] <= execution.MAX_JOURNAL_BYTES, 'Bounded operation journal required')
    replay = execution.audit_journal(journal.read_bytes(), selection)
    packet.need(packet.pin(root, journal) == journal_before and packet.pin(root, path) == before,
                'Status or journal changed during readback')
    packet.need(replay['counters'] == counters and
                replay['journal_records'] == value.get('journal_records') and
                replay['journal_bytes'] == value.get('journal_bytes') and
                replay['journal_sha256'] == value.get('journal_sha256') and
                replay['pending'] == value.get('pending') and replay['status'] == value.get('status'),
                'Saved status differs from durable journal replay')
    return {'status_receipt_present': True, 'status_schema_and_protocol_verified': True,
            'confirmed_counters': counters, 'journal_full_replay': True,
            'journal_independent_pixel_or_state_audit': False,
            'operation_outcome_unknown': replay['pending'] is not None or replay['status'] == 'JOURNAL_FAILED',
            'recorded_ledger_status': value['status'], 'status_complete_recorded': value.get('complete'),
            'recorded_phase': value.get('phase'), 'per_arm': replay['per_arm'],
            'quality_stops': replay['quality_stops'],
            'status_receipt': before, 'journal_receipt': journal_before}


def status(root, expected_sha):
    try:
        return _checked_status(root, expected_sha)
    except TimeoutError:
        raise
    except Exception as exc:
        return {'status_receipt_present': (root / packet.OUTPUT_RELATIVE / 'status.json').exists(),
                'status_schema_and_protocol_verified': False, 'confirmed_counters': None,
                'operation_outcome_unknown': True, 'journal_full_replay': False,
                'error_type': type(exc).__name__, 'status_read_error': str(exc)[:2000],
                'interrupted_status_write_present':
                    (root / packet.OUTPUT_RELATIVE / 'status.pending.json').exists()}


def quality_stop_evidence(result, failure, status_record, expected_sha):
    fatal = type(failure) is dict and failure.get('fatal') is True
    if type(result) is dict and not fatal:
        execution = result.get('execution')
        arms = result.get('quality_stops')
        stops = execution.get('quality_stops') if type(execution) is dict else None
        linked = type(arms) is dict and type(stops) is dict and bool(stops) and set(arms) == set(stops)
        if linked:
            for arm, row in arms.items():
                stop = stops[arm]
                linked = linked and (arm in ('A', 'B') and type(row) is dict and type(stop) is dict and
                    type(row.get('update')) is int and row['update'] == stop.get('checkpoint') and
                    row.get('reason') == stop.get('reason') and type(row.get('gate_receipt')) is dict and
                    row['gate_receipt'].get('sha256') == stop.get('receipt_sha256'))
        if (result.get('protocol_sha256') == expected_sha and result.get('model_qualified') is False and
                result.get('complete') is True and result.get('quality_stop') is True and
                linked and execution.get('status') == 'FINISHED' and
                execution.get('bookkeeping_complete') is True and
                status_record.get('status_schema_and_protocol_verified') is True and
                status_record.get('recorded_ledger_status') == 'FINISHED' and
                status_record.get('operation_outcome_unknown') is False and
                status_record.get('quality_stops') == stops):
            return {'quality_stop_recorded': True, 'quality_stop_source': 'conditioned_result_and_replayed_journal',
                    'recorded_quality_stop_arms': sorted(stops)}
    return {'quality_stop_recorded': False, 'quality_stop_source': None,
            'recorded_quality_stop_arms': [],
            'fatal_failure_present': fatal}


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def _export(root, expected_sha, start):
    def guard():
        packet.need(time.monotonic() - start < EXPORT_SECONDS, 'Finite600s export stop')
        packet.need(shutil.disk_usage(root).free >= _limits()['continuing_free_disk_bytes'],
                    'Keep4GiB export disk reserve')
    _export_context(root)
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
    packet.need(payload <= EXPORT_BYTES, 'Return payload exceeds12GiB')
    read_bytes = 0
    class BoundSource:
        def __init__(self, stream):
            self.stream = stream; self.count = 0; self.digest = hashlib.sha256()
        def read(self, size=-1):
            nonlocal read_bytes
            guard()
            block = self.stream.read(size)
            self.count += len(block); read_bytes += len(block); self.digest.update(block)
            packet.need(read_bytes <= EXPORT_BYTES, 'Read payload exceeds12GiB')
            return block
    class Limited:
        def __init__(self, stream):
            self.stream = stream; self.count = 0
        def write(self, block):
            guard(); self.count += len(block)
            packet.need(self.count <= EXPORT_BYTES, 'Return archive exceeds12GiB')
            return self.stream.write(block)
        def flush(self):
            return self.stream.flush()
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
    scientific_result = None
    result_read_error = None
    if result_path.exists():
        try:
            scientific_result = packet.strict_json(packet.regular(result_path).read_bytes())
            packet.need(type(scientific_result) is dict, 'Conditioned result object required')
        except TimeoutError:
            raise
        except Exception as exc:
            result_read_error = str(exc)[:2000]
    observed_status = status(root, expected_sha)
    result = {'complete': True, 'archive_sha256': digest.hexdigest(), 'bytes': archive.stat().st_size,
              'payload_files': len(inventory), 'payload_bytes': payload,
              'seconds': time.monotonic() - start, 'protocol_sha256': expected_sha,
              **observed_status, 'result_present': result_path.is_file(), 'result_read_error': result_read_error,
              'failure_present': failure_path.exists(), 'failure_read_error': failure_read_error,
              'worker_exit': worker_exit, 'worker_exit_receipt_error': worker_receipt_error,
              'quality_stop_exit2': worker_exit == 2,
              **quality_stop_evidence(scientific_result, failure, observed_status, expected_sha),
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
        raise TimeoutError('Absolute600s export stop includes all verification/inventory')
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
                    cleanup = _stop_group(child, _limits()['termination_grace_seconds'])
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
    command = [sys.executable, '-B', '-u', str(root / 'scripts/run_conditioned_sft_learning_v1.py'),
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
        print(json.dumps({'complete': True, 'static_files': len(p['files']), 'retained_quality_cases': 56,
                          'optimization_rolecases': 120, 'adaptive_monitors': 40, 'legacy_guards': 24,
                          'cached_U96_controls': 56, 'saved_terminal_u64_parity_cases': 4,
                          'maximum_full_forwards': 1736, 'maximum_new_updates_per_arm': 48,
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
        packet.need(type(result) is int and result in (0, 1, 2),
                    'Worker must return an exact declared exit code; unknown results are failures')
        return result
    return supervise(root, p, args.protocol_sha)


if __name__ == '__main__':
    raise SystemExit(main())

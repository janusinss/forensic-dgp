"""Static CLI, finite manual worker supervision, and PAX export for Conditioned SFT V2."""
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

from . import packet_v2 as packet

PREFIX = 'cctv-codeformer-conditioned-sft-learning-v2'
EXPORT_SECONDS = 600
EXPORT_EXTERNAL_SECONDS = 660
EXPORT_BYTES = 12 * 1024**3
ENV_SUPERVISED = 'FORENSIC_CONDITIONED_SFT_V2_SUPERVISED'
ENV_PROTOCOL = 'FORENSIC_CONDITIONED_SFT_V2_PROTOCOL'
ENV_PARENT = 'FORENSIC_CONDITIONED_SFT_V2_SUPERVISOR_PID'


def _limits():
    from .contract_v2 import LIMITS
    return LIMITS


def write_new(path: Path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def status(root: Path, expected_sha: str):
    status_path = root / packet.OUTPUT_RELATIVE / 'status.json'
    if not status_path.is_file():
        return {
            'status_receipt_present': False,
            'status_schema_and_protocol_verified': False,
            'recorded_status': None,
        }
    try:
        val = packet.strict_json(packet.regular(status_path).read_bytes())
        ok = (type(val) is dict and val.get('protocol_sha256') == expected_sha
              and val.get('model_qualified') is False)
        return {
            'status_receipt_present': True,
            'status_schema_and_protocol_verified': ok,
            'recorded_status': val.get('status'),
            'status_complete_recorded': val.get('complete'),
            'best_checkpoint_tag': val.get('best_checkpoint_tag'),
        }
    except Exception as exc:
        return {
            'status_receipt_present': True,
            'status_schema_and_protocol_verified': False,
            'error_type': type(exc).__name__,
            'status_read_error': str(exc)[:2000],
        }


def _export_context(root: Path):
    root = packet.directory(root)
    packet.need(
        platform.system() == 'Linux'
        and root == packet.directory(Path.home() / 'forensic-dgp' / packet.STAGE_NAME)
        and packet.directory(Path.cwd()) == root
        and Path(__file__).resolve() == root / packet.PACKAGE / 'entry_v2.py'
        and packet.directory(sys.prefix) == packet.directory(Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv')
        and bool(os.environ.get('TMUX', '').strip()),
        'Exact manual Linux Conditioned SFT V2 stage, retained venv, cwd, and tmux required for export',
    )


def _export(root: Path, expected_sha: str, start: float):
    def guard():
        packet.need(time.monotonic() - start < EXPORT_SECONDS, 'Finite 600s export stop')
        packet.need(shutil.disk_usage(root).free >= _limits()['continuing_free_disk_bytes'],
                    'Keep 4 GiB export disk reserve')

    _export_context(root)
    guard()
    p = packet.verify_static(root, expected_sha)
    guard()
    worker_evidence = ('supervisor.start.json', 'supervisor.worker.started.json',
                       'supervisor.worker.json', 'run.log', packet.OUTPUT_RELATIVE)
    if any((root / name).exists() for name in worker_evidence):
        worker_path = root / 'supervisor.worker.json'
        packet.need(worker_path.is_file(), 'Worker outcome unknown; refusing to export live writer')
        worker = packet.strict_json(packet.regular(worker_path).read_bytes())
        cleanup = worker.get('cleanup')
        packet.need(
            worker.get('complete') is True and worker.get('spawn_returned') is True
            and type(cleanup) is dict and cleanup.get('child_reaped') is True
            and cleanup.get('process_group_absent') is True
            and worker.get('orphan_cleanup_unknown', False) is False,
            'Known writer cleanup required before export',
        )

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
    packet.need(payload <= EXPORT_BYTES, 'Return payload exceeds 12 GiB')

    with archive.open('xb') as output:
        with gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for row in inventory:
                    guard()
                    source_path = packet.regular(root / row['path'])
                    info = tar.gettarinfo(str(source_path), arcname=p['stage_name'] + '/' + row['path'])
                    packet.need(info.isfile() and info.size == row['bytes'], 'Payload changed before read')
                    with source_path.open('rb') as original:
                        tar.addfile(info, original)
                    packet.need(packet.pin(root, source_path) == row, 'Payload changed after read')
        output.flush()
        os.fsync(output.fileno())

    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            guard()
            digest.update(block)

    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest.hexdigest() + '  ' + archive.name + '\n')
        stream.flush()
        os.fsync(stream.fileno())

    result_path = root / packet.OUTPUT_RELATIVE / 'result.json'
    failure_path = root / packet.OUTPUT_RELATIVE / 'failure.json'
    observed_status = status(root, expected_sha)
    result = {
        'complete': True,
        'archive_sha256': digest.hexdigest(),
        'bytes': archive.stat().st_size,
        'payload_files': len(inventory),
        'payload_bytes': payload,
        'seconds': time.monotonic() - start,
        'protocol_sha256': expected_sha,
        **observed_status,
        'result_present': result_path.is_file(),
        'failure_present': failure_path.exists(),
        'training_success_not_implied': True,
        'model_qualified': False,
    }
    write_new(root / 'export.json', result)
    write_new(receipt, result)
    print(json.dumps(result), flush=True)
    return result


def export(root: Path, expected_sha: str):
    start = time.monotonic()
    packet.need(platform.system() == 'Linux', 'Export manually on VM')
    failure = root / 'export_failure.json'
    packet.need(not failure.exists(), 'Prior failed export immutable')
    packet.need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'Do not replace an active timer')
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError('Absolute 600s export stop')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, EXPORT_SECONDS)
    try:
        return _export(root, expected_sha, start)
    except BaseException as exc:
        if not failure.exists():
            write_new(failure, {
                'complete': False,
                'protocol_sha256': expected_sha,
                'error_type': type(exc).__name__,
                'error': str(exc)[:4000],
                'seconds': time.monotonic() - start,
                **status(root, expected_sha),
                'partial_archive_retained': True,
                'model_qualified': False,
            })
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_group(child, grace: float):
    start = time.monotonic()
    actions = []
    if _group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGTERM)
            actions.append('SIGTERM')
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=max(0.01, min(10.0, grace - (time.monotonic() - start))))
    except subprocess.TimeoutExpired:
        pass
    if _group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL)
            actions.append('SIGKILL')
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=max(0.01, grace - (time.monotonic() - start)))
    except subprocess.TimeoutExpired:
        pass
    return {
        'actions': actions,
        'child_reaped': child.poll() is not None,
        'process_group_absent': not _group_alive(child.pid),
        'seconds': time.monotonic() - start,
    }


def _child(root: Path, command, *, phase: str, timeout: float, log_name: str, env):
    start = time.monotonic()
    child = None
    timed_out = False
    interrupted = None
    cleanup = None
    error = None
    code = None
    started_path = root / ('supervisor.' + phase + '.started.json')
    with (root / log_name).open('xb') as log:
        try:
            child = subprocess.Popen(
                command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=env,
            )
            write_new(started_path, {
                'complete': False,
                'phase': phase,
                'child_pid': child.pid,
                'process_group': child.pid,
                'external_cap_seconds': timeout,
                'command': command,
                'spawn_returned': True,
                'model_qualified': False,
            })
            print(json.dumps({
                'phase': phase,
                'child_pid': child.pid,
                'log': log_name,
                'external_cap_seconds': timeout,
            }), flush=True)
            code = child.wait(timeout=max(0.01, timeout - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            timed_out = True
            code = 124
        except (KeyboardInterrupt, SystemExit) as exc:
            interrupted = type(exc).__name__
            code = 130
        except TimeoutError:
            raise
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
            code = 1
        finally:
            if child is not None:
                try:
                    cleanup = _stop_group(child, _limits()['termination_grace_seconds'])
                except BaseException as exc:
                    cleanup = {
                        'actions': [],
                        'child_reaped': False,
                        'process_group_absent': False,
                        'error_type': type(exc).__name__,
                        'error': str(exc)[:2000],
                    }
    result = {
        'complete': True,
        'phase': phase,
        'exit_code': code,
        'timeout': timed_out,
        'interrupted': interrupted,
        'spawn_returned': child is not None,
        'child_pid': None if child is None else child.pid,
        'cleanup': cleanup,
        'error': error,
        'seconds': time.monotonic() - start,
        'training_success_not_implied': True,
        'model_qualified': False,
    }
    if cleanup is not None and (not cleanup['child_reaped'] or not cleanup['process_group_absent']):
        result['exit_code'] = code or 1
        result['orphan_cleanup_unknown'] = True
    return result


def supervise(root: Path, p, expected_sha: str):
    packet.require_vm(root)
    packet.need(
        not any((root / name).exists() for name in packet.RUNTIME_FILES - {'install.json'})
        and not (root / packet.OUTPUT_RELATIVE).exists(),
        'Prior launch immutable; no automatic retry',
    )
    command = [
        sys.executable, '-B', '-u', str(root / 'scripts/run_conditioned_sft_learning_v2.py'),
        '--root', str(root), '--protocol-sha', expected_sha, '--run',
    ]
    write_new(root / 'supervisor.start.json', {
        'complete': False,
        'protocol_sha256': expected_sha,
        'command': command,
        'worker_spawn_reserved': True,
        'model_qualified': False,
    })
    environment = dict(os.environ, **{
        ENV_SUPERVISED: '1',
        ENV_PROTOCOL: expected_sha,
        ENV_PARENT: str(os.getpid()),
        'PYTHONDONTWRITEBYTECODE': '1',
    })
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor interrupted; stopping owned child group')

    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    try:
        worker = _child(
            root, command, phase='worker', timeout=p['limits']['external_seconds'],
            log_name='run.log', env=environment,
        )
        write_new(root / 'supervisor.worker.json', worker)
        cleanup = worker.get('cleanup')
        writer_unknown = (
            worker.get('spawn_returned') is not True
            or type(cleanup) is not dict
            or cleanup.get('child_reaped') is not True
            or cleanup.get('process_group_absent') is not True
            or worker.get('orphan_cleanup_unknown', False) is not False
        )
        if writer_unknown:
            exported = {
                'complete': False,
                'phase': 'export',
                'skipped': True,
                'exit_code': None,
                'timeout': False,
                'active_writer_outcome_unknown': True,
                'partial_evidence_retained': True,
                'model_qualified': False,
            }
        else:
            export_command = command[:-1] + ['--export']
            exported = _child(
                root, export_command, phase='export', timeout=EXPORT_EXTERNAL_SECONDS,
                log_name='export.log', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            )
        write_new(root / 'supervisor.export.json', exported)
        result = {
            'complete': True,
            'protocol_sha256': expected_sha,
            'worker_exit': worker['exit_code'],
            'worker_timeout': worker['timeout'],
            'export_exit': exported['exit_code'],
            'export_timeout': exported['timeout'],
            'export_attempted': not writer_unknown,
            'export_skipped': writer_unknown,
            'seconds': time.monotonic() - start,
            **status(root, expected_sha),
            'training_success_not_implied': True,
            'model_qualified': False,
        }
        write_new(root / 'supervisor.json', result)
        print(json.dumps(result), flush=True)
        return 0 if result['worker_exit'] == result['export_exit'] == 0 else (result['worker_exit'] or result['export_exit'] or 1)
    except BaseException as exc:
        target = root / 'supervisor_failure.json'
        if not target.exists():
            write_new(target, {
                'complete': False,
                'protocol_sha256': expected_sha,
                'error_type': type(exc).__name__,
                'error': str(exc)[:2000],
                'partial_evidence_retained': True,
                **status(root, expected_sha),
                'model_qualified': False,
            })
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
    args = parser.parse_args()
    root = packet.directory(args.root)
    packet.need(Path(__file__).resolve() == root / packet.PACKAGE / 'entry_v2.py', 'Use own staged CLI source')
    if args.export:
        export(root, args.protocol_sha)
        return 0
    p = packet.verify_static(root, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps({
            'complete': True,
            'static_files': len(p['files']),
            'v2_optimization_cases': len(p['selection']['v2_optimization']),
            'v2_monitors': len(p['selection']['v2_monitors']),
            'v1_monitors': len(p['selection']['v1_monitors']),
            'retained_old_quality': len(p['selection']['old_quality']),
            'native_cctv_dev_cases': len(p['selection']['native_cctv_dev']),
            'phase1_epochs': p['comparison']['phase1_schedule']['epochs'],
            'phase2_updates': p['comparison']['phase2_schedule']['total_optimizer_updates'],
            'neural_gradient_optimizer_training_calls': 0,
            'actual_L4_parity_or_quality_verified': False,
        }), flush=True)
        return 0
    if args.status:
        print(json.dumps(status(root, args.protocol_sha)), flush=True)
        return 0
    if args.run:
        packet.need(
            os.environ.get(ENV_SUPERVISED) == '1'
            and os.environ.get(ENV_PROTOCOL) == args.protocol_sha
            and os.environ.get(ENV_PARENT) == str(os.getppid()),
            'Worker must be launched by its finite manual supervisor',
        )
        packet.require_vm(root)
        from . import worker_v2
        result = worker_v2.run(root, p, args.protocol_sha)
        packet.need(type(result) is int and result in (0, 1), 'Worker must return an exact declared exit code')
        return result
    return supervise(root, p, args.protocol_sha)


if __name__ == '__main__':
    raise SystemExit(main())

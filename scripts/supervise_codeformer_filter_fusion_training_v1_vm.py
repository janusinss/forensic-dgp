"""Human-invoked finite supervisor; never imports a neural framework itself."""
import argparse
import base64
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import transport as t
from research.codeformer_filter_fusion_training_v1 import worker as w

LOG_CAP, GRACE = 2 * 1024**2, 30
# Internal worker: <=300s preflight +4800s work +20s recovery.
# The outer deadline leaves180s for interpreter/startup/terminal closure.
OUTER_WORKER_SECONDS = 5300


def manual_root(path):
    path = t.no_links(Path(path).absolute())
    expected = Path.home() / 'forensic-dgp' / t.VM_NAME
    t.require(platform.system() == 'Linux' and path.resolve(strict=True) == expected.resolve(strict=True)
              and bool(os.environ.get('TMUX')), 'Exact manual Linux-home tmux root required')
    return path


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_own_child(process, grace):
    # Only this new start_new_session=True child may reach this function.
    if group_alive(process.pid):
        os.killpg(process.pid, signal.SIGTERM)
    end = time.monotonic() + grace
    while group_alive(process.pid) and time.monotonic() < end:
        process.poll()
        time.sleep(.05)
    if group_alive(process.pid):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)
    end = time.monotonic() + 10
    while group_alive(process.pid) and time.monotonic() < end:
        time.sleep(.05)
    t.require(not group_alive(process.pid), 'Own child group still live; export forbidden')


def stage(root, protocol_sha, flag, cap, log):
    command = [sys.executable, '-B', '-u', '-m', t.MODULE, '--root', str(root),
               '--protocol-sha', protocol_sha, flag]
    started, written, timed_out, capped = time.monotonic(), 0, False, False
    exception = None
    with t.no_links(log).open('xb') as output:
        child = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
        print(t.json_bytes({'stage': flag, 'child_pid': child.pid,
            'external_cap_seconds': cap, 'grace_seconds': GRACE,
            'log': log.relative_to(root).as_posix()}).decode().strip(), flush=True)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() - started > cap:
                        timed_out = True
                        stop_own_child(child, GRACE)
                        break
                    for key, _ in selector.select(timeout=.25):
                        payload = os.read(key.fileobj.fileno(), 65536)
                        if not payload:
                            selector.unregister(key.fileobj)
                            continue
                        room = max(0, LOG_CAP - written)
                        t.require(shutil.disk_usage(root).free >= t.LIMITS['reserve_free_bytes'] + min(room, len(payload)),
                                  'Supervisor log disk reserve exceeded')
                        output.write(payload[:room]); written += min(room, len(payload))
                        if len(payload) > room:
                            capped = True
                            stop_own_child(child, GRACE)
                            break
                    if capped:
                        break
            code = child.wait(timeout=10)
            if group_alive(child.pid):
                stop_own_child(child, 0)
        except BaseException as error:
            stop_own_child(child, GRACE)
            if not isinstance(error, Exception):
                raise
            exception = {'type': type(error).__name__, 'message': str(error)[:4096]}
            code = 125
        output.flush(); os.fsync(output.fileno())
    if timed_out:
        code = 124
    if capped:
        code = 125
    t.require(not group_alive(child.pid) and log.stat().st_size <= LOG_CAP,
              'Own stage/log terminal invariant failed')
    with log.open('rb') as saved:
        saved.seek(max(0, log.stat().st_size - 16000))
        print(saved.read().decode('utf-8', errors='replace'), flush=True)
    return {'stage': flag, 'exit_code': code, 'child_pid': child.pid,
        'seconds': time.monotonic() - started, 'external_cap_seconds': cap,
        'grace_seconds': GRACE, 'log': log.relative_to(root).as_posix(),
        'log_bytes': written, 'log_capped': capped, 'outer_watchdog_expired': timed_out,
        'worker_terminal': True, 'child_process_group_terminal': True,
        'supervisor_exception': exception}


def final_companion(root, protocol_sha, stages):
    """Close final logs outside the archive they describe; retain exact bytes."""
    with w.stage_deadline(180):
        home = t.no_links(Path.home())
        archive = home / (t.PREFIX + '-results.tar.gz')
        checksum = Path(str(archive) + '.sha256')
        receipt_path = home / (t.PREFIX + '-export.json')
        receipt = t.read_json(receipt_path)
        archive_pin = t.file_pin(archive)
        t.require(receipt.get('complete') is True and receipt['protocol_sha256'] == protocol_sha
                  and archive_pin == {'sha256': receipt['archive_sha256'], 'bytes': receipt['bytes']}
                  and checksum.read_bytes() == t.checksum_bytes(archive.name, receipt['archive_sha256']),
                  'Final archive/checksum/export identity differs')
        log = root / 'outputs/supervision/export.log'
        t.require(log.stat().st_size <= LOG_CAP, 'Final export log cap exceeded')
        finished = root / 'outputs/supervision/finished.json'
        value = {'format': 'codeformer-filter-fusion-supervisor-final-v1', 'complete': True,
            'protocol_sha256': protocol_sha, 'stages': stages,
            'archive': {'path': archive.name, **archive_pin},
            'export_receipt': {'path': receipt_path.name, **t.file_pin(receipt_path)},
            'checksum': {'path': checksum.name, **t.file_pin(checksum)},
            'finished': {'path': finished.relative_to(root).as_posix(), **t.file_pin(finished)},
            'export_log': {'path': log.relative_to(root).as_posix(), **t.file_pin(log),
                'base64': base64.b64encode(log.read_bytes()).decode('ascii')},
            'training_completion_not_model_acceptance': True, 'model_qualified': False,
            'closure_boundary': 'External record, no self hash, not inside the archive it closes'}
        t.require(len(t.json_bytes(value)) <= t.LIMITS['one_json_bytes'], 'Final companion cap exceeded')
        path = home / (t.PREFIX + '-supervisor-final.json')
        t.write_new_json(path, value)
        return {'path': str(path), **t.file_pin(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    root = manual_root(args.root)
    t.require(t.HEX.fullmatch(args.protocol_sha), 'Exact frozen protocol hash required')
    t.require(not (root / 'outputs').exists() and not (root / t.LOCK_NAME).exists(),
              'Preserve prior execution or lock; no retry or resume')
    with w.stage_deadline(300):
        verified = t.verify_transfer(root, args.protocol_sha)
        w.validate_plan(verified['protocol'])
        idle = w.idle_inventory()
    folder = root / 'outputs/supervision'
    folder.mkdir(parents=True)
    preflight = stage(root, args.protocol_sha, '--verify-transfer', 300, folder / 'preflight.log')
    t.require(preflight['exit_code'] == 0, 'Transfer preflight failed; no neural run')
    idle_again = w.idle_inventory()
    measured = stage(root, args.protocol_sha, '--run', OUTER_WORKER_SECONDS, folder / 'worker.log')
    t.write_new_json(folder / 'run_receipt.json', {'protocol_sha256': args.protocol_sha,
        'worker_terminal': True, 'child_process_group_terminal': True,
        'stages': [preflight, measured], 'maximum_training_updates': w.FINAL_UPDATE,
        'initial_idle': idle, 'before_worker_idle': idle_again,
        'actual_count_source': 'Worker terminal record; absent or malformed remains unknown',
        'no_automatic_retry_or_resume': True, 'model_qualified': False})
    t.write_new_json(folder / 'export_invocation.json', {'protocol_sha256': args.protocol_sha,
        'worker_terminal': True, 'export_failure_evidence': measured['exit_code'] != 0,
        'external_cap_seconds': t.LIMITS['export_seconds'],
        'closure_boundary': 'Final export log and finished record retained separately'})
    exported = stage(root, args.protocol_sha, '--export', t.LIMITS['export_seconds'], folder / 'export.log')
    stages = [preflight, measured, exported]
    t.write_new_json(folder / 'finished.json', {'complete': True, 'protocol_sha256': args.protocol_sha,
        'worker_terminal': True, 'stages': stages, 'model_qualified': False,
        'training_or_export_completion_not_model_acceptance': True})
    if exported['exit_code'] == 0:
        print(t.json_bytes(final_companion(root, args.protocol_sha, stages)).decode(), flush=True)
    return 0 if measured['exit_code'] == exported['exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

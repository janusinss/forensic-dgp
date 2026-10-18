"""Human-invoked finite L4 diagnostic and bounded evidence export. Never trains."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import stat
import sys
import tarfile
import time
import traceback

from . import contract as c


def manual_root(root, *, system=None, home=None, environment=None):
    system = platform.system() if system is None else system
    environment = os.environ if environment is None else environment
    c.require(system == 'Linux', 'Actual diagnostic runs manually on the Linux L4 VM')
    c.require(bool(environment.get('TMUX')), 'Launch inside the human-opened tmux session')
    root = Path(root).resolve(strict=True)
    home = Path.home() if home is None else Path(home)
    expected = home / 'forensic-dgp' / c.PACKAGE_DIR
    c.require(root == expected.resolve(strict=True) and not expected.is_symlink(),
              'Use the exact new diagnostic directory under ~/forensic-dgp')
    return root


def reject_unexpected_files(root):
    paths = []
    if not root.exists():
        return paths
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(folder) / name
            info = path.lstat()
            c.require(not path.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400,
                      'Result directory link forbidden')
        for name in names:
            path = Path(folder) / name
            info = path.lstat()
            c.require(stat.S_ISREG(info.st_mode) and not path.is_symlink() and
                      not getattr(info, 'st_file_attributes', 0) & 0x400, 'Regular evidence file required')
            c.require(path.suffix.lower() not in ('.pt', '.pth', '.pkl', '.pickle'),
                      'Diagnostic must not produce or export model/optimizer weights')
            paths.append(path)
    return sorted(paths)


def require_worker_terminal(root, protocol_sha, *, proc_base=Path('/proc')):
    """Receipt presence alone cannot authorize exporting a still-live worker."""
    launch = c.read_json(root / 'outputs/path_diagnostic_v1/manual_launch.json')
    c.require(launch.get('protocol_sha256') == protocol_sha and
              type(launch.get('pid')) is int and launch['pid'] > 0,
              'Manual launch identity missing or invalid')
    process = proc_base / str(launch['pid'])
    if not process.exists():
        return
    # A reused unrelated PID cannot write this packet. Matching either the
    # protocol argument or exact worker module is conservatively treated live.
    command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', errors='replace')
    c.require(protocol_sha not in command and
              'research.codeformer_path_diagnostic_v1.worker' not in command,
              'Recorded diagnostic worker is still live; wait for supervisor completion')


@contextmanager
def finite_signal_scope(seconds):
    def stopped(signum, frame):
        raise TimeoutError('Finite diagnostic signal/time stop: ' + str(signum))
    before = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
    if hasattr(signal, 'SIGALRM'):
        before[signal.SIGALRM] = signal.getsignal(signal.SIGALRM)
    for name in before:
        signal.signal(name, stopped)
    if hasattr(signal, 'setitimer'):
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        if hasattr(signal, 'setitimer'):
            signal.setitimer(signal.ITIMER_REAL, 0)
        for name, handler in before.items():
            signal.signal(name, handler)


def run(root, protocol_sha):
    root = manual_root(root)  # Must precede neural imports.
    p = c.verify_transfer(root, protocol_sha, verify_parent=True)
    out = root / p['outputs']
    c.require(not out.exists(), 'Preserve existing diagnostic evidence; do not relaunch')
    c.require(shutil.disk_usage(root).free >= p['limits']['initial_free_bytes'],
              'Require 6 GiB free after installation')
    out.mkdir(parents=True)
    started = time.monotonic()
    stamp = {'pid': os.getpid(), 'protocol_sha256': protocol_sha,
             'optimizer_updates': 0, 'automatic_training': False,
             'manual_tmux': True, 'start_unix_seconds': time.time()}
    c.write_new_json(out / 'manual_launch.json', stamp)
    try:
        with finite_signal_scope(p['limits']['worker_seconds']):
            # This is the only entry that may import neural execution code.
            from .neural import run as measure
            result = measure(root, p)
            c.require(isinstance(result, dict) and result.get('complete') is True and
                      result.get('counts') == p['call_caps'] and
                      result.get('state_before_after_equal') is True and
                      result.get('parameter_grad_fields_all_none') is True,
                      'Neural stage must return complete finite immutable-state evidence')
            c.require(sum(path.stat().st_size for path in reject_unexpected_files(out)) <=
                      p['limits']['result_bytes'], 'Diagnostic result payload cap exceeded')
            # Check parent file bytes again, independently of in-memory fingerprints.
            c.verify_transfer(root, protocol_sha, verify_parent=True)
            c.write_new_json(out / 'worker_complete.json', {
                **stamp, 'complete': True, 'seconds': time.monotonic() - started,
                'measurement_result': result, 'model_qualified': False,
                'next_training_authorized': False, 'automatic_app_promotion': False})
        return 0
    except BaseException as error:
        failure = out / 'worker_failure.json'
        if not failure.exists():
            c.write_new_json(failure, {**stamp, 'complete': False,
                'error_type': type(error).__name__, 'message': str(error),
                'traceback': traceback.format_exc(), 'seconds': time.monotonic() - started,
                'partial_evidence_retained': True, 'training_success_not_implied': True,
                'model_qualified': False, 'automatic_resume': False})
        traceback.print_exc()
        return 1


class BoundedArchive:
    """Count compressed bytes before writes; enforce export time at each write."""
    def __init__(self, stream, byte_cap, deadline):
        self.stream, self.byte_cap, self.deadline, self.bytes = stream, byte_cap, deadline, 0

    def write(self, payload):
        c.require(time.monotonic() <= self.deadline, 'Finite export time exceeded')
        c.require(self.bytes + len(payload) <= self.byte_cap, 'Compressed archive byte cap exceeded')
        written = self.stream.write(payload)
        self.bytes += written
        return written

    def flush(self):
        self.stream.flush()


def export(root, protocol_sha):
    root = manual_root(root)
    started = time.monotonic()
    p = c.verify_transfer(root, protocol_sha, verify_parent=True)
    out = root / p['outputs']
    c.require(out.is_dir(), 'No executed diagnostic evidence to export')
    c.require((out / 'worker_complete.json').is_file() or (out / 'worker_failure.json').is_file()
              or (root / 'outputs/supervision/run_receipt.json').is_file(),
              'No terminal worker/supervisor evidence; do not export a live measurement')
    require_worker_terminal(root, protocol_sha)
    prefix = p['export_prefix']
    archive = Path.home() / (prefix + '-results.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    receipt_path = Path.home() / (prefix + '-export.json')
    temporary = Path(str(archive) + '.writing')
    for path in (archive, checksum, receipt_path, temporary):
        c.require(not path.exists(), 'Preserve existing or partial export: ' + path.name)
    with finite_signal_scope(p['limits']['export_seconds']):
        result_files = reject_unexpected_files(out)
        c.require(sum(path.stat().st_size for path in result_files) <= p['limits']['result_bytes'],
                  'Retained result payload exceeds 256 MiB; no silent omission')
        relative_files = set(p['export_files']) | {'protocol.json'}
        relative_files.update(path.relative_to(root).as_posix() for path in result_files)
        for name in ('preflight.log', 'diagnostic.log', 'run_receipt.json'):
            relative = 'outputs/supervision/' + name
            if (root / relative).exists():
                relative_files.add(relative)
        files, sizes = {}, {}
        for name in sorted(relative_files):
            path = c.bounded(root, name)
            files[name] = c.sha256(path)
            sizes[name] = path.stat().st_size
        c.require(sum(sizes.values()) <= p['limits']['archive_bytes'],
                  'Uncompressed export accounting exceeds archive allowance')
        manifest_name = p['outputs'] + '/export_manifest.json'
        c.require(manifest_name not in files and not (root / manifest_name).exists(),
                  'Preserve prior export manifest')
        manifest = {'format': 'codeformer-path-diagnostic-return-v1',
                    'protocol_sha256': protocol_sha, 'files_sha256': files,
                    'files_bytes': sizes, 'parent_files_sha256': p['parent_files_sha256'],
                    'complete_measurement': (out / 'worker_complete.json').is_file(),
                    'failure_present': (out / 'worker_failure.json').is_file(),
                    'optimizer_updates': 0, 'model_qualified': False,
                    'native_CCTV_qualification': False, 'automatic_training': False}
        c.write_new_json(root / manifest_name, manifest)
        manifest_sha = c.sha256(root / manifest_name)
        with temporary.open('xb') as destination:
            bounded = BoundedArchive(destination, p['limits']['archive_bytes'],
                                     started + p['limits']['export_seconds'])
            with gzip.GzipFile(filename='', mode='wb', fileobj=bounded, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as bundle:
                    for name in sorted(relative_files | {manifest_name}):
                        path = c.bounded(root, name)
                        expected = manifest_sha if name == manifest_name else files[name]
                        c.require(c.sha256(path) == expected, 'Export evidence changed during packing')
                        info = bundle.gettarinfo(str(path), arcname=name)
                        c.require(info.isfile(), 'Only regular export members')
                        info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        with path.open('rb') as stream:
                            bundle.addfile(info, stream)
            destination.flush(); os.fsync(destination.fileno())
        c.require(temporary.stat().st_size <= p['limits']['archive_bytes'], 'Archive cap exceeded')
        temporary.rename(archive)
        archive_sha = c.sha256(archive)
        with checksum.open('x', encoding='ascii', newline='\n') as destination:
            destination.write(archive_sha + '  ' + archive.name + '\n')
        receipt = {'complete': True, 'archive_sha256': archive_sha,
                   'bytes': archive.stat().st_size, 'manifest_member': manifest_name,
                   'manifest_sha256': manifest_sha, 'protocol_sha256': protocol_sha,
                   'measurement_complete': manifest['complete_measurement'],
                   'failure_present': manifest['failure_present'],
                   'seconds': time.monotonic() - started, 'optimizer_updates': 0,
                   'model_qualified': False, 'training_success_not_implied': True}
        c.write_new_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify-transfer', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if args.verify_transfer:
        p = c.verify_transfer(args.root, args.protocol_sha, verify_parent=True)
        print(json.dumps({'complete': True, 'cases': len(p['cases']),
                          'parent_dependencies_verified': len(p['parent_files_sha256']),
                          'optimizer_updates': 0, 'neural_calls': 0, 'gradient_queries': 0}))
        return 0
    return export(args.root, args.protocol_sha) if args.export else run(args.root, args.protocol_sha)


if __name__ == '__main__':
    raise SystemExit(main())

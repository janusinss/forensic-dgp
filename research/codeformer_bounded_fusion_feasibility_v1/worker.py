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
            c.require(path.suffix.lower() not in ('.pt', '.pth', '.pkl', '.pickle', '.ckpt', '.safetensors'),
                      'Diagnostic must not produce or export model/optimizer weights')
            paths.append(path)
    return sorted(paths)


def require_worker_terminal(root, protocol_sha, *, proc_base=Path('/proc')):
    """Receipt presence alone cannot authorize exporting a still-live worker."""
    launch = c.read_json(root / 'outputs/bounded_fusion_feasibility_v1/manual_launch.json')
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
              'research.codeformer_bounded_fusion_feasibility_v1.worker' not in command,
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


class EvidenceWriter:
    """Exclusive, bounded CPU serialization callbacks; no neural code."""
    def __init__(self, root, protocol, clock, *, terminal=False):
        self.root, self.p, self.clock = Path(root), protocol, clock
        self.terminal = terminal
        self.out = self.root / protocol['outputs']
        self.records = {}

    def inventory(self):
        files = reject_unexpected_files(self.out)
        total = sum(path.stat().st_size for path in files)
        metadata = sum(path.stat().st_size for path in files if path.suffix.lower() in {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.sh'})
        return files, total, metadata

    def put(self, relative, payload, *, metadata=False):
        self.clock()
        c.canonical(relative)
        c.require(not relative.startswith('outputs/'), 'Callbacks use paths relative to this measurement output')
        target = c.bounded(self.out, relative, must_exist=False)
        files, total, meta = self.inventory()
        limits = self.p['limits']
        reserved_files, reserved_bytes = (0, 0) if self.terminal else (2, 1024**2)
        c.require(len(files) + 1 <= limits['regular_files'] - reserved_files and
                  total + len(payload) <= limits['result_bytes'] - reserved_bytes,
                  'Prospective evidence file/payload limit exceeded')
        c.require(not metadata or meta + len(payload) <= limits['metadata_bytes'] - reserved_bytes,
                  'Prospective metadata limit exceeded')
        c.require(shutil.disk_usage(self.root).free >= len(payload) + limits['reserve_free_bytes'],
                  'Preserve4GiB evidence write reserve')
        c.require(not target.exists(), 'Preserve previously written evidence')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        row = {'path': target.relative_to(self.root).as_posix(), 'sha256': c.sha256(target), 'bytes': target.stat().st_size}
        self.records[relative] = row
        return row

    def json(self, relative, value):
        c.require(Path(relative).suffix == '.json', 'JSON callback extension required')
        payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
        c.require(len(payload) <= 1024**2, 'One metadata record exceeds1MiB')
        return self.put(relative, payload, metadata=True)

    def array(self, relative, value):
        import io
        import numpy as np
        c.require(Path(relative).suffix == '.npy' and isinstance(value, np.ndarray) and
                  value.dtype in (np.dtype('float32'), np.dtype('int64')) and
                  value.flags.c_contiguous and np.isfinite(value).all(), 'Finite contiguous float32/int64 array required')
        c.require(value.nbytes + 4096 <= self.p['limits']['array_file_bytes'], 'Array exceeds per-file bound')
        stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
        return self.put(relative, stream.getvalue())

    def png(self, relative, pixels):
        import io
        import numpy as np
        from PIL import Image
        c.require(Path(relative).suffix == '.png' and isinstance(pixels, np.ndarray) and
                  pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8, 'ExactRGB256uint8PNG required')
        stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
        payload = stream.getvalue()
        c.require(len(payload) <= self.p['limits']['PNG_file_bytes'], 'PNG encoded byte cap exceeded')
        return self.put(relative, payload)


def run(root, protocol_sha):
    root = manual_root(root)  # Before neural/framework imports and output writes.
    started = time.monotonic()
    p = c.verify_transfer(root, protocol_sha, verify_parent=True)
    assets = c.resolve_assets(root, p)
    out = root / p['outputs']
    c.require(not out.exists(), 'Preserve existing measurement; no automatic rerun')
    c.require(shutil.disk_usage(root).free >= p['limits']['initial_free_bytes'], 'Require8GiB free after installation')
    out.mkdir(parents=True)
    stage_started = {'initialization': started}; stage_completed = {}; current_stage = ['initialization']
    counts = {name: 0 for name in p['call_caps']}; event = [0]
    def clock(stage=None):
        if stage is not None:
            c.require(stage in {'initialization', 'measurement'}, 'Unknown finite measurement stage')
            c.require(not (current_stage[0] == 'measurement' and stage == 'initialization'), 'Initialization cannot restart')
            if stage not in stage_started:
                stage_completed[current_stage[0]] = time.monotonic() - stage_started[current_stage[0]]
                stage_started[stage] = time.monotonic()
            current_stage[0] = stage
        now = time.monotonic(); limits = p['limits']
        c.require(now - started <= limits['worker_seconds'], 'Whole900sworker limit reached')
        c.require(now - stage_started[current_stage[0]] <= limits[current_stage[0] + '_seconds'], 'Stage finite time cap reached')
        c.require(shutil.disk_usage(root).free >= limits['reserve_free_bytes'], 'Preserve4GiB measurement reserve')
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            c.require(torch.cuda.memory_reserved() <= limits['max_reserved_GPU_bytes'] and
                      torch.cuda.max_memory_reserved() <= limits['max_reserved_GPU_bytes'], 'CUDA current/peak22GiB limit reached')
        return now - started
    writer = EvidenceWriter(root, p, clock)
    def reserve(kind, context):
        clock()
        expected_keys = ({'case_id', 'mode', 'section'} if kind == 'codeformer_forwards' else
                         {'case_id', 'mode', 'section', 'site', 'directory', 'weights', 'role'})
        c.require(kind in counts and isinstance(context, dict) and set(context) == expected_keys and
                  all(type(value) is str and bool(value) for value in context.values()) and
                  len(json.dumps(context, allow_nan=False, separators=(',', ':')).encode('utf-8')) <= 1024,
                  'Explicit finite structured call provenance required')
        c.require(counts[kind] < p['call_caps'][kind], 'Prospective call limit reached before ' + kind)
        counts[kind] += 1; event[0] += 1
        writer.json('call_trace/%04d.json' % event[0], {'event': event[0], 'before_call': kind,
            'context': context, 'counts': dict(counts), 'elapsed_seconds': clock()})
    stamp = {'pid': os.getpid(), 'protocol_sha256': protocol_sha, 'optimizer_updates': 0,
        'automatic_training': False, 'manual_tmux': True, 'start_unix_seconds': time.time()}
    writer.json('manual_launch.json', stamp)
    def guarded(where, protocol):
        c.require(manual_root(where) == root and protocol == p, 'Manual guarded neural input differs')
    try:
        with finite_signal_scope(p['limits']['worker_seconds']):
            from .neural import execute
            result = execute(root, p, assets['original'], assets['stopped'], {
                'guard': guarded, 'require': c.require, 'clock': clock, 'reserve': reserve,
                'write_json': writer.json, 'write_array': writer.array, 'write_png': writer.png})
            clock()
            c.require(isinstance(result, dict) and result.get('complete') is True and counts == p['call_caps'] and
                      result.get('counts') == counts and
                      result.get('state_before_after_equal') is True and
                      result.get('parameter_grad_fields_all_none') is True and
                      result.get('content_preservation_verified') is True and
                      result.get('repeat_zero_identity_canaries_pass') is True,
                      'Complete exact finite immutable measurement evidence required')
            c.verify_transfer(root, protocol_sha, verify_parent=True)
            terminal = EvidenceWriter(root, p, clock, terminal=True)
            terminal.json('worker_complete.json', {**stamp, 'complete': True, 'seconds': clock(),
                'measurement_result': result, 'counts': counts, 'stage_seconds': {
                    **stage_completed, current_stage[0]: time.monotonic() - stage_started[current_stage[0]]},
                'model_qualified': False, 'next_training_authorized': False, 'automatic_app_promotion': False})
        return 0
    except BaseException as error:
        failure = out / 'worker_failure.json'
        if not failure.exists():
            # Failure serialization must survive an exhausted measurement clock;
            # reserve a small exact terminal record without repeating neural work.
            terminal = EvidenceWriter(root, p, lambda: None, terminal=True)
            terminal.json('worker_failure.json', {**stamp, 'complete': False, 'error_type': type(error).__name__,
                'message': str(error), 'traceback': traceback.format_exc(), 'counts': counts,
                'seconds': time.monotonic() - started, 'partial_evidence_retained': True,
                'model_qualified': False, 'training_success_not_implied': True, 'automatic_retry': False})
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
                  'Retained result payload exceeds1.25GiB; no silent omission')
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
        c.require(len(sizes) + 1 <= p['limits']['regular_files'] and
                  sum(sizes.values()) + 8 * 1024**2 <= p['limits']['result_bytes'],
                  'Export result/source/manifest file or payload allowance exceeded')
        meta = sum(size for name, size in sizes.items() if Path(name).suffix.lower() in
                   {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.sh'})
        c.require(meta + 8 * 1024**2 <= p['limits']['metadata_bytes'], 'Export metadata/source allowance exceeded')
        manifest_name = p['outputs'] + '/export_manifest.json'
        c.require(manifest_name not in files and not (root / manifest_name).exists(),
                  'Preserve prior export manifest')
        manifest = {'format': 'codeformer-bounded-fusion-feasibility-return-v1',
                    'protocol_sha256': protocol_sha, 'files_sha256': files,
                    'files_bytes': sizes, 'parent_assets': p['assets'],
                    'retained_not_exported_sha256': {n: h for n, h in p['files_sha256'].items() if n not in files},
                    'complete_measurement': (out / 'worker_complete.json').is_file(),
                    'failure_present': (out / 'worker_failure.json').is_file(),
                    'optimizer_updates': 0, 'model_qualified': False,
                    'native_CCTV_qualification': False, 'automatic_training': False}
        c.write_new_json(root / manifest_name, manifest)
        manifest_sha = c.sha256(root / manifest_name)
        tar_upper = sum(((size + 511) // 512) * 512 + 4096 for size in sizes.values()) + 8 * 1024**2
        archive_upper = tar_upper + tar_upper // 1000 + 1024**2
        c.require(archive_upper <= p['limits']['archive_bytes'], 'Conservative TAR/gzip bound exceeds1.35GiB')
        c.require(shutil.disk_usage(root).free >= archive_upper + p['limits']['reserve_free_bytes'],
                  'Require conservative export bytes plus4GiB reserve')
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
                          'parent_assets_verified': len(p['assets']),
                          'optimizer_updates': 0, 'neural_calls': 0, 'gradient_queries': 0}))
        return 0
    return export(args.root, args.protocol_sha) if args.export else run(args.root, args.protocol_sha)


if __name__ == '__main__':
    raise SystemExit(main())

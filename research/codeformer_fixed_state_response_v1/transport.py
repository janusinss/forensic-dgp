"""Standard-library transfer checks and bounded exclusive zero-update measurement export.

No returned code, tensor file, model, optimizer or neural framework is loaded.
Parents remain read-only. New result checkpoints and model weights are forbidden.
"""
from datetime import datetime, timezone
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tarfile
import time
import unicodedata

from . import contract as c, budget


OUTPUTS, PREFIX, MODULE = c.OUTPUTS, c.PREFIX, c.MODULE
PAYLOAD_CAP = c.LIMITS['result_payload_bytes']
ARCHIVE_CAP = c.LIMITS['archive_bytes']
METADATA_CAP = c.LIMITS['metadata_bytes']
MEMBER_CAP = c.LIMITS['regular_files']
RESERVE = c.LIMITS['reserve_free_bytes']
EXPORT_SECONDS = c.LIMITS['export_seconds']
HEX = re.compile(r'^[a-f0-9]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)', re.I)
BINARY_SUFFIXES = {'.npy', '.npz', '.png'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.pkl', '.pickle', '.bin'}
SUPER_FILE_NAMES = {'preflight.log', 'diagnostic.log', 'run_receipt.json', 'export_invocation.json',
                    'export.log', 'finished.json'}


def canonical(name):
    c.require(isinstance(name, str) and bool(name), 'Missing portable relative path')
    p = PurePosixPath(name)
    c.require(not p.is_absolute() and p.as_posix() == name and p.parts and
              all(part not in ('', '.', '..') for part in p.parts), 'Noncanonical relative path')
    c.require(unicodedata.normalize('NFC', name) == name and
              not any(char in name for char in '\\:"<>|?*') and
              all(not DEVICE.match(part) and not part.endswith((' ', '.')) and
                  len(part.encode('utf-16-le')) <= 510 and
                  all(ord(char) >= 32 for char in part) for part in p.parts), 'Nonportable relative path')
    return p


def _no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            c.require(not stat.S_ISLNK(info.st_mode) and
                      not getattr(info, 'st_file_attributes', 0) & 1024, 'Linked/reparse path forbidden')
    return path


def _regular(root, name, *, unique_link=True):
    canonical(name)
    root = _no_links(root)
    path = _no_links(c.safe_path(root, name))
    info = path.lstat()
    c.require(stat.S_ISREG(info.st_mode), 'Nonregular file forbidden: ' + name)
    c.require(not unique_link or info.st_nlink == 1, 'Hard-linked packet/output forbidden: ' + name)
    return path


def _walk(root):
    """Inventory every new-root file without traversing links or hidden namespaces."""
    root = _no_links(root)
    c.require(root.is_dir(), 'Exact packet root must be a directory')
    files, portable = {}, {}
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            canonical(relative)
            folded = relative.casefold()
            c.require(folded not in portable, 'Duplicate/colliding portable namespace')
            portable[folded] = relative
            _no_links(path)
            info = path.lstat()
            if name in dirs:
                c.require(stat.S_ISDIR(info.st_mode), 'Non-directory namespace component')
            else:
                c.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                          'Only unlinked regular new-root files are permitted')
                files[relative] = path
    return files


def _binding_map(mapping, label):
    c.require(isinstance(mapping, dict), 'Missing ' + label + ' map')
    folded = set()
    for name, row in mapping.items():
        canonical(name)
        c.require(name.casefold() not in folded, 'Colliding ' + label + ' paths')
        folded.add(name.casefold())
        c.require(isinstance(row, dict) and set(row) == {'sha256', 'bytes'} and
                  isinstance(row['sha256'], str) and HEX.fullmatch(row['sha256']) and
                  type(row['bytes']) is int and row['bytes'] >= 0, 'Invalid ' + label + ' binding')


def row_ids(root, protocol):
    fitted = c.read_json(_regular(root, protocol['fitted_index_file']))
    heldout = c.read_json(_regular(root, protocol['new_pairs_file']))
    result = c.expected_row_ids(fitted, heldout)
    c.require(set(result) == {'fitted', 'heldout'} and len(result['fitted']) == 200 and len(result['heldout']) == 64,
              'Exact200fitted/64heldout row namespace required')
    for values in result.values():
        c.require(len(values) == len(set(values)) and all(isinstance(name, str) and
                  re.fullmatch(r'[A-Za-z0-9_.-]+', name) and name not in ('.', '..') for name in values),
                  'Unique canonical measurement row IDs required')
    c.require(not set(result['fitted']) & set(result['heldout']), 'Fitted/heldout row namespace overlap')
    return result


def verify_transfer(root, protocol_or_sha, protocol_sha=None):
    """Fresh bytes for every new static input/source and both read-only parents."""
    root = _no_links(root); supplied = protocol_or_sha if protocol_sha is not None else None
    protocol_sha = protocol_or_sha if protocol_sha is None else protocol_sha
    c.require(isinstance(protocol_sha, str) and HEX.fullmatch(protocol_sha), 'Exact protocol SHA required')
    protocol_path = _regular(root, 'protocol.json')
    c.require(protocol_path.stat().st_size <= 8 * 1024**2 and c.sha256(protocol_path) == protocol_sha,
              'Frozen protocol SHA/size differs')
    p = c.read_json(protocol_path); c.verify_protocol(p)
    c.require(supplied is None or c.equal(supplied, p), 'Caller protocol differs from frozen bytes')
    files, sizes = p['files_sha256'], p['files_bytes']; checked = {}; folded = set()
    pair_doc = c.read_json(_regular(root, p['new_pairs_file']))
    pair_paths = c.expected_pair_paths(pair_doc); pngs = c.expected_input_pngs(pair_doc)
    c.require(len(pair_paths) == 128 and len(pngs) == 64 and pngs <= pair_paths <= set(files),
              'Exact64PNG/64pair-record static input assets required')
    for name, digest in sorted(files.items()):
        canonical(name)
        c.require(name.casefold() not in folded and not name.startswith('outputs/') and name != c.LOCK_NAME and
                  (name in pngs or Path(name).suffix.lower() in {'.py', '.json', '.md', '.txt', '.sha256'}),
                  'Unexpected static source/input namespace')
        folded.add(name.casefold()); path = _regular(root, name); before = path.stat()
        c.require(before.st_size == sizes[name] and c.sha256(path) == digest and
                  (not name.endswith('.png') or name in pngs and before.st_size <= c.LIMITS['one256PNG_bytes']),
                  'Static bytes or exactPNG allowlist differs: ' + name)
        after = path.stat()
        c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                  (after.st_size, after.st_mtime_ns, after.st_ino), 'Static file changed while hashing')
        checked[name] = {'sha256': digest, 'bytes': sizes[name]}
    c.require(len(files) + 1 <= c.LIMITS['source_packet_files'] and
              sum(sizes.values()) + protocol_path.stat().st_size <= c.LIMITS['source_packet_bytes'],
              'Static source/input packet exceeds256files/32MiB')
    inventory = _walk(root)
    c.require({name for name in inventory if not name.startswith('outputs/') and name != c.LOCK_NAME} == set(files) | {'protocol.json'},
              'Unbound static/importable namespace')
    parents = c.resolve_parents(root)
    c.require(set(parents) == set(p['parent_roots']) == set(p['dependencies']) == {'tone_training', 'adain'},
              'Exact two read-only parent aliases required')
    parent_counts = {}
    for alias, mapping in p['dependencies'].items():
        _binding_map(mapping, alias); parent = parents[alias]
        old = _regular(parent, 'protocol.json', unique_link=False)
        c.require(c.sha256(old) == p['parent_protocol_sha256'][alias] and
                  mapping.get('protocol.json', {}).get('sha256') == p['parent_protocol_sha256'][alias],
                  'Exact parent protocol binding differs: ' + alias)
        for name, pin in sorted(mapping.items()):
            source = _regular(parent, name, unique_link=False); before = source.stat()
            c.require(before.st_size == pin['bytes'] and c.sha256(source) == pin['sha256'],
                      'Read-only parent hash/size differs: ' + alias + '/' + name)
            after = source.stat()
            c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                      (after.st_size, after.st_mtime_ns, after.st_ino), 'Read-only parent changed while hashing')
        parent_counts[alias] = len(mapping)
    for label, asset in p['assets'].items():
        c.require(asset['root'] in parents and p['dependencies'][asset['root']].get(asset['path']) ==
                  {key: asset[key] for key in ('sha256', 'bytes')}, 'Named immutable asset differs: ' + label)
    ids = row_ids(root, p)
    return {'format': 'codeformer-fixed-state-response-transfer-v1', 'complete': True,
        'protocol_sha256': protocol_sha, 'protocol': p, 'static_files': checked,
        'parent_roots': p['parent_roots'], 'parent_protocol_sha256': p['parent_protocol_sha256'],
        'dependencies': p['dependencies'], 'dependencies_verified_by_parent': parent_counts,
        'dependencies_verified': sum(parent_counts.values()), 'static_input_PNGs_verified': 64,
        'fitted_rows': len(ids['fitted']), 'heldout_rows': len(ids['heldout']), 'readonly': True,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_steps': 0}


def _measure_files(root, files):
    hashes, sizes, metadata = {}, {}, 0
    for name in sorted(files):
        path = _regular(root, name); size = path.stat().st_size
        c.require(not re.search(r'\.(?:pt|pth|ckpt|safetensors|pkl|pickle|bin)(?:$|[.\-])', path.name.lower()),
                  'New model/checkpoint serialization forbidden')
        if path.suffix.lower() in {'.npy', '.npz'}:
            c.require(size <= c.LIMITS['one_array_bytes'], 'Array file cap exceeded')
        elif path.suffix.lower() == '.png':
            c.require(size <= c.LIMITS['one256PNG_bytes'], 'PNG256 file cap exceeded')
        else: metadata += size
        hashes[name], sizes[name] = c.sha256(path), size
    c.require(len(sizes) <= MEMBER_CAP and sum(sizes.values()) <= PAYLOAD_CAP and metadata <= METADATA_CAP,
              'File/payload/metadata result cap exceeded; no silent omission')
    return hashes, sizes, metadata


def output_coverage(root, files, protocol):
    ids = row_ids(root, protocol); all_ids = ids['fitted'] + ids['heldout']; banks = {}; missing = []
    for bank in ('initial', 'final'):
        expected = {OUTPUTS + '/' + bank + '/' + row + '/' + name for row in all_ids
                    for name in ('raw.npy', 'native.png', 'record.json')}
        actual = {name for name in files if name.startswith(OUTPUTS + '/' + bank + '/')}
        c.require(actual <= expected, 'Unrecognized measurement row namespace')
        absent = sorted(expected - actual); missing.extend(absent)
        banks[bank] = {'expected_rows': 264, 'complete_rows': sum(all(OUTPUTS + '/' + bank + '/' + row + '/' + name in files
                      for name in ('raw.npy', 'native.png', 'record.json')) for row in all_ids),
                      'expected_members': len(expected), 'present_members': len(actual), 'missing_members': absent}
    parent_raw = {OUTPUTS + '/new/' + row + '/' + name for row in ids['heldout']
                  for name in ('original_raw.npy', 'base_raw.npy')}
    actual_raw = {name for name in files if name.startswith(OUTPUTS + '/new/')}
    c.require(actual_raw <= parent_raw, 'Unrecognized fresh parentRAW namespace')
    missing.extend(sorted(parent_raw - actual_raw))
    return {'complete': not missing, 'banks': banks, 'missing_members': missing,
        'head_RAW_PNG_pairs': sum(OUTPUTS + '/' + bank + '/' + row + '/raw.npy' in files and
                                 OUTPUTS + '/' + bank + '/' + row + '/native.png' in files
                                 for bank in banks for row in all_ids),
        'row_records': sum(OUTPUTS + '/' + bank + '/' + row + '/record.json' in files for bank in banks for row in all_ids),
        'fresh_parent_RAW_arrays': len(actual_raw), 'fixed_static_input_PNGs': 64,
        'scope': 'File/namespace coverage only; numerical/pixel/visual response audit remains separate'}


def _terminal_evidence(root, files, protocol_sha, protocol):
    success, failure = OUTPUTS + '/run_results.json', OUTPUTS + '/failure.json'
    supervisor = 'outputs/supervision/run_receipt.json'
    c.require(supervisor in files, 'Closed terminal supervisor receipt required')
    supervised = c.read_json(_regular(root, supervisor))
    c.require(supervised.get('worker_terminal') is True and supervised.get('protocol_sha256') == protocol_sha,
              'Terminal supervisor protocol binding differs')
    c.require(not (success in files and failure in files), 'Conflicting worker terminal records')
    terminal = success if success in files else failure if failure in files else None
    value = {} if terminal is None else c.read_json(_regular(root, terminal))
    counts = value.get('counts'); zero_keys = ('restorer_updates_this_run', 'head_updates_this_run', 'gradient_queries',
        'backward_calls', 'optimizer_constructions', 'optimizer_steps')
    if terminal is not None:
        c.require(value.get('protocol_sha256') == protocol_sha and type(value.get('measurement_complete')) is bool and
                  isinstance(counts, dict) and set(counts) == set(c.COUNTS), 'Exact measurement terminal schema required')
        c.require(all(type(counts[key]) is int and 0 <= counts[key] <= cap for key, cap in c.COUNTS.items()), 'Operation cap exceeded')
        c.require(all(type(value.get(key)) is int and value[key] == 0 for key in zero_keys), 'Zero-update/gradient/optimizer contract violated')
    coverage = output_coverage(root, files, protocol)
    if terminal == success:
        c.require(value.get('complete') is True and value['measurement_complete'] is True and counts == c.COUNTS and
                  coverage['complete'] and coverage['head_RAW_PNG_pairs'] == 528 and coverage['fresh_parent_RAW_arrays'] == 128 and
                  coverage['row_records'] == 528, 'Complete656call/528headpairs/128parentRAW measurement required')
        c.require(value.get('failure', False) is False, 'Conflicting successful measurement failure flag')
    elif terminal == failure:
        c.require(value.get('complete') is True and value.get('failure') is True and
                  value['measurement_complete'] is False, 'Explicit complete failure receipt required')
    return {'complete_measurement': terminal == success, 'failure_present': failure in files,
        'terminal_evidence': [name for name in (success, failure, supervisor) if name in files],
        'worker_recorded_counts': counts, 'worker_recorded_measurement_complete': value.get('measurement_complete'),
        **{key: value.get(key) for key in zero_keys}, 'counts_unknown_without_worker_receipt': terminal is None,
        'checkpoint_members': [], 'output_coverage': coverage}

def _process_rows():
    result = subprocess.run(['ps', '-eo', 'pid=,args='], check=True, capture_output=True, text=True, timeout=20)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        c.require(len(fields) == 2 and fields[0].isdigit(), 'Cannot classify full process inventory')
        rows.append((int(fields[0]), fields[1]))
    return rows


def require_worker_terminal(root):
    """A terminal receipt never overrides a live worker or retained PID lock."""
    lock = Path(root) / c.LOCK_NAME
    if lock.exists() or lock.is_symlink():
        _regular(root,c.LOCK_NAME)
        value = c.read_json(lock)
        pid = value.get('pid')
        c.require(type(pid) is int and pid > 0, 'Invalid retained worker lock PID')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError:
            c.require(False, 'Cannot establish retained worker PID is terminal')
        else:
            c.require(False, 'Worker PID is alive; preserve it and do not export')
    for pid, command in _process_rows():
        if pid == os.getpid():
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            c.require(MODULE not in command, 'Cannot classify potential diagnostic process')
            continue
        c.require(not (MODULE in words and '--run' in words),
                  'Diagnostic worker/wrapper remains live; do not export')


def _write_reserved_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    c.require(len(payload) <= c.LIMITS['one_JSON_bytes'], 'One export JSON exceeds declared2MiB cap')
    c.require(shutil.disk_usage(Path(path).parent).free >= RESERVE + len(payload) + 65536,
              'Metadata write would cross4GiB disk reserve')
    c.write_new_json(path, value)


class BoundedArchive:
    def __init__(self, stream, deadline, reserve_path):
        self.stream, self.deadline, self.reserve_path, self.bytes = stream, deadline, reserve_path, 0

    def check(self, size=0):
        c.require(time.monotonic() <= self.deadline, 'Finite export time exceeded')
        c.require(self.bytes + size <= ARCHIVE_CAP, 'Compressed archive cap exceeded')
        c.require(shutil.disk_usage(self.reserve_path).free >= RESERVE + size,
                  'Export disk reserve would be crossed; preserve partial evidence')

    def write(self, payload):
        self.check(len(payload))
        written = self.stream.write(payload)
        c.require(written == len(payload), 'Short archive write')
        self.bytes += written
        return written

    def flush(self):
        self.stream.flush()


class HashReader:
    def __init__(self, stream, bounded):
        self.stream, self.bounded, self.digest, self.bytes = stream, bounded, hashlib.sha256(), 0

    def read(self, size=-1):
        self.bounded.check()
        value = self.stream.read(size)
        self.digest.update(value); self.bytes += len(value)
        return value


@contextmanager
def _finite_export():
    """Bound hashing as well as streaming on the actual Linux export route."""
    if not hasattr(signal, 'setitimer'):
        yield  # CPU fixtures on Windows; actual route is guarded Linux-only.
        return
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame):
        raise TimeoutError('Finite export300second wall limit exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(EXPORT_SECONDS, previous_timer[0]) if previous_timer[0] > 0 else EXPORT_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        remaining = max(0.000001, previous_timer[0] - (time.monotonic() - started)) if previous_timer[0] > 0 else 0
        signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def export_results(root, protocol_sha):
    with _finite_export():
        return _export_results(root, protocol_sha)


def _export_results(root, protocol_sha):
    root = c.manual_root(root); started = time.monotonic()
    verification = verify_transfer(root, protocol_sha); p = verification['protocol']
    projection = budget.resource_ledger(len(p['files_bytes']) + 1,
        sum(p['files_bytes'].values()) + (root / 'protocol.json').stat().st_size)
    require_worker_terminal(root); inventory = _walk(root)
    status = _terminal_evidence(root, inventory, protocol_sha, p)
    for name in inventory:
        if name.startswith('outputs/'):
            c.require(name.startswith(OUTPUTS + '/') or name.startswith('outputs/supervision/') and
                      name[len('outputs/supervision/'):] in SUPER_FILE_NAMES, 'Unexpected output namespace')
    archive = _no_links(Path.home() / (PREFIX + '-results.tar.gz'))
    checksum = Path(str(archive) + '.sha256'); receipt_path = archive.with_name(PREFIX + '-export.json')
    for path in (archive, checksum, receipt_path):
        c.require(not path.exists() and not path.is_symlink(), 'Preserve existing/partial export: ' + path.name)
    out = root / OUTPUTS; out.mkdir(parents=True, exist_ok=True)
    dependency_name, manifest_name = OUTPUTS + '/dependency_verification.json', OUTPUTS + '/export_manifest.json'
    c.require(not (root / dependency_name).exists() and not (root / manifest_name).exists(),
              'Preserve prior dependency/export evidence; no automatic retry')
    dependency_receipt = dict(verification); dependency_receipt.pop('protocol')
    _write_reserved_json(root / dependency_name, dependency_receipt)
    inventory = _walk(root)
    excluded = {c.LOCK_NAME, 'outputs/supervision/export.log', 'outputs/supervision/finished.json'}
    hashes, sizes, metadata = _measure_files(root, set(inventory) - excluded)
    manifest = {'format': 'codeformer-fixed-state-response-return-v1', 'protocol_sha256': protocol_sha,
        'files_sha256': hashes, 'files_bytes': sizes, 'parent_roots': p['parent_roots'],
        'parent_protocol_sha256': p['parent_protocol_sha256'], 'dependencies': p['dependencies'], **status,
        'actual_count_source': 'Worker completion/failure receipt; missing remains unknown, never inferred zero',
        'new_checkpoint_serialization': False, 'parent_files_modified_or_copied': False,
        'model_qualified': False, 'automatic_training': False, 'native_CCTV_qualification': False,
        'prospective_source_resource_ledger': projection,
        'closure_boundary': {'immutable_pre_export_supervision_included': True,
            'separately_retained': ['outputs/supervision/export.log', 'outputs/supervision/finished.json',
                                   receipt_path.name, checksum.name, PREFIX + '-supervisor-final.json'],
            'external_final_companion': PREFIX + '-supervisor-final.json',
            'reason': 'Final companion binds exact export log and parsed finished receipt/hash after archive closure; no self-hash'},
        'UTC': datetime.now(timezone.utc).isoformat()}
    payload = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    c.require(len(payload) <= c.LIMITS['one_JSON_bytes'], 'Export manifest exceeds declared2MiB JSON cap')
    c.require(len(sizes) + 1 <= MEMBER_CAP and sum(sizes.values()) + len(payload) <= PAYLOAD_CAP and
              metadata + len(payload) <= METADATA_CAP, 'Manifest/source/input/result aggregate cap exceeded')
    tar_bytes = sum(512 + ((size + 511) // 512) * 512 for size in [*sizes.values(), len(payload)])
    tar_upper = ((tar_bytes + 1024 + 10239) // 10240) * 10240
    archive_upper = tar_upper + (tar_upper >> 12) + (tar_upper >> 14) + (tar_upper >> 25) + 64
    c.require(archive_upper <= ARCHIVE_CAP, 'Conservative USTAR/defaultgzip envelope exceeds archive cap')
    c.require(shutil.disk_usage(archive.parent).free >= archive_upper + RESERVE and
              shutil.disk_usage(root).free >= len(payload) + RESERVE, 'Require export envelope plus4GiB reserve')
    # Check USTAR filename representation before any HOME archive is created.
    for name, size in {**sizes, manifest_name: len(payload)}.items():
        info = tarfile.TarInfo(name); info.size, info.mode, info.mtime = size, 0o644, 0
        info.tobuf(format=tarfile.USTAR_FORMAT, encoding='utf-8', errors='strict')
    _write_reserved_json(root / manifest_name, manifest)
    c.require((root / manifest_name).stat().st_size == len(payload), 'Written manifest serialization differs')
    hashes = {**hashes, manifest_name: c.sha256(root / manifest_name)}; sizes = {**sizes, manifest_name: len(payload)}
    with archive.open('xb') as destination:
        bounded = BoundedArchive(destination, started + EXPORT_SECONDS, archive.parent)
        with gzip.GzipFile(filename='', mode='wb', fileobj=bounded, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as bundle:
                for name in sorted(hashes):
                    path = _regular(root, name); before = path.stat()
                    c.require(before.st_size == sizes[name], 'Export source size changed')
                    info = tarfile.TarInfo(name); info.size, info.mode, info.mtime = sizes[name], 0o644, 0
                    info.uid = info.gid = 0; info.uname = info.gname = ''
                    with path.open('rb') as source:
                        read = HashReader(source, bounded); bundle.addfile(info, read)
                        c.require(read.bytes == sizes[name] and read.digest.hexdigest() == hashes[name], 'Export source bytes changed')
                    after = path.stat()
                    c.require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                              (after.st_ino, after.st_size, after.st_mtime_ns), 'Export source changed during streaming')
        destination.flush(); os.fsync(destination.fileno())
    c.require(time.monotonic() - started <= EXPORT_SECONDS, 'Finite export time cap exceeded')
    archive_sha = c.sha256(archive)
    c.require(shutil.disk_usage(archive.parent).free >= RESERVE + 65536, 'Checksum/receipt reserve unavailable')
    with checksum.open('xb') as stream:
        stream.write((archive_sha + '  ' + archive.name + '\n').encode('ascii')); stream.flush(); os.fsync(stream.fileno())
    receipt = {'format': 'codeformer-fixed-state-response-export-v1', 'complete': True,
        'archive_sha256': archive_sha, 'bytes': archive.stat().st_size, 'manifest_member': manifest_name,
        'manifest_sha256': hashes[manifest_name], 'protocol_sha256': protocol_sha, **status,
        'regular_members': len(hashes), 'payload_bytes': sum(sizes.values()), 'seconds': time.monotonic() - started,
        'actual_count_source': manifest['actual_count_source'], 'new_checkpoint_serialization': False,
        'model_qualified': False, 'measurement_success_not_model_acceptance': True}
    _write_reserved_json(receipt_path, receipt)
    return receipt



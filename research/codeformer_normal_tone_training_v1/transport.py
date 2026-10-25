"""Standard-library transfer checks and bounded exclusive normal-tone export.

No returned code, tensor file, model, optimizer or neural framework is loaded.
Parent dependencies are read-only hash inputs and never exported.
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

from . import contract as c


OUTPUTS = 'outputs/normal_tone_training_v1'
PREFIX = 'cctv-codeformer-normal-tone-training-v1'
MODULE = 'research.codeformer_normal_tone_training_v1.worker'
PAYLOAD_CAP = 3758096384
ARCHIVE_CAP = 3865470566
METADATA_CAP = 128 * 1024**2
MEMBER_CAP = 7000
RESERVE = 4 * 1024**3
EXPORT_SECONDS = 600
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


def verify_transfer(root, protocol_sha):
    """Return a read-only metadata receipt containing the verified protocol."""
    root = _no_links(root)
    c.require(isinstance(protocol_sha, str) and HEX.fullmatch(protocol_sha), 'Exact protocol SHA-256 required')
    protocol_path = _regular(root, 'protocol.json')
    c.require(protocol_path.stat().st_size <= 8 * 1024**2, 'Protocol metadata exceeds8MiB')
    c.require(c.sha256(protocol_path) == protocol_sha, 'Frozen protocol hash differs')
    p = c.read_json(protocol_path)
    c.verify_protocol(p)
    files, sizes = p['files_sha256'], p['files_bytes']
    c.require(isinstance(files, dict) and bool(files) and isinstance(sizes, dict) and
              files.keys() == sizes.keys() and 'protocol.json' not in files, 'Exact static hash/size namespace required')
    checked, folded = {}, set()
    for name, digest in sorted(files.items()):
        canonical(name)
        c.require(name.casefold() not in folded and not name.startswith('outputs/') and
                  name != 'normal_tone_training_v1.lock' and
                  Path(name).suffix.lower() not in FORBIDDEN_SUFFIXES | BINARY_SUFFIXES | {'.tar', '.gz', '.zip'},
                  'Invalid new packet source namespace')
        folded.add(name.casefold())
        c.require(isinstance(digest, str) and HEX.fullmatch(digest) and
                  type(sizes[name]) is int and sizes[name] >= 0, 'Invalid static file binding')
        path = _regular(root, name)
        c.require(path.stat().st_size == sizes[name] and c.sha256(path) == digest, 'Static source/size differs: ' + name)
        checked[name] = {'sha256': digest, 'bytes': sizes[name]}
    c.require(len(files) + 1 <= MEMBER_CAP and sum(sizes.values()) + protocol_path.stat().st_size <= PAYLOAD_CAP and
              sum(size for name, size in sizes.items() if Path(name).suffix.lower() not in BINARY_SUFFIXES) +
              protocol_path.stat().st_size <= METADATA_CAP, 'Static packet file/payload/metadata cap exceeded')
    inventory = _walk(root)
    static = {name for name in inventory if not name.startswith('outputs/') and name != 'normal_tone_training_v1.lock'}
    c.require(static == set(files) | {'protocol.json'}, 'Unbound static/importable packet namespace')
    dependencies=p['dependencies']
    _binding_map(dependencies,'AdaIN parent dependency')
    parent=c.resolve_parent(root)
    parent_protocol=_regular(parent,'protocol.json',unique_link=False)
    c.require(c.sha256(parent_protocol)==p['parent_protocol_sha256'],'Exact AdaIN parent protocol differs')
    old=c.read_json(parent_protocol)
    for name,row in sorted(dependencies.items()):
        path=_regular(parent,name,unique_link=False)
        c.require(path.stat().st_size==row['bytes'] and c.sha256(path)==row['sha256'],
                  'Parent dependency hash/size differs: '+name)
    c.require(all(name in dependencies and dependencies[name]['sha256']==value
                  for name,value in old['files_sha256'].items()),
              'Entire frozen AdaIN immutable namespace must remain bound')
    c.require(p['assets']==old['assets'] and all(p[key]==old[key] for key in c.PARENT_CONTROLS),
              'Exact original assets/training/HQ/validation parity joins required')
    refs=c.read_json(_regular(parent,p['data_manifest'],unique_link=False))
    comparisons=c.read_json(_regular(parent,p['comparison_cases'],unique_link=False))
    training=c.read_json(_regular(parent,p['training_parity']['path'],unique_link=False))
    hq=c.read_json(_regular(parent,p['hq_cache_parity']['path'],unique_link=False))
    return {'format':'codeformer-normal-tone-training-transfer-v1','complete':True,
        'protocol_sha256':protocol_sha,'protocol':p,'static_files':checked,'parent_root':p['parent_dir'],
        'parent_protocol_sha256':p['parent_protocol_sha256'],'dependencies':dependencies,
        'dependencies_verified':len(dependencies),'reference_manifest_verified':p['data_manifest'],
        'comparison_manifest_verified':p['comparison_cases'],'training_parity_verified':p['training_parity'],
        'HQ_parity_verified':p['hq_cache_parity'],'readonly':True,'neural_calls':0,
        'gradient_queries':0,'optimizer_steps':0}


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
    lock = Path(root) / 'normal_tone_training_v1.lock'
    if lock.exists() or lock.is_symlink():
        _regular(root, 'normal_tone_training_v1.lock')
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


def checkpoint_kind(name):
    if name in c.CHECKPOINT_PATHS:return 'final'
    for base in c.CHECKPOINT_PATHS:
        if re.fullmatch(re.escape(base)+r'\.tmp-[0-9a-f]{32}',name):return 'temporary'
    return None


def _measure_files(root, files):
    hashes,sizes,metadata,temporary={}, {},0,0
    for name in sorted(files):
        path=_regular(root,name);size=path.stat().st_size;kind=checkpoint_kind(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(suffix+'.tmp-' in name.lower() for suffix in FORBIDDEN_SUFFIXES):
            c.require(kind is not None,'Checkpoint export outside exact filename gates forbidden')
        if kind is not None:
            cap=384*1024**2 if name==c.CHECKPOINT_PATHS[0] else 832*1024**2
            c.require(size<= (1024**3 if kind=='temporary' else cap),'Checkpoint file cap exceeded')
            if kind=='temporary':temporary+=size
        elif path.suffix.lower() in {'.npy','.npz'}:
            c.require(size<=c.LIMITS['array_file_bytes'],'Array file cap exceeded')
        elif path.suffix.lower()=='.png':
            cap=3*512*512+4096 if name==OUTPUTS+'/first_training_example/clean512.png' else c.LIMITS['PNG_file_bytes']
            c.require(size<=cap,'PNG file cap exceeded')
        elif path.suffix.lower() not in BINARY_SUFFIXES:
            metadata+=size
        hashes[name],sizes[name]=c.sha256(path),size
    c.require(temporary<=1024**3 and len(sizes)<=MEMBER_CAP and sum(sizes.values())<=PAYLOAD_CAP and metadata<=METADATA_CAP,
              'Result/checkpoint/file/payload/metadata cap exceeded; no silent omission')
    return hashes,sizes,metadata


def _write_reserved_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    c.require(len(payload) <= 8 * 1024**2, 'One export metadata record exceeds8MiB')
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


def _terminal_evidence(root,files,protocol_sha):
    success=OUTPUTS+'/run_results.json';failure=OUTPUTS+'/failure.json';supervision='outputs/supervision/run_receipt.json'
    c.require(supervision in files,'Closed terminal supervisor receipt required')
    supervised=c.read_json(_regular(root,supervision))
    c.require(supervised.get('worker_terminal') is True and supervised.get('protocol_sha256')==protocol_sha,
              'Terminal supervisor protocol binding differs')
    c.require(not(success in files and failure in files),'Conflicting worker terminal records')
    terminal=success if success in files else failure if failure in files else None
    value={} if terminal is None else c.read_json(_regular(root,terminal))
    c.require(terminal is None or value.get('protocol_sha256')==protocol_sha,'Terminal worker protocol binding differs')
    counts=value.get('counts');completed=value.get('completed_updates')
    inflight=value.get('optimizer_inflight')
    if completed is not None:
        c.require(type(completed) is int and 0<=completed<=50,'Training completed updates exceed finite cap')
    if counts is not None:
        c.require(isinstance(counts,dict),'Worker counts must be an object')
        for key,cap in c.COUNTS.items():
            c.require(type(counts.get(key)) is int and 0<=counts[key]<=cap,'Worker count exceeds finite cap: '+key)
        c.require(type(counts.get('optimizer_updates')) is int and counts['optimizer_updates']==completed and
                  completed<=counts['optimizer_step_attempts'],'Completed committed updates disagree')
    c.require(inflight is None or type(inflight) is bool,'Optimizer inflight must be explicit boolean or unknown')
    if terminal==success:
        c.require(value.get('complete') is True and completed==50 and counts is not None and
                  all(counts[key]==cap for key,cap in c.COUNTS.items()),'Successful worker census incomplete')
        c.require(all(name in files for name in c.CHECKPOINT_PATHS[:2]),
                  'Successful training must retain both candidate checkpoint endpoints')
    return {'complete_training':terminal==success,'failure_present':failure in files,
        'terminal_evidence':[n for n in (success,failure,supervision) if n in files],
        'worker_recorded_counts':counts,'completed_committed_training_updates':completed,
        'worker_recorded_optimizer_inflight_at_failure':inflight,
        'training_updates_unknown_without_worker_receipt':terminal is None,
        'candidate_checkpoint_members':[n for n in files if checkpoint_kind(n)=='final'],
        'incomplete_checkpoint_temporary_members':[n for n in files if checkpoint_kind(n)=='temporary']}


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
        raise TimeoutError('Finite export600second wall limit exceeded')
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
    """Export every immutable new-root result and source; preserve partial failures."""
    root = c.manual_root(root)
    started = time.monotonic()
    verification = verify_transfer(root, protocol_sha)
    p = verification['protocol']
    require_worker_terminal(root)
    inventory = _walk(root)
    status = _terminal_evidence(root, inventory, protocol_sha)
    for name in inventory:
        if name.startswith('outputs/'):
            c.require(name.startswith(OUTPUTS + '/') or
                      (name.startswith('outputs/supervision/') and
                       name[len('outputs/supervision/'):] in SUPER_FILE_NAMES), 'Unexpected output namespace')
    archive = _no_links(Path.home() / (PREFIX + '-results.tar.gz'))
    checksum, receipt_path = Path(str(archive) + '.sha256'), archive.with_name(PREFIX + '-export.json')
    for path in (archive, checksum, receipt_path):
        c.require(not path.exists() and not path.is_symlink(), 'Preserve existing/partial export: ' + path.name)
    out = root / OUTPUTS
    c.require(shutil.disk_usage(root).free >= RESERVE + 1024**2, 'Export metadata reserve unavailable')
    out.mkdir(parents=True, exist_ok=True)
    dependency_name = OUTPUTS + '/dependency_verification.json'
    manifest_name = OUTPUTS + '/export_manifest.json'
    c.require(not (root / dependency_name).exists() and not (root / manifest_name).exists(),
              'Preserve prior dependency/export evidence; no automatic export retry')
    dependency_receipt = dict(verification)
    dependency_receipt.pop('protocol')
    _write_reserved_json(root / dependency_name, dependency_receipt)
    inventory = _walk(root)
    # Export's own active stdout and post-export closure cannot contain their final
    # bytes inside the archive they describe. The immutable boundary is explicit.
    excluded = {'normal_tone_training_v1.lock', 'outputs/supervision/export.log', 'outputs/supervision/finished.json'}
    names = set(inventory) - excluded
    hashes, sizes, metadata = _measure_files(root, names)
    manifest = {'format': 'codeformer-normal-tone-return-v1', 'protocol_sha256': protocol_sha,
                'files_sha256': hashes, 'files_bytes': sizes, 'parent_dir':p['parent_dir'],'parent_protocol_sha256':p['parent_protocol_sha256'],'dependencies':p['dependencies'],
                **status, 'committed_training_updates':status['completed_committed_training_updates'],
                'actual_step_count_source':'worker completion/failure receipt; unknown is never inferred zero',
                'model_qualified': False, 'automatic_training': False, 'native_CCTV_qualification': False,
                'not_exported_parent_assets': True,
                'closure_boundary': {'immutable_pre_export_supervision_included': True,
                    'separately_retained': ['outputs/supervision/export.log', 'outputs/supervision/finished.json',
                                           receipt_path.name, checksum.name, PREFIX+'-supervisor-final.json'],
                    'external_final_companion': PREFIX+'-supervisor-final.json',
                    'reason': 'Supervisor final companion binds exact export log and finished bytes after archive closure; no self-hash'},
                'UTC': datetime.now(timezone.utc).isoformat()}
    payload = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    c.require(len(sizes) + 1 <= MEMBER_CAP and sum(sizes.values()) + len(payload) <= PAYLOAD_CAP and
              metadata + len(payload) <= METADATA_CAP, 'Manifest/source/result total cap exceeded')
    tar_upper = sum(((size + 511) // 512) * 512 + 8192 for size in [*sizes.values(), len(payload)]) + 10240
    archive_upper = tar_upper + tar_upper // 1000 + 1024**2
    c.require(archive_upper <= ARCHIVE_CAP, 'Conservative TAR/gzip bound exceeds archive cap')
    c.require(shutil.disk_usage(archive.parent).free >= archive_upper + RESERVE and
              shutil.disk_usage(root).free >= len(payload) + RESERVE,
              'Require conservative export space plus4GiB reserve')
    _write_reserved_json(root / manifest_name, manifest)
    manifest_size = (root / manifest_name).stat().st_size
    c.require(len(sizes) + 1 <= MEMBER_CAP and sum(sizes.values()) + manifest_size <= PAYLOAD_CAP and
              metadata + manifest_size <= METADATA_CAP, 'Actual written manifest total cap exceeded')
    hashes = {**hashes, manifest_name: c.sha256(root / manifest_name)}
    sizes = {**sizes, manifest_name: manifest_size}
    with archive.open('xb') as destination:
        bounded = BoundedArchive(destination, started + EXPORT_SECONDS, archive.parent)
        with gzip.GzipFile(filename='', mode='wb', fileobj=bounded, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as bundle:
                for name in sorted(hashes):
                    path = _regular(root, name)
                    before = path.stat()
                    c.require(before.st_size == sizes[name], 'Export source size changed')
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = sizes[name], 0o644, 0
                    info.uid = info.gid = 0
                    with path.open('rb') as source:
                        read = HashReader(source, bounded)
                        bundle.addfile(info, read)
                        c.require(read.bytes == sizes[name] and read.digest.hexdigest() == hashes[name],
                                  'Export source bytes changed during streaming')
                    after = path.stat()
                    c.require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                              (after.st_ino, after.st_size, after.st_mtime_ns), 'Export source replaced or changed')
        destination.flush(); os.fsync(destination.fileno())
    c.require(time.monotonic() - started <= EXPORT_SECONDS, 'Finite export time exceeded')
    archive_sha = c.sha256(archive)
    c.require(shutil.disk_usage(archive.parent).free >= RESERVE + 65536,
              'Checksum/receipt reserve unavailable; preserve complete archive')
    with checksum.open('xb') as output:
        output.write((archive_sha + '  ' + archive.name + '\n').encode('ascii'))
        output.flush(); os.fsync(output.fileno())
    receipt = {'format': 'codeformer-normal-tone-export-v1', 'complete': True,
               'archive_sha256': archive_sha, 'bytes': archive.stat().st_size,
               'manifest_member': manifest_name, 'manifest_sha256': hashes[manifest_name],
               'protocol_sha256': protocol_sha, **status, 'regular_members': len(hashes),
               'payload_bytes': sum(sizes.values()), 'seconds': time.monotonic() - started,
               'committed_training_updates':status['completed_committed_training_updates'],
               'actual_step_count_source': manifest['actual_step_count_source'],
               'model_qualified': False, 'training_success_not_implied': True}
    _write_reserved_json(receipt_path, receipt)
    return receipt

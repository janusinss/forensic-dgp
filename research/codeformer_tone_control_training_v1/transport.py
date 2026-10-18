"""Standard-library transfer checks and bounded exclusive composite tone-control export.

No returned code, tensor file, model, optimizer or neural framework is loaded.
Parent files remain read-only; only new composite state contains serialized frozen-base tensors.
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


def verify_transfer(root, protocol_or_sha, protocol_sha=None):
    """Hash exact source and both read-only parent maps without framework imports.

    Accept(root,sha)for CLI or(root,p,sha)for a caller's already parsed protocol.
    The latter must equal the independently read, hash-bound protocol bytes.
    """
    root=_no_links(root)
    supplied=protocol_or_sha if protocol_sha is not None else None
    protocol_sha=protocol_or_sha if protocol_sha is None else protocol_sha
    c.require(isinstance(protocol_sha,str) and HEX.fullmatch(protocol_sha),'Exact protocol SHA required')
    path=_regular(root,'protocol.json')
    c.require(path.stat().st_size<=8*1024**2 and c.sha256(path)==protocol_sha,'Frozen protocol hash/size differs')
    p=c.read_json(path);c.verify_protocol(p)
    c.require(supplied is None or c.equal(supplied,p),'Caller protocol differs from hash-bound bytes')
    files,sizes=p['files_sha256'],p['files_bytes'];checked={};folded=set()
    for name,digest in sorted(files.items()):
        canonical(name)
        c.require(name.casefold() not in folded and not name.startswith('outputs/') and name!=c.LOCK_NAME and
                  Path(name).suffix.lower() in {'.py','.json','.md','.txt','.sha256'},'Invalid new source-only namespace')
        folded.add(name.casefold());source=_regular(root,name)
        c.require(source.stat().st_size==sizes[name] and c.sha256(source)==digest,'Static source binding differs: '+name)
        checked[name]={'sha256':digest,'bytes':sizes[name]}
    c.require(len(files)+1<=64 and sum(sizes.values())+path.stat().st_size<=16*1024**2,
              'Source-only packet exceeds64files/16MiB')
    inventory=_walk(root)
    own={n for n in inventory if not n.startswith('outputs/') and n!=c.LOCK_NAME}
    c.require(own==set(files)|{'protocol.json'},'Unbound static/importable packet namespace')
    dependencies=p['dependencies'];retained=p['retained_output_dependencies']
    _binding_map(dependencies,'AdaIN parent');_binding_map(retained,'retained original/base outputs')
    c.require(not set(dependencies)&set(retained),'Parent maps overlap')
    parent=c.resolve_parent(root);oldpath=_regular(parent,'protocol.json',unique_link=False)
    c.require(c.sha256(oldpath)==p['parent_protocol_sha256'],'Exact parent protocol differs')
    old=c.read_json(oldpath)
    c.require(c.equal(old,p['historical_parent_protocol']),'Embedded parent provenance differs from actual parent')
    for label,mapping in [('dependencies',dependencies),('retained_output_dependencies',retained)]:
        for name,binding in sorted(mapping.items()):
            source=_regular(parent,name,unique_link=False);before=source.stat()
            c.require(before.st_size==binding['bytes'] and c.sha256(source)==binding['sha256'],
                      'Read-only parent hash/size differs: '+label+'/'+name)
            after=source.stat()
            c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
                      (after.st_size,after.st_mtime_ns,after.st_ino),'Parent changed while hashing')
    c.require(all(dependencies.get(n,{}).get('sha256')==v for n,v in old['files_sha256'].items()),
              'Entire frozen parent immutable namespace required')
    for label,asset in p['assets'].items():
        c.require(dependencies.get(asset['path'])=={k:asset[k] for k in ('sha256','bytes')},
                  'Named original/base asset binding differs: '+label)
    for key in c.PARENT_CONTROLS:
        c.require(c.equal(p[key],old[key]),'Parent data control join differs: '+key)
    cases=expected_case_ids(p)
    return {'format':'codeformer-tone-control-training-transfer-v1','complete':True,
        'protocol_sha256':protocol_sha,'protocol':p,'static_files':checked,
        'parent_root':p['parent_dir'],'parent_protocol_sha256':p['parent_protocol_sha256'],
        'dependencies':dependencies,'retained_output_dependencies':retained,
        'dependencies_verified':len(dependencies),'retained_output_dependencies_verified':len(retained),
        'total_readonly_bindings_verified':len(dependencies)+len(retained),'DEV_cases_verified':len(cases),
        'readonly':True,'neural_calls':0,'gradient_queries':0,'optimizer_steps':0}


def expected_case_ids(protocol):
    """The frozen1624two-bank RAW/PNG map defines exactly406 canonical cases."""
    roots=['outputs/adain_gradient_isolation_v1/snapshot_'+value+'/' for value in ('0000','0050')]
    ids=[]
    for root in roots:
        current=set()
        for name in protocol['retained_output_dependencies']:
            if name.startswith(root):
                suffix=name[len(root):];parts=suffix.split('/')
                c.require(len(parts)==2 and re.fullmatch(r'[A-Za-z0-9_.-]+',parts[0]) and
                          parts[1] in ('raw.npy','native.png'),'Unexpected retained case file')
                current.add(parts[0])
        c.require(len(current)==406,'Retained bank requires406exact case IDs')
        c.require({root+case+'/'+filename for case in current for filename in ('raw.npy','native.png')}<=
                  set(protocol['retained_output_dependencies']),'Retained bank RAW/PNG pair missing')
        ids.append(current)
    c.require(ids[0]==ids[1] and len(protocol['retained_output_dependencies'])==1624,
              'Retained original/base case namespace differs')
    return sorted(ids[0])


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


def checkpoint_kind(name):
    if name in c.CHECKPOINT_PATHS:return 'final'
    for base in c.CHECKPOINT_PATHS:
        if re.fullmatch(re.escape(base)+r'\.tmp-[0-9a-f]{32}',name):return 'temporary'
    return None


def _measure_files(root,files):
    hashes,sizes,metadata,temporary={}, {},0,0
    for name in sorted(files):
        path=_regular(root,name);size=path.stat().st_size;kind=checkpoint_kind(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(v+'.tmp-' in name.lower() for v in FORBIDDEN_SUFFIXES):
            c.require(kind is not None,'Composite checkpoint outside exact filename gates forbidden')
        if kind is not None:
            cap=c.DESIGN['state_and_return']['checkpoint_write_cap_bytes_each']
            c.require(size<=cap,
                      'Composite checkpoint file cap exceeded')
            if kind=='temporary':temporary+=size
        elif path.suffix.lower() in {'.npy','.npz'}:
            c.require(size<=c.LIMITS['one_array_bytes'],'Array file cap exceeded')
        elif path.suffix.lower()=='.png':
            c.require(size<=c.LIMITS['one256PNG_bytes'],'PNG256 file cap exceeded')
        else:metadata+=size
        hashes[name],sizes[name]=c.sha256(path),size
    c.require(temporary<=c.LIMITS['checkpoint_temporary_bytes'] and len(sizes)<=MEMBER_CAP and
              sum(sizes.values())<=PAYLOAD_CAP and metadata<=METADATA_CAP,
              'Result/composite/file/payload/metadata cap exceeded; no silent omission')
    budget.checkpoint_slot_accounting(sizes)
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


def output_coverage(files,protocol):
    cases=expected_case_ids(protocol);banks={};missing=[]
    for bank in ('original0','head0000','head0050'):
        filenames=('raw.npy','native.png','evidence.json')+(() if bank=='original0' else ('base_raw.npy',))
        expected={OUTPUTS+'/'+bank+'/'+case+'/'+name for case in cases for name in filenames}
        present=expected&set(files);absent=sorted(expected-set(files));missing.extend(absent)
        bank_prefix=OUTPUTS+'/'+bank+'/'
        actual={name for name in files if name.startswith(bank_prefix)}
        c.require(actual<=expected,'Unrecognized bank case/member namespace')
        complete=sum(all(OUTPUTS+'/'+bank+'/'+case+'/'+n in files for n in filenames) for case in cases)
        banks[bank]={'expected_cases':406,'complete_case_sets':complete,'expected_members':len(expected),
                     'present_members':len(present),'missing_members':absent}
    return {'complete':not missing,'banks':banks,'missing_members':missing,
        'RAW_PNG_pairs':sum(OUTPUTS+'/'+bank+'/'+case+'/raw.npy' in files and
                            OUTPUTS+'/'+bank+'/'+case+'/native.png' in files
                            for bank in banks for case in cases),
        'base_RAW_arrays':sum(OUTPUTS+'/'+bank+'/'+case+'/base_raw.npy' in files
                             for bank in ('head0000','head0050') for case in cases),
        'case_evidence_records':sum(OUTPUTS+'/'+bank+'/'+case+'/evidence.json' in files
                                  for bank in banks for case in cases),
        'scope':'File/namespace completeness only; independent pixel/mask/quality readback required'}


def _terminal_evidence(root,files,protocol_sha,protocol):
    success=OUTPUTS+'/run_results.json';failure=OUTPUTS+'/failure.json';supervision='outputs/supervision/run_receipt.json'
    c.require(supervision in files,'Closed terminal supervisor receipt required')
    supervised=c.read_json(_regular(root,supervision))
    c.require(supervised.get('worker_terminal') is True and supervised.get('protocol_sha256')==protocol_sha,
              'Terminal supervisor protocol binding differs')
    c.require(not(success in files and failure in files),'Conflicting worker terminal records')
    terminal=success if success in files else failure if failure in files else None
    value={} if terminal is None else c.read_json(_regular(root,terminal))
    c.require(terminal is None or value.get('protocol_sha256')==protocol_sha,'Terminal worker protocol binding differs')
    counts=value.get('counts');completed=value.get('completed_head_updates');inflight=value.get('head_optimizer_inflight')
    partial=value.get('partial_microbatches');restorer=value.get('restorer_updates_this_run')
    if terminal is not None:
        c.require(type(completed) is int and 0<=completed<=50 and type(partial) is int and 0<=partial<=4 and
                  type(inflight) is bool and type(restorer) is int and restorer==0,
                  'Actual head/restorer/partial/inflight terminal state differs')
        c.require(isinstance(counts,dict) and set(counts)==set(c.COUNTS),'Exact worker operation count keys required')
        for key,cap in c.COUNTS.items():
            c.require(type(counts[key]) is int and 0<=counts[key]<=cap,'Operation cap exceeded: '+key)
        c.require(completed<=counts['head_optimizer_step_attempts'],
                  'Head committed count exceeds attempted steps')
    coverage=output_coverage(files,protocol)
    if terminal==success:
        c.require(value.get('complete') is True and completed==50 and partial==0 and not inflight and
                  all(counts[k]==cap for k,cap in c.COUNTS.items()),'Successful head training census incomplete')
        c.require(coverage['complete'] and coverage['RAW_PNG_pairs']==1218 and coverage['base_RAW_arrays']==812,
                  'Successful training must retain all1218RAWPNG/812baseRAW case outputs')
        c.require(all(name in files for name in c.CHECKPOINT_PATHS),'All3genuine composite checkpoint slots required')
    return {'complete_training':terminal==success,'failure_present':failure in files,
        'terminal_evidence':[n for n in (success,failure,supervision) if n in files],
        'worker_recorded_counts':counts,'completed_head_updates':completed,
        'head_optimizer_inflight_at_failure':inflight,'partial_microbatches':partial,
        'restorer_updates_this_run':restorer,'head_updates_unknown_without_worker_receipt':terminal is None,
        'candidate_checkpoint_members':sorted(n for n in files if checkpoint_kind(n)=='final'),
        'incomplete_checkpoint_temporary_members':sorted(n for n in files if checkpoint_kind(n)=='temporary'),
        'DEV_output_coverage':coverage}


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
    status = _terminal_evidence(root, inventory, protocol_sha,p)
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
    excluded = {c.LOCK_NAME, 'outputs/supervision/export.log', 'outputs/supervision/finished.json'}
    names = set(inventory) - excluded
    hashes, sizes, metadata = _measure_files(root, names)
    manifest = {'format': 'codeformer-tone-control-return-v1', 'protocol_sha256': protocol_sha,
                'files_sha256': hashes, 'files_bytes': sizes, 'parent_dir':p['parent_dir'],'parent_protocol_sha256':p['parent_protocol_sha256'],'dependencies':p['dependencies'],'retained_output_dependencies':p['retained_output_dependencies'],
                **status, 'committed_head_updates':status['completed_head_updates'],
                'actual_step_count_source':'worker completion/failure receipt; unknown is never inferred zero',
                'model_qualified': False, 'automatic_training': False, 'native_CCTV_qualification': False,
                'parent_files_modified_or_copied':False,'frozen_base_tensors_serialized_inside_composite_checkpoints':True,
                'closure_boundary': {'immutable_pre_export_supervision_included': True,
                    'separately_retained': ['outputs/supervision/export.log', 'outputs/supervision/finished.json',
                                           receipt_path.name, checksum.name, PREFIX+'-supervisor-final.json'],
                    'external_final_companion': PREFIX+'-supervisor-final.json',
                    'reason': 'Supervisor final companion binds exact export log and parsed finished receipt/hash after archive closure; no self-hash'},
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
    # The prospective byte ledger assumes exactly one USTAR regular header per
    # file. Refuse unsupported names before creating a HOME archive; no PAX or
    # GNU extension headers can silently invalidate that resource premise.
    for name, size in sizes.items():
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = size, 0o644, 0
        info.tobuf(format=tarfile.USTAR_FORMAT, encoding='utf-8', errors='strict')
    with archive.open('xb') as destination:
        bounded = BoundedArchive(destination, started + EXPORT_SECONDS, archive.parent)
        with gzip.GzipFile(filename='', mode='wb', fileobj=bounded, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as bundle:
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
    receipt = {'format': 'codeformer-tone-control-export-v1', 'complete': True,
               'archive_sha256': archive_sha, 'bytes': archive.stat().st_size,
               'manifest_member': manifest_name, 'manifest_sha256': hashes[manifest_name],
               'protocol_sha256': protocol_sha, **status, 'regular_members': len(hashes),
               'payload_bytes': sum(sizes.values()), 'seconds': time.monotonic() - started,
               'committed_head_updates':status['completed_head_updates'],
               'actual_step_count_source': manifest['actual_step_count_source'],
               'model_qualified': False, 'training_success_not_implied': True}
    _write_reserved_json(receipt_path, receipt)
    return receipt

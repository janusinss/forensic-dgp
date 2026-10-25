"""Inert bounded transfer and terminal return for the new skip adapter pilot.

Importing this module never loads a neural framework, tensors or returned code.
The protocol freezes the whole static tree, exact output namespace and finite
operation budget before the human invokes the separate Linux tmux supervisor.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
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
import zlib

VM_NAME = 'cctv_codeformer_skip_feature_adapter_v1_vm'
MODULE = 'research.codeformer_skip_feature_adapter_v1.worker'
PREFIX = 'cctv-codeformer-skip-feature-adapter-v1'
OUTPUTS = 'outputs/skip_feature_adapter_v1'
LOCK_NAME = '.skip_feature_adapter_v1.lock'
GiB, MiB = 1024**3, 1024**2
WEIGHT_ASSETS = {
    'weights/codeformer_original.pth': {'bytes': 376637898,
        'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'},
    'weights/codeformer_r2.pth': {'bytes': 376609465,
        'sha256': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'},
    'weights/codeformer_parent.pth': {'bytes': 376643953,
        'sha256': '53778f54589697a14b067033eff8cffc2985985e8a33021092de3c14fbf443bc'}}
WEIGHT_ASSET_BYTES = sum(row['bytes'] for row in WEIGHT_ASSETS.values())
LIMITS = {
    'payload_bytes': 9*GiB, 'archive_bytes': 37*GiB//4,
    'metadata_bytes': 128*MiB, 'regular_files': 20000,
    'initial_free_bytes': 24*GiB, 'reserve_free_bytes': 4*GiB,
    'initial_checkpoint_bytes': 384*MiB, 'trained_checkpoint_bytes': 384*MiB,
    'checkpoint_temporary_bytes': 384*MiB,
    'one_json_bytes': 4*MiB, 'png256_bytes': MiB,
    'array_bytes': 40*MiB, 'source_packet_files': 4099,
    'source_packet_bytes': 512*MiB+WEIGHT_ASSET_BYTES,
    'nonweight_static_files': 4096, 'nonweight_static_bytes': 512*MiB,
    'export_seconds': 600,
}
CALL_CAPS = {
    'original_baseline_forwards': 478, 'parent_baseline_forwards': 478,
    'R2_baseline_forwards': 478, 'R2_clean_feature_teacher_forwards': 55,
    'original_clear_teacher_forwards': 55, 'R2_dark_teacher_forwards': 55,
    'candidate_snapshot_forwards': 1434, 'candidate_training_forwards': 440,
    'candidate_TRAIN_guard_forwards': 96, 'candidate_witness_forwards': 3,
    'R2_auxiliary_forwards': 18, 'candidate_auxiliary_forwards': 54,
    'backward_calls': 440, 'optimizer_step_attempts': 110, 'gradient_queries': 4}
CHECKPOINT_PATHS = tuple(OUTPUTS+'/'+name for name in (
    'snapshot_0000/checkpoint.pt', 'snapshot_0055/checkpoint.pt',
    'snapshot_0110/checkpoint.pt', 'stopped_checkpoint.pt'))
SUPERVISION_NAMES = tuple('outputs/supervision/'+name for name in (
    'preflight.log', 'worker.log', 'run_receipt.json', 'export_invocation.json',
    'export.log', 'finished.json'))
HEX = re.compile(r'^[0-9a-f]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)', re.I)
FORBIDDEN = re.compile(r'\.(?:pt|pth|ckpt|safetensors|pkl|pickle|bin)(?:$|[.\-])', re.I)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(name):
    require(isinstance(name, str) and name, 'Missing portable relative path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and path.parts and
            all(part not in ('', '.', '..') for part in path.parts), 'Noncanonical relative path')
    require(unicodedata.normalize('NFC', name) == name and
            not any(char in name for char in '\\:"<>|?*') and
            all(not DEVICE.match(part) and not part.endswith((' ', '.')) and
                len(part.encode('utf-16-le')) <= 510 and all(ord(ch) >= 32 for ch in part)
                for part in path.parts), 'Nonportable relative path')
    return path


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            row = item.lstat()
            require(not stat.S_ISLNK(row.st_mode) and
                    not getattr(row, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse path forbidden')
    return path


def regular(root, name):
    canonical(name)
    root = no_links(root)
    path = no_links(root.joinpath(*PurePosixPath(name).parts))
    require(path.is_relative_to(root), 'Path escapes exact root')
    row = path.lstat()
    require(stat.S_ISREG(row.st_mode) and row.st_nlink == 1, 'Unique regular file required')
    return path


def inventory(root):
    root = no_links(root)
    require(root.is_dir(), 'Exact existing root required')
    files, names = {}, set()
    for directory, folders, leaves in os.walk(root, followlinks=False):
        for name in sorted(folders+leaves):
            path = Path(directory)/name
            rel = path.relative_to(root).as_posix()
            canonical(rel)
            require(rel.casefold() not in names, 'Portable namespace collision')
            names.add(rel.casefold()); no_links(path)
            row = path.lstat()
            if name in folders:
                require(stat.S_ISDIR(row.st_mode), 'Non-directory namespace')
            else:
                require(stat.S_ISREG(row.st_mode) and row.st_nlink == 1,
                        'Linked/nonregular payload forbidden')
                files[rel] = path
    return files


def file_pin(path, check=None):
    path = no_links(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'Unique regular file required while hashing')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*MiB), b''):
            if check is not None:
                check()
            digest.update(block)
    after = path.lstat()
    require(stat.S_ISREG(after.st_mode) and after.st_nlink == 1 and
            (before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed while hashing')
    return {'sha256': digest.hexdigest(), 'bytes': before.st_size}


def json_bytes(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')


def parse_json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(payload.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read_json(path):
    path = no_links(path)
    row = path.lstat()
    require(stat.S_ISREG(row.st_mode) and row.st_nlink == 1 and
            row.st_size <= LIMITS['one_json_bytes'], 'Bounded unique regular JSON required')
    return parse_json(path.read_bytes())


def write_new_json(path, value, reserve=True):
    path = no_links(path); payload = json_bytes(value)
    require(len(payload) <= LIMITS['one_json_bytes'], 'Individual JSON cap exceeded')
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if reserve:
        require(shutil.disk_usage(ancestor).free >= LIMITS['reserve_free_bytes']+len(payload)+65536,
                'Receipt write reserve unavailable')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        require(stream.write(payload) == len(payload), 'Short receipt write')
        stream.flush(); os.fsync(stream.fileno())


def validate_binding_map(mapping, label='bindings'):
    require(type(mapping) is dict, 'Missing '+label)
    seen = set()
    for name, row in mapping.items():
        canonical(name)
        require(name.casefold() not in seen, 'Colliding '+label); seen.add(name.casefold())
        require(type(row) is dict and set(row) == {'sha256', 'bytes'} and
                isinstance(row['sha256'], str) and HEX.fullmatch(row['sha256']) and
                type(row['bytes']) is int and row['bytes'] >= 0, 'Invalid '+label+' record')


def operation_ledger():
    forwards = {key: value for key, value in CALL_CAPS.items() if key.endswith('_forwards')}
    require(sum(forwards.values()) == 3644, 'Exact finite forward census differs')
    return {'complete': True, 'prospective_only': True, 'by_counter': dict(CALL_CAPS),
        'top_level_forwards': 3644, 'backward_calls': 440, 'optimizer_attempts': 110,
        'gradient_queries': 4, 'reservation_events': sum(CALL_CAPS.values()),
        'old406_cases_each_model': 406, 'mixed72_cases_each_model': 72,
        'native_unpaired_per_model': 48, 'auxiliary_TRAIN_cases': 18,
        'training_completion_not_model_acceptance': True}


def resource_ledger(static_files=None, static_bytes=None, *, dependency_files=0, dependency_bytes=0):
    """Source-derived finite envelope, never a measured resource/quality claim.

    PNG bound is for the producer's RGB8 256x256 Pillow output (one zlib stream,
    no optional metadata), distinct from the looser per-file transport ceiling.
    """
    count = LIMITS['nonweight_static_files'] if static_files is None else static_files
    size = LIMITS['nonweight_static_bytes'] if static_bytes is None else static_bytes
    require(type(count) is int and 0 < count <= LIMITS['nonweight_static_files'] and
            type(size) is int and 0 <= size <= LIMITS['nonweight_static_bytes'] and
            dependency_files == dependency_bytes == 0, 'Self-contained finite static resource ledger required')
    rgb = 3*256*256*4+256; png = 198656
    features = sum(channels*side*side*4+256 for channels, side in ((256,32),(256,64),(128,128),(128,256)))
    categories = {
        'three_committed_states': {'files':3,'bytes':3*LIMITS['trained_checkpoint_bytes']},
        'one_retained_inflight_or_stopped_state': {'files':1,'bytes':LIMITS['checkpoint_temporary_bytes']},
        'all_snapshot_and_three_baseline_RAW_PNG': {'files':5736,'bytes':2868*(rgb+png)},
        'TRAIN_guard_RAW_PNG': {'files':192,'bytes':96*(rgb+png)},
        'auxiliary_RAW_PNG': {'files':144,'bytes':72*(rgb+png)},
        '55_clean_feature_teachers_four_sites': {'files':220,'bytes':55*features},
        '110_behavior_teachers_U_RAW_PNG': {'files':330,'bytes':110*(2*rgb+png)},
        'three_witness_U_RAW': {'files':6,'bytes':6*rgb},
        'all_metadata_logs': {'files':4500,'bytes':LIMITS['metadata_bytes']},
        'nonweight_static_packet': {'files':count,'bytes':size},
        'three_input_weights': {'files':3,'bytes':WEIGHT_ASSET_BYTES}}
    files = sum(row['files'] for row in categories.values())
    payload = sum(row['bytes'] for row in categories.values())
    tar = ((payload+files*1023+1024+10239)//10240)*10240
    compressed = tar+(tar>>12)+(tar>>14)+(tar>>25)+64
    simultaneous = LIMITS['payload_bytes']+LIMITS['archive_bytes']+LIMITS['reserve_free_bytes']
    require(files <= LIMITS['regular_files'] and payload <= LIMITS['payload_bytes'] and
            compressed <= LIMITS['archive_bytes'] and simultaneous < LIMITS['initial_free_bytes'],
            'Prospective whole return does not fit its finite disk admission')
    return {'complete':True,'prospective_only':True,'measured_peak_resources':False,
        'model515_FP32_bytes':376450828,'adapter24_FP32_bytes':51328*4,
        'Adam24_two_moments_bytes':51328*8,'Adam24_step_scalar_bytes':24*4,
        'categories':categories,'projected_files':files,'projected_payload_bytes':payload,
        'canonical_USTAR_upper_bytes':tar,'single_gzip_upper_bytes':compressed,
        'simultaneous_payload_archive_reserve_cap_bytes':simultaneous,
        'post_install_disk_margin_bytes':LIMITS['initial_free_bytes']-simultaneous,
        'source_feature_arrays_retained':220,'all440_training_native_arrays_retained':False,
        'training_scalar_hash_context_records_required':440,
        'selfcontained_input_weights_in_upload_and_return':True,
        'dependencies_copied_to_return':False,'model_qualified':False}


def checkpoint_kind(name):
    if name in CHECKPOINT_PATHS:
        return 'final'
    if any(re.fullmatch(re.escape(path)+r'\.tmp-[0-9a-f]{32}', name) for path in CHECKPOINT_PATHS):
        return 'temporary'
    return None


def checkpoint_accounting(sizes):
    slots = {name: [] for name in CHECKPOINT_PATHS}
    for name, size in sizes.items():
        require(type(size) is int and size >= 0, 'Invalid checkpoint size')
        kind = checkpoint_kind(name)
        if kind:
            base = name if kind == 'final' else next(path for path in slots if name.startswith(path+'.tmp-'))
            slots[base].append((name, size))
    for name, rows in slots.items():
        cap = LIMITS['initial_checkpoint_bytes'] if name == CHECKPOINT_PATHS[0] else LIMITS['trained_checkpoint_bytes']
        require(len(rows) <= 1 and sum(size for _, size in rows) <= cap,
                'One bounded final OR retained temporary per checkpoint slot required')
    require(not (slots[CHECKPOINT_PATHS[2]] and slots[CHECKPOINT_PATHS[3]] and
            all(checkpoint_kind(rows[0][0]) == 'final' for rows in
                (slots[CHECKPOINT_PATHS[2]], slots[CHECKPOINT_PATHS[3]]))),
            'Final110 and stopped state cannot both duplicate the same final state')
    temporary = sum(size for rows in slots.values() for name, size in rows if checkpoint_kind(name) == 'temporary')
    require(temporary <= LIMITS['checkpoint_temporary_bytes'], 'Aggregate retained checkpoint temporary cap exceeded')
    return {'slots':{name:[value for value,_ in rows] for name,rows in slots.items()},
        'temporary_bytes':temporary,'bytes':sum(size for rows in slots.values() for _,size in rows)}


def allowed_output_names(protocol):
    names = protocol.get('allowed_output_names')
    require(type(names) is list and len(names) <= LIMITS['regular_files'] and
            len(set(names)) == len(names), 'Frozen finite exact output names required')
    folded = set()
    for name in names:
        canonical(name)
        require(name.startswith(OUTPUTS+'/') or name in SUPERVISION_NAMES, 'Output allowance escapes new experiment')
        require(name.casefold() not in folded, 'Colliding output namespace'); folded.add(name.casefold())
        require(not FORBIDDEN.search(PurePosixPath(name).name) or checkpoint_kind(name) == 'final',
                'Arbitrary serialized model output forbidden')
    require(set(SUPERVISION_NAMES).issubset(names), 'Exact six finite supervision outputs required')
    return set(names)


def _bindings_from_protocol(protocol):
    require(type(protocol) is dict and
            protocol.get('format') == 'codeformer-skip-feature-adapter-finite-protocol-v1' and
            protocol.get('vm_name') == VM_NAME and protocol.get('outputs') == OUTPUTS,
            'Exact new protocol namespace required')
    source = protocol.get('files'); validate_binding_map(source, 'static source bindings')
    require('protocol.json' not in source and not any(name.startswith('outputs/') or name == LOCK_NAME for name in source),
            'Static packet cannot contain outputs, self-binding or runtime lock')
    require(protocol.get('weight_assets') == WEIGHT_ASSETS and all(source.get(name) == pin for name,pin in WEIGHT_ASSETS.items()) and
            not any(name.startswith('weights/') and name not in WEIGHT_ASSETS for name in source),
            'Exact three separately budgeted static weight assets required')
    require(protocol.get('dependencies') == {}, 'New packet must be self-contained with no mutable dependency tree')
    require(protocol.get('limits') == LIMITS and protocol.get('call_caps') == CALL_CAPS and
            protocol.get('max_updates') == 110 and protocol.get('snapshots') == [0,55,110] and
            protocol.get('guard_snapshots') == [0,20,55,110] and
            protocol.get('automatic_resume') is False and protocol.get('automatic_training_launch') is False,
            'Frozen finite caps/schedule/manual execution policy differ')
    allowed_output_names(protocol)
    required = protocol.get('required_success_output_names')
    require(type(required) is list and len(required) == len(set(required)) and
            set(required).issubset(allowed_output_names(protocol)) and all(name in required for name in CHECKPOINT_PATHS[:3]),
            'Frozen exact success coverage required')
    return source, {}


def verify_transfer(root, protocol_sha, *, dependency_base=None):
    """Hash every source/input before any framework import; no runtime activation."""
    root = no_links(root)
    require(isinstance(protocol_sha,str) and HEX.fullmatch(protocol_sha), 'Exact protocol SHA required')
    path = regular(root,'protocol.json')
    require(file_pin(path)['sha256'] == protocol_sha, 'Protocol hash differs')
    protocol = read_json(path); source, dependencies = _bindings_from_protocol(protocol)
    current = inventory(root)
    require({name for name in current if not name.startswith('outputs/') and name != LOCK_NAME} == set(source)|{'protocol.json'},
            'Unbound static/importable packet namespace')
    allowed = allowed_output_names(protocol)
    require(all(not name.startswith('outputs/') or name in allowed or checkpoint_kind(name) == 'temporary' for name in current),
            'Unexpected output namespace')
    for name,pin in sorted(source.items()):
        require(file_pin(regular(root,name)) == pin, 'Static packet bytes differ: '+name)
    nonweights = {name:pin for name,pin in source.items() if name not in WEIGHT_ASSETS}
    nonweight_bytes = sum(pin['bytes'] for pin in nonweights.values())+path.stat().st_size
    require(len(nonweights)+1 <= LIMITS['nonweight_static_files'] and nonweight_bytes <= LIMITS['nonweight_static_bytes'] and
            len(source)+1 <= LIMITS['source_packet_files'] and
            sum(pin['bytes'] for pin in source.values())+path.stat().st_size <= LIMITS['source_packet_bytes'],
            'Protocol-inclusive source packet envelope exceeded')
    checkpoint_accounting({name:item.stat().st_size for name,item in current.items() if checkpoint_kind(name)})
    return {'complete':True,'protocol_sha256':protocol_sha,'protocol':protocol,'static_files':len(source),
        'dependencies_verified':len(dependencies),'readonly_dependency_bytes':0,
        'neural_calls':0,'gradient_queries':0,'optimizer_updates':0}


def require_worker_terminal(root):
    lock = no_links(root)/LOCK_NAME
    if lock.exists() or lock.is_symlink():
        value = read_json(regular(root,LOCK_NAME)); pid = value.get('pid')
        require(type(pid) is int and pid > 0, 'Invalid retained worker lock PID')
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            pass
        except PermissionError:
            raise ValueError('Worker PID cannot be proved terminal')
        else:
            raise ValueError('Worker remains alive; export forbidden')
    result = subprocess.run(['ps','-eo','pid=,args='],check=True,capture_output=True,text=True,timeout=20)
    for line in result.stdout.splitlines():
        fields = line.strip().split(None,1)
        require(len(fields) == 2 and fields[0].isdigit(), 'Unclassifiable process inventory')
        if int(fields[0]) == os.getpid():
            continue
        try:
            words = shlex.split(fields[1])
        except ValueError:
            require(MODULE not in fields[1], 'Unclassifiable potential worker process'); continue
        require(not (MODULE in words and '--run' in words), 'Worker/run wrapper still alive')


def terminal_evidence(root, files, protocol_sha, protocol):
    receipt = read_json(regular(root,'outputs/supervision/run_receipt.json'))
    require(receipt.get('worker_terminal') is True and receipt.get('child_process_group_terminal') is True and
            receipt.get('protocol_sha256') == protocol_sha, 'Closed supervisor terminal receipt required')
    stages = receipt.get('stages')
    require(type(stages) is list and all(type(row) is dict for row in stages),
            'Explicit supervisor stage records required')
    runs = [row for row in stages if row.get('stage') == '--run']
    require(len(runs) == 1, 'Exactly one terminal worker attempt required; no automatic retry')
    run = runs[0]
    require(run.get('worker_terminal') is True and run.get('child_process_group_terminal') is True and
            type(run.get('child_pid')) is int and run['child_pid'] > 0 and type(run.get('exit_code')) is int and
            type(run.get('outer_watchdog_expired')) is bool and type(run.get('log_capped')) is bool and
            (run.get('supervisor_exception') is None or type(run['supervisor_exception']) is dict),
            'Authoritative worker exit/process-group/timeout evidence required')
    clean_exit = (run['exit_code'] == 0 and not run['outer_watchdog_expired'] and
                  not run['log_capped'] and run.get('supervisor_exception') is None)
    success, failure = OUTPUTS+'/run_results.json', OUTPUTS+'/failure.json'
    require(not (success in files and failure in files), 'Conflicting worker terminal receipts')
    terminal = success if success in files else failure if failure in files else None
    malformed = None; value = {}
    if terminal is not None:
        path = regular(root,terminal)
        try:
            value = read_json(path)
        except (ValueError,UnicodeError) as error:
            malformed = {'path':terminal,**file_pin(path),'error_type':type(error).__name__,'error':str(error)[:4096]}
            terminal = None
    completed, partial, inflight = (value.get(key) for key in ('completed_training_updates','partial_microbatches','optimizer_inflight'))
    counts, returned = value.get('counts'), value.get('actual_return_counts')
    if terminal:
        require(value.get('complete') is True and value.get('protocol_sha256') == protocol_sha and
                type(counts) is dict and set(counts) == set(CALL_CAPS) and type(returned) is dict and set(returned) == set(CALL_CAPS),
                'Exact worker reserved/returned counter schema differs')
        require(all(type(counts[key]) is int and 0 <= counts[key] <= CALL_CAPS[key] and
                type(returned[key]) is int and 0 <= returned[key] <= counts[key] for key in CALL_CAPS) and
                type(completed) is int and 0 <= completed <= 110 and type(partial) is int and 0 <= partial <= 4 and type(inflight) is bool,
                'Actual finite operation/commit counters differ')
        require(completed <= returned['optimizer_step_attempts'] and completed*4+partial <= returned['backward_calls'],
                'Committed state contradicts returned operation ledger')
    if terminal == success:
        require(completed == 110 and partial == 0 and inflight is False and counts == returned == CALL_CAPS and
                set(protocol['required_success_output_names']).issubset(files), 'Success requires all states and exact finite completed calls')
    if terminal == failure:
        require(value.get('failure') is True, 'Explicit failed/stopped status required')
    state, state_complete = value.get('state_checkpoint'), value.get('state_checkpoint_complete')
    if state is not None:
        require(type(state) is dict and set(state) == {'path','sha256','bytes'} and state['path'] in files and checkpoint_kind(state['path']) and
                file_pin(regular(root,state['path'])) == {key:state[key] for key in ('sha256','bytes')} and type(state_complete) is bool,
                'Terminal state must bind exact retained physical checkpoint bytes')
        if state_complete:
            require(checkpoint_kind(state['path']) == 'final', 'Temporary checkpoint is not a completed state')
        if completed == 110 and partial == 0 and inflight is False and state_complete:
            require(state['path'] in CHECKPOINT_PATHS[2:], 'Final110 state cannot bind an earlier snapshot')
    if terminal == success:
        require(state is not None and state_complete is True and state['path'] == CHECKPOINT_PATHS[2], 'Successful finite run requires final110 checkpoint')
    return {'worker_terminal_evidence':terminal,'failure_present':failure in files,
        'complete_training_process':terminal == success and clean_exit,
        'worker_run_exit_code':run['exit_code'],'worker_run_clean_exit':clean_exit,
        'worker_run_outer_watchdog_expired':run['outer_watchdog_expired'],
        'worker_run_log_capped':run['log_capped'],'worker_run_supervisor_exception':run.get('supervisor_exception'),
        'completed_training_updates':completed,'partial_microbatches':partial,'optimizer_inflight':inflight,
        'worker_recorded_counts':counts,'worker_actual_return_counts':returned,
        'activity_unknown_without_worker_receipt':terminal is None,'malformed_worker_terminal_preserved':malformed,
        'checkpoint_members':sorted(name for name in files if checkpoint_kind(name)),
        'state_checkpoint':state,'state_checkpoint_complete':state_complete,
        'model_qualified':False,'training_or_export_completion_not_model_acceptance':True}


@contextmanager
def finite_export():
    if not hasattr(signal,'setitimer'):
        yield; return
    started = time.monotonic()
    handler, timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    def expired(signum,frame):
        raise TimeoutError('Finite export600second wall limit exceeded')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,min(LIMITS['export_seconds'],timer[0]) if timer[0] else LIMITS['export_seconds'])
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,handler)
        signal.setitimer(signal.ITIMER_REAL,max(.000001,timer[0]-(time.monotonic()-started)) if timer[0] else 0,timer[1])


class _BoundedWriter:
    def __init__(self,stream,deadline,reserve_path):
        self.stream,self.deadline,self.reserve_path,self.bytes = stream,deadline,reserve_path,0
    def check(self,amount=0):
        require(time.monotonic() <= self.deadline and self.bytes+amount <= LIMITS['archive_bytes'], 'Finite export time/archive cap exceeded')
        require(shutil.disk_usage(self.reserve_path).free >= LIMITS['reserve_free_bytes']+amount, 'Archive would cross disk reserve; retain partial')
    def write(self,payload):
        self.check(len(payload)); written = self.stream.write(payload)
        require(written == len(payload),'Short archive write'); self.bytes += written; return written
    def flush(self):
        self.stream.flush()


class _HashReader:
    def __init__(self,stream,bounded):
        self.stream,self.bounded,self.digest,self.bytes = stream,bounded,hashlib.sha256(),0
    def read(self,size=-1):
        self.bounded.check(); value = self.stream.read(size); self.digest.update(value); self.bytes += len(value); return value


def measure_files(root,names):
    pins,metadata = {},0
    for name in sorted(names):
        path = regular(root,name); kind = checkpoint_kind(name); size = path.stat().st_size
        require(not FORBIDDEN.search(path.name) or kind or name in WEIGHT_ASSETS, 'Serialized model outside exact input/state slots')
        if not kind:
            if name in WEIGHT_ASSETS:
                require(file_pin(path) == WEIGHT_ASSETS[name], 'Static input-weight bytes changed')
                pins[name] = WEIGHT_ASSETS[name]; continue
            if path.suffix.lower() in ('.npy','.npz'):
                require(size <= LIMITS['array_bytes'], 'Individual retained array cap exceeded')
            elif path.suffix.lower() == '.png':
                require(size <= LIMITS['png256_bytes'], 'Individual PNG transport cap exceeded')
            else:
                metadata += size
                if path.suffix.lower() == '.json':
                    require(size <= LIMITS['one_json_bytes'], 'Individual JSON cap exceeded')
        pins[name] = file_pin(path)
    checkpoint_accounting({name:pin['bytes'] for name,pin in pins.items()})
    require(len(pins) <= LIMITS['regular_files'] and sum(pin['bytes'] for pin in pins.values()) <= LIMITS['payload_bytes'] and
            metadata <= LIMITS['metadata_bytes'], 'Whole result/file/metadata cap exceeded; no evidence omission permitted')
    return pins,metadata


def export_results(root,protocol_sha):
    with finite_export():
        return _export_results(root,protocol_sha)


def _export_results(root,protocol_sha):
    root = no_links(root); expected = no_links(Path.home()/'forensic-dgp'/VM_NAME)
    require(os.name == 'posix' and root.resolve(strict=True) == expected.resolve(strict=True), 'Only exact manual Linux VM root may export')
    started = time.monotonic(); verified = verify_transfer(root,protocol_sha); protocol = verified['protocol']
    require_worker_terminal(root); files = inventory(root); status = terminal_evidence(root,files,protocol_sha,protocol)
    archive = no_links(Path.home()/(PREFIX+'-results.tar.gz'))
    checksum,companion = Path(str(archive)+'.sha256'),archive.with_name(PREFIX+'-export.json')
    require(not any(path.exists() or path.is_symlink() for path in (archive,checksum,companion)), 'Preserve prior/partial export; no automatic retry')
    member = OUTPUTS+'/export_manifest.json'
    require(member not in files,'Preserve previous/partial export manifest')
    excluded = {LOCK_NAME,'outputs/supervision/export.log','outputs/supervision/finished.json'}
    pins,metadata = measure_files(root,set(files)-excluded)
    value = {'format':'codeformer-skip-feature-adapter-return-v1','protocol_sha256':protocol_sha,'files':pins,**status,
        'dependencies':{},'dependency_assets_copied':False,'UTC':datetime.now(timezone.utc).isoformat(),
        'separately_retained':[checksum.name,companion.name,PREFIX+'-supervisor-final.json',
            'outputs/supervision/export.log','outputs/supervision/finished.json']}
    encoded = json_bytes(value)
    require(len(encoded) <= LIMITS['one_json_bytes'] and len(pins)+1 <= LIMITS['regular_files'] and
            sum(pin['bytes'] for pin in pins.values())+len(encoded) <= LIMITS['payload_bytes'] and
            metadata+len(encoded) <= LIMITS['metadata_bytes'], 'Complete manifest envelope exceeded')
    tar = ((sum(512+((pin['bytes']+511)//512)*512 for pin in pins.values())+
        512+((len(encoded)+511)//512)*512+1024+10239)//10240)*10240
    bound = tar+(tar>>12)+(tar>>14)+(tar>>25)+64
    require(bound <= LIMITS['archive_bytes'] and shutil.disk_usage(archive.parent).free >= bound+LIMITS['reserve_free_bytes'],
            'Complete archive and physical reserve unavailable')
    write_new_json(root/member,value); pins[member] = file_pin(root/member)
    with archive.open('xb') as target:
        bounded = _BoundedWriter(target,started+LIMITS['export_seconds'],archive.parent)
        with gzip.GzipFile(filename='',mode='wb',fileobj=bounded,mtime=0,compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.USTAR_FORMAT) as tarstream:
                for name,pin in sorted(pins.items()):
                    path = regular(root,name); info = tarfile.TarInfo(name)
                    info.size,info.mode,info.mtime = pin['bytes'],0o644,0
                    info.uid = info.gid = 0; info.uname = info.gname = ''
                    with path.open('rb') as source:
                        reader = _HashReader(source,bounded); tarstream.addfile(info,reader)
                        require(reader.bytes == pin['bytes'] and reader.digest.hexdigest() == pin['sha256'],'Source changed while exporting')
        target.flush(); os.fsync(target.fileno())
    pin = file_pin(archive)
    require(time.monotonic()-started <= LIMITS['export_seconds'], 'Full export wall cap exceeded')
    with checksum.open('xb') as stream:
        stream.write(checksum_bytes(archive.name,pin['sha256'])); stream.flush(); os.fsync(stream.fileno())
    receipt = {'format':'codeformer-skip-feature-adapter-export-v1','complete':True,'archive_sha256':pin['sha256'],'bytes':pin['bytes'],
        'manifest_member':member,'manifest_sha256':pins[member]['sha256'],'protocol_sha256':protocol_sha,
        'regular_members':len(pins),'payload_bytes':sum(pin['bytes'] for pin in pins.values()),'seconds':time.monotonic()-started,**status}
    write_new_json(companion,receipt); return receipt


def checksum_bytes(archive_name,digest):
    canonical(archive_name)
    require('/' not in archive_name and isinstance(digest,str) and HEX.fullmatch(digest), 'Exact portable basename and SHA required')
    return (digest+'  '+archive_name+'\n').encode('ascii')

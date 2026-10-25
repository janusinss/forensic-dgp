"""Inert, finite transfer/return primitives for the filter-fusion pilot.

No model, optimizer, array or returned Python is deserialized.  This module is
not a launcher.  All names and operation reservations must be frozen by the
prospective protocol before a human starts the separate tmux supervisor.
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

VM_NAME = 'cctv_codeformer_filter_fusion_training_v1_vm'
MODULE = 'research.codeformer_filter_fusion_training_v1.worker'
PREFIX = 'cctv-codeformer-filter-fusion-training-v1'
OUTPUTS = 'outputs/filter_fusion_training_v1'
LOCK_NAME = '.filter_fusion_training_v1.lock'
GiB, MiB = 1024**3, 1024**2
MODEL_FP32_BYTES = 376450828
FUSION_FP32_BYTES = 42638336
ADAM_MOMENT_BYTES = 85276672
ADAM_STEP_BYTES = 288
SCHEMA_SHA256 = '5a926fc23c58916374d4ad69bf68fc4066bc469c440d920a2046b09a2761a994'
WEIGHT_ASSETS = {
    'weights/codeformer_original.pth': {'bytes': 376637898,
        'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'},
    'weights/codeformer_r2.pth': {'bytes': 376609465,
        'sha256': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'},
    'weights/codeformer_parent.pth': {'bytes': 376643953,
        'sha256': '53778f54589697a14b067033eff8cffc2985985e8a33021092de3c14fbf443bc'}}
WEIGHT_ASSET_BYTES = 1129891316
LIMITS = {
    'payload_bytes': 7*GiB, 'archive_bytes': 29*GiB//4,
    'metadata_bytes': 128*MiB, 'regular_files': 16000,
    'initial_free_bytes': 20*GiB, 'reserve_free_bytes': 4*GiB,
    'initial_checkpoint_bytes': 384*MiB, 'trained_checkpoint_bytes': 448*MiB,
    'checkpoint_temporary_bytes': 448*MiB,
    'one_json_bytes': 4*MiB, 'png256_bytes': 198656,
    'array_bytes': 32*MiB, 'source_packet_files': 2051,
    'source_packet_bytes': 512*MiB+WEIGHT_ASSET_BYTES,
    'nonweight_static_files': 2048, 'nonweight_static_bytes': 512*MiB,
    'export_seconds': 600,
}
CHECKPOINT_PATHS = tuple(OUTPUTS+'/'+n for n in (
    'snapshot_0000/checkpoint.pt', 'snapshot_0050/checkpoint.pt',
    'snapshot_0151/checkpoint.pt', 'stopped_checkpoint.pt'))
HEX = re.compile(r'^[0-9a-f]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)', re.I)
FORBIDDEN = re.compile(r'\.(?:pt|pth|ckpt|safetensors|pkl|pickle|bin)(?:$|[.\-])', re.I)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(name):
    require(isinstance(name, str) and name, 'Missing portable relative path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name and p.parts and
            all(v not in ('', '.', '..') for v in p.parts), 'Noncanonical relative path')
    require(unicodedata.normalize('NFC', name) == name and
            not any(v in name for v in '\\:"<>|?*') and
            all(not DEVICE.match(v) and not v.endswith((' ', '.')) and
                len(v.encode('utf-16-le')) <= 510 and
                all(ord(ch) >= 32 for ch in v) for v in p.parts),
            'Nonportable relative path')
    return p


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            st = item.lstat()
            require(not stat.S_ISLNK(st.st_mode) and
                    not getattr(st, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse path forbidden')
    return path


def regular(root, name):
    canonical(name)
    root = no_links(root)
    path = no_links(root.joinpath(*PurePosixPath(name).parts))
    require(path.is_relative_to(root), 'Path escapes exact root')
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1,
            'Unlinked regular file required: '+name)
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
            names.add(rel.casefold())
            no_links(path)
            st = path.lstat()
            if name in folders:
                require(stat.S_ISDIR(st.st_mode), 'Non-directory namespace')
            else:
                require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1,
                        'Linked/nonregular result forbidden')
                files[rel] = path
    return files


def file_pin(path, check=None):
    path = no_links(path)
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1,
            'Unique regular file required while hashing')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*MiB), b''):
            if check is not None:
                check()
            digest.update(block)
    end = path.stat()
    require((st.st_size, st.st_mtime_ns, st.st_ino) ==
            (end.st_size, end.st_mtime_ns, end.st_ino), 'File changed while hashing')
    return {'sha256': digest.hexdigest(), 'bytes': st.st_size}


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
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read_json(path):
    path = no_links(path)
    require(path.stat().st_size <= LIMITS['one_json_bytes'], 'Individual JSON cap exceeded')
    return parse_json(path.read_bytes())


def write_new_json(path, value, reserve=True):
    path = no_links(path)
    payload = json_bytes(value)
    require(len(payload) <= LIMITS['one_json_bytes'], 'Individual JSON cap exceeded')
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if reserve:
        require(shutil.disk_usage(ancestor).free >= LIMITS['reserve_free_bytes']+len(payload)+65536,
                'Receipt write reserve unavailable')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def validate_binding_map(mapping, label='bindings'):
    require(type(mapping) is dict, 'Missing '+label)
    names = set()
    for name, row in mapping.items():
        canonical(name)
        require(name.casefold() not in names, 'Colliding '+label)
        names.add(name.casefold())
        require(type(row) is dict and set(row) == {'sha256', 'bytes'} and
                isinstance(row['sha256'], str) and HEX.fullmatch(row['sha256']) and
                type(row['bytes']) is int and row['bytes'] >= 0,
                'Invalid '+label+' record')


def operation_ledger(witness_caps, teacher_sources=55, gradient_queries=7):
    """Exact non-witness workload; every witness category must be named by source."""
    require(type(witness_caps) is dict and witness_caps and
            all(isinstance(k, str) and k.endswith('_forwards') and
                type(v) is int and v >= 0 for k, v in witness_caps.items()),
            'Explicit source-enumerated witness forward reservations required')
    require(sum(witness_caps.values()) == 9,
            'Exactly nine prerequisite forwards; no unnamed contingency or replay calls')
    require(type(teacher_sources) is int and 0 < teacher_sources <= 55,
            'Exact finite admitted clear/dark source census required')
    require(type(gradient_queries) is int and gradient_queries == 7,
            'Exact seven source-enumerated prerequisite gradient queries required')
    calls = {'candidate_snapshot_forwards': 3*478,
        'candidate_training_forwards': 151*4,
        'original_clear_teacher_forwards': teacher_sources,
        'R2_dark_teacher_forwards': teacher_sources,
        'R2_baseline_forwards': 478,
        'original_baseline_forwards': 478,
        'parent_baseline_forwards': 478,
        'candidate_TRAIN_guard_forwards': 2*24}
    require(not set(calls).intersection(witness_caps), 'Witness counter aliases ordinary calls')
    calls.update(witness_caps)
    return {'complete': True, 'prospective_only': True, 'by_counter': calls,
        'top_level_forwards': sum(calls.values()),
        'before_witness_forwards': 3520+2*teacher_sources,
        'backward_calls': 604, 'optimizer_attempts': 151,
        'gradient_queries': gradient_queries,
        'reservation_events': sum(calls.values())+604+151+gradient_queries,
        'baseline_policy': 'All original/parent/R2 controls freshly run on the actual CUDA runtime; retained CPU results are additional evidence only',
        'old406_cases_each_model': 406, 'mixed72_cases_each_model': 72,
        'mixed72_split': {'TRAIN': 36, 'DEV': 36},
        'native_unpaired_per_model': 48, 'refused_controls_no_forward': 24}


def resource_ledger(static_files=None, static_bytes=None, *, dependency_files=0,
                    dependency_bytes=0, teacher_arrays=110, witness_arrays=36):
    """Source-derived payload/serialization/physical-disk envelope; no measured peak claim.

    Includes all 3x478 candidate outputs, 3x478 current-runtime baselines, 48
    TRAIN guards, all110 unique behavior teacher U256/RAW/PNG triples, 3 physical committed
    states and one additional retained in-flight checkpoint.  It does not retain
    all604 full native512 training arrays; scalar/hash records are in metadata.
    The separately bound exact three input weight copies are included in upload
    and return; historical sources remain untouched. Optional dependencies are
    read-only existing assets, not copied into the return.
    """
    n = LIMITS['nonweight_static_files'] if static_files is None else static_files
    b = LIMITS['nonweight_static_bytes'] if static_bytes is None else static_bytes
    require(type(n) is int and 0 < n <= LIMITS['nonweight_static_files'] and
            type(b) is int and 0 <= b <= LIMITS['nonweight_static_bytes'] and
            type(dependency_files) is int and dependency_files >= 0 and
            type(dependency_bytes) is int and dependency_bytes >= 0 and
            type(teacher_arrays) is int and 0 <= teacher_arrays <= 110 and
            type(witness_arrays) is int and 0 <= witness_arrays <= 36,
            'Invalid finite resource ledger input')
    rgb256 = 3*256*256*4+256
    categories = {
        'planned_committed_checkpoints': {'files': 3, 'bytes': LIMITS['initial_checkpoint_bytes']+2*LIMITS['trained_checkpoint_bytes']},
        'retained_inflight_checkpoint': {'files': 1, 'bytes': LIMITS['checkpoint_temporary_bytes']},
        'candidate_snapshot_RAW': {'files': 1434, 'bytes': 1434*rgb256},
        'candidate_snapshot_PNG': {'files': 1434, 'bytes': 1434*LIMITS['png256_bytes']},
        'all3_current_runtime_baseline_RAW': {'files': 1434, 'bytes': 1434*rgb256},
        'all3_current_runtime_baseline_PNG': {'files': 1434, 'bytes': 1434*LIMITS['png256_bytes']},
        'TRAIN_guard_RAW_PNG': {'files': 96, 'bytes': 48*(rgb256+LIMITS['png256_bytes'])},
        'fresh_teacher_U256_arrays': {'files': teacher_arrays, 'bytes': teacher_arrays*rgb256},
        'fresh_teacher_delivered_RAW_PNG': {'files': 2*teacher_arrays,
            'bytes': teacher_arrays*(rgb256+LIMITS['png256_bytes'])},
        'witness_capture_arrays': {'files': witness_arrays, 'bytes': witness_arrays*rgb256},
        'all_metadata_logs': {'files': 4500, 'bytes': LIMITS['metadata_bytes']},
        'nonweight_static_packet': {'files': n, 'bytes': b},
        'exact3_input_weight_copies': {'files': 3, 'bytes': WEIGHT_ASSET_BYTES}}
    files = sum(v['files'] for v in categories.values())
    payload = sum(v['bytes'] for v in categories.values())
    tar_upper = ((payload+files*1023+1024+10239)//10240)*10240
    gzip_upper = tar_upper+(tar_upper >> 12)+(tar_upper >> 14)+(tar_upper >> 25)+64
    require(files <= LIMITS['regular_files'] and payload <= LIMITS['payload_bytes'] and
            gzip_upper <= LIMITS['archive_bytes'], 'Prospective whole return exceeds frozen caps')
    simultaneous = LIMITS['payload_bytes']+LIMITS['archive_bytes']+LIMITS['reserve_free_bytes']
    require(simultaneous < LIMITS['initial_free_bytes'], 'Post-install disk admission cannot hold payload, archive and reserve')
    return {'complete': True, 'prospective_only': True, 'measured_peak_resources': False,
        'schema_sha256': SCHEMA_SHA256, 'model515_FP32_bytes': MODEL_FP32_BYTES,
        'fusion72_FP32_bytes': FUSION_FP32_BYTES,
        'Adam72_two_moments_bytes': ADAM_MOMENT_BYTES, 'Adam72_step_scalar_bytes': ADAM_STEP_BYTES,
        'trained_tensor_values_bytes_before_serialization': MODEL_FP32_BYTES+ADAM_MOMENT_BYTES+ADAM_STEP_BYTES,
        'categories': categories, 'projected_files': files, 'projected_payload_bytes': payload,
        'canonical_USTAR_upper_bytes': tar_upper, 'single_gzip_upper_bytes': gzip_upper,
        'readonly_dependency_files': dependency_files, 'readonly_dependency_bytes': dependency_bytes,
        'dependencies_copied_to_return': False,
        'selfcontained_input_weight_copies_included_upload_and_return': True,
        'simultaneous_payload_archive_reserve_cap_bytes': simultaneous,
        'post_install_disk_margin_bytes': LIMITS['initial_free_bytes']-simultaneous,
        'checkpoint_temporary_already_in_payload': True,
        'all604_training_native512_arrays_retained': False,
        'training_scalar_hash_context_records_required': 604,
        'serialization_headroom_bytes_per_trained_state': LIMITS['trained_checkpoint_bytes']-MODEL_FP32_BYTES-ADAM_MOMENT_BYTES-ADAM_STEP_BYTES}


def checkpoint_kind(name):
    if name in CHECKPOINT_PATHS:
        return 'final'
    if any(re.fullmatch(re.escape(v)+r'\.tmp-[0-9a-f]{32}', name) for v in CHECKPOINT_PATHS):
        return 'temporary'
    return None


def checkpoint_accounting(sizes):
    slots = {name: [] for name in CHECKPOINT_PATHS}
    for name, size in sizes.items():
        require(type(size) is int and size >= 0, 'Invalid checkpoint size')
        kind = checkpoint_kind(name)
        if kind:
            base = name if kind == 'final' else next(n for n in slots if name.startswith(n+'.tmp-'))
            slots[base].append((name, size))
    for base, rows in slots.items():
        cap = LIMITS['initial_checkpoint_bytes'] if base == CHECKPOINT_PATHS[0] else LIMITS['trained_checkpoint_bytes']
        require(len(rows) <= 1 and sum(v for _, v in rows) <= cap,
                'One exact final OR retained temporary per checkpoint slot required')
    require(not (slots[CHECKPOINT_PATHS[2]] and slots[CHECKPOINT_PATHS[3]] and
                 all(checkpoint_kind(rows[0][0]) == 'final' for rows in
                     (slots[CHECKPOINT_PATHS[2]], slots[CHECKPOINT_PATHS[3]]))),
            'Successful final151 and actual stopped state cannot both be duplicated')
    temporary = sum(v for rows in slots.values() for n, v in rows if checkpoint_kind(n) == 'temporary')
    require(temporary <= LIMITS['checkpoint_temporary_bytes'], 'Aggregate retained checkpoint temporary cap exceeded')
    require(sum(len(rows) for rows in slots.values()) <= 4, 'Excess checkpoint slots')
    return {'slots': {k: [n for n, _ in rows] for k, rows in slots.items()},
        'temporary_bytes': temporary, 'bytes': sum(v for rows in slots.values() for _, v in rows)}


def _bindings_from_protocol(p):
    require(type(p) is dict and p.get('vm_name') == VM_NAME and p.get('outputs') == OUTPUTS,
            'Exact new protocol namespace required')
    require(type(p.get('files_sha256')) is dict and type(p.get('files_bytes')) is dict and
            set(p['files_sha256']) == set(p['files_bytes']), 'Static source maps differ')
    source = {n: {'sha256': h, 'bytes': p['files_bytes'][n]} for n, h in p['files_sha256'].items()}
    validate_binding_map(source, 'static source bindings')
    require('protocol.json' not in source and not any(n.startswith('outputs/') or n == LOCK_NAME for n in source),
            'Static packet cannot contain results, protocol self-binding or worker lock')
    require(p.get('weight_assets') == WEIGHT_ASSETS and
            all(source.get(n) == pin for n, pin in WEIGHT_ASSETS.items()) and
            not any(n.startswith('weights/') and n not in WEIGHT_ASSETS for n in source),
            'Exact three separately budgeted static model assets required')
    dependencies = p.get('dependencies')
    require(type(dependencies) is dict, 'Exact read-only dependency map required')
    seen = set()
    for name, row in dependencies.items():
        canonical(name)
        require(name.casefold() not in seen and type(row) is dict and
                set(row) == {'root', 'path', 'sha256', 'bytes'}, 'Invalid dependency binding')
        seen.add(name.casefold())
        canonical(row['root']); canonical(row['path'])
        require(len(PurePosixPath(row['root']).parts) == 1,
                'Dependency root must be one exact sibling directory name')
        require(type(row['bytes']) is int and row['bytes'] >= 0 and
                isinstance(row['sha256'], str) and HEX.fullmatch(row['sha256']), 'Invalid dependency pin')
    return source, dependencies


def verify_transfer(root, protocol_sha, *, dependency_base=None):
    """Hash every source/input/dependency before importing a neural framework.

    Dependency roots are canonical siblings below forensic-dgp, never absolute
    path substitutions or an arbitrary parent.  Local fixtures may supply the
    containing directory; actual manual_root still belongs to the supervisor.
    """
    root = no_links(root)
    require(isinstance(protocol_sha, str) and HEX.fullmatch(protocol_sha), 'Exact protocol SHA required')
    pp = regular(root, 'protocol.json')
    require(file_pin(pp)['sha256'] == protocol_sha, 'Protocol hash differs')
    p = read_json(pp)
    source, deps = _bindings_from_protocol(p)
    current = inventory(root)
    require({n for n in current if not n.startswith('outputs/') and n != LOCK_NAME} == set(source)|{'protocol.json'},
            'Unbound static/importable packet namespace')
    for name, pin in sorted(source.items()):
        require(file_pin(regular(root, name)) == pin, 'Static packet bytes differ: '+name)
    nonweights = {n: pin for n, pin in source.items() if n not in WEIGHT_ASSETS}
    require(len(nonweights)+1 <= LIMITS['nonweight_static_files'] and
            sum(v['bytes'] for v in nonweights.values())+pp.stat().st_size <= LIMITS['nonweight_static_bytes'] and
            len(source)+1 <= LIMITS['source_packet_files'] and
            sum(v['bytes'] for v in source.values())+pp.stat().st_size <= LIMITS['source_packet_bytes'],
            'Protocol-inclusive source packet cap exceeded')
    base = no_links(root.parent if dependency_base is None else dependency_base)
    for name, pin in sorted(deps.items()):
        other = no_links(base.joinpath(*PurePosixPath(pin['root']).parts))
        require(other.is_relative_to(base) and other != root,
                'Dependency must remain a separate exact read-only sibling')
        require(file_pin(regular(other, pin['path'])) == {k: pin[k] for k in ('sha256', 'bytes')},
                'Read-only dependency differs: '+name)
    return {'complete': True, 'protocol_sha256': protocol_sha, 'protocol': p,
        'static_files': len(source), 'dependencies_verified': len(deps),
        'readonly_dependency_bytes': sum(v['bytes'] for v in deps.values()),
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0}


def require_worker_terminal(root):
    lock = no_links(root)/LOCK_NAME
    if lock.exists() or lock.is_symlink():
        value = read_json(regular(root, LOCK_NAME))
        pid = value.get('pid')
        require(type(pid) is int and pid > 0, 'Invalid retained worker lock PID')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError:
            raise ValueError('Worker PID cannot be proved terminal')
        else:
            raise ValueError('Worker remains alive; preserve it and do not export')
    result = subprocess.run(['ps', '-eo', 'pid=,args='], check=True,
        capture_output=True, text=True, timeout=20)
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        require(len(fields) == 2 and fields[0].isdigit(), 'Unclassifiable process inventory')
        if int(fields[0]) == os.getpid():
            continue
        try:
            words = shlex.split(fields[1])
        except ValueError:
            require(MODULE not in fields[1], 'Unclassifiable potential worker process')
            continue
        require(not (MODULE in words and '--run' in words), 'Worker or run wrapper is still alive')


def allowed_output_names(protocol):
    """Exact predeclared files, not a permissive arbitrary output directory."""
    names = protocol.get('allowed_output_names')
    require(type(names) is list and len(names) <= LIMITS['regular_files'] and
            len(set(names)) == len(names), 'Frozen finite output namespace required')
    folded = set()
    for name in names:
        canonical(name)
        require(name.startswith(OUTPUTS+'/') or name.startswith('outputs/supervision/'),
                'Output allowance escapes new experiment')
        require(name.casefold() not in folded, 'Colliding output namespace')
        folded.add(name.casefold())
        require(not FORBIDDEN.search(PurePosixPath(name).name) or checkpoint_kind(name) == 'final',
                'Arbitrary checkpoint/model output forbidden')
    return set(names)


def terminal_evidence(root, files, protocol_sha, protocol):
    """Report actual activity; absent or killed in-flight work stays unknown."""
    receipt = read_json(regular(root, 'outputs/supervision/run_receipt.json'))
    require(receipt.get('worker_terminal') is True and receipt.get('protocol_sha256') == protocol_sha,
            'Closed supervisor terminal receipt required')
    success, failure = OUTPUTS+'/run_results.json', OUTPUTS+'/failure.json'
    require(not (success in files and failure in files), 'Conflicting terminal receipts')
    terminal = success if success in files else failure if failure in files else None
    malformed = None
    if terminal is None:
        value = {}
    else:
        path = regular(root, terminal)
        try:
            value = read_json(path)
        except (ValueError, UnicodeError) as error:
            # An outer stop may interrupt a terminal JSON write. Preserve its
            # bytes and every partial artifact; do not invent zero activity or
            # a successful run from a truncated receipt.
            malformed = {'path': terminal, **file_pin(path),
                'error_type': type(error).__name__, 'error': str(error)[:4096]}
            terminal, value = None, {}
    completed, partial, inflight = (value.get(k) for k in
        ('completed_training_updates', 'partial_microbatches', 'optimizer_inflight'))
    counts = value.get('counts')
    if terminal:
        caps = protocol.get('call_caps')
        require(value.get('complete') is True and value.get('protocol_sha256') == protocol_sha and
                type(caps) is dict and type(counts) is dict and set(caps) == set(counts),
                'Exact terminal counter schema differs')
        require(all(type(v) is int and 0 <= v <= caps[k] for k, v in counts.items()) and
                type(completed) is int and 0 <= completed <= 151 and
                type(partial) is int and 0 <= partial <= 4 and type(inflight) is bool,
                'Actual finite activity/state counters differ')
        require(completed <= counts.get('optimizer_step_attempts', -1) and
                completed*4+partial <= counts.get('backward_calls', -1),
                'Committed state contradicts actual reservations')
    if terminal == success:
        required = protocol.get('required_success_output_names')
        require(type(required) is list and len(set(required)) == len(required) and
                set(required).issubset(allowed_output_names(protocol)),
                'Exact planned success coverage required')
        require(completed == 151 and partial == 0 and inflight is False and
                counts == protocol['call_caps'] and all(n in files for n in CHECKPOINT_PATHS[:3]) and
                set(required).issubset(files),
                'Success requires all planned states and exact completed counters')
    if terminal == failure:
        require(value.get('failure') is True, 'Explicit failed/stopped status required')
    state = value.get('state_checkpoint')
    state_complete = value.get('state_checkpoint_complete')
    if state is not None:
        require(type(state) is dict and set(state) == {'path', 'sha256', 'bytes'} and
                state['path'] in files and checkpoint_kind(state['path']) and
                file_pin(regular(root, state['path'])) == {k: state[k] for k in ('sha256', 'bytes')} and
                type(state_complete) is bool,
                'Terminal state checkpoint must bind exact retained physical bytes')
        if state_complete:
            require(checkpoint_kind(state['path']) == 'final',
                    'An interrupted temporary is not a completed checkpoint')
        if completed == 151 and partial == 0 and inflight is False and state_complete:
            require(state['path'] in (CHECKPOINT_PATHS[2], CHECKPOINT_PATHS[3]),
                    'Actual completed151 state cannot reference an earlier snapshot')
    if terminal == success:
        require(state is not None and state_complete is True and state['path'] == CHECKPOINT_PATHS[2],
                'Successful finite run requires the exact final151 state checkpoint')
    return {'worker_terminal_evidence': terminal, 'failure_present': failure in files,
        'complete_training_process': terminal == success,
        'completed_training_updates': completed, 'partial_microbatches': partial,
        'optimizer_inflight': inflight, 'worker_recorded_counts': counts,
        'activity_unknown_without_worker_receipt': terminal is None,
        'malformed_worker_terminal_preserved': malformed,
        'checkpoint_members': sorted(n for n in files if checkpoint_kind(n)),
        'state_checkpoint': state, 'state_checkpoint_complete': state_complete,
        'model_qualified': False, 'training_or_export_completion_not_model_acceptance': True}


@contextmanager
def finite_export():
    """Linux alarm covers hashing and streaming; caller has a separate outer timeout."""
    if not hasattr(signal, 'setitimer'):
        yield
        return
    before = time.monotonic()
    handler, timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame):
        raise TimeoutError('Finite export600second wall limit exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(LIMITS['export_seconds'], timer[0]) if timer[0] else LIMITS['export_seconds'])
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        left = max(0.000001, timer[0]-(time.monotonic()-before)) if timer[0] else 0
        signal.setitimer(signal.ITIMER_REAL, left, timer[1])


class _BoundedWriter:
    def __init__(self, stream, deadline, reserve_path):
        self.stream, self.deadline, self.reserve_path, self.bytes = stream, deadline, reserve_path, 0
    def check(self, amount=0):
        require(time.monotonic() <= self.deadline, 'Finite export wall cap exceeded')
        require(self.bytes+amount <= LIMITS['archive_bytes'], 'Compressed archive cap exceeded')
        require(shutil.disk_usage(self.reserve_path).free >= LIMITS['reserve_free_bytes']+amount,
                'Archive would cross disk reserve; preserve partial file')
    def write(self, payload):
        self.check(len(payload))
        written = self.stream.write(payload)
        require(written == len(payload), 'Short archive write')
        self.bytes += written
        return written
    def flush(self):
        self.stream.flush()


class _HashReader:
    def __init__(self, stream, bounded):
        self.stream, self.bounded, self.digest, self.bytes = stream, bounded, hashlib.sha256(), 0
    def read(self, size=-1):
        self.bounded.check()
        value = self.stream.read(size)
        self.digest.update(value)
        self.bytes += len(value)
        return value


def measure_files(root, names):
    pins, metadata = {}, 0
    for name in sorted(names):
        path = regular(root, name)
        kind = checkpoint_kind(name)
        require(not FORBIDDEN.search(path.name) or kind or name in WEIGHT_ASSETS,
                'Checkpoint outside exact slots or frozen input-weight category')
        size = path.stat().st_size
        if not kind:
            if name in WEIGHT_ASSETS:
                require(file_pin(path) == WEIGHT_ASSETS[name], 'Exact input-weight copy changed before export')
                pins[name] = WEIGHT_ASSETS[name]
                continue
            if path.suffix.lower() in ('.npy', '.npz'):
                require(size <= LIMITS['array_bytes'], 'Individual array cap exceeded')
            elif path.suffix.lower() == '.png':
                require(size <= LIMITS['png256_bytes'], 'Individual PNG256 cap exceeded')
            else:
                metadata += size
                if path.suffix.lower() == '.json':
                    require(size <= LIMITS['one_json_bytes'], 'Individual JSON cap exceeded')
        pins[name] = file_pin(path)
    checkpoint_accounting({n: v['bytes'] for n, v in pins.items()})
    require(len(pins) <= LIMITS['regular_files'] and
            sum(v['bytes'] for v in pins.values()) <= LIMITS['payload_bytes'] and
            metadata <= LIMITS['metadata_bytes'], 'Whole result/file/metadata cap exceeded; do not omit evidence')
    return pins, metadata


def export_results(root, protocol_sha):
    """Export only a terminal new run; leave every prior/partial destination intact."""
    with finite_export():
        return _export_results(root, protocol_sha)


def _export_results(root, protocol_sha):
    root = no_links(root)
    expected = no_links(Path.home()/'forensic-dgp'/VM_NAME)
    require(os.name == 'posix' and root.resolve(strict=True) == expected.resolve(strict=True),
            'Only exact existing manual VM root may export')
    start = time.monotonic()
    verified = verify_transfer(root, protocol_sha)
    protocol = verified['protocol']
    require_worker_terminal(root)
    files = inventory(root)
    status = terminal_evidence(root, files, protocol_sha, protocol)
    allowed = allowed_output_names(protocol)
    require(all(not n.startswith('outputs/') or n in allowed or checkpoint_kind(n) == 'temporary'
                for n in files), 'Unexpected output namespace')
    archive = no_links(Path.home()/(PREFIX+'-results.tar.gz'))
    checksum, companion = Path(str(archive)+'.sha256'), archive.with_name(PREFIX+'-export.json')
    require(not any(n.exists() or n.is_symlink() for n in (archive, checksum, companion)),
            'Preserve prior or partial export; no automatic retry')
    member = OUTPUTS+'/export_manifest.json'
    require(member not in files, 'Preserve previous/partial export manifest')
    excluded = {LOCK_NAME, 'outputs/supervision/export.log', 'outputs/supervision/finished.json'}
    pins, metadata = measure_files(root, set(files)-excluded)
    value = {'format': 'codeformer-filter-fusion-training-return-v1',
        'protocol_sha256': protocol_sha, 'files': pins, **status,
        'dependencies': protocol['dependencies'], 'dependency_assets_copied': False,
        'UTC': datetime.now(timezone.utc).isoformat(),
        'separately_retained': [checksum.name, companion.name, PREFIX+'-supervisor-final.json',
            'outputs/supervision/export.log', 'outputs/supervision/finished.json']}
    encoded = json_bytes(value)
    require(len(encoded) <= LIMITS['one_json_bytes'] and
            len(pins)+1 <= LIMITS['regular_files'] and
            sum(v['bytes'] for v in pins.values())+len(encoded) <= LIMITS['payload_bytes'] and
            metadata+len(encoded) <= LIMITS['metadata_bytes'], 'Complete manifest cap exceeded')
    tar_upper = ((sum(512+((v['bytes']+511)//512)*512 for v in pins.values())+
        512+((len(encoded)+511)//512)*512+1024+10239)//10240)*10240
    bound = tar_upper+(tar_upper >> 12)+(tar_upper >> 14)+(tar_upper >> 25)+64
    require(bound <= LIMITS['archive_bytes'] and
            shutil.disk_usage(archive.parent).free >= bound+LIMITS['reserve_free_bytes'],
            'Complete archive envelope/reserve unavailable')
    write_new_json(root/member, value)
    pins[member] = file_pin(root/member)
    with archive.open('xb') as target:
        bounded = _BoundedWriter(target, start+LIMITS['export_seconds'], archive.parent)
        with gzip.GzipFile(filename='', mode='wb', fileobj=bounded, mtime=0, compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                for name, pin in sorted(pins.items()):
                    path = regular(root, name)
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = pin['bytes'], 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as source:
                        reader = _HashReader(source, bounded)
                        tar.addfile(info, reader)
                        require(reader.bytes == pin['bytes'] and reader.digest.hexdigest() == pin['sha256'],
                                'Source changed while exporting')
        target.flush()
        os.fsync(target.fileno())
    pin = file_pin(archive)
    require(time.monotonic()-start <= LIMITS['export_seconds'], 'Finite full export cap exceeded')
    with checksum.open('xb') as stream:
        stream.write((pin['sha256']+'  '+archive.name+'\n').encode('ascii'))
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {'format': 'codeformer-filter-fusion-training-export-v1',
        'complete': True, 'archive_sha256': pin['sha256'], 'bytes': pin['bytes'],
        'manifest_member': member, 'manifest_sha256': pins[member]['sha256'],
        'protocol_sha256': protocol_sha, 'regular_members': len(pins),
        'payload_bytes': sum(v['bytes'] for v in pins.values()),
        'seconds': time.monotonic()-start, **status}
    write_new_json(companion, receipt)
    return receipt


def checksum_bytes(archive_name, digest):
    canonical(archive_name)
    require('/' not in archive_name and isinstance(digest, str) and HEX.fullmatch(digest),
            'Exact portable basename and SHA required')
    return (digest+'  '+archive_name+'\n').encode('ascii')


class StrictGzip:
    """Bounded one-frame decoder: no tarfile parser, pickle or returned code."""
    def __init__(self, path, expanded_cap, check=None):
        self.source = no_links(path).open('rb')
        self.decoder = zlib.decompressobj(31)
        self.pending, self.buffer = b'', bytearray()
        self.expanded, self.cap, self.ended = 0, expanded_cap, False
        self.check = check
    def _fill(self):
        if self.check is not None:
            self.check()
        if self.ended:
            return
        if self.decoder.eof:
            require(not self.decoder.unused_data and not self.pending and not self.source.read(1),
                    'Trailing gzip bytes or extra gzip frame')
            self.ended = True
            return
        payload = self.pending
        if not payload:
            payload = self.source.read(MiB)
            require(payload, 'Truncated gzip frame')
        block = self.decoder.decompress(payload, 4*MiB)
        self.pending = self.decoder.unconsumed_tail
        self.expanded += len(block)
        require(self.expanded <= self.cap, 'Expanded physical TAR cap exceeded')
        self.buffer.extend(block)
    def read(self, size):
        require(type(size) is int and 0 <= size <= 4*MiB, 'Bounded physical stream read required')
        while len(self.buffer) < size and not self.ended:
            self._fill()
        amount = min(size, len(self.buffer))
        result = bytes(self.buffer[:amount])
        del self.buffer[:amount]
        return result
    def exact(self, size):
        result = self.read(size)
        require(len(result) == size, 'Truncated physical TAR')
        return result
    def close(self):
        self.source.close()


def canonical_header(name, size):
    canonical(name)
    require(type(size) is int and size >= 0, 'Invalid physical member size')
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o644, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    header = info.tobuf(format=tarfile.USTAR_FORMAT, encoding='utf-8', errors='strict')
    require(len(header) == 512, 'Single canonical USTAR header required')
    return header


def verify_archive(archive, expected, *, extract_to=None, archive_pin=None, check=None):
    """Stream exact hash-bound canonical USTAR; extraction is exclusive and inert.

    The caller must independently validate the returned manifest/protocol and
    provide its exact expected map.  This does not execute/import any extracted
    code or deserialize model/array/optimizer contents.  A failed extraction is
    preserved at its distinct destination and never retried automatically.
    """
    if check is not None:
        check()
    validate_binding_map(expected, 'physical archive member bindings')
    require(0 < len(expected) <= LIMITS['regular_files'] and
            sum(v['bytes'] for v in expected.values()) <= LIMITS['payload_bytes'],
            'Exact complete archive envelope exceeded')
    pin = file_pin(archive, check=check)
    require(pin['bytes'] <= LIMITS['archive_bytes'] and
            (archive_pin is None or pin == archive_pin), 'Fresh archive identity differs')
    destination = None
    if extract_to is not None:
        destination = no_links(extract_to)
        require(not destination.exists() and not destination.is_symlink(),
                'Preserve prior/partial inert import; destination must not exist')
        require(shutil.disk_usage(destination.parent).free >=
                sum(v['bytes'] for v in expected.values())+LIMITS['reserve_free_bytes'],
                'Full exclusive import plus reserve does not fit')
        destination.mkdir()
    tar_upper = ((sum(512+((v['bytes']+511)//512)*512 for v in expected.values())+
        1024+10239)//10240)*10240
    stream = StrictGzip(archive, tar_upper, check=check)
    offset, members = 0, 0
    try:
        for name, binding in sorted(expected.items()):
            if check is not None:
                check()
            require(stream.exact(512) == canonical_header(name, binding['bytes']),
                    'Canonical USTAR header/order differs: '+name)
            offset += 512
            target = None
            if destination is not None:
                target = no_links(destination.joinpath(*PurePosixPath(name).parts))
                target.parent.mkdir(parents=True, exist_ok=True)
                target = target.open('xb')
            digest, remaining = hashlib.sha256(), binding['bytes']
            try:
                while remaining:
                    if check is not None:
                        check()
                    block = stream.exact(min(4*MiB, remaining))
                    digest.update(block)
                    if target is not None:
                        require(target.write(block) == len(block), 'Short inert import write')
                    remaining -= len(block)
                require(digest.hexdigest() == binding['sha256'], 'Physical archive payload hash differs: '+name)
                if target is not None:
                    target.flush()
                    os.fsync(target.fileno())
            finally:
                if target is not None:
                    target.close()
            offset += binding['bytes']
            padding = (-binding['bytes']) % 512
            require(stream.exact(padding) == b'\0'*padding, 'Nonzero member padding')
            offset += padding
            members += 1
        trailing = tar_upper-offset
        require(trailing >= 1024, 'Double end-of-archive required')
        while trailing:
            if check is not None:
                check()
            block = stream.exact(min(4*MiB, trailing))
            require(not any(block), 'Nonzero complete end-of-archive padding')
            trailing -= len(block)
        require(not stream.read(1), 'Extra TAR bytes beyond exact padding/full EOF')
    finally:
        stream.close()
    require(file_pin(archive, check=check) == pin, 'Archive changed during physical readback')
    if destination is not None:
        actual = inventory(destination)
        require(set(actual) == set(expected), 'Exclusive import namespace differs')
        require(all(file_pin(actual[n], check=check) == expected[n] for n in expected),
                'Fresh inert imported payload readback differs')
    if check is not None:
        check()
    return {'complete': True, 'archive': pin, 'members': members,
        'expanded_bytes': tar_upper, 'one_gzip_CRC_full_EOF': True,
        'canonical_USTAR_headers_order_padding_double_EOA': True,
        'exclusive_inert_import': destination is not None,
        'returned_code_or_tensors_executed': False,
        'training_or_archive_completion_not_model_acceptance': True}


def command_guide(protocol_sha, archive_sha, project='forensic-dgp-thesis',
                  instance='forensic-dgp-thesis', zone='us-central1-a', user='janusdominic0'):
    """Data-only exact command strings; no subprocess or training is invoked."""
    require(HEX.fullmatch(protocol_sha) and HEX.fullmatch(archive_sha), 'Exact sealed hashes required')
    require(all(re.fullmatch(r'[a-zA-Z0-9_.-]+', n) for n in (project, instance, zone, user)),
            'Unsafe gcloud identity')
    stem = PREFIX+'-execution.tar.gz'
    host = user+'@'+instance
    flags = '--project='+project+' --zone='+zone
    upload = ['cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"']
    upload += ['gcloud compute scp '+flags+' "'+name+'" "'+host+':/home/'+user+'/"'
               for name in (stem, stem+'.sha256')]
    install = ('cd ~ &&\necho '+archive_sha+'  '+stem+' | sha256sum -c - &&\nsha256sum -c '+stem+'.sha256 &&\n'
        'test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&\n'
        'test ! -e ~/forensic-dgp/'+VM_NAME+' &&\n'
        'tar -xzf '+stem+' -C ~/forensic-dgp')
    download = ['cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"']
    download += ['gcloud compute scp '+flags+' "'+host+':/home/'+user+'/'+PREFIX+suffix+'" "."'
        for suffix in ('-results.tar.gz', '-results.tar.gz.sha256', '-export.json', '-supervisor-final.json')]
    return {'upload_Windows_SDK_Shell': upload, 'install_VM_SSH': install,
        'tmux_VM_SSH': 'tmux new-session -A -s codeformer_filter_fusion_v1',
        'launch_inside_tmux': 'cd ~/forensic-dgp/'+VM_NAME+' &&\nsource ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&\npython -B -u scripts/supervise_codeformer_filter_fusion_training_v1_vm.py --root . --protocol-sha '+protocol_sha,
        'download_Windows_SDK_Shell_one_remote_source_per_command': download,
        'expected_archive_sha256': archive_sha,
        'manual_only': True, 'no_actual_commands_executed': True}

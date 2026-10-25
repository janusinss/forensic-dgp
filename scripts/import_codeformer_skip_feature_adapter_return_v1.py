"""Future inert import of the one sealed manual skip-feature pilot return.

Four downloaded companions and the frozen local protocol are authority. Returned
Python, NPY and checkpoints remain opaque bytes. No framework, returned module,
VM command, optimizer or neural operation is imported or executed. A failed or
partial import is retained exclusively and cannot be retried automatically.
"""
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import time
import traceback
import unicodedata
import zlib

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'cctv-codeformer-skip-feature-adapter-v1'
VM_NAME = 'cctv_codeformer_skip_feature_adapter_v1_vm'
OUTPUT = 'outputs/skip_feature_adapter_v1'
PROTOCOL_SHA = '0d358ab29e0344c7243b6312cb1d04ca6cc0e0a1de270bca7e6b4040dd566c90'
STAGE = ROOT / 'outputs' / VM_NAME
DEST = ROOT / 'outputs/codeformer_skip_feature_adapter_v1_return'
REVIEW = ROOT / 'outputs/codeformer_skip_feature_adapter_return_review_v1'
MANIFEST = OUTPUT + '/export_manifest.json'
GiB, MiB = 1024**3, 1024**2
ARCHIVE_CAP, PAYLOAD_CAP, MEMBER_CAP = 37 * GiB // 4, 9 * GiB, 20000
METADATA_CAP, JSON_CAP, LOG_CAP = 128 * MiB, 4 * MiB, 2 * MiB
ARRAY_CAP, PNG_CAP, CHECKPOINT_CAP, RESERVE = 40 * MiB, MiB, 384 * MiB, 4 * GiB
TAR_CAP = ((PAYLOAD_CAP + MEMBER_CAP * 1023 + 1024 + 10239) // 10240) * 10240
IMPORT_SECONDS = 1200
CHECKPOINTS = tuple(OUTPUT + '/' + name for name in (
    'snapshot_0000/checkpoint.pt', 'snapshot_0055/checkpoint.pt',
    'snapshot_0110/checkpoint.pt', 'stopped_checkpoint.pt'))
COUNTS = {'original_baseline_forwards':478, 'parent_baseline_forwards':478,
    'R2_baseline_forwards':478, 'R2_clean_feature_teacher_forwards':55,
    'original_clear_teacher_forwards':55, 'R2_dark_teacher_forwards':55,
    'candidate_snapshot_forwards':1434, 'candidate_training_forwards':440,
    'candidate_TRAIN_guard_forwards':96, 'candidate_witness_forwards':3,
    'R2_auxiliary_forwards':18, 'candidate_auxiliary_forwards':54,
    'backward_calls':440, 'optimizer_step_attempts':110, 'gradient_queries':4}
WEIGHTS = {
    'weights/codeformer_original.pth': {'bytes':376637898,
        'sha256':'1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'},
    'weights/codeformer_r2.pth': {'bytes':376609465,
        'sha256':'d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'},
    'weights/codeformer_parent.pth': {'bytes':376643953,
        'sha256':'53778f54589697a14b067033eff8cffc2985985e8a33021092de3c14fbf443bc'}}
SUPERVISION = tuple('outputs/supervision/' + name for name in (
    'preflight.log', 'worker.log', 'run_receipt.json', 'export_invocation.json',
    'export.log', 'finished.json'))
ARCHIVE_EXCLUSIONS = {'.skip_feature_adapter_v1.lock', SUPERVISION[4], SUPERVISION[5]}
HEX = re.compile(r'^[0-9a-f]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)', re.I)
BINARY = re.compile(r'\.(?:pt|pth|ckpt|safetensors|pkl|pickle|bin)(?:$|[.\-])', re.I)


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def compact(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def parse(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON key'); result[key] = value
        return result
    return json.loads(payload.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def unlinked(path, regular=False):
    path = Path(path).absolute()
    for current in (path, *path.parents):
        if current.exists() or current.is_symlink():
            row = current.lstat()
            need(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse import path refused')
    if regular:
        row = path.lstat()
        need(stat.S_ISREG(row.st_mode) and row.st_nlink == 1, 'Unique regular file required')
    return path


def pin(path):
    path = unlinked(path, True); before = path.lstat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * MiB), b''):
            digest.update(block)
    after = path.lstat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during hash')
    return {'bytes':before.st_size, 'sha256':digest.hexdigest()}


def read(path, cap=JSON_CAP):
    path = unlinked(path, True)
    need(path.stat().st_size <= cap, 'Bounded metadata read exceeded')
    return path.read_bytes()


def load(path):
    return parse(read(path))


def write_bytes(path, payload):
    path = unlinked(path)
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    need(shutil.disk_usage(ancestor).free >= RESERVE + len(payload) + 65536,
         'Local4GiB reserve reached; preserve partial import')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        need(stream.write(payload) == len(payload), 'Short exclusive import write')
        stream.flush(); os.fsync(stream.fileno())


def write_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    need(len(payload) <= JSON_CAP, 'Import evidence JSON cap exceeded')
    write_bytes(path, payload)


def portable(name):
    need(isinstance(name, str) and name, 'Nonempty portable member name required')
    value = PurePosixPath(name)
    need(not value.is_absolute() and value.as_posix() == name and
         all(part not in ('', '.', '..') for part in value.parts), 'Noncanonical relative archive name')
    need(unicodedata.normalize('NFC', name) == name and not any(c in name for c in '\\:"<>|?*') and
         all(not DEVICE.match(part) and not part.endswith((' ', '.')) and
             len(part.encode('utf-16-le')) <= 510 and all(ord(c) >= 32 for c in part)
             for part in value.parts), 'Nonportable archive name')
    return value


def binding_map(value):
    need(type(value) is dict and len(value) <= MEMBER_CAP, 'Bounded manifest binding map required')
    seen = set()
    for name, item in value.items():
        portable(name); need(name.casefold() not in seen, 'Case-colliding binding'); seen.add(name.casefold())
        need(type(item) is dict and set(item) == {'bytes', 'sha256'} and type(item['bytes']) is int and
             0 <= item['bytes'] <= PAYLOAD_CAP and isinstance(item['sha256'], str) and HEX.fullmatch(item['sha256']),
             'Exact physical byte/SHA binding required')


def checkpoint_kind(name):
    if name in CHECKPOINTS:
        return 'final'
    if any(re.fullmatch(re.escape(base) + r'\.tmp-[0-9a-f]{32}', name) for base in CHECKPOINTS):
        return 'temporary'
    return None


def checkpoint_limits(files):
    slots = {base:[] for base in CHECKPOINTS}; temporary_bytes = 0
    for name, item in files.items():
        kind = checkpoint_kind(name)
        if kind:
            base = name if kind == 'final' else next(base for base in slots if name.startswith(base + '.tmp-'))
            need(0 <= item['bytes'] <= CHECKPOINT_CAP, '384MiB checkpoint cap exceeded')
            slots[base].append(name)
            if kind == 'temporary':
                temporary_bytes += item['bytes']
    need(all(len(names) <= 1 for names in slots.values()) and temporary_bytes <= CHECKPOINT_CAP,
         'Duplicate final/temporary checkpoint slot or aggregate temporary cap')
    need(not (CHECKPOINTS[2] in files and CHECKPOINTS[3] in files), 'Duplicate final110/stopped checkpoint state')
    return {'slots':slots, 'temporary_bytes':temporary_bytes}


def canonical_header(name, size):
    """Reconstruct the producer's exact regular-file USTAR header."""
    portable(name); need(type(size) is int and 0 <= size <= PAYLOAD_CAP, 'Finite USTAR member size required')
    name = name.encode('ascii'); prefix = b''
    if len(name) > 100:
        choices = [i for i, char in enumerate(name) if char == 47 and i <= 155 and len(name) - i - 1 <= 100]
        need(choices, 'USTAR name cannot fit canonical header')
        i = choices[0]; prefix, name = name[:i], name[i+1:]
    raw = bytearray(512); raw[:len(name)] = name; raw[345:345+len(prefix)] = prefix
    for start, width, value in ((100,8,0o644), (108,8,0), (116,8,0), (124,12,size), (136,12,0)):
        field = (('%0*o' % (width-1, value)) + '\0').encode('ascii')
        need(len(field) == width, 'USTAR octal field overflow'); raw[start:start+width] = field
    raw[148:156] = b' ' * 8; raw[156] = 48; raw[257:263] = b'ustar\0'; raw[263:265] = b'00'
    raw[148:156] = ('%06o\0 ' % sum(raw)).encode('ascii')
    return bytes(raw)


class StrictGzip:
    """Exactly one gzip frame, canonical header, CRC/ISIZE and physical EOF."""
    def __init__(self, path, check):
        self.path = unlinked(path, True); self.source = self.path.open('rb'); self.check = check
        self.decoder = zlib.decompressobj(31); self.pending = b''; self.buffer = bytearray()
        self.digest = hashlib.sha256(); self.compressed = self.expanded = self.crc = 0; self.ended = False
        with self.path.open('rb') as source:
            need(source.read(10) == b'\x1f\x8b\x08\0\0\0\0\0\0\xff', 'Canonical level6 zero-time gzip required')
            source.seek(-8, 2); trailer = source.read(8)
            need(len(trailer) == 8, 'Gzip trailer missing'); self.trailer = struct.unpack('<II', trailer)

    def fill(self):
        self.check()
        if self.ended:
            return
        if self.decoder.eof:
            need(not self.decoder.unused_data and not self.pending and not self.source.read(1),
                 'Extra gzip frame or trailing compressed bytes refused')
            need(self.trailer == (self.crc, self.expanded % 2**32), 'Physical gzip CRC/ISIZE differs')
            self.ended = True; return
        payload = self.pending
        if not payload:
            payload = self.source.read(MiB); need(payload, 'Truncated gzip frame')
            self.digest.update(payload); self.compressed += len(payload)
            need(self.compressed <= ARCHIVE_CAP, 'Compressed9.25GiB cap exceeded')
        block = self.decoder.decompress(payload, 4 * MiB); self.pending = self.decoder.unconsumed_tail
        self.expanded += len(block); self.crc = zlib.crc32(block, self.crc)
        need(self.expanded <= TAR_CAP, 'Expanded TAR byte cap exceeded'); self.buffer.extend(block)

    def read(self, size):
        need(type(size) is int and 0 <= size <= 4 * MiB, 'Bounded decompression read required')
        while len(self.buffer) < size and not self.ended:
            self.fill()
        take = min(size, len(self.buffer)); result = bytes(self.buffer[:take]); del self.buffer[:take]
        return result

    def exact(self, size):
        result = self.read(size); need(len(result) == size, 'Truncated physical TAR'); return result

    def close(self):
        self.source.close()


def stream_tar(strict, member_policy, sink_factory, check):
    """Hash every byte; policies/sinks are caller-owned inert helpers."""
    files = {}; folded = set(); previous = ''; position = total = metadata = 0
    while True:
        check(); raw = strict.exact(512)
        if raw == b'\0' * 512:
            need(strict.exact(512) == b'\0' * 512, 'Two physical TAR end blocks required')
            padding = ((position + 1024 + 10239) // 10240) * 10240 - position - 1024
            need(strict.exact(padding) == b'\0' * padding and strict.read(1) == b'',
                 'Canonical TAR record padding and full EOF required')
            break
        def field(start, width):
            return raw[start:start+width].split(b'\0', 1)[0].decode('ascii')
        name = field(0,100); prefix = field(345,155); name = prefix + '/' + name if prefix else name
        portable(name); need(re.fullmatch(b'[0-7]{11}\0', raw[124:136]), 'Canonical TAR octal size required')
        size = int(raw[124:135], 8)
        need(raw == canonical_header(name, size), 'Canonical regular USTAR header required; links/PAX/devices refused')
        need(name not in files and name.casefold() not in folded and name > previous,
             'Duplicate, case-colliding or unsorted physical archive member')
        folded.add(name.casefold()); previous = name
        need(len(folded) <= MEMBER_CAP, '20000 regular-member cap exceeded')
        meta = member_policy(name, size)
        total += size; metadata += size if meta else 0
        need(total <= PAYLOAD_CAP and metadata <= METADATA_CAP, '9GiB payload/128MiB metadata cap exceeded')
        digest = hashlib.sha256(); remaining = size
        with sink_factory(name, size) as sink:
            while remaining:
                check(); block = strict.exact(min(remaining, MiB))
                need(sink.write(block) == len(block), 'Short inert member write')
                digest.update(block); remaining -= len(block)
        padding = (-size) % 512
        need(strict.exact(padding) == b'\0' * padding, 'Nonzero TAR member padding refused')
        files[name] = {'bytes':size, 'sha256':digest.hexdigest()}; position += 512 + size + padding
    need(strict.ended, 'Exact single-gzip EOF required')
    return files, total, metadata


def frozen_protocol():
    value = read(STAGE / 'protocol.json'); need(sha(value) == PROTOCOL_SHA, 'Frozen local protocol differs')
    protocol = parse(value); static = protocol['files']; binding_map(static)
    need(protocol['format'] == 'codeformer-skip-feature-adapter-finite-protocol-v1' and
         protocol['vm_name'] == VM_NAME and protocol['outputs'] == OUTPUT and len(static) == 876 and
         protocol['dependencies'] == {} and protocol['weight_assets'] == WEIGHTS and
         protocol['call_caps'] == COUNTS and protocol['max_updates'] == 110 and
         protocol['automatic_resume'] is False and protocol['automatic_training_launch'] is False,
         'Exact frozen new165/478 manual pilot contract differs')
    need(all(static.get(name) == item for name, item in WEIGHTS.items()), 'Three sealed input-weight pins differ')
    allowed = protocol['allowed_output_names']; required = protocol['required_success_output_names']
    need(type(allowed) is list and len(allowed) == len(set(allowed)) <= MEMBER_CAP and
         type(required) is list and len(required) == len(set(required)) and set(required).issubset(allowed),
         'Exact bounded frozen output names required')
    names = set(static) | {'protocol.json'} | set(allowed); folded = set()
    for name in names:
        portable(name); need(name.casefold() not in folded, 'Frozen namespace collision'); folded.add(name.casefold())
    need(all(name in allowed for name in CHECKPOINTS) and set(SUPERVISION).issubset(allowed) and
         MANIFEST in allowed and not set(static).intersection(allowed), 'Separate exact static/output namespaces required')
    return protocol, {**static, 'protocol.json':{'bytes':len(value), 'sha256':PROTOCOL_SHA}}, set(allowed)


def member_policy(static, allowed):
    def policy(name, size):
        need(name not in ARCHIVE_EXCLUSIONS, 'Lock/final external companion is forbidden inside return archive')
        kind = checkpoint_kind(name)
        need(name in static or name in allowed or kind == 'temporary', 'Unplanned returned member refused')
        suffix = PurePosixPath(name).suffix.lower()
        if name in static:
            need(size == static[name]['bytes'], 'Frozen static member size differs')
        elif kind:
            need(size <= CHECKPOINT_CAP, '384MiB retained checkpoint cap exceeded')
        else:
            need(not BINARY.search(PurePosixPath(name).name), 'Unplanned model binary refused')
            cap = ARRAY_CAP if suffix in ('.npy', '.npz') else PNG_CAP if suffix == '.png' else JSON_CAP if suffix == '.json' else LOG_CAP
            need(size <= cap, 'Per-member retained array/image/metadata cap exceeded')
        return suffix not in ('.npy', '.npz', '.png') and name not in WEIGHTS and not kind
    return policy


def validate_stages(stages):
    need(type(stages) is list and len(stages) == 3, 'One preflight/run/export sequence required')
    expected = (('--verify-transfer',300,SUPERVISION[0]), ('--run',5300,SUPERVISION[1]), ('--export',600,SUPERVISION[4]))
    for row, (flag, cap, log) in zip(stages, expected):
        need(type(row) is dict and row.get('stage') == flag and row.get('external_cap_seconds') == cap and
             row.get('grace_seconds') == 30 and row.get('log') == log and
             type(row.get('log_bytes')) is int and 0 <= row['log_bytes'] <= LOG_CAP and
             type(row.get('child_pid')) is int and row['child_pid'] > 0 and type(row.get('exit_code')) is int and
             row.get('worker_terminal') is True and row.get('child_process_group_terminal') is True and
             type(row.get('outer_watchdog_expired')) is bool and type(row.get('log_capped')) is bool and
             (row.get('supervisor_exception') is None or type(row['supervisor_exception']) is dict) and
             type(row.get('seconds')) in (int,float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
             'Typed actual finite supervisor stage/process evidence required')
    for row in (stages[0], stages[2]):
        need(row['exit_code'] == 0 and row['outer_watchdog_expired'] is False and row['log_capped'] is False and
             row.get('supervisor_exception') is None, 'Transfer preflight and export must actually close successfully')
    return stages[1]


def terminal_status(files, document, run, protocol):
    """Independently recompute activity/process labels; no returned code import."""
    success, failure = OUTPUT + '/run_results.json', OUTPUT + '/failure.json'
    need(not (success in files and failure in files), 'Conflicting worker terminal receipts')
    terminal = success if success in files else failure if failure in files else None
    malformed = None; value = {}
    if terminal is not None:
        try:
            value = document(terminal)
            need(type(value) is dict, 'Worker terminal JSON must be an object')
        except (ValueError, UnicodeError) as error:
            malformed = {'path':terminal, **files[terminal], 'error_type':type(error).__name__, 'error':str(error)[:4096]}
            terminal = None; value = {}
    completed, partial, inflight = (value.get(k) for k in ('completed_training_updates','partial_microbatches','optimizer_inflight'))
    counts, returned = value.get('counts'), value.get('actual_return_counts')
    if terminal:
        need(value.get('complete') is True and value.get('protocol_sha256') == PROTOCOL_SHA and
             type(counts) is dict and type(returned) is dict and set(counts) == set(returned) == set(COUNTS),
             'Actual worker counter schema differs')
        need(all(type(counts[k]) is int and type(returned[k]) is int and 0 <= returned[k] <= counts[k] <= COUNTS[k] for k in COUNTS) and
             type(completed) is int and 0 <= completed <= 110 and type(partial) is int and 0 <= partial <= 4 and
             type(inflight) is bool and completed <= returned['optimizer_step_attempts'] and
             completed * 4 + partial <= returned['backward_calls'], 'Inconsistent finite worker commit/operation evidence')
    if terminal == success:
        need(completed == 110 and partial == 0 and inflight is False and counts == returned == COUNTS and
             set(protocol['required_success_output_names']).issubset(files), 'Successful worker needs full110/count/output census')
    if terminal == failure:
        need(value.get('failure') is True, 'Explicit failed/stopped worker evidence required')
    state, state_complete = value.get('state_checkpoint'), value.get('state_checkpoint_complete')
    if state is not None:
        need(type(state) is dict and set(state) == {'path','bytes','sha256'} and state['path'] in files and
             checkpoint_kind(state['path']) and {k:state[k] for k in ('bytes','sha256')} == files[state['path']] and
             type(state_complete) is bool, 'Retained actual checkpoint bytes differ')
        if state_complete:
            need(checkpoint_kind(state['path']) == 'final', 'Temporary checkpoint is not complete')
        if completed == 110 and partial == 0 and inflight is False and state_complete:
            need(state['path'] in CHECKPOINTS[2:], 'Final110 cannot claim earlier state')
    if terminal == success:
        need(state is not None and state_complete is True and state['path'] == CHECKPOINTS[2], 'Success must bind exact final110 checkpoint')
    clean = run['exit_code'] == 0 and not run['outer_watchdog_expired'] and not run['log_capped'] and run.get('supervisor_exception') is None
    return {'worker_terminal_evidence':terminal, 'failure_present':failure in files,
        'complete_training_process':terminal == success and clean,
        'worker_run_exit_code':run['exit_code'], 'worker_run_clean_exit':clean,
        'worker_run_outer_watchdog_expired':run['outer_watchdog_expired'], 'worker_run_log_capped':run['log_capped'],
        'worker_run_supervisor_exception':run.get('supervisor_exception'),
        'completed_training_updates':completed, 'partial_microbatches':partial, 'optimizer_inflight':inflight,
        'worker_recorded_counts':counts, 'worker_actual_return_counts':returned,
        'activity_unknown_without_worker_receipt':terminal is None, 'malformed_worker_terminal_preserved':malformed,
        'checkpoint_members':sorted(n for n in files if checkpoint_kind(n)), 'state_checkpoint':state,
        'state_checkpoint_complete':state_complete, 'model_qualified':False,
        'training_or_export_completion_not_model_acceptance':True}


def import_return():
    started = time.monotonic()
    def check():
        need(time.monotonic() - started <= IMPORT_SECONDS, 'Finite1200s inert import limit reached')
    need(not any(n.split('.')[0] in ('torch','numpy','tensorflow','jax') for n in sys.modules), 'Fresh standard-library importer required')
    protocol, static, allowed = frozen_protocol()
    outputs = unlinked(ROOT / 'outputs'); unlinked(DEST); unlinked(REVIEW)
    need(not DEST.exists() and not REVIEW.exists(), 'Preserve prior complete/partial import and evidence; no replay')
    archive = unlinked(outputs / (PREFIX + '-results.tar.gz'), True)
    checksum = outputs / (PREFIX + '-results.tar.gz.sha256')
    export_path = outputs / (PREFIX + '-export.json'); final_path = outputs / (PREFIX + '-supervisor-final.json')
    companions = {p.name:pin(p) for p in (checksum, export_path, final_path)}
    exported, final = load(export_path), load(final_path)
    need(type(exported) is dict and exported.get('format') == 'codeformer-skip-feature-adapter-export-v1' and
         exported.get('complete') is True and exported.get('protocol_sha256') == PROTOCOL_SHA and
         exported.get('manifest_member') == MANIFEST and HEX.fullmatch(exported.get('manifest_sha256','')) and
         HEX.fullmatch(exported.get('archive_sha256','')) and type(exported.get('bytes')) is int and
         0 < exported['bytes'] <= ARCHIVE_CAP and type(exported.get('regular_members')) is int and
         len(static) < exported['regular_members'] <= MEMBER_CAP and type(exported.get('payload_bytes')) is int and
         sum(p['bytes'] for p in static.values()) < exported['payload_bytes'] <= PAYLOAD_CAP,
         'Bounded exact new-pilot export companion required')
    archive_before = archive.lstat()
    need(archive_before.st_size == exported['bytes'] and read(checksum,1024) ==
         (exported['archive_sha256'] + '  ' + archive.name + '\n').encode('ascii'), 'Actual archive size/ASCII-LF checksum differs')
    need(final.get('format') == 'codeformer-skip-feature-adapter-supervisor-final-v1' and final.get('complete') is True and
         final.get('protocol_sha256') == PROTOCOL_SHA and final.get('model_qualified') is False and
         final.get('archive') == {'path':archive.name, 'bytes':exported['bytes'], 'sha256':exported['archive_sha256']} and
         final.get('export_receipt') == {'path':export_path.name, **companions[export_path.name]} and
         final.get('checksum') == {'path':checksum.name, **companions[checksum.name]}, 'Noncircular final companion identity differs')
    stages = final['stages']; run = validate_stages(stages)
    log_meta = final['export_log']; need(type(log_meta) is dict and type(log_meta.get('base64')) is str and len(log_meta['base64']) <= 4*LOG_CAP//3+8,
                                        'Bounded base64 export log required')
    log = base64.b64decode(log_meta['base64'], validate=True)
    need(base64.b64encode(log).decode('ascii') == log_meta['base64'] and log_meta['path'] == SUPERVISION[4] and
         len(log) == log_meta['bytes'] <= LOG_CAP and sha(log) == log_meta['sha256'] and
         parse(log) == exported and stages[2]['log_bytes'] == len(log), 'Final export log bytes/receipt binding differs')
    finished = compact({'complete':True, 'protocol_sha256':PROTOCOL_SHA, 'worker_terminal':True,
        'stages':stages, 'model_qualified':False, 'training_or_export_completion_not_model_acceptance':True})
    need(final['finished'] == {'path':SUPERVISION[5], 'bytes':len(finished), 'sha256':sha(finished)}, 'Separately closed finished bytes differ')
    need(shutil.disk_usage(outputs).free >= exported['payload_bytes'] + len(log) + len(finished) + RESERVE,
         'Full return payload plus4GiB local reserve required')
    REVIEW.mkdir()
    write_json(REVIEW / 'import_started_v1.json', {'complete':False, 'protocol_sha256':PROTOCOL_SHA,
        'archive':{'path':archive.name, 'bytes':archive_before.st_size, 'sha256':exported['archive_sha256']},
        'companion_pins':companions, 'source':pin(__file__), 'partial_import_must_be_preserved':True,
        'returned_code_executed':False, 'neural_calls':0})
    try:
        DEST.mkdir(); strict = StrictGzip(archive, check)
        class Sink:
            def __init__(self, name, size):
                self.path = unlinked(DEST.joinpath(*portable(name).parts)); self.size = size
            def __enter__(self):
                need(self.path.resolve(strict=False).is_relative_to(DEST.resolve(strict=True)), 'Import target escaped exact root')
                need(shutil.disk_usage(outputs).free >= RESERVE + self.size + 65536, 'Import4GiB reserve reached')
                self.path.parent.mkdir(parents=True, exist_ok=True); self.stream = self.path.open('xb'); return self.stream
            def __exit__(self, kind, value, tb):
                try:
                    self.stream.flush(); os.fsync(self.stream.fileno())
                finally:
                    self.stream.close()
        try:
            files, total, metadata = stream_tar(strict, member_policy(static,allowed), Sink, check)
        finally:
            strict.close()
        need(strict.compressed == exported['bytes'] and strict.digest.hexdigest() == exported['archive_sha256'], 'Every compressed byte SHA differs')
        after = archive.lstat()
        need((archive_before.st_ino,archive_before.st_size,archive_before.st_mtime_ns) ==
             (after.st_ino,after.st_size,after.st_mtime_ns), 'Downloaded archive changed during import')
        need(len(files) == exported['regular_members'] and total == exported['payload_bytes'] and
             files.get('protocol.json') == static['protocol.json'] and
             files.get(MANIFEST,{}).get('sha256') == exported['manifest_sha256'], 'Actual member/payload/protocol/manifest census differs')
        need(set(static).issubset(files) and all(files[n] == p for n,p in static.items()), 'All877 frozen source/model/data bytes must match')
        checkpoint_inventory = checkpoint_limits(files)
        manifest = load(DEST / MANIFEST); bindings = manifest['files']; binding_map(bindings)
        need(manifest.get('format') == 'codeformer-skip-feature-adapter-return-v1' and manifest.get('protocol_sha256') == PROTOCOL_SHA and
             manifest.get('dependencies') == {} and manifest.get('dependency_assets_copied') is False and
             set(bindings) == set(files) - {MANIFEST} and all(bindings[n] == files[n] for n in bindings),
             'Exact self-excluded manifest binding for every physical member required')
        need(manifest.get('separately_retained') == [checksum.name,export_path.name,final_path.name,SUPERVISION[4],SUPERVISION[5]],
             'Noncircular separate companion declaration differs')
        def document(name):
            return load(DEST / name)
        status = terminal_status(files,document,run,protocol)
        for key,value in status.items():
            need(key in manifest and manifest[key] == value and key in exported and exported[key] == value,
                 'Independent process/state label differs from companion or manifest: ' + key)
        supervised = document(SUPERVISION[2]); invocation = document(SUPERVISION[3])
        need(supervised.get('protocol_sha256') == PROTOCOL_SHA and supervised.get('worker_terminal') is True and
             supervised.get('child_process_group_terminal') is True and supervised.get('stages') == stages[:2] and
             supervised.get('no_automatic_retry_or_resume') is True and supervised.get('maximum_training_updates') == 110 and
             supervised.get('model_qualified') is False, 'Archived actual run supervisor differs')
        need(invocation.get('protocol_sha256') == PROTOCOL_SHA and invocation.get('worker_terminal') is True and
             invocation.get('export_failure_evidence') == (run['exit_code'] != 0) and invocation.get('external_cap_seconds') == 600,
             'Archived export invocation differs')
        need(all(files[row['log']]['bytes'] == row['log_bytes'] for row in stages[:2]), 'Archived preflight/run log byte count differs')
        for path in (checksum,export_path,final_path):
            need(pin(path) == companions[path.name], 'Companion changed during inert import')
        need(sha(read(STAGE/'protocol.json')) == PROTOCOL_SHA, 'Frozen local authority changed')
        # External closed artifacts were intentionally absent from the archive.
        write_bytes(DEST / SUPERVISION[4],log); write_bytes(DEST / SUPERVISION[5],finished)
        reconstructed = {SUPERVISION[4]:{'bytes':len(log),'sha256':sha(log)}, SUPERVISION[5]:{'bytes':len(finished),'sha256':sha(finished)}}
        actual = {}; dirs = set(); folded = set()
        for directory,folders,leaves in os.walk(DEST,followlinks=False):
            for name in folders+leaves:
                path = unlinked(Path(directory)/name); rel = path.relative_to(DEST).as_posix(); portable(rel)
                need(rel.casefold() not in folded, 'Imported namespace case collision'); folded.add(rel.casefold())
                if name in folders:
                    need(stat.S_ISDIR(path.lstat().st_mode), 'Non-directory extraction namespace'); dirs.add(rel)
                else:
                    unlinked(path,True); actual[rel] = path.stat().st_size
        all_files = {**files,**reconstructed}
        expected_dirs = {q.as_posix() for name in all_files for q in portable(name).parents if q.as_posix() != '.'}
        need(set(actual) == set(all_files) and dirs == expected_dirs and all(actual[n] == p['bytes'] for n,p in all_files.items()),
             'Exclusive final inert namespace differs')
        need(not any(n.split('.')[0] in ('torch','numpy','tensorflow','jax') for n in sys.modules), 'No framework execution invariant failed')
        check()
        receipt = {'format':'codeformer-skip-feature-adapter-local-import-v1', 'complete':True, 'transport_pass':True,
            'UTC':datetime.now(timezone.utc).isoformat(), 'import_root':str(DEST), 'protocol_sha256':PROTOCOL_SHA,
            'archive':{'path':archive.name,'bytes':strict.compressed,'sha256':strict.digest.hexdigest()},
            'manifest':{'path':MANIFEST,**files[MANIFEST]}, 'files':files, 'regular_archive_members':len(files),
            'archive_payload_bytes':total, 'metadata_bytes':metadata, 'expanded_TAR_bytes':strict.expanded,
            'gzip_CRC32':strict.crc, 'gzip_ISIZE':strict.expanded % 2**32,
            'all877_frozen_static_files_exact':True, 'single_gzip_physical_EOF_canonical_USTAR':True,
            'export_manifest_self_excluded_exact':True, 'external_artifacts_reconstructed_from_bound_companion':reconstructed,
            'companion_pins':companions, 'checkpoint_inventory':checkpoint_inventory, 'independently_recomputed_status':status,
            'returned_code_executed':False, 'checkpoint_or_array_deserialized':False,
            'actual_local_neural_gradient_optimizer_VM_calls':0, 'source':pin(__file__),
            'model_qualified':False, 'independent_scientific_state_pixel_visual_review_required':True,
            'scope':'Exact inert transport closure only; retained training/process completion is distinct from useful restoration or app acceptance.',
            'seconds':time.monotonic()-started}
        write_json(REVIEW / 'return_import_v1.json',receipt)
        print(json.dumps({'complete':True,'transport_pass':True,'import_root':str(DEST),
            'regular_archive_members':len(files),'payload_bytes':total,'protocol_sha256':PROTOCOL_SHA,
            'complete_training_process':status['complete_training_process'],
            'completed_training_updates':status['completed_training_updates'],'failure_present':status['failure_present'],
            'activity_unknown_without_worker_receipt':status['activity_unknown_without_worker_receipt'],
            'model_qualified':False,'seconds':time.monotonic()-started}))
    except BaseException:
        failure = REVIEW / 'import_failure_v1.json'
        if not failure.exists():
            write_json(failure,{'complete':False,'traceback':traceback.format_exc(),'protocol_sha256':PROTOCOL_SHA,
                'partial_import_preserved':DEST.exists(),'returned_code_executed':False,
                'checkpoint_or_array_deserialized':False,'source':pin(__file__),'no_automatic_retry':True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import', dest='do_import', action='store_true', required=True,
                        help='Inertly import the four already downloaded companions; never train or run returned code.')
    parser.parse_args(); import_return()


if __name__ == '__main__':
    main()

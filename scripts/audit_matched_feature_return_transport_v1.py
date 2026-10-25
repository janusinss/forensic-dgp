"""Independent bounded return transport audit and data-only retention.

Never imports returned sources, decodes checkpoints/arrays, invokes a worker,
constructs a model, launches training or connects to a VM. Retained executables
are forbidden. Full gzip framing, tar payloads and actual compressed bytes are
verified before a complete transport receipt is emitted.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_matched_feature_learning_v1_vm'
RUN = 'outputs/matched_feature_learning_v1/'
PREFIX = 'cctv-codeformer-matched-feature-learning-v1'
PROTOCOL_SHA = 'afd59b90951138f94f0fcf0dea6ef308f8860042ae3c4f25b43c279d13d0fbd8'
LOCAL = ROOT / 'outputs' / STAGE
OUT = ROOT / 'outputs/codeformer_matched_feature_learning_v1_return_audit_v1'
MAX_BYTES = 12 * 1024**3
MAX_EXPANDED = MAX_BYTES + 64 * 1024**2
MAX_MEMBER = 384 * 1024**2
MAX_METADATA = 32 * 1024**2
MAX_FILES = 15000
MAX_SECONDS = 600
RUNTIME = frozenset({
    'install.json', 'install_failure.json', 'run.log', 'export.log',
    'supervisor.start.json', 'supervisor.worker.started.json', 'supervisor.worker.json',
    'supervisor.export.started.json', 'supervisor.export.json', 'supervisor.json',
    'supervisor_failure.json', 'export.json', 'export_failure.json',
})
FORBIDDEN_SUFFIX = ('.py', '.pyc', '.pyo', '.sh', '.bat', '.cmd', '.exe', '.dll', '.ps1')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    need(type(raw) is bytes and len(raw) <= MAX_METADATA, 'Bounded exact JSON bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key refused')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON token refused: ' + value)
    result = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(value, depth=0):
        need(depth <= 64, 'Bounded JSON nesting required')
        if type(value) is float:
            need(math.isfinite(value), 'Finite JSON number required')
        elif type(value) is dict:
            for child in value.values(): finite(child, depth + 1)
        elif type(value) is list:
            for child in value: finite(child, depth + 1)
    finite(result)
    return result


def canonical(value):
    need(type(value) is str and 0 < len(value) <= 512 and '\\' not in value and ':' not in value
         and all(ord(c) >= 32 for c in value), 'Canonical bounded relative path required')
    path = PurePosixPath(value)
    need(value != '.' and not path.is_absolute() and path.as_posix() == value
         and all(p not in ('', '.', '..') for p in path.parts), 'Path escape/alias refused')
    return value


def member_relative(name):
    need(type(name) is str and name.startswith(STAGE + '/'), 'Exact returned stage prefix required')
    return canonical(name[len(STAGE) + 1:])


def regular(path, *, maximum=MAX_BYTES, directory=False):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        info = part.lstat()
        wanted = stat.S_ISDIR(info.st_mode) if directory or part != path else stat.S_ISREG(info.st_mode)
        need(wanted and not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Ordinary unlinked path ancestry required')
        if part == path and not directory:
            need(info.st_nlink == 1 and 0 <= info.st_size <= maximum, 'Bounded single-link ordinary file required')
    need(path.resolve(strict=True) == path, 'Canonical path required')
    return path


def identity(path):
    state = path.stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


def filepin(path):
    path = regular(path)
    before = identity(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    need(identity(path) == before, 'File changed while hashing')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def retain_path(name):
    canonical(name)
    allowed = name == 'protocol.json' or name in RUNTIME or name.startswith(RUN)
    need(not (allowed and name.lower().endswith(FORBIDDEN_SUFFIX)), 'Executable in data retention namespace refused')
    return allowed


class SingleGzip:
    """Single CRC/ISIZE-checked gzip member, bounded chunks and expansion."""
    def __init__(self, stream, guard=lambda: None):
        self.stream, self.guard = stream, guard
        self.decoder = zlib.decompressobj(31)
        self.compressed = b''
        self.buffer = bytearray()
        self.tail = bytearray()
        self.expanded = 0
        self.done = False
    def read(self, size):
        need(type(size) is int and 0 <= size <= 1024**2, 'Bounded decompressor read required')
        while len(self.buffer) < size and not self.done:
            self.guard()
            chunk = self.compressed or self.stream.read(65536)
            need(bool(chunk), 'Truncated gzip frame')
            value = self.decoder.decompress(chunk, 65536)
            self.compressed = self.decoder.unconsumed_tail
            self.expanded += len(value)
            self.tail.extend(value)
            if len(self.tail) > 65536: del self.tail[:-65536]
            need(self.expanded <= MAX_EXPANDED, 'Finite expanded archive cap exceeded')
            self.buffer.extend(value)
            if self.decoder.eof:
                need(not self.decoder.unused_data and not self.compressed and self.stream.read(1) == b'',
                     'Concatenated gzip/trailing compressed bytes refused')
                self.done = True
        out = bytes(self.buffer[:size]); del self.buffer[:size]
        return out


def inspect_stream(raw, protocol, *, retained_root=None, guard=lambda: None):
    """Hash every compressed/member byte; retain only reviewed data namespaces."""
    class Hashed:
        count = 0
        digest = None
        def __init__(self): self.digest = hashlib.sha256()
        def read(self, size):
            guard(); value = raw.read(size)
            self.count += len(value); self.digest.update(value)
            need(self.count <= MAX_BYTES, 'Finite compressed archive cap exceeded')
            return value
    hashed = Hashed(); expanded = SingleGzip(hashed, guard)
    static = {name: row for name, row in protocol['files'].items() if not name.startswith('weights/')}
    index, metadata, seen_casefold = {}, {}, set()
    total, retained_bytes, last_end = 0, 0, 0
    if retained_root is not None:
        retained_root = Path(retained_root).absolute()
        need(not retained_root.exists(), 'Prior retained namespace immutable')
        retained_root.mkdir(parents=True)
        regular(retained_root, directory=True)
    with tarfile.open(fileobj=expanded, mode='r|') as archive:
        for member in archive:
            guard(); name = member_relative(member.name)
            need(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                 and not member.linkname and not member.sparse and
                 type(member.size) is int and 0 <= member.size <= MAX_MEMBER,
                 'Bounded ordinary non-sparse regular member required')
            need(set(member.pax_headers) <= {'path', 'mtime', 'atime', 'ctime'} and
                 ('path' not in member.pax_headers or member.pax_headers['path'] == member.name),
                 'Unexpected PAX metadata refused')
            need(name not in index and name.casefold() not in seen_casefold and len(index) < MAX_FILES,
                 'Duplicate/case-alias or excessive member count refused')
            need(name in static or name == 'protocol.json' or name in RUNTIME or name.startswith(RUN),
                 'Unexpected returned namespace: ' + name)
            need(not name.startswith('weights/'), 'Return must omit all initializer weight files')
            total += member.size
            need(total <= MAX_BYTES, 'Finite returned payload cap exceeded')
            kept = retain_path(name)
            destination = None
            if kept and retained_root is not None:
                path = retained_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                regular(path.parent, directory=True)
                need(path.is_relative_to(retained_root), 'Retention path escaped exact root')
                destination = path.open('xb')
            meta = kept and name.endswith(('.json', '.jsonl', '.log'))
            if meta: need(member.size <= MAX_METADATA, 'Bounded per-file retained metadata required')
            digest, size, pieces = hashlib.sha256(), 0, []
            source = archive.extractfile(member)
            need(source is not None, 'Regular member payload unavailable')
            try:
                for block in iter(lambda: source.read(1024**2), b''):
                    guard(); digest.update(block); size += len(block)
                    if destination is not None: destination.write(block)
                    if meta: pieces.append(block)
                if destination is not None:
                    destination.flush(); os.fsync(destination.fileno())
            finally:
                if destination is not None: destination.close()
            need(size == member.size, 'Truncated member payload')
            row = {'bytes': size, 'sha256': digest.hexdigest()}
            need(name not in static or row == static[name], 'Returned sealed static bytes differ: ' + name)
            if name == 'protocol.json': need(row['sha256'] == PROTOCOL_SHA, 'Returned protocol differs')
            index[name] = row; seen_casefold.add(name.casefold())
            if kept: retained_bytes += size
            if meta:
                value = b''.join(pieces)
                if name.endswith('.json'): metadata[name] = strict_json(value)
                elif name.endswith('.jsonl'):
                    need(not value or value.endswith(b'\n'), 'Partial JSONL line refused')
                    metadata[name] = [strict_json(line) for line in value.splitlines()]
                else: metadata[name] = value.decode('utf-8', errors='replace')
            last_end = ((member.offset_data + member.size + 511) // 512) * 512
            if retained_root is not None and len(index) % 500 == 0:
                print(json.dumps({'members_read': len(index), 'payload_bytes': total}), flush=True)
    while expanded.read(65536): guard()
    padding = expanded.expanded - last_end
    need(expanded.done and 1024 <= padding <= 65536 and padding % 512 == 0 and
         len(expanded.tail) >= padding and not any(expanded.tail[-padding:]),
         'Complete zero tar EOF padding required; hidden suffix refused')
    need(set(static) <= set(index) and 'protocol.json' in index, 'Required static payload missing')
    return index, metadata, {'compressed_bytes': hashed.count, 'compressed_sha256': hashed.digest.hexdigest(),
        'payload_bytes': total, 'retained_bytes': retained_bytes, 'expanded_bytes': expanded.expanded,
        'tar_zero_EOF_padding_bytes': padding, 'single_complete_gzip_CRC_ISIZE_verified': True,
        'returned_static_files_exact': len(static), 'all_payloads_hashed': True}


def direct_joins(metadata, index):
    joins = 0
    def visit(value):
        nonlocal joins
        if type(value) is dict:
            if set(value) == {'path', 'bytes', 'sha256'}:
                name = canonical(value['path'])
                need(type(value['bytes']) is int and value['bytes'] >= 0 and
                     type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
                     'Exact saved artifact binding required')
                # Original initializer files are deliberately omitted from returns.
                if name.startswith('weights/'):
                    return
                need(index.get(name) == {k: value[k] for k in ('bytes', 'sha256')},
                     'Saved runtime artifact binding differs: ' + name)
                joins += 1
            else:
                for child in value.values(): visit(child)
        elif type(value) is list:
            for child in value: visit(child)
    for name, value in metadata.items():
        if name.startswith(RUN) and name != RUN + 'operation_plan.json': visit(value)
    return joins


def trusted_helpers(protocol):
    required = [name for name in protocol['files'] if name.startswith(('research/', 'third_party/')) and name.endswith('.py')]
    for name in required:
        # Manifest authenticates the executable source held in the local sealed stage.
        actual = filepin(LOCAL / name)
        need({k: actual[k] for k in ('bytes', 'sha256')} == protocol['files'][name], 'Trusted local helper changed')
    need('research' not in sys.modules, 'Fresh trusted local helper namespace required')
    sys.path.insert(0, str(LOCAL))
    package = 'research.codeformer_matched_feature_learning_v1'
    execution = importlib.import_module(package + '.execution')
    contract = importlib.import_module(package + '.contract')
    for module, name in ((execution, 'execution'), (contract, 'contract')):
        need(Path(module.__file__).resolve() == LOCAL / 'research/codeformer_matched_feature_learning_v1' / (name + '.py'),
             'Trusted helper import origin differs')
    selection = contract.select_cases((LOCAL / protocol['admission']['path']).read_bytes(),
        (LOCAL / protocol['candidate_data']['path']).read_bytes(), (LOCAL / protocol['parent_protocol']['path']).read_bytes())
    need(selection == protocol['selection'], 'Exact source-pinned admission selection required')
    return execution, contract, selection


def validate_return(protocol, index, metadata, export, journal_raw):
    execution, contract, selection = trusted_helpers(protocol)
    replay = execution.audit_journal(journal_raw, selection)
    need(replay['bookkeeping_complete'] is True and replay['status'] == 'FINISHED' and replay['pending'] is None,
         'Completed known-outcome journal required')
    saved_plan = metadata[RUN + 'operation_plan.json']
    need(saved_plan['complete'] is True and saved_plan['calls'] == contract.operation_plan(selection)
         and saved_plan['selection'] == selection, 'Actual operation plan/selection differs')
    result, status = metadata[RUN + 'result.json'], metadata[RUN + 'status.json']
    need(result['protocol_sha256'] == status['protocol_sha256'] == PROTOCOL_SHA and
         result['complete'] is True and status['complete'] is True and
         result['model_qualified'] is status['model_qualified'] is False,
         'Finite result/status protocol and qualification boundaries differ')
    for value in (result['execution'], status):
        for field in ('counters', 'per_arm', 'pending', 'quality_stops', 'committed_updates',
                      'journal_records', 'journal_sha256', 'journal_bytes', 'returned_forward_slots', 'skipped_forward_slots'):
            need(value[field] == replay[field], 'Returned terminal journal field differs: ' + field)
    need(result['counters'] == replay['counters'] == export['confirmed_counters'], 'Export/result/replay counters differ')
    worker = metadata['supervisor.worker.json']
    need(worker['complete'] is True and worker['spawn_returned'] is True and
         type(worker['exit_code']) is int and worker['exit_code'] == result['worker_exit'] == export['worker_exit'] and
         worker['timeout'] is False and worker['cleanup']['child_reaped'] is True and
         worker['cleanup']['process_group_absent'] is True and not worker.get('orphan_cleanup_unknown', False),
         'Known worker exit/process-group cleanup required')
    need(export['protocol_sha256'] == PROTOCOL_SHA and export['status_receipt'] == {'path': RUN + 'status.json', **index[RUN + 'status.json']} and
         export['journal_receipt'] == {'path': RUN + 'operations.jsonl', **index[RUN + 'operations.jsonl']} and
         export['operation_outcome_unknown'] is False and export['journal_full_replay'] is True,
         'Export actual receipt joins differ')
    stops = replay['quality_stops']
    need(result['quality_stop'] is bool(stops) and worker['exit_code'] == (2 if stops else 0), 'Known scientific stop/exit must agree')
    for arm, stop in stops.items():
        actual = result['quality_stops'][arm]
        receipt = actual['gate_receipt']
        need(actual['update'] == stop['checkpoint'] == replay['committed_updates'][arm] and
             receipt['sha256'] == stop['receipt_sha256'] and actual['reason'] == stop['reason'] and
             index[receipt['path']] == {k: receipt[k] for k in ('bytes', 'sha256')}, 'Quality-stop exact receipt join differs')
        bank = metadata[receipt['path']]
        need(bank['gate']['pass'] is False and bank['gate']['update'] == stop['checkpoint'],
             'Quality stop requires failed saved gate (pixel recomputation remains separate)')
    need(export['payload_files'] == len(index) and export['payload_bytes'] == sum(row['bytes'] for row in index.values()),
         'Actual export inventory differs')
    unknown_final = [name for name in ('supervisor.export.json', 'supervisor.json', 'export.json') if name not in index]
    return {'worker_exit': worker['exit_code'], 'worker_seconds': worker['seconds'],
        'worker_timeout': worker['timeout'], 'worker_cleanup': worker['cleanup'],
        'journal_replay': replay, 'actual_saved_result_quality_stop': result['quality_stop'],
        'actual_gate_records_quality_recomputed': False, 'direct_runtime_binding_joins': direct_joins(metadata, index),
        'final_export_supervisor_receipts_absent': unknown_final,
        'final_export_process_exit_independently_established': False}


def write_new(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    need(path.read_bytes() == raw, 'New receipt readback differs')


def audit():
    began = time.monotonic()
    def guard(): need(time.monotonic() - began < MAX_SECONDS, 'Finite600s transport deadline')
    need(not OUT.exists(), 'Prior audit namespace immutable')
    archive = regular(ROOT / 'outputs' / (PREFIX + '-results.tar.gz'))
    checksum = regular(Path(str(archive) + '.sha256'))
    export_path = regular(ROOT / 'outputs' / (PREFIX + '-export.json'))
    protocol_raw = regular(LOCAL / 'protocol.json').read_bytes()
    need(sha(protocol_raw) == PROTOCOL_SHA, 'Exact retained sealed protocol required')
    protocol = strict_json(protocol_raw)
    exported_raw = export_path.read_bytes(); exported = strict_json(exported_raw)
    need(exported['complete'] is True and exported['protocol_sha256'] == PROTOCOL_SHA and
         type(exported['bytes']) is int and archive.stat().st_size == exported['bytes'],
         'Downloaded archive/export protocol size mismatch')
    expected = exported['archive_sha256']
    need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'Expected exact archive digest required')
    need(checksum.read_bytes() == (expected + '  ' + archive.name + '\n').encode('ascii'),
         'Exact downloaded LF checksum sidecar required')
    before = {name: identity(path) for name, path in (('archive', archive), ('checksum', checksum), ('export', export_path))}
    OUT.mkdir()
    retained = OUT / 'retained' / STAGE
    with archive.open('rb') as stream:
        index, metadata, framing = inspect_stream(stream, protocol, retained_root=retained, guard=guard)
    need(framing['compressed_sha256'] == expected and framing['compressed_bytes'] == exported['bytes'],
         'Actual full compressed archive hash differs')
    print(json.dumps({'archive_complete': True, 'members': len(index), 'retained_root': str(retained)}), flush=True)
    write_new(OUT / 'member_index.json', {'complete': True, 'archive_sha256': expected, 'members': index, 'framing': framing})
    journal = (retained / RUN / 'operations.jsonl').read_bytes()
    semantic = validate_return(protocol, index, metadata, exported, journal)
    need(before == {name: identity(path) for name, path in (('archive', archive), ('checksum', checksum), ('export', export_path))},
         'Downloaded input changed during audit')
    for name, row in index.items():
        if retain_path(name):
            actual = filepin(retained / name)
            need({k: actual[k] for k in ('bytes', 'sha256')} == row, 'Retained data readback differs')
    result = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'INDEPENDENT_TRANSPORT_AND_JOURNAL_AUDIT_NOT_MODEL_QUALIFICATION',
        'archive': {'path': archive.relative_to(ROOT).as_posix(), 'bytes': framing['compressed_bytes'], 'sha256': expected},
        'checksum': filepin(checksum), 'export': filepin(export_path),
        'protocol': {'bytes': len(protocol_raw), 'sha256': PROTOCOL_SHA},
        'member_index': filepin(OUT / 'member_index.json'), 'retained_root': retained.relative_to(ROOT).as_posix(),
        'members': len(index), 'framing': framing, **semantic,
        'seconds': time.monotonic() - began, 'returned_sources_imported_or_executed': False,
        'checkpoint_or_array_decoding': False, 'neural_gradient_optimizer_training_VM_browser_calls': 0,
        'model_qualified': False}
    write_new(OUT / 'transport.json', result)
    print(json.dumps({'complete': True, 'members': len(index), 'worker_exit': semantic['worker_exit'],
        'committed_updates': semantic['journal_replay']['committed_updates'], 'seconds': result['seconds']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    need(args.audit, 'Explicit --audit required')
    audit()


if __name__ == '__main__':
    main()

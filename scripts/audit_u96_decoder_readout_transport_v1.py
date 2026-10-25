"""Independent return transport/operation audit; no model or framework import.

Streaming single-frame gzip, bounded tar payloads and strict JSON. Never extract
untrusted paths, execute returned sources, import Torch or resume a worker.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
STAGE_NAME = 'cctv_codeformer_u96_decoder_readout_v1_vm'
RUN = 'outputs/u96_decoder_readout_v1/'
PROTOCOL_SHA = 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
ARCHIVE_SHA = '73c8f4bcca5814cb845058cbc7d553ad3ea1f29e97967449cad66ac848fed2c4'
MAX_BYTES = 2 * 1024**3
MAX_EXPANDED = MAX_BYTES + 64 * 1024**2
MAX_MEMBER = 128 * 1024**2
MAX_JSON = 24 * 1024**2
MAX_FILES = 8000
MAX_SECONDS = 240
EXPECTED_COUNTERS = {
    'full_forward_attempts': 712, 'full_forward_returns': 712,
    'reference_forwards': 120, 'candidate_snapshot_forwards': 336,
    'candidate_training_forwards': 256, 'gradient_queries': 0,
    'backward_attempts': 256, 'backward_returns': 256,
    'optimizer_constructions': 1, 'optimizer_construction_returns': 1,
    'optimizer_update_attempts': 64, 'optimizer_updates': 64,
    'controlled_hook_exception_probes': 1,
    'frozen_U96_adapter_site_reservations': 2848, 'frozen_U96_adapter_site_returns': 2848,
    'existing_RGB_readout_reservations': 592, 'existing_RGB_readout_returns': 592,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path, maximum=MAX_BYTES):
    path = Path(path).absolute()
    for parent in path.parents:
        info = parent.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary ancestry required')
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_nlink == 1
         and not getattr(info, 'st_file_attributes', 0) & 1024 and 0 < info.st_size <= maximum,
         'Bounded ordinary single-link input required')
    return path


def filepin(path):
    path = regular(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    after = path.stat()
    need((before.st_ino, before.st_mtime_ns, before.st_size) ==
         (after.st_ino, after.st_mtime_ns, after.st_size), 'Input changed while hashing')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': after.st_size, 'sha256': digest.hexdigest()}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON constant')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(item):
        if type(item) is float:
            need(math.isfinite(item), 'Nonfinite exponent JSON')
        elif type(item) is dict:
            for child in item.values():
                finite(child)
        elif type(item) is list:
            for child in item:
                finite(child)
    finite(value)
    return value


def member_relative(name):
    need(type(name) is str and name.startswith(STAGE_NAME + '/'), 'Exact stage prefix required')
    relative = name[len(STAGE_NAME) + 1:]
    path = PurePosixPath(relative)
    need(relative and len(relative) <= 512 and not path.is_absolute() and '\\' not in relative
         and ':' not in relative and path.as_posix() == relative
         and all(part not in ('', '.', '..') for part in path.parts), 'Canonical member path required')
    return relative


class SingleGzip:
    """Bounded decompression; CRC/ISIZE handled by zlib; reject concatenation."""
    def __init__(self, stream, guard=lambda: None):
        self.stream = stream
        self.guard = guard
        self.decoder = zlib.decompressobj(31)
        self.buffer = bytearray()
        self.compressed = b''
        self.expanded = 0
        self.final_tail = bytearray()
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
            self.final_tail.extend(value)
            if len(self.final_tail) > 65536:
                del self.final_tail[:-65536]
            need(self.expanded <= MAX_EXPANDED, 'Expanded return cap')
            self.buffer.extend(value)
            if self.decoder.eof:
                need(not self.decoder.unused_data and not self.compressed and self.stream.read(1) == b'',
                     'Extra gzip frame or trailing compressed bytes')
                self.done = True
        value = bytes(self.buffer[:size])
        del self.buffer[:size]
        return value


def verify_tar_end(stream, last_member_end):
    """Check padding from complete decompressed bytes, including tarfile buffers."""
    padding = stream.expanded - last_member_end
    need(stream.done and type(last_member_end) is int and last_member_end >= 0
         and 1024 <= padding <= 65536 and padding % 512 == 0,
         'Complete bounded tar EOF padding required')
    need(len(stream.final_tail) >= padding and not any(stream.final_tail[-padding:]),
         'Nonzero hidden tar payload after final member')
    return padding


def event_readback(rows):
    """Reconstruct reservations/returns from event order rather than final counts."""
    counters = {key: 0 for key in EXPECTED_COUNTERS}
    transitions = Counter()
    unknown = {'forward': None, 'backward': None, 'optimizer': None, 'construction': None}
    last_seconds = -1.
    expected_commits = 0
    for row in rows:
        need(type(row) is dict and type(row.get('kind')) is str, 'Explicit event kind required')
        seconds = row['seconds']
        need(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= last_seconds,
             'Monotone event time required')
        last_seconds = seconds
        kind = row['kind']
        transitions[kind] += 1
        if kind == 'forward_reserved':
            need(unknown['forward'] is None, 'Forward reserved while prior outcome unknown')
            mode = row['forward_kind']
            need(mode in ('reference_forwards', 'candidate_snapshot_forwards', 'candidate_training_forwards'),
                 'Known forward mode required')
            unknown['forward'] = mode
            counters['full_forward_attempts'] += 1
            counters[mode] += 1
            counters['frozen_U96_adapter_site_reservations'] += 4
            if mode != 'reference_forwards':
                counters['existing_RGB_readout_reservations'] += 1
        elif kind == 'forward_returned':
            mode = unknown['forward']
            need(mode == row['forward_kind'] and mode is not None, 'Unreserved forward return')
            need(row['sites'] == ['32', '64', '128', '256'], 'Exact four adapter sites required')
            counters['full_forward_returns'] += 1
            counters['frozen_U96_adapter_site_returns'] += 4
            if mode != 'reference_forwards':
                counters['existing_RGB_readout_returns'] += 1
            unknown['forward'] = None
        elif kind == 'controlled_hook_probe_reserved':
            counters['controlled_hook_exception_probes'] += 1
        elif kind == 'optimizer_construction_reserved':
            need(unknown['construction'] is None and counters['optimizer_constructions'] == 0,
                 'Only one fresh optimizer construction')
            unknown['construction'] = True
            counters['optimizer_constructions'] += 1
        elif kind == 'optimizer_construction_returned':
            need(unknown['construction'] is True, 'Unreserved optimizer construction return')
            counters['optimizer_construction_returns'] += 1
            unknown['construction'] = None
        elif kind == 'backward_reserved':
            need(unknown['backward'] is None, 'Backward outcome unknown')
            unknown['backward'] = (row['update'], row['slot'], row['id'])
            counters['backward_attempts'] += 1
        elif kind == 'backward_returned':
            need(unknown['backward'] == (row['update'], row['slot'], row['id']), 'Unreserved backward return')
            counters['backward_returns'] += 1
            unknown['backward'] = None
        elif kind == 'optimizer_reserved':
            need(unknown['optimizer'] is None and row['update'] == expected_commits + 1,
                 'Optimizer reservations must be a consecutive prefix')
            need(counters['backward_returns'] == row['update'] * 4, 'Four backwards before every commit')
            unknown['optimizer'] = row['update']
            counters['optimizer_update_attempts'] += 1
        elif kind == 'optimizer_returned':
            need(unknown['optimizer'] == row['update'], 'Unreserved optimizer return')
            counters['optimizer_updates'] += 1
            expected_commits += 1
            unknown['optimizer'] = None
        need(row['counters'] == counters, 'Durable event counters differ at ' + kind)
        need(all(value <= EXPECTED_COUNTERS[key] for key, value in counters.items()), 'Declared call cap exceeded')
    return {'events': len(rows), 'counts_by_kind': dict(transitions), 'independently_reconstructed_counters': counters,
            'unknown_operations': unknown, 'last_event_seconds': last_seconds}


def audit(archive, expected_sha, out):
    start = time.monotonic()
    checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        need(ok, message)
    def guard():
        need(time.monotonic() - start < MAX_SECONDS, 'Finite240s return transport audit stop')
    check(not out.exists(), 'Audit namespace immutable')
    archive_pin = filepin(archive)
    check(expected_sha == ARCHIVE_SHA == archive_pin['sha256'], 'Exact returned archive SHA differs')
    checksum = regular(Path(str(archive) + '.sha256'), 1024)
    check(checksum.read_bytes() == (expected_sha + '  ' + archive.name + '\n').encode('ascii'),
          'Exact LF return checksum differs')
    export_path = archive.with_name('cctv-codeformer-u96-decoder-readout-v1-export.json')
    export = strict_json(regular(export_path, 65536).read_bytes())
    check(export['complete'] is True and export['archive_sha256'] == expected_sha
          and export['bytes'] == archive_pin['bytes'] and export['protocol_sha256'] == PROTOCOL_SHA,
          'Downloaded export/archive/protocol join differs')
    stage = ROOT / 'outputs' / STAGE_NAME
    protocol_raw = regular(stage / 'protocol.json', MAX_JSON).read_bytes()
    check(sha(protocol_raw) == PROTOCOL_SHA, 'Sealed local protocol changed')
    protocol = strict_json(protocol_raw)
    index = {}
    metadata = {}
    retained = {}
    payload_total = 0
    last_member_end = 0
    with archive.open('rb') as raw:
        stream = SingleGzip(raw, guard)
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            for member in tar:
                guard()
                relative = member_relative(member.name)
                last_member_end = ((member.offset_data + member.size + 511) // 512) * 512
                check(member.isfile() and not member.issym() and not member.islnk(), 'Regular return member required')
                check(relative not in index and len(index) < MAX_FILES, 'Unique bounded members required')
                check(type(member.size) is int and 0 <= member.size <= MAX_MEMBER, 'Bounded member size required')
                check(not relative.startswith('weights/'), 'Frozen initializer bytes must be omitted from return')
                payload_total += member.size
                check(payload_total <= MAX_BYTES, 'Bounded return payload required')
                source = tar.extractfile(member)
                digest = hashlib.sha256()
                keep = relative.endswith(('.json', '.jsonl')) or relative == 'run.log'
                if keep:
                    check(member.size <= MAX_JSON, 'Bounded metadata/log size required')
                chunks = []
                count = 0
                while True:
                    guard()
                    chunk = source.read(1024**2)
                    if not chunk:
                        break
                    digest.update(chunk)
                    count += len(chunk)
                    if keep:
                        chunks.append(chunk)
                check(count == member.size, 'Member payload extent differs')
                index[relative] = {'path': relative, 'bytes': count, 'sha256': digest.hexdigest()}
                if keep:
                    value = b''.join(chunks)
                    retained[relative] = value
                    if relative.endswith('.json'):
                        metadata[relative] = strict_json(value)
                    elif relative.endswith('.jsonl'):
                        check(not value or value.endswith(b'\n'), 'Incomplete JSONL line')
                        metadata[relative] = [strict_json(line) for line in value.splitlines()]
        # tarfile buffers complete record blocks internally. Explicitly drain
        # remaining gzip bytes to validate CRC and refuse nonzero hidden payload.
        while True:
            tail = stream.read(65536)
            if not tail:
                break
            check(not any(tail), 'Nonzero trailing tar bytes')
        check(stream.done, 'Complete single gzip frame required')
        padding = verify_tar_end(stream, last_member_end)
        check(padding >= 1024, 'Both tar EOF blocks required')
    check(index['protocol.json']['sha256'] == PROTOCOL_SHA, 'Returned protocol differs')
    static = {name: binding for name, binding in protocol['files'].items() if not name.startswith('weights/')}
    for name, binding in static.items():
        check(index.get(name) == {'path': name, **binding}, 'Returned sealed static file differs: ' + name)
    for name in index:
        check(name in static or name in ('protocol.json', 'run.log', 'supervisor.json', 'install.json')
              or name.startswith(RUN), 'Unexpected return namespace: ' + name)
    check(export['payload_files'] == len(index) and export['payload_bytes'] == payload_total,
          'Export payload counts differ')
    joins = 0
    def visit(value):
        nonlocal joins
        if type(value) is dict:
            if set(value) == {'path', 'bytes', 'sha256'}:
                name = value['path']
                if name in index:
                    check(value == index[name], 'Saved direct binding differs: ' + name)
                    joins += 1
            else:
                for child in value.values():
                    visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)
    for name, value in metadata.items():
        if name.startswith(RUN) or name in ('supervisor.json', 'install.json'):
            visit(value)
    failure = metadata[RUN + 'failure.json']
    status = metadata[RUN + 'status.json']
    supervisor = metadata['supervisor.json']
    for label, row in (('failure', failure), ('status', status)):
        check(row['protocol_sha256'] == PROTOCOL_SHA and row['counters'] == EXPECTED_COUNTERS,
              'Complete counters/protocol differ: ' + label)
        check(row['optimizer_outcome_unknown'] is False, 'Unexpected unknown optimizer outcome')
    check(failure['status'] == 'QUALITY_STOP' and failure['phase'] == 'snapshot_u64_quality_gates'
          and failure['active_update'] == 64 and failure['model_qualified'] is False,
          'Terminal quality-stop identity differs')
    check(supervisor['worker_exit'] == 2 and supervisor['timeout'] is False
          and supervisor['optimizer_updates'] == supervisor['optimizer_update_attempts'] == 64,
          'Supervisor does not establish expected finite quality stop')
    check(supervisor['seconds'] <= protocol['limits']['external_seconds'] + 20
          and failure['elapsed_seconds'] <= protocol['limits']['internal_seconds'], 'Worker wall cap exceeded')
    check(RUN + 'result.json' not in index and not any('status.pending' in name for name in index),
          'Expected quality stop has result or pending status')
    event_result = event_readback(metadata[RUN + 'events.jsonl'])
    check(event_result['independently_reconstructed_counters'] == EXPECTED_COUNTERS
          and all(value is None for value in event_result['unknown_operations'].values()),
          'Full event trajectory is not a complete reserved/returned prefix')
    resources = metadata[RUN + 'resources.jsonl']
    check(bool(resources), 'Actual resource samples required')
    maximums = {key: max(row[key] for row in resources) for key in
                ('current_RSS_bytes', 'peak_RSS_bytes', 'CUDA_reserved_bytes', 'CUDA_peak_reserved_bytes')}
    check(maximums['peak_RSS_bytes'] <= protocol['limits']['maximum_RSS_bytes']
          and maximums['CUDA_peak_reserved_bytes'] <= protocol['limits']['maximum_CUDA_reserved_bytes'],
          'Observed peak memory cap exceeded')
    check(min(row['free_disk_bytes'] for row in resources) >= protocol['limits']['continuing_disk_reserve'],
          'Observed disk reserve crossed')
    check(max(row['seconds'] for row in resources) <= protocol['limits']['internal_seconds'], 'Observed runtime cap crossed')
    # Archive integrity/receipts establish saved claims; model-state arithmetic,
    # all pixel/gate recomputation and visual judgment are separate audits.
    check(filepin(archive) == archive_pin, 'Return archive changed during audit')
    out.mkdir()
    selected = ('failure.json', 'status.json', 'started.json', 'inherited_DEV_failures.json',
                'controlled_hook_exception.json', 'caller_RNG_restoration.json')
    for name in selected:
        key = RUN + name
        if key in retained:
            (out / name).write_bytes(retained[key])
    result = {'complete': True, 'transport_and_event_audit_pass': True,
              'UTC': datetime.now(timezone.utc).isoformat(), 'checks': checks,
              'seconds': time.monotonic() - start, 'source': filepin(Path(__file__).resolve()),
              'archive': archive_pin, 'checksum': filepin(checksum), 'export': filepin(export_path),
              'protocol_sha256': PROTOCOL_SHA, 'payload_files': len(index), 'payload_bytes': payload_total,
              'expanded_gzip_bytes': stream.expanded, 'sealed_nonweight_files': len(static),
              'tar_EOF_padding_bytes_verified_including_tarfile_buffer': padding,
              'saved_direct_bindings_checked': joins, 'index': index,
              'terminal_status': 'QUALITY_STOP_AT64_NOT_QUALIFIED',
              'failure': index[RUN + 'failure.json'], 'worker_exit': supervisor['worker_exit'],
              'events': event_result, 'resource_sample_count': len(resources),
              'observed_memory_maximums': maximums,
              'minimum_sampled_free_disk_bytes': min(row['free_disk_bytes'] for row in resources),
              'actual_optimizer_states_and_image_gates_independently_recomputed_here': False,
              'all_visual_sheets_reviewed': False, 'model_qualified': False,
              'new_model_neural_gradient_optimizer_training_VM_calls': 0}
    target = out / 'independent_transport_event_audit_v1.json'
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('complete', 'transport_and_event_audit_pass', 'checks', 'seconds', 'terminal_status')}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT / 'outputs/cctv-codeformer-u96-decoder-readout-v1-results.tar.gz')
    parser.add_argument('--archive-sha', required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/codeformer_u96_decoder_readout_v1_return_transport_audit')
    args = parser.parse_args()
    audit(args.archive, args.archive_sha, args.out)


if __name__ == '__main__':
    main()

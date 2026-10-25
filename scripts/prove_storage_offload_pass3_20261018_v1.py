"""Phase3 inert bounded gzip/TAR member retention proof. Never extracts or loads tensors.
No network, VM or removal. Actual upload/apply waits for root VM-phase closure."""
import collections
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'outputs/storage_offload_20261018_v1/final_local_pass_provisional_v1/scope.json'
SCOPE_SHA256 = 'a65b85501a001fdada3415bac478bcef0ac29c994530d7cb9e01cc45f72e99de'
OUT = ROOT / 'outputs/storage_offload_20261018_v1/pass3/member_retention_v1'
CAP = 900.0
CHUNK = 1024 * 1024
PREFIXES = {'outputs/cctv-dgp-bank-comparison-v1-r2-B-results.tar.gz': 'cctv_dgp_bank_comparison_v1_r2_B_vm', 'outputs/cctv-dgp-bank-comparison-v1-results.tar.gz': 'cctv_dgp_bank_comparison_v1_vm', 'outputs/cctv-dgp-bank-comparison-v1-r3-B-results.tar.gz': 'cctv_dgp_bank_comparison_v1_r3_B_vm'}
ALLOWED_PAX_KEYS = frozenset({'mtime', 'path'})


def mapped_member_name(member, prefix):
    """Match the old tarfile exporter: exact NAME prefix; mtime/path PAX only."""
    if not member.isreg() or member.issym() or member.islnk():
        raise ValueError('Only ordinary regular TAR members admitted: ' + member.name)
    full = name_ok(member.name)
    pieces = full.split('/')
    if len(pieces) < 2 or pieces[0] != prefix:
        raise ValueError('Unexpected exact TAR root prefix: ' + full)
    if not set(member.pax_headers).issubset(ALLOWED_PAX_KEYS):
        raise ValueError('Unexpected PAX metadata key: ' + full)
    if 'path' in member.pax_headers and member.pax_headers['path'] != full:
        raise ValueError('PAX path differs from parsed canonical name')
    if 'mtime' in member.pax_headers:
        m = member.pax_headers['mtime']
        if not isinstance(m, str) or len(m) > 64 or not math.isfinite(float(m)) or abs(float(m)) > 1e11:
            raise ValueError('Unbounded or invalid PAX mtime')
    return name_ok('/'.join(pieces[1:]))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def full_retention_verified(rows):
    return len(rows) == 3 and all(r.get('complete') is True and
                                 r.get('full_member_retention_verified') is True for r in rows)


def dump_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def meta(path):
    s = path.lstat()
    return {'bytes': s.st_size, 'inode': s.st_ino, 'dev': s.st_dev,
            'nlink': s.st_nlink, 'mtime_ns': s.st_mtime_ns, 'mode': s.st_mode,
            'attributes': getattr(s, 'st_file_attributes', None)}


def no_links(path, regular=True):
    p = path
    while True:
        s = p.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Link/reparse path refused: ' + str(p))
        if p == ROOT:
            break
        if ROOT not in p.parents:
            raise ValueError('Path outside workspace: ' + str(p))
        p = p.parent
    if regular and not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Regular file required: ' + str(path))


def name_ok(name):
    p = PurePosixPath(name)
    if (not name or '\\' in name or '\x00' in name or ':' in name or
            name.startswith('/') or any(x in ('', '.', '..') for x in name.split('/')) or
            p.is_absolute()):
        raise ValueError('Nonportable archive member: ' + repr(name))
    return name


def classification(name):
    lower = name.lower()
    if any(s in lower for s in ('checkpoint', 'weights/', '.pth', '.pt', '.onnx')):
        return 'retained_checkpoint_or_weight'
    if any(s in lower for s in ('failure', 'gate', 'exit_code', 'traceback')):
        return 'retained_failure_or_gate_record'
    if any(s in lower for s in ('cache', 'split', 'manifest', 'protocol', 'provenance')):
        return 'retained_research_metadata_or_cache'
    if lower.endswith(('.json', '.jsonl', '.csv', '.log', '.txt', '.md', '.py', '.sh')):
        return 'retained_source_metric_or_record'
    return 'retained_other_opaque_payload'


class HashedReader:
    def __init__(self, f, check):
        self.f = f
        self.check = check
        self.hash = hashlib.sha256()
        self.count = 0

    def read(self, n=-1):
        self.check()
        if n < 0:
            n = CHUNK
        b = self.f.read(n)
        self.hash.update(b)
        self.count += len(b)
        self.check()
        return b

    def seekable(self):
        return False


def prove(row, index):
    start = time.monotonic()
    deadline = start + CAP
    last_print = [start]
    current = [None]
    completed = 0
    member_bytes = 0
    compressed = None
    classes = collections.Counter()
    pax_keys = collections.Counter()
    seen = set()
    ledger_path = OUT / ('members_%02d.jsonl' % index)
    archive_path = ROOT / row['archive']
    retained = ROOT / row['unpacked_return_root']
    before = meta(archive_path)
    result = {'archive_path': row['archive'], 'archive_bytes': before['bytes'],
              'metadata_before': before, 'retained_root': row['unpacked_return_root'],
              'exact_archive_root_prefix': PREFIXES[row['archive']], 'allowed_PAX_keys': sorted(ALLOWED_PAX_KEYS),
              'cap_seconds': CAP, 'started_utc': utc(), 'complete': False,
              'full_member_retention_verified': False, 'archive_sha256': None}

    def check():
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError('900s per-archive proof limit reached; no retry')
        if now - last_print[0] >= 30:
            print(json.dumps({'archive_index': index, 'verified_members': completed,
                              'member_bytes': member_bytes,
                              'compressed_bytes_read': compressed.count if compressed else 0,
                              'seconds': round(now-start, 3), 'current_member': current[0]}), flush=True)
            last_print[0] = now

    try:
        check()
        no_links(archive_path)
        no_links(retained, regular=False)
        if before != row['fresh_Path_lstat']:
            raise ValueError('Archive metadata changed since exact scope inventory')
        import_path = ROOT / row['verified_import_receipt']
        no_links(import_path)
        import_bytes = import_path.read_bytes()
        receipt = json.loads(import_bytes)
        manifest_path = retained / 'export_manifest.json'
        no_links(manifest_path)
        manifest_before = meta(manifest_path)
        manifest_bytes = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_hash != receipt['export_manifest_sha256']:
            raise ValueError('Current retained manifest differs from verified import')
        if meta(manifest_path) != manifest_before:
            raise ValueError('Manifest changed while reading')
        manifest = json.loads(manifest_bytes)
        expected = dict(manifest['files'])
        expected['export_manifest.json'] = manifest_hash
        if len(expected) != receipt['returned_file_bindings']:
            raise ValueError('Expected member count differs from import receipt')
        if receipt['archive_bytes'] != before['bytes']:
            raise ValueError('Archive byte size differs from import receipt')
        if manifest['protocol_sha256'] != receipt['protocol_sha256']:
            raise ValueError('Retained protocol binding differs')
        casefolded = set()
        for name, h in expected.items():
            name_ok(name)
            if name.casefold() in casefolded or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('Expected namespace collision or invalid SHA')
            casefolded.add(name.casefold())
        result.update({'expected_archive_sha256': receipt['archive_sha256'],
                       'protocol_sha256': receipt['protocol_sha256'],
                       'expected_member_count': len(expected),
                       'expected_uncompressed_bytes': receipt['uncompressed_bytes'],
                       'import_receipt': {'path': row['verified_import_receipt'],
                         'sha256': hashlib.sha256(import_bytes).hexdigest(), 'bytes': len(import_bytes)},
                       'retained_manifest': {'path': str(manifest_path.relative_to(ROOT)).replace('\\', '/'),
                         'sha256': manifest_hash, 'bytes': len(manifest_bytes)}})
        with ledger_path.open('x', encoding='utf-8', newline='\n') as ledger:
            with archive_path.open('rb') as raw:
                compressed = HashedReader(raw, check)
                with gzip.GzipFile(fileobj=compressed, mode='rb') as decoded:
                    with tarfile.open(fileobj=decoded, mode='r|', bufsize=CHUNK) as archive:
                        for member in archive:
                            check()
                            current[0] = member.name
                            name = mapped_member_name(member, PREFIXES[row['archive']])
                            pax_keys.update(member.pax_headers.keys())
                            if name in seen or name not in expected:
                                raise ValueError('Duplicate or unexpected TAR member: ' + name)
                            local_path = retained / Path(*PurePosixPath(name).parts)
                            no_links(local_path)
                            local_before = meta(local_path)
                            if member.size != local_before['bytes']:
                                raise ValueError('TAR/local member byte count differs: ' + name)
                            th, lh = hashlib.sha256(), hashlib.sha256()
                            received = 0
                            with archive.extractfile(member) as payload, local_path.open('rb') as local:
                                while received < member.size:
                                    check()
                                    block = payload.read(min(CHUNK, member.size-received))
                                    if not block:
                                        raise ValueError('Truncated TAR payload: ' + name)
                                    local_block = local.read(len(block))
                                    if len(local_block) != len(block):
                                        raise ValueError('Truncated retained local file: ' + name)
                                    th.update(block)
                                    lh.update(local_block)
                                    received += len(block)
                                    check()
                                if local.read(1):
                                    raise ValueError('Retained local file grew: ' + name)
                            local_after = meta(local_path)
                            tar_hash, local_hash = th.hexdigest(), lh.hexdigest()
                            if local_after != local_before or tar_hash != local_hash or local_hash != expected[name]:
                                raise ValueError('Retained member identity/hash mismatch: ' + name)
                            kind = classification(name)
                            ledger.write(json.dumps({'archive_member': member.name, 'member': name, 'bytes': member.size,
                                'archive_payload_sha256': tar_hash, 'retained_local_sha256': local_hash,
                                'expected_manifest_sha256': expected[name],
                                'retained_path': str(local_path.relative_to(ROOT)).replace('\\', '/'),
                                'retained_metadata_after': local_after, 'classification': kind},
                                separators=(',', ':'), sort_keys=True) + '\n')
                            classes[kind] += 1
                            seen.add(name)
                            completed += 1
                            member_bytes += member.size
                            if completed % 64 == 0:
                                ledger.flush()
                                os.fsync(ledger.fileno())
                    # Drain gzip through its trailer/CRC. TAR trailing decoded bytes must be zero padding.
                    padding = 0
                    while True:
                        check()
                        b = decoded.read(CHUNK)
                        if not b:
                            break
                        padding += len(b)
                        if padding > 32 * CHUNK or any(b):
                            raise ValueError('Unexpected nonzero or excessive decoded TAR trailing bytes')
                check()
                if raw.read(1):
                    raise ValueError('Compressed archive not fully consumed')
            ledger.flush()
            os.fsync(ledger.fileno())
        if set(expected) != seen:
            raise ValueError('Missing expected TAR members')
        if member_bytes != receipt['uncompressed_bytes']:
            raise ValueError('Uncompressed byte total differs from import receipt')
        if compressed.count != before['bytes'] or compressed.hash.hexdigest() != receipt['archive_sha256']:
            raise ValueError('Full physical archive hash differs from import receipt')
        after = meta(archive_path)
        if after != before:
            raise ValueError('Archive metadata changed during proof')
        check()
        result.update({'complete': True, 'full_member_retention_verified': True,
                       'archive_sha256': compressed.hash.hexdigest(), 'metadata_after': after,
                       'all_archive_members_retained_locally': True,
                       'no_Drive_only_member_dependency': True})
    except Exception as e:
        result.update({'error': type(e).__name__ + ': ' + str(e),
                       'current_member_at_stop': current[0],
                       'metadata_after': meta(archive_path),
                       'partial_archive_sha256': compressed.hash.hexdigest() if compressed else None})
    result.update({'member_count': completed, 'uncompressed_bytes': member_bytes,
                   'compressed_bytes_read': compressed.count if compressed else 0,
                   'classifications': dict(classes), 'observed_PAX_keys': dict(pax_keys), 'seconds': time.monotonic()-start,
                   'finished_utc': utc(), 'no_extraction_or_tensor_loading': True,
                   'files_deleted': 0, 'VM_calls': 0, 'network_calls': 0})
    if ledger_path.exists():
        ledger_hash = hashlib.sha256()
        with ledger_path.open('rb') as f:
            for b in iter(lambda: f.read(CHUNK), b''):
                ledger_hash.update(b)
        result['per_member_ledger'] = {'path': str(ledger_path.relative_to(ROOT)).replace('\\', '/'),
                                     'bytes': ledger_path.stat().st_size, 'sha256': ledger_hash.hexdigest()}
    dump_new(OUT / ('archive_%02d_proof.json' % index), result)
    print(json.dumps({'archive_index': index, 'complete': result['complete'],
                      'member_count': completed, 'seconds': round(result['seconds'], 3),
                      'error': result.get('error')}), flush=True)
    return result


def main():
    scope_bytes = SCOPE.read_bytes()
    if hashlib.sha256(scope_bytes).hexdigest() != SCOPE_SHA256:
        raise ValueError('Exact prospectively reviewed metadata scope SHA required')
    scope = json.loads(scope_bytes)
    if not scope['complete'] or len(scope['candidates']) != 3 or scope['candidate_compressed_logical_bytes'] != 6661664658:
        raise ValueError('Exactly three complete frozen pass3 scope rows required')
    rows_to_check = []
    for item in scope['candidates']:
        metadata = dict(item['archive_metadata'])
        metadata['dev'] = metadata.pop('device')
        rows_to_check.append({'archive': item['archive_path'], 'fresh_Path_lstat': metadata,
            'unpacked_return_root': item['retained_root'],
            'verified_import_receipt': item['metadata_pins']['import_receipt']['path']})
    if {r['archive'] for r in rows_to_check} != set(PREFIXES):
        raise ValueError('Exact pass3 archive names required')
    OUT.mkdir(exist_ok=False)
    begin = time.monotonic()
    rows = []
    for i, row in enumerate(rows_to_check, 1):
        result = prove(row, i)
        rows.append(result)
        if not result['complete']:
            break
    report = {'format': 'pass3-inert-full-member-local-retention-proof-v1',
              'complete': len(rows) == 3 and all(r['complete'] for r in rows),
              'full_member_retention_verified': full_retention_verified(rows),
              'archives': rows,
              'scope': {'path': str(SCOPE.relative_to(ROOT)).replace('\\', '/'),
                        'sha256': hashlib.sha256(scope_bytes).hexdigest(), 'bytes': len(scope_bytes)},
              'script': {'path': str(Path(__file__).resolve().relative_to(ROOT)).replace('\\', '/'),
                         'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'all_3_full_member_retention_required_before_deletion': True,
              'no_automatic_retries': True, 'prior_pass1_sources_and_failures_preserved': True, 'cap_seconds_per_archive': CAP,
              'seconds': time.monotonic()-begin, 'UTC': utc(),
              'no_extraction_or_tensor_loading': True,
              'files_deleted': 0, 'VM_calls': 0, 'network_calls': 0}
    dump_new(OUT / 'retention_proof.json', report)
    print(json.dumps({'complete': report['complete'], 'archives_checked': len(rows),
                      'receipt': str((OUT/'retention_proof.json').relative_to(ROOT)).replace('\\', '/')}), flush=True)
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    sys.exit(main())

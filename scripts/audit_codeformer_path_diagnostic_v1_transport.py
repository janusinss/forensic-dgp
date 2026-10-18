"""Strict stdlib-only return transport audit. Importing performs no I/O.

No returned Python is imported, no checkpoint is loaded, and no neural,
gradient, optimizer, VM or network calls occur. Transport is not qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time

PROTOCOL_SHA = '2d3dcd1dbc3be371de5e10ed9c059e455ce9cc8376bac99f04a1a514acc0d49f'
ARCHIVE_SHA = 'c8ebadef8286ecd56c3080f7ec1311ef209fd3783072a542d46319bdca84b772'
ARCHIVE_BYTES = 79019264
MANIFEST_SHA = 'be4be94c889b38a0d08346cf9b0c386e889168b9a34d8392a4d25b36dd3b8ea4'
PREFIX = 'cctv-codeformer-path-diagnostic-v1'
OUTPUTS = 'outputs/path_diagnostic_v1'
MANIFEST = OUTPUTS + '/export_manifest.json'
RESULT_CAP = 256 * 1024**2
ARCHIVE_CAP = 272 * 1024**2
MAX_MEMBERS = 4096
MAX_JSON = 4 * 1024**2
RESERVE = 512 * 1024**2
SHA = re.compile(r'^[0-9a-f]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)', re.I)
SUPERVISION = {'outputs/supervision/' + x for x in ('preflight.log', 'diagnostic.log', 'run_receipt.json')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(name):
    require(isinstance(name, str) and name, 'Empty member path')
    p = PurePosixPath(name)
    require(p.parts and not p.is_absolute() and p.as_posix() == name and
            all(x not in ('', '.', '..') for x in p.parts), 'Noncanonical member path')
    require(not any(x in name for x in '\\:"<>|?*') and all(
        not DEVICE.match(x) and not x.endswith((' ', '.')) and
        len(x.encode('utf-16-le')) <= 510 and all(ord(c) >= 32 for c in x)
        for x in p.parts), 'Nonportable member path')
    require(p.suffix.casefold() not in {'.pt', '.pth'}, 'Checkpoint in diagnostic return')
    return name


def decode_json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    require(len(data) <= MAX_JSON, 'JSON exceeds finite limit')
    return json.loads(data.decode('utf-8-sig'), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Symlink/reparse path: ' + str(item))
    return path


def regular(path):
    info = no_links(path).lstat()
    require(stat.S_ISREG(info.st_mode), 'Expected regular file: ' + str(path))
    return info


def identity(info):
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev


def digest_file(path, tick=lambda: None):
    before = regular(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            tick(); h.update(block)
    require(identity(regular(path)) == identity(before), 'Local file changed while hashing')
    return h.hexdigest()


def json_file(path):
    info = regular(path)
    require(info.st_size <= MAX_JSON, 'Oversized JSON')
    data = Path(path).read_bytes()
    require(len(data) == info.st_size and identity(regular(path)) == identity(info), 'JSON changed')
    return decode_json(data)


def write_new(path, obj):
    with no_links(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def packet_contract(packet, expected_sha=PROTOCOL_SHA, tick=lambda: None):
    payload = (Path(packet) / 'protocol.json').read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_sha and
            digest_file(Path(packet) / 'protocol.json', tick) == expected_sha, 'Pinned protocol mismatch')
    p = decode_json(payload)
    require(p['format'] == 'codeformer-path-diagnostic-finite-manual-v1' and
            p['outputs'] == OUTPUTS and p['export_prefix'] == PREFIX and
            type(p['optimizer_updates']) is int and p['optimizer_updates'] == 0 and p['no_automatic_training'] is True,
            'Diagnostic protocol identity differs')
    require(p['limits']['result_bytes'] == RESULT_CAP and p['limits']['archive_bytes'] == ARCHIVE_CAP,
            'Protocol storage limits differ')
    files = p['files_sha256']
    require(isinstance(files, dict) and files, 'Missing packet hashes')
    casefold = set()
    for name, sha in files.items():
        canonical(name); require(SHA.fullmatch(sha), 'Invalid packet digest')
        require(name.casefold() not in casefold, 'Packet case collision')
        casefold.add(name.casefold())
        require(digest_file(Path(packet) / name, tick) == sha, 'Packet asset mismatch: ' + name)
    exports = p['export_files']
    require(isinstance(exports, list) and len(exports) == len(set(exports)) and
            set(exports) <= set(files), 'Export companions differ from protocol')
    return p, payload


def check_receipts(archive, sidecar, receipt, expected_sha, expected_bytes, expected_manifest):
    require(regular(archive).st_size == expected_bytes <= ARCHIVE_CAP, 'Archive byte count/cap mismatch')
    require(regular(sidecar).st_size <= 1024, 'Oversized checksum sidecar')
    expected = expected_sha + '  ' + Path(archive).name + '\n'
    require(Path(sidecar).read_bytes() == expected.encode('ascii'), 'Checksum sidecar differs')
    r = json_file(receipt)
    require(r.get('complete') is True and r.get('archive_sha256') == expected_sha and
            r.get('bytes') == expected_bytes and r.get('manifest_member') == MANIFEST and
            r.get('manifest_sha256') == expected_manifest and r.get('protocol_sha256') == PROTOCOL_SHA and
            type(r.get('optimizer_updates')) is int and r.get('optimizer_updates') == 0 and r.get('model_qualified') is False and
            r.get('training_success_not_implied') is True, 'Export receipt binding differs')
    return r


def member_allowed(name, exports):
    return (name == 'protocol.json' or name in exports or name in SUPERVISION or
            (name.startswith(OUTPUTS + '/') and PurePosixPath(name).suffix in {'.json', '.npy', '.png'}))


def validate_tar(handle, p, protocol_payload, receipt, expected_manifest, tick=lambda: None):
    """Read/hash every member before any imported destination is created."""
    observed = {}; folded = set(); total = 0; metadata = {}
    with tarfile.open(fileobj=handle, mode='r:gz') as tar:
        for member in tar:
            tick(); name = canonical(member.name)
            require(member.isreg() and not member.issparse() and not member.linkname,
                    'Nonregular/link/sparse member')
            require(not any(k.startswith('GNU.sparse') for k in member.pax_headers), 'Sparse extension')
            require(name not in observed and name.casefold() not in folded, 'Duplicate/case-colliding member')
            require(member_allowed(name, p['export_files']), 'Unexpected return namespace: ' + name)
            require(type(member.size) is int and 0 <= member.size <= RESULT_CAP, 'Invalid member size')
            total += member.size
            require(len(observed) < MAX_MEMBERS and total <= RESULT_CAP, 'Finite member/payload cap exceeded')
            h = hashlib.sha256(); read = 0; chunks = []
            source = tar.extractfile(member)
            require(source is not None, 'Missing regular member payload')
            for block in iter(lambda: source.read(1024**2), b''):
                tick(); h.update(block); read += len(block)
                if name in {MANIFEST, 'protocol.json'}:
                    require(read <= MAX_JSON, 'Oversized manifest/protocol'); chunks.append(block)
            require(read == member.size, 'Member truncated')
            observed[name] = {'sha256': h.hexdigest(), 'bytes': read}
            folded.add(name.casefold())
            if chunks:
                metadata[name] = b''.join(chunks)
    require(MANIFEST in observed and observed[MANIFEST]['sha256'] == expected_manifest, 'Manifest hash differs')
    require(metadata.get('protocol.json') == protocol_payload, 'Returned protocol differs')
    m = decode_json(metadata[MANIFEST]); hashes = m.get('files_sha256'); sizes = m.get('files_bytes')
    require(m.get('format') == 'codeformer-path-diagnostic-return-v1' and
            m.get('protocol_sha256') == hashlib.sha256(protocol_payload).hexdigest(), 'Manifest identity differs')
    require(isinstance(hashes, dict) and isinstance(sizes, dict) and
            set(hashes) == set(sizes) == set(observed) - {MANIFEST}, 'Manifest exact namespace differs')
    for name in hashes:
        require(SHA.fullmatch(hashes[name]) and type(sizes[name]) is int and sizes[name] >= 0 and
                observed[name] == {'sha256': hashes[name], 'bytes': sizes[name]}, 'Member hash/size differs: ' + name)
    require(m.get('parent_files_sha256') == p['parent_files_sha256'], 'Parent bindings differ')
    require(type(m.get('optimizer_updates')) is int and m.get('optimizer_updates') == 0 and m.get('model_qualified') is False and
            m.get('native_CCTV_qualification') is False and m.get('automatic_training') is False,
            'Diagnostic status differs')
    require(type(m.get('complete_measurement')) is bool and type(m.get('failure_present')) is bool and
            m.get('complete_measurement') is receipt.get('measurement_complete') and
            m.get('failure_present') is receipt.get('failure_present'), 'Receipt measurement status differs')
    for name in p['export_files']:
        require(observed.get(name, {}).get('sha256') == p['files_sha256'][name], 'Source companion differs: ' + name)
    # No filename may also be an ancestor directory of another file.
    for name in observed:
        require(not any(str(parent).casefold() in folded for parent in PurePosixPath(name).parents if str(parent) != '.'),
                'Member file/directory collision')
    return observed, m, total


def import_verified(handle, destination, observed, tick, journal):
    destination = no_links(destination)
    require(not destination.exists(), 'Exclusive import destination already exists')
    destination.mkdir(parents=False, exist_ok=False)
    imported = set()
    with tarfile.open(fileobj=handle, mode='r:gz') as tar:
        for member in tar:
            tick(); name = canonical(member.name)
            require(name in observed and name not in imported and member.isreg() and not member.issparse(),
                    'Archive changed before import')
            target = no_links(destination / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            require(no_links(target).is_relative_to(destination), 'Destination escape')
            h = hashlib.sha256(); count = 0
            with target.open('xb') as out:
                source = tar.extractfile(member)
                for block in iter(lambda: source.read(1024**2), b''):
                    tick(); h.update(block); count += len(block); out.write(block)
                out.flush(); os.fsync(out.fileno())
            require({'sha256': h.hexdigest(), 'bytes': count} == observed[name], 'Import bytes differ')
            require(digest_file(target, tick) == observed[name]['sha256'], 'Imported disk hash differs')
            imported.add(name); journal({'event': 'file_imported', 'name': name, **observed[name]})
    require(imported == set(observed), 'Imported namespace differs')
    return len(imported)


def audit(root):
    start = time.monotonic(); root = no_links(root)
    def tick():
        require(time.monotonic() - start <= 300, 'Local transport audit finite 300-second limit')
    packet = root / 'outputs/cctv_codeformer_path_diagnostic_v1_vm'
    out = root / 'outputs'; archive = out / (PREFIX + '-results.tar.gz')
    review = no_links(out / 'codeformer_path_diagnostic_v1_return_review')
    review.mkdir(exist_ok=True)
    receipt_path = review / 'transport_v1.json'
    require(not receipt_path.exists(), 'Transport receipt already exists; preserve it')
    journal_path = out / 'codeformer_path_diagnostic_v1_return_import_journal.jsonl'
    destination = out / 'codeformer_path_diagnostic_v1_return'
    require(not destination.exists(), 'Return destination already exists; preserve it')
    with journal_path.open('x', encoding='utf-8', newline='\n') as journal_file:
        def journal(event):
            journal_file.write(json.dumps(event, allow_nan=False) + '\n')
            journal_file.flush(); os.fsync(journal_file.fileno())
        journal({'event': 'audit_started', 'protocol_sha256': PROTOCOL_SHA})
        try:
            p, protocol_payload = packet_contract(packet, tick=tick)
            r = check_receipts(archive, out / (PREFIX + '-results.tar.gz.sha256'),
                               out / (PREFIX + '-export.json'), ARCHIVE_SHA, ARCHIVE_BYTES, MANIFEST_SHA)
            before = regular(archive)
            require(digest_file(archive, tick) == ARCHIVE_SHA, 'Actual archive SHA mismatch')
            with archive.open('rb') as handle:
                fd_before = identity(os.fstat(handle.fileno()))
                observed, manifest, total = validate_tar(handle, p, protocol_payload, r, MANIFEST_SHA, tick)
                require(identity(regular(archive)) == identity(before) and
                        identity(os.fstat(handle.fileno())) == fd_before, 'Archive changed during validation')
                journal({'event': 'all_members_verified_before_import', 'members': len(observed), 'bytes': total})
                require(shutil.disk_usage(out).free >= total + RESERVE, 'Insufficient local import space')
                handle.seek(0)
                count = import_verified(handle, destination, observed, tick, journal)
                require(identity(os.fstat(handle.fileno())) == fd_before and
                        identity(regular(archive)) == identity(before), 'Archive changed during import')
            result = {'complete': True, 'transport_verified': True, 'archive_sha256': ARCHIVE_SHA,
                      'archive_bytes': ARCHIVE_BYTES, 'protocol_sha256': PROTOCOL_SHA,
                      'manifest_sha256': MANIFEST_SHA, 'members': count, 'payload_bytes': total,
                      'packet_files_verified': len(p['files_sha256']), 'source_companions': len(p['export_files']),
                      'destination': str(destination), 'journal': str(journal_path),
                      'measurement_complete_reported': manifest['complete_measurement'],
                      'failure_present_reported': manifest['failure_present'], 'model_qualified': False,
                      'training_success_not_implied': True, 'local_neural_calls': 0, 'local_gradient_queries': 0,
                      'local_optimizer_updates': 0, 'VM_calls': 0, 'seconds': time.monotonic() - start}
            write_new(receipt_path, result); journal({'event': 'transport_complete', 'receipt': str(receipt_path)})
            return result
        except BaseException as exc:
            failure = {'complete': False, 'error_type': type(exc).__name__, 'error': str(exc),
                       'partial_import_preserved': destination.exists(), 'seconds': time.monotonic() - start,
                       'model_qualified': False, 'journal': str(journal_path)}
            journal({'event': 'audit_failed', **failure})
            if not receipt_path.exists():
                write_new(receipt_path, failure)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

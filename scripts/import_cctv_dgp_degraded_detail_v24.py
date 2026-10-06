"""Verify and safely import a user-returned V24 archive. Never execute it."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = '76ab24695a40b411832e2d678c79b0e2a80f6320d4525970b2dd32db2379e114'
PREFIX = 'cctv_dgp_degraded_detail_v24_return'
ARCHIVE_NAME = 'cctv-dgp-degraded-detail-v24-results.tar.gz'
MAX_MEMBERS = 1024
MAX_TOTAL_BYTES = 2 * 1024**3
MAX_FILE_BYTES = 32 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def relative_name(name):
    part = PurePosixPath(name)
    require(bool(part.parts) and not part.is_absolute() and str(part) == name,
            'Noncanonical relative archive path: ' + name)
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {'COM' + str(i) for i in range(1, 10)} | {'LPT' + str(i) for i in range(1, 10)}
    require(all(p not in {'.', '..'} and not p.endswith((' ', '.')) and
                p.split('.')[0].upper() not in reserved for p in part.parts) and
            not re.search(r'[\\:\x00-\x1f<>"|?*]', name), 'Unsafe archive path: ' + name)
    return part


def safe(root, name):
    part = relative_name(name)
    path = root.joinpath(*part.parts).resolve()
    require(path.is_relative_to(root.resolve()), 'Path escapes import root')
    return path


def inspect_archive(archive, expected_pin=PIN):
    """Read all members before making a directory; hashes bind the exact member set."""
    fingerprints = {}
    manifest = None
    total = 0
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            require(len(fingerprints) < MAX_MEMBERS, 'Too many returned archive members')
            part = relative_name(member.name)
            require(member.isfile() and not member.issparse(), 'Only nonsparse regular files are permitted')
            require(len(part.parts) > 1 and part.parts[0] == PREFIX, 'Returned archive prefix differs')
            name = '/'.join(part.parts[1:])
            require(name.casefold() not in {n.casefold() for n in fingerprints}, 'Duplicate Windows archive member')
            require(0 <= member.size <= MAX_FILE_BYTES, 'Returned member size exceeds cap')
            total += member.size
            require(total <= MAX_TOTAL_BYTES, 'Returned uncompressed archive exceeds cap')
            source = stream.extractfile(member)
            require(source is not None, 'Unreadable archive member')
            digest = hashlib.sha256()
            payload = bytearray() if name == 'export_manifest.json' else None
            count = 0
            for block in iter(lambda: source.read(1024**2), b''):
                digest.update(block); count += len(block)
                if payload is not None:
                    require(count <= 1024**2, 'Export manifest exceeds1MiB')
                    payload.extend(block)
            require(count == member.size, 'Truncated archive member')
            fingerprints[name] = digest.hexdigest()
            if payload is not None:
                # Use the same duplicate/nonfinite JSON rejection as ordinary receipts.
                def pairs(items):
                    result = {}
                    for key, value in items:
                        require(key not in result, 'Duplicate export manifest key')
                        result[key] = value
                    return result
                manifest = json.loads(payload.decode('utf-8'), object_pairs_hook=pairs,
                    parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite manifest: ' + v)))
    require(manifest is not None and manifest.get('complete') is True and
            manifest.get('protocol_sha256') == expected_pin, 'Missing or changed export manifest/protocol')
    declared = manifest.get('files_sha256')
    require(isinstance(declared, dict) and set(declared) == set(fingerprints) - {'export_manifest.json'},
            'Export manifest does not bind the exact returned file set')
    for name, digest in declared.items():
        relative_name(name)
        require(re.fullmatch(r'[0-9a-f]{64}', str(digest)) and fingerprints[name] == digest,
                'Returned member hash differs: ' + name)
    for name in ('protocol.json', 'schedule.json', 'scripts/cctv_dgp_degraded_detail_v24.py', 'scripts/run_v24.sh'):
        require(name in declared, 'Missing returned frozen source: ' + name)
    require(fingerprints['protocol.json'] == expected_pin, 'Returned protocol bytes differ')
    return manifest, fingerprints, total


def import_return(archive, sidecar, receipt_path, destination, allowed_root=ROOT, expected_sha=None, expected_bytes=None):
    started = time.monotonic()
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    require(destination.is_relative_to(Path(allowed_root).resolve()) and destination != Path(allowed_root).resolve(),
            'Import destination must stay inside the named workspace')
    require(not destination.exists(), 'Preserve existing return; no overwrite')
    require(archive.is_file() and archive.stat().st_size <= 1024**3, 'Missing or oversized return archive')
    actual = sha(archive)
    if expected_sha is not None:
        require(actual == expected_sha, 'Return differs from reported archive hash')
    if expected_bytes is not None:
        require(archive.stat().st_size == expected_bytes, 'Return differs from reported archive size')
    match = re.fullmatch(r'([0-9a-f]{64})  ([^\r\n]+)\n?', Path(sidecar).read_text(encoding='ascii'))
    require(match is not None and match.group(1) == actual and match.group(2) == ARCHIVE_NAME,
            'Return checksum hash/filename differs')
    receipt = read(receipt_path)
    require(receipt.get('complete') is True and receipt.get('archive_sha256') == actual and
            type(receipt.get('bytes')) is int and receipt['bytes'] == archive.stat().st_size,
            'Export receipt hash/size differs')
    require(receipt.get('training_success_not_implied') is True, 'Export must disclaim training acceptance')
    manifest, fingerprints, total = inspect_archive(archive)
    require(receipt.get('run_results_present') is ('outputs/results.json' in fingerprints) and
            receipt.get('failure_present') is ('outputs/failure.json' in fingerprints), 'Export presence receipt differs')
    destination.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            name = '/'.join(PurePosixPath(member.name).parts[1:])
            target = safe(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1024**2)
            require(sha(target) == fingerprints[name], 'Extraction read-back differs: ' + name)
    require(sha(archive) == actual, 'Archive changed during import')
    report = {'complete': True, 'scope': 'Transfer integrity and safe import only; no trained quality claim',
        'archive_sha256': actual, 'archive_bytes': archive.stat().st_size,
        'expected_reported_archive_sha256': expected_sha, 'expected_reported_archive_bytes': expected_bytes,
        'sidecar_sha256': sha(sidecar), 'export_receipt_sha256': sha(receipt_path),
        'protocol_sha256': PIN, 'members': len(fingerprints), 'uncompressed_bytes': total,
        'files_sha256': fingerprints, 'seconds': time.monotonic() - started,
        'neural_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False}
    write(destination.parent / (destination.name + '_import.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'outputs' / ARCHIVE_NAME)
    parser.add_argument('--sidecar', type=Path)
    parser.add_argument('--export-receipt', type=Path, default=ROOT / 'outputs/cctv-dgp-degraded-detail-v24-export.json')
    parser.add_argument('--destination', type=Path, default=ROOT / 'outputs' / PREFIX)
    parser.add_argument('--expected-sha', default='1d758362366f3e1f17ddf8239b0f41940b6b7c6ebf349a86ed76e0abdf63dd15')
    parser.add_argument('--expected-bytes', type=int, default=76492526)
    args = parser.parse_args()
    sidecar = args.sidecar or Path(str(args.archive) + '.sha256')
    report = import_return(args.archive, sidecar, args.export_receipt, args.destination,
                           expected_sha=args.expected_sha, expected_bytes=args.expected_bytes)
    print(json.dumps({k: report[k] for k in ('complete', 'members', 'archive_bytes', 'seconds', 'scope')}, indent=2))


if __name__ == '__main__':
    main()

"""Build and independently reread a finite manual diagnostic transfer archive."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_path_diagnostic_v1 import contract as c

ARCHIVE = 'cctv-codeformer-path-diagnostic-v1-execution.tar.gz'
EVIDENCE = ROOT / 'outputs/codeformer_path_diagnostic_v1_preparation'


def archive_readback(path, packet, protocol_sha):
    p = c.verify_transfer(packet, protocol_sha, verify_parent=False)
    expected = dict(p['files_sha256'], **{'protocol.json': protocol_sha})
    seen, total = set(), 0
    with tarfile.open(path, 'r:gz') as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            c.require(member.isfile() and not member.issparse() and member.name == name.as_posix() and
                      name.parts[0] == c.PACKAGE_DIR and len(name.parts) > 1,
                      'Only canonical regular packet archive members')
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            c.canonical(relative)
            c.require(relative in expected and relative not in seen,
                      'Unexpected or duplicate archive member')
            c.require(member.size == (packet / relative).stat().st_size, 'Archive member size differs')
            digest = hashlib.sha256()
            source = bundle.extractfile(member)
            c.require(source is not None, 'Missing archive payload')
            count = 0
            for block in iter(lambda: source.read(4 * 1024**2), b''):
                digest.update(block); count += len(block)
            c.require(count == member.size and digest.hexdigest() == expected[relative],
                      'Archive member content differs')
            seen.add(relative); total += count
    c.require(seen == set(expected), 'Archive omitted packet files')
    return {'members': len(seen), 'payload_bytes': total,
            'exact_namespace_and_every_member_hash': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    packet = ROOT / 'outputs' / c.PACKAGE_DIR
    p = c.verify_transfer(packet, args.protocol_sha, verify_parent=False)
    receipt_path = EVIDENCE / 'execution_archive.json'
    archive = ROOT / 'outputs' / ARCHIVE
    temporary = archive.with_name(archive.name + '.writing')
    checksum = archive.with_name(archive.name + '.sha256')
    for path in (receipt_path, archive, temporary, checksum):
        c.require(not path.exists(), 'Preserve existing or partial transfer artifact: ' + path.name)
    # Only hashes approved by the source readback can enter a protocol or archive.
    approved = c.read_json(packet / 'provenance/source_readiness.json')
    c.require(approved['complete'] and approved['approved_for_protocol_freeze'] and
              approved['training_authorized'] is False, 'Independent reviewed source required')
    for name, expected in approved['source_sha256'].items():
        c.require(p['files_sha256'].get(name) == expected, 'Sealed source differs from independent review')
    files = sorted(set(p['files_sha256']) | {'protocol.json'})
    with temporary.open('xb') as destination:
        with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as bundle:
                for relative in files:
                    path = c.bounded(packet, relative)
                    expected = args.protocol_sha if relative == 'protocol.json' else p['files_sha256'][relative]
                    c.require(c.sha256(path) == expected, 'Frozen packet file changed before archive')
                    info = bundle.gettarinfo(str(path), arcname=c.PACKAGE_DIR + '/' + relative)
                    c.require(info.isfile(), 'Execution assets must be regular files')
                    info.uid = info.gid = 0; info.uname = info.gname = ''
                    with path.open('rb') as stream:
                        bundle.addfile(info, stream)
        destination.flush(); os.fsync(destination.fileno())
    c.require(temporary.stat().st_size < 32 * 1024**2, 'Expected small diagnostic transfer below32MiB')
    readback = archive_readback(temporary, packet, args.protocol_sha)
    temporary.rename(archive)
    pin = c.sha256(archive)
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(pin + '  ' + archive.name + '\n')
    c.require(b'\r' not in checksum.read_bytes(), 'Checksum must use portable LF endings')
    receipt = {'complete': True, 'archive': archive.relative_to(ROOT).as_posix(),
               'archive_sha256': pin, 'bytes': archive.stat().st_size,
               'protocol_sha256': args.protocol_sha, **readback,
               'checksum_LF': True, 'parent_dependencies_not_reuploaded': True,
               'parent_dependencies_still_require_VM_verification': True,
               'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
               'VM_calls': 0, 'model_qualified': False, 'manual_launch_required': True}
    c.write_new_json(receipt_path, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

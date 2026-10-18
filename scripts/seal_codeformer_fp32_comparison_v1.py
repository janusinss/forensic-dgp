"""Create a regular-file-only manual transfer archive after source approval.

The protocol's dependency files are checked against the retained V2 parent;
they are installed by the new verified installer on the VM, never reuploaded.
This utility constructs no model and performs no neural, optimizer or VM call.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'cctv_codeformer_fp32_comparison_v1_vm'
PREFIX = 'cctv-codeformer-fp32-comparison-v1'
EVIDENCE = ROOT / 'outputs/codeformer_fp32_comparison_v1_preparation'
MAX_UPLOAD_BYTES = 160 * 1024**2
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)', re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and p.parts and not p.is_absolute() and p.as_posix() == name,
            'Noncanonical archive path')
    require(all(x not in ('.', '..') and not x.endswith((' ', '.')) and not DEVICE.match(x) for x in p.parts)
            and not any(c in name for c in '\\:"<>|?*') and all(ord(c) >= 32 for c in name),
            'Nonportable archive path')
    return p


def regular(path):
    path = Path(path)
    require(path.is_file(), 'Missing regular source: ' + str(path))
    for part in (path, *path.parents):
        info = part.lstat()
        require(not part.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 1024,
                'Link/reparse source forbidden')
    require(stat.S_ISREG(path.lstat().st_mode), 'Nonregular source forbidden')


def archive_readback(path, packet, expected):
    """Check complete exact namespace and payload; never extract or execute."""
    seen, folded, payload_bytes = set(), set(), 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            p = canonical(member.name)
            require(member.isreg() and not member.issparse() and not member.linkname and
                    not any(k.startswith('GNU.sparse') for k in member.pax_headers),
                    'Only ordinary regular archive members are permitted')
            require(len(p.parts) > 1 and p.parts[0] == PACKAGE, 'Archive package root differs')
            name = PurePosixPath(*p.parts[1:]).as_posix()
            require(name in expected and name not in seen and name.casefold() not in folded,
                    'Unexpected/duplicate/case-colliding member')
            source = Path(packet) / name
            require(member.size == source.stat().st_size, 'Member size differs')
            digest, count = hashlib.sha256(), 0
            with archive.extractfile(member) as stream:
                for data in iter(lambda: stream.read(1024**2), b''):
                    digest.update(data); count += len(data)
            require(count == member.size and digest.hexdigest() == expected[name], 'Member hash differs')
            payload_bytes += count; seen.add(name); folded.add(name.casefold())
    require(seen == set(expected), 'Archive namespace omits expected files')
    return {'members': len(seen), 'payload_bytes': payload_bytes, 'exact_namespace_and_every_member_hash': True}


def seal(root, protocol_sha):
    packet = root / 'outputs' / PACKAGE
    evidence = root / 'outputs/codeformer_fp32_comparison_v1_preparation'
    require(re.fullmatch('[a-f0-9]{64}', protocol_sha), 'Exact protocol SHA256 is required')
    require(sha(packet / 'protocol.json') == protocol_sha, 'Protocol hash differs')
    p = read(packet / 'protocol.json')
    approval_path = packet / 'source_readiness.json'
    approval = read(approval_path)
    require(approval.get('complete') and approval.get('approved_for_protocol_freeze') and
            approval.get('training_authorized') is False, 'Independent source approval required')
    sources = approval['source_sha256']
    require(isinstance(sources, dict) and sources, 'Nonempty reviewed source list required')
    required_sources = {'research/codeformer_fp32_comparison_v1/worker.py',
        'scripts/prepare_codeformer_fp32_comparison_v1.py',
        'scripts/supervise_codeformer_fp32_comparison_v1.py',
        'scripts/freeze_codeformer_fp32_comparison_v1.py',
        'scripts/seal_codeformer_fp32_comparison_v1.py'}
    require(required_sources <= set(sources), 'Every new execution/freeze source requires independent approval')
    for name, digest in sources.items():
        require(p['files_sha256'].get(name) == digest, 'Source differs from reviewed bytes: ' + name)
    dependency_hashes = p['dependency_files_sha256']
    require(set(dependency_hashes) <= set(p['files_sha256']), 'Dependency namespace not protocol bound')
    expected = {name: digest for name, digest in p['files_sha256'].items() if name not in dependency_hashes}
    expected['protocol.json'] = protocol_sha
    actual = {path.relative_to(packet).as_posix() for path in packet.rglob('*') if path.is_file()}
    require(actual == set(expected), 'Upload packet must contain exact overlay files only')
    parent = root / 'outputs/cctv_codeformer_adaptation_v2_vm'
    require(sha(parent / 'protocol.json') == p['parent_protocol_sha256'], 'Frozen parent protocol differs')
    parent_protocol = read(parent / 'protocol.json')
    for name, digest in dependency_hashes.items():
        require(digest == p['files_sha256'][name] == parent_protocol['files_sha256'].get(name),
                'Dependency map differs from immutable parent or complete protocol: ' + name)
    for name, digest in p['files_sha256'].items():
        canonical(name)
        file = parent / name if name in dependency_hashes else packet / name
        regular(file)
        require(sha(file) == digest, 'Bound file differs: ' + name)
    archive = root / 'outputs' / (PREFIX + '-execution.tar.gz')
    temporary = archive.with_name(archive.name + '.writing')
    checksum = archive.with_name(archive.name + '.sha256')
    receipt = evidence / 'execution_archive.json'
    for file in (archive, temporary, checksum, receipt):
        require(not file.exists(), 'Preserve existing transfer artifact: ' + file.name)
    with temporary.open('xb') as target:
        with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as bundle:
                for name in sorted(expected):
                    file = packet / name
                    require(sha(file) == expected[name], 'Frozen overlay changed')
                    info = bundle.gettarinfo(str(file), arcname=PACKAGE + '/' + name)
                    require(info.isreg(), 'Archive member not regular')
                    info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                    with file.open('rb') as stream:
                        bundle.addfile(info, stream)
        target.flush(); os.fsync(target.fileno())
    require(temporary.stat().st_size < MAX_UPLOAD_BYTES, 'Overlay exceeds expected160MiB transfer bound')
    result = archive_readback(temporary, packet, expected)
    temporary.rename(archive)
    archive_sha = sha(archive)
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(archive_sha + '  ' + archive.name + '\n')
    require(b'\r' not in checksum.read_bytes(), 'Checksum must use portable LF')
    result.update(complete=True, archive=str(archive.relative_to(root)), archive_sha256=archive_sha,
        bytes=archive.stat().st_size, protocol_sha256=protocol_sha, checksum_LF=True,
        dependency_files_not_reuploaded=len(dependency_hashes), parent_VM_assets_require_fresh_verification=True,
        neural_calls=0, gradient_queries=0, optimizer_updates=0, VM_calls=0,
        model_qualified=False, manual_launch_required=True, sealer_source_sha256=sha(__file__))
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha', required=True)
    arguments = parser.parse_args()
    print(json.dumps(seal(ROOT, arguments.protocol_sha), indent=2))

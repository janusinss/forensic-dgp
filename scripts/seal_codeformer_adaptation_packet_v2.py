"""Seal a reviewed V2 packet without editing any staged executable/protocol.

Requires independent packet approval and repeats source/copy/namespace checks.
No model, gradient, optimizer, remote connection or actual training is run.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_codeformer_adaptation_packet_v2 import (
    NAME, EXECUTION_ARCHIVE, EVIDENCE, DESIGN_SHA, HEX, require, sha256,
    read_json, write_json, write_new, bounded, canonical, exact_namespace,
    no_neural_imports, load_worker,
)


def write_lf_sidecar(path, digest, filename):
    require(HEX.fullmatch(digest) is not None and Path(filename).name == filename,
            'Malformed archive sidecar fields')
    # Bytes avoid Windows text translation; Linux sha256sum sees no trailing CR.
    write_new(path, (digest + '  ' + filename + '\n').encode('ascii'))


def inspect_archive(archive, prefix, members):
    found, folded, payload_bytes = {}, set(), 0
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            name = canonical(member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and not member.sparse,
                    'Execution archive contains a nonregular member')
            require(name.startswith(prefix + '/'), 'Wrong execution archive prefix')
            relative = canonical(name[len(prefix) + 1:])
            require(relative in members and relative not in found and relative.casefold() not in folded,
                    'Extra/duplicate/colliding execution member')
            require(member.size >= 0, 'Negative archive member size')
            folded.add(relative.casefold())
            digest = hashlib.sha256()
            size = 0
            stream = source.extractfile(member)
            for block in iter(lambda: stream.read(4 << 20), b''):
                digest.update(block); size += len(block)
            require(size == member.size and digest.hexdigest() == members[relative],
                    'Archive payload differs: ' + relative)
            found[relative] = digest.hexdigest(); payload_bytes += size
    require(found == members, 'Execution archive exact namespace differs')
    return {'members': len(found), 'payload_bytes': payload_bytes, 'all_members_streamed_and_verified': True}


def seal(root, approval_relative, approval_sha):
    started = time.monotonic()
    root = Path(root).resolve(strict=True)
    packet, evidence = root / 'outputs' / NAME, root / EVIDENCE
    archive = root / 'outputs' / EXECUTION_ARCHIVE
    receipt_path = root / 'outputs/codeformer_adaptation_v2_archive.json'
    require(not archive.exists() and not receipt_path.exists() and
            not archive.with_name(archive.name + '.sha256').exists(), 'Retain prior sealed archive/receipt')
    approval_path = bounded(root, approval_relative)
    require(HEX.fullmatch(approval_sha) and sha256(approval_path) == approval_sha, 'Independent packet approval hash differs')
    approval = read_json(approval_path)
    finalized = read_json(bounded(root, EVIDENCE + '/finalized.json'))
    protocol_path = bounded(packet, 'protocol.json')
    pin = sha256(protocol_path)
    require(finalized.get('complete') is True and finalized.get('protocol_sha256') == pin and
            finalized.get('design_sha256') == DESIGN_SHA, 'Finalized protocol differs')
    require(approval.get('complete') is True and approval.get('approved_for_archive_seal') is True and
            approval.get('protocol_sha256') == pin and approval.get('training_authorized') is False,
            'Independent review must approve this exact packet for sealing only')
    readiness_path = bounded(root, finalized['readiness_path'])
    require(sha256(readiness_path) == finalized['readiness_sha256'], 'Source readiness changed')
    readiness = read_json(readiness_path)
    for relative, expected in readiness['source_sha256'].items():
        require(sha256(bounded(root, relative)) == expected, 'Reviewed current source changed before seal')
        if relative.endswith('.py'):
            ast.parse((root / relative).read_text(encoding='utf-8'), feature_version=(3, 10))
    worker = load_worker(packet)
    with no_neural_imports():
        protocol, verification = worker.verify_transfer(packet, pin)
    members = dict(protocol['files_sha256'])
    members['protocol.json'] = pin
    require(exact_namespace(packet) == set(members), 'Unlisted/missing staged path before seal')
    for name, row in finalized['copy_bindings'].items():
        require(name in members and row['sha256'] == members[name] and
                sha256(bounded(root, row['original'])) == row['sha256'] and
                (root / row['original']).stat().st_size == row['bytes'], 'Original/copy chain changed before seal')
    total = sum(bounded(packet, name).stat().st_size for name in members)
    # A gzip stream need not compress; include TAR/PAX/header slack and 4GiB reserve.
    require(shutil.disk_usage(root).free >= total * 1.02 + len(members) * 4096 + 4 * 1024**3,
            'Require complete execution archive headroom plus4GiB local reserve')
    temporary = archive.with_name(archive.name + '.partial-' + uuid.uuid4().hex)
    try:
        with tarfile.open(temporary, 'x:gz', compresslevel=1) as destination:
            for relative, expected in sorted(members.items()):
                path = bounded(packet, relative)
                before = path.stat()
                require(sha256(path) == expected, 'Staged source changed before TAR write')
                info = destination.gettarinfo(str(path), arcname=NAME + '/' + relative)
                require(info.isfile(), 'Archive source is not regular')
                # Portable regular-file metadata; provenance file bytes remain exact.
                info.uid = info.gid = 0; info.uname = info.gname = ''
                info.mode = 0o644; info.mtime = 0; info.pax_headers = {}
                with path.open('rb') as source:
                    destination.addfile(info, source)
                after = path.stat()
                require((before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) ==
                        (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev), 'Archive source changed while reading')
        check = inspect_archive(temporary, NAME, members)
        # Verify the copied source chain again after writing, before immutable publish.
        for relative, expected in members.items():
            require(sha256(bounded(packet, relative)) == expected, 'Staged source changed during archive creation')
        for relative, expected in readiness['source_sha256'].items():
            require(sha256(bounded(root, relative)) == expected, 'Reviewed current source changed during seal')
        archive_sha = sha256(temporary)
        os.rename(temporary, archive)
        write_lf_sidecar(archive.with_name(archive.name + '.sha256'), archive_sha, archive.name)
        receipt = {'complete': True, 'archive': archive.relative_to(root).as_posix(),
            'archive_sha256': archive_sha, 'bytes': archive.stat().st_size, 'protocol_sha256': pin,
            'design_sha256': DESIGN_SHA, 'independent_packet_approval_sha256': approval_sha,
            'source_readiness_sha256': finalized['readiness_sha256'], 'verification': verification,
            **check, 'no_staged_source_or_protocol_modified_by_sealer': True,
            'training_started': False, 'model_calls': 0, 'gradient_queries': 0,
            'optimizer_updates': 0, 'VM_calls': 0, 'seconds': time.monotonic() - started}
        write_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        return receipt
    except BaseException as error:
        # Preserve partial archive and exact failed attempt; do not delete evidence.
        stop = evidence / ('seal_failure_' + uuid.uuid4().hex + '.json')
        write_json(stop, {'complete': False, 'error': str(error), 'traceback': traceback.format_exc(),
            'partial_archive': str(temporary), 'partial_archive_retained': temporary.exists(),
            'protocol_sha256': pin, 'model_calls': 0, 'VM_calls': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--approval-sha', required=True)
    args = parser.parse_args()
    seal(args.root, args.approval, args.approval_sha)


if __name__ == '__main__':
    main()

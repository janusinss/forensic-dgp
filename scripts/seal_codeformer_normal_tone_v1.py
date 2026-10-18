"""Seal and independently read back a source-only manual packet; no VM/model calls."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_v1 import contract as c
from research.codeformer_normal_tone_v1 import transport as t


def inventory(stage, protocol_sha):
    p = c.read_json(stage / 'protocol.json')
    c.require(c.sha256(stage / 'protocol.json') == protocol_sha, 'Protocol digest differs')
    c.verify_protocol(p)
    wanted = dict(p['files_sha256'])
    wanted['protocol.json'] = protocol_sha
    sizes = dict(p['files_bytes'])
    sizes['protocol.json'] = (stage / 'protocol.json').stat().st_size
    files = t._walk(stage)
    c.require(set(files) == set(wanted), 'Source stage must have exactly its frozen static members')
    c.require(len(files) <= 64 and sum(sizes.values()) <= 16 * 1024**2,
              'Source-only stage exceeds64files/16MiB')
    for name in sorted(files):
        c.require(not name.startswith('outputs/') and
                  Path(name).suffix.lower() in {'.py', '.json', '.md'}, 'Source-only member required')
        c.require(files[name].stat().st_size == sizes[name] and c.sha256(files[name]) == wanted[name],
                  'Stage byte binding differs: ' + name)
    return p, files, wanted, sizes


def verify_current_sources(stage, wanted):
    provenance = {
        'source_design_review_v1.md': 'outputs/codeformer_normal_tone_v1_design/source_design_review_v1.md',
        'source_design_review_v1.json': 'outputs/codeformer_normal_tone_v1_design/source_design_review_v1.json',
        'next_design_independent_review_v1.md': 'outputs/codeformer_loss_response_v1_return_review/next_design_independent_review_v1.md',
        'next_design_independent_review_v1.json': 'outputs/codeformer_loss_response_v1_return_review/next_design_independent_review_v1.json',
    }
    for name, digest in wanted.items():
        if name == 'protocol.json':
            continue
        if name == 'research/__init__.py':
            c.require((stage / name).read_bytes() == b'', 'Exact empty package marker required')
            continue
        if name == 'cases.json':
            origin = ROOT / 'outputs/codeformer_loss_response_v1_return/cases.json'
        elif name.startswith('instructions/'):
            origin = ROOT / name[len('instructions/'):]
        elif name.startswith('provenance/'):
            c.require(Path(name).name in provenance, 'Unknown provenance origin')
            origin = ROOT / provenance[Path(name).name]
        else:
            origin = ROOT / name
        c.require(c.sha256(c.no_links(origin)) == digest, 'Current source differs from stage: ' + name)


def verify_archive(path, top, wanted, sizes):
    """Require a single complete gzip stream and exact regular tar namespace."""
    import zlib
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    logical = 0
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(1024**2)
            if not chunk:
                break
            while chunk:
                block = inflater.decompress(chunk, 1024**2)
                logical += len(block)
                c.require(logical <= 32 * 1024**2, 'Expanded source archive exceeds32MiB')
                c.require(not inflater.unused_data, 'Trailing/concatenated gzip data forbidden')
                chunk = inflater.unconsumed_tail
        c.require(inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail,
                  'Incomplete or extra gzip stream')
    seen, folded = {}, set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            c.require(member.isfile() and not member.pax_headers and member.name.startswith(top + '/'),
                      'Only canonical regular source members allowed')
            name = member.name[len(top) + 1:]
            t.canonical(name)
            c.require(name in wanted and name not in seen and name.casefold() not in folded,
                      'Unknown/duplicate/colliding archive member')
            c.require(member.size == sizes[name], 'Archive member size differs')
            with archive.extractfile(member) as stream:
                digest = hashlib.sha256()
                actual = 0
                for payload in iter(lambda: stream.read(1024**2), b''):
                    digest.update(payload)
                    actual += len(payload)
                c.require(actual == sizes[name] and digest.hexdigest() == wanted[name],
                          'Archive member digest differs: ' + name)
            seen[name] = digest.hexdigest()
            folded.add(name.casefold())
    c.require(seen == wanted, 'Archive namespace incomplete')
    # tarfile deliberately stops at the first end marker. Independently consume
    # the complete physical stream, including canonical headers and zero padding.
    with gzip.open(path, 'rb') as source:
        consumed = 0
        for name in sorted(wanted):
            expected = tarfile.TarInfo(top + '/' + name)
            expected.size = sizes[name]
            expected.mode = 0o644
            expected.uid = expected.gid = expected.mtime = 0
            expected.uname = expected.gname = ''
            header = source.read(512)
            c.require(header == expected.tobuf(format=tarfile.USTAR_FORMAT), 'Noncanonical USTAR header/order')
            payload = source.read(sizes[name])
            c.require(len(payload) == sizes[name] and hashlib.sha256(payload).hexdigest() == wanted[name],
                      'Physical TAR payload differs')
            padding = (-sizes[name]) % 512
            c.require(source.read(padding) == b'\0' * padding, 'Nonzero TAR member padding')
            consumed += 512 + sizes[name] + padding
        expected_tail = ((consumed + 1024 + 10239) // 10240) * 10240 - consumed
        c.require(source.read(expected_tail) == b'\0' * expected_tail and source.read(1) == b'',
                  'Noncanonical/trailing TAR end records or blocking padding')
    return {'members': len(seen), 'payload_bytes': sum(sizes.values()),
            'single_gzip_stream_CRC_checked': True, 'all_member_bytes_independently_rehashed': True,
            'canonical_USTAR_headers_order_padding_and_full_EOF_checked': True,
            'extraction_or_execution': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    stage = c.no_links(args.stage).resolve(strict=True)
    c.require(stage.is_relative_to(ROOT / 'outputs') and stage.name == c.VM_NAME,
              'Exact new local canonical stage required')
    p, files, wanted, sizes = inventory(stage, args.protocol_sha)
    verify_current_sources(stage, wanted)
    readiness = ROOT / 'outputs/codeformer_normal_tone_v1_preparation/readiness_v1.json'
    ready = c.read_json(readiness)
    c.require(ready.get('complete') is True and ready.get('protocol_sha256') == args.protocol_sha and
              ready.get('source_files') == wanted and ready.get('local_neural_gradient_optimizer_VM_calls') == 0 and
              ready.get('independent_source_review_complete') is True and ready.get('tests_failed') == 0,
              'Closed hash-bound preparation/readiness receipt required')
    for name, digest in ready['evidence_sha256'].items():
        c.safe_relative(name)
        c.require(c.sha256(ROOT / name) == digest, 'Readiness evidence changed: ' + name)
    archive = ROOT / 'outputs/cctv-codeformer-normal-tone-v1-execution.tar.gz'
    part = archive.with_name(archive.name + '.part')
    checksum = archive.with_name(archive.name + '.sha256')
    receipt = ROOT / 'outputs/codeformer_normal_tone_v1_design/sealed_packet_v1.json'
    for path in (archive, part, checksum, receipt):
        c.require(not path.exists() and not path.is_symlink(), 'Preserve existing seal attempt: ' + str(path))
    with part.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as packed:
                for name in sorted(files):
                    info = tarfile.TarInfo(c.VM_NAME + '/' + name)
                    info.size = sizes[name]
                    info.mode = 0o644
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with files[name].open('rb') as source:
                        packed.addfile(info, source)
        raw.flush()
        os.fsync(raw.fileno())
    readback = verify_archive(part, c.VM_NAME, wanted, sizes)
    c.require(part.stat().st_size <= 16 * 1024**2, 'Compressed source cap exceeded')
    digest = c.sha256(part)
    os.replace(part, archive)
    line = (digest + '  ' + archive.name + '\n').encode('ascii')
    with checksum.open('xb') as stream:
        stream.write(line)
    c.require(checksum.read_bytes() == line and b'\r' not in line, 'Exact basename/ASCII/LF checksum required')
    c.write_new_json(receipt, {'complete': True, 'status': 'PREPARED_UNRUN',
        'UTC': datetime.now(timezone.utc).isoformat(), 'stage': str(stage),
        'protocol_sha256': args.protocol_sha, 'archive': str(archive), 'archive_sha256': digest,
        'archive_bytes': archive.stat().st_size, 'checksum_sha256': c.sha256(checksum),
        'readback': readback, 'source_files': wanted, 'source_bytes': sizes,
        'parent_dependency_counts': {key: len(value) for key, value in p['dependencies'].items()},
        'readiness_receipt_sha256': c.sha256(readiness),
        'sealer_source_sha256': c.sha256(__file__), 'actual_neural_gradient_optimizer_VM_calls': 0,
        'new_trained_model_created': False, 'model_qualified': False})
    print({'complete': True, 'protocol_sha256': args.protocol_sha, 'archive_sha256': digest,
           'bytes': archive.stat().st_size, **readback, 'neural_or_VM_calls': 0})


if __name__ == '__main__':
    main()

"""Seal an exact, reviewed source-only manual diagnostic packet; never execute it.

Readiness is mandatory. Existing stages, archives, checksums and receipts are
preserved. Gzip/tar headers are deterministic and independently read back.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import unicodedata
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_loss_response_v1 import contract as c

SOURCE_CAP = 16 * 1024**2
TAR_CAP = 32 * 1024**2
ARCHIVE_CAP = 16 * 1024**2
ARCHIVE_NAME = 'cctv-codeformer-loss-response-v1-execution.tar.gz'
BUILDER = 'scripts/prepare_codeformer_loss_response_v1.py'
REVIEW = 'outputs/codeformer_loss_response_design_v1'
SOURCE_MAP = {
    'research/codeformer_loss_response_v1/__init__.py': 'research/codeformer_loss_response_v1/__init__.py',
    'research/codeformer_loss_response_v1/contract.py': 'research/codeformer_loss_response_v1/contract.py',
    'research/codeformer_loss_response_v1/worker.py': 'research/codeformer_loss_response_v1/worker.py',
    'research/codeformer_loss_response_v1/transport.py': 'research/codeformer_loss_response_v1/transport.py',
    'scripts/supervise_codeformer_loss_response_v1.py': 'scripts/supervise_codeformer_loss_response_v1.py',
    **{'instructions/' + name: name for name in ('AGENTS.md', 'SYSTEM_WORKFLOW_AND_GOAL.md',
        'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'PROJECT_HANDOFF.md',
        'CCTV_CODEFORMER_LOSS_RESPONSE_V1_DESIGN.md')},
    **{'provenance/' + name: REVIEW + '/' + name for name in ('loss_contract_review.md',
        'train_pixels_findings_v1.md', 'train_pixels_finite_design_review_v1.md')},
}
OWN_NAMES = frozenset(SOURCE_MAP) | {'research/__init__.py', 'cases.json'}
ALL_NAMES = OWN_NAMES | {'protocol.json'}
FORBIDDEN = {'.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.npy', '.npz', '.safetensors',
             '.png', '.jpg', '.jpeg', '.bin', '.tar', '.gz', '.zip'}
DEVICE = re.compile(r'(?i)^(con|prn|aux|nul|conin\$|conout\$|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def parse_json(payload):
    def unique(items):
        result = {}
        for key, value in items:
            c.require(key not in result, 'Duplicate JSON key in sealing input')
            result[key] = value
        return result
    return json.loads(payload.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite sealing JSON')))


def portable(name):
    c.safe_relative(name)
    c.require(unicodedata.normalize('NFC', name) == name and
              not any(char in name for char in '\\:"<>|?*') and
              all(not DEVICE.match(part) and all(ord(char) >= 32 for char in part)
                  for part in PurePosixPath(name).parts), 'Nonportable source/archive path')
    return name.casefold()


def exact_regular(path):
    path = c.no_links(path)
    info = path.lstat()
    c.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
              'Exact singly linked regular file required: ' + str(path))
    c.require(info.st_size <= SOURCE_CAP, 'Source file exceeds16MiB cap')
    return path


def read_bound(path):
    path = exact_regular(path)
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    c.require((before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) ==
              (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) and
              len(payload) == after.st_size, 'File changed during source sealing read')
    return payload


def inventory(stage):
    stage = c.no_links(stage).resolve(strict=True)
    c.require(stage.is_dir(), 'Stage must be an exact directory')
    found, keys = {}, set()
    expected_directories = {str(parent) for name in ALL_NAMES for parent in PurePosixPath(name).parents
                            if str(parent) != '.'}
    for current, directories, files in os.walk(stage, followlinks=False):
        current = Path(current)
        for name in sorted(directories + files):
            path = c.no_links(current / name)
            relative = path.relative_to(stage).as_posix()
            key = portable(relative)
            c.require(key not in keys, 'Case/Unicode source namespace collision')
            keys.add(key)
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                c.require(relative in expected_directories, 'Unlisted/output directory in source stage')
            else:
                c.require(relative in ALL_NAMES, 'Unlisted source file in stage')
                c.require(path.suffix.lower() not in FORBIDDEN, 'Binary/model/array payload forbidden')
                found[relative] = read_bound(path)
    c.require(set(found) == ALL_NAMES and len(found) == 17, 'Exactly17 declared source/protocol files required')
    c.require(sum(map(len, found.values())) <= SOURCE_CAP, 'Source packet exceeds16MiB total payload cap')
    return found


def verify_readiness(value, protocol_sha, receipt_sha):
    c.require(all(value.get(key) is True for key in
        ('complete', 'source_data_review_pass', 'CPU_only', 'manual_unrun')), 'Current complete CPU-only/manual-unrun readiness required')
    c.require(value.get('protocol_sha256') == protocol_sha and
              value.get('stage_receipt_sha256') == receipt_sha, 'Readiness is stale for current protocol/stage receipt')
    c.require(type(value.get('actual_neural_gradient_optimizer_VM_calls')) is int and
              value['actual_neural_gradient_optimizer_VM_calls'] == 0, 'Readiness must bind zero actual local neural/VM calls')


def validate_stage(repo, stage_receipt, readiness):
    repo = c.no_links(repo).resolve(strict=True)
    receipt_payload = read_bound(stage_receipt)
    ready_payload = read_bound(readiness)
    receipt = parse_json(receipt_payload)
    ready = parse_json(ready_payload)
    c.require(receipt.get('complete') is True and receipt.get('execution_archive_sealed') is False and
              receipt.get('model_qualified') is False and
              type(receipt.get('actual_neural_gradient_optimizer_VM_calls')) is int and
              receipt['actual_neural_gradient_optimizer_VM_calls'] == 0, 'Complete unsealed CPU-only stage receipt required')
    stage = c.no_links(Path(receipt['stage'])).resolve(strict=True)
    c.require(stage.parent == repo / 'outputs' and
              re.fullmatch(re.escape(c.VM_NAME) + r'_r[1-9][0-9]*', stage.name),
              'Explicit revised stage required; original unsealed stage is preserved')
    source = inventory(stage)
    protocol_sha = digest(source['protocol.json'])
    c.require(receipt.get('protocol_sha256') == protocol_sha, 'Stage protocol differs from stage receipt')
    verify_readiness(ready, protocol_sha, digest(receipt_payload))
    protocol = parse_json(source['protocol.json'])
    c.verify_protocol(protocol)
    cases = c.verify_cases(parse_json(source['cases.json']))
    for row in cases['cases']:
        c.require(all(value in protocol['dependencies'] for key, value in row.items() if key.endswith('_path')),
                  'Canonical case path is not a bound read-only parent dependency')
        c.require(protocol['dependencies'][row['source_path']]['sha256'] == row['record']['source_sha256'] and
                  protocol['dependencies'][row['input_path']]['sha256'] == row['record']['saved_input_sha256'],
                  'Canonical case source/input fingerprint differs from parent binding')
    hashes = {name: digest(payload) for name, payload in source.items() if name != 'protocol.json'}
    sizes = {name: len(payload) for name, payload in source.items() if name != 'protocol.json'}
    c.require(set(protocol['files_sha256']) == OWN_NAMES and
              protocol['files_sha256'] == hashes == receipt.get('source_files') and
              protocol['files_bytes'] == sizes == receipt.get('source_bytes'),
              'Exact source hashes/sizes differ between stage/protocol/receipt')
    c.require(receipt.get('dependencies') == protocol['dependencies'], 'Stage dependency bindings differ from protocol')
    c.require(receipt.get('source_sha256') == digest(read_bound(repo / BUILDER)),
              'Current builder differs from stage receipt source hash')
    c.require(source['research/__init__.py'] == b'', 'Intentional empty research namespace initializer differs')
    for name, relative in SOURCE_MAP.items():
        c.require(source[name] == read_bound(repo / relative), 'Current copied repository source differs: ' + name)
    return {'repo': repo, 'stage': stage, 'source': source, 'protocol': protocol,
            'protocol_sha256': protocol_sha, 'stage_receipt_sha256': digest(receipt_payload),
            'readiness_sha256': digest(ready_payload), 'stage_receipt': Path(stage_receipt),
            'readiness': Path(readiness)}


class CappedWriter:
    def __init__(self, stream, cap):
        self.stream, self.cap, self.count = stream, cap, 0
    def write(self, payload):
        c.require(self.count + len(payload) <= self.cap, 'Archive/uncompressed stream exceeds finite cap')
        result = self.stream.write(payload)
        self.count += len(payload)
        return result
    def tell(self):
        return self.count
    def flush(self):
        return self.stream.flush()


def write_archive(path, source):
    """Exclusive deterministic archive; partial bytes are preserved on failure."""
    c.require(set(source) == ALL_NAMES, 'Archive input census differs')
    c.require(all(isinstance(payload, bytes) for payload in source.values()) and
              sum(map(len, source.values())) <= SOURCE_CAP, 'Source archive payload exceeds16MiB cap')
    path = Path(path)
    c.no_links(path.parent)
    c.require(not path.exists() and not path.is_symlink(), 'Retain existing archive; overwrite forbidden')
    with path.open('xb') as raw:
        compressed = CappedWriter(raw, ARCHIVE_CAP)
        with gzip.GzipFile(filename='', mode='wb', compresslevel=9, mtime=0, fileobj=compressed) as gz:
            uncompressed = CappedWriter(gz, TAR_CAP)
            with tarfile.open(fileobj=uncompressed, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted(source):
                    payload = source[name]
                    header = tarfile.TarInfo(c.VM_NAME + '/' + name)
                    header.size = len(payload); header.type = tarfile.REGTYPE
                    header.mode = 0o644; header.uid = header.gid = 0; header.mtime = 0
                    header.uname = header.gname = header.linkname = ''
                    archive.addfile(header, io.BytesIO(payload))
        raw.flush(); os.fsync(raw.fileno())
    return path


def readback_archive(path, source):
    """Independent decompression/header/payload/census validation; no extraction."""
    payload = read_bound(path)
    c.require(len(payload) <= ARCHIVE_CAP and payload[:10] == bytes((31,139,8,0,0,0,0,0,2,255)),
              'Deterministic gzip header differs')
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    tar_payload = decoder.decompress(payload, TAR_CAP + 1)
    c.require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
              'Exactly one complete gzip frame required; trailing/concatenated bytes forbidden')
    expected_tar_bytes = sum(512 + ((len(value) + 511) // 512) * 512 for value in source.values()) + 1024
    expected_tar_bytes = ((expected_tar_bytes + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE) * tarfile.RECORDSIZE
    c.require(len(tar_payload) == expected_tar_bytes and len(tar_payload) <= TAR_CAP,
              'Uncompressed tar bound/record padding differs')
    expected = [c.VM_NAME + '/' + name for name in sorted(source)]
    seen, member_receipts, end = set(), [], 0
    with tarfile.open(fileobj=io.BytesIO(tar_payload), mode='r:') as archive:
        members = archive.getmembers()
        c.require([member.name for member in members] == expected and len(members) == 17,
                  'Archive order/census/namespace differs')
        for member in members:
            portable(member.name)
            c.require(member.name.casefold() not in seen, 'Archive namespace collision')
            seen.add(member.name.casefold())
            c.require(member.isfile() and member.type == tarfile.REGTYPE and
                      member.mode == 0o644 and member.uid == member.gid == member.mtime == 0 and
                      member.uname == member.gname == member.linkname == '' and not member.pax_headers,
                      'Archive regular deterministic header differs')
            name = member.name[len(c.VM_NAME) + 1:]
            expected_payload = source[name]
            c.require(member.size == len(expected_payload), 'Archive member size differs')
            stream = archive.extractfile(member)
            actual = stream.read(member.size + 1)
            c.require(actual == expected_payload, 'Archive member payload/hash differs')
            member_receipts.append({'path': member.name, 'bytes': member.size, 'sha256': digest(actual)})
            end = member.offset_data + ((member.size + 511) // 512) * 512
    c.require(len(tar_payload) - end >= 1024 and not any(tar_payload[end:]), 'Nonzero/trailing archive payload forbidden')
    return {'archive_sha256': digest(payload), 'archive_bytes': len(payload),
            'uncompressed_tar_bytes': len(tar_payload), 'member_payload_bytes': sum(map(len, source.values())),
            'regular_members': len(member_receipts), 'members': member_receipts,
            'deterministic_gzip_and_regular_tar_headers': True, 'independent_payload_readback_pass': True}


def seal(repo, stage_receipt, readiness):
    validated = validate_stage(repo, stage_receipt, readiness)
    repo, source = validated['repo'], validated['source']
    destination = repo / 'outputs' / ARCHIVE_NAME
    checksum = Path(str(destination) + '.sha256')
    receipt_path = repo / REVIEW / 'seal_v1.json'
    for path in (destination, checksum, receipt_path):
        c.no_links(path.parent)
        c.require(not path.exists() and not path.is_symlink(), 'Retain existing archive/checksum/seal receipt')
    write_archive(destination, source)
    readback = readback_archive(destination, source)
    checksum_payload = (readback['archive_sha256'] + '  ' + destination.name + '\n').encode('ascii')
    with checksum.open('xb') as stream:
        stream.write(checksum_payload); stream.flush(); os.fsync(stream.fileno())
    c.require(read_bound(checksum) == checksum_payload and checksum_payload.count(b'\n') == 1 and b'\r' not in checksum_payload,
              'Portable LF SHA-256 basename checksum readback failed')
    # Detect changes after the initial read and bind the exact input receipts again.
    current = validate_stage(repo, stage_receipt, readiness)
    c.require(current['source'] == source and current['stage_receipt_sha256'] == validated['stage_receipt_sha256'] and
              current['readiness_sha256'] == validated['readiness_sha256'], 'Sealing inputs changed before final receipt')
    receipt = {'format': 'codeformer-loss-response-source-seal-v1', 'complete': True,
        'status': 'SEALED_PREPARED_UNRUN', 'protocol_sha256': validated['protocol_sha256'],
        'stage': str(validated['stage']), 'stage_receipt': str(validated['stage_receipt']),
        'stage_receipt_sha256': validated['stage_receipt_sha256'], 'readiness': str(validated['readiness']),
        'readiness_sha256': validated['readiness_sha256'], 'archive': str(destination),
        'checksum': str(checksum), 'checksum_sha256': digest(checksum_payload), **readback,
        'current_builder_sha256': digest(read_bound(repo / BUILDER)),
        'sealer_sha256': digest(read_bound(Path(__file__))), 'original_stage_preserved': True,
        'source_packet_only': True, 'dependencies_included_in_archive': False,
        'actual_neural_gradient_optimizer_VM_calls': 0, 'training_launched': False, 'model_qualified': False}
    c.write_new_json(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-receipt', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = seal(ROOT, args.stage_receipt, args.readiness)
    print(json.dumps({'complete': True, 'protocol_sha256': receipt['protocol_sha256'],
                      'archive_sha256': receipt['archive_sha256'], 'archive_bytes': receipt['archive_bytes'],
                      'regular_members': receipt['regular_members'], 'actual_neural_gradient_optimizer_VM_calls': 0}))


if __name__ == '__main__':
    main()

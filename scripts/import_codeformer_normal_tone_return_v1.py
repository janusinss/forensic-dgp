"""Bounded hash-bound return import; never execute returned code or load tensors."""
from datetime import datetime, timezone
import base64
import hashlib
import io
from pathlib import Path
import shutil
import sys
import tarfile
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_v1 import contract as c
from research.codeformer_normal_tone_v1.transport import canonical

PREFIX = 'cctv-codeformer-normal-tone-v1'
PROTOCOL = '6a3e162f6383cefac9d14fa11e08032d0149a7583d6ff60ba2ff893056335d22'
ARCHIVE_SHA = 'f5f8700fd5b02be9221c908689af5507d83c8b6d00204295799adb1ac5d01f92'
MANIFEST = 'outputs/normal_tone_v1/export_manifest.json'
REVIEW = ROOT / 'outputs/codeformer_normal_tone_v1_return_review'
DEST = ROOT / 'outputs/codeformer_normal_tone_v1_return'
PAYLOAD_CAP = 2 * 1024**3
TAR_CAP = PAYLOAD_CAP + 5000 * 8192 + 10240


class StrictGzip(io.RawIOBase):
    def __init__(self, path):
        self.source = path.open('rb')
        self.decoder = zlib.decompressobj(31)
        self.pending = b''
        self.buffer = bytearray()
        self.digest = hashlib.sha256()
        self.compressed = self.expanded = 0
        self.ended = False

    def readable(self):
        return True

    def fill(self):
        if self.ended:
            return
        if self.decoder.eof:
            c.require(not self.decoder.unused_data and not self.pending and not self.source.read(1),
                      'Single gzip stream without trailing data required')
            self.ended = True
            return
        payload = self.pending
        if not payload:
            payload = self.source.read(1024**2)
            c.require(payload, 'Truncated gzip frame')
            self.digest.update(payload)
            self.compressed += len(payload)
        output = self.decoder.decompress(payload, 4 * 1024**2)
        self.pending = self.decoder.unconsumed_tail
        self.expanded += len(output)
        c.require(self.expanded <= TAR_CAP, 'Expanded TAR cap exceeded')
        self.buffer.extend(output)

    def read(self, size=-1):
        c.require(type(size) is int and 0 <= size <= 4 * 1024**2, 'Bounded stream read required')
        while len(self.buffer) < size and not self.ended:
            self.fill()
        count = min(size, len(self.buffer))
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def close(self):
        self.source.close()
        super().close()


def pin(path):
    path = c.no_links(path)
    return {'sha256': c.sha256(path), 'bytes': path.stat().st_size}


def main():
    outputs = ROOT / 'outputs'
    archive = outputs / (PREFIX + '-results.tar.gz')
    checksum = outputs / (PREFIX + '-results.tar.gz.sha256')
    export_path = outputs / (PREFIX + '-export.json')
    final_path = outputs / (PREFIX + '-supervisor-final.json')
    stage = outputs / c.VM_NAME
    receipt_path = REVIEW / 'return_import_v1.json'
    c.require(not DEST.exists() and not receipt_path.exists(), 'Preserve previous import')
    REVIEW.mkdir(exist_ok=True)
    export = c.read_json(export_path)
    c.require(export['format'] == 'codeformer-normal-tone-export-v1' and export['complete'] is True and
              export['protocol_sha256'] == PROTOCOL and export['archive_sha256'] == ARCHIVE_SHA and
              export['manifest_member'] == MANIFEST and export['model_qualified'] is False,
              'Exact returned export identity required')
    archive_pin = pin(archive)
    c.require(archive_pin == {'sha256': ARCHIVE_SHA, 'bytes': 78886076} and
              archive_pin['bytes'] == export['bytes'], 'Actual archive bytes/hash differ')
    c.require(checksum.read_bytes() == (ARCHIVE_SHA + '  ' + archive.name + '\n').encode('ascii'),
              'Exact ASCII basename/single-LF checksum required')
    c.require(shutil.disk_usage(outputs).free >= PAYLOAD_CAP + 4 * 1024**3, 'Import reserve unavailable')
    c.require(c.sha256(stage / 'protocol.json') == PROTOCOL, 'Frozen stage protocol differs')
    protocol = c.read_json(stage / 'protocol.json')
    c.verify_protocol(protocol)
    final = c.read_json(final_path)
    c.require(final['complete'] is True and final['protocol_sha256'] == PROTOCOL and
              final['archive_sha256'] == ARCHIVE_SHA and final['archive_bytes'] == archive_pin['bytes'] and
              final['export_receipt_sha256'] == c.sha256(export_path) and
              final['export_receipt_bytes'] == export_path.stat().st_size and
              final['checksum_sha256'] == c.sha256(checksum), 'Final companion identities differ')
    log = base64.b64decode(final['export_log_base64'], validate=True)
    c.require(len(log) == final['export_log_bytes'] <= 2 * 1024**2 and
              hashlib.sha256(log).hexdigest() == final['export_log_sha256'], 'Exact final export-log bytes differ')
    finished = ( __import__('json').dumps(final['finished_json'], indent=2, allow_nan=False) + '\n').encode('utf-8')
    c.require(hashlib.sha256(finished).hexdigest() == final['finished_json_sha256'] and
              final['finished_json']['stages'] == final['stages'], 'Canonical finished receipt reconstruction differs')
    DEST.mkdir()
    hashes, sizes, folded = {}, {}, set()
    total = metadata = 0
    before = archive.stat()
    strict = StrictGzip(archive)
    try:
        with tarfile.open(fileobj=strict, mode='r|', errorlevel=2) as tar:
            for member in tar:
                name = member.name
                canonical(name)
                c.require(name.casefold() not in folded and member.isfile() and member.type == tarfile.REGTYPE and
                          member.linkname == '' and member.uid == member.gid == member.mtime == 0 and
                          member.mode == 0o644 and set(member.pax_headers) <= {'path'} and
                          ('path' not in member.pax_headers or member.pax_headers['path'] == name),
                          'Duplicate/nonregular/unexpected TAR namespace or metadata')
                folded.add(name.casefold())
                c.require(len(folded) <= 5000 and type(member.size) is int and member.size >= 0, 'File/count cap exceeded')
                suffix = Path(name).suffix.lower()
                c.require(suffix not in {'.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.bin', '.safetensors', '.npz'},
                          'Checkpoint/pickle/unplanned binary forbidden')
                cap = 256 * 1024**2 if suffix == '.npy' else 3 * 512 * 512 + 4096 if suffix == '.png' else 8 * 1024**2
                c.require(member.size <= cap, 'Per-file byte cap exceeded')
                if suffix not in {'.npy', '.png'}:
                    metadata += member.size
                    c.require(metadata <= 128 * 1024**2, 'Metadata cap exceeded')
                total += member.size
                c.require(total <= PAYLOAD_CAP and (name in protocol['files_sha256'] or name == 'protocol.json' or
                          name.startswith('outputs/normal_tone_v1/') or name in {
                              'outputs/supervision/preflight.log', 'outputs/supervision/diagnostic.log',
                              'outputs/supervision/run_receipt.json', 'outputs/supervision/export_invocation.json'}),
                          'Unbound return namespace or payload cap')
                target = c.safe_path(DEST, name, must_exist=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                source = tar.extractfile(member)
                remaining = member.size
                with target.open('xb') as destination:
                    while remaining:
                        block = source.read(min(remaining, 1024**2))
                        c.require(block, 'Short member payload')
                        destination.write(block)
                        digest.update(block)
                        remaining -= len(block)
                hashes[name], sizes[name] = digest.hexdigest(), member.size
            while True:
                padding = tar.fileobj.read(1024**2)
                if not padding:
                    break
                c.require(not any(padding), 'Nonzero bytes beyond TAR end')
        c.require(strict.ended and strict.digest.hexdigest() == ARCHIVE_SHA and
                  strict.compressed == archive_pin['bytes'], 'Gzip CRC/whole-stream digest/end differs')
    finally:
        strict.close()
    after = archive.stat()
    c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
              (after.st_size, after.st_mtime_ns, after.st_ino), 'Archive changed during import')
    c.require(hashes['protocol.json'] == PROTOCOL and hashes[MANIFEST] == export['manifest_sha256'],
              'Protocol/manifest fingerprint differs')
    manifest = c.read_json(DEST / MANIFEST)
    c.require(manifest['format'] == 'codeformer-normal-tone-return-v1' and
              manifest['protocol_sha256'] == PROTOCOL and manifest['dependencies'] == protocol['dependencies'] and
              manifest['parent_roots'] == protocol['parent_roots'] and manifest['model_qualified'] is False,
              'Manifest dependency identity differs')
    c.require(set(hashes) == set(manifest['files_sha256']) | {MANIFEST} and
              all(hashes[name] == digest and sizes[name] == manifest['files_bytes'][name]
                  for name, digest in manifest['files_sha256'].items()), 'Exact complete manifest mismatch')
    c.require(len(hashes) == export['regular_members'] == 199 and total == export['payload_bytes'] == 91344209,
              'Actual export file/byte census differs')
    for name, digest in protocol['files_sha256'].items():
        c.require(hashes[name] == digest and sizes[name] == protocol['files_bytes'][name] and
                  (DEST / name).read_bytes() == (stage / name).read_bytes(), 'Frozen source bytes differ')
    failure = c.read_json(DEST / 'outputs/normal_tone_v1/failure.json')
    expected = dict.fromkeys(c.COUNTS, 0)
    expected.update(candidate_source_forwards=16, coarse_gradient_queries=16, centered_normal_gradient_queries=8)
    c.require(failure['protocol_sha256'] == PROTOCOL and failure['counts'] == expected and
              failure['actual_disposable_Adam_steps'] == 0 and failure['model_checkpoint_written'] is False and
              not failure['optimizer_inflight_at_failure'], 'Terminal failure/count identity differs')
    c.require(final['stages'][:2] == c.read_json(DEST / 'outputs/supervision/run_receipt.json')['stages'],
              'Final versus archived supervisor stages differ')
    receipt = {'complete': True, 'format': 'codeformer-normal-tone-local-import-v1',
        'UTC': datetime.now(timezone.utc).isoformat(), 'archive_sha256': ARCHIVE_SHA,
        'archive_bytes': archive_pin['bytes'], 'checksum': pin(checksum), 'export_receipt': pin(export_path),
        'supervisor_final': pin(final_path), 'protocol_sha256': PROTOCOL, 'manifest_sha256': hashes[MANIFEST],
        'import_root': str(DEST), 'source_sha256': c.sha256(__file__), 'regular_files': len(hashes),
        'payload_bytes': total, 'metadata_bytes': metadata, 'expanded_tar_bytes': strict.expanded,
        'files_sha256': hashes, 'files_bytes': sizes, 'all20static_files_exact': True,
        'single_gzip_CRC_exact_manifest_and_companions_pass': True,
        'finished_JSON_canonical_bytes_reconstructed_exactly': True,
        'actual_returned_calls': expected, 'actual_local_neural_gradient_optimizer_VM_calls': 0,
        'returned_code_executed': False, 'checkpoint_loaded': False, 'model_qualified': False}
    c.write_new_json(receipt_path, receipt)
    print({'complete': True, 'files': len(hashes), 'payload_bytes': total, 'archive_sha256': ARCHIVE_SHA,
           'training_updates': 0, 'checkpoint_created': False})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        REVIEW.mkdir(exist_ok=True)
        path = REVIEW / 'import_failure_v1.json'
        if not path.exists():
            c.write_new_json(path, {'complete': False, 'error': str(error), 'traceback': traceback.format_exc(),
                                  'source_sha256': c.sha256(__file__), 'partial_import_preserved': DEST.exists()})
        raise

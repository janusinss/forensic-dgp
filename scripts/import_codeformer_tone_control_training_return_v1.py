"""Import the exact finite return as inert files; never execute returned code."""
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_tone_control_training_v1 import contract as c
from research.codeformer_tone_control_training_v1.transport import canonical, checkpoint_kind

PREFIX = c.PREFIX
PROTOCOL = 'f10ce6d52c2ec259632750af2d3afbdf97c5c925b99849fae2a2c4d9cd04a5b1'
ARCHIVE_SHA = '6a4386f84ebc27d6201ad188bfb19dc36bdb0a6fbab3c6f77c0662ecd4e547bf'
MANIFEST = c.OUTPUTS + '/export_manifest.json'
REVIEW = ROOT / 'outputs/codeformer_tone_control_training_v1_return_review'
DEST = ROOT / 'outputs/codeformer_tone_control_training_v1_return'
PAYLOAD_CAP = c.LIMITS['result_payload_bytes']
TAR_CAP = PAYLOAD_CAP + 9000 * 8192 + 10240


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
                      'Single gzip frame without trailing data required')
            self.ended = True
            return
        payload = self.pending
        if not payload:
            payload = self.source.read(1024 ** 2)
            c.require(payload, 'Truncated gzip frame')
            self.digest.update(payload)
            self.compressed += len(payload)
        output = self.decoder.decompress(payload, 4 * 1024 ** 2)
        self.pending = self.decoder.unconsumed_tail
        self.expanded += len(output)
        c.require(self.expanded <= TAR_CAP, 'Expanded TAR cap exceeded')
        self.buffer.extend(output)

    def read(self, size=-1):
        c.require(type(size) is int and 0 <= size <= 4 * 1024 ** 2, 'Bounded stream read required')
        while len(self.buffer) < size and not self.ended:
            self.fill()
        count = min(size, len(self.buffer))
        result = bytes(self.buffer[:count]); del self.buffer[:count]
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
    exported = c.read_json(export_path)
    c.require(exported['format'] == 'codeformer-tone-control-export-v1' and exported['complete'] is True and
              exported['protocol_sha256'] == PROTOCOL and exported['archive_sha256'] == ARCHIVE_SHA and
              exported['manifest_member'] == MANIFEST and exported['model_qualified'] is False,
              'Exact export companion identity required')
    archive_pin = pin(archive)
    c.require(archive_pin == {'sha256': ARCHIVE_SHA, 'bytes': 2858634754} and
              exported['bytes'] == archive_pin['bytes'], 'Actual archive hash/bytes differ')
    c.require(checksum.read_bytes() == (ARCHIVE_SHA + '  ' + archive.name + '\n').encode('ascii'),
              'Exact ASCII basename and single-LF checksum required')
    c.require(shutil.disk_usage(outputs).free >= PAYLOAD_CAP + 4 * 1024 ** 3, 'Import reserve unavailable')
    c.require(c.sha256(stage / 'protocol.json') == PROTOCOL, 'Frozen stage protocol differs')
    protocol = c.read_json(stage / 'protocol.json'); c.verify_protocol(protocol)
    final = c.read_json(final_path)
    c.require(final['complete'] is True and final['protocol_sha256'] == PROTOCOL and
              final['archive_sha256'] == ARCHIVE_SHA and final['archive_bytes'] == archive_pin['bytes'] and
              final['export_receipt_sha256'] == c.sha256(export_path) and
              final['export_receipt_bytes'] == export_path.stat().st_size and
              final['checksum_sha256'] == c.sha256(checksum), 'External final companion differs')
    log = base64.b64decode(final['export_log_base64'], validate=True)
    c.require(len(log) == final['export_log_bytes'] <= 2 * 1024 ** 2 and
              hashlib.sha256(log).hexdigest() == final['export_log_sha256'], 'External final export log differs')
    finished = (json.dumps(final['finished_json'], indent=2, allow_nan=False) + '\n').encode('utf-8')
    c.require(hashlib.sha256(finished).hexdigest() == final['finished_json_sha256'] and
              final['finished_json']['stages'] == final['stages'], 'Canonical final receipt differs')
    DEST.mkdir()
    hashes, sizes, folded = {}, {}, set()
    total = metadata = 0
    before = archive.stat(); strict = StrictGzip(archive)
    try:
        with tarfile.open(fileobj=strict, mode='r|', errorlevel=2) as tar:
            for member in tar:
                name = member.name; canonical(name)
                c.require(name.casefold() not in folded and member.isfile() and member.type == tarfile.REGTYPE and
                          not member.pax_headers and member.linkname == '' and
                          member.uid == member.gid == member.mtime == 0 and member.mode == 0o644,
                          'Duplicate, linked, nonregular or unexpected TAR member')
                folded.add(name.casefold())
                c.require(len(folded) <= 9000 and type(member.size) is int and member.size >= 0, 'Member/count cap exceeded')
                suffix, kind = Path(name).suffix.lower(), checkpoint_kind(name)
                c.require(suffix not in {'.pth', '.ckpt', '.pkl', '.pickle', '.bin', '.safetensors', '.npz'} and
                          (suffix != '.pt' or kind == 'final') and kind != 'temporary', 'Unexpected binary checkpoint')
                cap = 384 * 1024 ** 2 if kind == 'final' else 16 * 1024 ** 2 if suffix == '.npy' else \
                    c.LIMITS['one256PNG_bytes'] if suffix == '.png' else 8 * 1024 ** 2
                c.require(member.size <= cap, 'Per-file byte cap exceeded')
                if suffix not in {'.npy', '.png'} and kind != 'final':
                    metadata += member.size
                    c.require(metadata <= 128 * 1024 ** 2, 'Metadata cap exceeded')
                total += member.size
                c.require(total <= PAYLOAD_CAP and (name in protocol['files_sha256'] or name == 'protocol.json' or
                    name == c.LOCK_NAME or name.startswith(c.OUTPUTS + '/') or name in {
                    'outputs/supervision/preflight.log', 'outputs/supervision/diagnostic.log',
                    'outputs/supervision/run_receipt.json', 'outputs/supervision/export_invocation.json'}),
                    'Unbound output namespace or aggregate payload exceeded')
                target = c.safe_path(DEST, name, must_exist=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, source, remaining = hashlib.sha256(), tar.extractfile(member), member.size
                with target.open('xb') as destination:
                    while remaining:
                        block = source.read(min(remaining, 1024 ** 2))
                        c.require(block, 'Short member payload')
                        destination.write(block); digest.update(block); remaining -= len(block)
                hashes[name], sizes[name] = digest.hexdigest(), member.size
            while True:
                padding = tar.fileobj.read(1024 ** 2)
                if not padding: break
                c.require(not any(padding), 'Nonzero bytes beyond TAR end')
        c.require(strict.ended and strict.digest.hexdigest() == ARCHIVE_SHA and
                  strict.compressed == archive_pin['bytes'], 'Full gzip frame/CRC/digest/end differs')
    finally:
        strict.close()
    after = archive.stat()
    c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
              (after.st_size, after.st_mtime_ns, after.st_ino), 'Archive changed during import')
    c.require(hashes['protocol.json'] == PROTOCOL and hashes[MANIFEST] == exported['manifest_sha256'],
              'Manifest/protocol fingerprint differs')
    manifest = c.read_json(DEST / MANIFEST)
    c.require(set(c.CHECKPOINT_PATHS).issubset(hashes), 'All three genuine composite checkpoints required')
    c.require(manifest['format'] == 'codeformer-tone-control-return-v1' and
              manifest['protocol_sha256'] == PROTOCOL and manifest['model_qualified'] is False and
              manifest['parent_dir'] == protocol['parent_dir'] and
              manifest['parent_protocol_sha256'] == protocol['parent_protocol_sha256'] and
              c.equal(manifest['dependencies'], protocol['dependencies']) and
              c.equal(manifest['retained_output_dependencies'], protocol['retained_output_dependencies']),
              'Manifest parent bindings differ')
    c.require(set(hashes) == set(manifest['files_sha256']) | {MANIFEST} and
              all(hashes[n] == digest and sizes[n] == manifest['files_bytes'][n]
                  for n, digest in manifest['files_sha256'].items()), 'Complete manifest mismatch')
    c.require(len(hashes) == exported['regular_members'] == 8533 and total == exported['payload_bytes'] == 3209400295,
              'Actual return file/byte census differs')
    for name, digest in protocol['files_sha256'].items():
        c.require(hashes[name] == digest and sizes[name] == protocol['files_bytes'][name] and
                  (DEST / name).read_bytes() == (stage / name).read_bytes(), 'Frozen static bytes differ')
    failure = c.read_json(DEST / (c.OUTPUTS + '/failure.json'))
    c.require(failure['protocol_sha256'] == PROTOCOL and failure['counts'] == c.COUNTS and
              failure['completed_head_updates'] == 50 and failure['restorer_updates_this_run'] == 0 and
              failure['partial_microbatches'] == 0 and failure['head_optimizer_inflight'] is False,
              'Actual terminal count/failure identity differs')
    c.require(final['stages'][:2] == c.read_json(DEST / 'outputs/supervision/run_receipt.json')['stages'],
              'Archived/final supervisor stages differ')
    receipt = {'complete': True, 'format': 'codeformer-tone-control-local-import-v1',
        'UTC': datetime.now(timezone.utc).isoformat(), 'archive_sha256': ARCHIVE_SHA,
        'archive_bytes': archive_pin['bytes'], 'checksum': pin(checksum), 'export_receipt': pin(export_path),
        'supervisor_final': pin(final_path), 'protocol_sha256': PROTOCOL, 'manifest_sha256': hashes[MANIFEST],
        'import_root': str(DEST), 'source_sha256': c.sha256(__file__), 'regular_files': len(hashes),
        'payload_bytes': total, 'metadata_bytes': metadata, 'expanded_tar_bytes': strict.expanded,
        'files_sha256': hashes, 'files_bytes': sizes, 'all24static_files_exact': True,
        'single_gzip_CRC_exact_manifest_and_companions_pass': True,
        'finished_JSON_canonical_bytes_reconstructed_exactly': True,
        'actual_returned_counts': failure['counts'], 'completed_head_updates': 50, 'restorer_updates_this_run': 0,
        'actual_local_neural_gradient_optimizer_VM_calls': 0, 'returned_code_executed': False,
        'checkpoint_loaded': False, 'candidate_checkpoints_imported': list(c.CHECKPOINT_PATHS), 'model_qualified': False}
    c.write_new_json(receipt_path, receipt)
    print({'complete': True, 'files': len(hashes), 'payload_bytes': total,
           'head_updates': 50, 'restorer_updates': 0, 'model_qualified': False})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        REVIEW.mkdir(exist_ok=True)
        path = REVIEW / 'import_failure_v1.json'
        if not path.exists():
            c.write_new_json(path, {'complete': False, 'traceback': traceback.format_exc(),
                'source_sha256': c.sha256(__file__), 'partial_import_preserved': DEST.exists()})
        raise

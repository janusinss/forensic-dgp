"""Hash-bound bounded import; never execute returned Python or load a checkpoint."""
from datetime import datetime, timezone
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
from research.codeformer_loss_response_v1 import contract as c
from research.codeformer_loss_response_v1.transport import canonical

PREFIX = 'cctv-codeformer-loss-response-v1'
PROTOCOL = '23c19c394e34261861b8853cfd60aa60d1cfaefc91b12e92a2f8b8dce85112ab'
ARCHIVE_SHA = 'b0e2b6d0aa245fcebe334c21c1b50cfbfde3ca65d494bb1d1fd8a9d17730de86'
MANIFEST = 'outputs/loss_response_v1/export_manifest.json'
PAYLOAD_CAP = 4 * 1024**3
TAR_CAP = PAYLOAD_CAP + 12000 * 8192 + 10240
REVIEW = ROOT / 'outputs/codeformer_loss_response_v1_return_review'
DEST = ROOT / 'outputs/codeformer_loss_response_v1_return'


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
                      'Only one gzip member permitted; no trailing compressed bytes')
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
        c.require(self.expanded <= TAR_CAP, 'Expanded TAR exceeds fixed cap')
        self.buffer.extend(output)

    def read(self, size=-1):
        c.require(type(size) is int and 0 <= size <= 4 * 1024**2, 'Bounded read required')
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
    stage = outputs / 'cctv_codeformer_loss_response_v1_vm_r1'
    c.require(not DEST.exists() and not (REVIEW / 'return_import_v1.json').exists(), 'Preserve prior import')
    REVIEW.mkdir(exist_ok=True)
    export = c.read_json(export_path)
    c.require(export['format'] == 'codeformer-loss-response-export-v1' and export['complete'] is True and
              export['protocol_sha256'] == PROTOCOL and export['archive_sha256'] == ARCHIVE_SHA and
              export['manifest_member'] == MANIFEST and export['model_qualified'] is False,
              'Returned export identity differs')
    original = pin(archive)
    c.require(original == {'sha256': ARCHIVE_SHA, 'bytes': export['bytes']} and
              original['bytes'] <= 9 * 1024**3 // 2, 'Fresh archive bytes/hash differ')
    c.require(checksum.read_bytes() == (ARCHIVE_SHA + '  ' + archive.name + '\n').encode('ascii'),
              'Exact portable single-LF checksum differs')
    c.require(shutil.disk_usage(outputs).free >= PAYLOAD_CAP + 4 * 1024**3, 'Require import allowance plus4GiB reserve')
    c.require(c.sha256(stage / 'protocol.json') == PROTOCOL, 'Frozen local protocol differs')
    protocol = c.read_json(stage / 'protocol.json')
    c.verify_protocol(protocol)
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
                          member.linkname == '' and member.uid == member.gid == member.mtime == 0 and member.mode == 0o644,
                          'Duplicate/link/nonregular or unexpected header')
                c.require(set(member.pax_headers) <= {'path'} and
                          ('path' not in member.pax_headers or member.pax_headers['path'] == name), 'Unexpected PAX field')
                folded.add(name.casefold())
                c.require(len(folded) <= 12000 and type(member.size) is int and member.size >= 0, 'File/count cap exceeded')
                suffix = Path(name).suffix.lower()
                c.require(suffix not in {'.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.bin', '.safetensors', '.npz'},
                          'Checkpoint/pickle/unplanned binary forbidden')
                if suffix == '.npy':
                    c.require(member.size <= 256 * 1024**2, 'Array size cap exceeded')
                elif suffix == '.png':
                    c.require(member.size <= 3 * 512 * 512 + 4096, 'PNG size cap exceeded')
                else:
                    metadata += member.size
                    c.require(member.size <= 8 * 1024**2 and metadata <= 128 * 1024**2, 'Metadata cap exceeded')
                total += member.size
                c.require(total <= PAYLOAD_CAP and
                          (name in protocol['files_sha256'] or name == 'protocol.json' or
                           name.startswith('outputs/loss_response_v1/') or
                           name in {'outputs/supervision/preflight.log', 'outputs/supervision/diagnostic.log',
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
                        c.require(block, 'Short member read')
                        destination.write(block)
                        digest.update(block)
                        remaining -= len(block)
                hashes[name], sizes[name] = digest.hexdigest(), member.size
                if len(hashes) % 500 == 0:
                    print({'imported_files': len(hashes), 'payload_bytes': total}, flush=True)
            # Consume TAR stream read-ahead and gzip trailer. Extra nonzero data,
            # additional gzip frames and truncated/invalid CRCs are rejected.
            while True:
                padding = tar.fileobj.read(1024**2)
                if not padding:
                    break
                c.require(not any(padding), 'Nonzero bytes beyond TAR terminator')
        c.require(strict.ended and strict.digest.hexdigest() == ARCHIVE_SHA and strict.compressed == original['bytes'],
                  'Compressed stream hash/count/end differs')
    finally:
        strict.close()
    after = archive.stat()
    c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
              (after.st_size, after.st_mtime_ns, after.st_ino), 'Archive changed during import')
    c.require(hashes['protocol.json'] == PROTOCOL and hashes[MANIFEST] == export['manifest_sha256'],
              'Returned protocol/manifest hash differs')
    returned_protocol = c.read_json(DEST / 'protocol.json')
    c.require(returned_protocol == protocol, 'Returned protocol semantics differ')
    manifest = c.read_json(DEST / MANIFEST)
    c.require(manifest['format'] == 'codeformer-loss-response-return-v1' and manifest['protocol_sha256'] == PROTOCOL and
              manifest['parent_protocol_sha256'] == c.PARENT_PROTOCOL_SHA and
              manifest['dependencies'] == protocol['dependencies'] and manifest['model_qualified'] is False,
              'Returned manifest/dependency identity differs')
    c.require(set(hashes) == set(manifest['files_sha256']) | {MANIFEST} and
              all(hashes[n] == d and sizes[n] == manifest['files_bytes'][n] for n,d in manifest['files_sha256'].items()),
              'Every imported member must match exact manifest')
    c.require(len(hashes) == export['regular_members'] == 4609 and total == export['payload_bytes'] == 2881853749,
              'Export/actual namespace and bytes disagree')
    for name, digest in protocol['files_sha256'].items():
        c.require(hashes[name] == digest and sizes[name] == protocol['files_bytes'][name] and
                  (DEST / name).read_bytes() == (stage / name).read_bytes(), 'Frozen static source differs')
    completion = c.read_json(DEST / 'outputs/loss_response_v1/completion.json')
    c.require(completion['protocol_sha256'] == PROTOCOL and completion['complete'] is True and
              completion['counts'] == c.COUNTS and completion['actual_disposable_Adam_steps'] == 18 and
              completion['committed_training_updates'] == 0, 'Terminal receipt identity/count differs')
    receipt = {'complete': True, 'format': 'codeformer-loss-response-local-import-v1',
        'UTC': datetime.now(timezone.utc).isoformat(), 'archive': str(archive), 'archive_sha256': ARCHIVE_SHA,
        'archive_bytes': original['bytes'], 'checksum': pin(checksum), 'export_receipt': pin(export_path),
        'protocol_sha256': PROTOCOL, 'manifest_sha256': hashes[MANIFEST], 'import_root': str(DEST),
        'source_sha256': c.sha256(__file__), 'regular_files': len(hashes), 'payload_bytes': total,
        'metadata_bytes': metadata, 'expanded_tar_bytes': strict.expanded,
        'single_gzip_frame_CRC_and_exact_payload_hashes_pass': True,
        'files_sha256': hashes, 'files_bytes': sizes, 'all17static_files_exact': True,
        'actual_local_neural_gradient_optimizer_VM_calls': 0, 'returned_code_executed': False,
        'checkpoint_loaded': False, 'model_qualified': False}
    c.write_new_json(REVIEW / 'return_import_v1.json', receipt)
    print({'complete': True, 'regular_files': len(hashes), 'payload_bytes': total, 'archive_sha256': ARCHIVE_SHA})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        REVIEW.mkdir(exist_ok=True)
        failure = REVIEW / 'import_failure_v1.json'
        if not failure.exists():
            c.write_new_json(failure, {'complete': False, 'error': str(error), 'traceback': traceback.format_exc(),
                'source_sha256': c.sha256(__file__), 'partial_import_preserved': DEST.exists(),
                'actual_local_neural_gradient_optimizer_VM_calls': 0})
        raise

"""Seal a reviewed inert upload; no model import, inference, VM or training."""
import ast
import gzip
import hashlib
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import transport as t

PREP = ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation'
STAGE = ROOT / 'outputs' / t.VM_NAME


class CopyReader:
    def __init__(self, handle, deadline):
        self.handle, self.deadline = handle, deadline
        self.digest = hashlib.sha256(); self.size = 0

    def read(self, amount):
        t.require(time.monotonic() <= self.deadline, 'Source seal600second cap exceeded')
        value = self.handle.read(amount)
        self.digest.update(value); self.size += len(value)
        return value


def main():
    started = time.monotonic()
    def budget():
        t.require(time.monotonic() - started <= 600, 'Complete seal600second cap exceeded')
    stage_receipt = PREP / 'packet_stage_v1.json'
    stage = t.read_json(stage_receipt)
    t.require(stage.get('complete') is True and stage.get('static_transfer_verified') is True
              and Path(stage['stage']).resolve(strict=True) == STAGE.resolve(strict=True),
              'Exact complete inert stage required')
    protocol_sha = stage['protocol_sha256']
    expected = stage['actual_stage_pin_map']
    protocol_path = t.regular(STAGE, 'protocol.json')
    protocol_pin = t.file_pin(protocol_path, check=budget)
    t.require(protocol_pin['sha256'] == protocol_sha, 'Staged protocol changed')
    source_bindings, dependencies = t._bindings_from_protocol(t.read_json(protocol_path))
    t.require(dependencies == {} and expected == {**source_bindings, 'protocol.json': protocol_pin},
              'Exact self-contained static protocol binding map differs')
    t.require(set(t.inventory(STAGE)) == set(expected), 'Staged namespace differs')
    peer = t.read_json(PREP / 'packet_independent_preseal_v1.json')
    t.require(peer.get('complete') is True and peer.get('unresolved_findings') == []
              and peer.get('stage_receipt_sha256') == t.file_pin(stage_receipt, check=budget)['sha256']
              and peer.get('protocol_sha256') == protocol_sha and peer.get('files') == expected,
              'Exact independent physical stage review required before sealing')
    for name, pin in expected.items():
        budget()
        path = t.regular(STAGE, name)
        t.require(t.file_pin(path, check=budget) == pin, 'Changed staged file: ' + name)
        if name.endswith('.py'):
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    for name, source in stage['local_source_map'].items():
        if 'source' in source:
            path = t.no_links(ROOT / t.canonical(source['source']))
            if source.get('readonly_historical_original_weight_hardlink_exception') is True:
                state = path.lstat()
                t.require(name == 'weights/codeformer_original.pth' and
                          path == ROOT / 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth'
                          and state.st_nlink == 3 and stat.S_ISREG(state.st_mode),
                          'Only exact read-only original asset hardlink exception allowed')
                with path.open('rb') as handle:
                    reader = CopyReader(handle, started + 600)
                    while reader.read(4*t.MiB): pass
                actual = {'bytes': reader.size, 'sha256': reader.digest.hexdigest()}
                after = path.stat()
                t.require((state.st_ino, state.st_size, state.st_mtime_ns) ==
                          (after.st_ino, after.st_size, after.st_mtime_ns) and
                          actual == t.WEIGHT_ASSETS[name], 'Original historical asset changed')
            else:
                actual = t.file_pin(path, check=budget)
            t.require(actual == source['pin'] == expected[name],
                      'Current source changed after independent stage review: ' + name)
    archive = ROOT / 'outputs' / (t.PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    receipt_path = PREP / 'packet_seal_v1.json'
    t.require(not any(p.exists() or p.is_symlink() for p in (archive, checksum, receipt_path)),
              'Preserve any prior or partial archive/seal; no automatic retry')
    source_size = sum(v['bytes'] for v in expected.values())
    t.require(source_size <= t.LIMITS['source_packet_bytes'] and
              shutil.disk_usage(ROOT).free >= source_size + 4*t.GiB,
              'Full source archive plus4GiB local reserve required')
    with archive.open('xb') as handle:
        with gzip.GzipFile(filename='', fileobj=handle, mode='wb', mtime=0, compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                for name, pin in sorted(expected.items()):
                    t.require(shutil.disk_usage(ROOT).free >= 4*t.GiB, 'Local seal reserve crossed')
                    info = tarfile.TarInfo(t.VM_NAME + '/' + name)
                    info.size, info.mode, info.mtime = pin['bytes'], 0o644, 0
                    info.uid = info.gid = 0; info.uname = info.gname = ''
                    with t.regular(STAGE, name).open('rb') as source:
                        reader = CopyReader(source, started + 600)
                        tar.addfile(info, reader)
                        t.require(reader.size == pin['bytes'] and reader.digest.hexdigest() == pin['sha256'],
                                  'Source changed during physical archive write')
        handle.flush()
        import os
        os.fsync(handle.fileno())
    pin = t.file_pin(archive, check=budget)
    physical = t.verify_archive(archive,
        {t.VM_NAME + '/' + name: value for name, value in expected.items()}, archive_pin=pin, check=budget)
    t.require(time.monotonic() - started <= 600, 'Complete sealing/readback600second cap exceeded')
    with checksum.open('xb') as handle:
        handle.write(t.checksum_bytes(archive.name, pin['sha256']))
    receipt = {'format': 'codeformer-filter-fusion-training-source-seal-v1', 'complete': True,
        'archive': {'path': archive.relative_to(ROOT).as_posix(), **pin},
        'checksum': {'path': checksum.relative_to(ROOT).as_posix(), **t.file_pin(checksum)},
        'protocol_sha256': protocol_sha, 'static_files': len(expected), 'payload_bytes': source_size,
        'stage_receipt': t.file_pin(stage_receipt), 'physical_readback': physical,
        'seconds': time.monotonic() - started, 'source': t.file_pin(Path(__file__)),
        'neural_gradient_optimizer_VM_calls': 0, 'new_model_trained': False,
        'automatic_launch': False, 'model_qualified': False,
        'training_or_archive_completion_not_model_acceptance': True}
    t.write_new_json(receipt_path, receipt)
    print(t.json_bytes(receipt).decode(), flush=True)


if __name__ == '__main__': main()

"""Seal a reviewed source-only training packet; no execution or VM connection."""
import argparse
import gzip
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_training_v1 import contract as c
from research.codeformer_normal_tone_training_v1 import transport as t
from scripts.seal_codeformer_normal_tone_v1 import verify_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    stage = ROOT / 'outputs' / c.VM_NAME
    review = ROOT / 'outputs/codeformer_normal_tone_training_v1_preparation'
    ready_path = review / 'readiness_v1.json'
    ready = c.read_json(ready_path)
    p = c.read_json(stage / 'protocol.json')
    c.verify_protocol(p)
    c.require(c.sha256(stage / 'protocol.json') == args.protocol_sha == ready['protocol_sha256'] and
              ready['complete'] is True and ready['independent_source_review_complete'] is True and
              ready['tests_failed'] == 0 and ready['local_neural_gradient_optimizer_VM_calls'] == 0,
              'Closed exact-protocol readiness required')
    wanted = {**p['files_sha256'], 'protocol.json': args.protocol_sha}
    sizes = {**p['files_bytes'], 'protocol.json': (stage / 'protocol.json').stat().st_size}
    c.require(wanted == ready['source_files'] and sizes == ready['source_bytes'], 'Readiness namespace differs')
    for name, digest in ready['evidence_sha256'].items():
        c.safe_relative(name)
        c.require(c.sha256(ROOT / name) == digest, 'Readiness evidence changed: ' + name)
    files = t._walk(stage)
    c.require(set(files) == set(wanted) and len(files) <= 64 and sum(sizes.values()) <= 16 * 1024**2,
              'Exact bounded source-only namespace required')
    for name, path in files.items():
        c.require(Path(name).suffix in {'.py', '.json', '.md'} and not name.startswith('outputs/') and
                  path.stat().st_size == sizes[name] and c.sha256(path) == wanted[name],
                  'Frozen source-only member differs: ' + name)
    for name, digest in ready['current_sources_sha256'].items():
        c.safe_relative(name)
        c.require(c.sha256(ROOT / name) == digest, 'Current reviewed source changed: ' + name)
    archive = ROOT / 'outputs' / (c.PREFIX + '-execution.tar.gz')
    part = archive.with_name(archive.name + '.part')
    checksum = Path(str(archive) + '.sha256')
    receipt = review / 'sealed_packet_v1.json'
    for path in (archive, part, checksum, receipt):
        c.require(not path.exists() and not path.is_symlink(), 'Preserve prior seal: ' + str(path))
    with part.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as packed:
                for name in sorted(files):
                    info = tarfile.TarInfo(c.VM_NAME + '/' + name)
                    info.size, info.mode = sizes[name], 0o644
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with files[name].open('rb') as source:
                        packed.addfile(info, source)
        raw.flush()
        os.fsync(raw.fileno())
    readback = verify_archive(part, c.VM_NAME, wanted, sizes)
    digest = c.sha256(part)
    c.require(part.stat().st_size <= 16 * 1024**2, 'Source archive cap exceeded')
    os.replace(part, archive)
    with checksum.open('xb') as stream:
        stream.write((digest + '  ' + archive.name + '\n').encode('ascii'))
    c.write_new_json(receipt, {'complete': True, 'status': 'PREPARED_UNRUN',
        'protocol_sha256': args.protocol_sha, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
        'readiness_receipt_sha256': c.sha256(ready_path), 'source_files': wanted, 'source_bytes': sizes,
        'readback': readback, 'archive': str(archive), 'checksum_sha256': c.sha256(checksum),
        'sealer_source_sha256': c.sha256(__file__), 'neural_gradient_optimizer_VM_calls': 0,
        'actual_training_launched': False, 'candidate_checkpoint_created': False})
    print({'complete': True, 'archive_sha256': digest, 'protocol_sha256': args.protocol_sha,
           'bytes': archive.stat().st_size, 'members': len(files), 'training_launched': False})


if __name__ == '__main__':
    main()

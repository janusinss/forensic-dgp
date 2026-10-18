"""CPU-only source-packet seal and byte-exact archive readback; never launch."""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_tone_control_training_v1 import contract as c, budget, data


def main():
    preparation = ROOT / 'outputs/codeformer_tone_control_training_v1_preparation'
    stage_receipt = preparation / 'stage_v1.json'
    stage = c.read_json(stage_receipt)
    folder = Path(stage['stage'])
    p = c.read_json(folder / 'protocol.json'); c.verify_protocol(p)
    c.require(c.sha256(folder / 'protocol.json') == stage['protocol_sha256'], 'Staged protocol changed')
    preseal = c.read_json(preparation / 'independent_preseal_source_v1.json')
    c.require(preseal['complete'] is True and not preseal['unresolved_findings'] and
              preseal['protocol_sha256'] == stage['protocol_sha256'] and
              preseal['stage_receipt_sha256'] == c.sha256(stage_receipt) and
              preseal['stage_member_sha256'] == {**p['files_sha256'], 'protocol.json': stage['protocol_sha256']} and
              preseal['stage_member_bytes'] == {**p['files_bytes'], 'protocol.json': (folder / 'protocol.json').stat().st_size},
              'Current independent exact stage/source review required')
    checked = {}
    for name, expected in p['files_sha256'].items():
        path = c.safe_path(folder, name)
        c.require(c.sha256(path) == expected and path.stat().st_size == p['files_bytes'][name], 'Staged source differs')
        current = stage['current_sources'][name]
        if current is False:
            synthetic = stage['synthetic_sources'][name]
            c.require(name == 'research/__init__.py' and path.read_bytes() == b'' and
                      synthetic['packet_sha256'] == expected and synthetic['packet_bytes'] == 0 and
                      c.sha256(ROOT / synthetic['local_reference_path']) == synthetic['local_reference_sha256'],
                      'Undeclared synthetic namespace substitution')
        else:
            c.require(c.sha256(ROOT / current) == expected, 'Current executing or instruction source changed before seal')
        if name.endswith('.py'): ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
        checked[name] = expected
    for label in ('dependencies', 'retained_output_dependencies'):
        for name, pin in p[label].items():
            path = Path(stage['local_dependency_resolution'][name]); before = path.stat()
            c.require(before.st_size == pin['bytes'] and c.sha256(path) == pin['sha256'], 'Read-only parent binding differs')
            after = path.stat()
            c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                      (after.st_size, after.st_mtime_ns, after.st_ino), 'Parent changed during seal')
    index = data.load_index(folder, p, resolver=lambda name: stage['local_dependency_resolution'][name])
    c.require(index['binding_receipt']['TRAIN_cases'] == 200 and index['binding_receipt']['DEV_cases'] == 406,
              'Complete TRAIN/DEV closure required')
    ledger = budget.resource_ledger(p)
    for filename in ('independent_source_review_v1.json', 'independent_worker_review_v1.json'):
        review = c.read_json(preparation / filename)
        c.require(review['complete'] is True, 'Independent source review incomplete')
        # Independently read source maps are checked below in the separate seal audit.
    archive = ROOT / 'outputs' / (c.PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    c.require(not archive.exists() and not checksum.exists(), 'Preserve previous upload packet')
    members = sorted([*p['files_sha256'], 'protocol.json'])
    expected = {**p['files_sha256'], 'protocol.json': stage['protocol_sha256']}
    with archive.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for name in members:
                    path = folder / name
                    payload = path.read_bytes()
                    c.require(hashlib.sha256(payload).hexdigest() == expected[name], 'Stage changed during packing')
                    info = tarfile.TarInfo(c.VM_NAME + '/' + name)
                    info.size, info.mtime, info.mode = len(payload), 0, 0o644
                    tar.addfile(info, io.BytesIO(payload))
    c.require(archive.stat().st_size <= 16 * 1024**2, 'Source-only upload cap exceeded')
    rebuilt = {}
    with tarfile.open(archive, mode='r:gz') as tar:
        for item in tar:
            c.require(item.isfile() and item.name.startswith(c.VM_NAME + '/'), 'Nonregular or wrong-root archive member')
            name = item.name[len(c.VM_NAME) + 1:]; c.safe_relative(name)
            c.require(name not in rebuilt and name in expected, 'Extra/duplicate archive member')
            payload = tar.extractfile(item).read()
            c.require(len(payload) == item.size and hashlib.sha256(payload).hexdigest() == expected[name], 'Archive member differs')
            rebuilt[name] = expected[name]
    with gzip.open(archive, 'rb') as stream:
        while stream.read(1024**2): pass  # Complete CRC and end-of-stream validation.
    c.require(rebuilt == expected, 'Full regular-member archive namespace differs')
    digest = c.sha256(archive)
    with checksum.open('xb') as stream: stream.write((digest + '  ' + archive.name + '\n').encode('ascii'))
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'archive': str(archive), 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
        'protocol_sha256': stage['protocol_sha256'], 'static_members': len(members),
        'uncompressed_payload_bytes': sum((folder / name).stat().st_size for name in members),
        'members_sha256': expected, 'whole_archive_CRC_and_member_readback': True,
        'dependencies_freshly_verified': len(p['dependencies']),
        'retained_outputs_freshly_verified': len(p['retained_output_dependencies']),
        'stage_receipt_sha256': c.sha256(stage_receipt), 'source_ledger': ledger,
        'independent_preseal_review_sha256': c.sha256(preparation / 'independent_preseal_source_v1.json'),
        'all_TRAIN_DEV_bindings_checked': index['binding_receipt'],
        'seal_source_sha256': c.sha256(__file__), 'new_training_launched': False,
        'local_neural_gradient_optimizer_VM_calls': 0, 'model_qualified': False, 'status': 'PREPARED_UNRUN'}
    c.write_new_json(preparation / 'seal_v1.json', receipt)
    print({key: receipt[key] for key in ('complete', 'archive_bytes', 'archive_sha256', 'protocol_sha256', 'static_members', 'status')})


if __name__ == '__main__': main()

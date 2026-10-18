"""Seal the reviewed static source/input packet; no neural calls or VM connection."""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import io
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_fixed_state_response_v1 import contract as c, budget, data


def main():
    prep = ROOT / 'outputs/codeformer_fixed_state_response_v1_preparation'
    stage_path = prep / 'stage_v1.json'; stage = c.read_json(stage_path)
    folder = c.no_links(Path(stage['stage'])).resolve(strict=True)
    c.require(folder == (ROOT / 'outputs' / c.VM_NAME).resolve(strict=True), 'Exact reviewed stage required')
    protocol_path = folder / 'protocol.json'; p = c.read_json(protocol_path); c.verify_protocol(p)
    c.require(c.sha256(protocol_path) == stage['protocol_sha256'], 'Frozen staged protocol changed')
    expected = {**p['files_sha256'], 'protocol.json': stage['protocol_sha256']}
    expected_bytes = {**p['files_bytes'], 'protocol.json': protocol_path.stat().st_size}
    review_path = prep / 'independent_preseal_v1.json'; peer = c.read_json(review_path)
    c.require(peer['complete'] is True and peer['unresolved_findings'] == [] and
              peer['stage_receipt_sha256'] == c.sha256(stage_path) and
              peer['protocol_sha256'] == stage['protocol_sha256'] and
              peer['stage_member_sha256'] == expected and peer['stage_member_bytes'] == expected_bytes,
              'Exact independent actual-stage source/input review required')
    inventory = {path.relative_to(folder).as_posix() for path in folder.rglob('*') if path.is_file()}
    c.require(inventory == set(expected), 'Unbound stage namespace')
    for name, digest in expected.items():
        path = c.safe_path(folder, name)
        c.require(path.stat().st_size == expected_bytes[name] and c.sha256(path) == digest, 'Staged bytes changed: ' + name)
        if name.endswith('.py'): ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
        if name == 'protocol.json': continue
        current = stage['current_sources'][name]
        if current is False:
            if name == 'research/__init__.py':
                marker = stage['synthetic_sources'][name]
                c.require(path.read_bytes() == b'' and marker['packet_sha256'] == digest and marker['packet_bytes'] == 0 and
                          c.sha256(ROOT / marker['local_reference_path']) == marker['local_reference_sha256'],
                          'Undeclared empty namespace substitution')
            else:
                c.require(name in stage['generated_data_files'], 'Unbound generated input/index')
        else:
            c.require(c.sha256(ROOT / current) == digest, 'Current source/instruction changed since staging')
    resolved = stage['local_dependency_resolution']; dependency_counts = {}
    for alias, pins in p['dependencies'].items():
        c.require(stage['dependencies'][alias] == pins, 'Stage dependency set differs')
        for name, pin in pins.items():
            path = c.no_links(Path(resolved[alias][name])); before = path.stat()
            c.require(path.resolve(strict=True).is_relative_to(ROOT) and before.st_size == pin['bytes'] and
                      c.sha256(path) == pin['sha256'], 'Read-only parent identity changed')
            after = path.stat()
            c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                      (after.st_size, after.st_mtime_ns, after.st_ino), 'Parent changed during seal')
        dependency_counts[alias] = len(pins)
    index = data.load_index(folder, p, {}, resolver=lambda alias, name: Path(resolved[alias][name]))
    c.require(len(index['fitted']) == 200 and len(index['heldout']) == 64, 'Full264 source/input joins required')
    payload_bytes = sum(expected_bytes.values())
    ledger = budget.resource_ledger(len(expected), payload_bytes)
    c.require(stage['resource_ledger'] == ledger, 'Exact protocol-inclusive resource ledger changed')
    for filename in ('independent_source_closure_v1.json', 'independent_worker_review_v1.json'):
        c.require(c.read_json(prep / filename)['complete'] is True, 'Independent source review incomplete')
    archive = ROOT / 'outputs' / (c.PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    c.require(not archive.exists() and not checksum.exists() and not (prep / 'seal_v1.json').exists(),
              'Preserve every earlier/partial packet')
    with archive.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for name in sorted(expected):
                    payload = (folder / name).read_bytes()
                    c.require(len(payload) == expected_bytes[name] and hashlib.sha256(payload).hexdigest() == expected[name],
                              'Stage changed while sealing')
                    member = tarfile.TarInfo(c.VM_NAME + '/' + name)
                    member.size, member.mtime, member.mode = len(payload), 0, 0o644
                    tar.addfile(member, io.BytesIO(payload))
    c.require(archive.stat().st_size <= c.LIMITS['source_packet_bytes'], 'Compressed upload exceeds32MiB cap')
    readback = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            c.require(member.type == tarfile.REGTYPE and member.name.startswith(c.VM_NAME + '/') and
                      not member.pax_headers and member.mode == 0o644 and
                      member.uid == member.gid == member.mtime == 0 and member.linkname == '', 'Unexpected TAR entry')
            name = member.name[len(c.VM_NAME) + 1:]; c.safe_relative(name)
            c.require(name in expected and name not in readback and member.size == expected_bytes[name], 'Unexpected member namespace')
            c.require(hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[name], 'Packed member changed')
            readback[name] = expected[name]
    with gzip.open(archive, 'rb') as stream:
        while stream.read(1024 ** 2): pass
    c.require(readback == expected and not any(n in sys.modules for n in ('torch', 'tensorflow', 'jax')),
              'Full archive readback or no-framework invariant failed')
    digest = c.sha256(archive)
    with checksum.open('xb') as stream: stream.write((digest + '  ' + archive.name + '\n').encode('ascii'))
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'PREPARED_UNRUN',
        'archive': str(archive), 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
        'protocol_sha256': stage['protocol_sha256'], 'static_members': len(expected),
        'payload_bytes': payload_bytes, 'members_sha256': expected, 'members_bytes': expected_bytes,
        'stage_receipt_sha256': c.sha256(stage_path), 'independent_preseal_sha256': c.sha256(review_path),
        'dependencies_freshly_verified': dependency_counts, 'source_ledger': ledger,
        'all264_input_bindings_checked': index['binding_receipt'], 'seal_source_sha256': c.sha256(__file__),
        'CRC_and_member_readback': True, 'independent_single_frame_physical_TAR_review_pending': True,
        'local_neural_autograd_optimizer_VM_calls': 0, 'training_updates': 0, 'model_qualified': False}
    c.write_new_json(prep / 'seal_v1.json', receipt)
    print({key: receipt[key] for key in ('complete', 'status', 'archive_bytes', 'archive_sha256', 'protocol_sha256', 'static_members')})


if __name__ == '__main__': main()

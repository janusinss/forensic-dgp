"""Seal only the independently reviewed new manual-L4 source packet; no NN."""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_role_preserving_training_v1_r2 import contract as c, transport


def physical_readback(archive, expected, sizes):
    """Bounded single-frame gzip and exact canonical USTAR, including all padding."""
    decoder = zlib.decompressobj(31)
    compressed = archive.read_bytes()
    expanded = decoder.decompress(compressed, c.LIMITS['source_packet_bytes'] + 1024 * 1024)
    c.require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
              'Single complete gzip frame and bounded full EOF required')
    expanded += decoder.flush()
    c.require(len(expanded) <= c.LIMITS['source_packet_bytes'] + 1024 * 1024,
              'Expanded upload exceeds bounded source envelope')
    offset, seen = 0, {}
    for relative in sorted(expected):
        name = c.VM_NAME + '/' + relative
        member = tarfile.TarInfo(name)
        member.size, member.mode, member.mtime = sizes[relative], 0o644, 0
        member.uid = member.gid = 0
        member.uname = member.gname = ''
        header = member.tobuf(format=tarfile.USTAR_FORMAT, encoding='utf-8', errors='strict')
        c.require(len(header) == 512 and expanded[offset:offset+512] == header,
                  'Canonical USTAR header/order mismatch: ' + relative)
        offset += 512
        payload = expanded[offset:offset+sizes[relative]]
        c.require(len(payload) == sizes[relative] and hashlib.sha256(payload).hexdigest() == expected[relative],
                  'Packed payload differs: ' + relative)
        offset += sizes[relative]
        padding = (-sizes[relative]) % 512
        c.require(expanded[offset:offset+padding] == b'\0' * padding, 'Nonzero member padding')
        offset += padding
        seen[relative] = expected[relative]
    exact_size = ((offset + 1024 + 10239) // 10240) * 10240
    c.require(len(expanded) == exact_size and len(expanded)-offset >= 1024 and
              expanded[offset:] == b'\0' * (len(expanded)-offset),
              'Double EOA and complete record padding/full EOF required')
    return {'complete': True, 'single_gzip_CRC_full_EOF': True,
            'canonical_USTAR_headers_order_members_padding_double_EOA': True,
            'expanded_bytes': len(expanded), 'members': len(seen)}


def main():
    prep = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_preparation'
    stage_file = prep / 'stage_v1.json'
    stage = c.read_json(stage_file)
    folder = c.no_links(Path(stage['stage'])).resolve(strict=True)
    c.require(folder == (ROOT/'outputs'/c.VM_NAME).resolve(strict=True), 'Exact new stage required')
    protocol_path = folder/'protocol.json'
    p = c.read_json(protocol_path)
    c.verify_protocol(p)
    c.require(c.sha256(protocol_path) == stage['protocol_sha256'], 'Staged protocol changed')
    expected = {**p['files_sha256'], 'protocol.json': stage['protocol_sha256']}
    sizes = {**p['files_bytes'], 'protocol.json': protocol_path.stat().st_size}
    peer_file = prep/'independent_preseal_v1.json'
    peer = c.read_json(peer_file)
    c.require(peer.get('complete') is True and peer.get('unresolved_findings') == [] and
              peer.get('stage_receipt_sha256') == c.sha256(stage_file) and
              peer.get('protocol_sha256') == stage['protocol_sha256'] and
              peer.get('stage_member_sha256') == expected and peer.get('stage_member_bytes') == sizes,
              'Exact independently reviewed actual stage required')
    inventory = transport._walk(folder)
    c.require(set(inventory) == set(expected), 'Unbound stage namespace')
    for name, digest in expected.items():
        path = c.safe_path(folder, name)
        c.require(path.stat().st_size == sizes[name] and c.sha256(path) == digest, 'Staged bytes changed: '+name)
        if name.endswith('.py'):
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
        if name == 'protocol.json':
            continue
        current = stage['current_sources'][name]
        if current is False:
            c.require(name in stage['generated_data_files'] or name == 'research/__init__.py',
                      'Undeclared generated source/input')
            if name == 'research/__init__.py':
                marker = stage['synthetic_sources'][name]
                c.require(path.read_bytes() == b'' and c.sha256(ROOT/marker['local_reference_path']) ==
                          marker['local_reference_sha256'], 'Undeclared namespace substitution')
        else:
            c.require(c.sha256(ROOT/current) == digest, 'Current source/instruction changed: '+name)
    counts = {}
    resolution = stage['local_dependency_resolution']
    for key in ('dependencies', 'retained_output_dependencies', 'extra_returned_dependencies'):
        c.require(stage[key] == p[key], 'Physical dependency map changed')
        for name, pin in p[key].items():
            path = c.no_links(Path(resolution[name])); before = path.stat()
            c.require(path.resolve(strict=True).is_relative_to(ROOT) and path.is_file() and before.st_nlink == 1 and
                      before.st_size == pin['bytes'] and c.sha256(path) == pin['sha256'], 'Read-only parent changed: '+name)
            after = path.stat()
            c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                      (after.st_size, after.st_mtime_ns, after.st_ino), 'Parent changed during seal')
        counts[key] = len(p[key])
    bundle = c.read_json(folder/p['data_bundle_file'])
    joins = transport.verify_data_bundle(None, p, None, resolver=lambda n:Path(resolution[n]), bundle=bundle)
    numeric = stage['data_binding_receipt']
    c.require(numeric['complete'] is True and numeric['TRAIN_observations'] == 200 and
              numeric['HQ_selected_references'] == 100 and numeric['DEV_cases'] == 406 and numeric['neural_calls'] == 0,
              'Actual saved-pixel numeric stage verification required')
    ledger = transport.resource_ledger(len(expected), sum(sizes.values()))
    c.require(ledger == stage['resource_ledger'], 'Actual source-inclusive resource ledger changed')
    archive = ROOT/'outputs'/(c.PREFIX+'-execution.tar.gz')
    checksum = Path(str(archive)+'.sha256')
    seal_file = prep/'seal_v1.json'
    c.require(not any(path.exists() or path.is_symlink() for path in (archive, checksum, seal_file)),
              'Preserve prior/partial upload and seal')
    with archive.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for name in sorted(expected):
                    payload = (folder/name).read_bytes()
                    c.require(len(payload) == sizes[name] and hashlib.sha256(payload).hexdigest() == expected[name],
                              'Stage changed while sealing')
                    member = tarfile.TarInfo(c.VM_NAME+'/'+name)
                    member.size, member.mtime, member.mode = len(payload), 0, 0o644
                    tar.addfile(member, io.BytesIO(payload))
    c.require(archive.stat().st_size <= c.LIMITS['source_packet_bytes'], 'Bounded source upload required')
    physical = physical_readback(archive, expected, sizes)
    c.require(not any(n in sys.modules for n in ('torch', 'tensorflow', 'jax')), 'Local framework import forbidden')
    digest = c.sha256(archive)
    with checksum.open('xb') as stream:
        stream.write((digest+'  '+archive.name+'\n').encode('ascii'))
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'PREPARED_UNRUN',
        'archive': str(archive), 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
        'protocol_sha256': stage['protocol_sha256'], 'static_members': len(expected),
        'payload_bytes': sum(sizes.values()), 'members_sha256': expected, 'members_bytes': sizes,
        'stage_receipt_sha256': c.sha256(stage_file), 'independent_preseal_sha256': c.sha256(peer_file),
        'fresh_readonly_dependency_counts': counts, 'source_ledger': ledger, 'data_join_receipt': joins,
        'numeric_stage_receipt_bound': True, 'physical_readback': physical,
        'seal_source_sha256': c.sha256(__file__), 'local_neural_autograd_optimizer_VM_calls': 0,
        'training_updates': 0, 'model_qualified': False}
    c.write_new_json(seal_file, receipt)
    print({k:receipt[k] for k in ('complete','status','archive_bytes','archive_sha256','protocol_sha256','static_members')})


if __name__ == '__main__':
    main()

"""Seal a reviewed zero-update packet; CPU filesystem work only, never launch.

The freeze-and-seal entry validates a closed independently reviewed staging
and appends its exact receipts before creating the protocol and archive.
The readback entry consumes an existing protocol SHA. Neither imports code
from the staged packet or starts any process outside this CPU filesystem job.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_bounded_fusion_feasibility_v1 import contract as c
from research.codeformer_bounded_fusion_feasibility_v1 import neural as n

PACKAGE = c.PACKAGE_DIR
PREFIX = c.PREFIX
MAX_UPLOAD_BYTES = 96 * 1024**2
MAX_UPLOAD_PAYLOAD_BYTES = 96 * 1024**2
MAX_UPLOAD_FILES = 1000
MAX_SECONDS = 300
REQUIRED_REVIEWED_SOURCES = set(c.REQUIRED_SOURCES) | {
    'scripts/prepare_codeformer_bounded_fusion_feasibility_v1.py',
    'scripts/seal_codeformer_bounded_fusion_feasibility_v1.py'}


def fingerprint(path):
    path = c.no_links(path)
    c.require(stat.S_ISREG(path.stat().st_mode), 'Regular sealed source required')
    return {'sha256': c.sha256(path), 'bytes': path.stat().st_size}


def readiness(packet, p):
    """All readiness artifacts must themselves be protocol-bound regular JSON."""
    bindings = p.get('readiness')
    c.require(isinstance(bindings, dict) and set(bindings) == {'staging', 'data_resource', 'source'},
              'Closed staging/data-resource/source readiness required')
    records = {}
    for role, row in bindings.items():
        c.require(isinstance(row, dict) and set(row) == {'path', 'sha256'} and
                  p['files_sha256'].get(row['path']) == row['sha256'], 'Readiness hash is not protocol-bound')
        file = c.bounded(packet, row['path'])
        c.require(c.sha256(file) == row['sha256'], 'Readiness artifact changed')
        value = c.read_json(file)
        c.require(value.get('complete') is True and value.get('training_authorized', False) is False,
                  'Readiness is incomplete or authorizes training')
        if role == 'data_resource':
            c.require(value.get('approved_for_protocol_freeze') is True,
                      'Independent data/resource/source approval required')
        elif role == 'source':
            c.require(value.get('approved') is True and value.get('findings') == [],
                      'Final independent source review has open findings')
        records[role] = value
    sources = records['source'].get('source_sha256')
    c.require(isinstance(sources, dict) and REQUIRED_REVIEWED_SOURCES <= set(sources),
              'Every new executing/preparation/sealing source requires independent review')
    for name, digest in sources.items():
        c.require(p['files_sha256'].get(name) == digest and c.sha256(c.bounded(packet, name)) == digest,
                  'Reviewed source bytes differ: ' + name)
    for role in ('data_resource', 'source'):
        c.require(records[role].get('staging_sha256') == bindings['staging']['sha256'],
                  'Independent approval is not bound to this exact staging receipt')
    c.require(records['source'].get('resource_receipt_sha256') == bindings['data_resource']['sha256'],
              'Final source review does not bind this resource proof')
    policy = p.get('numerical_return_policy')
    c.require(isinstance(policy, dict) and isinstance(policy.get('path'), str) and
              p['files_sha256'].get(policy['path']) == policy.get('sha256') and
              c.sha256(c.bounded(packet, policy['path'])) == policy['sha256'],
              'Prospective numerical return policy is not exactly bound')
    c.require(records['source'].get('numerical_policy_sha256') == policy['sha256'],
              'Final source review does not bind this numerical return policy')
    validate_resource_proof(records['data_resource'])
    return records


def validate_resource_proof(value):
    ledger = value.get('resource_ledger', {})
    c.require(isinstance(ledger, dict) and c.exact(ledger.get('success_call_caps'), c.CALL_CAPS),
              'Source-derived exact successful call ledger required')
    for field, limit in [('result_payload_bytes', 'result_bytes'), ('archive_upper_bytes', 'archive_bytes'),
                         ('metadata_bytes', 'metadata_bytes'), ('regular_files', 'regular_files'),
                         ('temporary_bytes', 'temporary_bytes')]:
        count = ledger.get(field)
        c.require(type(count) is int and 0 < count <= c.LIMITS[limit],
                  'Source-derived finite resource proof exceeds/misses ' + field)
    # These retained array quantities follow the prospective design directly;
    # a proof below their strict minimum is not accepted as an approval claim.
    raw_minimum = 170 * ((512*512*3 + 256*256*3)*4 + 256*8 + 3*128) + 62914560 + 24*128
    c.require(ledger['result_payload_bytes'] >= raw_minimum and
              ledger['metadata_bytes'] >= 1130 and ledger['regular_files'] >= 170*6+24+1130,
              'Resource proof omits declared path arrays/records/call traces')


def checked_artifact(root, relative, expected_sha):
    c.require(isinstance(expected_sha, str) and c.HEX.fullmatch(expected_sha), 'Exact readiness SHA required')
    file = c.bounded(root, relative)
    c.require(c.sha256(file) == expected_sha, 'Readiness input differs: ' + relative)
    return file


def freeze_and_seal(root, approval_sha):
    """Append reviewed artifacts after exact282-file staging verification."""
    root = c.no_links(root)
    prep = root / 'outputs/codeformer_bounded_fusion_feasibility_v1_preparation'
    packet = root / 'outputs' / PACKAGE
    c.require(not (packet / 'protocol.json').exists(), 'Preserve frozen protocol; no automatic reseal')
    approval_file = checked_artifact(root,
        'outputs/codeformer_bounded_fusion_feasibility_v1_preparation/final_review_approval_v1.json', approval_sha)
    approval = c.read_json(approval_file)
    c.require(approval.get('complete') is True and approval.get('approved') is True and
              approval.get('findings') == [] and approval.get('training_authorized', False) is False,
              'Closed final independent review is required')
    staging_file = checked_artifact(root,
        'outputs/codeformer_bounded_fusion_feasibility_v1_preparation/staging_v1.json', approval.get('staging_sha256'))
    resource_file = checked_artifact(root,
        'outputs/codeformer_bounded_fusion_feasibility_v1_preparation/independent_data_resource_v1.json',
        approval.get('resource_receipt_sha256'))
    numeric_file = checked_artifact(root,
        'outputs/codeformer_bounded_fusion_feasibility_v1_preparation/numerical_policy_v1.json',
        approval.get('numerical_policy_sha256'))
    staging, resource = c.read_json(staging_file), c.read_json(resource_file)
    c.require(staging.get('complete') is True and staging.get('files_count') == 282 and
              staging.get('status') == 'STAGED / UNSEALED / UNRUN' and
              staging.get('local_neural_gradient_optimizer_VM_calls') == 0 and staging.get('model_qualified') is False,
              'Exact closed282-file original staging receipt required')
    c.require(resource.get('complete') is True and resource.get('approved_for_protocol_freeze') is True and
              resource.get('training_authorized') is False and
              resource.get('staging_sha256') == approval['staging_sha256'],
              'Closed staged-data/resource proof required')
    validate_resource_proof(resource)
    staged = staging.get('files', {})
    c.require(isinstance(staged, dict) and len(staged) == 282, 'Original staging map differs')
    actual = set()
    for folder, dirs, names in os.walk(packet, followlinks=False):
        for name in dirs:
            c.no_links(Path(folder) / name)
        for name in names:
            file = Path(folder) / name; c.no_links(file)
            c.require(stat.S_ISREG(file.lstat().st_mode), 'Only regular original staged files')
            actual.add(file.relative_to(packet).as_posix())
    c.require(actual == set(staged), 'Original282-file namespace changed before append')
    for name, row in staged.items():
        c.require(fingerprint(c.bounded(packet, name)) == row, 'Original staged bytes differ: ' + name)
    c.require(sum(row['bytes'] for row in staged.values()) == staging['payload_bytes'], 'Staging payload differs')
    draft_file = c.bounded(prep, 'protocol_draft.json')
    c.require(resource.get('protocol_draft_sha256') == c.sha256(draft_file), 'Resource proof binds another draft')
    sources = approval.get('source_sha256', {})
    c.require(isinstance(sources, dict) and REQUIRED_REVIEWED_SOURCES <= set(sources),
              'All new code requires exact independent approval')
    new_runnable = {name for name in staged if 'bounded_fusion_feasibility' in name and
                    Path(name).suffix.lower() in {'.py', '.sh'}}
    c.require(new_runnable <= set(sources), 'Every new staged executable requires independent source approval')
    for name, digest in sources.items():
        checked_artifact(root, name, digest)
        if name in staged:
            c.require(staged[name]['sha256'] == digest, 'Reviewed root source differs from original staged source')
    bound_inputs = approval.get('bound_inputs_sha256')
    c.require(isinstance(bound_inputs, dict) and bound_inputs, 'Final review must close its input receipt bindings')
    for name, digest in bound_inputs.items():
        checked_artifact(root, name, digest)
    # Verify large immutable assets locally by their fixed retained locations;
    # runtime still permits only the installed sibling FP32 asset namespace.
    for name, relative in [('original', 'outputs/cctv_codeformer_adaptation_v2_vm/weights/codeformer.pth'),
            ('stopped', 'outputs/codeformer_fp32_comparison_v1_return/outputs/fp32_comparison_v1/stopped_checkpoint.pt')]:
        actual_asset = fingerprint(c.bounded(root, relative))
        c.require(actual_asset == {key: c.ASSETS[name][key] for key in ('sha256','bytes')},
                  'Retained immutable ' + name + ' model asset differs')
    additions = {'provenance/readiness/staging_v1.json': staging_file,
        'provenance/readiness/independent_data_resource_v1.json': resource_file,
        'provenance/readiness/final_review_approval_v1.json': approval_file,
        'provenance/readiness/numerical_policy_v1.json': numeric_file}
    additions.update({name: c.bounded(root, name) for name in sources if name not in staged})
    c.require(not any(name.startswith('outputs/') or name == 'protocol.json' for name in additions),
              'Readiness source additions may not use the reserved result/protocol namespace')
    c.require(len(additions) <= 16 and sum(path.stat().st_size for path in additions.values()) <= 262144,
              'Additional sealing material exceeds approved256KiB/16-file reserve; refresh independent proof')
    # No original file can be rewritten. A partial append is retained and
    # blocks a retry, so a failed freeze requires explicit reviewed recovery.
    for name in additions:
        c.require(not c.bounded(packet, name, must_exist=False).exists(), 'Preserve prior readiness append')
    for name, source in additions.items():
        destination = c.bounded(packet, name, must_exist=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024**2); dst.flush(); os.fsync(dst.fileno())
        c.require(fingerprint(destination) == fingerprint(source), 'Readiness copy differs')
    p = c.read_json(draft_file)
    p['files_sha256'] = {name: row['sha256'] for name, row in staged.items()}
    p['files_sha256'].update({name: c.sha256(c.bounded(packet, name)) for name in additions})
    p['export_files'] = sorted(name for name in p['files_sha256'] if not name.startswith('data/'))
    p['readiness'] = {role: {'path': name, 'sha256': p['files_sha256'][name]} for role, name in
        [('staging','provenance/readiness/staging_v1.json'),
         ('data_resource','provenance/readiness/independent_data_resource_v1.json'),
         ('source','provenance/readiness/final_review_approval_v1.json')]}
    name = 'provenance/readiness/numerical_policy_v1.json'
    p['numerical_return_policy'] = {'path': name, 'sha256': p['files_sha256'][name]}
    c.validate_protocol(p); n.validate_protocol(p); readiness(packet, p)
    c.write_new_json(packet / 'protocol.json', p)
    result = seal(root, c.sha256(packet / 'protocol.json'))
    c.write_new_json(prep / 'sealing_v1.json', {**result, 'initial_staging_files_preserved': 282,
        'readiness_additions': {name: fingerprint(c.bounded(packet, name)) for name in additions}})
    return result


def archive_readback(archive_path, packet, expected, *, limit_bytes=MAX_UPLOAD_BYTES):
    """Stream every exact member and hash without extracting or executing it."""
    c.require(Path(archive_path).stat().st_size <= limit_bytes, 'Upload archive exceeds finite byte cap')
    c.require(len(expected) <= MAX_UPLOAD_FILES and expected, 'Finite nonempty upload namespace required')
    seen, folded, payload_bytes = set(), set(), 0
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            path = c.canonical(member.name)
            c.require(member.isreg() and not member.issparse() and not member.linkname and
                      not any(key.startswith('GNU.sparse') for key in member.pax_headers),
                      'Only ordinary regular archive members are allowed')
            c.require(len(path.parts) > 1 and path.parts[0] == PACKAGE, 'Exact upload package prefix required')
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            c.require(relative in expected and relative not in seen and relative.casefold() not in folded,
                      'Unexpected, duplicate or case-colliding upload member')
            file = c.bounded(packet, relative)
            c.require(type(member.size) is int and member.size >= 0 and member.size == file.stat().st_size,
                      'Upload member bytes differ')
            payload_bytes += member.size
            c.require(payload_bytes <= MAX_UPLOAD_PAYLOAD_BYTES, 'Upload payload byte cap exceeded')
            digest, count = hashlib.sha256(), 0
            with archive.extractfile(member) as stream:
                for payload in iter(lambda: stream.read(1024**2), b''):
                    digest.update(payload); count += len(payload)
            c.require(count == member.size and digest.hexdigest() == expected[relative], 'Upload member hash differs')
            seen.add(relative); folded.add(relative.casefold())
    c.require(seen == set(expected), 'Upload archive omits bound files')
    return {'members': len(seen), 'payload_bytes': payload_bytes,
            'exact_namespace_and_every_member_hash': True, 'returned_code_executed': False}


class FiniteArchive:
    def __init__(self, stream, deadline):
        self.stream, self.deadline, self.bytes = stream, deadline, 0

    def write(self, payload):
        c.require(time.monotonic() <= self.deadline, 'Finite sealing time exceeded')
        c.require(self.bytes + len(payload) <= MAX_UPLOAD_BYTES, 'Upload compressed byte cap exceeded')
        written = self.stream.write(payload); self.bytes += written
        return written

    def flush(self):
        self.stream.flush()


def seal(root, protocol_sha):
    root = c.no_links(root)
    packet = root / 'outputs' / PACKAGE
    evidence = root / 'outputs/codeformer_bounded_fusion_feasibility_v1_preparation'
    p = c.verify_transfer(packet, protocol_sha, verify_parent=False)
    n.validate_protocol(p)  # Pure schema/call-plan validation, not execute().
    proof = readiness(packet, p)
    expected = {**p['files_sha256'], 'protocol.json': protocol_sha}
    c.require(len(expected) <= MAX_UPLOAD_FILES, 'Upload file cap exceeded')
    total = sum(c.bounded(packet, name).stat().st_size for name in expected)
    c.require(total <= MAX_UPLOAD_PAYLOAD_BYTES, 'Upload payload cap exceeded')
    # Large parent weights remain external. Their exact hashes/bytes are fixed
    # by contract and must be checked fresh on the human-invoked VM preflight.
    c.require(not any(Path(name).suffix.lower() in {'.pth', '.pt', '.ckpt', '.safetensors'} for name in expected),
              'Do not upload copied model/optimizer weights')
    archive = root / 'outputs' / (PREFIX + '-execution.tar.gz')
    temporary = Path(str(archive) + '.writing')
    checksum = Path(str(archive) + '.sha256')
    receipt = evidence / 'execution_archive_v1.json'
    for file in (archive, temporary, checksum, receipt):
        c.no_links(file); c.require(not file.exists(), 'Preserve prior/partial sealed artifact: ' + file.name)
    started = time.monotonic()
    with temporary.open('xb') as stream:
        bounded = FiniteArchive(stream, started + MAX_SECONDS)
        with gzip.GzipFile(filename='', mode='wb', fileobj=bounded, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as bundle:
                for name in sorted(expected):
                    file = c.bounded(packet, name)
                    c.require(c.sha256(file) == expected[name], 'Sealed source changed while packing')
                    info = bundle.gettarinfo(str(file), arcname=PACKAGE + '/' + name)
                    c.require(info.isreg(), 'Regular upload source required')
                    info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                    info.mode = 0o644; info.pax_headers = {}
                    with file.open('rb') as source:
                        bundle.addfile(info, source)
        stream.flush(); os.fsync(stream.fileno())
    result = archive_readback(temporary, packet, expected)
    # Recheck all filesystem payloads after complete archive readback. Nothing
    # is silently omitted or repaired; a failure retains the .writing file.
    c.verify_transfer(packet, protocol_sha, verify_parent=False)
    temporary.rename(archive)
    archive_sha = c.sha256(archive)
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(archive_sha + '  ' + archive.name + '\n'); stream.flush(); os.fsync(stream.fileno())
    c.require(b'\r' not in checksum.read_bytes(), 'Portable LF checksum required')
    result.update(complete=True, protocol_sha256=protocol_sha, archive_sha256=archive_sha,
        archive=archive.relative_to(root).as_posix(), bytes=archive.stat().st_size, checksum_LF=True,
        parent_assets=p['assets'], parent_VM_assets_require_fresh_verification=True,
        readiness=p['readiness'], numerical_return_policy=p['numerical_return_policy'],
        source_approval_source_count=len(proof['source']['source_sha256']),
        sealer_source_sha256=c.sha256(__file__), seconds=time.monotonic()-started,
        manual_launch_required=True, neural_calls=0, gradient_queries=0, optimizer_updates=0,
        VM_calls=0, model_qualified=False, training_authorized=False)
    c.write_new_json(receipt, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--protocol-sha')
    group.add_argument('--freeze-and-seal', action='store_true')
    parser.add_argument('--approval-sha')
    args = parser.parse_args()
    if args.freeze_and_seal:
        c.require(args.approval_sha is not None, 'Exact final independent approval SHA required')
        result = freeze_and_seal(ROOT, args.approval_sha)
    else:
        c.require(args.approval_sha is None, 'Approval SHA applies only to first freeze-and-seal')
        result = seal(ROOT, args.protocol_sha)
    print(json.dumps(result, indent=2))

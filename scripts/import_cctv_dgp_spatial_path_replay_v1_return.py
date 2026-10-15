"""Import the manually returned replay evidence; never import returned VM code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

from cctv_dgp_spatial_path_replay_v1_contract import read, write, sha, verify_transfer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'
STEM = 'cctv-dgp-spatial-path-replay-v1'
NAME = 'cctv_dgp_spatial_path_replay_v1_vm'
PIN = '6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7'
RETURN = OUT / 'cctv_dgp_spatial_path_replay_v1_return'
ANALYSIS = OUT / 'cctv_dgp_spatial_path_replay_v1_analysis'
STAGE = 'outputs/spatial_path_replay_v1/'
MAX_MEMBERS = 4096
MAX_MANIFEST = 16 * 1024**2
RESERVE = 2 * 1024**3


def safe_name(name):
    assert isinstance(name, str) and name, 'Empty archive name'
    path = PurePosixPath(name)
    assert not path.is_absolute() and path.as_posix() == name
    assert '..' not in path.parts and ':' not in name and '\\' not in name
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {s + str(i) for s in ['COM', 'LPT'] for i in range(1, 10)}
    for part in path.parts:
        assert part and part == part.rstrip(' .') and part.split('.')[0].upper() not in reserved
        assert not any(ord(c) < 32 for c in part) and not any(c in '<>"|?*' for c in part)
    return name


def namespace(tar, byte_cap):
    members = tar.getmembers()
    assert 0 < len(members) <= MAX_MEMBERS, 'Archive member limit'
    names, folded = {}, set()
    for member in members:
        name = safe_name(member.name)
        assert member.isfile() and not member.issym() and not member.islnk(), 'Regular files only'
        assert 0 <= member.size <= byte_cap and name.casefold() not in folded
        names[name] = member
        folded.add(name.casefold())
    for name in names:
        assert not any(p.as_posix().casefold() in folded for p in PurePosixPath(name).parents
                       if p.as_posix() != '.'), 'File/directory collision'
    assert sum(m.size for m in members) <= byte_cap, 'Uncompressed archive limit'
    return names


def read_manifest(tar, names, pin, packet, output_cap):
    member = names['result_manifest.json']
    assert member.size <= MAX_MANIFEST
    with tar.extractfile(member) as stream:
        manifest = json.load(stream)
    assert manifest['protocol_sha256'] == pin
    assert manifest['model_quality_not_implied'] is True and manifest['new_optimizer_updates'] == 0
    files = manifest['files']
    assert set(names) == set(files) | {'result_manifest.json'}, 'Exact manifest namespace required'
    logs = {'review.log', 'review_exit.json', 'saved_audit.log', 'saved_audit_exit.json'}
    for name, row in files.items():
        safe_name(name)
        assert name in packet or name in logs or name.startswith(STAGE), 'Unexpected returned root file'
        assert type(row['bytes']) is int and row['bytes'] == names[name].size
        assert isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256'])
    assert all(name in files and files[name]['sha256'] == digest for name, digest in packet.items())
    assert sum(row['bytes'] for name, row in files.items() if name.startswith(STAGE)) <= output_cap
    return manifest


def check_payload(tar, names, manifest):
    """Read every payload before creating any extraction destination."""
    for name, row in manifest['files'].items():
        h = hashlib.sha256(); size = 0
        with tar.extractfile(names[name]) as stream:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(chunk); size += len(chunk)
        assert size == row['bytes'] and h.hexdigest() == row['sha256'], 'Payload differs: ' + name


def extract_checked(tar, names, manifest, destination):
    destination = Path(destination)
    assert not destination.exists() and not destination.is_symlink(), 'Retain previous import'
    destination.mkdir()
    for name, member in names.items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(exist_ok=True, parents=True)
        h = hashlib.sha256(); size = 0
        with tar.extractfile(member) as stream, target.open('xb') as output:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                output.write(chunk); h.update(chunk); size += len(chunk)
        assert size == member.size
        if name != 'result_manifest.json':
            assert h.hexdigest() == manifest['files'][name]['sha256']


def main():
    start = time.monotonic()
    archive = OUT / (STEM + '-results.tar.gz')
    sidecar = Path(str(archive) + '.sha256')
    exported_path = OUT / (STEM + '-export.json')
    assert all(p.is_file() and not p.is_symlink() for p in [archive, sidecar, exported_path]), \
        'Await all three manual download files; no VM launch or retrieval occurs'
    assert not RETURN.exists() and not ANALYSIS.exists(), 'Retain previous imported evidence'
    packet_root = OUT / NAME
    source = OUT / 'cctv_dgp_spatial_readout_v1_return'
    parent = OUT / 'cctv_dgp_bank_photometric_guard_v1_return'
    proof_path = OUT / 'cctv_dgp_spatial_path_replay_v1_design/independent_preparation_audit.json'
    proof = read(proof_path)
    assert proof['complete'] and proof['status'] == 'VERIFIED_PREPARED_UNRUN' and proof['protocol_sha256'] == PIN
    p, _, _ = verify_transfer(packet_root, source, parent, PIN)
    packet = {**p['assets_sha256'], 'protocol.json': PIN}
    exported = read(exported_path)
    assert exported['complete'] is True and exported['protocol_sha256'] == PIN
    assert exported['optimizer_updates'] == exported['gradient_queries'] == 0
    assert exported['training_success_not_implied'] is True and exported['model_qualified'] is False
    expected_sidecar = (exported['archive_sha256'] + '  ' + archive.name + '\n').encode('ascii')
    assert sidecar.read_bytes() == expected_sidecar, 'Exact LF SHA256 sidecar required'
    assert archive.stat().st_size == exported['bytes'] and sha(archive) == exported['archive_sha256']
    protected = {name: sha(ROOT / name) for name in [
        'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md',
        'CCTV_DGP_NEXT_PILOT.md', 'PROJECT_HANDOFF.md', 'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_VM.md',
        'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth']}
    with tarfile.open(archive, 'r:gz') as tar:
        names = namespace(tar, p['budgets']['output_byte_cap'] + 16 * 1024**2)
        manifest = read_manifest(tar, names, PIN, packet, p['budgets']['output_byte_cap'])
        assert (STAGE + 'run_results.json' in names) == exported['run_results_present']
        assert (STAGE + 'failure.json' in names) == exported['failure_present']
        assert exported['run_results_present'] or exported['failure_present']
        total = sum(member.size for member in names.values())
        assert shutil.disk_usage(OUT).free >= total + RESERVE, 'Local import space/reserve stop'
        check_payload(tar, names, manifest)
        assert sha(archive) == exported['archive_sha256'], 'Archive changed during preflight'
        ANALYSIS.mkdir()
        write(ANALYSIS / 'protected_before.json', dict(complete=True, files=protected))
        extract_checked(tar, names, manifest, RETURN)
    assert sha(archive) == exported['archive_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    assert sha(RETURN / 'protocol.json') == PIN
    receipt = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        archive_sha256=exported['archive_sha256'], archive_bytes=exported['bytes'],
        protocol_sha256=PIN, manifest_sha256=sha(RETURN / 'result_manifest.json'),
        returned_file_bindings=len(names), uncompressed_bytes=total,
        source_bindings_verified=410, parent_bindings_verified=203,
        all_payloads_checked_before_extraction=True, all_extracted_files_rehashed=True,
        every_member_regular_namespace_size_hash_verified=True,
        run_results_present=exported['run_results_present'], failure_present=exported['failure_present'],
        saved_output_audit_pending=True, every_planned_sheet_review_pending=True,
        neural_calls=0, optimizer_updates=0, gradient_queries=0,
        returned_VM_code_imported_or_executed=False, agent_VM_calls=0,
        model_qualified=False, goal_complete=False, seconds=time.monotonic() - start,
        original_checkpoint_unchanged=True, preparation_proof_sha256=sha(proof_path), checker_sha256=sha(__file__))
    write(ANALYSIS / 'import.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

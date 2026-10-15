"""Hash-bound local import of manual VM evidence. Returned code is never executed."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tarfile
import time

from cctv_dgp_decoder_hinge_repair_v1_contract import (
    NAME, STEM, STAGE as STAGE_FOLDER, CHECKPOINT, read, write, sha, verify_transfer,
)
from import_cctv_dgp_spatial_path_replay_v1_return import (
    safe_name, namespace, check_payload, extract_checked,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'outputs'
RETURN = OUT/'cctv_dgp_decoder_hinge_repair_v1_return'
ANALYSIS = OUT/'cctv_dgp_decoder_hinge_repair_v1_analysis'
DESIGN = OUT/'cctv_dgp_decoder_hinge_repair_v1_design'
STAGE = STAGE_FOLDER+'/'
MAX_MEMBERS = 4096
MAX_MANIFEST = 16*1024**2
RESERVE = 2*1024**3


def _updates(value):
    assert type(value) is int and 0 <= value <= 50, 'Finite VM optimizer-update count required'
    return value


def validate_export_receipt(exported, pin, archive, sidecar_bytes):
    assert exported['complete'] is True and exported['protocol_sha256'] == pin
    assert exported['training_success_not_implied'] is True and exported['model_quality_not_implied'] is True
    assert exported['model_qualified'] is False
    _updates(exported['optimizer_updates'])
    digest = exported['archive_sha256']
    assert isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
    assert type(exported['bytes']) is int and exported['bytes'] == archive.stat().st_size
    assert sidecar_bytes == (digest+'  '+archive.name+'\n').encode('ascii'), 'Exact LF SHA256 sidecar required'
    assert sha(archive) == digest, 'Returned archive differs from its export receipt'
    assert type(exported['run_results_present']) is bool and type(exported['failure_present']) is bool
    assert exported['run_results_present'] or exported['failure_present'], 'A completed result or retained failure is required'


def read_manifest(tar, names, pin, packet, output_cap, exported=None):
    assert 'result_manifest.json' in names
    member = names['result_manifest.json']; assert member.size <= MAX_MANIFEST
    with tar.extractfile(member) as stream: manifest = json.load(stream)
    assert manifest['complete'] is True and manifest['protocol_sha256'] == pin
    assert manifest['training_success_not_implied'] is True and manifest['model_quality_not_implied'] is True
    assert manifest['model_qualified'] is False
    updates = _updates(manifest['optimizer_updates'])
    if exported is not None: assert updates == exported['optimizer_updates']
    files = manifest['files']; assert isinstance(files, dict)
    assert set(names) == set(files) | {'result_manifest.json'}, 'Exact manifest namespace required'
    assert packet['protocol.json'] == pin
    logs = {'run.log', 'run_exit.json', 'saved_audit.log', 'saved_audit_exit.json'}
    for name, row in files.items():
        safe_name(name)
        assert name in packet or name in logs or name.startswith(STAGE), 'Unexpected returned root file'
        assert set(row) == {'bytes', 'sha256'}
        assert type(row['bytes']) is int and row['bytes'] == names[name].size
        assert isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256'])
    assert all(name in files and files[name]['sha256'] == digest for name, digest in packet.items()), 'Every frozen packet asset is required'
    assert sum(row['bytes'] for name, row in files.items() if name.startswith(STAGE)) <= output_cap
    return manifest


def check_result_metadata(tar, names, manifest, exported):
    """Bind the exported VM update count without importing any returned module."""
    found = []
    for filename, flag in [('run_results.json', 'run_results_present'), ('failure.json', 'failure_present')]:
        name = STAGE+filename
        assert (name in names) == exported[flag]
        if name not in names: continue
        assert names[name].size <= MAX_MANIFEST
        with tar.extractfile(names[name]) as stream: result = json.load(stream)
        assert result['complete'] is (filename == 'run_results.json')
        assert _updates(result['counts']['optimizer_updates']) == manifest['optimizer_updates'] == exported['optimizer_updates']
        if filename == 'run_results.json': assert result['model_qualified'] is False
        found.append(filename)
    assert found, 'No saved result/failure receipt'
    return found


def local_preparation():
    """Check the actual frozen preparation without requiring return downloads."""
    prepared_path, proof_path = DESIGN/'preparation.json', DESIGN/'preparation_audit.json'
    prepared, proof = read(prepared_path), read(proof_path)
    assert prepared['complete'] and prepared['status'] == 'PREPARED_UNRUN_PENDING_INDEPENDENT_AUDIT'
    assert proof['complete'] and proof['status'] == 'VERIFIED_PREPARED_UNRUN'
    pin = prepared['protocol_sha256']; assert pin == proof['protocol_sha256']
    execution_archive = OUT/(STEM+'-execution.tar.gz')
    assert prepared['archive_sha256'] == proof['archive_sha256'] == sha(execution_archive)
    assert prepared['archive_bytes'] == proof['archive_bytes'] == execution_archive.stat().st_size
    execution_sidecar = Path(str(execution_archive)+'.sha256')
    assert execution_sidecar.read_bytes() == (prepared['archive_sha256']+'  '+execution_archive.name+'\n').encode('ascii')
    source = OUT/'cctv_dgp_spatial_readout_v1_return'
    parent = OUT/'cctv_dgp_bank_photometric_guard_v1_return'
    replay = OUT/'cctv_dgp_spatial_path_replay_v1_return'
    packet_root = OUT/NAME
    p, _, _ = verify_transfer(packet_root, source, parent, pin, replay=replay)
    return pin, p, prepared_path, proof_path


def main():
    start = time.monotonic()
    archive = OUT/(STEM+'-results.tar.gz'); sidecar = Path(str(archive)+'.sha256')
    exported_path = OUT/(STEM+'-export.json')
    assert all(path.is_file() and not path.is_symlink() for path in [archive, sidecar, exported_path]), \
        'Await all three manually downloaded return files; no VM retrieval or launch occurs'
    assert not RETURN.exists() and not ANALYSIS.exists(), 'Retain previous imported evidence'
    pin, p, prepared_path, proof_path = local_preparation()
    packet = {**p['assets_sha256'], 'protocol.json': pin}
    exported = read(exported_path)
    validate_export_receipt(exported, pin, archive, sidecar.read_bytes())
    protected_names = [
        'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md',
        'CCTV_DGP_NEXT_PILOT.md', 'PROJECT_HANDOFF.md', 'CCTV_DGP_DECODER_HINGE_REPAIR_V1_VM.md',
        'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',
    ]
    protected = {name: sha(ROOT/name) for name in protected_names}
    assert protected[protected_names[-1]] == CHECKPOINT
    with tarfile.open(archive, 'r:gz') as tar:
        names = namespace(tar, p['budgets']['output_byte_cap']+MAX_MANIFEST)
        assert len(names) <= MAX_MEMBERS
        manifest = read_manifest(tar, names, pin, packet, p['budgets']['output_byte_cap'], exported)
        check_result_metadata(tar, names, manifest, exported)
        total = sum(member.size for member in names.values())
        assert shutil.disk_usage(OUT).free >= total+RESERVE, 'Local import space/reserve stop'
        check_payload(tar, names, manifest)
        assert sha(archive) == exported['archive_sha256'], 'Archive changed during preflight'
        ANALYSIS.mkdir()
        write(ANALYSIS/'protected_before.json', dict(complete=True, files=protected))
        extract_checked(tar, names, manifest, RETURN)
    assert sha(archive) == exported['archive_sha256']
    assert all(sha(ROOT/name) == digest for name, digest in protected.items())
    assert sha(RETURN/'protocol.json') == pin
    receipt = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        archive_sha256=exported['archive_sha256'], archive_bytes=exported['bytes'],
        protocol_sha256=pin, manifest_sha256=sha(RETURN/'result_manifest.json'),
        returned_file_bindings=len(names), uncompressed_bytes=total,
        source_bindings_verified=len(p['source_required_sha256']), parent_bindings_verified=len(p['parent_required_sha256']),
        replay_bindings_verified=len(p['replay_required_sha256']),
        all_payloads_checked_before_extraction=True, all_extracted_files_rehashed=True,
        every_member_regular_namespace_size_hash_verified=True,
        run_results_present=exported['run_results_present'], failure_present=exported['failure_present'],
        reported_VM_optimizer_updates=exported['optimizer_updates'], new_local_optimizer_updates=0,
        saved_output_audit_pending=True, every_planned_sheet_review_pending=True,
        local_neural_calls=0, local_gradient_queries=0, returned_VM_code_imported_or_executed=False, agent_VM_calls=0,
        model_qualified=False, goal_complete=False, seconds=time.monotonic()-start, original_checkpoint_unchanged=True,
        preparation_sha256=sha(prepared_path), independent_preparation_proof_sha256=sha(proof_path),
        execution_archive_unchanged=True, checker_sha256=sha(__file__))
    write(ANALYSIS/'import.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()

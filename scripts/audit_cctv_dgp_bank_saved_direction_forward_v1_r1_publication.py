"""Close documentation, retained failure and exact manual R2 command bindings."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import re
import tarfile

from import_cctv_dgp_bank_saved_direction_forward_v1_r1_return import (
    ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write,
)


def fences(value):
    return re.findall(rb'```[^\r\n]*\r?\n.*?```',value,re.S)


def main():
    receipt=ANALYSIS/'publication_audit.json'
    assert not receipt.exists()
    publication=read(ANALYSIS/'publication.json')
    assert publication['complete'] and publication['goal_status']=='active' and not publication['goal_complete']
    documents={}
    for name,binding in publication['documents'].items():
        path,backup=ROOT/name,Path(binding['backup'])
        assert sha(path)==binding['after_sha256'] and sha(backup)==binding['before_sha256']
        assert path.read_bytes()[binding['prefix_bytes']:]==backup.read_bytes()
        assert fences(path.read_bytes())==fences(backup.read_bytes())
        assert binding['expected_hash_checked_under_exclusive_lock']
        documents[name]=sha(path)
    for name,digest in publication['new_documents_sha256'].items():
        assert sha(ROOT/name)==digest
        documents[name]=digest
    for name,digest in publication['evidence_sha256'].items():
        assert sha(ROOT/name)==digest and read(ROOT/name)['complete']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert len(protected)==12 and all(sha(ROOT/name)==digest for name,digest in protected.items())
    imported=read(ANALYSIS/'import.json')
    failure=read(ANALYSIS/'failure_audit.json')
    visual=read(ANALYSIS/'visual_review.json')
    assert sha(OUT/'cctv-dgp-bank-saved-direction-forward-v1-r1-results.tar.gz')==imported['archive_sha256']
    manifest=read(RETURN/'export_manifest.json')
    assert manifest['failure_present'] and not manifest['complete_diagnostic']
    assert len(manifest['files'])+1==37 and all(sha(RETURN/name)==digest for name,digest in manifest['files'].items())
    assert failure['failure_is_processing_not_a_quality_gate'] and failure['reported_counts']['optimizer_updates']==0
    assert failure['reported_counts']['B']==21 and failure['reported_counts']['recognizer']==0
    assert visual['all25_sheet_cells_and5_partial_PNGs_inspected'] and not visual['model_qualified']
    assert sha(ROOT/visual['sheet'])==visual['sheet_sha256']==failure['comparison_sheet_sha256']
    prep=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_preparation.json')
    packet=OUT/prep['name']
    p=read(packet/'protocol.json')
    assert sha(packet/'protocol.json')==prep['protocol_sha256']
    prior=read(RETURN/'protocol.json')
    for key in ['proposals','case_ids','learning_names','parent_required_sha256','preserved_scientific_thresholds',
                'worker_seconds','measurement_seconds','export_seconds','external_seconds','export_external_seconds',
                'peak_VRAM_bytes','minimum_free_bytes','disk_reserve_bytes','output_byte_cap']:
        assert p[key]==prior[key]
    assert p['optimizer_updates']==p['gradient_queries']==p['backwards']==p['full_epochs']==0
    assert p['native_pixels_used']==p['final_identity_pixels']==0
    assert p['maximum_temporary_parameter_proposals']==4 and p['model_forward_cap']==100 and p['recognizer_forward_cap']==160
    assert not p['automatic_follow_on'] and not p['app_promotion'] and not p['failed_run_resume_allowed']
    assert not (packet/'outputs').exists()
    for name,digest in p['assets_sha256'].items():
        assert sha(packet/name)==digest
    archive=OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive)==prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_bytes()==(prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        assert len(members)==5
        for m in members:
            assert m.isfile() and not m.islnk() and not m.issym()
            assert stream.extractfile(m).read()==(OUT/m.name).read_bytes()
    worker=packet/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r2_vm.py'
    ast.parse(worker.read_text(encoding='utf-8'),feature_version=(3,10))
    for name,digest in prep['prospective_return_checker_files_sha256'].items():
        assert sha(ROOT/name)==digest
    checker=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_return_checker_preparation.json')
    assert checker['protocol_sha256']==prep['protocol_sha256'] and checker['actual_saved_control_cases_checked']==100
    assert checker['raw_PNG_metric_rows_checked']==200 and len(checker['original_capacity_logic_six_boundary_tests'])==6
    assert checker['importer_sha256']==sha(ROOT/'scripts/import_cctv_dgp_bank_saved_direction_forward_v1_r2_return.py')
    assert checker['auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_bank_saved_direction_forward_v1_r2_return.py')
    transfer=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_transfer_audit.json')
    assert transfer['all100_grid_hashes_match_retained_alignment'] and transfer['directions_exactly_unchanged']
    assert len(transfer['independent_geometry_equations_checked'])==100 and len(transfer['invalid_matrix_boundary_checks'])==3
    bash=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_bash_syntax.json')
    assert bash['exit_code']==0 and bash['checked_shell_sha256']==sha(packet/'scripts/run_forward.sh')
    guide=(ROOT/'CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R2_VM.md').read_text(encoding='utf-8')
    assert 'forward-v1-r1-execution' not in guide and PIN not in guide
    assert len(re.findall(r'^\d\. ',guide,re.M))==5
    scp=re.findall(r'^gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "([^"]+)" "([^"]+)"$',guide,re.M)
    assert len(scp)==5
    remote='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'
    uploads=[(archive.name,remote),(archive.name+'.sha256',remote)]
    files=[prep['stem']+'-results.tar.gz',prep['stem']+'-results.tar.gz.sha256',prep['stem']+'-export.json']
    downloads=[(remote+name,name) for name in files]
    assert scp==uploads+downloads
    assert 'tr -d \'\\r\' < '+archive.name+'.sha256 | sha256sum -c -' in guide
    assert 'test ! -e ~/forensic-dgp/'+prep['name'] in guide
    assert 'tmux new-session -A -s dgp_saved_direction_v1_r2' in guide
    command='python -B -u scripts/'+worker.name+' --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha '+prep['protocol_sha256']+' --verify-transfer'
    assert command in guide and 'bash scripts/run_forward.sh '+prep['protocol_sha256'] in guide
    assert 'cd ~/forensic-dgp/'+prep['name']+' &&' in guide
    assert 'fixed_target_affines_checked: 100' in guide
    assert prep['archive_sha256'] in guide and 'unrun' in guide
    old_prep=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_preparation.json')
    assert sha(OUT/(old_prep['stem']+'-execution.tar.gz'))==old_prep['archive_sha256']
    earliest=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_preparation.json')
    assert sha(OUT/(earliest['stem']+'-execution.tar.gz'))==earliest['archive_sha256']
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(ANALYSIS/'publication.json'),documents_sha256=documents,
        previous5_document_payloads_and_command_fences_preserved_exactly=True,
        returned37_file_hashes_and_partial5_output_visual_bindings_rechecked=True,
        R1_processing_failure_classification_preserved=True,R1_and_V1_execution_archives_preserved=True,
        R2_five_member_archive_LF_sidecar_and_actual_source_hashes_checked=True,
        independent100_geometry_and_three_invalid_matrix_receipts_bound=True,
        future100_case_200_metric_six_gate_fixture_checker_bound=True,
        exact5_manual_steps_two_uploads_three_single_remote_downloads_checked=True,
        current_checkpoint_app_examples_12_protected_hashes_rechecked=True,
        no_agent_VM_connection_or_launch=True,no_local_model_gradient_optimizer_or_displacement=True,
        R2_actual_VM_forward_unrun=True,model_qualified=False,app_promotion=False,
        more_epochs_not_qualified=True,goal_status='active',goal_complete=False))
    print(dict(complete=True,documents_checked=7,manual_steps_checked=5,
        returned_files=37,next_manual_R2_unrun=True,goal_complete=False))


if __name__=='__main__':
    main()

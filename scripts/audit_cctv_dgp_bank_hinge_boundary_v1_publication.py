"""Close returned evidence, published history and manual R1 command bindings."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import re
import tarfile
from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


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
        documents[name]=sha(path)
    for name,digest in publication['new_documents_sha256'].items():
        assert sha(ROOT/name)==digest
        documents[name]=digest
    for name,digest in publication['evidence_sha256'].items():
        assert sha(ROOT/name)==digest and read(ROOT/name)['complete']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert len(protected)==12 and all(sha(ROOT/name)==digest for name,digest in protected.items())
    import_receipt=read(ANALYSIS/'import.json')
    recovery=read(OUT/'cctv_dgp_bank_hinge_boundary_v1_filename_recovery.json')
    assert sha(recovery['source'])==sha(recovery['target'])==import_receipt['archive_sha256']
    assert Path(recovery['source']).stat().st_size==Path(recovery['target']).stat().st_size==import_receipt['archive_bytes']
    manifest=read(RETURN/'export_manifest.json')
    assert all(sha(RETURN/name)==digest for name,digest in manifest['files'].items())
    assert len(manifest['files'])+1==135
    audit=read(ANALYSIS/'independent_audit.json')
    assert audit['inactive_initial_guard_zero_gradient_all20_independently_checked'] and audit['diagnostic_complete']
    assert audit['raw_case_replays_checked']==200 and audit['counts_recomputed']['optimizer_updates']==0
    review=read(ANALYSIS/'loss_direction_review.json')
    assert review['independent_audit_sha256']==sha(ANALYSIS/'independent_audit.json')
    assert review['group_summaries']['all']['initial']['nominal_single_reference_Adam_degraded_MSE_ascents']==15
    assert all(row['excluding_ArcFace_and_ArcFace_barrier_Gram_bit_exact'] for row in review['cross_run_scalar_and_nonbrightness_Gram_checks'])
    prep_path=OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_preparation.json'
    prep=read(prep_path)
    packet=OUT/prep['name']
    p=read(packet/'protocol.json')
    assert sha(packet/'protocol.json')==prep['protocol_sha256']
    assert p['PNG_export_policy'].startswith('Retained frozen_raw_metrics.deliver: floor RGB255')
    assert p['optimizer_updates']==p['gradient_queries']==p['backwards']==p['full_epochs']==0
    assert p['maximum_temporary_parameter_proposals']==4
    assert p['model_forward_cap']==100 and p['recognizer_forward_cap']==160
    assert p['native_pixels_used']==p['final_identity_pixels']==0
    assert not p['automatic_follow_on'] and not p['app_promotion'] and not p['failed_run_resume_allowed']
    assert p['preserved_scientific_thresholds']==read(PARENT/'protocol.json')['scientific_thresholds']
    assert not (packet/'outputs').exists()
    for name,digest in p['assets_sha256'].items():
        assert sha(packet/name)==digest
    archive=OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive)==prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_bytes()==(prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        expected={prep['name']+'/'+name for name in p['assets_sha256']}|{prep['name']+'/protocol.json'}
        assert len(members)==5 and {m.name for m in members}==expected
        for member in members:
            assert member.isfile() and not member.islnk() and not member.issym()
            assert stream.extractfile(member).read()==(OUT/member.name).read_bytes()
    transfer=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_transfer_audit.json')
    assert transfer['preparation_sha256']==sha(prep_path) and transfer['protocol_sha256']==prep['protocol_sha256']
    bash=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_bash_syntax.json')
    assert bash['checked_shell_sha256']==sha(packet/'scripts/run_forward.sh') and bash['exit_code']==0
    checker=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_return_checker_preparation.json')
    assert checker['protocol_sha256']==prep['protocol_sha256']
    for key,name in [('importer_sha256','import_cctv_dgp_bank_saved_direction_forward_v1_r1_return.py'),
                     ('auditor_sha256','audit_cctv_dgp_bank_saved_direction_forward_v1_r1_return.py')]:
        assert sha(ROOT/'scripts'/name)==checker[key]
        ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
    assert checker['actual_saved_control_cases_checked']==100 and checker['raw_PNG_metric_rows_checked']==200
    assert len(checker['original_capacity_logic_six_boundary_tests'])==6
    assert checker['synthetic_boundary_fixtures_are_not_restoration_results'] and checker['full_future_return_audit_unrun']
    original=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_preparation.json')
    assert sha(OUT/(original['stem']+'-execution.tar.gz'))==original['archive_sha256']
    assert not (OUT/original['name']/'outputs').exists()
    assert p['superseded_unrun_packet_sha256']==original['archive_sha256']
    png=read(OUT/'cctv_dgp_bank_saved_direction_v1_analysis/png_policy_repair.json')
    assert png['corrected_R1_worker_sha256']==sha(packet/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py')
    assert png['corrected_floor_policy_matches_all100_retained_PNGs'] and png['old_rounding_policy_mismatching_cases']==100

    guide=(ROOT/'CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ',guide,re.M)==list('12345')
    assert prep['protocol_sha256'] in guide and prep['archive_sha256'] in guide
    assert "tr -d '\\r'" in guide
    assert 'tmux new-session -A -s dgp_saved_direction_v1_r1' in guide
    assert 'tmux attach -t dgp_saved_direction_v1_r1' in guide
    assert 'scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha '+prep['protocol_sha256']+' --verify-transfer' in guide
    assert 'bash scripts/run_forward.sh '+prep['protocol_sha256'] in guide
    calls=[line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(calls)==5
    for call in calls:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in call
        assert call.count('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/')==1
    for call,suffix in zip(calls[2:],['-results.tar.gz','-results.tar.gz.sha256','-export.json'],strict=True):
        assert call.endswith('"'+prep['stem']+suffix+'"')
        assert 'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+prep['stem']+suffix+'"' in call
    report=(ROOT/'CCTV_DGP_BANK_HINGE_BOUNDARY_V1_RESULTS.md').read_text(encoding='utf-8')
    prose=' '.join(report.split())
    assert f"{audit['VM_seconds']:.2f} seconds" in prose
    assert 'do not independently re-run all 40 neural derivatives' in prose
    assert f"{prep['archive_bytes']:,}-byte R1 packet" in prose
    assert 'zero optimizer updates/epochs' in prose and 'Goal **ACTIVE**' in report
    assert '20 photographic TRAIN references per state' in prose
    assert 'All five milestones remain the completion criterion' in prose
    assert 'matched corrected-current-versus-replacement comparison remains unresolved' in prose
    assert 'all 400 finite raw case outputs' in prose and '20 comparison sheets' in prose
    assert '112.24 seconds' in prose
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(ANALYSIS/'publication.json'),current_document_bindings=documents,
        evidence_receipts_checked=len(publication['evidence_sha256']),returned_files_rechecked=135,
        original_and_recovered_download_sha256=import_receipt['archive_sha256'],original_download_preserved=True,
        protected_bindings_rechecked=12,exact_document_history_and_historical_commands_preserved=True,
        corrected_manual_R1_archive_members_checked=5,explicit_download_destination_filenames_checked=True,
        no_multiple_remote_sources_per_SCP=True,future_return_checker_hash_and_AST_checked=True,
        superseded_unrun_V1_with_rounding_failure_evidence_retained=True,
        actual_hinge_gradient_defect_resolved_but_model_quality_not_qualified=True,
        next_manual_R1_finite_proposals_unrun=True,no_agent_VM_launch=True,
        all_five_milestones_and_seven_covering_families_pending=True,goal_status='active',goal_complete=False))
    print(dict(complete=True,documents=7,returned_files=135,protected_bindings=12,next_manual_R1_unrun=True))


if __name__=='__main__':
    main()

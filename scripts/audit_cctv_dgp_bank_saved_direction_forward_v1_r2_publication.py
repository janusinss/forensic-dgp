"""Close reviewed evidence and publication with fresh protected/packet hashes."""
import ast
from datetime import datetime, timezone
import re
from import_cctv_dgp_bank_saved_direction_forward_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def main():
    publication = ANALYSIS/'publication.json'
    published = read(publication)
    assert published['complete'] and published['goal_status']=='active' and not published['goal_complete']
    assert sha(ROOT/'scripts/publish_cctv_dgp_bank_saved_direction_forward_v1_r2_results.py')==published['publisher_sha256']
    for name,digest in published['evidence_sha256'].items():
        assert sha(ROOT/name)==digest and read(ROOT/name)['complete'], name
    historical_fences = 0
    for name,record in published['documents'].items():
        path = ROOT/name;backup = __import__('pathlib').Path(record['backup'])
        assert sha(path)==record['after_sha256'] and sha(backup)==record['before_sha256']
        old,current = backup.read_bytes(),path.read_bytes()
        assert current[record['prefix_bytes']:]==old and record['old_bytes_preserved_as_exact_suffix']
        assert record['expected_hash_checked_under_exclusive_lock']
        old_code = re.findall(rb'```[^\r\n]*\r?\n.*?```',old,re.S)
        assert all(fence in current for fence in old_code)
        historical_fences += len(old_code)
    for name,digest in published['new_documents_sha256'].items():
        assert sha(ROOT/name)==digest
    audit, visual, review = [read(ANALYSIS/n) for n in ['independent_audit.json','visual_review.json','finite_direction_review.json']]
    assert audit['complete'] and visual['complete'] and review['complete']
    assert visual['all20_sheets_inspected'] and visual['all800_bound_cells_inspected'] and visual['all400_candidate_PNGs_inspected']
    assert len(visual['reviewed_sheets'])==len(audit['comparison_sheets'])==20
    assert all(sha(ROOT/r['file'])==r['sha256'] for r in visual['reviewed_sheets'])
    assert len(review['first_order_vs_finite'])==80 and len(review['sampled_scalar_recomputation'])==10
    assert not review['model_qualified'] and not review['more_epochs_qualified']
    assert review['only2_of20_TRAIN_references_covered_by_gradient_projection'] and review['detail_constraint_and_gate_both_use_four_degraded_profiles']
    assert all(not p['gates'][stage]['preview_pass'] for p in review['proposals'].values() for stage in ['raw','png'])
    finite = read(ANALYSIS/'finite_metrics.json')
    assert len(finite['baseline_rows'])==100 and len(finite['proposals'])==4
    assert sum(len(v['rows']) for v in finite['proposals'].values())==400
    assert all(len(v['groups'][stage])==17 for v in finite['proposals'].values() for stage in ['raw','png'])
    manifest = read(RETURN/'export_manifest.json')
    assert len(manifest['files'])+1==596
    assert all(sha(RETURN/n)==h for n,h in manifest['files'].items())
    imported = read(ANALYSIS/'import.json')
    old_archive = OUT/'cctv-dgp-bank-saved-direction-forward-v1-r2-results.tar.gz'
    assert sha(old_archive)==imported['archive_sha256'] and old_archive.stat().st_size==imported['archive_bytes']
    earlier_publication = read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_analysis/publication.json')
    earlier_fences = __import__('pathlib').Path(earlier_publication['documents']['CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R1_VM.md']['backup'])
    assert earlier_fences.exists()
    assert (OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_return/outputs/saved_direction_forward_v1/failure.json').is_file()
    assert (OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_analysis/failure_audit.json').is_file()
    original_checker = ROOT/'scripts/audit_cctv_dgp_bank_saved_direction_forward_v1_r2_return.py'
    timing = read(ANALYSIS/'timing_audit_repair.json')
    assert sha(original_checker)==timing['original_checker_sha256']==audit['original_prepared_checker_sha256']
    assert sha(ROOT/timing['revised_checker'])==timing['revised_checker_sha256']==audit['checker_sha256']
    old_numeric_reviewer = ROOT/'scripts/review_cctv_dgp_bank_saved_direction_forward_v1_r2.py'
    assert sha(old_numeric_reviewer)==visual['reviewer_sha256']
    assert sha(ROOT/'scripts/review_cctv_dgp_bank_saved_direction_forward_v1_r2_r1.py')==review['reviewer_sha256']
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert len(protected)==12 and all(sha(ROOT/n)==h for n,h in protected.items())
    prep = read(OUT/'cctv_dgp_bank_group_gradients_v1_preparation.json')
    transfer = read(OUT/'cctv_dgp_bank_group_gradients_v1_transfer_audit.json')
    packet = OUT/prep['name'];p = read(packet/'protocol.json')
    assert sha(packet/'protocol.json')==prep['protocol_sha256']==transfer['protocol_sha256']
    assert all(sha(packet/n)==h for n,h in p['assets_sha256'].items())
    assert len(p['parent_required_sha256'])==5852 and all(sha(PARENT/n)==h for n,h in p['parent_required_sha256'].items())
    assert not (packet/'outputs').exists() and not (OUT/'cctv_dgp_bank_group_gradients_v1_return').exists()
    assert sha(ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_return.py')==prep['importer_sha256']
    future_checker = ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_return.py'
    assert sha(future_checker)==transfer['future_auditor_sha256']
    assert p['maximum_gradient_queries']==320 and p['model_forward_cap']==40 and p['recognizer_forward_cap']==20
    assert p['optimizer_updates']==p['backwards']==p['full_epochs']==p['parameter_displacements']==0
    assert len(p['reference_blocks'])==20 and len(p['gradient_groups'])==32
    assert p['preserved_scientific_thresholds']==read(PARENT/'protocol.json')['scientific_thresholds']
    archive = OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    guide = (ROOT/'CCTV_DGP_BANK_GROUP_GRADIENTS_V1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ',guide,re.M)==['1','2','3','4','5']
    assert 'cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"' in guide
    commands = re.findall(r'^gcloud compute scp .+$',guide,re.M)
    assert len(commands)==5
    uploads,downloads = commands[:2],commands[2:]
    assert all('--project=forensic-dgp-thesis --zone=us-central1-a' in c for c in commands)
    assert all(c.count('janusdominic0@forensic-dgp-thesis:')==1 for c in commands)
    for command,suffix in zip(downloads,['-results.tar.gz','-results.tar.gz.sha256','-export.json']):
        filename=prep['stem']+suffix
        assert command.endswith('"'+filename+'"') and ':/home/janusdominic0/'+filename+'"' in command
    assert 'tmux new-session -A -s dgp_group_gradients_v1' in guide
    assert prep['protocol_sha256'] in guide and prep['archive_sha256'] in guide
    assert 'bash scripts/run_groups.sh '+prep['protocol_sha256'] in guide
    assert '--verify-transfer &&' in guide and 'test ! -e ~/forensic-dgp/'+prep['name'] in guide
    for name in ['SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','PROJECT_HANDOFF.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']:
        text=(ROOT/name).read_text(encoding='utf-8')
        prefix=text.split('\n---\n',1)[0]
        assert 'Goal ACTIVE' in prefix and 'Allfive milestones' in prefix and 'allseven automatic' in prefix
        assert 'Training/gradients/displacements remain human-manual L4 work' in prefix and 'maintenance-only' in prefix
        assert 'reserved final' in prefix and 'Clear glasses' in prefix and 'original/mask/one plausible estimate' in prefix
    write(ANALYSIS/'publication_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(publication),updated_documents=6,new_documents=2,historical_code_fences_preserved=historical_fences,
        all6_previous_document_bytes_preserved_as_exact_suffix=True,
        all20_visual_sheets_and400_candidate_PNG_review_bindings_checked=True,
        finite400_raw_PNG_outputs_800_metric_rows_and17_groups_bound=True,original_R1_failure_and_both_local_checkers_preserved=True,
        protected12_checkpoint_app_example_hashes_freshly_matched=True,parent5852_hashes_freshly_matched=True,
        sealed6_member_packet_and_importer_future_auditor_Bash_bindings_unchanged=True,
        five_exact_manual_steps_two_uploads_three_single_remote_explicit_downloads_checked=True,
        new_manual_diagnostic_prepared_unrun=True,actual_new_training_or_VM_launch=False,
        no_local_neural_gradient_optimizer_or_parameter_displacements=True,
        full_goal_active_and_unqualified_scope_preserved=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,documents=8,protected_hashes=12,parent_hashes=5852,manual_VM_packet_unrun=True,goal_complete=False))


if __name__ == '__main__':
    main()

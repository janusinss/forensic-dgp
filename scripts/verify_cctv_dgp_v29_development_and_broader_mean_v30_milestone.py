"""Independent readback of the new milestone, prior chain and unchanged app."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_v29_development_and_broader_mean_v30_milestone'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    started=time.monotonic();m=read(OUT/'milestone.json');current=verify(m['new_evidence_sha256'])
    path=ROOT/'outputs/dgp_v28_preservation_diagnostic_and_mean_centered_v29_milestone/milestone.json'
    assert sha(path)==m['previous_milestone_sha256']=='4f81d4a7dca45ab0e8e5c86b6a6a61b63c771531b9acb363f2011078eec0e7be'
    previous=read(path);counts=[verify(previous['new_evidence_sha256'],m['previous371_original_locations'])]
    for name,key in [
        ('outputs/dgp_v28_return_and_preservation_diagnostic_v1_milestone/milestone.json','previous1041_original_locations'),
        ('outputs/dgp_original_decoder_r2_audit_and_active_decoder_v28_milestone/milestone.json','previous586_original_locations'),
        ('outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/milestone.json','previous50_original_locations'),
        ('outputs/dgp_feature_skips_v27_audit_milestone/milestone.json','previous668_original_locations')]:
        path=ROOT/name;assert sha(path)==previous['previous_milestone_sha256'];older=read(path)
        counts.append(verify(older['new_evidence_sha256'],previous[key]));previous=older
    v27=previous;path=ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json'
    assert sha(path)==v27['cleanup_closure_sha256'];cleanup=read(path)
    counts.append(verify(cleanup['new_evidence_sha256'],v27['cleanup60_original_locations']))
    path=ROOT/'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json'
    assert sha(path)==v27['previous_milestone_sha256'];previous=read(path)
    counts.append(verify(previous['new_evidence_sha256'],v27['previous309_original_locations']))
    for name,key in [
        ('outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json','previous66_original_locations'),
        ('outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json','previous692_original_locations'),
        ('outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json','previous299_original_locations'),
        ('outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json','previous697_original_locations'),
        ('outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json','previous513_original_locations')]:
        path=ROOT/name;assert sha(path)==previous['previous_milestone_sha256'];older=read(path)
        counts.append(verify(older['new_evidence_sha256'],previous[key]));previous=older
    assert counts==[371,1041,586,50,668,60,309,66,692,299,697,513]
    for name,backup in m['previous371_original_locations'].items():
        before=(ROOT/backup).read_bytes();after=(ROOT/name).read_bytes();split=before.index(b'\n')+1
        assert after.startswith(before[:split]) and after.endswith(before[split:])
        prefix=after[split:len(after)-len(before[split:])].decode('utf-8')
        for literal in ['7 October 2026','TRAIN capacity audited','21 fixed group/metric','Reject V29 app promotion',
                        'all24 frozen ChokePoint','Actual V30 training has not started','Human gcloud upload/SSH/tmux',
                        'No final reserved pixels','Goal active/incomplete','V29 pending/not-started language']:
            assert literal.lower() in prefix.lower(),(name,literal)
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';assert sha(app_path)==m['app_record_sha256']
    app=read(app_path);assert verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    imported=read(ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_return_import.json')
    assert verify({'outputs/cctv_dgp_mean_centered_decoder_v29_return/'+name:digest for name,digest in imported['files_sha256'].items()})==837
    audit=read(ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json')
    assert audit['complete'] and audit['checker_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_mean_centered_decoder_v29_return.py')
    assert audit['selected12_preflight_complete'] and audit['training_completed800'] and audit['necessary_capacity_pass']
    assert audit['members_verified']==837 and audit['saved_gradient_values_checked']==38394279 and not audit['VM_failure_retained']
    assert audit['archive_sha256']==m['V29_return_archive_sha256']=='ecf88626b31dee6b0ceb65f6a2a76e534cea8079183e0efd143c6bcb16fb5ae2'
    assert read(ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_visual_review/visual_review.json')['cases_reviewed']==50
    paired=ROOT/'outputs/cctv_dgp_v29_paired_development_v1';r=read(paired/'results.json');a=read(paired/'saved_output_audit.json');v=read(paired/'visual_review.json')
    assert a['complete'] and a['cases_recomputed']==520 and a['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_v29_paired_development_v1.py')
    assert len(r['rows'])==m['V29_paired_DEV_cases_audited']==520
    assert len(r['diagnostic_preservation_failures'])==m['V29_DEV_preservation_failures']==21
    assert sum(row['metric']=='ArcFace_observed_fixed' for row in r['diagnostic_preservation_failures'])==17
    assert v['preview_faces_reviewed']==m['V29_DEV_preview_faces_reviewed']==50 and v['results_sha256']==sha(paired/'results.json')
    assert r['degraded_structure_gain_fraction']['dataset/asian_faces/degraded']<0
    native=ROOT/'outputs/cctv_dgp_v29_native_development_v1';na=read(native/'saved_output_audit.json');nv=read(native/'visual_review.json')
    assert na['complete'] and na['all144_sheet_cells_exact'] and nv['cases_reviewed']==m['V29_native_faces_audited_and_reviewed']==24
    assert not nv['restoration_qualified'] and nv['native_evidence_unpaired'] and not nv['captured_country_or_ethnicity_inferred']
    assert not m['V29_development_restoration_qualified']
    prep=ROOT/'outputs/cctv_dgp_broader_mean_v30_preparation';packet=read(prep/'independent_packet_audit.json')
    assert packet['complete'] and packet['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_broader_mean_v30_packet_r1.py')
    assert packet['original_checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_broader_mean_v30_packet.py')
    assert packet['external_Bash_receipt_sha256']==sha(prep/'bash_external_syntax_receipt.json')
    assert packet['return_boundary_regressions_passed']==m['V30_return_regressions_passed']==7
    assert packet['runbook_sha256']==sha(ROOT/'CCTV_DGP_BROADER_MEAN_V30_VM.md') and packet['five_manual_steps_verified']
    assert packet['prospective_return_auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_broader_mean_v30_return.py')
    bundle=ROOT/'outputs/cctv_dgp_broader_mean_vm_v30';p=read(bundle/'protocol.json')
    assert sha(bundle/'protocol.json')==packet['protocol_sha256']==m['V30_protocol_sha256']
    assert len(p['case_rows'])==m['V30_training_cases']==3905 and len(p['training_references'])==m['V30_training_references']==781
    assert p['optimizer_updates']==m['V30_updates_bound']==800 and p['epochs']==800/781 and p['component_gradient_calls']==70
    assert verify({'outputs/cctv_dgp_broader_mean_vm_v30/'+name:digest for name,digest in p['assets_sha256'].items()})==7
    assert verify({'outputs/cctv_dgp_mixed_vm_v9_r2/'+name:digest for name,digest in p['mixed_TRAIN_assets_sha256'].items()})==m['V30_TRAIN_assets_verified']==5467
    assert verify(p['local_basis_sha256'])==9
    archive=ROOT/'outputs/cctv-dgp-broader-mean-v30-execution.tar.gz'
    assert sha(archive)==m['V30_archive_sha256']==packet['archive_sha256'] and archive.stat().st_size==m['V30_archive_bytes']==746356
    assert read(prep/'test_receipt.json')['tests_passed']==7 and read(prep/'bash_syntax_receipt.json')['read_only']
    failure=ROOT/'outputs/cctv_dgp_broader_mean_v30_packet_audit_setup_failure'
    assert sha(failure/'original_checker.py')==packet['original_checker_sha256']
    diagnosis=read(failure/'environment_diagnosis_r1.json')
    assert diagnosis['original_failure_sha256']==sha(failure/'original_failure.json') and not diagnosis['training_packet_or_gates_changed']
    for key in ['V30_actual_VM_training_started','VM_actions','reserved_final_used','independent_final_review_complete','app_promotion','goal_complete']:
        assert m[key] is False,key
    assert m['local_gradient_calls']==m['local_backward_calls']==m['local_optimizer_updates']==0
    assert m['human_manual_VM_execution_required'] and m['all_old_failed_gates_preserved'] and m['goal_status']=='active'
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'milestone_sha256':sha(OUT/'milestone.json'),
             'new_bindings_verified':current,'historical_bindings_verified':counts,'app22_bindings_verified':22,
             'full_document_bodies_preserved':3,'V29_capacity_and_rejected_DEV_native_qualification_separated':True,
             'V30_packet_original_setup_failure_R1_guard_tests_runbook_and_finite_bounds_verified':True,
             'local_neural_or_gradient_calls':0,'local_optimizer_updates':0,'VM_actions':False,
             'app_promotion':False,'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-started}
    with (OUT/'independent_readback.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

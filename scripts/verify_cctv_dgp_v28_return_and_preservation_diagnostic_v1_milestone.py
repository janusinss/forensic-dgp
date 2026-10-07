"""Independent full evidence/history/document readback; no neural or gradient calls."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from diagnose_cctv_dgp_active_original_decoder_v28_mean_control import sha,read,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_v28_return_and_preservation_diagnostic_v1_milestone'


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    start=time.monotonic();m=read(OUT/'milestone.json');current=verify(m['new_evidence_sha256'])
    previous_path=ROOT/'outputs/dgp_original_decoder_r2_audit_and_active_decoder_v28_milestone/milestone.json'
    assert sha(previous_path)==m['previous_milestone_sha256']=='8347872326a0f9cd2c79b9ebfa86c8ce46b0698deb966f5eaba427f1847118cf'
    previous=read(previous_path);counts=[verify(previous['new_evidence_sha256'],m['previous586_original_locations'])]
    path=ROOT/'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/milestone.json'
    assert sha(path)==previous['previous_milestone_sha256'];older=read(path)
    counts.append(verify(older['new_evidence_sha256'],previous['previous50_original_locations']));previous=older
    path=ROOT/'outputs/dgp_feature_skips_v27_audit_milestone/milestone.json'
    assert sha(path)==previous['previous_milestone_sha256'];v27=read(path)
    counts.append(verify(v27['new_evidence_sha256'],previous['previous668_original_locations']))
    path=ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json'
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
    assert counts==[586,50,668,60,309,66,692,299,697,513]
    for name,backup in m['previous586_original_locations'].items():
        before=(ROOT/backup).read_bytes();after=(ROOT/name).read_bytes();split=before.index(b'\n')+1
        assert after.startswith(before[:split]) and after.endswith(before[split:])
        prefix=after[split:len(after)-len(before[split:])].decode('utf-8')
        for literal in ['V28 completed800 but failed acceptance','necessary_capacity_pass remains false','all50 final faces reviewed',
                        'zero optimizer updates/epochs','No actual VM run has occurred yet','No new native or reserved-final pixels',
                        'Goal active/incomplete','only the distinct\nzero-update diagnostic is ready']:
            assert literal in prefix,(name,literal)
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';assert sha(app_path)==m['app_record_sha256']
    app=read(app_path);assert verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    audit_path=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_independent_audit.json';audit=read(audit_path)
    assert audit['complete'] and audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert audit['checker_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_active_original_decoder_v28_return.py')
    assert audit['archive_sha256']==m['V28_archive_sha256']=='3196e8ab797763e7ced75a57411b973afc3b1b12654632b7662fab3e2e168aac'
    assert audit['archive_bytes']==m['V28_archive_bytes']==291926255 and audit['members_verified']==838
    imported=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return_import.json')
    assert verify({'outputs/cctv_dgp_active_original_decoder_v28_return/'+k:v for k,v in imported['files_sha256'].items()})==838
    result=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return/outputs/results.json')
    assert result['optimizer_updates']==m['V28_updates']==800 and result['epochs']==m['V28_epochs']==80
    assert len(result['preservation_failures'])==m['V28_gate_failures_retained']==3 and not result['necessary_capacity_pass']
    assert audit['saved_gradient_values_checked']==m['V28_saved_gradient_values_audited']==38394279
    assert [r['update'] for r in audit['snapshots_audited']]==[0,50,400,800]
    assert audit['CPU_original_DGP_forwards']==50 and audit['CPU_candidate_DGP_forwards']==200
    assert audit['CPU_initial_fixed_recognizer_forwards']==50 and audit['CPU_snapshot_recognizer_forwards']==200
    visual_dir=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_visual_review'
    prep=read(visual_dir/'preparation.json');visual=read(visual_dir/'visual_review.json');plan=read(visual_dir/'plan.json')
    assert visual['complete'] and visual['cases_reviewed']==m['V28_training_cases_visually_reviewed']==50
    assert visual['preparation_sha256']==sha(visual_dir/'preparation.json') and prep['plan_sha256']==sha(visual_dir/'plan.json')
    assert visual['sheet_sha256']==prep['sheet_sha256'] and [r['id'] for r in visual['rows']]==plan['case_ids']
    assert verify(prep['source_bindings_sha256'])==len(prep['source_bindings_sha256'])
    cells=0
    for sheet in prep['sheets']:
        assert sha(visual_dir/sheet['file'])==prep['sheet_sha256'][sheet['file']]
        with Image.open(visual_dir/sheet['file']) as im:canvas=np.asarray(im).copy()
        for cell in sheet['cells']:
            x,y=cell['xy'];actual=canvas[y:y+256,x:x+256]
            with Image.open(ROOT/cell['source']) as im:expected=np.asarray(im.convert('RGB')).copy()
            assert np.array_equal(actual,expected);cells+=1
    assert cells==visual['exact_source_cells']==200 and not visual['independent_final_review']
    control_dir=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1'
    c=read(control_dir/'results.json');cp=read(control_dir/'plan.json');checked=read(control_dir/'independent_readback.json')
    assert checked['complete'] and checked['results_sha256']==sha(control_dir/'results.json')
    assert checked['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_active_original_decoder_v28_mean_control.py')
    assert verify(cp['source_bindings_sha256'])==checked['source_bindings_verified']==500
    assert verify({'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1/'+k:v for k,v in c['output_sha256'].items()})==150
    assert len(c['preservation_failures'])==checked['gate_failures_retained']==m['mean_control_gate_failures_retained']==5
    assert c['necessary_control_gate_pass'] is m['mean_control_gate_pass'] is False and checked['fixed_control_is_insufficient']
    assert checked['CPU_fixed_recognizer_forwards']==50 and checked['metrics_and_all17_groups_verified']
    dp=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_preparation';packet=read(dp/'independent_packet_audit.json')
    assert packet['complete'] and packet['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_v28_preservation_diagnostic_v1_packet.py')
    assert packet['prospective_return_auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_v28_preservation_diagnostic_v1_return.py')
    assert packet['prospective_return_tests_sha256']==sha(ROOT/'tests/test_cctv_dgp_v28_preservation_diagnostic_v1.py')
    assert packet['return_regressions_passed']==m['return_regressions_passed']==9 and packet['five_manual_steps_verified']
    assert packet['actual_Windows_rejection_before_neural_or_gradients'] and not packet['actual_VM_execution_started']
    assert read(dp/'test_receipt.json')['tests_passed']==9 and read(dp/'actual_Windows_rejection.json')['outputs_created'] is False
    bash=read(dp/'bash_syntax_receipt.json');assert bash['complete'] and bash['exit_code']==0
    bundle=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_vm';p=read(bundle/'protocol.json')
    assert sha(bundle/'protocol.json')==packet['protocol_sha256']==m['diagnostic_protocol_sha256']
    assert bash['shell_sha256']==sha(bundle/'scripts/run_v28_preservation.sh')
    assert p['component_gradient_calls']==m['diagnostic_gradient_queries_bound']==100 and p['optimizer_updates']==p['epochs']==0
    assert verify({('outputs/cctv_dgp_v28_preservation_diagnostic_v1_vm/'+k):v for k,v in p['assets_sha256'].items()})==2
    assert verify(p['local_basis_sha256'])==5
    assert verify({'outputs/cctv_dgp_active_original_decoder_v28_return/'+k:v for k,v in p['closed_V28_evidence_sha256'].items()})==307
    archive=ROOT/'outputs/cctv-dgp-v28-preservation-diagnostic-v1-execution.tar.gz'
    assert sha(archive)==packet['archive_sha256']==m['diagnostic_archive_sha256']
    assert archive.stat().st_size==packet['archive_bytes']==m['diagnostic_archive_bytes']==28332
    assert sha(ROOT/'CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_VM.md')==packet['runbook_sha256']
    assert m['human_manual_VM_execution_required'] and m['all_old_failed_gates_preserved']
    assert m['local_gradient_calls']==m['local_backward_calls']==m['local_optimizer_updates']==m['diagnostic_optimizer_updates']==0
    for key in ['diagnostic_actual_VM_execution_started','new_finite_training_protocol_created','VM_actions','native_or_reserved_used',
                'independent_final_review_complete','app_promotion','goal_complete']:assert m[key] is False
    assert m['goal_status']=='active'
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'milestone_sha256':sha(OUT/'milestone.json'),
        'new_bindings_verified':current,'historical_bindings_verified':counts,'full_document_bodies_preserved':3,'app22_bindings_verified':22,
        'V28_execution_audit_and_three_failed_gates_verified':True,'all50_visual_rows_and200_exact_cells_verified':True,
        'fixed_mean_control_and_five_failed_gates_verified':True,'zero_update_VM_packet_and_five_manual_steps_verified':True,
        'prospective_return_regressions_passed':9,'local_neural_or_gradient_calls':0,'local_optimizer_updates':0,'VM_actions':False,
        'app_promotion':False,'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    write(OUT/'independent_readback.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

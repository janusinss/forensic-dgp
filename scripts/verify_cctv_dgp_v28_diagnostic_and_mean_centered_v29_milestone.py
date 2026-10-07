"""Independent new evidence, full history, handoff bodies and app readback."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_v28_preservation_diagnostic_and_mean_centered_v29_milestone'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    started=time.monotonic();m=read(OUT/'milestone.json');current=verify(m['new_evidence_sha256'])
    path=ROOT/'outputs/dgp_v28_return_and_preservation_diagnostic_v1_milestone/milestone.json'
    assert sha(path)==m['previous_milestone_sha256']=='5e163e129b438081214b0914845a3204431abc70751457b101dcd368ce87ac86'
    previous=read(path);counts=[verify(previous['new_evidence_sha256'],m['previous1041_original_locations'])]
    for name,key in [
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
    assert counts==[1041,586,50,668,60,309,66,692,299,697,513]
    for name,backup in m['previous1041_original_locations'].items():
        before=(ROOT/backup).read_bytes();after=(ROOT/name).read_bytes();split=before.index(b'\n')+1
        assert after.startswith(before[:split]) and after.endswith(before[split:])
        prefix=after[split:len(after)-len(before[split:])].decode('utf-8')
        for literal in ['diagnostic audited','zero optimizer updates/backwards/epochs','one-ULP','V28\'s18.0595%',
                        'still\nfails five','BEFORE unchanged losses','Actual V29 training has not started','Goal active/incomplete',
                        'No new native or reserved-final pixels','only the distinct V29 pilot is ready']:
            assert literal in prefix,(name,literal)
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';assert sha(app_path)==m['app_record_sha256']
    app=read(app_path);assert verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    imported=read(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return_import.json')
    assert verify({'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return/'+name:digest for name,digest in imported['files_sha256'].items()})==322
    audit=read(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_independent_audit_r1.json')
    assert audit['complete'] and audit['diagnostic_complete'] and not audit['VM_failure_retained']
    assert audit['checker_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_v28_preservation_diagnostic_v1_return_r1.py')
    assert audit['original_checker_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_v28_preservation_diagnostic_v1_return.py')
    assert audit['original_rounding_failure_retained'] and audit['arithmetic_rtol']==1e-12 and audit['arithmetic_atol']==1e-14
    assert audit['saved_gradient_values_checked']==m['diagnostic_saved_values_audited']==54848970
    assert audit['archive_sha256']==m['diagnostic_archive_sha256'] and audit['archive_bytes']==m['diagnostic_archive_bytes']==211650756
    assert audit['CPU_forward_counts']=={'original_DGP':50,'final_DGP':50,'recognizer':100}
    source=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return/outputs'
    result=read(source/'results.json');assert result['component_gradient_calls']==m['diagnostic_gradient_queries']==100
    assert result['optimizer_updates']==result['backwards']==result['epochs']==0 and not result['new_checkpoint_created']
    assert len(result['fresh_same_batch_replay_rows'])==50
    assert all(row['final_PNG_exact'] and all(row[k]==0 for k in ['final_raw_maximum_error','initial_raw_maximum_error','final_vector_maximum_error','truth_vector_maximum_error']) for row in result['fresh_same_batch_replay_rows'])
    a=read(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_analysis/results.json')
    g=np.load(source/'gradient_components.npy',allow_pickle=False)
    assert np.array_equal(-(g@g[:7].sum(0)),np.asarray(a['directional_derivative_along_negative_original_objective_gradient']))
    assert a['mean_anchor_increases_along_plain_SGD_direction_at_final_state'] and a['clear_SSIM_and_ArcFace_hinges_decrease_along_same_direction']
    original=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return/outputs/results.json')
    control=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1/independent_readback.json')
    assert not original['necessary_capacity_pass'] and len(original['preservation_failures'])==m['V28_rejected_gates_retained']==3
    assert control['fixed_control_is_insufficient'] and control['gate_failures_retained']==m['mean_control_rejected_gates_retained']==5
    prep=ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_preparation';packet=read(prep/'independent_packet_audit.json')
    assert packet['complete'] and packet['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_mean_centered_decoder_v29_packet.py')
    assert packet['return_and_projection_regressions_passed']==m['V29_regressions_passed']==17
    assert packet['actual_Windows_rejection_before_neural_or_gradients'] and packet['all50_projection_reference_arithmetic_verified']
    assert packet['five_manual_steps_verified'] and packet['loss_weights_optimizer_schedule_and_capacity_gates_unchanged']
    assert packet['prospective_return_auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_mean_centered_decoder_v29_return.py')
    assert packet['runbook_sha256']==sha(ROOT/'CCTV_DGP_MEAN_CENTERED_DECODER_V29_VM.md')
    bundle=ROOT/'outputs/cctv_dgp_mean_centered_decoder_vm_v29';p=read(bundle/'protocol.json');old=read(ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28/protocol.json')
    assert sha(bundle/'protocol.json')==packet['protocol_sha256']==m['V29_protocol_sha256']
    for key in ['optimizer','optimizer_updates','epochs','terms','budgets','parameter_layout','retained_capacity_gates','case_rows']:
        assert p[key]==old[key],key
    assert p['optimizer_updates']==m['V29_update_bound']==800 and p['epochs']==m['V29_epoch_bound']==80
    assert p['component_gradient_calls']==m['V29_preflight_gradient_queries_bound']==70
    archive=ROOT/'outputs/cctv-dgp-mean-centered-decoder-v29-execution.tar.gz'
    assert archive.stat().st_size==m['V29_archive_bytes']==packet['archive_bytes']==30293
    assert sha(archive)==m['V29_archive_sha256']==packet['archive_sha256']
    assert verify({'outputs/cctv_dgp_mean_centered_decoder_vm_v29/'+name:digest for name,digest in p['assets_sha256'].items()})==6
    bash=read(prep/'bash_syntax_receipt.json');assert bash['complete'] and bash['exit_code']==0 and bash['read_only']
    assert bash['shell_sha256']==sha(bundle/'scripts/run_v29.sh')
    assert read(prep/'test_receipt.json')['tests_passed']==17 and read(prep/'all50_projection_arithmetic.json')['cases']==50
    for key in ['V29_actual_VM_training_started','VM_actions','native_or_reserved_used','independent_final_review_complete','app_promotion','goal_complete']:
        assert m[key] is False,key
    assert m['local_gradient_calls']==m['local_backward_calls']==m['local_optimizer_updates']==m['diagnostic_optimizer_updates']==0
    assert m['human_manual_VM_execution_required'] and m['all_old_failed_gates_preserved'] and m['goal_status']=='active'
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'milestone_sha256':sha(OUT/'milestone.json'),
        'new_bindings_verified':current,'historical_bindings_verified':counts,'full_document_bodies_preserved':3,'app22_bindings_verified':22,
        'diagnostic322_files_and54million_value_audit_verified':True,'original_arithmetic_failure_retained':True,
        'first_order_measurement_and_limits_verified':True,'V28_and_mean_control_failed_gates_retained':True,
        'V29_packet_seventeen_regressions_five_manual_steps_and_bounds_verified':True,
        'local_neural_or_gradient_calls':0,'local_optimizer_updates':0,'VM_actions':False,'app_promotion':False,
        'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-started}
    write=OUT/'independent_readback.json'
    with write.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

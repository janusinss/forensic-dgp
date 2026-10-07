"""Independently read back R2 milestone, historical bindings and unchanged application."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_feature_fusion_v32_r2_milestone'
PREVIOUS=ROOT/'outputs/cctv_dgp_feature_fusion_v32_r1_milestone'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(mapping,aliases=None):
    for name,digest in mapping.items():
        path=(ROOT/(aliases or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path)==digest,name
    return len(mapping)


def main():
    start=time.monotonic();m=read(OUT/'milestone.json');new=verify(m['new_evidence_sha256'])
    assert m['complete'] and m['routing_revision']==2 and m['goal_status']=='active' and not m['goal_complete']
    assert m['root_and_transfer_regressions_passed']==14 and m['R1_optimizer_updates']==0
    assert m['manual_VM_required'] and not m['actual_VM_training_started_by_agent']
    assert m['local_neural_calls']==m['local_gradient_calls']==m['local_optimizer_updates']==0
    assert not any(m[k] for k in ['app_changes','native_or_reserved_used','quality_qualification','app_promotion'])
    doc=m['document'];before=(ROOT/doc['before_path']).read_bytes();after=(ROOT/doc['name']).read_bytes()
    split=before.index(b'\n')+1
    assert after[:split]+after[split+doc['addition_bytes']:] == before
    assert hashlib.sha256(before).hexdigest()==doc['before_sha256'] and sha(ROOT/doc['name'])==doc['after_sha256']
    prior=read(PREVIOUS/'milestone.json');assert sha(PREVIOUS/'milestone.json')==m['previous_milestone_sha256']
    old=verify(prior['new_evidence_sha256'],m['previous_document_locations'])
    closure=read(PREVIOUS/'independent_closure_audit.json')
    assert closure['complete'] and closure['milestone_sha256']==m['previous_milestone_sha256']
    assert closure['previous_actual_Windows_backup_receipt_preserved']==4431
    assert closure['prior756_returned_files_audit_preserved'] and closure['prior200_CPU_outputs_audit_preserved']
    final=read(PREVIOUS/'final_readback.json');assert final['complete']
    # Prior readback document aliases use the newly archived full handoff.
    final_count=verify(final['evidence_sha256'],m['previous_document_locations'])
    p=read(ROOT/'outputs/cctv_dgp_feature_fusion_vm_v32_r2/protocol.json')
    assert sha(ROOT/'outputs/cctv_dgp_feature_fusion_vm_v32_r2/protocol.json')==m['protocol_sha256']
    bindings={key:verify(p[key]) for key in ['local_basis_sha256','prior_R1_packet_evidence_sha256','closed_R1_launch_failure_evidence_sha256']}
    prep=ROOT/'outputs/cctv_dgp_feature_fusion_v32_r2_preparation/independent_audit_r1'
    checked=read(prep/'independent_packet_audit.json');fixture=read(prep/'fixture_correction_receipt.json')
    assert checked['complete'] and checked['protocol_sha256']==m['protocol_sha256'] and checked['archive_sha256']==m['archive_sha256']
    assert checked['root_regressions_passed']==8 and checked['transfer_dispatch_regressions_passed']==6
    assert checked['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_feature_fusion_v32_routing_r2_audit_r1.py')
    assert fixture['complete'] and fixture['packet_and_protocol_unchanged'] and fixture['only_test_fixture_changed']
    failure=read(ROOT/'outputs/cctv_dgp_feature_fusion_v32_r1_failure_v1/host_access_retry/verification.json')
    assert failure['complete'] and failure['optimizer_updates']==0 and not failure['raw_gradient_values_independently_audited']
    assert failure['VM_writes']==0 and not failure['training_started_by_agent']
    app=read(ROOT/'outputs/dgp_app_v3_integration_record.json')
    app_count=verify({n:d for n,d in app['sources_sha256'].items() if n!='static/face_workflow.js'})
    assert sha(ROOT/'static/face_workflow.js')=='9be6cfab57e0041f87571cd38ba6d0d1e0850742b8fafd498f1b1d126137a060'
    assert sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth')==app['checkpoint_sha256']
    early=read(ROOT/'outputs/cctv_dgp_profile_batches_v31_return/outputs/early_structure_stop.json')
    assert early['minimum']==.01 and not early['pass']
    assert sha(ROOT/'SYSTEM_WORKFLOW_AND_GOAL.md')=='2aad6635473dad349c85ab177ec0d1fc09d1089c350bf620298f4a06f67d2299'
    assert sha(ROOT/'PRACTICAL_OUTPUT_SCOPE.md')=='47a02049970719379727aca9f5a6707177e5894a35c211814076aa7c628bdb2b'
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'milestone_sha256':sha(OUT/'milestone.json'),
             'new_bindings_verified':new,'previous_R1_bindings_preserved':old,'previous_final_readback_bindings_preserved':final_count,
             'packet_evidence_binding_counts':bindings,'full_previous_handoff_preserved':True,
             'R1_original_zero_update_failure_preserved':True,'R1_full_gradient_audit_pending':True,
             'root_and_transfer_regressions_passed':14,'scientific_behavior_unchanged':True,
             'previous_actual_Windows_backup_receipt_preserved':4431,'previous_756_return_audit_preserved':True,
             'V31_failed_one_percent_gate_retained':True,'unchanged_app_sources':app_count,
             'original_app_checkpoint_preserved':True,'manual_VM_required':True,'actual_VM_training_started_by_agent':False,
             'local_neural_calls':0,'local_gradient_calls':0,'local_optimizer_updates':0,
             'quality_qualification':False,'goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_closure_audit.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()

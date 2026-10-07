"""Independent complete evidence/history/source/command readback; no neural calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_original_decoder_r2_audit_and_active_decoder_v28_milestone'
PIN='27c140430673880f1aaab47a9df9af9b33758cc5d8adec53822cd9e05b13bbc8'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    start=time.monotonic();target=OUT/'independent_readback.json';assert not target.exists()
    path=OUT/'milestone.json';m=read(path);current=verify(m['new_evidence_sha256'])
    previous_path=ROOT/'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/milestone.json'
    assert sha(previous_path)==m['previous_milestone_sha256']=='6b97fcbaf2f4e3709b7b2a4989ff60ffd86464c5120af6ef8752c82fd41a8e4f'
    previous=read(previous_path);counts=[verify(previous['new_evidence_sha256'],m['previous50_original_locations'])]
    v27_path=ROOT/'outputs/dgp_feature_skips_v27_audit_milestone/milestone.json'
    assert sha(v27_path)==previous['previous_milestone_sha256'];v27=read(v27_path)
    counts.append(verify(v27['new_evidence_sha256'],previous['previous668_original_locations']))
    cleanup_path=ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json'
    assert sha(cleanup_path)==v27['cleanup_closure_sha256']
    cleanup=read(cleanup_path);counts.append(verify(cleanup['new_evidence_sha256'],v27['cleanup60_original_locations']))
    previous_path=ROOT/'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json'
    assert sha(previous_path)==v27['previous_milestone_sha256'];previous=read(previous_path)
    counts.append(verify(previous['new_evidence_sha256'],v27['previous309_original_locations']))
    for name,key in [
        ('outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json','previous66_original_locations'),
        ('outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json','previous692_original_locations'),
        ('outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json','previous299_original_locations'),
        ('outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json','previous697_original_locations'),
        ('outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json','previous513_original_locations')]:
        next_path=ROOT/name;assert sha(next_path)==previous['previous_milestone_sha256']
        older=read(next_path);counts.append(verify(older['new_evidence_sha256'],previous[key]));previous=older
    assert counts==[50,668,60,309,66,692,299,697,513]
    for name,backup in m['previous50_original_locations'].items():
        original=(ROOT/backup).read_bytes();updated=(ROOT/name).read_bytes();split=original.index(b'\n')+1
        assert updated.startswith(original[:split]) and updated.endswith(original[split:])
        prefix=updated[split:len(updated)-len(original[split:])].decode('utf-8')
        for literal in ['R2 all14 failure audited','R2 remains failed and must not rerun','498,627 trainable',
                        'parameters in12 tensors','800 updates/80 epochs','Training remains',
                        'human transfer/SSH/tmux only','Actual V28 training is pending','Goal active/incomplete']:
            assert literal in prefix,(name,literal)
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';assert sha(app_path)==m['app_record_sha256']
    app=read(app_path);assert verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    audit_path=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json';audit=read(audit_path)
    assert audit['complete'] and not audit['VM_proof_complete'] and audit['VM_failure_retained']
    assert audit['saved_gradient_values_independently_checked']==m['R2_saved_gradient_values_audited']==46909863
    assert audit['archive_sha256']==m['R2_return_sha256']=='dfd8e661126d144e87875f236a7eedb339f7803a65f57b92aa0fd128110d4e3e'
    assert audit['archive_bytes']==m['R2_archive_bytes']==93557954 and audit['members_verified']==172
    assert audit['checker_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_original_decoder_gradient_v1_r2_return.py')
    imported=read(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return_import.json')
    assert verify({('outputs/cctv_dgp_original_decoder_gradient_v1_r2_return/'+k):v for k,v in imported['files_sha256'].items()})==172
    for directory,checker,count in [
       ('cctv_dgp_decoder_app_normalization_v1','verify_cctv_dgp_decoder_app_normalization_v1.py',252),
       ('cctv_dgp_original_decoder_head4_review_v1','verify_cctv_dgp_original_decoder_head4_v1.py',425),
       ('cctv_dgp_active_original_decoder_v28_review','verify_cctv_dgp_active_original_decoder_v28.py',254)]:
        results=ROOT/'outputs'/directory/'results.json';r=read(results);checked=read(results.with_name('independent_readback.json'))
        assert r['complete'] and checked['complete'] and checked['results_sha256']==sha(results)
        assert checked['checker_sha256']==sha(ROOT/'scripts'/checker) and verify(r['source_bindings_sha256'])==count
    head=read(ROOT/'outputs/cctv_dgp_original_decoder_head4_review_v1/results.json')
    assert head['zero_VM_gradient_tensors_all_batches']==m['inactive_tensors']==['head4.block0.weight','head4.block1.weight']
    assert head['all_head4_weights_nonzero_subnormal'] and head['unchanged_CPU_head4_outputs_exact_zero_cases']==50
    assert head['CPU_fourth_map_nonzero_cases']==head['float64_maximum_head4_below_float32_subnormal_cases']==50
    prep=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_preparation';packet=read(prep/'independent_packet_audit_final.json')
    tests=read(prep/'test_receipt.json');cpu=read(prep/'cpu_initial_replay_regression.json')
    assert packet['complete'] and packet['protocol_sha256']==m['V28_protocol_sha256']==PIN
    assert packet['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_active_original_decoder_v28_packet.py')
    assert packet['prospective_return_audit_sha256']==cpu['auditor_sha256']==tests['return_auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_active_original_decoder_v28_return.py')
    assert cpu['complete'] and not cpu['actual_V28_VM_result'] and not cpu['V28_return_files_created']
    assert cpu['CPU_original_DGP_forwards']==cpu['CPU_fixed_recognizer_forwards']==cpu['cohort_rows']==50
    assert cpu['maximum_CPU_CUDA_raw_error']<=1e-5 and cpu['maximum_CPU_CUDA_target_vector_error']<=5e-5
    assert tests['complete'] and tests['tests_passed']==m['V28_return_regressions_passed']==9 and tests['test_exit_code']==tests['bash_syntax_exit_code']==0
    assert tests['test_source_sha256']==sha(ROOT/'tests/test_cctv_dgp_active_original_decoder_v28_audit.py')
    assert tests['synthetic_arrays_are_not_VM_results'] and not tests['actual_V28_VM_training_started']
    bundle=ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28';p=read(bundle/'protocol.json')
    assert sha(bundle/'protocol.json')==PIN and verify(p['local_basis_sha256'])==10
    assert verify({('outputs/cctv_dgp_active_original_decoder_vm_v28/'+k):v for k,v in p['assets_sha256'].items()})==7
    archive=ROOT/'outputs/cctv-dgp-active-original-decoder-v28-execution.tar.gz'
    assert sha(archive)==m['V28_archive_sha256']==packet['archive_sha256']=='e1484a675ad4330e4615d2f58a70f66ba8a8ad287b78cae66f8555ec4c4b1614'
    assert archive.stat().st_size==m['V28_archive_bytes']==packet['archive_bytes']==29527
    runbook=(ROOT/'CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ',runbook,flags=re.MULTILINE)==['1','2','3','4','5']
    cloud='gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a '
    expected=[cloud+'"'+name+'" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"' for name in ['cctv-dgp-active-original-decoder-v28-execution.tar.gz','cctv-dgp-active-original-decoder-v28-execution.tar.gz.sha256']]
    expected+=[cloud+'"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+name+'" "."' for name in ['cctv-dgp-active-original-decoder-v28-results.tar.gz','cctv-dgp-active-original-decoder-v28-results.tar.gz.sha256','cctv-dgp-active-original-decoder-v28-export.json']]
    assert [line for line in runbook.splitlines() if line.startswith('gcloud compute scp ')]==expected
    for literal in ['cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"','tmux new-session -A -s dgp_active_original_decoder_v28',
                    'test ! -e ~/forensic-dgp/cctv_dgp_active_original_decoder_vm_v28',
                    '--protocol-sha '+PIN+' --verify-transfer','bash scripts/run_v28.sh '+PIN]:assert literal in runbook,literal
    tree=ast.parse((bundle/'cctv_dgp_app_input_v28.py').read_text(encoding='utf-8'));app_tree=ast.parse((ROOT/'dgp_face_workflow_v3.py').read_text(encoding='utf-8'))
    function=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_tensor')
    assert ast.dump(function(tree),include_attributes=False)==ast.dump(function(app_tree),include_attributes=False)
    assert m['V28_trainable_parameters']==498627 and m['V28_trainable_tensors']==12 and m['V28_updates_bound']==800
    assert not m['V28_actual_training_started'] and m['human_manual_VM_execution_required'] and m['new_finite_training_protocol_created']
    assert m['all_old_failed_gates_preserved'] and m['R2_proof_remains_failed']
    assert m['local_gradient_calls']==m['local_backward_calls']==m['local_optimizer_updates']==0
    for key in ['VM_actions','native_or_reserved_used','independent_final_review_complete','app_promotion','goal_complete']:assert m[key] is False
    assert m['goal_status']=='active'
    record={'complete':True,'checker_sha256':sha(Path(__file__)),'milestone_sha256':sha(path),'new_bindings_verified':current,
            'prior50_668_cleanup60_deeper309_66_692_299_697_513_bindings_verified':counts,'full_document_bodies_preserved':3,
            'app22_bindings_verified':22,'R2_failure_and_all14_gate_retained':True,'head4_trace_and_independent_arrays_verified':True,
            'actual_app_CPU_equivalence_and_active12_initial_parity_verified':True,'new_finite_V28_packet_and_five_manual_steps_verified':True,
            'prospective_return_audit_regressions_passed':9,'actual_V28_training_pending':True,
            'local_neural_or_gradient_calls':0,'local_optimizer_updates':0,'VM_actions':False,'app_promotion':False,
            'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()

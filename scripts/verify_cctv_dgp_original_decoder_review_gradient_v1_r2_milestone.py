"""Independent complete preparation/history/manual-command readback."""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone'
PIN='81127e45a205c44ef685646a4f82911d02b2355dfe24c63772c23e62e66a41f2'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    start=time.monotonic();path=OUT/'milestone.json';m=read(path)
    target=OUT/'independent_readback.json';assert not target.exists()
    current=verify(m['new_evidence_sha256'])
    previous_path=ROOT/'outputs/dgp_feature_skips_v27_audit_milestone/milestone.json'
    assert sha(previous_path)==m['previous_milestone_sha256']=='24b85fc75565b3f485a59e8a04c786685569a4cc9dc7aa48163d94b20af3c491'
    v27=read(previous_path);counts=[verify(v27['new_evidence_sha256'],m['previous668_original_locations'])]
    cleanup_path=ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json'
    assert sha(cleanup_path)==v27['cleanup_closure_sha256'];cleanup=read(cleanup_path)
    counts.append(verify(cleanup['new_evidence_sha256'],v27['cleanup60_original_locations']))
    previous_path=ROOT/'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json'
    assert sha(previous_path)==v27['previous_milestone_sha256'];previous=read(previous_path)
    counts.append(verify(previous['new_evidence_sha256'],v27['previous309_original_locations']))
    for name,key in [
        ('outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json','previous66_original_locations'),
        ('outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json','previous692_original_locations'),
        ('outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json','previous299_original_locations'),
        ('outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json','previous697_original_locations'),
        ('outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json','previous513_original_locations')]:
        previous_path=ROOT/name;assert sha(previous_path)==previous['previous_milestone_sha256']
        old=read(previous_path);counts.append(verify(old['new_evidence_sha256'],previous[key]));previous=old
    assert counts==[668,60,309,66,692,299,697,513]
    for name,backup in m['previous668_original_locations'].items():
        original=(ROOT/backup).read_bytes();updated=(ROOT/name).read_bytes();split=original.index(b'\n')+1
        assert updated.startswith(original[:split]) and updated.endswith(original[split:])
        prefix=updated[split:len(updated)-len(original[split:])].decode('utf-8')
        for literal in ['selected original-decoder review verified','609,219 parameters in14 tensors',
                        'zero optimizer construction/updates/epochs','worker600s, supervisor660s+30s grace','Goal active/incomplete']:
            assert literal in prefix,(name,literal)
    assert len(m['previous668_original_locations'])==m['original_document_bodies_preserved']==3
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';assert sha(app_path)==m['app_record_sha256']
    app=read(app_path);assert verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    review_path=ROOT/'outputs/cctv_dgp_original_decoder_review_v1/review.json'
    review=read(review_path);readback=read(review_path.with_name('independent_readback.json'))
    assert readback['complete'] and readback['review_sha256']==sha(review_path)
    assert readback['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_original_decoder_review_v1.py')
    assert verify(review['source_bindings_sha256'])==251
    assert review['neural_forward_counts']=={'original':50,'candidate':51}
    assert review['local_gradient_calls']==review['local_backward_calls']==review['local_optimizer_updates']==0
    prep=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_preparation'
    packet=read(prep/'independent_packet_audit.json');test=read(prep/'test_receipt.json')
    assert packet['complete'] and packet['protocol_sha256']==m['released_protocol_sha256']==PIN
    assert packet['checker_sha256']==sha(ROOT/'scripts/verify_cctv_dgp_original_decoder_gradient_v1_r2.py')
    assert packet['prospective_return_audit_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_original_decoder_gradient_v1_r2_return.py')
    assert packet['actual_Windows_execution_rejected_before_neural_work'] and packet['packet_unchanged_by_host_guard']
    assert packet['original_filter_initializer_retained'] and packet['full_worker_AST_delta_and_historical_shell_behavior_verified']
    assert test['complete'] and test['tests_passed']==m['source_return_regressions_passed']==8 and test['bash_syntax_exit_code']==0
    assert test['tests_sha256']==sha(ROOT/'tests/test_cctv_dgp_original_decoder_gradient_v1_r2_audit.py')
    assert test['synthetic_arrays_are_not_VM_results'] and test['local_neural_or_gradient_calls']==0
    archive=ROOT/'outputs/cctv-dgp-original-decoder-gradient-v1-r2-execution.tar.gz'
    assert sha(archive)==m['released_archive_sha256']==packet['archive_sha256'] and archive.stat().st_size==m['released_archive_bytes']==13816
    p=read(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/protocol.json')
    assert sha(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/protocol.json')==PIN
    verify(p['local_basis_sha256'])
    runbook=(ROOT/'CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ',runbook,flags=re.MULTILINE)==['1','2','3','4','5']
    cloud='gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a '
    lines=[line for line in runbook.splitlines() if line.startswith('gcloud compute scp ')]
    names=['cctv-dgp-original-decoder-gradient-v1-r2-execution.tar.gz','cctv-dgp-original-decoder-gradient-v1-r2-execution.tar.gz.sha256']
    expected=[cloud+'"'+name+'" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"' for name in names]
    downloads=['cctv-dgp-original-decoder-gradient-v1-r2-results.tar.gz','cctv-dgp-original-decoder-gradient-v1-r2-results.tar.gz.sha256','cctv-dgp-original-decoder-gradient-v1-r2-export.json']
    expected+=[cloud+'"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+name+'" "."' for name in downloads]
    assert lines==expected
    for literal in ['cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"','tmux new-session -A -s dgp_original_decoder_gradient_v1_r2',
        'test ! -e ~/forensic-dgp/cctv_dgp_original_decoder_gradient_v1_r2_vm',
        '--protocol-sha '+PIN+' --verify-transfer','bash scripts/run_decoder_gradient.sh '+PIN]:
        assert literal in runbook,literal
    assert ' --preflight' not in runbook
    assert p['optimizer_updates']==p['epochs']==0 and p['component_gradient_calls']==70 and p['batches']==10
    assert m['initial_raw_PNG_parity_cases']==50 and m['decoder_parameter_tensors']==14
    assert m['actual_L4_gradient_proof_pending'] and m['human_manual_VM_execution_required'] and m['prospective_full_return_audit_prepared']
    assert m['local_gradient_calls']==m['local_backward_calls']==m['local_optimizer_updates']==0
    assert not m['VM_actions'] and not m['new_training_recipe_created'] and not m['native_or_reserved_used']
    assert not m['app_promotion'] and not m['independent_final_review_complete'] and not m['goal_complete'] and m['goal_status']=='active'
    result={'complete':True,'checker_sha256':sha(Path(__file__)),'milestone_sha256':sha(path),'new_bindings_verified':current,
        'prior668_cleanup60_deeper309_66_692_299_697_513_bindings_verified':counts,
        'full_document_bodies_preserved':3,'app22_bindings_verified':22,'initial_exact_parity_cases':50,
        'released_R2_four_file_packet_and_five_manual_steps_verified':True,'regressions_passed':8,
        'all_original_quality_and_failed_gates_retained':True,'actual_L4_gradient_proof_pending':True,
        'local_neural_or_gradient_calls':0,'optimizer_updates':0,'VM_actions':False,'app_promotion':False,
        'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

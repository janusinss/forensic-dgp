"""Independent closure of preserved history and manual-only diagnostic scope."""
from pathlib import Path
import time
from cctv_dgp_actual_step_review_v1_contract import NAME,STEM,read,write,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_prepared_milestone'
PRIOR=ROOT/'outputs/cctv_dgp_v41_preservation_trace_milestone'


def main():
    start=time.monotonic();m=read(OUT/'milestone.json');assert m['complete'] and m['prepared_only']
    for name,digest in m['new_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    before=(ROOT/m['previous_handoff_path']).read_bytes();addition=(OUT/'new_handoff_section.md').read_bytes();after=(ROOT/'PROJECT_HANDOFF.md').read_bytes()
    at=before.index(b'\n')+1;assert after==before[:at]+addition+before[at:]
    parent=read(PRIOR/'milestone.json');closed=read(PRIOR/'independent_closure_audit.json')
    assert parent['complete'] and closed['complete'] and closed['milestone_sha256']==sha(PRIOR/'milestone.json')==m['previous_milestone_sha256']
    assert sha(PRIOR/'independent_closure_audit.json')==m['previous_closure_sha256']
    assert parent['new_evidence_sha256']['PROJECT_HANDOFF.md']==sha(ROOT/m['previous_handoff_path'])
    # Previous full return closure is retained; preparation changes no old image/model source.
    full=read(ROOT/'outputs/cctv_dgp_v41_return_review_milestone/independent_closure_audit.json')
    assert full['complete'] and full['milestone_sha256']==sha(ROOT/'outputs/cctv_dgp_v41_return_review_milestone/milestone.json')
    packet=read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation/independent_packet_audit.json')
    prep=read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_packet_preparation.json');p=read(ROOT/'outputs'/NAME/'protocol.json')
    assert packet['complete'] and packet['members_verified']==261 and len(packet['regressions'])==25
    assert packet['protocol_sha256']==prep['protocol_sha256']==m['protocol_sha256']==sha(ROOT/'outputs'/NAME/'protocol.json')
    assert packet['archive_sha256']==m['archive_sha256']==sha(ROOT/'outputs'/(STEM+'-execution.tar.gz'))
    assert p['optimizer_updates']==p['gradient_queries']==0 and not p['new_trained_checkpoint'] and not p['app_promotion']
    assert p['manual_VM_execution_required'] and not p['automatic_follow_on'] and not p['native_or_DEV_or_reserved_final_used']
    assert len(p['cases'])==145 and len(p['references'])==29 and all(c['role']=='train' for c in p['cases'])
    assert p['post_outcome_TRAIN_failure_probe_selection'] and p['fixed_cohorts_unchanged'] and p['no_damping_sweep']
    for name,digest in m['app_bindings_sha256'].items():assert sha(ROOT/name)==digest,name
    assert len(m['app_bindings_sha256'])==14
    retained=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json')
    failed_root=ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return'
    for name in ['outputs/failure.json','outputs/capacity_update50.json','outputs/stopped_spatial_decoder.pth']:
        assert sha(failed_root/name)==retained['files_sha256'][name]
    gates=read(failed_root/'outputs/capacity_update50.json');assert not gates['pass'] and gates['relative_feature_gain']<.01
    assert len(gates['preservation_failures'])==1
    failures=[ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation_failure_r1/failure.json',
              ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation_failure_r2/failure.json']
    assert all(path.exists() and path.relative_to(ROOT).as_posix() in m['new_evidence_sha256'] for path in failures)
    second=read(failures[1]);partial=failures[1].parent/'partial_packet'
    assert len(second['files'])==153
    for row in second['files']:
        path=partial/row['path'];assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    assert not m['new_training_recipe_prepared'] and not m['historical_training_code_modified'] and not m['new_trained_checkpoint']
    assert m['new_neural_model_forwards']==m['gradient_queries']==m['optimizer_updates']==m['VM_calls']==0
    assert not m['app_promotion'] and not m['native_or_DEV_or_reserved_final_used'] and not m['independent_final_review'] and not m['goal_complete']
    assert not (ROOT/'outputs'/NAME/'outputs').exists() and not (ROOT/'outputs'/(STEM+'-results.tar.gz')).exists()
    assert time.monotonic()-start<300
    write(OUT/'independent_closure_audit.json',{'complete':True,'milestone_sha256':sha(OUT/'milestone.json'),'checker_sha256':sha(Path(__file__)),
       'new_bindings_verified':len(m['new_evidence_sha256']),'entire_previous_handoff_preserved':True,'prior_full_historical_closure_receipt_checked':True,
       'new_full_historical_reaudit_performed':False,'original_V41_gate_and_checkpoint_and_failure_unchanged':True,
       'both_preparation_failures_and153_partial_files_preserved':True,'all14_DGP_primary_app_bindings_unchanged':True,
       'manual_only_inference_packet_verified':True,'optimizer_updates':0,'gradient_queries':0,'VM_calls':0,
       'new_trained_checkpoint':False,'app_promotion':False,'independent_final_review':False,'goal_complete':False,
       'seconds':time.monotonic()-start,'cap_seconds':300})
    print({'complete':True,'bindings':len(m['new_evidence_sha256']),'app_bindings':14,'goal_complete':False,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()

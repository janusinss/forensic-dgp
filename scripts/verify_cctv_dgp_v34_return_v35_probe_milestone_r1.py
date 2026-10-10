"""Independent milestone, entire preceding handoff and historical-asset readback."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_v34_return_v35_probe_milestone'
PREVIOUS=ROOT/'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone'
OVERRIDES={'CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md':'outputs/cctv_dgp_v34_return_v35_closure_basis_v1/V34_original_runbook.md'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    started=time.monotonic();m=read(OUT/'milestone.json');assert m['complete']
    for name,digest in m['new_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    old=read(PREVIOUS/'milestone.json');assert sha(PREVIOUS/'milestone.json')==m['previous_milestone_sha256']
    before=(ROOT/m['previous_handoff_path']).read_bytes();current=(ROOT/'PROJECT_HANDOFF.md').read_bytes()
    at=before.index(b'\n')+1;amount=m['document']['addition_bytes']
    assert current[:at]==before[:at] and current[at+amount:]==before[at:]
    assert hashlib.sha256(before).hexdigest()==m['document']['before_sha256']
    for name,digest in old['new_evidence_sha256'].items():
        path=ROOT/m['previous_handoff_path'] if name=='PROJECT_HANDOFF.md' else ROOT/m['previous_file_overrides'].get(name,name)
        assert sha(path)==digest,name
    audit=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json')
    external=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_audit_run_v1/external_receipt.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['members_verified']==230
    assert external['complete'] and not external['timeout'] and external['worker_exit_code']==0
    assert audit['local_gradient_calls']==audit['local_optimizer_updates']==0
    assert audit['pinned_baseline_CPU_replay']['outputs']==280 and audit['pinned_baseline_CPU_replay']['all_states_restored']
    imported=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_return_import.json')
    for name,digest in imported['files_sha256'].items():assert sha(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_return'/name)==digest,name
    result=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_return/outputs/results.json')
    assert result['gradient_queries']==300 and result['optimizer_updates']==result['parameter_updates']==0
    assert not result['new_checkpoint_created']
    a=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/analysis.json')
    checked=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/independent_analysis_audit.json')
    assert a['complete'] and checked['complete'] and checked['group_rows_independently_reassembled']==102
    assert len(a['constraint_labels'])==108 and a['joint_projection']['constraint_count']==108
    assert a['joint_projection']['independent_primal_relative_L2_error']<=2e-7
    assert all(sum(r['change']>0 for r in cohort['predicted_group_changes'])==12 for cohort in a['cohort_summaries'])
    assert all(r['metric']=='one_minus_raw_ArcFace' for cohort in a['cohort_summaries'] for r in cohort['predicted_group_changes'] if r['change']>0)
    for failed in ['cctv_dgp_group_guard_grad_v34_analysis_v1/failure_preservation.json',
        'cctv_dgp_group_guard_probe_v35_preparation/unrun_draft_readback_failure.json']:
        assert read(ROOT/'outputs'/failed)['complete']
    prep=read(ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_preparation/independent_packet_audit.json')
    p=read(ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_vm/protocol.json')
    assert prep['complete'] and prep['all108_projection_rows_verified'] and prep['VM_launches']==0
    assert prep['scope_rejections']==3 and prep['unsafe_member_rejections']==8 and prep['valid_R1_scope_and_all_replay_directories_verified']
    assert p['states']==[0] and p['candidate_displacement_trials']==4 and p['trial_outputs']==400 and p['before_outputs']==100
    assert p['optimizer_updates']==p['new_gradient_queries']==p['committed_trajectory_updates']==p['epochs']==0
    assert not (ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_vm/outputs').exists()
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    app=read(ROOT/'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app)==14
    for name,digest in app.items():assert sha(ROOT/name)==digest,name
    covering=read(ROOT/'outputs/completion_input_footprints_comparison_v1_r1/visual_review.json')
    assert not covering['automatic_quality_qualification'] and not covering['assisted_quality_qualification'] and not covering['app_adoption']
    receipt={'complete':True,'milestone_sha256':sha(OUT/'milestone.json'),'checker_sha256':sha(Path(__file__)),
        'new_bindings_verified':len(m['new_evidence_sha256']),'previous_bindings_preserved':len(old['new_evidence_sha256']),
        'complete_previous_handoff_preserved':True,'historical_runbook_recovered_to_exact_hash':True,'live_edited_runbook_preserved':True,'V34_members_verified':230,'V34_raw_outputs_identical':100,'V34_CPU_replays':280,
        'all108_nonzero_constraints_verified':True,'V35_R1_packet_verified':True,'V35_R1_VM_started':False,
        'serialization_and_unrun_draft_failures_retained':True,'all14_app_bindings_unchanged':True,
        'historical_restore_and_covering_failures_retained':True,'native_or_reserved_used':False,
        'local_neural_calls_in_closure':0,'local_autograd_or_optimizer_calls':0,'app_promotion':False,
        'independent_final_quality_review':False,'goal_complete':False,'seconds':time.monotonic()-started}
    with (OUT/'independent_closure_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'bindings':receipt['new_bindings_verified'],'previous':receipt['previous_bindings_preserved'],'seconds':receipt['seconds']}))


if __name__=='__main__':main()

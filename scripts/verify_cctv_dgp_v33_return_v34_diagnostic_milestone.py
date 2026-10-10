"""Independent milestone readback, failure preservation and unchanged-app checks."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone'
PREVIOUS=ROOT/'outputs/completion_input_footprints_v1_milestone'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    started=time.monotonic();m=read(OUT/'milestone.json');assert m['complete']
    for n,h in m['new_evidence_sha256'].items():assert sha(ROOT/n)==h,n
    old=read(PREVIOUS/'milestone.json');assert sha(PREVIOUS/'milestone.json')==m['previous_milestone_sha256']
    saved=ROOT/m['previous_handoff_path'];before=saved.read_bytes();current=(ROOT/'PROJECT_HANDOFF.md').read_bytes()
    at=before.index(b'\n')+1;amount=m['document']['addition_bytes']
    assert current[:at]==before[:at] and current[at+amount:]==before[at:]
    assert sha(saved)==m['document']['before_sha256'] and sha(ROOT/'PROJECT_HANDOFF.md')==m['document']['after_sha256']
    for n,h in old['new_evidence_sha256'].items():assert sha(saved if n=='PROJECT_HANDOFF.md' else ROOT/n)==h,n
    imported=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_return_import.json')
    assert imported['complete'] and imported['members']==5914
    for n,h in imported['files_sha256'].items():assert sha(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_return'/n)==h,n
    audit=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json')
    external=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_audit_r2_run_v1/external_receipt.json')
    assert external['complete'] and external['worker_exit_code']==0 and not external['timeout']
    assert audit['complete'] and audit['finite_probe_complete'] and audit['CPU_replay']['outputs']==280
    assert len(audit['finite_output_arithmetic'])==28 and audit['members_verified']==5914
    assert audit['local_gradient_calls']==audit['local_backward_calls']==audit['local_optimizer_updates']==0
    assert audit['quality_gates_and_original_case_tolerances_unchanged'] and audit['original_R1_failure_retained']
    assert audit['CPU_replay']['raw_maximum_error']<=1e-5 and audit['CPU_replay']['PNG_maximum_byte_error']<=1
    a=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1/analysis.json')
    v=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1/visual_review.json')
    assert len(a['variants'])==24 and len(a['before_states'])==4
    assert v['all4_pages_actually_viewed_at_original_detail'] and v['viewed_case_count']==20 and v['viewed_comparison_cells']==160
    assert not v['all1400_visually_reviewed'] and not v['independent_final_review'] and not v['app_adoption']
    for page in a['visual_pages']:assert sha(ROOT/page['path'])==page['sha256']
    assert all(r['preservation_against_original']['failures'] for r in a['variants'] if r['state']==0)
    assert all(r['preservation_against_original']['failures'] for r in a['variants'] if r['state']==50 and r['cohort']=='unexposed')
    before_bad=next(r for r in a['before_states'] if r['state']==50 and r['cohort']=='unexposed')['preservation_against_original']['failures']
    assert {(r['group'],r['metric']) for r in before_bad}=={('dataset/asian_faces/blur_lr24','ArcFace_observed_fixed'),('dataset/thumbnails128x128/compound_lr24','ArcFace_observed_fixed')}
    cone=next(r for r in a['variants'] if r['state']==50 and r['cohort']=='unexposed' and r['variant']=='cone_1')
    assert cone['new_failure_keys_relative_to_before']==[['dataset/asian_faces/clear','ArcFace_observed_fixed']]
    import numpy as np
    for cohort in ['exposed','unexposed']:
        g=np.load(ROOT/f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state0_{cohort}/gradient_components.npy',allow_pickle=False)
        assert g.shape==(7,978243) and np.count_nonzero(g[3:])==0
    p=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/protocol.json')
    prep=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_preparation/preparation_receipt.json')
    checked=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_preparation/independent_packet_audit_r1.json')
    assert checked['complete'] and checked['protocol_sha256']==prep['protocol_sha256']==sha(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/protocol.json')
    assert p['gradient_queries']==300 and p['optimizer_updates']==p['parameter_updates']==0 and p['states']==[0]
    assert checked['actual_gradient_queries']==checked['actual_optimizer_updates']==checked['actual_neural_calls']==checked['VM_launches']==0
    assert checked['Bash_readonly_syntax_verified'] and checked['Python310_syntax_verified']
    assert checked['wrong_root_and_instance_rejections']==3 and checked['unsafe_archive_boundary_regressions']==8
    assert checked['group_cancellation_reorder_count_regressions']==6
    assert not (ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/outputs').exists()
    for n,h in p['local_basis_sha256'].items():assert sha(ROOT/n)==h,n
    app=read(ROOT/'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app)==14
    for n,h in app.items():assert sha(ROOT/n)==h,n
    covering=read(ROOT/'outputs/completion_input_footprints_comparison_v1_r1/visual_review.json')
    assert not covering['automatic_quality_qualification'] and not covering['assisted_quality_qualification'] and not covering['app_adoption']
    closure={'complete':True,'milestone_sha256':sha(OUT/'milestone.json'),'checker_sha256':sha(Path(__file__)),
        'new_bindings_verified':len(m['new_evidence_sha256']),'previous_bindings_preserved':len(old['new_evidence_sha256']),
        'full_previous_handoff_preserved':True,'V33_members_verified':5914,'V33_saved_outputs_audited':1400,'V33_CPU_replays':280,
        'V33_all24_comparison_failures_retained':True,'original_zero_preservation_gradients_verified':True,
        'V34_prepared_queries':300,'V34_started':False,'all14_app_bindings_unchanged':True,
        'historical_covering_family_failures_retained':True,'local_neural_calls_in_closure':0,'local_gradient_or_optimizer_calls':0,
        'app_promotion':False,'independent_final_quality_review':False,'goal_complete':False,'seconds':time.monotonic()-started}
    with (OUT/'independent_closure_audit.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(closure,indent=2)+'\n')
    print(json.dumps({'complete':True,'bindings':closure['new_bindings_verified'],'previous':closure['previous_bindings_preserved'],'seconds':closure['seconds']}))


if __name__=='__main__':main()

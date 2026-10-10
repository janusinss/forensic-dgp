"""Independent full retention closure after the human endpoint diagnostic return."""
from pathlib import Path
import hashlib
import sys
import time
import numpy as np
from cctv_dgp_spatial_fit_v40_contract import read, write, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_v40_signal_return_v41_prepared_milestone'


def verify(mapping,start,handoff=None,overrides=None):
    for name,digest in mapping.items():
        path=ROOT/handoff if name=='PROJECT_HANDOFF.md' and handoff else ROOT/(overrides or {}).get(name,name)
        assert path.resolve().is_relative_to(ROOT) and sha(path)==digest,name
        assert time.monotonic()-start<600


def main():
    start=time.monotonic();m=read(OUT/'milestone.json');assert m['complete'];verify(m['new_evidence_sha256'],start)
    before=(ROOT/m['previous_handoff_path']).read_bytes();current=(ROOT/'PROJECT_HANDOFF.md').read_bytes();at=before.index(b'\n')+1
    assert hashlib.sha256(before).hexdigest()==m['document']['before_sha256']
    assert current[:at]==before[:at] and current[at+m['document']['addition_bytes']:]==before[at:]
    folders=['cctv_dgp_v40_learning_signal_v1_vm_availability_milestone','cctv_dgp_v40_learning_signal_v1_milestone',
        'cctv_dgp_v40_return_completion_milestone','cctv_dgp_v39_return_v40_prepared_milestone','cctv_dgp_v39_prepared_milestone_v1',
        'cctv_dgp_v38_return_development_milestone_v1','cctv_dgp_v37_return_v38_probe_milestone',
        'cctv_dgp_v36_return_v37_diagnostic_milestone','completion_conditioning_union_v1_milestone',
        'cctv_dgp_v35_return_v36_probe_milestone','cctv_dgp_v34_return_v35_probe_milestone','cctv_dgp_v33_return_v34_diagnostic_milestone']
    child=m;counts=[]
    for folder in folders:
        parent=read(ROOT/'outputs'/folder/'milestone.json');assert child['previous_milestone_sha256']==sha(ROOT/'outputs'/folder/'milestone.json')
        verify(parent['new_evidence_sha256'],start,child['previous_handoff_path'],child.get('previous_file_overrides'))
        closed=read(ROOT/'outputs'/folder/'independent_closure_audit.json');assert closed['complete'] and closed['milestone_sha256']==sha(ROOT/'outputs'/folder/'milestone.json')
        counts.append(len(parent['new_evidence_sha256']));child=parent
        print({'historical_folder':folder,'bindings_checked':counts[-1]},flush=True)
    assert counts==[14,578,28136,11283,536,9157,734,2562,539,2230,316,5961]
    diagnostic=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_independent_audit.json')
    imported=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return_import.json');first=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return_environment_failure/return_import.json')
    assert diagnostic['complete'] and diagnostic['diagnostic_complete'] and diagnostic['members_verified']==imported['members']==first['members']==957
    assert imported['files_sha256']==first['files_sha256'] and not imported['returned_code_executed']
    assert diagnostic['gradient_readback']['component_queries_readback']==280 and diagnostic['gradient_readback']['tensor_partitions']==57
    assert diagnostic['saved_composition']['raw_compositions']==200 and diagnostic['saved_composition']['PNG_compositions_exact']==300
    assert diagnostic['CPU_replay']['cases']==40 and diagnostic['CPU_replay']['states_unchanged']
    assert diagnostic['local_gradient_calls']==diagnostic['local_optimizer_updates']==diagnostic['VM_calls']==0
    assert not diagnostic['training_capacity_pass'] and not diagnostic['goal_complete']
    passed=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return_audit_execution_r1.json');failed=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return_audit_execution.json')
    assert passed['complete'] and passed['checker_changed'] is False and failed['exit_code']==1
    assert 'ModuleNotFoundError: No module named' in (ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return_audit.log').read_text()
    old=ROOT/'scripts/verify_cctv_dgp_v40_learning_signal_v1_analysis.py';new=ROOT/'scripts/verify_cctv_dgp_v40_learning_signal_v1_analysis_r1.py'
    needle="            assert row['identity_to_landmark_gradient_norm_ratio'] == norms[6]/norms[0]"
    replacement="            np.testing.assert_allclose(row['identity_to_landmark_gradient_norm_ratio'], norms[6]/norms[0], rtol=2e-12, atol=0)"
    assert new.read_text().replace(replacement,needle)==old.read_text()
    original_failure=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis/independent_analysis_original_failure.json')
    assert original_failure['original_checker_sha256']==sha(old) and max(v['absolute_difference'] for v in original_failure['diagnosis'])==8.881784197001252e-16
    rejected=False
    try:np.testing.assert_allclose(1.00001,1.,rtol=2e-12,atol=0)
    except AssertionError:rejected=True
    assert rejected
    a=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis/analysis.json');checked=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis/independent_analysis_audit.json')
    assert checked['complete'] and checked['analysis_sha256']==sha(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis/analysis.json')
    assert checked['saved_batches']==40 and checked['exact_unscaled_sheet_cells']==200 and a['all40_saved_batches_first_order_nonincrease']
    visual=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis/visual_review.json')
    assert visual['all50_optimized_TRAIN_cases_and200_cells_actually_viewed'] and len(visual['pages'])==10
    assert visual['analysis_sha256']==sha(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis/analysis.json')
    assert [cid for page in visual['pages'] for cid in page['ids']]==read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm/protocol.json')['cohorts']['optimized']
    assert visual['all_visible_features_considered']==['eyes','nose','mouth','face_outline','visible_appearance']
    assert not visual['convincing_incremental_structure_gain'] and not visual['input_only_criteria_or_case_labels_changed'] and not visual['app_promotion']
    p=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41/protocol.json');old_p=read(ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json')
    packet=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_preparation/independent_packet_audit.json');prepared=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_preparation/prepared.json')
    assert packet['complete'] and prepared['complete'] and packet['protocol_sha256']==prepared['protocol_sha256']==sha(ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41/protocol.json')
    assert packet['members_checked']==prepared['packet_files']==5493 and prepared['archive_bytes']==444567499
    verify(p['sources_sha256'],start)
    assert p['retained_capacity_gates']==old_p['retained_capacity_gates'] and p['cases']==old_p['cases'] and p['optimizer']==old_p['optimizer']
    assert p['gradient_combination']['queries_bound']==5600 and len(p['gradient_orders'])==800
    assert len(packet['saved_step_corruption_rejections'])==7 and packet['unsafe_return_archive_rejections']==9 and packet['scope_rejections']==3
    assert packet['Windows_pre_neural_rejection'] and packet['Bash_readonly_syntax'] and packet['prepared_only']
    assert not (ROOT/'outputs/cctv-dgp-pcgrad-fit-v41-results.tar.gz').exists() and not (ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41/outputs').exists()
    stop=read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json');assert stop['failure_retained'] and not stop['necessary_capacity_pass'] and not stop['gates'][0]['pass']
    completion=read(ROOT/'outputs/completion_feature_fusion_off_v1/visual_review.json')
    assert not completion['app_adoption'] and not completion['automatic_quality_qualification'] and not completion['assisted_quality_qualification']
    dev=read(ROOT/'outputs/cctv_dgp_v38_quarter_paired_development_v1/saved_output_audit.json');assert len(dev['diagnostic_preservation_failures'])==1
    app=read(ROOT/'outputs/completion_feature_fusion_off_v1/protocol.json')['app_preservation_sha256'];assert len(app)==14;verify(app,start)
    assert m['human_VM_diagnostic_audited'] and m['diagnostic_optimizer_updates']==0 and m['V41_prepared_only'] and not m['V41_training_started']
    assert m['local_gradient_calls']==m['local_optimizer_updates']==m['VM_calls_here']==0
    assert not m['app_promotion'] and not m['independent_final_review'] and not m['goal_complete'] and 'torch' not in sys.modules
    assert time.monotonic()-start<600
    write(OUT/'independent_closure_audit.json',{'complete':True,'milestone_sha256':sha(OUT/'milestone.json'),'checker_sha256':sha(Path(__file__)),
        'new_bindings_verified':len(m['new_evidence_sha256']),'historical_bindings_preserved':counts,'entire_previous_handoff_preserved':True,
        'human_diagnostic_return_and_all280_saved_queries_bound':True,'all50_optimized_TRAIN_visual_notes_bound':True,
        'original_environment_and_exact_ratio_failures_retained':True,'unchanged_prospective_return_checker_and_narrow_analysis_R1_verified':True,
        'V41_distinct_PCGrad_treatment_prepared_not_run':True,'all_original_gate_split_and_checkpoint_bindings_retained':True,
        'all14_DGP_primary_app_bindings_unchanged':True,'VM_calls_here':0,'local_gradient_calls':0,'local_optimizer_updates':0,
        'app_promotion':False,'independent_final_review':False,'goal_complete':False,'seconds':time.monotonic()-start,'cap_seconds':600})
    print({'complete':True,'new_bindings':len(m['new_evidence_sha256']),'historical':counts,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()

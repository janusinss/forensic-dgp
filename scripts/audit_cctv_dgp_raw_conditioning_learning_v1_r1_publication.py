"""Check published numbers, retained failed parity and exact history boundaries."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

from import_cctv_dgp_raw_conditioning_learning_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write
from cctv_dgp_raw_conditioning_learning_v1_r1_contract import decision


def main():
    folder=ANALYSIS/'publication'; published=read(folder/'publication.json')
    assert published['complete'] and not published['original_prospective_checker_passed']
    assert published['returned_accepted_updates']==0 and published['actual_rejected_trials']==3
    assert not any(published[k] for k in ['more_epochs_qualified','new_runnable_packet','model_qualified','goal_complete'])
    assert len(published['documents'])==6 and len(published['new_documents_sha256'])==2
    for name,row in published['documents'].items():
        backup=Path(row['backup']);assert backup.resolve().is_relative_to(folder.resolve()) and backup.name==name
        assert sha(backup)==row['before_sha256'] and sha(ROOT/name)==row['after_sha256']
        assert (ROOT/name).read_bytes()[row['prefix_bytes']:]==backup.read_bytes()
    for name,h in {**published['evidence_sha256'],**published['new_documents_sha256']}.items(): assert sha(ROOT/name)==h,name
    imported=read(ANALYSIS/'import.json'); audited=read(ANALYSIS/'failure_forensic_audit.json')
    prediction=read(ANALYSIS/'gradient_prediction_review.json');visual=read(ANALYSIS/'visual_review.json')
    assert all(v['complete'] for v in [imported,audited,prediction,visual])
    assert published['returned_protocol_sha256']==PIN==imported['protocol_sha256']==sha(RETURN/'protocol.json')
    assert published['returned_archive_sha256']==imported['archive_sha256']=='b589020e0fe69ebe97e2063008c4048f9c52c5db3862fc31c040cfa17eed9e4a'
    assert imported['archive_bytes']==862975919 and imported['returned_file_bindings']==632 and imported['parent_file_bindings']==5852
    out=RETURN/'outputs/raw_conditioning_learning_v1_r1'
    assert not (out/'run_results.json').exists() and not list(out.glob('accepted*'))
    assert audited['failure']['counts']==dict(B=100,recognizer=180,gradient_queries=320,backwards=0,accepted_updates=0,parameter_proposals=3)
    assert audited['completed_snapshots_audited']==['initial','r00_trial00','r00_trial01','r00_trial02']
    assert audited['accepted_trials_verified']==[] and not audited['original_prospective_checker_passed']
    assert not audited['cross_run_derivative_parity_pass'] and audited['original_cross_run_bound_not_relaxed']
    assert audited['underlying_scientific_metric_replay_bound_unchanged']==1e-13
    original=ROOT/'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_r1_return.py'
    assert sha(original)==audited['original_checker_sha256']==sha(ANALYSIS/'prospective_checker_original.py')
    prepared=read(ANALYSIS/'forensic_checker_preparation.json')
    recreated=original.read_text(encoding='utf-8')
    for row in prepared['exact_source_replacements']:
        assert recreated.count(row['before'])==1
        recreated=recreated.replace(row['before'],row['after'],1)
    supplement=ROOT/'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_r1_forensic_return.py'
    assert recreated==supplement.read_text(encoding='utf-8') and sha(supplement)==audited['checker_sha256']
    parity=audited['gradient_audits'][0]['cross_run_original_bound_results']
    assert len(parity)==32 and sum(r['elements'] for r in parity)==82587648
    assert sum(r['mismatched_elements'] for r in parity)==1 and all(r['norm_check_pass'] for r in parity)
    missed=[r for r in parity if not r['original_parity_check_pass']];assert len(missed)==1
    assert missed[0]['key']=='RAW/dataset/thumbnails128x128/lowlight_lr32/ArcFace_loss'
    v=missed[0]['violations'][0];assert v['index']==2485488 and v['absolute_difference']>v['original_bound']
    regression=read(ANALYSIS/'cross_run_report_check.json')
    assert not regression['known_original_stop_still_fails']['original_parity_check_pass']
    assert regression['invalid_inputs_rejected']==['NaN','Inf','shape']
    assert visual['all20_sheets_and600_cells_directly_viewed'] and len(visual['sheets'])==20
    assert sum(r['cells'] for r in visual['sheets'])==600
    assert {r['file']:r['sha256'] for r in visual['sheets']}=={r['file']:r['sha256'] for r in audited['all_planned_visual_sheets']}
    assert not visual['native_CCTV_used'] and not visual['independent_final_review'] and not visual['automatic_or_assisted_completion_qualification']
    for row in visual['sheets']:
        assert sha(ROOT/row['file'])==row['sha256'] and row['all30_cells_directly_viewed']
        assert row['requested_image_detail']=='original' and row['observations']
    report=(ROOT/'CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_R1_RESULTS.md').read_text(encoding='utf-8')
    baseline=read(out/'initial/snapshot.json')['groups']; gradient=read(out/'round00/gradients.json')
    gram=np.asarray(gradient['certificate']['normalized_Gram']);eigen=np.linalg.eigvalsh(gram)
    assert np.isclose(eigen[-1]/eigen[0],prediction['RAW32_Gram_condition_number'],rtol=1e-12)
    assert prediction['original_cross_run_derivative_parity_pass'] is False
    assert len(prediction['RAW_group_trial_comparisons'])==96
    for t in prediction['trials']:
        saved=read(out/t['trial']/'snapshot.json')['groups'];receipt=read(out/t['trial']/'decision.json')
        assert decision(baseline,baseline,saved)==receipt['decision']
        assert not receipt['decision']['accept_step'] and not receipt['decision']['one_percent_checkpoint_pass']
        for s in ['raw','png']:
            value=receipt['decision']['stages'][s]
            assert t['stages'][s]['actual_global_feature_gain']==value['relative_feature_gain']
            assert t['stages'][s]['failed_checks']==len(value['initial_preservation_failures'])
            assert f"{value['relative_feature_gain']*100:.6f}%" in report
        for item in [r for r in prediction['RAW_group_trial_comparisons'] if r['trial']==t['trial']]:
            fields={'MSE':'MSE','SSIM_loss':'SSIM','ArcFace_loss':'ArcFace_observed_fixed','feature_MSE':'landmark_high_frequency_MSE'}
            key=item['source']+'/'+item['profile'];field=fields[item['metric']]
            before,after=baseline['raw'][key][field],saved['raw'][key][field]
            if item['metric'] in ['SSIM_loss','ArcFace_loss']:before,after=1-before,1-after
            assert item['before_loss']==before and item['after_loss']==after
            assert item['actual_loss_delta']==after-before and item['actual_rounded_parameter_change_dot']<0
    assert '**did not pass**' in report and 'zero updates were accepted' in report.lower()
    design=(ROOT/'CCTV_DGP_POST_RAW_CONDITIONING_REVIEW_V1.md').read_text(encoding='utf-8')
    assert 'no runnable' in design.lower() or 'not a runnable' in design.lower() or 'No such verified transfer packet' in design
    assert 'https://docs.pytorch.org/docs/2.9/notes/gradcheck.html' in design
    assert 'https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html' in design
    for name in [*published['documents'],*published['new_documents_sha256']]:
        for target in re.findall(r'\]\(<(C:/[^>]+)>\)',(ROOT/name).read_text(encoding='utf-8')[:12000]):assert Path(target).is_file(),target
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    for file in [Path(__file__),ROOT/'scripts/publish_cctv_dgp_raw_conditioning_learning_v1_r1_results.py',supplement]:
        ast.parse(file.read_text(encoding='utf-8'),feature_version=(3,10))
    write(folder/'publication_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__),publication_sha256=sha(folder/'publication.json'),
        exact_history_documents_verified=6,all20_sheets600_cells_hash_bound=True,
        all_actual_zero_updates_and_three_rejected_trials_verified=True,
        original_checker_stop_and_exact_supplement_differences_retained=True,original_cross_run_bound_still_failed=True,
        scientific_quality_thresholds_and_metric_replay_bounds_unchanged=True,
        original_protected_app_checkpoint_unchanged=True,new_runnable_packet=False,more_epochs_qualified=False,model_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=6,reviewed_sheets=20,accepted_updates=0,original_cross_run_parity='failed and retained',new_runnable_packet=False))


if __name__=='__main__':main()

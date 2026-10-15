"""Verify published result against retained failures, exact histories and every visual sheet."""
from datetime import datetime,timezone
from pathlib import Path
import re
import numpy as np

from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import ROOT,OUT,RETURN,ANALYSIS,PIN,read,sha,write


def main():
    stage=ANALYSIS/'publication';published=read(stage/'publication.json')
    assert published['complete'] and published['protocol_sha256']==PIN
    assert len(published['documents'])==10 and len(published['new_documents_sha256'])==2
    assert all(sha(ROOT/n)==h for n,h in published['evidence_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in published['new_documents_sha256'].items())
    manifest=read(RETURN/'export_manifest.json');assert len(manifest['files'])==1144 and manifest['accepted_updates']==0 and manifest['failure_present']
    assert not manifest['complete_pilot'] and all(sha(RETURN/n)==h for n,h in manifest['files'].items())
    audit=read(ANALYSIS/'failure_audit.json');analysis=read(ANALYSIS/'finite_response_review.json');visual=read(ANALYSIS/'visual_review.json')
    assert audit['complete'] and not audit['completed_pilot'] and audit['accepted_trials']==[] and audit['trials_replayed']==3
    assert len(audit['snapshot_labels'])==7 and len(audit['complete_gradient_sites'])==3 and audit['certificate_replay']['all96_original_dot_inequalities_pass']
    out=RETURN/'outputs/nearby_gradient_learning_v1';base=read(out/'initial/snapshot.json')['groups']
    assert len(analysis['initial_RAW_group_trial_comparisons'])==96 and len(analysis['nearby_gradient_pairs'])==32
    gains=[];failures=[]
    for i,t in enumerate(analysis['trials']):
        saved=read(out/f'trial{i:02d}/decision.json');snapshot=read(out/f'trial{i:02d}/snapshot.json')
        assert not saved['decision']['accept_step'] and not saved['decision']['one_percent_checkpoint_pass']
        counts=[]
        for s in ['raw','png']:
            gain=1-snapshot['groups'][s]['degraded']['landmark_high_frequency_MSE']/base[s]['degraded']['landmark_high_frequency_MSE']
            assert gain==t['stages'][s]['global_structure_gain'] and gain<.01
            fs=saved['decision']['stages'][s]['initial_preservation_failures'];assert fs==t['stages'][s]['failures']
            counts.append(len(fs));gains.append(gain)
        failures.append(counts)
    assert failures==[[32,32],[24,25],[14,9]]
    np.testing.assert_allclose(gains,[.006874946900924606,.0070691725734144395,.0057860369024899105,.005978446194028875,.003452585466485081,.0036102051191142515],rtol=0,atol=0)
    assert analysis['previous_RAW32_direction_cosine']==.9995393076560194
    parity=[r for r in audit['original_initial_gradient_parity_reports'] if not r['original_parity_check_pass']]
    assert len(parity)==1 and parity[0]['mismatched_elements']==1 and all(r['norm_check_pass'] for r in audit['original_initial_gradient_parity_reports'])
    assert not audit['all_original_initial_parity_checks_pass']
    regression=read(ANALYSIS/'retained_failure_regression.json');assert regression['complete'] and len(regression['actual_three_trial_rejections_checked'])==3
    assert regression['actual_original_parity_miss_reproduced']['mismatched_elements']==1 and not regression['actual_original_parity_miss_reproduced']['original_parity_check_pass']
    assert visual['all20_sheets_and900_cells_directly_viewed'] and len(visual['reviewed_sheets'])==20
    assert sum(r['cells'] for r in visual['reviewed_sheets'])==900
    for v,a in zip(visual['reviewed_sheets'],audit['visual_sheets']):
        assert v['file']==a['file'] and v['sha256']==a['sha256']==sha(ANALYSIS/v['file'])
        assert v['direct_original_resolution_review'] and v['all_five_rows_and_nine_columns_reviewed']
        assert not v['useful_visible_gain_over_initial_established']
    r=read(out/'rollback.json');assert r['complete'] and r['fresh_state']==r['initial_model_state'] and r['RNG_restored']
    with np.load(out/'initial_RNG.npz',allow_pickle=False) as a,np.load(out/'restored_RNG.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[n],b[n]) for n in a.files)
    assert not list(out.glob('accepted*')) and read(out/'failure_restoration.json')['complete']
    for n,b in published['documents'].items():
        before=Path(b['backup']);assert sha(before)==b['before_sha256'] and sha(ROOT/n)==b['after_sha256']
        assert (ROOT/n).read_bytes()[b['prefix_bytes']:]==before.read_bytes()
        prefix=(ROOT/n).read_bytes()[:b['prefix_bytes']].decode('utf-8')
        assert prefix.startswith('# Nearby-gradient V1 R1 audited and rejected') and 'USED / FAILED' in prefix and 'Goal incomplete.' in prefix
    old=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_publication/publication.json')['exact_command_blocks']
    guide=(ROOT/'CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_R1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'```(?:bat|bash)\n(.*?)\n```',guide,re.S)==list(old.values())
    report=(ROOT/'CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_R1_RESULTS.md').read_text(encoding='utf-8')
    assert '| trial00 | 0.687495% | 0.706917% | 32 / 32 | Rejected |' in report
    assert '| trial02 | 0.345259% | 0.361021% | 14 / 9 | Rejected |' in report
    assert '0.999539308' in report and '2,163,166,224' in report and '0b9fc3fb95c2e1d4334165fd9024e32fa7fcb57e1142f8c07fd46471d4f156d3' in report
    links=[]
    for n in list(published['documents'])+list(published['new_documents_sha256']):
        prefixbytes=published['documents'][n]['prefix_bytes'] if n in published['documents'] else None
        text=(ROOT/n).read_bytes()[:prefixbytes].decode('utf-8')
        for target in re.findall(r'\]\(<([^>]+)>\)',text):assert Path(target).is_file(),target;links.append(target)
    question=read(ANALYSIS/'diagnostic_question.json');assert question['human_answer_pending'] and question['human_answer'] is None
    assert question['training_recipe_modification_stopped'] and len(question['related_unsuccessful_recipes'])==3 and sha(ROOT/'AGENTS.md')==question['source_rule_sha256']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256'];assert len(protected)==12 and all(sha(ROOT/n)==h for n,h in protected.items())
    assert published['next_training_recipe_not_selected_or_prepared'] and not published['automatic_VM_launch'] and published['local_neural_or_training_calls']==0
    write(stage/'publication_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=PIN,publication_sha256=sha(stage/'publication.json'),all1144_return_bindings_rechecked=True,
        all10_history_suffixes_and2_new_reports_checked=True,all5_old_manual_blocks_unchanged=True,
        exact_three_trial_RAW_PNG_gain_and_failure_counts_recounted=failures,
        all20_original_resolution_visual_sheets_bound=True,actual_one_parity_bound_miss_retained=True,
        original_RNG_and_protected12_files_exact=True,verified_new_local_links=len(links),
        required_human_diagnostic_question_pending=True,next_training_recipe_not_selected_or_prepared=True,
        local_neural_or_training_calls=0,automatic_VM_launch=False,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=10,new_reports=2,rejections=failures,sheets=20,protected=True,human_diagnostic_question_pending=True))


if __name__=='__main__':main()

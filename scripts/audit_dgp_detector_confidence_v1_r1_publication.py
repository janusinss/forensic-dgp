"""Recount published detector evidence using saved arrays, with no neural forward."""
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
from PIL import Image

from audit_dgp_detector_confidence_v1_r1 import ROOT, OUT, read, sha, write


def main():
    stage=OUT/'publication';published=read(stage/'publication.json')
    p=read(OUT/'protocol.json');audit=read(OUT/'audit.json');result=read(OUT/'results.json')
    summary=read(OUT/'summary.json')['summary'];visual=read(OUT/'visual_review.json')
    assert published['complete'] and published['protocol_sha256']==sha(OUT/'protocol.json')
    assert len(published['documents'])==5 and len(published['new_documents_sha256'])==1
    assert len(p['assets_sha256'])==118 and all(sha(ROOT/n)==h for n,h in p['assets_sha256'].items())
    assert len(result['artifacts_sha256'])==114 and all(sha(OUT/n)==h for n,h in result['artifacts_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in published['evidence_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in published['new_documents_sha256'].items())
    assert audit['complete'] and audit['results_sha256']==sha(OUT/'results.json')
    assert audit['all36_prior_app_proposal_pixels_and_statistics_exact'] and audit['all6_sheet_cells_match_bound_inputs_outputs']
    assert result['counts']==dict(detector=36,completion=0,restorer=0,gradients=0,optimizer_updates=0)
    assert result['detector_model_state_before']==result['detector_model_state_after']
    assert result['detector_container_includes_unused_generator_weights'] and result['generator_forward_calls']==0
    assert not result['native_CCTV_or_reserved_final_pixels_used']
    eligible={c:[] for c in ['original_photo','synthetic_degraded_photo']}
    empty={c:[] for c in eligible};controls=[];exclusions=[];zero_seed=[]
    by_id={r['id']:r for r in result['rows']}
    pairs={}
    for c in p['cases']:
        row=by_id[c['id']];condition=c['condition']
        assert condition in eligible and row['condition']==condition
        probability=np.load(OUT/row['probability'],allow_pickle=False)
        with Image.open(OUT/row['raw']) as im:raw=np.asarray(im)==255
        with Image.open(OUT/row['proposal']) as im:proposal=np.asarray(im)==255
        with Image.open(ROOT/c['reviewed']) as im:operator=np.asarray(im)==255
        np.testing.assert_array_equal(raw,probability>=np.float32(.5))
        assert int(raw.sum())==row['raw_pixels'] and int(proposal.sum())==row['proposal_pixels']
        assert int(operator.sum())==row['operator_pixels']
        assert raw.any()==proposal.any()
        assert c['clear_control']==next(r['clear_control'] for r in audit['rows'] if r['id']==c['id'])
        stem=c['id'][:-len('_native')] if condition=='original_photo' else c['id'][:-len('_degraded')]
        pair=pairs.setdefault(stem,{})
        assert condition not in pair;pair[condition]=(c,operator,proposal)
        if c['input_review']!='usable':
            exclusions.append(c['id'])
        elif c['clear_control']:
            assert not operator.any();controls.append(dict(id=c['id'],condition=condition,
                raw_pixels=int(raw.sum()),proposal_pixels=int(proposal.sum())))
        else:
            assert operator.any();eligible[condition].append(c['id'])
            if not proposal.any():empty[condition].append(c['id'])
            if proposal.any() and not (raw&operator).any():zero_seed.append(c['id'])
    assert len(pairs)==18 and len(controls)==4 and len(exclusions)==4
    for pair in pairs.values():
        assert set(pair)==set(eligible)
        a,b=pair['original_photo'],pair['synthetic_degraded_photo']
        assert a[0]['input_review']==b[0]['input_review'] and a[0]['clear_control']==b[0]['clear_control']
        np.testing.assert_array_equal(a[1],b[1])
    assert [len(eligible[c]) for c in eligible]==[14,14]
    assert [len(empty[c]) for c in empty]==[4,11]
    for condition in eligible:
        assert summary[condition]['eligible_covering_cases']==len(eligible[condition])
        assert summary[condition]['empty_case_ids']==empty[condition]
        assert summary[condition]['empty_proposal_cases']==len(empty[condition])
        families={c['family'] for c in p['cases'] if c['id'] in eligible[condition]}
        assert set(summary[condition]['recorded_family_tags'])==families
        for family in families:
            ids=[c['id'] for c in p['cases'] if c['id'] in eligible[condition] and c['family']==family]
            assert summary[condition]['recorded_family_tags'][family]==dict(cases=len(ids),empty_proposals=sum(i in empty[condition] for i in ids))
    assert summary['clear_controls']==controls and [r['id'] for r in summary['exclusions']]==exclusions
    assert zero_seed==['val_362_hair_eye_native']
    assert summary['original_photo']['nonempty_proposal_with_no_selected_operator_pixels']==zero_seed
    assert summary['synthetic_degraded_photo']['nonempty_proposal_with_no_selected_operator_pixels']==[]
    assert next(r['proposal_pixels'] for r in controls if r['id']=='08_uncovered_native')==11683
    assert summary['image_views']==36 and summary['matched_photograph_pairs']==18 and not summary['native_CCTV_evidence']
    assert not summary['automatic_or_assisted_family_qualified'] and not summary['model_qualified'] and not summary['goal_complete']
    assert visual['complete'] and visual['all6_sheets_and180_cells_directly_viewed']
    assert not visual['independent_final_human_review'] and len(visual['reviewed_sheets'])==6
    assert sum(r['cells'] for r in visual['reviewed_sheets'])==180
    for observed,planned in zip(visual['reviewed_sheets'],audit['planned_visual_sheets']):
        assert observed['file']==planned['file'] and observed['sha256']==planned['sha256']==sha(OUT/observed['file'])
        assert observed['case_ids']==planned['case_ids'] and observed['direct_original_resolution_review']
        assert observed['all_six_rows_and_five_columns_reviewed'] and observed['cells']==30 and observed['observation']
    old=ROOT/'outputs/dgp_detector_confidence_v1';old_p=read(old/'protocol.json')
    assert all(sha(ROOT/n)==h for n,h in old_p['assets_sha256'].items())
    failure=read(old/'failure.json');assert failure['type']=='ModuleNotFoundError' and failure['message']=="No module named 'psutil'"
    assert not any(failure['counts'].values()) and read(old/'supervisor.json')['child_exit_code']==1
    assert not (old/'results.json').exists() and read(old/'failure_audit.json')['complete']
    analysis=ROOT/'outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis'
    question=read(analysis/'diagnostic_question.json')
    assert question['human_answer_pending'] and question['human_answer'] is None and question['training_recipe_modification_stopped']
    protected=read(analysis/'local_protected_before.json')['files_sha256']
    assert len(protected)==12 and all(sha(ROOT/n)==h for n,h in protected.items())
    assert read(analysis/'failure_audit.json')['accepted_trials']==[]
    links=[]
    for n,b in published['documents'].items():
        before=Path(b['backup'])
        assert sha(before)==b['before_sha256'] and sha(ROOT/n)==b['after_sha256']
        current=(ROOT/n).read_bytes();assert current[b['prefix_bytes']:]==before.read_bytes()
        text=current[:b['prefix_bytes']].decode('utf-8')
        assert text.startswith('# Current status: detector confidence audited; restorer discussion pending')
        assert 'USED / FAILED' in text and 'Goal incomplete.' in text and '11/14' in text
        for target in re.findall(r'\]\(<([^>]+)>\)',text):assert Path(target).is_file();links.append(target)
    report=(ROOT/'CCTV_DGP_DETECTOR_CONFIDENCE_V1_R1_RESULTS.md').read_text(encoding='utf-8')
    for fragment in ['| Original photographs | 14 | 4 |','| Matched synthetic degradations | 14 | 11 |',
        '**not native CCTV**','11,683','all 180','unused generator tensors','Goal incomplete.']:
        assert fragment in report,fragment
    for target in re.findall(r'\]\(<([^>]+)>\)',report):assert Path(target).is_file();links.append(target)
    assert published['human_diagnostic_question_pending'] and published['next_training_recipe_not_selected_or_prepared']
    assert not published['automatic_VM_launch'] and published['publication_neural_or_training_calls']==0
    assert not published['model_qualified'] and not published['goal_complete']
    write(stage/'publication_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),publication_sha256=sha(stage/'publication.json'),
        all118_current_and_original_failure_source_bindings_rechecked=True,all114_output_bindings_rechecked=True,
        all36_probability_arrays_and_raw_masks_recounted=True,paired_operator_masks_exact=True,
        eligible_original_and_degraded_views=[14,14],empty_original_and_degraded_views=[4,11],
        all4_clear_controls_and4_exclusions_separate=True,false_clear_control_proposal_pixels=11683,
        all6_original_resolution_sheet_review_bindings_exact=True,all180_cells_previously_reconstructed=True,
        all5_current_document_prefixes_and_exact_history_suffixes_checked=True,new_report_and_local_links_checked=len(links),
        original_zero_call_startup_failure_preserved=True,restorer_question_pending=True,
        protected12_hashes_exact=True,publication_neural_or_training_calls=0,automatic_VM_launch=False,
        model_qualified=False,goal_complete=False))
    print(dict(complete=True,source_bindings=118,outputs=114,eligible=[14,14],empty=[4,11],sheets=6,documents=5,
               original_failure_preserved=True,model_qualified=False))


if __name__=='__main__':main()

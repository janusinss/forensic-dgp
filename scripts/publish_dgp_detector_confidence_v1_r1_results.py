"""Publish frozen inference evidence without model, mask or training changes."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_dgp_detector_confidence_v1_r1 import ROOT, OUT, read, sha, write
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link

ANALYSIS=ROOT/'outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis'
REPORT='CCTV_DGP_DETECTOR_CONFIDENCE_V1_R1_RESULTS.md'
DOCUMENTS=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
           'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']


def summarize(rows,cases):
    """Keep condition, eligibility, assisted footprint and matched-pair provenance exact."""
    allowed={'original_photo','synthetic_degraded_photo'}
    ids=[c['id'] for c in cases]
    assert len(ids)==len(set(ids)) and len(rows)==len(cases)
    by_id={r['id']:r for r in rows}
    assert len(by_id)==len(rows) and set(by_id)==set(ids)
    pairs={}
    for c in cases:
        r=by_id[c['id']]
        assert c['condition'] in allowed
        for k in ['id','family','condition','input_review','clear_control']:assert r[k]==c[k],k
        assert c['input_review'] in {'usable','out_of_scope','needs_clearer'}
        for k in ['raw_pixels','proposal_pixels','operator_pixels']:
            assert type(r[k]) is int and 0<=r[k]<=256*256
        assert r['proposal_pixels']>=r['raw_pixels']
        assert (r['raw_pixels']==0)==(r['proposal_pixels']==0), 'Margin cannot create a seed'
        op=r['scores']['operator']
        if c['clear_control']:
            assert c['input_review']=='usable' and r['operator_pixels']==0
            assert op['maximum'] is None and op['selected_fraction'] is None
        else:
            assert r['operator_pixels']>0 and 0<=op['maximum']<=1 and 0<=op['selected_fraction']<=1
            assert (op['maximum']<.5)==(op['selected_fraction']==0)
        suffix='_native' if c['condition']=='original_photo' else '_degraded'
        assert c['id'].endswith(suffix)
        pair=pairs.setdefault(c['id'][:-len(suffix)],{})
        assert c['condition'] not in pair
        pair[c['condition']]=r
    for pair in pairs.values():
        assert set(pair)==allowed
        a,b=pair['original_photo'],pair['synthetic_degraded_photo']
        for k in ['family','input_review','clear_control','operator_pixels']:assert a[k]==b[k],k
    result=dict(image_views=len(rows),matched_photograph_pairs=len(pairs),
        native_CCTV_evidence=False,independent_identity_count_not_established=True,
        automatic_generation_outputs=0,assisted_generation_outputs=0,
        automatic_or_assisted_family_qualified=False,model_qualified=False,goal_complete=False)
    for condition in sorted(allowed):
        eligible=[r for r in rows if r['condition']==condition and r['input_review']=='usable' and not r['clear_control']]
        empty=[r['id'] for r in eligible if r['proposal_pixels']==0]
        result[condition]=dict(eligible_covering_cases=len(eligible),empty_proposal_cases=len(empty),
            empty_case_ids=empty,
            nonempty_proposal_with_no_selected_operator_pixels=[r['id'] for r in eligible
                if r['proposal_pixels'] and r['scores']['operator']['selected_fraction']==0],
            recorded_family_tags={f:dict(cases=len(g),empty_proposals=sum(r['proposal_pixels']==0 for r in g))
                for f in sorted({r['family'] for r in eligible})
                for g in [[r for r in eligible if r['family']==f]]})
    result['clear_controls']=[{k:r[k] for k in ['id','condition','raw_pixels','proposal_pixels']}
                             for r in rows if r['clear_control']]
    result['exclusions']=[{k:r[k] for k in ['id','condition','input_review','raw_pixels','proposal_pixels']}
                         for r in rows if r['input_review']!='usable']
    result['eligible_pairs_original_nonempty_degraded_empty']=[stem for stem,pair in pairs.items()
        if pair['original_photo']['input_review']=='usable' and not pair['original_photo']['clear_control']
        and pair['original_photo']['proposal_pixels'] and not pair['synthetic_degraded_photo']['proposal_pixels']]
    return result


def main():
    stage=OUT/'publication';(stage/'before').mkdir(parents=True,exist_ok=True)
    assert not (stage/'publication.json').exists()
    p=read(OUT/'protocol.json');result=read(OUT/'results.json');audit=read(OUT/'audit.json')
    stop=read(OUT/'supervisor.json');old=ROOT/'outputs/dgp_detector_confidence_v1'
    old_audit=read(old/'failure_audit.json');startup=read(ROOT/'outputs/dgp_detector_confidence_v1_r1_startup_review.json')
    assert audit['complete'] and audit['results_sha256']==sha(OUT/'results.json')
    assert result['counts']==dict(detector=36,completion=0,restorer=0,gradients=0,optimizer_updates=0)
    assert stop['child_exit_code']==0 and not stop['timeout'] and stop['seconds']<210
    assert audit['all36_prior_app_proposal_pixels_and_statistics_exact']
    assert audit['source_bindings']==len(p['assets_sha256'])==118
    assert audit['output_bindings']==len(result['artifacts_sha256'])==114
    assert all(sha(ROOT/n)==h for n,h in p['assets_sha256'].items())
    assert all(sha(OUT/n)==h for n,h in result['artifacts_sha256'].items())
    assert old_audit['complete'] and old_audit['failure']['type']=='ModuleNotFoundError'
    assert not any(old_audit['failure']['counts'].values())
    assert startup['original_failure_audit_sha256']==sha(old/'failure_audit.json')
    assert startup['original_recorder_sha256']==sha(ROOT/'scripts/record_dgp_detector_confidence_v1.py')
    assert startup['corrected_recorder_sha256']==sha(ROOT/'scripts/record_dgp_detector_confidence_v1_r1.py')
    old_p=read(old/'protocol.json')
    assert all(sha(ROOT/n)==h for n,h in old_p['assets_sha256'].items())
    checker=read(ROOT/'outputs/dgp_detector_confidence_summary_v1_check.json')
    assert checker['complete'] and len(checker['all_seven_false_cohort_claims_rejected'])==7
    question=read(ANALYSIS/'diagnostic_question.json')
    assert question['human_answer_pending'] and question['human_answer'] is None
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert len(protected)==12 and all(sha(ROOT/n)==h for n,h in protected.items())
    prior=read(ANALYSIS/'publication/publication.json')
    expected={n:prior['documents'][n]['after_sha256'] for n in DOCUMENTS}
    assert all(sha(ROOT/n)==h or ((stage/'before'/n).exists() and sha(stage/'before'/n)==h) for n,h in expected.items())
    summary=summarize(audit['rows'],p['cases'])
    assert summary['image_views']==36 and summary['matched_photograph_pairs']==18
    assert summary['original_photo']['eligible_covering_cases']==summary['synthetic_degraded_photo']['eligible_covering_cases']==14
    assert summary['original_photo']['empty_proposal_cases']==4 and summary['synthetic_degraded_photo']['empty_proposal_cases']==11
    write(OUT/'summary.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        summarizer_sha256=sha(__file__),audit_sha256=sha(OUT/'audit.json'),summary=summary,
        source_footprints_approximate_not_segmentation_truth=True,threshold_sweep_not_performed=True,
        no_unique_architecture_or_training_data_root_cause_proved=True))
    observations=[
        'Cloth and pink mask proposals include fragments and visible-feature/background selections. The degraded pink-mask proposal is empty. The strongly turned mask-with-clear-glasses pair remains out of scope.',
        'Both dark-sunglasses pairs and the white-glare pair have empty raw and delivered proposals. The score maps are visually dark; the operator footprint identifies the approximate covered lens regions, not hidden truth.',
        'Mirrored lenses are partly detected, with holes/extra selections. Hand-over-mask is almost wholly missed. The uncovered original falsely selects background, face boundaries and eye pixels; its degraded pair has no selection.',
        'Clear glasses remain unselected in both views. The original hand-over-eyes image selects portions of fingers plus non-obstructing head eyewear, hair/edges and visible mouth; the degraded proposal is empty. Hand-over-mouth is sparse originally and empty after degradation.',
        'Obstructing hair is missed at the covered eye; the original has a tiny separate hair selection. The two original scarf proposals are fragmented and the knit-scarf proposal reaches visible eye/nose regions. Both degraded scarf proposals are empty.',
        'Flower-over-mouth is undetected in both views. The original leaf proposal mainly selects isolated edges/ordinary hair; the degraded leaf proposal covers much more of the leaf but remains imperfect. Both nearly-hidden-hand views remain needs_clearer and do not qualify generation.',
    ]
    reviewed=[]
    for sheet,observation in zip(audit['planned_visual_sheets'],observations):
        assert sha(OUT/sheet['file'])==sheet['sha256'] and sheet['cells']==30
        reviewed.append(dict(file=sheet['file'],sha256=sheet['sha256'],cells=30,image_size=[1280,1752],
            case_ids=sheet['case_ids'],direct_original_resolution_review=True,
            all_six_rows_and_five_columns_reviewed=True,observation=observation))
    assert len(reviewed)==6 and sum(s['cells'] for s in reviewed)==180
    write(OUT/'visual_review.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        results_sha256=sha(OUT/'results.json'),audit_sha256=sha(OUT/'audit.json'),reviewed_sheets=reviewed,
        all6_sheets_and180_cells_directly_viewed=True,independent_final_human_review=False,
        no_face_completion_outputs_or_hidden_ground_truth=True,automatic_or_assisted_family_qualified=False,
        no_threshold_mask_app_or_model_change=True,model_qualified=False,goal_complete=False))
    table='\n'.join('| '+r['id']+' | '+str(r['proposal_pixels'])+' |' for r in summary['clear_controls'])
    report=f'''# Detector confidence diagnostic V1 R1 — 14 October 2026

**The diagnostic completed; automatic covering detection remains inadequate.**
The unchanged detector made empty proposals on 4/14 eligible original-photo
covering views and 11/14 matched synthetic-degraded views. An uncovered original
also received an incorrect 11,683-pixel proposal. More mask margin cannot fix
empty predictions. No new detector, restorer, completed face or training update
was created. {link('Independent saved-array audit','outputs/dgp_detector_confidence_v1_r1/audit.json')}.

## Evidence and input scope

One frozen CPU inference run recorded 36 score maps, 36 raw masks and 36 app
proposals in **{result['seconds']:.2f} seconds**. These are 18 original photographs
and their matched synthetic degradations, not 36 independent people. Thirty-two
views are input-usable, including four clear controls; only 28 covered views
(14 original/degraded pairs) enter the covering counts. Four excluded views
retain their input-only out-of-scope or needs-clearer decisions. The legacy
`_native` filenames denote original photographs here, **not native CCTV**.
They provide no Zamboanga performance, ethnicity or untouched final evidence.

The reviewed operator masks are approximate removal footprints, shared between
matched views. They are not segmentation ground truth or an aligned clean hidden
face. These sources are already exposed development/training material; external
pretraining overlap is unknown. Current scores are uncalibrated classifier
outputs, not validated probabilities. Small exposed counts describe these exact
cases and cannot establish general population accuracy.

| View condition | Eligible covered views | Empty proposals |
|---|---:|---:|
| Original photographs | 14 | 4 |
| Matched synthetic degradations | 14 | 11 |

{link('All counts, pairs, exclusions and recorded family tags','outputs/dgp_detector_confidence_v1_r1/summary.json')}.
Composite tags such as hand-over-mask and sunglasses-with-glare are retained;
the agreed seven covering families are still required separately.

## Where detection fails

The .5 threshold and documented three-pixel margin at 256×256 were unchanged.
All 36 proposals equal the prior app masks by pixels, and the saved measurements
also reproduce the original evidence. Thus this result explains existing masks;
it does not introduce a new segmentation policy or improve generation.

The four empty original covering cases are the two dark-sunglasses examples,
white lens glare and flower-over-mouth. Eleven degraded cases are empty, spanning
pink mask, sunglasses, glare, hands, obstructing hair, scarves and flower. Their
raw threshold masks are already empty before margin processing. For example,
the operator-region maximum score is 4.6871e-8 on the degraded pink mask and
1.0080e-7 on the degraded knit scarf. These specific misses are not predictions
just below .5. No threshold was fitted or swept on these exposed images.

Inference from the exact masks: binary dilation needs selected seed pixels;
it cannot recover these empty predictions. The low region scores and extra
selections on visible features demonstrate detector scoring/localization
limitations on these cases. They do not establish one unique architectural,
loss or dataset root cause, nor justify guessing a lower threshold. This
diagnostic does not test the completion generator or explain all generated
central defects.

The original obstructing-hair view has a nonempty 70-pixel proposal but **zero
raw selected pixels inside its reviewed obstruction footprint**. A nonempty
proposal alone is not evidence that the covered feature was detected. Mirrored
lenses and the degraded leaf have larger proposals, while other covered views
receive fragments or extra selections on visible eye, mouth, hair and background
regions. All six sheets were directly reviewed at 1280×1752, including all 180
input/score/raw/proposal/operator cells.
{link('Every-sheet observations','outputs/dgp_detector_confidence_v1_r1/visual_review.json')}.

| Clear control | Proposal pixels |
|---|---:|
{table}

The uncovered original's false proposal is 17.826843% of its 256×256 canvas,
including background and visible facial pixels. The clear-glasses pair is
correctly unselected in this small review. Zero overlap with the limited supplied
protected masks does not prove preservation of every visible pixel; this run
generated no faces. Clear glasses and ordinary hair remain protected by scope.

## Location, cause and corrected startup

Location: the original V1 recorder, `scripts/record_dgp_detector_confidence_v1.py:142`.
Cause: unavailable optional `psutil`, before any model import or neural forward.
Fix: a separate R1 recorder uses the built-in Windows process working-set API.
A touched 64 MiB allocation verified that reader. No environment package was
installed. The original frozen protocol, source, traceback, supervisor and zero-
call failure remain intact; R1 keeps the same images, checkpoint, threshold,
mask processing and budgets. The supervisor enforces 210 seconds, with an
internal 180-second/4 GiB memory/32 MiB output limit.
{link('Preserved startup failure','outputs/dgp_detector_confidence_v1/failure_audit.json')}.
{link('Separate repair binding','outputs/dgp_detector_confidence_v1_r1_startup_review.json')}.
[Microsoft process-memory API](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo).

The detector checkpoint is `{result['detector_checkpoint_sha256']}`, trained
container epoch 2; this is **not the primary DGP's epoch count**. The local runtime
was `{result['torch']}` on CPU. The CompletionNet container loads segmenter and
unused generator tensors; only the segmenter executes. The primary restorer and
separate app completion backend were not loaded. Counts are detector 36,
completion 0, restorer 0, gradients 0 and optimizer updates 0.
Before/after state receipts match the checkpoint tensor schema exactly.

The independent checker reconstructs thresholding and the seven-pixel-square
dilation with SciPy, recalculates region statistics and all prior app proposal
comparisons, and reconstructs every image cell. All 118 source and 114 output
bindings pass. It does not repeat neural inference. Prospective array fixtures
reject eight saved-score/mask corruptions; the cohort checker rejects seven
false claims, including relabeling photographs as CCTV and counting excluded
or clear images as covered cases. Completion of these audits does not qualify
the detector or model. Result-time pending flags remain historical; the later
audit and visual sidecars record their actual completion.

## Consequence for the full goal

The automatic detector needs a justified representation/label/degradation
review before any new finite training design. Assisted masks and completion
generator quality remain separate obligations: correcting an area does not
establish a plausible hidden face. No detector training recipe, threshold
change, mask expansion, new restorer, app adoption or epoch extension is selected
by this diagnostic. All automatic and assisted covering families remain
unqualified; original/mask/result previews, downloads and insufficient-
information requests remain required.

The joint, RAW-only and nearby-gradient restoration recipes remain failed.
The required reconstruction-path-versus-gradient design question is still
pending; no fourth training recipe is prepared while awaiting that discussion.
{link('Restorer design boundary','CCTV_DGP_POST_NEARBY_LEARNING_REVIEW_V1.md')}.
The original DGP checkpoint remains the permanent baseline. It is a fine-tuned
DeblurGAN-v2-compatible restorer; the external frozen StyleGAN2 bank is not an
own-trained prior or qualified replacement. All five milestones remain active,
with native unpaired, synthetic paired and final evidence kept separate.
Actual training/gradients stay human-manual on the L4 in tmux; direct agent VM
access stays inventory-first hash-bound maintenance. Goal incomplete.
'''
    retained_new(ROOT/REPORT,report)
    prefix=f'''# Current status: detector confidence audited; restorer discussion pending — 14 October 2026

The separate CPU diagnostic completed 36 frozen detector forwards in 10.20s;
all 118 source / 114 output bindings and all six sheets / 180 cells are audited
and directly reviewed. All 36 proposals reproduce the previous app masks.
Empty proposals remain 4/14 original-photo covering views and 11/14 matched
synthetic-degraded views. One uncovered original is incorrectly marked at
11,683 pixels. These legacy `_native` files are photographs, not native CCTV.
{link('Current detector diagnosis and exact evidence',REPORT)}.

Missed regions already have very low scores before threshold/margin; dilation
cannot create a mask from an empty seed. Detection has demonstrated scoring/
localization limitations on these exposed cases. No unique architecture or
dataset cause is proved; no threshold fitting, app change or face generation.
The original pre-inference psutil failure is preserved separately from R1.
Detector/assisted completion and all seven families remain unqualified.

The latest restorer recipe remains USED / FAILED: best 0.687495% RAW / 0.706917%
PNG, with 32/32 preservation failures and zero accepted updates. After three
related failed recipes, the required diagnostic question is pending. Stop
restorer training-code changes; no fourth recipe or epoch extension is prepared.
{link('Required discussion and preserved gates','CCTV_DGP_POST_NEARBY_LEARNING_REVIEW_V1.md')}.

Keep the original DGP, checkpoints, splits, caches, provenance, full states and
failures. Manual L4 tmux training, maintenance-only direct VM access, all five
milestones, native unpaired/synthetic paired separation, untouched final
identities, visible appearance, clear glasses, ordinary hair and insufficient-
information requests remain binding. Original app design remains unchanged.
Goal incomplete. This dated status supersedes older UNRUN detector statements;
exact prior document follows.

'''
    bindings={n:finish_prepend(ROOT/n,prefix,stage/'before'/n,h) for n,h in expected.items()}
    files=[OUT/n for n in ['protocol.json','binding.json','preparation.json','execution.json','supervisor.json',
        'results.json','audit.json','summary.json','visual_review.json']]
    files += [old/n for n in ['protocol.json','binding.json','failure.json','failure_audit.json','supervisor.json']]
    files += [ROOT/'outputs'/n for n in ['dgp_detector_confidence_v1_r1_startup_review.json',
        'dgp_detector_confidence_v1_r1_checker.json','dgp_detector_process_memory_v1_check.json',
        'dgp_detector_confidence_summary_v1_check.json']]
    files += [ANALYSIS/n for n in ['diagnostic_question.json','local_protected_before.json','failure_audit.json',
                                 'publication/publication.json','publication/publication_audit.json']]
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),documents=bindings,
        new_documents_sha256={REPORT:sha(ROOT/REPORT)},
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files},
        local_detector_inference_images=36,publication_neural_or_training_calls=0,
        automatic_generation_outputs=0,assisted_generation_outputs=0,
        no_app_model_checkpoint_threshold_or_mask_change=True,original_failure_preserved=True,
        human_diagnostic_question_pending=True,next_training_recipe_not_selected_or_prepared=True,
        automatic_VM_launch=False,model_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=5,new_report=REPORT,empty_original=4,empty_degraded=11,
               generation_outputs=0,human_restorer_question_pending=True))


if __name__=='__main__':main()

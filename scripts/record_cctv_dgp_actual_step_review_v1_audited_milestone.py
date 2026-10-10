"""Record already performed visual review and preserve the preceding handoff."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis'
AUDIT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit'
OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_audited_milestone'
PRIOR = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_prepared_milestone'
TAIL = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_vm'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    assert not path.exists(), str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


PROFILE_NOTES = {
    'clear': 'Broad eyes, nose, mouth and outline remain recognizable. DGP output softens fine visible detail compared with the paired input; neither proposal adds convincing structure.',
    'blur_lr24': 'Central eyes, nose and mouth remain heavily soft. Recorded and cone proposals show no convincing added definition over the same before-state.',
    'lowlight_lr32': 'Lighting and central facial detail remain soft; neither proposal establishes clearer eye, nose or mouth structure over the before-state.',
    'motion_lr48': 'Broad facial structure is more discernible than in the stronger blur/compound cases, but neither proposal visibly improves it over the before-state.',
    'compound_lr24': 'Strong degradation leaves central facial structure very soft. Neither proposal establishes useful additional clarity; no hidden detail is treated as recovered truth.',
}

REFERENCE_NOTES = {
    'v9_tr_asian_00048': 'Visible moustache, facial hair and broad outline remain; eyelid, nostril and lip definition stay soft under degradation.',
    'v9_tr_ffhq_00084': 'Clear glasses remain visible in the clear control; degraded lenses, eyes and mouth remain soft. Clear glasses are appearance to preserve, not an automatic removal target.',
    'v9_tr_asian_02292': 'Eye makeup, mouth outline and surrounding hair remain broadly similar; the proposals do not add convincing central definition.',
    'v9_tr_ffhq_37935': 'Blue lighting/background and the visible microphone remain similar. Degraded eye and mouth detail remains soft; this is not a completion-removal test.',
    'v9_tr_asian_09839': 'Mildly turned pose and broad appearance remain similar; the proposals add no convincing central clarity.',
    'v9_tr_ffhq_03147': 'Warm red lighting and a hand near the image edge remain part of the input; no covered-region completion is assessed.',
    'v9_tr_ffhq_38018': 'Warm appearance and surrounding hairstyle remain broadly similar; degraded central structure stays soft.',
    'v9_tr_ffhq_55484': 'Visible head decoration and smiling mouth remain broadly similar; no convincing extra eye, nose or tooth definition is established.',
    'v9_tr_ffhq_33529': 'Graduation cap and smile remain broadly similar; teeth, eye and lip boundaries soften. A second printed face near the lower edge remains input context; this diagnostic does not validate the app single-face guard.',
    'v9_tr_asian_08558': 'Turned pose and grey side padding remain similar. Degraded eye, nostril and lip detail stays soft.',
    'v9_tr_ffhq_40530': 'Visible blond hair and broad toothy smile remain similar; degraded eyes and individual tooth boundaries stay soft.',
    'v9_tr_asian_03456': 'Visible moustache, beard and hair remain; degraded eyelids, nose and mouth remain soft.',
    'v9_tr_asian_07274': 'Clear glasses and smile remain visible in the clear control; strong degradations obscure central detail, with no convincing proposal benefit.',
}


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Existing milestone must not be overwritten'
    assert not (ANALYSIS / 'visual_review.json').exists()
    old = read(PRIOR / 'milestone.json')
    old_closure = read(PRIOR / 'independent_closure_audit_r1.json')
    assert old_closure['complete'] and old_closure['new_bindings_verified'] == 703
    assert sha(PRIOR / 'milestone.json') == old_closure['milestone_sha256']
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    assert sha(handoff) == old['new_evidence_sha256']['PROJECT_HANDOFF.md']
    before = handoff.read_bytes()
    for name, expected in old['new_evidence_sha256'].items():
        assert sha(ROOT / name) == expected, name
    science = read(AUDIT / 'independent_audit.json')
    plan = read(ANALYSIS / 'prospective_review.json')
    analysis = read(ANALYSIS / 'analysis.json')
    sheet_audit = read(ANALYSIS / 'independent_arithmetic_and_sheets_audit.json')
    tail = read(TAIL / 'protocol.json')
    tail_audit = read(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_preparation/independent_packet_audit.json')
    assert science['complete'] and science['conditions_count'] == 29
    assert science['CPU_replay']['cases'] == 145 and not science['full_3150_review_complete']
    assert len(science['derived_floored_ratio_recomputation_exception']) == 3
    assert science['categorical_decisions_exact'] and science['original_storage_failure_retained']
    assert sheet_audit['complete'] and sheet_audit['all1350_cells_exact']
    assert sheet_audit['analysis_sha256'] == sha(ANALYSIS / 'analysis.json')
    assert len(plan['visual_rows']) == 225 and len(analysis['sheets']) == 45
    assert tail_audit['complete'] and tail['prepared_only'] and tail['manual_VM_execution_required']
    assert not list((ROOT / 'outputs').glob('cctv-dgp-actual-step-tail-v1-results*'))
    for name, expected in old['app_bindings_sha256'].items():
        assert sha(ROOT / name) == expected, name
    rows = []
    sheets = []
    for index, sheet in enumerate(analysis['sheets']):
        assert sha(ROOT / sheet['path']) == sheet['sha256']
        sheets.append({'page': index + 1, 'path': sheet['path'], 'sha256': sheet['sha256'],
                       'actually_viewed': True, 'review_method': 'view_image with original detail, six unscaled 256x256 cells per row'})
        for selected in plan['visual_rows'][index * 5:index * 5 + 5]:
            ref = selected['id'][:-len('_' + selected['profile'])]
            rows.append({**selected, 'sheet': sheet['path'], 'actually_viewed': True,
                         'columns_viewed': plan['columns'], 'observation': PROFILE_NOTES[selected['profile']],
                         'visible_appearance_note': REFERENCE_NOTES[ref],
                         'recorded_step_added_useful_structure_demonstrated': False,
                         'cone_step_added_useful_structure_demonstrated': False,
                         'independent_final_review': False, 'model_qualification': False})
    assert len(rows) == 225
    visual = {
        'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'Implementing assistant /root; development diagnostic review, not an independent final reviewer',
        'review_basis': 'All45 exact sheets were actually inspected through view_image before this record was written; the earlier arithmetic checker checks pixels only.',
        'prospective_selection_sha256': sha(ANALYSIS / 'prospective_review.json'),
        'analysis_sha256': sha(ANALYSIS / 'analysis.json'),
        'independent_pixel_audit_sha256': sha(ANALYSIS / 'independent_arithmetic_and_sheets_audit.json'),
        'sheets': sheets, 'rows': rows, 'sheets_viewed': 45, 'rows_viewed': 225,
        'exact256_pixel_cells_viewed': 1350, 'completed_triad_updates': [10, 12, 22, 27, 37, 38, 39, 42, 44],
        'source_labels_are_dataset_names_not_ethnicity': True,
        'paired_photographic_TRAIN_diagnostic_only': True, 'raw_and_PNG_numeric_results_remain_separate': True,
        'original_analysis_unchanged': True, 'full_3150_review_complete': False,
        'update45_cone_not_visually_qualified': True, 'no_new_app_input_sufficiency_decision': True,
        'no_new_removal_margin_policy': True, 'no_covering_completion_quality_claim': True,
        'native_or_DEV_or_reserved_final_used': False, 'independent_final_review': False,
        'model_forwards': 0, 'optimizer_updates': 0, 'gradient_queries': 0, 'VM_calls': 0,
        'app_promotion': False, 'goal_complete': False,
    }
    write(ANALYSIS / 'visual_review.json', visual)
    report_path = ROOT / 'CCTV_DGP_ACTUAL_STEP_REVIEW_V1_RESULTS.md'
    assert not report_path.exists()
    table_parts = []
    for proposal, label in [('recorded', 'Recorded actual step: ten complete states'), ('cone', 'Frozen cone proposal: nine complete states')]:
        lines = [f'**{label}.** Relative landmark-feature improvement over the same before-state:',
                 '', '| TRAIN cohort | Raw mean change (%) | PNG mean change (%) | Preservation passes: raw / PNG |',
                 '| --- | ---: | ---: | --- |']
        for role, title in [('not_yet_optimized', 'Fixed not-yet-optimized'), ('optimized', 'Fixed previously optimized'), ('current_batch', 'Current five-profile batch')]:
            rr = next(r for r in analysis['summaries'] if (r['proposal'], r['role'], r['stage']) == (proposal, role, 'raw'))
            pp = next(r for r in analysis['summaries'] if (r['proposal'], r['role'], r['stage']) == (proposal, role, 'PNG'))
            lines.append(f"| {title} | {100 * rr['feature_gain_mean']:+.8f} | {100 * pp['feature_gain_mean']:+.8f} | {rr['finite_preservation_pass_updates']} / {pp['finite_preservation_pass_updates']} |")
        table_parts.append('\n'.join(lines))
    report = '''# Actual-step diagnostic: audited partial return and pending final-state review

The downloaded return is valid, but the finite diagnostic stopped before its
last condition. Neither tested proposal qualifies for another training recipe
or application promotion from the complete conditions reviewed so far.
V41's original 50/800 structure and preservation failure remains binding.

**Location:** the diagnostic worker's encoded-return guard.
**Cause:** retained output files reached the frozen 3 GiB budget minus its
16 MiB worker margin. This was a protocol storage stop, not a demonstrated
full-disk error. The run retained 29 complete conditions and an incomplete
update45/cone folder containing66 arrays and130 PNGs, with no metrics receipt.
**Fix:** retain the failed run; finish only that state in a separate small,
bounded inference packet with exact old-output overlap validation.

The original return archive has3,204,580,880 bytes, SHA256
`d380ce08cf858eebc334fcad3c6f5590e01d526084239bbf807bd4b465e8c604`.
Strict import verifies all9,369 regular files without executing returned code.
The separate R1 scientific audit verifies every3,045 complete raw/PNG/mean-only
slot and145 frozen CPU current-batch replays. Maximum raw replay error is
2.5033950805664062e-6; PNG replay differs by at most one byte, within the original
declared allowance. Persistent model states remain exact. No optimizer,
gradient, backward or new trained checkpoint occurs in this diagnostic.

The first partial checker failed on a recomputed mean-only quotient. The
failure, checker and exploratory reports remain. A separate R1 uses an explicit
float64 roundoff bound only when the quotient denominator is floored at1e-12
and both degraded-MSE comparisons already fail. The three exceptions have
group-MSE differences3.469446951953614e-18. Saved-row quotient arithmetic,
all image/replay tolerances and every categorical gate remain unchanged.
Six regressions verify acceptance of bounded arithmetic and rejection of an
excessive discrepancy, gate flip, corrupted saved quotient or wrong branch.
This corrects an audit calculation; it does not convert a failed model gate.

'''
    report += '\n\n'.join(table_parts) + '''

These means summarize repeated cases at diagnostic states selected after TRAIN
outcomes. They are not population estimates or independent evaluation. The
preservation comparison is against each same before-state; it is not the
full-corpus capacity test against the retained original DGP. Both fixed cohorts
fail preservation for every complete recorded/cone state in raw and PNG forms.
The current batch passes raw preservation only at27; no current PNG comparison
passes. The recorded step increases the raw landmark term in9 of10 batches.
Five of9 cone batches also increase it by small finite amounts, despite the
local saved-array proposal constraint. This demonstrates that the tested local
proposal is insufficient; it does not isolate one unique cause of the whole
restoration limitation or establish that quantization alone caused the failures.

All45 predetermined comparison sheets were actually viewed:225 rows and1,350
unscaled256-pixel cells. The paired input/target, original DGP, same before-state,
recorded step and cone proposal remain visually close across the degraded cases.
The clear controls broadly retain visible appearance, while DGP softens fine
detail. Strong blur, low light and compound degradation retain soft eyes, nose
and mouth; no convincing proposal gain in whole-face structure is established.
Clear glasses, ordinary hair and facial hair are visible appearance to preserve.
This assistant review is exposed paired photographic TRAIN evidence. It is not
native CCTV, a completion-family audit or independent final thesis review.
The original analysis and pixel-checker receipts remain unchanged; the separate
visual record documents actual inspection and does not promote a checkpoint.

The storage audit rechecks1,050 original-RGB arrays:145 unique cases and905
byte-identical duplicates. Reusing exact original arrays could save at least
570.36 MiB of compressed payload. This is a bound on a possible future format;
no deduplication or file deletion was performed. The new final-state packet
avoids repeating the whole3,150-slot run. It repeats210 complete control slots
only to validate exact overlap and reviews105 cone slots in a separate root.

The verified tail packet is43,992 bytes and remains unrun. It requires2 GiB free;
estimated VM runtime is3–7 minutes plus download. Its prospective checker must
audit all315 raw/PNG/mean-only slots, all828 old decoded arrays/pixels/control
metrics and15 frozen CPU replays. Storage512 MiB, cache120s, review300s,
worker600s/external630s, export300s/external330s and20 GiB allocated VRAM are
bounded. The original recipe, models, inputs, failures and gates stay unchanged.
Manual transfer/tmux/download instructions are ready; no automatic follow-on
or new training is launched.

The14 DGP-primary app files/checkpoint bindings remain exact. Original research
assets, splits, caches/local backup and failed gates remain retained. No new
native, development or reserved-final images are opened. Unpaired CCTV evidence
stays separate from paired metrics; dataset labels do not establish ethnicity
or Zamboanga performance. Useful native restoration, automatic/assisted quality
for all seven covering families, independent final review and qualified local
application flow still remain required. Goal active/incomplete.

[Scientific R1 audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_actual_step_review_v1_partial_audit/independent_audit.json>)
[Actual visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_actual_step_review_v1_partial_analysis/visual_review.json>)
[Exact pixel/arithmetic audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_actual_step_review_v1_partial_analysis/independent_arithmetic_and_sheets_audit.json>)
[Storage evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_actual_step_review_v1_storage_analysis/lossless_storage_evidence.json>)
[Five manual final-state steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md>)
'''
    report_path.write_text(report, encoding='utf-8')
    section = '''**Research milestone — 9 October 2026: actual-step partial return audited and visually reviewed; final state pending.**

The human returned the finite inference diagnostic. Its3,204,580,880-byte archive
matches SHA256 d380ce08cf858eebc334fcad3c6f5590e01d526084239bbf807bd4b465e8c604.
Strict import verifies all9,369 files. The original run stopped at its frozen
3GiB encoded-output limit, preserving29 complete conditions; update45/cone
has66 arrays/130 PNGs but no complete metrics receipt. This is a protocol storage
stop, not a demonstrated full-disk error. No optimizer, gradient, backward or
new trained checkpoint occurs. Preserve the stop; the full3,150-slot diagnostic
is not complete and the original full checker remains import-only.

Separate R1 scientific verification checks every3,045 complete raw/PNG/mean-only
slot and145 frozen CPU replays. All categorical decisions and original
pixel/objective/replay tolerances remain unchanged. The failed first partial
checker is retained. R1 allows only a bounded recomputation of a mean-only
quotient whose denominator is floored at1e-12 and both MSE gates already fail.
Three exceptions have3.469446951953614e-18 group-MSE roundoff; all six corruption/
branch/decision regressions pass. This audit recovery does not repair model gates.

Neither recorded nor cone proposals pass preservation on either fixed TRAIN
cohort at any complete state, in raw or PNG form. Current-batch raw preservation
passes only at27; none of its PNG comparisons pass. The recorded landmark term
increases in9/10 batches; cone finite changes can also increase it slightly.
This rejects the tested local proposal as sufficient evidence for a new recipe;
no unique whole-model cause or PNG-only attribution is established.
All45 fixed sheets/225 rows/1,350 exact256-pixel cells were actually viewed by
the implementing assistant. Proposals remain visibly close to the earlier DGP,
with soft degraded central features and no convincing added facial structure.
This is exposed paired photographic TRAIN diagnostic evidence, not an
independent final, native CCTV or completion-family quality review.

A lossless-storage audit verifies905 repeated original-RGB copies, with at least
570.36MiB potential compressed-payload savings. Nothing is deleted or deduplicated.
The smaller43,992-byte final-state packet remains prepared/unrun. It evaluates
update45 only, repeats210 controls for exact overlap validation and adds105 cone
slots. Its checker must verify315 outputs,828 exact decoded overlap files and
15 frozen CPU replays. Require2GiB free; estimate3–7minutes plus download.
Protocol SHA256: ec6981ce0e6961b765fb5dc1e82f191bc3511db2a1ca6b87e2849e83ea40eef9
Execution archive SHA256: 0436e9477b1ce277856d9c54a1655d2915091c204637ab78e1c1fffe85d68950
All4 transfer members,3 packet assets and260 original parent assets pass;
scope/storage/timing-only source changes have exact inverse proof. Python3.10,
pre-Torch Windows guards and read-only Bash syntax pass. Cache120s, review300s,
worker600s/external630s, export300s/external330s,20GiB VRAM and512MiB returns
are enforced. Manual upload/install/tmux/download steps are prepared. No new
actual training recipe or automatic launch occurs; every historical failure stays.

All14 DGP-primary app bindings remain exact. Original checkpoints, sources,
splits, research caches/local backup and provenance remain retained. No new
native/DEV/reserved-final images are opened and no ethnicity or Zamboanga claim
is made. Clear glasses, ordinary hair and every visible facial feature remain
protected. Useful native restoration, all seven automatic/assisted covering
families, independent final review and qualified full local app flow are still
required. Goal active/incomplete. Prior functional Playwright evidence is
retained; no app change justifies a new browser run for this evidence milestone.
The complete previous handoff is archived and preserved byte-for-byte below.

[Diagnostic findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_REVIEW_V1_RESULTS.md>)
[Five manual final-state steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md>)

'''
    OUT.mkdir()
    backup = OUT / 'before_docs/PROJECT_HANDOFF.md'
    backup.parent.mkdir()
    backup.write_bytes(before)
    (OUT / 'new_handoff_section.md').write_text(section, encoding='utf-8')
    handoff.write_bytes(section.encode('utf-8') + before)
    files = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_actual_step_review_v1_audited_milestone.py',
             report_path, handoff, backup, OUT / 'new_handoff_section.md',
             PRIOR / 'milestone.json', PRIOR / 'independent_closure_audit_r1.json',
             ROOT / 'CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md',
             ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json',
             ROOT / 'outputs/cctv_dgp_actual_step_review_v1_independent_audit.json',
             ROOT / 'outputs/cctv_dgp_actual_step_review_v1_vm_return/outputs/failure.json',
             ROOT / 'outputs/cctv_dgp_actual_step_review_v1_vm_return/supervisor_receipt.json',
             ROOT / 'outputs/cctv_dgp_actual_step_review_v1_vm_return/export_manifest.json',
             ROOT / 'outputs/cctv-dgp-actual-step-review-v1-results.tar.gz.sha256',
             ROOT / 'outputs/cctv-dgp-actual-step-review-v1-export.json',
             ROOT / 'outputs/cctv-dgp-actual-step-tail-v1-execution.tar.gz',
             ROOT / 'outputs/cctv-dgp-actual-step-tail-v1-execution.tar.gz.sha256',
             ROOT / 'scripts/prepare_cctv_dgp_actual_step_tail_v1.py',
             ROOT / 'scripts/verify_cctv_dgp_actual_step_tail_v1_packet.py',
             ROOT / 'scripts/cctv_dgp_actual_step_tail_v1_vm.py',
             ROOT / 'scripts/audit_cctv_dgp_actual_step_tail_v1_return.py',
             ROOT / 'scripts/audit_cctv_dgp_actual_step_review_v1_storage.py']
    for folder in [ANALYSIS, AUDIT, TAIL, ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_preparation',
                   ROOT / 'outputs/cctv_dgp_actual_step_review_v1_storage_analysis']:
        files.extend(p for p in folder.rglob('*') if p.is_file())
    for pattern in ['*actual_step_review_v1_partial*py', '*actual_step_review_v1_partial*analysis*py']:
        files.extend((ROOT / 'scripts').glob(pattern))
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}
    write(OUT / 'milestone.json', {
        'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'previous_milestone_sha256': sha(PRIOR / 'milestone.json'),
        'previous_R1_closure_sha256': sha(PRIOR / 'independent_closure_audit_r1.json'),
        'previous_binding_count': 703, 'previous_handoff_path': backup.relative_to(ROOT).as_posix(),
        'previous_handoff_sha256': sha(backup), 'new_evidence_sha256': bindings,
        'app_bindings_sha256': old['app_bindings_sha256'],
        'returned_archive_sha256': science['archive_sha256'], 'returned_archive_bytes': 3204580880,
        'complete_conditions': 29, 'complete_forward_slots': 3045, 'frozen_CPU_replays': 145,
        'selected_visual_rows_reviewed': 225, 'sheets_reviewed': 45, 'exact256_cells_reviewed': 1350,
        'original_partial_checker_failure_retained': True, 'original_VM_storage_stop_retained': True,
        'full_3150_review_complete': False, 'original_V41_training_gates_still_failed': True,
        'tail_protocol_sha256': sha(TAIL / 'protocol.json'),
        'tail_execution_archive_sha256': tail_audit['execution_archive_sha256'],
        'tail_prepared_only': True, 'manual_tail_execution_required': True,
        'new_training_recipe_prepared': False, 'new_trained_checkpoint': False,
        'historical_training_code_modified': False, 'optimizer_updates': 0,
        'gradient_queries': 0, 'backward_calls': 0, 'additional_model_forwards_in_this_recorder': 0,
        'VM_calls_in_this_recorder': 0, 'file_deletions': 0, 'app_promotion': False,
        'native_or_DEV_or_reserved_final_used': False, 'independent_final_review': False,
        'goal_complete': False, 'seconds': time.monotonic() - start, 'cap_seconds': 300,
    })
    print({'complete': True, 'visual_rows_recorded': 225, 'new_bindings': len(bindings),
           'tail_prepared_only': True, 'goal_complete': False}, flush=True)


if __name__ == '__main__':
    main()

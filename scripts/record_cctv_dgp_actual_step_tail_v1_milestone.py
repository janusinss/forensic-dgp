"""Record performed visual review, completed diagnostic and exact handoff history."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_analysis'
PARTIAL = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis'
PRIOR = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261009_v1'
OUT = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_audited_milestone'

PROFILE_NOTES = {
    'clear': 'Broad visible facial appearance remains. DGP softens finer eyelid, nose and mouth detail; neither proposal establishes convincing added structure.',
    'blur_lr24': 'Central eye, nose and mouth boundaries remain very soft. The two proposals show no convincing added definition over the same before-state.',
    'lowlight_lr32': 'Visible appearance remains broadly similar while central lighting/detail stays soft. Neither proposal demonstrates clearer facial structure.',
    'motion_lr48': 'Broad structure is discernible, but fine eye, nose and lip boundaries remain soft. No convincing proposal benefit is visible.',
    'compound_lr24': 'Central eye, nose and mouth detail remains very limited. Neither proposal establishes useful additional clarity or verified hidden detail.',
}
REFERENCE_NOTES = {
    'v9_tr_asian_00048': 'Moustache, facial hair and broad outline remain visible; degraded eyelid, nostril and lip definition stay soft.',
    'v9_tr_ffhq_00084': 'Clear glasses remain in the clear control; degraded lens, eye, nose and mouth detail stays soft. Clear glasses and ordinary hair remain visible appearance to preserve.',
    'v9_tr_asian_02292': 'Eye makeup, surrounding hair and mouth outline remain broadly similar. Degraded central structure stays soft.',
    'v9_tr_ffhq_37935': 'Blue lighting, hair and visible microphone remain context; degraded eyes and mouth stay soft. This is not a covered-object completion test.',
    'v9_tr_ffhq_52071': 'Broad jaw, short dark hair, red jacket and outdoor context remain. Fine brow, eyelid, nose and lip structure stays soft, particularly in blur and compound cases.',
}


def main():
    start = time.monotonic()

    def sha(path):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert time.monotonic() - start < 600
        return digest

    read = lambda path: json.loads(path.read_text(encoding='utf-8'))

    def write(path, value):
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    assert not OUT.exists() and not (ANALYSIS / 'visual_review.json').exists()
    old = read(PRIOR / 'milestone.json')
    old_closure = read(PRIOR / 'independent_closure_audit.json')
    assert old_closure['complete'] and sha(PRIOR / 'milestone.json') == old_closure['milestone_sha256']
    for name, expected in old['new_evidence_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in old['app_bindings_sha256'].items():
        assert sha(ROOT / name) == expected, name
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    before = handoff.read_bytes()
    assert sha(handoff) == old['new_evidence_sha256']['PROJECT_HANDOFF.md']
    science_path = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_independent_audit.json'
    science = read(science_path)
    plan = read(ANALYSIS / 'prospective_review.json')
    analysis = read(ANALYSIS / 'analysis.json')
    pixel_audit = read(ANALYSIS / 'independent_arithmetic_and_sheets_audit.json')
    original_plan = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_analysis/prospective_review.json')
    old_visual = read(PARTIAL / 'visual_review.json')
    assert science['complete'] and science['all315_outputs_checked']
    assert science['overlap_files_independently_checked'] == 828
    assert science['CPU_replay']['cases'] == 15 and science['original_storage_failure_retained']
    assert pixel_audit['complete'] and pixel_audit['all150_new_cells_exact']
    assert pixel_audit['analysis_sha256'] == sha(ANALYSIS / 'analysis.json')
    assert analysis['unique_completed_diagnostic_slots'] == 3150 and len(analysis['logical_conditions']) == 30
    assert plan['visual_rows'] == original_plan['visual_rows'][-25:]
    assert old_visual['complete'] and old_visual['rows_viewed'] == 225 and old_visual['sheets_viewed'] == 45
    assert [dict((k, row[k]) for k in ['update', 'role', 'id', 'source', 'profile']) for row in old_visual['rows']] == original_plan['visual_rows'][:225]
    assert not science['app_promotion'] and not science['new_trained_checkpoint']
    assert len(analysis['sheets']) == 5 and len(plan['visual_rows']) == 25
    OUT.mkdir()
    sheets, rows = [], []
    for index, sheet in enumerate(analysis['sheets']):
        assert sha(ROOT / sheet['path']) == sheet['sha256']
        sheets.append({'page': index + 46, 'path': sheet['path'], 'sha256': sheet['sha256'],
            'actually_viewed': True, 'review_method': 'view_image original detail; six unscaled256-pixel cells per row'})
        for selected in plan['visual_rows'][index * 5:index * 5 + 5]:
            ref = selected['id'][:-len('_' + selected['profile'])]
            rows.append({**selected, 'sheet': sheet['path'], 'actually_viewed': True,
                'columns_viewed': plan['columns'], 'observation': PROFILE_NOTES[selected['profile']],
                'visible_appearance_note': REFERENCE_NOTES[ref],
                'recorded_step_added_useful_structure_demonstrated': False,
                'cone_step_added_useful_structure_demonstrated': False,
                'independent_final_review': False, 'model_qualification': False})
    visual = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'Implementing assistant /root; exposed TRAIN diagnostic, not independent final review',
        'review_basis': 'All five final sheets were actually inspected through view_image before this record was written',
        'prospective_selection_sha256': sha(ANALYSIS / 'prospective_review.json'),
        'analysis_sha256': sha(ANALYSIS / 'analysis.json'),
        'independent_pixel_audit_sha256': sha(ANALYSIS / 'independent_arithmetic_and_sheets_audit.json'),
        'sheets': sheets, 'rows': rows, 'sheets_viewed': 5, 'rows_viewed': 25, 'exact256_pixel_cells_viewed': 150,
        'previous_visual_review_sha256': sha(PARTIAL / 'visual_review.json'),
        'previous_sheets_viewed': 45, 'previous_rows_viewed': 225, 'previous_cells_viewed': 1350,
        'combined_sheets_viewed_across_both_reviews': 50, 'combined_rows_viewed': 250, 'combined_cells_viewed': 1500,
        'all_frozen_visual_rows_reviewed_across_both_turns': True,
        'source_labels_are_dataset_names_not_ethnicity': True,
        'paired_photographic_TRAIN_diagnostic_only': True, 'original_analysis_and_prior_review_unchanged': True,
        'diagnostic_coverage_complete': True, 'model_qualified': False,
        'no_new_app_input_sufficiency_decision': True, 'no_new_removal_margin_policy': True,
        'no_covering_completion_quality_claim': True, 'native_or_DEV_or_reserved_final_used': False,
        'independent_final_review': False, 'model_forwards_in_visual_record': 0,
        'optimizer_updates': 0, 'gradient_queries': 0, 'VM_calls': 0,
        'app_promotion': False, 'goal_complete': False}
    write(ANALYSIS / 'visual_review.json', visual)
    write(OUT / 'primary_sources.json', {'complete': True,
        'sources_checked': [
            {'title': 'Gradient Episodic Memory for Continual Learning', 'authors': 'Lopez-Paz and Ranzato',
             'url': 'https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf',
             'use': 'Section3 local linearity and representative-memory assumptions; no CCTV validation'},
            {'title': 'Simple Baselines for Image Restoration', 'authors': 'Chen et al.',
             'url': 'https://arxiv.org/abs/2204.04676',
             'use': 'Gated architecture motivation; our spatial block differs; no pretrained weight use or CCTV performance inference'},
            {'title': 'The Perception-Distortion Tradeoff', 'authors': 'Blau and Michaeli',
             'url': 'https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html',
             'use': 'Separate reference distortion and visual quality; no identity/gate relaxation or optimality claim'}],
        'research_does_not_qualify_our_model': True})
    section = '''**Research milestone — 9 October 2026: final-state return audited; actual-step diagnostic complete, proposals unqualified.**

The human manually ran the final-state tail. Its review/export stages both
completed. The324,974,655-byte archive matches SHA256
4a0037a5029aeb854ca7264e5179b0a06f110c6f6ebfc9c2ba3aa96561516616.
Strict local import checks959 members without executing returned code.
The prospective scientific auditor checks315 raw/PNG/mean-only slots,
828 exact decoded overlap files and15 frozen CPU replays. Persistent states,
original stored-row arithmetic and all categorical decisions remain exact.
One predeclared floored mean-only quotient roundoff allowance is bounded;
both underlying degraded-MSE gates still fail. No model-quality gate changes.

The original storage-stopped run remains failed and preserved. Together the
separate audits cover30 conditions and3,150 unique planned slots:
3,045 +315 -210 repeated controls. CPU replays total160 executions/150 unique
slots. The original full checker is not relabelled as passing. All50 frozen
sheets/250 rows/1,500 exact256-pixel cells have been actually viewed across
the prior and final assistant reviews. The five new sheets add25 rows/150 cells.

Neither recorded nor cone proposals pass preservation on either fixed TRAIN
cohort at any state in raw or PNG form. Current-batch raw passes only at27;
every current PNG fails. The recorded raw landmark term increases in9/10
batches; cone increases in5/10. Degraded eyes, nose and mouth remain soft;
no convincing added whole-face clarity is established. The finite diagnostic
is complete, but it supplies no trained candidate or app qualification.

A separate all145-case saved-array target oracle is completed and independently
audited with zero neural/gradient/optimizer calls. An ideal bounded, mean-centered
correction has96.4534% raw/96.3637% PNG degraded landmark-error reduction.
These are oracle arithmetic, not model performance: clean target access is
unavailable in the app; ArcFace/complete preservation/network capacity are
untested. Mean/amplitude geometry alone does not explain the tiny learned gains.

The selected next design review examines reconstruction-first supervision of
our own spatial DGP decoder before another executable pilot. All earlier
restoration-only/group/PNG-guard failures remain binding. No unchanged retry,
new training protocol, transfer packet, VM launch or app promotion occurs.
Actual training remains manual on the existing L4 under a distinct verified
finite packet with unchanged preservation/structure requirements.

All14 DGP-primary local app bindings and original checkpoint/source/split/failure
assets remain exact. No new native/DEV/reserved-final data are opened. This is
paired photographic TRAIN diagnostic evidence, not native CCTV, covering-family
qualification or independent final review. No ethnicity, Zamboanga performance
or recovered hidden-identity claim is made. Useful native restoration, all
seven automatic/assisted completion families, independent final review and
qualified full app flow remain outstanding. Goal active/incomplete.
The complete preceding handoff, including the cleanup milestone, is archived
and preserved byte-for-byte below. The cleanup free-space reading is historical;
this audit does not report a new live VM free-space measurement.

[Completed diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_TAIL_V1_RESULTS.md>)
[Next design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_ACTUAL_STEP_ARCHITECTURE_REVIEW.md>)

'''
    intro = section.encode('utf-8')
    with (OUT / 'PROJECT_HANDOFF_before_tail.md').open('xb') as stream:
        stream.write(before)
    handoff.write_bytes(intro + before)
    evidence = {}

    def bind(path, expected=None):
        value = sha(path)
        if expected is not None:
            assert value == expected, str(path)
        evidence[path.relative_to(ROOT).as_posix()] = value

    for name, expected in old['new_evidence_sha256'].items():
        if name != 'PROJECT_HANDOFF.md':
            bind(ROOT / name, expected)
    for path in [handoff, Path(__file__), ROOT / 'scripts/verify_cctv_dgp_actual_step_tail_v1_milestone.py',
        ROOT / 'scripts/analyze_cctv_dgp_actual_step_tail_v1.py', ROOT / 'scripts/verify_cctv_dgp_actual_step_tail_v1_analysis.py',
        ROOT / 'scripts/analyze_cctv_dgp_output_geometry_v1.py', ROOT / 'scripts/verify_cctv_dgp_output_geometry_v1.py',
        ROOT / 'CCTV_DGP_ACTUAL_STEP_TAIL_V1_RESULTS.md', ROOT / 'CCTV_DGP_POST_ACTUAL_STEP_ARCHITECTURE_REVIEW.md',
        science_path, ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_return_import.json',
        ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json',
        PRIOR / 'milestone.json', PRIOR / 'independent_closure_audit.json',
        ROOT / 'scratch/cctv_dgp_actual_step_tail_v1_audit_stdout.log',
        ROOT / 'scratch/cctv_dgp_actual_step_tail_v1_audit_stderr.log']:
        bind(path)
    for folder in [ANALYSIS, PARTIAL, ROOT / 'outputs/cctv_dgp_output_geometry_v1', OUT]:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                bind(path)
    for prefix in ['cctv_dgp_actual_step_tail_v1_vm', 'cctv_dgp_actual_step_review_v1_vm']:
        bundle = ROOT / 'outputs' / prefix
        p = read(bundle / 'protocol.json')
        bind(bundle / 'protocol.json')
        for name, expected in p['assets_sha256'].items():
            bind(bundle / name, expected)
    for stem in ['cctv-dgp-actual-step-tail-v1', 'cctv-dgp-actual-step-review-v1']:
        for path in sorted((ROOT / 'outputs').glob(stem + '-results*')):
            if path.is_file():
                bind(path)
        for path in sorted((ROOT / 'outputs').glob(stem + '-export.json')):
            bind(path)
    tail_import = read(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_return_import.json')
    for name, expected in tail_import['files_sha256'].items():
        bind(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_vm_return' / name, expected)
    milestone = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'previous_milestone_sha256': sha(PRIOR / 'milestone.json'),
        'previous_closure_sha256': sha(PRIOR / 'independent_closure_audit.json'),
        'previous_handoff_sha256': hashlib.sha256(before).hexdigest(),
        'previous_handoff_exact_bytes_preserved': True, 'new_intro_bytes': len(intro),
        'new_evidence_sha256': evidence, 'app_bindings_sha256': old['app_bindings_sha256'],
        'tail_archive_sha256': science['archive_sha256'], 'tail_return_members': 959,
        'all_planned_conditions_covered': 30, 'unique_output_slots_audited': 3150,
        'tail_output_slots_checked': 315, 'overlap_files_checked': 828,
        'CPU_replay_executions': 160, 'unique_CPU_replay_slots': 150,
        'combined_visual_sheets': 50, 'combined_visual_rows': 250, 'combined_visual_cells': 1500,
        'original_storage_failure_retained': True, 'original_full_checker_not_relabelled_as_pass': True,
        'oracle_case_count': 145, 'oracle_is_not_model_performance': True,
        'diagnostic_coverage_complete': True, 'new_training_protocol_prepared': False,
        'selected_next_design': 'Reconstruction-first supervision review of own spatial DGP path; unchanged acceptance gates',
        'training_or_diagnostic_launched_on_VM_by_agent': False, 'optimizer_updates': 0,
        'gradient_queries': 0, 'new_trained_checkpoint': False, 'native_or_DEV_or_reserved_final_used': False,
        'app_promotion': False, 'independent_final_review': False, 'model_qualification': False,
        'goal_complete': False, 'seconds': time.monotonic() - start, 'cap_seconds': 600}
    write(OUT / 'milestone.json', milestone)
    print({'complete': True, 'evidence_bindings': len(evidence), 'diagnostic_slots': 3150,
        'sheets_actually_viewed_across_both_reviews': 50, 'app_bindings': 14,
        'new_training_protocol': False, 'goal_complete': False, 'seconds': milestone['seconds']}, flush=True)


if __name__ == '__main__':
    main()

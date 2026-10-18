"""Bind independent factual review of the final AdaIN return report, CPU only."""
from collections import Counter
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1.return_archive import read, sha, require, write_new, PROTOCOL_SHA


def main():
    review = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review'
    report = ROOT / 'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_RETURN_REVIEW.md'
    text = report.read_text(encoding='utf-8'); report_sha = sha(report)
    names = ['transport_v1.json', 'state_v1/state_audit.json', 'metrics_v1/metrics_audit.json',
        'metrics_v1/guards.json', 'metrics_v1/metrics.json', 'scientific_readback_v1.json',
        'SCIENTIFIC_READBACK_V1.md', 'clear_error_decomposition_v1.json',
        'independent_scientific_readback_v1.json', 'independent_summary_render_readback_v1.json',
        'independent_next_design_readback_v1.json', 'next_design_review_v1.json', 'next_design_review_v1.md',
        'visual_review_v1.json', 'independent_root_visual_record_v1.json',
        'input_usability_requirement_v1.md', 'input_usability_requirement_v1.json', 'all27_CPU_tests_v1.json']
    evidence = {str((review / name).relative_to(ROOT)).replace('\\', '/'): sha(review / name) for name in names}
    evidence[str(Path(__file__).relative_to(ROOT)).replace('\\', '/')] = sha(__file__)
    transport = read(review / 'transport_v1.json'); state = read(review / 'state_v1/state_audit.json')
    summary = read(review / 'scientific_readback_v1.json'); dec = read(review / 'clear_error_decomposition_v1.json')
    usage = read(review / 'input_usability_requirement_v1.json'); visual = read(review / 'visual_review_v1.json')
    record = read(review / 'independent_root_visual_record_v1.json'); tests = read(review / 'all27_CPU_tests_v1.json')
    require(transport['protocol_sha256'] == state['protocol_sha256'] == PROTOCOL_SHA and
        state['evidence_consistency_pass'] and not state['training_finished'] and not state['resume_authorized'],
        'Report cannot promote quality-stopped training')
    require(str(transport['archive_bytes']) not in text or '2,864,673,701' in text, 'Archive byte formatting differs')
    for value in ('2,864,673,701', '6,558', '3,143,169,457', '1,939', '31 local overlay',
                  '50 committed Adam', '200 TRAIN', '1,614', '2,073', '350 trainable', '165 frozen', '391 HQ', '50 behavior'):
        require(value in text, 'Missing factual run/census claim: ' + value)
    require(transport['archive_sha256'] in text and PROTOCOL_SHA in text and
        state['stop_reason'] in text and '43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b' in text,
        'Pinned archive/protocol/failure/stopped-checkpoint identity absent')
    require(state['sections']['trajectory']['unique_TRAIN_references'] == 100, 'Photographic training count differs')
    for row in summary['guard_summary']:
        if row['state'] != 'isolated_update50': continue
        for value in (row['mean_delta'], row['worst_delta']): require(format(value, '.6f') in text, 'Report guard numeric differs')
        require('%d/%d' % (len(row['case_failures']), row['cases']) in text, 'Report failure denominator differs')
    require('0.01' in text and '0.03' in text, 'Preservation thresholds omitted')
    for rid in summary['clear_failed_case_ids']['isolated_update50']: require(rid in text, 'Report misses failed clear case')
    hq = [row for row in summary['full_source_profile_deltas'] if row['source'] == 'FFHQ validation' and
          row['view'] == 'native_png' and row['comparison'] == 'isolated_update50_minus_original_update0']
    for profile in ('compound', 'lowlight'):
        row = next(row for row in hq if row['profile'] == profile)
        require(format(abs(row['mean_delta_mae']), '.6f') in text and format(row['mean_delta_ssim'], '.6f') in text,
            'Paired HQ improvement size differs')
    require(summary['all_six_HQ_profiles_PNG_mean_MAE_slightly_worse_than_prior_FP32'] and
        summary['all_six_HQ_profiles_PNG_mean_SSIM_slightly_lower_than_prior_FP32'] and
        'All six HQ profile mean PNG MAEs are slightly worse' in text, 'Previous FP32 comparison claim differs')
    encoder = state['sections']['prior_FP32_weight_comparison']['partitions']['encoder']
    for value, digits in ((encoder['new_vs_prior_delta_direction_cosine'], 5), (encoder['new_change_l2'], 6),
                           (encoder['prior_change_l2'], 6)):
        require(format(value, '.' + str(digits) + 'f') in text, 'Encoder state comparison numeric differs')
    runtime = read(ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return/outputs/adain_gradient_isolation_v1/runtime.json')
    flags = runtime['algorithm_settings']
    require(flags['deterministic_algorithms_enabled'] is flags['cudnn_deterministic'] is False and
        'global deterministic execution is not claimed' in text, 'Report algorithm precision claim differs')
    require(record['complete'] and record['root_visual_review_sha256'] == sha(review / 'visual_review_v1.json') and
        record['all55_sheet_hashes_paths_case_ids_checked'] and not record['personal_visual_verdict_by_this_verifier'],
        'Independent record verification cannot invent a personal image review')
    require(visual['root_personally_viewed_all_planned_sheets'] and len(visual['sheets']) == 55 and
        not visual['visual_acceptance'] and '1,988 source cells' in text and '1,222 output cells' in text and
        '15 critical figure cells' in text, 'Report root display/review scope differs')
    require(sha(review / 'input_usability_requirement_v1.md') ==
        '5ad7aecd82e0d71eb560c817b3039dd98327defc277a3f494167c6f4c6245337' and
        sha(review / 'input_usability_requirement_v1.json') ==
        'e20c772c7ee86902939dc23ad8982e51a68d8165ec4ee76bef41e5409fc0ee5d',
        'Clarified visible-feature requirement revision differs')
    require(usage['complete'] and not usage['app_changed'] and not usage['automatic_structure_gate_qualified'] and
        usage['visible_feature_interpretation']['all_features_need_not_be_visible'] and
        usage['visible_feature_interpretation']['covered_mouth_or_nose_alone_not_automatic_rejection'],
        'Coverage must evaluate actually visible relationships, not demand every feature')
    require(len(usage['source_and_evidence_sha256']) == 22, 'Usability source census differs')
    for name, digest in usage['source_and_evidence_sha256'].items():
        require(sha(ROOT / name) == digest, 'Current usability source binding changed'); evidence[name] = digest
    old = read(ROOT / 'outputs/cctv_native_development_v2/input_review.json')
    require(old['reviewed_before_model_outputs'] and not old['reserved_inputs_viewed'], 'Native input labels lack prior input-only review')
    labels = {row['id']: row for row in old['rows']}
    selected = usage['page48_metadata_reference']['cases']
    counts = Counter()
    for row in selected:
        source = labels[row['id'].removeprefix('comparison_')]
        require(row['preexisting_input_structure_review'] == source['input_structure_review'] and
            row['preexisting_pose_review'] == source['pose_review'] and
            row['preexisting_input_only_note'] == source['input_only_note'], 'Page48 prior input-only label differs')
        counts[source['input_structure_review']] += 1
    require(dict(counts) == {'insufficient': 5, 'partial': 1, 'weak': 2}, 'Page48 label count differs')
    require('does not yet implement a calibrated automatic structural threshold' in text and
        'Coverage alone does not reject a face with enough useful visible context' in text and
        'No numerical cutoff, app change or gate qualification is claimed' in text,
        'Report overclaims current usability implementation')
    require(tests['complete'] and tests['CPU_tests'] == 27 and tests['returncode'] == 0,
        'All27 current bounded CPU fixtures required')
    for name, digest in tests['source_sha256'].items(): require(sha(ROOT / name) == digest, 'Current test/audit source changed')
    for target in re.findall(r'\]\(([^)]+)\)', text):
        require(not target.startswith(('http:', 'https:')) and (ROOT / target).exists(), 'Missing report evidence link')
    expected_limits = ('not recovered-identity, ethnicity or Zamboanga claims',
        'Reserved FINAL pixels were not read', 'does not prove CE/HQ alone caused the failure',
        'not a decomposition of the MAE acceptance gate', 'No next packet, recipe, new training',
        'no returned model was instantiated', 'Separate detector/completion development remains deferred')
    for value in expected_limits: require(value in text, 'Missing scientific limitation: ' + value)
    require(sha(report) == report_sha, 'Report changed during factual review')
    result = {'complete': True, 'approved_for_publication': True, 'report_sha256': report_sha,
        'visual_review_sha256': sha(review / 'visual_review_v1.json'),
        'evidence_sha256': evidence, 'review_scope': 'Full report personally read for factual/source consistency; no independent personal visual judgment.',
        'all27_CPU_tests_current_source_bound_pass': True, 'input_usability_sources_freshly_hashed': 22,
        'page48_preexisting_label_join_independently_checked': dict(counts),
        'visible_features_need_not_all_be_uncovered': True,
        'automatic_usability_gate_recorded_not_implemented_or_calibrated': True,
        'historical406_and61clear53blur_denominators_unchanged': True,
        'no_unqualified_illumination_causality_or_identity_claim': True,
        'root_personal_visual_scope55sheets_pluscritical_figure_preserved': True,
        'no_neural_gradient_optimizer_VM_or_app_execution': True,
        'model_qualified': False, 'training_authorized': False, 'overall_goal_complete': False}
    write_new(review / 'independent_report_readback_v1.json', result)
    print({'complete': True, 'approved_for_publication': True, 'report_sha256': report_sha,
           'receipt_sha256': sha(review / 'independent_report_readback_v1.json')})


if __name__ == '__main__': main()

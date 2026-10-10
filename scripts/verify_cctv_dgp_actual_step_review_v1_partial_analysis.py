"""Read-only arithmetic, original-source and exact visual-cell verification."""
from pathlib import Path
import hashlib
import time
import numpy as np
from PIL import Image
from cctv_dgp_actual_step_review_v1_contract import NAME, PROPOSALS, ROLES, read, write, sha, output_prefix

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs' / NAME
RETURN = ROOT / 'outputs' / (NAME + '_return')
OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis'


def pixels(path):
    with Image.open(path) as image:
        assert image.mode == 'RGB' and image.size == (256, 256)
        return np.asarray(image).copy()


def main():
    start = time.monotonic()
    assert not (OUT / 'independent_arithmetic_and_sheets_audit.json').exists()
    plan = read(OUT / 'prospective_review.json')
    report = read(OUT / 'analysis.json')
    p = read(BUNDLE / 'protocol.json')
    imported = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json')
    audit_path = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit/independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and not audit['full_3150_review_complete']
    assert audit['conditions_count'] == 29 and audit['CPU_replay']['cases'] == 145
    assert report['complete'] and not report['full_3150_review_complete']
    assert report['independent_audit_sha256'] == sha(audit_path)
    assert report['prospective_selection_sha256'] == sha(OUT / 'prospective_review.json')
    assert not report['visual_review_complete'] and not report['app_promotion']
    assert report['protocol_sha256'] == plan['protocol_sha256'] == sha(BUNDLE / 'protocol.json')
    assert report['model_forwards'] == report['gradient_queries'] == report['optimizer_updates'] == 0
    for bindings in [plan['source_bindings'], report['source_evidence_sha256']]:
        for name, digest in bindings.items():
            assert sha(ROOT / name) == digest, name
    partial_script = ROOT / 'scripts/analyze_cctv_dgp_actual_step_review_v1_partial_return.py'
    original_script = ROOT / 'scripts/analyze_cctv_dgp_actual_step_review_v1_return.py'
    inverse = partial_script.read_text(encoding='utf-8')
    for change in reversed(plan['source_changes']):
        assert inverse.count(change['after']) == 1
        inverse = inverse.replace(change['after'], change['before'], 1)
    assert inverse == original_script.read_text(encoding='utf-8')
    prior = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_analysis/prospective_review.json')
    assert sha(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_analysis/prospective_review.json') == plan['original_full_visual_plan_sha256']
    assert plan['selected_fixed_cohort_ids'] == prior['selected_fixed_cohort_ids']
    assert plan['visual_rows'] == prior['visual_rows'][:225] and plan['columns'] == prior['columns']
    assert len(plan['visual_rows']) == 225 and plan['sheets'] == 45 and plan['exact256_pixel_cells'] == 1350
    assert not plan['outcome_based_case_or_probe_selection_added_here']
    completed = {(r['update'], r['proposal']) for r in audit['conditions']}
    conditions = {}
    expected_rows, expected_terms = [], []
    for probe in p['probes']:
        for proposal in PROPOSALS:
            if (probe['update'], proposal) not in completed:
                continue
            path = RETURN / output_prefix(probe['update'], proposal) / 'metrics.json'
            assert sha(path) == imported['files_sha256'][path.relative_to(RETURN).as_posix()]
            conditions[(probe['update'], proposal)] = read(path)
        baseline = conditions[(probe['update'], 'zero')]
        for proposal in PROPOSALS[1:]:
            if (probe['update'], proposal) not in completed:
                continue
            candidate = conditions[(probe['update'], proposal)]
            delta = np.array(candidate['objective_term_means']['current_batch']) - np.array(baseline['objective_term_means']['current_batch'])
            expected_terms.append({'update': probe['update'], 'proposal': proposal,
                'current_batch_raw_objective_term_changes': delta.tolist(), 'sum_of_seven_term_changes': float(delta.sum()),
                'landmark_term_nonincrease': bool(delta[0] <= 0)})
            for role in ROLES:
                for stage in ['raw', 'PNG']:
                    expected_rows.append({'update': probe['update'], 'proposal': proposal, 'role': role, 'stage': stage,
                        'cases': candidate['groups'][role][stage]['all']['cases'],
                        **candidate['comparison_to_same_before_state'][role][stage]})
    assert len(expected_rows) == 114 and expected_rows == report['all114_role_stage_state_proposal_rows']
    assert len(expected_terms) == 19 and expected_terms == report['all19_current_batch_raw_term_changes']
    expected_summaries = []
    for proposal in PROPOSALS[1:]:
        for role in ROLES:
            for stage in ['raw', 'PNG']:
                subset = [row for row in expected_rows if (row['proposal'], row['role'], row['stage']) == (proposal, role, stage)]
                assert len(subset) == (10 if proposal == 'recorded' else 9)
                gains = np.array([row['relative_feature_gain'] for row in subset])
                expected_summaries.append({'proposal': proposal, 'role': role, 'stage': stage, 'states': len(subset),
                    'finite_preservation_pass_updates': [row['update'] for row in subset if row['finite_preservation_pass']],
                    'finite_preservation_failure_updates': [row['update'] for row in subset if not row['finite_preservation_pass']],
                    'positive_feature_gain_updates': [row['update'] for row in subset if row['relative_feature_gain'] > 0],
                    'feature_gain_mean': float(gains.mean()), 'feature_gain_minimum': float(gains.min()), 'feature_gain_maximum': float(gains.max()),
                    'mean_is_repeated_case_diagnostic_summary_not_independent_population_estimate': True})
    assert len(expected_summaries) == 12 and report['summaries'] == expected_summaries
    cases = {row['id']: row for row in p['cases']}
    cell_count = 0
    for sheet_index, sheet in enumerate(report['sheets']):
        assert time.monotonic() - start < 300
        path = ROOT / sheet['path']
        assert sha(path) == sheet['sha256'] and not sheet['actually_viewed']
        with Image.open(path) as image:
            assert image.mode == 'RGB' and image.size == (1816, 1504)
            sheet_pixels = np.asarray(image).copy()
        selected = plan['visual_rows'][sheet_index * 5:sheet_index * 5 + 5]
        assert len(sheet['cells']) == 30
        for slot, row in enumerate(selected):
            case = cases[row['id']]
            camera = pixels(BUNDLE / case['input'])
            target = pixels(BUNDLE / case['target'])
            with Image.open(BUNDLE / case['observed']) as image:
                mask = np.asarray(image.convert('L')) > 0
            zero_path = RETURN / output_prefix(row['update'], 'zero') / row['role'] / (row['id'] + '.npz')
            with np.load(zero_path, allow_pickle=False) as data:
                original = np.floor(data['original_rgb'] * np.float32(255)).astype(np.uint8)
            original[~mask] = camera[~mask]
            expected_images = [camera, target, original]
            expected_images += [pixels(RETURN / output_prefix(row['update'], proposal) / row['role'] / (row['id'] + '.png')) for proposal in PROPOSALS]
            for column, expected in enumerate(expected_images):
                cell = sheet['cells'][slot * 6 + column]
                x, y = 280 + 256 * column, 64 + 288 * slot
                assert cell['row'] == row and cell['column'] == plan['columns'][column]
                assert cell['box'] == [x, y, x + 256, y + 256]
                assert np.array_equal(sheet_pixels[y:y + 256, x:x + 256], expected)
                assert cell['pixel_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
                cell_count += 1
    assert len(report['sheets']) == 45 and cell_count == 1350
    app = read(ROOT / 'outputs/cctv_dgp_pcgrad_fit_v41_audit_recovery_r1/app_preservation_readback.json')
    assert app['complete'] and app['app_bindings_verified'] == 14
    for name, digest in app['files_sha256'].items():
        assert sha(ROOT / name) == digest, name
    write(OUT / 'independent_arithmetic_and_sheets_audit.json', {
        'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'), 'prospective_plan_sha256': sha(OUT / 'prospective_review.json'),
        'source_inverse_verified': True, 'numeric_summary_rows_checked': 114, 'term_change_rows_checked': 19,
        'summaries_checked': 12, 'all45_sheets_checked': True, 'all1350_cells_exact': True,
        'visual_rows': 225, 'actually_viewed': False, 'app_bindings_verified': 14,
        'full_3150_review_complete': False, 'original_failure_retained': True,
        'model_forwards': 0, 'optimizer_updates': 0, 'gradient_queries': 0, 'VM_calls': 0,
        'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - start,
        'cap_seconds': 300, 'checker_sha256': sha(Path(__file__)),
    })
    print({'complete': True, 'summary_rows': 114, 'sheets': 45, 'exact_cells': 1350, 'actually_viewed': False, 'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    main()

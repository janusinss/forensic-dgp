"""Independently check all700 original-size cells and the finite-analysis decisions."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1'
RETURN = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_return'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    a, v, p = read(OUT / 'analysis.json'), read(OUT / 'visual_review.json'), read(RETURN / 'protocol.json')
    assert a['complete'] and v['complete'] and v['analysis_sha256'] == sha(OUT / 'analysis.json')
    for path, digest in a['bindings_sha256'].items():
        assert sha(ROOT / path) == digest, path
    seen, cell_count = [], 0
    for page in a['visual_pages']:
        assert sha(ROOT / page['path']) == page['sha256']
        with Image.open(ROOT / page['path']) as image:
            canvas = np.asarray(image.convert('RGB')).copy()
        assert canvas.shape == (1480, 1792, 3) and len(page['cells']) == 35
        label = Path(page['path']).stem.rsplit('_', 1)[0]
        cohort = next(c for c in p['cohorts'] if c['name'] == label)
        index = int(Path(page['path']).stem.rsplit('_', 1)[1]) - 1
        cases = cohort['cases'][index * 5:(index + 1) * 5]
        assert page['case_ids'] == [c['id'] for c in cases]
        for row, case in enumerate(cases):
            seen.append(case['id'])
            for column, variant in enumerate(['input', 'target', 'before', 'joint_1', 'joint_half', 'joint_quarter', 'joint_eighth']):
                expected_path = MIXED / case[variant] if column < 2 else RETURN / f'outputs/state0_{label}' / variant / (case['id'] + '.png')
                cell = page['cells'][row * 7 + column]
                x, y = column * 256, 79 + row * 284
                assert cell['id'] == case['id'] and cell['column'] == column and cell['xy'] == [x, y]
                assert ROOT / cell['path'] == expected_path and sha(expected_path) == cell['sha256']
                with Image.open(expected_path) as source:
                    pixels = np.asarray(source.convert('RGB')).copy()
                assert np.array_equal(canvas[y:y + 256, x:x + 256], pixels)
                cell_count += 1
    assert seen == [case['id'] for cohort in p['cohorts'] for case in cohort['cases']]
    assert len(seen) == len(set(seen)) == 100 and cell_count == 700
    assert len(v['pages']) == 20 and v['actually_viewed_cases'] == 100 and v['model_output_cells_reviewed'] == 500
    assert all(r['actually_viewed'] and r['native_cell_size'] == [256, 256] and r['image_detail'] == 'original' for r in v['pages'])
    assert not v['independent_final_review'] and not v['app_adoption']
    failures = []
    for row in a['variants']:
        label, variant = row['cohort'], row['variant']
        cmp = read(RETURN / f'outputs/state0_{label}/{variant}/comparison.json')
        assert row['preservation_against_original'] == cmp['preservation_against_original']
        assert row['degraded_PNG_structure_gain_percent'] == 100 * cmp['incremental_degraded_PNG_structure_gain']
        assert row['all108_linear_function_changes'] == cmp['all108_linear_function_changes']
        assert len(row['case_changes']) == 50 and len(row['first_order_to_finite_raw_checks']) == 34
        assert row['descriptive_raw_MSE_ArcFace_failures'] and row['preservation_against_original']['failures']
        assert not row['subset_preservation_source_brightness_pass'] and not row['training_capacity_pass']
        failures.append({'cohort': label, 'variant': variant,
            'PNG_failures': len(row['preservation_against_original']['failures']),
            'descriptive_raw_MSE_ArcFace_failures': len(row['descriptive_raw_MSE_ArcFace_failures'])})
    assert len(failures) == 8 and a['jointly_eligible_subset_variants'] == []
    old = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1'
    failed = read(old / 'failure_preservation.json')
    for name, digest in failed['partial_pages_sha256'].items():
        assert sha(old / name) == digest and sha(OUT / name) == digest
    assert len(failed['partial_pages_sha256']) == 10
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14
    for name, digest in app.items():
        assert sha(ROOT / name) == digest, name
    receipt = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'),
        'visual_review_sha256': sha(OUT / 'visual_review.json'), 'checker_sha256': sha(Path(__file__)),
        'all700_cells_exact': True, 'all100_cases_in_frozen_order': True, 'pages_verified': 20,
        'all8_failed_comparisons_retained': failures, 'all10_partial_pages_preserved_and_identical': True,
        'all14_app_bindings_unchanged': True, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
        'independent_final_visual_review': False, 'training_capacity_pass': False, 'app_promotion': False,
        'goal_complete': False, 'seconds': time.monotonic() - started}
    assert receipt['seconds'] <= 60
    with (OUT / 'independent_analysis_page_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'exact_cells': cell_count, 'pages': 20, 'seconds': receipt['seconds']}))


if __name__ == '__main__':
    main()

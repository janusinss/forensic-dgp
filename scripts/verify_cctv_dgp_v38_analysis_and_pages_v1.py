"""Verify all 700 original-size cells, saved decisions and prior-output parity."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1'
RETURN = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return'
V36 = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return'
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
    for path, digest in a['bindings_sha256'].items(): assert sha(ROOT / path) == digest, path
    seen, cell_count = [], 0
    for page, reviewed in zip(a['visual_pages'], v['pages'], strict=True):
        assert sha(ROOT / page['path']) == page['sha256'] == reviewed['sha256']
        assert page['path'] == reviewed['path'] and reviewed['case_ids'] == page['case_ids']
        assert reviewed['actually_viewed'] and reviewed['native_cell_size'] == [256, 256] and reviewed['image_detail'] == 'original'
        with Image.open(ROOT / page['path']) as image: canvas = np.asarray(image.convert('RGB')).copy()
        assert canvas.shape == (1480, 1792, 3) and len(page['cells']) == 35
        label = Path(page['path']).stem.rsplit('_', 1)[0]
        cohort = next(c for c in p['cohorts'] if c['name'] == label)
        index = int(Path(page['path']).stem.rsplit('_', 1)[1]) - 1
        cases = cohort['cases'][index * 5:(index + 1) * 5]
        assert page['case_ids'] == [c['id'] for c in cases]
        for row, case in enumerate(cases):
            seen.append(case['id'])
            for column, variant in enumerate(['input', 'target', 'before'] + [r['name'] for r in p['variants']]):
                expected_path = MIXED / case[variant] if column < 2 else RETURN / f'outputs/state0_{label}' / variant / (case['id'] + '.png')
                cell = page['cells'][row * 7 + column]
                x, y = column * 256, 79 + row * 284
                assert cell['id'] == case['id'] and cell['column'] == column and cell['xy'] == [x, y]
                assert ROOT / cell['path'] == expected_path and sha(expected_path) == cell['sha256']
                with Image.open(expected_path) as source: pixels = np.asarray(source.convert('RGB')).copy()
                assert np.array_equal(canvas[y:y + 256, x:x + 256], pixels)
                cell_count += 1
    assert seen == [case['id'] for cohort in p['cohorts'] for case in cohort['cases']]
    assert len(seen) == len(set(seen)) == 100 and cell_count == 700
    assert v['actually_viewed_cases'] == 100 and v['model_output_cells_reviewed'] == 500
    assert not v['independent_final_review'] and not v['app_adoption']
    prior = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/analysis.json')
    failures, raw_count, png_count, earlier_png_count, eligible = [], 0, 0, 0, []
    for row in a['variants']:
        label, variant = row['cohort'], row['variant']
        cmp = read(RETURN / f'outputs/state0_{label}/{variant}/comparison.json')
        assert row['preservation_against_original'] == cmp['preservation_against_original']
        assert row['degraded_PNG_structure_gain_percent'] == 100 * cmp['incremental_degraded_PNG_structure_gain']
        assert row['all210_linear_function_changes'] == cmp['all210_linear_function_changes']
        assert len(row['case_changes']) == 50 and len(row['first_order_to_finite_raw_checks']) == 34
        previous = next(r for r in prior['variants'] if r['cohort'] == label and r['scale'] == row['scale'])
        assert row['V36_same_scale_structure_gain_percent'] == previous['degraded_PNG_structure_gain_percent']
        assert row['change_from_V36_structure_gain_percentage_points'] == row['degraded_PNG_structure_gain_percent'] - previous['degraded_PNG_structure_gain_percent']
        before = read(RETURN / f'outputs/state0_{label}/before/receipt.json')
        current = read(RETURN / f'outputs/state0_{label}/{variant}/receipt.json')
        measured_failures = []
        for key in before['groups']:
            indices = [i for i, r in enumerate(before['rows']) if key in {'all', 'clear' if r['profile'] == 'clear' else 'degraded',
                r['source'] + '/all', r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'), r['source'] + '/' + r['profile']}]
            for metric in ['raw_MSE', 'raw_ArcFace']:
                old_value = float(np.mean([before['rows'][i]['metrics'][metric] for i in indices]))
                new_value = float(np.mean([current['rows'][i]['metrics'][metric] for i in indices]))
                if (new_value > old_value + 1e-12 if metric == 'raw_MSE' else new_value < old_value - 1e-6):
                    measured_failures.append({'group': key, 'metric': metric, 'baseline': old_value, 'candidate': new_value})
        keyed = lambda values: {(r['group'], r['metric']): (r['baseline'], r['candidate']) for r in values}
        assert keyed(measured_failures) == keyed(row['descriptive_raw_MSE_ArcFace_failures'])
        decision = row['preservation_against_original']
        expected_pass = not decision['failures'] and decision['brightness_gate_pass'] and all(z >= 0 for z in decision['source_structure_gains'].values())
        assert row['subset_preservation_source_brightness_pass'] == expected_pass and not row['training_capacity_pass']
        raw_count += len(measured_failures)
        png_count += len(decision['failures'])
        earlier_png_count += len(previous['preservation_against_original']['failures'])
        failures.append({'cohort': label, 'variant': variant, 'PNG_failures': len(decision['failures']),
            'descriptive_raw_MSE_ArcFace_failures': len(measured_failures), 'subset_preservation_source_brightness_pass': expected_pass})
    for variant in p['variants']:
        if all(row['subset_preservation_source_brightness_pass'] for row in a['variants'] if row['variant'] == variant['name']): eligible.append(variant['name'])
    assert a['jointly_eligible_subset_variants'] == eligible
    assert v['subset_preservation_passing_variants'] == eligible
    for cohort in p['cohorts']:
        for case in cohort['cases']:
            for suffix in ['.png', '.npy']:
                name = f'outputs/state0_{cohort["name"]}/before/{case["id"]}{suffix}'
                assert sha(V36 / name) == sha(RETURN / name), name
    assert a['all100_V36_original_PNG_and_raw_files_exact']
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14
    for name, digest in app.items(): assert sha(ROOT / name) == digest, name
    receipt = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'),
        'visual_review_sha256': sha(OUT / 'visual_review.json'), 'checker_sha256': sha(Path(__file__)),
        'all700_cells_exact': True, 'all100_cases_in_frozen_order': True, 'pages_verified': 20,
        'all8_comparisons_retained': failures, 'V38_PNG_failures': png_count, 'V36_PNG_failures': earlier_png_count,
        'V38_descriptive_raw_MSE_ArcFace_failures': raw_count, 'jointly_eligible_subset_variants': eligible,
        'all100_V36_original_PNG_and_raw_files_exact': True, 'all14_app_bindings_unchanged': True,
        'neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0,
        'independent_final_visual_review': False, 'training_capacity_pass': False, 'app_promotion': False,
        'goal_complete': False, 'seconds': time.monotonic() - started}
    assert receipt['seconds'] <= 60
    with (OUT / 'independent_analysis_page_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'exact_cells': cell_count, 'pages': 20, 'seconds': receipt['seconds']}))


if __name__ == '__main__':
    main()

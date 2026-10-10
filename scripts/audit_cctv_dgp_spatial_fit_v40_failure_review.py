"""Independent preview selection, pixel-cell and TRAIN exposure audit; no models."""
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
from PIL import Image
from cctv_dgp_spatial_fit_v40_contract import read, write, sha, validate_schedule, PROFILES

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'
RETURNED = ROOT/'outputs/cctv_dgp_spatial_fit_v40_return'
OUT = ROOT/'outputs/cctv_dgp_spatial_fit_v40_failure_review'


def pixels(path):
    with Image.open(path) as im: return np.array(im.convert('RGB'))


def main():
    start = time.monotonic(); p = read(BUNDLE/'protocol.json'); plan = read(OUT/'plan.json'); ready = read(OUT/'preparation.json')
    a_path = ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json'; a = read(a_path)
    assert ready['complete'] and a['complete'] and a['failure_retained'] and not a['necessary_capacity_pass']
    assert plan['return_audit_sha256'] == sha(a_path) and plan['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    assert plan['source_sha256'] == sha(ROOT/'scripts/prepare_cctv_dgp_spatial_fit_v40_failure_review.py') and plan['checker_sha256'] == sha(Path(__file__))
    assert ready['plan_sha256'] == sha(OUT/'plan.json') and plan['case_ids'] == p['preview_case_ids']
    assert plan['cell_size'] == [256, 256] and not plan['image_processing_or_resize']
    assert plan['columns'] == ['Input256', 'Retained DGP / update0', 'V40 stopped50', 'Mean-shift-only diagnostic', 'Paired TRAIN target']
    for name, digest in ready['source_bindings_sha256'].items(): assert sha(ROOT/name) == digest
    schedule = read(BUNDLE/'schedule.json')['batches']; validate_schedule(p['cases'], schedule)
    cases = {c['id']: c for c in p['cases']}; exposed = {p['cases'][i]['id'] for batch in schedule[:50] for i in batch}
    assert len(exposed) == ready['unique_cases_optimized_by50'] == 250
    initial, stopped = [read(RETURNED/('outputs/update'+str(n)+'/metrics.json')) for n in [0, 50]]
    before, after = [{row['id']: row for row in v['rows']} for v in [initial, stopped]]
    for label, bucket in ready['exposure_buckets'].items():
        selected = [cid for cid in before if cases[cid]['profile'] != 'clear' and
                    (label == 'all_TRAIN' or label == 'exposed_by50_TRAIN' and cid in exposed or
                     label == 'not_yet_exposed_TRAIN' and cid not in exposed or label == 'fixed50_TRAIN' and cid in plan['case_ids'])]
        assert len(selected) == bucket['degraded_cases']
        baseline, current = [float(np.mean([table[cid]['metrics']['landmark_high_frequency_MSE'] for cid in selected])) for table in [before, after]]
        assert baseline == bucket['baseline_feature_MSE'] and current == bucket['stopped50_feature_MSE'] and 1-current/baseline == bucket['relative_feature_gain']
        assert bucket['all_are_TRAIN_not_held_out_evaluation']
    assert ready['early_gate'] == a['gates'][0] and not ready['early_gate']['pass']
    assert ready['fixed50_cases_optimized_by50'] == len(exposed&set(plan['case_ids']))
    checked_cells = 0; viewed_ids = []
    for page in ready['pages']:
        assert sha(OUT/page['path']) == page['sha256']; canvas = pixels(OUT/page['path'])
        assert canvas.shape == (1516, 1336, 3) and len(page['cells']) == 25 and len(page['ids']) == 5
        assert {cases[cid]['source_person_or_reference'] for cid in page['ids']} == {page['reference']}
        assert [cases[cid]['profile'] for cid in page['ids']] == PROFILES
        for cell in page['cells']:
            c = cases[cell['id']]; cid = c['id']; column = cell['column']; row_index = page['ids'].index(cid)
            expected = [BUNDLE/c['input'], RETURNED/('outputs/update0/'+cid+'.png'), RETURNED/('outputs/update50/'+cid+'.png'),
                        RETURNED/('outputs/update50/'+cid+'_mean_only.png'), BUNDLE/c['target']][column]
            assert cell['source'] == expected.relative_to(ROOT).as_posix() and cell['xy'] == [12+column*264, 88+row_index*288]
            source = pixels(expected); assert source.shape == (256, 256, 3) and hashlib.sha256(source.tobytes()).hexdigest() == cell['RGB_sha256']
            x, y = cell['xy']; np.testing.assert_array_equal(canvas[y:y+256, x:x+256], source)
            checked_cells += 1
        viewed_ids.extend(page['ids'])
    assert viewed_ids == plan['case_ids'] and checked_cells == 250 and len(ready['pages']) == 10
    for row in ready['rows']:
        cid = row['id']; c = cases[cid]
        assert row['source'] == c['source'] and row['profile'] == c['profile'] and row['optimized_by50'] == (cid in exposed)
        assert row['baseline_metrics'] == before[cid]['metrics'] and row['stopped50_metrics'] == after[cid]['metrics']
        with Image.open(BUNDLE/c['observed']) as im: mask = np.array(im) > 0
        r0, r50 = [np.load(RETURNED/('outputs/update'+str(n)+'/'+cid+'.npy'), allow_pickle=False) for n in [0, 50]]
        difference = (r50-r0)[mask].astype(np.float64)
        assert row['raw_correction_RMS'] == float(np.sqrt(np.square(difference).mean()))
        np.testing.assert_array_equal(row['mean_postclip_RGB_shift'], difference.mean(0))
        delta = pixels(RETURNED/('outputs/update50/'+cid+'.png')).astype(np.int16)-pixels(RETURNED/('outputs/update0/'+cid+'.png')).astype(np.int16)
        assert row['maximum_PNG_byte_change'] == int(np.abs(delta).max()) and row['changed_PNG_pixels'] == int(np.any(delta != 0, -1).sum())
    assert not any(n in sys.modules for n in ['torch', 'dgp_frozen_inference_v2', 'pretrained_completion'])
    assert time.monotonic()-start < 180
    write(OUT/'independent_preparation_audit.json', {'complete': True, 'checker_sha256': sha(Path(__file__)),
          'plan_sha256': sha(OUT/'plan.json'), 'preparation_sha256': sha(OUT/'preparation.json'),
          'prospective50_ids_verified': True, 'exact_page_cells': checked_cells, 'pages': 10,
          'exposure_buckets_and_raw_deltas_verified': True, 'unchanged_failed_early_gate': True,
          'model_forwards': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'app_promotion': False,
          'native_or_reserved_used': False, 'independent_final_review': False,
          'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 180})
    print({'complete': True, 'cells': checked_cells, 'pages': 10, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()

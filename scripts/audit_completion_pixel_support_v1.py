"""Independent recount from original photos, masks and saved pipeline arrays."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_pixel_support_v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def image(path, mode):
    with Image.open(path) as source: return np.asarray(source.convert(mode)).copy()


def main():
    p, r = read(OUT / 'plan.json'), read(OUT / 'results.json')
    assert r['complete'] and r['plan_sha256'] == sha(OUT / 'plan.json')
    for n, h in p['sources_sha256'].items(): assert sha(ROOT / n) == h, n
    for n, h in r['artifacts_sha256'].items(): assert sha(OUT / n) == h, n
    assert r['eligible_cases_recounted'] == 32 and r['ROI_recounts'] == 8 and len(r['rows']) == 36
    assert p['ROIs_selected_after_previous_development_output_review'] and p['ROIs_are_diagnostic_rectangles_not_segmentation_truth']
    assert not r['quality_qualification'] and not r['independent_final_review'] and not r['native_CCTV_or_reserved_final_used']
    checked, regions = 0, 0
    for c in p['cases']:
        row = next(x for x in r['rows'] if x['id'] == c['id'])
        assert c['rejected'] == (c['operator_input_review'] != 'usable')
        if c['rejected']:
            assert row == {'id': c['id'], 'rejected_before_previous_inference': True}; continue
        original = image(ROOT / c['input'], 'RGB'); mask = image(ROOT / c['reviewed'], 'L') == 255
        output, context = image(ROOT / c['output'], 'RGB'), image(ROOT / c['context6'], 'RGB')
        with np.load(ROOT / c['stages'], allow_pickle=False) as data: completion = data['completion'][0].transpose(1, 2, 0)
        raw = np.floor(completion * np.float32(255)).astype(np.uint8)
        if read(ROOT / c['metadata'])['display_processing']['colour_policy']['applied']:
            raw = np.repeat(cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
        assert np.array_equal(output[mask], raw[mask]) and np.array_equal(output[~mask], original[~mask])
        assert np.array_equal(context[~mask], original[~mask])
        assert row['source_support_pixels'] == int(np.count_nonzero(~mask))
        assert row['estimate_support_pixels'] == int(np.count_nonzero(mask)) and row['source_changed_pixels'] == 0
        checked += 1
        for roi in p['ROIs']:
            if c['id'] not in [roi['base_id'] + '_native', roi['base_id'] + '_degraded']: continue
            evidence = next(x for x in r['regions'] if x['id'] == c['id'] and x['name'] == roi['name'])
            assert evidence['xyxy'] == roi['xyxy'] and evidence['hidden_ground_truth'] is None
            x0, y0, x1, y1 = roi['xyxy']; ix = (slice(y0, y1), slice(x0, x1))
            selected = mask[ix]; src, delivered, old = original[ix], output[ix], context[ix]
            assert evidence['rectangle_pixels'] == (x1 - x0) * (y1 - y0)
            assert evidence['estimate_support_pixels'] == int(np.count_nonzero(selected))
            assert evidence['source_support_pixels'] == int(np.count_nonzero(~selected))
            assert evidence['estimate_fraction'] == float(np.count_nonzero(selected) / selected.size)
            assert evidence['source_changed_pixels'] == 0 and np.array_equal(delivered[~selected], src[~selected])
            assert evidence['estimated_pixels_different_from_uploaded_covering'] == int(np.count_nonzero(np.max(delivered[selected] != src[selected], axis=-1)))
            assert evidence['context6_changed_estimated_pixels'] == int(np.count_nonzero(np.max(delivered[selected] != old[selected], axis=-1)))
            regions += 1
    assert checked == 32 and regions == 8 and 'torch' not in sys.modules
    preparation_failure = read(ROOT / 'outputs/completion_pixel_support_v1_preparation_failure/failure.json')
    assert not preparation_failure['main_output_directory_created'] and preparation_failure['new_inference_calls'] == 0
    assert sha(ROOT / 'outputs/completion_pixel_support_v1_preparation_failure/diagnostic_source_before_correction.py') == preparation_failure['original_source_sha256']
    for key in ['model_forwards', 'local_gradients', 'optimizer_updates']: assert r[key] == 0
    assert not r['mask_edits'] and not r['app_changes'] and r['hidden_reference_metrics'] is None
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'results_sha256': sha(OUT / 'results.json'),
               'sources_verified': len(p['sources_sha256']), 'eligible_Off_compositions_independently_verified': checked,
               'support_ROIs_independently_recounted': regions, 'source_pixels_exact_in_current_and_context6': True,
               'post_hoc_rectangles_not_ground_truth': True, 'model_calls': 0, 'local_gradients': 0,
               'original_preparation_failure_and_operator_review_precedence_verified': True,
               'mask_edits': False, 'app_changes': False, 'hidden_metrics': None,
               'independent_quality_evaluation': False, 'goal_complete': False}
    with (OUT / 'independent_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()

"""Independent review/support-window recount; no neural calls or new masks."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_conditioning_union_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rgb(path):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.array(im.convert('RGB'))


def binary(path):
    with Image.open(path) as im:
        assert im.size == (256, 256) and im.mode == 'L'
        values = np.array(im)
    assert set(np.unique(values)) <= {0, 255}
    return values != 0


def main():
    started = time.monotonic()
    p, r, v, d, audit = [read(OUT / name) for name in ['protocol.json', 'results.json', 'visual_review.json',
                                                     'pixel_support_diagnostic.json', 'independent_saved_output_audit.json']]
    assert v['complete'] and d['complete'] and audit['complete']
    assert v['protocol_sha256'] == d['protocol_sha256'] == audit['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert v['results_sha256'] == d['results_sha256'] == audit['results_sha256'] == sha(OUT / 'results.json')
    assert v['pixel_support_diagnostic_sha256'] == sha(OUT / 'pixel_support_diagnostic.json')
    assert v['independent_saved_output_audit_sha256'] == sha(OUT / 'independent_saved_output_audit.json')
    assert v['all32_outputs_and_same_final_baselines_actually_viewed'] and v['all8_pages_actually_viewed_at_original256_cell_detail']
    assert not v['automatic_quality_qualification'] and not v['assisted_quality_qualification'] and not v['app_adoption'] and not v['independent_final_review']
    assert v['no_output_based_reannotation_or_seed_search'] and v['no_delivered_mask_expansion']
    assert v['quality_criteria_unchanged'] == p['quality_criteria'] and v['pages'] == r['pages']
    mapping = {c['id']: c for c in p['cases']}
    assert [row['id'] for row in v['rows']] == [c['id'] for c in p['cases']]
    eligible = []
    for row in v['rows']:
        c = mapping[row['id']]
        assert row['condition'] == c['condition'] and row['excluded_before_generation'] == c['rejected']
        if c['rejected']:
            assert row['input_review'] == c['input_review']
        else:
            assert row['family'] == c['family'] and row['visible_outside_final_and_protected_bytes_exact']
            assert row['hidden_reference'] is None and not row['independent_final_quality_review']
            assert row['observed_comparison'] and row['development_decision']
            eligible.append(c['id'])
    viewed = [entry['id'] for page in v['pages'] for entry in page['entries']]
    assert len(viewed) == len(set(viewed)) == len(eligible) == 32 and set(viewed) == set(eligible)
    for page in v['pages']:
        assert sha(OUT / page['path']) == page['sha256']
    assert d['not_covering_labels_or_hidden_face_truth'] and d['no_mask_or_annotation_changed']
    assert d['scope_counts_not_covering_miss_rates'] and len(d['rows']) == 8
    exact_window_copied_pixels = 0
    for row in d['rows']:
        c = mapping[row['id']]
        x1, y1, x2, y2 = row['rectangle_xyxy']
        assert all(isinstance(value, int) for value in [x1, y1, x2, y2]) and 0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256
        mask = binary(ROOT / c['masks']['removal'])[y1:y2, x1:x2]
        new, old, source = [rgb(path)[y1:y2, x1:x2] for path in [OUT / 'images' / (c['id'] + '.png'), ROOT / c['baseline_output'], ROOT / c['input']]]
        assert row['window_pixels'] == mask.size and row['generated_support_pixels'] == int(mask.sum())
        assert row['copied_support_pixels'] == int((~mask).sum())
        assert row['changed_estimated_pixels'] == int(np.any(new != old, axis=-1)[mask].sum())
        assert row['inside_mask_pixels_coincidentally_equal_source'] == int(np.all(new == source, axis=-1)[mask].sum())
        np.testing.assert_array_equal(new[~mask], source[~mask])
        np.testing.assert_array_equal(old[~mask], source[~mask])
        exact_window_copied_pixels += int((~mask).sum())
    for name, digest in {**p['sources_sha256'], **p['app_preservation_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': sha(OUT / 'protocol.json'),
               'visual_review_sha256': sha(OUT / 'visual_review.json'), 'pixel_support_diagnostic_sha256': sha(OUT / 'pixel_support_diagnostic.json'),
               'all32_reviewed_ids_and8_pages_verified': True, 'all4_input_exclusions_retained': True,
               'post_output_processing_windows': 8, 'exact_copied_window_pixels_in_both_outputs': exact_window_copied_pixels,
               'windows_not_new_masks_or_semantic_truth': True, 'all14_app_bindings_unchanged': True,
               'model_forwards': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'app_adoption': False,
               'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
               'independent_final_review': False, 'seconds': time.monotonic() - started, 'cap_seconds': 180}
    assert receipt['seconds'] < 180
    with (OUT / 'independent_review_support_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ['complete', 'seconds', 'post_output_processing_windows',
                     'exact_copied_window_pixels_in_both_outputs']}), flush=True)


if __name__ == '__main__':
    main()

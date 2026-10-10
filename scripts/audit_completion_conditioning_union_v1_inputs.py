"""Independent saved-mask/page audit and pre-inference protocol freeze."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_conditioning_union_v1'
BASE = ROOT / 'outputs/completion_input_footprints_comparison_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def rgb(path):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.array(im.convert('RGB'))


def binary(path):
    with Image.open(path) as im:
        assert im.mode == 'L' and im.size == (256, 256)
        values = np.array(im)
    assert set(np.unique(values)) <= {0, 255}
    return values != 0


def main():
    started = time.monotonic()
    assert not (OUT / 'protocol.json').exists() and not (OUT / 'execution.json').exists()
    draft, parent, review = read(OUT / 'draft.json'), read(BASE / 'protocol.json'), read(OUT / 'input_visual_review.json')
    assert review['complete'] and review['draft_sha256'] == sha(OUT / 'draft.json')
    assert review['all9_pages_actually_viewed_at_original_detail'] and review['all36_inputs_reviewed']
    assert not review['output_based_reannotation'] and not review['quality_qualification']
    for name, digest in {**draft['sources_sha256'], **draft['app_preservation_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    assert draft['parent_protocol_sha256'] == sha(BASE / 'protocol.json')
    assert draft['parent_results_sha256'] == sha(BASE / 'results.json')
    assert draft['parent_model_state'] == read(BASE / 'results.json')['state_before']
    assert draft['quality_criteria'] == parent['quality_criteria']
    assert draft['seed'] == parent['seed'] and draft['search_trials_per_case'] == 1
    assert draft['max_forwards'] == {'completion': 29, 'internal512': 29, 'DGP': 0, 'detector': 0}
    assert draft['max_requests'] == 37 and draft['cap_seconds'] == 600 and draft['external_timeout_seconds'] == 630
    assert draft['artifact_cap_bytes'] == 268435456 and draft['expected_trial_forwards'] == 28
    assert len(draft['cases']) == len(parent['cases']) == 36
    pairs, recounted, exclusions = {}, [], 0
    for c, old_c in zip(draft['cases'], parent['cases']):
        assert all(c[key] == value for key, value in old_c.items()), c['id']
        pairs.setdefault(c['base_id'], []).append(c)
        if c['rejected']:
            assert c['input_review'] in ['out_of_scope', 'needs_clearer'] and 'conditioning' not in c
            exclusions += 1
            continue
        source = rgb(ROOT / c['input'])
        core, final, face, protected = [binary(ROOT / c['masks'][key]) for key in ['core', 'removal', 'face', 'protected']]
        old, union = binary(ROOT / c['reviewed']), binary(ROOT / c['conditioning'])
        expected = old | final
        np.testing.assert_array_equal(union, expected)
        assert not (final & ~union).any() and not (final & protected).any() and not (final & ~face).any()
        assert float(union.mean()) < .85
        if core.any():
            np.testing.assert_array_equal(final, core | ((distance_transform_edt(~core) <= 2) & face & ~protected))
        else:
            assert not final.any() and not union.any() and protected.all()
        counts = {'final_pixels': int(final.sum()), 'conditioning_pixels': int(union.sum()),
                  'extra_conditioning_pixels': int((union & ~final).sum()),
                  'hidden_protected_context_pixels': int((union & protected).sum()),
                  'hidden_outside_face_context_pixels': int((union & ~face).sum()),
                  'delivered_protected_overlap_pixels': 0,
                  'visible_context_rgb_sha256': hashlib.sha256(source[~union].tobytes()).hexdigest()}
        assert counts == c['geometry_counts']
        recounted.append({'id': c['id'], **counts})
    assert len(recounted) == 32 and exclusions == 4 and sum(not row['final_pixels'] for row in recounted) == 4
    assert len(pairs) == 18
    for values in pairs.values():
        assert len(values) == 2 and {v['condition'] for v in values} == {'original_photo', 'synthetic_degraded_photo'}
        assert values[0]['rejected'] == values[1]['rejected'] and values[0]['input_review'] == values[1]['input_review']
        if not values[0]['rejected']:
            for field in ['conditioning']:
                np.testing.assert_array_equal(binary(ROOT / values[0][field]), binary(ROOT / values[1][field]))
            np.testing.assert_array_equal(binary(ROOT / values[0]['masks']['removal']), binary(ROOT / values[1]['masks']['removal']))
    assert recounted == draft['mask_counts']
    assert dict(Counter(c['family'] for c in draft['cases'] if not c['rejected'])) == draft['family_case_counts']
    actual_ids, cells = [], 0
    mapping = {c['id']: c for c in draft['cases']}
    assert len(draft['input_pages']) == len(review['pages']) == 9
    for page, observed_page in zip(draft['input_pages'], review['pages']):
        assert page['path'] == observed_page['path'] and page['sha256'] == observed_page['sha256'] == sha(OUT / page['path'])
        with Image.open(OUT / page['path']) as im:
            sheet = np.array(im.convert('RGB'))
        for entry in page['entries']:
            c = mapping[entry['id']]
            source = rgb(ROOT / c['input'])
            if c['rejected']:
                assert sheet.shape == (324, 1024, 3)
                x = entry['column'] * 256
                np.testing.assert_array_equal(sheet[24:280, x:x + 256], source)
                cells += 1
            else:
                assert sheet.shape == (1214, 1024, 3)
                final, union = binary(ROOT / c['masks']['removal']), binary(ROOT / c['conditioning'])
                expected_cells = [source]
                for support, colour in [(final, (16, 185, 129)), (union, (240, 140, 32)), (union & ~final, (210, 64, 180))]:
                    values = source.astype(np.float64)
                    values[support] = values[support] * .55 + np.array(colour) * .45
                    expected_cells.append(np.floor(values + .5).astype(np.uint8))
                y = 30 + entry['row'] * 296
                for k, expected_cell in enumerate(expected_cells):
                    np.testing.assert_array_equal(sheet[y:y + 256, k * 256:(k + 1) * 256], expected_cell)
                    cells += 1
            actual_ids.append(c['id'])
    assert len(actual_ids) == len(set(actual_ids)) == 36 and cells == 132
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'draft_sha256': sha(OUT / 'draft.json'),
               'input_visual_review_sha256': sha(OUT / 'input_visual_review.json'), 'mask_pairs': 18,
               'exact_union_masks': 32, 'unchanged_final_masks': 32, 'exact_input_page_cells': cells,
               'all4_exclusions_retained': True, 'all4_empty_controls_retained': True,
               'all14_app_bindings_unchanged': len(draft['app_preservation_sha256']) == 14,
               'hidden_protected_context_pixels_reported': sum(row['hidden_protected_context_pixels'] for row in recounted),
               'delivered_protected_overlap_pixels': 0, 'source_bindings': len(draft['sources_sha256']),
               'model_forwards': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'app_changes': False,
               'native_CCTV_or_reserved_final_used': False, 'quality_qualification': False,
               'seconds': time.monotonic() - started, 'cap_seconds': 180}
    assert receipt['seconds'] < 180
    write(OUT / 'independent_input_audit.json', receipt)
    protocol = dict(draft)
    protocol.update({'format': 'completion-conditioning-union-frozen-v1',
                     'draft_sha256': sha(OUT / 'draft.json'),
                     'input_audit_sha256': sha(OUT / 'independent_input_audit.json'),
                     'input_visual_review_sha256': sha(OUT / 'input_visual_review.json'),
                     'input_review_before_freezing_complete': True})
    write(OUT / 'protocol.json', protocol)
    print(json.dumps({'complete': True, 'cells': cells, 'masks': 32, 'seconds': receipt['seconds'],
                      'protocol_sha256': sha(OUT / 'protocol.json')}), flush=True)


if __name__ == '__main__':
    main()

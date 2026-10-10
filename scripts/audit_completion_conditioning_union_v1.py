"""Independent full saved-input/raw/PNG arithmetic audit; no model construction."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

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
        assert im.mode == 'L' and im.size == (256, 256)
        values = np.array(im)
    assert set(np.unique(values)) <= {0, 255}
    return values != 0


def stage_audit(source, union, stage_path):
    with np.load(stage_path, allow_pickle=False) as stages:
        assert set(stages.files) == {'completion', 'internal512', 'neural_input512'}
        composed, net, actual_input = (stages[key] for key in ['completion', 'internal512', 'neural_input512'])
        assert composed.shape == (1, 3, 256, 256) and net.shape == actual_input.shape == (1, 3, 512, 512)
        assert composed.dtype == net.dtype == actual_input.dtype == np.float32
        assert np.isfinite(composed).all() and np.isfinite(net).all() and np.isfinite(actual_input).all()
        assert composed.min() >= 0 and composed.max() <= 1 and actual_input.min() >= -1 and actual_input.max() <= 1
        canonical = torch.from_numpy((source.astype(np.float32) / np.float32(255)).transpose(2, 0, 1).copy())[None]
        mask = torch.from_numpy(union.astype(np.float32))[None, None]
        visible = 1 - mask
        with torch.inference_mode():
            support = F.interpolate(visible, size=(512, 512), mode='bilinear', align_corners=False)
            context = F.interpolate(canonical * visible, size=(512, 512), mode='bilinear', align_corners=False)
            normal = context / support.clamp_min(1e-8)
            high_mask = F.interpolate(mask, size=(512, 512), mode='nearest')
            expected_input = (normal * (1 - high_mask) + high_mask) * 2 - 1
            resized = F.interpolate(((torch.from_numpy(net) + 1) / 2).clamp(0, 1),
                                    size=(256, 256), mode='bilinear', align_corners=False)
            expected_raw = canonical * (1 - mask) + resized * mask
        np.testing.assert_array_equal(actual_input, expected_input.numpy())
        np.testing.assert_array_equal(composed, expected_raw.numpy())
        np.testing.assert_array_equal(actual_input.transpose(0, 2, 3, 1)[high_mask.numpy()[:, 0].astype(bool)],
                                      np.ones((int(high_mask.sum()), 3), dtype=np.float32))
        return composed[0].transpose(1, 2, 0).copy()


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    p, r, outer = read(OUT / 'protocol.json'), read(OUT / 'results.json'), read(OUT / 'external_receipt.json')
    assert r['complete'] and outer['complete'] and outer['worker_exit_code'] == 0 and not outer['timeout']
    assert r['protocol_sha256'] == outer['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert outer['log_sha256'] == sha(OUT / 'inference.log')
    assert 0 < r['seconds'] <= p['cap_seconds'] == 600 and 0 < outer['external_seconds'] <= p['external_timeout_seconds'] == 630
    assert r['state_before'] == r['state_after'] == p['parent_model_state']
    assert r['forwards'] == p['max_forwards'] == {'completion': 29, 'internal512': 29, 'DGP': 0, 'detector': 0}
    assert r['requests'] == 37 and r['optimizer_updates'] == r['backward_calls'] == r['gradient_calls'] == 0
    assert not r['app_changes'] and not r['new_checkpoint'] and not r['automatic_forwards']
    assert p['input_audit_sha256'] == sha(OUT / 'independent_input_audit.json') and read(OUT / 'independent_input_audit.json')['complete']
    assert p['input_visual_review_sha256'] == sha(OUT / 'input_visual_review.json')
    for name, digest in {**p['sources_sha256'], **p['app_preservation_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    for name, digest in r['artifacts_sha256'].items():
        assert sha(OUT / name) == digest, name
    assert sum((OUT / name).stat().st_size for name in r['artifacts_sha256']) == r['artifact_bytes'] <= p['artifact_cap_bytes'] == 268435456
    parity = next(c for c in p['cases'] if c['id'] == p['parity_case'])
    source, mask = rgb(ROOT / parity['input']), binary(ROOT / parity['masks']['removal'])
    stage_audit(source, mask, OUT / 'parity/stages.npz')
    np.testing.assert_array_equal(rgb(OUT / 'parity/estimate.png'), rgb(ROOT / parity['baseline_output']))
    with np.load(ROOT / parity['baseline_stages'], allow_pickle=False) as old, np.load(OUT / 'parity/stages.npz', allow_pickle=False) as fresh:
        for key in ['internal512', 'completion']:
            np.testing.assert_array_equal(old[key], fresh[key])
    parity_receipt = read(OUT / 'parity/receipt.json')
    assert parity_receipt['complete'] and parity_receipt['case'] == parity['id'] and not parity_receipt['new_tolerance']
    mapping = {c['id']: c for c in p['cases']}
    assert [row['id'] for row in r['rows']] == [c['id'] for c in p['cases']]
    visible_bytes = protected_bytes = generated = bypasses = rejections = 0
    rows = []
    for row in r['rows']:
        c = mapping[row['id']]
        assert row['rejected_before_neural'] == c['rejected']
        if c['rejected']:
            assert row['input_review'] == c['input_review'] and row['input_review'] in ['needs_clearer', 'out_of_scope']
            assert not (OUT / 'images' / (c['id'] + '.png')).exists() and not (OUT / 'stages' / (c['id'] + '.npz')).exists()
            rejections += 1
            continue
        source, output = rgb(ROOT / c['input']), rgb(OUT / row['output'])
        final, union, protected = binary(ROOT / c['masks']['removal']), binary(ROOT / c['conditioning']), binary(ROOT / c['masks']['protected'])
        np.testing.assert_array_equal(union, final | binary(ROOT / c['reviewed']))
        assert not (final & ~union).any() and not (final & protected).any()
        meta = read(OUT / row['metadata'])
        expected = source.copy()
        if final.any():
            raw = stage_audit(source, union, OUT / row['stages'])
            expected[final] = np.floor(raw[final] * np.float32(255)).astype(np.uint8)
            assert meta['completion']['weights_sha256'] == 'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
            generated += 1
        else:
            with np.load(OUT / row['stages'], allow_pickle=False) as empty:
                assert not empty.files and meta['completion'] == {'bypassed': 'empty mask'}
            assert not union.any()
            bypasses += 1
        old_colour = read(ROOT / c['baseline_metadata'])['display_processing']['colour_policy']
        colour = meta['display_processing']['colour_policy']
        assert colour == old_colour
        # Derive palette selection independently from FINAL visible support.
        support = cv2.erode((~final).astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
        blurred = cv2.GaussianBlur(source.astype(np.float32), (5, 5), 1)
        mean_chroma = float((blurred.max(axis=-1) - blurred.min(axis=-1))[support].mean()) if int(support.sum()) >= 512 else None
        assert colour['mean_visible_channel_range_255'] == mean_chroma
        assert colour['grayscale_input'] == colour['applied'] == (mean_chroma is not None and mean_chroma <= 4)
        if colour['applied']:
            gray = cv2.cvtColor(expected, cv2.COLOR_RGB2GRAY)
            expected[final] = np.repeat(gray[..., None], 3, axis=-1)[final]
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(output[~final], source[~final])
        np.testing.assert_array_equal(output[protected], source[protected])
        assert meta['restoration_requested'] == 'off' and not meta['restoration_applied']
        assert meta['mask_source'] == 'assisted_reviewed'
        assert meta['final_mask_sha256'] == hashlib.sha256(final.astype(np.uint8).tobytes()).hexdigest()
        assert meta['conditioning_mask_sha256'] == hashlib.sha256(union.astype(np.uint8).tobytes()).hexdigest()
        assert meta['original_rgb_sha256'] == hashlib.sha256(source.tobytes()).hexdigest()
        assert meta['output_rgb_sha256'] == hashlib.sha256(output.tobytes()).hexdigest()
        assert meta['optimizer_updates'] == meta['backward_calls'] == meta['gradient_calls'] == 0
        baseline = rgb(ROOT / c['baseline_output'])
        np.testing.assert_array_equal(output[~final], baseline[~final])
        changed = int(np.any(output != baseline, axis=-1)[final].sum())
        assert changed == row['changed_estimated_pixels_vs_same_final_baseline']
        assert row['mask_pixels'] == int(final.sum()) and row['conditioning_pixels'] == int(union.sum()) and not row['outside_final_changed_pixels']
        visible_bytes += int((~final).sum()) * 3
        protected_bytes += int(protected.sum()) * 3
        rows.append({'id': c['id'], 'condition': c['condition'], 'family': c['family'], 'empty_bypass': not bool(final.any()),
                     'unchanged_final_mask': True, 'changed_pixels_vs_baseline_within_final': changed,
                     'source_changed_pixels_outside_final': 0, 'protected_changed_pixels': 0,
                     'neural_input_and512_to256_exact': bool(final.any())})
        assert time.monotonic() - started < 180
    assert len(rows) == 32 and generated == 28 and bypasses == rejections == 4
    page_ids, exact_cells = [], 0
    assert len(r['pages']) == 8
    for page in r['pages']:
        assert page['sha256'] == sha(OUT / page['path']) and len(page['entries']) == 4
        with Image.open(OUT / page['path']) as im:
            assert im.size == (1280, 1214)
            values = np.array(im.convert('RGB'))
        for entry in page['entries']:
            c = mapping[entry['id']]
            source = rgb(ROOT / c['input'])
            cells = [source]
            for support, colour, next_image in [(binary(ROOT / c['masks']['removal']), (16, 185, 129), rgb(ROOT / c['baseline_output'])),
                                                 (binary(ROOT / c['conditioning']), (240, 140, 32), rgb(OUT / 'images' / (c['id'] + '.png')))]:
                over = source.astype(np.float64)
                over[support] = over[support] * .55 + np.array(colour) * .45
                cells.extend([np.floor(over + .5).astype(np.uint8), next_image])
            y = 30 + entry['row'] * 296
            for k, expected_cell in enumerate(cells):
                np.testing.assert_array_equal(values[y:y + 256, k * 256:(k + 1) * 256], expected_cell)
                exact_cells += 1
            page_ids.append(c['id'])
    assert len(page_ids) == len(set(page_ids)) == 32 and exact_cells == 160
    assert not any(name in sys.modules for name in ['dgp_face_workflow_v3', 'pretrained_completion', 'third_party.codeformer.codeformer_arch'])
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': sha(OUT / 'protocol.json'),
               'results_sha256': sha(OUT / 'results.json'), 'external_receipt_sha256': sha(OUT / 'external_receipt.json'),
               'input_audit_sha256': sha(OUT / 'independent_input_audit.json'), 'seconds': time.monotonic() - started,
               'cap_seconds': 180, 'same_pipeline_parity_PNG_and_raw_exact': True, 'exact_trial_outputs': 32,
               'exact_union_neural_inputs_and_raw_compositions': 28, 'exact_empty_bypasses': 4, 'input_exclusions_retained': 4,
               'visible_source_bytes_exact': visible_bytes, 'protected_source_bytes_exact': protected_bytes,
               'same_final_support_as_baseline': True, 'exact_page_cells': exact_cells,
               'source_bindings': len(p['sources_sha256']), 'artifact_bindings': len(r['artifacts_sha256']), 'rows': rows,
               'all14_app_bindings_unchanged': True, 'model_forwards': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
               'app_changes': False, 'hidden_metrics': None, 'native_CCTV_or_reserved_final_used': False,
               'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
               'independent_final_review': False, 'goal_complete': False}
    with (OUT / 'independent_saved_output_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ['complete', 'seconds', 'exact_trial_outputs',
                      'exact_union_neural_inputs_and_raw_compositions', 'visible_source_bytes_exact', 'exact_page_cells']}), flush=True)


if __name__ == '__main__':
    main()

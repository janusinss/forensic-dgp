"""Independent saved-array composition, margin-boundary and exact-page checks."""
import hashlib
import json
from math import sqrt
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/completion_conditioning_union_v1'
OUT = ROOT / 'outputs/completion_margin_feather_v1'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def rgb(path):
    with Image.open(path) as im:
        assert im.size == (256, 256); return np.array(im.convert('RGB'))


def binary(path):
    with Image.open(path) as im:
        assert im.size == (256, 256) and im.mode == 'L'; v = np.array(im)
    assert set(np.unique(v)) <= {0, 255}; return v != 0


def independent_weights(core, final):
    # Enumerate only integer offsets in the existing radius2 Euclidean disk.
    # Sliced shifts avoid wrapped pixels at image boundaries.
    best = np.full((256, 256), np.inf, np.float64)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx * dx + dy * dy > 4: continue
            y0, y1, x0, x1 = max(0, dy), min(256, 256 + dy), max(0, dx), min(256, 256 + dx)
            near = core[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            view = best[y0:y1, x0:x1]
            view[near] = np.minimum(view[near], sqrt(dx * dx + dy * dy))
    assert np.isfinite(best[final]).all() and (best[final] <= 2).all()
    weight = np.zeros((256, 256), np.float32)
    weight[final] = np.float32(1.) - best[final].astype(np.float32) / np.float32(3.)
    return weight


def boundary(output, final, ring):
    total, count = 0., 0
    for a, b in [((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                 ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))]:
        support = (final[a] != final[b]) & (ring[a] | ring[b])
        values = np.abs(output[a].astype(np.int16) - output[b].astype(np.int16)).astype(np.float64).mean(axis=-1)
        total += float(values[support].sum()); count += int(support.sum())
    return {'edge_pairs':count, 'mean_abs_channel_jump_255':total / count if count else None,
            'processing_continuity_only_not_quality':True}


def main():
    started = time.monotonic(); p, r = read(OUT / 'protocol.json'), read(OUT / 'results.json')
    assert r['complete'] and r['protocol_sha256'] == sha(OUT / 'protocol.json') and r['seconds'] < p['cap_seconds'] == 180
    for name, digest in {**p['sources_sha256'], **p['app_preservation_sha256']}.items(): assert sha(ROOT / name) == digest, name
    for name, digest in r['artifacts_sha256'].items(): assert sha(OUT / name) == digest, name
    parent = read(OLD / 'protocol.json'); old_results = read(OLD / 'results.json')
    assert p['cases'] == parent['cases'] and p['quality_criteria'] == parent['quality_criteria']
    assert p['parent_protocol_sha256'] == sha(OLD / 'protocol.json') and p['parent_results_sha256'] == sha(OLD / 'results.json')
    assert old_results['state_before'] == old_results['state_after'] and read(OLD / 'independent_saved_output_audit.json')['complete']
    assert sum((OUT / name).stat().st_size for name in r['artifacts_sha256']) == r['artifact_bytes'] <= p['artifact_cap_bytes']
    assert [c['id'] for c in p['cases']] == [c['id'] for c in r['rows']]
    mapping = {c['id']:c for c in p['cases']}; exact_core = exact_visible = exact_protected = changed_ring = 0
    outputs, bypasses, exclusions, decreased = 0, 0, 0, 0
    for row in r['rows']:
        c = mapping[row['id']]; assert c['rejected'] == row['rejected_before_processing']
        if c['rejected']:
            assert row['input_review'] == c['input_review'] and not (OUT / 'images' / (c['id'] + '.png')).exists()
            assert not (OUT / 'weights' / (c['id'] + '.npy')).exists(); exclusions += 1; continue
        source, old, actual = rgb(ROOT / c['input']), rgb(OLD / 'images' / (c['id'] + '.png')), rgb(OUT / 'images' / (c['id'] + '.png'))
        core, final, protected, face = [binary(ROOT / c['masks'][key]) for key in ['core', 'removal', 'protected', 'face']]
        assert not (core & ~final).any() and not (final & ~face).any() and not (final & protected).any()
        if final.any():
            with np.load(OLD / 'stages' / (c['id'] + '.npz'), allow_pickle=False) as stages:
                raw = stages['completion']; assert raw.dtype == np.float32 and raw.shape == (1, 3, 256, 256)
                assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
                values = raw[0].transpose(1, 2, 0)
                expected_old = source.copy(); expected_old[final] = np.floor(values[final] * np.float32(255)).astype(np.uint8)
            colour = read(OLD / 'metadata' / (c['id'] + '.json'))['display_processing']['colour_policy']
            if colour['applied']:
                gray = cv2.cvtColor(expected_old, cv2.COLOR_RGB2GRAY)
                expected_old[final] = np.repeat(gray[..., None], 3, axis=-1)[final]
            np.testing.assert_array_equal(old, expected_old)
        else:
            with np.load(OLD / 'stages' / (c['id'] + '.npz'), allow_pickle=False) as empty: assert not empty.files
            np.testing.assert_array_equal(old, source); bypasses += 1
        weight = independent_weights(core, final)
        np.testing.assert_array_equal(np.load(OUT / 'weights' / (c['id'] + '.npy'), allow_pickle=False), weight)
        mixed = old.astype(np.float32) * weight[..., None] + source.astype(np.float32) * (np.float32(1.) - weight[..., None])
        expected = np.floor(mixed + np.float32(.5)).astype(np.uint8)
        expected[~final] = source[~final]; expected[core] = old[core]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[core], old[core]); np.testing.assert_array_equal(actual[~final], source[~final])
        np.testing.assert_array_equal(actual[protected], source[protected])
        ring = final & ~core; change = np.any(actual != old, axis=-1)
        assert not (change & ~ring).any() and int(change.sum()) == row['changed_pixels']
        assert row['core_pixels'] == int(core.sum()) and row['margin_pixels'] == int(ring.sum())
        assert row['old_boundary_jump'] == boundary(old, final, ring) and row['new_boundary_jump'] == boundary(actual, final, ring)
        old_jump, new_jump = [row[key]['mean_abs_channel_jump_255'] for key in ['old_boundary_jump', 'new_boundary_jump']]
        if old_jump is not None: decreased += int(new_jump < old_jump)
        exact_core += int(core.sum()) * 3; exact_visible += int((~final).sum()) * 3; exact_protected += int(protected.sum()) * 3
        changed_ring += int(change.sum()); outputs += 1
        assert time.monotonic() - started < 180
    assert (outputs, bypasses, exclusions) == (32, 4, 4)
    page_ids, cells = [], 0
    assert len(r['pages']) == 8
    for page in r['pages']:
        assert sha(OUT / page['path']) == page['sha256']
        with Image.open(OUT / page['path']) as im: image = np.array(im.convert('RGB')); assert im.size == (1280, 1214)
        for entry in page['entries']:
            c = mapping[entry['id']]; source = rgb(ROOT / c['input']); core, final = [binary(ROOT / c['masks'][key]) for key in ['core', 'removal']]
            over = source.astype(np.float64); over[core] = .55 * over[core] + .45 * np.array([16, 185, 129])
            over[final & ~core] = .55 * over[final & ~core] + .45 * np.array([240, 140, 32])
            weight = independent_weights(core, final); grey = np.floor(weight * np.float32(255) + np.float32(.5)).astype(np.uint8)
            wanted = [source, np.floor(over + .5).astype(np.uint8), rgb(OLD / 'images' / (c['id'] + '.png')),
                      rgb(OUT / 'images' / (c['id'] + '.png')), np.repeat(grey[..., None], 3, axis=-1)]
            y = 30 + 296 * entry['row']
            for index, value in enumerate(wanted):
                np.testing.assert_array_equal(image[y:y + 256, index * 256:(index + 1) * 256], value); cells += 1
            page_ids.append(c['id'])
    assert len(page_ids) == len(set(page_ids)) == 32 and cells == 160
    from completion_margin_feather_v1 import feather
    source = np.full((256, 256, 3), 60, np.uint8); estimate = np.full_like(source, 180)
    core = np.zeros((256, 256), bool); core[1, 1] = True
    final = np.zeros_like(core)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx * dx + dy * dy <= 4 and 0 <= 1 + dy < 256 and 0 <= 1 + dx < 256: final[1 + dy, 1 + dx] = True
    fixture, w = feather(source, estimate, core, final)
    assert tuple(fixture[1, 1]) == (180, 180, 180) and tuple(fixture[1, 2]) == (140, 140, 140)
    assert tuple(fixture[1, 3]) == (100, 100, 100) and tuple(fixture[2, 2]) == (123, 123, 123)
    assert not w[255].any() and not w[:, 255].any()
    empty = np.zeros_like(core); bypass, w = feather(source, estimate, empty, empty)
    np.testing.assert_array_equal(bypass, source); assert not w.any()
    rejected = 0
    bad_final = final.copy(); bad_final[1, 4] = True
    for args in [(source.astype(np.float32), estimate, core, final), (source, estimate, core, empty),
                 (source, estimate, core, bad_final), (source, estimate, empty, final)]:
        try: feather(*args)
        except ValueError: rejected += 1
        else: raise AssertionError('Invalid composition must stop')
    assert rejected == 4 and not any(name in sys.modules for name in ['torch', 'pretrained_completion', 'dgp_face_workflow_v3'])
    receipt = {'complete':True, 'protocol_sha256':sha(OUT / 'protocol.json'), 'results_sha256':sha(OUT / 'results.json'),
               'checker_sha256':sha(Path(__file__)), 'exact_outputs':outputs, 'exact_bypasses':bypasses, 'prior_input_exclusions_retained':exclusions,
               'core_estimate_bytes_exact':exact_core, 'visible_source_bytes_exact':exact_visible, 'protected_source_bytes_exact':exact_protected,
               'old_raw_to_PNG_compositions_reverified':28, 'changed_pixels_confined_to_existing_margin':changed_ring,
               'exact_page_cells':cells, 'boundary_jump_decreased_cases':decreased, 'boundary_measurement_not_quality':True,
               'known_distance_quantization_and_no_wrap_fixture':True, 'empty_bypass_fixture':True, 'invalid_input_rejections':rejected,
               'source_bindings':len(p['sources_sha256']), 'artifact_bindings':len(r['artifacts_sha256']), 'all14_app_bindings_unchanged':True,
               'model_forwards':0, 'gradient_calls':0, 'optimizer_updates':0, 'app_adoption':False,
               'automatic_quality_qualification':False, 'assisted_quality_qualification':False, 'independent_final_review':False,
               'goal_complete':False, 'seconds':time.monotonic() - started, 'cap_seconds':180}
    assert receipt['seconds'] < 180
    with (OUT / 'independent_saved_output_audit.json').open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key:receipt[key] for key in ['complete', 'exact_outputs', 'core_estimate_bytes_exact', 'boundary_jump_decreased_cases', 'seconds']}))


if __name__ == '__main__': main()

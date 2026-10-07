"""Independent numerical review of saved loss evidence; only frozen CPU recognition."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1_r1'
RETURNED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    started = time.monotonic()
    plan, analysis, p, parent = map(read, [OUT / 'plan.json', OUT / 'analysis.json', BUNDLE / 'protocol.json', PARENT / 'protocol.json'])
    assert analysis['complete'] and analysis['plan_sha256'] == sha(OUT / 'plan.json')
    assert analysis['source_bindings_sha256'] == plan['source_bindings_sha256']
    for name, digest in plan['source_bindings_sha256'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    original_source = ROOT / 'scripts/review_cctv_dgp_post_v32_saved_losses_v1.py'
    assert sha(original_source) == '05f057f6cca63e88128fee7de2a45bbdc8c23dbc19ba8894299523ebe4987000'
    source_tree = ast.parse(original_source.read_text(encoding='utf-8'))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ['grad', 'backward', 'step', 'AdamW', 'SGD'] for n in ast.walk(source_tree))
    assert analysis['analyzer_sha256'] == sha(ROOT / 'scripts/review_cctv_dgp_post_v32_saved_losses_v1_r1.py')
    assert analysis['CPU_recognizer_images'] == plan['maximum_CPU_recognizer_images'] == 100
    assert analysis['DGP_forwards'] == analysis['local_gradient_calls'] == analysis['local_backward_calls'] == analysis['optimizer_updates'] == 0
    assert not analysis['local_training'] and not analysis['native_or_reserved_used']
    assert analysis['seconds'] < plan['budget_seconds'] == 180
    failure = read(ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1/analysis_failure.json')
    assert failure['exception'] == "KeyError: 'reference'" and failure['CPU_recognizer_images'] == failure['optimizer_updates'] == 0
    cases = {c['id']: c for c in parent['cases']}
    assert plan['case_ids'] == p['preview_case_ids'] == [r['id'] for r in analysis['rows']]
    schedule = read(BUNDLE / 'schedule.json')['batches']
    exposed = {p['case_rows'][i]['id'] for batch in schedule[:50] for i in batch}
    assert not set(exposed) & set(plan['case_ids'])
    kernel = np.exp(-.5 * (np.arange(-6, 7, dtype=np.float64) / 2) ** 2); kernel /= kernel.sum()
    luma = np.array([.299, .587, .114], np.float64)
    def high(rgb):
        y = (rgb.astype(np.float64) * luma).sum(2)
        return y - convolve1d(convolve1d(y, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    def score(a, b, mask):
        a, b = a.astype(np.float64), b.astype(np.float64)
        pool = lambda x: uniform_filter(x, size=(7, 7, 1), mode='constant')
        u, v = pool(a), pool(b)
        va, vb = (pool(a * a) - u * u) * (49 / 48), (pool(b * b) - v * v) * (49 / 48)
        cov = (pool(a * b) - u * v) * (49 / 48)
        s = ((2 * u * v + .01 ** 2) * (2 * cov + .03 ** 2)) / ((u * u + v * v + .01 ** 2) * (va + vb + .03 ** 2))
        return float(s[mask].mean())
    items = []
    worst_pixel = worst_feature = worst_ssim = worst_term = 0.
    regions_verified = 0
    normal = read(RETURNED / 'outputs/cohort_loss_setup.json')
    for row in analysis['rows']:
        assert time.monotonic() - started < 180
        c = cases[row['id']]
        assert c['role'] == row['role'] == 'train' and c['profile'] == row['profile'] and c['source'] == row['source']
        with Image.open(PARENT / c['target']) as im: target8 = np.asarray(im.convert('RGB')).copy()
        with Image.open(PARENT / c['observed']) as im: observed = np.asarray(im).copy() > 0
        target = target8.astype(np.float32) / np.float32(255)
        def interior(radius):
            # Independent min filter, rather than the producer's cumulative sum.
            from scipy.ndimage import minimum_filter
            return minimum_filter(observed, size=2 * radius + 1, mode='constant', cval=0)
        core, valid = interior(6), interior(3)
        patches = {}
        combined = np.zeros((256, 256), bool)
        for key, point in zip(['eye0', 'eye1', 'nose', 'mouth0', 'mouth1'], c['landmarks5_canvas_xy'], strict=True):
            x, y = np.floor(point).astype(int)
            patch = np.zeros_like(combined)
            patch[max(0, y - 12):min(256, y + 12), max(0, x - 12):min(256, x + 12)] = True
            patches[key] = patch & core
            combined |= patches[key]
        patches['outside_five_patches_in_observed_interior'] = core & ~combined
        before = np.load(RETURNED / 'outputs/initial_baseline' / (c['id'] + '.npy'), allow_pickle=False)
        after = np.load(RETURNED / 'outputs/update50' / (c['id'] + '.npy'), allow_pickle=False)
        h0, h1 = high(before) - high(target), high(after) - high(target)
        movement = high(after) - high(before)
        delta = after.astype(np.float64) - before.astype(np.float64)
        for key, mask in {'feature': combined, 'interior': core, **patches}.items():
            item = row['raw_numpy_float64'][key] if key in ['feature', 'interior'] else row['raw_numpy_float64']['regions'][key]
            a, b = float(np.square(h0[mask]).mean()), float(np.square(h1[mask]).mean())
            dot, squared = float((h0[mask] * movement[mask]).mean()), float(np.square(movement[mask]).mean())
            for measured, saved in [(a, item['baseline_MSE']), (b, item['stopped50_MSE']), (dot, item['error_dot_output_movement']), (squared, item['movement_MSE'])]:
                assert abs(measured - saved) < 1e-12
            assert abs(1 - b / a - item['relative_gain']) < 1e-12
            assert abs((a + 2 * dot + squared) - b) < 1e-12
            cosine = -dot / max(np.sqrt(a * squared), 1e-30)
            assert abs(cosine - item['cosine_with_negative_baseline_error']) < 1e-12
            regions_verified += 1
        assert np.allclose(delta[observed].mean(0), row['raw_numpy_float64']['postclip_mean_RGB_shift'], atol=1e-15, rtol=0)
        for update, rgb in [(0, before), (50, after)]:
            replay = row['training_float32_CPU_replay'][str(update)]
            pixel = float(np.square(rgb.astype(np.float64) - target.astype(np.float64))[observed].mean())
            feature = float(np.square((h0 if update == 0 else h1)[combined]).mean())
            inner = float(np.square((h0 if update == 0 else h1)[core]).mean())
            ssim = score(rgb, target, valid)
            worst_pixel = max(worst_pixel, abs(pixel - replay['pixel_MSE']))
            worst_feature = max(worst_feature, abs(feature - replay['feature_MSE']), abs(inner - replay['interior_MSE']))
            worst_ssim = max(worst_ssim, abs(ssim - replay['SSIM']))
            assert abs(pixel - replay['pixel_MSE']) < 1e-8 and abs(feature - replay['feature_MSE']) < 1e-8 and abs(inner - replay['interior_MSE']) < 1e-8
            assert abs(ssim - replay['SSIM']) < 1e-4, 'Independent float64 SSIM comparison, not a gate tolerance'
            with Image.open(RETURNED / ('outputs/update' + str(update)) / (c['id'] + '.png')) as im: png = np.asarray(im.convert('RGB'))
            png_high = high(png.astype(np.float64) / 255) - high(target8.astype(np.float64) / 255)
            assert abs(np.square(png_high[combined]).mean() - row['delivered_PNG'][str(update)]['landmark_high_frequency_MSE']) < 1e-12
        items.append((row, c, before, after, observed))
    import torch
    sys.path.insert(0, str(PARENT)); sys.path.insert(0, str(ROOT))
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    torch.set_num_threads(4)
    identity = FixedObservedIdentity(PARENT / 'weights/w600k_r50.onnx', 'cpu')
    initial_state = state_hash(identity)
    assert initial_state == analysis['identity_state_before_after'] == p['frozen_recognizer_state']
    references = {r['id']: r for r in parent['references']}
    count, worst_identity = 0, 0.
    with torch.no_grad():
        for begin in range(0, 50, 5):
            assert time.monotonic() - started < 180
            batch = items[begin:begin + 5]
            a = torch.from_numpy(np.stack([i[2] for i in batch])).permute(0, 3, 1, 2)
            b = torch.from_numpy(np.stack([i[3] for i in batch])).permute(0, 3, 1, 2)
            mask = torch.from_numpy(np.stack([i[4] for i in batch]).astype(np.float32))[:, None]
            grid = torch.from_numpy(np.stack([grid112(references[i[1]['source_person_or_reference']]['matrix112']) for i in batch]))
            vectors = identity.embedding(torch.cat([a, b]), torch.cat([mask, mask]), torch.cat([grid, grid]))
            truth = torch.from_numpy(np.stack([np.load(RETURNED / 'outputs/initial_baseline' / (i[1]['id'] + '_target_embedding.npy'), allow_pickle=False) for i in batch]))
            cosines = [(vectors[:5] * truth).sum(1), (vectors[5:] * truth).sum(1)]
            for slot, (row, c, before, after, observed) in enumerate(batch):
                original = row['training_float32_CPU_replay']['0']
                for update, cos in zip([0, 50], cosines, strict=True):
                    q = row['training_float32_CPU_replay'][str(update)]
                    worst_identity = max(worst_identity, abs(float(cos[slot]) - q['ArcFace']))
                    assert abs(float(cos[slot]) - q['ArcFace']) < 1e-6
                    degraded = c['profile'] != 'clear'
                    anchor = float(np.square((before if update == 0 else after).astype(np.float64) - before.astype(np.float64))[observed].mean())
                    expected = {'degraded_landmark_detail': 1.25 * q['feature_MSE'] / normal['feature_normalizer'] if degraded else 0.,
                                'degraded_observed_detail': .3125 * q['interior_MSE'] / normal['interior_normalizer'] if degraded else 0.,
                                'degraded_pixel': .0625 * q['pixel_MSE'] / max(original['pixel_MSE'], 1e-5) if degraded else 0.,
                                'clear_baseline_anchor': .05 * anchor / max(original['pixel_MSE'], 1e-5) if not degraded else 0.,
                                'pixel_regression': 2 * max(0., (q['pixel_MSE'] - original['pixel_MSE']) / max(original['pixel_MSE'], 1e-5)),
                                'SSIM_regression': 5 * max(0., original['SSIM'] - q['SSIM']),
                                'ArcFace_regression': 5 * max(0., float(cosines[0][slot]) - float(cos[slot]))}
                    assert list(expected) == p['terms']
                    for term, value in expected.items():
                        worst_term = max(worst_term, abs(value - q['terms'][term]))
                        assert abs(value - q['terms'][term]) < 5e-7
            count += 10
    assert count == 100 and state_hash(identity) == initial_state
    assert all(not value.requires_grad and value.grad is None for value in identity.parameters())
    for name, group in analysis['groups'].items():
        records = [r for r in analysis['rows'] if name == 'all' or name == ('clear' if r['profile'] == 'clear' else 'degraded') or name == r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded')]
        assert len(records) == group['cases']
        for update in [0, 50]:
            for term in p['terms']:
                assert abs(float(np.mean([r['training_float32_CPU_replay'][str(update)]['terms'][term] for r in records])) - group['training_terms_mean'][str(update)][term]) < 1e-12
        for term in p['terms'][3:]:
            assert sum(r['training_float32_CPU_replay']['50']['terms'][term] > 1e-7 for r in records) == group['active_preservation_terms_at50'][term]
        for key, field in [('raw_feature_gain', 'raw_numpy_float64'), ('PNG_feature_gain', 'delivered_PNG')]:
            if field == 'raw_numpy_float64':
                before = sum(r[field]['feature']['baseline_MSE'] for r in records)
                after = sum(r[field]['feature']['stopped50_MSE'] for r in records)
            else:
                before = sum(r[field]['0']['landmark_high_frequency_MSE'] for r in records)
                after = sum(r[field]['50']['landmark_high_frequency_MSE'] for r in records)
            assert abs(1 - after / before - group[key]) < 1e-12
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'analysis_sha256': sha(OUT / 'analysis.json'),
               'plan_sha256': sha(OUT / 'plan.json'), 'bindings_verified': len(plan['source_bindings_sha256']),
               'cases_verified': 50, 'independent_raw_feature_region_quantities_verified': regions_verified,
               'independent_raw_recognizer_images': count, 'maximum_pixel_MSE_difference': worst_pixel,
               'maximum_high_pass_MSE_difference': worst_feature, 'maximum_float64_SSIM_difference': worst_ssim,
               'maximum_identity_cosine_difference': worst_identity, 'maximum_term_assembly_difference': worst_term,
               'first_preparation_schema_failure_preserved': True, 'fixed50_not_optimized_by50': True,
               'DGP_forwards': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'native_or_reserved_used': False,
               'causal_gradient_or_optimizer_history_proof': False, 'PNG_gate_thresholds_unchanged': True,
               'goal_complete': False, 'seconds': time.monotonic() - started}
    assert receipt['seconds'] < 180
    with (OUT / 'independent_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

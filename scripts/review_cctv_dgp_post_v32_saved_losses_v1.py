"""Finite forward-only loss review of already audited R2 TRAIN arrays; no fitting."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def masks(support, landmarks):
    import numpy as np
    def erode(radius):
        size = 2 * radius + 1
        padded = np.pad(support.astype(np.int64), radius)
        summed = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return summed[size:, size:] - summed[:-size, size:] - summed[size:, :-size] + summed[:-size, :-size] == size * size
    interior = erode(6)
    patches = {}
    feature = np.zeros((256, 256), bool)
    for name, point in zip(['eye0', 'eye1', 'nose', 'mouth0', 'mouth1'], landmarks, strict=True):
        x, y = np.floor(point).astype(int)
        patch = np.zeros_like(feature)
        patch[max(0, y - 12):min(256, y + 12), max(0, x - 12):min(256, x + 12)] = True
        patch &= interior
        assert patch.any()
        patches[name] = patch
        feature |= patch
    patches['outside_five_patches_in_observed_interior'] = interior & ~feature
    return feature, interior, erode(3), patches


def numpy_measure(base, pred, target, support, feature, interior, patches):
    import numpy as np
    from scipy.ndimage import convolve1d
    kernel = np.exp(-.5 * (np.arange(-6, 7, dtype=np.float64) / 2) ** 2)
    kernel /= kernel.sum()
    luma = np.array([.299, .587, .114], np.float64)
    def high(value):
        y = (value.astype(np.float64) * luma).sum(2)
        return y - convolve1d(convolve1d(y, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    error, movement = high(base) - high(target), high(pred) - high(base)
    def component(mask):
        before, delta = error[mask], movement[mask]
        initial = float(np.square(before).mean())
        dot = float((before * delta).mean())
        square = float(np.square(delta).mean())
        final = float(np.square(before + delta).mean())
        cosine = -dot / max(float(np.sqrt(initial * square)), 1e-30)
        return {'baseline_MSE': initial, 'stopped50_MSE': final, 'relative_gain': 1 - final / initial,
                'error_dot_output_movement': dot, 'movement_MSE': square,
                'cosine_with_negative_baseline_error': cosine}
    delta = pred.astype(np.float64) - base.astype(np.float64)
    baseline_pixel = float(np.square(base.astype(np.float64) - target.astype(np.float64))[support].mean())
    stopped_pixel = float(np.square(pred.astype(np.float64) - target.astype(np.float64))[support].mean())
    return {'feature': component(feature), 'interior': component(interior),
            'regions': {name: component(mask) for name, mask in patches.items()},
            'baseline_pixel_MSE': baseline_pixel, 'stopped50_pixel_MSE': stopped_pixel,
            'output_change_RMS': float(np.sqrt(np.square(delta[support]).mean())),
            'postclip_mean_RGB_shift': delta[support].mean(0).tolist(),
            'baseline_output_boundary_channel_fraction': float(((base[support] == 0) | (base[support] == 1)).mean()),
            'stopped_output_boundary_channel_fraction': float(((pred[support] == 0) | (pred[support] == 1)).mean()),
            'unchanged_delivered_RGB_channel_fraction': float((np.floor(base[support] * np.float32(255)) == np.floor(pred[support] * np.float32(255))).mean())}


def main():
    started = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json'
    import_path = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_import.json'
    audit, imported, p, base_p = map(read, [audit_path, import_path, BUNDLE / 'protocol.json', PARENT / 'protocol.json'])
    assert audit['complete'] and audit['VM_failure_retained'] and audit['protocol_sha256'] == sha(BUNDLE / 'protocol.json')
    assert not audit['necessary_capacity_pass'] and len(p['preview_case_ids']) == 50
    cases = {c['id']: c for c in base_p['cases']}
    rows = [cases[cid] for cid in p['preview_case_ids']]
    assert len(rows) == 50 and {c['role'] for c in rows} == {'train'}
    setup_path = RETURNED / 'outputs/cohort_loss_setup.json'
    setup = read(setup_path)
    assert setup['cases'] == 50 and setup['degraded_cases'] == 40 and setup['clear_controls'] == 10
    files = [Path(__file__), audit_path, import_path, BUNDLE / 'protocol.json', PARENT / 'protocol.json', setup_path,
             RETURNED / 'outputs/gradient_summary.json', RETURNED / 'outputs/update0/metrics.json', RETURNED / 'outputs/update50/metrics.json',
             PARENT / 'cctv_dgp_pilot.py', PARENT / 'cctv_dgp_batchmatched_identity_v26.py',
             PARENT / 'cctv_dgp_degraded_objective_v24.py', PARENT / 'weights/w600k_r50.onnx', ROOT / 'models/identity_loss.py']
    for c in rows:
        files.extend(PARENT / c[name] for name in ['input', 'target', 'observed'])
        files.extend(RETURNED / 'outputs' / folder / (c['id'] + suffix)
                     for folder, suffixes in [('initial_baseline', ['.npy', '_target_embedding.npy']), ('update0', ['.npy', '.png']), ('update50', ['.npy', '.png'])]
                     for suffix in suffixes)
    bindings = {}
    for path in files:
        digest = sha(path)
        bindings[path.relative_to(ROOT).as_posix()] = digest
        if path.is_relative_to(RETURNED):
            assert digest == imported['files_sha256'][path.relative_to(RETURNED).as_posix()]
        elif path.is_relative_to(PARENT) and path.relative_to(PARENT).as_posix() in base_p['assets_sha256']:
            assert digest == base_p['assets_sha256'][path.relative_to(PARENT).as_posix()]
    assert not OUT.exists()
    OUT.mkdir()
    plan = {'case_ids': [c['id'] for c in rows], 'roles': ['train'], 'source_bindings_sha256': bindings,
            'budget_seconds': 180, 'maximum_CPU_recognizer_images': 100, 'DGP_forwards': 0,
            'local_gradient_calls': 0, 'local_backward_calls': 0, 'optimizer_updates': 0,
            'native_or_reserved_used': False, 'gate_changes': False, 'app_changes': False,
            'question': 'Which raw losses activate at the audited stop; how do raw/PNG structure, regions and finite output movement differ?',
            'limits': 'Fixed50 preflight/normalization TRAIN previews; none optimized by50. Forward quantities do not determine gradient conflict or unsaved AdamW history.'}
    write(OUT / 'plan.json', plan)
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, str(PARENT))
    sys.path.insert(0, str(ROOT))
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from cctv_dgp_degraded_objective_v24 import assemble_terms
    torch.set_num_threads(4)
    identity = FixedObservedIdentity(PARENT / 'weights/w600k_r50.onnx', 'cpu')
    identity_before = state_hash(identity)
    assert identity_before == p['frozen_recognizer_state']
    references = {r['id']: r for r in base_p['references']}
    def tensor(a):
        return torch.from_numpy(np.asarray(a).copy()).permute(2, 0, 1)[None]
    def mean(error, mask):
        return (error * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * error.shape[1]).clamp_min(1)
    z = torch.arange(-6, 7, dtype=torch.float32)
    kernel = torch.exp(-.5 * (z / 2).square()); kernel /= kernel.sum()
    indices = torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1)))
    def high(value):
        v = F.conv2d(value.index_select(2, indices), kernel.view(1, 1, 13, 1))
        v = F.conv2d(v.index_select(3, indices), kernel.view(1, 1, 1, 13))
        return value - v
    def ssim(value, target, valid):
        pool = lambda a: F.avg_pool2d(a, 7, 1, 3)
        u, v = pool(value), pool(target)
        va, vb = (pool(value.square()) - u.square()) * (49 / 48), (pool(target.square()) - v.square()) * (49 / 48)
        covariance = (pool(value * target) - u * v) * (49 / 48)
        return mean(((2 * u * v + .01 ** 2) * (2 * covariance + .03 ** 2)) /
                    ((u.square() + v.square() + .01 ** 2) * (va + vb + .03 ** 2)), valid)
    setup_rows = {r['id']: r for r in setup['rows']}
    saved = {update: {r['id']: r for r in read(RETURNED / ('outputs/update' + str(update) + '/metrics.json'))['rows']} for update in [0, 50]}
    items = []
    for c in rows:
        assert time.monotonic() - started < plan['budget_seconds']
        with Image.open(PARENT / c['target']) as im:
            target8 = np.asarray(im.convert('RGB')).copy()
        with Image.open(PARENT / c['observed']) as im:
            support = np.asarray(im).copy() > 0
        feature, interior, valid7, patches = masks(support, c['landmarks5_canvas_xy'])
        original = np.load(RETURNED / 'outputs/initial_baseline' / (c['id'] + '.npy'), allow_pickle=False)
        stopped = np.load(RETURNED / 'outputs/update50' / (c['id'] + '.npy'), allow_pickle=False)
        target = target8.astype(np.float32) / np.float32(255)
        assert original.shape == stopped.shape == target.shape == (256, 256, 3) and original.dtype == stopped.dtype == np.float32
        assert np.array_equal(original, np.load(RETURNED / 'outputs/update0' / (c['id'] + '.npy'), allow_pickle=False))
        for value in [original, stopped]:
            assert np.isfinite(value).all() and (value >= 0).all() and (value <= 1).all()
        measure = numpy_measure(original, stopped, target, support, feature, interior, patches)
        for update, raw in [(0, original), (50, stopped)]:
            with Image.open(RETURNED / ('outputs/update' + str(update)) / (c['id'] + '.png')) as im:
                png = np.asarray(im.convert('RGB')).copy()
            # Same fixed high-pass and feature area, evaluated on delivered values.
            metric = numpy_measure(png.astype(np.float64) / 255, png.astype(np.float64) / 255,
                                   target8.astype(np.float64) / 255, support, feature, interior, patches)
            assert abs(metric['feature']['baseline_MSE'] - saved[update][c['id']]['metrics']['landmark_high_frequency_MSE']) < 1e-10
        m = lambda a: torch.from_numpy(a.astype(np.float32))[None, None]
        items.append({'case': c, 'measure': measure, 'base': tensor(original), 'pred': tensor(stopped), 'target': tensor(target),
                      'mask': m(support), 'feature': m(feature), 'interior': m(interior), 'valid7': m(valid7),
                      'grid': torch.from_numpy(grid112(references[c['reference']]['matrix112']))[None],
                      'truth': torch.from_numpy(np.load(RETURNED / 'outputs/initial_baseline' / (c['id'] + '_target_embedding.npy'), allow_pickle=False))[None]})
    normalizers = tuple(torch.tensor(setup[name], dtype=torch.float32) for name in ['feature_normalizer', 'interior_normalizer'])
    records = []
    identity_images = 0
    with torch.no_grad():
        for begin in range(0, 50, 5):
            assert time.monotonic() - started < plan['budget_seconds']
            group = items[begin:begin + 5]
            b = {name: torch.cat([item[name] for item in group]) for name in ['base', 'pred', 'target', 'mask', 'feature', 'interior', 'valid7', 'grid', 'truth']}
            vectors = identity.embedding(torch.cat((b['base'], b['pred']), 0), torch.cat((b['mask'], b['mask']), 0), torch.cat((b['grid'], b['grid']), 0))
            identity_images += 10
            baseline_cos, stopped_cos = (vectors[:5] * b['truth']).sum(1), (vectors[5:] * b['truth']).sum(1)
            baseline_pixel = mean((b['base'] - b['target']).square(), b['mask'])
            baseline_score = ssim(b['base'], b['target'], b['valid7'])
            per_update = {}
            for update, value, cosine in [(0, b['base'], baseline_cos), (50, b['pred'], stopped_cos)]:
                weights = value.new_tensor([.299, .587, .114])[None, :, None, None]
                delta = high((value * weights).sum(1, keepdim=True)) - high((b['target'] * weights).sum(1, keepdim=True))
                feature, interior = mean(delta.square(), b['feature']), mean(delta.square(), b['interior'])
                pixel = mean((value - b['target']).square(), b['mask'])
                score = ssim(value, b['target'], b['valid7'])
                anchor = mean((value - b['base']).square(), b['mask'])
                clear = value.new_tensor([float(i['case']['profile'] == 'clear') for i in group])
                terms = assemble_terms(feature, interior, pixel, baseline_pixel, score, baseline_score, cosine, baseline_cos,
                                       anchor, 1.25 * (1 - clear), clear, normalizers)
                assert list(terms) == p['terms']
                per_update[update] = {'terms': terms, 'feature': feature, 'interior': interior, 'pixel': pixel, 'SSIM': score, 'ArcFace': cosine, 'anchor': anchor}
            for slot, item in enumerate(group):
                cid = item['case']['id']
                zero = per_update[0]
                assert abs(float(zero['feature'][slot]) - setup_rows[cid]['feature_MSE']) < 3e-9
                assert abs(float(zero['interior'][slot]) - setup_rows[cid]['interior_MSE']) < 3e-9
                record = {'id': cid, 'source': item['case']['source'], 'profile': item['case']['profile'], 'role': 'train',
                          'raw_numpy_float64': item['measure'], 'delivered_PNG': {str(u): saved[u][cid]['metrics'] for u in [0, 50]},
                          'training_float32_CPU_replay': {str(u): {'terms': {k: float(v[slot]) for k, v in q['terms'].items()},
                                                                'feature_MSE': float(q['feature'][slot]), 'interior_MSE': float(q['interior'][slot]),
                                                                'pixel_MSE': float(q['pixel'][slot]), 'SSIM': float(q['SSIM'][slot]),
                                                                'ArcFace': float(q['ArcFace'][slot]), 'baseline_anchor_MSE': float(q['anchor'][slot])}
                                                        for u, q in per_update.items()}}
                records.append(record)
            print(json.dumps({'reviewed_cases': begin + 5, 'of': 50, 'CPU_recognizer_images': identity_images}), flush=True)
    assert identity_images == 100 and state_hash(identity) == identity_before
    assert all(not value.requires_grad and value.grad is None for value in identity.parameters())
    groups = {}
    for name in ['all', 'clear', 'degraded'] + sorted({c['source'] + '/' + k for c in rows for k in ['clear', 'degraded']}):
        selected = [r for r in records if name == 'all' or name == ('clear' if r['profile'] == 'clear' else 'degraded') or name == r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded')]
        groups[name] = {'cases': len(selected), 'raw_feature_gain': 1 - sum(r['raw_numpy_float64']['feature']['stopped50_MSE'] for r in selected) / sum(r['raw_numpy_float64']['feature']['baseline_MSE'] for r in selected),
                        'PNG_feature_gain': 1 - sum(r['delivered_PNG']['50']['landmark_high_frequency_MSE'] for r in selected) / sum(r['delivered_PNG']['0']['landmark_high_frequency_MSE'] for r in selected),
                        'raw_regions': {region: {'relative_gain': 1 - sum(r['raw_numpy_float64']['regions'][region]['stopped50_MSE'] for r in selected) / sum(r['raw_numpy_float64']['regions'][region]['baseline_MSE'] for r in selected)} for region in records[0]['raw_numpy_float64']['regions']},
                        'training_terms_mean': {str(u): {term: float(np.mean([r['training_float32_CPU_replay'][str(u)]['terms'][term] for r in selected])) for term in p['terms']} for u in [0, 50]},
                        'active_preservation_terms_at50': {term: sum(r['training_float32_CPU_replay']['50']['terms'][term] > 1e-7 for r in selected) for term in p['terms'][3:]},
                        'output_change_RMS_mean': float(np.mean([r['raw_numpy_float64']['output_change_RMS'] for r in selected])),
                        'output_boundary_fraction_mean': float(np.mean([r['raw_numpy_float64']['stopped_output_boundary_channel_fraction'] for r in selected]))}
    assert all(groups['all']['training_terms_mean']['0'][t] == 0 for t in p['terms'][3:])
    summary = read(RETURNED / 'outputs/gradient_summary.json')
    for term, value in zip(p['terms'], summary['component_values'], strict=True):
        assert abs(groups['all']['training_terms_mean']['0'][term] - value) < 1e-7
    result = {'complete': True, 'analyzer_sha256': sha(Path(__file__)), 'plan_sha256': sha(OUT / 'plan.json'),
              'source_bindings_sha256': bindings, 'rows': records, 'groups': groups,
              'all50_fixed_PREVIEW_TRAIN_not_optimized_by50': True, 'identity_state_before_after': identity_before,
              'CPU_recognizer_images': identity_images, 'DGP_forwards': 0, 'local_gradient_calls': 0, 'local_backward_calls': 0,
              'optimizer_updates': 0, 'local_training': False, 'native_or_reserved_used': False,
              'raw_boundary_fraction_is_not_preclip_saturation_or_gradient_proof': True,
              'AdamW_history_reconstructed': False, 'full3905_raw_loss_replay': False,
              'early_one_percent_failure_preserved': True, 'app_promotion': False, 'independent_final_review': False,
              'goal_complete': False, 'seconds': time.monotonic() - started}
    assert result['seconds'] < plan['budget_seconds']
    write(OUT / 'analysis.json', result)
    print(json.dumps({'complete': True, 'groups': groups, 'CPU_recognizer_images': identity_images, 'optimizer_updates': 0, 'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    main()

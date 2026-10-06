"""V24 training objective only; labels never enter the restoration forward path."""
import torch
from torch.nn import functional as F


def cohort_normalizers(items, head):
    """Frozen baseline cohort scalars; VM runtime owns this fixed-filter preflight."""
    rows = []
    with torch.no_grad():
        for item in items:
            weights = item['base'].new_tensor([.299, .587, .114])[None, :, None, None]
            delta = head.high((item['base'] * weights).sum(1, keepdim=True)) - head.high((item['target'] * weights).sum(1, keepdim=True))
            feature = (delta.square() * item['feature']).sum() / item['feature'].sum()
            interior = (delta.square() * item['interior']).sum() / item['interior'].sum()
            clear = item['case']['profile'] == 'clear'
            item['degraded_weight'] = item['base'].new_tensor([0. if clear else 1.25])
            item['clear_weight'] = item['base'].new_tensor([1. if clear else 0.])
            rows.append({'id': item['case']['id'], 'clear': clear, 'feature_MSE': float(feature), 'interior_MSE': float(interior)})
        degraded = [r for r in rows if not r['clear']]
        assert len(rows) == 50 and len(degraded) == 40 and all(r['feature_MSE'] > 0 and r['interior_MSE'] > 0 for r in rows)
        feature = items[0]['base'].new_tensor(sum(r['feature_MSE'] for r in degraded) / 40).clamp_min(1e-6)
        interior = items[0]['base'].new_tensor(sum(r['interior_MSE'] for r in degraded) / 40).clamp_min(1e-6)
    assert not feature.requires_grad and not interior.requires_grad
    return (feature, interior), {'complete': True, 'cases': 50, 'clear_controls': 10, 'degraded_cases': 40,
        'feature_normalizer': float(feature), 'interior_normalizer': float(interior), 'normalizer_floor': 1e-6,
        'clear_reward': False, 'degraded_weight': 1.25, 'clear_baseline_anchor_weight': .05,
        'fixed_high_pass_calls': 100, 'optimizer_constructed': False, 'rows': rows}


def assemble_terms(feature, interior, pixel, base_pixel, score, base_score, cosine, base_cosine,
                   baseline_anchor, degraded_weight, clear_weight, normalizers):
    """Pure term assembly, separately contract-checked without models or gradients."""
    assert feature.shape == interior.shape == pixel.shape == base_pixel.shape == score.shape == base_score.shape == cosine.shape == base_cosine.shape
    assert degraded_weight.shape == clear_weight.shape == feature.shape
    assert torch.all((degraded_weight == 0) | (degraded_weight == 1.25)) and torch.all((clear_weight == 0) | (clear_weight == 1))
    assert torch.equal(degraded_weight / 1.25 + clear_weight, torch.ones_like(clear_weight))
    normalizer_feature, normalizer_interior = normalizers
    return {
        'degraded_landmark_detail': degraded_weight * feature / normalizer_feature,
        'degraded_observed_detail': degraded_weight * .25 * interior / normalizer_interior,
        'degraded_pixel': degraded_weight * .05 * pixel / base_pixel.clamp_min(1e-5),
        'clear_baseline_anchor': clear_weight * .05 * baseline_anchor / base_pixel.clamp_min(1e-5),
        'pixel_regression': 2 * F.relu((pixel - base_pixel) / base_pixel.clamp_min(1e-5)),
        'SSIM_regression': 5 * F.relu(base_score - score),
        'ArcFace_regression': 5 * F.relu(base_cosine - cosine),
    }


def objective_terms(b, pred, identity, mean, feature_errors, ssim, normalizers):
    feature, interior = feature_errors(pred, b['target'], b['feature'], b['interior'])
    pixel = mean((pred - b['target']).square(), b['mask'])
    base_pixel = mean((b['base'] - b['target']).square(), b['mask'])
    cosine = (identity.embedding(pred * b['mask'] + b['x'] * (1 - b['mask']), b['mask'], b['grid']) * b['truth']).sum(1)
    score = ssim(pred, b['target'], b['valid7']); base_score = ssim(b['base'], b['target'], b['valid7'])
    anchor = mean((pred - b['base']).square(), b['mask'])
    return assemble_terms(feature, interior, pixel, base_pixel, score, base_score, cosine, b['base_cosine'], anchor,
                          b['degraded_weight'], b['clear_weight'], normalizers)

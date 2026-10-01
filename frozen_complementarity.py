"""Two frozen masks, fixed compositions and explicitly label-informed bounds.

No router, optimizer, training, deployment or validation selection is implemented.
"""
from collections import defaultdict
import hashlib
import numpy as np
import torch

COUNT_FIELDS = ('tp', 'fp', 'fn', 'visible', 'covered_cases', 'empty_mask_cases',
                'negative_cases', 'negative_false_positive_cases', 'ignored_positive_pixels')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matching_binary(*arrays):
    require(arrays and all(isinstance(a, np.ndarray) and a.dtype == np.bool_ and a.ndim == 2
                           and a.size > 0 and a.shape == arrays[0].shape for a in arrays),
            'Matching binary image-sized masks required')


def counts(prediction, target, valid):
    matching_binary(prediction, target, valid)
    require(valid.any() and not (target & ~valid).any(), 'Known target lies outside supervision support')
    has = bool(target.any()); pred = bool((prediction & valid).any())
    return {'tp': int((prediction & target & valid).sum()),
            'fp': int((prediction & ~target & valid).sum()),
            'fn': int((~prediction & target & valid).sum()),
            'visible': int((~target & valid).sum()), 'covered_cases': int(has),
            'empty_mask_cases': int(has and not pred), 'negative_cases': int(not has),
            'negative_false_positive_cases': int(not has and pred),
            'ignored_positive_pixels': int((prediction & ~valid).sum())}


def aggregate(rows):
    require(rows and all(set(r) == set(COUNT_FIELDS) and
            all(type(r[k]) is int and r[k] >= 0 for k in COUNT_FIELDS) for r in rows), 'Complete nonnegative pixel/case counts required')
    totals = {k: sum(r[k] for r in rows) for k in COUNT_FIELDS}
    require(totals['empty_mask_cases'] <= totals['covered_cases']
            and totals['negative_false_positive_cases'] <= totals['negative_cases'], 'Case count exceeds denominator')
    tp, fp, fn = (totals[k] for k in ('tp', 'fp', 'fn'))
    return {**totals, 'cases': len(rows), 'iou': tp/max(1, tp+fp+fn),
            'missed_fraction': fn/max(1, tp+fn), 'visible_false_positive': fp/max(1, totals['visible'])}


def fixed_compositions(parent, candidate):
    """Executable mask compositions: no targets or source IDs accepted."""
    matching_binary(parent, candidate)
    return {'union': parent | candidate, 'intersection': parent & candidate}


def diagnostic_oracles(parent, candidate, target, valid, pool):
    """Optimistic bounds only. Both target and pool access forbid deployment."""
    require(pool in ('real', 'replay', 'reflection'), 'Training-only diagnostic pool required')
    a, b = counts(parent, target, valid), counts(candidate, target, valid)
    dominates = b['tp'] >= a['tp'] and b['fp'] <= a['fp'] and (b['tp'] > a['tp'] or b['fp'] < a['fp'])
    return {'candidate_dominates': dominates,
            'dominance_oracle': (candidate if dominates else parent).copy(),
            'pool_oracle': (parent if pool == 'replay' else candidate).copy(),
            'pixel_oracle': np.where(target, parent | candidate, parent & candidate)}


def conflicting_inputs(items):
    groups = defaultdict(list); conflicts = []
    require(items and len({r['id'] for r in items}) == len(items), 'Unique training case IDs required')
    for row in items:
        rgb = row['rgb']; target, valid = row['target'], row['valid']
        require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (*target.shape, 3), 'Byte RGB geometry differs')
        counts(target, target, valid)
        key = (rgb.shape, hashlib.sha256(rgb.tobytes()).hexdigest())
        for previous in groups[key]:
            common = valid & previous['valid']
            different = (target != previous['target']) & common
            if different.any():
                conflicts.append({'cases': [previous['id'], row['id']], 'input_byte_sha256': key[1],
                                  'common_supervised_pixels': int(common.sum()), 'conflicting_pixels': int(different.sum())})
        groups[key].append(row)
    return conflicts


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        require(tensor.device.type == 'cpu' and torch.isfinite(tensor).all(), 'Finite CPU-only frozen state required')
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_predictions(model, items, *, batch_size=8, progress=True):
    require(type(batch_size) is int and batch_size >= 1 and items
            and len({r['id'] for r in items}) == len(items), 'Unique CPU inference cases and positive batch required')
    model.requires_grad_(False).eval(); before = state_digest(model); predictions = {}
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start:start+batch_size]; arrays = [r['rgb'] for r in batch]
            require(all(isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.ndim == 3
                        and a.shape[2] == 3 and a.shape == arrays[0].shape for a in arrays), 'Matching byte RGB input required')
            x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float()/255
            logits = model.detect(x)
            require(isinstance(logits, torch.Tensor) and logits.device.type == 'cpu'
                    and logits.shape == (len(batch), 1, *arrays[0].shape[:2])
                    and torch.isfinite(logits).all(), 'Finite matching CPU logits required')
            masks = (logits.sigmoid()[:, 0] >= .5).numpy()
            predictions.update({r['id']: m.copy() for r, m in zip(batch, masks)})
            if progress and (start % 64 == 0 or start+len(batch) == len(items)):
                print('Frozen CPU prediction', start+len(batch), '/', len(items), flush=True)
    after = state_digest(model)
    require(after == before and all(p.grad is None for p in model.parameters()), 'Frozen inference changed state or produced parameter gradients')
    return predictions, {'forward_images': len(items), 'state_before_sha256': before,
                         'state_after_sha256': after, 'model_state_unchanged': True,
                         'threshold': .5, 'optimizer_constructed': False, 'optimizer_updates_locally': 0}

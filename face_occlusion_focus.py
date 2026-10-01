"""Target-only component balancing for a fixed detector-loss experiment.

This auxiliary BCE is a project-specific hypothesis, not the published distance
boundary loss. Nothing here starts training, downloads weights or changes labels.
"""
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F

RADIUS = 3
WEIGHT = .25


def require(condition, message):
    if not condition: raise ValueError(message)


def component_weights(target, priority=None):
    require(isinstance(target, np.ndarray) and target.dtype == np.bool_ and target.ndim == 2 and
            min(target.shape) > 0, 'Require a nonempty binary 2D target')
    if priority is None: priority = np.zeros_like(target)
    require(isinstance(priority, np.ndarray) and priority.dtype == np.bool_ and priority.shape == target.shape
            and not np.any(priority & ~target), 'Priority must be a binary subset of the reviewed target')
    regions = []
    # Reviewed reflection additions get their own region even when they touch a
    # mouth-mask strap. The foreground union and all supervised labels stay exact.
    for section in (priority, target & ~priority):
        number, labels = cv2.connectedComponents(np.ascontiguousarray(section, dtype='uint8'), connectivity=8)
        regions.extend(labels == label for label in range(1, number))
    components = len(regions); weights = np.zeros(target.shape, dtype='float32')
    positive_sizes = []; ring_sizes = []
    kernel = np.ones((2 * RADIUS + 1, 2 * RADIUS + 1), dtype='uint8')
    for covered in regions:
        ring = (cv2.dilate(covered.astype('uint8'), kernel) > 0) & ~target
        foreground = int(covered.sum()); visible = int(ring.sum())
        positive_sizes.append(foreground); ring_sizes.append(visible)
        # An all-covered image has no visible ring and keeps unit positive mass.
        weights[covered] += (0.5 if visible else 1.) / (components * foreground)
        if visible: weights[ring] += .5 / (components * visible)
    require(np.isfinite(weights).all() and np.isclose(weights.sum(), float(components > 0), atol=1e-6),
            'Invalid component weight normalization')
    return weights, {'components': components, 'component_pixels': positive_sizes,
                     'ring_pixels': ring_sizes, 'radius': RADIUS,
                     'positive_pixels': int(target.sum()), 'priority_pixels': int(priority.sum()),
                     'visible_pixels': int((~target).sum())}


def focus_loss(logits, masks, weights):
    require(all(isinstance(x, torch.Tensor) and x.ndim == 4 and x.is_floating_point()
                for x in (logits, masks, weights)) and logits.shape == masks.shape == weights.shape and
            logits.shape[1] == 1 and min(logits.shape) > 0 and
            logits.device == masks.device == weights.device, 'Require matching floating N1HW tensors')
    require(all(torch.isfinite(x).all() for x in (logits, masks, weights)) and
            ((masks == 0) | (masks == 1)).all() and (weights >= 0).all(), 'Nonfinite logits or invalid target/map')
    positive = masks.flatten(1).sum(1) > 0
    require(torch.allclose(weights.flatten(1).sum(1), positive.to(weights.dtype), atol=1e-6, rtol=1e-6),
            'Weight map mass differs from target')
    bce = F.binary_cross_entropy_with_logits(logits, masks, reduction='none').flatten(1)
    terms = []
    for i in range(logits.shape[0]):
        if positive[i]: terms.append((bce[i] * weights[i].flatten()).sum())
        else: terms.append(bce[i].topk(max(1, math.ceil(bce.shape[1] * .1))).values.mean())
    return torch.stack(terms).mean()


def validate_source(payload):
    fixed = {'initialization': 'pretrained', 'epoch': 30, 'optimizer_updates': 630,
             'additional_epoch': 20, 'fresh_optimizer_updates': 420}
    require(all(type(payload.get(k)) is type(v) and payload[k] == v for k, v in fixed.items()),
            'Require the verified pretrained epoch30 source')


def validate_optimizer(payload, groups, model_sha):
    require(set(payload) == {'state', 'model_sha256', 'fresh_optimizer_updates', 'cumulative_model_updates'} and
            payload['model_sha256'] == model_sha and type(payload['fresh_optimizer_updates']) is int and
            type(payload['cumulative_model_updates']) is int and payload['fresh_optimizer_updates'] == 420 and
            payload['cumulative_model_updates'] == 630, 'Optimizer source binding/counters differ')
    state = payload['state']; require(set(state) == {'state', 'param_groups'}, 'Optimizer fields differ')
    require(len(state['param_groups']) == len(groups) == 2, 'Optimizer group count differs')
    ids = []; parameters = []
    allowed = {'params', 'lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'maximize',
               'capturable', 'differentiable', 'foreach', 'fused', 'decoupled_weight_decay'}
    for stored, expected in zip(state['param_groups'], groups):
        require(set(stored).issubset(allowed) and stored.get('lr') == expected['lr'] and
                stored.get('weight_decay') == 1e-4 and tuple(stored.get('betas', ())) == (.9, .999) and
                stored.get('eps') == 1e-8 and stored.get('decoupled_weight_decay', True) is True,
                'Optimizer rate/decay/settings differ')
        require(all(stored.get(k) is False for k in ('amsgrad', 'maximize', 'capturable', 'differentiable')) and
                stored.get('foreach') is None and stored.get('fused') is None,
                'Optimizer execution flags differ')
        require(isinstance(stored.get('params'), list) and len(stored['params']) == len(expected['params']),
                'Optimizer parameter group length differs')
        ids.extend(stored['params']); parameters.extend(expected['params'])
    require(all(type(i) is int for i in ids) and ids == list(range(len(parameters))) and
            set(state['state']) == set(ids), 'Optimizer parameter IDs differ')
    for i, parameter in zip(ids, parameters):
        moment = state['state'][i]
        require(set(moment) == {'step', 'exp_avg', 'exp_avg_sq'} and isinstance(moment['step'], torch.Tensor) and
                moment['step'].numel() == 1 and torch.isfinite(moment['step']).all() and moment['step'].item() == 420,
                'Optimizer moment fields/step differ')
        require(all(isinstance(moment[k], torch.Tensor) and moment[k].shape == parameter.shape and
                    moment[k].dtype == parameter.dtype and torch.isfinite(moment[k]).all()
                    for k in ('exp_avg', 'exp_avg_sq')) and (moment['exp_avg_sq'] >= 0).all(),
                'Optimizer moments invalid')
    return {'parameter_states': len(ids), 'parameter_elements': sum(p.numel() for p in parameters),
            'restored_step': 420, 'cumulative_model_updates': 630}

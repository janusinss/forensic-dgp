"""Candidate-scoped AdaIN gradient routing; imports no neural framework.

The only changed operation is detaching the existing style tensor. Numerical
forward arithmetic remains in the original, hash-bound architecture callable.
Real autograd queries are invoked only by the manually guarded VM worker.
"""
from contextlib import contextmanager
import hashlib
import math


POLICY = 'candidate-AdaIN-style-detach-only-v1'
IMAGE_TERMS = frozenset(('pixel_l1_rgb', 'lpips_vgg', 'coarse_rgb_l1', 'behavior_rgb_l1'))
SUPERVISED_TERMS = frozenset(('token_ce', 'latent_mse', 'behavior_token_kl'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def isolated_style(architecture):
    """Restore the exact module callable even if a candidate forward fails."""
    original = architecture.adaptive_instance_normalization
    require(callable(original) and not getattr(original, '_isolated_style_override', False),
            'Nested or invalid AdaIN override is forbidden')
    record = {'policy': POLICY, 'calls': [], 'original_callable_restored': False}

    def detached_style(content, style):
        detached = style.detach()
        record['calls'].append({'style_shape': list(style.shape),
            'content_requires_grad': bool(content.requires_grad),
            'style_requires_grad_before': bool(style.requires_grad),
            'style_requires_grad_after': bool(detached.requires_grad),
            'dtype': str(style.dtype), 'device': str(style.device),
            'detached_shares_storage': style.data_ptr() == detached.data_ptr(),
            'shape_dtype_device_unchanged': style.shape == detached.shape and
                style.dtype == detached.dtype and style.device == detached.device})
        return original(content, detached)

    detached_style._isolated_style_override = True
    architecture.adaptive_instance_normalization = detached_style
    try:
        yield record
    finally:
        current = architecture.adaptive_instance_normalization
        architecture.adaptive_instance_normalization = original
        record['original_callable_restored'] = architecture.adaptive_instance_normalization is original
        require(current is detached_style, 'AdaIN callable changed inside the isolated candidate scope')


def validate_capture(record):
    require(record.get('policy') == POLICY and record.get('original_callable_restored') is True and
            len(record.get('calls', [])) == 1, 'Exactly one restored candidate AdaIN invocation required')
    call = record['calls'][0]
    require(call.get('style_shape') == [1, 256, 16, 16] and
            call.get('content_requires_grad') is False and
            call.get('style_requires_grad_before') is True and
            call.get('style_requires_grad_after') is False and
            call.get('dtype') == 'torch.float32' and call.get('device') == 'cuda:0' and
            call.get('detached_shares_storage') is True and
            call.get('shape_dtype_device_unchanged') is True,
            'Candidate AdaIN detach changed values/geometry/precision or retained the style gradient')
    return record


def parameter_partition(name):
    if name.startswith('encoder.'):
        return 'encoder'
    if name.startswith('fuse_convs_dict.'):
        return 'fusion'
    require(name.split('.')[0] in {'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer'},
            'Undeclared trainable parameter in route witness: ' + name)
    return 'predictor'


def summarize_gradients(names, gradients, np):
    """Finite CPU readout of VM-computed gradients, not a local recomputation."""
    require(len(names) == len(gradients) and len(set(names)) == len(names),
            'Gradient query parameter census differs')
    rows, groups = {}, {key: {'tensors': 0, 'none': 0, 'nonzero': 0,
                             'squared_l2': 0.0, 'maximum_absolute': 0.0}
                        for key in ('encoder', 'predictor', 'fusion')}
    for name, gradient in zip(names, gradients):
        group = groups[parameter_partition(name)]
        group['tensors'] += 1
        if gradient is None:
            rows[name] = {'is_none': True, 'nonzero': False, 'l2': 0.0,
                          'maximum_absolute': 0.0, 'sha256': None}
            group['none'] += 1
            continue
        array = np.ascontiguousarray(gradient.detach().float().cpu().numpy())
        require(array.dtype == np.float32 and np.isfinite(array).all(), 'Nonfinite/non-FP32 witness gradient')
        squared = float(np.square(array.astype(np.float64)).sum(dtype=np.float64))
        maximum = float(np.max(np.abs(array)))
        rows[name] = {'is_none': False, 'nonzero': maximum > 0,
            'l2': math.sqrt(squared), 'maximum_absolute': maximum,
            'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
            'shape': list(array.shape), 'dtype': str(array.dtype)}
        group['nonzero'] += int(maximum > 0)
        group['squared_l2'] += squared
        group['maximum_absolute'] = max(group['maximum_absolute'], maximum)
    for group in groups.values():
        group['l2'] = math.sqrt(group.pop('squared_l2'))
        require(math.isfinite(group['l2']), 'Nonfinite aggregate witness gradient')
    return {'parameters': rows, 'groups': groups,
            'full_gradient_vectors_exported': False,
            'norms_are_source_bound_VM_readouts_not_independent_gradient_replay': True}


def validate_witness_gradients(mode, image, supervised):
    require(mode in {'original', 'isolated'}, 'Unknown route-witness mode')
    for summary in (image, supervised):
        require(sum(row['tensors'] for row in summary['groups'].values()) == 350,
                'All350 trainable tensors must be queried')
        require(all(math.isfinite(row['l2']) for row in summary['groups'].values()),
                'Nonfinite witness aggregate')
    img, sup = image['groups'], supervised['groups']
    require(img['predictor']['none'] == img['predictor']['tensors'],
            'Hard-code image route unexpectedly reaches predictor')
    require(img['fusion']['l2'] > 0 and img['fusion']['nonzero'] > 0,
            'Image objective must retain finite nonzero fusion gradients')
    if mode == 'isolated':
        require(img['encoder']['none'] == img['encoder']['tensors'],
                'Isolated image objective must have no encoder connection, not merely zero values')
    else:
        require(img['encoder']['l2'] > 0 and img['encoder']['nonzero'] > 0,
                'Original image-to-encoder route was not witnessed')
    require(sup['encoder']['l2'] > 0 and sup['encoder']['nonzero'] > 0 and
            sup['predictor']['l2'] > 0 and sup['predictor']['nonzero'] > 0,
            'CE/HQ objective must retain encoder and predictor learning')
    return {'pass': True, 'mode': mode, 'queried_tensors': 350,
            'image_encoder_route_absent': mode == 'isolated',
            'supervised_encoder_predictor_and_image_fusion_routes_nonzero': True}


def validate_forward_parity(original, isolated, np):
    require(set(original) == set(isolated) and set(original) ==
            {'native', 'raw', 'logits', 'latent', 'tokens', 'terms'}, 'Incomplete forward parity evidence')
    for name in ('native', 'raw', 'logits', 'latent', 'tokens'):
        require(original[name].shape == isolated[name].shape and
                original[name].dtype == isolated[name].dtype and
                np.array_equal(original[name], isolated[name]),
                'Bit-exact original/isolated forward parity failed: ' + name)
    require(original['terms'] == isolated['terms'], 'Bit-exact original/isolated loss parity failed')
    return {'pass': True, 'native_raw_logits_latent_tokens_exact': True,
            'all_loss_terms_exact': True, 'floating_metadata_tolerance': 0.0}

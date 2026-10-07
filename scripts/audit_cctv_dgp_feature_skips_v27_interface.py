"""CPU inference-only wiring checks with fixed, non-fitted probe coefficients.

The probes do not use clean targets, estimate derivatives or create checkpoints.
Their only purpose is to check that every declared input-feature connection works.
"""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
SAVED = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
OUT = ROOT / 'outputs/cctv_dgp_feature_skips_v27_preparation/interface_inference_audit.json'
PIN = '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from audit_cctv_dgp_batchmatched_identity_v26 import make_head as original_head, state_hash

    started = time.monotonic(); assert not OUT.exists()
    assert sha(BUNDLE / 'protocol.json') == PIN
    p = read(BUNDLE / 'protocol.json')
    sources = {}

    def bind(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)

    for name, digest in p['assets_sha256'].items():
        assert sha(BUNDLE / name) == digest
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    code = ast.parse((BUNDLE / 'cctv_dgp_feature_skips_v27.py').read_text(encoding='utf-8'))
    cls = next(node for node in code.body if isinstance(node, ast.ClassDef))
    namespace = {'torch': torch, 'nn': torch.nn, 'F': F}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<local-pinned-V27-head-only>', 'exec'), namespace)
    torch.manual_seed(p['design']['seed'])
    head = namespace['SpatialFeatureHead']().cpu().eval().requires_grad_(False)
    state = head.state_dict()
    old = original_head(ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26').state_dict()
    assert len(list(head.parameters())) == 36 and sum(value.numel() for value in head.parameters()) == 55524
    assert len(state) == len(old) + 10
    assert all(torch.equal(state[name], value) for name, value in old.items()), 'Original CPU initialization changed'
    assert all(torch.count_nonzero(value) == 0 for name, value in state.items() if name.startswith('feature_skips.'))
    prior_path = SAVED / 'outputs/update0/head.pth'; bind(prior_path)
    vm_initial = torch.load(prior_path, map_location='cpu', weights_only=True)
    maximum_initial_difference = max(float((state[name].double() - value.double()).abs().max()) for name, value in vm_initial.items())
    assert maximum_initial_difference <= 1e-8, 'Original seeded CPU/VM compatibility exceeded'
    assert state_hash(vm_initial) == p['closed_V26_initial_head_state']
    for name, value in vm_initial.items():
        state[name] = value
    head.load_state_dict(state, strict=True)
    before = state_hash(head.state_dict())
    cache = read(SAVED / 'outputs/frozen_DGP_features.json'); bind(SAVED / 'outputs/frozen_DGP_features.json')
    cache_folder = SAVED / 'outputs/frozen_DGP_features'

    def item(case):
        paths = [BUNDLE / case[name] for name in ['input', 'observed', 'raw_dgp', 'png_dgp']]
        for path in paths:
            bind(path)
        with Image.open(paths[0]) as image:
            camera = np.asarray(image).copy()
        with Image.open(paths[1]) as image:
            mask8 = np.asarray(image).copy() > 0
        base = np.load(paths[2], allow_pickle=False)
        with Image.open(paths[3]) as image:
            baseline8 = np.asarray(image).copy()
        values = []
        for index in range(5):
            name = case['id'] + '_fpn' + str(index) + '.npy'; path = cache_folder / name
            assert sha(path) == cache['files_sha256'][name]; bind(path)
            values.append(torch.from_numpy(np.load(path, allow_pickle=False).copy())[None])
        return {'x': torch.from_numpy(camera.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None],
            'base': torch.from_numpy(base.copy()).permute(2, 0, 1)[None],
            'mask': torch.from_numpy(mask8.astype(np.float32))[None, None],
            'fpn': tuple(values), 'camera': camera, 'mask8': mask8, 'baseline8': baseline8}

    rows = []; forwards = 0; probe_item = None
    for case in p['cases']:
        assert time.monotonic() - started < 180, 'Inference wiring audit cap180s'
        b = item(case)
        prediction = head(b['x'], b['base'], b['mask'], b['fpn']); forwards += 1
        assert torch.equal(prediction, b['base'])
        raw = prediction[0].permute(1, 2, 0).numpy()
        delivered = np.where(b['mask8'][..., None], np.floor(raw * np.float32(255)), b['camera']).astype(np.uint8)
        assert np.array_equal(delivered, b['baseline8'])
        rows.append({'id': case['id'], 'raw_initial_exact_baseline': True, 'PNG_initial_exact_baseline': True})
        if probe_item is None and case['profile'] != 'clear':
            probe_item = (case['id'], b)
    assert state_hash(head.state_dict()) == before
    probes = []; cid, b = probe_item
    for index in range(5):
        # Fixed input-only wiring coefficient, not fitted to any target or metric.
        head.feature_skips[index].weight[0, 0, 0, 0] = .01
        prediction = head(b['x'], b['base'], b['mask'], b['fpn']); forwards += 1
        difference = prediction - b['base']
        assert torch.count_nonzero(difference) > 0
        disabled = list(b['fpn']); disabled[index] = torch.zeros_like(disabled[index])
        control = head(b['x'], b['base'], b['mask'], tuple(disabled)); forwards += 1
        assert torch.equal(control, b['base']), 'Shortcut is not connected solely to its declared feature scale'
        probes.append({'scale': index, 'case': cid, 'preset_red_channel0_coefficient': .01,
            'response_RMS': float(difference.square().mean().sqrt()),
            'response_maximum': float(difference.abs().max()), 'zero_feature_control_exact_baseline': True,
            'fitted': False, 'quality_or_derivative_estimate': False})
        head.feature_skips[index].weight[0, 0, 0, 0] = 0
    head.feature_skips[0].weight[0, 0, 0, 0] = .01
    partial = torch.zeros_like(b['mask']); partial[:, :, 64:192, 64:192] = 1
    prediction = head(b['x'], b['base'], partial, b['fpn']); forwards += 1
    outside = (partial == 0).expand_as(prediction)
    assert torch.equal(prediction[outside], b['base'][outside])
    assert torch.isfinite(prediction).all() and 0 <= prediction.min() <= prediction.max() <= 1
    assert float((prediction - b['base']).abs().max()) <= .1 + 1e-7
    head.feature_skips[0].weight[0, 0, 0, 0] = 0
    assert state_hash(head.state_dict()) == before
    assert not torch.is_grad_enabled() and all(not value.requires_grad and value.grad is None for value in head.parameters())
    empty_rejected = False
    try:
        head(b['x'], b['base'], torch.zeros_like(b['mask']), b['fpn'])
    except AssertionError:
        empty_rejected = True
    assert empty_rejected
    for path in [Path(__file__), BUNDLE / 'protocol.json', BUNDLE / 'cctv_dgp_feature_skips_v27.py']:
        bind(path)
    result = {'complete': True, 'date': '2026-10-06', 'protocol_sha256': PIN,
        'source_bindings_sha256': sources, 'shared_original_CPU_tensors_exact': len(old),
        'original_CPU_VM_initialization_maximum_difference': maximum_initial_difference,
        'new_feature_skip_tensors_exactly_zero': 10, 'initial_cases': rows, 'probes': probes,
        'head_forwards': forwards, 'DGP_recognizer_forwards': 0,
        'partial_mask_outside_exact_and_output_bound_verified': True, 'empty_support_rejected': True,
        'final_head_state_unchanged': before, 'no_checkpoint_or_probe_images_created': True,
        'preset_coefficients_use_no_target_no_fit_no_derivative_estimate': True,
        'scope': 'Initial parity and five-scale inference wiring, not restoration quality or L4 gradient proof',
        'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'native_or_reserved_pixels_used': False, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - started}
    with OUT.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['source_bindings_sha256', 'initial_cases']}, indent=2))


if __name__ == '__main__':
    main()

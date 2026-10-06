"""Bounded CPU inference contracts on two exposed training inputs; no learning."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_degraded_detail_v24 as a


def main():
    import copy
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_spatial_features_v25 import SpatialFeatureHead, frozen_fpn_features, tensor_receipt
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    started = time.monotonic(); torch.set_num_threads(4)
    out = ROOT / 'outputs/cctv_dgp_spatial_features_v25_interface'; out.mkdir()
    ids = ['v9_tr_asian_00048_blur_lr24', 'v9_tr_ffhq_00084_motion_lr48']
    module = ROOT / 'scripts/cctv_dgp_spatial_features_v25.py'
    p = a.verify_bundle(a.BUNDLE)
    plan = {'date': '2026-10-06', 'scope': 'Own-DGP feature interface and untrained forward contracts only, two already-exposed TRAINING inputs',
        'runner_sha256': a.sha(Path(__file__)), 'module_sha256': a.sha(module), 'case_ids': ids,
        'old_protocol_sha256': a.PIN, 'user_decision_sha256': a.sha(ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json'),
        'DGP_forward_cap': 4, 'head_forward_cap': 5, 'seconds_cap': 300,
        'initial_parity': 'Exact fresh frozen CPU DGP raw/PNG; cached historical CUDA differences recorded separately, not relabeled canonical parity.',
        'disposable_weights': 'Two fixed non-training fixtures: direct smooth signal and a specified feature0 connection. Never stored as a trained candidate.',
        'backward_calls': 0, 'optimizer_updates': 0, 'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False}
    a.write(out / 'plan.json', plan)
    torch.manual_seed(20261005)
    head = SpatialFeatureHead().cpu().eval().requires_grad_(False)
    initial_state = a.state_hash(head.state_dict())
    counts = {'DGP_forwards': 0, 'head_forwards': 0}
    head.register_forward_hook(lambda *_: counts.__setitem__('head_forwards', counts['head_forwards'] + 1))
    net, provenance = load_frozen_dgp_restorer(a.BUNDLE / 'weights/dgp_v2.pth',
        expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'], device='cpu')
    net.net.register_forward_hook(lambda *_: counts.__setitem__('DGP_forwards', counts['DGP_forwards'] + 1))
    before = a.state_hash(net.net.state_dict())
    rows = []; bindings = {}; captured_fixture = None
    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = a.sha(path)
    bind(module); bind(a.BUNDLE / 'protocol.json'); bind(a.BUNDLE / 'weights/dgp_v2.pth')
    for cid in ids:
        assert time.monotonic() - started < 300
        case = next(c for c in p['cases'] if c['id'] == cid)
        camera = a.rgb(a.BUNDLE / case['input']); mask = a.observed(a.BUNDLE / case['observed'])
        for key in ['input', 'observed', 'raw_dgp']:
            bind(a.BUNDLE / case[key])
        x = torch.from_numpy(camera.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
        # Explicit CPU byte-scalar convention for comparison with inherited cache.
        dgp_x = torch.from_numpy(camera.copy()).permute(2, 0, 1).float()[None] / 255
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        base, features = frozen_fpn_features(net.net, dgp_x)
        with torch.inference_mode():
            unhooked = net.net(dgp_x)
            initial = head(x, base, m, features)
        assert torch.equal(base, unhooked) and torch.equal(initial, base)
        assert len(net.net.fpn._forward_hooks) == 0
        arrays = {f'fpn{index}': value[0].cpu().numpy().copy() for index, value in enumerate(features)}
        np.savez(out / (cid + '_features.npz'), **arrays)
        raw = base[0].permute(1, 2, 0).numpy().copy()
        np.save(out / (cid + '_baseline.npy'), raw, allow_pickle=False)
        np.save(out / (cid + '_initial.npy'), initial[0].permute(1, 2, 0).numpy(), allow_pickle=False)
        baseline_png = a.png(raw, camera, mask)
        Image.fromarray(baseline_png).save(out / (cid + '_initial.png'))
        cached = a.raw_rgb(a.BUNDLE / case['raw_dgp'])
        rows.append({'id': cid, 'features': [tensor_receipt(value) for value in features],
            'raw_initial_maximum_error': 0., 'raw_hook_vs_unhooked_maximum_error': 0.,
            'cached_CUDA_raw_maximum_difference': float(np.max(np.abs(raw - cached))),
            'PNG_initial_exact': True, 'capture_hook_removed': True})
        if captured_fixture is None:
            captured_fixture = (x, base, m, features)
    assert a.state_hash(net.net.state_dict()) == before and a.state_hash(head.state_dict()) == initial_state
    assert not any(v.grad is not None or v.requires_grad for v in list(net.net.parameters()) + list(head.parameters()))
    disposable = copy.deepcopy(head)
    with torch.no_grad():
        for value in disposable.parameters(): value.zero_()
        disposable.direct.weight[:, 0, 1, 1] = .1
    u = torch.arange(256, dtype=torch.float32)[None, None, None, :]
    scene = (.5 + .15 * torch.sin(2 * torch.pi * u / 128)).expand(1, 3, 256, 256).clone()
    base = torch.full_like(scene, .5); mask = torch.zeros(1, 1, 256, 256); mask[:, :, 32:224, 16:240] = 1
    zeros = tuple(torch.zeros(1, channels, size, size) for channels, size in zip([64, 128, 128, 128, 128], [128, 64, 32, 16, 8]))
    with torch.inference_mode():
        actual = disposable(scene, base, mask, zeros)
        q = (.05 * torch.tanh(.1 * scene[:, :1])).expand_as(scene)
        expected = base + (q - (q * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True)) * mask
    outside = (~mask.bool()).expand_as(base)
    smooth_error = float((actual - expected).abs().max())
    assert smooth_error <= 1e-7 and torch.equal(actual[outside], base[outside])
    np.save(out / 'disposable_smooth_raw.npy', actual[0].permute(1, 2, 0).numpy(), allow_pickle=False)
    with torch.no_grad():
        for value in disposable.parameters(): value.zero_()
        disposable.projections[0].weight[0, 0, 0, 0] = 1
        disposable.decode0.weight[0, 24, 1, 1] = .2
        disposable.fuse.weight[0, 0, 1, 1] = .2
        disposable.tail.weight[0, 0, 1, 1] = .2
    x, base, mask, features = captured_fixture
    changed = [value.clone() for value in features]
    v = torch.arange(128, dtype=torch.float32)[None, :]
    changed[0][0, 0] += .1 * torch.sin(2 * torch.pi * v / 64).expand(128, 128)
    with torch.inference_mode():
        original = disposable(x, base, mask, features)
        perturbed = disposable(x, base, mask, tuple(changed))
    feature_response = float((original - perturbed).abs().max())
    assert feature_response > 0 and torch.isfinite(perturbed).all()
    assert counts == {'DGP_forwards': 4, 'head_forwards': 5}
    assert a.state_hash(net.net.state_dict()) == before and a.state_hash(head.state_dict()) == initial_state
    assert not any(v.grad is not None or v.requires_grad for v in disposable.parameters())
    assert time.monotonic() - started <= 300
    torch.save(head.state_dict(), out / 'untrained_initial_head.pth')
    artifacts = {path.name: a.sha(path) for path in out.iterdir() if path.is_file()}
    result = {'complete': True, 'date': '2026-10-06', 'seconds': time.monotonic() - started,
        'runner_sha256': a.sha(Path(__file__)), 'plan_sha256': a.sha(out / 'plan.json'),
        'source_bindings_sha256': bindings, 'artifacts_sha256': artifacts,
        'rows': rows, 'counts': counts, 'trainable_design_parameters': sum(v.numel() for v in head.parameters()),
        'initial_head_state': initial_state, 'frozen_DGP_state_before_after': before, 'DGP_provenance': provenance,
        'fixed_smooth_signal_maximum_contract_error': smooth_error,
        'fixed_smooth_signal_outside_values_exact': int(outside.sum()),
        'disposable_feature0_connection_maximum_response': feature_response,
        'feature_tensors_are_detached_ordinary_tensors': True, 'capture_hooks_removed': True,
        'learned_capacity_or_quality_proven': False, 'disposable_weights_are_not_candidates': True,
        'VM_gradient_or_timing_evidence': False, 'backward_calls': 0, 'optimizer_updates': 0,
        'VM_actions': False, 'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False}
    a.write(out / 'results.json', result)
    print(json.dumps({key: result[key] for key in ['complete', 'seconds', 'counts', 'trainable_design_parameters',
        'fixed_smooth_signal_maximum_contract_error', 'disposable_feature0_connection_maximum_response', 'learned_capacity_or_quality_proven']}, indent=2))


if __name__ == '__main__':
    main()

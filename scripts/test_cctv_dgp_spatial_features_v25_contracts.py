"""Feature-interface failure regressions, without backward or optimization."""
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_spatial_features_v25 import SpatialFeatureHead, frozen_fpn_features, tensor_receipt
from cctv_dgp_spatial_features_v25_prepare import prepare_features


def main():
    start = time.monotonic(); torch.set_num_threads(4)
    out = ROOT / 'outputs/cctv_dgp_spatial_features_v25_contract_regressions'
    assert not out.exists(), 'Keep previous regression evidence'
    passed = []; counts = {'successful_head_forwards': 0, 'fake_provider_attempts': 0}
    def rejects(name, action, expected=AssertionError):
        try:
            action()
        except expected:
            passed.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    head = SpatialFeatureHead().eval().requires_grad_(False)
    x = torch.full((1, 3, 256, 256), .5); mask = torch.ones(1, 1, 256, 256)
    shapes = [(1, c, s, s) for c, s in zip([64,128,128,128,128], [128,64,32,16,8])]
    features = tuple(torch.zeros(shape) for shape in shapes)
    rejects('missing_feature_scale', lambda: head(x, x, mask, features[:-1]))
    bad = list(features); bad[4] = torch.zeros(1, 128, 9, 8)
    rejects('wrong_coarse_feature_shape', lambda: head(x, x, mask, tuple(bad)))
    bad = list(features); bad[0] = bad[0].requires_grad_(True)
    rejects('feature_encoder_gradient_forbidden', lambda: head(x, x, mask, tuple(bad)))
    features[0].requires_grad_(False)
    with torch.inference_mode():
        bad = list(features); bad[0] = torch.zeros(shapes[0])
    rejects('inference_tensor_cannot_feed_trainable_convolution', lambda: head(x, x, mask, tuple(bad)))
    rejects('empty_observed_support', lambda: head(x, x, torch.zeros_like(mask), features))
    bad = list(features); bad[2] = bad[2].double()
    rejects('incompatible_feature_dtype', lambda: head(x, x, mask, tuple(bad)))
    class FPN(nn.Module):
        def forward(self, value):
            return tuple(torch.zeros(shape) for shape in shapes)
    class Provider(nn.Module):
        def __init__(self):
            super().__init__(); self.fpn = FPN(); self.fail = False
        def forward(self, value):
            counts['fake_provider_attempts'] += 1
            self.fpn(value)
            if self.fail: raise RuntimeError('intentional post-hook failure')
            return value.clone()
    net = Provider().eval()
    with torch.inference_mode():
        raw, captured = frozen_fpn_features(net, x)
    assert not torch.is_inference(raw) and all(not torch.is_inference(v) and not v.requires_grad for v in captured)
    assert not net.fpn._forward_hooks and torch.equal(raw, x)
    passed.append('ordinary_detached_features_inside_outer_inference_mode')
    net.fail = True
    rejects('capture_hook_removed_on_provider_exception', lambda: frozen_fpn_features(net, x), RuntimeError)
    assert not net.fpn._forward_hooks
    rejects('training_provider_rejected_before_forward', lambda: frozen_fpn_features(net.train(), x))
    net.eval(); net.fail = False
    rejects('input_gradient_rejected_before_forward', lambda: frozen_fpn_features(net, x.clone().requires_grad_(True)))
    bad = torch.full((1,), float('nan'))
    rejects('nonfinite_feature_receipt_rejected', lambda: tensor_receipt(bad))
    def blocked_guard(*_args, **_kwargs):
        raise RuntimeError('blocked before neural/optimizer work')
    rejects('prepare_guard_before_GPU_or_model_loading', lambda: prepare_features(ROOT, {}, blocked_guard), RuntimeError)
    assert counts == {'successful_head_forwards': 0, 'fake_provider_attempts': 2}
    assert all(value.grad is None for value in head.parameters())
    out.mkdir()
    sha = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
    record = {'complete': True, 'date': '2026-10-06', 'regressions_passed': passed,
        'count': len(passed), 'counts': counts,
        'source_bindings_sha256': {file.relative_to(ROOT).as_posix(): sha(file) for file in
            [Path(__file__), ROOT/'scripts/cctv_dgp_spatial_features_v25.py', ROOT/'scripts/cctv_dgp_spatial_features_v25_prepare.py']},
        'actual_DGP_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'VM_actions': False,
        'quality_or_training_pass': False, 'seconds': time.monotonic()-start}
    with (out/'results.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()

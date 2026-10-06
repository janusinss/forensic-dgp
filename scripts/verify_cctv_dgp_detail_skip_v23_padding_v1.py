"""Fixed, no-target synthetic padding contract; CPU forwards only, no fitting."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_detail_prior_v22_r1 as a
from verify_cctv_dgp_detail_skip_v23_preparation import head_from


def main():
    import numpy as np
    import torch
    torch.set_num_threads(4)
    started = time.monotonic()
    bundle = ROOT / 'outputs/cctv_dgp_detail_skip_vm_v23'
    evidence = ROOT / 'outputs/cctv_dgp_detail_skip_v23_preparation'
    protocol = a.read(bundle / 'protocol.json')
    prepared = a.read(evidence / 'preparation.json')
    source = bundle / 'scripts/cctv_dgp_detail_skip_v23.py'
    a.require(a.sha(bundle / 'protocol.json') == prepared['protocol_sha256'] and
              a.sha(source) == protocol['assets_sha256']['scripts/cctv_dgp_detail_skip_v23.py'], 'Frozen source differs')
    plan = {
        'date': '2026-10-06', 'scope': 'Synthetic contract, no photographs, targets, optimization or quality selection',
        'protocol_sha256': prepared['protocol_sha256'], 'checker_sha256': a.sha(Path(__file__)),
        'observed_rectangle_yx': [24, 232, 32, 224], 'base_RGB': [.5, .5, .5],
        'camera_bytes': '32 + ((x + 3*y + channel*7) modulo 193)',
        'fixed_nonzero_parameter': 'direct.weight[0,6,1,1] = 0.1',
        'mean_error_maximum': 2e-8, 'forwards': 2, 'time_cap_seconds': 30,
        'backward_calls': 0, 'optimizer_updates': 0, 'coefficient_search': False,
    }
    a.write(evidence / 'nonempty_padding_plan_v1.json', plan)
    head = head_from(source)
    before = a.state_hash(head.state_dict())
    yy, xx = np.indices((256, 256))
    camera = np.stack([32 + (xx + 3 * yy + channel * 7) % 193 for channel in range(3)], axis=2).astype(np.uint8)
    base = np.full((256, 256, 3), np.float32(.5), dtype=np.float32)
    observed = np.zeros((256, 256), bool)
    observed[24:232, 32:224] = True
    x = torch.from_numpy(camera.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
    b = torch.from_numpy(base).permute(2, 0, 1)[None]
    m = torch.from_numpy(observed.astype(np.float32))[None, None]
    with torch.inference_mode():
        initial = head(x, b, m)[0].permute(1, 2, 0).numpy().copy()
        a.require(np.array_equal(initial, base), 'Synthetic zero head changes base')
        head.direct.weight[0, 6, 1, 1] = .1
        changed = head(x, b, m)[0].permute(1, 2, 0).numpy().copy()
        head.direct.weight[0, 6, 1, 1] = 0
    a.require(np.isfinite(changed).all() and changed.min() >= 0 and changed.max() <= 1, 'Nonfinite/out-of-range result')
    a.require(np.array_equal(changed[~observed], base[~observed]), 'Nonempty padding changes in raw output')
    delivered = a.png(changed, camera, observed)
    a.require(np.array_equal(delivered[~observed], camera[~observed]), 'Delivered padding changes')
    delta = (changed - base)[observed].astype(np.float64)
    mean = np.abs(delta.mean(0))
    a.require(float(mean.max()) <= 2e-8 and float(np.abs(delta).max()) > 1e-6, 'Nonzero detail/observed mean contract fails')
    a.require(a.state_hash(head.state_dict()) == before and not any(p.requires_grad or p.grad is not None for p in head.parameters()), 'Disposable fixed test changes state/gradients')
    seconds = time.monotonic() - started
    a.require(seconds <= 30, 'Fixed CPU contract exceeds cap')
    receipt = {
        'complete': True, 'date': '2026-10-06', 'checker_sha256': a.sha(Path(__file__)),
        'source_sha256': a.sha(source), 'plan_sha256': a.sha(evidence / 'nonempty_padding_plan_v1.json'),
        'observed_pixels': int(observed.sum()), 'padding_pixels': int((~observed).sum()),
        'exact_zero_baseline': True, 'raw_padding_exact': True, 'PNG_padding_exact': True,
        'maximum_absolute_detail': float(np.abs(delta).max()), 'absolute_observed_mean_RGB': mean.tolist(),
        'finite_range': [float(changed.min()), float(changed.max())], 'state_restored': True,
        'new_head_CPU_forwards': 2, 'recognizer_DGP_forwards': 0, 'optimizer_updates': 0, 'backward_calls': 0,
        'seconds': seconds, 'VM_actions': False, 'quality_claim': False, 'goal_complete': False,
    }
    a.write(evidence / 'nonempty_padding_contract_v1.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

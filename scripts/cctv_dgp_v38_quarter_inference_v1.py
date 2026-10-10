"""Read the independently audited quarter state into disposable frozen models.

No checkpoint is written, no optimizer is created and no learning is performed.
The fixed state was chosen from TRAIN results before new development inference.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return'
PROTOCOL_SHA = 'caea39469005b7066d7cdbd74e073ba674b9b9b6fb9324f3110fe827a06948ab'
ORIGINAL_SHA = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
STATE = 'b7aad53d93d4fa58ca826be6162667fff2faf85578da4f421e938b8711bc49ba'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_basis(original_path):
    audit = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_independent_audit.json')
    review = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/visual_review.json')
    checked = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/independent_analysis_page_audit.json')
    assert audit['complete'] and audit['finite_probe_complete'] and not audit['failure_retained']
    assert audit['protocol_sha256'] == sha(RETURN / 'protocol.json') == PROTOCOL_SHA
    assert checked['complete'] and checked['all700_cells_exact']
    assert sha(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/visual_review.json') == checked['visual_review_sha256']
    assert review['actually_viewed_cases'] == 100 and not review['app_adoption']
    assert 'margin_quarter' in checked['jointly_eligible_subset_variants']
    assert sha(original_path) == ORIGINAL_SHA
    p = read(RETURN / 'protocol.json')
    for name in ['theta_before.npy', 'projected_displacement.npy']:
        assert sha(RETURN / name) == p['assets_sha256'][name]
    for label in ['exposed', 'unexposed']:
        receipt = read(RETURN / f'outputs/state0_{label}/margin_quarter/receipt.json')
        comparison = read(RETURN / f'outputs/state0_{label}/margin_quarter/comparison.json')
        assert receipt['candidate_state'] == STATE
        assert comparison['preservation_against_original']['all17_preservation_groups_pass']
    return p


def load_v38_quarter(original_path, *, device='cpu'):
    import numpy as np
    import torch
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from dgp_mean_centered_inference_v29 import MeanCenteredDGPInferenceV29
    p = verify_basis(original_path)
    original, provenance = load_frozen_dgp_restorer(original_path, expected_sha256=ORIGINAL_SHA, device=device)
    candidate, _ = load_frozen_dgp_restorer(original_path, expected_sha256=ORIGINAL_SHA, device=device)
    initial = {name: value.detach().clone() for name, value in candidate.net.state_dict().items()}
    selected = dict(candidate.net.named_parameters())
    theta = np.load(RETURN / 'theta_before.npy', allow_pickle=False)
    direction = np.load(RETURN / 'projected_displacement.npy', allow_pickle=False)
    assert theta.dtype == np.float32 and direction.dtype == np.float64
    assert theta.shape == direction.shape == (978243,) and np.isfinite(theta).all() and np.isfinite(direction).all()
    assert len(p['parameter_layout']) == 23
    with torch.inference_mode():
        actual = torch.cat([selected[row['name']].reshape(-1) for row in p['parameter_layout']]).cpu().numpy()
        assert np.array_equal(actual, theta)
        fixed = (theta.astype(np.float64) - .25 * direction).astype(np.float32)
        for row in p['parameter_layout']:
            value = selected[row['name']]
            assert list(value.shape) == row['shape'] and row['end'] - row['start'] == value.numel()
            value.copy_(torch.from_numpy(fixed[row['start']:row['end']].copy()).to(device).reshape(value.shape))
    names = {row['name'] for row in p['parameter_layout']}
    for name, value in candidate.net.state_dict().items():
        if name not in names:
            assert torch.equal(value, initial[name]), name
    assert state_hash(original.net) == 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
    assert state_hash(candidate.net) == STATE
    model = MeanCenteredDGPInferenceV29(original, candidate).to(device).eval().requires_grad_(False)
    assert all(not v.requires_grad and v.grad is None for v in model.parameters())
    return model, {'backend': 'our-DGP-disposable-V38-quarter', 'protocol_sha256': PROTOCOL_SHA,
        'original_checkpoint_sha256': ORIGINAL_SHA, 'candidate_state': STATE, 'scale': .25,
        'selection': 'Largest step passing both TRAIN cohorts; fixed before new development outputs',
        'original': provenance, 'same_input_original_baseline_required': True,
        'mean_centering': 'Same float32 per-case observed delta centering used by audited V38',
        'selected_tensors': 23, 'selected_parameter_values': 978243,
        'fixed_normalization_layers_per_DGP': 5, 'all_unselected_tensors_and_buffers_exact': True,
        'training': False, 'inference_only': True, 'gradient_calls': 0, 'optimizer_updates': 0,
        'checkpoint_written': False, 'enhancement': 'none', 'automatic_app_promotion': False,
        'training_capacity_pass': False, 'quality_claim': 'Finite TRAIN preservation pass only; useful development structure unqualified'}

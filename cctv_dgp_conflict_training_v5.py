"""Fixed V5 lineage and detached two-objective gradient arithmetic; no training."""
import copy
import math
from pathlib import Path

import torch

from cctv_dgp_pilot import read, sha
from cctv_dgp_perceptual_training_v3 import (START_SHA, START_STATE,
    CalibratedPerceptualV3, starting_weights_v3, load_starting_state)
from cctv_dgp_objective_diagnostic_v4 import verify_recipe, TEACHERS

ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT
PERCEPTUAL_DIR = ROOT
DIAGNOSTIC_DIR = ROOT
PARENT_RETURN = V3_RETURN = V4_RETURN = None
V4_RESULTS_SHA = 'e73b789f437483d046763cca60b07c8f67d81238dd993eab1976d689b5ff4ebb'
V4_AUDIT_SHA = '1a4a1bd1c904b2d9c08f42837b91bf95828931b821d6832b19ff03aeaa113b05'
RUNTIME_CAP_SECONDS = 1800
ARMS = [
    {'id': 'identity_weight04', 'lambda_identity': .4, 'feature_policy': 'postactivation',
     'gradient_policy': 'weighted_sum'},
    {'id': 'two_objective_pcgrad', 'lambda_identity': .1, 'feature_policy': 'postactivation',
     'gradient_policy': 'two_objective_pcgrad'},
]
REQUIRED_ASSETS = {
    'cctv_dgp_conflict_training_v5.py', 'scripts/run_cctv_dgp_conflict_vm_v5.py',
    'scripts/run_cctv_dgp_conflict_vm_v5.sh', 'scripts/audit_cctv_dgp_conflict_results_v5.py',
    'scripts/derive_cctv_dgp_conflict_v5.py', 'scripts/prepare_cctv_dgp_conflict_v5.py',
    'tests/test_cctv_dgp_conflict_v5.py', 'CCTV_DGP_CONFLICT_V5.md',
    'v4_local_audit_for_training.json',
}


def configure_paths(bundle_dir, parent_return=None, perceptual_bundle_dir=None,
                    diagnostic_bundle_dir=None, v3_return=None, v4_return=None):
    global BUNDLE_DIR, PERCEPTUAL_DIR, DIAGNOSTIC_DIR, PARENT_RETURN, V3_RETURN, V4_RETURN
    BUNDLE_DIR = Path(bundle_dir).resolve()
    PERCEPTUAL_DIR = Path(perceptual_bundle_dir).resolve() if perceptual_bundle_dir else ROOT
    DIAGNOSTIC_DIR = Path(diagnostic_bundle_dir).resolve() if diagnostic_bundle_dir else ROOT
    PARENT_RETURN = Path(parent_return).resolve() if parent_return else None
    V3_RETURN = Path(v3_return).resolve() if v3_return else None
    V4_RETURN = Path(v4_return).resolve() if v4_return else None


def protocol_sha_v5(root):
    return sha(BUNDLE_DIR / 'conflict_protocol_v5.json')


def starting_weights_v5(root):
    return starting_weights_v3(root)


def verify_protocol_v5(root):
    root = Path(root).resolve()
    parent, diagnostic = verify_recipe(root, DIAGNOSTIC_DIR, PERCEPTUAL_DIR, PARENT_RETURN, V3_RETURN)
    manifest_path = BUNDLE_DIR / 'conflict_protocol_v5.json'
    if (BUNDLE_DIR / 'conflict_protocol_v5.sha256').read_bytes() != (sha(manifest_path)+'\n').encode('ascii'):
        raise ValueError('V5 protocol fingerprint/LF differs')
    recipe = read(manifest_path)
    expected = copy.deepcopy(parent)
    expected['arms'] = ARMS
    if (recipe['format'] != 'cctv-two-objective-conflict-pilot-v5'
            or recipe['diagnostic_protocol_sha256'] != sha(DIAGNOSTIC_DIR / 'objective_diagnostic_protocol_v4.json')
            or recipe['diagnostic_results_sha256'] != V4_RESULTS_SHA
            or recipe['diagnostic_local_audit_sha256'] != V4_AUDIT_SHA
            or recipe['starting_checkpoint_sha256'] != START_SHA or recipe['starting_state_hash'] != START_STATE
            or recipe['runtime_cap_seconds'] != RUNTIME_CAP_SECONDS or recipe['protocol'] != expected
            or recipe['expected_training_autograd_grad_calls'] != 904 or recipe['expected_preflight_autograd_grad_calls'] != 8
            or recipe['native_cases_used'] != 0 or recipe['native_reserved_used'] or recipe['production_promotion_permitted']):
        raise ValueError('V5 fixed objective/cohort/start/budget differs')
    if set(recipe['assets_sha256']) != REQUIRED_ASSETS:
        raise ValueError('V5 executable/lineage inventory differs')
    for name, pin in recipe['assets_sha256'].items():
        asset = (BUNDLE_DIR / name).resolve()
        if not asset.is_relative_to(BUNDLE_DIR) or sha(asset) != pin:
            raise ValueError('Changed/escaped V5 asset: '+name)
        if name.endswith('.py') and sha(ROOT / name) != pin:
            raise ValueError('Executed V5 source differs: '+name)
    returned = V4_RETURN or root / 'outputs/cctv_dgp_objective_diagnostic_v4'
    if sha(returned / 'results.json') != V4_RESULTS_SHA:
        raise ValueError('Completed V4 diagnostic changed')
    capsule = BUNDLE_DIR / 'v4_local_audit_for_training.json'
    if sha(capsule) != V4_AUDIT_SHA:
        raise ValueError('Independent V4 receipt changed')
    # Rebuild the diagnostic arithmetic/source/state audit before admitting training.
    from scripts.audit_cctv_dgp_objective_diagnostic_v4 import audit
    if read(capsule) != audit(root, DIAGNOSTIC_DIR, returned, PERCEPTUAL_DIR, PARENT_RETURN, V3_RETURN):
        raise ValueError('Independent V4 receipt does not reproduce')
    return recipe['protocol']


def rebuild_policy_summary(policy, dimension, gram):
    """Rebuild a two-objective receipt from weighted-gradient dot products only."""
    if policy not in ('weighted_sum', 'two_objective_pcgrad') or type(dimension) is not int or dimension < 1:
        raise ValueError('Unknown gradient policy or invalid dimension')
    if len(gram) != 2 or any(len(row) != 2 for row in gram):
        raise ValueError('Two-objective Gram required')
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in gram for v in row):
        raise ValueError('Nonfinite objective Gram')
    r2, dot, i2 = float(gram[0][0]), float(gram[0][1]), float(gram[1][1])
    if r2 < 0 or i2 < 0 or dot != gram[1][0] or dot*dot > r2*i2 + 1e-10*max(1., r2*i2):
        raise ValueError('Invalid objective Gram symmetry/positive semidefiniteness')
    if (r2 == 0 or i2 == 0) and dot != 0:
        raise ValueError('Zero gradient has a nonzero cross product')
    conflict = dot < 0
    cr = ci = 1.
    if policy == 'two_objective_pcgrad' and conflict:
        # Project each original objective against the OTHER ORIGINAL gradient, then sum.
        cr, ci = 1.-dot/r2, 1.-dot/i2
    square = cr*cr*r2 + 2*cr*ci*dot + ci*ci*i2
    return {'policy': policy, 'dimension': dimension, 'gram_matrix': [[r2, dot], [dot, i2]],
            'cosine': dot/math.sqrt(r2*i2) if r2*i2 else None, 'conflict': conflict,
            'coefficients': [cr, ci], 'combined_l2': math.sqrt(max(0., square)),
            'combined_reconstruction_dot': cr*r2+ci*dot,
            'combined_identity_dot': cr*dot+ci*i2,
            'basis': 'Weighted parameter gradients before clipping/Adam; no output-preservation guarantee'}


def combine_gradients(reconstruction, identity, policy):
    """Detached arithmetic only; autograd and optimizer invocation belong to the guarded VM runner."""
    if not reconstruction or len(reconstruction) != len(identity):
        raise ValueError('Nonempty matching parameter gradient lists required')
    if any(not isinstance(r, torch.Tensor) or not isinstance(i, torch.Tensor) or r.shape != i.shape
           or r.device != i.device or r.dtype != i.dtype or not r.is_floating_point() or not r.numel()
           for r, i in zip(reconstruction, identity)):
        raise ValueError('Gradient parameter shapes/devices/dtypes differ')
    with torch.no_grad():
        r = torch.cat([v.detach().reshape(-1) for v in reconstruction])
        i = torch.cat([v.detach().reshape(-1) for v in identity])
        if not torch.isfinite(r).all() or not torch.isfinite(i).all():
            raise ValueError('Nonfinite objective gradients')
        rd, id_ = r.double(), i.double()
        r2, dot, i2 = float(rd.dot(rd)), float(rd.dot(id_)), float(id_.dot(id_))
        receipt = rebuild_policy_summary(policy, r.numel(), [[r2, dot], [dot, i2]])
        cr, ci = receipt['coefficients']
        merged = tuple((cr*a.detach()+ci*b.detach()) for a, b in zip(reconstruction, identity))
    return merged, receipt

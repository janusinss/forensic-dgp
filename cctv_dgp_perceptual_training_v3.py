"""Explicit V3 recipe validation and calibrated perceptual loss; no optimizer here."""
import copy
import math
from pathlib import Path

from torch.nn import functional as F

from cctv_dgp_pilot import read, sha, state_hash, verify_bundle
from cctv_dgp_perceptual_v3 import PerceptualV3

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6'
START_SHA = 'b30aeabecc60dff2fbd289575ce8691be9e670c653cacb6721bc154b618c3916'
START_STATE = 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
RETURN_SHA = '61c537cf451542889d52d5b4ae820548e04970c9c0435c5ef2b760a8b8882725'
AUDIT_SHA = '8eb3ec00c31e201689a22d4682c6fa21345653990529bf9d0986133a50eae685'
EXISTING_RUNTIME_ASSETS = {
    'cctv_dgp_frozen_norm.py': '61c7ddc6f002a5ab87856ae43923e0ec8a1f5613c718d0fd6f68f1cab2936060',
    'scripts/run_cctv_dgp_normfix_vm.py': '85869bb4cdb2facf1ce6323743f78675c5badeb58f090498e174c4bfe5c427d2',
    'scripts/audit_cctv_dgp_normfix_results.py': '72d1eae872a2f375557babce327d101a225104d75e9fb7162b62c9815fa97b25',
}
REQUIRED_NEW_ASSETS = {
    'cctv_dgp_perceptual_v3.py', 'cctv_dgp_perceptual_training_v3.py',
    'scripts/run_cctv_dgp_perceptual_vm_v3.py', 'scripts/audit_cctv_dgp_perceptual_results_v3.py',
    'scripts/run_cctv_dgp_perceptual_vm_v3.sh', 'CCTV_DGP_PERCEPTUAL_V3.md',
    'parent_v2_local_audit.json', 'perceptual_scales_v3.json', 'calibration_protocol_v3.json',
}
BUNDLE_DIR = ROOT
PARENT_RETURN = None
ARMS = [{'id':'continued_postactivation', 'lambda_identity':.1, 'feature_policy':'postactivation'},
        {'id':'calibrated_preactivation', 'lambda_identity':.1, 'feature_policy':'preactivation'}]


def configure_paths(bundle_dir, parent_return=None):
    global BUNDLE_DIR, PARENT_RETURN
    BUNDLE_DIR = Path(bundle_dir).resolve()
    PARENT_RETURN = Path(parent_return).resolve() if parent_return else None


def protocol_path_v3(root):
    return BUNDLE_DIR / 'perceptual_protocol_v3.json'


def protocol_sha_v3(root):
    return sha(protocol_path_v3(root))


def starting_weights_v3(root):
    directory = PARENT_RETURN or Path(root) / 'outputs/cctv_dgp_pilot'
    return directory / 'camera_identity/best.pth'


def load_starting_state(root):
    import torch
    path = starting_weights_v3(root)
    if sha(path) != START_SHA:
        raise ValueError('Audited epoch2 starting checkpoint changed')
    weights = torch.load(path, map_location='cpu', weights_only=True)
    if state_hash(weights) != START_STATE:
        raise ValueError('Audited starting tensor state differs')
    return weights


def validate_scales(scales):
    if (not isinstance(scales, (list, tuple)) or len(scales) != 4 or
            any(type(v) not in (float, int) or not math.isfinite(v) or not 0 < v <= 10 for v in scales)):
        raise ValueError('Four finite positive fixed perceptual scales are required')
    return tuple(float(v) for v in scales)


def verify_protocol_v3(root):
    root = Path(root).resolve()
    parent = verify_bundle(root)
    if sha(root / 'protocol.json') != PARENT_SHA:
        raise ValueError('Original camera pilot protocol changed')
    if (BUNDLE_DIR / 'perceptual_protocol_v3.sha256').read_text(encoding='ascii').strip() != protocol_sha_v3(root):
        raise ValueError('V3 protocol fingerprint differs')
    recipe = read(protocol_path_v3(root))
    if (recipe['format'] != 'cctv-calibrated-perceptual-pilot-v3' or
            recipe['parent_protocol_sha256'] != PARENT_SHA or
            recipe['starting_checkpoint_sha256'] != START_SHA or
            recipe['starting_state_hash'] != START_STATE or
            recipe['parent_return_results_sha256'] != RETURN_SHA or
            recipe['runtime_cap_seconds'] != 5400):
        raise ValueError('V3 parent/start/budget differs')
    if not REQUIRED_NEW_ASSETS <= set(recipe['new_assets_sha256']):
        raise ValueError('V3 executable/calibration assets are missing')
    if recipe['existing_runtime_assets_sha256'] != EXISTING_RUNTIME_ASSETS:
        raise ValueError('Existing normalization/audit runtime pins differ')
    for name, pin in EXISTING_RUNTIME_ASSETS.items():
        if sha(ROOT / name) != pin:
            raise ValueError('Executed normalization/audit runtime differs: ' + name)
    for name, pin in recipe['new_assets_sha256'].items():
        asset = (BUNDLE_DIR / name).resolve()
        if not asset.is_relative_to(BUNDLE_DIR) or sha(asset) != pin:
            raise ValueError('Changed/escaped V3 asset: ' + name)
        if name.endswith('.py') and sha(ROOT / name) != pin:
            raise ValueError('Executed V3 source differs: ' + name)
    for name, pin in parent['assets_sha256'].items():
        if name.startswith('models/') or name in ('cctv_dgp_pilot.py',
                'scripts/run_cctv_dgp_pilot_vm.py', 'scripts/audit_cctv_dgp_pilot_results.py'):
            if sha(ROOT / name) != pin:
                raise ValueError('Executed original source differs: ' + name)
    if sha(starting_weights_v3(root)) != START_SHA:
        raise ValueError('Audited epoch2 starting checkpoint changed')
    directory = starting_weights_v3(root).parents[1]
    if sha(directory / 'results.json') != RETURN_SHA:
        raise ValueError('Parent completed return changed')
    if sha(BUNDLE_DIR / 'parent_v2_local_audit.json') != AUDIT_SHA:
        raise ValueError('Pinned parent independent audit changed')
    receipt = read(BUNDLE_DIR / 'parent_v2_local_audit.json')
    if not receipt['complete'] or receipt['returned_results_sha256'] != RETURN_SHA or receipt['update_trace_checked'] != 452:
        raise ValueError('Incomplete parent independent audit')
    calibration = read(BUNDLE_DIR / 'perceptual_scales_v3.json')
    calibration_plan = read(BUNDLE_DIR / 'calibration_protocol_v3.json')
    if (not calibration['complete'] or not calibration['teacher_states_unchanged'] or
            calibration['protocol_sha256'] != sha(BUNDLE_DIR / 'calibration_protocol_v3.json') or
            calibration['starting_checkpoint_sha256'] != START_SHA or
            calibration['starting_state_hash'] != START_STATE or
            calibration['reference_count'] != 64 or calibration['validation_references_used'] != 0 or
            calibration['native_cases_used'] != 0 or calibration['backward_calls'] != 0 or
            calibration['optimizer_updates'] != 0 or calibration['postactivation_scales'] != [1., 1., 1., 1.] or
            calibration['gradient_scale_equivalence_claimed'] or calibration['trained_improvement_claimed']):
        raise ValueError('Training-only scale calibration differs')
    chosen = [c for source in sorted({r['source'] for r in parent['references']})
              for c in [c for c in parent['training_epochs']['1'] if c['source'] == source][:32]]
    if calibration_plan['cases'] != chosen or len({c['reference_id'] for c in chosen}) != 64:
        raise ValueError('Fixed training-only calibration cases differ')
    validate_scales(calibration['preactivation_scales'])
    expected = copy.deepcopy(parent)
    expected['arms'] = ARMS
    expected['weights']['dgp'] = 'outputs/cctv_dgp_pilot/camera_identity/best.pth'
    if recipe['protocol'] != expected:
        raise ValueError('V3 changed data/order/seed/optimization/identity budget beyond the declared feature-policy comparison')
    return recipe['protocol']


class CalibratedPerceptualV3(PerceptualV3):
    def __init__(self, path, device):
        self.fixed_pre_scales = validate_scales(read(BUNDLE_DIR / 'perceptual_scales_v3.json')['preactivation_scales'])
        super().__init__(path, device, 'postactivation')

    def forward(self, x, target):
        import torch
        actual = self.taps(x)
        with torch.no_grad():
            expected = self.taps(target)
        scales = self.fixed_pre_scales if self.feature_policy == 'preactivation' else (1., 1., 1., 1.)
        return sum(w*s*F.l1_loss(a,b) for w,s,a,b in zip((.1,.2,1.,1.),scales,actual,expected))

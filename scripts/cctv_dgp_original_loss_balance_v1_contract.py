"""Finite saved-direction experiment; no neural import or automatic execution."""
import hashlib
import json
from pathlib import Path

NAME = 'cctv_dgp_original_loss_balance_v1_vm'
STEM = 'cctv-dgp-original-loss-balance-v1'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
STATE = 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
PARENT_PIN = 'ccc30870959c8f4f4f448c1d1e06bb0ae0403d450b81143c6bcb9032d122e94b'
TERMS = ['degraded_landmark_structure', 'observed_RGB', 'absolute_identity']
SCOPES = ['feature_identity', 'balanced_1', 'balanced_2']
FRACTIONS = [1e-5, 1e-4, 1e-3]
BUDGETS = {'cache_seconds': 120, 'gradient_seconds': 0, 'trial_seconds': 600,
    'worker_seconds': 1200, 'external_seconds': 1230, 'kill_grace_seconds': 30,
    'export_seconds': 300, 'export_external_seconds': 330,
    'minimum_free_disk_bytes': 4*1024**3, 'disk_reserve_bytes': 512*1024**2,
    'return_uncompressed_bytes': int(1.75*1024**3), 'peak_vram_bytes': 20*1024**3,
    'original_forward_calls': 20, 'candidate_forward_calls': 200,
    'recognizer_forward_calls': 220}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def directions(layout, aggregate):
    import numpy as np
    masks = {k: np.zeros(1996035, bool) for k in ['decoder_control', 'feature_only']}
    for row in layout: masks[row['partition']][row['start']:row['end']] = True
    d = -np.where(masks['decoder_control'], aggregate[TERMS[0]], 0.)
    f = -np.where(masks['feature_only'], aggregate[TERMS[2]], 0.)
    assert np.linalg.norm(d) > 0 and np.linalg.norm(f) > 0
    d /= np.linalg.norm(d); f /= np.linalg.norm(f)
    adverse = float(aggregate[TERMS[2]]@d); protection = -float(aggregate[TERMS[2]]@f)
    assert adverse > 0 and protection > 0
    ratio = adverse/protection
    return {'feature_identity': f, 'balanced_1': d+ratio*f, 'balanced_2': d+2*ratio*f}, ratio, masks


def validate(p):
    assert p['format'] == 'own-DGP-original-loss-balance-finite-saved-direction-v1'
    assert p['parent_protocol_sha256'] == PARENT_PIN
    assert p['original_checkpoint_sha256'] == CHECKPOINT and p['original_state'] == STATE
    assert p['budgets'] == BUDGETS and p['terms'] == TERMS and p['scopes'] == SCOPES
    assert p['relative_displacement_fractions'] == FRACTIONS and p['trial_variants'] == 9
    assert p['scale_weight_partition'] == 'decoder_control'
    assert p['canonical_direction_arithmetic_atol'] == 1e-14 and p['weight_norm_arithmetic_atol'] == 1e-10
    assert 0 < p['measured_balance_ratio'] < 1 and p['initial_decoder_weight_L2'] > 0
    assert p['gradient_queries'] == p['optimizer_updates'] == p['epochs'] == p['committed_trajectory_updates'] == 0
    assert p['saved_parent_gradient_queries'] == 30
    assert p['native_or_DEV_or_final_used'] is p['model_qualification'] is p['automatic_follow_on'] is p['app_promotion'] is False
    assert p['scientific_thresholds'] == {'early_structure_gain': .01, 'final_structure_gain': .1,
        'MSE_regression_tolerance': 1e-12, 'SSIM_ArcFace_regression_tolerance': 1e-6, 'brightness_fraction_maximum': .2}
    assert p['CPU_replay_tolerances'] == {'raw_max_abs': 3e-6, 'PNG_byte_max': 1, 'embedding_max_abs': 5e-5}
    assert len(p['cases']) == 100 and len(p['references']) == 20
    assert len({c['id'] for c in p['cases']}) == 100 and all(c['role'] == 'train' for c in p['cases'])
    assert len(p['cohorts']) == 2 and all(len(co['case_ids']) == 50 for co in p['cohorts'])
    assert set(p['cohorts'][0]['case_ids']).isdisjoint(p['cohorts'][1]['case_ids'])
    assert {c['id'] for c in p['cases']} == {cid for co in p['cohorts'] for cid in co['case_ids']}
    by = {c['id']: c for c in p['cases']}
    for co in p['cohorts']:
        rows = [by[cid] for cid in co['case_ids']]
        assert len({c['source_person_or_reference'] for c in rows}) == 10
        for i in range(0, 50, 5):
            assert [r['profile'] for r in rows[i:i+5]] == ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
            assert len({r['source_person_or_reference'] for r in rows[i:i+5]}) == 1
    layout = p['parameter_layout']; assert len(layout) == len({r['name'] for r in layout}) == 158
    end = 0
    for r in layout:
        assert r['start'] == end and r['end']-r['start'] == r['elements']
        assert r['partition'] in ['decoder_control', 'feature_only']; end = r['end']
    assert end == 1996035 and sum(r['elements'] for r in layout if r['partition'] == 'decoder_control') == 498627


def verified_assets(root, pin):
    root = Path(root).resolve(); assert sha(root/'protocol.json') == pin
    p = read(root/'protocol.json'); validate(p)
    declared = set(p['assets_sha256']) | {'protocol.json'}
    for name, digest in p['assets_sha256'].items():
        q = root/name
        assert q.resolve().is_relative_to(root) and q.is_file() and not q.is_symlink() and sha(q) == digest, name
    for q in root.rglob('*'):
        assert not q.is_symlink()
        if q.is_file():
            n = q.relative_to(root).as_posix()
            assert n in declared or n.startswith('outputs/') or n in [
                'diagnostic.log', 'diagnostic_exit_code.txt', 'supervisor_receipt.json', 'export_manifest.json'], n
    return p

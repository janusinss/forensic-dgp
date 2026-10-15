"""Finite manual reconstruction comparison; no neural imports or construction."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys

NAME = 'cctv_dgp_spatial_readout_v1_vm'
STEM = 'cctv-dgp-spatial-readout-v1'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
PARENT_PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
THRESHOLDS = dict(early_structure_gain=.01, final_structure_gain=.1,
    MSE_regression_tolerance=1e-12, SSIM_ArcFace_regression_tolerance=1e-6,
    brightness_fraction_maximum=.2)
BUDGETS = dict(worker_seconds=3000, external_seconds=3030, export_seconds=600,
    export_external_seconds=630, kill_grace_seconds=30, cache_seconds=240,
    preflight_seconds=180, arm_fit_seconds=600, snapshot_seconds=300,
    minimum_free_bytes=8*1024**3, available_RAM_before_import_bytes=8*1024**3,
    disk_reserve_bytes=1024**3, output_byte_cap=3*1024**3,
    peak_VRAM_bytes=20*1024**3, peak_RSS_bytes=12*1024**3,
    maximum_optimizer_updates=100, maximum_gradient_queries=4,
    maximum_backwards=100, original_batch_calls=20, prior_feature_batch_calls=20,
    decoder_batch_calls=184, recognizer_batch_calls=340)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require_manual(root):
    assert platform.system() == 'Linux', 'Human manual Linux L4 tmux execution only'
    assert platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing declared VM only'
    assert Path(root).resolve() == (Path.home()/'forensic-dgp'/NAME).resolve()
    assert Path.cwd().resolve() == Path(root).resolve() and os.environ.get('TMUX'), 'Human manual tmux required'
    assert sys.prefix == str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')


def verify_paths(folder, bindings):
    folder = Path(folder).resolve()
    for name, digest in bindings.items():
        assert isinstance(name, str) and '\\' not in name and ':' not in name
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = folder/name
        assert path.resolve().is_relative_to(folder) and not path.is_symlink(), name
        assert stat.S_ISREG(path.stat().st_mode) and sha(path) == digest, name


def center_kernel(weight):
    import numpy as np
    assert weight.dtype == np.float32 and weight.ndim == 4 and weight.shape[2:] == (1, 1)
    assert np.isfinite(weight).all()
    result = np.zeros(weight.shape[:2]+(3, 3), dtype=np.float32)
    result[:, :, 1, 1] = weight[:, :, 0, 0]
    return result


def common_radius(norms):
    assert len(norms) == 3 and all(math.isfinite(v) and v > 0 for v in norms)
    return .001*min(norms)


def project_original_ball(initial, proposal, radius):
    import numpy as np
    assert initial.dtype == proposal.dtype == np.float32
    assert initial.ndim == proposal.ndim == 1 and initial.shape == proposal.shape
    assert np.isfinite(initial).all() and np.isfinite(proposal).all() and math.isfinite(radius) and radius > 0
    delta = proposal.astype(np.float64)-initial.astype(np.float64)
    distance = float(np.linalg.norm(delta))
    reserve = float(2*np.linalg.norm(np.spacing(initial).astype(np.float64)))
    scale = 1. if distance <= radius else max(0., radius-reserve)/distance
    actual = proposal.copy() if scale == 1. else (initial.astype(np.float64)+scale*delta).astype(np.float32)
    actual_distance = float(np.linalg.norm(actual.astype(np.float64)-initial.astype(np.float64)))
    assert actual_distance <= radius, 'Float32 projection exceeds original bound; retain stop'
    return actual, dict(unconstrained_distance=distance, actual_distance=actual_distance,
        radius=radius, scale=scale, float32_roundoff_reserve=reserve,
        bound_reference='original selected vector, never previous step', scientific_bound_relaxed=False)


def stage_gate(before, after):
    assert set(before) == set(after) and len(before) == 17
    failures = []
    for key in sorted(before):
        old, new = before[key], after[key]
        assert old['cases'] == new['cases']
        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            a, b = new[metric], old[metric]
            assert math.isfinite(a) and math.isfinite(b)
            bad = a > b+1e-12 if metric == 'MSE' else a < b-1e-6
            if bad:
                failures.append(dict(group=key, metric=metric, baseline=b, candidate=a))
    old, new = before['degraded'], after['degraded']
    assert old['landmark_high_frequency_MSE'] > 0
    gain = 1-new['landmark_high_frequency_MSE']/old['landmark_high_frequency_MSE']
    sources = {k: 1-after[k]['landmark_high_frequency_MSE']/before[k]['landmark_high_frequency_MSE']
               for k in before if k.endswith('/degraded')}
    assert len(sources) == 2 and math.isfinite(gain) and all(math.isfinite(v) for v in sources.values())
    brightness = max(0., old['MSE']-new['constant_mean_shift_only_MSE'])/max(old['MSE']-new['MSE'], 1e-12)
    if brightness > .2:
        failures.append(dict(group='degraded', metric='brightness_gain_fraction', candidate=brightness, maximum=.2))
    return dict(**{'pass': not failures and gain >= .01 and all(v >= .01 for v in sources.values())},
        global_structure_gain=gain, source_structure_gains=sources, brightness_gain_fraction=brightness,
        preservation_failures=failures, scientific_thresholds=THRESHOLDS,
        numerical_sample_gate_only=True, model_qualified=False, automatic_extra_epochs=False)


def validate_protocol(p):
    assert p['name'] == NAME and p['parent_protocol_sha256'] == PARENT_PIN
    assert p['original_checkpoint_sha256'] == CHECKPOINT and p['scientific_thresholds'] == THRESHOLDS
    assert p['budgets'] == BUDGETS and p['selected_routes'] == ['A', 'B']
    assert type(p['updates_per_route']) is int and p['updates_per_route'] == 50
    assert p['routes'] == {'A': {'tensors': 15, 'elements': 609219}, 'B': {'tensors': 25, 'elements': 2176419}}
    assert len(p['case_ids']) == len(set(p['case_ids'])) == 100 and len(p['reference_blocks']) == 20
    assert len(p['schedule']) == 50 and set(p['schedule'][:20]) == set(range(20))
    assert p['schedule'][20:40] == p['schedule'][:20] and p['schedule'][40:] == p['schedule'][:10]
    assert all(p['reference_blocks'][i]['source'] != p['reference_blocks'][j]['source']
               for i, j in zip(p['schedule'][::2], p['schedule'][1::2]))
    assert p['conditioning_trainable'] is False and p['observation_trainable'] is False
    assert p['prior_trainable'] is False and p['normalization_trainable'] is False
    assert p['B_readout_kernel'] == 3 and p['initial_parity_policy'] == 'exact_RAW_and_PNG_all100'
    assert p['optimizer'] == dict(name='Adam', lr=.0001, betas=[.9, .999], eps=1e-8,
        weight_decay=1e-5, gradient_clip_L2=1., projection='common_original_ball_after_each_step')
    assert p['distance_fraction_maximum'] == .001 and p['common_radius_basis'] == 'minimum_initial_A_B_historical_conditioning_L2'
    assert p['unconstrained_proposal_execution'] == 'shadow_parameters_never_model_forward'
    assert p['reconstruction_weights'] == [.2, 1., 1., 1.] and p['regression_barrier_coefficient'] == 5
    assert p['photometric_coefficient'] == 10. and p['brightness_fraction_maximum'] == .2
    assert p['paired_TRAIN_cases'] == 100 and p['full_corpus_epochs'] == 0
    for key in ['native_pixels_used', 'DEV_pixels_used', 'final_identity_pixels', 'completion_forwards']:
        assert p[key] == 0
    for key in ['app_promotion', 'automatic_follow_on', 'failed_recipe_resume_allowed', 'agent_training_allowed']:
        assert p[key] is False
    return p


def verify_transfer(root, parent, pin):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    assert sha(root/'protocol.json') == pin
    p = validate_protocol(read(root/'protocol.json'))
    assert sha(parent/'protocol.json') == PARENT_PIN
    old = read(parent/'protocol.json')
    assert p['case_ids'] == old['preview_case_ids']
    assert p['normalizers'] == old['normalizers'] and p['active_anchor_normalizers'] == old['active_anchor_normalizers']
    assert p['scientific_thresholds'] == old['scientific_thresholds']
    verify_paths(root, p['assets_sha256'])
    verify_paths(parent, p['parent_required_sha256'])
    cases = {c['id']: c for c in old['cases']}
    for block in p['reference_blocks']:
        selected = [cases[c] for c in block['case_ids']]
        assert len(selected) == 5 and [c['profile'] for c in selected] == PROFILES
        assert all(c['role'] == 'train' and c['reference_id'] == block['reference'] for c in selected)
    return p, old

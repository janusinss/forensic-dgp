"""Finite manual measurement of one retained decoder direction; no new learning."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import sys

NAME = 'cctv_dgp_decoder_activation_v1_vm'
STEM = 'cctv-dgp-decoder-activation-v1'
STAGE = 'outputs/decoder_activation_v1'
REPAIR_NAME = 'cctv_dgp_decoder_hinge_repair_v1_vm'
SOURCE_NAME = 'cctv_dgp_spatial_readout_v1_vm'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
REPLAY_NAME = 'cctv_dgp_spatial_path_replay_v1_vm'
REPAIR_PIN = 'a4d57c933214345a6d1cab5639007d15599f85e0db0450a0aad11aea5d715518'
SOURCE_PIN = '3bfbaf4b4a519f2926f0503731cd9d22390a7fc0c09da20e2f022e6a1b93c97e'
PARENT_PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
REPLAY_PIN = '6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
CORRECTED_HELPER_SHA = '8368f97c13b4703834c6ea249a85a6970322dce45416bfecd23d8a299f6706f8'
STATES = ['initial', 'probe_1_16', 'probe_1_4', 'fit_u001']
AMPLITUDES = [0., 1/16, 1/4, 1.]
BUDGETS = dict(worker_seconds=1200, external_seconds=1230, cache_seconds=240,
    gradient_seconds=240, snapshot_seconds=180, saved_audit_seconds=300,
    saved_audit_external_seconds=330, export_seconds=300, export_external_seconds=330,
    kill_grace_seconds=30, minimum_free_bytes=3*1024**3, disk_reserve_bytes=1024**3,
    available_RAM_before_import_bytes=8*1024**3, output_byte_cap=768*1024**2,
    emergency_metadata_bytes=8*1024**2,
    peak_VRAM_bytes=20*1024**3, peak_RSS_bytes=12*1024**3,
    maximum_optimizer_updates=0, maximum_backwards=0, maximum_gradient_queries=70,
    original_batch_calls=20, prior_feature_batch_calls=20,
    decoder_batch_calls=82, recognizer_batch_calls=82)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_paths(folder, bindings):
    folder = Path(folder).resolve()
    for name, expected in bindings.items():
        relative = PurePosixPath(name)
        assert isinstance(name, str) and ':' not in name and '\\' not in name
        assert not relative.is_absolute() and '..' not in relative.parts and str(relative) == name
        path = folder/name
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert stat.S_ISREG(path.stat().st_mode) and sha(path) == expected, name


def validate(p, design):
    assert all(p.get(key) == value for key, value in design.items()), 'Frozen design differs'
    assert p['name'] == NAME and p['repair_protocol_sha256'] == REPAIR_PIN
    assert p['source_protocol_sha256'] == SOURCE_PIN and p['parent_protocol_sha256'] == PARENT_PIN
    assert p['replay_protocol_sha256'] == REPLAY_PIN and p['checkpoint_sha256'] == CHECKPOINT
    assert p['state_labels'] == STATES and p['retained_amplitudes'] == AMPLITUDES
    assert p['budgets'] == BUDGETS and p['derivative_reference_indices'] == [12, 4]
    assert p['derivative_reference_ids'] == ['tr_asian_00176', 'tr_ffhq_53945']
    assert p['gradient_layout'] == [
        *[f'MSE_{i}' for i in range(5)], *[f'SSIM_loss_{i}' for i in range(5)],
        *[f'ArcFace_loss_{i}' for i in range(5)], *[f'landmark_structure_{i}' for i in range(1,5)],
        *[f'channel_mean_{i}_{c}' for i in range(5) for c in range(3)], 'objective']
    assert len(p['gradient_layout']) == 35 and p['trainable_elements'] == 609219
    assert p['trainable_tensors'] == 15 and p['common_radius_L2'] == 0.04018470287178282
    assert p['new_optimizer_updates'] == p['new_parameter_proposals'] == p['new_full_corpus_epochs'] == 0
    assert p['scientific_thresholds'] == dict(structure_gain=.01, brightness_gain_fraction=.2,
                                             MSE_slack=1e-12, SSIM_ArcFace_slack=1e-6)
    assert len(p['case_ids']) == len(set(p['case_ids'])) == 100 and len(p['reference_blocks']) == 20
    for key in ['automatic_launch', 'failed_recipe_resume_allowed', 'model_qualified',
                'new_incumbent_selection', 'local_neural_construction_allowed', 'direction_search',
                'new_amplitudes', 'native_pixels_used', 'DEV_pixels_used', 'final_identity_pixels']:
        assert p[key] is False, key
    assert p['diagnostic_only'] is True and p['training_objective_batch'] == 5
    assert p['historical_diagnostic_embedding_batch'] == 15
    assert p['clipped_gradient_replay_bound'] == dict(max_abs=2e-7, cosine_min=.999999)
    assert p['objective_replay_max_abs'] == 1e-10
    assert p['gradient_norm_metadata_bound'] == dict(rtol=2e-6, atol=1e-12)
    assert p['clip_factor_metadata_max_ULPs'] == 2


def verify_transfer(root, repair, source, parent, replay, pin):
    import numpy as np
    root, repair, source, parent, replay = [Path(path).resolve() for path in [root,repair,source,parent,replay]]
    p, design = read(root/'protocol.json'), read(root/'design.json')
    assert sha(root/'protocol.json') == pin and sha(root/'design.json') == p['design_sha256']
    validate(p, design)
    for folder, field in [(root,'assets_sha256'), (repair,'repair_required_sha256'),
                          (source,'source_required_sha256'), (parent,'parent_required_sha256'),
                          (replay,'replay_required_sha256')]:
        verify_paths(folder, p[field])
    repaired, old, data = read(repair/'protocol.json'), read(source/'protocol.json'), read(parent/'protocol.json')
    for folder, expected in [(repair,REPAIR_PIN), (source,SOURCE_PIN), (parent,PARENT_PIN), (replay,REPLAY_PIN)]:
        assert sha(folder/'protocol.json') == expected
    assert sha(repair/'cctv_dgp_bank_hinge_boundary_v1_loss.py') == CORRECTED_HELPER_SHA
    assert p['case_ids'] == repaired['case_ids'] == old['case_ids'] == data['preview_case_ids']
    assert p['reference_blocks'] == repaired['reference_blocks'] == old['reference_blocks']
    assert p['derivative_reference_indices'] == repaired['schedule'][:2]
    assert p['trainable_parameter_names'] == old['trainable_parameter_names']['A']
    result = read(repair/'outputs/decoder_hinge_repair_v1/run_results.json')
    assert result['complete'] and result['training_updates'] == 1 and result['stop_reason'] == 'quality_stop'
    vectors = [np.load(repair/f'outputs/decoder_hinge_repair_v1/{label}_vector.npy',allow_pickle=False) for label in STATES]
    assert all(v.dtype == np.float32 and v.shape == (609219,) and np.isfinite(v).all() for v in vectors)
    assert np.array_equal(vectors[0],np.load(source/'outputs/spatial_readout_v1/A_initial_vector.npy',allow_pickle=False))
    for vector in vectors[1:]:
        assert 0 < float(np.linalg.norm(vector.astype(np.float64)-vectors[0].astype(np.float64))) <= p['common_radius_L2']
    cases = {case['id']: case for case in data['cases']}
    for block in p['reference_blocks']:
        assert len(block['case_ids']) == 5
        assert all(cases[c]['role'] == 'train' and cases[c]['reference_id'] == block['reference'] for c in block['case_ids'])
        assert [cases[c]['profile'] for c in block['case_ids']] == ['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
    return p, repaired, old, data


def require_manual(root):
    assert platform.system() == 'Linux', 'Human manual Linux L4 tmux only'
    assert platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert Path(root).resolve() == (Path.home()/'forensic-dgp'/NAME).resolve()
    assert Path.cwd().resolve() == Path(root).resolve() and os.environ.get('TMUX')
    assert sys.prefix == str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')

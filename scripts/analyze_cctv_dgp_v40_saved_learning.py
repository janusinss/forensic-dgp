"""Analyze audited saved weights/gradients only; no model construction or derivatives."""
import ast
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from cctv_dgp_spatial_fit_v40_contract import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v40_saved_learning_analysis'
BUNDLE = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'
RETURNED = ROOT/'outputs/cctv_dgp_spatial_fit_v40_return'
GRADIENT = ROOT/'outputs/cctv_dgp_spatial_decoder_v39_return'


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        assert not value.requires_grad and value.grad is None
        value = value.detach().cpu().contiguous()
        h.update(name.encode()+b'\0'+str(value.dtype).encode()+b'\0')
        h.update(str(tuple(value.shape)).encode()+b'\0'+value.numpy().tobytes())
    return h.hexdigest()


def main():
    start = time.monotonic(); assert not OUT.exists()
    p = read(BUNDLE/'protocol.json'); a = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json')
    old = read(ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39/protocol.json')
    old_audit = read(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json')
    assert a['complete'] and a['failure_retained'] and not a['necessary_capacity_pass'] and old_audit['complete']
    assert p['terms'] == old['terms'] and p['parameter_layout'] == old['parameter_layout']
    c39, c40 = [read(path) for path in [GRADIENT/'outputs/cohort_loss_setup.json', RETURNED/'outputs/cohort_loss_setup.json']]
    assert all(c39[k] == c40[k] for k in ['feature_normalizer', 'interior_normalizer', 'normalizer_floor', 'degraded_weight', 'clear_baseline_anchor_weight'])
    old_cases = {c['id']: c for c in old['cases']}; cases = {c['id']: c for c in p['cases']}
    assert list(old_cases) == p['preview_case_ids']
    for cid in p['preview_case_ids']:
        for key in ['input', 'target', 'observed']:
            assert sha(BUNDLE/cases[cid][key]) == sha(ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39'/old_cases[cid][key])
        assert cases[cid]['landmarks5_canvas_xy'] == old_cases[cid]['landmarks5_canvas_xy']
    seed_path = BUNDLE/'untrained_initial_decoder.pth'
    after_path = RETURNED/'outputs/update50/spatial_decoder.pth'; stop_path = RETURNED/'outputs/stopped_spatial_decoder.pth'
    states = [torch.load(path, map_location='cpu', weights_only=True) for path in [seed_path, after_path, stop_path]]
    initial, current, stopped = states
    assert set(initial) == set(current) == set(stopped) == {row['name'] for row in p['parameter_layout']}
    assert state_digest(initial) == p['initial_states']['decoder']
    assert state_digest(current) == state_digest(stopped) == read(RETURNED/'outputs/update50/metrics.json')['decoder_state']
    partitions = []; pieces = []
    for row in p['parameter_layout']:
        name = row['name']; arrays = [s[name].numpy() for s in states]
        assert all(v.shape == tuple(row['shape']) and v.dtype == np.float32 and np.isfinite(v).all() for v in arrays)
        np.testing.assert_array_equal(arrays[1], arrays[2])
        delta = (arrays[1].astype(np.float64)-arrays[0].astype(np.float64)).reshape(-1); pieces.append(delta)
        partitions.append({'name': name, 'changed_values': int(np.count_nonzero(delta)), 'values': len(delta),
                           'change_norm': float(np.linalg.norm(delta)), 'change_RMS': float(np.sqrt(np.square(delta).mean())),
                           'maximum_absolute_change': float(np.abs(delta).max())})
    delta = np.concatenate(pieces); assert delta.shape == (17952,) and all(row['changed_values'] > 0 for row in partitions)
    gradient_path = GRADIENT/'outputs/gradient_components.npy'; gradient = np.load(gradient_path, allow_pickle=False)
    assert gradient.shape == (7, 17952) and gradient.dtype == np.float64 and np.isfinite(gradient).all()
    assert sha(gradient_path) == read(GRADIENT/'outputs/gradient_summary.json')['gradient_sha256']
    observations = []
    for name, g in zip(p['terms'], gradient):
        norm = float(np.linalg.norm(g)); dot = float(g@delta)
        observations.append({'term': name, 'initial_gradient_norm': norm, 'gradient_dot_weight_change': dot,
                             'negative_gradient_alignment': -dot/(norm*float(np.linalg.norm(delta))) if norm else None})
    OUT.mkdir(); np.save(OUT/'parameter_delta.npy', delta, allow_pickle=False)
    sources = [Path(__file__), BUNDLE/'protocol.json', ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39/protocol.json',
               ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json', ROOT/'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json',
               seed_path, after_path, stop_path, gradient_path, GRADIENT/'outputs/gradient_summary.json',
               GRADIENT/'outputs/cohort_loss_setup.json', RETURNED/'outputs/cohort_loss_setup.json', RETURNED/'outputs/update50/metrics.json']
    assert time.monotonic()-start < 120
    write(OUT/'results.json', {'complete': True, 'sources_sha256': {q.relative_to(ROOT).as_posix(): sha(q) for q in sources},
          'initial_state': state_digest(initial), 'stopped_state': state_digest(stopped), 'stop_and_snapshot50_tensors_exact': True,
          'parameter_layout': p['parameter_layout'], 'tensors_changed': 57, 'parameter_values': 17952,
          'parameter_delta_norm': float(np.linalg.norm(delta)), 'delta_sha256': sha(OUT/'parameter_delta.npy'), 'partitions': partitions,
          'initial_gradient_observations': observations, 'normalizers_exact': True, 'all50_initial_gradient_data_inputs_targets_support_exact': True,
          'scope': 'Same fixed50 TRAIN cohort at initial state; actual finite weight change across50 updates. Not per-step AdamW or stopped-state gradients.',
          'limitation': 'Initial tangent projections do not reconstruct optimizer moments, nonlinear loss changes or establish a unique failure cause.',
          'learning_failure_unchanged': True, 'new_model_constructions': 0, 'new_model_forwards': 0,
          'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0, 'app_promotion': False,
          'native_or_reserved_used': False, 'independent_final_review': False, 'goal_complete': False,
          'seconds': time.monotonic()-start, 'cap_seconds': 120})
    print({'complete': True, 'all57_tensors_changed': True, 'change_norm': float(np.linalg.norm(delta)), 'observations': observations}, flush=True)


if __name__ == '__main__': main()

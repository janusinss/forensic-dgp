"""Independent saved checkpoint delta and local-tangent arithmetic verification."""
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from cctv_dgp_spatial_fit_v40_contract import read, write, sha
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v40_saved_learning_analysis'


def main():
    start = time.monotonic(); r = read(OUT/'results.json'); assert r['complete']
    for name, value in r['sources_sha256'].items(): assert sha(ROOT/name) == value
    initial = torch.load(ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    stopped = torch.load(ROOT/'outputs/cctv_dgp_spatial_fit_v40_return/outputs/stopped_spatial_decoder.pth', map_location='cpu', weights_only=True)
    snapshot = torch.load(ROOT/'outputs/cctv_dgp_spatial_fit_v40_return/outputs/update50/spatial_decoder.pth', map_location='cpu', weights_only=True)
    h0 = hashlib.sha256(); h1 = hashlib.sha256()
    for name in sorted(initial):
        for state, h in [(initial, h0), (stopped, h1)]:
            v = state[name]; assert not v.requires_grad and v.grad is None
            h.update(name.encode()+b'\0'+str(v.dtype).encode()+b'\0'+str(tuple(v.shape)).encode()+b'\0'+v.contiguous().numpy().tobytes())
        assert torch.equal(stopped[name], snapshot[name])
    assert h0.hexdigest() == r['initial_state'] and h1.hexdigest() == r['stopped_state']
    delta = np.load(OUT/'parameter_delta.npy', allow_pickle=False); assert sha(OUT/'parameter_delta.npy') == r['delta_sha256']
    assert delta.dtype == np.float64 and delta.shape == (17952,)
    for layout, reported in zip(r['parameter_layout'], r['partitions']):
        assert layout['name'] == reported['name']
        values = stopped[layout['name']].numpy().astype(np.float64)-initial[layout['name']].numpy().astype(np.float64)
        block = delta[layout['start']:layout['end']]; np.testing.assert_array_equal(block, values.reshape(-1))
        assert reported['changed_values'] == int(np.count_nonzero(block)) > 0 and reported['values'] == len(block)
        assert reported['change_norm'] == float(np.linalg.norm(block)) and reported['change_RMS'] == float(np.sqrt(np.square(block).mean()))
        assert reported['maximum_absolute_change'] == float(np.abs(block).max())
    assert r['parameter_delta_norm'] == float(np.linalg.norm(delta)) and r['tensors_changed'] == 57
    g = np.load(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_return/outputs/gradient_components.npy', allow_pickle=False)
    for row, vector in zip(r['initial_gradient_observations'], g):
        norm = float(np.linalg.norm(vector)); dot = float(vector@delta)
        assert row['initial_gradient_norm'] == norm and row['gradient_dot_weight_change'] == dot
        expected = -dot/(norm*float(np.linalg.norm(delta))) if norm else None
        assert row['negative_gradient_alignment'] == expected
    assert r['new_model_forwards'] == r['new_model_constructions'] == r['local_gradient_calls'] == r['local_optimizer_updates'] == r['VM_calls'] == 0
    assert r['learning_failure_unchanged'] and not r['app_promotion'] and not r['goal_complete'] and time.monotonic()-start < 120
    write(OUT/'independent_analysis_audit.json', {'complete': True, 'checker_sha256': sha(Path(__file__)), 'results_sha256': sha(OUT/'results.json'),
          'all57_stop_snapshot_tensors_and17952_delta_values_verified': True, 'all7_initial_tangent_rows_exact': True,
          'not_an_optimizer_trajectory_reconstruction': True, 'local_model_constructions': 0, 'local_model_forwards': 0,
          'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0, 'app_promotion': False,
          'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 120})
    print({'complete': True, 'parameters_checked': 17952, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()

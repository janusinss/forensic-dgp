"""Recompute all empirical margins and full-row projection certificates."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v37_delivered_margins_v1'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    import numpy as np
    a = read(OUT / 'analysis.json'); assert a['complete'] and a['full_rows'] == 210
    for name, digest in a['basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, digest in a['artifact_sha256'].items(): assert sha(OUT / name) == digest, name
    p = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/protocol.json')
    rawroot = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1'
    pngroot = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1'
    raw = np.load(rawroot / 'group_guard_matrix.npy', allow_pickle=False, mmap_mode='r')
    png = np.load(pngroot / 'PNG_group_guard_matrix.npy', allow_pickle=False, mmap_mode='r')
    extras = [np.load(ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state0_{c["name"]}/gradient_components.npy',
                     allow_pickle=False, mmap_mode='r')[:3] for c in p['cohorts']]
    full = np.concatenate([raw, *extras, png]); assert full.shape == (210, 978243)
    proposal = np.load(rawroot / 'mean_original_displacement.npy', allow_pickle=False)
    theta = np.load(rawroot / 'theta_before.npy', allow_pickle=False)
    v36direction = np.load(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/projected_displacement.npy', allow_pickle=False)
    old = np.asarray(read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/projection.json')['clearance_targets'], np.float64)
    targets = np.concatenate([old, np.zeros(102, np.float64)]); recalibrated = 0
    for ci, cohort in enumerate(p['cohorts']):
        root = ROOT / f'outputs/cctv_dgp_finite_clearance_probe_v36_return/outputs/state0_{cohort["name"]}'
        before = read(root / 'before/receipt.json')
        rows = a['constraint_labels'][108 + 51 * ci:108 + 51 * (ci + 1)]
        maxima = np.zeros(51)
        for variant, scale in [('clearance_1', 1.), ('clearance_half', .5), ('clearance_quarter', .25), ('clearance_eighth', .125)]:
            after = read(root / variant / 'receipt.json')
            effective = theta.astype(np.float64) - (theta.astype(np.float64) - scale * v36direction).astype(np.float32).astype(np.float64)
            linear = -(png[51 * ci:51 * (ci + 1)] @ effective)
            for index, row in enumerate(rows):
                key = {'PNG_MSE': 'MSE', 'one_minus_PNG_SSIM': 'SSIM', 'one_minus_PNG_ArcFace': 'ArcFace_observed_fixed'}[row['metric']]
                before_value, after_value = before['groups'][row['group']][key], after['groups'][row['group']][key]
                finite = after_value - before_value if key == 'MSE' else before_value - after_value
                maxima[index] = max(maxima[index], (finite - linear[index]) / scale)
                recalibrated += 1
        targets[108 + 51 * ci:108 + 51 * (ci + 1)] = 2 * maxima
    saved = np.load(OUT / 'clearance_targets.npy', allow_pickle=False)
    assert np.allclose(saved, targets, rtol=2e-10, atol=1e-11)
    assert np.array_equal(saved[:108], old) and np.count_nonzero(saved[:108]) == 65
    assert np.count_nonzero(saved[108:]) == 97 and recalibrated == 408
    proof = a['projection']; assert proof['geometry_qualified_for_disposable_probe']
    direction = np.load(OUT / 'candidate_displacement.npy', allow_pickle=False)
    assert direction.shape == proposal.shape == (978243,) and direction.dtype == np.float64
    norms = np.linalg.norm(full, axis=1); assert (norms > 0).all()
    unit = full / norms[:, None]; multipliers = np.asarray(proof['multipliers'], np.float64)
    slack = unit @ direction - targets / norms
    stationarity = float(np.linalg.norm(direction - proposal - multipliers @ unit))
    assert multipliers.shape == (210,) and (multipliers >= 0).all()
    assert stationarity <= 1e-12 and slack.min() >= -proof['KKT_tolerance']
    assert float(np.max(np.abs(multipliers * slack))) <= 1e-12
    assert np.allclose(full @ direction, proof['raw_dots_after'], rtol=2e-10, atol=1e-11)
    assert np.linalg.norm(direction) / np.linalg.norm(proposal) <= 2
    assert proof['independent_primal_dual_relative_L2_error'] <= 2e-7
    for row in a['float32_copy_checks']:
        effective = theta.astype(np.float64) - (theta.astype(np.float64) - row['scale'] * direction).astype(np.float32).astype(np.float64)
        predicted = -(full @ effective)
        assert np.allclose(predicted[:102], row['linear_guard_loss_changes'], rtol=2e-10, atol=1e-11)
        assert np.allclose(predicted[102:108], row['linear_restoration_loss_changes'], rtol=2e-10, atol=1e-11)
        assert np.allclose(predicted[108:], row['coarse_PNG_loss_changes'], rtol=2e-10, atol=1e-11)
    del full, unit
    from cctv_dgp_delivered_margin_geometry_v38 import project_affine
    desired = np.array([.2, .2]); guards = np.array([[1., 0.], [0., 1.]])
    margin = np.array([.3, .4]); value, certificate = project_affine(guards, desired, margin)
    assert np.allclose(value, [.3, .4], atol=1e-10) and certificate['geometry_qualified_for_disposable_probe']
    changed, _ = project_affine(guards[[1, 0]] * np.array([[3.], [7.]]), desired, margin[[1, 0]] * np.array([3., 7.]))
    assert np.allclose(changed, value, atol=1e-10)
    infeasible, blocked = project_affine(np.array([[1., 0.], [-1., 0.]]), desired, np.array([.1, .1]))
    assert infeasible is None and not blocked['linear_feasible'] and not blocked['geometry_qualified_for_disposable_probe']
    rejected = 0
    for g, q, t in [(np.zeros((1, 2)), desired, np.zeros(1)), (guards, desired, np.array([-.1, 0.])),
                    (guards, np.array([np.nan, .2]), np.zeros(2)), (np.ones((211, 2)), desired, np.zeros(211))]:
        try: project_affine(g, q, t)
        except ValueError: rejected += 1
        else: raise AssertionError('Unsafe geometry must stop')
    assert rejected == 4 and not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'analysis_sha256': sha(OUT / 'analysis.json'),
               'all210_rows_and408_calibration_observations_verified': True, 'all65_V36_clearances_unchanged': True,
               'new97_PNG_margins_recalibrated': True, 'KKT_stationarity_error': stationarity,
               'minimum_direct_unit_clearance_residual': float(slack.min()), 'magnitude_limit_retained': True,
               'positive_scaling_reorder_and_infeasibility_fixtures': 3, 'invalid_geometry_rejections': rejected,
               'geometry_qualified_for_disposable_probe': True, 'finite_preservation_implied': False,
               'neural_calls': 0, 'local_gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0,
               'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - started, 'cap_seconds': 300}
    assert receipt['seconds'] < 300
    with (OUT / 'independent_geometry_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ['complete', 'KKT_stationarity_error', 'minimum_direct_unit_clearance_residual', 'seconds']}), flush=True)


if __name__ == '__main__': main()

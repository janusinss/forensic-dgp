"""Check an affine guard design using saved arrays only; no neural training or gradients."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v35_finite_clearance_v1'
BASIS = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1'
V35 = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import numpy as np
    from scipy.optimize import LinearConstraint, linprog, minimize, nnls
    started = time.monotonic()
    assert not OUT.exists()
    checked = read(V35 / 'independent_analysis_page_audit.json')
    assert checked['complete'] and checked['all700_cells_exact']
    a = read(BASIS / 'analysis.json')
    finite = read(V35 / 'analysis.json')
    p = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_vm/protocol.json')
    assert finite['jointly_eligible_subset_variants'] == []
    for name, digest in a['bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    matrix = np.load(BASIS / 'group_guard_matrix.npy', allow_pickle=False)
    extra = [np.load(ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state0_{c["name"]}/gradient_components.npy', allow_pickle=False)[:3]
             for c in p['cohorts']]
    gradients = np.concatenate([matrix, *extra], axis=0)
    proposal = np.load(BASIS / 'mean_original_displacement.npy', allow_pickle=False)
    theta = np.load(BASIS / 'theta_before.npy', allow_pickle=False)
    assert gradients.shape == (108, 978243) and proposal.shape == theta.shape == (978243,)
    norms = np.linalg.norm(gradients, axis=1)
    assert np.isfinite(gradients).all() and (norms > 0).all()
    targets, calibration = np.zeros(108, np.float64), []
    for index, label in enumerate(a['constraint_labels'][:102]):
        if label['metric'] == 'one_minus_raw_SSIM':
            continue  # skimage PNG SSIM and the VM derivative are different functions.
        row = next(r for r in finite['variants'] if r['cohort'] == label['cohort'] and r['variant'] == 'joint_1')
        raw = next(r for r in row['first_order_to_finite_raw_checks'] if r['group'] == label['group'] and r['metric'] == label['metric'])
        folder = ROOT / f'outputs/cctv_dgp_group_guard_probe_v35_r1_return/outputs/state0_{label["cohort"]}'
        old = read(folder / 'before/receipt.json')['groups'][label['group']]
        new = read(folder / 'joint_1/receipt.json')['groups'][label['group']]
        key = 'MSE' if label['metric'] == 'raw_MSE' else 'ArcFace_observed_fixed'
        png_change = new[key] - old[key] if key == 'MSE' else old[key] - new[key]
        residual = max(0., raw['finite_minus_linear'], png_change - raw['linear_loss_change'])
        targets[index] = 2 * residual
        calibration.append({**label, 'linear_loss_change': raw['linear_loss_change'],
            'finite_raw_loss_change': raw['finite_loss_change'], 'finite_PNG_loss_change': png_change,
            'measured_residual': residual, 'required_linear_clearance': float(targets[index])})
    length = float(np.linalg.norm(proposal))
    unit = gradients / norms[:, None]
    gram = unit @ unit.T
    values, vectors = np.linalg.eigh(gram)
    keep = values > 1e-12
    assert values.min() >= -2e-10
    A = vectors[:, keep] * np.sqrt(values[keep])[None, :]
    dots = unit @ (proposal / length)
    y = (vectors[:, keep].T @ dots) / np.sqrt(values[keep])
    b = targets / (norms * length)
    assert np.max(np.abs(A @ y - dots)) <= 2e-10
    feasibility = linprog(np.zeros(len(y)), A_ub=-A, b_ub=-b,
        bounds=[(None, None)] * len(y), method='highs', options={'maxiter': 2000, 'time_limit': 60})
    result = {'complete': True, 'policy': 'Empirical finite-clearance affine design, not a validated loss bound',
        'original_rows': 108, 'guard_rows': 102, 'rank': int(keep.sum()), 'calibration': calibration,
        'nonzero_clearance_rows': int(np.count_nonzero(targets)), 'linear_feasible': bool(feasibility.success),
        'LP_status': int(feasibility.status), 'LP_message': str(feasibility.message),
        'SSIM_and_restoration_rows_retained_with_zero_clearance': True,
        'finite_preservation_implied': False, 'geometry_qualified_for_disposable_probe': False,
        'optimizer_updates': 0, 'neural_or_gradient_calls': 0, 'new_VM_calls': 0,
        'original_checkpoints_modified': False, 'native_or_reserved_used': False,
        'training_capacity_pass': False, 'app_promotion': False, 'goal_complete': False}
    OUT.mkdir()
    np.save(OUT / 'clearance_targets.npy', targets, allow_pickle=False)
    if feasibility.success:
        primal = minimize(lambda z: .5 * float((z - y) @ (z - y)), feasibility.x,
            jac=lambda z: z - y, constraints=[LinearConstraint(A, b, np.inf)],
            method='SLSQP', options={'ftol': 1e-14, 'maxiter': 2000})
        slack = A @ primal.x - b
        active = np.flatnonzero(slack <= 2e-8)
        multipliers = np.zeros(108)
        multipliers[active], stationarity = nnls(A[active].T, primal.x - y, maxiter=10800)
        # Separate dual formulation, different variables and feasible initial point.
        dual = minimize(lambda z: .5 * float(z @ gram @ z) + float((dots - b) @ z), np.zeros(108),
            jac=lambda z: gram @ z + dots - b, bounds=[(0., None)] * 108,
            method='SLSQP', options={'ftol': 1e-14, 'maxiter': 2000})
        independent_error = float(np.linalg.norm(primal.x - (y + A.T @ dual.x)))
        candidate = proposal + length * ((primal.x - y) @ np.linalg.pinv(A) @ unit)
        actual = unit @ candidate - targets / norms
        complementarity = float(np.max(np.abs(multipliers * slack)))
        stable = bool(primal.success and dual.success and slack.min() >= -2e-10
                      and stationarity <= 2e-10 and complementarity <= 2e-10
                      and independent_error <= 2e-7 and actual.min() >= -2e-10 * length)
        magnitude_ratio = float(np.linalg.norm(candidate) / length)
        roundoff = []
        for scale in [1., .5, .25, .125]:
            copied = theta.astype(np.float64) - (theta.astype(np.float64) - scale * candidate).astype(np.float32).astype(np.float64)
            residual = gradients @ copied - scale * targets
            roundoff.append({'scale': scale, 'minimum_raw_clearance_residual': float(residual.min()),
                             'maximum_float32_linear_guard_loss_change': float((-(gradients @ copied)).max())})
        result.update({'primal_success': bool(primal.success), 'dual_success': bool(dual.success),
            'primal_iterations': int(primal.nit), 'dual_iterations': int(dual.nit),
            'minimum_normalized_slack': float(slack.min()), 'KKT_stationarity_relative_error': float(stationarity),
            'KKT_complementarity_relative_error': complementarity, 'independent_primal_dual_relative_L2_error': independent_error,
            'candidate_magnitude_ratio': magnitude_ratio, 'maximum_permitted_magnitude_ratio': 2.,
            'geometry_qualified_for_disposable_probe': stable and magnitude_ratio <= 2.,
            'float32_copy_checks': roundoff, 'multipliers': multipliers.tolist()})
        np.save(OUT / 'candidate_displacement.npy', candidate, allow_pickle=False)
    result['basis_sha256'] = {q.relative_to(ROOT).as_posix(): sha(q) for q in [
        BASIS / 'analysis.json', BASIS / 'group_guard_matrix.npy', BASIS / 'mean_original_displacement.npy',
        BASIS / 'theta_before.npy', V35 / 'analysis.json', V35 / 'independent_analysis_page_audit.json',
        Path(__file__)]}
    result['artifact_sha256'] = {q.name: sha(q) for q in OUT.glob('*.npy')}
    result['seconds'] = time.monotonic() - started
    assert result['seconds'] <= 300
    with (OUT / 'analysis.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['complete', 'linear_feasible', 'geometry_qualified_for_disposable_probe', 'seconds']}))


if __name__ == '__main__':
    main()

"""Saved-gradient arithmetic only: no models, neural calls or parameter updates."""
from datetime import datetime, timezone
import time

import numpy as np
from scipy.optimize import nnls, minimize

from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write


def main():
    start = time.monotonic()
    audit_path = ANALYSIS / 'independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['protocol_sha256'] == PIN
    assert audit['all32_aggregate_vector_norm_dot_mean_and_loss_bindings_checked']
    out = RETURN / 'outputs/group_gradients_v1_r1'
    run_path = out / 'run_results.json'
    assert sha(run_path) == audit['run_results_sha256']
    run = read(run_path)
    dest = ANALYSIS / 'common_direction'
    assert not dest.exists(), 'Retain earlier arithmetic'
    groups = run['groups']
    assert len(groups) == 32 and run['parameters'] == 3364643
    vectors, bindings = [], []
    for row in groups:
        path = out / row['file']
        assert sha(path) == row['sha256']
        vector = np.load(path, allow_pickle=False, mmap_mode='r')
        assert vector.shape == (run['parameters'],) and vector.dtype == np.float64
        vectors.append(vector)
        bindings.append(dict(key=row['key'], sha256=row['sha256'], file=row['file']))
    norms = np.array([r['gradient_norm'] for r in groups], dtype=np.float64)
    losses = np.array([r['value'] for r in groups], dtype=np.float64)
    assert np.all(norms > 0) and np.all(losses > 0)
    gram = np.zeros((32, 32), dtype=np.float64)
    for begin in range(0, run['parameters'], 65536):
        block = np.stack([v[begin:begin+65536] for v in vectors]) / norms[:, None]
        assert np.isfinite(block).all()
        gram += block @ block.T
    assert np.allclose(gram, gram.T, rtol=0, atol=1e-12)
    assert np.allclose(np.diag(gram), 1, rtol=0, atol=1e-10)
    eigen = np.linalg.eigvalsh(gram)
    assert eigen[0] > 1e-9, 'No numerically independent common direction; preserve diagnostic'
    # Minimum L2 direction with one unit of fractional structure improvement
    # for EACH source and a tenth-unit preservation margin for ALL 30 groups.
    # These are first-order design targets, never relaxed output-quality gates.
    fractions = np.array([1 if r['metric'] == 'feature_MSE' else .1 for r in groups])
    targets = fractions * losses / norms
    chol = np.linalg.cholesky(gram)
    multiplier, residual = nnls(chol.T, np.linalg.solve(chol, targets), maxiter=10000)
    products = gram @ multiplier
    slack = targets - products
    assert multiplier.min() >= 0 and slack.max() < 1e-9
    assert np.max(np.abs(multiplier * slack)) < 1e-9
    # A separately formulated primal QP in the gradient span checks the solve.
    objective = lambda c: .5 * float(c @ gram @ c)
    result = minimize(objective, multiplier.copy(), jac=lambda c: gram @ c,
                      method='SLSQP', constraints=[dict(type='ineq', fun=lambda c: gram @ c-targets,
                      jac=lambda c: gram)], options=dict(ftol=1e-12, maxiter=2000))
    assert result.success, result.message
    difference = result.x - multiplier
    discrepancy = float(np.sqrt(max(0, difference @ gram @ difference)))
    assert discrepancy < 1e-6 and np.max(targets-gram@result.x) < 1e-9
    direction = np.empty(run['parameters'], dtype=np.float32)
    for begin in range(0, run['parameters'], 65536):
        block = np.stack([v[begin:begin+65536] for v in vectors]) / norms[:, None]
        direction[begin:begin+65536] = -(multiplier @ block).astype(np.float32)
    actual_dots = np.array([float(v @ direction.astype(np.float64)) for v in vectors])
    fractional_descent = -actual_dots / losses
    assert np.all(fractional_descent >= fractions - 1e-6)
    old_direction = read(RETURN / 'direction_review.json')
    assert old_direction['names'] == run['learning_names']
    shapes = old_direction['shapes']
    sizes = [int(np.prod(s)) for s in shapes]
    assert sum(sizes) == run['parameters']
    offsets = np.cumsum([0] + sizes)
    norm = float(np.linalg.norm(direction.astype(np.float64)))
    initial_norm = old_direction['initial_parameter_norm']
    rates = [.005, .01, .02, .04]
    assert max(rates)*norm/initial_norm < .001, 'Finite proposal exceeds 0.1 percent parameter-norm bound'
    dest.mkdir()
    file = dest / 'directions.npz'
    arrays = {f'balanced_p{i:02d}': direction[offsets[i]:offsets[i+1]].reshape(shape)
              for i, shape in enumerate(shapes)}
    np.savez_compressed(file, **arrays)
    with np.load(file, allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)
        assert all(np.array_equal(archive[k], v) for k, v in arrays.items())
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n) == h for n, h in protected.items())
    records = []
    for i, row in enumerate(groups):
        records.append(dict(key=row['key'], source=row['source'], profile=row['profile'], metric=row['metric'],
            loss=row['value'], gradient_norm=row['gradient_norm'], dot=float(actual_dots[i]),
            fractional_descent_per_unit_rate=float(fractional_descent[i]), target_fraction_per_unit_rate=float(fractions[i]),
            predicted_percent_improvement={str(rate): float(100*rate*fractional_descent[i]) for rate in rates},
            old_direction_dots=row['saved_direction_dot'], first_order_prediction_only=True))
    write(dest / 'direction_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        reviewer_sha256=sha(__file__), audited_return_sha256=sha(audit_path), run_results_sha256=sha(run_path),
        input_protocol_sha256=PIN, aggregate_gradient_bindings=bindings,
        names=run['learning_names'], shapes=shapes, sizes=sizes, parameters=run['parameters'],
        direction_file_sha256=sha(file), direction_file_bytes=file.stat().st_size,
        initial_file_sha256=old_direction['initial_file_sha256'], initial_parameter_norm=initial_norm,
        direction_norm=norm, rate_candidates=rates, normalized_Gram=gram.tolist(), normalized_targets=targets.tolist(),
        dual_multipliers=multiplier.tolist(), smallest_Gram_eigenvalue=float(eigen[0]),
        Gram_condition_number=float(eigen[-1]/eigen[0]), NNLS_residual=float(residual),
        maximum_primal_violation=float(slack.max()), maximum_complementarity_error=float(np.max(np.abs(multiplier*slack))),
        independent_primal_QP_direction_discrepancy=discrepancy, float32_group_predictions=records,
        maximum_prospective_displacement_L2=max(rates)*norm,
        maximum_prospective_displacement_relative_to_parameter_L2=max(rates)*norm/initial_norm,
        old_directions_initially_regressing_group_counts={k: sum(r['saved_direction_dot'][k]>0 for r in groups)
                                                         for k in ['ordinary','constrained']},
        method='minimum L2 vector with source/profile first-order inequality constraints; NNLS dual and SLSQP primal',
        research_basis='https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html',
        distinction_from_MGDA='inequality-constrained design, not the paper implementation or a convergence guarantee',
        no_local_neural_calls_gradients_optimizer_or_parameter_displacement=True,
        scope='20 paired photographic TRAIN references / 100 cases; no native CCTV or final identities',
        finite_forward_validation_required=True, model_qualified=False, more_epochs_qualified=False,
        goal_complete=False, seconds=time.monotonic()-start))
    print(dict(complete=True, groups=32, smallest_eigenvalue=float(eigen[0]), direction_norm=norm,
               maximum_relative_displacement=max(rates)*norm/initial_norm, local_neural_calls=0))


if __name__ == '__main__':
    main()

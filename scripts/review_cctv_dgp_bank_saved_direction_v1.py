"""Project a saved hypothetical direction; no inference, autograd or model update."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
import torch

from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def main():
    dest = OUT / 'cctv_dgp_bank_saved_direction_v1_analysis'
    assert not dest.exists()
    audit_path = ANALYSIS / 'independent_audit.json'
    review_path = ANALYSIS / 'loss_direction_review.json'
    assert read(audit_path)['complete'] and read(review_path)['complete']
    run_path = RETURN / 'outputs/loss_alignment_v1/run_results.json'
    assert sha(run_path) == read(audit_path)['run_results_sha256']
    run = read(run_path)
    torch.set_grad_enabled(False)
    initial_path = PARENT / 'initializers/initial_B.pth'
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    names = run['learning_names']
    shapes = [tuple(initial[name].shape) for name in names]
    sizes = [int(np.prod(s)) for s in shapes]
    offsets = np.cumsum([0]+sizes)
    values = np.concatenate([initial[name].numpy().reshape(-1) for name in names])
    assert values.size == 3364643
    rows = [r for r in run['rows'] if r['state'] == 'initial' and r['sampled_gradient_file']]
    assert len(rows) == 2
    gradients, bindings = [], []
    for row in rows:
        path = RETURN / 'outputs/loss_alignment_v1' / row['sampled_gradient_file']
        with np.load(path, allow_pickle=False) as archive:
            gradients.append([np.concatenate([archive[f'g{j:02d}_p{i:02d}'].reshape(-1) for i in range(37)])
                              for j in range(9)])
        bindings.append(dict(reference=row['reference'], source=row['source'], begin=row['begin'],
                             file=str(path), sha256=sha(path)))
    # This averages two saved reference batches, not the historical five-batch update.
    objective = [0, 1, 2, 3, 5, 6]
    total = sum(sum(g[j] for j in objective) for g in gradients)/np.float32(2)
    norm = float(np.linalg.norm(total.astype(np.float64)))
    clip = np.float32(min(1, 1/(norm+1e-6)))
    effective = total*clip+values*np.float32(1e-5)
    m, v = np.float32(.1)*effective, np.float32(.001)*np.square(effective)
    direction = -(m/np.float32(.1))/(np.sqrt(v/np.float32(.001))+np.float32(1e-8))
    indices = [3, 7, 8, 1, 2]
    constraint_rows = np.stack([g[j] for g in gradients for j in indices]).astype(np.float64)
    norms = np.linalg.norm(constraint_rows, axis=1)
    assert np.all(norms > 0)
    normalized = constraint_rows/norms[:, None]
    gram = normalized@normalized.T
    assert np.allclose(gram, gram.T, rtol=0, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(gram)
    assert eigenvalues[0] > 1e-9, 'Insufficient independent directions; no proposal'
    # Prospectively fixed local targets at nominal LR, not restored-image claims.
    nominal_lr = 1e-4
    fractions = np.array([.01, .0001, .0001, .0001, .0001]*2)
    scalars = np.array([r['component_values'][j] for r in rows for j in indices])
    target = -fractions*scalars/(nominal_lr*norms)
    b = normalized@direction.astype(np.float64)-target
    chol = np.linalg.cholesky(gram)
    multiplier, residual = nnls(chol.T, np.linalg.solve(chol, b), maxiter=1000)
    projected = direction.astype(np.float64)-normalized.T@multiplier
    projected32 = projected.astype(np.float32)
    constraint_products = normalized@projected
    slack = constraint_products-target
    # Independent dual/primal arithmetic checks; these are not quality tolerances.
    assert np.max(slack) < 1e-7 and np.all(multiplier >= 0)
    assert np.max(np.abs(multiplier*slack)) < 1e-5
    assert np.linalg.norm(projected-(direction.astype(np.float64)-normalized.T@multiplier)) < 1e-9
    ratio = float(np.linalg.norm(projected)/np.linalg.norm(direction.astype(np.float64)))
    assert ratio < 2, 'Projection exceeds finite proposal direction bound'
    directions = dict(ordinary=direction, constrained=projected32)
    dest.mkdir()
    arrays = {}
    for label, value in directions.items():
        for i, shape in enumerate(shapes):
            arrays[f'{label}_p{i:02d}'] = value[offsets[i]:offsets[i+1]].reshape(shape)
    proposal_path = dest/'directions.npz'
    np.savez_compressed(proposal_path, **arrays)
    flat = {}
    with np.load(proposal_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)
        for name, value in arrays.items():
            assert np.array_equal(archive[name], value) and value.dtype == np.float32 and np.isfinite(value).all()
        for label in directions:
            flat[label] = np.concatenate([archive[f'{label}_p{i:02d}'].reshape(-1) for i in range(37)])
    float32_products = constraint_rows@flat['constrained'].astype(np.float64)
    predicted_fraction = -nominal_lr*float32_products/scalars
    records = []
    for k, (row, j) in enumerate((r, j) for r in rows for j in indices):
        records.append(dict(reference=row['reference'], component=run['component_names'][j],
            nominal_ordinary_direction_product=float(constraint_rows[k]@direction.astype(np.float64)),
            nominal_constrained_direction_product=float(float32_products[k]),
            required_local_fraction=float(fractions[k]), nominal_predicted_fraction=float(predicted_fraction[k]),
            is_first_order_prediction_not_a_finite_output=True))
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == digest for name, digest in protected.items())
    write(dest/'direction_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        reviewer_sha256=sha(__file__), parent_audit_sha256=sha(audit_path), parent_loss_review_sha256=sha(review_path),
        run_results_sha256=sha(run_path), initial_file_sha256=sha(initial_path), sampled_gradient_bindings=bindings,
        names=names, shapes=[list(s) for s in shapes], sizes=sizes, direction_file_sha256=sha(proposal_path),
        direction_file_bytes=proposal_path.stat().st_size, parameters=values.size, initial_parameter_norm=float(np.linalg.norm(values.astype(np.float64))),
        constraint_components=[run['component_names'][j] for j in indices], constraint_records=records,
        normalized_Gram=gram.tolist(), normalized_targets=target.tolist(), dual_multipliers=multiplier.tolist(),
        smallest_Gram_eigenvalue=float(eigenvalues[0]), dual_NNLS_residual=float(residual),
        maximum_primal_slack=float(np.max(slack)), maximum_complementarity_error=float(np.max(abs(multiplier*slack))),
        direction_norm_ratio=ratio, nominal_learning_rate=nominal_lr,
        displacement_l2_bounds={label: float(nominal_lr*np.linalg.norm(value.astype(np.float64))) for label,value in directions.items()},
        objective_omits_absolute_anchors_but_retains_regression_and_corrected_brightness_losses=True,
        projection_applied_to_saved_Adam_direction_after_preconditioning=True,
        sample_scope_two_TRAIN_references_ten_cases=True, not_the_actual_five_reference_accumulated_update=True,
        all20_training_references_and_native_outputs_not_validated=True,
        derived_direction_is_not_a_checkpoint_or_model_update=True,
        no_local_neural_calls_gradients_optimizer_or_parameter_displacement=True,
        finite_forward_and_training_usefulness_unproven=True, protected_bindings_unchanged=True,
        model_qualified=False, goal_complete=False))
    print(dict(complete=True, direction_bytes=proposal_path.stat().st_size, direction_norm_ratio=ratio,
        minimum_predicted_detail_fraction=float(min(predicted_fraction[::5])), local_optimizer_updates=0))


if __name__ == '__main__':
    main()

"""Compare saved derivative predictions to three already executed trial outputs.

No model construction, local gradients, solver, weight proposals or training.
This reads the returned vectors, actual rounded component arrays and image scores.
"""
from collections import Counter
from datetime import datetime, timezone
import time

import numpy as np

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, RETURN, ANALYSIS, PIN, read, sha, write


def score_loss(groups, row):
    values = groups[row['metric_stage'].lower()][row['source'] + '/' + row['profile']]
    if row['metric'] == 'MSE':
        return values['MSE']
    if row['metric'] == 'SSIM_loss':
        return 1 - values['SSIM']
    if row['metric'] == 'ArcFace_loss':
        return 1 - values['ArcFace_observed_fixed']
    assert row['metric'] == 'feature_MSE'
    return values['landmark_high_frequency_MSE']


def main():
    started = time.monotonic()
    audit = read(ANALYSIS / 'failure_audit.json')
    p = read(RETURN / 'protocol.json')
    assert audit['complete'] and audit['protocol_sha256'] == PIN
    assert audit['completed_trials'] == 3 and audit['accepted_trials_verified'] == []
    assert len(audit['gradient_audits']) == 1 and audit['gradient_audits'][0]['full_vector_KKT_and_float32_dots_checked']
    out = RETURN / 'outputs/conditioning_learning_v1_r4'
    gradient_file = out / 'round00/gradients.json'
    gradients = read(gradient_file)
    records = gradients['groups']
    assert len(records) == 64
    direction = np.load(out / 'round00/direction.npy', allow_pickle=False).astype(np.float64)
    initial = np.load(out / 'initial_conditioning.npy', allow_pickle=False).astype(np.float64)
    labels = ['r00_trial00', 'r00_trial01', 'r00_trial02']
    trials = [read(out / label / 'decision.json') for label in labels]
    snapshots = [read(out / label / 'snapshot.json') for label in labels]
    baseline = read(out / 'initial/snapshot.json')['groups']
    deltas = np.column_stack([np.load(out / label / 'conditioning.npy', allow_pickle=False).astype(np.float64) - initial
                              for label in labels])
    vectors = []
    for row in records:
        path = out / row['file']
        assert sha(path) == row['sha256']
        v = np.load(path, allow_pickle=False, mmap_mode='r')
        assert v.shape == direction.shape and v.dtype == np.float64 and np.isfinite(v).all()
        vectors.append(v)
    direction_dots = np.zeros(64)
    rounded_step_dots = np.zeros((64, 3))
    for begin in range(0, len(direction), 32768):
        assert time.monotonic() - started < 180
        block = np.vstack([v[begin:begin + 32768] for v in vectors])
        direction_dots += np.einsum('ij,j->i', block, direction[begin:begin + 32768], optimize=True)
        rounded_step_dots += np.einsum('ij,jk->ik', block, deltas[begin:begin + 32768], optimize=True)
    np.testing.assert_allclose(direction_dots, gradients['certificate']['direction_dots'], rtol=1e-8, atol=1e-10)
    assert (direction_dots < 0).all()
    gram = np.asarray(gradients['certificate']['normalized_Gram'], np.float64)
    eigensystems = {}
    for name, matrix in [('RAW32', gram[:32, :32]), ('PNG32', gram[32:, 32:]), ('joint64', gram)]:
        eigen = np.linalg.eigvalsh(matrix)
        assert eigen[0] > 0
        eigensystems[name] = dict(smallest_eigenvalue=float(eigen[0]), largest_eigenvalue=float(eigen[-1]),
                                 condition_number=float(eigen[-1] / eigen[0]))
    correlations = np.diag(gram[:32, 32:])
    ledger = []
    summaries = []
    for col, (label, trial, snapshot) in enumerate(zip(labels, trials, snapshots)):
        counts = Counter()
        by_stage = {'raw': Counter(), 'png': Counter()}
        parameter_step = direction * trial['rate']
        rounding_ratio = float(np.linalg.norm(deltas[:, col] - parameter_step) / np.linalg.norm(parameter_step))
        for index, row in enumerate(records):
            before = score_loss(baseline, row)
            after = score_loss(snapshot['groups'], row)
            bound = 1e-6 if row['metric'] == 'ArcFace_loss' else 1e-13
            assert abs(row['value'] - before) <= bound
            linear_delta = float(direction_dots[index] * trial['rate'])
            rounded_delta = float(rounded_step_dots[index, col])
            actual_delta = float(after - before)
            scientific_tolerance = (1e-12 if row['metric'] == 'MSE' else
                                    1e-6 if row['metric'] in ['SSIM_loss', 'ArcFace_loss'] else 0.)
            reversed_safe_prediction = rounded_delta < 0 and actual_delta > scientific_tolerance
            counts['groups'] += 1
            counts['rounding_changes_predicted_descent_sign'] += rounded_delta >= 0
            counts['actual_loss_improved'] += actual_delta < 0
            counts['actual_preservation_loss_failed'] += row['metric'] != 'feature_MSE' and actual_delta > scientific_tolerance
            counts['predicted_safe_descent_but_actual_loss_regressed'] += reversed_safe_prediction
            by_stage[row['metric_stage'].lower()]['groups'] += 1
            by_stage[row['metric_stage'].lower()]['prediction_sign_reversal_beyond_original_tolerance'] += reversed_safe_prediction
            ledger.append(dict(trial=label, group=row['key'], stage=row['metric_stage'].lower(),
                source=row['source'], profile=row['profile'], metric=row['metric'], before_loss=before,
                after_loss=after, saved_gradient_loss=row['value'], direction_dot=float(direction_dots[index]),
                rate=trial['rate'], predicted_linear_loss_delta=linear_delta,
                predicted_delta_for_actual_rounded_parameter_change=rounded_delta,
                actual_scientific_loss_delta=actual_delta,
                finite_minus_linear_prediction_residual=actual_delta - rounded_delta,
                preservation_loss_regressed_beyond_original_tolerance=actual_delta > scientific_tolerance,
                original_metric_regression_tolerance=scientific_tolerance,
                gradient_descent_prediction_reversed=reversed_safe_prediction,
                no_claim_of_exact_Hessian_or_neural_derivative_validation=True))
        actual_gate = trial['decision']
        predictions = {}
        for stage in ['raw', 'png']:
            indices = [i for i, row in enumerate(records) if row['metric_stage'].lower() == stage
                       and row['metric'] == 'feature_MSE']
            assert len(indices) == 2
            source_weights = np.array([baseline[stage][records[i]['source'] + '/degraded']['cases'] for i in indices])
            predicted_global = -float(np.average(rounded_step_dots[indices, col], weights=source_weights)) / baseline[stage]['degraded']['landmark_high_frequency_MSE']
            predictions[stage] = dict(predicted_global_detail_gain=predicted_global,
                actual_global_detail_gain=actual_gate['stages'][stage]['relative_feature_gain'],
                actual_source_detail_gains=actual_gate['stages'][stage]['source_feature_gains'],
                actual_preservation_failed_checks=len(actual_gate['stages'][stage]['initial_preservation_failures']),
                actual_brightness_fraction=actual_gate['stages'][stage]['brightness_gain_fraction'])
        summaries.append(dict(trial=label, rate=trial['rate'],
            cumulative_relative_parameter_distance=trial['cumulative_relative_parameter_distance'],
            norm_of_float32_rounding_error_relative_to_intended_step=rounding_ratio,
            counts=dict(counts), stages={k: dict(v) for k, v in by_stage.items()}, detail_and_gates=predictions,
            actual_accept_step=False, one_percent_checkpoint_pass=False))
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), analyst_sha256=sha(__file__),
        protocol_sha256=PIN, failure_audit_sha256=sha(ANALYSIS / 'failure_audit.json'),
        gradient_receipt_sha256=sha(gradient_file), exact_float32_direction_dots_recomputed=True,
        actual_rounded_saved_parameter_changes_used_for_prediction=True, eigensystems=eigensystems,
        paired_RAW_PNG_gradient_cosines=dict(minimum=float(correlations.min()), maximum=float(correlations.max()),
                                           mean=float(correlations.mean())),
        trials=summaries, real_group_trial_comparisons=ledger,
        conclusion='The saved descent certificate did not establish finite preservation at any tested rate. '
                   'Actual parameter-rounding predictions, curvature, approximate loss derivatives and '
                   'RAW/PNG constraint conditioning must be distinguished before another learning pilot.',
        smaller_unexecuted_steps_or_future_success_not_inferred=True,
        original_gates_states_models_caches_splits_and_failed_trials_preserved=True,
        selected_incumbent_is_still_original=True, accepted_updates=0,
        new_solver_or_neural_model_forwards_gradients_updates_or_parameter_proposals=0,
        native_CCTV_or_final_identities_used=False, seconds=time.monotonic() - started,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False)
    write(ANALYSIS / 'gradient_prediction_review.json', record)
    print(dict(complete=True, saved_group_trial_comparisons=len(ledger), trials=summaries,
               gradient_Gram_condition_numbers={k: v['condition_number'] for k, v in eigensystems.items()},
               new_neural_or_training_calls=0), flush=True)


if __name__ == '__main__':
    main()

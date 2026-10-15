"""Saved-vector and already executed trial analysis; never construct a model."""
from collections import Counter
from datetime import datetime, timezone
import time

import numpy as np

from import_cctv_dgp_raw_conditioning_learning_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write


def loss(groups, row):
    values = groups['raw'][row['source'] + '/' + row['profile']]
    fields = {'MSE': 'MSE', 'SSIM_loss': 'SSIM', 'ArcFace_loss': 'ArcFace_observed_fixed',
              'feature_MSE': 'landmark_high_frequency_MSE'}
    value = values[fields[row['metric']]]
    return 1 - value if row['metric'] in ['SSIM_loss', 'ArcFace_loss'] else value


def main():
    started = time.monotonic()
    audited = read(ANALYSIS / 'failure_forensic_audit.json')
    assert audited['complete'] and audited['protocol_sha256'] == PIN
    assert audited['completed_trials'] == 3 and audited['accepted_trials_verified'] == []
    assert len(audited['gradient_audits']) == 1
    assert audited['gradient_audits'][0]['full_vector_KKT_and_float32_dots_checked']
    out = RETURN / 'outputs/raw_conditioning_learning_v1_r1'
    gradient = read(out / 'round00/gradients.json')
    records = gradient['groups']; assert len(records) == 32
    baseline = read(out / 'initial/snapshot.json')['groups']
    initial = np.load(out / 'initial_conditioning.npy', allow_pickle=False).astype(np.float64)
    direction = np.load(out / 'round00/direction.npy', allow_pickle=False).astype(np.float64)
    labels = ['r00_trial00', 'r00_trial01', 'r00_trial02']
    trials = [read(out / label / 'decision.json') for label in labels]
    snapshots = [read(out / label / 'snapshot.json')['groups'] for label in labels]
    deltas = np.column_stack([np.load(out / label / 'conditioning.npy', allow_pickle=False).astype(np.float64) - initial
                             for label in labels])
    vectors = []
    for row in records:
        path = out / row['file']; assert sha(path) == row['sha256']
        vector = np.load(path, allow_pickle=False, mmap_mode='r')
        assert vector.shape == direction.shape and vector.dtype == np.float64 and np.isfinite(vector).all()
        vectors.append(vector)
    rounded_dots = np.zeros((32, 3)); dots = np.zeros(32)
    for begin in range(0, len(direction), 32768):
        assert time.monotonic() - started < 180
        block = np.vstack([v[begin:begin + 32768] for v in vectors])
        dots += np.einsum('ij,j->i', block, direction[begin:begin + 32768], optimize=True)
        rounded_dots += np.einsum('ij,jk->ik', block, deltas[begin:begin + 32768], optimize=True)
    np.testing.assert_allclose(dots, gradient['certificate']['direction_dots'], rtol=1e-8, atol=1e-10)
    assert (dots < 0).all()
    gram = np.asarray(gradient['certificate']['normalized_Gram'], np.float64)
    eigen = np.linalg.eigvalsh(gram); assert eigen[0] > 0
    rows = []; summaries = []
    for column, (label, trial, groups) in enumerate(zip(labels, trials, snapshots)):
        counts = Counter(); indices = []
        for index, row in enumerate(records):
            before, after = loss(baseline, row), loss(groups, row)
            alignment = 1e-6 if row['metric'] == 'ArcFace_loss' else 1e-13
            assert abs(before - row['value']) <= alignment
            tolerance = 1e-12 if row['metric'] == 'MSE' else 1e-6 if row['metric'] in ['SSIM_loss', 'ArcFace_loss'] else 0.
            predicted = float(rounded_dots[index, column]); actual = float(after - before)
            counts['RAW_groups'] += 1
            counts['rounding_changes_predicted_descent_sign'] += predicted >= 0
            counts['actual_RAW_preservation_loss_failed'] += row['metric'] != 'feature_MSE' and actual > tolerance
            rows.append(dict(trial=label, key=row['key'], source=row['source'], profile=row['profile'], metric=row['metric'],
                             before_loss=before, after_loss=after, direction_dot=float(dots[index]),
                             actual_rounded_parameter_change_dot=predicted, actual_loss_delta=actual,
                             finite_minus_linear_residual=actual - predicted,
                             original_regression_bound=tolerance, regression_beyond_bound=actual > tolerance))
            if row['metric'] == 'feature_MSE': indices.append(index)
        assert len(indices) == 2
        weights = np.array([baseline['raw'][records[i]['source'] + '/degraded']['cases'] for i in indices])
        predicted_gain = -float(np.average(rounded_dots[indices, column], weights=weights)) / baseline['raw']['degraded']['landmark_high_frequency_MSE']
        intended = direction * trial['rate']
        stages = {}
        for stage, value in trial['decision']['stages'].items():
            failures = value['initial_preservation_failures']
            stages[stage] = dict(actual_global_feature_gain=value['relative_feature_gain'],
                actual_source_feature_gains=value['source_feature_gains'], failed_checks=len(failures),
                failed_metrics=dict(Counter(v['metric'] for v in failures)),
                failed_groups=sorted(set(v['group'] for v in failures)), brightness_fraction=value['brightness_gain_fraction'])
        summaries.append(dict(trial=label, rate=trial['rate'], predicted_RAW_feature_gain=predicted_gain,
            rounding_error_relative_to_intended_step=float(np.linalg.norm(deltas[:, column] - intended) / np.linalg.norm(intended)),
            cumulative_relative_parameter_distance=trial['cumulative_relative_parameter_distance'], counts=dict(counts), stages=stages,
            actual_accept_step=trial['decision']['accept_step'], endpoint_pass=trial['decision']['one_percent_checkpoint_pass']))
    previous = OUT / 'cctv_dgp_conditioning_learning_v1_r4_analysis/gradient_prediction_review.json'
    old = read(previous); assert old['complete'] and old['accepted_updates'] == 0
    old_return = OUT / 'cctv_dgp_conditioning_learning_v1_r4_return/outputs/conditioning_learning_v1_r4'
    old_initial = np.load(old_return / 'initial_conditioning.npy', allow_pickle=False)
    np.testing.assert_array_equal(old_initial, initial.astype(np.float32))
    assert read(old_return / 'initial/snapshot.json')['groups'] == baseline
    old_direction = np.load(old_return / 'round00/direction.npy', allow_pickle=False).astype(np.float64)
    cosine = float(direction @ old_direction / (np.linalg.norm(direction) * np.linalg.norm(old_direction)))
    compared = [dict(trial=label, previous_RAW_gain=old['trials'][i]['detail_and_gates']['raw']['actual_global_detail_gain'],
                     previous_PNG_gain=old['trials'][i]['detail_and_gates']['png']['actual_global_detail_gain'],
                     current_RAW_gain=summaries[i]['stages']['raw']['actual_global_feature_gain'],
                     current_PNG_gain=summaries[i]['stages']['png']['actual_global_feature_gain']) for i, label in enumerate(labels)]
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == h for name, h in protected.items())
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), analyst_sha256=sha(__file__), protocol_sha256=PIN,
        failure_audit_sha256=sha(ANALYSIS / 'failure_forensic_audit.json'), gradient_receipt_sha256=sha(out / 'round00/gradients.json'),
        original_cross_run_derivative_parity_pass=audited['cross_run_derivative_parity_pass'],
        RAW32_Gram_condition_number=float(eigen[-1] / eigen[0]), previous_joint64_Gram_condition_number=old['eigensystems']['joint64']['condition_number'],
        previous_saved_review_sha256=sha(previous), current_previous_direction_cosine=cosine,
        exact_same_initial_component_and_scientific_baseline_checked=True, trials=summaries, RAW_group_trial_comparisons=rows,
        previous_executed_trial_comparison=compared, accepted_updates=0,
        conclusion='Removing approximate PNG optimizer constraints did not establish finite preservation or a qualified structure gain. '
                   'Neither parameter rounding nor the joint Gram condition number alone explains the failure. '
                   'Review data support and nonlinear conditioning response before another training recipe.',
        unexecuted_rates_epochs_or_future_quality_not_inferred=True,
        no_new_neural_derivatives_models_solver_direction_or_weight_proposals=True,
        native_CCTV_or_final_identity_pixels_used=False, seconds=time.monotonic() - started,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False)
    write(ANALYSIS / 'gradient_prediction_review.json', record)
    print(dict(complete=True, comparisons=len(rows), condition_number=record['RAW32_Gram_condition_number'],
               previous_direction_cosine=cosine, trials=summaries, new_neural_or_training_calls=0), flush=True)


if __name__ == '__main__': main()

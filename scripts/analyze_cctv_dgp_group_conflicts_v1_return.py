"""Post-hoc saved-vector/finite-output attribution; no neural or derivative calls."""
from pathlib import Path
import json
import time
import numpy as np
from prepare_cctv_dgp_group_conflicts_v1_visual_review import ROOT, BUNDLE, RETURN, AUDIT, PIN, read, sha

OUT = ROOT / 'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/finite_direction_analysis.json'


def main():
    started = time.monotonic()
    assert not OUT.exists()
    assert sha(BUNDLE / 'protocol.json') == PIN
    protocol, result, audit = read(BUNDLE / 'protocol.json'), read(RETURN / 'outputs/results.json'), read(AUDIT)
    assert audit['complete'] and audit['all32_saved_group_vectors_and_certificate_arithmetic_verified']
    assert audit['all_stored_row_comparisons_and_failure_decisions_exact']
    assert result['protocol_sha256'] == PIN and result['optimizer_updates'] == result['epochs'] == 0
    folder = RETURN / 'outputs'
    baseline = read(folder / 'baseline/metrics.json')
    baseline_rows = {row['id']: row for row in baseline['rows']}
    initial = np.load(folder / 'initial_parameters.npy', allow_pickle=False).astype(np.float64)
    with np.load(folder / 'group_gradients.npz', allow_pickle=False) as loaded:
        gradients = loaded['gradients']
    assert gradients.shape == (32, 1996035) and gradients.dtype == np.float64
    certificate = read(folder / 'direction_certificate.json')
    previous = read(folder / 'previous_direction_group_predictions.json')
    previous_negative = [{'index': i, 'group': protocol['group_losses'][i],
        'descent_cosine': previous['descent_cosines'][i]} for i in previous['negative_cosine_group_indices']]
    metric_keys = {'MSE': 'MSE', 'SSIM_loss': 'SSIM',
        'ArcFace_loss': 'ArcFace_observed_fixed', 'landmark_structure': 'landmark_high_frequency_MSE'}

    def loss_mean(rows, group):
        values = [rows[cid]['raw'][metric_keys[group['metric']]] for cid in group['case_ids']]
        if group['metric'] in ['SSIM_loss', 'ArcFace_loss']:
            values = [1. - value for value in values]
        return float(np.mean(values))

    trials, decisions = [], []
    for trial in result['trial_summaries']:
        label = trial['variant']
        parameters = np.load(folder / (label + '_parameters.npy'), allow_pickle=False).astype(np.float64)
        predicted = gradients @ (parameters - initial)
        assert np.allclose(predicted, trial['gradient_dot_actual_displacement'], rtol=1e-12, atol=1e-12)
        saved = read(folder / label / 'metrics.json')
        candidate_rows = {row['id']: row for row in saved['rows']}
        groups = []
        for i, group in enumerate(protocol['group_losses']):
            before, after = loss_mean(baseline_rows, group), loss_mean(candidate_rows, group)
            allowance = 1e-12 if group['metric'] == 'MSE' else 1e-6 if group['metric'] in ['SSIM_loss', 'ArcFace_loss'] else 0.
            groups.append({'index': i, 'source': group['source'], 'profile': group['profile'],
                'metric': group['metric'], 'cases': len(group['case_ids']),
                'predicted_initial_loss_delta': float(predicted[i]),
                'observed_scientific_raw_loss_delta': after - before,
                'initial_loss_mean': before, 'finite_loss_mean': after,
                'observation_exceeds_existing_metric_allowance': bool(after - before > allowance),
                'SSIM_derivative_is_declared_surrogate': group['metric'] == 'SSIM_loss'})
        trials.append({'variant': label, 'all32_initial_derivative_predictions_improve': bool(np.all(predicted < 0)),
            'sampled_group_finite_regressions_beyond_retained_allowance': sum(group['observation_exceeds_existing_metric_allowance'] for group in groups),
            'groups': groups})
        for cohort, stages in trial['comparisons'].items():
            for stage, gate in stages.items():
                decisions.append({'variant': label, 'cohort': cohort, 'stage': stage,
                    'structure_gain_percent': 100 * gate['relative_feature_gain'],
                    'preservation_failure_count': len(gate['preservation_failures']),
                    'failed_pairs': [[failure['group'], failure['metric']] for failure in gate['preservation_failures']],
                    'pass': gate['pass']})
        assert time.monotonic() - started < 120
    parent = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_vm_return/outputs/baseline'
    original_baseline_exact = True
    for cid in baseline_rows:
        original_baseline_exact &= sha(folder / 'baseline' / (cid + '.npy')) == sha(parent / (cid + '.npy'))
        original_baseline_exact &= sha(folder / 'baseline' / (cid + '.png')) == sha(parent / (cid + '.png'))
    assert original_baseline_exact and len(decisions) == 12 and all(not gate['pass'] for gate in decisions)
    report = {'complete': True, 'post_hoc_saved_evidence_analysis': True,
        'protocol_sha256': PIN, 'results_sha256': sha(folder / 'results.json'),
        'independent_audit_sha256': sha(AUDIT), 'initial100_baseline_raw_and_PNG_exact_to_prior_balance': True,
        'previous_balanced_direction_negative_descent_groups': previous_negative,
        'common_direction_minimum_normalized_descent_cosine': certificate['minimum_normalized_descent_cosine'],
        'common_direction_found': certificate['common_direction_found'], 'finite_group_predictions': trials,
        'unchanged_scientific_decisions': decisions,
        'scope': 'Saved aggregate derivative arithmetic versus finite raw metrics; SSIM derivative is a surrogate. Two photographic TRAIN cohorts, no full corpus, DEV, native CCTV or final evidence. Individual autograd/aggregation not replayed locally.',
        'optimizer_trajectory_tested': False, 'new_neural_calls': 0, 'new_gradient_queries': 0,
        'optimizer_updates': 0, 'scientific_thresholds_changed': False, 'model_qualification': False,
        'goal_complete': False, 'analyzer_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({'complete': True, 'previous_negative_group_count': len(previous_negative),
        'baseline100_exact': True, 'finite_regressions': {trial['variant']: trial['sampled_group_finite_regressions_beyond_retained_allowance'] for trial in trials},
        'all12_decisions_fail': True, 'seconds': report['seconds']}, flush=True)


if __name__ == '__main__':
    main()

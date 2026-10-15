"""Summarize independently replayed saved responses; no neural execution or solve."""
from collections import Counter
from datetime import datetime, timezone
import math
from statistics import median

from import_cctv_dgp_conditioning_response_v1_return import (
    ANALYSIS, OUT, PIN, RETURN, read, sha, write,
)


def main():
    audit = read(ANALYSIS / 'independent_audit.json')
    imported = read(ANALYSIS / 'import.json')
    assert audit['complete'] and audit['full_diagnostic_returned']
    assert audit['protocol_sha256'] == PIN and len(audit['central_response_comparisons']) == 96
    assert audit['accepted_updates'] == audit['optimizer_updates'] == 0
    base = RETURN / 'outputs/conditioning_response_v1'
    p = read(RETURN / 'protocol.json')
    initial = read(base / 'initial/snapshot.json')
    gradient = read(base / 'repeat00/gradients.json')
    points = []
    derivative_changes = []
    for step in p['step_plan']:
        record = read(base / step['label'] / 'response.json')
        snapshot = read(base / step['label'] / 'snapshot.json')
        stages = {}
        for s in ['raw', 'png']:
            gates = record['scientific_decision_information_only']['stages'][s]
            failures = gates['initial_preservation_failures']
            stages[s] = dict(
                structure_gain_fraction=gates['relative_feature_gain'],
                source_gains=gates['source_feature_gains'],
                preservation_failures=len(failures),
                failures_by_metric=dict(Counter(r['metric'] for r in failures)),
                failures=failures,
                all_original_scientific_decision_fields=gates,
            )
        points.append(dict(label=step['label'], signed_rate=record['nominal_rate'],
                           relative_parameter_distance=record['cumulative_relative_parameter_distance'],
                           stages=stages, accepted_update=False, candidate_selection=False))
        pair = int(step['label'][-2:])
        for g in gradient['groups']:
            source_key = g['source'] + '/' + g['profile']
            field = {'MSE': 'MSE', 'SSIM_loss': 'SSIM',
                     'ArcFace_loss': 'ArcFace_observed_fixed',
                     'feature_MSE': 'landmark_high_frequency_MSE'}[g['metric']]
            before, after = initial['groups']['raw'][source_key][field], snapshot['groups']['raw'][source_key][field]
            if g['metric'] in ['SSIM_loss', 'ArcFace_loss']:
                before, after = 1-before, 1-after
            central = next(r for r in audit['central_response_comparisons']
                           if r['pair'] == pair and r['key'] == g['key'])
            sign = -1 if step['label'].startswith('minus') else 1
            prediction = central['even_linear_prediction'] + sign * central['predicted_two_sided_difference']/2
            derivative_changes.append(dict(label=step['label'], key=g['key'], metric=g['metric'],
                                           before=before, after=after, observed_change=after-before,
                                           rounded_direction_prediction=prediction,
                                           even_response=central['even_response'],
                                           diagnostic_central_resolved=central['resolved']))
    summaries = []
    for pair in range(3):
        for metric in ['MSE', 'SSIM_loss', 'ArcFace_loss', 'feature_MSE']:
            rows = [r for r in audit['central_response_comparisons']
                    if r['pair'] == pair and r['metric'] == metric]
            resolved = [r for r in rows if r['resolved']]
            relative_errors = [abs(r['central_prediction_residual']) / abs(r['predicted_two_sided_difference'])
                               for r in resolved]
            summaries.append(dict(pair=pair, magnitude=rows[0]['magnitude'], metric=metric, groups=len(rows),
                                  resolved=len(resolved), unresolved=len(rows)-len(resolved),
                                  within_diagnostic_bound=sum(r['central_comparison_within_diagnostic_bound'] is True for r in rows),
                                  outside_diagnostic_bound=sum(r['central_comparison_within_diagnostic_bound'] is False for r in rows),
                                  median_relative_central_error=median(relative_errors) if relative_errors else None,
                                  maximum_relative_central_error=max(relative_errors) if relative_errors else None,
                                  maximum_absolute_even_response=max(abs(r['even_response']) for r in rows),
                                  comparison_is_not_a_quality_gate=True))
    parity = []
    for repeat in audit['repeat_vector_checks']:
        for comparison in ['previous stopped R1', 'repeat00 versus repeat01']:
            rows = [r for r in repeat['original_unchanged_parity_bounds'] if r['comparison'] == comparison]
            if rows:
                parity.append(dict(repeat=repeat['repeat'], comparison=comparison, groups=len(rows),
                                   passing_groups=sum(r['original_parity_check_pass'] for r in rows),
                                   mismatched_elements=sum(r['mismatched_elements'] for r in rows),
                                   maximum_relative_vector_difference=max(r['relative_vector_difference'] for r in rows),
                                   maximum_absolute_difference=max(r['maximum_absolute_difference'] for r in rows),
                                   failed_groups=[r for r in rows if not r['original_parity_check_pass']]))
    previous = OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_return/outputs/raw_conditioning_learning_v1_r1/r00_trial02/snapshot.json'
    previous_snapshot = read(previous)
    plus00 = read(base / 'plus00/snapshot.json')
    # This prior finite point uses the same actual rate and initial component.
    score_differences = []
    assert set(previous_snapshot['groups']) == set(plus00['groups'])
    for s, groups in plus00['groups'].items():
        assert set(groups) == set(previous_snapshot['groups'][s])
        for name, values in groups.items():
            for field, value in values.items():
                prior = previous_snapshot['groups'][s][name][field]
                assert math.isfinite(value) and math.isfinite(prior)
                score_differences.append(abs(value-prior))
    run = read(base / 'run_results.json')
    summary = dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), analyzer_sha256=sha(__file__),
        protocol_sha256=PIN, independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        import_sha256=sha(ANALYSIS/'import.json'), run_sha256=sha(base/'run_results.json'),
        diagnostic_counts=run['counts'], runtime_seconds=run['seconds'],
        signed_points=points, central_summaries=summaries, original_bound_parity_summaries=parity,
        RAW_group_point_changes=derivative_changes,
        prior_smallest_trial_comparison=dict(prior_snapshot_sha256=sha(previous), current_snapshot_sha256=sha(base/'plus00/snapshot.json'),
                                            compared_group_scalars=len(score_differences), maximum_absolute_score_difference=max(score_differences)),
        evidence_scope='100 already exposed paired photographic TRAIN cases; not native CCTV, final identities, or covering qualification',
        actual_saved_arrays_scores_only=True, local_neural_gradient_or_training_calls=0,
        new_direction_or_optimizer_or_rate_search=False, accepted_updates=0,
        no_unique_all_dimensional_root_cause_proof=True, model_qualified=False,
        more_epochs_qualified=False, goal_complete=False,
    )
    write(ANALYSIS/'response_review.json', summary)
    print(dict(complete=True, parity=parity, central=summaries,
               signed_points=[dict(label=r['label'],raw=r['stages']['raw']['structure_gain_fraction'],
                                   png=r['stages']['png']['structure_gain_fraction'],
                                   failures=[r['stages'][s]['preservation_failures'] for s in ['raw','png']]) for r in points],
               previous_point_maximum_difference=max(score_differences)))


if __name__ == '__main__':
    main()

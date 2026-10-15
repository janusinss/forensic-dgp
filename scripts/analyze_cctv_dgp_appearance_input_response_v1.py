"""Interpret audited, saved response arrays only; no neural execution or new direction."""
from datetime import datetime, timezone

import numpy as np

from import_cctv_dgp_appearance_input_response_v1_return import (
    ANALYSIS, BANK, PIN, RETURN, ROOT, SOURCE, read, sha, write,
)
from cctv_dgp_conditioning_response_v1_contract import response_receipt


def main():
    audited = read(ANALYSIS / 'independent_audit.json')
    imported = read(ANALYSIS / 'import.json')
    assert audited['complete'] and audited['full_diagnostic_returned'] and not audited['failure']
    assert audited['protocol_sha256'] == imported['protocol_sha256'] == PIN
    assert audited['all_original_parity_checks_pass'] and len(audited['per_case_gradient_parity_original_bounds']) == 100
    p = read(RETURN / 'protocol.json')
    base = RETURN / 'outputs/appearance_input_response_v1'
    parent = SOURCE / 'outputs/conditioning_response_v1'
    repeats = [read(base / f'repeat{k:02d}/repeat.json') for k in range(2)]
    run = read(base / 'run_results.json')
    old = read(BANK / 'protocol.json')
    initial = read(parent / 'initial/snapshot.json')
    initial_rows = {r['id']: r for r in initial['rows']}
    signs = {label: read(parent / label / 'snapshot.json') for label in p['signed_labels']}
    signed_rows = {label: {r['id']: r for r in s['rows']} for label, s in signs.items()}
    parameter = read(RETURN / 'evidence/parameter_response_audit.json')['central_response_comparisons']
    precision = read(SOURCE / 'outputs/conditioning_response_v1/repeat00/gradients.json')['groups']
    case_doubles = {}
    input_bindings = {}
    maximum_scalar_reporting_difference = 0.
    for block in p['reference_blocks']:
        stem = f"b{block['begin']//5:03d}"
        cases = old['cases'][block['begin']:block['begin'] + 5]
        for label in ['initial'] + p['signed_labels']:
            path = parent / label / (stem + '_embeddings.npz')
            input_bindings[path.relative_to(SOURCE).as_posix()] = sha(path)
            with np.load(path, allow_pickle=False) as f:
                v = f['raw']
                assert v.shape == (15, 512) and v.dtype == np.float32 and np.isfinite(v).all()
                for i, case in enumerate(cases):
                    double = 1 - float(v[i].astype(np.float64) @ v[10 + i].astype(np.float64))
                    canonical = (initial_rows if label == 'initial' else signed_rows[label])[case['id']]['raw']['ArcFace_observed_fixed']
                    maximum_scalar_reporting_difference = max(maximum_scalar_reporting_difference, abs(double - (1 - canonical)))
                    case_doubles[(case['id'], label)] = double
    summaries, group_records, case_records = [], [], []
    for repeat, record in enumerate(repeats):
        for group in p['appearance_groups']:
            rows = [r for r in record['rows'] if r['source'] == group['source'] and r['profile'] == group['profile']]
            assert len(rows) == 10
            cached_before = 1 - initial['groups']['raw'][group['source'] + '/' + group['profile']]['ArcFace_observed_fixed']
            fresh_values = [r['derivative_scalar'] for r in rows]
            before_double = float(np.mean([case_doubles[(r['id'], 'initial')] for r in rows]))
            spread = abs(float(np.mean(fresh_values)) - before_double)
            for pair in range(3):
                plus, minus = f'plus{pair:02d}', f'minus{pair:02d}'
                actual_double = {label: float(np.mean([case_doubles[(r['id'], label)] for r in rows])) for label in [plus, minus]}
                projected = {label: float(np.mean([next(x['input_projection'] for x in r['projections'] if x['label'] == label) for r in rows])) for label in [plus, minus]}
                prior = next(r for r in parameter if r['key'] == 'RAW/' + group['key'] and r['pair'] == pair)
                canonical = next(r for r in audited['RGB_input_and_parameter_boundary_comparisons']
                                 if r['key'] == 'RAW/' + group['key'] and r['pair'] == pair and r['repeat'] == repeat)
                diagnostic = response_receipt(before_double, actual_double[plus], actual_double[minus],
                                              prior['magnitude'], projected[plus], projected[minus], spread)
                individual = []
                for row in rows:
                    pp = next(v['input_projection'] for v in row['projections'] if v['label'] == plus)
                    pm = next(v['input_projection'] for v in row['projections'] if v['label'] == minus)
                    before = 1 - initial_rows[row['id']]['raw']['ArcFace_observed_fixed']
                    losses = {label: 1 - signed_rows[label][row['id']]['raw']['ArcFace_observed_fixed'] for label in [plus, minus]}
                    other = next(v for v in repeats[1 - repeat]['rows'] if v['id'] == row['id'])
                    receipt = response_receipt(before, losses[plus], losses[minus], prior['magnitude'], pp, pm,
                        max(abs(row['derivative_scalar'] - before), abs(other['derivative_scalar'] - before),
                            abs(row['derivative_scalar'] - other['derivative_scalar'])))
                    fresh = response_receipt(case_doubles[(row['id'], 'initial')], case_doubles[(row['id'], plus)],
                        case_doubles[(row['id'], minus)], prior['magnitude'], pp, pm,
                        abs(row['derivative_scalar'] - case_doubles[(row['id'], 'initial')]))
                    item = dict(repeat=repeat, pair=pair, key='RAW/' + group['key'], id=row['id'],
                        RGB_input_canonical=receipt, saved_double_reporting=fresh,
                        predicted_two_sided_difference=pp - pm)
                    case_records.append(item); individual.append(item)
                absolute = sum(abs(v['predicted_two_sided_difference']) for v in individual)
                summed = sum(v['predicted_two_sided_difference'] for v in individual)
                ratio = abs(summed) / absolute if absolute else None
                original = canonical['RGB_input_response']
                group_records.append(dict(repeat=repeat, pair=pair, key=canonical['key'],
                    canonical_RGB_input_response=original, saved_double_reporting=diagnostic,
                    retained_parameter_response=prior,
                    individual_resolved=sum(v['RGB_input_canonical']['resolved'] for v in individual),
                    individual_canonical_pass=sum(v['RGB_input_canonical']['central_comparison_within_diagnostic_bound'] is True for v in individual),
                    individual_canonical_fail=sum(v['RGB_input_canonical']['central_comparison_within_diagnostic_bound'] is False for v in individual),
                    individual_double_pass=sum(v['saved_double_reporting']['central_comparison_within_diagnostic_bound'] is True for v in individual),
                    individual_double_fail=sum(v['saved_double_reporting']['central_comparison_within_diagnostic_bound'] is False for v in individual),
                    sum_absolute_case_predictions=absolute, absolute_group_sum_prediction=abs(summed),
                    group_to_absolute_case_prediction_ratio=ratio,
                    signed_case_cancellation_fraction=None if ratio is None else 1 - ratio,
                    earlier_path_prediction_difference=original['predicted_two_sided_difference'] - prior['predicted_two_sided_difference'],
                    recognizer_input_response_residual=original['central_prediction_residual']))
        for pair in range(3):
            rows = [v for v in group_records if v['repeat'] == repeat and v['pair'] == pair]
            fail = lambda r, k: r[k]['central_comparison_within_diagnostic_bound'] is False
            summaries.append(dict(repeat=repeat, pair=pair, groups=len(rows),
                input_canonical_pass=sum(r['canonical_RGB_input_response']['central_comparison_within_diagnostic_bound'] is True for r in rows),
                input_canonical_fail=sum(fail(r, 'canonical_RGB_input_response') for r in rows),
                input_double_pass=sum(r['saved_double_reporting']['central_comparison_within_diagnostic_bound'] is True for r in rows),
                input_double_fail=sum(fail(r, 'saved_double_reporting') for r in rows),
                parameter_pass=sum(r['retained_parameter_response']['central_comparison_within_diagnostic_bound'] is True for r in rows),
                parameter_fail=sum(fail(r, 'retained_parameter_response') for r in rows),
                input_only_fail=[r['key'] for r in rows if fail(r, 'canonical_RGB_input_response') and not fail(r, 'retained_parameter_response')],
                parameter_only_fail=[r['key'] for r in rows if not fail(r, 'canonical_RGB_input_response') and fail(r, 'retained_parameter_response')],
                joint_fail=[r['key'] for r in rows if fail(r, 'canonical_RGB_input_response') and fail(r, 'retained_parameter_response')]))
    assert len(group_records) == 60 and len(case_records) == 600 and len(summaries) == 6
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    for folder, bindings in [(RETURN, read(RETURN / 'export_manifest.json')['files']),
                             (SOURCE, p['response_required_sha256']), (BANK, p['bank_required_sha256'])]:
        assert all(sha(folder / n) == h for n, h in bindings.items())
    write(ANALYSIS / 'boundary_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        analyzer_sha256=sha(__file__), independent_audit_sha256=sha(ANALYSIS / 'independent_audit.json'), protocol_sha256=PIN,
        group_records=group_records, case_records=case_records, summaries=summaries,
        saved_embedding_bindings=input_bindings, maximum_scalar_float_vs_saved_double_reporting_difference=maximum_scalar_reporting_difference,
        all100_RGB_gradient_repeats_pass_original_bounds=True,
        zero_RGB_gradient_cases=sum(r['gradient_norm'] == 0 for r in repeats[0]['rows']),
        maximum_repeat_absolute_gradient_difference=max(r['maximum_absolute_difference'] for r in audited['per_case_gradient_parity_original_bounds']),
        runtime_seconds=run['seconds'], export_seconds=6.269175778999852, counts=run['counts'],
        no_restorer_forward_or_new_image=True, no_new_visual_sheet_needed=True,
        conclusion='Sampled finite appearance disagreement exists already at the frozen recognizer RGB-input boundary; earlier path contributes in some groups.',
        unique_root_cause_correct_backward_full_precision_or_all_dimensional_parity_not_proved=True,
        saved_double_arithmetic_is_not_FP64_neural_execution=True,
        diagnostic_bound_is_not_scientific_preservation=True, quality_gates_unchanged=True,
        original_proposals_and_PNG_failures_remain_binding=True,
        local_neural_calls=0, gradient_queries=0, optimizer_updates=0, accepted_updates=0,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, group_comparisons=60, case_comparisons=600,
        canonical_input_fail=[v['input_canonical_fail'] for v in summaries if v['repeat'] == 0],
        saved_double_input_fail=[v['input_double_fail'] for v in summaries if v['repeat'] == 0],
        model_qualified=False))


if __name__ == '__main__':
    main()

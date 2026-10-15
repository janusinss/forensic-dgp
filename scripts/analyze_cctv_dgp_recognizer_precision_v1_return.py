"""Interpret independently checked precision/activation arrays without neural calls."""
from collections import Counter
from datetime import datetime, timezone
import numpy as np

from import_cctv_dgp_recognizer_precision_v1_return import ANALYSIS, OUT, PIN, RETURN, ROOT, read, sha, write


def main():
    audit = read(ANALYSIS / 'independent_audit.json')
    imported = read(ANALYSIS / 'import.json')
    assert audit['complete'] and audit['full_diagnostic_returned'] and audit['failure'] is None
    assert audit['protocol_sha256'] == imported['protocol_sha256'] == PIN
    assert audit['all_original_parity_checks_pass'] and len(audit['original_bound_parity_reports']) == 30
    assert len(audit['precision_responses']) == 20
    pairs = []
    rows = audit['precision_responses']
    for before in [r for r in rows if r['precision'] == 'FP32']:
        after = next(r for r in rows if r['precision'] == 'FP64' and r['id'] == before['id'])
        a, b = [r['full_network_precision_response'] for r in [before, after]]
        assert a['resolved'] and b['resolved']
        pairs.append(dict(id=before['id'], source=before['source'], profile=before['profile'],
            FP32=a, FP64=b,
            FP32_relative_residual=abs(a['central_prediction_residual'] / a['predicted_two_sided_difference']),
            FP64_relative_residual=abs(b['central_prediction_residual'] / b['predicted_two_sided_difference']),
            observed_response_difference=b['observed_two_sided_difference'] - a['observed_two_sided_difference'],
            predicted_response_difference=b['predicted_two_sided_difference'] - a['predicted_two_sided_difference']))
    totals = []
    for precision in ['FP32', 'FP64']:
        for label in ['plus02', 'minus02', 'repeat']:
            maps = [r for r in audit['activation_transitions'] if r['precision'] == precision and r['label'] == label]
            assert len(maps) == 50
            totals.append(dict(precision=precision, label=label, module_reference_maps=50,
                units=sum(r['units'] for r in maps), changed_sign_units=sum(r['changed_sign_units'] for r in maps),
                maps_with_changes=sum(r['changed_sign_units'] > 0 for r in maps),
                per_case_changes={case['id']: sum(r['per_case_changed_sign_units'][i] for r in maps if r['begin'] == block['begin'])
                    for block in read(RETURN / 'protocol.json')['reference_blocks']
                    for i, case in enumerate([dict(id=n) for n in block['case_ids']])}))
    source_bindings = {}
    precision_pairs = []
    p = read(RETURN / 'protocol.json'); base = RETURN / 'outputs/recognizer_precision_v1'
    for block in p['reference_blocks']:
        stem = f"b{block['begin']//5:03d}"
        for label in p['labels']:
            arrays = []
            for precision in ['FP32', 'FP64']:
                path = base / precision / (stem + '_' + label + '_embeddings.npz')
                source_bindings[path.relative_to(RETURN).as_posix()] = sha(path)
                with np.load(path, allow_pickle=False) as f:
                    arrays.append(f['raw'].astype(np.float64))
            precision_pairs.append(dict(begin=block['begin'], label=label,
                maximum_embedding_precision_difference=float(np.max(np.abs(arrays[0] - arrays[1]))),
                vector_L2_differences=np.linalg.norm(arrays[0] - arrays[1], axis=1).tolist()))
    run = read(base / 'run_results.json')
    exported = read(OUT / 'cctv-dgp-recognizer-precision-v1-export.json')
    assert exported['archive_sha256'] == imported['archive_sha256']
    failures = {precision: sum(r[precision]['central_comparison_within_diagnostic_bound'] is False for r in pairs)
                for precision in ['FP32', 'FP64']}
    assert failures == dict(FP32=6, FP64=5)
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(ANALYSIS / 'precision_activation_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        analyzer_sha256=sha(__file__), independent_audit_sha256=sha(ANALYSIS / 'independent_audit.json'), protocol_sha256=PIN,
        paired_precision_responses=pairs, sampled_response_failure_counts=failures,
        activation_transition_totals=totals, embedding_precision_comparisons=precision_pairs,
        saved_embedding_bindings=source_bindings, parity_groups=dict(Counter(r['precision'] for r in audit['original_bound_parity_reports'])),
        all_original_parity_bounds_pass=True,
        runtime_seconds=run['seconds'], export_seconds=exported['seconds'], counts=run['counts'],
        precision_alone_does_not_resolve_sampled_finite_prediction_error=True,
        activation_region_changes_are_observed_not_exclusive_causation_proof=True,
        converted_backward_bug_and_all_dimensional_correctness_not_proved=True,
        source_selection_is_exposed_TRAIN_failure_diagnosis_not_independent_population=True,
        no_new_restored_image_or_visual_sheet=True, official_FP32_and_PNG_gates_unchanged=True,
        all_prior_failed_restoration_decisions_remain_binding=True,
        local_neural_calls=0, gradient_queries=0, optimizer_updates=0,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, response_failures=failures,
        activation_changed=[r['changed_sign_units'] for r in totals], model_qualified=False))


if __name__ == '__main__':
    main()

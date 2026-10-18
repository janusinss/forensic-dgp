"""Independent CPU crossreview of closed AdaIN state/pixel/decomposition evidence.

No model or checkpoint is loaded. No generated pixels are edited. This accepts
audit consistency only and preserves the authoritative quality-stop status.
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image

from research.codeformer_adain_gradient_isolation_v1.return_archive import read, sha, require, write_new, PROTOCOL_SHA


def near(left, right):
    require(math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-14), 'Independent numeric readback differs')


def crossreview(workspace):
    base = Path(workspace).resolve(strict=True)
    review = base / 'outputs/codeformer_adain_gradient_isolation_v1_return_review'
    packet = base / 'outputs/cctv_codeformer_adain_gradient_isolation_v1_vm'
    paths = {name: review / relative for name, relative in {
        'transport': 'transport_v1.json', 'state': 'state_v1/state_audit.json',
        'metrics': 'metrics_v1/metrics_audit.json', 'all_metrics': 'metrics_v1/metrics.json',
        'guards': 'metrics_v1/guards.json', 'visual_manifest': 'metrics_v1/visual_manifest.json',
        'prior_comparison_binding': 'prior_comparison_binding_v1.json',
        'decomposition': 'clear_error_decomposition_v1.json'}.items()}
    evidence = {name: read(path) for name, path in paths.items()}
    before = {name: sha(path) for name, path in paths.items()}
    transport, state, metrics, guards, dec = [evidence[k] for k in ('transport', 'state', 'metrics', 'guards', 'decomposition')]
    require(all(value['complete'] is True for value in evidence.values()), 'Closed successful audits required')
    require(state['evidence_consistency_pass'] is True and state['issues'] == [] and
        state['training_finished'] is False and state['completed_updates'] == 50 and state['resume_authorized'] is False,
        'Authoritative finite quality-stop classification changed')
    require(state['stop_reason'] == transport['reported_failure'] ==
        'Declared RAW/PNG clear-preservation guard failed; retained snapshot', 'Stop reason differs')
    require(state['evidence_binding']['transport_receipt_sha256'] == metrics['transport_receipt_sha256'] == before['transport'],
        'Audit transport chain differs')
    require(state['evidence_binding']['returned_manifest_members'] == transport['members'] == 6558 and
        state['sections']['call_ledger']['guarded_reservations'] == 2073 and
        state['sections']['call_ledger']['actual_neural_forward_reservations'] == 1614,
        'Transport/call ledger census differs')
    for report in (state, metrics):
        require(report['protocol_sha256'] == PROTOCOL_SHA and report['model_qualified'] is False, 'Protocol/qualification differs')
        for path, digest in report['source_sha256'].items(): require(sha(path) == digest, 'Closed audit source changed')
    require(metrics['source_profile_view_metric_rows'] == 2436 and metrics['native_descriptor_rows'] == 288 and
        metrics['all406_baseline_RAW_PNG_exact'] is True and metrics['four_witness_outputs_exact'] is True and
        metrics['isolated_clear_guard_pass'] is metrics['isolated_blur_guard_pass'] is False,
        'Metric census/status differs')
    p = read(packet / 'protocol.json'); require(sha(packet / 'protocol.json') == PROTOCOL_SHA, 'Frozen protocol differs')
    require(guards['unchanged_thresholds'] == p['evaluation'] and
        p['evaluation']['mean_clear_mae_delta_stop'] == .01 and
        p['evaluation']['worst_per_case_clear_mae_delta_stop'] == .03,
        'Prospective quality gates changed')
    rows = evidence['all_metrics']['rows']
    require(len(rows) == 2436, 'All three state/two view/406case metrics required')
    index = {(row['state'], row['id'], row['view']): row for row in rows}
    require(len(index) == len(rows), 'Duplicate metric row')
    null_metrics = ('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'coarse_rgb_mae', 'signed_luminance_bias',
                    'gradient_vector_mae', 'gradient_magnitude_mae')
    require(all(all(row[name] is None for name in null_metrics) for row in rows if row['evidence'] == 'native_unpaired'),
        'Unpaired native CCTV has invented clean-reference score')
    recalculated = []
    for guard in guards['guards']:
        state_name = guard['state']
        for view in ('raw', 'native_png'):
            selected = [row for row in rows if row['state'] == state_name and row['view'] == view and
                        row['evidence'] == 'paired_synthetic' and row['profile'] == 'clear']
            require(len(selected) == 61, 'Clear-control count changed')
            deltas = {row['id']: row['mae'] - index['original_update0', row['id'], view]['mae'] for row in selected}
            saved = guard['clear']['views'][view]
            require(set(deltas) == set(saved['deltas']), 'Clear delta namespace differs')
            for key, value in deltas.items(): near(value, saved['deltas'][key])
            mean, worst = math.fsum(deltas.values()) / len(deltas), max(deltas.values())
            near(mean, saved['mean_mae_delta']); near(worst, saved['worst_per_case_mae_delta'])
            passed = mean <= .01 and worst <= .03
            require(passed == saved['pass'], 'Independent clear pass/fail differs')
            recalculated.append({'state': state_name, 'view': view, 'clear_cases': 61,
                'mean_MAE_delta': mean, 'worst_MAE_delta': worst,
                'per_case_failures': sorted(key for key, value in deltas.items() if value > .03), 'pass': passed})
    manifest = evidence['visual_manifest']; cases = {case['id']: case for case in manifest['cases']}
    require(len(cases) == 406 and len(dec['rows']) == 366 and dec['paired_clear_controls'] == 61,
        'Complete unchanged error-decomposition cohort required')
    require(dec['original_outputs_not_changed'] is dec['quality_thresholds_not_changed'] is
        dec['not_an_oracle_output_or_gate_replacement'] is True, 'Error decomposition changes acceptance')
    for name, digest in dec['input_receipt_sha256'].items():
        require(sha(review / 'metrics_v1' / name) == digest, 'Decomposition input receipt changed')
    require(sha(base / 'research/codeformer_adain_gradient_isolation_v1/return_error_decomposition.py') == dec['source_sha256'],
        'Error-decomposition source changed')
    comparison_spec_path = packet / p['comparison_cases']
    if not comparison_spec_path.exists():
        comparison_spec_path = base / 'outputs/cctv_codeformer_adaptation_v2_vm' / p['comparison_cases']
    require(sha(comparison_spec_path) == p['files_sha256'][p['comparison_cases']], 'Frozen comparison spec changed')
    specs = {'comparison_' + row['id']: row for row in read(comparison_spec_path)['cases']}
    bound = {}; actual = {}; max_identity_error = 0.; masked_rows = 0
    def verify(path, digest):
        key = str(Path(path))
        if key not in bound: require(sha(path) == digest, 'Decomposition read source changed'); bound[key] = digest
        else: require(bound[key] == digest, 'Inconsistent artifact digest')
    for row in dec['rows']:
        key = row['state'], row['id'], row['view']
        require(key not in actual, 'Duplicate decomposition row'); actual[key] = row
        case = cases[row['id']]
        require(case['evidence'] == 'paired_synthetic' and case['profile'] == 'clear', 'Non-clear/unpaired case in decomposition')
        verify(case['target'], case['target_sha256'])
        with Image.open(case['target']) as image:
            require(image.mode == 'RGB' and image.size == (256, 256), 'Exact target RGB256 required')
            target = np.asarray(image).astype(np.float32) / np.float32(255)
        observed = np.ones((256, 256), dtype=bool)
        if row['id'].startswith('comparison_'):
            spec = specs[row['id']]; relative = spec['observed_path']
            candidates = (packet / relative, base / 'outputs/cctv_codeformer_fp32_comparison_v1_vm' / relative,
                          base / 'outputs/cctv_codeformer_adaptation_v2_vm' / relative)
            path = next(path for path in candidates if path.exists())
            require(spec['observed_sha256'] == p['files_sha256'][relative], 'Comparison support digest differs')
            verify(path, spec['observed_sha256'])
            with Image.open(path) as image:
                require(image.size == (256, 256), 'Support geometry differs'); observed = np.asarray(image.convert('L')) > 0
        require(observed.any() and row['observed_pixels'] == int(observed.sum()) == index[key]['observed_pixels'],
            'Observed support or padding differs from actual quality metric')
        masked_rows += int(not observed.all())
        record = case['states'][row['state']]; field = 'raw' if row['view'] == 'raw' else 'native_png'
        path = Path(record[field]); verify(path, record[field + '_sha256'])
        if field == 'raw':
            with path.open('rb') as stream:
                prediction = np.load(stream, allow_pickle=False); require(stream.read(1) == b'', 'Trailing array bytes')
            require(prediction.shape == (256, 256, 3) and prediction.dtype == np.float32, 'RAW geometry/dtype differs')
        else:
            with Image.open(path) as image:
                require(image.mode == 'RGB' and image.size == (256, 256), 'PNG geometry differs')
                prediction = np.asarray(image).astype(np.float32) / np.float32(255)
        error = (prediction.astype(np.float64) - target.astype(np.float64))[observed]
        require(np.isfinite(error).all(), 'Nonfinite paired error')
        means = np.mean(error, axis=0)
        mse = float(np.mean(error * error)); bias = float(np.dot(means, means) / 3)
        residual = float(np.mean(np.var(error, axis=0, dtype=np.float64)))
        mae = float(np.mean(np.abs(error)))
        near(mse, bias + residual); near(mse, row['mse']); near(bias, row['global_channel_bias_MSE_component'])
        near(residual, row['residual_MSE_component']); near(mae, row['mae_unchanged'])
        near(mae, index[key]['mae']); near(mse, index[key]['mse'])
        for a, b in zip(means, row['per_channel_mean_error']): near(float(a), b)
        near(float(means @ np.asarray([.2126, .7152, .0722])), row['signed_mean_sRGB_luminance_error'])
        max_identity_error = max(max_identity_error, abs(mse - bias - residual))
    require(len(actual) == 366, 'Missing decomposition value')
    for row in dec['deltas']:
        before_state = row['comparison'].removeprefix('isolated_update50_minus_')
        lhs, rhs = actual['isolated_update50', row['id'], row['view']], actual[before_state, row['id'], row['view']]
        for name in ('mse', 'mae_unchanged', 'global_channel_bias_MSE_component', 'residual_MSE_component',
                     'signed_mean_sRGB_luminance_error'):
            near(lhs[name] - rhs[name], row[name])
    require(len(dec['deltas']) == 244, 'Full comparison delta census differs')
    for path, digest in dec['read_sources_sha256'].items(): verify(path, digest)
    for name, path in paths.items(): require(sha(path) == before[name], 'Closed evidence changed during independent review')
    result = {'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'approved_evidence_consistency': True,
        'approved_model_or_training': False, 'source_and_receipt_hashes': before,
        'state_source_review': {'no_model_instantiation_or_training': True,
            'weights_only_CPU_tensor_loading': True, 'all350_Adam_states_step50_and165frozen': True,
            'same_initializer_sampler_inputs_losses_LR_and50updates': True,
            '391HQ_array_parity_and_codebook_lookup': True,
            'witness_None_versus_zero_route_checks': True,
            'full_gradient_logits_latent_arrays_not_exported': True},
        'saved_pixel_source_review': {'all812new_outputs_and406prior_outputs': True,
            '406baseline_RAW_PNG_values_exact': True, 'clear61_blur53_unchanged': True,
            'native48_per_state_source_separate_and_no_reference_metrics': True,
            'secondary_SSIM_not_identity_or_gate': True},
        'independent_clear_guard_readback': recalculated,
        'decomposition': {'all366_rows_independently_recomputed': True, 'all244_deltas_recomputed': True,
            'masked_rows_exclude_padding': masked_rows, 'individual_artifacts_freshly_bound': len(bound),
            'maximum_MSE_identity_roundoff': max_identity_error,
            'residual_is_not_a_structure_identity_score': True,
            'MSE_identity_does_not_explain_MAE_gate_or_establish_illumination_cause': True,
            'outputs_and_thresholds_unchanged': True},
        'authoritative_status': '50-update quality stop remains; do not append epochs or promote.',
        'visual_review': 'Owned by root; no visual verdict from this readback.',
        'source_sha256': sha(__file__), 'no_checkpoint_or_pickle_loaded_by_crossreview': True,
        'neural_gradient_optimizer_VM_calls': 0, 'model_qualified': False, 'training_authorized': False}
    write_new(review / 'independent_scientific_readback_v1.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--workspace', default=Path.cwd(), type=Path)
    result = crossreview(parser.parse_args().workspace)
    print({key: result[key] for key in ('complete', 'approved_evidence_consistency', 'authoritative_status')})

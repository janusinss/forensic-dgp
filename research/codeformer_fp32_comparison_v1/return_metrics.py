"""Independent CPU saved-pixel/guard comparison of finite FP32 and BF16 returns.

No restorer, neural framework, gradient, optimizer, VM connection or output
enhancement. Native CCTV has no invented clean-reference quality metric.
"""
from collections import Counter, defaultdict
import csv
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from research.codeformer_adaptation_v1 import return_metrics as v1
from research.codeformer_adaptation_v2 import return_metrics as v2


PROTOCOL_SHA = '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'
PARENT_SHA = v2.PROTOCOL_SHA
require, sha, read, write = v1.require, v1.sha, v1.read, v1.write
VIEWS = ('raw', 'native_png')
METRICS = ('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'coarse_rgb_mae',
           'mean_luminance', 'target_mean_luminance', 'signed_luminance_bias',
           'luminance_contrast', 'target_luminance_contrast', 'contrast_absolute_error',
           'gradient_vector_mae', 'gradient_magnitude_mae')
SSIM_POLICY = {'name': 'paired-observed-RGB-SSIM-win7-v1', 'input_dtype': 'float64 RGB[0,1]',
    'data_range': 1.0, 'win_size': 7, 'gaussian_weights': False, 'channel_axis': 2,
    'use_sample_covariance': True, 'K1': .01, 'K2': .03,
    'reduction': 'mean all3 channels at observed support eroded by7x7ones, border_value0',
    'native_clean_reference_metrics': 'not computed', 'quality_gate_changed': False}


def paired_secondary(prediction, target, observed, *, evidence):
    if evidence == 'native_unpaired':
        require(target is None, 'Native CCTV cannot have invented clean targets')
        return {name: None for name in ('ssim', 'mean_luminance', 'target_mean_luminance',
            'signed_luminance_bias', 'luminance_contrast', 'target_luminance_contrast',
            'contrast_absolute_error', 'gradient_vector_mae', 'gradient_magnitude_mae')}
    require(evidence == 'paired_synthetic' and prediction.shape == target.shape == (256, 256, 3) and
            observed.shape == (256, 256) and observed.dtype == np.bool_ and observed.any() and
            np.isfinite(prediction).all() and np.isfinite(target).all(), 'Paired secondary metric geometry differs')
    p, t = prediction.astype(np.float64), target.astype(np.float64)
    valid7 = binary_erosion(observed, structure=np.ones((7, 7), bool), border_value=0)
    valid3 = binary_erosion(observed, structure=np.ones((3, 3), bool), border_value=0)
    require(valid7.any() and valid3.any(), 'Insufficient observed windows for paired metrics')
    _, ssim_map = structural_similarity(p, t, data_range=1., win_size=7, channel_axis=2,
        gaussian_weights=False, use_sample_covariance=True, K1=.01, K2=.03, full=True)
    weights = np.asarray([.2126, .7152, .0722], np.float64)
    yp, yt = p @ weights, t @ weights
    def gradients(y):
        gx, gy = np.zeros_like(y), np.zeros_like(y)
        gx[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2
        gy[1:-1] = (y[2:] - y[:-2]) / 2
        return gx[valid3], gy[valid3]
    xp, yp_gradient = gradients(yp); xt, yt_gradient = gradients(yt)
    std_p, std_t = float(yp[observed].std()), float(yt[observed].std())
    return {'ssim': float(ssim_map[valid7].mean(dtype=np.float64)),
        'mean_luminance': float(yp[observed].mean()), 'target_mean_luminance': float(yt[observed].mean()),
        'signed_luminance_bias': float((yp[observed] - yt[observed]).mean()),
        'luminance_contrast': std_p, 'target_luminance_contrast': std_t,
        'contrast_absolute_error': abs(std_p - std_t),
        'gradient_vector_mae': float(((np.abs(xp - xt) + np.abs(yp_gradient - yt_gradient)) / 2).mean()),
        'gradient_magnitude_mae': float(np.abs(np.hypot(xp, yp_gradient) - np.hypot(xt, yt_gradient)).mean())}


def compare_saved(values, saved, spec, label):
    require(set(saved) == ({'mae', 'mse', 'psnr', 'luminance_bias', 'coarse_rgb_mae'}
        if spec['source'] == 'FFHQ validation' and spec['profile'] == 'blur' else {'mae', 'mse', 'psnr'}),
        'Saved metric extension schema differs: ' + label)
    for name, value in saved.items():
        if value is None:
            require(values[name] is None, 'Native/perfect metric NULL differs')
        else:
            require(v2.near(values[name], value), 'Saved scalar differs: ' + label + '/' + name)


def case_specs(packet, parent, protocol):
    """All immutable DEV dependencies resolve without copying or regeneration."""
    hashes = protocol['files_sha256']; dependency = protocol['dependency_files_sha256']
    checked = {}
    def bound(relative, expected=None):
        require(relative in hashes and (expected is None or hashes[relative] == expected), 'Unbound DEV dependency')
        root = packet if (packet / relative).exists() else parent
        require(root == packet or dependency.get(relative) == hashes[relative], 'Missing overlay dependency')
        path = v1.checked_path(root, relative)
        require(sha(path) == hashes[relative], 'DEV dependency hash differs: ' + relative)
        checked[relative] = hashes[relative]
        return path
    refs = read(bound(protocol['data_manifest']))['references']
    require(Counter(x['historical_role'] for x in refs) == {'train': 391, 'validation': 53}, 'Reference roles changed')
    validation = {row['id']: row for row in refs if row['historical_role'] == 'validation'}
    saved = read(bound(protocol['validation_saved']['manifest'], protocol['validation_saved']['sha256']))
    original = read(bound(saved['retained_original_manifest_path'], saved['retained_original_manifest_sha256']))
    require(saved['case_count'] == len(saved['items']) == 318 and not saved['locally_regenerated_development_pixels'],
            'Saved DEV namespace/reuse policy differs')
    require({row['id'] for row in saved['items']} == {'validation_' + ref + '_' + profile
        for ref in validation for profile in protocol['evaluation']['profiles']}, 'DEV identities/profiles differ')
    specs, receipts = {}, []
    for row in saved['items']:
        input_path = bound(row['input_path'], row['input_sha256'])
        record = read(bound(row['pair_record_path'], row['pair_record_sha256']))
        ref = validation[record['reference_id']]
        require(record['role'] == 'validation' and record['source_sha256'] == ref['source_sha256'] and
                record['reference_manifest_sha256'] == hashes[protocol['data_manifest']] and
                record['clean512_rgb8_sha256'] == ref['proposed_512_rgb_sha256'] and
                row['id'] == 'validation_' + ref['id'] + '_' + record['profile'], 'DEV role/source binding differs')
        require(original['cases'][row['id']]['sha256'] == row['input_sha256'], 'Original V1 VM input binding differs')
        target_path = bound(row['target_path'], row['target_sha256'])
        require(row['target_path'] == ref['existing_target_path'] and row['target_sha256'] == ref['existing_target_sha256'],
                'Direct validation target differs')
        pixels = v1.rgb8(input_path); floats = pixels.astype(np.float32) / np.float32(255)
        require(v1.data.pixel_sha256(pixels) == record['input256_rgb8_sha256'] and
                v1.data.pixel_sha256(floats) == record['input256_float32_sha256'], 'Saved DEV float32 input binding differs')
        specs[row['id']] = {'id': row['id'], 'source': 'FFHQ validation', 'evidence': 'paired_synthetic',
            'profile': record['profile'], 'input': input_path, 'input_relative': row['input_path'],
            'input_sha256': row['input_sha256'], 'input_pixels': pixels, 'target': target_path,
            'target_sha256': row['target_sha256'], 'target_pixels': v1.rgb8(target_path).astype(np.float32) / np.float32(255),
            'observed': np.ones((256, 256), dtype=bool)}
        receipts.append({'id': row['id'], 'input_sha256': row['input_sha256'],
                         'pair_record_sha256': row['pair_record_sha256'], 'saved_float32_pixels_exact': True})
    comparisons = read(bound(protocol['comparison_cases']))['cases']
    require(len(comparisons) == 88 and Counter(x['evidence'] for x in comparisons) ==
            {'paired_synthetic': 40, 'native_unpaired': 48}, 'Comparison roles/counts changed')
    for row in comparisons:
        input_path = bound(row['prepared_path'], row['prepared_sha256'])
        observed = v1.mask256(bound(row['observed_path'], row['observed_sha256']))
        target_path = bound(row['target_path'], row['target_sha256']) if row['evidence'] == 'paired_synthetic' else None
        require(row['evidence'] != 'native_unpaired' or not row.get('target_path'), 'Invented native clean target')
        identifier = 'comparison_' + row['id']
        require(identifier not in specs, 'Duplicate comparison case')
        specs[identifier] = {'id': identifier, 'source': row['source'], 'evidence': row['evidence'], 'profile': row['profile'],
            'input': input_path, 'input_relative': row['prepared_path'], 'input_sha256': row['prepared_sha256'],
            'input_pixels': v1.rgb8(input_path), 'target': target_path, 'target_sha256': row.get('target_sha256'),
            'target_pixels': v1.rgb8(target_path).astype(np.float32) / np.float32(255) if target_path else None,
            'observed': observed}
    require(len(specs) == 406, 'Full406 namespace required')
    return specs, receipts, checked, {r['id']: r for r in saved['items']}


def groups(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row['state'], row['view'], row['source'], row['evidence'], row['profile'])].append(row)
    results = []
    for key, items in sorted(buckets.items()):
        result = dict(zip(('state', 'view', 'source', 'evidence', 'profile'), key), cases=len(items))
        for name in METRICS:
            values = [x[name] for x in items if x.get(name) is not None]
            result['mean_' + name] = sum(values) / len(values) if values else None
        results.append(result)
    return results


def deltas(candidate, baseline, label):
    require(set(candidate) == set(baseline), 'Compared metric namespace differs')
    result = []
    for key in sorted(candidate):
        a, b = candidate[key], baseline[key]
        row = {name: a[name] for name in ('id', 'source', 'evidence', 'profile', 'view')}
        row['comparison'] = label
        row.update({name: a[name] - b[name] if a.get(name) is not None and b.get(name) is not None else None for name in METRICS})
        result.append(row)
    return result


def delta_groups(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row['comparison'], row['view'], row['source'], row['evidence'], row['profile'])].append(row)
    results = []
    for key, items in sorted(buckets.items()):
        result = dict(zip(('comparison', 'view', 'source', 'evidence', 'profile'), key), cases=len(items))
        for name in METRICS:
            values = [x[name] for x in items if x.get(name) is not None]
            result['mean_delta_' + name] = sum(values) / len(values) if values else None
            result['cases_lower_' + name] = sum(value < 0 for value in values) if values else None
            result['cases_higher_' + name] = sum(value > 0 for value in values) if values else None
        results.append(result)
    return results


def audit_return(imported, packet, parent, bf16_return, outputdir):
    imported, packet, parent, prior = [Path(value).resolve(strict=True)
                                       for value in (imported, packet, parent, bf16_return)]
    out = Path(outputdir).resolve()
    require(not out.exists(), 'Preserve prior metrics review; use a new audit directory')
    require(sha(imported / 'protocol.json') == sha(packet / 'protocol.json') == PROTOCOL_SHA, 'Frozen FP32 protocol differs')
    require(sha(parent / 'protocol.json') == sha(prior / 'protocol.json') == PARENT_SHA, 'Retained V2 protocol differs')
    p, old = read(packet / 'protocol.json'), read(parent / 'protocol.json')
    require(p['evaluation'] == old['evaluation'], 'Historical quality criteria changed')
    started = time.monotonic(); out.mkdir(parents=True)
    specs, input_receipts, hashed, saved_entries = case_specs(packet, parent, p)
    require(all(old['files_sha256'].get(name) == digest for name, digest in hashed.items()), 'FP32 DEV differs from BF16 DEV')
    run = imported / p['outputs']
    reuse = read(run / 'validation_inputs.json')
    require(reuse['complete'] and reuse['saved_V1_VM_pixels_reused_for_all_snapshots'] and
            reuse['cases_count'] == 318 and reuse['cases'] == saved_entries, 'Returned saved DEV input reuse differs')
    require({x.name for x in run.iterdir() if x.is_dir() and x.name.startswith('snapshot_')} == {'snapshot_0000', 'snapshot_0050'},
            'Exact two prospective snapshots required')
    all_rows, visual, decoded, summaries, cache = [], [], {}, [], {}
    prior_receipts = {u: read(prior / old['outputs'] / ('snapshot_%04d' % u) / 'metrics.json') for u in (0, 50)}
    for u in (0, 50):
        receipt = read(run / ('snapshot_%04d' % u) / 'metrics.json')
        require(receipt['complete'] and receipt['update'] == u and len(receipt['rows']) == 406, 'Complete406 snapshot required')
        saved = {x['id']: x for x in receipt['rows']}
        require(set(saved) == set(specs) and len(saved) == 406, 'FP32 snapshot namespace differs')
        prior_saved = {x['id']: x for x in prior_receipts[u]['rows']}
        require(set(prior_saved) == set(specs), 'BF16 snapshot namespace differs')
        states = [('FP32_update%d' % u, run / ('snapshot_%04d' % u), saved),
                  ('BF16_update%d' % u, prior / old['outputs'] / ('snapshot_%04d' % u), prior_saved)]
        for state, directory, records in states:
            require({x.name for x in directory.iterdir() if x.is_dir()} == set(specs), 'Output case directory namespace differs')
            clear, blur = {}, {}
            for rid in sorted(specs):
                spec, record = specs[rid], records[rid]
                require(all(record[k] == spec[k] for k in ('source', 'profile', 'evidence')) and
                        record['observed_pixels'] == int(spec['observed'].sum()), 'Output source/profile/support differs')
                if rid.startswith('validation_'):
                    require(record['input_path'] == spec['input_relative'] and record['input_sha256'] == spec['input_sha256'],
                            'Output does not bind identical saved DEV input')
                raw_path, png_path = directory / rid / 'raw.npy', directory / rid / 'native.png'
                raw = v1.load_raw(raw_path, record['raw_sha256'])
                require(sha(png_path) == record['native_png_sha256'], 'Native PNG hash differs')
                png = v1.rgb8(png_path)
                require(np.array_equal(png, v1.expected_native_png(raw, spec['input_pixels'], spec['observed'])),
                        'Native PNG floor/visible protected padding differs')
                values = {}
                for view, value in [('raw', raw), ('native_png', png.astype(np.float32) / np.float32(255))]:
                    m = v1.paired_metrics(value, spec['target_pixels'], spec['observed'], evidence=spec['evidence'])
                    if spec['target_pixels'] is not None and spec['observed'].all():
                        m.update(v2.photometry(value, spec['target_pixels']))
                    compare_saved(m, record[view], spec, state + '/' + rid + '/' + view)
                    m.update(paired_secondary(value, spec['target_pixels'], spec['observed'], evidence=spec['evidence']))
                    values[view] = m
                    row = {'state': state, 'snapshot': u, 'id': rid, 'view': view,
                        'source': spec['source'], 'evidence': spec['evidence'], 'profile': spec['profile'],
                        'observed_pixels': int(spec['observed'].sum()), **{name: m.get(name) for name in METRICS}}
                    all_rows.append(row); cache[(state, rid, view)] = row
                if spec['evidence'] == 'paired_synthetic' and spec['profile'] == 'clear':
                    clear[rid] = {view: values[view]['mae'] for view in VIEWS}
                if spec['source'] == 'FFHQ validation' and spec['profile'] == 'blur':
                    blur[rid] = {view: {name: values[view][name] for name in v2.BLUR_NAMES} for view in VIEWS}
                decoded[state, rid] = (raw_path, record['raw_sha256'], png_path, record['native_png_sha256'])
                if state.startswith('FP32'):
                    visual.append({'snapshot': u, 'snapshot_complete': True, 'id': rid,
                        'source': spec['source'], 'evidence': spec['evidence'], 'profile': spec['profile'],
                        'input': str(spec['input']), 'input_sha256': spec['input_sha256'],
                        'target': str(spec['target']) if spec['target'] else None, 'target_sha256': spec['target_sha256'],
                        'raw': str(raw_path), 'raw_sha256': record['raw_sha256'],
                        'native_png': str(png_path), 'native_png_sha256': record['native_png_sha256']})
            summaries.append({'state': state, 'snapshot': u, 'clear': clear, 'blur': blur,
                               'receipt_sha256': sha(directory / 'metrics.json'), 'case_count': len(saved)})
            saved_receipt = receipt if state.startswith('FP32') else prior_receipts[u]
            require(set(clear) == set(saved_receipt['clear']) and set(blur) == set(saved_receipt['blur']),
                    'Saved individual quality-control namespace differs')
            for rid in clear:
                require(all(v2.near(clear[rid][view], saved_receipt['clear'][rid][view]) for view in VIEWS),
                        'Saved clear-control scalar differs')
            for rid in blur:
                require(all(v2.near(blur[rid][view][name], saved_receipt['blur'][rid][view][name])
                            for view in VIEWS for name in v2.BLUR_NAMES), 'Saved blur-control scalar differs')
            if u == 0:
                require(saved_receipt['quality_guard'] is None and saved_receipt['normal_blur_photometric_guard'] is None,
                        'Untouched update0 cannot have candidate quality gate')
    baseline_by_state = {row['state']: row for row in summaries}
    gates, failure_cases = [], []
    for prefix, receipts in [('FP32', {u: read(run / ('snapshot_%04d' % u) / 'metrics.json') for u in (0, 50)}),
                             ('BF16', prior_receipts)]:
        base, current = baseline_by_state[prefix + '_update0'], baseline_by_state[prefix + '_update50']
        clear = v1.independently_clear_gate(current['clear'], base['clear'], p['evaluation'])
        blur = v2.independent_blur_gate(current['blur'], base['blur'], p['evaluation'])
        clear_status = v2.check_gate(clear, receipts[50]['quality_guard'], p['evaluation'], kind='clear')
        blur_status = v2.check_gate(blur, receipts[50]['normal_blur_photometric_guard'], p['evaluation'], kind='blur')
        require(clear_status['status'] == blur_status['status'] == 'MATCH', 'Guard numerical boundary ambiguity must retain review')
        gates.append({'state': prefix + '_update50', 'independent_clear_guard': clear, 'independent_blur_guard': blur,
                       'clear_saved_comparison': clear_status, 'blur_saved_comparison': blur_status})
        for rid in sorted(base['clear']):
            failed_views = [view for view in VIEWS if clear['views'][view]['deltas'][rid] > p['evaluation']['worst_per_case_clear_mae_delta_stop']]
            if failed_views:
                failure_cases.append({'state': prefix + '_update50', 'id': rid, 'failed_views': failed_views,
                    'raw_mae_delta': clear['views']['raw']['deltas'][rid],
                    'native_png_mae_delta': clear['views']['native_png']['deltas'][rid],
                    'input': str(specs[rid]['input']), 'target': str(specs[rid]['target']),
                    'native_png': str(decoded[prefix + '_update50', rid][2])})
    parity = []
    for rid in sorted(specs):
        a, b = decoded['FP32_update0', rid], decoded['BF16_update0', rid]
        av, bv = v1.load_raw(a[0], a[1]), v1.load_raw(b[0], b[1])
        error = np.abs(av.astype(np.float64) - bv.astype(np.float64))
        parity.append({'id': rid, 'RAW_exact_pixels': bool(np.array_equal(av, bv)),
            'RAW_maximum_absolute_difference': float(error.max()), 'RAW_mean_absolute_difference': float(error.mean()),
            'PNG_exact_pixels': bool(np.array_equal(v1.rgb8(a[2]), v1.rgb8(b[2])))})
        for row in (x for x in visual if x['id'] == rid):
            old50 = decoded['BF16_update50', rid]
            row.update({'BF16_update50_raw': str(old50[0]), 'BF16_update50_raw_sha256': old50[1],
                        'BF16_update50_native_png': str(old50[2]), 'BF16_update50_native_png_sha256': old50[3]})
    comparisons = []
    for left, right in [('FP32_update50', 'FP32_update0'), ('FP32_update50', 'BF16_update50'),
                         ('BF16_update50', 'BF16_update0')]:
        comparisons.extend(deltas({(rid, view): cache[left, rid, view] for rid in specs for view in VIEWS},
                                  {(rid, view): cache[right, rid, view] for rid in specs for view in VIEWS}, left + '_minus_' + right))
    write(out / 'input_audit.json', {'complete': True, 'saved_DEV_318': input_receipts, 'comparison_cases': 88,
        'native_unpaired': 48, 'all406_input_target_hashes_shared_with_V2': True,
        'DEV_regeneration_calls': 0, 'TRAIN_inputs_owned_by_independent_trajectory_audit': True})
    write(out / 'metrics.json', {'complete': True, 'rows': all_rows, 'groups': groups(all_rows),
        'SSIM_policy': SSIM_POLICY, 'native_reference_metrics_all_null': True})
    write(out / 'guards.json', {'complete': True, 'guards': gates, 'clear_case_failures': failure_cases,
        'unchanged_thresholds': p['evaluation'], 'model_qualified': False})
    write(out / 'comparisons.json', {'complete': True, 'deltas': comparisons, 'source_profile_deltas': delta_groups(comparisons),
        'baseline_parity': parity,
        'same50_update_budgets': True, 'precision_only_causal_claim_requires_target_cache_and_trajectory_review': True})
    write(out / 'visual_manifest.json', {'complete': True, 'cases': visual, 'planned_cases': 406,
        'comparison_views': ['input', 'target_or_native_blank', 'original_CF_FP32_update0', 'BF16_update50', 'FP32_update50'],
        'all812_actual_FP32_outputs_included': True, 'visual_review_pending': True, 'model_qualified': False})
    with (out / 'metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    result = {'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'snapshot_cases': {0: 406, 50: 406},
        'new_FP32_RAW_PNG_pairs': 812, 'retained_BF16_RAW_PNG_pairs_rechecked': 812,
        'source_profile_view_metric_rows': len(all_rows), 'paired_cases_per_state': 358, 'native_cases_per_state': 48,
        'exact_floor_and_protected_padding_checked': True, 'unchanged_clear_controls': 61, 'unchanged_blur_controls': 53,
        'FP32_clear_guard_pass': gates[0]['independent_clear_guard']['pass'],
        'FP32_blur_guard_pass': gates[0]['independent_blur_guard']['pass'],
        'all406_baseline_RAW_exact': all(x['RAW_exact_pixels'] for x in parity),
        'all406_baseline_PNG_exact': all(x['PNG_exact_pixels'] for x in parity),
        'native_reference_metrics_null': True, 'SSIM_is_secondary_not_identity_or_gate': True,
        'TRAIN_anchor_and_checkpoint_audit_separate': True, 'visual_review_pending': True,
        'source_sha256': sha(__file__), 'seconds': time.monotonic() - started,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
        'DEV_regeneration_calls': 0, 'reserved_final_pixel_reads': 0, 'model_qualified': False}
    write(out / 'metrics_audit.json', result)
    return result


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--imported', required=True)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--bf16-return', required=True)
    parser.add_argument('--outputdir', required=True)
    a = parser.parse_args()
    print(json.dumps(audit_return(a.imported, a.packet, a.parent, a.bf16_return, a.outputdir), indent=2))

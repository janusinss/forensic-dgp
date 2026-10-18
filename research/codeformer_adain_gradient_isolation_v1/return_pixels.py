"""Independent CPU saved-pixel audit of the finite AdaIN route comparison.

No neural imports/forward/gradient/optimizer, VM access, regenerated DEV, or
qualification. Immutable predecessor CPU metric primitives are reused and
source-hashed. Native CCTV never receives an invented clean target.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from research.codeformer_fp32_comparison_v1 import return_metrics as old
from research.codeformer_bounded_fusion_feasibility_v1.return_pixels import delivery_check

PROTOCOL_SHA = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
PARENT_SHA = old.PROTOCOL_SHA
require, sha, read, write = old.require, old.sha, old.read, old.write
VIEWS, METRICS = old.VIEWS, old.METRICS
STATES = ('original_update0', 'isolated_update50', 'retained_FP32_update50')
WEIGHTS = np.asarray([.2126, .7152, .0722], np.float64)


def load_array(path, shape, dtype):
    with Path(path).open('rb') as stream:
        value = np.load(stream, allow_pickle=False)
        require(stream.read(1) == b'', 'Trailing witness NPY bytes')
    require(value.shape == shape and value.dtype == np.dtype(dtype) and
            value.flags.c_contiguous and np.isfinite(value).all(), 'Witness geometry/dtype/finite differs')
    return value


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def unpaired_descriptors(prediction, observed, input_pixels):
    """Input-relative descriptors are not clean-reference quality/identity scores."""
    require(prediction.shape == input_pixels.shape == (256, 256, 3) and observed.any(),
            'Native descriptor geometry/support differs')
    p = prediction.astype(np.float64)
    y = p @ WEIGHTS
    inp = input_pixels.astype(np.float32) / np.float32(255)
    return {'observed_mean_sRGB_luminance': float(y[observed].mean()),
        'observed_luminance_std': float(y[observed].std()),
        'observed_saturated_scalar_fraction': float(((p[observed] == 0) | (p[observed] == 1)).mean()),
        'observed_MAE_change_from_input_not_target': float(np.abs(p[observed] - inp[observed]).mean()),
        'clean_reference_metrics_computed': False, 'is_quality_or_identity_evidence': False}


def make_failure_rows(gate, kind, state):
    rows = []
    for view in VIEWS:
        metrics = gate['views'][view] if kind == 'blur' else {'mae': gate['views'][view]}
        for name, summary in metrics.items():
            for rid, delta in summary['deltas'].items():
                if delta > .03:
                    rows.append({'state': state, 'control': kind, 'view': view, 'metric': name,
                        'id': rid, 'delta': delta, 'per_case_limit': .03})
    return rows


def audit_witness(imported, run):
    directory = run / 'gradient_route_witness'
    witness = read(directory / 'witness.json')
    require(witness['complete'] and witness['pass'] and len(witness['forwards']) == 4 and
            len(witness['parity']) == 2 and witness['optimizer_constructed'] is False and
            witness['optimizer_updates'] == witness['backward_calls'] == 0, 'Pre-Adam witness did not pass')
    rows, pixels, visuals = [], {}, []
    for row in witness['forwards']:
        slot, mode = row['slot'], row['mode']
        require(slot in (1, 2) and mode in ('original', 'isolated') and row['update'] == 3,
                'Witness is not the prospectively declared two inputs/four paths')
        case = directory / ('update_0003_slot%d_%s' % (slot, mode))
        require(read(case / 'record.json') == row, 'Witness record/report differs')
        require(set(row['files_sha256']) == {'native512.npy', 'raw256.npy', 'actual_tokens.npy', 'native.png'},
                'Witness file namespace differs')
        for name, digest in row['files_sha256'].items():
            require(sha(case / name) == digest, 'Witness artifact hash differs')
        native = load_array(case / 'native512.npy', (512, 512, 3), 'float32')
        raw = load_array(case / 'raw256.npy', (256, 256, 3), 'float32')
        tokens = load_array(case / 'actual_tokens.npy', (256,), 'int64')
        png = old.v1.rgb8(case / 'native.png')
        require(raw.min() >= 0 and raw.max() <= 1 and tokens.min() >= 0 and tokens.max() < 1024,
                'Witness RAW/token range differs')
        require(pixel_sha(raw.transpose(2, 0, 1)[None]) == row['array_sha256']['raw'] and
                pixel_sha(tokens) == row['array_sha256']['tokens'], 'Witness array value digest differs')
        require(np.array_equal(png, np.floor(raw * np.float32(255)).astype(np.uint8)),
                'Witness delivered PNG is not the exact floor of RAW')
        delivery = delivery_check(native, raw)
        pixels[slot, mode] = (native, raw, tokens, png)
        rows.append({'slot': slot, 'mode': mode, 'file_hashes_checked': 4,
            'native512_RGB_float32_sha256': pixel_sha(native), 'raw256_RGB_float32_sha256': pixel_sha(raw),
            'token_int64_sha256': pixel_sha(tokens), 'exact_floor_PNG': True, 'delivery': delivery,
            'signed_native_BCHW_hash_not_reconstructed_from_lossy_RGB_transform': True,
            'logits_latent_gradients_are_hash_or_source_readouts_not_exported_arrays': True})
        visuals.append({'slot': slot, 'mode': mode, 'reference_id': row['reference_id'],
            'native_png': str(case / 'native.png'), 'native_png_sha256': row['files_sha256']['native.png'],
            'raw': str(case / 'raw256.npy'), 'raw_sha256': row['files_sha256']['raw256.npy']})
        input_record = read(run / 'training_inputs' / 'update_0003' / ('slot_%d' % slot) / 'pair_record.json')
        input_path = old.v1.checked_path(imported, input_record['saved_input_path'])
        require(input_record['reference_id'] == row['reference_id'] and input_record['role'] == 'train' and
                sha(input_path) == input_record['saved_input_sha256'] and
                pixel_sha(old.v1.rgb8(input_path)) == input_record['input256_rgb8_sha256'],
                'Witness displayed input is not the saved matching TRAIN observation')
        visuals[-1].update(input=str(input_path), input_sha256=input_record['saved_input_sha256'])
    for slot in (1, 2):
        require(all(np.array_equal(a, b) for a, b in zip(pixels[slot, 'original'], pixels[slot, 'isolated'])),
                'Witness original/isolated exported native/RAW/token/PNG values differ')
    return {'complete': True, 'four_forward_outputs': rows, 'both_slots_exported_pixels_tokens_exact': True,
        'forward_parity_does_not_independently_replay_VM_gradients': True,
        'witness_json_sha256': sha(directory / 'witness.json'), 'visuals': visuals}


def native_metric_nulls(row):
    return all(row.get(name) is None for name in METRICS)


def audit_return(imported, packet, dependency, prior_return, transport_receipt, outputdir):
    imported, packet, dependency, prior = [Path(value).resolve(strict=True)
        for value in (imported, packet, dependency, prior_return)]
    out = Path(outputdir).resolve()
    require(not out.exists(), 'Preserve prior pixel audit; select a fresh version')
    require(sha(imported / 'protocol.json') == sha(packet / 'protocol.json') == PROTOCOL_SHA,
            'Exact frozen AdaIN protocol differs')
    require(sha(prior / 'protocol.json') == PARENT_SHA, 'Retained FP32 protocol differs')
    transport = read(transport_receipt)
    require(transport['complete'] and transport['protocol_sha256'] == PROTOCOL_SHA and
            transport['whole_archive_and_all_member_hashes_verified_before_import'] is True and
            transport['exact_namespace_verified'] is True,
            'Completed exact transport verification is required')
    p, oldp = read(packet / 'protocol.json'), read(prior / 'protocol.json')
    require(p['evaluation'] == oldp['evaluation'], 'Prospective preservation criteria changed')
    require(p['evaluation']['mean_clear_mae_delta_stop'] == p['evaluation']['mean_blur_photometric_delta_stop'] == .01 and
            p['evaluation']['worst_per_case_clear_mae_delta_stop'] == p['evaluation']['worst_per_case_blur_photometric_delta_stop'] == .03,
            'No threshold relaxation allowed')
    started = time.monotonic(); out.mkdir(parents=True)
    specs, input_receipts, hashed, saved_entries = old.case_specs(packet, dependency, p)
    require(all(oldp['files_sha256'].get(name) == digest for name, digest in hashed.items()),
            'AdaIN and retained FP32 do not bind identical DEV inputs/targets')
    run, prior_run = imported / p['outputs'], prior / oldp['outputs']
    reuse = read(run / 'validation_inputs.json')
    require(reuse['complete'] and reuse['saved_V1_VM_pixels_reused_for_all_snapshots'] and
            reuse['cases_count'] == 318 and reuse['cases'] == saved_entries, 'Returned DEV input reuse differs')
    require({x.name for x in run.iterdir() if x.is_dir() and x.name.startswith('snapshot_')} ==
            {'snapshot_0000', 'snapshot_0050'}, 'Exact two finite snapshots required')
    state_dirs = {'original_update0': run / 'snapshot_0000', 'isolated_update50': run / 'snapshot_0050',
                  'retained_FP32_update50': prior_run / 'snapshot_0050'}
    records, receipts = {}, {}
    for state, directory in state_dirs.items():
        receipt = read(directory / 'metrics.json')
        require(receipt['complete'] and receipt['update'] == (0 if state == 'original_update0' else 50) and
                len(receipt['rows']) == 406, 'Complete406 snapshot required')
        rows = {row['id']: row for row in receipt['rows']}
        require(len(rows) == 406 and set(rows) == set(specs) and
                {x.name for x in directory.iterdir() if x.is_dir()} == set(specs), 'Snapshot namespace differs')
        records[state], receipts[state] = rows, receipt
    old_zero = read(prior_run / 'snapshot_0000/metrics.json')
    old_zero_records = {row['id']: row for row in old_zero['rows']}
    require(old_zero['complete'] and set(old_zero_records) == set(specs), 'Retained baseline namespace differs')
    all_rows, native_rows, visual, decoded, summaries, cache = [], [], [], {}, [], {}
    for state, directory in state_dirs.items():
        clear, blur = {}, {}
        for index, rid in enumerate(sorted(specs), 1):
            spec, record = specs[rid], records[state][rid]
            require(all(record[key] == spec[key] for key in ('source', 'profile', 'evidence')) and
                    record['observed_pixels'] == int(spec['observed'].sum()), 'Output evidence/support differs')
            if rid.startswith('validation_'):
                require(record['input_path'] == spec['input_relative'] and record['input_sha256'] == spec['input_sha256'],
                        'Saved DEV input source differs')
            raw_path, png_path = directory / rid / 'raw.npy', directory / rid / 'native.png'
            raw = old.v1.load_raw(raw_path, record['raw_sha256'])
            require(sha(png_path) == record['native_png_sha256'], 'Delivered PNG hash differs')
            png = old.v1.rgb8(png_path)
            require(np.array_equal(png, old.v1.expected_native_png(raw, spec['input_pixels'], spec['observed'])),
                    'Exact floor/protected padding differs')
            values = {}
            for view, value in [('raw', raw), ('native_png', png.astype(np.float32) / np.float32(255))]:
                metrics = old.v1.paired_metrics(value, spec['target_pixels'], spec['observed'], evidence=spec['evidence'])
                if spec['target_pixels'] is not None and spec['observed'].all():
                    metrics.update(old.v2.photometry(value, spec['target_pixels']))
                old.compare_saved(metrics, record[view], spec, state + '/' + rid + '/' + view)
                metrics.update(old.paired_secondary(value, spec['target_pixels'], spec['observed'], evidence=spec['evidence']))
                row = {'state': state, 'snapshot': 0 if state == 'original_update0' else 50,
                    'id': rid, 'view': view, 'source': spec['source'], 'evidence': spec['evidence'],
                    'profile': spec['profile'], 'observed_pixels': int(spec['observed'].sum()),
                    **{name: metrics.get(name) for name in METRICS}}
                require(spec['evidence'] != 'native_unpaired' or native_metric_nulls(row),
                        'Native CCTV has an invented reference quality score')
                if spec['evidence'] == 'native_unpaired':
                    native_rows.append({key: row[key] for key in ('state', 'snapshot', 'id', 'view', 'source', 'profile')} |
                        unpaired_descriptors(value, spec['observed'], spec['input_pixels']))
                values[view] = metrics
                all_rows.append(row); cache[state, rid, view] = row
            if spec['evidence'] == 'paired_synthetic' and spec['profile'] == 'clear':
                clear[rid] = {view: values[view]['mae'] for view in VIEWS}
            if spec['source'] == 'FFHQ validation' and spec['profile'] == 'blur':
                blur[rid] = {view: {name: values[view][name] for name in old.v2.BLUR_NAMES} for view in VIEWS}
            decoded[state, rid] = {'raw': str(raw_path), 'raw_sha256': record['raw_sha256'],
                'native_png': str(png_path), 'native_png_sha256': record['native_png_sha256']}
            if index % 100 == 0:
                print(json.dumps({'CPU_saved_pixel_state': state, 'cases': index, 'of': 406}), flush=True)
        require(len(clear) == 61 and len(blur) == 53, 'Complete clear/blur controls required')
        saved = receipts[state]
        require(set(clear) == set(saved['clear']) and set(blur) == set(saved['blur']), 'Saved control namespace differs')
        require(all(old.v2.near(clear[rid][view], saved['clear'][rid][view]) for rid in clear for view in VIEWS),
                'Saved clear control differs')
        require(all(old.v2.near(blur[rid][view][name], saved['blur'][rid][view][name])
            for rid in blur for view in VIEWS for name in old.v2.BLUR_NAMES), 'Saved blur control differs')
        summaries.append({'state': state, 'clear': clear, 'blur': blur, 'receipt_sha256': sha(directory / 'metrics.json')})
    require(receipts['original_update0']['quality_guard'] is None and
            receipts['original_update0']['normal_blur_photometric_guard'] is None, 'Baseline cannot be quality-stopped')
    summaries = {row['state']: row for row in summaries}
    gates, failures = [], []
    for state in ('isolated_update50', 'retained_FP32_update50'):
        base, current = summaries['original_update0'], summaries[state]
        clear = old.v1.independently_clear_gate(current['clear'], base['clear'], p['evaluation'])
        blur = old.v2.independent_blur_gate(current['blur'], base['blur'], p['evaluation'])
        clear_status = old.v2.check_gate(clear, receipts[state]['quality_guard'], p['evaluation'], kind='clear')
        blur_status = old.v2.check_gate(blur, receipts[state]['normal_blur_photometric_guard'], p['evaluation'], kind='blur')
        require(clear_status['status'] == blur_status['status'] == 'MATCH', 'Numerical guard boundary needs review')
        gates.append({'state': state, 'clear': clear, 'blur': blur, 'saved_clear_comparison': clear_status,
            'saved_blur_comparison': blur_status})
        failures.extend(make_failure_rows(clear, 'clear', state)); failures.extend(make_failure_rows(blur, 'blur', state))
    parity = []
    for rid, spec in sorted(specs.items()):
        original = decoded['original_update0', rid]
        raw0 = old.v1.load_raw(original['raw'], original['raw_sha256'])
        png0 = old.v1.rgb8(original['native_png'])
        old_record = old_zero_records[rid]
        old_raw = old.v1.load_raw(prior_run / 'snapshot_0000' / rid / 'raw.npy', old_record['raw_sha256'])
        old_png_path = prior_run / 'snapshot_0000' / rid / 'native.png'
        require(sha(old_png_path) == old_record['native_png_sha256'], 'Prior baseline PNG hash differs')
        old_png = old.v1.rgb8(old_png_path)
        require(np.array_equal(raw0, old_raw) and np.array_equal(png0, old_png),
                'Cannot make same-initializer comparison with nonidentical baseline values')
        parity.append({'id': rid, 'RAW_exact': True, 'PNG_exact': True,
            'prior_RAW_sha256': old_record['raw_sha256'], 'prior_PNG_sha256': old_record['native_png_sha256']})
        visual.append({'id': rid, 'source': spec['source'], 'evidence': spec['evidence'], 'profile': spec['profile'],
            'input': str(spec['input']), 'input_sha256': spec['input_sha256'],
            'target': str(spec['target']) if spec['target'] else None, 'target_sha256': spec['target_sha256'],
            'states': {state: decoded[state, rid] for state in STATES}})
    comparisons = []
    for left, right in [('isolated_update50', 'original_update0'), ('isolated_update50', 'retained_FP32_update50'),
                        ('retained_FP32_update50', 'original_update0')]:
        comparisons.extend(old.deltas({(rid, view): cache[left, rid, view] for rid in specs for view in VIEWS},
            {(rid, view): cache[right, rid, view] for rid in specs for view in VIEWS}, left + '_minus_' + right))
    witness = audit_witness(imported, run)
    write(out / 'input_audit.json', {'complete': True, 'saved_DEV_318': input_receipts, 'comparison_cases': 88,
        'native_unpaired': 48, 'all406_inputs_targets_identical_to_retained_FP32': True,
        'input_files_sha256': hashed, 'DEV_regeneration_calls': 0, 'FINAL_pixel_reads': 0})
    write(out / 'metrics.json', {'complete': True, 'rows': all_rows, 'groups': old.groups(all_rows),
        'SSIM_policy': old.SSIM_POLICY, 'native_reference_metrics_all_null': True})
    write(out / 'native_descriptors.json', {'complete': True, 'rows': native_rows,
        'sources_kept_separate': True, 'no_aligned_clean_reference': True, 'not_quality_identity_or_local_performance': True})
    write(out / 'guards.json', {'complete': True, 'guards': gates, 'per_case_failures': failures,
        'unchanged_thresholds': p['evaluation'], 'model_qualified': False})
    write(out / 'comparisons.json', {'complete': True, 'deltas': comparisons,
        'source_profile_deltas': old.delta_groups(comparisons), 'baseline_parity': parity,
        'causal_interpretation_requires_separate_trajectory_audit': True})
    write(out / 'witness_pixels.json', witness)
    write(out / 'visual_manifest.json', {'complete': True, 'cases': visual, 'planned_cases': 406,
        'comparison_views': ['input', 'target_or_native_blank', *STATES],
        'all812_AdaIN_snapshot_outputs_included': True, 'all406_retained_FP32_u50_included': True,
        'witness': witness['visuals'], 'visual_review_pending': True, 'model_qualified': False})
    with (out / 'metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    modules = [Path(__file__), Path(old.__file__), Path(old.v1.__file__), Path(old.v2.__file__),
               Path(__import__(delivery_check.__module__, fromlist=['delivery_check']).__file__)]
    result = {'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'snapshot_cases': {'0': 406, '50': 406},
        'new_RAW_PNG_pairs': 812, 'retained_FP32_u50_RAW_PNG_pairs_rechecked': 406,
        'retained_FP32_u0_RAW_PNG_pairs_exact_value_rechecked': 406,
        'source_profile_view_metric_rows': len(all_rows), 'native_descriptor_rows': len(native_rows),
        'paired_cases_per_state': 358, 'native_cases_per_state': 48, 'unchanged_clear_controls': 61,
        'unchanged_blur_controls': 53, 'isolated_clear_guard_pass': gates[0]['clear']['pass'],
        'isolated_blur_guard_pass': gates[0]['blur']['pass'], 'exact_floor_and_padding': True,
        'all406_baseline_RAW_PNG_exact': True, 'four_witness_outputs_exact': True,
        'native_reference_metrics_null': True, 'SSIM_is_secondary_not_identity_or_gate': True,
        'visual_review_pending': True, 'source_sha256': {str(path): sha(path) for path in modules},
        'transport_receipt_sha256': sha(transport_receipt), 'seconds': time.monotonic() - started,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
        'DEV_regeneration_calls': 0, 'reserved_final_pixel_reads': 0, 'model_qualified': False}
    write(out / 'metrics_audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('imported', 'packet', 'dependency', 'prior-return', 'transport-receipt', 'outputdir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_return(args.imported, args.packet, args.dependency, args.prior_return,
        args.transport_receipt, args.outputdir), indent=2))


if __name__ == '__main__':
    main()

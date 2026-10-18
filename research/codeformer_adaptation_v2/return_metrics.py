"""Independent V2 saved-pixel audit; CPU only, no neural or VM operations.

V1's immutable strict RAW/PNG/masked-metric helpers are reused. V2 DEV inputs
are the exact saved V1 VM PNGs; no photographic degradation is regenerated.
Actual source roles, RAW/PNG decisions and partial/missing outputs stay explicit.
"""
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

from research.codeformer_adaptation_v1 import return_metrics as base

PROTOCOL_SHA = '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
REL_TOL, ABS_TOL = 2e-11, 2e-12
VIEWS = ('raw', 'native_png')
BLUR_NAMES = ('luminance_bias', 'coarse_rgb_mae')
require, sha, read, write = base.require, base.sha, base.read, base.write
checked_path, rgb8, mask256 = base.checked_path, base.rgb8, base.mask256
load_raw, expected_native_png, paired_metrics = base.load_raw, base.expected_native_png, base.paired_metrics


def near(a, b):
    return (isinstance(a, (int, float)) and not isinstance(a, bool)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL))


def photometry(prediction, target):
    """Full-support aligned RGB256; never applied to native or padded GT."""
    require(prediction.shape == target.shape == (256, 256, 3), 'Photometry geometry differs')
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), 'Nonfinite photometry')
    p, t = prediction.astype(np.float64), target.astype(np.float64)
    y = np.asarray([.2126, .7152, .0722], dtype=np.float64)
    yp, yt = p @ y, t @ y
    pp = p.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))
    pt = t.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))
    return {'luminance_bias': float(abs((yp - yt).mean())),
            'coarse_rgb_mae': float(np.abs(pp - pt).mean()),
            'mean_luminance': float(yp.mean()), 'target_mean_luminance': float(yt.mean()),
            'luminance_contrast': float(yp.std()), 'target_luminance_contrast': float(yt.std())}


def independent_blur_gate(candidate, baseline, evaluation):
    require(set(candidate) == set(baseline) and len(candidate) == 53, 'Blur namespace must contain exact53 views')
    result = {'pass': True, 'views': {}, 'qualification_not_implied': True}
    for view in VIEWS:
        metrics = {}
        for name in BLUR_NAMES:
            deltas = {identifier: candidate[identifier][view][name] - baseline[identifier][view][name]
                      for identifier in sorted(candidate)}
            require(all(math.isfinite(x) for x in deltas.values()), 'Nonfinite blur regression')
            mean, worst = sum(deltas.values()) / len(deltas), max(deltas.values())
            passed = (mean <= evaluation['mean_blur_photometric_delta_stop']
                      and worst <= evaluation['worst_per_case_blur_photometric_delta_stop'])
            metrics[name] = {'cases': len(deltas), 'mean_delta': mean,
                             'worst_per_case_delta': worst, 'deltas': deltas, 'pass': passed}
            result['pass'] = result['pass'] and passed
        result['views'][view] = metrics
    return result


def check_gate(rebuilt, saved, evaluation, *, kind):
    """Compare derived floats with tolerance; thresholds themselves are untouched.

    A decision difference within numeric tolerance is explicitly ambiguous,
    never silently promoted to a pass. Saved summary/decision must agree itself.
    """
    require(isinstance(saved, dict) and set(saved['views']) == set(VIEWS), 'Saved guard schema differs')
    ambiguous = []
    saved_pass = True
    for view in VIEWS:
        names = BLUR_NAMES if kind == 'blur' else (None,)
        for name in names:
            a = rebuilt['views'][view][name] if name else rebuilt['views'][view]
            b = saved['views'][view][name] if name else saved['views'][view]
            mean_key, worst_key = ('mean_delta', 'worst_per_case_delta') if name else ('mean_mae_delta', 'worst_per_case_mae_delta')
            mean_limit = evaluation['mean_blur_photometric_delta_stop' if name else 'mean_clear_mae_delta_stop']
            worst_limit = evaluation['worst_per_case_blur_photometric_delta_stop' if name else 'worst_per_case_clear_mae_delta_stop']
            require(a['cases'] == b['cases'] and set(a['deltas']) == set(b['deltas']), 'Guard namespace differs')
            require(near(a[mean_key], b[mean_key]) and near(a[worst_key], b[worst_key]), 'Guard summary differs')
            require(all(near(v, b['deltas'][k]) for k, v in a['deltas'].items()), 'Guard per-case delta differs')
            decision = b[mean_key] <= mean_limit and b[worst_key] <= worst_limit
            require(b['pass'] is decision, 'Saved guard decision contradicts saved scalars')
            saved_pass = saved_pass and decision
            if a['pass'] != b['pass']:
                require(near(a[mean_key], mean_limit) or near(a[worst_key], worst_limit),
                        'Guard decision differs beyond derived numerical tolerance')
                ambiguous.append({'view': view, 'metric': name, 'saved_pass': b['pass'], 'rebuilt_pass': a['pass']})
    require(saved['pass'] is saved_pass, 'Saved aggregate guard decision differs')
    return {'status': 'NEAR_THRESHOLD_REQUIRES_REVIEW' if ambiguous else 'MATCH',
            'saved_pass': saved_pass, 'rebuilt_pass': rebuilt['pass'],
            'numeric_boundary_differences': ambiguous, 'thresholds_changed': False}


def _bound_file(packet, protocol, relative, expected=None):
    path = checked_path(packet, relative)
    actual = sha(path)
    require(actual == protocol['files_sha256'][relative], 'Protocol-bound input file differs')
    if expected is not None:
        require(actual == expected, 'Declared input hash differs')
    return path


def case_specs(packet, protocol):
    """Load the exact saved318/88 input namespace without any regeneration."""
    refs_path = _bound_file(packet, protocol, protocol['data_manifest'])
    refs = read(refs_path)['references']
    require(Counter(r['historical_role'] for r in refs) == {'train': 391, 'validation': 53}, 'Reference roles differ')
    validation = {r['id']: r for r in refs if r['historical_role'] == 'validation'}
    saved = read(_bound_file(packet, protocol, protocol['validation_saved']['manifest'],
                            protocol['validation_saved']['sha256']))
    require(saved['format'] == 'codeformer-adaptation-saved-validation-v1'
            and saved['case_count'] == len(saved['items']) == 318, 'Saved validation namespace incomplete')
    original = read(_bound_file(packet, protocol, saved['retained_original_manifest_path'],
                               saved['retained_original_manifest_sha256']))
    expected_ids = {'validation_' + rid + '_' + profile
                    for rid in validation for profile in protocol['evaluation']['profiles']}
    require({r['id'] for r in saved['items']} == expected_ids and len(expected_ids) == 318,
            'Saved DEV identity/profile namespace differs')
    specs = {}
    receipts = []
    for row in saved['items']:
        input_path = _bound_file(packet, protocol, row['input_path'], row['input_sha256'])
        pair_path = _bound_file(packet, protocol, row['pair_record_path'], row['pair_record_sha256'])
        pair = read(pair_path)
        reference = validation[pair['reference_id']]
        require(pair['role'] == 'validation' and pair['source_sha256'] == reference['source_sha256']
                and pair['reference_manifest_sha256'] == protocol['files_sha256'][protocol['data_manifest']]
                and pair['clean512_rgb8_sha256'] == reference['proposed_512_rgb_sha256']
                and row['id'] == 'validation_' + reference['id'] + '_' + pair['profile'], 'DEV pair role/target/source differs')
        old = original['cases'][row['id']]
        require(old['sha256'] == row['input_sha256'] and row['original_VM_paths'] ==
                {'input': old['path'], 'pair_record': old['pair_record_path']}, 'Exact V1 input provenance differs')
        target_path = _bound_file(packet, protocol, row['target_path'], row['target_sha256'])
        require(row['target_path'] == reference['existing_target_path']
                and row['target_sha256'] == reference['existing_target_sha256'], 'DEV target binding differs')
        pixels = rgb8(input_path)
        floats = pixels.astype(np.float32) / np.float32(255)
        require(base.data.pixel_sha256(pixels) == pair['input256_rgb8_sha256']
                and base.data.pixel_sha256(floats) == pair['input256_float32_sha256'], 'Exact saved DEV pixel hash differs')
        specs[row['id']] = {'id': row['id'], 'evidence': 'paired_synthetic', 'source': 'FFHQ validation',
            'profile': pair['profile'], 'input_path': str(input_path), 'input_relative': row['input_path'],
            'input_sha256': row['input_sha256'], 'target_path': str(target_path),
            'target_sha256': row['target_sha256'], 'input_pixels': pixels,
            'target': rgb8(target_path).astype(np.float32) / np.float32(255),
            'observed': np.ones((256, 256), dtype=bool)}
        receipts.append({'id': row['id'], 'input_file_sha256': row['input_sha256'],
                         'pair_record_sha256': row['pair_record_sha256'], 'pixel_hashes_exact': True})
    comparisons = read(_bound_file(packet, protocol, protocol['comparison_cases']))['cases']
    require(len(comparisons) == 88 and Counter(r['evidence'] for r in comparisons) ==
            {'paired_synthetic': 40, 'native_unpaired': 48}, 'Comparison roles/counts differ')
    for row in comparisons:
        path = _bound_file(packet, protocol, row['prepared_path'], row['prepared_sha256'])
        observed = mask256(_bound_file(packet, protocol, row['observed_path'], row['observed_sha256']))
        target_path = None
        if row['evidence'] == 'paired_synthetic':
            target_path = _bound_file(packet, protocol, row['target_path'], row['target_sha256'])
        else:
            require(not row.get('target_path'), 'Native CCTV must remain unpaired')
        identifier = 'comparison_' + row['id']
        require(identifier not in specs, 'Duplicate comparison ID')
        specs[identifier] = {'id': identifier, 'evidence': row['evidence'], 'source': row['source'],
            'profile': row['profile'], 'input_path': str(path), 'input_relative': row['prepared_path'],
            'input_sha256': row['prepared_sha256'], 'input_pixels': rgb8(path), 'observed': observed,
            'target_path': str(target_path) if target_path else None,
            'target_sha256': row.get('target_sha256'),
            'target': rgb8(target_path).astype(np.float32) / np.float32(255) if target_path else None}
    require(len(specs) == 406, 'Full406 input namespace differs')
    return specs, receipts, {r['id']: r for r in saved['items']}


def _metric_check(values, saved, spec, label):
    basic = {k: saved[k] for k in ('mae', 'mse', 'psnr')}
    base.compare_metric_dict({k: values[k] for k in basic}, basic, label)
    extras = set(saved) - set(basic)
    expected_extra = set(BLUR_NAMES) if spec['source'] == 'FFHQ validation' and spec['profile'] == 'blur' else set()
    require(extras == expected_extra, 'Saved metric extension schema differs')
    for name in extras:
        require(near(values[name], saved[name]), 'Saved photometric metric differs')


def _groups(rows, metric_names, *, deltas=False):
    buckets = defaultdict(list)
    for row in rows:
        for profile in {row['profile'], 'all_clear' if row['profile'] == 'clear' else 'all_degraded'}:
            buckets[(row['snapshot'], row['snapshot_complete'], row['view'], row['source'], row['evidence'], profile)].append(row)
    groups = []
    for (update, complete, view, source, evidence, profile), values in sorted(buckets.items()):
        group = {'snapshot': update, 'snapshot_complete': complete, 'view': view, 'source': source,
                 'evidence': evidence, 'profile': profile, 'cases': len(values)}
        for name in metric_names:
            available = [r[name] for r in values if r.get(name) is not None]
            group['mean_' + name] = sum(available) / len(available) if available else None
            if deltas:
                group['cases_lower_' + name] = sum(v < 0 for v in available) if available else None
                group['cases_higher_' + name] = sum(v > 0 for v in available) if available else None
        groups.append(group)
    return groups


def audit_return(imported, packet, outputdir, *, prior_return=None):
    imported, packet, out = map(lambda p: Path(p).resolve(), (imported, packet, outputdir))
    require(sha(imported / 'protocol.json') == sha(packet / 'protocol.json') == PROTOCOL_SHA, 'Frozen V2 protocol differs')
    protocol = read(packet / 'protocol.json')
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), 'Keep previous pixel review; use a new empty output directory')
    started = time.monotonic()
    run = imported / protocol['outputs']
    require(run.is_dir(), 'No imported V2 run evidence')
    specs, input_receipts, saved_entries = case_specs(packet, protocol)
    returned_inputs = run / 'validation_inputs.json'
    if returned_inputs.exists():
        rinputs = read(returned_inputs)
        require(rinputs['complete'] and rinputs['cases_count'] == 318 and
                rinputs['saved_V1_VM_pixels_reused_for_all_snapshots'] is True and
                rinputs['cases'] == saved_entries, 'Returned saved-input reuse declaration differs')
    write(out / 'input_audit.json', {'complete': True, 'exact_saved_DEV_cases': input_receipts,
          'saved_DEV_regenerated': False, 'returned_input_manifest_present': returned_inputs.exists(),
          'comparisons': 88, 'native_clean_targets': 0})

    all_rows, visual, summaries, decoded = [], [], [], {}
    baseline_clear = baseline_blur = None
    directories = [p for p in run.iterdir() if p.is_dir() and p.name.startswith('snapshot_')]
    for snapshot in sorted(directories):
        require(re.fullmatch(r'snapshot_[0-9]{4}', snapshot.name), 'Malformed snapshot directory')
        update = int(snapshot.name[-4:])
        require(update in protocol['training']['snapshots'], 'Undeclared snapshot update')
        complete_file, partial_file = snapshot / 'metrics.json', snapshot / 'partial_metrics.json'
        require(not (complete_file.exists() and partial_file.exists()), 'Ambiguous snapshot receipts')
        receipt_path = complete_file if complete_file.exists() else partial_file if partial_file.exists() else None
        complete = complete_file.exists()
        receipt = read(receipt_path) if receipt_path else None
        if receipt:
            require(receipt['update'] == update and receipt['complete'] is complete, 'Snapshot status differs')
        else:
            require((run / 'failure.json').exists(), 'Unrecorded snapshot requires retained failure')
        saved_rows = receipt['rows'] if receipt else []
        saved = {r['id']: r for r in saved_rows}
        require(len(saved) == len(saved_rows) and set(saved) <= set(specs), 'Duplicate/unknown output rows')
        if complete:
            require(set(saved) == set(specs), 'Complete snapshot must contain exact406 cases')
        case_dirs = {p.name: p for p in snapshot.iterdir() if p.is_dir()}
        require(set(case_dirs) <= set(specs), 'Unknown snapshot case directory')
        artifacts = {rid for rid, folder in case_dirs.items() if (folder / 'raw.npy').exists() or (folder / 'native.png').exists()}
        require(not complete or artifacts == set(specs), 'Complete snapshot has missing/unrecorded output artifacts')
        clear, blur, orphan = {}, {}, []
        for rid in sorted(set(saved) | artifacts):
            spec, record = specs[rid], saved.get(rid)
            folder = snapshot / rid
            raw_path, png_path = folder / 'raw.npy', folder / 'native.png'
            require(raw_path.exists(), 'Native PNG or metric row without preceding RAW')
            raw_hash = record['raw_sha256'] if record else sha(raw_path)
            raw = load_raw(raw_path, raw_hash)
            png = None
            if png_path.exists():
                require(sha(png_path) == (record['native_png_sha256'] if record else sha(png_path)), 'Saved PNG hash differs')
                png = rgb8(png_path)
                require(np.array_equal(png, expected_native_png(raw, spec['input_pixels'], spec['observed'])), 'PNG floor/protected padding differs')
            require(record is None or png is not None, 'Recorded metric row lacks PNG')
            if record:
                require(all(record[k] == spec[k] for k in ('evidence', 'profile', 'source')) and
                        record['observed_pixels'] == int(spec['observed'].sum()), 'Row role/support differs')
                if rid.startswith('validation_'):
                    require(record['input_sha256'] == spec['input_sha256'] and
                            record['input_path'] == spec['input_relative'], 'Snapshot exact DEV input differs')
            else:
                require(not complete, 'Unrecorded output in complete snapshot')
                orphan.append({'id': rid, 'raw_present': True, 'PNG_present': png is not None,
                               'VM_metric_row_missing': True, 'raw_sha256': raw_hash})
            values = {}
            for view, prediction in [('raw', raw), ('native_png', png.astype(np.float32) / np.float32(255) if png is not None else None)]:
                if prediction is None:
                    continue
                metrics = paired_metrics(prediction, spec['target'], spec['observed'], evidence=spec['evidence'])
                extra = photometry(prediction, spec['target']) if spec['target'] is not None and spec['observed'].all() else {}
                metrics.update(extra)
                values[view] = metrics
                if record:
                    _metric_check(metrics, record[view], spec, f'{snapshot.name}/{rid}/{view}')
                all_rows.append({'snapshot': update, 'snapshot_complete': complete, 'VM_metric_row_present': record is not None,
                    'id': rid, 'source': spec['source'], 'evidence': spec['evidence'], 'profile': spec['profile'],
                    'view': view, 'observed_pixels': int(spec['observed'].sum()),
                    **{k: metrics.get(k) for k in ('mae', 'mse', 'psnr', *BLUR_NAMES,
                       'mean_luminance', 'target_mean_luminance', 'luminance_contrast', 'target_luminance_contrast')}})
            if record and spec['evidence'] == 'paired_synthetic' and spec['profile'] == 'clear':
                clear[rid] = {v: values[v]['mae'] for v in VIEWS}
            if record and spec['source'] == 'FFHQ validation' and spec['profile'] == 'blur':
                blur[rid] = {v: {k: values[v][k] for k in BLUR_NAMES} for v in VIEWS}
            visual.append({'snapshot': update, 'snapshot_complete': complete, 'id': rid,
                'source': spec['source'], 'evidence': spec['evidence'], 'profile': spec['profile'],
                'input': spec['input_path'], 'input_sha256': spec['input_sha256'],
                'target': spec['target_path'], 'target_sha256': spec['target_sha256'],
                'raw': str(raw_path), 'raw_sha256': raw_hash,
                'native_png': str(png_path) if png is not None else None,
                'native_png_sha256': sha(png_path) if png is not None else None,
                'VM_metric_row_present': record is not None})
            decoded[update, rid] = (raw_path, raw_hash, png_path if png is not None else None)
        if receipt:
            require(set(clear) == set(receipt['clear']), 'Saved clear namespace differs')
            for rid in clear:
                require(all(near(clear[rid][v], receipt['clear'][rid][v]) for v in VIEWS), 'Saved clear values differ')
        gate = blur_gate = None
        verified_clear = verified_blur = None
        if update == 0 and complete:
            require(len(clear) == 61 and len(blur) == 53 and receipt['quality_guard'] is None and
                    receipt['normal_blur_photometric_guard'] is None, 'Baseline control receipt differs')
            baseline_clear, baseline_blur = clear, blur
        elif complete:
            require(baseline_clear is not None and baseline_blur is not None, 'Missing complete update0 baseline')
            gate = base.independently_clear_gate(clear, baseline_clear, protocol['evaluation'])
            blur_gate = independent_blur_gate(blur, baseline_blur, protocol['evaluation'])
            verified_clear = check_gate(gate, receipt['quality_guard'], protocol['evaluation'], kind='clear')
            verified_blur = check_gate(blur_gate, receipt['normal_blur_photometric_guard'], protocol['evaluation'], kind='blur')
        if complete:
            require(set(blur) == set(receipt['blur']), 'Saved blur namespace differs')
            for rid in blur:
                require(all(near(blur[rid][v][k], receipt['blur'][rid][v][k]) for v in VIEWS for k in BLUR_NAMES), 'Saved blur values differ')
        summaries.append({'snapshot': update, 'complete': complete, 'recorded_rows': len(saved),
            'actual_output_cases': len(artifacts), 'missing_case_ids': sorted(set(specs) - artifacts),
            'partial_unrecorded_outputs': orphan, 'metrics_receipt_missing': receipt is None,
            'clear_controls': len(clear), 'blur_controls': len(blur),
            'independent_clear_guard': gate, 'independent_blur_guard': blur_gate,
            'clear_guard_saved_comparison': verified_clear, 'blur_guard_saved_comparison': verified_blur,
            'receipt_sha256': sha(receipt_path) if receipt_path else None})

    metrics_names = ('mae', 'mse', 'psnr', *BLUR_NAMES, 'mean_luminance', 'luminance_contrast')
    baseline_rows = {(r['id'], r['view']): r for r in all_rows if r['snapshot'] == 0}
    deltas = []
    for row in all_rows:
        if row['snapshot'] == 0 or (row['id'], row['view']) not in baseline_rows:
            continue
        initial = baseline_rows[row['id'], row['view']]
        delta = {k: row[k] for k in ('snapshot', 'snapshot_complete', 'id', 'source', 'evidence', 'profile', 'view')}
        delta.update({k: row[k] - initial[k] if row.get(k) is not None and initial.get(k) is not None else None for k in metrics_names})
        deltas.append(delta)
    # V1 comparison is a separate historical development trajectory, not the
    # prospective same-VM V2 guard baseline. Missing prior pixels remain explicit.
    prior_return = Path(prior_return).resolve() if prior_return else packet.parent / 'codeformer_adaptation_v1_return'
    prior_metrics = {}
    prior_parity, prior_changes = [], []
    if prior_return.is_dir():
        require(sha(prior_return / 'protocol.json') == base.PROTOCOL_SHA, 'Historical V1 protocol changed')
        require(sha(prior_return / 'references.json') == sha(packet / protocol['data_manifest'])
                and sha(prior_return / 'comparison_cases.json') == sha(packet / protocol['comparison_cases']),
                'Historical V1 source/comparison roles differ')
        for at in (0, 200):
            f = prior_return / 'outputs/adaptation_v1' / f'snapshot_{at:04d}' / 'metrics.json'
            if f.exists():
                if at == 0:
                    require(sha(f) == protocol['prior_baseline']['sha256'] ==
                            sha(_bound_file(packet, protocol, protocol['prior_baseline']['metrics'])),
                            'Retained historical baseline metrics changed')
                prior_metrics[at] = {r['id']: r for r in read(f)['rows']}
        for (at, rid), (raw_path, raw_hash, png_path) in decoded.items():
            for prior_at in (0, 200):
                if rid not in prior_metrics.get(prior_at, {}):
                    continue
                oldrow = prior_metrics[prior_at][rid]
                require(all(oldrow[k] == specs[rid][k] for k in ('source', 'profile', 'evidence')),
                        'Historical V1 output role differs')
                if rid.startswith('validation_'):
                    require(oldrow['input_sha256'] == specs[rid]['input_sha256'],
                            'Historical V1 did not use the exact same saved DEV pixels')
                olddir = prior_return / 'outputs/adaptation_v1' / f'snapshot_{prior_at:04d}' / rid
                oldraw = load_raw(olddir / 'raw.npy', oldrow['raw_sha256'])
                oldpng_path = olddir / 'native.png'
                require(sha(oldpng_path) == oldrow['native_png_sha256'], 'V1 historical PNG changed')
                oldpng = rgb8(oldpng_path)
                current_raw = load_raw(raw_path, raw_hash)
                current_png = rgb8(png_path) if png_path else None
                raw_error = np.abs(current_raw.astype(np.float64) - oldraw.astype(np.float64))
                record = {'V2_snapshot': at, 'V1_snapshot': prior_at, 'id': rid,
                    'source': specs[rid]['source'], 'profile': specs[rid]['profile'], 'evidence': specs[rid]['evidence'],
                    'raw_maximum_absolute_difference': float(raw_error.max()),
                    'raw_mean_absolute_difference': float(raw_error.mean()),
                    'RAW_exact_pixels': bool(np.array_equal(current_raw, oldraw)),
                    'PNG_exact_pixels': bool(np.array_equal(current_png, oldpng)) if current_png is not None else None,
                    'PNG_maximum_abs_255': int(np.abs(current_png.astype(np.int16)-oldpng.astype(np.int16)).max()) if current_png is not None else None}
                if at == 0 and prior_at == 0:
                    prior_parity.append(record)
                if prior_at == 200:
                    prior_changes.append(record)
                item = next(v for v in visual if v['snapshot'] == at and v['id'] == rid)
                item['V1_update' + str(prior_at) + '_raw'] = str(olddir / 'raw.npy')
                item['V1_update' + str(prior_at) + '_native_png'] = str(oldpng_path)
                item['V1_update' + str(prior_at) + '_raw_sha256'] = oldrow['raw_sha256']
                item['V1_update' + str(prior_at) + '_native_png_sha256'] = oldrow['native_png_sha256']
    write(out / 'prior_V1_comparison.json', {'complete': True, 'prior_return_present': prior_return.is_dir(),
          'V2_update0_vs_V1_update0': prior_parity, 'V2_vs_stopped_V1_update200': prior_changes,
          'V2_same_VM_update0_is_guard_baseline': True, 'cross_runtime_pixel_equality_not_assumed': True,
          'quality_or_identity_not_implied': True})
    write(out / 'metrics.json', {'complete': True, 'rows': all_rows, 'groups': _groups(all_rows, metrics_names),
          'native_reference_metrics_null': True, 'paired_synthetic_separate': True})
    write(out / 'candidate_vs_update0.json', {'complete': True, 'case_deltas': deltas,
          'source_profile_deltas': _groups(deltas, metrics_names, deltas=True), 'thresholds_unchanged': True})
    if all_rows:
        with (out / 'metrics.csv').open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    write(out / 'visual_manifest.json', {'complete': True, 'cases': visual,
          'all_actual_artifact_cases_included': True, 'visual_review_pending': True, 'candidate_qualified': False})
    audit = {'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'snapshot_summaries': summaries,
        'missing_planned_snapshots': sorted(set(protocol['training']['snapshots']) - {s['snapshot'] for s in summaries}),
        'exact_saved_DEV_inputs': 318, 'comparison_cases': 88, 'metric_rows': len(all_rows),
        'native_reference_metrics_null': True, 'exact_floor_and_protected_padding_checked': True,
        'derived_float_tolerance': {'relative': REL_TOL, 'absolute': ABS_TOL, 'does_not_change_guard_thresholds': True},
        'prior_V1_baseline_pixel_comparisons': len(prior_parity), 'prior_V1_candidate_comparisons': len(prior_changes),
        'review_source_sha256': sha(__file__), 'seconds': time.monotonic()-started,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
        'DEV_regeneration_calls': 0, 'reserved_final_pixel_reads': 0,
        'visual_review_pending': True, 'model_qualified': False, 'trajectory_and_anchors_audit_separate': True}
    write(out / 'metrics_audit.json', audit)
    return audit

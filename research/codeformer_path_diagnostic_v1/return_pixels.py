"""Independent saved-pixel/token replay of the finite path diagnostic.

CPU arrays and Torch bilinear image geometry only. This module never imports a
restorer, constructs a neural module, executes a forward, calculates a gradient,
loads a checkpoint, creates an optimizer or connects to a VM. Its outputs are
diagnostic comparisons on eight already observed TRAIN cases, not candidates.
"""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .contract import bounded, read_json, require, sha256, write_new_json


PROTOCOL_SHA = '2d3dcd1dbc3be371de5e10ed9c059e455ce9cc8376bac99f04a1a514acc0d49f'
MODES = ('E32', 'E32_grad', 'T32', 'TBF16')
STATES = ('original', 'stopped50')
ARMS = ('original_moments__current_SFT', 'current_moments__original_SFT',
        'original_moments__original_SFT')
# Derived CPU/GPU geometry scalars, never a quality threshold. Arrays, hashes,
# selected indices and delivered PNG conversion remain exact.
ABS_TOL, REL_TOL = 1e-8, 1e-7
METRIC_NAMES = ('luminance_mean', 'luminance_std', 'target_luminance_mean',
                'target_luminance_std', 'signed_luminance_bias', 'RGB_MAE',
                'RGB_MSE', 'mean_absolute_first_difference_error',
                'delivered_RGB_clip_fraction')


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def strict_array(path, shape, dtype):
    """Reject hidden trailing payloads, wrong layout/dtype and nonfinite data."""
    with Path(path).open('rb') as stream:
        value = np.load(stream, allow_pickle=False)
        require(stream.read(1) == b'', 'Trailing NPY data: ' + str(path))
    require(value.shape == shape and value.dtype == np.dtype(dtype), 'Array shape/dtype differs: ' + str(path))
    require(value.flags.c_contiguous and np.isfinite(value).all(), 'Invalid array layout/nonfinite data: ' + str(path))
    return value


def rgb8(path, size=(256, 256)):
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == size,
                'Expected exact RGB PNG geometry: ' + str(path))
        return np.asarray(image).copy()


def delivered_png(raw):
    require(raw.dtype == np.float32 and raw.ndim == 3 and raw.shape[-1] == 3 and
            np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1, 'Invalid delivered RAW')
    return np.floor(np.clip(raw, 0, 1) * 255).astype(np.uint8)


def target256(clean):
    """Image geometry only: exact retained float32 target -> CPU bilinear256."""
    import torch
    from torch.nn import functional as F
    value = torch.from_numpy(np.ascontiguousarray(clean).copy()).permute(2, 0, 1).unsqueeze(0)
    require(value.device.type == 'cpu' and not value.requires_grad, 'Geometry must stay CPU without autograd')
    return F.interpolate(value, (256, 256), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).contiguous().numpy()


def metrics(raw, target):
    require(raw.shape == target.shape and raw.ndim == 3 and raw.shape[-1] == 3,
            'Paired metric geometry differs')
    require(np.isfinite(raw).all() and np.isfinite(target).all(), 'Nonfinite paired pixels')
    x, y = raw.astype(np.float64), target.astype(np.float64)
    weights = np.asarray([.2126, .7152, .0722], dtype=np.float64)
    lx, ly = x @ weights, y @ weights
    error = x - y
    derivative = .5 * (np.abs(np.diff(x, axis=0) - np.diff(y, axis=0)).mean() +
                       np.abs(np.diff(x, axis=1) - np.diff(y, axis=1)).mean())
    return {'luminance_mean': float(lx.mean()), 'luminance_std': float(lx.std()),
            'target_luminance_mean': float(ly.mean()), 'target_luminance_std': float(ly.std()),
            'signed_luminance_bias': float((lx - ly).mean()), 'RGB_MAE': float(np.abs(error).mean()),
            'RGB_MSE': float(np.square(error).mean()),
            'mean_absolute_first_difference_error': float(derivative),
            'delivered_RGB_clip_fraction': float(((raw <= 0) | (raw >= 1)).mean())}


def scalar_equal(actual, reported, name):
    require(type(actual) in (int, float) and type(reported) in (int, float) and
            math.isfinite(actual) and math.isfinite(reported), 'Nonfinite/non-numeric scalar: ' + name)
    require(math.isclose(actual, reported, rel_tol=REL_TOL, abs_tol=ABS_TOL),
            'Saved/rebuilt scalar differs: %s: %.17g vs %.17g' % (name, actual, reported))


def token_statistics(tokens, probabilities, indices, logits_top2):
    require(tokens.shape == (256,) and tokens.dtype == np.int64 and
            np.all((tokens >= 0) & (tokens < 1024)), 'Invalid captured actual tokens')
    require(probabilities.shape == indices.shape == logits_top2.shape == (256, 2), 'Top2 geometry differs')
    require(probabilities.dtype == logits_top2.dtype == np.float32 and indices.dtype == np.int64,
            'Top2 dtype differs')
    require(np.isfinite(probabilities).all() and np.isfinite(logits_top2).all(), 'Nonfinite top2')
    require(np.all((probabilities >= 0) & (probabilities <= 1)) and
            np.all((indices >= 0) & (indices < 1024)) and np.all(indices[:, 0] != indices[:, 1]),
            'Invalid probability/index top2')
    require(np.all(probabilities[:, 0] >= probabilities[:, 1]) and
            np.all(logits_top2[:, 0] >= logits_top2[:, 1]), 'Top2 must be descending')
    margins = probabilities[:, 0] - probabilities[:, 1]
    nonfirst = tokens != indices[:, 0]
    require(np.all(margins[nonfirst] == 0), 'Captured selection disagrees with a strict native probability winner')
    return {'native_probability_top2_margin_minimum': float(margins.min()),
            'native_probability_top2_zero_margin_tokens': int(np.count_nonzero(margins == 0)),
            'selected_index_not_top2_first_due_to_ties': int(np.count_nonzero(nonfirst)),
            'FP32_logits_top2_margin_minimum': float((logits_top2[:, 0] - logits_top2[:, 1]).min()),
            'FP32_logits_top2_zero_margin_tokens': int(np.count_nonzero(logits_top2[:, 0] == logits_top2[:, 1])),
            'actual_selection_independently_recomputed_from_full_logits': False,
            'FP32_argmax_not_substituted_for_captured_native_selection': True}


def differences(a, b, tokens_a, tokens_b):
    error = a.astype(np.float64) - b.astype(np.float64)
    return {'token_disagreements': int(np.count_nonzero(tokens_a != tokens_b)),
            'RAW_maximum_absolute': float(np.abs(error).max()),
            'RAW_mean_absolute': float(np.abs(error).mean()),
            'PNG_pixel_values_different': int(np.count_nonzero(delivered_png(a) != delivered_png(b)))}


def factorial_contrasts(arms):
    """Conditional2x2 contrasts at fixed stopped content/selected tokens.

    Values are scalar metric effects, not independent causal training results.
    C=current moments/current fusion; M=original moments/current fusion;
    F=current moments/original fusion; MF=both original substitutions.
    """
    require(set(arms) == {'C', 'M', 'F', 'MF'}, 'Incomplete conditional2x2')
    names = ('luminance_mean', 'luminance_std', 'signed_luminance_bias', 'RGB_MAE',
             'RGB_MSE', 'mean_absolute_first_difference_error')
    return {name: {'M_minus_C': arms['M'][name] - arms['C'][name],
                   'F_minus_C': arms['F'][name] - arms['C'][name],
                   'MF_minus_C': arms['MF'][name] - arms['C'][name],
                   'interaction_MF_minus_M_minus_F_plus_C': arms['MF'][name] - arms['M'][name] -
                                                        arms['F'][name] + arms['C'][name]}
            for name in names}


def expected_conditions(cases):
    rows = []
    for state in STATES:
        for case in cases:
            for mode in MODES:
                rows.append(('path', state, case, mode, None,
                             'paths/%s/%s/%s' % (state, case['id'], mode)))
    for state in STATES:
        for case in cases:
            if (case['update'], case['slot']) in {(1, 1), (1, 3)}:
                for mode in MODES:
                    rows.append(('repeat', state, case, mode, None,
                                 'repeats/%s/%s/%s' % (state, case['id'], mode)))
    for case in cases:
        for arm in ARMS:
            rows.append(('intervention', 'stopped50', case, 'E32', arm,
                         'interventions/%s/%s' % (case['id'], arm)))
    require(len(rows) == 104 and len({row[-1] for row in rows}) == 104, 'Expected104 exact conditions')
    return rows


def _aggregate(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: {'count': len(values), 'metrics_mean': {
        metric: sum(row['metrics_raw'][metric] for row in values) / len(values)
        for metric in METRIC_NAMES},
        'native_probability_ties_total': sum(row['token_statistics']['native_probability_top2_zero_margin_tokens']
                                             for row in values)}
        for name, values in sorted(groups.items())}


def audit(return_root, packet_root, output_dir):
    """Audit only a transport-verified import, without modifying return/packet."""
    started = time.monotonic()
    root = Path(return_root).resolve(strict=True)
    packet = Path(packet_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    require(not output.exists() and not output.is_relative_to(root) and not output.is_relative_to(packet),
            'Preserve existing audit, return and frozen packet')
    protocol = read_json(root / 'protocol.json')
    require(sha256(root / 'protocol.json') == sha256(packet / 'protocol.json') == PROTOCOL_SHA,
            'Protocol differs from reviewed packet')
    manifest = read_json(root / protocol['outputs'] / 'export_manifest.json')
    require(manifest['protocol_sha256'] == PROTOCOL_SHA, 'Export protocol differs')
    file_hashes = manifest['files_sha256']
    checks = []

    def checked(relative):
        path = bounded(root, relative)
        require(relative in file_hashes and sha256(path) == file_hashes[relative], 'Return file hash differs: ' + relative)
        checks.append(relative)
        return path

    def packet_bound(relative, digest):
        # The result exporter intentionally omits source observations/targets.
        # Read the retained original packet, whose protocol bytes match return.
        path = bounded(packet, relative)
        require(protocol['files_sha256'][relative] == digest and sha256(path) == digest,
                'Exact original packet input binding differs: ' + relative)
        checks.append('original-packet/' + relative)
        return path

    cases = protocol['cases']
    require(len(cases) == 8 and {(c['update'], c['slot']) for c in cases} ==
            {(u, s) for u in (1, 24) for s in range(1, 5)} and
            len({c['reference_id'] for c in cases}) == 4, 'TRAIN cohort differs')
    loaded, input_rows = {}, []
    output.mkdir(parents=True)
    for case in cases:
        pixels = rgb8(packet_bound(case['input_path'], case['input_sha256']))
        clean = strict_array(packet_bound(case['clean512_path'], case['clean512_sha256']), (512, 512, 3), np.float32)
        require(clean.min() >= 0 and clean.max() <= 1, 'Clean target outside[0,1]')
        record = read_json(packet_bound(case['pair_record_path'], case['pair_record_sha256']))
        require(record['role'] == 'train' and record['reference_id'] == case['reference_id'] and
                record['update'] == case['update'] and record['slot'] == case['slot'] and
                record['input256_rgb8_sha256'] == array_sha(pixels) and
                record['input256_float32_sha256'] == array_sha(pixels.astype(np.float32) / np.float32(255)) and
                record['clean512_float32_sha256'] == array_sha(clean), 'Exact saved TRAIN/target tensor binding differs')
        target = target256(clean)
        target_path = output / 'targets' / (case['reference_id'] + '.png')
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(delivered_png(target)).save(target_path, format='PNG')
        else:
            require(np.array_equal(rgb8(target_path), delivered_png(target)), 'Repeated clean target differs')
        loaded[case['id']] = {'target': target, 'record': record, 'input': packet / case['input_path'],
                               'target_png': target_path}
        if case['behavior']:
            behavior = case['behavior']
            anchor = strict_array(packet_bound(behavior['delivered_path'], behavior['delivered_sha256']),
                                  (1, 3, 256, 256), np.float32)
            require(anchor.min() >= 0 and anchor.max() <= 1, 'Saved delivered behavior anchor outside[0,1]')
            strict_array(packet_bound(behavior['logits_path'], behavior['logits_sha256']),
                         (1, 256, 1024), np.float32)
            loaded[case['id']]['behavior_anchor'] = np.ascontiguousarray(anchor[0].transpose(1, 2, 0))
        input_rows.append({'case_id': case['id'], 'reference_id': case['reference_id'], 'role': 'train',
                           'input_sha256': case['input_sha256'], 'clean512_sha256': case['clean512_sha256'],
                           'input_tensor_exact': True, 'clean512_tensor_exact': True,
                           'profile': record['profile'], 'long_edge': record['parameters']['long_edge']})
    expected = expected_conditions(cases)
    prefix = protocol['outputs'] + '/'
    wanted = {prefix + row[-1] + '/summary.json' for row in expected}
    actual = {path.relative_to(root).as_posix() for section in ('paths', 'repeats', 'interventions')
              for path in (root / protocol['outputs'] / section).rglob('summary.json')}
    require(actual == wanted, 'Saved condition namespace differs from all104 planned conditions')
    rows, cache, visual = [], {}, []
    for kind, state, case, mode, arm, condition in expected:
        directory = prefix + condition + '/'
        summary_path = checked(directory + 'summary.json')
        saved = read_json(summary_path)
        require(saved['case_id'] == case['id'] and saved['state'] == state and saved['mode'] == mode and
                saved['intervention'] == arm and saved['input_sha256'] == case['input_sha256'] and
                saved['clean512_sha256'] == case['clean512_sha256'], 'Condition provenance differs')
        for name, relative in [('raw', 'raw.npy'), ('native_png', 'native.png'), ('tokens', 'actual_tokens.npy')]:
            require(saved[name]['path'] == directory + relative and
                    saved[name]['sha256'] == file_hashes[directory + relative], 'Summary artifact binding differs')
        raw_path, png_path = checked(directory + 'raw.npy'), checked(directory + 'native.png')
        require(saved['raw']['bytes'] == raw_path.stat().st_size and
                saved['tokens']['bytes'] == bounded(root, directory + 'actual_tokens.npy').stat().st_size,
                'Saved array byte count differs')
        raw = strict_array(raw_path, (256, 256, 3), np.float32)
        require(np.array_equal(rgb8(png_path), delivered_png(raw)), 'Native PNG is not exact floor RAW conversion')
        tokens = strict_array(checked(directory + 'actual_tokens.npy'), (256,), np.int64)
        arrays = {name: strict_array(checked(directory + name + '.npy'), shape, dtype)
                  for name, shape, dtype in [('native_probability_top2', (256, 2), np.float32),
                    ('native_probability_top2_indices', (256, 2), np.int64),
                    ('FP32_logits_top2', (256, 2), np.float32),
                    ('encoder_mean', (1, 256, 1, 1), np.float32),
                    ('encoder_std', (1, 256, 1, 1), np.float32)]}
        require(np.all(arrays['encoder_std'] > 0), 'Encoder standard deviation must be positive')
        token = token_statistics(tokens, arrays['native_probability_top2'],
                                 arrays['native_probability_top2_indices'], arrays['FP32_logits_top2'])
        for name in ('native_probability_top2_margin_minimum', 'native_probability_top2_zero_margin_tokens',
                     'selected_index_not_top2_first_due_to_ties'):
            scalar_equal(token[name], saved[name], name)
        execution = {'candidate_training': mode in ('T32', 'TBF16'), 'quantizer_training': False,
                     'generator_training': False, 'grad_enabled': mode != 'E32', 'inference_mode': mode == 'E32'}
        require(saved['execution'] == execution, 'Saved execution mode receipt differs')
        require(saved['logits_dtype'] == ('torch.bfloat16' if mode == 'TBF16' else 'torch.float32') and
                saved['native_probability_dtype'] == 'torch.float32', 'Saved logits/probability dtype differs')
        computed = metrics(raw, loaded[case['id']]['target'])
        for name, value in computed.items():
            scalar_equal(value, saved[name], name)
        preclip = saved['preclip_RGB']
        require(all(type(preclip[name]) in (int, float) and math.isfinite(preclip[name])
                    for name in ('minimum', 'maximum', 'mean', 'outside_range_fraction')) and
                preclip['minimum'] <= preclip['mean'] <= preclip['maximum'] and
                0 <= preclip['outside_range_fraction'] <= 1, 'Invalid reported preclip summary')
        row = {'condition': condition, 'condition_kind': kind, 'state': state, 'case_id': case['id'],
               'mode': mode, 'arm': arm, 'profile': loaded[case['id']]['record']['profile'],
               'metrics_raw': computed, 'metrics_native_png': metrics(rgb8(png_path).astype(np.float64) / 255,
                                                                        loaded[case['id']]['target']),
               'token_statistics': token, 'preclip_reported_only': preclip,
               'preclip_not_recomputed_without_native512_array': True,
               'summary_sha256': sha256(summary_path)}
        rows.append(row)
        cache[condition] = (raw, tokens, arrays, row)
        target_png = loaded[case['id']]['target_png']
        visual.append({**{key: row[key] for key in ('condition', 'condition_kind', 'state', 'case_id', 'mode', 'arm', 'profile')},
                       'input': str(loaded[case['id']]['input']), 'input_sha256': case['input_sha256'],
                       'target': str(target_png), 'target_sha256': sha256(target_png),
                       'raw': str(raw_path), 'raw_sha256': sha256(raw_path),
                       'native_png': str(png_path), 'native_png_sha256': sha256(png_path)})
    paths = []
    for state in STATES:
        for case in cases:
            base = cache['paths/%s/%s/E32' % (state, case['id'])]
            for mode in MODES[1:]:
                other = cache['paths/%s/%s/%s' % (state, case['id'], mode)]
                paths.append({'state': state, 'case_id': case['id'], 'baseline': 'E32', 'mode': mode,
                              **differences(other[0], base[0], other[1], base[1])})
    saved_paths = read_json(checked(prefix + 'path_comparisons.json'))['comparisons']
    require(len(saved_paths) == len(paths), 'Path comparison count differs')
    for actual_row, saved_row in zip(paths, saved_paths):
        for name in ('state', 'case_id', 'baseline', 'mode', 'token_disagreements'):
            require(actual_row[name] == saved_row[name], 'Path comparison provenance/token mismatch')
        for name in ('RAW_maximum_absolute', 'RAW_mean_absolute'):
            scalar_equal(actual_row[name], saved_row[name], name)
    repeats = []
    for kind, state, case, mode, arm, condition in expected:
        if kind == 'repeat':
            repeated, base = cache[condition], cache['paths/%s/%s/%s' % (state, case['id'], mode)]
            diff = differences(repeated[0], base[0], repeated[1], base[1])
            repeats.append({'state': state, 'case_id': case['id'], 'mode': mode,
                            'tokens_equal': diff['token_disagreements'] == 0, **diff})
    saved_repeats = read_json(checked(prefix + 'repeat_stability.json'))['comparisons']
    require(len(saved_repeats) == len(repeats), 'Repeat count differs')
    for actual_row, saved_row in zip(repeats, saved_repeats):
        for name in ('state', 'case_id', 'mode', 'tokens_equal'):
            require(actual_row[name] == saved_row[name], 'Repeat provenance/token mismatch')
        scalar_equal(actual_row['RAW_maximum_absolute'], saved_row['RAW_maximum_absolute'], 'Repeat RAW')
    interventions = []
    for case in cases:
        baseline = cache['paths/stopped50/%s/E32' % case['id']]
        arms, arm_rows = {'C': baseline[3]['metrics_raw']}, {}
        for key, arm in zip(('M', 'F', 'MF'), ARMS):
            item = cache['interventions/%s/%s' % (case['id'], arm)]
            require(np.array_equal(item[1], baseline[1]), 'Conditional intervention changed captured actual codes')
            require(np.array_equal(item[2]['encoder_mean'], baseline[2]['encoder_mean']) and
                    np.array_equal(item[2]['encoder_std'], baseline[2]['encoder_std']),
                    'Conditional intervention changed captured stopped encoder moments')
            arms[key] = item[3]['metrics_raw']
            arm_rows[key] = {'arm': arm, 'actual_codes_equal_to_stopped_E32': True,
                             **differences(item[0], baseline[0], item[1], baseline[1])}
        interventions.append({'case_id': case['id'], 'arms': arms, 'differences': arm_rows,
                              'conditional_metric_contrasts': factorial_contrasts(arms),
                              'training_causal_or_quality_claim': False})
    behavior_bridges = []
    for case in cases:
        if not case['behavior']:
            continue
        anchor = loaded[case['id']]['behavior_anchor']
        by_mode = {}
        for state in STATES:
            for mode in MODES:
                value = cache['paths/%s/%s/%s' % (state, case['id'], mode)][0]
                error = np.abs(value.astype(np.float64) - anchor.astype(np.float64))
                by_mode[state + '/' + mode] = {'maximum_absolute': float(error.max()),
                                               'mean_absolute': float(error.mean())}
        saved = read_json(checked(prefix + 'teacher_anchor_comparisons/' + case['id'] + '.json'))
        require(saved['case_id'] == case['id'], 'Teacher anchor comparison case differs')
        for name in ('maximum_absolute', 'mean_absolute'):
            scalar_equal(by_mode['original/E32'][name], saved['differences']['delivered'][name],
                         'Original E32 vs exact saved behavior image ' + name)
        behavior_bridges.append({'case_id': case['id'], 'delivered_image_gaps_recomputed': by_mode,
                                  'new_original_full_logits_gap_reported_only': saved['differences']['logits'],
                                  'full_new_logits_not_saved_cannot_recompute_logit_gap': True,
                                  'exact_recorded_training_anchor_was_not_replaced': True})
    require(len(behavior_bridges) == 2, 'Two exact original behavior anchor bridges required')
    summary = {'by_condition': _aggregate(rows, lambda row: row['condition_kind'] + '/' + row['state'] + '/' +
                                                              (row['arm'] or row['mode'])),
               'by_case': _aggregate(rows, lambda row: row['case_id']),
               'paths': paths, 'repeats': repeats, 'interventions': interventions,
               'exact_saved_behavior_anchor_bridges': behavior_bridges,
               'repeat_RAW_exact_all': all(row['RAW_maximum_absolute'] == 0 for row in repeats),
               'repeat_tokens_exact_all': all(row['tokens_equal'] for row in repeats)}
    write_new_json(output / 'input_bindings.json', {'cases': input_rows, 'clean_references': 4,
                                                  'paired_synthetic_TRAIN_only': True})
    write_new_json(output / 'condition_metrics.json', {'conditions': rows})
    write_new_json(output / 'comparisons.json', summary)
    write_new_json(output / 'visual_manifest.json', {'format': 'codeformer-path-diagnostic-visual-v1',
        'conditions': visual, 'all_planned_conditions': 104, 'neural_calls_or_training_performed': 0,
        'display_policy': 'saved native floor PNG; no enhancement; audit target geometry declared',
        'root_visual_review_required': True, 'quality_or_identity_claim': False})
    result = {'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'conditions_verified': 104,
        'RAW_exact_native_PNG_conversions': 104, 'captured_actual_token_arrays': 104,
        'all_expected_TRAIN_inputs': 8, 'exact_clean512_targets': 4,
        'exact_saved_behavior_anchor_pairs': 2,
        'path_comparisons': len(paths), 'repeat_comparisons': len(repeats), 'conditional_2x2_cases': 8,
        'hashed_reads_unique': len(set(checks)), 'replayed_scalar_tolerance': {'absolute': ABS_TOL, 'relative': REL_TOL},
        'preclip_native512_not_saved_not_independently_recomputed': True,
        'selected_codes_are_captured_native_lookup_not_FP32_argmax_reconstruction': True,
        'neural_or_gradient_or_optimizer_calls': 0, 'model_qualified': False,
        'seconds': time.monotonic() - started, 'source_sha256': sha256(__file__)}
    write_new_json(output / 'pixel_audit.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--return-root', required=True)
    parser.add_argument('--packet-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(audit(args.return_root, args.packet_root, args.output_dir), indent=2))

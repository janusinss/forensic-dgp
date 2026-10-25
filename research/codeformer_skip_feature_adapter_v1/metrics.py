"""Saved-pixel metrics and unchanged preservation gates; no neural imports.

Photographic targets and PNG conversion retain the historical float32 RGB8/255
convention. RAW and PNG are distinct views. Native CCTV receives no clean scores.
"""
import math


def require(value, message):
    if not value:
        raise ValueError(message)


def metrics(raw, png, target, support, evidence):
    import numpy as np
    require(raw.shape == png.shape == (256, 256, 3) and raw.dtype == np.float32 and
            png.dtype == np.uint8 and support.shape == (256, 256) and support.dtype == np.bool_ and
            support.any() and np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1,
            'Finite delivered RGB256/support required')
    names = ('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'signed_luminance_bias',
             'coarse_rgb_mae', 'adjacent_rgb_error')
    if evidence == 'native_unpaired':
        require(target is None, 'Native CCTV may not acquire a clean target')
        return {view: dict.fromkeys(names) for view in ('raw', 'native_png')}
    require(evidence in ('paired_synthetic', 'paired_synthetic_filter_stress', 'paired_synthetic_TRAIN') and
            target is not None and target.shape == raw.shape and target.dtype == np.float32 and
            np.isfinite(target).all() and 0 <= target.min() <= target.max() <= 1,
            'Exact paired photographic target required')
    truth = target.astype(np.float64)
    answer = {}
    for view, value in (('raw', raw), ('native_png', png.astype(np.float32) / np.float32(255))):
        predicted = value.astype(np.float64)
        residual = predicted - truth
        differences = residual[support]
        mse = float(np.square(differences).mean())
        signed = float((differences @ np.asarray([.2126, .7152, .0722])).mean())
        row = {'mae': float(np.abs(differences).mean()), 'mse': mse,
               'psnr': None if mse == 0 else -10 * math.log10(mse), 'ssim': None,
               'luminance_bias': abs(signed), 'signed_luminance_bias': signed,
               'coarse_rgb_mae': None, 'adjacent_rgb_error': None}
        if support.all():
            from skimage.metrics import structural_similarity
            pool = lambda a: a.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3), dtype=np.float64)
            row['coarse_rgb_mae'] = float(np.abs(pool(predicted) - pool(truth)).mean())
            row['ssim'] = float(structural_similarity(predicted, truth, data_range=1.0, channel_axis=2))
            row['adjacent_rgb_error'] = float(.5 * (
                np.abs(residual[:, 1:] - residual[:, :-1]).mean() +
                np.abs(residual[1:] - residual[:-1]).mean()))
        require(all(v is None or math.isfinite(v) for v in row.values()), 'Nonfinite paired metric')
        answer[view] = row
    return answer


def preservation(bank, original):
    """Historical61 clear and53 normally blurred, same0.01/0.03 limits."""
    require(set(bank) == set(original) and len(bank) == 406, 'All406 historical DEV slots required')
    clear = sorted(k for k, r in bank.items() if r['evidence'] == 'paired_synthetic' and r['profile'] == 'clear')
    blur = sorted(k for k, r in bank.items() if r['source'] == 'FFHQ validation' and r['profile'] == 'blur')
    require(len(clear) == 61 and len(blur) == 53, 'Unchanged61clear/53blur cohort required')
    report = {'pass': True, 'quality_gate_thresholds_changed': False, 'groups': {}}
    for label, ids, fields in (('clear', clear, ('mae',)),
                              ('normal_blur', blur, ('luminance_bias', 'coarse_rgb_mae'))):
        group = {'cases': ids, 'views': {}, 'pass': True}
        for view in ('raw', 'native_png'):
            group['views'][view] = {}
            for field in fields:
                delta = {k: bank[k]['metrics'][view][field] - original[k]['metrics'][view][field] for k in ids}
                require(all(math.isfinite(v) for v in delta.values()), 'Finite original-relative gate required')
                mean = sum(delta.values()) / len(delta)
                worst = max(delta.values())
                passed = mean <= .01 and worst <= .03
                group['views'][view][field] = {'deltas': delta, 'mean_delta': mean, 'worst_delta': worst,
                    'mean_limit': .01, 'worst_limit': .03, 'cases_over_0_03': sum(v > .03 for v in delta.values()),
                    'pass': passed}
                group['pass'] = group['pass'] and passed
        report['groups'][label] = group
        report['pass'] = report['pass'] and group['pass']
    return report


def dark_retention(bank, control):
    require(set(bank) == set(control) and len(bank) == 406, 'All406 dark-control banks required')
    report = {'pass': True, 'profiles': {}}
    for profile in ('lowlight', 'compound'):
        ids = sorted(k for k, r in bank.items() if r['source'] == 'FFHQ validation' and r['profile'] == profile)
        require(len(ids) == 53, 'Unchanged53-case lowlight/compound cohort required')
        means = {name: {field: sum(rows[k]['metrics']['native_png'][field] for k in ids) / 53
                         for field in ('mae', 'ssim')}
                 for name, rows in (('candidate', bank), ('control', control))}
        require(all(math.isfinite(v) for r in means.values() for v in r.values()), 'Finite dark means required')
        passed = means['candidate']['mae'] <= means['control']['mae'] and means['candidate']['ssim'] >= means['control']['ssim']
        report['profiles'][profile] = {**means, 'cases': ids, 'pass': passed}
        report['pass'] = report['pass'] and passed
    return report


def interim(bank, original, initial):
    """Additional no-new-regression stop, never a qualification replacement.

    Python float64 arithmetic on saved FP32 metrics. Exact <=, no new epsilon.
    Each case and each group mean is compared independently against max(limit,r0).
    """
    current, start = preservation(bank, original), preservation(initial, original)
    result = {'pass': True, 'qualification': current, 'additional_stop_only': True,
              'arithmetic': 'Python float64; exact <=; no added tolerance', 'groups': {}}
    for label, group in current['groups'].items():
        result['groups'][label] = {}
        for view, fields in group['views'].items():
            result['groups'][label][view] = {}
            for field, row in fields.items():
                old = start['groups'][label]['views'][view][field]
                limits = {k: max(.03, old['deltas'][k]) for k in row['deltas']}
                bad = [k for k in row['deltas'] if row['deltas'][k] > limits[k]]
                mean_limit = max(.01, old['mean_delta'])
                passed = not bad and row['mean_delta'] <= mean_limit
                result['groups'][label][view][field] = {'pass': passed, 'case_limits': limits,
                    'failed_case_ids': bad, 'mean_limit': mean_limit, 'mean_delta': row['mean_delta']}
                result['pass'] = result['pass'] and passed
    return result


def filter_summary(rows):
    """Retain each profile/role/source; no post-hoc best-model/default selector."""
    from collections import Counter
    filtered = [r for r in rows if r.get('cohort') in
                ('filter_probe', 'retained_filter72_mixed_roles')]
    require(len(filtered) == 72 and Counter(r['split'] for r in filtered) == {'TRAIN': 36, 'DEV': 36},
            'All72 mixed-role filter probes required')
    answer = {'comparisons': 72, 'role_counts': {'TRAIN': 36, 'DEV': 36}, 'groups': {},
              'acceptance_established': False, 'independent_identity_count': None}
    for split in ('TRAIN', 'DEV'):
        for profile in sorted({r['profile'] for r in filtered}):
            cases = [r for r in filtered if r['split'] == split and r['profile'] == profile]
            if not cases:
                continue
            values = {}
            for view in ('raw', 'native_png'):
                values[view] = {field: sum(r['metrics'][view][field] for r in cases) / len(cases)
                    for field in ('mae', 'ssim', 'adjacent_rgb_error')}
            answer['groups'][split + '/' + profile] = {'cases': [r['id'] for r in cases], 'metrics': values}
    return answer


def guard_interim(current, initial):
    """Additional repeated-TRAIN stops; never substitute for full DEV gates.

    Eight clear and eight brightness-only anchors compare delivered behavior
    against frozen teachers. Eight filter probes compare paired MAE and adjacent
    residual error with snapshot zero. RAW and PNG remain separate. No target is
    invented for a brightness anchor and no independent holdout claim is made.
    """
    from collections import Counter
    require(type(current) is dict and type(initial) is dict and
            set(current) == set(initial) and len(current) == 24,
            'Exact same24 TRAIN guard IDs required')
    require(Counter(r['role'] for r in current.values()) == {'clear': 8, 'dark': 8, 'filter': 8},
            'Eight clear/dark/filter guards required')
    result = {'pass': True, 'additional_TRAIN_stop_only': True,
              'model_qualified': False, 'groups': {},
              'arithmetic': 'Python float64; exact <=; no added tolerance'}
    for key, row in current.items():
        require(row['role'] == initial[key]['role'], 'Guard role changed')
        require(row.get('split') == initial[key].get('split') == 'TRAIN',
                'Guard probes must remain TRAIN')
    for role in ('clear', 'dark', 'filter'):
        ids = sorted(k for k, r in current.items() if r['role'] == role)
        group = {'pass': True, 'cases': ids, 'views': {}}
        for view in ('raw', 'native_png'):
            group['views'][view] = {}
            fields = ('mae', 'adjacent_rgb_error') if role == 'filter' else ('behavior_distance',)
            for field in fields:
                def value(rows, identifier):
                    entry = rows[identifier]
                    v = entry['behavior_distance'][view] if field == 'behavior_distance' else entry['metrics'][view][field]
                    require(type(v) in (int, float) and math.isfinite(v) and v >= 0,
                            'Finite nonnegative TRAIN guard measurement required')
                    return float(v)
                start = {k: value(initial, k) for k in ids}
                now = {k: value(current, k) for k in ids}
                if role == 'filter':
                    deltas = {k: now[k] - start[k] for k in ids}
                    limits = dict.fromkeys(ids, .03)
                    mean = sum(deltas.values()) / len(ids)
                    mean_limit = .01
                else:
                    deltas = now
                    limits = {k: max(.03, start[k]) for k in ids}
                    mean = sum(now.values()) / len(ids)
                    mean_limit = max(.01, sum(start.values()) / len(ids))
                bad = [k for k in ids if deltas[k] > limits[k]]
                passed = not bad and mean <= mean_limit
                group['views'][view][field] = {'pass': passed,
                    'current': now, 'initial': start, 'case_limits': limits,
                    'failed_case_ids': bad, 'mean_measurement': mean,
                    'mean_limit': mean_limit,
                    'measurement': 'increase from initial' if role == 'filter' else 'distance from frozen teacher'}
                group['pass'] = group['pass'] and passed
        result['groups'][role] = group
        result['pass'] = result['pass'] and group['pass']
    return result

"""Inert saved-pixel metrics and unchanged inherited TRAIN preservation gates.

NumPy is supplied by the caller. No model, framework, image decoder, filesystem,
optimizer or worker is imported or executed. Proposal feasibility is additional;
the full pilot's four one-percent endpoints remain separately reported.
"""
import math


VIEWS = ('raw', 'native_png')
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
REGIONS = ('whole', 'central128', 'outside')
FIELDS = ('mae', 'adjacent_rgb_error')
LUMINANCE = (.2126, .7152, .0722)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def ordered_float_sum(values):
    """Explicit sequential Python float arithmetic, including on CPython3.13."""
    total = 0.0
    for value in values:
        total = total + float(value)
    return total


def _array(value, *, np):
    need(isinstance(value, np.ndarray) and value.shape == (256, 256, 3) and
         value.dtype.kind == 'f' and np.isfinite(value).all() and
         value.min() >= 0 and value.max() <= 1,
         'Finite floating HWC RGB256 in [0,1] required')
    return value


def _png(value, *, np):
    need(isinstance(value, np.ndarray) and value.shape == (256, 256, 3) and
         value.dtype == np.uint8, 'Exact RGB8 HWC256 PNG pixels required')
    return value.astype(np.float32) / np.float32(255)


def pixel_metrics(raw, target_or_teacher, *, np):
    """Exact inherited whole-region MAE and mean-of-two-means edge arithmetic."""
    _array(raw, np=np)
    if target_or_teacher is None:
        return dict.fromkeys(FIELDS, None)
    _array(target_or_teacher, np=np)
    residual = raw.astype(np.float64) - target_or_teacher.astype(np.float64)
    return {'mae': float(np.abs(residual).mean()),
            'adjacent_rgb_error': float(.5 * (
                np.abs(residual[:, 1:] - residual[:, :-1]).mean() +
                np.abs(residual[1:] - residual[:-1]).mean()))}


def regional_metrics(raw, target, *, np):
    """Exact parent whole/half-open-central/outside metrics, excluding seam edges."""
    _array(raw, np=np)
    if target is None:
        return None
    _array(target, np=np)
    residual = raw.astype(np.float64) - target.astype(np.float64)
    whole = np.ones((256, 256), dtype=np.bool_)
    inside = np.zeros_like(whole)
    inside[64:192, 64:192] = True
    result = {'whole': pixel_metrics(raw, target, np=np)}
    for name, region in (('central128', inside), ('outside', ~inside)):
        horizontal = region[:, :-1] & region[:, 1:]
        vertical = region[:-1] & region[1:]
        # Preserve parent's sum-then-divide arithmetic for non-whole regions.
        adjacent = float((np.abs(np.diff(residual, axis=1))[horizontal].sum() +
                          np.abs(np.diff(residual, axis=0))[vertical].sum()) /
                         (int(horizontal.sum() + vertical.sum()) * 3))
        result[name] = {'mae': float(np.abs(residual[region]).mean()),
                        'adjacent_rgb_error': adjacent}
    return result


def behavior_distance(raw, teacher, *, np):
    _array(raw, np=np)
    _array(teacher, np=np)
    return float(np.abs(raw.astype(np.float64) - teacher.astype(np.float64)).mean())


def illumination_distance(raw, teacher, *, np):
    _array(raw, np=np)
    _array(teacher, np=np)
    weights = np.asarray(LUMINANCE, dtype=np.float64)
    observed = (raw.astype(np.float64) @ weights).reshape(16, 16, 16, 16).mean(axis=(1, 3))
    reference = (teacher.astype(np.float64) @ weights).reshape(16, 16, 16, 16).mean(axis=(1, 3))
    return float(np.abs(observed - reference).mean())


def record_metrics(raw, png, target, row, *, np, teacher_raw=None,
                   teacher_png=None, baseline_raw=None, baseline_png=None):
    """Metric fields for a worker-owned record; no files or generated teachers.

    PNG arguments are actual RGB8 decoded pixels. Raw and target are floating
    RGB256 arrays. Omitted baseline pair means this is an initializer self-pair;
    both U96 movement distances are then exactly zero, never teacher distances.
    """
    _array(raw, np=np)
    quantized = _png(png, np=np)
    need(type(row) is dict and row.get('role') in (*ROLES, 'filter', 'dark'),
         'Declared paired or legacy role required')
    need(target is not None or row['role'] == 'dark', 'Only inherited dark guards lack clean targets')
    if target is not None:
        _array(target, np=np)
    need((baseline_raw is None) == (baseline_png is None), 'Both initial RAW/PNG arrays or neither')
    need((teacher_raw is None) == (teacher_png is None), 'Both teacher RAW/PNG arrays or neither')
    need((row.get('behavior_teacher') is not None) == (teacher_raw is not None),
         'Pinned behavior teacher pair required exactly when declared by row')
    result = {'metrics': {}, 'regional_metrics': {}, 'U96_behavior_distance': {},
              'U96_illumination_distance': {}}
    references = None if baseline_raw is None else {
        'raw': _array(baseline_raw, np=np), 'native_png': _png(baseline_png, np=np)}
    teachers = None if teacher_raw is None else {
        'raw': _array(teacher_raw, np=np), 'native_png': _png(teacher_png, np=np)}
    if teachers is not None:
        result['behavior_distance'] = {}
    for view, value in (('raw', raw), ('native_png', quantized)):
        result['metrics'][view] = pixel_metrics(value, target, np=np)
        result['regional_metrics'][view] = regional_metrics(value, target, np=np)
        result['U96_behavior_distance'][view] = (
            0.0 if references is None else behavior_distance(value, references[view], np=np))
        result['U96_illumination_distance'][view] = (
            0.0 if references is None else illumination_distance(value, references[view], np=np))
        if teachers is not None:
            result['behavior_distance'][view] = behavior_distance(value, teachers[view], np=np)
    return result


def _number(value):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
         'Finite nonnegative scalar metric required')
    return float(value)


def _bank(value):
    if type(value) is list:
        need(all(type(row) is dict and type(row.get('id')) is str for row in value),
             'Complete identified records required')
        converted = {row['id']: row for row in value}
        need(len(converted) == len(value), 'Duplicate record ID refused')
        value = converted
    need(type(value) is dict and all(type(key) is str and type(row) is dict and
         row.get('id') == key for key, row in value.items()), 'ID-bound record bank required')
    return value


def _quality_selection(protocol):
    need(type(protocol) is dict and type(protocol.get('cases')) is list and
         type(protocol.get('guards')) is list, 'Inherited cases/guards protocol required')
    policy = protocol['quality_policy']
    for key, expected in {
        'RAW_and_PNG_separate': True, 'monitor_metric_delta_mean_maximum': .01,
        'monitor_metric_delta_individual_maximum': .03,
        'clear_behavior_and_dark_illumination_mean_maximum': .01,
        'clear_behavior_and_dark_illumination_individual_maximum': .03,
        'terminal64_adjacent_gain_fraction_minimum': .01,
        'terminal64_dark_MAE_and_adjacent_nonregression': True,
        'small_TRAIN_pass_qualifies_DEV_or_FINAL': False,
    }.items():
        need(key in policy and type(policy[key]) is type(expected) and policy[key] == expected,
             'Unchanged inherited quality policy required: ' + key)
    monitors = [row for row in protocol['cases'] if row.get('cohort') == 'monitor']
    guards = protocol['guards']
    need(len(monitors) == 32 and len(guards) == 24, 'All32 monitors and24 guards required')
    expected = {row['id']: row for row in monitors + guards}
    need(len(expected) == 56 and all(row['split'] == 'TRAIN' for row in expected.values()),
         'Exact unique56 TRAIN quality rows required')
    need(all(sum(row['role'] == role for row in monitors) == 8 for role in ROLES) and
         all(sum(row['role'] == role for row in guards) == 8 for role in ('clear', 'dark', 'filter')),
         'All eight members per monitor/legacy role required')
    return monitors, guards, expected


def _validate_bank(bank, expected):
    need(set(bank) == set(expected), 'Complete exact56 quality IDs required; no extras or omissions')
    for key, reference in expected.items():
        row = bank[key]
        need(all(row.get(field) == reference[field] for field in ('id', 'reference_id', 'role', 'split')),
             'Record metadata differs from declared TRAIN row')
        for view in VIEWS:
            if reference.get('target') is None:
                need(reference['role'] == 'dark' and row['regional_metrics'][view] is None and
                     all(row['metrics'][view][field] is None for field in FIELDS),
                     'Unpaired legacy dark target/metric nulls must remain null')
            else:
                for region in REGIONS:
                    for field in FIELDS:
                        _number(row['regional_metrics'][view][region][field])
                for field in FIELDS:
                    need(_number(row['metrics'][view][field]) ==
                         row['regional_metrics'][view]['whole'][field], 'Whole metric aliases differ')
            if reference.get('behavior_teacher') is not None:
                _number(row['behavior_distance'][view])
            if reference.get('cohort') == 'monitor':
                _number(row['U96_behavior_distance'][view])
                _number(row['U96_illumination_distance'][view])


def evaluate_bank(bank, baseline, protocol, *, np=None, state='proposal'):
    """Evaluate EXACT56 quality records; NumPy is unused by scalar gate logic.

    Initial state checks inherited legacy and ordinary monitor guards only.
    Proposal states additionally require strict global/local adjacent gain and
    dark MAE/adjacent nonregression. The full1% endpoints are reporting only here.
    """
    need(state in ('initializer', 'proposal', 'proposal_A', 'proposal_B'), 'Declared diagnostic state required')
    monitors, guards, expected = _quality_selection(protocol)
    bank = _bank(bank)
    _validate_bank(bank, expected)
    baseline = bank if baseline is None and state == 'initializer' else _bank(baseline)
    _validate_bank(baseline, expected)
    report = {'pass': True, 'state': state, 'TRAIN_only_not_qualification': True,
              'legacy': {'pass': True, 'thresholds_reset_to_U96': False, 'groups': {}},
              'monitor': {'pass': True, 'TRAIN_monitor_only_not_DEV': True, 'groups': {}},
              'strict_feasibility': {'pass': True, 'evaluated': state != 'initializer', 'groups': {}},
              'full_pilot_endpoints': {'pass': True, 'binding_for_this_diagnostic': False,
                                       'required_gain_fraction': .01, 'groups': {}}}
    inherited = {row['id']: row for row in guards}
    for role in ('clear', 'dark', 'filter'):
        ids = sorted(key for key, row in inherited.items() if row['role'] == role)
        group = {'pass': True, 'views': {}}
        for view in VIEWS:
            checks = {}
            for field in (FIELDS if role == 'filter' else ('behavior_distance',)):
                metric = lambda rows, key: _number(rows[key]['behavior_distance'][view]
                    if field == 'behavior_distance' else rows[key]['metrics'][view][field])
                start = {key: metric(inherited, key) for key in ids}
                now = {key: metric(bank, key) for key in ids}
                if role == 'filter':
                    values = {key: now[key] - start[key] for key in ids}
                    case_limits = dict.fromkeys(ids, .03)
                    mean_limit = .01
                else:
                    values = now
                    case_limits = {key: max(.03, start[key]) for key in ids}
                    mean_limit = max(.01, ordered_float_sum(start.values()) / 8)
                mean = ordered_float_sum(values.values()) / 8
                bad = [key for key in ids if values[key] > case_limits[key]]
                passed = not bad and mean <= mean_limit
                checks[field] = {'pass': passed, 'initial': start, 'current': now,
                    'measurements': values, 'case_limits': case_limits, 'mean_limit': mean_limit,
                    'mean': mean, 'failed_ids': bad}
                group['pass'] = group['pass'] and passed
            group['views'][view] = checks
        report['legacy']['groups'][role] = group
        report['legacy']['pass'] = report['legacy']['pass'] and group['pass']
    for role in ROLES:
        ids = sorted(row['id'] for row in monitors if row['role'] == role)
        group = {'pass': True, 'views': {}}
        strict = {'pass': True, 'views': {}}
        endpoint = {'pass': True, 'views': {}}
        for view in VIEWS:
            checks = {}
            for region in (REGIONS if role == ROLES[0] else ('whole',)):
                for field in FIELDS:
                    start = {key: baseline[key]['regional_metrics'][view][region][field] for key in ids}
                    now = {key: bank[key]['regional_metrics'][view][region][field] for key in ids}
                    delta = {key: now[key] - start[key] for key in ids}
                    mean = ordered_float_sum(delta.values()) / 8
                    checks[region + '/' + field] = {'pass': max(delta.values()) <= .03 and mean <= .01,
                        'initial': start, 'current': now, 'deltas': delta, 'mean': mean,
                        'mean_limit': .01, 'worst_limit': .03}
            if role in ('clear', 'lowlight_linear_0p35'):
                field = 'U96_behavior_distance' if role == 'clear' else 'U96_illumination_distance'
                distances = {key: bank[key][field][view] for key in ids}
                checks[field] = {'pass': ordered_float_sum(distances.values()) / 8 <= .01 and
                    max(distances.values()) <= .03, 'values': distances, 'mean_limit': .01, 'worst_limit': .03}
            group['views'][view] = checks
            group['pass'] = group['pass'] and all(check['pass'] for check in checks.values())
            strict_checks = {}
            endpoint_checks = {}
            if role in ROLES[:2]:
                region = 'central128' if role == ROLES[0] else 'whole'
                start = ordered_float_sum(baseline[key]['regional_metrics'][view][region]['adjacent_rgb_error'] for key in ids) / 8
                now = ordered_float_sum(bank[key]['regional_metrics'][view][region]['adjacent_rgb_error'] for key in ids) / 8
                strict_checks['blur_adjacent'] = {'pass': now < start, 'initial_mean': start,
                    'current_mean': now, 'region': region}
                gain = (start - now) / start if start > 0 else None
                endpoint_checks['one_percent_blur'] = {'pass': gain is not None and gain >= .01,
                    'gain_fraction': gain, 'initial_mean': start, 'current_mean': now, 'region': region}
            if role == 'lowlight_linear_0p35':
                for field in FIELDS:
                    start = ordered_float_sum(baseline[key]['regional_metrics'][view]['whole'][field] for key in ids) / 8
                    now = ordered_float_sum(bank[key]['regional_metrics'][view]['whole'][field] for key in ids) / 8
                    check = {'pass': now <= start, 'initial_mean': start, 'current_mean': now}
                    strict_checks['dark_' + field] = check
                    endpoint_checks['dark_' + field] = dict(check)
            strict['views'][view] = strict_checks
            strict['pass'] = strict['pass'] and all(check['pass'] for check in strict_checks.values())
            endpoint['views'][view] = endpoint_checks
            endpoint['pass'] = endpoint['pass'] and all(check['pass'] for check in endpoint_checks.values())
        report['monitor']['groups'][role] = group
        report['monitor']['pass'] = report['monitor']['pass'] and group['pass']
        report['strict_feasibility']['groups'][role] = strict
        report['full_pilot_endpoints']['groups'][role] = endpoint
        report['full_pilot_endpoints']['pass'] = report['full_pilot_endpoints']['pass'] and endpoint['pass']
        if state != 'initializer':
            report['strict_feasibility']['pass'] = report['strict_feasibility']['pass'] and strict['pass']
    report['pass'] = report['legacy']['pass'] and report['monitor']['pass'] and report['strict_feasibility']['pass']
    return report


def bank_gate(bank, baseline, protocol, selection, *, np=None, state=None):
    """Convenience wrapper for complete72 core banks, then evaluate exact56."""
    bank = _bank(bank)
    rows = selection['optimization'] + selection['monitors'] + selection['guards']
    need(len(rows) == 72 and set(bank) == {row['id'] for row in rows}, 'Complete exact72 core records required')
    if baseline is not None:
        baseline = _bank(baseline)
        need(set(baseline) == set(bank), 'Same complete72 initializer records required')
    quality = {row['id'] for row in selection['monitors'] + selection['guards']}
    state = ('initializer' if baseline is None else 'proposal') if state is None else state
    return evaluate_bank({key: bank[key] for key in quality},
        None if baseline is None else {key: baseline[key] for key in quality}, protocol, np=np, state=state)

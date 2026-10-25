"""Pure saved-metric selection and detached NumPy candidate arithmetic.

This prototype implements no model, autograd, parameter application, restoration
journal, worker, transfer package, optimizer, VM command or launch entrypoint.
Four complete optimization-only candidate banks are required before selection.
"""
import math
import numbers
import re

import numpy as np

ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'adjacent_rgb_error')
CANDIDATE_MAXABS = (2e-5, 1.5e-5, 1e-5, 5e-6)
NUMERICAL_HEADROOM = .99
OLD_MAXABS_STEP = 2.5e-5
MAX_CUMULATIVE = 1e-4
MAX_COORDINATES = 168960
PROVENANCE = ('id', 'reference_id', 'role', 'split', 'cohort',
              'optimization_permitted', 'metric_region', 'input', 'target')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _binding(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'Exact source/target byte binding required')
    name = value['path']
    need(type(name) is str and name and ':' not in name and '\\' not in name
         and not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/')),
         'Confined canonical input/target path required')
    need(type(value['bytes']) is int and value['bytes'] > 0 and type(value['sha256']) is str
         and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Positive file size and exact SHA256 required')


def validate_optimization_metadata(metadata):
    """Bind64 TRAIN rows; supplying monitor/guard/adaptive candidates is refused."""
    need(type(metadata) is list and len(metadata) == 64 and all(type(r) is dict for r in metadata),
         'Exact ordered64 optimization provenance rows required')
    need(len({r.get('id') for r in metadata}) == 64, 'Unique64 optimizing IDs required')
    references = []
    for index in range(0, 64, 4):
        rows = metadata[index:index + 4]
        reference = rows[0].get('reference_id')
        need(type(reference) is str and reference and reference not in references,
             'Sixteen distinct photograph-major references required')
        references.append(reference)
        need([r.get('role') for r in rows] == list(ROLES)
             and all(r.get('reference_id') == reference for r in rows), 'Matched four-role photograph-major order required')
        need(all(r.get('target') == rows[0].get('target') for r in rows), 'Same aligned clean target for four matched roles required')
        for row in rows:
            need(all(key in row for key in PROVENANCE)
                 and type(row['id']) is str and row['id']
                 and row['split'] == 'TRAIN' and row['cohort'] == 'optimize'
                 and row['optimization_permitted'] is True,
                 'Only explicitly admitted optimization TRAIN rows allowed; no monitor or guard')
            need(row['metric_region'] == ('central128' if row['role'] == ROLES[0] else 'whole'),
                 'Original central128/whole gate ROI binding required')
            _binding(row['input']); _binding(row['target'])
    return {row['id']: row for row in metadata}


def _bank_means(bank, metadata):
    need(type(bank) is dict and set(bank) == set(metadata), 'Complete exact64 optimization bank required; no monitor/guard extras')
    checked = {}
    for key, reference in metadata.items():
        row = bank[key]
        need(type(row) is dict and all(row.get(field) == reference[field] for field in PROVENANCE),
             'Per-case ID/role/ROI/source/target provenance differs: ' + key)
        metrics = row.get('metrics')
        need(type(metrics) is dict and set(metrics) == set(VIEWS), 'Both actual RAW and PNG metric views required')
        checked[key] = {}
        for view in VIEWS:
            need(type(metrics[view]) is dict and set(metrics[view]) == set(FIELDS), 'Exact MAE/adjacent metric fields required')
            checked[key][view] = {}
            for field in FIELDS:
                value = metrics[view][field]
                need(isinstance(value, numbers.Real) and not isinstance(value, bool)
                     and math.isfinite(float(value)) and value >= 0, 'Finite nonnegative actual metric required')
                checked[key][view][field] = float(value)
    return {view: {role: {field: math.fsum(checked[key][view][field]
        for key in sorted(metadata) if metadata[key]['role'] == role) / 16.
        for field in FIELDS} for role in ROLES} for view in VIEWS}


def select_candidate(original_bank, trials, optimization_metadata):
    """Evaluate all four fixed candidates; first eligible index or None.

    trials is a four-element list of {'maximum_absolute_coordinate_bound': cap,
    'bank': complete64bank}. Per-case metrics must already use the bound ROI and
    actual durable delivered RAW/PNG. This scalar-only helper authenticates no
    image/model bytes; a future worker and independent auditor must do so.
    """
    metadata = validate_optimization_metadata(optimization_metadata)
    need(type(trials) is list and len(trials) == 4, 'All four complete registered trial banks required before selection')
    need(all(type(t) is dict and set(t) == {'maximum_absolute_coordinate_bound', 'bank'} for t in trials),
         'Only fixed cap and optimization bank belong in selector trial input')
    need(tuple(t['maximum_absolute_coordinate_bound'] for t in trials) == CANDIDATE_MAXABS,
         'Exact immutable four-scale order required; no appended/reordered scale')
    baseline = _bank_means(original_bank, metadata)
    results = []
    # No early return: even a first eligible trial requires all four banks to be
    # complete and valid. An invariant/provenance failure is fatal, not ineligible.
    for index, trial in enumerate(trials):
        means = _bank_means(trial['bank'], metadata)
        checks, strict = [], []
        for view in VIEWS:
            for role in ROLES:
                for field in FIELDS:
                    now, start = means[view][role][field], baseline[view][role][field]
                    checks.append({'view': view, 'role': role, 'field': field,
                        'original': start, 'current': now, 'delta': now - start, 'pass': now <= start})
        for role in ROLES[:2]:
            now, start = means['native_png'][role]['adjacent_rgb_error'], baseline['native_png'][role]['adjacent_rgb_error']
            strict.append({'view': 'native_png', 'role': role, 'field': 'adjacent_rgb_error',
                'original': start, 'current': now, 'delta': now - start, 'pass': now < start})
        results.append({'candidate_index': index, 'maximum_absolute_coordinate_bound': CANDIDATE_MAXABS[index],
            'eligible': all(r['pass'] for r in checks + strict), 'all16_nonincrease_checks': checks,
            'both_blur_PNG_strict_checks': strict, 'means': means})
    chosen = next((r['candidate_index'] for r in results if r['eligible']), None)
    return {'complete': True, 'all_four_trials_evaluated': True, 'candidate_registry': list(CANDIDATE_MAXABS),
        'selected_index': chosen, 'selected_maximum_absolute_coordinate_bound': None if chosen is None else CANDIDATE_MAXABS[chosen],
        'no_eligible_candidate': chosen is None, 'trial_results': results, 'original_means': baseline,
        'quality_tolerance': 0, 'monitor_or_guard_input_accepted': False,
        'scalar_only_actual_saved_pixels_not_verified_here': True,
        'invariant_failure_is_fatal_not_ineligible': True, 'post_monitor_fallback_authorized': False,
        'worker_or_nine_application_journal_implemented': False,
        'parameter_applied': False, 'manual_packet_ready': False, 'model_qualified': False}


def propose_from_original_fp32(original, direction, pooled_raw_gradients, maximum_absolute_coordinate_bound):
    """Detached synthetic/saved-vector arithmetic; no actual model copies.

    Independently starts from original for each call. A future worker must bind
    exact36-head schema/state, actual copy/restore journal and RAW/PNG outcomes.
    Only one registered cap is accepted, with no adaptation or retry.
    """
    cap = maximum_absolute_coordinate_bound
    need(type(cap) in (float, int) and cap in CANDIDATE_MAXABS and cap <= OLD_MAXABS_STEP,
         'Only four registered bounds below original2.5e-5 per-step cap permitted')
    need(isinstance(original, np.ndarray) and original.ndim == 1 and original.dtype == np.float32
         and 0 < original.size <= MAX_COORDINATES, 'Finite original FP32 flat vector required')
    need(isinstance(direction, np.ndarray) and direction.shape == original.shape and direction.dtype == np.float64,
         'Same-coordinate FP64 direction required')
    need(isinstance(pooled_raw_gradients, np.ndarray) and pooled_raw_gradients.shape == (8, original.size)
         and pooled_raw_gradients.dtype == np.float64, 'All eight current-original RAW gradient vectors required')
    need(all(np.isfinite(v).all() for v in (original, direction, pooled_raw_gradients)), 'Finite arithmetic inputs required')
    before = [v.tobytes() for v in (original, direction, pooled_raw_gradients)]
    peak = float(np.max(np.abs(direction)))
    need(peak > 0, 'Nonzero direction required')
    alpha = NUMERICAL_HEADROOM * float(cap) / peak
    need(math.isfinite(alpha) and alpha > 0, 'Finite positive fixed scalar required')
    candidate = (original.astype(np.float64) + alpha * direction).astype(np.float32)
    need(np.isfinite(candidate).all(), 'Finite rounded FP32 candidate required')
    delta = candidate.astype(np.float64) - original.astype(np.float64)
    actual_peak = float(np.max(np.abs(delta)))
    need(actual_peak <= cap and actual_peak <= OLD_MAXABS_STEP and actual_peak <= MAX_CUMULATIVE,
         'Actual rounded candidate exceeds registered/old-step/cumulative bound; no retry')
    dots = [float(np.sum(row * delta, dtype=np.float64)) for row in pooled_raw_gradients]
    need(all(math.isfinite(value) and value < 0 for value in dots),
         'Actual rounded candidate lacks all eight strict RAW descent dots; no retry')
    need(before == [v.tobytes() for v in (original, direction, pooled_raw_gradients)], 'Input bytes changed')
    return {'candidate': candidate, 'actual_delta': delta, 'certificate': {
        'pass': True, 'registered_bound': float(cap), 'fixed_numerical_headroom': NUMERICAL_HEADROOM,
        'alpha': alpha, 'actual_maxabs_delta_from_original': actual_peak,
        'original_per_step_cap': OLD_MAXABS_STEP, 'original_cumulative_cap': MAX_CUMULATIVE,
        'all8_actual_RAW_dots': dots, 'independent_original_initialization': True,
        'inputs_unchanged': True, 'parameter_applied': False, 'metric_adaptive_scalar': False,
        'retry_or_fallback': False, 'finite_PNG_gain_certified': False,
        'restore_or_nine_application_accounting_proved': False}}

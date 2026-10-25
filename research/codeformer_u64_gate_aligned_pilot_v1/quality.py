"""Pure added role-mean checks; inherited quality logic remains untouched.

Inputs use the existing cohort_means schema. A caller must evaluate every
original bank/individual/teacher/terminal gate and pass its complete result.
This wrapper preserves that result and can only add reasons to stop.
"""
import copy
import math
import numbers

ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'adjacent_rgb_error')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _means(value):
    need(type(value) is dict and set(value) == set(VIEWS), 'Exact RAW/PNG mean views required')
    result = {}
    for view in VIEWS:
        need(type(value[view]) is dict and set(value[view]) == set(ROLES), 'All four photographic roles required')
        result[view] = {}
        for role in ROLES:
            row = value[view][role]
            need(type(row) is dict and set(row) == set(FIELDS), 'Exact photographic MAE/adjacent fields required')
            result[view][role] = {}
            for field in FIELDS:
                number = row[field]
                need(isinstance(number, numbers.Real) and not isinstance(number, bool)
                     and math.isfinite(float(number)) and number >= 0,
                     'Finite nonnegative real metric required')
                result[view][role][field] = float(number)
    return result


def evaluate_quality(current, original, prior, terminal, inherited_gate_result):
    """Add exact role-mean nonincrease and terminal strict MAE requirements.

    current/original/prior are cohort_means outputs: raw/native_png, four roles,
    mae/adjacent_rgb_error, with local central128 and other roles whole256.
    inherited_gate_result['pass'] must already include all original individual,
    behavior, teacher and guard gates. At terminal=True its separate
    original_terminal['pass'] is also binding, even when the inherited u4
    top-level pass did not bind the original1percent blur/dark terminal gates.
    No tolerance, independent holdout claim or per-case nonincrease is added.
    """
    need(type(terminal) is bool, 'Explicit terminal boolean required')
    need(type(inherited_gate_result) is dict
         and type(inherited_gate_result.get('pass')) is bool,
         'Explicit complete inherited gate result with boolean pass required')
    terminal_pass = None
    if terminal:
        original_terminal = inherited_gate_result.get('original_terminal')
        need(type(original_terminal) is dict and type(original_terminal.get('pass')) is bool,
             'Terminal4 requires explicit inherited original_terminal boolean pass')
        terminal_pass = original_terminal['pass']
    now, start, previous = _means(current), _means(original), _means(prior)
    nonincrease, terminal_checks = [], []
    for view in VIEWS:
        for role in ROLES:
            for field in FIELDS:
                measured = now[view][role][field]
                for label, baseline in (('original_u64', start), ('prior_state', previous)):
                    reference = baseline[view][role][field]
                    nonincrease.append({'view': view, 'role': role, 'field': field,
                        'reference': label, 'baseline': reference, 'current': measured,
                        'delta': measured - reference, 'pass': measured <= reference})
        if terminal:
            for role in ('clear', 'lowlight_linear_0p35'):
                reference, measured = start[view][role]['mae'], now[view][role]['mae']
                terminal_checks.append({'view': view, 'role': role, 'field': 'mae',
                    'baseline_original_u64': reference, 'current': measured,
                    'delta': measured - reference, 'pass': measured < reference})
    inherited = copy.deepcopy(inherited_gate_result)
    additional_pass = all(row['pass'] for row in nonincrease + terminal_checks)
    passed = inherited['pass'] and additional_pass and (terminal_pass if terminal else True)
    return {'pass': passed, 'terminal': terminal,
        'additional_role_mean_checks_pass': additional_pass,
        'inherited_gate_result': inherited, 'inherited_gate_result_preserved': True,
        'inherited_original_terminal_binding': {'required': terminal, 'pass': terminal_pass},
        'role_mean_nonincrease_checks': nonincrease,
        'terminal_clear_lowlight_strict_MAE_checks': terminal_checks,
        'cohort_means': {'current': now, 'original_u64': start, 'prior_state': previous},
        'metric_region': {r: 'central128' if r == ROLES[0] else 'whole' for r in ROLES},
        'zero_added_tolerance': True, 'individual_case_nonincrease_added': False,
        'original_individual_case_gates_required_unchanged': True,
        'wrapper_recomputed_original_gates': False,
        'scalar_evidence_only': True, 'pixels_or_model_state_verified_here': False,
        'quality_stop': not passed,
        'earlier_state_fallback_or_retry_authorized': False,
        'finite_TRAIN_result_qualifies_model': False, 'model_qualified': False}

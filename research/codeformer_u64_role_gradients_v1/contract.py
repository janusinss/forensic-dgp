"""Pure finite measurement contract; no model, framework, gradient or worker."""
import copy
import hashlib
import json
import math

FORMAT = 'u64-role-gradients-manual-L4-v1'
STAGE = 'cctv_codeformer_u64_role_gradients_v1_vm'
PARENT_STAGE = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PACKAGE = 'research/codeformer_u64_role_gradients_v1'
OUTPUT = 'outputs/u64_role_gradients_v1'
PREFIX = 'cctv-codeformer-u64-role-gradients-v1'
PARENT_BYTES = 279879
PARENT_SHA256 = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
STANDARDIZED_BYTES = 197079
STANDARDIZED_SHA256 = 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
HEAD_CANONICAL_SHA256 = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
QUERIES = ('objective_div4', 'gate_mae', 'gate_adjacent')
LIMITS = {
    'full_forward_attempts': 72, 'full_forward_returns': 72,
    'gradient_query_attempts': 192, 'gradient_query_returns': 192,
    'parity_forwards': 4, 'repeat_forwards': 4, 'diagnostic_forwards': 64,
    'backward_attempts': 0, 'backward_returns': 0,
    'optimizer_constructions': 0, 'optimizer_update_attempts': 0, 'optimizer_updates': 0,
    'parameter_applications': 0, 'worker_seconds': 900, 'external_seconds': 960,
    'termination_grace_seconds': 20, 'per_phase_seconds': 180,
    'failure_capture_seconds': 30, 'failure_receipt_seconds': 5, 'RNG_restore_seconds': 5,
    'export_seconds': 180, 'export_external_seconds': 210,
    'initial_free_disk_bytes': 8 * 1024**3, 'continuing_free_disk_bytes': 4 * 1024**3,
    'maximum_RSS_bytes': 10 * 1024**3, 'maximum_CUDA_reserved_bytes': 18 * 1024**3,
    'maximum_output_bytes': 2 * 1024**3, 'maximum_archive_bytes': 2 * 1024**3,
    'automatic_retry_resume': 0,
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    need(type(raw) is bytes, 'Actual exact protocol bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(item, depth=0):
        need(depth <= 30, 'Bounded JSON depth')
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON number required')
        elif type(item) in (dict, list):
            for child in item.values() if type(item) is dict else item:
                finite(child, depth + 1)
    finite(value)
    return value


def head_shapes():
    shapes = {}
    for site, channels in (('32', 256), ('64', 256), ('128', 128), ('256', 128)):
        for name, shape in {
            'encoder.weight': (32, channels, 1, 1), 'encoder.bias': (32,),
            'decoder.weight': (32, channels, 1, 1), 'fused.weight': (32, channels, 1, 1),
            'coarse.weight': (32, 256, 1, 1), 'spatial.weight': (32, 32, 3, 3),
            'spatial.bias': (32,), 'output.weight': (channels, 32, 1, 1),
            'output.bias': (channels,),
        }.items():
            shapes['heads.' + site + '.' + name] = shape
    need(len(shapes) == 36 and sum(math.prod(shape) for shape in shapes.values()) == 168960,
         'Exact36-head/168960-coordinate schema required')
    return dict(sorted(shapes.items()))


def metric_region(role):
    need(role in ROLES, 'Known paired TRAIN role required')
    return 'central128' if role == ROLES[0] else 'whole'


def validate_selection(selection):
    need(type(selection) is dict and selection.get('parent_protocol_sha256') == PARENT_SHA256
         and selection.get('standardized_parent_protocol_sha256') == STANDARDIZED_SHA256,
         'Exact bound two-parent selection required')
    order, monitors = selection['optimization_reference_order'], selection['monitor_reference_order']
    need(type(order) is list and len(order) == len(set(order)) == 16
         and type(monitors) is list and len(monitors) == len(set(monitors)) == 8
         and not set(order) & set(monitors), 'Distinct16 optimization/8 adaptive monitor photographs required')
    train, monitor, guards = (selection[key] for key in ('optimization', 'monitors', 'guards'))
    need((len(train), len(monitor), len(guards)) == (64, 32, 24), 'Exact64/32/24 rolecase scope required')
    need([(row['reference_id'], row['role']) for row in train] == [(ref, role) for ref in order for role in ROLES]
         and [(row['reference_id'], row['role']) for row in monitor] == [(ref, role) for ref in monitors for role in ROLES],
         'Exact photograph-major four-role measurement order required')
    need(len({row['id'] for row in train + monitor + guards}) == 120
         and all(row['split'] == 'TRAIN' for row in train + monitor + guards), 'Unique TRAIN-only membership required')
    need(all(row['cohort'] == 'optimize' and row['optimization_permitted'] is True and row.get('target') is not None for row in train)
         and all(row['cohort'] == 'monitor' and row['optimization_permitted'] is False for row in monitor),
         'Only admitted optimization rows receive gradient queries')
    for rows in (train, monitor):
        for offset in range(0, len(rows), 4):
            need(all(row['target'] == rows[offset]['target'] for row in rows[offset:offset + 4]), 'Matched photographic clean target across four roles required')
    guard_refs = {row['reference_id'] for row in guards}
    need(len(guard_refs) == 8 and not guard_refs & set(order + monitors)
         and all(row.get('target') is None for row in guards if row['role'] == 'dark'),
         'Legacy guards stay disjoint and null dark targets stay null')
    need(selection['independent_holdout'] is False and selection['DEV_native_wild_FINAL_optimized'] is False,
         'Adaptive TRAIN and unopened evaluation boundaries required')
    return selection


def select_cases(parent_protocol_bytes, standardized_parent_bytes):
    need(type(parent_protocol_bytes) is bytes and (len(parent_protocol_bytes), hashlib.sha256(parent_protocol_bytes).hexdigest())
         == (PARENT_BYTES, PARENT_SHA256), 'Exact sealed retained-u64 continuation protocol required')
    need(type(standardized_parent_bytes) is bytes and (len(standardized_parent_bytes), hashlib.sha256(standardized_parent_bytes).hexdigest())
         == (STANDARDIZED_BYTES, STANDARDIZED_SHA256), 'Exact original standardized TRAIN protocol required')
    parent, standard = strict_json(parent_protocol_bytes), strict_json(standardized_parent_bytes)
    need(parent['stage_name'] == PARENT_STAGE, 'Exact retained parent stage required')
    for name in ('model', 'data', 'cases', 'guards', 'quality_policy', 'inherited_DEV_evidence'):
        need(parent[name] == standard[name], 'Immutable original parent field differs: ' + name)
    need(parent['parent_protocol'] == {'path': 'evidence/u64_parent_protocol.json',
         'bytes': STANDARDIZED_BYTES, 'sha256': STANDARDIZED_SHA256}, 'Exact standardized parent binding required')
    learning = standard['learning']
    need(tuple(learning['slot_roles']) == ROLES, 'Exact original role order required')
    order, monitors = learning['optimization_reference_order'], learning['monitor_reference_order']
    by = {(row['reference_id'], row['role']): row for row in parent['cases']}
    need(len(by) == len(parent['cases']) == 96, 'Exact96 original paired cases required')
    selection = {'parent_protocol_sha256': PARENT_SHA256,
        'standardized_parent_protocol_sha256': STANDARDIZED_SHA256,
        'optimization_reference_order': list(order), 'monitor_reference_order': list(monitors),
        'optimization': [by[(ref, role)] for ref in order for role in ROLES],
        'monitors': [by[(ref, role)] for ref in monitors for role in ROLES], 'guards': parent['guards'],
        'independent_holdout': False, 'DEV_native_wild_FINAL_optimized': False,
        'photograph_separation_is_not_identity_or_pretraining_proof': True}
    validate_selection(selection)
    need([row['id'] for row in parent['u64_parity']] == [row['id'] for row in selection['monitors'][:4]],
         'Exact four historical-u64 parity witnesses required')
    return copy.deepcopy(selection)


def operation_plan(selection):
    validate_selection(selection)
    calls = []
    def append(row, phase, queries):
        calls.append({'slot': len(calls) + 1, 'id': row['id'], 'reference_id': row['reference_id'],
            'role': row['role'], 'phase': phase, 'metric_region': metric_region(row['role']),
            'gradient_queries': list(queries)})
    for phase in ('historical_u64_parity', 'deterministic_same_u64_repeat'):
        for row in selection['monitors'][:4]:
            append(row, phase, ())
    for row in selection['optimization']:
        append(row, 'paired_TRAIN_gradient_diagnostic', QUERIES)
    need(len(calls) == 72 and sum(len(row['gradient_queries']) for row in calls) == 192,
         'Exact72-forward/192-query finite diagnostic required')
    return calls


def prospective_contract():
    return {'status': 'INERT_ROLE_GRADIENT_MEASUREMENT_NOT_A_TRAINING_RECIPE',
        'initializer': 'Exact retained trained-u64 terminal64, never failed-new8; complete575 values unchanged',
        'initial_head_canonical_sha256': HEAD_CANONICAL_SHA256,
        'total_tensors': 575, 'unchanged_tensors': 575, 'frozen_R2_U96_tensors': 539,
        'gradient_tensors': 36, 'gradient_scalars': 168960,
        'roles': list(ROLES), 'query_order': list(QUERIES),
        'objective_weighting': 'objective_div4 gradients already contain exactly one division by4; sum four supplied vectors without another division',
        'metric_regions': {role: metric_region(role) for role in ROLES},
        'metric_domain': 'Connected delivered RAW: clamp native512 RGB before bilinear256; no PNG derivative',
        'cohort': 'All16 admitted optimization photographs times four matched roles; monitors/guards never queried',
        'zero_vectors': 'Retain measured zero and cancellation; cosine undefined at zero norm. Caller authenticates explicitly unused coordinates/zeros in query receipts; never drop/impute a mandatory query.',
        'first_order_scope': 'Local gradient-dot-direction arithmetic only; no finite-step, Adam-trajectory or PNG-quality prediction',
        'hypothetical_fresh_Adam': {'vector_arithmetic_only': True, 'empty_moments': True,
            'LR': 0.000025, 'clip_max_norm': 1.0, 'clip_denominator_epsilon': 0.000001,
            'Adam_denominator_epsilon': 0.00000001, 'parameter_application': False},
        'old_schedule_control': 'Recombine the same common-state vectors in the original16update offsets0/4/8/12, without changing state',
        'no_common_descent_solver': True, 'no_PCGrad_or_reweighting': True,
        'no_optimizer_or_parameter_application': True, 'all_original_quality_gates_unchanged': True,
        'limits': copy.deepcopy(LIMITS), 'manual_execution_only': True,
        'model_qualified': False, 'sealed_packet_or_launch_guide_ready': False}

"""Pure, exact scope for a new manually launched four-update L4 pilot."""

import copy
import hashlib

from research.codeformer_u64_role_gradients_v1 import contract as _roles
from research.codeformer_u64_gate_aligned_v1 import contract as _design

FORMAT = 'u64-gate-aligned-pilot-manual-L4-v1'
STAGE = 'cctv_codeformer_u64_gate_aligned_pilot_v1_vm'
PARENT_STAGE = _roles.PARENT_STAGE
PACKAGE = 'research/codeformer_u64_gate_aligned_pilot_v1'
OUTPUT = 'outputs/u64_gate_aligned_pilot_v1'
PREFIX = 'cctv-codeformer-u64-gate-aligned-pilot-v1'
ROLES = _roles.ROLES
QUERIES = ('gate_mae', 'gate_adjacent')
HEAD_CANONICAL_SHA256 = _design.HEAD_CANONICAL_SHA256
FULL_STATE_CANONICAL_SHA256 = _design.FULL_STATE_CANONICAL_SHA256
GRADIENT_PROTOCOL_SHA256 = _design.PARENT_GRADIENT_PROTOCOL_SHA256
PARENT_SHA256 = _roles.PARENT_SHA256
STANDARDIZED_SHA256 = _roles.STANDARDIZED_SHA256
SCALAR_SAFETY_FACTOR = .99
ENV = 'FORENSIC_U64_GATE_ALIGNED_PILOT_'
CATEGORIES = {'initial_quality_forwards': 56, 'initial_repeat_forwards': 4,
              'optimization_forwards': 256, 'quality_forwards': 224}
LIMITS = {**_design.operation_limits(), **CATEGORIES,
          'failure_capture_seconds': 30, 'failure_receipt_seconds': 5,
          'RNG_restore_seconds': 5}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def head_shapes():
    return _roles.head_shapes()


def select_cases(parent_protocol_bytes, standardized_parent_bytes):
    return _roles.select_cases(parent_protocol_bytes, standardized_parent_bytes)


def operation_plan(selection):
    _roles.validate_selection(selection)
    calls = []

    def forward(row, update, phase, category, queries=()):
        calls.append({'slot': len(calls) + 1, 'kind': 'forward',
                      'new_update': update, 'id': row['id'],
                      'reference_id': row['reference_id'], 'role': row['role'],
                      'phase': phase, 'category': category,
                      'metric_region': 'central128' if row['role'] == ROLES[0] else 'whole',
                      'gradient_queries': list(queries)})

    bank = selection['monitors'] + selection['guards']
    for row in bank:
        forward(row, 0, 'initial_u64_quality_bank', 'initial_quality_forwards')
    for row in selection['monitors'][:4]:
        forward(row, 0, 'deterministic_same_u64_repeat', 'initial_repeat_forwards')
    for update in range(1, 5):
        for row in selection['optimization']:
            forward(row, update, 'paired_TRAIN_gate_gradients', 'optimization_forwards', QUERIES)
        calls.append({'slot': len(calls) + 1, 'kind': 'parameter_application',
                      'new_update': update, 'id': 'apply_u' + str(update).zfill(4),
                      'phase': 'gate_aligned_parameter_application',
                      'gradient_queries': []})
        for row in bank:
            forward(row, update, 'current_quality_bank', 'quality_forwards')
    need(len(calls) == 544, 'Exactly540 forward slots plus4 parameter applications required')
    need(sum(row['kind'] == 'forward' for row in calls) == 540
         and sum(len(row['gradient_queries']) for row in calls) == 512,
         'Exact finite operation budget required')
    for category, count in CATEGORIES.items():
        need(sum(row.get('category') == category for row in calls) == count,
             'Exact forward category budget required')
    return calls


def prospective_contract(parent_bytes, standard_bytes, gradient_bytes):
    selection = select_cases(parent_bytes, standard_bytes)
    need(type(gradient_bytes) is bytes
         and hashlib.sha256(gradient_bytes).hexdigest() == GRADIENT_PROTOCOL_SHA256,
         'Exact previously returned diagnostic protocol required')
    design = _design.prospective_contract(gradient_bytes, standard_bytes)
    need(selection == design['selection'], 'Original optimization/monitor/guard membership differs')
    return {
        'format': FORMAT,
        'status': 'DISTINCT_FINITE_MANUAL_GATE_ALIGNED_PILOT_REQUIRES_REVIEWED_PACKET',
        'original_source_only_design': design,
        'initializer': copy.deepcopy(design['initializer']),
        'roles': list(ROLES), 'gradient_query_order': list(QUERIES),
        'limits': copy.deepcopy(LIMITS),
        'step_scalar_safety_factor': SCALAR_SAFETY_FACTOR,
        'step_scalar_reason': 'One fixed0.99 numerical headroom factor before FP32 rounding, independent of images or scores; no retries or metric-dependent line search.',
        'original_quality_policy_exact': copy.deepcopy(design['quality']['inherited_policy_exact']),
        'initial_quality': 'Exact56 original-u64 bank plus four full deterministic repeats; frozen U96 references and original teachers stay unchanged.',
        'per_update_quality': 'All original inherited gates plus16 role-mean RAW/PNG endpoints nonincrease versus originalu64 AND prior accepted state.',
        'terminal_quality': 'Original terminal1percent requirements bind at4; clear/dark photographic MAE strictly improve in both RAW/PNG; no earlier failed-state selection.',
        'all_gradients_current_state': True,
        'query_region': {role: 'central128' if role == ROLES[0] else 'whole' for role in ROLES},
        'same_current_state_16_photos_pooled_before_one_normalization': True,
        'finite_step_validation': 'Actual rounded FP32 candidate displacement in FP64 must satisfy every coordinate/cumulative bound and all8 strictly negative current pooled gradient dots before copy.',
        'optimizer_state': 'No optimizer object or momentum exists; explicit null optimizer-state receipt, plain certified direction applications only.',
        'RNG_and_sampler': 'Initial and every applied head checkpoint preserve Python/NumPy/CPU/CUDA RNG and exact next plan slot. No automatic resume.',
        'retained_full575_states': 2,
        'retained_head_states': 5,
        'retained_each_forward': 'DeliveredRAW256 and floorPNG256, source/target/current-state bindings and metrics. Unclipped/native/logits/LQ arrays only for8 initial parity/repeat witnesses.',
        'retained_each_query': 'All36 connected finite FP32 head tensors and exact scalar/role/slot/state/schema/canonical binding; unused or all-zero mandatory query stops after receipt.',
        'retained_each_update': 'All128 query files; pooled8 vectors; common direction, exhaustive-face certificate, FP32 candidate, actual roundeddelta and pre/post head bindings.',
        'worker_return_codes': {'screen_pass': 0, 'quality_stop': 2, 'other_failure': 1},
        'partial_parameter_application': 'Durable reservation remains unknown if interrupted between36 tensor copies; never roll back or claim an applied return.',
        'manual_launch_only': True, 'automatic_retry_resume': False,
        'same_failed_recipe_repeated': False,
        'previous_Adam8_failure_retained': True,
        'inherited_broader_DEV_failures_remain_binding': True,
        'native_wild_FINAL_opened_or_optimized': False,
        'model_qualified': False, 'app_incumbent_changed': False,
    }

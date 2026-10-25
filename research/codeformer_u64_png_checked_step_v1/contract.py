"""Pure exact finite scope for a prospective manual PNG-checked step packet.

This contract constructs no model, optimizer, autograd query or parameter
application. Every trial/restore/final copy is an explicit future operation.
"""
from research.codeformer_u64_role_gradients_v1 import contract as _roles

FORMAT = 'u64-png-checked-step-manual-L4-v1'
STAGE = 'cctv_codeformer_u64_png_checked_step_v1_vm'
PARENT_STAGE = _roles.PARENT_STAGE
PACKAGE = 'research/codeformer_u64_png_checked_step_v1'
OUTPUT = 'outputs/u64_png_checked_step_v1'
PREFIX = 'cctv-codeformer-u64-png-checked-step-v1'
ENV = 'FORENSIC_U64_PNG_CHECKED_STEP_'
ROLES = _roles.ROLES
QUERIES = ()
PARENT_BYTES = _roles.PARENT_BYTES
PARENT_SHA256 = _roles.PARENT_SHA256
STANDARDIZED_BYTES = _roles.STANDARDIZED_BYTES
STANDARDIZED_SHA256 = _roles.STANDARDIZED_SHA256
HEAD_CANONICAL_SHA256 = _roles.HEAD_CANONICAL_SHA256
FULL_STATE_CANONICAL_SHA256 = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
FROZEN539_CANONICAL_SHA256 = 'f096108dc16dff56687a61a57d7ee367f8dd50accc385d2d72be8b53d4510cdf'
# Deliberately duplicated pure constants keep static bootstrap/--help free of
# NumPy and neural imports. Pure-array tests bind them to the selector registry.
CANDIDATE_MAXABS = (2e-5, 1.5e-5, 1e-5, 5e-6)
NUMERICAL_HEADROOM = .99
CATEGORIES = {'initial_quality_forwards': 56, 'initial_repeat_forwards': 4,
    'initial_optimization_forwards': 64, 'trial_optimization_forwards': 256,
    'selected_quality_forwards': 56}
LIMITS = {'full_forward_attempts': 436, 'full_forward_returns': 436,
    'parameter_application_attempts': 9, 'parameter_applications': 9,
    'trial_parameter_applications': 4, 'original_restoration_applications': 4,
    'selected_parameter_commits': 1, 'selected_training_updates': 1,
    'gradient_query_attempts': 0, 'gradient_query_returns': 0,
    'backward_attempts': 0, 'backward_returns': 0,
    'optimizer_constructions': 0, 'optimizer_update_attempts': 0, 'optimizer_updates': 0,
    'worker_seconds': 1200, 'external_seconds': 1260, 'termination_grace_seconds': 20,
    'per_phase_seconds': 180, 'export_seconds': 300, 'export_external_seconds': 330,
    'failure_capture_seconds': 30, 'failure_receipt_seconds': 5, 'RNG_restore_seconds': 5,
    'initial_free_disk_bytes': 8 * 1024**3, 'continuing_free_disk_bytes': 4 * 1024**3,
    'maximum_RSS_bytes': 10 * 1024**3, 'maximum_CUDA_reserved_bytes': 18 * 1024**3,
    'maximum_output_bytes': 2 * 1024**3, 'maximum_archive_bytes': 2 * 1024**3,
    'original_maximum_absolute_coordinate_step': 2.5e-5,
    'maximum_cumulative_absolute_coordinate_delta': 1e-4,
    'automatic_retry_resume': 0, **CATEGORIES}
NO_ELIGIBLE_PREFIX = {'plan_rows': 388, 'full_forward_attempts': 380,
    'full_forward_returns': 380, 'parameter_application_attempts': 8,
    'parameter_applications': 8, 'selected_parameter_commits': 0,
    'final_head_canonical_sha256': HEAD_CANONICAL_SHA256}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def head_shapes():
    return _roles.head_shapes()


def metric_region(role):
    # The56-case quality bank also contains inherited dark/filter guard roles.
    need(role in ROLES or role in ('dark', 'filter'), 'Known paired or inherited guard role required')
    return 'central128' if role == ROLES[0] else 'whole'


def select_cases(parent_protocol_bytes, standardized_parent_bytes):
    return _roles.select_cases(parent_protocol_bytes, standardized_parent_bytes)


def operation_plan(selection):
    _roles.validate_selection(selection)
    calls = []
    def forward(row, phase, category, trial_index=None):
        calls.append({'slot': len(calls) + 1, 'kind': 'forward', 'id': row['id'],
            'reference_id': row['reference_id'], 'role': row['role'],
            'metric_region': metric_region(row['role']), 'phase': phase,
            'category': category, 'trial_index': trial_index, 'gradient_queries': []})
    def application(action, trial_index):
        label = 'selected' if trial_index is None else 'trial_' + str(trial_index)
        calls.append({'slot': len(calls) + 1, 'kind': 'parameter_application',
            'id': action + '_' + label, 'action': action, 'trial_index': trial_index,
            'phase': action, 'gradient_queries': [],
            'source_state': 'candidate_trial_' + str(trial_index) if action == 'trial_restore' else 'original_u64',
            'destination_state': 'original_u64' if action == 'trial_restore' else
                ('candidate_trial_' + str(trial_index) if action == 'trial_apply' else 'selected_eligible_candidate'),
            'destination_head_canonical_sha256': HEAD_CANONICAL_SHA256 if action == 'trial_restore' else None,
            'worker_must_record_exact_before_after_canonical': True})
    bank = selection['monitors'] + selection['guards']
    for row in bank:
        forward(row, 'initial_u64_quality_bank', 'initial_quality_forwards')
    for row in selection['monitors'][:4]:
        forward(row, 'deterministic_same_u64_repeat', 'initial_repeat_forwards')
    for row in selection['optimization']:
        forward(row, 'initial_optimization_bank', 'initial_optimization_forwards')
    for trial_index in range(4):
        application('trial_apply', trial_index)
        for row in selection['optimization']:
            forward(row, 'trial_optimization_bank', 'trial_optimization_forwards', trial_index)
        application('trial_restore', trial_index)
    need(len(calls) == NO_ELIGIBLE_PREFIX['plan_rows'], 'Exact380forward/eightapplication no-eligible prefix required')
    application('selected_commit', None)
    for row in bank:
        forward(row, 'selected_quality_bank', 'selected_quality_forwards')
    need(len(calls) == 445 and sum(c['kind'] == 'forward' for c in calls) == 436
         and sum(c['kind'] == 'parameter_application' for c in calls) == 9
         and not any(c['gradient_queries'] for c in calls), 'Exact436forward/nineapplication/zeroquery plan required')
    for category, count in CATEGORIES.items():
        need(sum(c.get('category') == category for c in calls) == count, 'Exact bounded forward category required')
    return calls


def prospective_contract():
    return {'status': 'FINITE_PNG_CHECKED_STEP_REQUIRES_REVIEWED_MANUAL_PACKET',
        'format': FORMAT, 'stage_name': STAGE,
        'original_full575_canonical_sha256': FULL_STATE_CANONICAL_SHA256,
        'original_head36_canonical_sha256': HEAD_CANONICAL_SHA256,
        'frozen539_canonical_sha256': FROZEN539_CANONICAL_SHA256,
        'initializer': 'Exact original trained-u64 terminal64; neither failed pilotu2 nor earlieru1.',
        'total_tensors': 575, 'trainable_head_tensors': 36, 'head_coordinates': 168960,
        'frozen_R2_U96_tensors': 539, 'roles': list(ROLES),
        'metric_regions': {r: metric_region(r) for r in ROLES},
        'direction': 'Authenticate and reconstruct original-u64 eight-RAW pooled direction from audited saved gradient witnesses; no new gradient/autograd query or16-vector replacement.',
        'candidate_maximum_absolute_coordinate_bounds': list(CANDIDATE_MAXABS),
        'fixed_numerical_headroom': NUMERICAL_HEADROOM,
        'scalar': '0.99*registered_bound/maxabs(direction), fixed before any metric; no adaptive retries or appended scales.',
        'candidate_independence': 'Every candidate and final selected value is formed independently from exact originalu64, never previous trial.',
        'actual_candidate_guards': 'FP32 actual delta below registered bound AND original2.5e-5 step cap AND1e-4 cumulative cap; all eight saved-original RAW gradient dots strictly negative before candidate/final copy.',
        'restore_semantics': 'Four explicit original-state copies, including after fourthtrial; journal actual before/after hashes and exact destination originalhead canonical. Restores are counted applications, not descent steps; no RAW descent dot requirement on restoration.',
        'selection': 'Require all four complete64-ID optimization TRAIN banks with original role/ROI/input/target provenance; no monitor or guard API. All16 actual RAW/PNG means nonincrease vs originalu64 and both blur PNGadjacent means strictly smaller; zero tolerance; first eligible fixed order or noeligible.',
        'no_eligible': dict(NO_ELIGIBLE_PREFIX),
        'no_eligible_action': 'Stop after fourth restore with exact originalhead state; no selected commit or56 final quality forwards.',
        'post_selection': 'Commit once and evaluate56 monitor/guard cases exactly once; monitor failure rejects entire test with no next candidate or oldstate fallback.',
        'quality': 'All original teachers, baseline, individual behavior limits,1percent blur structure vs frozenU96, clear/lowlight requirements and16RAW/PNG rolemean retention vs originalu64 remain binding.',
        'ineligible_vs_fatal': 'Only valid actual finite quality ineligibility advances to next preregistered trial. Any state/provenance/rounding/sign/nonfinite/RNG/resource/durable failure stops; interrupted copies remain unknown and do not gain retrospective clean restore claims.',
        'evidence': 'Initial/final575; every candidate/restore/selectedhead value and actualdelta/prepostcanonicals; all436 planned RAWPNG pairs, individual regressions, fixedsampler/RNG/nulloptimizer and durable operation/resource/failure records.',
        'all_original_quality_gates_unchanged': True, 'no_new_gradient_queries': True,
        'actual_trial_and_restore_applications_counted': True,
        'limits': dict(LIMITS), 'manual_execution_only': True,
        'automatic_retry_resume': False, 'prior_quality_failures_remain_binding': True,
        'DEV_native_wild_FINAL_optimized': False, 'model_qualified': False,
        'app_incumbent_changed': False, 'worker_implemented_by_this_contract': False,
        'sealed_packet_or_launch_guide_ready_by_this_contract': False}

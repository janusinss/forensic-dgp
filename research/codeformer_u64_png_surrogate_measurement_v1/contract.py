"""Pure bounded scope for a separate zero-update measurement, not a trainer."""
from research.codeformer_u64_role_gradients_v1 import contract as _roles

FORMAT = 'u64-png-surrogate-measurement-manual-L4-v1'
STAGE = 'cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
PARENT_STAGE = _roles.PARENT_STAGE
PACKAGE = 'research/codeformer_u64_png_surrogate_measurement_v1'
OUTPUT = 'outputs/u64_png_surrogate_measurement_v1'
PREFIX = 'cctv-codeformer-u64-png-surrogate-measurement-v1'
ENV = 'FORENSIC_U64_PNG_SURROGATE_MEASUREMENT_'
ROLES = _roles.ROLES
QUERIES = ('raw_mae', 'raw_adjacent', 'png_surrogate_mae', 'png_surrogate_adjacent')
PARENT_BYTES = _roles.PARENT_BYTES
PARENT_SHA256 = _roles.PARENT_SHA256
STANDARDIZED_BYTES = _roles.STANDARDIZED_BYTES
STANDARDIZED_SHA256 = _roles.STANDARDIZED_SHA256
HEAD_CANONICAL_SHA256 = _roles.HEAD_CANONICAL_SHA256
FULL_STATE_CANONICAL_SHA256 = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
FROZEN539_CANONICAL_SHA256 = 'f096108dc16dff56687a61a57d7ee367f8dd50accc385d2d72be8b53d4510cdf'
LIMITS = {'full_forward_attempts': 72, 'full_forward_returns': 72,
    'gradient_query_attempts': 256, 'gradient_query_returns': 256,
    'parity_forwards': 4, 'repeat_forwards': 4, 'diagnostic_forwards': 64,
    'backward_attempts': 0, 'backward_returns': 0,
    'optimizer_constructions': 0, 'optimizer_update_attempts': 0, 'optimizer_updates': 0,
    'parameter_application_attempts': 0, 'parameter_applications': 0,
    'worker_seconds': 900, 'external_seconds': 960, 'termination_grace_seconds': 20,
    'per_phase_seconds': 180, 'failure_capture_seconds': 30, 'failure_receipt_seconds': 5,
    'RNG_restore_seconds': 5, 'export_seconds': 300, 'export_external_seconds': 330,
    'initial_free_disk_bytes': 8 * 1024**3, 'continuing_free_disk_bytes': 4 * 1024**3,
    'maximum_RSS_bytes': 10 * 1024**3, 'maximum_CUDA_reserved_bytes': 18 * 1024**3,
    'maximum_output_bytes': 2 * 1024**3, 'maximum_archive_bytes': 2 * 1024**3,
    'automatic_retry_resume': 0}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def head_shapes():
    return _roles.head_shapes()


def metric_region(role):
    return _roles.metric_region(role)


def validate_selection(selection):
    return _roles.validate_selection(selection)


def select_cases(parent_protocol_bytes, standardized_parent_bytes):
    return _roles.select_cases(parent_protocol_bytes, standardized_parent_bytes)


def operation_plan(selection):
    validate_selection(selection)
    calls = []
    def append(row, phase, queries):
        calls.append({'slot': len(calls) + 1, 'id': row['id'],
            'reference_id': row['reference_id'], 'role': row['role'], 'phase': phase,
            'metric_region': metric_region(row['role']), 'gradient_queries': list(queries)})
    for phase in ('historical_u64_parity', 'deterministic_same_u64_repeat'):
        for row in selection['monitors'][:4]:
            append(row, phase, ())
    for row in selection['optimization']:
        append(row, 'paired_TRAIN_gradient_diagnostic', QUERIES)
    need(len(calls) == 72 and sum(len(c['gradient_queries']) for c in calls) == 256,
         'Exact72-forward/256-query zero-update plan required')
    return calls


def prospective_contract():
    return {'status': 'ZERO_UPDATE_PNG_SURROGATE_MEASUREMENT_NOT_A_TRAINING_RECIPE',
        'format': FORMAT, 'stage_name': STAGE,
        'initializer': 'Exact original trained-u64 terminal64; never failed gate-alignedu2 or earlieru1.',
        'full575_canonical_sha256': FULL_STATE_CANONICAL_SHA256,
        'head36_canonical_sha256': HEAD_CANONICAL_SHA256,
        'frozen539_canonical_sha256': FROZEN539_CANONICAL_SHA256,
        'total_tensors': 575, 'unchanged_tensors': 575, 'frozen_R2_U96_tensors': 539,
        'gradient_tensors': 36, 'gradient_scalars': 168960,
        'roles': list(ROLES), 'query_order': list(QUERIES),
        'metric_regions': {role: metric_region(role) for role in ROLES},
        'query_cohort': 'All16 admitted optimization TRAIN photographs times four matched roles; monitors/guards never receive gradient queries.',
        'RAW_domain': 'Clamp native512 RGB before bilinear256, retaining the unchanged delivered FP32 path.',
        'PNG_forward_reference': 'Actual durable unchanged floor-RGB8 PNG decoded on CPU to FP32 and divided by FP32(255); worker binds file/pixels/scalars and supplies tensor on delivered device.',
        'surrogate': {'forward': 'Detached clone of caller-supplied actual saved-PNG decoded tensor; exact bitwise values, no quantization recalculation or cancellation expression.',
            'backward': 'Identity grad_output for delivered RAW input; None for decoded-PNG reference.',
            'biased_estimator': True, 'actual_PNG_derivative': False,
            'true_floor_derivative': 'Zero almost everywhere and undefined at quantization boundaries.',
            'caller_must_bind_actual_saved_PNG_reference': True},
        'gradient_policy': 'All four mandatory query36-tensor witnesses retained; measured zeros and explicitly unused coordinates remain distinguishable, with no missing-query imputation.',
        'analysis': 'Per-photo and pooled16 vector norms, pairwise geometry, RAW versus PNG-surrogate alignment and cancellation only; no common-direction solver or parameter directions.',
        'pooled_vectors': 16, 'no_eight_vector_solver_on_16': True,
        'initial_and_closing575_arrays_unchanged_required': True,
        'native_logits_LQ_triples': 'Only four original parity and four repeat witnesses; all72 durable RAW/PNG pairs retained.',
        'limits': dict(LIMITS), 'all_original_quality_gates_unchanged': True,
        'manual_execution_only': True, 'automatic_retry_resume': False,
        'DEV_native_wild_FINAL_optimized': False, 'inherited_broader_failures_retained': True,
        'finite_step_or_PNG_success_certified': False, 'new_training_recipe_selected': False,
        'sealed_packet_or_launch_guide_ready': False, 'model_qualified': False,
        'app_incumbent_changed': False}

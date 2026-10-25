"""Inert prospective four-step gate-aligned design; no runnable worker exists."""
import copy
import hashlib
import json

FORMAT = 'u64-gate-aligned-source-only-design-v1'
PARENT_GRADIENT_PROTOCOL_SHA256 = '07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3'
STANDARD_PROTOCOL_BYTES = 197079
STANDARD_PROTOCOL_SHA256 = 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
HEAD_CANONICAL_SHA256 = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
FULL_STATE_CANONICAL_SHA256 = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
DOMAINS = ('RAW', 'PNG')
GATES = ('gate_mae', 'gate_adjacent')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def operation_limits():
    # 32 paired adaptive TRAIN monitors +24 unchanged legacy guards =56.
    # Each update recomputes64 optimization forwards and128 gate queries, then
    # evaluates every one of56 quality-bank cases. Four initial repeats are
    # additional to the complete initial56 quality bank.
    return {'updates': 4, 'initial_quality_bank_forwards': 56,
        'initial_repeat_forwards': 4, 'optimization_forwards_per_update': 64,
        'gate_queries_per_optimization_forward': 2, 'quality_bank_forwards_per_update': 56,
        'full_forward_attempts': 56 + 4 + 4 * (64 + 56),
        'full_forward_returns': 56 + 4 + 4 * (64 + 56),
        'gradient_query_attempts': 4 * 64 * 2, 'gradient_query_returns': 4 * 64 * 2,
        'parameter_application_attempts': 4, 'parameter_applications': 4,
        'optimizer_constructions': 0, 'backward_attempts': 0, 'backward_returns': 0,
        'solver_variables': 8, 'solver_faces_per_update_maximum': 255,
        'solver_seconds_per_update_maximum': 10.,
        'maximum_absolute_coordinate_step': 2.5e-5,
        'maximum_cumulative_absolute_coordinate_delta': 1e-4,
        'worker_seconds': 1800, 'external_seconds': 1860,
        'termination_grace_seconds': 20, 'per_phase_seconds': 180,
        'export_seconds': 300, 'export_external_seconds': 330,
        'maximum_RSS_bytes': 10 * 1024**3,
        'maximum_CUDA_reserved_bytes': 18 * 1024**3,
        'maximum_output_bytes': 2 * 1024**3,
        'maximum_archive_bytes': 2 * 1024**3,
        'initial_free_disk_bytes': 8 * 1024**3,
        'continuing_free_disk_bytes': 4 * 1024**3,
        'automatic_retry_resume': 0}


def prospective_contract(gradient_protocol_bytes, standard_protocol_bytes):
    """Bind the unchanged cohort, legacy guards and original quality policies.

    Source-only design output. A separately reviewed future worker would need
    to establish every run bound and state/renderer/gradient receipt; this
    contract neither implements nor authorizes that worker or a VM launch.
    """
    need(type(gradient_protocol_bytes) is bytes and hashlib.sha256(gradient_protocol_bytes).hexdigest()
         == PARENT_GRADIENT_PROTOCOL_SHA256, 'Exact sealed returned gradient protocol required')
    need(type(standard_protocol_bytes) is bytes and len(standard_protocol_bytes) == STANDARD_PROTOCOL_BYTES
         and hashlib.sha256(standard_protocol_bytes).hexdigest() == STANDARD_PROTOCOL_SHA256,
         'Exact original standardized TRAIN protocol required')
    gradient = json.loads(gradient_protocol_bytes)
    standard = json.loads(standard_protocol_bytes)
    selection = gradient['selection']
    need(tuple(gradient['diagnostic']['roles']) == ROLES
         and len(selection['optimization']) == 64 and len(selection['monitors']) == 32
         and len(selection['guards']) == 24, 'Original64/32/24 cohort required')
    need(all(row['split'] == 'TRAIN' and row['cohort'] == 'optimize'
             and row['optimization_permitted'] is True for row in selection['optimization']),
         'Only original64 admitted TRAIN rolecases may receive gradient queries')
    need(standard['quality_policy']['terminal64_adjacent_gain_fraction_minimum'] == .01,
         'Original1percent structure gate required')
    limits = operation_limits()
    need(limits['full_forward_attempts'] == 540 and limits['gradient_query_attempts'] == 512,
         'Exact finite540forward/512query design required')
    return {'format': FORMAT, 'status': 'INERT_SOURCE_ONLY_DESIGN_NOT_A_MANUAL_PACKET',
        'initializer': {'description': 'Exact original retained trained-u64 terminal64; never failed-new8',
            'full575_canonical_sha256': FULL_STATE_CANONICAL_SHA256,
            'head36_canonical_sha256': HEAD_CANONICAL_SHA256,
            'all_tensors': 575, 'trainable_head_tensors': 36, 'trainable_head_coordinates': 168960,
            'frozen_R2_U96_tensors': 539},
        'parent_gradient_protocol_sha256': PARENT_GRADIENT_PROTOCOL_SHA256,
        'standard_protocol_sha256': STANDARD_PROTOCOL_SHA256,
        'selection': copy.deepcopy(selection), 'roles': list(ROLES),
        'metric_regions': {r: 'central128' if r == ROLES[0] else 'whole' for r in ROLES},
        'connected_gradient_domain': 'Delivered RAW: clamp native512 RGB before bilinear256; no PNG derivative',
        'gradient_measurements': {'queries': list(GATES), 'per_update': 128,
            'all_64_original_optimization_rolecases': True,
            'pool': 'Arithmetic mean16 per role/query at the same current state; normalize each of eight pooled vectors once.',
            'missing_zero_nonfinite_disconnected_mandatory_query': 'STOP; never omit, impute, drop role or fallback.'},
        'direction_rule': {'method': 'Exhaustive nonempty simplex-face minimum-norm convex combination of eight unit pooled gate gradients',
            'KKT_method': 'Symmetric8x8 Gram; bounded np.linalg.lstsq on each face; singular faces require residual/weight/normalization checks.',
            'maximum_faces': 255, 'maximum_seconds': 10.,
            'strict_actual_normalized_margin_minimum_exclusive': 1e-6,
            'all_eight_actual_full_coordinate_dots_strict_negative_required': True,
            'no_feasible_certificate': 'STOP without fallback or parameter application.',
            'direction': 'Negative minimum-norm combination normalized to unit L2.'},
        'prospective_step': {'method': 'Four finite plain gradient-descent direction applications; recompute all gradients at every current state.',
            'scalar': 'Positive scalar selected only by maximum absolute coordinate step2.5e-5; no metric-dependent line search.',
            'maximum_absolute_coordinate_step': 2.5e-5,
            'maximum_cumulative_absolute_coordinate_delta_from_original_u64': 1e-4,
            'no_Adam_or_coordinate_preconditioning': True,
            'reason': 'Coordinate preconditioning can destroy the measured common-descent signs.',
            'no_router_or_dark_drop_or_adaptive_recipe_or_line_search': True,
            'FP32_rounding_precommit_guard': 'Reconstruct actual rounded FP32 candidate displacement in FP64; require every committed36-head coordinate delta maxabs<=2.5e-5 and cumulative delta from originalu64 maxabs<=1e-4.',
            'rounded_displacement_gradient_guard': 'Before commit, require actual rounded displacement dot each of eight current pooled gradients strictly negative. The unrounded direction certificate does not imply this.',
            'rounding_guard_failure': 'STOP with all evidence retained; no adaptive alpha retries, line search, dropped gate or fallback.',
            'bounds_are_prospective_requirements_not_locally_measured_model_deltas': True},
        'quality': {'initial_and_after_every_update_complete56_case_bank': True,
            'RAW_and_PNG_checked_separately': True,
            'inherited_policy_exact': copy.deepcopy(standard['quality_policy']),
            'inherited_legacy24_guard_definitions_exact': copy.deepcopy(standard['guards']),
            'thresholds_teachers_and_case_limits_never_rebased': True,
            'all8_role_mean_endpoints': 'Four roles times MAE/adjacent, in each of RAW and PNG, nonincrease at every update relative to preceding accepted current state AND originalu64.',
            'individual_case_MAE_regressions': 'Report every individual regression; unchanged original individual-case and behavior limits still apply.',
            'final4_clear_and_lowlight_role_mean_MAE': 'Strictly better than originalu64 in BOTH RAW and PNG.',
            'final4_blur_structure': 'Original1percent minimum gain in global and central-local adjacent endpoints in BOTH RAW and PNG.',
            'final4_lowlight_MAE_adjacent_nonregression': True,
            'historical64_structure_gate_is_evaluated_at_prospective_terminal4_without_relaxation': True,
            'inherited_later_training_snapshots_are_not_executed': True,
            'any_quality_stop': 'Retain all states and failure evidence; no retrospective earlier-state selection, continuation or automatic rerun.'},
        'limits': limits, 'new_raw_renderer_or_mask': False,
        'prospective_evidence': {'all512_saved_gradients': 'Every query retains all36 FP32 tensors, exact canonical/file/schema/scalar/query/role/state receipts, including measured zeros and explicitly unused coordinates.',
            'full575_states': 'Exactly initial and closing complete575-array witnesses, including closing after any quality stop; preserve originalu64 and every failed state.',
            'head_states': 'Retain initial and every one of at most4 applied36-head states, including failed quality endpoints; no retrospective earlier-state selection.',
            'every_forward_RAW_PNG': 'Save durable delivered RAW and PNG plus source-input/target pins, metrics and output hashes for every successful forward.',
            'native_logits_LQ': 'Save only four selected initial-bank parity cases and their four repeat witnesses, maximum8 triples; compare exact values/schema/hash. Do not retain these large stock tensors for all540 forwards.',
            'capture_failure': 'STOP; incomplete evidence does not qualify success or authorize retry.'},
        'planning': {'unmeasured_runtime_estimate_minutes': [8, 16],
            'estimate_is_not_a_benchmark_or_execution_receipt': True,
            'reviewed_external_VM_supervisor_required': True,
            'external_VM_supervisor_implemented_here': False,
            'resource_caps_are_prospective_not_observed': True},
        'DEV_native_wild_FINAL_opened_or_optimized': False,
        'prior_failures_remain_binding': True, 'paired_TRAIN_is_not_independent_holdout': True,
        'worker_implemented': False, 'trainer_implemented': False,
        'bootstrap_or_transfer_package_implemented': False, 'manual_packet_ready': False,
        'pilot_selected_or_authorized': False, 'VM_launched': False,
        'headspace_local_feasibility_is_not_finite_step_or_PNG_quality': True,
        'model_qualified': False, 'app_incumbent_changed': False}

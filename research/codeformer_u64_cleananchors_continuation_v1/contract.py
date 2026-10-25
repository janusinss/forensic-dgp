"""Inert finite u64 warm-start design; no worker, CLI or automatic launch."""
import copy
import hashlib
import json
import math
from .objective import ROLES, formula_contract

STATUS = 'INERT_U64_WARM_START_SOURCE_NOT_A_SEALED_EXECUTION_PACKET'
PARENT_BYTES = 197079
PARENT_SHA256 = 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
HEAD_BYTES = 685818
HEAD_FILE_SHA256 = '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'
HEAD_CANONICAL_SHA256 = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
SNAPSHOTS = (0, 4, 8, 16)
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'adjacent_rgb_error')
LIMITS = {
    'full_forward_attempts': 292, 'full_forward_returns': 292,
    'initial_quality_bank_forwards': 56, 'deterministic_repeat_forwards': 4,
    'candidate_snapshot_forwards_after_initializer': 168,
    'candidate_training_forwards': 64,
    'backward_attempts': 64, 'backward_returns': 64,
    'gradient_queries': 0, 'optimizer_constructions': 1,
    'optimizer_update_attempts': 16, 'optimizer_updates': 16,
    'worker_seconds': 1200, 'external_seconds': 1260,
    'termination_grace_seconds': 20, 'per_phase_seconds': 180,
    'failure_capture_seconds': 30, 'failure_receipt_seconds': 5,
    'RNG_restore_seconds': 5, 'export_seconds': 300,
    'export_external_seconds': 330,
    'initial_free_disk_bytes': 8 * 1024**3,
    'continuing_free_disk_bytes': 4 * 1024**3,
    'maximum_RSS_bytes': 10 * 1024**3,
    'maximum_CUDA_reserved_bytes': 18 * 1024**3,
    'maximum_output_bytes': 2 * 1024**3,
    'maximum_archive_bytes': 2 * 1024**3,
    'automatic_retry_resume': 0,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def select_parent_cases(raw):
    need(type(raw) is bytes and len(raw) == PARENT_BYTES
         and hashlib.sha256(raw).hexdigest() == PARENT_SHA256,
         'Exact retained standardized parent protocol required')
    p = json.loads(raw)
    need(len(p['cases']) == 96 and len(p['guards']) == 24,
         'Keep exact 96 paired TRAIN rows and 24 historical guards')
    learning = p['learning']
    optimization = tuple(learning['optimization_reference_order'])
    monitors = tuple(learning['monitor_reference_order'])
    need(len(set(optimization)) == 16 and len(set(monitors)) == 8
         and not set(optimization) & set(monitors)
         and tuple(learning['slot_roles']) == ROLES,
         'Retained photograph partitions and four-role order required')
    by = {(r['reference_id'], r['role']): r for r in p['cases']}
    need(len(by) == 96 and all(r['split'] == 'TRAIN' for r in p['cases'] + p['guards']),
         'Unique TRAIN-only membership required')
    train = [by[(reference, role)] for reference in optimization for role in ROLES]
    monitor = [by[(reference, role)] for reference in monitors for role in ROLES]
    need(all(r['cohort'] == 'optimize' and r['optimization_permitted'] is True for r in train),
         'Use only prior admitted optimization rows')
    need(all(r['cohort'] == 'monitor' and r['optimization_permitted'] is False for r in monitor),
         'Adaptive TRAIN monitors remain non-optimization rows')
    for reference in optimization + monitors:
        need(len({json.dumps(by[(reference, role)]['target'], sort_keys=True) for role in ROLES}) == 1,
             'Four roles must use the same aligned photographic target')
    guard_refs = {r['reference_id'] for r in p['guards']}
    need(len(guard_refs) == 8 and not guard_refs & set(optimization + monitors),
         'Legacy guards stay photograph-disjoint and evaluation-only')
    need(all(r.get('target') is None for r in p['guards'] if r['role'] == 'dark'),
         'Legacy dark null targets stay null, never enter clean-anchor optimization')
    return copy.deepcopy({'optimization': train, 'monitors': monitor, 'guards': p['guards'],
        'optimization_reference_order': list(optimization), 'monitor_reference_order': list(monitors),
        'independent_holdout': False, 'photograph_separation_is_not_identity_or_pretraining_proof': True,
        'DEV_native_wild_FINAL_optimized': False})


def operation_plan(selection):
    train, monitor, guards = (selection[k] for k in ('optimization', 'monitors', 'guards'))
    need((len(train), len(monitor), len(guards)) == (64, 32, 24), 'Exact64/32/24 scope required')
    need(len({r['id'] for r in train + monitor + guards}) == 120, 'Unique retained case IDs required')
    by = {(r['reference_id'], r['role']): r for r in train}
    order = selection['optimization_reference_order']
    bank = monitor + guards
    result = []
    def append(row, phase, update, backward=False):
        result.append({'slot': len(result)+1, 'id': row['id'], 'phase': phase,
                       'new_update': update, 'backward': backward,
                       'optimizer_application_after_this_forward': backward and row['role'] == ROLES[-1]})
    for row in bank:
        append(row, 'initial_u64_quality_bank', 0)
    for row in monitor[:4]:
        append(row, 'deterministic_same_u64_repeat', 0)
    for update in range(1, 17):
        for role, offset in zip(ROLES, (0, 4, 8, 12)):
            append(by[(order[(update-1+offset) % 16], role)], 'paired_TRAIN_optimization', update, True)
        if update in SNAPSHOTS[1:]:
            for row in bank:
                append(row, 'new_quality_snapshot', update)
    need(len(result) == 292 and sum(row['backward'] for row in result) == 64
         and sum(row['optimizer_application_after_this_forward'] for row in result) == 16,
         'Exact finite call/backward/application plan differs')
    return result


def prospective_contract():
    return {
        'status': STATUS,
        'initializer': 'Exact R2(515) + U96 adapters(24) + retained trained u64 heads(36); no zero-head reset',
        'head_file': {'bytes': HEAD_BYTES, 'sha256': HEAD_FILE_SHA256},
        'head_canonical_sha256': HEAD_CANONICAL_SHA256,
        'total_tensors': 575, 'frozen_tensors': 539,
        'trainable_tensors': 36, 'trainable_scalars': 168960,
        'objective': formula_contract(), 'maximum_new_updates': 16,
        'snapshot_updates': list(SNAPSHOTS), 'microbatches_per_commit': 4,
        'correlated_optimization_rolecases': 64, 'rolecase_table_passes_maximum': 1,
        'optimizer': {'name': 'Adam', 'fresh_state_required': True, 'max_lr': 0.0001,
                      'betas': [0.9, 0.99], 'eps': 1e-8, 'weight_decay': 0.0,
                      'warmup_new_updates': 4, 'gradient_clip_max_norm': 1.0,
                      'foreach': False, 'fused': False, 'capturable': False,
                      'differentiable': False, 'amsgrad': False, 'maximize': False},
        'fresh_optimizer_reason': 'Changed supervision starts a new finite protocol; old terminal64 is closed and old moments retained, not resumed',
        'objective_only_causality_claimed': False,
        'initialization_parity': 'Initial new-candidate RAW/PNG must match four saved terminal-u64 monitor-role witnesses exactly',
        'deterministic_repeats': 'Four extra same-candidate forwards compare native/logits/LQ/RAW/PNG with its own initializer calls; not independent historical stock parity',
        'historical_stock_native_logits_LQ_parity_claimed': False,
        'cached_frozen_U96_controls': 56,
        'early_u4': 'Original legacy and ordinary TRAIN movement/behavior guards; retain all measured changes',
        'early_u8': 'Original guards plus paired clear MAE strictly lower and dark MAE nonincreasing against initial u64, in RAW and PNG',
        'terminal_u16': 'All original guards/1% endpoints plus each of four u64 structure gains retained, clear/dark MAE lower and local-center/global-whole blur MAE nonincreasing, in RAW and PNG',
        'same_original_references_and_quality_thresholds': True,
        'no_retrospective_checkpoint_selection': True, 'no_automatic_retry_resume': True,
        'state_retention': 'Full575 baseline and terminal payloads; every committed36-head/Adam/RNG/sampler state and exact frozen539 value bindings; retain partial failures without overwrite',
        'failure_extra_full_state_guaranteed': False,
        'limits': copy.deepcopy(LIMITS), 'model_qualified': False,
        'manual_finite_L4_worker_required': True, 'sealed_packet_or_launch_guide_ready': False,
    }


def retention_checks(initial_u64, current, frozen_U96, *, phase):
    """ADDITIONAL scalar gates only; inherited complete-bank checks remain required.

    Inputs are already-bound eight-photo cohort means at local central128 or
    other-role whole256. This arithmetic cannot establish cohort membership or
    pixel integrity. A future worker/auditor must bind and recompute both.
    """
    need(phase in ('initializer', 'u4', 'u8', 'u16'), 'Explicit prospective checkpoint required')
    for bank in (initial_u64, current, frozen_U96):
        need(type(bank) is dict and set(bank) == set(VIEWS), 'Both RAW/PNG views required')
        for view in VIEWS:
            need(set(bank[view]) == set(ROLES), 'All four roles required separately')
            for role in ROLES:
                values = bank[view][role]
                need(type(values) is dict and set(values) == set(FIELDS)
                     and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in values.values()),
                     'Exact finite nonnegative cohort means required')
    checks = {}
    for view in VIEWS:
        for role in ROLES[:2]:
            reference = frozen_U96[view][role]['adjacent_rgb_error']
            start = initial_u64[view][role]['adjacent_rgb_error']
            now = current[view][role]['adjacent_rgb_error']
            need(reference > 0, 'Positive frozen U96 structure denominator required')
            key = view + '/' + role
            if phase == 'initializer':
                checks[key + '/retained_original_one_percent'] = (reference-start)/reference >= 0.01
            if phase == 'u16':
                checks[key + '/retained_original_one_percent'] = (reference-now)/reference >= 0.01
                checks[key + '/retain_u64_structure'] = now <= start
                checks[key + '/blur_target_MAE_nonincrease'] = current[view][role]['mae'] <= initial_u64[view][role]['mae']
        if phase in ('u8', 'u16'):
            checks[view + '/clear_target_MAE_improves'] = current[view]['clear']['mae'] < initial_u64[view]['clear']['mae']
            now, start = current[view][ROLES[-1]]['mae'], initial_u64[view][ROLES[-1]]['mae']
            checks[view + '/dark_target_MAE'] = now < start if phase == 'u16' else now <= start
    return {'pass': all(checks.values()), 'phase': phase, 'checks': checks,
            'additional_gates_only': True, 'inherited_complete_bank_gates_required': True,
            'quality_tolerance_added': False, 'model_qualified': False}

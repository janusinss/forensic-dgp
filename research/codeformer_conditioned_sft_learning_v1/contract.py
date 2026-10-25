"""Inert Conditioned SFT V1 specification; no worker or launch entry point.

Exact admitted bytes determine the 120 optimization cases, 40 new monitor cases,
and 56 inherited quality cases. Both arms start from all 575 trained-u64 tensors
(Arm A: 36 spatial heads trainable; Arm B: 24 U96 pre-SFT adapters + 36 spatial
heads = 60 tensors trainable, 515 R2 frozen) with input-only degradation/exposure
conditioning, multi-site Clean-SFT encoder feature distillation at sites
32/64/128/256, and role-balanced conflict-projected gradient aggregation.
"""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path

from . import conditioning
from .objective import ROLES, formula_contract

STATUS = 'CONDITIONED_SFT_SOURCE_CONTRACT_NOT_A_SEALED_EXECUTION_PACKET'
STAGE = 'cctv_codeformer_conditioned_sft_learning_v1_vm'
OUTPUT = 'outputs/conditioned_sft_learning_v1'
PREFIX = 'cctv-codeformer-conditioned-sft-learning-v1'
ARMS = ('A', 'B')
SNAPSHOTS = (0, 8, 24, 48)
UPDATES_PER_ARM = 48
PHOTOGRAPH_PASSES = 2
ORDER_SEED = '20261025-conditioned-sft-learning-v1-photograph-order'
PREPARATION = 'outputs/codeformer_matched_feature_learning_v1_preparation/'
PINS = {
    'admission': (9653, '888b191bf6639760e324f2a7d77885fd6b9d13e439a6bbf64251372513fc89b0'),
    'candidate': (323833, '23c402f82c9dea0b52f32a828446b5f961c52d249a7c9881e260459e0147e9ce'),
    'parent': (197079, 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'),
}
PARENT_CONTRACT = {
    'path': 'research/codeformer_u64_cleananchors_continuation_v1/contract.py',
    'bytes': 10844,
    'sha256': '99e1bbef911750c5ef0622d350e203bcafed58559b3c2a2c500744ef76102bb0',
}
PARENT_OBJECTIVE = {
    'path': 'research/codeformer_u64_cleananchors_continuation_v1/objective.py',
    'bytes': 8205,
    'sha256': 'cffafed3feae10ce91890cfd68ee5de96c8db8edad00041420615cd071f2b7ad',
}
OPTIMIZATION_REFERENCES = (
    'tr_ffhq_10113', 'tr_ffhq_01210', 'tr_ffhq_12002', 'tr_ffhq_04476',
    'tr_ffhq_03089', 'tr_ffhq_05302', 'tr_ffhq_06809', 'tr_ffhq_06995',
    'tr_ffhq_04436', 'tr_ffhq_11549', 'tr_ffhq_06781', 'tr_ffhq_01639',
    'tr_ffhq_04759', 'tr_ffhq_03212', 'tr_ffhq_09531', 'tr_ffhq_07536',
    'tr_ffhq_07778', 'tr_ffhq_02421', 'tr_ffhq_01672', 'tr_ffhq_03483',
    'tr_ffhq_07430', 'tr_ffhq_07386', 'tr_ffhq_11684', 'tr_ffhq_02665',
)
MONITOR_REFERENCES = (
    'tr_ffhq_08698', 'tr_ffhq_02147', 'tr_ffhq_05179', 'tr_ffhq_08949',
    'tr_ffhq_11039', 'tr_ffhq_11154', 'tr_ffhq_08064', 'tr_ffhq_08094',
)
LIMITS = {
    'full_forward_attempts': 1736, 'full_forward_returns': 1736,
    'training_forwards': 480, 'initializer_forwards': 432,
    'deterministic_repeat_forwards': 8, 'later_quality_forwards': 576,
    'terminal_optimization_readout_forwards': 240,
    'backward_attempts': 480, 'backward_returns': 480,
    'gradient_queries': 0, 'optimizer_constructions': 2,
    'optimizer_update_attempts': 96, 'optimizer_updates': 96,
    'worker_seconds': 5400, 'external_seconds': 5460,
    'per_arm_seconds': 2700, 'per_phase_seconds': 300,
    'termination_grace_seconds': 20, 'failure_capture_seconds': 45,
    'failure_receipt_seconds': 5, 'export_seconds': 600,
    'export_external_seconds': 660,
    'initial_free_disk_bytes': 32 * 1024**3,
    'continuing_free_disk_bytes': 4 * 1024**3,
    'maximum_RSS_bytes': 10 * 1024**3,
    'maximum_CUDA_reserved_bytes': 18 * 1024**3,
    'maximum_output_bytes': 12 * 1024**3,
    'maximum_archive_bytes': 12 * 1024**3,
    'automatic_retry_resume': 0,
}
OPTIMIZER = {
    'name': 'Adam', 'fresh_state_required': True,
    'maximum_learning_rate': 1e-5, 'warmup_updates': 4,
    'betas': [0.9, 0.99], 'eps': 1e-8, 'weight_decay': 0.0,
    'gradient_clip_max_norm': 1.0,
    'foreach': False, 'fused': False, 'capturable': False,
    'differentiable': False, 'amsgrad': False, 'maximize': False,
    'EMA': False,
    'maximum_actual_step_Linf': 1e-4,
    'maximum_actual_cumulative_Linf_from_initializer': 1e-3,
    'role_conflict_gradient_projection': True,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _pinned_json(raw, label):
    size, digest = PINS[label]
    need(type(raw) is bytes and len(raw) == size and
         hashlib.sha256(raw).hexdigest() == digest,
         'Exact retained ' + label + ' bytes required before decoding')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key refused')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON token refused: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    need(type(value) is dict, 'Protocol object required')
    return value


def _old_selection(parent_raw):
    root = Path(__file__).resolve().parents[2]
    for pin in (PARENT_CONTRACT, PARENT_OBJECTIVE):
        source = root / pin['path']
        need(source.resolve(strict=True) == source.absolute(), 'Unlinked inherited source required')
        raw = source.read_bytes()
        need(len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256'],
             'Exact inherited pure selection source required')
    source = root / PARENT_CONTRACT['path']
    module = importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.contract')
    need(Path(module.__file__).resolve() == source, 'Inherited selection origin differs')
    objective = importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.objective')
    need(Path(objective.__file__).resolve() == root / PARENT_OBJECTIVE['path'],
         'Inherited objective origin differs')
    return module.select_parent_cases(parent_raw)


def _asset(item, expected, staged):
    need(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'} and
         item['path'] == expected and type(item['bytes']) is int and item['bytes'] > 0 and
         type(item['sha256']) is str and len(item['sha256']) == 64 and
         all(c in '0123456789abcdef' for c in item['sha256']),
         'Exact bounded prepared asset pin required')
    return {'path': staged, 'bytes': item['bytes'], 'sha256': item['sha256']}


def select_cases(admission_raw, candidate_raw, parent_raw):
    """Bind prepared bytes and normalize the admitted 120/40/56 cohorts."""
    admission = _pinned_json(admission_raw, 'admission')
    candidate = _pinned_json(candidate_raw, 'candidate')
    _pinned_json(parent_raw, 'parent')
    old = _old_selection(parent_raw)
    need(admission['complete'] is True and admission['new_optimization_data_admission'] is True
         and admission['monitor_optimization_permitted'] is False
         and admission['prior96_dataset_or_splits_changed'] is False
         and admission['all160_inputs_root_reviewed'] is True,
         'Exact reviewed data admission required')
    need(candidate['selection']['sha256'] == admission['selection']['sha256'] ==
         '1cd22c1fea890527186b5f7a81e5f21e4aaa8f6b9be9c0f579d9611ff09a7ea0',
         'Exact selected photograph task required')
    need(tuple(old['optimization_reference_order']) == OPTIMIZATION_REFERENCES[:16]
         and tuple(old['monitor_reference_order']) == MONITOR_REFERENCES,
         'Original16 optimization/eight adaptive monitor partitions required')
    optimize_ids = [reference + '__' + role for reference in OPTIMIZATION_REFERENCES for role in ROLES]
    monitor_ids = [reference + '__' + role for reference in MONITOR_REFERENCES for role in ROLES]
    need(admission['optimization_case_ids'] == optimize_ids and
         admission['adaptive_monitor_case_ids'] == monitor_ids,
         'Exact120 optimization/40 monitor admission order required')
    rows = candidate['cases']
    need(type(rows) is list and len(rows) == 160 and
         [row['id'] for row in rows] == optimize_ids + monitor_ids,
         'Exact32-photograph five-role prepared bank required')
    new = []
    for row in rows:
        reference, role = row['reference_id'], row['role']
        optimize = reference in OPTIMIZATION_REFERENCES
        need(row['id'] == reference + '__' + role and role in ROLES and
             row['historical_split'] == 'TRAIN' and row['native_CCTV'] is False and
             row['paired_synthetic'] is True and row['optimization_permitted'] is False and
             row['input_only_decision'] == 'PENDING' and
             row['cohort'] == ('optimize' if optimize else 'adaptive_monitor'),
             'Historical candidate metadata must stay intact; admission is separate')
        item = {
            'id': row['id'], 'reference_id': reference, 'role': role, 'split': 'TRAIN',
            'cohort': 'optimize' if optimize else 'monitor',
            'optimization_permitted': optimize,
            'input': _asset(row['input'], PREPARATION + 'inputs/' + row['id'] + '.png',
                            'data/inputs/' + row['id'] + '.png'),
            'target': _asset(row['target'], PREPARATION + 'targets/' + reference + '.png',
                             'data/targets/' + reference + '.png'),
            'prepared_input_RGB8_sha256': row['input_RGB8_sha256'],
            'prepared_target_RGB8_sha256': row['target_RGB8_sha256'],
            'historical_candidate_workspace_input': copy.deepcopy(row['input']),
            'historical_candidate_workspace_target': copy.deepcopy(row['target']),
            'synthetic_corruption_mask_is_app_preparation': False,
        }
        if role == 'clear':
            need(item['input']['sha256'] == item['target']['sha256'] and
                 item['prepared_input_RGB8_sha256'] == item['prepared_target_RGB8_sha256'],
                 'Clear observation stays exact photographic target')
        new.append(item)
    guard_refs = {row['reference_id'] for row in old['guards']}
    need(not guard_refs & set(OPTIMIZATION_REFERENCES + MONITOR_REFERENCES),
         'Legacy guard photographs stay disjoint from new32')
    result = {
        'source_pins': {key: {'bytes': value[0], 'sha256': value[1]} for key, value in PINS.items()},
        'new_optimization': new[:120], 'new_monitors': new[120:],
        'old_monitors': old['monitors'], 'old_guards': old['guards'],
        'old_quality': old['monitors'] + old['guards'],
        'optimization_reference_order': list(OPTIMIZATION_REFERENCES),
        'monitor_reference_order': list(MONITOR_REFERENCES),
        'independent_holdout': False, 'DEV_native_wild_FINAL_optimized': False,
        'photograph_separation_is_not_identity_or_pretraining_proof': True,
    }
    _selection(result)
    return copy.deepcopy(result)


def _selection(selection):
    need(type(selection) is dict and selection.get('source_pins') ==
         {key: {'bytes': value[0], 'sha256': value[1]} for key, value in PINS.items()},
         'Selection must be rederived from pinned protocol bytes')
    need(tuple(selection['optimization_reference_order']) == OPTIMIZATION_REFERENCES and
         tuple(selection['monitor_reference_order']) == MONITOR_REFERENCES,
         'Exact fixed photograph orders required')
    for key, refs, cohort, allowed in (
            ('new_optimization', OPTIMIZATION_REFERENCES, 'optimize', True),
            ('new_monitors', MONITOR_REFERENCES, 'monitor', False)):
        rows = selection[key]
        need(type(rows) is list and len(rows) == len(refs) * len(ROLES) and
             [row['id'] for row in rows] == [r + '__' + role for r in refs for role in ROLES] and
             all(row['id'] == row['reference_id'] + '__' + row['role'] and
                 row['split'] == 'TRAIN' and row['cohort'] == cohort and
                 row['optimization_permitted'] is allowed for row in rows),
             'Exact admitted optimize/evaluation-only cohorts required')
    old = selection['old_quality']
    need(len(old) == 56 and len(selection['old_monitors']) == 32 and
         len(selection['old_guards']) == 24 and old == selection['old_monitors'] + selection['old_guards'] and
         len({row['id'] for row in old}) == 56 and
         all(row['split'] == 'TRAIN' and row.get('optimization_permitted') is not True for row in old),
         'Unchanged56 TRAIN quality rows required')
    return selection


def photograph_order(epoch):
    need(type(epoch) is int and epoch in (1, 2), 'Only two declared photograph passes required')
    def rank(reference):
        text = ORDER_SEED + ':' + str(epoch) + ':' + reference
        return hashlib.sha256(text.encode('ascii')).hexdigest(), reference
    return tuple(sorted(OPTIMIZATION_REFERENCES, key=rank))


def learning_rate(update):
    need(type(update) is int and 1 <= update <= UPDATES_PER_ARM, 'Finite new update1..48 required')
    return OPTIMIZER['maximum_learning_rate'] * min(update / OPTIMIZER['warmup_updates'], 1.0)


def operation_plan(selection):
    """Exact maximum forward order; stops retain a declared ordered subsequence."""
    _selection(selection)
    result = []
    quality = ([('old_quality', row) for row in selection['old_quality']] +
               [('new_monitors', row) for row in selection['new_monitors']])
    training = selection['new_optimization']
    by = {(row['reference_id'], row['role']): row for row in training}
    def append(arm, row, phase, update, *, bank, backward=False, epoch=None):
        result.append({'slot': len(result) + 1, 'arm': arm, 'id': row['id'],
                       'bank': bank, 'case_key': bank + ':' + row['id'],
                       'phase': phase, 'new_update': update, 'epoch': epoch,
                       'backward': backward,
                       'optimizer_application_after_this_forward': backward and row['role'] == ROLES[-1]})
    for arm in ARMS:
        for bank, row in quality:
            append(arm, row, 'initial_quality', 0, bank=bank)
        for row in training:
            append(arm, row, 'initial_optimization_readout', 0, bank='new_optimization')
        for row in selection['old_monitors'][:4]:
            append(arm, row, 'same_state_deterministic_repeat', 0, bank='old_quality')
    for arm in ARMS:
        for epoch in (1, 2):
            for position, reference in enumerate(photograph_order(epoch), 1):
                update = (epoch - 1) * 24 + position
                for role in ROLES:
                    append(arm, by[(reference, role)], 'photographic_optimization', update,
                           bank='new_optimization', backward=True, epoch=epoch)
                if update in SNAPSHOTS[1:]:
                    for bank, row in quality:
                        append(arm, row, 'quality_snapshot', update, bank=bank)
        for row in training:
            append(arm, row, 'terminal_optimization_readout', 48, bank='new_optimization')
    need(len(result) == LIMITS['full_forward_attempts'] and
         sum(row['backward'] for row in result) == LIMITS['backward_attempts'] and
         sum(row['optimizer_application_after_this_forward'] for row in result) == LIMITS['optimizer_updates'],
         'Exact1736 forwards/480 backwards/96 commits differ')
    return result


def resolve_case(selection, operation):
    """Resolve a bank-bound call without merging historical/new clear IDs."""
    _selection(selection)
    need(type(operation) is dict and operation.get('bank') in
         ('old_quality', 'new_monitors', 'new_optimization') and
         operation.get('case_key') == operation['bank'] + ':' + operation.get('id', ''),
         'Explicit scope-qualified call key required')
    bank = selection[operation['bank']]
    matches = [row for row in bank if row['id'] == operation['id']]
    need(len(matches) == 1, 'One exact row in the declared observation bank required')
    if operation.get('backward'):
        need(operation['bank'] == 'new_optimization' and matches[0]['optimization_permitted'] is True,
             'Only admitted new optimization rows may backpropagate')
    return copy.deepcopy(matches[0])


def displacement_checks(before, after, initial):
    """Pure actual FP32-value scalar check; never project or apply a proposal."""
    need(type(before) is list and type(after) is list and type(initial) is list and
         0 < len(before) == len(after) == len(initial), 'Same nonempty bound coordinate scope required')
    for bank in (before, after, initial):
        need(all(type(v) in (int, float) and math.isfinite(v) for v in bank),
             'Finite actual parameter values required')
    step = max(abs(a - b) for a, b in zip(after, before))
    cumulative = max(abs(a - b) for a, b in zip(after, initial))
    return {'actual_step_Linf': step, 'actual_cumulative_Linf': cumulative,
            'pass': 0 < step <= OPTIMIZER['maximum_actual_step_Linf'] and
                    0 < cumulative <= OPTIMIZER['maximum_actual_cumulative_Linf_from_initializer']}


def prospective_contract():
    return {
        'status': STATUS, 'arms': list(ARMS), 'snapshots': list(SNAPSHOTS),
        'initializer': 'Same complete original575 trained-u64 tensors; no head or adapter reset',
        'arm_A': {'trainable_tensors': 36, 'trainable_coordinates': 168960, 'frozen_tensors': 539},
        'arm_B': {'trainable_tensors': 60, 'trainable_coordinates': 220288, 'frozen_tensors': 515},
        'conditioning': conditioning.conditioning_contract(),
        'trainable_contrast': 'Only B adds gradients through24 existing U96 adapters + Clean-SFT distillation; same input-conditioned forward equations',
        'coarse32_context': 'Detached for both; B used encoder stays live through stock SFT, head.encoder, and multi-site Clean-SFT distillation',
        'same_task_and_sampler_for_both_arms': True, 'role_order': list(ROLES),
        'new_updates_per_arm': 48, 'photograph_passes_per_arm': 2,
        'training_exposures_per_arm': 240, 'singleton_microbatches_per_commit': 5,
        'objective': formula_contract(), 'external_microbatch_divisor_exactly_once': 5,
        'optimizer': copy.deepcopy(OPTIMIZER),
        'initial_parity': 'Both arms exactly match each other and u64 at update 0 (gamma=0); four saved original-u64 RAW/PNG witnesses and exact old56 recorded metrics bind',
        'repeat_parity': 'Four self-repeats per arm compare native/logits/LQ/RAW/PNG',
        'quality': {'0': 'Original initializer56; supplemental40 payload exact',
                    '8': 'Inherited u8 strict clear MAE/dark nonincrease plus40 rolewise MAE nonincrease',
                    '24': 'Inherited terminal all RAW/PNG1%/u64retention/preservation; extra40 MAE/adjacent/coarse nonincrease',
                    '48': 'Same terminal rules; no lower checkpoint rescue'},
        'limits': copy.deepcopy(LIMITS),
        'two_arm_result_is_quality_qualification': False,
        'broader_DEV_and_independent_FINAL_still_required': True,
        'automatic_preparation_or_completion_development': False,
        'new_model_or_gradient_optimizer_training_VM_calls': 0,
        'worker_transfer_and_launch_guide_ready': False, 'model_qualified': False,
    }

"""Prospective finite contract only: no CLI, I/O, framework or execution path."""
import copy
import hashlib
import json

from .objective import ROLES, TASK_NAMES


STATUS = 'INERT_PROTOTYPE_UNSEALED_UNRUN_NOT_A_TRAINED_MODEL'
PARENT_PROTOCOL_BYTES = 197079
PARENT_PROTOCOL_SHA256 = 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
OPTIMIZATION_REFERENCES = ('tr_ffhq_10113', 'tr_ffhq_01210', 'tr_ffhq_12002', 'tr_ffhq_04476')
MONITOR_REFERENCES = ('tr_ffhq_08698', 'tr_ffhq_02147', 'tr_ffhq_05179', 'tr_ffhq_08949',
                      'tr_ffhq_11039', 'tr_ffhq_11154', 'tr_ffhq_08064', 'tr_ffhq_08094')
STEP_MAXABS = (1e-5, 5e-5)
PARITY_IDS = ('tr_ffhq_10113__global_gaussian_1p2',
              'tr_ffhq_10113__lowlight_linear_0p35')
LIMITS = {
    'full_forward_attempts': 218, 'full_forward_returns': 218,
    'common_initializer_component_forwards': 16, 'gradient_queries': 32,
    'initial_monitor_guard_forwards': 56, 'independent_parity_forwards': 2,
    'proposal_forwards': 144, 'proposal_applications': 2,
    'optimizer_constructions': 0, 'optimizer_steps': 0,
    'automatic_retry_resume': 0, 'worker_seconds': 1200, 'external_seconds': 1260,
    'termination_grace_seconds': 20, 'per_phase_seconds': 180,
    'export_seconds': 300, 'export_external_seconds': 330,
    'initial_free_disk_bytes': 8 * 1024**3, 'continuing_free_disk_bytes': 4 * 1024**3,
    'maximum_RSS_bytes': 10 * 1024**3, 'maximum_CUDA_reserved_bytes': 18 * 1024**3,
    'maximum_output_bytes': 2 * 1024**3, 'maximum_archive_bytes': 2 * 1024**3,
    'complete_model_state_witnesses': 3,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def select_parent_cases(parent_bytes):
    """Read the exact prior seal; select cohorts without consulting new results.

    All 96 parent cases remain retained. Only the first four already ordered
    optimization photographs enter this diagnostic's gradients. The old eight
    monitor photographs are adaptive TRAIN, never independent validation.
    """
    need(type(parent_bytes) is bytes and len(parent_bytes) == PARENT_PROTOCOL_BYTES and
         hashlib.sha256(parent_bytes).hexdigest() == PARENT_PROTOCOL_SHA256,
         'Exact immutable parent protocol required')
    parent = json.loads(parent_bytes)
    learning = parent['learning']
    need(tuple(learning['optimization_reference_order'][:4]) == OPTIMIZATION_REFERENCES and
         tuple(learning['monitor_reference_order']) == MONITOR_REFERENCES and
         tuple(learning['slot_roles']) == ROLES, 'Parent selection/order differs')
    rows = parent['cases']
    need(len(rows) == 96 and len({row['id'] for row in rows}) == 96, 'Exact96 unique paired cases required')
    lookup = {(row['reference_id'], row['role']): row for row in rows}
    need(len(lookup) == 96 and all(row['split'] == 'TRAIN' for row in rows), 'TRAIN role table differs')
    optimization = [lookup[(ref, role)] for ref in OPTIMIZATION_REFERENCES for role in ROLES]
    monitors = [lookup[(ref, role)] for ref in MONITOR_REFERENCES for role in ROLES]
    need(all(row['cohort'] == 'optimize' and row['optimization_permitted'] is True
             for row in optimization), 'Selected photograph lacks prior optimization permission')
    need(all(row['cohort'] == 'monitor' and row['optimization_permitted'] is False
             for row in monitors), 'Monitor permission differs')
    guards = parent['guards']
    need(len(guards) == 24 and len({row['id'] for row in guards}) == 24 and
         all(row['split'] == 'TRAIN' for row in guards), 'Exact24 TRAIN legacy guards required')
    opt_refs = set(OPTIMIZATION_REFERENCES)
    mon_refs = set(MONITOR_REFERENCES)
    guard_refs = {row['reference_id'] for row in guards}
    need(len(guard_refs) == 8 and not opt_refs & mon_refs and not opt_refs & guard_refs and
         not mon_refs & guard_refs, 'Photograph cohorts overlap')
    for ref in OPTIMIZATION_REFERENCES + MONITOR_REFERENCES:
        need(len({json.dumps(lookup[(ref, role)]['target'], sort_keys=True) for role in ROLES}) == 1,
             'All four roles require the same aligned photographic target')
    return {'optimization': copy.deepcopy(optimization), 'monitors': copy.deepcopy(monitors),
            'guards': copy.deepcopy(guards), 'parent_case_count': 96,
            'new_optimization_case_count': 16, 'adaptive_monitor_case_count': 32,
            'independent_holdout': False, 'FINAL_or_native_pixels_used': False,
            'inherited_quality_policy': copy.deepcopy(parent['quality_policy'])}


def operation_plan(selection):
    """Prospective ordered calls; legacy guard roles never become gradient tasks."""
    optimization = selection['optimization']
    monitors = selection['monitors']
    guards = selection['guards']
    need(len(optimization) == 16 and len(monitors) == 32 and len(guards) == 24,
         'Exact16/32/24 bank required')
    core = ([{'id': row['id'], 'kind': 'optimization', 'role': row['role']} for row in optimization] +
            [{'id': row['id'], 'kind': 'adaptive_TRAIN_monitor', 'role': row['role']} for row in monitors] +
            [{'id': row['id'], 'kind': 'legacy_guard', 'role': row['role']} for row in guards])
    need(len({row['id'] for row in core}) == 72 and
         set(PARITY_IDS) <= {row['id'] for row in optimization}, 'Bank IDs/parity differ')
    calls = []
    for state in ('initializer', 'proposal_A', 'proposal_B'):
        for row in core:
            calls.append({'slot': len(calls) + 1, 'state': state, **row,
                          'gradient_components': ['rgb', 'adjacent']
                          if state == 'initializer' and row['kind'] == 'optimization' else []})
        if state == 'initializer':
            for case_id in PARITY_IDS:
                calls.append({'slot': len(calls) + 1, 'state': 'independent_frozen_U96_parity',
                              'id': case_id, 'kind': 'parity', 'gradient_components': []})
    need(len(calls) == 218 and sum(len(row['gradient_components']) for row in calls) == 32,
         'Forward/query census differs')
    return calls


def permit_next_proposal(*, attempted, initial_checks_passed, direction_certified,
                         actual_FP32_direction_passed, prior_proposal_passed):
    """Pure fail-closed control decision; not a worker or scientific metric gate."""
    need(type(attempted) is int and attempted in (0, 1, 2), 'At most two applications')
    flags = (initial_checks_passed, direction_certified, actual_FP32_direction_passed,
             prior_proposal_passed)
    need(all(type(value) is bool for value in flags), 'Explicit known boolean outcomes required')
    return attempted < 2 and all(flags)


def prospective_contract():
    """A design contract cannot be mistaken for an executable/sealed packet."""
    assert 16 + 56 + 2 + 144 == LIMITS['full_forward_attempts'] == 218
    assert 16 * 2 == LIMITS['gradient_queries']
    return {
        'status': STATUS, 'manual_L4_only': True, 'runtime_observed': False,
        'initializer': 'Exact complete retained U96 from R2 plus24 adapters; not failed readout-u64.',
        'trainable': 'Existing generator final ResBlock22 and RGBConv24 only:10 tensors/75843 scalars.',
        'frozen_tensors': 529, 'full_model_tensors': 539,
        'optimization_reference_order': list(OPTIMIZATION_REFERENCES),
        'adaptive_TRAIN_monitor_reference_order': list(MONITOR_REFERENCES),
        'roles': list(ROLES), 'component_tasks': list(TASK_NAMES),
        'selection': 'First four entries of the exact previously admitted optimization order; fixed before new results.',
        'gradient_graphs': 'One same-initializer forward per optimization case; two separate RGB/adjacent queries.',
        'direction': 'Certified common-descent proposal from eight unit-normalized mean component gradients; two-tensor readout is analytical subspace control.',
        'steps': {'maximum_absolute_FP32_displacements': list(STEP_MAXABS),
                  'independent_from_same_initializer': True, 'cumulative_training': False,
                  'momentum': 0, 'adaptive_preconditioning': False,
                  'actual_FP32_displacement_derivatives_required': True},
        'parity_case_ids': list(PARITY_IDS),
        'early_stop': 'Any initial/parity/resource/certificate/FP32-direction failure stops; proposalA quality failure stops beforeB. No retries or added epochs.',
        'joint_acceptance': 'BOTH predeclared proposals must pass; no best-of-two or selective promotion.',
        'legacy_guard_routing': 'Keep inherited filter/clear/dark role semantics and null dark clean targets. No guard gradients or substitution by four paired roles.',
        'actual_feasibility_requirements': [
            'Global whole and local central128 adjacent means strictly lower in RAW and PNG.',
            'Dark mean MAE and adjacent error do not increase in RAW or PNG.',
            'Every inherited legacy guard and ordinary monitor movement guard passes unchanged.',
            'Every complete frozen529 value matches initialization in both full trial state witnesses.',
            'Passing is TRAIN feasibility only; full four1%endpoints, broaderDEV, human acceptance and FINAL remain binding.'
        ],
        'complete_full_state_exports': 'All539 values at initializer and after each applied proposal; partial/failure states retained.',
        'call_accounting': '72 initial +72 proposalA +72 proposalB +2 independent parity =218 maximum; stops can yield fewer.',
        'extra_neural_hook_probes': 0,
        'limits': copy.deepcopy(LIMITS),
        'estimated_worker_minutes': [12, 20],
        'estimate_basis': 'Prior712call worker24.6min; fewer calls but new block gradients and state exports are unmeasured.',
        'worker_supervisor_exporter_installer_builder_and_guide_ready': False,
        'model_qualified': False, 'app_or_checkpoint_promotion': False,
        'new_training_or_VM_executed': False,
    }

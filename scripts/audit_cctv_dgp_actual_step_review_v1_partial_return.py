"""Independently audit the 29 complete conditions in the stopped VM return.

This never changes the original prospective auditor, failure, protocol, model
checkpoint or gate. The incomplete thirtieth condition stays incomplete.
"""
from pathlib import Path
import ast
import time
import traceback
from cctv_dgp_actual_step_review_v1_contract import NAME, read, write, sha, verify

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs' / NAME
RETURN = ROOT / 'outputs' / (NAME + '_return')
OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit'
CAP = 2400


def main():
    start = time.monotonic()
    scope_path = OUT / 'prospective_partial_scope.json'
    scope = read(scope_path)
    receipt_path = OUT / 'independent_audit.json'
    assert not receipt_path.exists() and not (OUT / 'failure.json').exists(), 'Retain every audit attempt'
    p = read(BUNDLE / 'protocol.json')
    pin = sha(BUNDLE / 'protocol.json')
    assert pin == scope['protocol_sha256'] and not scope['full_3150_review_complete']
    assert scope['conditions_count'] == 29 and scope['raw_PNG_mean_only_slots'] == 3045
    assert scope['CPU_current_batch_replays'] == 145 and scope['independent_audit_cap_seconds'] == CAP
    assert scope['original_prospective_tolerances'] == p['prospective_audit']
    assert scope['original_categorical_decisions_and_capacity_gates_unchanged']
    for name, digest in scope['source_bindings'].items():
        assert sha(ROOT / name) == digest, name
    original_path = ROOT / 'scripts/audit_cctv_dgp_actual_step_review_v1_return.py'
    assert sha(original_path) == p['source_evidence_sha256'][original_path.relative_to(ROOT).as_posix()]
    original = original_path.read_text(encoding='utf-8')
    generated_path = ROOT / 'scripts/cctv_dgp_actual_step_review_v1_partial_checks.py'
    generated = generated_path.read_text(encoding='utf-8')
    source_tree = ast.parse(original)
    original_cpu = ast.get_source_segment(original, next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'CPU_replay'))
    partial_tree = ast.parse(generated)
    generated_cpu = ast.get_source_segment(generated, next(n for n in partial_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'CPU_partial_replay'))
    block_begin = original.index('    sys.path.insert(0,str(BUNDLE))\n', original.index('def main():'))
    block_end = original.index('    replay=CPU_replay(p,start)', block_begin)
    generated_begin = generated.index('    sys.path.insert(0,str(BUNDLE))\n', generated.index('def scientific_partial('))
    generated_end = generated.index('    return term_errors,condition_list,checked\n', generated_begin)
    for altered, expected, changes in [
        (generated[generated_begin:generated_end], original[block_begin:block_end], scope['source_changes']['scientific_block']),
        (generated_cpu, original_cpu, scope['source_changes']['CPU_replay']),
    ]:
        for change in reversed(changes):
            assert altered.count(change['after']) == 1
            altered = altered.replace(change['after'], change['before'], 1)
        assert altered == expected
    verify(BUNDLE, pin)
    imported = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json')
    assert imported['complete'] and not imported['returned_code_executed']
    assert {path.relative_to(RETURN).as_posix() for path in RETURN.rglob('*') if path.is_file()} == set(imported['files_sha256'])
    for index, (name, digest) in enumerate(imported['files_sha256'].items(), 1):
        assert time.monotonic() - start < CAP
        assert sha(RETURN / name) == digest, name
        if index % 2000 == 0:
            print({'partial_import_files_reverified': index, 'of': imported['members']}, flush=True)
    assert sha(RETURN / 'protocol.json') == pin and not (RETURN / 'outputs/results.json').exists()
    failure = read(RETURN / 'outputs/failure.json')
    supervisor = read(RETURN / 'supervisor_receipt.json')
    manifest = read(RETURN / 'export_manifest.json')
    assert not failure['complete'] and failure['error'] == 'Return-storage limit; retain partial review'
    assert failure['restoration'] == {'attempted': True, 'decoder_restored': True}
    assert not failure['app_promotion'] and not failure['new_trained_checkpoint']
    progress = failure['progress']
    assert progress['optimizer_updates'] == progress['gradient_queries'] == progress['backward_calls'] == 0
    assert not progress['optimizer_constructed'] and not progress['new_trained_checkpoint']
    assert progress['completed_conditions'] == 29 and progress['forward_slots'] == 3110
    assert progress['parameter_proposals_loaded'] == 30
    completed = {(row['update'], row['proposal']) for row in scope['conditions']}
    assert [(row['update'], row['proposal']) for row in failure['completed_conditions']] == [(row['update'], row['proposal']) for row in scope['conditions']]
    assert not manifest['diagnostic_completed'] and not supervisor['complete'] and supervisor['review_exit_code'] == 1
    assert not supervisor['automatic_training_follow_on'] and not supervisor['app_promotion']
    assert len(supervisor['calls']) == 1 and supervisor['calls'][0]['flag'] == '--review' and supervisor['calls'][0]['exit_code'] == 1
    assert not supervisor['calls'][0]['timeout'] and supervisor['calls'][0]['external_cap_seconds'] == p['budgets']['worker_seconds'] + 30
    preflight = read(RETURN / 'outputs/preflight.json')
    cache = read(RETURN / 'outputs/cache_receipt.json')
    timing = read(RETURN / 'outputs/timing_projection.json')
    assert preflight['complete'] and preflight['states'] == p['initial_states'] and preflight['initial_exact_case_parity'] == 145
    assert 'L4' in preflight['gpu'] and preflight['grad_enabled'] == False
    assert preflight['optimizer_updates'] == preflight['gradient_queries'] == 0 and not preflight['new_trained_checkpoint']
    assert preflight['source_forward_counts'] == {'original_DGP': 58, 'decoder': 29, 'reference_decoder': 29, 'recognizer': 29}
    assert cache['complete'] and cache['cases'] == 145 and cache['references'] == 29 and cache['original_reference_and_recognizer_frozen']
    assert 0 < cache['seconds'] <= cache['cap_seconds'] == p['budgets']['cache_seconds'] and cache['optimizer_updates'] == 0
    assert timing['completed_probe_states'] == 1 and timing['remaining_probe_states'] == 9 and timing['completed_forward_slots'] == 315
    assert timing['cap_seconds'] == p['budgets']['review_seconds'] and timing['safety_factor'] == p['budgets']['timing_safety_factor']
    assert abs(timing['projected_review_seconds'] - timing['seconds'] * 10 * timing['safety_factor']) <= 1e-9
    assert 0 < timing['projected_review_seconds'] <= timing['cap_seconds']
    from cctv_dgp_actual_step_review_v1_partial_checks import scientific_partial, CPU_partial_replay
    errors, conditions, checked = scientific_partial(p, failure, start, completed)
    replay = CPU_partial_replay(p, start, completed)
    assert checked == 3045 and len(conditions) == 29 and replay['cases'] == 145
    assert time.monotonic() - start < CAP
    write(receipt_path, {
        'complete': True, 'audit_scope': 'All29 fully completed conditions;3045 raw/PNG/mean-only slots and145 frozen CPU current-batch replays',
        'full_3150_review_complete': False, 'original_storage_failure_retained': True,
        'protocol_sha256': pin, 'archive_sha256': imported['archive_sha256'],
        'members_verified': imported['members'], 'source_inverse_verified': True,
        'conditions_count': 29, 'all3045_raw_PNG_and_mean_only_outputs_checked': True,
        'raw_objective_maximum_errors': errors.tolist(), 'CPU_replay': replay, 'conditions': conditions,
        'excluded_unfinished_condition': scope['excluded_unfinished_condition'],
        'unfinished_files_retained_without_scientific_qualification': True,
        'categorical_decisions_exact': True, 'original_tolerances_unchanged': True,
        'prospective_partial_scope_sha256': sha(scope_path), 'checker_sha256': sha(Path(__file__)),
        'generated_checks_sha256': sha(generated_path), 'original_checker_sha256': sha(original_path),
        'paired_photographic_TRAIN_diagnostic_only': True, 'full_TRAIN_capacity_pass': False,
        'native_or_DEV_or_reserved_final_used': False, 'returned_code_executed': False,
        'optimizer_updates': 0, 'gradient_queries': 0, 'backward_calls': 0,
        'new_trained_checkpoint': False, 'VM_calls': 0, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - start, 'cap_seconds': CAP,
    })
    print({'partial_scientific_audit_complete': True, 'conditions': 29, 'slots': 3045, 'CPU_replay_cases': 145, 'full_3150_complete': False, 'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists() and not (OUT / 'failure.json').exists():
            write(OUT / 'failure.json', {'complete': False, 'type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(), 'partial_scientific_audit_complete': False,
                'original_storage_failure_retained': True, 'new_trained_checkpoint': False, 'app_promotion': False,
                'optimizer_updates': 0, 'gradient_queries': 0, 'checker_sha256': sha(Path(__file__))})
        raise

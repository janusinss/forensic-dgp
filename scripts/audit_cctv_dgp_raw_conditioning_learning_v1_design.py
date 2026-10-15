"""Independently verify the optimizer change and all unchanged output requirements."""
import ast

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha


def audit(packet, p):
    assert packet.name == 'cctv_dgp_raw_conditioning_learning_v1_vm'
    old = read(RETURN / 'protocol.json')
    prepared = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_preparation.json')
    source = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_source_preparation_r1.json')
    assert prepared['complete'] and source['complete']
    assert prepared['source_preparation_sha256'] == sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_source_preparation_r1.json')
    assert sha(RETURN / 'protocol.json') == PIN == p['comparison_parent_protocol_sha256']
    assert all(sha(RETURN / name) == digest for name, digest in old['assets_sha256'].items())
    assert all(sha(packet / name) == digest for name, digest in p['assets_sha256'].items())
    for name in prepared['inherited_scientific_and_state_resource_fields_checked']:
        assert p[name] == old[name], name
    assert p['RAW_groups_per_round'] == 32 and p['PNG_groups_per_round'] == 0
    assert p['maximum_gradient_queries'] == 640 and p['recognizer_forward_cap'] == 320
    assert p['refresh_all32_RAW_gradients_after_every_accepted_update']
    assert not p['refresh_all64_gradients_after_every_accepted_update']
    assert not p['PNG_surrogate_used_for_optimization']
    assert p['exact_PNG_delivery_and_all17_scientific_acceptance_groups_retained']
    assert p['optimizer_constraints'] == '32 RAW groups; PNG measured by actual unchanged acceptance gates'
    worker = packet / 'scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py'
    text = worker.read_text(encoding='utf-8')
    restored = text.replace('raw_conditioning_learning_v1', 'conditioning_learning_v1_r4')
    restored = restored.replace('cctv-dgp-raw-conditioning-learning-v1', 'cctv-dgp-conditioning-learning-v1-r4')
    for before, after in reversed(source['changes']):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == (RETURN / 'scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py').read_text(encoding='utf-8')
    assert sha(worker) == sha(ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py')
    original = (RETURN / 'conditioning_contract.py').read_text(encoding='utf-8')
    contract = (packet / 'conditioning_contract.py').read_text(encoding='utf-8')
    begin, end = contract.index('def solve_direction('), contract.index('def bounded_rate(')
    solver = contract[begin:end]
    for before, after in [
        ('len(vectors) == len(records) == 32', 'len(vectors) == len(records) == 64'),
        ('np.zeros((32,32))', 'np.zeros((64,64))'), ('rank_bound = 32*', 'rank_bound = 64*'),
        ('Retain all32 RAW loss dimensions', 'Retain all64 loss dimensions'),
        ('np.zeros(32)', 'np.zeros(64)'),
        ('all32_RAW_loss_dimensions_retained', 'all64_loss_dimensions_retained'),
    ]:
        assert before in solver
        solver = solver.replace(before, after)
    restored_contract = contract[:begin] + solver + contract[end:]
    assert restored_contract == original
    assert sha(packet / 'conditioning_contract.py') == sha(ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_contract.py')
    for name in ['conditioning_model.py', 'scripts/run_learning.sh']:
        restored = (packet / name).read_text(encoding='utf-8').replace('raw_conditioning_learning_v1', 'conditioning_learning_v1_r4')
        restored = restored.replace('cctv-dgp-raw-conditioning-learning-v1', 'cctv-dgp-conditioning-learning-v1-r4')
        assert restored == (RETURN / name).read_text(encoding='utf-8')
    changed = {'conditioning_contract.py', 'conditioning_model.py', 'scripts/run_learning.sh',
               'scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py'}
    for name, digest in old['assets_sha256'].items():
        if name not in changed:
            assert sha(packet / name) == digest
    initial = read(ANALYSIS / 'failure_audit.json')
    assert initial['complete'] and initial['actual_update_count_reported'] == 0
    assert len(initial['gradient_audits']) == 1 and initial['gradient_audits'][0]['full_vector_KKT_and_float32_dots_checked']
    prediction = read(packet / 'gradient_prediction_review.json')
    assert sha(packet / 'gradient_prediction_review.json') == p['reviewed_prediction_sha256'] == sha(ANALYSIS / 'gradient_prediction_review.json')
    assert prediction['complete'] and len(prediction['real_group_trial_comparisons']) == 192
    assert prediction['eigensystems']['joint64']['condition_number'] > 100000 * prediction['eigensystems']['RAW32']['condition_number']
    assert sha(packet / 'diagnostic_decision.json') == p['human_diagnostic_decision_sha256'] == sha(ANALYSIS / 'diagnostic_decision.json')
    assert read(packet / 'diagnostic_decision.json')['human_answer'] == 'apply best approach'
    green = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_solver_green.json')
    precision = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_precision_green.json')
    assert green['complete'] and precision['complete']
    assert green['contract_sha256'] == precision['contract_sha256'] == sha(packet / 'conditioning_contract.py')
    assert green['checker_sha256'] == sha(ROOT / 'scripts/check_cctv_dgp_raw_conditioning_solver_v1.py')
    assert precision['checker_sha256'] == sha(ROOT / 'scripts/check_cctv_dgp_raw_conditioning_precision_v1.py')
    assert not read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_solver_red.json')['complete']
    assert not read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_precision_red.json')['complete']
    stop = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_draft_stop/stop.json')
    assert sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_draft_stop/conditioning_contract.py') == stop['original_contract_sha256']
    assert stop['packet_not_sealed_uploaded_or_trained']
    assert stop['corrected_contract_sha256'] == sha(packet / 'conditioning_contract.py')
    pixels = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_pixel_fixture_green.json')
    assert pixels['complete'] and pixels['protocol_sha256'] == sha(packet / 'protocol.json')
    assert pixels['actual_fixture_shape'] == [1, 3, 256, 256]
    assert pixels['worker_sha256'] == sha(worker) and pixels['helper_sha256'] == sha(packet / 'conditioning_losses.py')
    future = (ROOT / 'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_return.py').read_text(encoding='utf-8')
    assert "for stage,index in [('raw',0),('png',1)]" in future
    assert "keys=['RAW/'+g['key']" in future and "gradient_queries=320*len(gradient_audits)" in future
    helper = (ROOT / 'scripts/raw_conditioning_decision_receipt_check_v1.py').read_text(encoding='utf-8')
    assert helper.replace('raw_conditioning_learning_v1', 'conditioning_learning_v1_r4') == (
        ROOT / 'scripts/conditioning_decision_receipt_check_v1.py').read_text(encoding='utf-8')
    assert 'png_losses' not in text
    assert "stages=[('RAW',pred,raw_losses(pred,target,mask,support))]" in text
    assert "if checked['accept_step']:" in text and "qualified=checked['one_percent_checkpoint_pass']" in text
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    for path in [worker, ROOT / 'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_return.py']:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    return dict(complete=True, checker_sha256=sha(__file__),
        original_R4_worker_restored_exactly_by_reversing_declared_optimizer_and_namespace_changes=True,
        original_solver_restored_exactly_by_reversing32_constraint_sizing=True,
        original_float64_Gram_rank_and_dot_math_preserved=True,
        bounded_rate_and_entire_actual_RAW_PNG_decision_source_unchanged=True,
        same_initial_full_state12_conditioners100_cases20_references2_sources=True,
        all17_RAW_and17_PNG_groups_and_1percent_endpoint_and_10percent_full_TRAIN_gates_unchanged=True,
        no_approximate_PNG_constraints_in_direction_solver_explicitly_declared=True,
        synthetic_full_rank_rank_deficient_nonfinite_and_cumulative_distance_fixtures_checked=True,
        unsealed_precision_draft_stop_retained_and_fixed_before_training=True,
        all256_PNG_byte_levels_checked_without_models=True,
        original_R4_failure_and_rejected_states_and_checkers_preserved=True,
        local_neural_gradient_optimizer_or_actual_model_displacement_calls=0,
        current_new_CUDA_training_not_run=True, model_qualified=False, goal_complete=False)

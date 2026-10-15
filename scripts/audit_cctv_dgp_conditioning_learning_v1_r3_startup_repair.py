"""Independently check the demonstrated startup repair and unchanged recipe."""
from import_cctv_dgp_conditioning_learning_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha


def audit(packet, p):
    previous = read(RETURN / 'protocol.json')
    startup = read(ANALYSIS / 'startup_review.json')
    red = read(ANALYSIS / 'pixel_fixture_red.json')
    green_path = OUT / 'cctv_dgp_conditioning_learning_v1_r3_pixel_fixture_green.json'
    green = read(green_path)
    assert startup['complete'] and p['startup_repair_of_protocol_sha256'] == sha(RETURN / 'protocol.json') == PIN
    assert p['startup_review_sha256'] == sha(ANALYSIS / 'startup_review.json') == sha(packet / 'startup_review.json')
    assert not any(startup['reported_counts'].values()) and not startup['finite_structure_gate_reached']
    assert sha(ANALYSIS / 'pixel_fixture_red.json') == startup['fixture_red_sha256']
    assert not red['complete'] and red['actual_fixture_shape'] == [1, 3, 16, 16]
    assert green['complete'] and green['actual_fixture_shape'] == [1, 3, 256, 256]
    assert green['all256_levels_canonical_RGB32_and_feature64']
    assert green['protocol_sha256'] == sha(packet / 'protocol.json')
    assert green['helper_sha256'] == red['helper_sha256'] == previous['assets_sha256']['conditioning_losses.py']
    assert green['checker_sha256'] == red['checker_sha256'] == sha(ROOT / 'scripts/check_cctv_dgp_conditioning_preflight_pixels_v1.py')
    old_worker = RETURN / 'scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py'
    new_worker = packet / 'scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py'
    assert green['worker_sha256'] == sha(new_worker) == p['assets_sha256']['scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py']
    restored = new_worker.read_text(encoding='utf-8').replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r2')
    restored = restored.replace('cctv-dgp-conditioning-learning-v1-r3', 'cctv-dgp-conditioning-learning-v1-r2')
    assert restored.count(green['actual_fixture_expression']) == 1
    restored = restored.replace(green['actual_fixture_expression'], red['actual_fixture_expression'])
    assert restored == old_worker.read_text(encoding='utf-8'), 'Only the fixture and namespace may change'
    assert sha(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py') == sha(new_worker)
    for n, old_hash in previous['assets_sha256'].items():
        target = n.replace('conditioning_learning_v1_r2', 'conditioning_learning_v1_r3')
        if n.startswith('scripts/cctv_dgp_'):
            continue
        if n in ['scripts/run_learning.sh', 'conditioning_model.py']:
            restored = (packet / target).read_text(encoding='utf-8').replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r2')
            restored = restored.replace('cctv-dgp-conditioning-learning-v1-r3', 'cctv-dgp-conditioning-learning-v1-r2')
            assert restored == (RETURN / n).read_text(encoding='utf-8')
        else:
            assert sha(packet / target) == old_hash == sha(RETURN / n)
    excluded = {'format', 'name', 'UTC', 'assets_sha256', 'startup_repair_of_protocol_sha256',
                'startup_repair_of_return_archive_sha256', 'startup_review_sha256', 'repair_scope'}
    assert {k: v for k, v in previous.items() if k not in excluded} == {k: v for k, v in p.items() if k not in excluded}
    # The named flattening order is independently bound to the already returned
    # activation schema. No neural constructor is used for this check.
    from audit_cctv_dgp_conditioning_learning_v1_r3_return import schema
    state = schema(packet)
    assert [n for n in state if n in p['conditioning_names']] == p['conditioning_names']
    assert sha(packet / 'conditioning_contract.py') == sha(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r3_contract.py')
    assert sha(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r3_contract.py') == sha(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r2_contract.py')
    return dict(complete=True, checker_sha256=sha(__file__), red_regression_sha256=sha(ANALYSIS / 'pixel_fixture_red.json'),
                green_regression_sha256=sha(green_path), reproduced_original_helper_failure=True,
                all256_byte_levels_on256x256_check_passed_on_CPU=True,
                exact_worker_diff_is_fixture_and_namespace_only=True,
                helper_losses_contract_readouts_state_and_all_training_limits_unchanged=True,
                saved_conditioning_parameter_order_matches_protocol=True,
                original_failure_and_all_R2_packet_members_retained=True,
                manual_L4_training_not_yet_run=True,
                local_neural_CUDA_gradient_optimizer_or_parameter_displacement_calls=0,
                model_qualified=False, more_epochs_qualified=False, goal_complete=False)

"""Independent narrow receipt correction plus retained RAW32 design audit."""
import ast

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, read, sha


def audit(packet, p):
    from audit_cctv_dgp_raw_conditioning_learning_v1_design import audit as previous_audit
    original = OUT / 'cctv_dgp_raw_conditioning_learning_v1_vm'
    old = read(original / 'protocol.json')
    old_transfer = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_transfer_audit.json')
    previous = previous_audit(original, old)
    assert previous == old_transfer['RAW_optimizer_and_unchanged_quality_gate_design_audit']
    prep = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_preparation.json')
    assert prep['complete'] and prep['source_protocol_sha256'] == sha(original / 'protocol.json')
    excluded = {'format', 'name', 'UTC', 'assets_sha256', 'transfer_receipt_repair_of_protocol_sha256',
                'transfer_receipt_red_sha256', 'transfer_receipt_fields_read_directly_from_protocol',
                'namespace_and_transfer_summary_only_changed_from_RAW32_V1'}
    assert {k:v for k,v in p.items() if k not in excluded} == {k:v for k,v in old.items() if k not in excluded}
    assert p['transfer_receipt_repair_of_protocol_sha256'] == sha(original / 'protocol.json')
    assert all(sha(original / n) == h for n, h in old['assets_sha256'].items())
    assert all(sha(packet / n) == h for n, h in p['assets_sha256'].items())
    worker = packet / 'scripts/cctv_dgp_raw_conditioning_learning_v1_r1_vm.py'
    text = worker.read_text(encoding='utf-8')
    restored = text.replace('raw_conditioning_learning_v1_r1', 'raw_conditioning_learning_v1')
    restored = restored.replace('cctv-dgp-raw-conditioning-learning-v1-r1', 'cctv-dgp-raw-conditioning-learning-v1')
    new_line = "cases=p['paired_TRAIN_cases'],maximum_accepted_updates=p['maximum_accepted_updates'],maximum_gradient_queries=p['maximum_gradient_queries'],actual_neural_calls=0,"
    old_line = 'cases=100,maximum_accepted_updates=2,maximum_gradient_queries=1280,actual_neural_calls=0,'
    assert restored.count(new_line) == 1
    restored = restored.replace(new_line, old_line)
    assert restored == (original / 'scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py').read_text(encoding='utf-8')
    assert sha(worker) == sha(ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_r1_vm.py')
    for name in ['conditioning_model.py', 'scripts/run_learning.sh']:
        restored = (packet / name).read_text(encoding='utf-8').replace('raw_conditioning_learning_v1_r1', 'raw_conditioning_learning_v1')
        restored = restored.replace('cctv-dgp-raw-conditioning-learning-v1-r1', 'cctv-dgp-raw-conditioning-learning-v1')
        assert restored == (original / name).read_text(encoding='utf-8')
    for name, h in old['assets_sha256'].items():
        if name not in ['conditioning_model.py', 'scripts/run_learning.sh', 'scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py']:
            assert sha(packet / name) == h
    red = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_transfer_receipt_red.json')
    green = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_transfer_receipt_green.json')
    assert not red['complete'] and green['complete'] and not green['differences']
    assert green['protocol_sha256'] == sha(packet / 'protocol.json')
    assert green['worker_sha256'] == sha(worker)
    assert green['checker_sha256'] == red['checker_sha256'] == sha(ROOT / 'scripts/check_cctv_dgp_manual_transfer_receipt_v1.py')
    assert green['actual_hash_only_worker_receipt']['maximum_gradient_queries'] == p['maximum_gradient_queries'] == 640
    assert green['actual_hash_only_worker_receipt']['maximum_accepted_updates'] == p['maximum_accepted_updates'] == 2
    assert sha(packet / 'transfer_receipt_review.json') == p['transfer_receipt_red_sha256']
    pixels = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_pixel_fixture_green.json')
    assert pixels['complete'] and pixels['worker_sha256'] == sha(worker)
    assert pixels['protocol_sha256'] == sha(packet / 'protocol.json')
    assert pixels['actual_fixture_shape'] == [1, 3, 256, 256]
    assert sha(ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_r1_contract.py') == sha(packet / 'conditioning_contract.py')
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    for file in packet.rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8'), feature_version=(3, 10))
    return dict(complete=True, checker_sha256=sha(__file__),
        retained_RAW32_design_audit=previous, old_protocol_packet_checkers_and_red_summary_retained=True,
        worker_diff_only_namespace_and_hash_only_receipt_fields=True,
        optimizer_architecture_states_cases_32_RAW_groups_all17_RAW_PNG_gates_and_finite_limits_unchanged=True,
        actual_fresh_hash_only_receipt_matches640_queries2_updates100_cases_and16_assets=True,
        pixel_fixture_checked_on_actual_R1_worker_without_models=True,
        local_neural_gradient_optimizer_or_actual_model_displacement_calls=0,
        current_new_CUDA_training_not_run=True, model_qualified=False, goal_complete=False)

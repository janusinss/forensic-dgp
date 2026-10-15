"""Independent source-diff, recorded regression and unchanged scientific gates."""
import ast
from import_cctv_dgp_conditioning_learning_v1_r3_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha


def audit(packet, p):
    old = read(RETURN / 'protocol.json')
    red, green = [read(ANALYSIS / n) for n in ['loss_forward_red.json', 'loss_forward_green.json']]
    cause = read(ANALYSIS / 'loss_alignment_review.json')
    assert cause['complete'] and green['complete'] and not red['complete']
    assert len(red['failures']) == 1 and not green['failures']
    assert red['real_saved_loss_checks'] == green['real_saved_loss_checks'] == 600
    assert red['checker_sha256'] == green['checker_sha256'] == sha(ROOT / 'scripts/check_cctv_dgp_conditioning_loss_forward_v1.py')
    assert p['loss_forward_repair_of_protocol_sha256'] == sha(RETURN / 'protocol.json') == PIN
    assert p['loss_alignment_review_sha256'] == sha(ANALYSIS / 'loss_alignment_review.json') == sha(packet / 'loss_alignment_review.json')
    assert green['raw_helper_sha256'] == sha(packet / 'conditioning_raw_losses.py') == sha(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r4_raw_losses.py')
    assert red['png_helper_sha256'] == green['png_helper_sha256'] == sha(packet / 'conditioning_losses.py')
    assert red['raw_helper_sha256'] == sha(PARENT / 'cctv_dgp_group_conflicts_v1_losses.py')
    # Independently remove the already audited PNG canonical-value correction;
    # the remaining RAW source must match the original derivative graph exactly.
    png_text = (RETURN / 'conditioning_losses.py').read_text(encoding='utf-8')
    start = png_text.index('    # The scientific forward uses the retained float32 scikit-image evaluator.')
    end = png_text.index("    return {'MSE': mse", start)
    canonical = png_text[start:end]
    helper = (packet / 'conditioning_raw_losses.py').read_text(encoding='utf-8')
    additions = 'import cv2\nimport numpy as np\nfrom skimage.metrics import structural_similarity\n'
    old_return = "    return {'MSE': mse, 'SSIM_loss': 1. - ssim, 'landmark_structure': feature}\n"
    new_return = "    return {'MSE': mse, 'SSIM_loss': ssim_loss, 'landmark_structure': feature}\n"
    assert helper.count(canonical + new_return) == 1
    restored = helper.replace('import torch\n' + additions, 'import torch\n').replace(canonical + new_return, old_return)
    assert restored == (PARENT / 'cctv_dgp_group_conflicts_v1_losses.py').read_text(encoding='utf-8')
    worker = packet / 'scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py'
    text = worker.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    diagnostics = [n for n in ast.walk(tree) if isinstance(n, ast.If) and any(
        isinstance(c, ast.Constant) and c.value == 'loss_alignment_failure.json' for c in ast.walk(n))]
    assert len(diagnostics) == 1
    node = diagnostics[0]
    restored = ''.join(lines[:node.lineno - 1] + lines[node.end_lineno:])
    restored = restored.replace('conditioning_learning_v1_r4', 'conditioning_learning_v1_r3').replace(
        'cctv-dgp-conditioning-learning-v1-r4', 'cctv-dgp-conditioning-learning-v1-r3')
    restored = restored.replace('from conditioning_raw_losses import pixel_and_structure_losses as raw_losses',
                                'from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses as raw_losses')
    assert restored == (RETURN / 'scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py').read_text(encoding='utf-8')
    assert sha(worker) == sha(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py')
    excluded = {'format', 'name', 'UTC', 'assets_sha256', 'repair_scope',
                'loss_forward_repair_of_protocol_sha256', 'loss_forward_repair_of_return_archive_sha256',
                'loss_alignment_review_sha256', 'RAW_SSIM_exact_scientific_forward_with_explicit_approximate_float64_backward',
                'RAW_SSIM_backward_graph_unchanged_from_R3', 'alignment_failure_exports_actual_scalar_case_stage_and_bound'}
    assert {k: v for k, v in old.items() if k not in excluded} == {k: v for k, v in p.items() if k not in excluded}
    for n, h in old['assets_sha256'].items():
        new_n = n.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r4')
        if n.startswith('scripts/cctv_dgp_'):
            continue
        if n in ['conditioning_model.py', 'scripts/run_learning.sh']:
            restored = (packet / new_n).read_text(encoding='utf-8').replace('conditioning_learning_v1_r4', 'conditioning_learning_v1_r3')
            restored = restored.replace('cctv-dgp-conditioning-learning-v1-r4', 'cctv-dgp-conditioning-learning-v1-r3')
            assert restored == (RETURN / n).read_text(encoding='utf-8')
        else:
            assert sha(packet / new_n) == h == sha(RETURN / n)
    pixels = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_pixel_fixture_green.json')
    assert pixels['complete'] and pixels['protocol_sha256'] == sha(packet / 'protocol.json')
    assert pixels['worker_sha256'] == sha(worker) and pixels['actual_fixture_shape'] == [1, 3, 256, 256]
    assert pixels['helper_sha256'] == sha(packet / 'conditioning_losses.py')
    assert pixels['checker_sha256'] == sha(ROOT / 'scripts/check_cctv_dgp_conditioning_preflight_pixels_v1.py')
    assert sha(packet / 'conditioning_contract.py') == sha(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r4_contract.py')
    return dict(complete=True, checker_sha256=sha(__file__),
        old_CPU_forward_regression_sha256=sha(ANALYSIS / 'loss_forward_red.json'),
        corrected_CPU_forward_regression_sha256=sha(ANALYSIS / 'loss_forward_green.json'),
        actual600_RAW_PNG_saved_loss_checks_passed=True, failed_GPU_scalar_not_known=True,
        canonical_RAW_SSIM_forward_and_explicit_unchanged_approximate_backward=True,
        original_RAW_graph_restored_exactly_by_removing_canonical_value_correction=True,
        worker_diff_only_RAW_import_failure_instrumentation_and_namespace=True,
        scientific_quality_and_alignment_bounds_limits_cohort_and_initial_state_unchanged=True,
        original_R3_run_and_all_packet_members_retained=True,
        current256_pixel_fixture_checked_without_models=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        new_manual_training_unrun=True, model_qualified=False, goal_complete=False)

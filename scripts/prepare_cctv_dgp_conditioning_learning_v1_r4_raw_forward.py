"""Correct only RAW SSIM's scientific forward, using the retained PNG convention."""
import ast
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_learning_v1_r3_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def main():
    red = read(ANALYSIS / 'loss_forward_red.json')
    assert red['complete'] is False and len(red['failures']) == 1
    row = red['failures'][0]
    assert row['id'] == 'v9_tr_asian_00180_blur_lr24' and row['term'] == 'SSIM_loss' and row['stage'] == 'raw'
    assert row['absolute_difference'] > row['retained_alignment_bound'] == 1e-6
    p = read(RETURN / 'protocol.json')
    parent_source = PARENT / 'cctv_dgp_group_conflicts_v1_losses.py'
    png_source = RETURN / 'conditioning_losses.py'
    assert sha(parent_source) == p['parent_required_sha256']['cctv_dgp_group_conflicts_v1_losses.py'] == red['raw_helper_sha256']
    assert sha(png_source) == p['assets_sha256']['conditioning_losses.py'] == red['png_helper_sha256']
    original = parent_source.read_text(encoding='utf-8')
    assert original.count('import torch\n') == 1
    additions = 'import cv2\nimport numpy as np\nfrom skimage.metrics import structural_similarity\n'
    text = original.replace('import torch\n', 'import torch\n' + additions)
    source = png_source.read_text(encoding='utf-8')
    begin = source.index('    # The scientific forward uses the retained float32 scikit-image evaluator.')
    end = source.index("    return {'MSE': mse", begin)
    canonical = source[begin:end]
    old_return = "    return {'MSE': mse, 'SSIM_loss': 1. - ssim, 'landmark_structure': feature}\n"
    new_return = "    return {'MSE': mse, 'SSIM_loss': ssim_loss, 'landmark_structure': feature}\n"
    assert text.count(old_return) == 1
    text = text.replace(old_return, canonical + new_return)
    # Exact inverse verifies no pooling, mask, MSE, feature or derivative graph
    # expression has been changed. The added detach correction's derivative is
    # the original surrogate derivative, explicitly not the discrete evaluator.
    restored = text.replace('import torch\n' + additions, 'import torch\n').replace(canonical + new_return, old_return)
    assert restored == original
    ast.parse(text, feature_version=(3, 10))
    target = ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r4_raw_losses.py'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    write(OUT / 'cctv_dgp_conditioning_learning_v1_r4_RAW_forward_preparation.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__),
        failed_regression_sha256=sha(ANALYSIS / 'loss_forward_red.json'),
        original_RAW_helper_sha256=sha(parent_source), corrected_RAW_helper_sha256=sha(target),
        canonical_code_source_sha256=sha(png_source),
        correction='Canonical float32 scientific SSIM forward plus unchanged float64 approximate derivative',
        inverse_source_diff_exactly_restores_original=True,
        no_mask_MSE_feature_derivative_architecture_or_quality_threshold_change=True,
        local_neural_gradient_optimizer_parameter_displacement_calls=0,
        real_all_case_forward_check_pending=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, corrected_helper=str(target), real_all_case_check_pending=True))


if __name__ == '__main__':
    main()

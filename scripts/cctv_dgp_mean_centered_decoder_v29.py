"""A single fixed mean-centering path inside the own-DGP decoder optimization.

The frozen original DGP supplies the baseline for the same prepared input.
All profiles use this path. No target, profile, source or identity enters it.
Clipping can reintroduce mean shift; retained delivered-output gates decide it.
"""
import torch

from cctv_dgp_active_original_decoder_v28 import ActiveOriginalDecoderV28
from cctv_dgp_original_decoder_candidate_v1 import require_gradient_vm


def center_observed_delta(prediction, baseline, image, support):
    if torch.is_grad_enabled():
        require_gradient_vm(prediction)
    assert prediction.dtype == baseline.dtype == image.dtype == torch.float32
    assert prediction.shape == baseline.shape == image.shape and prediction.ndim == 4 and prediction.shape[1] == 3
    assert support.shape == (prediction.shape[0], 1, prediction.shape[2], prediction.shape[3])
    assert prediction.device == baseline.device == image.device == support.device
    assert not baseline.requires_grad and not image.requires_grad and not support.requires_grad
    assert bool(((support == 0) | (support == 1)).all()) and bool((support.sum((1, 2, 3)) > 0).all())
    for value in [prediction, baseline, image]:
        assert bool(torch.isfinite(value).all()) and bool(((value >= 0) & (value <= 1)).all())
    mask = support.to(dtype=torch.float32)
    delta = prediction - baseline
    mean = (delta * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True)
    centered = baseline + delta - mean
    return torch.where(support.bool(), centered.clamp(0, 1), image)


class MeanCenteredOriginalDecoderV29(ActiveOriginalDecoderV28):
    def forward(self, image, support, baseline):
        # The inherited original-decoder VM guard runs before any neural work.
        prediction = super().forward(image, support)
        return center_observed_delta(prediction, baseline, image, support)

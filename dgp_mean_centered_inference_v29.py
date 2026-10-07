"""Explicitly selected V29 decoder plus same-input frozen original DGP.

Inference only. Loading a checkpoint never qualifies or promotes it in the app.
The mean-centering path is required at inference because it was used in training.
"""
from pathlib import Path

import torch
from torch import nn

from dgp_face_restoration import sha
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer

ORIGINAL_SHA256 = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
FINAL800_SHA256 = 'd8bf23337282b44b5a530d9b4eeec05688fb16114e7be1f190c04cfff6c4ea38'
PROTOCOL_SHA256 = '77565ba437959305f22cff4dd967fc6c3caadbf9dbd4ac91abf8a366577fa72f'
SELECTED = (
    'head1.block0.weight', 'head1.block1.weight',
    'head2.block0.weight', 'head2.block1.weight',
    'head3.block0.weight', 'head3.block1.weight',
    'smooth.0.weight', 'smooth.0.bias', 'smooth2.0.weight', 'smooth2.0.bias',
    'final.weight', 'final.bias',
)


def validate_image_support(image, support):
    if not isinstance(image, torch.Tensor) or image.dtype != torch.float32 or image.ndim != 4 or image.shape[1:] != (3, 256, 256) or not len(image):
        raise ValueError('Prepare a nonempty float32 N x3 x256 x256 RGB batch')
    if image.requires_grad or not bool(torch.isfinite(image).all()) or not bool(((image >= 0) & (image <= 1)).all()):
        raise ValueError('Use finite observed RGB in[0,1] without input gradients')
    if not isinstance(support, torch.Tensor) or support.shape != (len(image), 1, 256, 256) or support.device != image.device or support.requires_grad:
        raise ValueError('Use a matching binary observed support without gradients')
    if not bool(((support == 0) | (support == 1)).all()) or not bool((support.sum((1, 2, 3)) > 0).all()):
        raise ValueError('Observed support must be binary and nonempty in every case')


@torch.inference_mode()
def project_observed_delta(prediction, baseline, image, support):
    """Same float32 operation order as the frozen VM training path."""
    validate_image_support(image, support)
    for value in (prediction, baseline):
        if value.shape != image.shape or value.dtype != torch.float32 or value.device != image.device or value.requires_grad:
            raise ValueError('Use matching frozen predictions')
        if not bool(torch.isfinite(value).all()) or not bool(((value >= 0) & (value <= 1)).all()):
            raise FloatingPointError('Invalid frozen DGP output')
    mask = support.to(dtype=torch.float32)
    delta = prediction - baseline
    mean = (delta * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True)
    centered = baseline + delta - mean
    return torch.where(support.bool(), centered.clamp(0, 1), image)


class MeanCenteredDGPInferenceV29(nn.Module):
    def __init__(self, original, candidate):
        super().__init__()
        for model in (original, candidate):
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError('Only strictly frozen inference models are permitted')
        self.original, self.candidate = original, candidate
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def forward_components(self, image, support):
        validate_image_support(image, support)
        self.eval()
        baseline = self.original(image)
        candidate = torch.where(support.bool(), self.candidate(image), image)
        result = project_observed_delta(candidate, baseline, image, support)
        return {'original_raw': baseline, 'candidate_unprojected': candidate, 'result': result}

    @torch.inference_mode()
    def forward(self, image, support):
        return self.forward_components(image, support)['result']


def load_mean_centered_dgp_v29(original_path, candidate_path, *, device='cpu'):
    """Only the audited original and final800 states; no fallback or selection."""
    original_path, candidate_path = Path(original_path), Path(candidate_path)
    if sha(original_path) != ORIGINAL_SHA256 or sha(candidate_path) != FINAL800_SHA256:
        raise ValueError('Require the independently audited original and V29 final800 fingerprints')
    original, op = load_frozen_dgp_restorer(original_path, expected_sha256=ORIGINAL_SHA256, device=device)
    candidate, cp = load_frozen_dgp_restorer(candidate_path, expected_sha256=FINAL800_SHA256, device=device)
    before, after = original.net.state_dict(), candidate.net.state_dict()
    if set(before) != set(after):
        raise ValueError('Original DGP schema changed')
    for name in before:
        if before[name].shape != after[name].shape or before[name].dtype != after[name].dtype:
            raise ValueError('Original DGP tensor schema changed: ' + name)
        if name not in SELECTED and not torch.equal(before[name], after[name]):
            raise ValueError('Frozen encoder/head4/evaluation buffer changed: ' + name)
    if sum(before[n].numel() for n in SELECTED) != 498627:
        raise ValueError('Selected12 decoder layout changed')
    model = MeanCenteredDGPInferenceV29(original, candidate).to(device).eval()
    provenance = {
        'backend': 'our-trained-dgp-v29', 'policy': 'mean-centered-observed-DGP-v29',
        'protocol_sha256': PROTOCOL_SHA256, 'original_weights_sha256': ORIGINAL_SHA256,
        'candidate_weights_sha256': FINAL800_SHA256, 'selected_snapshot': 800,
        'original': op, 'candidate': cp, 'same_input_original_baseline_required': True,
        'mean_centering': 'Per-case/channel observed(candidate-original) mean removed before clamping',
        'clipping_may_reintroduce_mean_shift': True, 'projection_learned_parameters': 0,
        'fixed_normalization_layers_per_DGP': 5, 'frozen_encoder_head4_and_buffers_exact': True,
        'training': False, 'inference_only': True, 'enhancement': 'none',
        'target_profile_source_identity_routing': False, 'automatic_app_promotion': False,
        'quality_claim': 'TRAIN capacity passed; development/native and independent final review still required',
    }
    return model, provenance

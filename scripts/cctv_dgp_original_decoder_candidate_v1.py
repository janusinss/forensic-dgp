"""Separate copy of the retained own-DGP decoder; no optimizer or checkpoint writer.

Local execution permits forward-only review. Differentiation requires the existing
L4 VM. The original inference adapter and checkpoint are never modified.
"""
import copy
from pathlib import Path
import platform
import sys
import urllib.request

import torch
from torch import nn

DECODER_PREFIXES = ('head1.', 'head2.', 'head3.', 'head4.', 'smooth.', 'smooth2.', 'final.')
DECODER_PARAMETERS = 609219
DECODER_TENSORS = 14


def require_gradient_vm(image):
    if sys.platform != 'linux' or platform.node().split('.')[0] != 'forensic-dgp-thesis':
        raise RuntimeError('Decoder differentiation requires the existing Linux VM')
    root = Path(__file__).resolve().parent
    if not root.is_relative_to((Path.home() / 'forensic-dgp').resolve()):
        raise RuntimeError('Decoder differentiation requires ~/forensic-dgp')
    if image.device.type != 'cuda' or not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(image.device):
        raise RuntimeError('Decoder differentiation requires NVIDIA L4 CUDA')
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                                     headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=3) as response:
        if not response.read().decode().rstrip().endswith('/g2-standard-4'):
            raise RuntimeError('Decoder differentiation requires existing g2-standard-4')


class OriginalDecoderCandidate(nn.Module):
    """Use the original DGPSynthesizer forward with only its reconstruction weights enabled."""
    def __init__(self, original):
        super().__init__()
        assert not original.training and all(not v.requires_grad for v in original.parameters())
        self.net = copy.deepcopy(original)
        self.net.eval().requires_grad_(False)
        for name, value in self.net.named_parameters():
            if name.startswith(DECODER_PREFIXES):
                value.requires_grad_(True)
        selected = [(name, value) for name, value in self.net.named_parameters() if value.requires_grad]
        assert len(selected) == DECODER_TENSORS and sum(v.numel() for _, v in selected) == DECODER_PARAMETERS
        assert {name.split('.')[0] for name, _ in selected} == {name[:-1] for name in DECODER_PREFIXES}
        assert all(not value.requires_grad for value in self.net.fpn.parameters())
        # Shared FPN names inside a single model are retained, but nothing aliases the original.
        for left, right in zip(original.parameters(), self.net.parameters()):
            assert left is not right and left.data_ptr() != right.data_ptr() and torch.equal(left, right)
        for left, right in zip(original.buffers(), self.net.buffers()):
            assert left is not right and left.data_ptr() != right.data_ptr() and torch.equal(left, right)
        norms = [layer for layer in self.net.modules() if isinstance(layer, nn.InstanceNorm2d)]
        assert len(norms) == 5 and all(layer.track_running_stats and not layer.training for layer in norms)
        self.eval()

    def train(self, mode=True):
        # Evaluation normalization stays fixed even when decoder weights permit derivatives.
        return super().train(False)

    def forward(self, image, support):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1:] != (3, 256, 256) or not len(image):
            raise ValueError('Use a prepared nonempty N x3 x256 x256 image')
        if image.dtype != torch.float32 or not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
            raise ValueError('Use finite float32 RGB in [0,1]')
        if image.requires_grad:
            raise ValueError('Observed input gradients are outside this decoder review')
        if not isinstance(support, torch.Tensor) or support.shape != (len(image), 1, 256, 256) or support.device != image.device:
            raise ValueError('Use a matching observed support')
        if not bool(((support == 0) | (support == 1)).all()) or not bool((support.sum((2, 3)) > 0).all()):
            raise ValueError('Observed support must be binary and nonempty in every case')
        if self.net.training or any(layer.training for layer in self.net.modules()):
            raise RuntimeError('Stored normalization must stay in evaluation mode')
        if torch.is_grad_enabled():
            require_gradient_vm(image)
        raw = self.net(image)
        if raw.shape != image.shape or raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()):
            raise FloatingPointError('Invalid original-decoder output')
        return torch.where(support.bool(), raw, image)

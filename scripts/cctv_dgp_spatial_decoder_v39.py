"""Prospective own-DGP spatial decoder; frozen locally, learning on the existing VM only.

The original DGP supplies input-specific multiscale features. A new two-scale
decoder subtracts its fixed initial response to preserve exact starting output.
No pretrained restoration weights or image-space sharpening are used here.
"""
import copy
from pathlib import Path
import platform
import sys

import torch
from torch import nn
from torch.nn import functional as F

SEED = 390039
WIDTH = 16


def require_learning_vm(root):
    root = Path(root).resolve()
    assert sys.platform == 'linux', 'All gradients/learning require the existing Linux VM'
    assert platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing forensic-dgp-thesis VM only'
    assert root == (Path.home() / 'forensic-dgp/cctv_dgp_spatial_decoder_vm_v39').resolve(), 'Distinct V39 VM root required'


class ChannelNorm(nn.Module):
    """Per-pixel channel normalization: no running or cross-image statistics."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image):
        centered = image - image.mean(1, keepdim=True)
        return centered * torch.rsqrt(centered.square().mean(1, keepdim=True) + 1e-6) * self.weight + self.bias


class SpatialGate(nn.Module):
    """Local gated residual block inspired by the primary NAFNet design, not its model."""
    def __init__(self, channels):
        super().__init__()
        self.norm1 = ChannelNorm(channels)
        self.expand1 = nn.Conv2d(channels, 2 * channels, 1)
        self.local = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.channel = nn.Conv2d(channels, channels, 1)
        self.reduce1 = nn.Conv2d(channels, channels, 1)
        self.norm2 = ChannelNorm(channels)
        self.expand2 = nn.Conv2d(channels, 2 * channels, 1)
        self.reduce2 = nn.Conv2d(channels, channels, 1)

    def forward(self, image):
        a, b = self.local(self.expand1(self.norm1(image))).chunk(2, dim=1)
        gate = a * b
        image = image + .1 * self.reduce1(gate * self.channel(gate.mean((2, 3), keepdim=True)))
        a, b = self.expand2(self.norm2(image)).chunk(2, dim=1)
        return image + .1 * self.reduce2(a * b)


class SpatialDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv2d(102, WIDTH, 1)
        self.high = SpatialGate(WIDTH)
        self.down = nn.Conv2d(WIDTH, 2 * WIDTH, 2, stride=2)
        self.low = SpatialGate(2 * WIDTH)
        self.up = nn.Conv2d(2 * WIDTH, WIDTH, 1)
        self.fuse = nn.Conv2d(2 * WIDTH, WIDTH, 1)
        self.finish = SpatialGate(WIDTH)
        # A constant RGB bias is removed by observed mean centering; omit it.
        self.rgb = nn.Conv2d(WIDTH, 3, 3, padding=1, bias=False)

    def forward(self, context):
        high = self.high(self.embed(context))
        low = self.up(self.low(self.down(high)))
        low = F.interpolate(low, size=(256, 256), mode='bilinear', align_corners=False)
        return self.rgb(self.finish(self.fuse(torch.cat([high, low], dim=1))))


class SpatialDGPCandidateV39(nn.Module):
    def __init__(self, original, initial_decoder_state=None):
        super().__init__()
        assert not original.training and all(not v.requires_grad for v in original.parameters())
        self.original = original
        # Initialize on CPU, with a separate seed context. Initial weights are
        # explicitly transferred later; GPU RNG/version is not a state source.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            self.decoder = SpatialDecoder().eval().requires_grad_(False)
        if initial_decoder_state is not None:
            self.decoder.load_state_dict(initial_decoder_state, strict=True)
        self.reference_decoder = copy.deepcopy(self.decoder).eval().requires_grad_(False)
        device = next(original.parameters()).device
        self.decoder.to(device)
        self.reference_decoder.to(device)
        self.learning_root = None
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        # Normalization and the original network always use evaluation behavior.
        return super().train(False)

    def enable_vm_learning(self, root):
        require_learning_vm(root)
        assert next(self.original.parameters()).device.type == 'cuda' and torch.cuda.is_available()
        assert 'L4' in torch.cuda.get_device_name(0)
        self.learning_root = Path(root).resolve()
        self.decoder.requires_grad_(True)
        self.original.requires_grad_(False)
        self.reference_decoder.requires_grad_(False)
        self.eval()

    def forward_components(self, image, support):
        from dgp_mean_centered_inference_v29 import validate_image_support
        validate_image_support(image, support)
        if any(v.requires_grad for v in self.decoder.parameters()) or image.requires_grad:
            assert self.learning_root is not None
            require_learning_vm(self.learning_root)
        assert all(not v.requires_grad for v in self.original.parameters())
        assert all(not v.requires_grad for v in self.reference_decoder.parameters())
        captured = {}; hooks = []
        for key, module in [('map0', self.original.net.fpn.lateral0), ('smooth2', self.original.net.smooth2)]:
            def collect(_module, _inputs, result, key=key):
                assert key not in captured
                captured[key] = result.detach()
            hooks.append(module.register_forward_hook(collect))
        try:
            raw = self.original(image)
        finally:
            for hook in hooks:
                hook.remove()
        assert set(captured) == {'map0', 'smooth2'}
        assert captured['map0'].shape == (len(image), 64, 128, 128)
        assert captured['smooth2'].shape == (len(image), 32, 128, 128)
        baseline = torch.where(support.bool(), raw, image)
        # Fresh regular tensors outside the adapter's inference context can be
        # saved by a trainable decoder on the VM; no encoder gradient is needed.
        context = torch.cat([image, baseline,
            F.interpolate(captured['map0'], size=(256, 256), mode='bilinear', align_corners=False),
            F.interpolate(captured['smooth2'], size=(256, 256), mode='bilinear', align_corners=False)], dim=1).detach().clone()
        assert context.shape == (len(image), 102, 256, 256)
        with torch.no_grad():
            initial_response = self.reference_decoder(context)
        current_response = self.decoder(context)
        delta = .5 * torch.tanh(current_response - initial_response)
        mask = support.to(dtype=torch.float32)
        centered = delta - (delta * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True)
        result = torch.where(support.bool(), (baseline + centered).clamp(0, 1), image)
        assert bool(torch.isfinite(result).all())
        return {'original_raw': baseline, 'spatial_delta': delta, 'result': result}

    def forward(self, image, support):
        return self.forward_components(image, support)['result']

"""Own-DGP feature-conditioned spatial reconstruction. No training entry point.

The frozen DGP remains the pixel baseline and feature provider. Only the new
decoder is trainable in a separately guarded manual-VM capacity experiment.
"""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class SpatialFeatureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(channels, 16, 1) for channels in [64, 128, 128, 128, 128]])
        self.decode3 = nn.Conv2d(32, 24, 3, padding=1)
        self.decode2 = nn.Conv2d(40, 24, 3, padding=1)
        self.decode1 = nn.Conv2d(40, 24, 3, padding=1)
        self.decode0 = nn.Conv2d(40, 24, 3, padding=1)
        self.camera = nn.Conv2d(13, 16, 3, padding=1)
        self.fuse = nn.Conv2d(40, 24, 3, padding=1)
        self.tail = nn.Conv2d(24, 3, 3, padding=1)
        self.direct = nn.Conv2d(13, 3, 3, padding=1)
        for module in [self.tail, self.direct]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        z = torch.arange(-6, 7, dtype=torch.float32)
        kernel = torch.exp(-.5 * (z / 2).square()); kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)
        self.register_buffer('reflect_indices', torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))))

    def blur(self, value):
        channels = value.shape[1]
        vertical = self.kernel.view(1, 1, 13, 1).expand(channels, 1, 13, 1)
        horizontal = self.kernel.view(1, 1, 1, 13).expand(channels, 1, 1, 13)
        value = F.conv2d(value.index_select(2, self.reflect_indices), vertical, groups=channels)
        return F.conv2d(value.index_select(3, self.reflect_indices), horizontal, groups=channels)

    def high(self, value):
        # The fixed diagnostic/input filter remains; no output high-pass is used.
        return value - self.blur(value)

    def forward(self, x, base, mask, fpn):
        assert x.shape == base.shape and x.ndim == 4 and x.shape[1:] == (3, 256, 256)
        assert mask.shape == (x.shape[0], 1, 256, 256) and len(fpn) == 5
        sizes = [128, 64, 32, 16, 8]
        channels = [64, 128, 128, 128, 128]
        for feature, channels_at_scale, size in zip(fpn, channels, sizes):
            assert feature.shape == (x.shape[0], channels_at_scale, size, size)
            assert feature.dtype == x.dtype and feature.device == x.device
            assert not feature.requires_grad and not torch.is_inference(feature)
        assert bool((mask.sum((2, 3)) > 0).all())
        projected = [F.silu(layer(feature)) for layer, feature in zip(self.projections, fpn)]
        up = lambda value, target: F.interpolate(value, size=target.shape[-2:], mode='bilinear', align_corners=False)
        value = F.silu(self.decode3(torch.cat((up(projected[4], projected[3]), projected[3]), 1)))
        value = F.silu(self.decode2(torch.cat((up(value, projected[2]), projected[2]), 1)))
        value = F.silu(self.decode1(torch.cat((up(value, projected[1]), projected[1]), 1)))
        value = F.silu(self.decode0(torch.cat((up(value, projected[0]), projected[0]), 1)))
        observed = torch.cat((x, base, self.high(x), self.high(base), mask), 1)
        camera = F.silu(self.camera(observed))
        full = F.silu(self.fuse(torch.cat((up(value, camera), camera), 1)))
        q = .05 * torch.tanh(self.direct(observed) + self.tail(full))
        mean = (q * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True)
        correction = (q - mean) * mask
        return (base + correction).clamp(0, 1)


def frozen_fpn_features(net, x):
    """One unchanged frozen-DGP forward; remove its read-only hook on every exit.

    Return ordinary detached tensors. Inference tensors cannot supply convolution
    weight gradients, even when the feature encoder itself stays frozen.
    """
    assert not net.training and all(not value.requires_grad for value in net.parameters())
    assert hasattr(net, 'fpn') and not x.requires_grad
    captured = []
    def capture(_module, _inputs, output):
        assert not captured and isinstance(output, tuple) and len(output) == 5
        captured.extend(output)
    hook = net.fpn.register_forward_hook(capture)
    try:
        with torch.inference_mode(False), torch.no_grad():
            raw = net(x)
            features = tuple(value.detach().clone() for value in captured)
            raw = raw.detach().clone()
    finally:
        hook.remove()
    assert len(features) == 5 and all(not value.requires_grad and not torch.is_inference(value) for value in features)
    return raw, features


def tensor_receipt(value):
    assert value.dtype == torch.float32 and not value.requires_grad
    array = value.detach().cpu().contiguous().numpy()
    assert __import__('numpy').isfinite(array).all()
    return {'shape': list(value.shape), 'dtype': 'float32',
        'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
        'requires_grad': False, 'inference_tensor': bool(torch.is_inference(value))}

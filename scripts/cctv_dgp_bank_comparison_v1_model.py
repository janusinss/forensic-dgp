"""Own256 reconstruction with declared frozen GAN features; manual L4 learning.

Route A retains the parity-preserving repaired current decoder. Route B adds
continuous input/style and spatial conditioning plus multiscale feature fusion.
Both start at the retained DGP output. New fusion projections start at zero;
conditioning gradients are expected after their first update, not at parity.
No pretrained restoration encoder, RGB prior subtraction or display filter.
"""
import copy
import os
from pathlib import Path
import platform
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F

NAME = 'cctv_dgp_bank_comparison_v1_vm'
CURRENT_NAMES = ['net.head4.block0.weight', 'net.head4.block1.weight', 'live_fusion4'] + [
    f'net.head{i}.block{j}.weight' for i in [1, 2, 3] for j in [0, 1]] + [
    'live_fusion_fine', 'net.smooth.0.bias', 'net.smooth2.0.weight', 'net.smooth2.0.bias',
    'net.final.weight', 'net.final.bias']


def require_vm(root, value):
    assert platform.system() == 'Linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root is not None and Path(root).resolve() == (Path.home() / 'forensic-dgp' / NAME).resolve()
    assert os.environ.get('TMUX'), 'Human manual tmux is required'
    assert value.device.type == 'cuda' and torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == 'NVIDIA L4'


class CorrectedCurrentDGP(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert not original.training and all(not p.requires_grad for p in original.parameters())
        self.net = copy.deepcopy(original).eval().requires_grad_(False)
        self.vm_root = None
        for name, value in [('original_head4_0', original.head4.block0.weight),
                ('original_head4_1', original.head4.block1.weight),
                ('original_fusion4', original.smooth[0].weight[:, :64]),
                ('anchor_head4_0', original.head3.block0.weight),
                ('anchor_head4_1', original.head3.block1.weight),
                ('anchor_fusion4', original.smooth[0].weight[:, 64:128])]:
            self.register_buffer(name, value.detach().clone().contiguous())
        with torch.no_grad():
            self.net.head4.block0.weight.copy_(self.anchor_head4_0)
            self.net.head4.block1.weight.copy_(self.anchor_head4_1)
        self.live_fusion4 = nn.Parameter(self.anchor_fusion4.clone(), requires_grad=False)
        self.live_fusion_fine = nn.Parameter(original.smooth[0].weight[:, 64:].detach().clone().contiguous(), requires_grad=False)
        assert sum(p.numel() for p in self.learning_parameters()) == 609219

    def train(self, mode=True):
        return super().train(False)

    def learning_names(self):
        return CURRENT_NAMES

    def learning_parameters(self):
        values = dict(self.named_parameters())
        return [values[n] for n in self.learning_names()]

    def enable_vm_learning(self, root):
        require_vm(root, self.live_fusion4)
        request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                                         headers={'Metadata-Flavor': 'Google'})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.read().decode().rstrip().endswith('/g2-standard-4')
        self.vm_root = Path(root).resolve()
        self.requires_grad_(False)
        for p in self.learning_parameters():
            p.requires_grad_(True)

    @staticmethod
    def fixed_head(x, w0, w1):
        return F.relu(F.conv2d(F.relu(F.conv2d(x, w0, padding=1)), w1, padding=1))

    def observe(self, image):
        if torch.is_grad_enabled():
            require_vm(self.vm_root, self.live_fusion4)
        assert image.dtype == torch.float32 and image.shape[1:] == (3, 256, 256)
        assert not image.requires_grad and torch.isfinite(image).all()
        assert image.min() >= 0 and image.max() <= 1
        assert all(not m.training for m in self.net.modules())
        return self.net.fpn(image * 2 - 1)

    def reconstruct(self, image, maps, extras=None):
        map0, map1, map2, map3, map4 = maps
        base4 = F.interpolate(self.fixed_head(map4, self.original_head4_0, self.original_head4_1), scale_factor=8, mode='nearest')
        live4 = F.interpolate(self.net.head4(map4), scale_factor=8, mode='nearest')
        anchor4 = F.interpolate(self.fixed_head(map4, self.anchor_head4_0, self.anchor_head4_1), scale_factor=8, mode='nearest')
        if extras is not None:
            map1 = map1 + extras['64']
            map2 = map2 + extras['32']
            map3 = map3 + extras['16']
        h3 = F.interpolate(self.net.head3(map3), scale_factor=4, mode='nearest')
        h2 = F.interpolate(self.net.head2(map2), scale_factor=2, mode='nearest')
        h1 = self.net.head1(map1)
        basew = torch.cat([self.original_fusion4, self.live_fusion_fine], dim=1)
        base = F.conv2d(torch.cat([base4, h3, h2, h1], dim=1), basew, self.net.smooth[0].bias, padding=1)
        delta4 = F.conv2d(live4, self.live_fusion4, padding=1) - F.conv2d(anchor4, self.anchor_fusion4, padding=1)
        y = self.net.smooth[2](self.net.smooth[1](base + delta4))
        y = F.interpolate(y, scale_factor=2, mode='nearest')
        context = y + map0
        if extras is not None:
            context = context + extras['128']
        y = self.net.smooth2(context)
        y = F.interpolate(y, scale_factor=2, mode='nearest')
        if extras is not None:
            y = y + extras['256']
        x = image * 2 - 1
        return ((torch.tanh(self.net.final(y)) + x).clamp(-1, 1) + 1) / 2

    def forward(self, image):
        return self.reconstruct(image, self.observe(image))


class ConditionedBankDGP(nn.Module):
    def __init__(self, original, prior, mean_style):
        super().__init__()
        assert not prior.training and all(not p.requires_grad for p in prior.parameters())
        assert mean_style.shape == (512,) and mean_style.dtype == torch.float32
        assert torch.isfinite(mean_style).all()
        self.current = CorrectedCurrentDGP(original)
        self.prior = prior.eval().requires_grad_(False)
        self.vm_root = None
        self.register_buffer('mean_style', mean_style.detach().clone())
        # Fixed, declared random initializer for our new conditioning only.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.style_conditioner = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                nn.LeakyReLU(.2), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                nn.Linear(128 * 16, 256), nn.LeakyReLU(.2), nn.Linear(256, 13 * 512))
            self.spatial_conditioners = nn.ModuleDict({str(size): nn.Conv2d(128, 512, 1)
                for size in [8, 16, 32]})
            self.fusions = nn.ModuleDict({str(size): nn.Conv2d(inp, dest, 1)
                for size, inp, dest in [(16, 512, 128), (32, 512, 128), (64, 256, 128),
                                        (128, 128, 64), (256, 64, 32)]})
            # Spatial conditioning is additive; no learned BN/IN statistics.
            for layer in self.spatial_conditioners.values():
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            for layer in self.fusions.values():
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def learning_names(self):
        return ['current.' + n for n in CURRENT_NAMES] + [n for n, _ in self.named_parameters()
            if n.startswith(('style_conditioner.', 'spatial_conditioners.', 'fusions.'))]

    def learning_parameters(self):
        values = dict(self.named_parameters())
        return [values[n] for n in self.learning_names()]

    def enable_vm_learning(self, root):
        require_vm(root, self.mean_style)
        self.current.enable_vm_learning(root)
        self.vm_root = Path(root).resolve()
        self.requires_grad_(False)
        for p in self.learning_parameters():
            p.requires_grad_(True)
        assert all(not p.requires_grad for p in self.prior.parameters())

    def conditioned_features(self, maps):
        if torch.is_grad_enabled():
            require_vm(self.vm_root, self.mean_style)
        _, map1, map2, map3, map4 = maps
        styles = self.mean_style[None, None] + self.style_conditioner(map4).reshape(-1, 13, 512)
        noises = [getattr(self.prior.noises, f'noise{i}') for i in range(13)]
        out = self.prior.constant_input(len(styles))
        out = self.prior.style_conv1(out, styles[:, 0], noise=noises[0])
        conditioning = {'8': map4, '16': map3, '32': map2}
        features = {}
        # Pretrained feature blocks through256 only. Original RGB/512 blocks remain
        # loaded/frozen and unused; there is no hidden resize or RGB generator tail.
        for index in range(6):
            size = 2 ** (index + 3)
            out = self.prior.style_convs[2 * index](out, styles[:, 1 + 2 * index], noise=noises[1 + 2 * index])
            out = self.prior.style_convs[2 * index + 1](out, styles[:, 2 + 2 * index], noise=noises[2 + 2 * index])
            if str(size) in conditioning:
                out = out + self.spatial_conditioners[str(size)](conditioning[str(size)])
            if size >= 16:
                features[str(size)] = out
        assert set(features) == {'16', '32', '64', '128', '256'}
        return features

    def forward(self, image, bank_enabled=True, return_features=False):
        maps = self.current.observe(image)
        if not bank_enabled:
            return self.current.reconstruct(image, maps)
        features = self.conditioned_features(maps)
        extras = {name: self.fusions[name](value) for name, value in features.items()}
        result = self.current.reconstruct(image, maps, extras)
        return (result, features) if return_features else result

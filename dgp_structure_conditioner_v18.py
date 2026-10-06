"""DGP pixel base plus our spatial residual decoder; local calls are inference-only.

Frozen R2 code predictions supply frozen CodeFormer generator features. The
new learned decoder never substitutes the prior's RGB face for the DGP image.
Initial zero residual is exact DGP parity, not an improved trained result.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def rgb(value):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.ndim != 4 or value.shape[1:] != (3, 256, 256) or not len(value)
            or value.requires_grad or not torch.isfinite(value).all()
            or value.min() < 0 or value.max() > 1):
        raise ValueError('Require detached finite nonempty float32 RGB256 in [0,1]')


def block(inputs, outputs, stride=1):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1),
                         nn.GroupNorm(8, outputs), nn.SiLU())


class DGPStructureResidualHead(nn.Module):
    """Own image/prior fusion with direct DGP pixel skip and full RGB supervision."""

    def __init__(self):
        super().__init__()
        self.image_encoder = nn.ModuleList([block(6, 24, 2), block(24, 48, 2),
                                            block(48, 96, 2), block(96, 128, 2)])
        self.prior_projection = nn.Sequential(nn.Conv2d(256, 32, 1),
                                             nn.GroupNorm(8, 32), nn.SiLU())
        self.fuse16 = block(160, 128)
        self.decode32 = block(224, 96)
        self.decode64 = block(176, 64)
        self.decode128 = block(88, 32)
        self.decode256 = block(38, 16)
        self.final = nn.Conv2d(16, 3, 3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        self._training_root = None
        nn.Module.train(self, False)
        self.requires_grad_(False)

    def train(self, mode=True):
        if mode:
            raise RuntimeError('Use enable_vm_training(root); local training forbidden')
        nn.Module.train(self, False)
        self.requires_grad_(False)
        self._training_root = None
        return self

    def enable_vm_training(self, root):
        from cctv_dgp_targets_v6 import require_vm
        require_vm(root)
        self._training_root = Path(root).resolve()
        nn.Module.train(self, True)
        self.requires_grad_(True)
        return self

    def forward(self, camera, dgp_base, prior64, *, prior_ablation=False):
        if self.training or any(p.requires_grad for p in self.parameters()):
            if self._training_root is None:
                raise RuntimeError('Trainable forward requires explicit existing-L4 VM authorization')
            from cctv_dgp_targets_v6 import require_vm
            require_vm(self._training_root)
        rgb(camera)
        rgb(dgp_base)
        if (not isinstance(prior64, torch.Tensor)
                or prior64.shape != (len(camera), 256, 64, 64) or prior64.dtype != torch.float32
                or prior64.requires_grad or not torch.isfinite(prior64).all()
                or camera.device != dgp_base.device or camera.device != prior64.device
                or len(camera) != len(dgp_base)):
            raise ValueError('Require detached finite matched N x256 x64 x64 prior features')
        image = torch.cat([camera, dgp_base], 1)
        features = []
        x = image
        for layer in self.image_encoder:
            x = layer(x)
            features.append(x)
        # Ablation removes prior input only, retaining the same trained decoder.
        prior = self.prior_projection(torch.zeros_like(prior64) if prior_ablation else prior64)
        x = self.fuse16(torch.cat([features[3], F.avg_pool2d(prior, 4)], 1))
        for size, feature, layer in [(32, features[2], self.decode32),
                                     (64, features[1], self.decode64),
                                     (128, features[0], self.decode128)]:
            x = F.interpolate(x, (size, size), mode='bilinear', align_corners=False)
            parts = [x, feature] + ([prior] if size == 64 else [])
            x = layer(torch.cat(parts, 1))
        x = self.decode256(torch.cat([F.interpolate(x, (256, 256), mode='bilinear',
                                                    align_corners=False), image], 1))
        raw = (dgp_base + .5 * self.final(x).tanh()).clamp(0, 1)
        if not torch.isfinite(raw).all():
            raise FloatingPointError('Nonfinite DGP structure residual')
        return raw


@torch.no_grad()
def prior_features64(prior, logits):
    """Freeze the declared prior and stop at its exact64 feature interface.

No AdaIN, original-input feature fusion, RGB decoder tail or target/teacher is
used. The R2 code head is frozen; RGB loss trains only our new residual decoder.
"""
    if (prior.training or any(p.requires_grad for p in prior.parameters())
            or logits.dtype != torch.float32 or logits.ndim != 3
            or logits.shape[1:] != (256, 1024) or not len(logits)
            or logits.requires_grad or not torch.isfinite(logits).all()):
        raise ValueError('Require frozen prior and detached finite R2 logits')
    x = prior.quantize.get_codebook_feat(logits.argmax(2), shape=[len(logits), 16, 16, 256])
    last = prior.fuse_generator_block['64']
    for index, layer in enumerate(prior.generator.blocks):
        x = layer(x)
        if index == last:
            if x.shape != (len(logits), 256, 64, 64) or not torch.isfinite(x).all():
                raise ValueError('Frozen CodeFormer feature64 schema differs')
            return x.detach()
    raise ValueError('Frozen generator feature64 interface missing')

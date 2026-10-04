"""Our direct code/statistics conditioner with a declared frozen face prior.

V14 changes code logits directly instead of backpropagating feature regression
through the frozen CodeFormer classifier. Only our two-head conditioner is
trainable. All training graphs/backward preflights require the existing L4 VM.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dgp_face_code_conditioner_v11 import DGPFaceCodePrior, check_rgb, from_prior_canvas
from third_party.codeformer.codeformer_arch import calc_mean_std


class DirectCodeConditioner(nn.Module):
    """Predict residual code logits and clean mean/log-standard-deviation."""

    def __init__(self):
        super().__init__()
        layers = []
        for a, b in zip([6, 32, 64, 128], [32, 64, 128, 256]):
            layers.extend([nn.Conv2d(a, b, 3, stride=2, padding=1), nn.SiLU()])
        self.image_features = nn.Sequential(*layers)
        self.fusion = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(256, 256, 3, padding=1), nn.SiLU())
        self.code_projection = nn.Conv2d(256, 1024, 1)
        self.stats_hidden = nn.Sequential(nn.Linear(256, 256), nn.SiLU())
        self.stats_projection = nn.Linear(256, 512)
        for layer in [self.code_projection, self.stats_projection]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self._training_root = None
        nn.Module.train(self, False)
        self.requires_grad_(False)

    def train(self, mode=True):
        if mode:
            raise RuntimeError('Use enable_vm_training(root); local training is forbidden')
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

    def forward(self, image, dgp_image, original_features, base_logits):
        if self.training or any(p.requires_grad for p in self.parameters()):
            from cctv_dgp_targets_v6 import require_vm
            if self._training_root is None:
                raise RuntimeError('Direct code graph requires explicit VM authorization')
            require_vm(self._training_root)
        check_rgb(image)
        check_rgb(dgp_image)
        n = len(image)
        if (original_features.shape != (n, 256, 16, 16) or
                base_logits.shape != (n, 256, 1024) or
                not torch.isfinite(original_features).all() or
                not torch.isfinite(base_logits).all() or
                any(v.dtype != torch.float32 or v.requires_grad for v in
                    [image, dgp_image, original_features, base_logits])):
            raise ValueError('Expected detached finite float32 cached inputs/features/logits')
        x = self.image_features(torch.cat([image, dgp_image], 1))
        x = self.fusion(torch.cat([original_features, x], 1))
        logits = base_logits + self.code_projection(x).flatten(2).transpose(1, 2)
        delta = self.stats_projection(self.stats_hidden(x.mean((2, 3))))
        mean, std = calc_mean_std(original_features)
        mean, std = mean.flatten(1), std.flatten(1)
        predicted_mean = mean + delta[:, :256]
        predicted_logstd = std.log() + delta[:, 256:]
        # Multiplicative delta preserves the exact observed std at initialization.
        predicted_std = std * torch.exp(delta[:, 256:])
        return {'logits': logits, 'mean': predicted_mean, 'logstd': predicted_logstd,
                'std': predicted_std, 'observed_mean': mean, 'observed_std': std}


@torch.inference_mode()
def render_codes(prior, codes, *, mean=None, std=None):
    """Frozen w0 renderer; statistics may be observed, predicted or omitted."""
    if (codes.dtype != torch.int64 or codes.ndim != 2 or codes.shape[1] != 256 or
            not len(codes) or codes.min() < 0 or codes.max() >= 1024):
        raise ValueError('Expected integer codes N x 256 in [0,1023]')
    if (mean is None) != (std is None):
        raise ValueError('Both rendering statistics are required together')
    x = prior.quantize.get_codebook_feat(codes, [len(codes), 16, 16, 256])
    if mean is not None:
        if (mean.shape != (len(codes), 256) or std.shape != mean.shape or
                not torch.isfinite(mean).all() or not torch.isfinite(std).all() or
                std.min() <= 0):
            raise ValueError('Invalid finite positive rendering statistics')
        cm, cs = calc_mean_std(x)
        x = (x - cm) / cs * std[:, :, None, None] + mean[:, :, None, None]
    for block in prior.generator.blocks:
        x = block(x)
    return from_prior_canvas(x)


class DGPDirectFaceCodePrior(nn.Module):
    """Our trainable conditioner; original DGP and CodeFormer remain frozen.

    The prior encoder always sees the original crop. Frozen DGP RGB supplies
    extra conditioning channels. No clean target or teacher is needed at inference.
    """

    def __init__(self, dgp_net, prior_net):
        super().__init__()
        self.core = DGPFaceCodePrior(dgp_net, prior_net)
        self.conditioner = DirectCodeConditioner()
        nn.Module.train(self, False)

    def train(self, mode=True):
        if mode:
            raise RuntimeError('Use conditioner.enable_vm_training(root) on the existing VM')
        nn.Module.train(self, False)
        return self

    @torch.no_grad()
    def frozen_inputs(self, image):
        check_rgb(image)
        dgp = self.core.dgp(image).clamp(0, 1)
        features, _ = self.core.encode(image)
        logits = self.core.logits(features)
        return dgp, features, logits

    @torch.inference_mode()
    def forward(self, image, statistics='predicted'):
        if statistics not in ['observed', 'none', 'predicted']:
            raise ValueError('Unknown rendering-statistics mode')
        dgp, features, logits = self.frozen_inputs(image)
        predicted = self.conditioner(image, dgp, features, logits)
        if not all(torch.isfinite(predicted[k]).all() for k in ['logits','mean','logstd','std']):
            raise FloatingPointError('Nonfinite direct conditioning prediction')
        codes = predicted['logits'].argmax(2)
        if statistics == 'none':
            return render_codes(self.core.prior, codes)
        mean = predicted['mean'] if statistics == 'predicted' else predicted['observed_mean']
        std = predicted['std'] if statistics == 'predicted' else predicted['observed_std']
        return render_codes(self.core.prior, codes, mean=mean, std=std)

"""Our feature conditioner with a declared frozen CodeFormer face prior.

The pretrained source/weights are retained separately under third_party/codeformer.
This interface adapts its encoder -> transformer -> codebook -> decoder sequence
without modifying that source. Local calls are inference-only. Training-code
graphs and backward preflights require the existing forensic-dgp-thesis L4 VM.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from third_party.codeformer.codeformer_arch import adaptive_instance_normalization


class FaceCodeConditioner(nn.Module):
    """Learn a residual at the prior's 16x16 face-code feature grid."""

    def __init__(self):
        super().__init__()
        channels = [6, 32, 64, 128, 256]
        layers = []
        for a, b in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(a, b, 3, stride=2, padding=1), nn.SiLU()])
        self.features = nn.Sequential(*layers)
        self.projection = nn.Conv2d(256, 256, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, image, dgp_image):
        return self.projection(self.features(torch.cat([image, dgp_image], 1)))


def check_rgb(image):
    if (image.dtype != torch.float32 or image.ndim != 4 or
            image.shape[1:] != (3, 256, 256) or not len(image) or
            not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError('Expected finite nonempty float32 RGB N x 3 x 256 x 256 in [0,1]')


def to_prior_canvas(image):
    return F.interpolate(image, (512, 512), mode='bilinear', align_corners=False) * 2 - 1


def from_prior_canvas(image):
    if image.ndim != 4 or image.shape[1:] != (3, 512, 512) or not torch.isfinite(image).all():
        raise FloatingPointError('Invalid prior decoder output')
    return F.interpolate(((image + 1) / 2).clamp(0, 1), (256, 256),
                         mode='bilinear', align_corners=False)


class DGPFaceCodePrior(nn.Module):
    """Frozen DGP signal features + frozen face prior + our trainable conditioner.

    The prior encoder always receives the original crop. DGP RGB supplies three
    extra conditioning channels; it is not substituted for the prior input.
    The residual changes code prediction only. Original prior features supply
    AdaIN and fidelity connections, keeping the rendering interface explicit.
    """

    def __init__(self, dgp_net, prior_net):
        super().__init__()
        self.dgp = dgp_net.eval().requires_grad_(False)
        self.prior = prior_net.eval().requires_grad_(False)
        self.conditioner = FaceCodeConditioner().requires_grad_(False)
        self._training_root = None
        nn.Module.train(self, False)

    def train(self, mode=True):
        if mode:
            raise RuntimeError('Use enable_vm_training(root); local training is forbidden')
        nn.Module.train(self, False)
        self.conditioner.requires_grad_(False)
        self._training_root = None
        return self

    def enable_vm_training(self, root):
        # Before enabling parameters or constructing any differentiable graph.
        from cctv_dgp_targets_v6 import require_vm
        require_vm(root)
        self._training_root = Path(root).resolve()
        nn.Module.train(self, False)
        self.conditioner.train(True).requires_grad_(True)
        return self

    def trainable_parameters(self):
        return [p for p in self.conditioner.parameters() if p.requires_grad]

    def encode(self, image):
        """Frozen input encoder; no inference tensors enter a training graph."""
        with torch.no_grad():
            x = to_prior_canvas(image)
            skips = {}
            wanted = [self.prior.fuse_encoder_block[s] for s in self.prior.connect_list]
            for i, block in enumerate(self.prior.encoder.blocks):
                x = block(x)
                if i in wanted:
                    skips[str(x.shape[-1])] = x.clone()
        if x.shape[1:] != (256, 16, 16):
            raise ValueError('CodeFormer encoder feature schema differs')
        return x, skips

    def logits(self, features):
        pos = self.prior.position_emb.unsqueeze(1).repeat(1, len(features), 1)
        query = self.prior.feat_emb(features.flatten(2).permute(2, 0, 1))
        for layer in self.prior.ft_layers:
            query = layer(query, query_pos=pos)
        return self.prior.idx_pred_layer(query).permute(1, 0, 2)

    def training_codes(self, image):
        # Check again at the graph entry, even after an earlier VM preflight.
        from cctv_dgp_targets_v6 import require_vm
        if self._training_root is None:
            raise RuntimeError('Training-code graph requires explicit VM authorization')
        require_vm(self._training_root)
        check_rgb(image)
        if not self.conditioner.training or not self.trainable_parameters():
            raise RuntimeError('VM conditioner parameters are not enabled')
        with torch.no_grad():
            dgp = self.dgp(image).clamp(0, 1)
        features, _ = self.encode(image)
        conditioned = features + self.conditioner(image, dgp)
        return self.logits(conditioned), conditioned

    @torch.inference_mode()
    def decode(self, codes, original_features=None, skips=None, *, fidelity=0.0):
        if not 0 <= fidelity <= 1:
            raise ValueError('Fidelity must be in [0,1]')
        if (codes.dtype != torch.int64 or codes.ndim != 2 or codes.shape[1] != 256 or not len(codes) or
                codes.min() < 0 or codes.max() >= 1024):
            raise ValueError('Expected integer face codes N x 256 in [0,1023]')
        x = self.prior.quantize.get_codebook_feat(codes, shape=[len(codes), 16, 16, 256])
        if original_features is not None:
            x = adaptive_instance_normalization(x, original_features)
        wanted = [self.prior.fuse_generator_block[s] for s in self.prior.connect_list]
        for i, block in enumerate(self.prior.generator.blocks):
            x = block(x)
            if i in wanted and fidelity > 0:
                size = str(x.shape[-1])
                x = self.prior.fuse_convs_dict[size](skips[size].detach(), x, fidelity)
        return from_prior_canvas(x)

    @torch.inference_mode()
    def forward(self, image, fidelity=1.0):
        check_rgb(image)
        if not 0 <= fidelity <= 1:
            raise ValueError('Fidelity must be in [0,1]')
        dgp = self.dgp(image).clamp(0, 1)
        original, skips = self.encode(image)
        conditioned = original + self.conditioner(image, dgp)
        logits = self.logits(conditioned)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite conditioned code logits')
        # Same softmax/top-1 convention as the pinned original, for zero-delta parity.
        indices = torch.topk(F.softmax(logits, dim=2), 1, dim=2)[1].squeeze(-1)
        return self.decode(indices, original, skips, fidelity=fidelity)

    @torch.inference_mode()
    def clean_code_oracle(self, codes):
        """Capacity diagnostic only: clean teacher codes, no target skips/AdaIN."""
        return self.decode(codes, fidelity=0.0)


@torch.no_grad()
def teacher_codes(teacher, clean_image):
    """Official clean-image VQGAN teacher; this never receives native CCTV truth."""
    check_rgb(clean_image)
    features = teacher.encoder(to_prior_canvas(clean_image))
    quantized, _, statistics = teacher.quantize(features)
    codes = statistics['min_encoding_indices'].view(len(clean_image), 256)
    if quantized.shape[1:] != (256, 16, 16) or not torch.isfinite(quantized).all():
        raise FloatingPointError('Invalid clean teacher representation')
    return codes, quantized


def load_teacher(path, expected_sha256, device='cpu'):
    from dgp_face_restoration import sha
    from third_party.codeformer.vqgan_arch import VQAutoEncoder
    if sha(path) != expected_sha256:
        raise ValueError('Clean teacher checkpoint fingerprint differs')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'params_ema' not in state:
        raise ValueError('Expected official teacher params_ema')
    net = VQAutoEncoder(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], 1024)
    net.load_state_dict(state['params_ema'], strict=True)
    if not all(torch.isfinite(v).all() for v in net.state_dict().values()):
        raise ValueError('Nonfinite clean teacher parameters')
    return net.to(device).eval().requires_grad_(False)

"""Direct covered-region logits from a separately initialized face U-Net.

No pretrained weights are downloaded and no training starts on import. The
original visible-face head is retained for provenance, never used as an occluder.
"""
import copy
from collections.abc import Mapping

import torch
from torch import nn

FORMAT = 'dgp-face-occlusion-adapter-v1'
TARGET = 'covered_region_is_one'
SOURCE_SHA = '01d3c3939c28e47a45acb9a5ea8f8ee460e5ecf046c0caa8404e05915e10901b'
SMP_VERSION = '0.5.0'


def strip_source_prefix(state):
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Require a nonempty source tensor state')
    if any(not isinstance(k, str) or not k.startswith('module.') or
           not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
           for k, v in state.items()):
        raise ValueError('Source must contain finite tensors with module. prefix')
    return {k.removeprefix('module.'): v for k, v in state.items()}


def make_network():
    import segmentation_models_pytorch as smp
    if smp.__version__ != SMP_VERSION:
        raise ValueError(f'Require segmentation-models-pytorch {SMP_VERSION}')
    return smp.Unet(encoder_name='resnet18', encoder_weights=None, classes=1, activation=None)


class FaceOcclusionAdapter(nn.Module):
    target = TARGET

    def __init__(self, network, new_head):
        super().__init__()
        if not all(hasattr(network, k) for k in ('encoder', 'decoder', 'segmentation_head')):
            raise ValueError('Require an encoder/decoder segmentation network')
        self.network = network
        self.reference_visible_head = copy.deepcopy(network.segmentation_head)
        self.reference_visible_head.requires_grad_(False).eval()
        self.network.segmentation_head = new_head
        self.network.requires_grad_(True)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.reference_visible_head.eval()
        # Fixed statistics avoid small mixed-batch drift; affine weights remain
        # trainable in both initialization arms. Control statistics stay at defaults.
        for module in self.network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def detect(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] < 1 or not image.is_floating_point():
            raise ValueError('Require a nonempty floating RGB NCHW batch')
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Require finite RGB values in [0,1]')
        logits = self.network(image)
        if logits.shape != (image.shape[0], 1, *image.shape[2:]) or not torch.isfinite(logits).all():
            raise ValueError('Require finite direct occlusion logits with matching N1HW shape')
        return logits

    def forward(self, image):
        return self.detect(image)


def initialize_pair(source, *, factory=None, seed=42):
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('Invalid initialization seed')
    state = strip_source_prefix(source)
    factory = make_network if factory is None else factory
    # Use the CPU generator only; no mutation of the caller's CUDA RNG or downloads.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        prototype = factory()
        random = copy.deepcopy(prototype)
        pretrained = copy.deepcopy(prototype)
        pretrained.load_state_dict(state, strict=True)
        torch.default_generator.manual_seed(seed + 1)
        head = copy.deepcopy(prototype.segmentation_head)
        convolutions = [m for m in head.modules() if isinstance(m, nn.Conv2d)]
        if len(convolutions) != 1 or convolutions[0].out_channels != 1:
            raise ValueError('Require a single-logit convolution head')
        convolutions[0].reset_parameters()
        nn.init.zeros_(convolutions[0].bias)
        return {'pretrained': FaceOcclusionAdapter(pretrained, copy.deepcopy(head)),
                'random': FaceOcclusionAdapter(random, copy.deepcopy(head))}


def parameter_groups(model):
    encoder = list(model.network.encoder.parameters())
    decoder_head = list(model.network.decoder.parameters()) + list(model.network.segmentation_head.parameters())
    if {id(p) for p in encoder + decoder_head} != {id(p) for p in model.network.parameters()}:
        raise ValueError('Unexpected network parameters outside encoder/decoder/head')
    return [{'params': encoder, 'lr': 1e-5}, {'params': decoder_head, 'lr': 1e-4}]


def load_adapter(path, device='cpu', *, allow_initial=False, factory=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(payload, Mapping) or payload.get('format') != FORMAT or
        payload.get('target') != TARGET or payload.get('architecture') != 'resnet18-unet' or
        payload.get('smp_version') != SMP_VERSION or payload.get('initialization') not in ('pretrained','random')):
        raise ValueError('Unexpected versioned direct-occlusion checkpoint')
    updates = payload.get('optimizer_updates')
    if not isinstance(updates, int) or updates < 0 or (not allow_initial and updates < 1):
        raise ValueError('Checkpoint requires completed optimizer updates')
    state = payload.get('model')
    if not isinstance(state, Mapping) or not state or any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Require finite adapter state tensors')
    network = (make_network if factory is None else factory)()
    model = FaceOcclusionAdapter(network, copy.deepcopy(network.segmentation_head))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload

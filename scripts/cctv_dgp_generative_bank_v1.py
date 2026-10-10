"""Declared external generative bank; no trained CCTV conditioner is included.

Pure PyTorch resampling preserves the pinned BasicSR CPU reference. GPU speed
and numerical parity require a separate manual L4 check before any training.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import sys

import torch


NAME = 'StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require_manual_vm():
    if platform.system() != 'Linux' or not os.environ.get('TMUX'):
        raise RuntimeError('Gradient-enabled generative-bank work requires manual VM tmux')
    if not Path.cwd().resolve().is_relative_to(Path.home() / 'forensic-dgp'):
        raise RuntimeError('Gradient-enabled work must stay in ~/forensic-dgp')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Gradient-enabled work requires the declared NVIDIA L4')


def load_prior(source_root, implementation_root, expected_sha):
    source_root, implementation_root = Path(source_root), Path(implementation_root)
    checkpoint = source_root / NAME
    if sha(checkpoint) != expected_sha:
        raise ValueError('Standalone generator checkpoint fingerprint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if set(state) != {'params', 'params_ema'} or len(state['params_ema']) != 139:
        raise ValueError('Expected official standalone generator schema')
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) and
               torch.isfinite(v).all() for k, v in state['params_ema'].items()):
        raise ValueError('Nonfinite or unsupported standalone prior tensors')
    package = '_cctv_dgp_generative_bank_reference_v1'
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, implementation_root / '__init__.py',
            submodule_search_locations=[str(implementation_root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    module = __import__(package + '.generator', fromlist=['StyleGAN2Generator'])
    prior = module.StyleGAN2Generator(512, num_style_feat=512, num_mlp=8,
        channel_multiplier=1, resample_kernel=(1, 3, 3, 1), lr_mlp=.01)
    prior.load_state_dict(state['params_ema'], strict=True)
    prior.eval().requires_grad_(False)
    if sha(checkpoint) != expected_sha:
        raise ValueError('Checkpoint changed during loading')
    return prior


def bank_features(prior, z):
    """Return the original RGB path and internal16–256 features, not restoration.

    No input image, clean target, codebook, RGB-difference correction, fidelity
    mixer or pretrained restoration encoder is used. The future conditioner
    needs its own verified training and preservation evidence.
    """
    if torch.is_grad_enabled():
        require_manual_vm()
    if z.shape != (1, 512) or z.dtype != torch.float32 or not torch.isfinite(z).all():
        raise ValueError('This bounded bank probe uses one finite float32 Z512')
    if any(p.requires_grad for p in prior.parameters()) or prior.training:
        raise ValueError('External generative prior must remain frozen/eval')
    style = prior.get_latent(z)
    latent = style[:, None].repeat(1, prior.num_latent, 1)
    noise = [getattr(prior.noises, f'noise{i}') for i in range(prior.num_layers)]
    out = prior.constant_input(1)
    out = prior.style_conv1(out, latent[:, 0], noise=noise[0])
    skip = prior.to_rgb1(out, latent[:, 1])
    features = {}
    index = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(prior.style_convs[::2],
            prior.style_convs[1::2], noise[1::2], noise[2::2], prior.to_rgbs):
        out = conv1(out, latent[:, index], noise=noise1)
        out = conv2(out, latent[:, index + 1], noise=noise2)
        skip = to_rgb(out, latent[:, index + 2], skip)
        resolution = out.shape[-1]
        if resolution in [16, 32, 64, 128, 256]:
            features[str(resolution)] = out
        index += 2
    return skip, features

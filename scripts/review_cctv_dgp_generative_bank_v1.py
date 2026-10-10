"""Finite local inference prerequisite: source, geometry and frozen-state checks.

Uses two fixed generator seeds and one repeat, no CCTV or photographic images.
Acquisition and generated faces are not own-trained restoration evidence.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from cctv_dgp_generative_bank_v1 import NAME, bank_features, load_prior, sha


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def tensor_state(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        data = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(data.dtype).encode())
        digest.update(str(data.shape).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def prepare_implementation(source, destination):
    """Keep generator source body exact; replace only imports/registration.

    Fused activation uses the bias/leaky-ReLU/sqrt2 formula in PyTorch.
    Resampling uses the upstream CPU reference body verbatim. No bilinear
    substitute or clean-generator conversion is applied to the checkpoint.
    """
    destination.mkdir()
    original = (source / 'stylegan2_arch.py').read_text(encoding='utf-8')
    tree = ast.parse(original)
    generator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StyleGAN2Generator')
    text = '\n'.join(original.splitlines()[:generator.end_lineno]) + '\n'
    for line in ['from basicsr.ops.fused_act import FusedLeakyReLU, fused_leaky_relu\n',
                 'from basicsr.ops.upfirdn2d import upfirdn2d\n',
                 'from basicsr.utils.registry import ARCH_REGISTRY\n',
                 '@ARCH_REGISTRY.register()\n']:
        assert text.count(line) == 1
        text = text.replace(line, '')
    header = '# Adapted from pinned BasicSR for finite CPU reference inference.\n'
    header += '# Changes: imports/registration, pure PyTorch operator adapters, omitted discriminator.\n'
    header += '# Retain LICENSE_BasicSR.txt and LICENSE_StyleGAN2_NVIDIA.txt with this component.\n'
    header += 'from .native_ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d\n'
    (destination / 'generator.py').write_text(header + text, encoding='utf-8', newline='\n')
    source_ops = (source / 'upfirdn2d.py').read_text(encoding='utf-8')
    native = next(n for n in ast.parse(source_ops).body if isinstance(n, ast.FunctionDef) and n.name == 'upfirdn2d_native')
    body = ast.get_source_segment(source_ops, native)
    ops = """# Pure PyTorch adapters; native filter function below is retained verbatim from BasicSR.
import torch
from torch import nn
from torch.nn import functional as F

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    shape = [1, -1] + [1] * (input.ndim - 2)
    return F.leaky_relu(input + bias.view(*shape), negative_slope=negative_slope) * scale

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope, self.scale = negative_slope, scale
    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

"""
    (destination / 'native_ops.py').write_text(ops + body + '\n', encoding='utf-8', newline='\n')
    (destination / '__init__.py').write_text('# Standalone generative bank reference only.\n', encoding='utf-8')
    for name in ['LICENSE_BasicSR.txt', 'LICENSE_GFPGAN.txt', 'LICENSE_StyleGAN2_NVIDIA.txt']:
        shutil.copyfile(source / name, destination / name)


def run(root, source):
    root, source = root.resolve(), source.resolve()
    workspace = Path(__file__).resolve().parents[1]
    assert root.is_relative_to(workspace / 'outputs') and not root.exists()
    acquisition = json.loads((source / 'acquisition.json').read_text(encoding='utf-8'))
    assert acquisition['complete'] and acquisition['neural_calls'] == 0
    for name, record in acquisition['bindings'].items():
        assert sha(source / name) == record['sha256']
    root.mkdir(parents=True)
    checkpoint_sha = acquisition['bindings'][NAME]['sha256']
    baseline = workspace / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    assert sha(baseline) == '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    protected_paths = [baseline, workspace / 'models/dgp_synthesizer.py', workspace / 'models/fpn_mobilenet.py',
        workspace / 'dgp_face_restoration.py', workspace / 'cctv_dgp_frozen_norm.py',
        workspace / 'CCTV_DGP_MULTISCALE_CALIBRATION_V1_RESULTS.md',
        workspace / 'outputs/cctv_dgp_multiscale_calibration_vm_v1/protocol.json']
    protected = {p.relative_to(workspace).as_posix(): sha(p) for p in protected_paths}
    plan = {'format': 'standalone-generative-bank-local-prerequisite-v1',
        'seeds': [0, 1, 0], 'CPU_threads': 4, 'worker_seconds_cap': 1200,
        'prior_generator_forward_limit': 6, 'current_DGP_forward_limit': 0,
        'source_root': str(source), 'source_acquisition_sha256': sha(source / 'acquisition.json'),
        'standalone_prior_sha256': checkpoint_sha, 'prior_parameter_container': 'params_ema',
        'prior_output_size': 512, 'display_output_size': 256, 'prior_feature_sizes': [16, 32, 64, 128, 256],
        'expected_feature_channels': [512, 512, 256, 128, 64],
        'stock_vs_feature_wrapper_max_error': 0.0, 'repeat_max_error': 0.0,
        'raw_PNG_policy': 'clip(-1,1), map[0,1]; torch bilinear256; floor255',
        'paired_TRAIN_images': 0, 'native_DEV_images': 0, 'reserved_final_images': 0,
        'optimizer_updates': 0, 'backwards': 0, 'gradient_queries': 0,
        'output_bytes_cap': 200 * 1024**2, 'local_disk_reserve_bytes': 1024**3,
        'generated_faces_are_restoration_evidence': False, 'app_promotion': False,
        'protected_before': protected}
    write(root / 'plan.json', plan)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    calls = 0

    def clock():
        assert time.monotonic() - start < 1200, 'Finite1200s local inference stop'
        assert shutil.disk_usage(root).free >= 1024**3, 'Local disk1GiB reserve stop'
        assert sum(f.stat().st_size for f in root.rglob('*') if f.is_file()) < 200 * 1024**2, 'Finite200MiB output stop'

    try:
        implementation = root / 'implementation'
        prepare_implementation(source, implementation)
        prior = load_prior(source, implementation, checkpoint_sha)
        before = tensor_state(prior.state_dict())
        rows = []
        tiles = []
        first_raw = None
        with torch.inference_mode():
            for index, seed in enumerate(plan['seeds']):
                clock()
                z = torch.randn(1, 512, generator=torch.Generator().manual_seed(seed))
                stock = prior([z], randomize_noise=False)[0]
                calls += 1
                raw, features = bank_features(prior, z)
                calls += 1
                assert stock.shape == raw.shape == (1, 3, 512, 512)
                assert torch.equal(stock, raw), 'Feature extraction changed original RGB path'
                assert torch.isfinite(raw).all()
                for resolution, channels in zip(plan['prior_feature_sizes'], plan['expected_feature_channels']):
                    feature = features[str(resolution)]
                    assert feature.shape == (1, channels, resolution, resolution) and torch.isfinite(feature).all()
                    if index < 2:
                        np.save(root / f'seed{seed}_feature{resolution}.npy', feature.cpu().numpy(), allow_pickle=False)
                if index == 0:
                    first_raw = raw.clone()
                elif index == 2:
                    assert torch.equal(raw, first_raw), 'Fixed-noise seed replay changed'
                np.save(root / f'call{index}_z.npy', z.cpu().numpy(), allow_pickle=False)
                np.save(root / f'call{index}_raw512.npy', raw[0].permute(1, 2, 0).cpu().numpy(), allow_pickle=False)
                display = F.interpolate(((raw.clamp(-1, 1) + 1) / 2), (256, 256), mode='bilinear', align_corners=False)
                array = display[0].permute(1, 2, 0).cpu().numpy()
                np.save(root / f'call{index}_display256.npy', array, allow_pickle=False)
                png = np.floor(array * 255).astype(np.uint8)
                Image.fromarray(png).save(root / f'call{index}_display256.png')
                if index < 2:
                    tiles.append(png)
                rows.append({'call': index, 'seed': seed, 'stock_wrapper_max_error': float((stock - raw).abs().max()),
                    'raw_range': [float(raw.min()), float(raw.max())],
                    'feature_shapes': {k: list(v.shape) for k, v in features.items()}})
                print(json.dumps({'seed': seed, 'prior_forwards': calls, 'seconds': time.monotonic() - start}), flush=True)
        assert calls == 6 and tensor_state(prior.state_dict()) == before
        Image.fromarray(np.concatenate(tiles, axis=1)).save(root / 'generated_prior_seeds_not_restoration.png')
        protected_after = {p.relative_to(workspace).as_posix(): sha(p) for p in protected_paths}
        assert protected_after == protected
        clock()
        write(root / 'results.json', {'complete': True, 'plan_sha256': sha(root / 'plan.json'),
            'seconds': time.monotonic() - start, 'prior_generator_forwards': calls,
            'source_state_sha256': before, 'frozen_state_unchanged': True,
            'unique_parameter_tensors': len(list(prior.parameters())),
            'unique_parameter_elements': sum(p.numel() for p in prior.parameters()),
            'trainable_parameter_elements': sum(p.numel() for p in prior.parameters() if p.requires_grad),
            'state_entries': len(prior.state_dict()), 'rows': rows, 'repeat_raw_max_error': 0.0,
            'protected_after': protected_after, 'torch': torch.__version__, 'device': 'cpu',
            'CUDA_parity_verified': False, 'optimizer_updates': 0, 'backwards': 0,
            'gradient_queries': 0, 'restoration_qualification': False, 'goal_complete': False})
        write(root / 'manifest.json', {f.relative_to(root).as_posix(): sha(f)
            for f in sorted(root.rglob('*')) if f.is_file()})
    except Exception as exc:
        write(root / 'failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'seconds': time.monotonic() - start, 'prior_forwards': calls, 'optimizer_updates': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.source)

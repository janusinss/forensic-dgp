"""Frozen CPU-only Conditioned SFT V1 inference adapter; inert until explicit loading.

Supports both:
1. Baseline 575-tensor initialization (`R2` + `U96` + `terminal_u0064_heads.npz`)
2. Trained 60-tensor Conditioned SFT V1 Arm-B `trainable.npz` (`24` `U96` pre-SFT
   adapters + `36` post-fusion spatial heads, `515` `R2` tensors frozen)

Applies input-only exposure/degradation conditioning (`g_dark(x)`, `g_deg(x)`) and
16x16 coarse block anchoring (`beta = 0.005`) on normal-lit inputs while preserving
zero-mean spatial detail (`pred - B16(pred)`) across eyes, brows, nose, mouth, and jaw.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import threading
import weakref
import zipfile

from research.codeformer_conditioned_sft_learning_v1 import conditioning

ROOT = Path(__file__).resolve().parents[1]
SITES = ('32', '64', '128', '256')
CHANNELS = {'32': 256, '64': 256, '128': 128, '256': 128}
WIDTH = 32
R2_FILE = {'bytes': 376609465, 'sha256': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'}
U96_FILE = {'bytes': 212827, 'sha256': '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'}
HEAD_FILE = {'bytes': 685818, 'sha256': '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'}
R2_STATE = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_STATE = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
HEAD_STATE = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
PROTOCOL_SHA = '273b8367fbe645206b4684cddeee6cbdc4435fd0587250855424d229d3843c53'
ARCHIVE_SHA = '168d8c97d9352e381b69d531651b8e223e9440683b68b6a7e18e5812330a34e8'
_OWNER_ATTRIBUTE = '_forensic_conditioned_sft_exclusive_owner'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ordinary_file(path, maximum=512 * 1024**2):
    path = Path(path).absolute()
    for parent in path.parents:
        info = parent.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary source/artifact parent')
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_nlink == 1
         and not getattr(info, 'st_file_attributes', 0) & 1024 and 0 < info.st_size <= maximum,
         'Bounded unlinked ordinary source/artifact required')
    return path


def file_pin(path, maximum=512 * 1024**2):
    path = ordinary_file(path, maximum)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024**2), b''):
            h.update(b)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino),
         'Source/artifact changed during hash read')
    return {'bytes': before.st_size, 'sha256': h.hexdigest()}


def adapter_shapes():
    return {
        'adapters.' + site + '.' + name: shape
        for site, c in CHANNELS.items()
        for name, shape in {
            'down.weight': (32, c, 1, 1), 'down.bias': (32,),
            'depthwise.weight': (32, 1, 3, 3), 'depthwise.bias': (32,),
            'up.weight': (c, 32, 1, 1), 'up.bias': (c,),
        }.items()
    }


def head_shapes():
    out = {}
    for site, c in CHANNELS.items():
        for name, shape in {
            'encoder.weight': (32, c, 1, 1), 'encoder.bias': (32,),
            'decoder.weight': (32, c, 1, 1), 'fused.weight': (32, c, 1, 1),
            'coarse.weight': (32, 256, 1, 1), 'spatial.weight': (32, 32, 3, 3),
            'spatial.bias': (32,), 'output.weight': (c, 32, 1, 1), 'output.bias': (c,),
        }.items():
            out['heads.' + site + '.' + name] = shape
    need(len(out) == 36 and sum(math.prod(s) for s in out.values()) == 168960, 'Exact36/168960 head schema')
    return out


def trainable_60_shapes():
    combined = {**adapter_shapes(), **head_shapes()}
    need(len(combined) == 60 and sum(math.prod(s) for s in combined.values()) == 220288,
         'Exact60/220288 conditioned-sft Arm-B trainable schema')
    return combined


def inspect_trainable_npz(raw):
    """Parse either a 36-tensor head NPZ or a 60-tensor Arm-B trainable NPZ."""
    need(type(raw) is bytes and 0 < len(raw) < 4 * 1024**2, 'Bounded trainable NPZ bytes')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        items = archive.infolist()
        names = archive.namelist()
        keys = {n[:-4] for n in names}
        if keys == set(head_shapes()):
            expected_schema = head_shapes()
        elif keys == set(trainable_60_shapes()):
            expected_schema = trainable_60_shapes()
        else:
            raise ValueError('NPZ must match exact 36-head or 60-tensor Arm-B schema')
        need(len(items) == len(set(names)) == len(expected_schema)
             and all('/' not in n and '\\' not in n and ':' not in n and n.endswith('.npy') for n in names),
             'Exact unique flat array keys required')
        import numpy as np
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            result = {}
            for name, shape in expected_schema.items():
                a = z[name]
                need(a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all(),
                     'Finite exact FP32 array values required: ' + name)
                result[name] = np.ascontiguousarray(a).copy()
            return result


def numpy_state_sha(state):
    """Canonical FP32 array digest matching the sealed tensor-state algorithm."""
    h = hashlib.sha256()
    for name, a in sorted(state.items()):
        header = json.dumps([name, 'torch.float32', list(a.shape)], separators=(',', ':')).encode()
        h.update(len(header).to_bytes(8, 'big'))
        h.update(header)
        h.update(a.tobytes(order='C'))
    return h.hexdigest()


def source_contract():
    """Read-only static contract of the Conditioned SFT V1 CPU inference adapter."""
    return {
        'format': 'conditioned-sft-v1-frozen-CPU-inference-source-v1',
        'R2_file': dict(R2_FILE),
        'U96_file': dict(U96_FILE),
        'head_NPZ_file': dict(HEAD_FILE),
        'canonical_states': {
            'R2_515': R2_STATE,
            'U96_24': U96_STATE,
            'terminal_heads_36': HEAD_STATE,
        },
        'protocol_sha256': PROTOCOL_SHA,
        'archive_sha256': ARCHIVE_SHA,
        'conditioning': conditioning.conditioning_contract(),
        'frozen_tensors': 575,
        'sites': list(SITES),
        'arm_A_trainable_parameters': 168960,
        'arm_B_trainable_parameters': 220288,
        'policy': 'CPU FP32 singleton512, w1, detach16, AdaIN, input-only exposure/degradation conditioning',
        'no_training_entrypoint': True,
        'model_qualified': False,
    }


def apply_conditioned_postprocessing_numpy(raw_pred_rgb, input_rgb, *, np,
                                           baseline_rgb=None,
                                           beta=conditioning.DEFAULT_ANCHOR_BETA,
                                           gamma=1.0):
    """Apply Conditioned SFT V1 postprocessing and floor RGB8 quantization in NumPy."""
    conditioned_raw, gates = conditioning.condition_output_numpy(
        raw_pred_rgb, input_rgb, np=np, baseline_rgb=baseline_rgb, beta=beta, gamma=gamma
    )
    rgb8 = (np.clip(conditioned_raw.astype(np.float32), 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
    return conditioned_raw, rgb8, gates


def load_frozen_conditioned_cpu_restorer(r2_artifact, u96_artifact, head_or_trainable_artifact,
                                         *, beta=conditioning.DEFAULT_ANCHOR_BETA, gamma=1.0):
    """Construct a frozen CPU restorer with Conditioned SFT V1 conditioning."""
    paths = [ordinary_file(p) for p in (r2_artifact, u96_artifact, head_or_trainable_artifact)]
    need(file_pin(paths[0]) == R2_FILE, 'Exact R2 artifact bytes required')
    need(file_pin(paths[1]) == U96_FILE, 'Exact U96 artifact bytes required')
    npz_arrays = inspect_trainable_npz(paths[2].read_bytes())

    import numpy as np
    import torch
    from torch.nn import functional as F
    from third_party.codeformer.codeformer_arch import CodeFormer

    need(torch.get_default_dtype() == torch.float32, 'FP32 default dtype required')
    saved_r2 = torch.load(paths[0], map_location='cpu', weights_only=True, mmap=True)
    saved_u96 = torch.load(paths[1], map_location='cpu', weights_only=True, mmap=True)
    r2 = saved_r2['params_ema']
    u96 = dict(saved_u96['adapter_state'])
    for name in list(adapter_shapes()):
        if name in npz_arrays:
            u96[name] = torch.from_numpy(npz_arrays[name].copy())
    head_dict = {name: torch.from_numpy(npz_arrays[name].copy()) for name in head_shapes()}

    class ResidualSkipAdapter(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            options = {'device': 'cpu', 'dtype': torch.float32}
            self.down = torch.nn.Conv2d(channels, 32, 1, bias=True, **options)
            self.depthwise = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=True, **options)
            self.up = torch.nn.Conv2d(32, channels, 1, bias=True, **options)
            self.activation = torch.nn.SiLU(inplace=False)
        def forward(self, observed):
            return observed + self.up(self.activation(self.depthwise(self.activation(self.down(observed)))))

    class SpatialHead(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            options = {'device': 'cpu', 'dtype': torch.float32}
            self.encoder = torch.nn.Conv2d(channels, WIDTH, 1, bias=True, **options)
            self.decoder = torch.nn.Conv2d(channels, WIDTH, 1, bias=False, **options)
            self.fused = torch.nn.Conv2d(channels, WIDTH, 1, bias=False, **options)
            self.coarse = torch.nn.Conv2d(256, WIDTH, 1, bias=False, **options)
            self.spatial = torch.nn.Conv2d(WIDTH, WIDTH, 3, padding=1, bias=True, **options)
            self.output = torch.nn.Conv2d(WIDTH, channels, 1, bias=True, **options)
            self.activation = torch.nn.GELU()
        def forward(self, used, decoder, fused, coarse):
            context = self.coarse(coarse.detach())
            context = F.interpolate(context, size=used.shape[-2:], mode='bilinear', align_corners=False)
            mixed = (self.encoder(used.detach()) + self.decoder(decoder) + self.fused(fused) + context) / 2.0
            logits = self.output(self.activation(self.spatial(self.activation(mixed))))
            return fused + torch.tanh(logits) * (fused - decoder)

    class FrozenConditionedCPURestorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(SITES))
            self.base.load_state_dict(r2, strict=True)
            self.adapters = torch.nn.ModuleDict({site: ResidualSkipAdapter(CHANNELS[site]) for site in SITES})
            self.heads = torch.nn.ModuleDict({site: SpatialHead(CHANNELS[site]) for site in SITES})
            self.adapters.load_state_dict({n.removeprefix('adapters.'): v for n, v in u96.items()}, strict=True)
            self.heads.load_state_dict({n.removeprefix('heads.'): v for n, v in head_dict.items()}, strict=True)
            self._forward_lock = threading.Lock()
            self.beta = float(beta)
            self.gamma = float(gamma)
            self.eval()
            self.requires_grad_(False)
            setattr(self.base, _OWNER_ATTRIBUTE, weakref.ref(self))

        def restore_rgb256(self, input_rgb_f32):
            need(not torch.is_grad_enabled(), 'Caller torch.no_grad required')
            need(isinstance(input_rgb_f32, np.ndarray) and input_rgb_f32.shape == (256, 256, 3)
                 and input_rgb_f32.dtype == np.float32, 'FP32 HWC 256x256 input in [0,1] required')
            input_256 = torch.from_numpy(input_rgb_f32.transpose(2, 0, 1).copy())[None]
            sample = F.interpolate(input_256, (512, 512), mode='bilinear', align_corners=False) * 2.0 - 1.0
            handles = []
            pending = {}
            coarse = None
            def pre_hook(site):
                def _pre(module, args):
                    nonlocal coarse
                    encoder, decoder, w_val = args
                    used = self.adapters[site](encoder.detach())
                    if site == '32':
                        coarse = used.detach()
                    pending[site] = (used, decoder)
                    return used, decoder, w_val
                return _pre
            def post_hook(site):
                def _post(module, args, fused):
                    used, decoder = pending.pop(site)
                    return self.heads[site](used, decoder, fused, coarse)
                return _post
            with self._forward_lock:
                try:
                    for site in SITES:
                        mod = self.base.fuse_convs_dict[site]
                        handles.append(mod.register_forward_pre_hook(pre_hook(site)))
                        handles.append(mod.register_forward_hook(post_hook(site)))
                    triple = self.base(sample, w=1.0, detach_16=True, adain=True)
                finally:
                    for h in reversed(handles):
                        h.remove()
            normalized = (triple[0] + 1.0) / 2.0
            base_raw = F.interpolate(normalized.clamp(0.0, 1.0), (256, 256), mode='bilinear', align_corners=False)
            raw_hwc = base_raw[0].numpy().transpose(1, 2, 0).copy()
            return apply_conditioned_postprocessing_numpy(
                raw_hwc, input_rgb_f32, np=np, beta=self.beta, gamma=self.gamma
            )

    with torch.no_grad(), torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261025)
        net = FrozenConditionedCPURestorer()
    return net

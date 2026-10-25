"""Portable CPU/CUDA Conditioned SFT V2 multi-model inference adapter; inert on import.

Supports side-by-side 256x256 restoration across:
1. `u64` (`575` frozen tensors: `515` `R2` + `24` `U96` + `36` `terminal_u0064_heads.npz`)
2. `V1 Arm B` (`Conditioned SFT V1 u0008` with coarse dark-drift alignment)
3. `Conditioned SFT V2` (`575` frozen `u64` backbone + `V2TrainableBundle` loaded from
   `trainable_v2.npz` or zero-initialized step-0 `u64`-parity bundle)
"""
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import zipfile

from research import codeformer_conditioned_sft_inference_v1 as inf_v1
from research.codeformer_conditioned_sft_learning_v1 import conditioning as cond_v1
from research.codeformer_conditioned_sft_learning_v2 import (
    architecture_v2 as arch_v2,
    conditioning_v2 as cond_v2,
)

SITES = ('32', '64', '128', '256')
R2_FILE = {'bytes': 376609465, 'sha256': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'}
U96_FILE = {'bytes': 212827, 'sha256': '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'}
HEAD_FILE = {'bytes': 685818, 'sha256': '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def source_contract_v2():
    return {
        'format': 'conditioned-sft-v2-multi-model-inference-source-v1',
        'R2_file': dict(R2_FILE),
        'U96_file': dict(U96_FILE),
        'head_NPZ_file': dict(HEAD_FILE),
        'architecture_v2': arch_v2.architecture_v2_contract(),
        'conditioning_v2': cond_v2.conditioning_v2_contract(),
        'no_training_entrypoint': True,
        'model_qualified': False,
    }


def save_v2_trainable_npz_bytes(bundle, *, np) -> bytes:
    """Serialize V2TrainableBundle parameters into deterministic uncompressed NPZ bytes."""
    arrays = arch_v2.export_v2_trainable_npz_dict(bundle, np=np)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def inspect_v2_trainable_npz(raw: bytes, *, torch, np):
    """Validate and load a serialized V2 trainable_v2.npz payload."""
    need(type(raw) is bytes and 0 < len(raw) < 16 * 1024**2, 'Bounded V2 trainable NPZ bytes required')
    ref_bundle = arch_v2.build_v2_trainable_modules(torch=torch)
    expected = {name: tuple(p.shape) for name, p in sorted(ref_bundle.named_parameters())}
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        need(len(names) == len(set(names)) == len(expected)
             and all(n.endswith('.npy') for n in names),
             'Exact V2 trainable NPZ entry count required')
        keys = {n[:-4] for n in names}
        need(keys == set(expected.keys()), 'Exact V2 trainable tensor keyset required')
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        out = {}
        for name, shape in expected.items():
            arr = z[name]
            need(arr.shape == shape and arr.dtype == np.float32 and np.isfinite(arr).all(),
                 'Invalid or nonfinite V2 tensor in NPZ: ' + name)
            out[name] = np.ascontiguousarray(arr).copy()
        return out


def central_laplacian_sharpness_numpy(rgb_f32, *, np) -> float:
    w = np.asarray(cond_v2.LUMA_WEIGHTS, dtype=np.float64)
    luma = rgb_f32.astype(np.float64) @ w
    c = luma[64:192, 64:192]
    lx = np.abs(c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]).mean()
    ly = np.abs(c[2:, :] - 2.0 * c[1:-1, :] + c[:-2, :]).mean()
    return float(0.5 * (lx + ly))


def load_conditioned_sft_v2_restorer(
    r2_path,
    u96_path,
    u64_heads_path,
    *,
    v2_trainable_path=None,
    v1_b8_trainable_path=None,
    device: str = 'cpu',
):
    """Load frozen 575-tensor u64 backbone + V2 trainable bundle (+ optional V1-B8 comparison arm)."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from third_party.codeformer.codeformer_arch import CodeFormer

    need(inf_v1.file_pin(r2_path) == R2_FILE, 'Exact codeformer_r2.pth bytes required')
    need(inf_v1.file_pin(u96_path) == U96_FILE, 'Exact codeformer_u96_adapter.pth bytes required')
    need(inf_v1.file_pin(u64_heads_path) == HEAD_FILE, 'Exact terminal_u0064_heads.npz bytes required')

    dev = torch.device(device)
    base = CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=['32', '64', '128', '256'],
    )
    r2_ckpt = torch.load(str(r2_path), map_location='cpu', weights_only=True)
    r2_params = r2_ckpt['params_ema'] if 'params_ema' in r2_ckpt else r2_ckpt
    base.load_state_dict(r2_params, strict=True)
    base.eval().to(dev)
    for p in base.parameters():
        p.requires_grad_(False)
    need(arch_v2.tensor_state_sha256(dict(base.named_parameters()), torch=torch) == arch_v2.R2_CANONICAL_SHA256,
         'Exact 515-tensor R2 canonical SHA256 required')

    u96_adapters, u64_heads = arch_v2.build_frozen_u96_and_u64_modules(torch=torch)
    u96_ckpt = torch.load(str(u96_path), map_location='cpu', weights_only=True)
    u96_sd = u96_ckpt.get('adapter_state', u96_ckpt.get('adapter_state_dict', u96_ckpt))
    u96_adapters.load_state_dict(
        {k[len('adapters.'):] if k.startswith('adapters.') else k: v for k, v in u96_sd.items()},
        strict=True,
    )
    u64_np = inf_v1.inspect_trainable_npz(Path(u64_heads_path).read_bytes())
    u64_heads.load_state_dict(
        {k[len('heads.'):]: torch.from_numpy(v) for k, v in u64_np.items()},
        strict=True,
    )
    u96_adapters.eval().to(dev)
    u64_heads.eval().to(dev)
    for mod in (u96_adapters, u64_heads):
        for p in mod.parameters():
            p.requires_grad_(False)

    # Optional V1 Arm B comparison modules
    b8_adapters = b8_heads = None
    b8_gamma = 0.0
    if v1_b8_trainable_path is not None and Path(v1_b8_trainable_path).is_file():
        b8_adapters, b8_heads = arch_v2.build_frozen_u96_and_u64_modules(torch=torch)
        b8_np = inf_v1.inspect_trainable_npz(Path(v1_b8_trainable_path).read_bytes())
        b8_adapters.load_state_dict(
            {k[len('adapters.'):]: torch.from_numpy(v) for k, v in b8_np.items() if k.startswith('adapters.')},
            strict=True,
        )
        b8_heads.load_state_dict(
            {k[len('heads.'):]: torch.from_numpy(v) for k, v in b8_np.items() if k.startswith('heads.')},
            strict=True,
        )
        b8_adapters.eval().to(dev)
        b8_heads.eval().to(dev)
        for mod in (b8_adapters, b8_heads):
            for p in mod.parameters():
                p.requires_grad_(False)
        b8_gamma = 1.0

    # V2 trainable bundle (zero-initialized step-0 u64 parity unless checkpoint supplied)
    v2_bundle = arch_v2.build_v2_trainable_modules(torch=torch).to(dev)
    v2_init_dict = arch_v2.export_v2_trainable_npz_dict(v2_bundle, np=np)
    v2_loaded_checkpoint = None
    if v2_trainable_path is not None and Path(v2_trainable_path).is_file():
        raw_v2 = Path(v2_trainable_path).read_bytes()
        v2_arrays = inspect_v2_trainable_npz(raw_v2, torch=torch, np=np)
        arch_v2.load_v2_trainable_npz_dict(v2_bundle, v2_arrays, torch=torch, np=np)
        v2_loaded_checkpoint = {
            'path': str(v2_trainable_path),
            'bytes': len(raw_v2),
            'sha256': hashlib.sha256(raw_v2).hexdigest(),
        }
    v2_bundle.eval()
    for p in v2_bundle.parameters():
        p.requires_grad_(False)
    v2_disp = arch_v2.max_linf_displacement_from_init(v2_bundle, v2_init_dict, np=np)
    v2_gamma = cond_v2.displacement_gamma(v2_disp)

    class ConditionedSFTV2Restorer:
        def __init__(self):
            self.device = dev
            self.v2_checkpoint = v2_loaded_checkpoint
            self.v2_max_linf_displacement = v2_disp
            self.v2_gamma = v2_gamma
            self.has_v1_b8 = b8_adapters is not None
            self.v2_parameter_count = sum(p.numel() for p in v2_bundle.parameters())
            self.v2_tensor_count = sum(1 for _ in v2_bundle.parameters())

        @staticmethod
        def _validate_rgb_u8(rgb_u8):
            need(isinstance(rgb_u8, np.ndarray) and rgb_u8.shape == (256, 256, 3) and rgb_u8.dtype == np.uint8,
                 '256x256 RGB uint8 image required')

        def _forward_u64_or_b8_raw(self, sample_512, adapters_mod, heads_mod):
            handles = []
            pending = {}
            coarse = None

            def pre_hook(site):
                def _pre(module, args):
                    nonlocal coarse
                    enc, dec, w_val = args
                    used = adapters_mod[site](enc.detach())
                    if site == '32':
                        coarse = used.detach()
                    pending[site] = (used, dec)
                    return used, dec, w_val
                return _pre

            def post_hook(site):
                def _post(module, args, fused):
                    used, dec = pending.pop(site)
                    return heads_mod[site](used, dec, fused, coarse)
                return _post

            try:
                for s in SITES:
                    mod = base.fuse_convs_dict[s]
                    handles.append(mod.register_forward_pre_hook(pre_hook(s)))
                    handles.append(mod.register_forward_hook(post_hook(s)))
                with torch.no_grad():
                    triple = base(sample_512, w=1.0, detach_16=True, adain=True)
            finally:
                for h in reversed(handles):
                    h.remove()
            norm = (triple[0] + 1.0) / 2.0
            raw256 = F.interpolate(norm.clamp(0.0, 1.0), (256, 256), mode='bilinear', align_corners=False)
            return raw256[0].detach().cpu().numpy().transpose(1, 2, 0).copy()

        def _forward_v2_raw(self, sample_512, input_nchw_256):
            cond_t = cond_v2.compute_v2_conditioning_torch(input_nchw_256, torch=torch)
            handles = []
            pending = {}
            coarse = None
            prev_h = None

            def pre_hook(site):
                def _pre(module, args):
                    nonlocal coarse, prev_h
                    enc, dec, w_val = args
                    e_u96 = u96_adapters[site](enc.detach())
                    used, prev_h = v2_bundle.refine_site_sequential(site, e_u96, cond_t, prev_h)
                    if site == '32':
                        coarse = used.detach()
                    pending[site] = (used, dec)
                    return used, dec, w_val
                return _pre

            def post_hook(site):
                def _post(module, args, fused):
                    used, dec = pending.pop(site)
                    base_post = u64_heads[site](used, dec, fused, coarse)
                    return v2_bundle.refine_post_sft(site, base_post, used, dec, fused, coarse, cond_t)
                return _post

            try:
                for s in SITES:
                    mod = base.fuse_convs_dict[s]
                    handles.append(mod.register_forward_pre_hook(pre_hook(s)))
                    handles.append(mod.register_forward_hook(post_hook(s)))
                with torch.no_grad():
                    triple = base(sample_512, w=1.0, detach_16=True, adain=True)
            finally:
                for h in reversed(handles):
                    h.remove()
            norm = (triple[0] + 1.0) / 2.0
            raw256 = F.interpolate(norm.clamp(0.0, 1.0), (256, 256), mode='bilinear', align_corners=False)
            return raw256[0].detach().cpu().numpy().transpose(1, 2, 0).copy()

        def restore_v2(self, rgb_u8, *, cached_u64_raw=None):
            self._validate_rgb_u8(rgb_u8)
            inp_f32 = rgb_u8.astype(np.float32) / np.float32(255.0)
            inp_t = torch.from_numpy(rgb_u8.transpose(2, 0, 1).copy()).to(device=dev, dtype=torch.float32)[None] / 255.0
            sample_512 = F.interpolate(inp_t, (512, 512), mode='bilinear', align_corners=False) * 2.0 - 1.0
            u64_raw = cached_u64_raw if cached_u64_raw is not None else self._forward_u64_or_b8_raw(sample_512, u96_adapters, u64_heads)
            if self.v2_gamma == 0.0:
                v2_raw = u64_raw.copy()
                cond_info = cond_v2.compute_v2_conditioning_numpy(inp_f32, np=np)
            else:
                v2_raw_neural = self._forward_v2_raw(sample_512, inp_t)
                v2_raw, cond_info = cond_v2.condition_output_v2_numpy(
                    v2_raw_neural, inp_f32, np=np, baseline_rgb=u64_raw,
                    beta=cond_v2.DEFAULT_ANCHOR_BETA, gamma=self.v2_gamma, dark_drift_align=True,
                )
            v2_u8 = (np.clip(v2_raw, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
            return {
                'raw_f32': v2_raw,
                'rgb8': v2_u8,
                'u64_raw_f32': u64_raw,
                'u64_rgb8': (np.clip(u64_raw, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8),
                'conditioning': cond_info,
                'v2_gamma': self.v2_gamma,
                'v2_checkpoint_loaded': self.v2_checkpoint is not None,
            }

        def restore_compare_all(self, rgb_u8):
            """Run Input -> u64, V1-B8 (if loaded), and V2 side-by-side with full sharpness telemetry."""
            self._validate_rgb_u8(rgb_u8)
            inp_f32 = rgb_u8.astype(np.float32) / np.float32(255.0)
            inp_t = torch.from_numpy(rgb_u8.transpose(2, 0, 1).copy()).to(device=dev, dtype=torch.float32)[None] / 255.0
            sample_512 = F.interpolate(inp_t, (512, 512), mode='bilinear', align_corners=False) * 2.0 - 1.0

            u64_raw = self._forward_u64_or_b8_raw(sample_512, u96_adapters, u64_heads)
            u64_u8 = (np.clip(u64_raw, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)

            b8_u8 = None
            b8_sharp = None
            if self.has_v1_b8:
                b8_neural = self._forward_u64_or_b8_raw(sample_512, b8_adapters, b8_heads)
                b8_cond, b8_u8, _ = inf_v1.apply_conditioned_postprocessing_numpy(
                    b8_neural, inp_f32, np=np, baseline_rgb=u64_raw,
                    beta=cond_v1.DEFAULT_ANCHOR_BETA, gamma=b8_gamma, dark_drift_align=True,
                )
                b8_sharp = central_laplacian_sharpness_numpy(b8_cond, np=np)

            v2_res = self.restore_v2(rgb_u8, cached_u64_raw=u64_raw)
            cond_clean = {
                k: v for k, v in v2_res['conditioning'].items()
                if k != 'local_map_16'
            }
            return {
                'input_rgb8': rgb_u8.copy(),
                'u64_rgb8': u64_u8,
                'v1_b8_rgb8': b8_u8,
                'v2_rgb8': v2_res['rgb8'],
                'conditioning': cond_clean,
                'sharpness': {
                    'input': central_laplacian_sharpness_numpy(inp_f32, np=np),
                    'u64': central_laplacian_sharpness_numpy(u64_raw, np=np),
                    'v1_b8': b8_sharp,
                    'v2': central_laplacian_sharpness_numpy(v2_res['raw_f32'], np=np),
                },
                'v2_checkpoint': self.v2_checkpoint,
                'v2_gamma': self.v2_gamma,
            }

    return ConditionedSFTV2Restorer()

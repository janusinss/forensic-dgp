"""CodeFormer Restoration V3 — Unified Single-Pass Neural Architecture.

Architecture blueprint (#333):
- 100% frozen official CodeFormer backbone (codeformer.pth v0.1.0, 515 tensors).
- Mask-aware border-replicate padding for non-square prepare_crop canvases.
- Module A: V3DegradationEncoder (5-ch un-gated input + 4-stage inverted-residual FPN)
            + PhotoCorrectNet (bounded 32x32 gain/gamma/bias, identity-initialized)
            + per-site spatial w_map(s) heads + auxiliary degradation/gate heads.
- Module B: DegGatedTransformerLoRA (r=16, alpha=16, zero-init B on all 9 frozen
            TransformerSALayer blocks + 16x16 FiLM) + MaskedAdaIN.
- Module C: V3PyramidSFT (top-down 32->64->128->256 dilated + directional 1x5/5x1
            strip SE blocks, zero-init residual) + single-pass frozen Fuse_sft_block
            with native spatial w_map(s) broadcast + V3FreqHead (zero-init HP5 + spatial).
- Module D: V3OutputComposer (antialiased 512->256 bicubic + pixel-wise alpha_bypass
            + clamped r_out in [-0.1, 0.1]) + exact ~observed preservation.
- Flat zero-pickle float32 .npz export/load with strict header validation.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import stat
import zipfile
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from third_party.codeformer.codeformer_arch import CodeFormer

SITES: Tuple[str, ...] = ("32", "64", "128", "256")
CHANNELS: Dict[str, int] = {"32": 256, "64": 256, "128": 128, "256": 128}
D_CHANNELS: Dict[str, int] = {"16": 128, "32": 96, "64": 64, "128": 48, "256": 32}
PYR_WIDTHS: Dict[str, int] = {"32": 160, "64": 144, "128": 80, "256": 48}
FREQ_WIDTHS: Dict[str, int] = {"32": 96, "64": 80, "128": 48, "256": 32}
Z_DIM: int = 128
BASE_TENSOR_COUNT: int = 515

LUMA_WEIGHTS: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722)

# PhotoCorrectNet bounds (V3.0/3.1/3.2 default)
GAIN_MIN, GAIN_MAX = 0.85, 3.20
GAMMA_MIN, GAMMA_MAX = 0.55, 1.35
BIAS_MIN, BIAS_MAX = -0.15, 0.25

# V3.3 tightened PhotoCorrectNet bounds & calibrated w_maps floors
GAIN_MIN_V33, GAIN_MAX_V33 = 0.92, 2.80
GAMMA_MIN_V33, GAMMA_MAX_V33 = 0.60, 1.15
BIAS_MIN_V33, BIAS_MAX_V33 = -0.05, 0.15
_WMAP_FLOORS_V33: Dict[str, float] = {"32": 0.50, "64": 0.35, "128": 0.15, "256": 0.08}
ARCH_VARIANTS: Tuple[str, ...] = ("v30", "v33", "v34")


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and (channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def replicate_pad_observed(x_256: Tensor, observed_mask: Tensor) -> Tensor:
    """Replace ~observed bars with nearest-border replicate fill from the observed bbox.

    Args:
        x_256: (B, 3, H, W) float tensor in [0, 1].
        observed_mask: (B, 1, H, W) float or bool tensor (> 0.5 means observed).

    Returns:
        (B, 3, H, W) tensor with observed pixels unchanged and ~observed pixels
        filled by nearest-border replication from the observed bounding box.
    """
    b, _, h, w = x_256.shape
    mask_bool = observed_mask > 0.5
    if mask_bool.all():
        return x_256

    out_list = []
    ar_y = torch.arange(h, device=x_256.device)
    ar_x = torch.arange(w, device=x_256.device)
    for i in range(b):
        mi = mask_bool[i, 0]
        if mi.all() or not mi.any():
            out_list.append(x_256[i])
            continue
        ys = torch.nonzero(mi.any(dim=1), as_tuple=False).squeeze(1)
        xs = torch.nonzero(mi.any(dim=0), as_tuple=False).squeeze(1)
        y0, y1 = int(ys[0].item()), int(ys[-1].item())
        x0, x1 = int(xs[0].item()), int(xs[-1].item())
        y_idx = ar_y.clamp(y0, y1)
        x_idx = ar_x.clamp(x0, x1)
        rep = x_256[i][:, y_idx][:, :, x_idx]
        out_list.append(torch.where(mi.unsqueeze(0), x_256[i], rep))
    return torch.stack(out_list, dim=0)


class _MBConvBlock(nn.Module):
    """Lightweight Mobile-Inverted-Residual block with GroupNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 2):
        super().__init__()
        mid_ch = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.pw1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.gn1 = _gn(mid_ch)
        self.dw = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False)
        self.gn2 = _gn(mid_ch)
        self.pw2 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.gn3 = _gn(out_ch)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.gn1(self.pw1(x)))
        h = self.act(self.gn2(self.dw(h)))
        h = self.gn3(self.pw2(h))
        return x + h if self.use_res else h


class V3DegradationEncoder(nn.Module):
    """Module A (part 1): 5-ch un-gated degradation encoder + 4-stage FPN + heads.

    Input channels (256x256):
        [R, G, B, observed_mask, luma_hf_resid = luma - avg3x3(luma)]
    """

    def __init__(self, z_dim: int = Z_DIM, arch_variant: str = "v30"):
        super().__init__()
        if arch_variant not in ARCH_VARIANTS:
            raise ValueError(f"Unsupported arch_variant={arch_variant!r}; expected one of {ARCH_VARIANTS}")
        self.z_dim = z_dim
        self.arch_variant = str(arch_variant)
        self.register_buffer(
            "_luma_w",
            torch.tensor(LUMA_WEIGHTS, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # Bottom-up 4-stage inverted-residual encoder: widths [32, 48, 64, 96, 128]
        self.stem = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1, bias=False),
            _gn(32),
            nn.SiLU(inplace=False),
        )
        self.stage1 = nn.Sequential(
            _MBConvBlock(32, 48, stride=2, expand=2),
            _MBConvBlock(48, 48, stride=1, expand=2),
        )
        self.stage2 = nn.Sequential(
            _MBConvBlock(48, 64, stride=2, expand=2),
            _MBConvBlock(64, 64, stride=1, expand=2),
        )
        self.stage3 = nn.Sequential(
            _MBConvBlock(64, 96, stride=2, expand=2),
            _MBConvBlock(96, 96, stride=1, expand=2),
        )
        self.stage4 = nn.Sequential(
            _MBConvBlock(96, 128, stride=2, expand=2),
            _MBConvBlock(128, 128, stride=1, expand=2),
        )

        # Top-down FPN lateral projections and fusion (256x256 depthwise-separable for CPU speed)
        self.up_16_32 = nn.Conv2d(128, 96, 1, bias=True)
        self.fuse_32 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            _gn(96),
            nn.SiLU(inplace=False),
        )
        self.up_32_64 = nn.Conv2d(96, 64, 1, bias=True)
        self.fuse_64 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            _gn(64),
            nn.SiLU(inplace=False),
        )
        self.up_64_128 = nn.Conv2d(64, 48, 1, bias=True)
        self.fuse_128 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            _gn(48),
            nn.SiLU(inplace=False),
        )
        self.up_128_256 = nn.Conv2d(48, 32, 1, bias=True)
        self.fuse_256 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 32, 1, bias=False),
            _gn(32),
            nn.SiLU(inplace=False),
        )

        # Masked global avg + max pool -> z_deg (B, 128)
        self.z_proj = nn.Sequential(
            nn.Linear(256, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(inplace=False),
            nn.Linear(z_dim, z_dim),
        )

        # Auxiliary scalar heads
        self.deg_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.SiLU(inplace=False),
            nn.Linear(64, 4),
        )
        self.alpha_adain_head = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, 1),
        )
        self.lora_gate_head = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, 1),
        )
        self.sufficiency_head = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, 1),
        )

        # Per-site spatial w_map(s) heads for s in {32, 64, 128, 256}
        self.w_heads = nn.ModuleDict()
        for site in SITES:
            d_ch = D_CHANNELS[site]
            if site == "256":
                head = nn.Sequential(
                    nn.Conv2d(d_ch, d_ch, 3, padding=1, groups=d_ch, bias=False),
                    nn.Conv2d(d_ch, 32, 1, bias=False),
                    _gn(32),
                    nn.SiLU(inplace=False),
                    nn.Conv2d(32, 1, 1, bias=True),
                )
            else:
                head = nn.Sequential(
                    nn.Conv2d(d_ch, 32, 3, padding=1, bias=False),
                    _gn(32),
                    nn.SiLU(inplace=False),
                    nn.Conv2d(32, 1, 1, bias=True),
                )
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, 2.2)  # sigmoid(2.2) ~= 0.90 at step 0
            self.w_heads[site] = head

        # Sensible step-0 gate initializations
        nn.init.zeros_(self.alpha_adain_head[-1].weight)
        nn.init.constant_(self.alpha_adain_head[-1].bias, 2.0)  # ~= 0.88
        nn.init.zeros_(self.lora_gate_head[-1].weight)
        nn.init.constant_(self.lora_gate_head[-1].bias, -2.0)  # ~= 0.12 (LoRA B=0 anyway)

    def forward(self, x_pad_256: Tensor, observed_mask_256: Tensor) -> Dict[str, Any]:
        w = self._luma_w.to(device=x_pad_256.device, dtype=x_pad_256.dtype)
        luma = (x_pad_256 * w).sum(dim=1, keepdim=True)
        luma_avg = F.avg_pool2d(F.pad(luma, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        luma_hf = (luma - luma_avg).mul(10.0).clamp(-1.0, 1.0)

        mask_f = observed_mask_256.to(dtype=x_pad_256.dtype)
        inp5 = torch.cat([x_pad_256, mask_f, luma_hf], dim=1)

        f256 = self.stem(inp5)
        f128 = self.stage1(f256)
        f64 = self.stage2(f128)
        f32 = self.stage3(f64)
        f16 = self.stage4(f32)

        d16 = f16
        d32 = self.fuse_32(f32 + F.interpolate(self.up_16_32(d16), size=(32, 32), mode="bilinear", align_corners=False))
        d64 = self.fuse_64(f64 + F.interpolate(self.up_32_64(d32), size=(64, 64), mode="bilinear", align_corners=False))
        d128 = self.fuse_128(f128 + F.interpolate(self.up_64_128(d64), size=(128, 128), mode="bilinear", align_corners=False))
        d256 = self.fuse_256(f256 + F.interpolate(self.up_128_256(d128), size=(256, 256), mode="bilinear", align_corners=False))
        d_maps = {"16": d16, "32": d32, "64": d64, "128": d128, "256": d256}

        # Masked global avg + max pooling over observed_mask_16
        m16 = F.interpolate(mask_f, size=(16, 16), mode="area")
        m16_bin = (m16 > 0.05).to(dtype=d16.dtype)
        valid_counts = m16_bin.sum(dim=(2, 3))  # (B, 1)
        has_obs = valid_counts > 0.5
        m16_eff = torch.where(has_obs.view(-1, 1, 1, 1), m16_bin, torch.ones_like(m16_bin))
        denom = m16_eff.sum(dim=(2, 3)).clamp(min=1.0)
        avg_pool = (d16 * m16_eff).sum(dim=(2, 3)) / denom
        max_pool = d16.masked_fill(m16_eff < 0.5, -1e4).amax(dim=(2, 3))
        z_deg = self.z_proj(torch.cat([avg_pool, max_pool], dim=1))

        deg_raw = torch.sigmoid(self.deg_head(z_deg))
        deg_preds = {
            "exposure": deg_raw[:, 0:1],
            "blur": deg_raw[:, 1:2],
            "noise": deg_raw[:, 2:3],
            "codec_lr": deg_raw[:, 3:4],
        }
        alpha_adain = torch.sigmoid(self.alpha_adain_head(z_deg)).view(-1, 1, 1, 1)
        lora_gate = torch.sigmoid(self.lora_gate_head(z_deg))
        sufficiency = torch.sigmoid(self.sufficiency_head(z_deg))

        if self.arch_variant == "v33":
            w_maps = {
                site: _WMAP_FLOORS_V33[site]
                + (1.0 - _WMAP_FLOORS_V33[site]) * torch.sigmoid(self.w_heads[site](d_maps[site]))
                for site in SITES
            }
        else:
            w_maps = {site: torch.sigmoid(self.w_heads[site](d_maps[site])) for site in SITES}
        return {
            "z_deg": z_deg,
            "d_maps": d_maps,
            "w_maps": w_maps,
            "deg_preds": deg_preds,
            "alpha_adain": alpha_adain,
            "lora_gate": lora_gate,
            "sufficiency": sufficiency,
            "observed_mask_16": m16_eff,
        }


class PhotoCorrectNet(nn.Module):
    """Module A (part 2): Smooth 32x32 bounded photometric pre-corrector.

    Identity-initialized so at step 0: gain=1.0, gamma=1.0, bias=0.0 => x_photo == x_in.
    """

    def __init__(self, d32_dim: int = D_CHANNELS["32"], z_dim: int = Z_DIM, arch_variant: str = "v30"):
        super().__init__()
        if arch_variant not in ARCH_VARIANTS:
            raise ValueError(f"Unsupported arch_variant={arch_variant!r}; expected one of {ARCH_VARIANTS}")
        self.arch_variant = str(arch_variant)
        self.register_buffer(
            "_luma_w",
            torch.tensor(LUMA_WEIGHTS, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.z_proj = nn.Linear(z_dim, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(d32_dim + 32, 64, 3, padding=1, bias=False),
            _gn(64),
            nn.SiLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            _gn(64),
            nn.SiLU(inplace=False),
        )
        self.param_head = nn.Conv2d(64, 9, 3, padding=1, bias=True)
        nn.init.zeros_(self.param_head.weight)
        nn.init.zeros_(self.param_head.bias)

        self.g_photo_head = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.g_photo_head[-1].weight)
        nn.init.zeros_(self.g_photo_head[-1].bias)  # sigmoid(0) = 0.5 so param_head gets grad

        if self.arch_variant == "v33":
            self.gain_min, self.gain_max = GAIN_MIN_V33, GAIN_MAX_V33
            self.gamma_min, self.gamma_max = GAMMA_MIN_V33, GAMMA_MAX_V33
            self.bias_min, self.bias_max = BIAS_MIN_V33, BIAS_MAX_V33
        else:
            self.gain_min, self.gain_max = GAIN_MIN, GAIN_MAX
            self.gamma_min, self.gamma_max = GAMMA_MIN, GAMMA_MAX
            self.bias_min, self.bias_max = BIAS_MIN, BIAS_MAX

        self._gain_offset = _logit((1.0 - self.gain_min) / (self.gain_max - self.gain_min))
        self._gamma_offset = _logit((1.0 - self.gamma_min) / (self.gamma_max - self.gamma_min))
        self._bias_offset = _logit((0.0 - self.bias_min) / (self.bias_max - self.bias_min))

    def forward(
        self,
        x_pad_256: Tensor,
        d32: Tensor,
        z_deg: Tensor,
        *,
        observed_mask: Optional[Tensor] = None,
        sufficiency: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        b = x_pad_256.shape[0]
        z_sp = self.z_proj(z_deg).view(b, 32, 1, 1).expand(-1, -1, 32, 32)
        raw = self.param_head(self.conv(torch.cat([d32, z_sp], dim=1)))

        raw_gain, raw_gamma, raw_bias = raw.chunk(3, dim=1)
        gain_32 = self.gain_min + (self.gain_max - self.gain_min) * torch.sigmoid(raw_gain + self._gain_offset)
        gamma_32 = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw_gamma + self._gamma_offset)
        bias_32 = self.bias_min + (self.bias_max - self.bias_min) * torch.sigmoid(raw_bias + self._bias_offset)

        gain = F.interpolate(gain_32, size=(256, 256), mode="bilinear", align_corners=False)
        gamma = F.interpolate(gamma_32, size=(256, 256), mode="bilinear", align_corners=False)
        bias = F.interpolate(bias_32, size=(256, 256), mode="bilinear", align_corners=False)

        # Smooth gamma curve that is exact identity at gamma=1.0 even at x=0.0
        eps = 1e-5
        x_f = x_pad_256.float().clamp(0.0, 1.0)
        g_f = gamma.float()
        eps_g = eps ** g_f
        one_g = (1.0 + eps) ** g_f
        x_pow = ((x_f + eps).pow(g_f) - eps_g) / (one_g - eps_g).clamp(min=1e-6)
        x_corr = (gain.float() * x_pow + bias.float()).clamp(0.0, 1.0).to(dtype=x_pad_256.dtype)

        g_photo = torch.sigmoid(self.g_photo_head(z_deg)).view(b, 1, 1, 1)
        if self.arch_variant == "v33" and sufficiency is not None:
            # Smoothly damp g_photo on bright low-sufficiency inputs (mean_luma > 0.45 & suff < 0.45)
            lw = self._luma_w.to(device=x_f.device, dtype=torch.float32)
            luma = (x_f * lw).sum(dim=1, keepdim=True)
            if observed_mask is not None:
                m_f = observed_mask.to(device=x_f.device, dtype=torch.float32)
                if m_f.ndim == 3:
                    m_f = m_f.unsqueeze(1)
                mean_luma = (luma * m_f).sum(dim=(1, 2, 3)) / m_f.sum(dim=(1, 2, 3)).clamp(min=1.0)
            else:
                mean_luma = luma.mean(dim=(1, 2, 3))
            suff_f = sufficiency.float().reshape(b)
            bright_f = torch.sigmoid((mean_luma - 0.45) / 0.06)
            low_suff_f = torch.sigmoid((0.45 - suff_f) / 0.08)
            damp = (1.0 - 0.85 * bright_f * low_suff_f).clamp(0.15, 1.0).view(b, 1, 1, 1)
            g_photo = g_photo * damp.to(dtype=g_photo.dtype)
        x_photo = (x_pad_256 + g_photo * (x_corr - x_pad_256)).clamp(0.0, 1.0)
        return x_photo, g_photo


class _LoRALinear(nn.Module):
    """Low-rank adapter with zero-init B so step-0 delta is strictly zero."""

    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.scale = float(alpha) / float(r)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.B(self.A(x)) * self.scale


class DegGatedTransformerLoRA(nn.Module):
    """Module B (part 1): Rank-16 degradation-gated LoRA on 9 frozen TransformerSALayer blocks
    plus zero-init 16x16 FiLM from (z_deg, D_16).

    Never mutates base CodeFormer weights.
    """

    def __init__(
        self,
        n_layers: int = 9,
        embed_dim: int = 512,
        dim_mlp: int = 1024,
        n_head: int = 8,
        r: int = 16,
        alpha: float = 16.0,
        z_dim: int = Z_DIM,
        d16_dim: int = D_CHANNELS["16"],
    ):
        super().__init__()
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head

        self.z_to_d16 = nn.Linear(z_dim, d16_dim)
        self.film_net = nn.Sequential(
            nn.Conv2d(d16_dim, 32, 3, padding=1, bias=False),
            _gn(32),
            nn.SiLU(inplace=False),
            nn.Conv2d(32, embed_dim * 2, 1, bias=True),
        )
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "lora_q": _LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha),
                "lora_k": _LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha),
                "lora_v": _LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha),
                "lora_out": _LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha),
                "lora_fc1": _LoRALinear(embed_dim, dim_mlp, r=r, alpha=alpha),
                "lora_fc2": _LoRALinear(dim_mlp, embed_dim, r=r, alpha=alpha),
            })
            for _ in range(n_layers)
        ])

    def forward(
        self,
        query_emb: Tensor,
        pos_emb: Tensor,
        ft_layers: nn.Sequential,
        z_deg: Tensor,
        d16: Tensor,
        lora_gate: Tensor,
    ) -> Tensor:
        s_len, b, e_dim = query_emb.shape
        gate_seq = lora_gate.view(1, b, 1)

        # 16x16 FiLM before Transformer layer 0 (zero-init => identity at step 0)
        cond16 = d16 + self.z_to_d16(z_deg).view(b, -1, 1, 1)
        film = self.film_net(cond16)  # (B, 2*E, 16, 16)
        film_scale, film_shift = film.chunk(2, dim=1)
        film_scale = film_scale.flatten(2).permute(2, 0, 1)  # (256, B, E)
        film_shift = film_shift.flatten(2).permute(2, 0, 1)
        tgt = query_emb * (1.0 + gate_seq * film_scale) + gate_seq * film_shift

        for layer, lora in zip(ft_layers, self.layers):
            # Self-attention (matches TransformerSALayer.forward exactly when LoRA=0)
            tgt2 = layer.norm1(tgt)
            q_in = tgt2 + pos_emb
            k_in = q_in
            v_in = tgt2

            attn = layer.self_attn
            w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)

            q = F.linear(q_in, w_q, b_q) + gate_seq * lora["lora_q"](q_in)
            k = F.linear(k_in, w_k, b_k) + gate_seq * lora["lora_k"](k_in)
            v = F.linear(v_in, w_v, b_v) + gate_seq * lora["lora_v"](v_in)

            qh = q.view(s_len, b, self.n_head, self.head_dim).permute(1, 2, 0, 3)
            kh = k.view(s_len, b, self.n_head, self.head_dim).permute(1, 2, 0, 3)
            vh = v.view(s_len, b, self.n_head, self.head_dim).permute(1, 2, 0, 3)

            ctx = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0)
            ctx = ctx.permute(2, 0, 1, 3).reshape(s_len, b, e_dim)

            attn_out = F.linear(ctx, attn.out_proj.weight, attn.out_proj.bias) + gate_seq * lora["lora_out"](ctx)
            tgt = tgt + attn_out

            # FFN
            tgt2 = layer.norm2(tgt)
            fc1_out = F.linear(tgt2, layer.linear1.weight, layer.linear1.bias) + gate_seq * lora["lora_fc1"](tgt2)
            h = layer.activation(fc1_out)
            fc2_out = F.linear(h, layer.linear2.weight, layer.linear2.bias) + gate_seq * lora["lora_fc2"](h)
            tgt = tgt + fc2_out

        return tgt


class MaskedAdaIN(nn.Module):
    """Module B (part 2): AdaIN computing lq_feat statistics strictly over observed_mask_16 in FP32."""

    def forward(
        self,
        quant_feat: Tensor,
        lq_feat: Tensor,
        observed_mask_16: Tensor,
        alpha_adain: Tensor,
        eps: float = 1e-5,
    ) -> Tensor:
        out_dtype = quant_feat.dtype
        q_f = quant_feat.float()
        l_f = lq_feat.float()
        m = observed_mask_16.to(device=l_f.device, dtype=torch.float32)
        m_sum = m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        style_mean = (l_f * m).sum(dim=(2, 3), keepdim=True) / m_sum
        style_var = (((l_f - style_mean) ** 2) * m).sum(dim=(2, 3), keepdim=True) / m_sum
        style_std = (style_var + eps).sqrt()

        content_mean = q_f.mean(dim=(2, 3), keepdim=True)
        content_var = q_f.var(dim=(2, 3), keepdim=True, unbiased=False)
        content_std = (content_var + eps).sqrt()

        adain_feat = (q_f - content_mean) / content_std * style_std + style_mean
        out = q_f + alpha_adain.float() * (adain_feat - q_f)
        return out.to(dtype=out_dtype)


class _V3DilatedStripSEBlock(nn.Module):
    """Multi-scale dilated (1,2,4) + directional (1x5/5x1) depthwise strip SE residual block."""

    def __init__(self, width: int):
        super().__init__()
        assert width % 4 == 0, f"width must be divisible by 4, got {width}"
        q = width // 4
        self.q = q
        self.dw1 = nn.Conv2d(q, q, 3, padding=1, dilation=1, groups=q, bias=True)
        self.dw2 = nn.Conv2d(q, q, 3, padding=2, dilation=2, groups=q, bias=True)
        self.dw4 = nn.Conv2d(q, q, 3, padding=4, dilation=4, groups=q, bias=True)
        self.dw_h = nn.Conv2d(q, q, (1, 5), padding=(0, 2), groups=q, bias=True)
        self.dw_v = nn.Conv2d(q, q, (5, 1), padding=(2, 0), groups=q, bias=True)
        self.pw = nn.Conv2d(width, width, 1, bias=True)
        se_hidden = max(16, width // 4)
        self.se1 = nn.Conv2d(width, se_hidden, 1, bias=True)
        self.se2 = nn.Conv2d(se_hidden, width, 1, bias=True)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        x0, x1, x2, x3 = x.chunk(4, dim=1)
        y0 = self.dw1(x0)
        y1 = self.dw2(x1)
        y2 = self.dw4(x2)
        y3 = self.dw_h(x3) + self.dw_v(x3)
        h = self.act(self.pw(torch.cat([y0, y1, y2, y3], dim=1)))
        se = torch.sigmoid(self.se2(self.act(self.se1(h.mean(dim=(2, 3), keepdim=True)))))
        return x + h * se


class _V3PyramidStage(nn.Module):
    def __init__(
        self,
        enc_ch: int,
        deg_ch: int,
        width: int,
        prev_width: int = 0,
        arch_variant: str = "v30",
    ):
        super().__init__()
        self.arch_variant = str(arch_variant)
        in_ch = enc_ch + deg_ch + prev_width
        self.in_proj = nn.Conv2d(in_ch, width, 1, bias=True)
        self.block1 = _V3DilatedStripSEBlock(width)
        self.block2 = _V3DilatedStripSEBlock(width)
        self.out_proj = nn.Conv2d(width, enc_ch, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        enc: Tensor,
        d_map: Tensor,
        prev_h: Optional[Tensor],
        w_map: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        parts = [enc, d_map]
        if prev_h is not None:
            parts.append(F.interpolate(prev_h, size=enc.shape[-2:], mode="bilinear", align_corners=False))
        h = self.in_proj(torch.cat(parts, dim=1))
        h = self.block2(self.block1(h))
        delta = self.out_proj(h)
        if self.arch_variant == "v33":
            # Soft norm cap: rms(delta_capped) <= 0.35 * rms(enc) per sample
            enc_rms = enc.float().pow(2).mean(dim=(1, 2, 3), keepdim=True).add(1e-8).sqrt()
            delta_rms = delta.float().pow(2).mean(dim=(1, 2, 3), keepdim=True).add(1e-8).sqrt()
            max_rms = (0.35 * enc_rms).clamp_min(1e-8)
            cap_scale = (max_rms * torch.tanh(delta_rms / max_rms)) / delta_rms.clamp_min(1e-8)
            delta_capped = delta * cap_scale.to(dtype=delta.dtype)
            if w_map is not None:
                gate_sft = ((w_map.to(dtype=enc.dtype) - 0.55) / 0.35).clamp(0.0, 1.0)
                return enc + gate_sft * delta_capped, h
            return enc + delta_capped, h
        return enc + delta, h


class V3PyramidSFT(nn.Module):
    """Module C (part 1): Top-down 32->64->128->256 dilated + strip SE pyramid.

    Zero-init residual so step-0 enc_used[s] == enc_feat[s].
    In arch_variant='v33', gates residual by clamp((w_map - 0.55) / 0.35, 0, 1)
    and soft-caps delta RMS to <= 0.35 * rms(enc_s).
    """

    def __init__(self, arch_variant: str = "v30"):
        super().__init__()
        if arch_variant not in ARCH_VARIANTS:
            raise ValueError(f"Unsupported arch_variant={arch_variant!r}; expected one of {ARCH_VARIANTS}")
        self.arch_variant = str(arch_variant)
        self.stages = nn.ModuleDict()
        prev_w = 0
        for site in SITES:
            w = PYR_WIDTHS[site]
            self.stages[site] = _V3PyramidStage(
                CHANNELS[site], D_CHANNELS[site], w, prev_width=prev_w, arch_variant=self.arch_variant
            )
            prev_w = w

    def forward(
        self,
        enc_feat_dict: Mapping[str, Tensor],
        d_maps: Mapping[str, Tensor],
        w_maps: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        enc_used: Dict[str, Tensor] = {}
        prev_h: Optional[Tensor] = None
        for site in SITES:
            w_s = w_maps[site] if (w_maps is not None and site in w_maps) else None
            used, prev_h = self.stages[site](enc_feat_dict[site], d_maps[site], prev_h, w_map=w_s)
            enc_used[site] = used
        return enc_used


def _hp5(x: Tensor) -> Tensor:
    low = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    return x - low


class _V3FreqHeadSite(nn.Module):
    """Per-site zero-init high-frequency injection + spatial residual head (v30)
    or pure non-negative HP5 VQ detail transfer only (v33 / hp5_only).
    """

    def __init__(self, ch: int, d_ch: int, width: int, arch_variant: str = "v30"):
        super().__init__()
        self.arch_variant = str(arch_variant)
        gate_mid = max(16, ch // 4)
        self.freq_gate = nn.Sequential(
            nn.Conv2d(2 * ch, gate_mid, 1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(gate_mid, ch, 1, bias=True),
        )
        nn.init.zeros_(self.freq_gate[-1].weight)
        if self.arch_variant == "v33":
            nn.init.constant_(self.freq_gate[-1].bias, -2.0)  # 0.75 * sigmoid(-2.0) ~= 0.09 gentle non-negative start
            return

        nn.init.zeros_(self.freq_gate[-1].bias)
        self.enc_p = nn.Conv2d(ch, width, 1, bias=True)
        self.dec_p = nn.Conv2d(ch, width, 1, bias=False)
        self.fus_p = nn.Conv2d(ch, width, 1, bias=False)
        self.coa_p = nn.Conv2d(CHANNELS["32"], width, 1, bias=False)
        self.deg_p = nn.Conv2d(d_ch, width, 1, bias=False)

        self.dw1 = nn.Conv2d(width, width, 3, padding=1, dilation=1, groups=width, bias=True)
        self.dw2 = nn.Conv2d(width, width, 3, padding=2, dilation=2, groups=width, bias=True)
        self.pw = nn.Conv2d(2 * width, width, 1, bias=True)
        self.pw2 = nn.Conv2d(width, width, 1, bias=True)
        self.out_proj = nn.Conv2d(width, ch, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.act = nn.GELU()

    def forward(
        self,
        enc_used: Tensor,
        dec: Tensor,
        fused: Tensor,
        coarse_32: Tensor,
        d_map: Tensor,
        *,
        return_gate: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if self.arch_variant == "v33":
            gate = 0.75 * torch.sigmoid(self.freq_gate(torch.cat([dec, fused], dim=1)))
            out = fused + gate * _hp5(dec - fused)
            if return_gate:
                return out, gate
            return out

        gate = torch.tanh(self.freq_gate(torch.cat([dec, fused], dim=1)))
        hf_term = gate * _hp5(dec - fused)

        coa = F.interpolate(self.coa_p(coarse_32), size=dec.shape[-2:], mode="bilinear", align_corners=False)
        mixed = self.act((self.enc_p(enc_used) + self.dec_p(dec) + self.fus_p(fused) + coa + self.deg_p(d_map)) * 0.5)
        sp = self.act(self.pw(torch.cat([self.dw1(mixed), self.dw2(mixed)], dim=1)))
        sp = sp + self.act(self.pw2(sp))
        r_spatial = self.out_proj(sp)
        out = fused + hf_term + r_spatial
        if return_gate:
            return out, gate
        return out


class V3FreqHead(nn.Module):
    """Module C (part 2): Multi-site frequency-selective detail injection (+ spatial residual in v30; hp5_only in v33)."""

    def __init__(self, arch_variant: str = "v30"):
        super().__init__()
        if arch_variant not in ARCH_VARIANTS:
            raise ValueError(f"Unsupported arch_variant={arch_variant!r}; expected one of {ARCH_VARIANTS}")
        self.arch_variant = str(arch_variant)
        self.sites = nn.ModuleDict({
            site: _V3FreqHeadSite(CHANNELS[site], D_CHANNELS[site], FREQ_WIDTHS[site], arch_variant=self.arch_variant)
            for site in SITES
        })

    def forward_site(
        self,
        site: str,
        enc_used: Tensor,
        dec: Tensor,
        fused: Tensor,
        coarse_32: Tensor,
        d_map: Tensor,
        *,
        return_gate: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        return self.sites[site](enc_used, dec, fused, coarse_32, d_map, return_gate=return_gate)


class V3OutputComposer(nn.Module):
    """Module D: Antialiased 512->256 downsampling + adaptive clean-input bypass + clamped residual."""

    def __init__(self, d256_dim: int = D_CHANNELS["256"], width: int = 32):
        super().__init__()
        in_ch = 3 + 3 + 3 + d256_dim + 1  # dec_256, x_pad_256, diff, D_256, observed_mask
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, width, 1, bias=False),
            _gn(width),
            nn.SiLU(inplace=False),
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.Conv2d(width, width, 1, bias=False),
            _gn(width),
            nn.SiLU(inplace=False),
        )
        self.out_conv = nn.Conv2d(width, 4, 1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        with torch.no_grad():
            self.out_conv.bias.copy_(torch.tensor([-2.2, 0.0, 0.0, 0.0], dtype=torch.float32))

    def forward(
        self,
        dec_512: Tensor,
        x_in_256: Tensor,
        d256: Tensor,
        observed_mask_256: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dec_256 = F.interpolate(
            dec_512.float(), size=(256, 256), mode="bicubic", align_corners=False, antialias=True
        ).clamp(0.0, 1.0).to(dtype=dec_512.dtype)
        mask_f = observed_mask_256.to(dtype=dec_256.dtype)
        inp = torch.cat([dec_256, x_in_256, dec_256 - x_in_256, d256, mask_f], dim=1)
        out = self.out_conv(self.net(inp))
        alpha_bypass = torch.sigmoid(out[:, 0:1])
        r_out = 0.1 * torch.tanh(out[:, 1:4])
        y_256 = ((1.0 - alpha_bypass) * (dec_256 + r_out) + alpha_bypass * x_in_256).clamp(0.0, 1.0)
        return y_256, alpha_bypass, r_out


# ---------------------------------------------------------------------------
# V3.4 ("Pre-Conditioned Frozen CodeFormer") Front-End Modules
# ---------------------------------------------------------------------------


def _compute_v34_hf_channels(x_rgb: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute (luma_hf, chroma_hf_1px, chroma_hf_coarse) from RGB tensor in [0, 1].

    Computed strictly in FP32 (disabling outer bf16/fp16 autocast) so 7-bit bf16
    mantissa never truncates small chroma high-frequency differences.
    """
    with torch.amp.autocast(device_type=x_rgb.device.type, enabled=False):
        xf = x_rgb.float()
        r, g, b = xf[:, 0:1], xf[:, 1:2], xf[:, 2:3]
        luma = LUMA_WEIGHTS[0] * r + LUMA_WEIGHTS[1] * g + LUMA_WEIGHTS[2] * b
        luma_avg = F.avg_pool2d(F.pad(luma, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        luma_hf = (luma - luma_avg).mul(10.0).clamp(-1.0, 1.0)

        cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
        cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
        cbcr = torch.cat([cb, cr], dim=1)

        cbcr_avg3 = F.avg_pool2d(F.pad(cbcr, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        chroma_hf_1px = (cbcr - cbcr_avg3).abs().mean(dim=1, keepdim=True).mul(20.0).clamp(0.0, 1.0)

        cbcr_avg2 = F.avg_pool2d(F.pad(cbcr, (0, 1, 0, 1), mode="reflect"), kernel_size=2, stride=1)
        cbcr_avg6 = F.avg_pool2d(F.pad(cbcr, (2, 3, 2, 3), mode="reflect"), kernel_size=6, stride=1)
        chroma_hf_coarse = (cbcr_avg2 - cbcr_avg6).abs().mean(dim=1, keepdim=True).mul(20.0).clamp(0.0, 1.0)

    return luma_hf, chroma_hf_1px, chroma_hf_coarse


class V34DegradationEncoder(nn.Module):
    """V3.4 Front-End Module 1: 7-ch degradation encoder (~0.29M params, NO FPN decoder).

    Input channels (256x256):
        [R, G, B, observed_mask, luma_hf, chroma_hf_1px, chroma_hf_coarse]
    """

    W_MIN: float = 0.60
    W_MAX: float = 0.98
    W_INIT: float = 0.82

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim

        # 4-stage MBConv encoder with widths [24, 40, 56, 80, 112] (stem s=2 -> 128 -> 64 -> 32 -> 16 -> 8)
        self.stem = nn.Sequential(
            nn.Conv2d(7, 24, 3, stride=2, padding=1, bias=False),
            _gn(24),
            nn.SiLU(inplace=False),
        )
        self.stage1 = nn.Sequential(
            _MBConvBlock(24, 40, stride=2, expand=2),
            _MBConvBlock(40, 40, stride=1, expand=2),
        )
        self.stage2 = nn.Sequential(
            _MBConvBlock(40, 56, stride=2, expand=2),
            _MBConvBlock(56, 56, stride=1, expand=2),
        )
        self.stage3 = nn.Sequential(
            _MBConvBlock(56, 80, stride=2, expand=2),
            _MBConvBlock(80, 80, stride=1, expand=2),
        )
        self.stage4 = nn.Sequential(
            _MBConvBlock(80, 112, stride=2, expand=2),
            _MBConvBlock(112, 112, stride=1, expand=2),
        )

        # Masked global avg + max pool -> z_deg (B, 128)
        self.z_proj = nn.Sequential(
            nn.Linear(224, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(inplace=False),
            nn.Linear(z_dim, z_dim),
        )

        # Scalar heads from z_deg
        self.deg_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.SiLU(inplace=False),
            nn.Linear(64, 4),
        )
        self.g_photo_head = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, 1),
        )
        self.g_pre_head = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, 1),
        )
        self.w_head = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, 1),
        )
        self.clean_head = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.SiLU(inplace=False),
            nn.Linear(32, 1),
        )

        # Sensible step-0 gate initializations
        nn.init.zeros_(self.g_photo_head[-1].weight)
        nn.init.constant_(self.g_photo_head[-1].bias, -2.0)
        nn.init.zeros_(self.g_pre_head[-1].weight)
        nn.init.constant_(self.g_pre_head[-1].bias, -2.0)
        nn.init.zeros_(self.w_head[-1].weight)
        nn.init.constant_(self.w_head[-1].bias, _logit((self.W_INIT - self.W_MIN) / (self.W_MAX - self.W_MIN)))
        nn.init.zeros_(self.clean_head[-1].weight)
        nn.init.constant_(self.clean_head[-1].bias, 0.0)

    def forward(self, x_pad_256: Tensor, observed_mask_256: Tensor) -> Dict[str, Any]:
        mask_f = observed_mask_256.to(dtype=x_pad_256.dtype)
        luma_hf, chroma_hf_1px, chroma_hf_coarse = _compute_v34_hf_channels(x_pad_256)
        inp7 = torch.cat(
            [
                x_pad_256,
                mask_f,
                luma_hf.to(dtype=x_pad_256.dtype),
                chroma_hf_1px.to(dtype=x_pad_256.dtype),
                chroma_hf_coarse.to(dtype=x_pad_256.dtype),
            ],
            dim=1,
        )

        f128 = self.stem(inp7)
        f64 = self.stage1(f128)
        f32 = self.stage2(f64)
        f16 = self.stage3(f32)
        f8 = self.stage4(f16)
        f4 = F.adaptive_avg_pool2d(f8, (4, 4))

        # Masked global avg + max pooling over f8 (8x8)
        m8 = F.interpolate(mask_f, size=f8.shape[-2:], mode="area")
        m8_bin = (m8 > 0.05).to(dtype=f8.dtype)
        has_obs_8 = m8_bin.sum(dim=(2, 3)) > 0.5
        m8_eff = torch.where(has_obs_8.view(-1, 1, 1, 1), m8_bin, torch.ones_like(m8_bin))
        denom8 = m8_eff.sum(dim=(2, 3)).clamp(min=1.0)
        avg_pool = (f8 * m8_eff).sum(dim=(2, 3)) / denom8
        max_pool = f8.masked_fill(m8_eff < 0.5, -1e4).amax(dim=(2, 3))
        z_deg = self.z_proj(torch.cat([avg_pool, max_pool], dim=1))

        # Observed mask at 16x16 for MaskedAdaIN
        m16 = F.interpolate(mask_f, size=(16, 16), mode="area")
        m16_bin = (m16 > 0.05).to(dtype=f8.dtype)
        has_obs_16 = m16_bin.sum(dim=(2, 3)) > 0.5
        m16_eff = torch.where(has_obs_16.view(-1, 1, 1, 1), m16_bin, torch.ones_like(m16_bin))

        with torch.amp.autocast(device_type=z_deg.device.type, enabled=False):
            z_f = z_deg.float()
            deg_raw = torch.sigmoid(self.deg_head(z_f))
            p_exp = deg_raw[:, 0:1]
            p_blur = deg_raw[:, 1:2]
            p_noise = deg_raw[:, 2:3]
            p_grid = deg_raw[:, 3:4]
            deg_preds = {
                "exposure": p_exp,
                "blur": p_blur,
                "noise": p_noise,
                "codec_lr": p_grid,
            }

            g_photo = torch.sigmoid(self.g_photo_head(z_f))
            g_pre = torch.sigmoid(self.g_pre_head(z_f))
            w_pred = self.W_MIN + (self.W_MAX - self.W_MIN) * torch.sigmoid(self.w_head(z_f))

            p_clean = (
                torch.sigmoid(self.clean_head(z_f))
                * (1.0 - p_exp)
                * (1.0 - p_blur)
                * (1.0 - p_noise)
                * (1.0 - p_grid)
            )
            if self.training:
                alpha_clean = torch.sigmoid((p_clean - 0.80) * 18.0)
            else:
                t = ((p_clean - 0.72) / 0.20).clamp(0.0, 1.0)
                alpha_clean = t * t * (3.0 - 2.0 * t)

        return {
            "z_deg": z_deg,
            "f4": f4,
            "deg_preds": deg_preds,
            "g_photo": g_photo,
            "g_pre": g_pre,
            "w_pred": w_pred,
            "p_clean": p_clean,
            "alpha_clean": alpha_clean,
            "observed_mask_16": m16_eff,
        }


class V34PhotoCorrectNet(nn.Module):
    """V3.4 Front-End Module 2: Highlight-pinned black-level dehaze + Reinhard rational tone curve
    + gamma + luma-neutral WB (~0.04M params).

    Exact identity at zero-init and mathematically pinned at f(1) = 1 in FP32.
    """

    B_HAZE_MAX: float = 0.42
    G_TONE_MIN: float = 0.80
    G_TONE_MAX: float = 5.00
    GAMMA_MIN: float = 0.45
    GAMMA_MAX: float = 1.25
    WB_MIN: float = 0.80
    WB_MAX: float = 1.25

    def __init__(self, f4_dim: int = 112, z_dim: int = Z_DIM):
        super().__init__()
        self.z_proj = nn.Linear(z_dim, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(f4_dim + 32, 24, 3, padding=1, bias=False),
            _gn(24),
            nn.SiLU(inplace=False),
            nn.Conv2d(24, 24, 3, padding=1, bias=False),
            _gn(24),
            nn.SiLU(inplace=False),
        )
        self.param_head = nn.Conv2d(24, 5, 1, bias=True)
        nn.init.zeros_(self.param_head.weight)
        nn.init.zeros_(self.param_head.bias)

        self.wb_head = nn.Linear(z_dim, 3, bias=True)
        nn.init.zeros_(self.wb_head.weight)
        nn.init.zeros_(self.wb_head.bias)

        self._g_tone_offset = _logit((1.0 - self.G_TONE_MIN) / (self.G_TONE_MAX - self.G_TONE_MIN))
        self._gamma_offset = _logit((1.0 - self.GAMMA_MIN) / (self.GAMMA_MAX - self.GAMMA_MIN))
        self._wb_offset = _logit((1.0 - self.WB_MIN) / (self.WB_MAX - self.WB_MIN))

    def forward(
        self,
        x_pad_256: Tensor,
        f4: Tensor,
        z_deg: Tensor,
        g_photo: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        b = x_pad_256.shape[0]
        z_sp = self.z_proj(z_deg).view(b, 32, 1, 1).expand(-1, -1, 4, 4)
        raw = self.param_head(self.conv(torch.cat([f4, z_sp], dim=1)))  # (B, 5, 4, 4)
        raw_wb = self.wb_head(z_deg)  # (B, 3)

        with torch.amp.autocast(device_type=x_pad_256.device.type, enabled=False):
            raw_f = raw.float()
            raw_b = raw_f[:, 0:3]
            raw_g = raw_f[:, 3:4]
            raw_gam = raw_f[:, 4:5]

            b_haze_4 = self.B_HAZE_MAX * torch.sigmoid(12.0 * raw_b - 11.0)
            g_tone_4 = self.G_TONE_MIN + (self.G_TONE_MAX - self.G_TONE_MIN) * torch.sigmoid(
                raw_g + self._g_tone_offset
            )
            gamma_4 = self.GAMMA_MIN + (self.GAMMA_MAX - self.GAMMA_MIN) * torch.sigmoid(
                raw_gam + self._gamma_offset
            )
            photo_params_4x4 = torch.cat([b_haze_4, g_tone_4, gamma_4], dim=1)

            b_haze = F.interpolate(b_haze_4, size=(256, 256), mode="bicubic", align_corners=False).clamp(
                0.0, self.B_HAZE_MAX
            )
            g_tone = F.interpolate(g_tone_4, size=(256, 256), mode="bicubic", align_corners=False).clamp(
                self.G_TONE_MIN, self.G_TONE_MAX
            )
            gamma = F.interpolate(gamma_4, size=(256, 256), mode="bicubic", align_corners=False).clamp(
                self.GAMMA_MIN, self.GAMMA_MAX
            )

            wb_raw = self.WB_MIN + (self.WB_MAX - self.WB_MIN) * torch.sigmoid(raw_wb.float() + self._wb_offset)
            wb_geo = (wb_raw[:, 0:1] * wb_raw[:, 1:2] * wb_raw[:, 2:3]).clamp(min=1e-6).pow(1.0 / 3.0)
            wb = (wb_raw / wb_geo).view(b, 3, 1, 1)

            x_f = x_pad_256.float().clamp(0.0, 1.0)
            # 1. Black-level dehaze (strictly <= x_f, pinned at f(1)=1)
            x1 = ((x_f - b_haze) / (1.0 - b_haze).clamp(min=0.55)).clamp(0.0, 1.0)
            # 2. Highlight-pinned Reinhard rational tone + smooth gamma (exact identity at g=1, gamma=1; pinned f(1)=1)
            r = (g_tone * x1) / (1.0 + (g_tone - 1.0) * x1).clamp(min=1e-6)
            eps = 1e-4
            eps_g = eps ** gamma
            one_g = (1.0 + eps) ** gamma
            x2 = ((r + eps).pow(gamma) - eps_g) / (one_g - eps_g).clamp(min=1e-6)
            # 3. Luma-neutral WB + scalar gate
            x_corr = (x2 * wb).clamp(0.0, 1.0)
            g_p = g_photo.float().view(b, 1, 1, 1)
            x_photo = (x_f + g_p * (x_corr - x_f)).clamp(0.0, 1.0).to(dtype=x_pad_256.dtype)

        return x_photo, photo_params_4x4


class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors computed in FP32."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.float()
        mean = x_f.mean(dim=1, keepdim=True)
        var = (x_f - mean).pow(2).mean(dim=1, keepdim=True)
        out = (x_f - mean) / torch.sqrt(var + self.eps)
        out = out * self.weight.float() + self.bias.float()
        return out.to(dtype=x.dtype)


class _NAFBlock(nn.Module):
    """NAFNet building block with SimpleGate, Simplified Channel Attention (SCA),
    LayerNorm2d, and zero-init residual scaling (beta, gamma = 0).
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch = c * dw_expand
        ffn_ch = c * ffn_expand
        self.norm1 = _LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=True)
        self.sca = nn.Conv2d(dw_ch // 2, dw_ch // 2, 1, bias=True)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1, bias=True)

        self.norm2 = _LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_ch, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    @staticmethod
    def _simple_gate(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self._simple_gate(y)
        y = y * self.sca(y.mean(dim=(2, 3), keepdim=True))
        y = self.conv3(y)
        x = x + y * self.beta

        y2 = self.norm2(x)
        y2 = self.conv4(y2)
        y2 = self._simple_gate(y2)
        y2 = self.conv5(y2)
        return x + y2 * self.gamma


class V34NAFPreConditionNet(nn.Module):
    """V3.4 Front-End Module 3: 3-level NAFNet-Tiny UNet at 256x256 (~1.95M params).

    Removes non-optical corruptions (confetti, NN pixelation, macroblocks, interlace)
    into a clean intermediate `x_pre_256` before invoking frozen CodeFormer.
    Exact identity at step 0 via zero-init 3x3 output head.
    """

    def __init__(
        self,
        c: int = 32,
        enc_blks: Tuple[int, ...] = (1, 2, 2),
        mid_blks: int = 2,
        dec_blks: Tuple[int, ...] = (2, 2, 1),
        z_dim: int = Z_DIM,
    ):
        super().__init__()
        self.intro = nn.Conv2d(6, c, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = c
        for n_blk in enc_blks:
            self.encoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n_blk)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
            ch *= 2

        self.middle = nn.Sequential(*[_NAFBlock(ch) for _ in range(mid_blks)])
        self.mid_film = nn.Linear(z_dim, ch * 2, bias=True)
        nn.init.zeros_(self.mid_film.weight)
        nn.init.zeros_(self.mid_film.bias)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n_blk in dec_blks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            ch //= 2
            self.decoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n_blk)]))

        self.out_conv = nn.Conv2d(c, 3, 3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x_photo_256: Tensor,
        observed_mask_256: Tensor,
        z_deg: Tensor,
        g_pre: Tensor,
    ) -> Tensor:
        b = x_photo_256.shape[0]
        mask_f = observed_mask_256.to(dtype=x_photo_256.dtype)
        _, chroma_hf_1px, chroma_hf_coarse = _compute_v34_hf_channels(x_photo_256)
        inp6 = torch.cat(
            [
                x_photo_256,
                mask_f,
                chroma_hf_1px.to(dtype=x_photo_256.dtype),
                chroma_hf_coarse.to(dtype=x_photo_256.dtype),
            ],
            dim=1,
        )

        h = self.intro(inp6)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.middle(h)
        film_scale, film_shift = self.mid_film(z_deg).chunk(2, dim=1)
        h = h * (1.0 + film_scale.view(b, -1, 1, 1)) + film_shift.view(b, -1, 1, 1)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h) + skip
            h = dec(h)

        delta_pre = 0.5 * torch.tanh(self.out_conv(h))
        g_p = g_pre.view(b, 1, 1, 1).to(dtype=x_photo_256.dtype)
        return (x_photo_256 + g_p * delta_pre).clamp(0.0, 1.0)


def delivered_views_v3(
    raw: np.ndarray,
    canvas: np.ndarray,
    observed: np.ndarray,
    *,
    rounding: str = "floor",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose exact prepared pixels outside observed support for both FP32 RAW and uint8 PNG.

    Matches integrated_app/implementation/runtime.py:delivered_views when rounding='floor'.
    """
    if not (
        isinstance(raw, np.ndarray) and raw.shape == (256, 256, 3) and raw.dtype == np.float32
        and np.isfinite(raw).all() and float(raw.min()) >= 0.0 and float(raw.max()) <= 1.0
    ):
        raise ValueError("Expected finite float32 (256, 256, 3) raw image in [0, 1].")
    if not (
        isinstance(canvas, np.ndarray) and canvas.shape == (256, 256, 3) and canvas.dtype == np.uint8
    ):
        raise ValueError("Expected uint8 (256, 256, 3) RGB canvas.")
    if not (
        isinstance(observed, np.ndarray) and observed.shape == (256, 256) and observed.dtype == bool
    ):
        raise ValueError("Expected bool (256, 256) observed mask.")

    raw_out = raw.copy()
    raw_out[~observed] = canvas[~observed].astype(np.float32) / np.float32(255.0)
    if rounding == "floor":
        pixels = np.floor(raw_out * np.float32(255.0)).astype(np.uint8)
    elif rounding == "rint":
        pixels = np.clip(np.rint(raw_out * np.float32(255.0)), 0.0, 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported rounding mode: {rounding}")
    pixels[~observed] = canvas[~observed]
    return raw_out, pixels


class CodeFormerV3Restorer(nn.Module):
    """Unified single-pass CodeFormer Restoration V3 model.

    Holds:
      - self.base: 100% frozen official CodeFormer (515 tensors, requires_grad=False, eval mode)
      - self.v3:   nn.ModuleDict of all trainable V3 modules (~3.55M params)
    """

    def __init__(
        self,
        codeformer_weights_path: Optional[str | Path] = None,
        *,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        grad_checkpoint: bool = False,
        arch_variant: str = "v30",
    ):
        super().__init__()
        if arch_variant not in ARCH_VARIANTS:
            raise ValueError(f"Unsupported arch_variant={arch_variant!r}; expected one of {ARCH_VARIANTS}")
        self.arch_variant = str(arch_variant)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.base = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=list(SITES),
            fix_modules=None,
        )
        if codeformer_weights_path is not None:
            self.load_official_codeformer(codeformer_weights_path)
        self._freeze_base()

        if self.arch_variant == "v34":
            v3_mods: Dict[str, nn.Module] = {
                "deg_encoder": V34DegradationEncoder(z_dim=Z_DIM),
                "photo_net": V34PhotoCorrectNet(f4_dim=112, z_dim=Z_DIM),
                "pre_net": V34NAFPreConditionNet(
                    c=32, enc_blks=(1, 2, 2), mid_blks=2, dec_blks=(2, 2, 1), z_dim=Z_DIM
                ),
            }
        else:
            v3_mods = {
                "deg_encoder": V3DegradationEncoder(z_dim=Z_DIM, arch_variant=self.arch_variant),
                "photo_net": PhotoCorrectNet(d32_dim=D_CHANNELS["32"], z_dim=Z_DIM, arch_variant=self.arch_variant),
            }
            if self.arch_variant == "v30":
                v3_mods["transformer_lora"] = DegGatedTransformerLoRA(
                    n_layers=9,
                    embed_dim=512,
                    dim_mlp=1024,
                    n_head=8,
                    r=lora_rank,
                    alpha=lora_alpha,
                    z_dim=Z_DIM,
                    d16_dim=D_CHANNELS["16"],
                )
            v3_mods["pyramid_sft"] = V3PyramidSFT(arch_variant=self.arch_variant)
            v3_mods["freq_head"] = V3FreqHead(arch_variant=self.arch_variant)
            v3_mods["output_composer"] = V3OutputComposer(d256_dim=D_CHANNELS["256"], width=32)
        self.v3 = nn.ModuleDict(v3_mods)
        self.masked_adain = MaskedAdaIN()

        self._enc_out_indices = {self.base.fuse_encoder_block[s] for s in SITES}
        self._gen_fuse_map = {self.base.fuse_generator_block[s]: s for s in SITES}

    def _freeze_base(self) -> None:
        self.base.eval()
        for param in self.base.parameters():
            param.requires_grad = False

    def load_official_codeformer(self, weights_path: str | Path) -> None:
        saved = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
        sd = saved["params_ema"] if isinstance(saved, dict) and "params_ema" in saved else saved
        if not isinstance(sd, dict) or len(sd) != BASE_TENSOR_COUNT:
            raise ValueError(f"Expected {BASE_TENSOR_COUNT} tensors in official codeformer.pth, got {len(sd) if isinstance(sd, dict) else type(sd)}.")
        self.base.load_state_dict(sd, strict=True)
        self._freeze_base()

    def train(self, mode: bool = True) -> "CodeFormerV3Restorer":
        super().train(mode)
        self.base.eval()  # Official backbone is always frozen in eval mode
        return self

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def base_parameter_count(self) -> int:
        return sum(p.numel() for p in self.base.parameters())

    def module_param_breakdown(self) -> Dict[str, int]:
        return {
            name: sum(p.numel() for p in mod.parameters() if p.requires_grad)
            for name, mod in self.v3.items()
        }

    def encode_gt_teacher(
        self,
        gt_512: Tensor,
        observed_mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Run frozen official CodeFormer encoder + transformer on clean GT 512x512 in [0, 1].

        When observed_mask is provided, upsamples it to 512x512 (nearest) and applies
        replicate_pad_observed before encoding so teacher targets match student geometry.
        """
        with torch.no_grad():
            gt_in = gt_512
            if observed_mask is not None:
                obs = observed_mask.to(device=gt_512.device, dtype=gt_512.dtype)
                if obs.ndim == 3:
                    obs = obs.unsqueeze(1)
                if obs.shape[-2:] != gt_512.shape[-2:]:
                    obs = F.interpolate(obs, size=gt_512.shape[-2:], mode="nearest")
                gt_in = replicate_pad_observed(gt_512, obs)
            x = gt_in * 2.0 - 1.0
            enc_gt: Dict[str, Tensor] = {}
            for i, block in enumerate(self.base.encoder.blocks):
                x = block(x)
                if i in self._enc_out_indices:
                    enc_gt[str(x.shape[-1])] = x.detach()
            lq_gt = x.detach()
            pos_emb = self.base.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
            q_emb = self.base.feat_emb(lq_gt.flatten(2).permute(2, 0, 1))
            for layer in self.base.ft_layers:
                q_emb = layer(q_emb, query_pos=pos_emb)
            logits_gt = self.base.idx_pred_layer(q_emb).permute(1, 0, 2)
            top_idx_gt = logits_gt.argmax(dim=2)  # (B, 256)
            return {
                "enc_gt": enc_gt,
                "lq_feat_16_gt": lq_gt,
                "code_indices_gt": top_idx_gt,
                "code_logits_gt": logits_gt,
            }

    def _forward_v34(
        self,
        x_pad_256: Tensor,
        obs_mask: Tensor,
        *,
        skip_decoder: bool = False,
    ) -> Dict[str, Any]:
        b = x_pad_256.shape[0]
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()

        # 1. Front-end Module 1: V34DegradationEncoder
        deg_out = self.v3["deg_encoder"](x_pad_256, obs_mask)
        z_deg = deg_out["z_deg"]
        obs_mask_16 = deg_out["observed_mask_16"]

        # 2. Front-end Module 2: V34PhotoCorrectNet
        x_photo_256, photo_params_4x4 = self.v3["photo_net"](
            x_pad_256, deg_out["f4"], z_deg, deg_out["g_photo"]
        )

        # 3. Front-end Module 3: V34NAFPreConditionNet
        x_pre_256 = self.v3["pre_net"](x_photo_256, obs_mask, z_deg, deg_out["g_pre"])

        # 4. Frozen Official CodeFormer Encoder + Transformer on x_pre_512
        x_pre_512 = F.interpolate(x_pre_256, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
        enc_feat_dict: Dict[str, Tensor] = {}
        x = x_pre_512
        for i, block in enumerate(self.base.encoder.blocks):
            x = _grad_ckpt(block, x, use_reentrant=False) if use_ckpt else block(x)
            if i in self._enc_out_indices:
                enc_feat_dict[str(x.shape[-1])] = x
        lq_feat_16 = x  # (B, 256, 16, 16)

        pos_emb = self.base.position_emb.unsqueeze(1).repeat(1, b, 1)
        query_emb = self.base.feat_emb(lq_feat_16.flatten(2).permute(2, 0, 1))
        for layer in self.base.ft_layers:
            if use_ckpt:
                query_emb = _grad_ckpt(
                    lambda q, p, _l=layer: _l(q, query_pos=p), query_emb, pos_emb, use_reentrant=False
                )
            else:
                query_emb = layer(query_emb, query_pos=pos_emb)
        code_logits = self.base.idx_pred_layer(query_emb).permute(1, 0, 2)  # (B, 256, 1024)

        if skip_decoder:
            return {
                "y_256": None,
                "y_512": None,
                "dec_256": None,
                "x_pad_256": x_pad_256,
                "x_photo_256": x_photo_256,
                "photo_params_4x4": photo_params_4x4,
                "x_pre_256": x_pre_256,
                "code_logits": code_logits,
                "lq_feat_16": lq_feat_16,
                "deg_preds": deg_out["deg_preds"],
                "g_photo": deg_out["g_photo"],
                "g_pre": deg_out["g_pre"],
                "w_pred": deg_out["w_pred"],
                "p_clean": deg_out["p_clean"],
                "alpha_clean": deg_out["alpha_clean"],
                "alpha_bypass": deg_out["alpha_clean"],
            }

        # 5. Quantization: hard top-1 z_hard + training-only FP32 Softmax STE (tau=1.0)
        soft_one_hot = F.softmax(code_logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        z_hard = self.base.quantize.get_codebook_feat(top_idx, shape=[b, 16, 16, 256]).detach()
        if self.training and torch.is_grad_enabled():
            with torch.amp.autocast(device_type=code_logits.device.type, enabled=False):
                probs = F.softmax(code_logits.float(), dim=2)
                emb_w = self.base.quantize.embedding.weight.float()
                z_soft = torch.matmul(probs, emb_w).permute(0, 2, 1).reshape(b, 256, 16, 16)
                quant_feat = (z_hard.float() + (z_soft - z_soft.detach())).to(dtype=z_hard.dtype)
        else:
            quant_feat = z_hard

        # 6. MaskedAdaIN with fixed alpha_adain = 1.0
        alpha_adain_one = torch.ones((b, 1, 1, 1), device=x_pad_256.device, dtype=quant_feat.dtype)
        quant_feat = self.masked_adain(quant_feat, lq_feat_16, obs_mask_16, alpha_adain_one)

        # 7. Untouched Frozen CF Generator + Fuse_sft_block with scalar w_pred
        x_dec = quant_feat
        w_scalar = deg_out["w_pred"].view(b, 1, 1, 1).to(dtype=x_dec.dtype)
        for i, block in enumerate(self.base.generator.blocks):
            x_dec = _grad_ckpt(block, x_dec, use_reentrant=False) if use_ckpt else block(x_dec)
            if i in self._gen_fuse_map:
                site = self._gen_fuse_map[i]
                fuse_blk = self.base.fuse_convs_dict[site]
                if use_ckpt:
                    x_dec = _grad_ckpt(
                        lambda e, d, w_t, _f=fuse_blk: _f(e, d, w_t),
                        enc_feat_dict[site],
                        x_dec,
                        w_scalar,
                        use_reentrant=False,
                    )
                else:
                    x_dec = fuse_blk(enc_feat_dict[site], x_dec, w_scalar)

        y_512 = ((x_dec + 1.0) * 0.5).clamp(0.0, 1.0)
        dec_256 = F.interpolate(
            y_512.float(), size=(256, 256), mode="bicubic", align_corners=False, antialias=True
        ).clamp(0.0, 1.0).to(dtype=y_512.dtype)

        alpha_clean = deg_out["alpha_clean"].view(b, 1, 1, 1).to(dtype=dec_256.dtype)
        y_256 = ((1.0 - alpha_clean) * dec_256 + alpha_clean * x_pad_256).clamp(0.0, 1.0)

        return {
            "y_256": y_256,
            "y_512": y_512,
            "dec_256": dec_256,
            "x_pad_256": x_pad_256,
            "x_photo_256": x_photo_256,
            "photo_params_4x4": photo_params_4x4,
            "x_pre_256": x_pre_256,
            "code_logits": code_logits,
            "lq_feat_16": lq_feat_16,
            "deg_preds": deg_out["deg_preds"],
            "g_photo": deg_out["g_photo"],
            "g_pre": deg_out["g_pre"],
            "w_pred": deg_out["w_pred"],
            "p_clean": deg_out["p_clean"],
            "alpha_clean": deg_out["alpha_clean"],
            "alpha_bypass": deg_out["alpha_clean"],
        }

    def forward(
        self,
        x_256: Tensor,
        observed_mask: Optional[Tensor] = None,
        *,
        stage1_only: bool = False,
        skip_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Unified single-pass forward.

        Args:
            x_256: (B, 3, 256, 256) float tensor in [0, 1].
            observed_mask: optional (B, 1, 256, 256) float/bool mask (>0.5 observed).
            stage1_only: if True, skips the frozen generator & Module D for fast Stage-1 training.
            skip_decoder: V3.4 alias for skipping frozen generator during Stage-1v4 warmup.
        """
        b, c, h, w = x_256.shape
        if (c, h, w) != (3, 256, 256):
            raise ValueError(f"Expected x_256 shape (B, 3, 256, 256), got {tuple(x_256.shape)}")

        if observed_mask is None:
            obs_mask = torch.ones((b, 1, 256, 256), device=x_256.device, dtype=x_256.dtype)
        else:
            obs_mask = observed_mask.to(device=x_256.device, dtype=x_256.dtype)
            if obs_mask.ndim == 3:
                obs_mask = obs_mask.unsqueeze(1)

        # 1. Mask-aware border-replicate padding
        x_pad_256 = replicate_pad_observed(x_256, obs_mask)

        if self.arch_variant == "v34":
            return self._forward_v34(x_pad_256, obs_mask, skip_decoder=bool(skip_decoder or stage1_only))

        # 2. Module A: Degradation Encoder + PhotoCorrectNet
        deg_out = self.v3["deg_encoder"](x_pad_256, obs_mask)
        z_deg = deg_out["z_deg"]
        d_maps = deg_out["d_maps"]
        w_maps = deg_out["w_maps"]
        obs_mask_16 = deg_out["observed_mask_16"]

        x_photo_256, g_photo = self.v3["photo_net"](
            x_pad_256,
            d_maps["32"],
            z_deg,
            observed_mask=obs_mask,
            sufficiency=deg_out["sufficiency"],
        )

        # 3. Frozen Official CodeFormer Encoder on x_photo_512
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        x_photo_512 = F.interpolate(x_photo_256, size=(512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
        enc_feat_dict: Dict[str, Tensor] = {}
        x = x_photo_512
        for i, block in enumerate(self.base.encoder.blocks):
            x = _grad_ckpt(block, x, use_reentrant=False) if use_ckpt else block(x)
            if i in self._enc_out_indices:
                enc_feat_dict[str(x.shape[-1])] = x
        lq_feat_16 = x  # (B, 256, 16, 16)

        # 4. Module B: Deg-Gated Transformer LoRA (v30) or pure frozen CF Transformer (v33) + top-1 quantization + MaskedAdaIN
        pos_emb = self.base.position_emb.unsqueeze(1).repeat(1, b, 1)
        feat_emb = self.base.feat_emb(lq_feat_16.flatten(2).permute(2, 0, 1))
        if "transformer_lora" in self.v3:
            query_emb = self.v3["transformer_lora"](
                feat_emb,
                pos_emb,
                self.base.ft_layers,
                z_deg,
                d_maps["16"],
                deg_out["lora_gate"],
            )
            lora_gate_out = deg_out["lora_gate"]
        else:
            query_emb = feat_emb
            for layer in self.base.ft_layers:
                if use_ckpt:
                    query_emb = _grad_ckpt(lambda q, p, _l=layer: _l(q, query_pos=p), query_emb, pos_emb, use_reentrant=False)
                else:
                    query_emb = layer(query_emb, query_pos=pos_emb)
            lora_gate_out = deg_out["lora_gate"].detach() * 0.0
        code_logits = self.base.idx_pred_layer(query_emb).permute(1, 0, 2)  # (B, 256, 1024)

        # 5. Module C (part 1): V3PyramidSFT on encoder skips
        enc_used = self.v3["pyramid_sft"](enc_feat_dict, d_maps, w_maps=w_maps)

        if stage1_only:
            return {
                "y_256": None,
                "y_512": None,
                "x_photo_256": x_photo_256,
                "code_logits": code_logits,
                "lq_feat_16": lq_feat_16,
                "enc_used": enc_used,
                "alpha_bypass": None,
                "w_maps": w_maps,
                "deg_preds": deg_out["deg_preds"],
                "sufficiency": deg_out["sufficiency"],
                "lora_gate": lora_gate_out,
                "alpha_adain": deg_out["alpha_adain"],
                "g_photo": g_photo,
                "freq_gates": None,
            }

        soft_one_hot = F.softmax(code_logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.base.quantize.get_codebook_feat(top_idx, shape=[b, 16, 16, 256]).detach()
        quant_feat = self.masked_adain(quant_feat, lq_feat_16, obs_mask_16, deg_out["alpha_adain"])

        # 6. Single-pass Frozen CF Generator + Adaptive-w Fuse_sft_block + V3FreqHead
        x_dec = quant_feat
        coarse_32 = enc_used["32"]
        freq_gates: Dict[str, Tensor] = {}
        for i, block in enumerate(self.base.generator.blocks):
            x_dec = _grad_ckpt(block, x_dec, use_reentrant=False) if use_ckpt else block(x_dec)
            if i in self._gen_fuse_map:
                site = self._gen_fuse_map[i]
                dec_s = x_dec
                fused_s = self.base.fuse_convs_dict[site](enc_used[site], dec_s, w_maps[site])
                x_dec, gate_s = self.v3["freq_head"].forward_site(
                    site, enc_used[site], dec_s, fused_s, coarse_32, d_maps[site], return_gate=True
                )
                freq_gates[site] = gate_s

        y_512 = ((x_dec + 1.0) * 0.5).clamp(0.0, 1.0)

        # 7. Module D: V3OutputComposer (use x_pad_256 to prevent gray-bar step-edge bleed)
        y_256, alpha_bypass, r_out = self.v3["output_composer"](y_512, x_pad_256, d_maps["256"], obs_mask)

        return {
            "y_256": y_256,
            "y_512": y_512,
            "x_photo_256": x_photo_256,
            "code_logits": code_logits,
            "lq_feat_16": lq_feat_16,
            "enc_used": enc_used,
            "alpha_bypass": alpha_bypass,
            "r_out": r_out,
            "w_maps": w_maps,
            "deg_preds": deg_out["deg_preds"],
            "sufficiency": deg_out["sufficiency"],
            "lora_gate": lora_gate_out,
            "alpha_adain": deg_out["alpha_adain"],
            "g_photo": g_photo,
            "freq_gates": freq_gates,
        }

    @torch.no_grad()
    def restore_canvas(
        self,
        canvas_uint8: np.ndarray,
        observed_bool: Optional[np.ndarray] = None,
        *,
        return_delivered: bool = False,
        rounding: str = "floor",
    ):
        """Fast single-pass inference on a (256, 256, 3) uint8 RGB canvas.

        Preserves ~observed pixels bit-exact via delivered_views_v3.
        Returns raw_f32 (256, 256, 3) in [0, 1] by default, or (raw_f32, delivered_u8)
        when return_delivered=True.
        """
        if not (isinstance(canvas_uint8, np.ndarray) and canvas_uint8.shape == (256, 256, 3) and canvas_uint8.dtype == np.uint8):
            raise ValueError("Expected uint8 (256, 256, 3) RGB canvas.")
        if observed_bool is None:
            obs_np = np.ones((256, 256), dtype=bool)
        else:
            if not (isinstance(observed_bool, np.ndarray) and observed_bool.shape == (256, 256) and observed_bool.dtype == bool):
                raise ValueError("Expected bool (256, 256) observed_bool mask.")
            obs_np = observed_bool

        device = next(self.v3.parameters()).device
        inp_t = torch.from_numpy(canvas_uint8.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32)[None] / 255.0
        obs_t = torch.from_numpy(obs_np.astype(np.float32))[None, None].to(device=device)

        was_training = self.training
        self.eval()
        try:
            out = self.forward(inp_t, obs_t)
        finally:
            if was_training:
                self.train(True)

        raw_f32 = out["y_256"][0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32, copy=True)
        raw_out, pixels_u8 = delivered_views_v3(raw_f32, canvas_uint8, obs_np, rounding=rounding)
        if return_delivered:
            return raw_out, pixels_u8
        return raw_out


# ---------------------------------------------------------------------------
# Flat zero-pickle float32 .npz export / inspection / load
# ---------------------------------------------------------------------------

_V3_SHAPES_CACHE: Dict[str, Dict[str, Tuple[int, ...]]] = {}


def v3_shapes(
    model: Optional[CodeFormerV3Restorer] = None,
    *,
    arch_variant: str = "v30",
) -> Dict[str, Tuple[int, ...]]:
    """Return exact expected tensor name -> shape mapping for V3 trainable modules."""
    if model is not None:
        return {name: tuple(tensor.shape) for name, tensor in model.v3.state_dict().items()}
    if arch_variant not in ARCH_VARIANTS:
        raise ValueError(f"Unsupported arch_variant={arch_variant!r}; expected one of {ARCH_VARIANTS}")
    cached = _V3_SHAPES_CACHE.get(arch_variant)
    if cached is None:
        ref = CodeFormerV3Restorer(arch_variant=arch_variant)
        cached = {name: tuple(tensor.shape) for name, tensor in ref.v3.state_dict().items()}
        _V3_SHAPES_CACHE[arch_variant] = cached
    return dict(cached)


def detect_v3_npz_variant(raw_or_path: bytes | str | Path) -> str:
    """Auto-detect 'v30' vs 'v33' vs 'v34' from flat zero-pickle .npz tensor key set."""
    raw = raw_or_path if isinstance(raw_or_path, bytes) else Path(raw_or_path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
    if not all(n.endswith(".npy") for n in names):
        raise ValueError("Expected flat .npy entries in V3 NPZ archive.")
    keys = {n[:-4] for n in names}
    if any(k.startswith("pre_net.") for k in keys) and keys == set(v3_shapes(arch_variant="v34")):
        return "v34"
    for variant in ("v33", "v30"):
        if keys == set(v3_shapes(arch_variant=variant)):
            return variant
    raise ValueError(
        f"Unrecognized V3 NPZ key set ({len(keys)} tensors): matches none of 'v30' "
        f"({len(v3_shapes(arch_variant='v30'))}), 'v33' ({len(v3_shapes(arch_variant='v33'))}), "
        f"or 'v34' ({len(v3_shapes(arch_variant='v34'))})."
    )


def v3_state_sha(state: Mapping[str, np.ndarray | Tensor]) -> str:
    """Canonical SHA-256 over sorted float32 V3 tensor names, shapes, and C-order bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        arr = (
            value.detach().cpu().contiguous().to(dtype=torch.float32).numpy()
            if isinstance(value, torch.Tensor)
            else np.ascontiguousarray(value, dtype=np.float32)
        )
        header = json.dumps([name, "torch.float32", list(arr.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def export_v3_npz(model: CodeFormerV3Restorer, path: str | Path) -> str:
    """Export trainable V3 module weights as a flat, zero-pickle float32 .npz file.

    Returns the canonical v3_state_sha hex digest.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for name, tensor in model.v3.state_dict().items():
        arr = np.ascontiguousarray(tensor.detach().cpu().to(dtype=torch.float32).numpy())
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in V3 tensor {name}")
        arrays[name] = arr
    np.savez(out_path, **arrays)
    return v3_state_sha(arrays)


def inspect_v3_npz(
    raw: bytes,
    expected_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
    *,
    max_bytes: int = 32 * 1024**2,
) -> Dict[str, np.ndarray]:
    """Validate flat zero-pickle float32 V3 .npz archive headers before allocation.

    When `expected_shapes` is None, auto-detects 'v30' vs 'v33' from exact key set.
    """
    if not (type(raw) is bytes and 0 < len(raw) < max_bytes):
        raise ValueError("Bounded V3 NPZ bytes required.")
    if expected_shapes is None:
        variant = detect_v3_npz_variant(raw)
        expected_shapes = v3_shapes(arch_variant=variant)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        items = archive.infolist()
        names = archive.namelist()
        if not (
            len(items) == len(set(names)) == len(expected_shapes)
            and {n[:-4] for n in names} == set(expected_shapes)
            and all("/" not in n and "\\" not in n and ":" not in n and n.endswith(".npy") for n in names)
        ):
            raise ValueError(f"Expected exactly {len(expected_shapes)} unique flat V3 arrays.")
        if not (
            all(not (item.flag_bits & 1) and not stat.S_ISLNK(item.external_attr >> 16) for item in items)
            and sum(item.file_size for item in items) < max_bytes
        ):
            raise ValueError("Bounded ordinary expanded V3 NPZ required.")
        for member in items:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("Unsupported V3 NPY version.")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                key = member.filename[:-4]
                if not (shape == expected_shapes[key] and dtype == np.dtype("float32") and not dtype.hasobject and not fortran):
                    raise ValueError(f"Exact float32 V3 schema required for {key}.")
                if member.file_size != stream.tell() + math.prod(shape) * 4:
                    raise ValueError(f"Incorrect V3 array size for {key}.")

    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        result: Dict[str, np.ndarray] = {}
        for name, shape in expected_shapes.items():
            arr = saved[name]
            if not (arr.shape == shape and arr.dtype == np.float32 and np.isfinite(arr).all()):
                raise ValueError(f"Finite exact V3 values required for {name}.")
            result[name] = np.ascontiguousarray(arr).copy()
        return result


def load_v3_npz(
    path: str | Path,
    model: Optional[CodeFormerV3Restorer] = None,
    *,
    partial: bool = False,
    skip_prefixes: Tuple[str, ...] = (),
) -> Dict[str, np.ndarray]:
    """Load and validate a flat zero-pickle float32 V3 .npz checkpoint.

    Auto-detects 'v30' vs 'v33' when `model` is None or `partial=True`.
    If `model` is provided and `partial=False`, loads into `model.v3` with strict=True.
    If `partial=True`, loads only matching-shape tensors whose keys do not start with `skip_prefixes`.
    """
    raw = Path(path).read_bytes()
    if model is None or partial:
        src_variant = detect_v3_npz_variant(raw)
        expected = v3_shapes(arch_variant=src_variant)
    else:
        expected = v3_shapes(model)
    arrays = inspect_v3_npz(raw, expected)
    if model is not None:
        device = next(model.v3.parameters()).device
        if partial:
            cur_sd = model.v3.state_dict()
            load_sd: Dict[str, Tensor] = {}
            for k, v in arrays.items():
                if skip_prefixes and any(k.startswith(pref) for pref in skip_prefixes):
                    continue
                if k in cur_sd and tuple(cur_sd[k].shape) == tuple(v.shape):
                    load_sd[k] = torch.from_numpy(v.copy()).to(device=device, dtype=torch.float32)
            model.v3.load_state_dict(load_sd, strict=False)
        else:
            sd = {k: torch.from_numpy(v.copy()).to(device=device, dtype=torch.float32) for k, v in arrays.items()}
            model.v3.load_state_dict(sd, strict=True)
    return arrays


# Aliases for backward/spec naming compatibility
export_v3_modules_npz = export_v3_npz
load_v3_modules_npz = load_v3_npz

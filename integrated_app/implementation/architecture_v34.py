"""Self-contained inference-only Pre-Conditioned Frozen CodeFormer V3.4 architecture.

Components:
- Frozen official S-Lab CodeFormer backbone (`codeformer.pth`, 515 tensors, eval-only).
- `V34DegradationEncoder`: 7-ch MBConv encoder + scalar degradation/gate heads.
- `V34PhotoCorrectNet`: highlight-pinned black-level dehaze + Reinhard tone curve + gamma + WB.
- `V34NAFPreConditionNet`: 3-level NAFNet-Tiny UNet with SimpleGate, SCA, LayerNorm2d, and z-FiLM.
- `MaskedAdaIN`: FP32 AdaIN computed strictly over observed support at 16x16.
- `replicate_pad_observed_torch`: mask-aware nearest-border replicate fill for non-square crops.
- Strict flat zero-pickle float32 `.npz` header validation (`inspect_v34_npz`, `load_v34_modules_npz`).

Zero `research/` imports and zero training/loss dependencies.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import stat
from typing import Any, Dict, Mapping, Optional, Tuple
import zipfile

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from third_party.codeformer.codeformer_arch import CodeFormer

SITES: Tuple[str, ...] = ("32", "64", "128", "256")
Z_DIM: int = 128
BASE_TENSOR_COUNT: int = 515
V34_TENSOR_COUNT: int = 344
V34_PARAM_COUNT: int = 2241395
LUMA_WEIGHTS: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and (channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def replicate_pad_observed_torch(x_256: Tensor, observed_mask: Tensor) -> Tensor:
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


def _compute_v34_hf_channels(x_rgb: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute (luma_hf, chroma_hf_1px, chroma_hf_coarse) from RGB tensor in [0, 1] in FP32."""
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


class V34DegradationEncoder(nn.Module):
    """V3.4 Front-End Module 1: 7-ch degradation encoder (~0.29M params, NO FPN decoder)."""

    W_MIN: float = 0.60
    W_MAX: float = 0.98
    W_INIT: float = 0.82

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim

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

        self.z_proj = nn.Sequential(
            nn.Linear(224, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(inplace=False),
            nn.Linear(z_dim, z_dim),
        )

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

        m8 = F.interpolate(mask_f, size=f8.shape[-2:], mode="area")
        m8_bin = (m8 > 0.05).to(dtype=f8.dtype)
        has_obs_8 = m8_bin.sum(dim=(2, 3)) > 0.5
        m8_eff = torch.where(has_obs_8.view(-1, 1, 1, 1), m8_bin, torch.ones_like(m8_bin))
        denom8 = m8_eff.sum(dim=(2, 3)).clamp(min=1.0)
        avg_pool = (f8 * m8_eff).sum(dim=(2, 3)) / denom8
        max_pool = f8.masked_fill(m8_eff < 0.5, -1e4).amax(dim=(2, 3))
        z_deg = self.z_proj(torch.cat([avg_pool, max_pool], dim=1))

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
    """V3.4 Front-End Module 2: Highlight-pinned black-level dehaze + Reinhard tone + gamma + WB."""

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
        raw = self.param_head(self.conv(torch.cat([f4, z_sp], dim=1)))
        raw_wb = self.wb_head(z_deg)

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
            x1 = ((x_f - b_haze) / (1.0 - b_haze).clamp(min=0.55)).clamp(0.0, 1.0)
            r = (g_tone * x1) / (1.0 + (g_tone - 1.0) * x1).clamp(min=1e-6)
            eps = 1e-4
            eps_g = eps ** gamma
            one_g = (1.0 + eps) ** gamma
            x2 = ((r + eps).pow(gamma) - eps_g) / (one_g - eps_g).clamp(min=1e-6)
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
    """NAFNet building block with SimpleGate, Simplified Channel Attention (SCA), and LayerNorm2d."""

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
    """V3.4 Front-End Module 3: 3-level NAFNet-Tiny UNet at 256x256 (~1.95M params)."""

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


class MaskedAdaIN(nn.Module):
    """AdaIN computing lq_feat statistics strictly over observed_mask_16 in FP32."""

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


def delivered_views_v34(
    raw: np.ndarray,
    canvas: np.ndarray,
    observed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose exact prepared pixels outside observed support for both FP32 RAW and floor uint8 PNG."""
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
    pixels = np.floor(raw_out * np.float32(255.0)).astype(np.uint8)
    pixels[~observed] = canvas[~observed]
    return raw_out, pixels


class PreConditionedCodeFormerV34(nn.Module):
    """Frozen single-pass Pre-Conditioned CodeFormer V3.4 inference architecture."""

    def __init__(self, base: Optional[CodeFormer] = None):
        super().__init__()
        self.arch_variant = "v34"
        self.base = base if base is not None else CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=list(SITES),
            fix_modules=None,
        )
        self.v3 = nn.ModuleDict({
            "deg_encoder": V34DegradationEncoder(z_dim=Z_DIM),
            "photo_net": V34PhotoCorrectNet(f4_dim=112, z_dim=Z_DIM),
            "pre_net": V34NAFPreConditionNet(
                c=32, enc_blks=(1, 2, 2), mid_blks=2, dec_blks=(2, 2, 1), z_dim=Z_DIM
            ),
        })
        self.masked_adain = MaskedAdaIN()
        self._enc_out_indices = {self.base.fuse_encoder_block[s] for s in SITES}
        self._gen_fuse_map = {self.base.fuse_generator_block[s]: s for s in SITES}
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "PreConditionedCodeFormerV34":
        if mode is not False:
            raise ValueError("Training mode is unavailable in this frozen V3.4 runtime.")
        return super().train(False)

    def forward(
        self,
        x_256: Tensor,
        observed_mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        b, c, h, w = x_256.shape
        if (c, h, w) != (3, 256, 256):
            raise ValueError(f"Expected x_256 shape (B, 3, 256, 256), got {tuple(x_256.shape)}")

        if observed_mask is None:
            obs_mask = torch.ones((b, 1, 256, 256), device=x_256.device, dtype=x_256.dtype)
        else:
            obs_mask = observed_mask.to(device=x_256.device, dtype=x_256.dtype)
            if obs_mask.ndim == 3:
                obs_mask = obs_mask.unsqueeze(1)

        x_pad_256 = replicate_pad_observed_torch(x_256, obs_mask)

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
            x = block(x)
            if i in self._enc_out_indices:
                enc_feat_dict[str(x.shape[-1])] = x
        lq_feat_16 = x

        pos_emb = self.base.position_emb.unsqueeze(1).repeat(1, b, 1)
        query_emb = self.base.feat_emb(lq_feat_16.flatten(2).permute(2, 0, 1))
        for layer in self.base.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        code_logits = self.base.idx_pred_layer(query_emb).permute(1, 0, 2)

        # 5. Hard top-1 codebook quantization
        soft_one_hot = F.softmax(code_logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.base.quantize.get_codebook_feat(top_idx, shape=[b, 16, 16, 256]).detach()

        # 6. MaskedAdaIN with fixed alpha_adain = 1.0
        alpha_adain_one = torch.ones((b, 1, 1, 1), device=x_pad_256.device, dtype=quant_feat.dtype)
        quant_feat = self.masked_adain(quant_feat, lq_feat_16, obs_mask_16, alpha_adain_one)

        # 7. Untouched Frozen CF Generator + Fuse_sft_block with scalar w_pred
        x_dec = quant_feat
        w_scalar = deg_out["w_pred"].view(b, 1, 1, 1).to(dtype=x_dec.dtype)
        for i, block in enumerate(self.base.generator.blocks):
            x_dec = block(x_dec)
            if i in self._gen_fuse_map:
                site = self._gen_fuse_map[i]
                x_dec = self.base.fuse_convs_dict[site](enc_feat_dict[site], x_dec, w_scalar)

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
        }


_V34_SHAPES_CACHE: Optional[Dict[str, Tuple[int, ...]]] = None


def v34_shapes(model: Optional[PreConditionedCodeFormerV34] = None) -> Dict[str, Tuple[int, ...]]:
    """Return exact expected tensor name -> shape mapping for V3.4 front-end modules (344 tensors)."""
    global _V34_SHAPES_CACHE
    if model is not None:
        return {name: tuple(tensor.shape) for name, tensor in model.v3.state_dict().items()}
    if _V34_SHAPES_CACHE is None:
        v3_mods = nn.ModuleDict({
            "deg_encoder": V34DegradationEncoder(z_dim=Z_DIM),
            "photo_net": V34PhotoCorrectNet(f4_dim=112, z_dim=Z_DIM),
            "pre_net": V34NAFPreConditionNet(
                c=32, enc_blks=(1, 2, 2), mid_blks=2, dec_blks=(2, 2, 1), z_dim=Z_DIM
            ),
        })
        _V34_SHAPES_CACHE = {name: tuple(tensor.shape) for name, tensor in v3_mods.state_dict().items()}
    return dict(_V34_SHAPES_CACHE)


def v34_state_sha(state: Mapping[str, np.ndarray | Tensor]) -> str:
    """Canonical SHA-256 over sorted float32 V3.4 tensor names, shapes, and C-order bytes."""
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


def inspect_v34_npz(
    raw: bytes,
    expected_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
    *,
    max_bytes: int = 16 * 1024**2,
) -> Dict[str, np.ndarray]:
    """Validate flat zero-pickle float32 V3.4 .npz archive headers before allocation."""
    if not (type(raw) is bytes and 0 < len(raw) < max_bytes):
        raise ValueError("Bounded V3.4 NPZ bytes required.")
    shapes = expected_shapes if expected_shapes is not None else v34_shapes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        items = archive.infolist()
        names = archive.namelist()
        if not (
            len(items) == len(set(names)) == len(shapes)
            and {n[:-4] for n in names} == set(shapes)
            and all("/" not in n and "\\" not in n and ":" not in n and n.endswith(".npy") for n in names)
        ):
            raise ValueError(f"Expected exactly {len(shapes)} unique flat V3.4 arrays.")
        if not (
            all(not (item.flag_bits & 1) and not stat.S_ISLNK(item.external_attr >> 16) for item in items)
            and sum(item.file_size for item in items) < max_bytes
        ):
            raise ValueError("Bounded ordinary expanded V3.4 NPZ required.")
        for member in items:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("Unsupported V3.4 NPY version.")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                key = member.filename[:-4]
                if not (shape == shapes[key] and dtype == np.dtype("float32") and not dtype.hasobject and not fortran):
                    raise ValueError(f"Exact float32 V3.4 schema required for {key}.")
                if member.file_size != stream.tell() + math.prod(shape) * 4:
                    raise ValueError(f"Incorrect V3.4 array size for {key}.")

    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        result: Dict[str, np.ndarray] = {}
        for name, shape in shapes.items():
            arr = saved[name]
            if not (arr.shape == shape and arr.dtype == np.float32 and np.isfinite(arr).all()):
                raise ValueError(f"Finite exact V3.4 values required for {name}.")
            result[name] = np.ascontiguousarray(arr).copy()
        return result


def load_v34_modules_npz(
    path: str | Path,
    model: Optional[PreConditionedCodeFormerV34] = None,
) -> Dict[str, np.ndarray]:
    """Load and strictly validate flat zero-pickle float32 V3.4 module weights."""
    raw = Path(path).read_bytes()
    expected = v34_shapes(model)
    arrays = inspect_v34_npz(raw, expected)
    if model is not None:
        sd = {k: torch.from_numpy(v.copy()).to(device="cpu", dtype=torch.float32) for k, v in arrays.items()}
        model.v3.load_state_dict(sd, strict=True)
    return arrays

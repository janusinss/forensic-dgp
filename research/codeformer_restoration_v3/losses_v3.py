"""CodeFormer Restoration V3 — Loss Functions & Spectral-Norm PatchGAN Discriminator.

All loss computations execute in FP32 (disabling outer bf16/fp16 autocast) and mask
pixel/feature objectives strictly to `observed_mask` so neutral-128 padding bars never
contaminate gradients.

Implements:
- Stage1Loss: L_photo (low-freq 16x16 L1 + CIE-Lab ab + clear identity lock),
  L_deg_aux (exposure, blur, noise, codec/lr), L_code (GT codebook CE ls=0.05 +
  16x16 cosine), L_clean_sft (multi-scale L1 + 0.25*(1-cos) + 0.30*GradL1).
- Stage2Loss: Charbonnier L1 (1.00), 1-MS_SSIM7 (0.35), YCbCr Chroma L1 (0.15),
  FrozenLPIPSVGG16 (0.25), differentiable ArcFaceIdentityLoss (0.35 / 0.45 in S3),
  MultiScale_Laplacian_L1 (0.25), Log_FFT_Amp_L1 (0.10), Clear-Anchor, aux 512x512
  loss (0.20), and Stage-1 retention terms (0.15*L_code + 0.10*L_clean_sft + 0.05*L_photo).
- PatchGANDiscriminator (Spectral-Norm) + lazy R1 penalty (gamma=10 every 16 steps)
  + Stage3Loss (degraded-only GAN at lambda_gan=0.01, clear samples masked out).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

# Default FFHQ 5-point canonical landmark coordinates normalized to [0, 1] for
# aligned 256x256 / 512x512 FFHQ crops when per-sample landmarks are not supplied.
FFHQ_CANONICAL_5PT_NORM = torch.tensor(
    [
        [0.3725, 0.4685],  # left eye
        [0.6275, 0.4685],  # right eye
        [0.5000, 0.6215],  # nose tip
        [0.3980, 0.7420],  # left mouth
        [0.6020, 0.7420],  # right mouth
    ],
    dtype=torch.float32,
)

ARCFACE_TEMPLATE_112 = torch.tensor(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=torch.float32,
)


def _ensure_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    """Return a float32 (B, 1, H, W) mask resized to match `ref` spatial dims."""
    b, _, h, w = ref.shape
    if mask is None:
        return torch.ones((b, 1, h, w), device=ref.device, dtype=torch.float32)
    m = mask.float()
    if m.ndim == 3:
        m = m.unsqueeze(1)
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m, size=(h, w), mode="nearest")
    return m.clamp(0.0, 1.0)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute mean of `tensor` (B, C, H, W) strictly inside `mask` (B, 1, H, W)."""
    m = _ensure_mask(mask, tensor)
    if m.shape[1] == 1 and tensor.shape[1] != 1:
        m = m.expand_as(tensor)
    denom = m.sum().clamp_min(eps)
    return (tensor * m).sum() / denom


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return masked_mean((pred - target).abs(), _ensure_mask(mask, pred))


def masked_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    diff = pred - target
    val = torch.sqrt(diff * diff + eps * eps)
    return masked_mean(val, _ensure_mask(mask, pred))


def masked_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Channel-wise (1 - cosine_similarity) averaged strictly over observed spatial sites."""
    m = _ensure_mask(mask, pred)
    p_norm = pred / pred.norm(dim=1, keepdim=True).clamp_min(eps)
    t_norm = target / target.norm(dim=1, keepdim=True).clamp_min(eps)
    cos = (p_norm * t_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    return masked_mean(1.0 - cos, m)


def masked_grad_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """First-order spatial gradient L1 loss masked to pairs of adjacent observed pixels."""
    m = _ensure_mask(mask, pred)
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    mx = m[:, :, :, 1:] * m[:, :, :, :-1]

    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    my = m[:, :, 1:, :] * m[:, :, :-1, :]

    lx = masked_mean((dx_p - dx_t).abs(), mx)
    ly = masked_mean((dy_p - dy_t).abs(), my)
    return 0.5 * (lx + ly)


def rgb_to_lab_ab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB [0, 1] -> CIE-Lab (a*, b*) channels normalized by 100.

    Uses smooth clamping so backward gradients are finite everywhere including black pixels.
    """
    x = rgb.clamp(0.0, 1.0)
    # Approximate sRGB gamma linearization smoothly (avoiding kink at 0.04045)
    lin = torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow(((x + 0.055) / 1.055).clamp_min(1e-6), 2.4),
    )
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    # D65 RGB -> XYZ normalized by reference white (0.95047, 1.0, 1.08883)
    X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    Y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    Z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    delta = 6.0 / 29.0
    delta3 = delta ** 3
    factor = 1.0 / (3.0 * delta * delta)
    offset = 4.0 / 29.0

    def _f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(
            t > delta3,
            torch.pow(t.clamp_min(delta3), 1.0 / 3.0),
            factor * t + offset,
        )

    fx, fy, fz = _f(X), _f(Y), _f(Z)
    a_star = 5.0 * (fx - fy)  # scaled by 1/100 (500/100 = 5.0)
    b_star = 2.0 * (fy - fz)  # scaled by 1/100 (200/100 = 2.0)
    return torch.cat([a_star, b_star], dim=1)


def rgb_to_ycbcr_chroma(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable BT.601 RGB [0, 1] -> (Cb, Cr) centered around 0."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([cb, cr], dim=1)


def ssim7_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact 7x7 uniform box-filter SSIM map matching adviser_three_model_comparison.py."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_p = F.avg_pool2d(pred, kernel_size=7, stride=1, padding=0)
    mu_t = F.avg_pool2d(target, kernel_size=7, stride=1, padding=0)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    var_p = (F.avg_pool2d(pred * pred, kernel_size=7, stride=1, padding=0) - mu_p2).clamp_min(0.0)
    var_t = (F.avg_pool2d(target * target, kernel_size=7, stride=1, padding=0) - mu_t2).clamp_min(0.0)
    cov = F.avg_pool2d(pred * target, kernel_size=7, stride=1, padding=0) - mu_pt

    num = (2.0 * mu_pt + c1) * (2.0 * cov + c2)
    den = (mu_p2 + mu_t2 + c1) * (var_p + var_t + c2)
    return (num / den.clamp_min(1e-12)).clamp(-1.0, 1.0)


def masked_ms_ssim7_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scales: int = 3,
) -> torch.Tensor:
    """Multi-scale (1 - SSIM7) loss masked to observed pixels."""
    m = _ensure_mask(mask, pred)
    weights = [0.5, 0.3, 0.2][:scales]
    total_w = sum(weights)
    p, t, cm = pred, target, m
    loss = pred.new_tensor(0.0)
    for w in weights:
        if min(p.shape[-2:]) < 11:
            break
        s_map = ssim7_map(p, t)
        # Valid 7x7 interior mask
        m_valid = cm[:, :, 3:-3, 3:-3]
        loss = loss + (w / total_w) * masked_mean(1.0 - s_map, m_valid)
        p = F.avg_pool2d(p, kernel_size=2, stride=2)
        t = F.avg_pool2d(t, kernel_size=2, stride=2)
        cm = F.avg_pool2d(cm, kernel_size=2, stride=2)
    return loss


class MultiScaleLaplacianL1(nn.Module):
    """Multi-scale 3x3 Laplacian high-frequency L1 loss (masked to observed_mask)."""

    def __init__(self, scales: int = 3):
        super().__init__()
        self.scales = scales
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))

    def _lap(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.kernel.to(x.device), groups=3)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = _ensure_mask(mask, pred)
        p, t, cm = pred.float(), target.float(), m
        weights = [0.5, 0.3, 0.2][: self.scales]
        total_w = sum(weights)
        loss = p.new_tensor(0.0)
        for w in weights:
            if min(p.shape[-2:]) < 16:
                break
            lp = self._lap(p)
            lt = self._lap(t)
            # Erode mask by 1px so boundary transitions don't create artificial edges
            m_inner = F.avg_pool2d(cm, kernel_size=3, stride=1, padding=1)
            m_inner = (m_inner > 0.99).float()
            if m_inner.sum() < 1.0:
                m_inner = cm
            loss = loss + (w / total_w) * masked_mean((lp - lt).abs(), m_inner)
            p = F.avg_pool2d(p, kernel_size=2, stride=2)
            t = F.avg_pool2d(t, kernel_size=2, stride=2)
            cm = F.avg_pool2d(cm, kernel_size=2, stride=2)
        return loss


def log_fft_amp_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP32 Log-FFT amplitude L1 loss; fills ~observed with target to prevent pad ringing."""
    m = _ensure_mask(mask, pred)
    p = pred.float() * m + target.float() * (1.0 - m)
    t = target.float()
    fp = torch.fft.rfft2(p, norm="ortho")
    ft = torch.fft.rfft2(t, norm="ortho")
    amp_p = torch.log1p(torch.sqrt(fp.real.square() + fp.imag.square() + 1e-8))
    amp_t = torch.log1p(torch.sqrt(ft.real.square() + ft.imag.square() + 1e-8))
    return (amp_p - amp_t).abs().mean()


# ---------------------------------------------------------------------------
# V3.1 Perceptual / Sharpness / Sufficiency Loss Primitives (FP32, NaN-Safe)
# ---------------------------------------------------------------------------

_LPIPS_TAPS: Tuple[int, ...] = (3, 8, 15, 22, 29)
_LPIPS_TAP_TO_IDX: Dict[int, int] = {tap: i for i, tap in enumerate(_LPIPS_TAPS)}
_GRAM_TAPS: Tuple[int, ...] = (3, 8, 15, 22)
_GRAM_WEIGHTS: Dict[int, float] = {3: 0.25, 8: 0.30, 15: 0.30, 22: 0.15}


def compute_lpips_and_gram(
    lpips_model: nn.Module,
    pred_m11: torch.Tensor,
    target_m11: torch.Tensor,
    *,
    compute_gram: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-pass LPIPS + RMS-normalized 4-layer VGG16 Gram style loss in FP32.

    Reuses `FrozenLPIPSVGG16.features` in one forward pass on `pred_m11` (and
    `target_m11` under `torch.no_grad()`), with `sqrt(sum(feat^2) + 1e-12)` for
    NaN-free backward on pure-black pixels and detached target RMS normalization
    for O(1) Gram matrices across `relu1_2, relu2_2, relu3_3, relu4_3`.
    """
    with torch.amp.autocast(device_type=pred_m11.device.type, enabled=False):
        p_in = pred_m11.float()
        t_in = target_m11.float()
        if not (
            hasattr(lpips_model, "features")
            and hasattr(lpips_model, "shift")
            and hasattr(lpips_model, "scale")
            and hasattr(lpips_model, "calibration")
        ):
            l_lpips = lpips_model(p_in, t_in).float()
            return l_lpips, p_in.new_tensor(0.0)

        shift = lpips_model.shift.to(device=p_in.device, dtype=torch.float32)
        scale = lpips_model.scale.to(device=p_in.device, dtype=torch.float32)

        p = (p_in - shift) / scale
        with torch.no_grad():
            t = (t_in - shift) / scale

        lpips_dists = []
        l_gram = p_in.new_tensor(0.0)
        for index, layer in enumerate(lpips_model.features):
            p = layer(p)
            with torch.no_grad():
                t = layer(t)

            if index in _LPIPS_TAP_TO_IDX:
                tap_idx = _LPIPS_TAP_TO_IDX[index]
                norm_p = torch.sqrt(p.pow(2).sum(dim=1, keepdim=True) + 1e-12)
                norm_t = torch.sqrt(t.pow(2).sum(dim=1, keepdim=True) + 1e-12)
                pn = p / (norm_p + 1e-10)
                tn = t / (norm_t + 1e-10)
                w_cal = lpips_model.calibration[tap_idx].to(device=p.device, dtype=torch.float32)
                lpips_dists.append(F.conv2d((pn - tn) ** 2, w_cal).mean(dim=(2, 3), keepdim=True))

                if compute_gram and index in _GRAM_WEIGHTS:
                    rms_t = t.pow(2).mean(dim=(1, 2, 3), keepdim=True).add(1e-6).sqrt().detach()
                    fp = (p / rms_t).flatten(2)  # (B, C, HW)
                    ft = (t / rms_t).flatten(2)
                    hw = float(fp.shape[2])
                    gp = torch.bmm(fp, fp.transpose(1, 2)) / hw
                    gt_g = torch.bmm(ft, ft.transpose(1, 2)) / hw
                    l_gram = l_gram + _GRAM_WEIGHTS[index] * F.smooth_l1_loss(gp, gt_g, beta=0.1)

        l_lpips = sum(lpips_dists).mean()
        return l_lpips, l_gram


class MultiScaleHFEnergyLoss(nn.Module):
    """2-Scale phase-invariant masked log-HF-energy loss (zero std/sqrt(0) NaN trap).

    Converts pred/target to BT.709 luma, computes reflect-padded 3x3 Laplacian
    energy `E = sqrt(L^2 + 1e-8) - 1e-4`, and aligns masked local patch averages
    (`k=16*s, stride=8*s` @ 0.45; `k=32*s, stride=16*s` @ 0.35) plus global
    observed mean (@ 0.20) in log space at scales 1x (0.65) and 0.5x (0.35).
    """

    def __init__(self):
        super().__init__()
        luma_w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32).view(1, 3, 1, 1)
        lap_k = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("luma_w", luma_w)
        self.register_buffer("lap_k", lap_k)

    def _luma(self, x: torch.Tensor) -> torch.Tensor:
        w = self.luma_w.to(device=x.device, dtype=torch.float32)
        return (x.float() * w).sum(dim=1, keepdim=True)

    def _lap_energy(self, y: torch.Tensor) -> torch.Tensor:
        k = self.lap_k.to(device=y.device, dtype=torch.float32)
        lap = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="reflect"), k)
        return torch.sqrt(lap * lap + 1e-8) - 1e-4

    def _scale_loss(self, yp: torch.Tensor, yt: torch.Tensor, cm: torch.Tensor) -> torch.Tensor:
        ep = self._lap_energy(yp)
        et = self._lap_energy(yt)
        m_inner = (F.avg_pool2d(cm, kernel_size=3, stride=1, padding=1) > 0.99).float()
        if m_inner.sum() < 1.0:
            m_inner = cm

        h, w = yp.shape[-2:]
        s = max(1, h // 256)
        terms = []
        weights = []

        for k_base, st_base, wt in ((16, 8, 0.45), (32, 16, 0.35)):
            k = k_base * s
            st = st_base * s
            if min(h, w) >= k:
                m_pool = F.avg_pool2d(m_inner, kernel_size=k, stride=st)
                valid = m_pool > 0.25
                if valid.any():
                    ep_p = F.avg_pool2d(ep * m_inner, kernel_size=k, stride=st) / m_pool.clamp_min(0.25)
                    et_p = F.avg_pool2d(et * m_inner, kernel_size=k, stride=st) / m_pool.clamp_min(0.25)
                    lp = F.smooth_l1_loss(
                        torch.log(ep_p[valid] + 1e-3),
                        torch.log(et_p[valid] + 1e-3),
                        beta=0.05,
                    )
                    terms.append(lp)
                    weights.append(wt)

        # Global observed-mean energy (weight 0.20)
        m_sum = m_inner.sum(dim=(1, 2, 3))
        valid_g = m_sum > 1.0
        if valid_g.any():
            ep_g = (ep * m_inner).sum(dim=(1, 2, 3))[valid_g] / m_sum[valid_g].clamp_min(1.0)
            et_g = (et * m_inner).sum(dim=(1, 2, 3))[valid_g] / m_sum[valid_g].clamp_min(1.0)
            lg = F.smooth_l1_loss(torch.log(ep_g + 1e-3), torch.log(et_g + 1e-3), beta=0.05)
            terms.append(lg)
            weights.append(0.20)

        if not terms:
            return yp.sum() * 0.0
        w_sum = sum(weights)
        return sum((wt / w_sum) * term for wt, term in zip(weights, terms))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            m = _ensure_mask(mask, pred)
            yp = self._luma(pred)
            yt = self._luma(target)
            l_1x = self._scale_loss(yp, yt, m)
            if min(yp.shape[-2:]) >= 32:
                yp_half = F.avg_pool2d(yp, kernel_size=2, stride=2)
                yt_half = F.avg_pool2d(yt, kernel_size=2, stride=2)
                m_half = F.avg_pool2d(m, kernel_size=2, stride=2)
                l_half = self._scale_loss(yp_half, yt_half, m_half)
                return 0.65 * l_1x + 0.35 * l_half
            return l_1x


_WMAP_CAP_SLOPES: Dict[str, float] = {"32": 0.35, "64": 0.55, "128": 0.75, "256": 0.85}
_WMAP_FLOORS: Dict[str, float] = {"32": 0.45, "64": 0.25, "128": 0.08, "256": 0.03}
_WMAP_FLOORS_V33: Dict[str, float] = {"32": 0.50, "64": 0.35, "128": 0.15, "256": 0.08}
_WMAP_BLUR_CAPS_V33: Dict[str, float] = {"32": 0.65, "64": 0.45, "128": 0.28, "256": 0.18}


def _extract_deg_components(
    deg_targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (exposure, blur, codec_lr) as FP32 (B,) tensors."""
    if isinstance(deg_targets, dict):
        exp = deg_targets["exposure"].float().reshape(-1).to(device)
        blur = deg_targets["blur"].float().reshape(-1).to(device)
        codec = deg_targets.get("codec_lr", deg_targets.get("codec")).float().reshape(-1).to(device)
        return exp, blur, codec
    dt = deg_targets.float().to(device)
    if dt.ndim == 1:
        dt = dt.unsqueeze(0)
    return dt[:, 0], dt[:, 1], dt[:, 3]


def _extract_deg_components_4(
    deg_targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (exposure, blur, noise, codec_lr) as FP32 (B,) tensors."""
    if isinstance(deg_targets, dict):
        exp = deg_targets["exposure"].float().reshape(-1).to(device)
        blur = deg_targets["blur"].float().reshape(-1).to(device)
        noise = deg_targets.get("noise", torch.zeros_like(exp)).float().reshape(-1).to(device)
        codec = deg_targets.get("codec_lr", deg_targets.get("codec")).float().reshape(-1).to(device)
        return exp, blur, noise, codec
    dt = deg_targets.float().to(device)
    if dt.ndim == 1:
        dt = dt.unsqueeze(0)
    return dt[:, 0], dt[:, 1], dt[:, 2], dt[:, 3]


def compute_wmap_loss(
    w_maps: Optional[Dict[str, torch.Tensor]],
    deg_targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    observed_mask: Optional[torch.Tensor] = None,
    is_clear: Optional[torch.Tensor] = None,
    sufficiency: Optional[torch.Tensor] = None,
    ref_tensor: Optional[torch.Tensor] = None,
    blur_mask: Optional[torch.Tensor] = None,
    v33_mode: bool = False,
) -> torch.Tensor:
    """Per-site `w_maps` sufficiency ceiling/floor hinge + `sufficiency` aux loss.

    When `v33_mode=True`, applies per-pixel spatial supervision using `blur_mask`
    (`m_blur(x, y)`): blurred regions push `w_maps['64'/'128'/'256']` down toward
    low ceilings (`<= 0.45 / 0.28 / 0.18`, cleanly below the `0.55` `gate_sft` cutoff),
    while sharp regions encourage `w_maps >= 0.75` and clear samples `>= 0.88`.
    """
    if not w_maps or deg_targets is None:
        if ref_tensor is not None:
            return ref_tensor.new_tensor(0.0)
        return torch.tensor(0.0)

    first_w = next(iter(w_maps.values()))
    device = first_w.device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        _, blur, noise, codec_lr = _extract_deg_components_4(deg_targets, device)
        d_sp = (torch.maximum(blur, codec_lr) + 0.35 * torch.minimum(blur, codec_lr)).clamp(0.0, 1.0)
        clear_f = (
            is_clear.float().reshape(-1).to(device)
            if is_clear is not None
            else torch.zeros_like(d_sp)
        )

        site_losses = []
        if v33_mode:
            # Per-pixel spatial w_maps supervision using m_blur(x, y)
            b_size = d_sp.shape[0]
            if blur_mask is not None:
                bm_full = blur_mask.float().to(device)
                if bm_full.ndim == 3:
                    bm_full = bm_full.unsqueeze(1)
                bm_full = bm_full.clamp(0.0, 1.0)
            else:
                bm_full = d_sp.view(b_size, 1, 1, 1)

            # Sharp region eligibility: not heavily noisy or codec-compressed
            sharp_sample_ok = ((codec_lr < 0.20) & (noise < 0.20)).float().view(b_size, 1, 1, 1)
            clear_4d = clear_f.view(b_size, 1, 1, 1)

            for site, w_cap_blur in _WMAP_BLUR_CAPS_V33.items():
                if site not in w_maps:
                    continue
                wm = w_maps[site].float()
                ms = _ensure_mask(observed_mask, wm)
                if bm_full.shape[-2:] != wm.shape[-2:]:
                    mb = F.interpolate(bm_full, size=wm.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    mb = bm_full
                if mb.shape[-2:] != wm.shape[-2:]:
                    mb = mb.expand_as(wm)

                # Per-pixel ceiling: 0.95 when sharp (mb=0), smoothly down to w_cap_blur when blurred (mb=1)
                w_cap_px = 0.95 - (0.95 - w_cap_blur) * mb
                w_floor = _WMAP_FLOORS_V33[site]

                # Per-pixel penalties inside observed mask
                cap_pen = F.relu(wm - w_cap_px)
                floor_pen = F.relu(w_floor - wm)
                # Sharp non-degraded pixels encourage high w_maps >= 0.75 (0.88 on clear)
                sharp_px = (mb < 0.08).float() * sharp_sample_ok * (1.0 - clear_4d)
                sharp_pen = sharp_px * F.relu(0.75 - wm)
                clear_pen = clear_4d * F.relu(0.88 - wm)

                site_losses.append(masked_mean(cap_pen + floor_pen + 0.50 * sharp_pen + clear_pen, ms))
        else:
            for site, slope in _WMAP_CAP_SLOPES.items():
                if site not in w_maps:
                    continue
                wm = w_maps[site].float()
                ms = _ensure_mask(observed_mask, wm)
                denom = ms.sum(dim=(1, 2, 3)).clamp_min(1e-6)
                w_mean = (wm * ms).sum(dim=(1, 2, 3)) / denom
                w_cap = 0.95 - slope * d_sp
                w_floor = _WMAP_FLOORS[site]
                hinge = (
                    F.relu(w_mean - w_cap)
                    + F.relu(w_floor - w_mean)
                    + clear_f * F.relu(0.88 - w_mean)
                ).mean()
                site_losses.append(hinge)

        loss = sum(site_losses) / float(len(site_losses)) if site_losses else first_w.new_tensor(0.0)
        if sufficiency is not None:
            suf_pred = sufficiency.float().reshape(-1)
            suf_tgt = (1.0 - d_sp).clamp(0.05, 1.0).to(device=suf_pred.device, dtype=torch.float32)
            loss = loss + 0.25 * F.smooth_l1_loss(suf_pred, suf_tgt)
        return loss


class DifferentiableArcFaceLoss(nn.Module):
    """Differentiable FP32 ArcFace cosine identity loss using w600k_r50.onnx via onnx2torch.

    Guarantees:
    - Frozen encoder weights, eval() mode preserved even when parent calls .train().
    - Clean FP32 execution under outer bf16/fp16 autocast.
    - Non-zero gradients flowing into `pred` (both with explicit landmarks or canonical FFHQ alignment).
    """

    def __init__(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        encoder: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        if encoder is None:
            import onnx
            from onnx2torch import convert

            path = Path(onnx_path or "~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"ArcFace ONNX weights missing: {path}")
            graph = onnx.load(str(path))
            shape = graph.graph.input[0].type.tensor_type.shape.dim
            if [d.dim_value for d in shape[1:]] != [3, 112, 112]:
                raise ValueError("Expected RGB ArcFace input with shape N x 3 x 112 x 112")
            encoder = convert(graph)
        self.encoder = encoder.to(device).float().eval().requires_grad_(False)
        self.register_buffer("template", ARCFACE_TEMPLATE_112.clone().to(device))
        self.register_buffer("canonical_5pt", FFHQ_CANONICAL_5PT_NORM.clone().to(device))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _to_5pt(self, landmarks: Optional[torch.Tensor], b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        if landmarks is None:
            pts = self.canonical_5pt.to(device).unsqueeze(0).expand(b, -1, -1).clone()
            pts[:, :, 0] *= float(w)
            pts[:, :, 1] *= float(h)
            return pts
        lm = landmarks.detach().float().to(device)
        if lm.shape[1] == 68:
            return torch.stack(
                [
                    lm[:, 36:42].mean(1),
                    lm[:, 42:48].mean(1),
                    lm[:, 30],
                    lm[:, 48],
                    lm[:, 54],
                ],
                dim=1,
            )
        if lm.shape[1] == 5:
            return lm
        raise ValueError(f"Expected 5 or 68 landmarks, got shape {tuple(lm.shape)}")

    def alignment_grid(
        self,
        landmarks: Optional[torch.Tensor],
        b: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        points = self._to_5pt(landmarks, b, height, width, device)
        valid = (
            torch.isfinite(points).all(dim=(1, 2))
            & ((points[:, 0] - points[:, 1]).norm(dim=1) >= 4.0)
            & (points[:, :, 0].amin(1) >= -width * 0.25)
            & (points[:, :, 0].amax(1) <= width * 1.25)
            & (points[:, :, 1].amin(1) >= -height * 0.25)
            & (points[:, :, 1].amax(1) <= height * 1.25)
        )
        p = points[valid]
        if len(p) == 0:
            return points.new_empty((0, 112, 112, 2)), valid
        x, y = p.unbind(-1)
        ones, zeros = torch.ones_like(x), torch.zeros_like(x)
        a = torch.stack(
            [
                torch.stack([x, -y, ones, zeros], dim=-1),
                torch.stack([y, x, zeros, ones], dim=-1),
            ],
            dim=2,
        ).reshape(-1, 10, 4)
        target = self.template.to(p).reshape(1, 10, 1).expand(len(p), -1, -1)
        fit = torch.linalg.lstsq(a, target).solution.squeeze(-1)
        scale, rot, tx, ty = fit.unbind(-1)
        denom = (scale.square() + rot.square()).clamp_min(1e-8)
        yy, xx = torch.meshgrid(
            torch.arange(112, device=p.device, dtype=p.dtype),
            torch.arange(112, device=p.device, dtype=p.dtype),
            indexing="ij",
        )
        u, v = xx[None] - tx[:, None, None], yy[None] - ty[:, None, None]
        src_x = (scale[:, None, None] * u + rot[:, None, None] * v) / denom[:, None, None]
        src_y = (-rot[:, None, None] * u + scale[:, None, None] * v) / denom[:, None, None]
        grid = torch.stack([2.0 * (src_x + 0.5) / width - 1.0, 2.0 * (src_y + 0.5) / height - 1.0], dim=-1)
        return grid, valid

    def embed(
        self,
        images: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return L2-normalized ArcFace embeddings (N_valid, 512) and boolean `valid` mask."""
        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            img_f = images.float()
            b, _, h, w = img_f.shape
            grid, valid = self.alignment_grid(landmarks, b, h, w, img_f.device)
            if not valid.any():
                return img_f.new_empty((0, 512)), valid
            crop = F.grid_sample(
                img_f[valid],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            emb = self.encoder(crop * 2.0 - 1.0).float()
            return F.normalize(emb, dim=1, eps=1e-8), valid

    def cosine_similarity(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            m = _ensure_mask(mask, pred)
            p_clean = pred.float() * m + target.float() * (1.0 - m)
            t_clean = target.float()
            emb_p, valid = self.embed(p_clean, landmarks)
            if not valid.any():
                return pred.new_empty((0,)), valid
            with torch.no_grad():
                emb_t, _ = self.embed(t_clean, landmarks)
            cos = (emb_p * emb_t).sum(dim=1).clamp(-1.0, 1.0)
            return cos, valid

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            cos, valid = self.cosine_similarity(pred, target, landmarks=landmarks, mask=mask)
            if not valid.any():
                return pred.float().sum() * 0.0
            return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# Stage 1 Loss
# ---------------------------------------------------------------------------


class Stage1Loss(nn.Module):
    """Stage-1 Photometric, Degradation, Codebook & Clean-SFT Alignment Loss.

    Operates purely on Module A/B/C outputs vs frozen GT CodeFormer teacher outputs.
    All math executes in FP32 and is masked to `observed_mask`.
    """

    SFT_WEIGHTS: Dict[str, float] = {"32": 0.15, "64": 0.25, "128": 0.30, "256": 0.30}

    def __init__(
        self,
        w_photo: float = 1.0,
        w_deg_aux: float = 0.5,
        w_code: float = 1.0,
        w_clean_sft: float = 1.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.w_photo = float(w_photo)
        self.w_deg_aux = float(w_deg_aux)
        self.w_code = float(w_code)
        self.w_clean_sft = float(w_clean_sft)
        self.label_smoothing = float(label_smoothing)

    @staticmethod
    def _masked_pool16(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = _ensure_mask(mask, x)
        num = F.adaptive_avg_pool2d(x * m, (16, 16))
        den = F.adaptive_avg_pool2d(m, (16, 16))
        pooled = num / den.clamp_min(1e-4)
        valid_16 = (den > 1e-3).float()
        return pooled, valid_16

    def compute_photo_loss(
        self,
        x_photo: torch.Tensor,
        x_in_512: torch.Tensor,
        gt_512: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
        is_clear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = _ensure_mask(observed_mask, x_photo)
        # 1) Low-frequency 16x16 L1 (exposure / illumination / haze alignment)
        p16, valid16 = self._masked_pool16(x_photo, m)
        g16, _ = self._masked_pool16(gt_512, m)
        l_lowfreq = masked_mean((p16 - g16).abs(), valid16)

        # 2) CIE-Lab ab chroma loss (color cast correction)
        ab_p = rgb_to_lab_ab(x_photo)
        ab_g = rgb_to_lab_ab(gt_512)
        l_lab_ab = masked_mean((ab_p - ab_g).abs(), m)

        # 3) Strict clear identity lock: |x_photo - x_in| on clear samples
        if is_clear is not None and is_clear.any():
            cm = is_clear.bool()
            l_clear_id = masked_l1(x_photo[cm], x_in_512[cm], m[cm])
        else:
            l_clear_id = x_photo.new_tensor(0.0)

        return l_lowfreq + 0.5 * l_lab_ab + 2.0 * l_clear_id

    @staticmethod
    def compute_deg_aux_loss(
        pred_deg: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        target_deg: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if pred_deg is None or target_deg is None:
            return ref_tensor.new_tensor(0.0)
        if isinstance(pred_deg, dict) and isinstance(target_deg, dict):
            keys = [k for k in ("exposure", "blur", "noise", "codec_lr", "codec") if k in pred_deg and k in target_deg]
            if not keys:
                return ref_tensor.new_tensor(0.0)
            losses = [F.smooth_l1_loss(pred_deg[k].float().reshape(-1), target_deg[k].float().reshape(-1)) for k in keys]
            return sum(losses) / len(losses)
        if isinstance(pred_deg, dict):
            pred_vec = torch.stack(
                [pred_deg[k].float().reshape(-1) for k in ("exposure", "blur", "noise", "codec_lr") if k in pred_deg],
                dim=-1,
            )
        else:
            pred_vec = pred_deg.float()
        if isinstance(target_deg, dict):
            tgt_vec = torch.stack(
                [target_deg[k].float().reshape(-1) for k in ("exposure", "blur", "noise", "codec_lr") if k in target_deg],
                dim=-1,
            )
        else:
            tgt_vec = target_deg.float()
        n_dim = min(pred_vec.shape[-1], tgt_vec.shape[-1])
        return F.smooth_l1_loss(pred_vec[..., :n_dim], tgt_vec[..., :n_dim])

    def compute_code_loss(
        self,
        logits: Optional[torch.Tensor],
        gt_indices: Optional[torch.Tensor],
        lq_feat_16: Optional[torch.Tensor],
        gt_feat_16: Optional[torch.Tensor],
        observed_mask: Optional[torch.Tensor],
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        l_ce = ref_tensor.new_tensor(0.0)
        l_cos = ref_tensor.new_tensor(0.0)

        if logits is not None and gt_indices is not None:
            # Normalize logits to (B, 1024, 256) and targets to (B, 256)
            if logits.ndim == 4:
                # (B, 1024, 16, 16) -> (B, 1024, 256)
                b, c, h, w = logits.shape
                log_flat = logits.float().reshape(b, c, h * w)
                n_tok = h * w
            elif logits.ndim == 3:
                # (B, 256, 1024) -> (B, 1024, 256)
                if logits.shape[-1] == 1024:
                    log_flat = logits.float().transpose(1, 2)
                    n_tok = logits.shape[1]
                else:
                    log_flat = logits.float()
                    n_tok = logits.shape[2]
            else:
                raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")

            tgt_flat = gt_indices.long().reshape(log_flat.shape[0], n_tok)
            ce_map = F.cross_entropy(
                log_flat,
                tgt_flat,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )  # (B, 256)
            if observed_mask is not None:
                side = int(math.isqrt(n_tok))
                m16 = F.interpolate(
                    _ensure_mask(observed_mask, observed_mask if observed_mask.ndim == 4 else observed_mask.unsqueeze(1)),
                    size=(side, side),
                    mode="nearest",
                ).reshape(log_flat.shape[0], n_tok)
                l_ce = (ce_map * m16).sum() / m16.sum().clamp_min(1.0)
            else:
                l_ce = ce_map.mean()

        if lq_feat_16 is not None and gt_feat_16 is not None:
            l_cos = masked_cosine_loss(lq_feat_16.float(), gt_feat_16.float(), observed_mask)

        return l_ce + 0.5 * l_cos

    def compute_clean_sft_loss(
        self,
        enc_used: Optional[Dict[str, torch.Tensor]],
        gt_enc_feats: Optional[Dict[str, torch.Tensor]],
        observed_mask: Optional[torch.Tensor],
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not enc_used or not gt_enc_feats:
            return ref_tensor.new_tensor(0.0)
        total = ref_tensor.new_tensor(0.0)
        w_sum = 0.0
        for scale_key, weight in self.SFT_WEIGHTS.items():
            if scale_key not in enc_used or scale_key not in gt_enc_feats:
                continue
            p = enc_used[scale_key].float()
            t = gt_enc_feats[scale_key].float().detach()
            m = _ensure_mask(observed_mask, p)
            l1 = masked_l1(p, t, m)
            cos = masked_cosine_loss(p, t, m)
            grad = masked_grad_l1(p, t, m)
            total = total + weight * (l1 + 0.25 * cos + 0.30 * grad)
            w_sum += weight
        return total / max(w_sum, 1e-6)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        teacher: Dict[str, torch.Tensor],
        x_in: torch.Tensor,
        gt: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        is_clear: Optional[torch.Tensor] = None,
        target_deg: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device_type = x_in.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_photo = (
                outputs["x_photo_256"] if "x_photo_256" in outputs else outputs["x_photo_512"]
            ).float()
            x_in_f = x_in.float()
            gt_f = gt.float()
            if x_in_f.shape[-2:] != x_photo.shape[-2:]:
                x_in_f = F.interpolate(x_in_f, size=x_photo.shape[-2:], mode="bilinear", align_corners=False)
            if gt_f.shape[-2:] != x_photo.shape[-2:]:
                gt_f = F.interpolate(gt_f, size=x_photo.shape[-2:], mode="bilinear", align_corners=False)

            l_photo = self.compute_photo_loss(
                x_photo, x_in_f, gt_f, observed_mask, is_clear=is_clear
            )
            pred_deg = outputs.get("deg_preds") if "deg_preds" in outputs else outputs.get("pred_deg")
            l_deg_aux = self.compute_deg_aux_loss(pred_deg, target_deg, ref_tensor=x_photo)

            logits = outputs.get("code_logits") if "code_logits" in outputs else outputs.get("logits")
            gt_indices = teacher.get("code_indices_gt") if "code_indices_gt" in teacher else teacher.get("gt_indices")
            gt_feat_16 = teacher.get("lq_feat_16_gt") if "lq_feat_16_gt" in teacher else teacher.get("gt_feat_16")
            l_code = self.compute_code_loss(
                logits=logits,
                gt_indices=gt_indices,
                lq_feat_16=outputs.get("lq_feat_16"),
                gt_feat_16=gt_feat_16,
                observed_mask=observed_mask,
                ref_tensor=x_photo,
            )
            gt_enc_feats = teacher.get("enc_gt") if "enc_gt" in teacher else teacher.get("gt_enc_feats")
            l_clean_sft = self.compute_clean_sft_loss(
                enc_used=outputs.get("enc_used"),
                gt_enc_feats=gt_enc_feats,
                observed_mask=observed_mask,
                ref_tensor=x_photo,
            )
            total = (
                self.w_photo * l_photo
                + self.w_deg_aux * l_deg_aux
                + self.w_code * l_code
                + self.w_clean_sft * l_clean_sft
            )
            metrics = {
                "loss_total": float(total.detach().item()),
                "l_photo": float(l_photo.detach().item()),
                "l_deg_aux": float(l_deg_aux.detach().item()),
                "l_code": float(l_code.detach().item()),
                "l_clean_sft": float(l_clean_sft.detach().item()),
            }
            return total, metrics


# ---------------------------------------------------------------------------
# Stage 2 Loss
# ---------------------------------------------------------------------------


def load_frozen_lpips(
    perceptual_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Load FrozenLPIPSVGG16 from a directory containing vgg16_features.pth & lpips_vgg_v01.pth.

    Supports both full manifest-verified directories (perceptual.load_perceptual) and
    flat weight directories (e.g., VM weights/ with direct or symlinked .pth files).
    """
    from research.codeformer_adaptation_v1.perceptual import FrozenLPIPSVGG16, load_perceptual

    root = Path(perceptual_dir)
    feat_path = root / "vgg16_features.pth"
    cal_path = root / "lpips_vgg_v01.pth"
    if not (feat_path.is_file() and cal_path.is_file()):
        raise FileNotFoundError(
            f"LPIPS weights (vgg16_features.pth, lpips_vgg_v01.pth) not found in {root}"
        )

    # Prefer strict manifest-verified loader when manifest.json is present and files are unlinked
    if (root / "manifest.json").is_file() and not feat_path.is_symlink() and not cal_path.is_symlink():
        try:
            return load_perceptual(root, device=device)
        except ValueError:
            pass

    payload = torch.load(feat_path, map_location="cpu", weights_only=True)
    features = payload["params"] if isinstance(payload, dict) and "params" in payload else payload
    calibration = torch.load(cal_path, map_location="cpu", weights_only=True)

    model = FrozenLPIPSVGG16()
    model.features.load_state_dict(
        {k.removeprefix("features."): v for k, v in features.items()}, strict=True
    )
    with torch.no_grad():
        for idx, weight in enumerate(model.calibration):
            src = calibration[f"lin{idx}.model.1.weight"]
            weight.copy_(src.float())
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Non-finite LPIPS weights")
    return model.to(device).eval().requires_grad_(False)


class Stage2Loss(nn.Module):
    """Stage-2 / Stage-2B / Stage-3B End-to-End Multi-Objective Fidelity + Identity + Sharpness Loss."""

    def __init__(
        self,
        perceptual_dir: Optional[Union[str, Path]] = None,
        arcface_onnx: Optional[Union[str, Path]] = None,
        lpips_model: Optional[nn.Module] = None,
        arcface_model: Optional[DifferentiableArcFaceLoss] = None,
        device: Union[str, torch.device] = "cpu",
        w_charb: float = 1.00,
        w_ssim: float = 0.35,
        w_chroma: float = 0.15,
        w_lpips: float = 0.25,
        w_arcface: float = 0.35,
        w_lap: float = 0.25,
        w_fft: float = 0.10,
        w_aux512: float = 0.20,
        w_ret_code: float = 0.15,
        w_ret_sft: float = 0.10,
        w_ret_photo: float = 0.05,
        charb_eps: float = 1e-3,
        w_vgg_gram: float = 0.0,
        w_hf_energy: float = 0.0,
        w_wmap: float = 0.0,
        w_ret_deg: float = 0.0,
        w_adain_floor: float = 0.0,
        w_pos_gate: float = 0.0,
        w_clear_l1: float = 2.0,
        v31_clear_anchor: bool = False,
        v31_aux512: bool = False,
        v33_mode: bool = False,
    ):
        super().__init__()
        self.w_charb = float(w_charb)
        self.w_ssim = float(w_ssim)
        self.w_chroma = float(w_chroma)
        self.w_lpips = float(w_lpips)
        self.w_arcface = float(w_arcface)
        self.w_lap = float(w_lap)
        self.w_fft = float(w_fft)
        self.w_aux512 = float(w_aux512)
        self.w_ret_code = float(w_ret_code)
        self.w_ret_sft = float(w_ret_sft)
        self.w_ret_photo = float(w_ret_photo)
        self.charb_eps = float(charb_eps)
        self.w_vgg_gram = float(w_vgg_gram)
        self.w_hf_energy = float(w_hf_energy)
        self.w_wmap = float(w_wmap)
        self.w_ret_deg = float(w_ret_deg)
        self.w_adain_floor = float(w_adain_floor)
        self.w_pos_gate = float(w_pos_gate)
        self.w_clear_l1 = float(w_clear_l1)
        self.v31_clear_anchor = bool(v31_clear_anchor)
        self.v31_aux512 = bool(v31_aux512)
        self.v33_mode = bool(v33_mode)

        self.lap_loss = MultiScaleLaplacianL1(scales=3).to(device)
        self.hf_energy_loss = MultiScaleHFEnergyLoss().to(device)
        self.stage1_helper = Stage1Loss()

        if lpips_model is not None:
            self.lpips = lpips_model.to(device).eval().requires_grad_(False)
        elif perceptual_dir is not None:
            self.lpips = load_frozen_lpips(perceptual_dir, device=device)
        else:
            self.lpips = None

        if arcface_model is not None:
            self.arcface = arcface_model.to(device).eval().requires_grad_(False)
        elif arcface_onnx is not None:
            self.arcface = DifferentiableArcFaceLoss(onnx_path=arcface_onnx, device=device)
        else:
            self.arcface = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.lpips is not None:
            self.lpips.eval()
        if self.arcface is not None:
            self.arcface.eval()
        return self

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_256: torch.Tensor,
        gt_512: Optional[torch.Tensor] = None,
        x_in_512: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        is_clear: Optional[torch.Tensor] = None,
        landmarks: Optional[torch.Tensor] = None,
        teacher: Optional[Dict[str, torch.Tensor]] = None,
        deg_targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        blur_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device_type = gt_256.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            y_256 = outputs["y_256"].float()
            gt256_f = gt_256.float()
            m256 = _ensure_mask(observed_mask, y_256)

            l_charb = masked_charbonnier(y_256, gt256_f, m256, eps=self.charb_eps)
            l_ssim = masked_ms_ssim7_loss(y_256, gt256_f, m256, scales=3)
            l_chroma = masked_mean(
                (rgb_to_ycbcr_chroma(y_256) - rgb_to_ycbcr_chroma(gt256_f)).abs(), m256
            )
            l_lap = self.lap_loss(y_256, gt256_f, m256)
            l_fft = log_fft_amp_l1(y_256, gt256_f, m256)

            # LPIPS + optional single-pass VGG16 Gram style on [-1, 1] with ~observed replaced by GT
            l_lpips = y_256.new_tensor(0.0)
            l_vgg_gram = y_256.new_tensor(0.0)
            if self.lpips is not None and (self.w_lpips > 0 or self.w_vgg_gram > 0):
                y_masked = y_256 * m256 + gt256_f * (1.0 - m256)
                if self.w_vgg_gram > 0:
                    l_lpips_raw, l_vgg_gram = compute_lpips_and_gram(
                        self.lpips,
                        y_masked * 2.0 - 1.0,
                        gt256_f * 2.0 - 1.0,
                        compute_gram=True,
                    )
                    if self.w_lpips > 0:
                        l_lpips = l_lpips_raw
                elif self.w_lpips > 0:
                    # Preserve exact V3.0 numerical equivalence when w_vgg_gram == 0
                    l_lpips = self.lpips(y_masked * 2.0 - 1.0, gt256_f * 2.0 - 1.0)

            # 2-Scale phase-invariant masked log-HF-energy loss
            if self.w_hf_energy > 0:
                l_hf_energy = self.hf_energy_loss(y_256, gt256_f, m256)
            else:
                l_hf_energy = y_256.new_tensor(0.0)

            # Per-site w_maps sufficiency ceiling/floor hinge + sufficiency aux
            if self.w_wmap > 0 and deg_targets is not None:
                l_wmap = compute_wmap_loss(
                    w_maps=outputs.get("w_maps"),
                    deg_targets=deg_targets,
                    observed_mask=observed_mask,
                    is_clear=is_clear,
                    sufficiency=outputs.get("sufficiency"),
                    ref_tensor=y_256,
                    blur_mask=blur_mask,
                    v33_mode=self.v33_mode,
                )
            else:
                l_wmap = y_256.new_tensor(0.0)

            # Differentiable FP32 ArcFace identity loss
            if self.arcface is not None and self.w_arcface > 0:
                l_arcface = self.arcface(y_256, gt256_f, landmarks=landmarks, mask=m256)
            else:
                l_arcface = y_256.new_tensor(0.0)

            # Clear-Anchor (V3.0 legacy vs V3.1 photometric-aware boost vs V3.3 per-pixel suppression)
            alpha_bypass = outputs.get("alpha_bypass")
            l_clear_anchor = y_256.new_tensor(0.0)
            if is_clear is not None:
                cm = is_clear.bool()
                dm = ~cm
                if cm.any():
                    l_c = self.w_clear_l1 * masked_l1(y_256[cm], gt256_f[cm], m256[cm])
                    if alpha_bypass is not None:
                        ab_c = alpha_bypass[cm].float()
                        w_clear_ab = 0.10 if self.v33_mode else 0.50
                        l_c = l_c + w_clear_ab * masked_mean((1.0 - ab_c).abs(), _ensure_mask(m256[cm], ab_c))
                    l_clear_anchor = l_clear_anchor + l_c
                if dm.any() and alpha_bypass is not None:
                    ab_d = alpha_bypass[dm].float()
                    m_d = _ensure_mask(m256[dm], ab_d)
                    if self.v33_mode:
                        # Per-pixel alpha_bypass suppression wherever m_blur > 0.10 OR exp > 0.10 OR noise > 0.08
                        if deg_targets is not None:
                            exp_all, blur_all, noise_all, _ = _extract_deg_components_4(deg_targets, y_256.device)
                            exp_d = exp_all[dm].view(-1, 1, 1, 1)
                            noise_d = noise_all[dm].view(-1, 1, 1, 1)
                            if blur_mask is not None:
                                mb_d = _ensure_mask(blur_mask[dm], ab_d)
                            else:
                                mb_d = blur_all[dm].view(-1, 1, 1, 1).expand_as(ab_d)
                            deg_px = ((mb_d > 0.10) | (exp_d > 0.10) | (noise_d > 0.08)).float()
                        elif blur_mask is not None:
                            mb_d = _ensure_mask(blur_mask[dm], ab_d)
                            deg_px = (mb_d > 0.10).float()
                        else:
                            deg_px = torch.ones_like(ab_d)
                        m_deg_px = deg_px * m_d
                        if m_deg_px.sum() > 0.5:
                            l_clear_anchor = l_clear_anchor + 1.50 * masked_mean(ab_d.abs() + ab_d * ab_d, m_deg_px)
                    elif self.v31_clear_anchor:
                        if deg_targets is not None:
                            exp_all, _, _ = _extract_deg_components(deg_targets, y_256.device)
                            exp_d = exp_all[dm]
                            w_photo_boost = (1.0 + 1.5 * (exp_d > 0.10).float()).view(-1, 1, 1, 1)
                        else:
                            w_photo_boost = 1.0
                        l_clear_anchor = l_clear_anchor + 1.20 * masked_mean(
                            w_photo_boost * (ab_d.abs() + ab_d * ab_d), m_d
                        )
                    else:
                        l_clear_anchor = l_clear_anchor + 0.2 * masked_mean(ab_d.abs(), m_d)

            # Aux 512x512 loss
            y_512 = outputs.get("y_512")
            if y_512 is not None and gt_512 is not None and self.w_aux512 > 0:
                y512_f = y_512.float()
                gt512_f = gt_512.float()
                m512 = _ensure_mask(observed_mask, y512_f)
                if self.v31_aux512:
                    l_aux512 = (
                        0.50 * masked_charbonnier(y512_f, gt512_f, m512, eps=self.charb_eps)
                        + 0.15 * self.lap_loss(y512_f, gt512_f, m512)
                        + 0.35 * self.hf_energy_loss(y512_f, gt512_f, m512)
                    )
                else:
                    l_aux512 = masked_charbonnier(y512_f, gt512_f, m512, eps=self.charb_eps) + 0.25 * self.lap_loss(
                        y512_f, gt512_f, m512
                    )
            else:
                l_aux512 = y_256.new_tensor(0.0)

            # Retention terms: L_code + L_clean_sft + L_photo + optional V3.1 L_ret_deg
            l_ret_code = y_256.new_tensor(0.0)
            l_ret_sft = y_256.new_tensor(0.0)
            l_ret_photo = y_256.new_tensor(0.0)
            l_ret_deg = y_256.new_tensor(0.0)
            if teacher is not None:
                logits = outputs.get("code_logits") if "code_logits" in outputs else outputs.get("logits")
                gt_indices = teacher.get("code_indices_gt") if "code_indices_gt" in teacher else teacher.get("gt_indices")
                gt_feat_16 = teacher.get("lq_feat_16_gt") if "lq_feat_16_gt" in teacher else teacher.get("gt_feat_16")
                gt_enc_feats = teacher.get("enc_gt") if "enc_gt" in teacher else teacher.get("gt_enc_feats")
                l_ret_code = self.stage1_helper.compute_code_loss(
                    logits=logits,
                    gt_indices=gt_indices,
                    lq_feat_16=outputs.get("lq_feat_16"),
                    gt_feat_16=gt_feat_16,
                    observed_mask=observed_mask,
                    ref_tensor=y_256,
                )
                l_ret_sft = self.stage1_helper.compute_clean_sft_loss(
                    enc_used=outputs.get("enc_used"),
                    gt_enc_feats=gt_enc_feats,
                    observed_mask=observed_mask,
                    ref_tensor=y_256,
                )
            x_photo = outputs.get("x_photo_256")
            if x_photo is None:
                x_photo = outputs.get("x_photo_512")
            if x_photo is not None and x_in_512 is not None and self.w_ret_photo > 0:
                xp = x_photo.float()
                xin = x_in_512.float()
                gt_p = gt256_f if xp.shape[-2:] == gt256_f.shape[-2:] else (gt_512.float() if gt_512 is not None else gt256_f)
                if xin.shape[-2:] != xp.shape[-2:]:
                    xin = F.interpolate(xin, size=xp.shape[-2:], mode="bilinear", align_corners=False)
                if gt_p.shape[-2:] != xp.shape[-2:]:
                    gt_p = F.interpolate(gt_p, size=xp.shape[-2:], mode="bilinear", align_corners=False)
                l_ret_photo = self.stage1_helper.compute_photo_loss(
                    xp,
                    xin,
                    gt_p,
                    observed_mask,
                    is_clear=is_clear,
                )
            if self.w_ret_deg > 0 and deg_targets is not None:
                pred_deg = outputs.get("deg_preds") if "deg_preds" in outputs else outputs.get("pred_deg")
                l_ret_deg = self.stage1_helper.compute_deg_aux_loss(
                    pred_deg, deg_targets, ref_tensor=y_256
                )

            # V3.2 AdaIN floor & non-negative V3FreqHead gate regularizers
            alpha_adain = outputs.get("alpha_adain")
            if self.w_adain_floor > 0 and alpha_adain is not None:
                l_adain_floor = F.relu(0.85 - alpha_adain.float()).mean()
            else:
                l_adain_floor = y_256.new_tensor(0.0)

            freq_gates = outputs.get("freq_gates")
            if self.w_pos_gate > 0 and freq_gates:
                gate_penalties = [F.relu(-g.float()).mean() for g in freq_gates.values()]
                l_pos_gate = sum(gate_penalties) / float(len(gate_penalties))
            else:
                l_pos_gate = y_256.new_tensor(0.0)

            total = (
                self.w_charb * l_charb
                + self.w_ssim * l_ssim
                + self.w_chroma * l_chroma
                + self.w_lpips * l_lpips
                + self.w_vgg_gram * l_vgg_gram
                + self.w_hf_energy * l_hf_energy
                + self.w_wmap * l_wmap
                + self.w_arcface * l_arcface
                + self.w_lap * l_lap
                + self.w_fft * l_fft
                + l_clear_anchor
                + self.w_aux512 * l_aux512
                + self.w_ret_code * l_ret_code
                + self.w_ret_sft * l_ret_sft
                + self.w_ret_photo * l_ret_photo
                + self.w_ret_deg * l_ret_deg
                + self.w_adain_floor * l_adain_floor
                + self.w_pos_gate * l_pos_gate
            )
            metrics = {
                "loss_total": float(total.detach().item()),
                "l_charb": float(l_charb.detach().item()),
                "l_ssim": float(l_ssim.detach().item()),
                "l_chroma": float(l_chroma.detach().item()),
                "l_lpips": float(l_lpips.detach().item()),
                "l_vgg_gram": float(l_vgg_gram.detach().item()),
                "l_hf_energy": float(l_hf_energy.detach().item()),
                "l_wmap": float(l_wmap.detach().item()),
                "l_arcface": float(l_arcface.detach().item()),
                "l_lap": float(l_lap.detach().item()),
                "l_fft": float(l_fft.detach().item()),
                "l_clear_anchor": float(l_clear_anchor.detach().item()),
                "l_aux512": float(l_aux512.detach().item()),
                "l_ret_code": float(l_ret_code.detach().item()),
                "l_ret_sft": float(l_ret_sft.detach().item()),
                "l_ret_photo": float(l_ret_photo.detach().item()),
                "l_ret_deg": float(l_ret_deg.detach().item()),
                "l_adain_floor": float(l_adain_floor.detach().item()),
                "l_pos_gate": float(l_pos_gate.detach().item()),
            }
            return total, metrics


# ---------------------------------------------------------------------------
# Stage 3 Spectral-Norm PatchGAN Discriminator & Lazy R1 Penalty
# ---------------------------------------------------------------------------


class PatchGANDiscriminator(nn.Module):
    """Spectral-Norm PatchGAN discriminator for Stage-3 degraded-only texture polish.

    Uses zero BatchNorm layers so R1 gradient penalty is exact and batch-independent.
    Expects RGB images in [-1, 1] of shape (B, 3, 256, 256).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.net(x.float())


def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def d_nonsaturating_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def g_nonsaturating_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(-fake_logits).mean()


def compute_r1_penalty(real_logits: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty: 0.5 * E[||grad_x D(x)||_2^2] in FP32."""
    grad_real = torch.autograd.grad(
        outputs=real_logits.float().sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return 0.5 * grad_real.float().pow(2).reshape(grad_real.shape[0], -1).sum(dim=1).mean()


class Stage3GANHelper:
    """Degraded-only Stage-3 GAN + lazy R1 (gamma=10 every 16 steps) helper."""

    def __init__(
        self,
        discriminator: PatchGANDiscriminator,
        lambda_gan: float = 0.01,
        r1_gamma: float = 10.0,
        r1_every: int = 16,
        gan_mode: str = "hinge",
    ):
        if gan_mode not in ("hinge", "non_saturating"):
            raise ValueError(f"Unsupported gan_mode: {gan_mode}")
        self.discriminator = discriminator
        self.lambda_gan = float(lambda_gan)
        self.r1_gamma = float(r1_gamma)
        self.r1_every = int(r1_every)
        self.gan_mode = gan_mode

    def generator_gan_loss(
        self,
        y_256: torch.Tensor,
        gt_256: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
        is_clear: torch.Tensor,
        lambda_scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply GAN loss strictly to degraded samples (`clear` masked out!)."""
        eff_lambda = self.lambda_gan * float(lambda_scale)
        deg_mask = ~is_clear.bool()
        if eff_lambda <= 0.0 or not deg_mask.any():
            return y_256.float().sum() * 0.0
        with torch.amp.autocast(device_type=y_256.device.type, enabled=False):
            m = _ensure_mask(observed_mask, y_256)[deg_mask]
            fake_deg = (y_256[deg_mask].float() * m + gt_256[deg_mask].float() * (1.0 - m)) * 2.0 - 1.0
            fake_logits = self.discriminator(fake_deg)
            g_loss = g_hinge_loss(fake_logits) if self.gan_mode == "hinge" else g_nonsaturating_loss(fake_logits)
            return eff_lambda * g_loss

    def discriminator_step_loss(
        self,
        y_256_detached: torch.Tensor,
        gt_256: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
        is_clear: torch.Tensor,
        step: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Discriminator loss + lazy R1 strictly on degraded samples."""
        deg_mask = ~is_clear.bool()
        if not deg_mask.any():
            return None, {"d_loss": 0.0, "d_r1": 0.0}
        with torch.amp.autocast(device_type=gt_256.device.type, enabled=False):
            m = _ensure_mask(observed_mask, gt_256)[deg_mask]
            real_deg = (gt_256[deg_mask].float() * 2.0 - 1.0).detach()
            fake_deg = (
                (y_256_detached[deg_mask].float() * m + gt_256[deg_mask].float() * (1.0 - m)) * 2.0 - 1.0
            ).detach()

            do_r1 = self.r1_every > 0 and (step % self.r1_every == 0)
            if do_r1:
                real_deg.requires_grad_(True)

            real_logits = self.discriminator(real_deg)
            fake_logits = self.discriminator(fake_deg)
            d_loss = (
                d_hinge_loss(real_logits, fake_logits)
                if self.gan_mode == "hinge"
                else d_nonsaturating_loss(real_logits, fake_logits)
            )
            r1_val = 0.0
            if do_r1:
                r1 = compute_r1_penalty(real_logits, real_deg)
                # compute_r1_penalty already includes the 0.5 factor; scale by r1_gamma * r1_every
                d_loss = d_loss + (self.r1_gamma * self.r1_every) * r1
                r1_val = float(r1.detach().item())
            return d_loss, {"d_loss": float(d_loss.detach().item()), "d_r1": r1_val}


# ---------------------------------------------------------------------------
# V3.4 ("Pre-Conditioned Frozen CodeFormer") Stage-1v4 & Stage-2v4 Losses
# ---------------------------------------------------------------------------

V34_HARD_NEG_REGIMES: Tuple[str, ...] = (
    "local_partial_blur",
    "mild_transition",
    "daylight_sensor_noise",
)


def _tv_4x4(params_4x4: Optional[torch.Tensor], ref_tensor: torch.Tensor) -> torch.Tensor:
    if params_4x4 is None:
        return ref_tensor.new_tensor(0.0)
    p = params_4x4.float()
    tv_x = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
    tv_y = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
    return 0.5 * (tv_x + tv_y)


def compute_v34_struct_clean_loss(
    x_pre: torch.Tensor,
    x_struct_clean: torch.Tensor,
    observed_mask: Optional[torch.Tensor] = None,
    charb_eps: float = 1e-3,
) -> torch.Tensor:
    """Charb + 0.25*GradL1 + 0.20*Lab_ab between x_pre_256 and x_struct_clean_256 in FP32."""
    m = _ensure_mask(observed_mask, x_pre)
    xp = x_pre.float()
    xt = x_struct_clean.float()
    l_charb = masked_charbonnier(xp, xt, m, eps=charb_eps)
    l_grad = masked_grad_l1(xp, xt, m)
    l_lab = masked_mean((rgb_to_lab_ab(xp) - rgb_to_lab_ab(xt)).abs(), m)
    return l_charb + 0.25 * l_grad + 0.20 * l_lab


def compute_v34_id_lock_loss(
    x_pad: torch.Tensor,
    x_photo: torch.Tensor,
    x_pre: torch.Tensor,
    g_photo: torch.Tensor,
    g_pre: torch.Tensor,
    deg_targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
    observed_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Strict identity locks on photo_net (when exp < 0.05) and pre_net (when noise < 0.05 & grid < 0.05)
    plus SmoothL1 gate supervision on g_photo and g_pre.
    """
    device = x_pre.device
    exp, _, noise, grid = _extract_deg_components_4(deg_targets, device)
    m = _ensure_mask(observed_mask, x_pre)

    # 1) |x_photo - x_pad| on exposure_sev < 0.05
    no_exp = exp < 0.05
    if no_exp.any():
        l_photo_id = masked_l1(x_photo[no_exp].float(), x_pad[no_exp].float(), m[no_exp])
    else:
        l_photo_id = x_pre.new_tensor(0.0)

    # 2) |x_pre - x_photo| on noise_sev < 0.05 & grid_sev < 0.05
    no_ng = (noise < 0.05) & (grid < 0.05)
    if no_ng.any():
        l_pre_id = masked_l1(x_pre[no_ng].float(), x_photo[no_ng].float(), m[no_ng])
    else:
        l_pre_id = x_pre.new_tensor(0.0)

    # 3) Continuous gate supervision
    g_photo_tgt = (exp / 0.15).clamp(0.0, 1.0)
    g_pre_tgt = (torch.maximum(noise, grid) / 0.15).clamp(0.0, 1.0)
    l_g_photo = F.smooth_l1_loss(g_photo.float().reshape(-1), g_photo_tgt)
    l_g_pre = F.smooth_l1_loss(g_pre.float().reshape(-1), g_pre_tgt)

    return l_photo_id + l_pre_id + l_g_photo + l_g_pre


def compute_v34_clean_bce(
    p_clean: Optional[torch.Tensor],
    is_clear: Optional[torch.Tensor],
    ref_tensor: torch.Tensor,
    *,
    regimes: Optional[Sequence[str]] = None,
    is_nn_pixelated: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted BCE(p_clean, is_clear) with 2x weight on hard negatives."""
    if p_clean is None or is_clear is None:
        return ref_tensor.new_tensor(0.0)
    pc = p_clean.float().reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    tgt = is_clear.float().reshape(-1).to(device=pc.device)
    bce_per = F.binary_cross_entropy(pc, tgt, reduction="none")

    weights = torch.ones_like(bce_per)
    if regimes is not None:
        hard_reg = torch.tensor(
            [str(r) in V34_HARD_NEG_REGIMES for r in regimes],
            device=pc.device,
            dtype=torch.bool,
        )
        if hard_reg.shape == weights.shape:
            weights = torch.where(hard_reg & (tgt < 0.5), torch.full_like(weights, 2.0), weights)
    if is_nn_pixelated is not None:
        nn_mask = is_nn_pixelated.bool().reshape(-1).to(device=pc.device)
        if nn_mask.shape == weights.shape:
            weights = torch.where(nn_mask & (tgt < 0.5), torch.full_like(weights, 2.0), weights)

    return (bce_per * weights).sum() / weights.sum().clamp_min(1.0)


class V34Stage1Loss(nn.Module):
    """V3.4 Stage-1v4 Pre-Conditioner + Codebook Warmup Loss (all FP32, skips frozen decoder)."""

    def __init__(
        self,
        w_struct_clean: float = 1.00,
        w_photo: float = 0.50,
        w_id_lock: float = 2.00,
        w_deg_aux: float = 0.50,
        w_clean_bce: float = 1.00,
        w_code: float = 0.35,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.w_struct_clean = float(w_struct_clean)
        self.w_photo = float(w_photo)
        self.w_id_lock = float(w_id_lock)
        self.w_deg_aux = float(w_deg_aux)
        self.w_clean_bce = float(w_clean_bce)
        self.w_code = float(w_code)
        self.stage1_helper = Stage1Loss(label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        teacher: Optional[Dict[str, torch.Tensor]],
        x_struct_clean: torch.Tensor,
        x_photo_tgt: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        is_clear: Optional[torch.Tensor] = None,
        deg_targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        regimes: Optional[Sequence[str]] = None,
        is_nn_pixelated: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_pre = outputs["x_pre_256"]
        with torch.amp.autocast(device_type=x_pre.device.type, enabled=False):
            xp = x_pre.float()
            xph = outputs["x_photo_256"].float()
            xpad = outputs["x_pad_256"].float()
            m = _ensure_mask(observed_mask, xp)

            l_struct = compute_v34_struct_clean_loss(xp, x_struct_clean.float(), m)

            # Photo loss: LowFreq16 + 0.40*Lab_ab + 0.05*TV_4x4
            p16, valid16 = Stage1Loss._masked_pool16(xph, m)
            g16, _ = Stage1Loss._masked_pool16(x_photo_tgt.float(), m)
            l_lowfreq = masked_mean((p16 - g16).abs(), valid16)
            l_lab_ph = masked_mean((rgb_to_lab_ab(xph) - rgb_to_lab_ab(x_photo_tgt.float())).abs(), m)
            l_tv = _tv_4x4(outputs.get("photo_params_4x4"), xp)
            l_photo = l_lowfreq + 0.40 * l_lab_ph + 0.05 * l_tv

            if deg_targets is not None:
                l_id_lock = compute_v34_id_lock_loss(
                    x_pad=xpad,
                    x_photo=xph,
                    x_pre=xp,
                    g_photo=outputs["g_photo"],
                    g_pre=outputs["g_pre"],
                    deg_targets=deg_targets,
                    observed_mask=m,
                )
                l_deg_aux = Stage1Loss.compute_deg_aux_loss(outputs.get("deg_preds"), deg_targets, ref_tensor=xp)
            else:
                l_id_lock = xp.new_tensor(0.0)
                l_deg_aux = xp.new_tensor(0.0)

            l_clean_bce = compute_v34_clean_bce(
                outputs.get("p_clean"),
                is_clear,
                ref_tensor=xp,
                regimes=regimes,
                is_nn_pixelated=is_nn_pixelated,
            )

            if teacher is not None and self.w_code > 0:
                l_code = self.stage1_helper.compute_code_loss(
                    logits=outputs.get("code_logits"),
                    gt_indices=teacher.get("code_indices_gt"),
                    lq_feat_16=outputs.get("lq_feat_16"),
                    gt_feat_16=teacher.get("lq_feat_16_gt"),
                    observed_mask=m,
                    ref_tensor=xp,
                )
            else:
                l_code = xp.new_tensor(0.0)

            total = (
                self.w_struct_clean * l_struct
                + self.w_photo * l_photo
                + self.w_id_lock * l_id_lock
                + self.w_deg_aux * l_deg_aux
                + self.w_clean_bce * l_clean_bce
                + self.w_code * l_code
            )
            metrics = {
                "loss_total": float(total.detach().item()),
                "l_struct_clean": float(l_struct.detach().item()),
                "l_photo": float(l_photo.detach().item()),
                "l_id_lock": float(l_id_lock.detach().item()),
                "l_deg_aux": float(l_deg_aux.detach().item()),
                "l_clean_bce": float(l_clean_bce.detach().item()),
                "l_code": float(l_code.detach().item()),
            }
            return total, metrics


class V34Stage2Loss(nn.Module):
    """V3.4 Stage-2v4 End-to-End Through Frozen CodeFormer Loss (all FP32, zero PatchGAN)."""

    def __init__(
        self,
        perceptual_dir: Optional[Union[str, Path]] = None,
        arcface_onnx: Optional[Union[str, Path]] = None,
        lpips_model: Optional[nn.Module] = None,
        arcface_model: Optional[DifferentiableArcFaceLoss] = None,
        device: Union[str, torch.device] = "cpu",
        w_lpips: float = 0.75,
        w_vgg_gram: float = 0.20,
        w_arcface: float = 0.50,
        w_charb: float = 0.30,
        w_chroma: float = 0.15,
        w_struct_clean: float = 0.35,
        w_code: float = 0.20,
        w_id_lock: float = 1.50,
        w_clean_bce: float = 1.00,
        w_clear_l1: float = 2.00,
        w_w_reg: float = 0.25,
        w_margin: float = 0.06,
        charb_eps: float = 1e-3,
    ):
        super().__init__()
        self.w_lpips = float(w_lpips)
        self.w_vgg_gram = float(w_vgg_gram)
        self.w_arcface = float(w_arcface)
        self.w_charb = float(w_charb)
        self.w_chroma = float(w_chroma)
        self.w_struct_clean = float(w_struct_clean)
        self.w_code = float(w_code)
        self.w_id_lock = float(w_id_lock)
        self.w_clean_bce = float(w_clean_bce)
        self.w_clear_l1 = float(w_clear_l1)
        self.w_w_reg = float(w_w_reg)
        self.w_margin = float(w_margin)
        self.charb_eps = float(charb_eps)
        self.stage1_helper = Stage1Loss(label_smoothing=0.05)

        if lpips_model is not None:
            self.lpips = lpips_model.to(device).eval().requires_grad_(False)
        elif perceptual_dir is not None:
            self.lpips = load_frozen_lpips(perceptual_dir, device=device)
        else:
            self.lpips = None

        if arcface_model is not None:
            self.arcface = arcface_model.to(device).eval().requires_grad_(False)
        elif arcface_onnx is not None:
            self.arcface = DifferentiableArcFaceLoss(onnx_path=arcface_onnx, device=device)
        else:
            self.arcface = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.lpips is not None:
            self.lpips.eval()
        if self.arcface is not None:
            self.arcface.eval()
        return self

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_256: torch.Tensor,
        x_struct_clean: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        is_clear: Optional[torch.Tensor] = None,
        landmarks: Optional[torch.Tensor] = None,
        teacher: Optional[Dict[str, torch.Tensor]] = None,
        deg_targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        regimes: Optional[Sequence[str]] = None,
        is_nn_pixelated: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        y_256 = outputs["y_256"]
        with torch.amp.autocast(device_type=y_256.device.type, enabled=False):
            y_f = y_256.float()
            gt_f = gt_256.float()
            m256 = _ensure_mask(observed_mask, y_f)

            l_charb = masked_charbonnier(y_f, gt_f, m256, eps=self.charb_eps)
            l_chroma = masked_mean((rgb_to_ycbcr_chroma(y_f) - rgb_to_ycbcr_chroma(gt_f)).abs(), m256)

            l_lpips = y_f.new_tensor(0.0)
            l_vgg_gram = y_f.new_tensor(0.0)
            if self.lpips is not None and (self.w_lpips > 0 or self.w_vgg_gram > 0):
                y_masked = y_f * m256 + gt_f * (1.0 - m256)
                l_lp_raw, l_gr_raw = compute_lpips_and_gram(
                    self.lpips,
                    y_masked * 2.0 - 1.0,
                    gt_f * 2.0 - 1.0,
                    compute_gram=self.w_vgg_gram > 0,
                )
                if self.w_lpips > 0:
                    l_lpips = l_lp_raw
                if self.w_vgg_gram > 0:
                    l_vgg_gram = l_gr_raw

            if self.arcface is not None and self.w_arcface > 0:
                l_arcface = self.arcface(y_f, gt_f, landmarks=landmarks, mask=m256)
            else:
                l_arcface = y_f.new_tensor(0.0)

            if x_struct_clean is not None and self.w_struct_clean > 0:
                l_struct = compute_v34_struct_clean_loss(
                    outputs["x_pre_256"].float(), x_struct_clean.float(), m256, charb_eps=self.charb_eps
                )
            else:
                l_struct = y_f.new_tensor(0.0)

            if teacher is not None and self.w_code > 0:
                l_code = self.stage1_helper.compute_code_loss(
                    logits=outputs.get("code_logits"),
                    gt_indices=teacher.get("code_indices_gt"),
                    lq_feat_16=outputs.get("lq_feat_16"),
                    gt_feat_16=teacher.get("lq_feat_16_gt"),
                    observed_mask=m256,
                    ref_tensor=y_f,
                )
            else:
                l_code = y_f.new_tensor(0.0)

            if deg_targets is not None and self.w_id_lock > 0:
                l_id_lock = compute_v34_id_lock_loss(
                    x_pad=outputs["x_pad_256"].float(),
                    x_photo=outputs["x_photo_256"].float(),
                    x_pre=outputs["x_pre_256"].float(),
                    g_photo=outputs["g_photo"],
                    g_pre=outputs["g_pre"],
                    deg_targets=deg_targets,
                    observed_mask=m256,
                )
            else:
                l_id_lock = y_f.new_tensor(0.0)

            l_clean_bce = compute_v34_clean_bce(
                outputs.get("p_clean"),
                is_clear,
                ref_tensor=y_f,
                regimes=regimes,
                is_nn_pixelated=is_nn_pixelated,
            )

            if is_clear is not None and is_clear.any() and self.w_clear_l1 > 0:
                cm = is_clear.bool()
                l_clear_l1 = masked_l1(y_f[cm], gt_f[cm], m256[cm])
            else:
                l_clear_l1 = y_f.new_tensor(0.0)

            # Scalar w_pred deadzone regularizer (margin=0.06)
            w_pred = outputs.get("w_pred")
            if w_pred is not None and deg_targets is not None and self.w_w_reg > 0:
                _, blur, noise, grid = _extract_deg_components_4(deg_targets, y_f.device)
                w_tgt = (0.95 - 0.25 * torch.maximum(blur, grid) - 0.15 * noise).clamp(0.65, 0.95)
                excess = F.relu((w_pred.float().reshape(-1) - w_tgt).abs() - self.w_margin)
                l_w_reg = F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.05)
            else:
                l_w_reg = y_f.new_tensor(0.0)

            total = (
                self.w_lpips * l_lpips
                + self.w_vgg_gram * l_vgg_gram
                + self.w_arcface * l_arcface
                + self.w_charb * l_charb
                + self.w_chroma * l_chroma
                + self.w_struct_clean * l_struct
                + self.w_code * l_code
                + self.w_id_lock * l_id_lock
                + self.w_clean_bce * l_clean_bce
                + self.w_clear_l1 * l_clear_l1
                + self.w_w_reg * l_w_reg
            )
            metrics = {
                "loss_total": float(total.detach().item()),
                "l_lpips": float(l_lpips.detach().item()),
                "l_vgg_gram": float(l_vgg_gram.detach().item()),
                "l_arcface": float(l_arcface.detach().item()),
                "l_charb": float(l_charb.detach().item()),
                "l_chroma": float(l_chroma.detach().item()),
                "l_struct_clean": float(l_struct.detach().item()),
                "l_code": float(l_code.detach().item()),
                "l_id_lock": float(l_id_lock.detach().item()),
                "l_clean_bce": float(l_clean_bce.detach().item()),
                "l_clear_l1": float(l_clear_l1.detach().item()),
                "l_w_reg": float(l_w_reg.detach().item()),
            }
            return total, metrics

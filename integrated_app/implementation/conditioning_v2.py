"""Pure input-only V2 conditioning and V2 pyramid/head modules for IMPLEMENTATION.

Importing this module loads zero neural frameworks (`torch` / `numpy` passed explicitly).
All conditioning signals depend strictly on the observed 256x256 RGB input x in [0, 1].
"""
from __future__ import annotations

import math

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
SQRT2 = 1.4142135623730951
IMAGE_SIZE = 256
BLOCK_SIZE = 16
GRID_SIZE = 16
TOP1PCT_PIXELS = 655

DARK_HIGH_LOW = 0.640
DARK_HIGH_HIGH = 0.680
CLEAR_MIN_DIR_LAP_CUTOFF = 0.00915
BLUR_MIN_DIR_LAP_LOW = 0.00480
DARK_SMOOTH_LAP_HIGH = 0.00260
DARK_SMOOTH_LAP_LOW = 0.00120
DARK_NOISE_DIFF_LOW = 0.00760
DARK_NOISE_DIFF_HIGH = 0.01250
PATCH_DIR_LAP_HIGH = 0.00880
PATCH_DIR_LAP_LOW = 0.00380

DEFAULT_ANCHOR_BETA = 0.005
DEFAULT_DISPLACEMENT_TAU = 1e-6
SITES = ("32", "64", "128", "256")
CHANNELS = {"32": 256, "64": 256, "128": 128, "256": 128}
PYR_WIDTH = 96
PYR_BRANCH = 32
PYR_SE_HIDDEN = 24
DARK_WIDTH = 48
HEAD_V2_WIDTH = 48
INIT_SEED_V2 = 20261026


def need(condition, message):
    if not condition:
        raise ValueError(message)


def displacement_gamma(max_linf_displacement, *, tau=DEFAULT_DISPLACEMENT_TAU):
    need(type(max_linf_displacement) in (int, float)
         and math.isfinite(max_linf_displacement)
         and float(max_linf_displacement) >= 0.0,
         "Finite nonnegative parameter displacement required.")
    need(type(tau) in (int, float) and math.isfinite(tau) and float(tau) > 0.0,
         "Positive displacement threshold tau required.")
    return float(min(1.0, float(max_linf_displacement) / float(tau)))


def block16_mean_numpy(array, *, np):
    need(isinstance(array, np.ndarray) and array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
         and array.dtype.kind == "f" and np.isfinite(array).all(),
         "Finite floating HWC RGB256 array required.")
    blocks = array.astype(np.float64).reshape(
        GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE, 3
    ).mean(axis=(1, 3), keepdims=True)
    return np.repeat(np.repeat(blocks, BLOCK_SIZE, axis=1), BLOCK_SIZE, axis=3).reshape(
        IMAGE_SIZE, IMAGE_SIZE, 3
    )


def smooth_coarse_256_numpy(diff_hwc, *, np):
    """Smooth 16x16 coarse exposure drift via bilinear upsampling + 2-pass 17x17 reflect box smoothing."""
    need(isinstance(diff_hwc, np.ndarray) and diff_hwc.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
         and diff_hwc.dtype.kind == "f" and np.isfinite(diff_hwc).all(),
         "Finite floating HWC RGB256 array required.")
    b16 = diff_hwc.astype(np.float64).reshape(
        GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE, 3
    ).mean(axis=(1, 3))
    coords = np.clip((np.arange(IMAGE_SIZE, dtype=np.float64) + 0.5) / float(BLOCK_SIZE) - 0.5, 0.0, float(GRID_SIZE - 1))
    i0 = np.floor(coords).astype(np.int32)
    i1 = np.minimum(i0 + 1, GRID_SIZE - 1)
    w1 = (coords - i0)[:, None]
    w0 = 1.0 - w1
    vy = w0[:, None, :] * b16[i0] + w1[:, None, :] * b16[i1]
    up = w0[None, :, :] * vy[:, i0, :] + w1[None, :, :] * vy[:, i1, :]
    for _ in range(2):
        px = np.pad(up, ((0, 0), (8, 8), (0, 0)), mode="reflect")
        csx = np.pad(np.cumsum(px, axis=1), ((0, 0), (1, 0), (0, 0)))
        hx = (csx[:, 17:, :] - csx[:, :-17, :]) / 17.0
        py = np.pad(hx, ((8, 8), (0, 0), (0, 0)), mode="reflect")
        csy = np.pad(np.cumsum(py, axis=0), ((1, 0), (0, 0), (0, 0)))
        up = (csy[17:, :, :] - csy[:-17, :, :]) / 17.0
    return up


def _binomial3_2d_numpy(luma, *, np):
    padded = np.pad(luma, ((1, 1), (1, 1)), mode="reflect")
    horiz = 0.25 * padded[:, :-2] + 0.50 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    vert = 0.25 * horiz[:-2, :] + 0.50 * horiz[1:-1, :] + 0.25 * horiz[2:, :]
    return vert


def _directional_lap_maps_numpy(luma, *, np):
    lx = np.pad(np.abs(luma[:, 2:] - 2.0 * luma[:, 1:-1] + luma[:, :-2]), ((0, 0), (1, 1)), mode="edge")
    ly = np.pad(np.abs(luma[2:, :] - 2.0 * luma[1:-1, :] + luma[:-2, :]), ((1, 1), (0, 0)), mode="edge")
    ld1 = np.pad(np.abs(luma[2:, 2:] - 2.0 * luma[1:-1, 1:-1] + luma[:-2, :-2]) / SQRT2, ((1, 1), (1, 1)), mode="edge")
    ld2 = np.pad(np.abs(luma[2:, :-2] - 2.0 * luma[1:-1, 1:-1] + luma[:-2, 2:]) / SQRT2, ((1, 1), (1, 1)), mode="edge")
    return lx, ly, ld1, ld2


def central_laplacian_sharpness_numpy(rgb_f32, *, np) -> float:
    w = np.asarray(LUMA_WEIGHTS, dtype=np.float64)
    luma = rgb_f32.astype(np.float64) @ w
    c = luma[64:192, 64:192]
    lx = np.abs(c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]).mean()
    ly = np.abs(c[2:, :] - 2.0 * c[1:-1, :] + c[:-2, :]).mean()
    return float(0.5 * (lx + ly))


def compute_v2_conditioning_numpy(input_rgb, *, np):
    """Compute g_dark, 4-directional g_blur, g_sensor_noise, and 16x16 spatial blur gate in pure NumPy."""
    need(isinstance(input_rgb, np.ndarray) and input_rgb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
         and input_rgb.dtype.kind == "f" and np.isfinite(input_rgb).all()
         and float(input_rgb.min()) >= 0.0 and float(input_rgb.max()) <= 1.0,
         "Finite floating HWC RGB256 input in [0, 1] required.")
    x = input_rgb.astype(np.float64)
    w = np.asarray(LUMA_WEIGHTS, dtype=np.float64)
    luma = x @ w
    mean_luma = float(luma.mean())
    sorted_luma = np.sort(luma.reshape(-1))
    top1pct_luma = float(sorted_luma[-TOP1PCT_PIXELS:].mean())
    g_dark = float(np.clip(
        (DARK_HIGH_HIGH - top1pct_luma) / (DARK_HIGH_HIGH - DARK_HIGH_LOW),
        0.0, 1.0,
    ))

    c = luma[64:192, 64:192]
    c_lx = float(np.abs(c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]).mean())
    c_ly = float(np.abs(c[2:, :] - 2.0 * c[1:-1, :] + c[:-2, :]).mean())
    c_ld1 = float(np.abs(c[2:, 2:] - 2.0 * c[1:-1, 1:-1] + c[:-2, :-2]).mean() / SQRT2)
    c_ld2 = float(np.abs(c[2:, :-2] - 2.0 * c[1:-1, 1:-1] + c[:-2, 2:]).mean() / SQRT2)
    raw_central_lap = float(0.5 * (c_lx + c_ly))
    min_dir_lap = float(min(c_lx, c_ly, c_ld1, c_ld2))

    smooth = _binomial3_2d_numpy(luma, np=np)
    sx = np.pad(np.abs(smooth[:, 2:] - 2.0 * smooth[:, 1:-1] + smooth[:, :-2]), ((0, 0), (1, 1)), mode="edge")
    sy = np.pad(np.abs(smooth[2:, :] - 2.0 * smooth[1:-1, :] + smooth[:-2, :]), ((1, 1), (0, 0)), mode="edge")
    local_smooth_lap = 0.5 * (sx + sy)
    central_smooth_lap = float(local_smooth_lap[64:192, 64:192].mean())
    noise_hf_diff = float(max(0.0, raw_central_lap - central_smooth_lap))

    if g_dark == 0.0:
        g_blur = float(np.clip(
            (CLEAR_MIN_DIR_LAP_CUTOFF - min_dir_lap) / (CLEAR_MIN_DIR_LAP_CUTOFF - BLUR_MIN_DIR_LAP_LOW),
            0.0, 1.0,
        ))
        g_sensor_noise = 0.0
    else:
        g_blur = float(np.clip(
            (DARK_SMOOTH_LAP_HIGH - central_smooth_lap) / (DARK_SMOOTH_LAP_HIGH - DARK_SMOOTH_LAP_LOW),
            0.0, 1.0,
        ))
        g_sensor_noise = float(g_dark * np.clip(
            (noise_hf_diff - DARK_NOISE_DIFF_LOW) / (DARK_NOISE_DIFF_HIGH - DARK_NOISE_DIFF_LOW),
            0.0, 1.0,
        ))

    if g_blur == 0.0:
        effective_16 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        outer_ring_lap = raw_central_lap
    else:
        lx, ly, ld1, ld2 = _directional_lap_maps_numpy(luma, np=np)
        iso_256 = 0.5 * (lx + ly)
        outer_sum = float(iso_256.sum() - iso_256[64:192, 64:192].sum())
        outer_ring_lap = float(outer_sum / float(IMAGE_SIZE * IMAGE_SIZE - 128 * 128))
        p_lx = lx.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(axis=(1, 3))
        p_ly = ly.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(axis=(1, 3))
        p_ld1 = ld1.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(axis=(1, 3))
        p_ld2 = ld2.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).mean(axis=(1, 3))
        p_min = np.minimum(np.minimum(p_lx, p_ly), np.minimum(p_ld1, p_ld2))
        local_16 = np.clip((PATCH_DIR_LAP_HIGH - p_min) / (PATCH_DIR_LAP_HIGH - PATCH_DIR_LAP_LOW), 0.0, 1.0)
        p16 = np.pad(local_16, ((1, 1), (1, 1)), mode="edge")
        smooth_16 = (
            p16[:-2, :-2] + p16[:-2, 1:-1] + p16[:-2, 2:]
            + p16[1:-1, :-2] + p16[1:-1, 1:-1] + p16[1:-1, 2:]
            + p16[2:, :-2] + p16[2:, 1:-1] + p16[2:, 2:]
        ) / 9.0
        gate_activate = min(1.0, g_blur / 0.20)
        effective_16 = (np.maximum(smooth_16, 0.65 * g_blur) * gate_activate).astype(np.float32)

    return {
        "g_dark": g_dark,
        "g_blur": g_blur,
        "g_sensor_noise": g_sensor_noise,
        "mean_luma": mean_luma,
        "top1pct_luma": top1pct_luma,
        "raw_central_laplacian": raw_central_lap,
        "min_directional_laplacian": min_dir_lap,
        "outer_ring_laplacian": outer_ring_lap,
        "central_smooth_laplacian": central_smooth_lap,
        "noise_hf_diff": noise_hf_diff,
        "max_local_blur": float(effective_16.max()),
        "local_map_16": effective_16,
    }


def compute_v2_conditioning_torch(input_nchw, *, torch):
    """Compute detached V2 conditioning & per-site spatial maps (32, 64, 128, 256) in Torch."""
    F = torch.nn.functional
    need(isinstance(input_nchw, torch.Tensor) and input_nchw.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
         and input_nchw.dtype == torch.float32,
         "FP32 NCHW RGB256 input tensor required.")
    with torch.no_grad():
        x = input_nchw.detach()
        w = torch.tensor(LUMA_WEIGHTS, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        luma = (x * w).sum(dim=1, keepdim=True)
        sorted_luma, _ = torch.sort(luma.reshape(-1))
        top1pct_luma = float(sorted_luma[-TOP1PCT_PIXELS:].mean().item())
        g_dark = float(min(1.0, max(0.0, (DARK_HIGH_HIGH - top1pct_luma) / (DARK_HIGH_HIGH - DARK_HIGH_LOW))))

        c = luma[:, :, 64:192, 64:192]
        c_lx = float((c[:, :, :, 2:] - 2.0 * c[:, :, :, 1:-1] + c[:, :, :, :-2]).abs().mean().item())
        c_ly = float((c[:, :, 2:, :] - 2.0 * c[:, :, 1:-1, :] + c[:, :, :-2, :]).abs().mean().item())
        c_ld1 = float(((c[:, :, 2:, 2:] - 2.0 * c[:, :, 1:-1, 1:-1] + c[:, :, :-2, :-2]).abs().mean().item()) / SQRT2)
        c_ld2 = float(((c[:, :, 2:, :-2] - 2.0 * c[:, :, 1:-1, 1:-1] + c[:, :, :-2, 2:]).abs().mean().item()) / SQRT2)
        raw_central_lap = float(0.5 * (c_lx + c_ly))
        min_dir_lap = float(min(c_lx, c_ly, c_ld1, c_ld2))

        k1d = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float32, device=x.device)
        k2d = (k1d[:, None] * k1d[None, :])[None, None]
        smooth = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode="reflect"), k2d)
        sx = F.pad((smooth[:, :, :, 2:] - 2.0 * smooth[:, :, :, 1:-1] + smooth[:, :, :, :-2]).abs(), (1, 1, 0, 0), mode="replicate")
        sy = F.pad((smooth[:, :, 2:, :] - 2.0 * smooth[:, :, 1:-1, :] + smooth[:, :, :-2, :]).abs(), (0, 0, 1, 1), mode="replicate")
        local_smooth_lap = 0.5 * (sx + sy)
        central_smooth_lap = float(local_smooth_lap[:, :, 64:192, 64:192].mean().item())
        noise_hf_diff = float(max(0.0, raw_central_lap - central_smooth_lap))

        if g_dark == 0.0:
            g_blur = float(min(1.0, max(0.0, (CLEAR_MIN_DIR_LAP_CUTOFF - min_dir_lap) / (CLEAR_MIN_DIR_LAP_CUTOFF - BLUR_MIN_DIR_LAP_LOW))))
            g_sensor_noise = 0.0
        else:
            g_blur = float(min(1.0, max(0.0, (DARK_SMOOTH_LAP_HIGH - central_smooth_lap) / (DARK_SMOOTH_LAP_HIGH - DARK_SMOOTH_LAP_LOW))))
            g_sensor_noise = float(g_dark * min(1.0, max(0.0, (noise_hf_diff - DARK_NOISE_DIFF_LOW) / (DARK_NOISE_DIFF_HIGH - DARK_NOISE_DIFF_LOW))))

        if g_blur == 0.0:
            effective_16 = torch.zeros((1, 1, GRID_SIZE, GRID_SIZE), dtype=torch.float32, device=x.device)
        else:
            lx = F.pad((luma[:, :, :, 2:] - 2.0 * luma[:, :, :, 1:-1] + luma[:, :, :, :-2]).abs(), (1, 1, 0, 0), mode="replicate")
            ly = F.pad((luma[:, :, 2:, :] - 2.0 * luma[:, :, 1:-1, :] + luma[:, :, :-2, :]).abs(), (0, 0, 1, 1), mode="replicate")
            ld1 = F.pad((luma[:, :, 2:, 2:] - 2.0 * luma[:, :, 1:-1, 1:-1] + luma[:, :, :-2, :-2]).abs() / SQRT2, (1, 1, 1, 1), mode="replicate")
            ld2 = F.pad((luma[:, :, 2:, :-2] - 2.0 * luma[:, :, 1:-1, 1:-1] + luma[:, :, :-2, 2:]).abs() / SQRT2, (1, 1, 1, 1), mode="replicate")
            p_lx = F.avg_pool2d(lx, kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
            p_ly = F.avg_pool2d(ly, kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
            p_ld1 = F.avg_pool2d(ld1, kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
            p_ld2 = F.avg_pool2d(ld2, kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
            p_min = torch.minimum(torch.minimum(p_lx, p_ly), torch.minimum(p_ld1, p_ld2))
            local_16 = torch.clamp((PATCH_DIR_LAP_HIGH - p_min) / (PATCH_DIR_LAP_HIGH - PATCH_DIR_LAP_LOW), 0.0, 1.0)
            box3 = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=x.device) / 9.0
            smooth_16 = F.conv2d(F.pad(local_16, (1, 1, 1, 1), mode="replicate"), box3)
            gate_activate = min(1.0, g_blur / 0.20)
            effective_16 = torch.clamp(smooth_16, min=0.65 * g_blur) * float(gate_activate)

        site_maps = {
            s: F.interpolate(effective_16, size=(int(s), int(s)), mode="bilinear", align_corners=False)
            for s in SITES
        }
        return {
            "g_dark": g_dark,
            "g_blur": g_blur,
            "g_sensor_noise": g_sensor_noise,
            "top1pct_luma": top1pct_luma,
            "raw_central_laplacian": raw_central_lap,
            "min_directional_laplacian": min_dir_lap,
            "central_smooth_laplacian": central_smooth_lap,
            "noise_hf_diff": noise_hf_diff,
            "max_local_blur": float(effective_16.max().item()),
            "local_map_16": effective_16,
            "site_maps": site_maps,
        }


def condition_output_v2_numpy(pred_rgb, input_rgb, *, np, baseline_rgb=None,
                              beta=DEFAULT_ANCHOR_BETA, gamma=1.0, dark_drift_align=True):
    """Apply V2 exposure-decoupled block16 anchoring and smooth dark coarse-drift protection."""
    cond = compute_v2_conditioning_numpy(input_rgb, np=np)
    if float(gamma) == 0.0:
        return pred_rgb.astype(np.float32).copy(), cond
    pred = pred_rgb.astype(np.float64)
    inp = input_rgb.astype(np.float64)
    base = pred if baseline_rgb is None else baseline_rgb.astype(np.float64)
    g_dark = cond["g_dark"]
    g_blur = cond["g_blur"]
    normal_w = float(gamma) * (1.0 - g_dark)
    drift_suppression = 1.0 - 0.70 * min(1.0, g_blur / 0.25)
    coarse_shift = -drift_suppression * block16_mean_numpy(pred - base, np=np) + float(beta) * block16_mean_numpy(inp - base, np=np)
    out = np.clip(pred + normal_w * coarse_shift, 0.0, 1.0)
    if dark_drift_align and baseline_rgb is not None and g_dark > 0.0:
        out = np.clip(out - g_dark * smooth_coarse_256_numpy(out - base, np=np), 0.0, 1.0)
    return out.astype(np.float32), cond


def build_v2_trainable_modules(*, torch, seed: int = INIT_SEED_V2):
    """Build deterministic V2 trainable modules with zero-initialized residual projections."""
    nn = torch.nn
    F = torch.nn.functional

    cpu_rng = torch.get_rng_state().clone()
    try:
        torch.manual_seed(int(seed))

        class MultiScaleDilatedSEBlock(nn.Module):
            def __init__(self, width: int = PYR_WIDTH, branch: int = PYR_BRANCH, se_hidden: int = PYR_SE_HIDDEN):
                super().__init__()
                self.branch = branch
                self.dw1 = nn.Conv2d(branch, branch, kernel_size=3, padding=1, dilation=1, groups=branch, bias=True)
                self.dw2 = nn.Conv2d(branch, branch, kernel_size=3, padding=2, dilation=2, groups=branch, bias=True)
                self.dw4 = nn.Conv2d(branch, branch, kernel_size=3, padding=4, dilation=4, groups=branch, bias=True)
                self.pw = nn.Conv2d(width, width, kernel_size=1, bias=True)
                self.se1 = nn.Conv2d(width, se_hidden, kernel_size=1, bias=True)
                self.se2 = nn.Conv2d(se_hidden, width, kernel_size=1, bias=True)
                self.act = nn.SiLU()

            def forward(self, h):
                b1, b2, b3 = torch.split(h, self.branch, dim=1)
                ms = torch.cat([self.act(self.dw1(b1)), self.act(self.dw2(b2)), self.act(self.dw4(b3))], dim=1)
                fused = self.pw(ms)
                pooled = fused.mean(dim=(-2, -1), keepdim=True)
                se = torch.sigmoid(self.se2(self.act(self.se1(pooled))))
                return h + se * fused

        class PyramidSiteStage(nn.Module):
            def __init__(self, channels: int, has_top_down: bool):
                super().__init__()
                in_ch = channels + 1 + (PYR_WIDTH if has_top_down else 0)
                self.in_proj = nn.Conv2d(in_ch, PYR_WIDTH, kernel_size=1, bias=True)
                self.act = nn.SiLU()
                self.block1 = MultiScaleDilatedSEBlock()
                self.block2 = MultiScaleDilatedSEBlock()
                self.out_proj = nn.Conv2d(PYR_WIDTH, channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)

            def forward(self, feat, site_map, prev_hidden=None):
                parts = [feat, site_map]
                if prev_hidden is not None:
                    up_prev = F.interpolate(prev_hidden, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                    parts.append(up_prev)
                h = self.act(self.in_proj(torch.cat(parts, dim=1)))
                h = self.block2(self.block1(h))
                delta = site_map * self.out_proj(self.act(h))
                return delta, h

        class TopDownMultiScaleDilatedSFTPyramid(nn.Module):
            def __init__(self):
                super().__init__()
                self.stages = nn.ModuleDict({
                    s: PyramidSiteStage(CHANNELS[s], has_top_down=(s != "32"))
                    for s in SITES
                })

            def step_site(self, site: str, feat, site_map, prev_hidden):
                if float(site_map.detach().max().item()) <= 0.0:
                    delta = feat * 0.0
                    next_h = torch.zeros(
                        (feat.shape[0], PYR_WIDTH, int(site), int(site)),
                        dtype=feat.dtype, device=feat.device,
                    )
                    return delta, next_h
                return self.stages[site](feat, site_map, prev_hidden)

        class DarkNoiseSiteStage(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.in_proj = nn.Conv2d(channels, DARK_WIDTH, kernel_size=1, bias=True)
                self.dw1 = nn.Conv2d(DARK_WIDTH, DARK_WIDTH, kernel_size=3, padding=1, dilation=1, groups=DARK_WIDTH, bias=True)
                self.dw2 = nn.Conv2d(DARK_WIDTH, DARK_WIDTH, kernel_size=3, padding=2, dilation=2, groups=DARK_WIDTH, bias=True)
                self.pw = nn.Conv2d(2 * DARK_WIDTH, DARK_WIDTH, kernel_size=1, bias=True)
                self.act = nn.SiLU()
                self.out_proj = nn.Conv2d(DARK_WIDTH, channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)

            def forward(self, feat, g_sensor_noise):
                max_g = float(g_sensor_noise.detach().max().item()) if isinstance(g_sensor_noise, torch.Tensor) else float(g_sensor_noise)
                if max_g <= 0.0:
                    return feat * 0.0
                h = self.act(self.in_proj(feat))
                ms = torch.cat([self.act(self.dw1(h)), self.act(self.dw2(h))], dim=1)
                h = h + self.act(self.pw(ms))
                scale = g_sensor_noise if isinstance(g_sensor_noise, torch.Tensor) else float(g_sensor_noise)
                return scale * self.out_proj(h)

        class DarkNoiseSFTRefiner(nn.Module):
            def __init__(self):
                super().__init__()
                self.stages = nn.ModuleDict({s: DarkNoiseSiteStage(CHANNELS[s]) for s in SITES})

        class FrequencySelectiveSiteHeadV2(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.freq_gate = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.freq_gate.weight)
                nn.init.zeros_(self.freq_gate.bias)
                self.enc_p = nn.Conv2d(channels, HEAD_V2_WIDTH, kernel_size=1, bias=True)
                self.dec_p = nn.Conv2d(channels, HEAD_V2_WIDTH, kernel_size=1, bias=False)
                self.fus_p = nn.Conv2d(channels, HEAD_V2_WIDTH, kernel_size=1, bias=False)
                self.coa_p = nn.Conv2d(256, HEAD_V2_WIDTH, kernel_size=1, bias=False)
                self.dw1 = nn.Conv2d(HEAD_V2_WIDTH, HEAD_V2_WIDTH, kernel_size=3, padding=1, dilation=1, groups=HEAD_V2_WIDTH, bias=True)
                self.dw2 = nn.Conv2d(HEAD_V2_WIDTH, HEAD_V2_WIDTH, kernel_size=3, padding=2, dilation=2, groups=HEAD_V2_WIDTH, bias=True)
                self.pw = nn.Conv2d(2 * HEAD_V2_WIDTH, HEAD_V2_WIDTH, kernel_size=1, bias=True)
                self.act = nn.SiLU()
                self.out_proj = nn.Conv2d(HEAD_V2_WIDTH, channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)

            @staticmethod
            def _highpass5(z):
                lp = F.avg_pool2d(z, kernel_size=5, stride=1, padding=2, count_include_pad=False)
                return z - lp

            def forward(self, base_post, used, decoder, fused, coarse, gate_map):
                if float(gate_map.detach().max().item()) <= 0.0:
                    return base_post
                hp_diff = self._highpass5(decoder) - self._highpass5(base_post)
                g_freq = torch.tanh(self.freq_gate(torch.cat([decoder, base_post], dim=1)))
                c_term = self.coa_p(coarse)
                if c_term.shape[-2:] != fused.shape[-2:]:
                    c_term = F.interpolate(c_term, size=fused.shape[-2:], mode="bilinear", align_corners=False)
                h = self.act(self.enc_p(used) + self.dec_p(decoder) + self.fus_p(fused) + c_term)
                ms = torch.cat([self.act(self.dw1(h)), self.act(self.dw2(h))], dim=1)
                r_spatial = self.out_proj(self.act(h + self.pw(ms)))
                return base_post + gate_map * (g_freq * hp_diff + r_spatial)

        class V2TrainableBundle(nn.Module):
            def __init__(self):
                super().__init__()
                self.pyramid = TopDownMultiScaleDilatedSFTPyramid()
                self.dark_refiner = DarkNoiseSFTRefiner()
                self.head_v2 = nn.ModuleDict({s: FrequencySelectiveSiteHeadV2(CHANNELS[s]) for s in SITES})

            def refine_site_sequential(self, site: str, e_u96_s, cond_info, prev_hidden):
                d_pyr, next_h = self.pyramid.step_site(site, e_u96_s, cond_info["site_maps"][site], prev_hidden)
                d_dark = self.dark_refiner.stages[site](e_u96_s, cond_info["g_sensor_noise"])
                return e_u96_s + d_pyr + d_dark, next_h

            def refine_sft_features(self, e_u96_dict, cond_info):
                used = {}
                prev_h = None
                for s in SITES:
                    used[s], prev_h = self.refine_site_sequential(s, e_u96_dict[s], cond_info, prev_h)
                return used

            def refine_post_sft(self, site: str, base_post, used, decoder, fused, coarse, cond_info):
                g_sn = cond_info["g_sensor_noise"]
                sn_term = 0.50 * g_sn if isinstance(g_sn, torch.Tensor) else 0.50 * float(g_sn)
                m_s = torch.clamp(cond_info["site_maps"][site] + sn_term, 0.0, 1.0)
                return self.head_v2[site](base_post, used, decoder, fused, coarse, m_s)

        return V2TrainableBundle()
    finally:
        torch.set_rng_state(cpu_rng)

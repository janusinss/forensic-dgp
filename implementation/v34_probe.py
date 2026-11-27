"""Cached, thread-safe in-process V3.4 DegradationEncoder + PhotoCorrectNet probe.

Loads ONLY the lightweight 0.29M-parameter ``V34DegradationEncoder`` and 0.04M-parameter
``V34PhotoCorrectNet`` from the already-bundled ``v34_precond_modules.npz`` (0 MB added weights;
never loads the 376.6 MB CodeFormer backbone). Degrades cleanly to ``None`` if torch or the
bundled ``.npz`` is absent, corrupted, or unavailable.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import threading
from typing import Any, Callable, Sequence
import zipfile
import zlib

import numpy as np
from PIL import Image, ImageFilter

from .runtime import MODEL_DIR, V34_FILE, ordinary_file, prepare_crop

_ASSETS_NPZ = Path(__file__).with_name("assets") / "v34_precond_modules.npz"
_LOCK = threading.Lock()
_LOADED = False
_PROBE_BUNDLE: dict[str, Any] | None = None

CLEAN_P_CLEAN_MIN = 0.82
CLEAN_MAX_DEG_CAP = 0.20
PHOTO_NET_DAMP_THRESHOLD = 0.65


def reset_probe_cache() -> None:
    """Clear the cached V3.4 probe modules (primarily for deterministic unit testing)."""
    global _LOADED, _PROBE_BUNDLE
    with _LOCK:
        _LOADED = False
        _PROBE_BUNDLE = None


def _resolve_v34_npz_path() -> Path | None:
    for candidate in (_ASSETS_NPZ, MODEL_DIR / "v34_precond_modules.npz"):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _load_probe_bundle() -> dict[str, Any] | None:
    """Lazy, thread-safe loader for V34DegradationEncoder + V34PhotoCorrectNet."""
    global _LOADED, _PROBE_BUNDLE
    if _LOADED:
        return _PROBE_BUNDLE
    with _LOCK:
        if _LOADED:
            return _PROBE_BUNDLE
        bundle: dict[str, Any] | None = None
        npz_path = _resolve_v34_npz_path()
        if npz_path is not None:
            try:
                import torch
                from .architecture_v34 import (
                    V34DegradationEncoder,
                    V34PhotoCorrectNet,
                    inspect_v34_npz,
                    replicate_pad_observed_torch,
                )

                verified_path = ordinary_file(npz_path, maximum=32 * 1024**2)
                raw = verified_path.read_bytes()
                if (
                    len(raw) == V34_FILE["bytes"]
                    and hashlib.sha256(raw).hexdigest() == V34_FILE["sha256"]
                ):
                    arrays = inspect_v34_npz(raw)
                    deg_sd = {
                        k[len("deg_encoder."):]: torch.from_numpy(v.copy())
                        for k, v in arrays.items()
                        if k.startswith("deg_encoder.")
                    }
                    photo_sd = {
                        k[len("photo_net."):]: torch.from_numpy(v.copy())
                        for k, v in arrays.items()
                        if k.startswith("photo_net.")
                    }
                    enc = V34DegradationEncoder()
                    enc.load_state_dict(deg_sd, strict=True)
                    enc.eval().requires_grad_(False)

                    photo = V34PhotoCorrectNet()
                    photo.load_state_dict(photo_sd, strict=True)
                    photo.eval().requires_grad_(False)

                    bundle = {
                        "torch": torch,
                        "deg_encoder": enc,
                        "photo_net": photo,
                        "replicate_pad": replicate_pad_observed_torch,
                    }
            except (
                ImportError,
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                zipfile.BadZipFile,
                zlib.error,
            ):
                bundle = None
        _PROBE_BUNDLE = bundle
        _LOADED = True
        return _PROBE_BUNDLE


def _finite_unit(val: float, default: float = 0.0) -> float:
    fval = float(val)
    if not math.isfinite(fval):
        return default
    return round(min(1.0, max(0.0, fval)), 4)


def _format_single_output(
    out: dict[str, Any],
    idx: int,
    *,
    photo_stats: dict[str, float] | None = None,
) -> dict[str, float]:
    dp = out["deg_preds"]
    res: dict[str, float] = {
        "exposure": _finite_unit(dp["exposure"][idx].item()),
        "blur": _finite_unit(dp["blur"][idx].item()),
        "noise": _finite_unit(dp["noise"][idx].item()),
        "codec_lr": _finite_unit(dp["codec_lr"][idx].item()),
        "g_photo": _finite_unit(out["g_photo"][idx].item()),
        "g_pre": _finite_unit(out["g_pre"][idx].item()),
        "w_pred": _finite_unit(out["w_pred"][idx].item()),
        "p_clean": _finite_unit(out["p_clean"][idx].item()),
        "alpha_clean": _finite_unit(out["alpha_clean"][idx].item()),
    }
    if photo_stats is not None:
        res.update(photo_stats)
    return res


def _is_valid_rgb_array(rgb: Any) -> bool:
    return (
        isinstance(rgb, np.ndarray)
        and rgb.ndim == 3
        and rgb.shape[2] == 3
        and rgb.dtype == np.uint8
        and min(rgb.shape[:2]) >= 8
    )


def probe_batch(
    rgb_list: Sequence[np.ndarray],
    *,
    include_photo: bool = False,
) -> list[dict[str, float]] | None:
    """Score a batch of native RGB uint8 crops in one batched ``V34DegradationEncoder`` forward.

    Returns a list of JSON-safe probe dicts (one per input image), or ``None`` if the probe
    is unavailable or inputs are invalid.
    """
    if not rgb_list:
        return None
    bundle = _load_probe_bundle()
    if bundle is None:
        return None
    canvases = []
    masks = []
    for rgb in rgb_list:
        if not _is_valid_rgb_array(rgb):
            return None
        canvas_256, observed, _, _ = prepare_crop(rgb)
        canvases.append(canvas_256)
        masks.append(observed)

    torch = bundle["torch"]
    enc = bundle["deg_encoder"]
    photo = bundle["photo_net"]
    replicate_pad = bundle["replicate_pad"]
    try:
        x_np = np.stack([c.astype(np.float32) / np.float32(255.0) for c in canvases], axis=0)
        x = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()
        m_np = np.stack([obs.astype(np.float32) for obs in masks], axis=0)
        m = torch.from_numpy(m_np).unsqueeze(1).contiguous()
        with torch.inference_mode():
            x_pad = replicate_pad(x, m)
            out = enc(x_pad, m)
            photo_list: list[dict[str, float]] | None = None
            if include_photo:
                x_photo, params_4x4 = photo(x_pad, out["f4"], out["z_deg"], out["g_photo"])
                photo_list = []
                for idx in range(len(rgb_list)):
                    obs_i = m[idx : idx + 1] > 0.5
                    diff_i = (x_photo[idx : idx + 1] - x_pad[idx : idx + 1]).abs()
                    if bool(obs_i.any()):
                        delta_mean = float(diff_i.masked_select(obs_i.expand_as(diff_i)).mean().item())
                    else:
                        delta_mean = float(diff_i.mean().item())
                    b_haze_mean = float(params_4x4[idx, 0:3].mean().item())
                    g_tone_mean = float(params_4x4[idx, 3:4].mean().item())
                    photo_list.append({
                        "photo_delta_mean": round(max(0.0, delta_mean), 4) if math.isfinite(delta_mean) else 0.0,
                        "photo_b_haze_mean": _finite_unit(b_haze_mean),
                        "photo_g_tone_mean": round(float(g_tone_mean), 4) if math.isfinite(g_tone_mean) else 1.0,
                    })
        return [
            _format_single_output(out, i, photo_stats=photo_list[i] if photo_list is not None else None)
            for i in range(len(rgb_list))
        ]
    except (RuntimeError, ValueError, TypeError):
        return None


def probe_v34(rgb: np.ndarray, *, include_photo: bool = False) -> dict[str, float] | None:
    """Run single-crop ``V34DegradationEncoder`` (+ optional ``V34PhotoCorrectNet``) probe."""
    results = probe_batch([rgb], include_photo=include_photo)
    if not results:
        return None
    return results[0]


def is_clean_control_probe(probe: dict[str, float] | None) -> bool:
    """Return True when V3.4 strongly confirms a clean, artifact-free face crop."""
    if not isinstance(probe, dict):
        return False
    p_clean = float(probe.get("p_clean", 0.0))
    max_deg = max(
        float(probe.get("exposure", 1.0)),
        float(probe.get("blur", 1.0)),
        float(probe.get("noise", 1.0)),
        float(probe.get("codec_lr", 1.0)),
    )
    return p_clean >= CLEAN_P_CLEAN_MIN and max_deg < CLEAN_MAX_DEG_CAP


def photometric_damping_factor(probe: dict[str, float] | None) -> float:
    """Return damping multiplier in [0.35, 1.0] to prevent double-correction with V34PhotoCorrectNet."""
    if not isinstance(probe, dict):
        return 1.0
    g_photo = float(probe.get("g_photo", 0.0))
    if not math.isfinite(g_photo) or g_photo < PHOTO_NET_DAMP_THRESHOLD:
        return 1.0
    # Smooth linear damping from 0.72 at g_photo=0.65 down to 0.35 at g_photo=1.0.
    t = min(1.0, max(0.0, (g_photo - PHOTO_NET_DAMP_THRESHOLD) / (1.0 - PHOTO_NET_DAMP_THRESHOLD)))
    return round(0.72 - 0.37 * t, 3)


def stage_reflection_accept(
    before_probe: dict[str, float] | None,
    after_probe: dict[str, float] | None,
    stage_name: str,
) -> tuple[bool, str]:
    """Verify via V3.4 probe that a non-mosaic Step 2 stage did not harm a clean face or spike artifacts."""
    if not isinstance(before_probe, dict) or not isinstance(after_probe, dict):
        return True, "probe_unavailable"

    b_clean = float(before_probe.get("p_clean", 0.0))
    a_clean = float(after_probe.get("p_clean", 0.0))
    b_blur = float(before_probe.get("blur", 0.0))
    a_blur = float(after_probe.get("blur", 0.0))
    b_noise = float(before_probe.get("noise", 0.0))
    a_noise = float(after_probe.get("noise", 0.0))

    # Veto marginal cosmetic stages on strongly verified clean portraits.
    if is_clean_control_probe(before_probe) and stage_name in (
        "chroma_debleeding",
        "low_light_recovery",
        "glare_recovery",
    ):
        return False, "clean_control_veto"

    # Never allow chroma_debleeding to degrade p_clean on already-clean inputs.
    if stage_name == "chroma_debleeding" and b_clean >= 0.50 and (a_clean < b_clean - 0.08):
        return False, "clean_probability_drop"

    if stage_name == "noise_denoise":
        noise_drop = b_noise - a_noise
        blur_jump = a_blur - b_blur
        if blur_jump > 0.22 and noise_drop < 0.05:
            return False, "collateral_blur_spike"
        if b_clean >= 0.82 and a_clean < b_clean - 0.15 and noise_drop < 0.08:
            return False, "clean_control_veto"
        return True, "accepted"

    if stage_name in (
        "chroma_debleeding",
        "cast_dehaze",
        "color_cast_balance",
        "haze_dehaze",
        "low_light_recovery",
        "glare_recovery",
        "motion_deblur",
    ):
        if a_blur > b_blur + 0.15:
            return False, "collateral_blur_spike"
        if a_noise > b_noise + 0.15:
            return False, "collateral_noise_spike"
        if stage_name != "motion_deblur" and b_clean >= 0.72 and a_clean < b_clean - 0.15:
            return False, "clean_probability_drop"

    return True, "accepted"


def _sharpness_256(rgb: np.ndarray) -> float:
    """Exact quality_signals blur_variance on the observed 256x256 canvas."""
    import cv2

    canvas, observed, _, _ = prepare_crop(rgb)
    support = cv2.erode(observed.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    support[:4] = False
    support[-4:] = False
    support[:, :4] = False
    support[:, -4:] = False
    if not bool(support.any()):
        support = observed
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0.6)
    return float(cv2.Laplacian(smoothed, cv2.CV_32F)[support].var())


def _block_residual_score(rgb: np.ndarray) -> float:
    """Fast 256-scale periodic seam residual for candidate mosaic sigma ranking."""
    from .automatic_preparation import pixelation_signal

    h, w = rgb.shape[:2]
    if min(h, w) < 24:
        return 0.0
    sig = pixelation_signal(rgb, (0, 0, w, h))
    return float(sig.get("score", 0.0)) if sig.get("suspected") else 0.0


def choose_closed_loop_mosaic_sigma(
    base_crop: np.ndarray,
    base_radius_native: float,
    *,
    geometric_pattern: str | None = None,
    detected_period: float | None = None,
    apply_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
    probe_batch_fn: Callable[[Sequence[np.ndarray]], list[dict[str, float]] | None] | None = None,
) -> tuple[float, np.ndarray | None, dict[str, Any]]:
    """Closed-loop batched V3.4 sigma refinement for uniform whole-crop mosaic smoothing.

    Sweeps up to 5 bounded candidate sigmas around ``base_radius_native`` in ONE batched
    ``probe_batch`` call when the baseline sigma over-smooths (``sharp_prep < 0.85`` or
    ``v34 exposure > 0.55``) or leaves excessive residual block energy. Always deterministic
    and bounded; returns ``(selected_radius, selected_rgb, report)``.
    """
    radius0 = round(float(max(0.5, base_radius_native)), 4)
    if apply_fn is None:
        from .automatic_preparation import apply_uniform_crop_gaussian

        def _default_apply(img: np.ndarray, sig: float) -> np.ndarray:
            return apply_uniform_crop_gaussian(img, sig, geometric_pattern=geometric_pattern)

        apply_fn = _default_apply

    if probe_batch_fn is None:
        probe_batch_fn = probe_batch

    base_out = apply_fn(base_crop, radius0)
    h, w = base_crop.shape[:2]
    if min(h, w) < 64:
        return radius0, base_out, {"refined": False, "reason": "small_crop_passthrough", "selected_radius_native": radius0}

    base_sharp = _sharpness_256(base_out)
    raw_probe_list = probe_batch_fn([base_crop, base_out])
    if not raw_probe_list or len(raw_probe_list) != 2:
        return radius0, base_out, {"refined": False, "reason": "probe_unavailable", "selected_radius_native": radius0}

    raw_probe, base_probe = raw_probe_list[0], raw_probe_list[1]
    base_exp = float(base_probe.get("exposure", 0.0))
    raw_exp = float(raw_probe.get("exposure", 0.0))
    base_codec = float(base_probe.get("codec_lr", 0.0))

    over_smoothed = (
        base_sharp < 0.50
        or (base_sharp < 0.70 and base_exp > 0.60 and base_exp > raw_exp + 0.20)
    )
    if not over_smoothed:
        return radius0, base_out, {
            "refined": False,
            "reason": "baseline_within_bounds",
            "selected_radius_native": radius0,
            "baseline_sharp_256": round(base_sharp, 3),
            "baseline_v34_exposure": base_exp,
            "baseline_v34_blur": float(base_probe.get("blur", 0.0)),
            "baseline_v34_codec_lr": base_codec,
        }

    # Construct 4 tighter bounded candidates plus baseline (5 total -> one batched forward).
    if detected_period is not None and detected_period >= 4.0:
        min_sigma = max(0.85, 0.24 * float(detected_period))
    else:
        min_sigma = max(0.85, 0.36 * radius0)
    min_sigma = min(min_sigma, radius0 * 0.90)

    multipliers = (0.45, 0.58, 0.72, 0.85)
    cand_radii: list[float] = []
    for mult in multipliers:
        r_c = round(float(max(min_sigma, radius0 * mult)), 4)
        if r_c < radius0 - 0.05 and all(abs(r_c - prev) >= 0.08 for prev in cand_radii):
            cand_radii.append(r_c)
    if not cand_radii:
        return radius0, base_out, {"refined": False, "reason": "no_tighter_candidates", "selected_radius_native": radius0}

    cand_imgs = [apply_fn(base_crop, r_c) for r_c in cand_radii]
    cand_probes = probe_batch_fn(cand_imgs)
    if not cand_probes or len(cand_probes) != len(cand_imgs):
        return radius0, base_out, {"refined": False, "reason": "batch_probe_failed", "selected_radius_native": radius0}

    all_entries: list[dict[str, Any]] = []
    for r_c, img_c, pr_c in [*zip(cand_radii, cand_imgs, cand_probes), (radius0, base_out, base_probe)]:
        sh_c = _sharpness_256(img_c)
        blk_c = _block_residual_score(img_c)
        exp_c = float(pr_c.get("exposure", 0.0))
        blur_c = float(pr_c.get("blur", 0.0))
        codec_c = float(pr_c.get("codec_lr", 0.0))
        # Penalty balances dissolving periodic blocks cleanly while preserving sharp_prep >= 0.5
        # and keeping V3.4 exposure drift minimal.
        block_pen = 1.6 * blk_c + 0.8 * max(0.0, codec_c - 0.35)
        smooth_pen = 2.5 * max(0.0, 0.65 - sh_c) / 0.65
        exp_pen = 2.2 * max(0.0, exp_c - max(0.45, raw_exp + 0.12))
        under_blur_pen = 1.4 * max(0.0, 0.75 - blur_c)
        cost = block_pen + smooth_pen + exp_pen + under_blur_pen
        all_entries.append({
            "radius_native": r_c,
            "img": img_c,
            "sharp_256": round(sh_c, 3),
            "block_residual": round(blk_c, 3),
            "v34_exposure": exp_c,
            "v34_blur": blur_c,
            "v34_codec_lr": codec_c,
            "cost": round(cost, 4),
        })

    # Prefer candidates that keep blocks dissolved (blk_c == 0) and avoid sharp collapse (< 0.50).
    best = min(all_entries, key=lambda e: (e["cost"], -e["sharp_256"]))
    best_r = float(best["radius_native"])
    refined = abs(best_r - radius0) >= 0.05
    summary_candidates = [
        {k: v for k, v in e.items() if k != "img"}
        for e in all_entries
    ]
    return best_r, best["img"], {
        "refined": refined,
        "reason": "closed_loop_v34_sigma_sweep" if refined else "baseline_optimal",
        "baseline_radius_native": radius0,
        "selected_radius_native": best_r,
        "selected_sharp_256": best["sharp_256"],
        "selected_v34_exposure": best["v34_exposure"],
        "selected_v34_blur": best["v34_blur"],
        "candidates": summary_candidates,
    }

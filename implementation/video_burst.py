"""Video / RTSP Lucky-Frame selection and 5-frame ECC Huber-trimmed temporal burst fusion."""
from __future__ import annotations

import math
import time
from typing import Sequence

import cv2
import numpy as np

from .fiqa_scorer import score_face_crops_fiqa

_ECC_ERRORS = (cv2.error, RuntimeError, ValueError)
_FIQA_ERRORS = (cv2.error, RuntimeError, ValueError, TypeError, OSError, ImportError)


def _ncc_core(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """Compute zero-mean normalized cross-correlation on the inner 76% frame core."""
    h, w = a_gray.shape[:2]
    y0, y1 = int(h * 0.12), int(h * 0.88)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    if (y1 - y0) < 4 or (x1 - x0) < 4:
        y0, y1, x0, x1 = 0, h, 0, w
    a = a_gray[y0:y1, x0:x1].astype(np.float64).ravel()
    b = b_gray[y0:y1, x0:x1].astype(np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if not math.isfinite(denom) or denom < 1e-8:
        return 0.0
    ncc = float(np.dot(a, b) / denom)
    return ncc if math.isfinite(ncc) else 0.0


def _valid_rgb_u8(arr: object) -> bool:
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim == 3
        and arr.shape[2] == 3
        and arr.dtype == np.uint8
        and min(arr.shape[:2]) >= 16
    )


def _laplacian_sharpness(rgb: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        val = float(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0.6), cv2.CV_32F).var())
        return val if math.isfinite(val) else 0.0
    except _ECC_ERRORS:
        return 0.0


def ecc_align_to_anchor(
    anchor_rgb: np.ndarray,
    frame_rgb: np.ndarray,
    *,
    max_ecc_side: int = 160,
    ncc_min: float = 0.88,
) -> tuple[np.ndarray | None, float, bool]:
    """Align `frame_rgb` onto `anchor_rgb` via ECC affine (with translation fallback) and NCC gate."""
    if not (_valid_rgb_u8(anchor_rgb) and _valid_rgb_u8(frame_rgb)):
        return None, 0.0, False
    ncc_thresh = float(ncc_min)
    if not math.isfinite(ncc_thresh):
        return None, 0.0, False
    ah, aw = anchor_rgb.shape[:2]
    try:
        if frame_rgb.shape[:2] != (ah, aw):
            frame_rgb = cv2.resize(frame_rgb, (aw, ah), interpolation=cv2.INTER_CUBIC)
        a_gray = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        a_ecc = cv2.GaussianBlur(a_gray, (3, 3), 0.8)
        f_ecc = cv2.GaussianBlur(f_gray, (3, 3), 0.8)
    except _ECC_ERRORS:
        return None, 0.0, False

    max_side = max(16, int(max_ecc_side))
    scale = min(1.0, float(max_side) / float(max(ah, aw)))
    sx = 1.0
    sy = 1.0
    if scale < 1.0:
        sw = max(16, int(round(aw * scale)))
        sh = max(16, int(round(ah * scale)))
        sx = float(sw) / float(aw)
        sy = float(sh) / float(ah)
        try:
            ag_s = cv2.resize(a_ecc, (sw, sh), interpolation=cv2.INTER_AREA)
            fg_s = cv2.resize(f_ecc, (sw, sh), interpolation=cv2.INTER_AREA)
        except _ECC_ERRORS:
            return None, 0.0, False
    else:
        ag_s, fg_s = a_ecc, f_ecc

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 45, 1e-4)
    try:
        _, warp = cv2.findTransformECC(ag_s, fg_s, warp, cv2.MOTION_AFFINE, criteria, None, 3)
    except _ECC_ERRORS:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ag_s, fg_s, warp, cv2.MOTION_TRANSLATION, criteria, None, 3)
        except _ECC_ERRORS:
            return None, 0.0, False

    if not isinstance(warp, np.ndarray) or warp.shape != (2, 3) or not bool(np.isfinite(warp).all()):
        return None, 0.0, False

    if scale < 1.0:
        warp = warp.copy()
        warp[0, 1] *= sy / sx
        warp[1, 0] *= sx / sy
        warp[0, 2] /= sx
        warp[1, 2] /= sy

    det = float(warp[0, 0] * warp[1, 1] - warp[0, 1] * warp[1, 0])
    max_shift = 0.35 * float(max(aw, ah))
    if not math.isfinite(det) or not (0.75 <= det <= 1.35) or abs(float(warp[0, 2])) > max_shift or abs(float(warp[1, 2])) > max_shift:
        return None, 0.0, False

    try:
        aligned = cv2.warpAffine(
            frame_rgb,
            warp,
            (aw, ah),
            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        aligned_gray = cv2.GaussianBlur(
            cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY).astype(np.float32),
            (3, 3),
            0.8,
        )
    except _ECC_ERRORS:
        return None, 0.0, False

    ncc = _ncc_core(a_ecc, aligned_gray)
    if not math.isfinite(ncc) or ncc < ncc_thresh:
        safe_ncc = round(float(ncc), 4) if math.isfinite(ncc) else 0.0
        return None, safe_ncc, False
    return aligned, round(float(ncc), 4), True


def huber_temporal_burst_fuse(
    anchor_rgb: np.ndarray,
    neighbor_rgbs: Sequence[np.ndarray],
    neighbor_weights: Sequence[float] | None = None,
    *,
    ncc_min: float = 0.88,
    huber_tau: float = 12.0,
) -> tuple[np.ndarray, dict]:
    """Align neighbors via ECC, reject NCC < ncc_min, and fuse via per-pixel Huber trimming."""
    t0 = time.perf_counter()
    if not _valid_rgb_u8(anchor_rgb):
        raise ValueError("Anchor frame must be a valid uint8 RGB image (>= 16px per side).")
    ah, aw = anchor_rgb.shape[:2]
    raw_tau = float(huber_tau)
    if not math.isfinite(raw_tau) or raw_tau <= 0.0:
        raise ValueError("huber_tau must be a finite positive number.")
    tau = max(1e-3, raw_tau)
    anchor_f = anchor_rgb.astype(np.float32)
    anchor_guide = cv2.GaussianBlur(anchor_f, (5, 5), 1.2)

    num = anchor_f.copy()
    den = np.ones((ah, aw, 1), dtype=np.float32)

    accepted = 0
    rejected = 0
    nccs: list[dict] = []
    neighbors = list(neighbor_rgbs)
    if neighbor_weights is None:
        weights: list[float] = [1.0] * len(neighbors)
    else:
        weights = list(neighbor_weights)
        if len(weights) < len(neighbors):
            weights.extend([1.0] * (len(neighbors) - len(weights)))

    for nbr, q_w in zip(neighbors, weights):
        aligned, ncc, ok = ecc_align_to_anchor(anchor_rgb, nbr, ncc_min=ncc_min)
        nccs.append({"ncc": ncc, "accepted": ok})
        if not ok or aligned is None:
            rejected += 1
            continue
        accepted += 1
        al_f = aligned.astype(np.float32)
        al_guide = cv2.GaussianBlur(al_f, (5, 5), 1.2)
        res = np.mean(np.abs(al_guide - anchor_guide), axis=2, keepdims=True)
        huber_w = np.where(res <= tau, 1.0, (tau / np.maximum(res, 1e-4)) ** 2).astype(np.float32)
        huber_w = np.where(res > 2.2 * tau, 0.0, huber_w)
        q_val = float(q_w) if (q_w is not None and math.isfinite(float(q_w))) else 1.0
        w_eff = float(max(0.35, min(1.0, q_val))) * huber_w
        num += w_eff * al_f
        den += w_eff

    fused = np.clip(np.rint(num / np.maximum(den, 1e-4)), 0, 255).astype(np.uint8)
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return fused, {
        "frames_total_with_anchor": 1 + accepted,
        "neighbors_accepted": accepted,
        "neighbors_rejected": rejected,
        "nccs": nccs,
        "fusion_ms": ms,
    }


def select_and_fuse_burst(
    frames_rgb: Sequence[np.ndarray],
    *,
    max_neighbors: int = 4,
    ncc_min: float = 0.88,
    huber_tau: float = 12.0,
) -> tuple[np.ndarray, dict]:
    """Select Lucky-Frame anchor (eDifFIQA-T with Laplacian fallback + consensus fallback) and fuse."""
    frames = list(frames_rgb)
    if not frames:
        raise ValueError("At least one frame is required for temporal burst fusion.")
    if not _valid_rgb_u8(frames[0]):
        raise ValueError("Primary burst frame must be a valid uint8 RGB image (>= 16px per side).")
    if len(frames) == 1:
        return frames[0].copy(), {
            "burst_used": False,
            "frames_input": 1,
            "anchor_index": 0,
            "scorer": "passthrough",
            "consensus_anchor_fallback": False,
            "frames_total_with_anchor": 1,
            "neighbors_accepted": 0,
            "neighbors_rejected": 0,
            "nccs": [],
            "fusion_ms": 0.0,
        }

    cap_neighbors = max(0, int(max_neighbors))
    valid_flags = [_valid_rgb_u8(f) for f in frames]
    try:
        fiqa_scores = score_face_crops_fiqa(frames)
    except _FIQA_ERRORS:
        fiqa_scores = None
    rank_scores: list[float] | None = None
    if isinstance(fiqa_scores, (list, tuple)) and len(fiqa_scores) == len(frames):
        parsed: list[float] = []
        any_valid_fiqa = False
        for ok_frame, s in zip(valid_flags, fiqa_scores):
            if not ok_frame or s is None:
                parsed.append(float("-inf"))
                continue
            try:
                val = float(s)
            except (TypeError, ValueError):
                parsed.append(float("-inf"))
                continue
            if math.isfinite(val):
                parsed.append(val)
                any_valid_fiqa = True
            else:
                parsed.append(float("-inf"))
        if any_valid_fiqa:
            rank_scores = parsed
    if rank_scores is not None:
        scorer = "ediffiqa_tiny"
    else:
        scorer = "laplacian_fallback"
        rank_scores = [_laplacian_sharpness(f) if ok_frame else float("-inf") for ok_frame, f in zip(valid_flags, frames)]

    order = sorted(range(len(frames)), key=lambda idx: (-rank_scores[idx], idx))
    finite_scores = [s for s in rank_scores if math.isfinite(s)]
    lo_s = min(finite_scores) if finite_scores else 0.0
    hi_s = max(finite_scores) if finite_scores else 0.0
    span_s = hi_s - lo_s

    def _weights_for(idxs: list[int]) -> list[float]:
        if span_s <= 1e-8:
            return [1.0] * len(idxs)
        out = []
        for idx in idxs:
            sc = rank_scores[idx]
            norm = (sc - lo_s) / span_s if math.isfinite(sc) else 0.0
            out.append(float(max(0.35, min(1.0, 0.55 + 0.45 * norm))))
        return out

    anchor_idx = order[0]
    nbr_idxs = [idx for idx in order[1 : 1 + cap_neighbors] if _valid_rgb_u8(frames[idx])]
    fused, fmeta = huber_temporal_burst_fuse(
        frames[anchor_idx],
        [frames[idx] for idx in nbr_idxs],
        _weights_for(nbr_idxs),
        ncc_min=ncc_min,
        huber_tau=huber_tau,
    )
    consensus_fallback = False
    if fmeta["neighbors_accepted"] == 0 and len(order) >= 2 and _valid_rgb_u8(frames[order[1]]):
        alt_idx = order[1]
        alt_nbrs = [idx for idx in order if idx != alt_idx and _valid_rgb_u8(frames[idx])][:cap_neighbors]
        alt_fused, alt_meta = huber_temporal_burst_fuse(
            frames[alt_idx],
            [frames[idx] for idx in alt_nbrs],
            _weights_for(alt_nbrs),
            ncc_min=ncc_min,
            huber_tau=huber_tau,
        )
        if alt_meta["neighbors_accepted"] > 0:
            anchor_idx = alt_idx
            fused = alt_fused
            fmeta = alt_meta
            consensus_fallback = True

    meta = {
        **fmeta,
        "burst_used": bool(fmeta["neighbors_accepted"] > 0),
        "frames_input": len(frames),
        "anchor_index": int(anchor_idx),
        "scorer": scorer,
        "consensus_anchor_fallback": consensus_fallback,
    }
    return fused, meta

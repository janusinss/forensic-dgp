"""Portable frozen R2 completion on an already prepared/restored RGB256 canvas.

Importing this module loads no Torch model, detector, restoration or research
service. Completion preserves the supplied canvas outside the reviewed mask.
If that canvas was restored, the preservation reference is the restored canvas,
not the earlier upload. Original/prepared/restored evidence belongs to the app.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from . import completion_assets as assets

ROOT = Path(__file__).resolve().parents[1]
LOGICAL_SHA256 = "300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0"
ORIGINAL_SHA256 = "b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
PROTOCOL_SHA256 = "463c1977e9cd74f0a4fd0240df7a784f48b613043ef667714f1ef94adb4cd60a"
PARENT_R1_SHA256 = "7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0"
ARCHITECTURE_SOURCES = {
    "third_party.codeformer_completion.codeformer_arch": ("codeformer_arch.py", "29c40cee39414c1117d1ae55fc15d6cdd42aed09e03019522c7f55771b1b84af"),
    "third_party.codeformer_completion.vqgan_arch": ("vqgan_arch.py", "01d7d9a796489716439cbd59fbd9e305278d762d97e6582aafd54b9279e73265"),
    "third_party.codeformer_completion.registry": ("registry.py", "21c23bd3bf727eb2decd99035120fc47316db92486d73116cca18e9fa8b6c854"),
}


class CompletionRefused(ValueError):
    """Insufficient reviewed facial support; ask for a clearer/less-covered crop."""


def need(ok, message):
    if not ok:
        raise ValueError(message)


def model_contract():
    entry = assets.load_manifest()["assets"]["completion_r2"]
    return {"model_id": "completion_r2", "model_label": "Trained CodeFormer completion R2",
            "backend": "codeformer-inpainting-finetune-r2", "weights_sha256": entry["sha256"],
            "original_checkpoint_sha256": ORIGINAL_SHA256, "logical_state_sha256": LOGICAL_SHA256,
            "parent_checkpoint_sha256": PARENT_R1_SHA256,
            "training_protocol_sha256": PROTOCOL_SHA256, "source_revision": entry["source_revision"],
            "device": "cpu", "precision": "float32", "internal_resolution": 512,
            "delivered_resolution": 256, "input_policy": "visible-normalized-resize-v1",
            "compositing_policy": "mask-only-v1", "additional_mask_expansion": 0,
            "model_qualified": False, "hidden_identity_verified": False,
            "claim": "One plausible estimate in the reviewed removal area; hidden identity is not verified."}


def validate_arrays(rgb, mask, prepared_bounds=None):
    need(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (256, 256, 3),
         "Expected the exact prepared/restored RGB256 uint8 canvas.")
    need(isinstance(mask, np.ndarray) and mask.dtype in (np.dtype("uint8"), np.dtype("bool"))
         and mask.shape == (256, 256), "Expected a matching binary256 removal mask.")
    values = set(np.unique(mask).tolist())
    need(values.issubset({0, 1}) or values.issubset({0, 255}), "Use only binary0/1 or0/255 removal values.")
    removal = (mask != 0).astype(np.uint8)
    bounds = [0, 0, 256, 256] if prepared_bounds is None else prepared_bounds
    need(isinstance(bounds, (list, tuple)) and len(bounds) == 4 and all(type(v) is int for v in bounds),
         "prepared_bounds must contain four integer content coordinates.")
    left, top, right, bottom = bounds
    need(0 <= left < right <= 256 and 0 <= top < bottom <= 256, "Invalid prepared content bounds.")
    unavailable = np.ones((256, 256), np.uint8)
    unavailable[top:bottom, left:right] = 0
    observed = unavailable == 0
    need(not removal[unavailable != 0].any(), "Keep prepared padding outside the removal area.")
    unavailable = np.maximum(unavailable, removal)
    review = visibility_check256(unavailable, observed=observed)
    if review["rejected"]:
        raise CompletionRefused("Too little visible facial structure remains. Use a clearer or less-covered face crop.")
    return removal, list(bounds), review


def visibility_check256(unavailable, observed=None):
    """Retained frontal-crop geometry heuristic; not a face/quality detector."""
    need(isinstance(unavailable, np.ndarray) and unavailable.shape == (256, 256), "Expected256 visibility support.")
    m = unavailable != 0
    yy, xx = np.mgrid[:256, :256]
    face = ((xx - 127.5) / 83) ** 2 + ((yy - 137) / 104) ** 2 <= 1
    bands = {"left_eye": (60, 84, 119, 133), "right_eye": (137, 84, 196, 133),
             "nose": (108, 119, 149, 166), "mouth": (82, 165, 174, 208)}
    fractions = {name: float(m[y1:y2, x1:x2].mean()) for name, (x1, y1, x2, y2) in bands.items()}
    if observed is not None:
        need(isinstance(observed, np.ndarray) and observed.shape == (256, 256) and observed.dtype == np.bool_,
             "Expected boolean256 observed support.")
        obs_face = face & observed
        obs_face_fraction = float(obs_face.sum()) / float(max(1, int(face.sum())))
        covered = float(m[obs_face].mean()) if obs_face.any() else 1.0
        insufficient_observed_face = obs_face_fraction < 0.45
    else:
        obs_face_fraction = 1.0
        covered = float(m[face].mean())
        insufficient_observed_face = False
    all_features = (fractions["left_eye"] >= .85 and fractions["right_eye"] >= .85
                    and fractions["nose"] >= .9 and fractions["mouth"] >= .9 and covered >= .65)
    return {"rejected": bool(insufficient_observed_face or covered >= .82 or all_features or m.mean() >= .85),
            "face_covered_fraction": covered, "observed_face_fraction": obs_face_fraction,
            "feature_covered_fractions": fractions,
            "padding_counts_as_unavailable": True,
            "scope": "Approximate frontal/mild-turn crop geometry conditional on the reviewed mask; not a detector or proof of adequate detail."}


_FFHQ_5PT_256 = np.array(
    [
        [96.49069, 119.97354],
        [159.45139, 120.09680],
        [128.31708, 157.00967],
        [100.63058, 185.70522],
        [156.54453, 185.57559],
    ],
    dtype=np.float32,
)
_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
_YUNET_BYTES_CACHE: bytes | None = None


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Compute 2x3 similarity transform M mapping src -> dst via Umeyama least-squares."""
    src_f = np.asarray(src, dtype=np.float64)
    dst_f = np.asarray(dst, dtype=np.float64)
    if src_f.ndim != 2 or dst_f.ndim != 2 or src_f.shape != dst_f.shape or src_f.shape[1] != 2 or src_f.shape[0] < 2:
        return None
    src_mean = src_f.mean(axis=0)
    dst_mean = dst_f.mean(axis=0)
    src_c = src_f - src_mean
    dst_c = dst_f - dst_mean
    src_var = float((src_c ** 2).sum() / src_f.shape[0])
    if src_var < 1e-6 or not np.isfinite(src_var):
        return None
    cov = (dst_c.T @ src_c) / float(src_f.shape[0])
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2, dtype=np.float64)
    if np.linalg.det(cov) < 0.0:
        d[1] = -1.0
    rot = u @ np.diag(d) @ vt
    scale = float((s * d).sum() / src_var)
    if not np.isfinite(scale) or scale <= 1e-4:
        return None
    t = dst_mean - scale * (rot @ src_mean)
    mat = np.empty((2, 3), dtype=np.float32)
    mat[:, :2] = (scale * rot).astype(np.float32)
    mat[:, 2] = t.astype(np.float32)
    return mat if np.isfinite(mat).all() else None


def _load_yunet_bytes() -> bytes | None:
    """Load and SHA-256 verify the bundled 0.22 MB YuNet face detector once."""
    global _YUNET_BYTES_CACHE
    if _YUNET_BYTES_CACHE is not None:
        return _YUNET_BYTES_CACHE
    path = Path(__file__).with_name("assets") / "face_detection_yunet_2023mar.onnx"
    try:
        verified = assets.ordinary_file(path)
        raw = verified.read_bytes()
    except (OSError, assets.CompletionAssetError):
        return None
    if hashlib.sha256(raw).hexdigest() != _YUNET_SHA256:
        return None
    _YUNET_BYTES_CACHE = raw
    return raw


def _detect_yunet_landmarks_256(rgb: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Detect primary face 5-point landmarks on RGB256 using bundled YuNet."""
    raw = _load_yunet_bytes()
    if raw is None:
        return None
    import cv2
    try:
        det = cv2.FaceDetectorYN.create("onnx", raw, b"", (256, 256), 0.35, 0.3, 50)
        _, faces = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except (cv2.error, OSError, RuntimeError, ValueError):
        return None
    if faces is None or len(faces) == 0:
        return None
    # Pick best face closest to canvas center with high score and reasonable size.
    best_pts = None
    best_score = -1.0
    for row in faces:
        bx, by, bw, bh = (float(row[k]) for k in range(4))
        score = float(row[14]) if len(row) >= 15 else 0.5
        if bw < 28.0 or bh < 28.0 or score < 0.45:
            continue
        pts = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        if not np.isfinite(pts).all():
            continue
        re, le, nose, rm, lm = pts
        d_eye = float(np.hypot(le[0] - re[0], le[1] - re[1]))
        emx, emy = 0.5 * (re[0] + le[0]), 0.5 * (re[1] + le[1])
        mmx, mmy = 0.5 * (rm[0] + lm[0]), 0.5 * (rm[1] + lm[1])
        d_em = float(np.hypot(mmx - emx, mmy - emy))
        if d_eye < 16.0 or d_em < 14.0 or d_em < 0.32 * d_eye or d_eye < 0.25 * d_em:
            continue
        dist_center = float(np.hypot((bx + 0.5 * bw) - 127.5, (by + 0.5 * bh) - 127.5))
        rank = score * (bw * bh) ** 0.35 / (1.0 + dist_center / 128.0)
        if rank > best_score:
            best_score = rank
            best_pts = (pts, score)
    return best_pts


def _estimate_ffhq_alignment(rgb: np.ndarray, removal: np.ndarray, observed: np.ndarray) -> dict:
    """Estimate safe Umeyama Sim(2) alignment to canonical FFHQ 5-point geometry.

    Prefers InsightFace 2d106det 5-point landmarks via lazy import, falling back
    seamlessly to bundled YuNet. Bypasses cleanly on already-canonical faces
    (|roll| < 5 deg, offset <= 12 px, |scale-1| <= 12%) or unreliable landmarks.
    """
    bypass = {"applied": False, "roll_deg": 0.0, "scale": 1.0, "offset_px": 0.0, "reproj_err_px": 0.0, "anchors_used": 0}
    pts = None
    score = 0.0
    try:
        from .automatic_covering import _run_landmarks_106, landmarks_5pt_from_106

        pts106 = _run_landmarks_106(rgb)
        if pts106 is not None:
            pts5 = landmarks_5pt_from_106(pts106)
            if pts5 is not None:
                pts = pts5
                score = 0.85
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        pts = None

    if pts is None:
        det = _detect_yunet_landmarks_256(rgb)
        if det is None:
            return bypass
        pts, score = det

    import cv2
    mask_pad = cv2.dilate((removal != 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    vis_idx: list[int] = []
    for idx in range(5):
        px_i, py_i = int(round(float(pts[idx, 0]))), int(round(float(pts[idx, 1])))
        if 0 <= px_i < 256 and 0 <= py_i < 256 and observed[py_i, px_i] and not mask_pad[py_i, px_i]:
            vis_idx.append(idx)

    # Prefer >= 3 unoccluded landmarks; fall back to unoccluded eye pair (0,1) or nose+mouth (2,3,4),
    # or all 5 high-confidence YuNet points when the detector strongly tracks the face geometry.
    if len(vis_idx) >= 3:
        use_idx = vis_idx
    elif 0 in vis_idx and 1 in vis_idx:
        use_idx = [0, 1]
    elif score >= 0.72:
        use_idx = [0, 1, 2, 3, 4]
    else:
        return bypass

    src_pts = pts[use_idx]
    dst_pts = _FFHQ_5PT_256[use_idx]
    mat = _umeyama_similarity(src_pts, dst_pts)
    if mat is None:
        return bypass

    scale = float(np.hypot(float(mat[0, 0]), float(mat[0, 1])))
    roll_deg = float(np.degrees(np.arctan2(float(mat[1, 0]), float(mat[0, 0]))))
    # Compute landmark-centroid translation shift and landmark reprojection error.
    ones = np.ones((len(use_idx), 1), dtype=np.float32)
    proj = (np.hstack([src_pts, ones]) @ mat.T).astype(np.float32)
    reproj_err = float(np.linalg.norm(proj - dst_pts, axis=1).mean())
    src_centroid = src_pts.mean(axis=0)
    dst_centroid = dst_pts.mean(axis=0)
    offset_px = float(np.hypot(float(src_centroid[0] - dst_centroid[0]), float(src_centroid[1] - dst_centroid[1])))

    info = {
        "applied": False,
        "roll_deg": round(roll_deg, 2),
        "scale": round(scale, 4),
        "offset_px": round(offset_px, 2),
        "reproj_err_px": round(reproj_err, 2),
        "anchors_used": int(len(use_idx)),
    }
    # Safety bounds: reject extreme angles/scales or high reprojection residual.
    if abs(roll_deg) > 35.0 or not (0.70 <= scale <= 1.45) or (len(use_idx) >= 3 and reproj_err >= 6.0):
        return info
    # Canonical bypass: zero warp on already-aligned faces.
    if abs(roll_deg) < 5.0 and offset_px <= 12.0 and abs(scale - 1.0) <= 0.12:
        return info

    info["applied"] = True
    info["matrix"] = mat
    return info


def _solve_harmonic_lab_field(mask_roi: np.ndarray, boundary_mask: np.ndarray, boundary_delta: np.ndarray, screening: float = 0.0025) -> np.ndarray:
    """Solve 2D discrete screened harmonic interpolation inside mask_roi for (L, a, b) deltas."""
    h, w = mask_roi.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    if not mask_roi.any():
        return out
    out[boundary_mask] = boundary_delta[boundary_mask]
    interior = mask_roi & (~boundary_mask)
    ys, xs = np.nonzero(interior)
    n_int = int(ys.size)
    if n_int == 0:
        return out

    bnd_mean = boundary_delta[boundary_mask].mean(axis=0).astype(np.float64) if boundary_mask.any() else np.zeros(3, dtype=np.float64)
    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(n_int, dtype=np.int32)

    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        rows = [np.arange(n_int, dtype=np.int32)]
        cols = [np.arange(n_int, dtype=np.int32)]
        diag = np.full(n_int, float(screening), dtype=np.float64)
        rhs = np.broadcast_to(bnd_mean * float(screening), (n_int, 3)).copy()
        data_parts: list[np.ndarray] = []

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = ys + dy
            nx = xs + dx
            in_bounds = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            valid_i = np.nonzero(in_bounds)[0]
            if valid_i.size == 0:
                continue
            ny_v = ny[valid_i]
            nx_v = nx[valid_i]
            is_int = interior[ny_v, nx_v]
            is_bnd = boundary_mask[ny_v, nx_v]

            # Active neighbors (interior or boundary) contribute to the discrete Laplacian stencil.
            active = is_int | is_bnd
            diag[valid_i[active]] += 1.0

            if is_int.any():
                ii = valid_i[is_int]
                jj = idx_map[ny_v[is_int], nx_v[is_int]]
                rows.append(ii)
                cols.append(jj)
                data_parts.append(np.full(ii.size, -1.0, dtype=np.float64))

            if is_bnd.any():
                bi = valid_i[is_bnd]
                rhs[bi] += boundary_delta[ny_v[is_bnd], nx_v[is_bnd]]

        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_data = np.concatenate([diag, *data_parts]) if data_parts else diag
        mat = sparse.coo_matrix((all_data, (all_rows, all_cols)), shape=(n_int, n_int)).tocsc()
        sol = spsolve(mat, rhs)
        if sol.ndim == 1:
            sol = sol[:, None]
        if np.isfinite(sol).all():
            out[ys, xs] = sol.astype(np.float32)
            return out
    except (ImportError, ValueError, RuntimeError):
        pass

    # Vectorized Jacobi fallback if scipy is unavailable.
    field = np.broadcast_to(bnd_mean.astype(np.float32), (h, w, 3)).copy()
    field[boundary_mask] = boundary_delta[boundary_mask]
    active_f = mask_roi.astype(np.float32)[..., None]
    for _ in range(48):
        acc = np.zeros_like(field)
        cnt = np.zeros((h, w, 1), dtype=np.float32)
        acc[1:] += field[:-1] * active_f[:-1]; cnt[1:] += active_f[:-1]
        acc[:-1] += field[1:] * active_f[1:]; cnt[:-1] += active_f[1:]
        acc[:, 1:] += field[:, :-1] * active_f[:, :-1]; cnt[:, 1:] += active_f[:, :-1]
        acc[:, :-1] += field[:, 1:] * active_f[:, 1:]; cnt[:, :-1] += active_f[:, 1:]
        upd = (acc + np.float32(screening) * bnd_mean.astype(np.float32)) / np.maximum(cnt + np.float32(screening), np.float32(1.0))
        field[interior] = upd[interior]
    out[interior] = field[interior]
    return out


def _apply_monochrome_guard(orig_rgb, blended_f, removal, observed):
    """Attenuate hallucinated chroma inside mask on B&W / near-grayscale portraits (0 MB, <0.5 ms).

    Operates on float32 [0, 1] before encode_output so raw.npy and completion.png remain
    100% replay-consistent and outside-mask pixels stay bit-exact.
    """
    mask_bool = removal != 0
    if not mask_bool.any():
        return blended_f
    vis = observed & (~mask_bool)
    if int(vis.sum()) < 200:
        return blended_f
    import cv2

    orig_lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    vis_chroma = np.hypot(
        orig_lab[..., 1][vis] - np.float32(128.0),
        orig_lab[..., 2][vis] - np.float32(128.0),
    )
    if float(np.median(vis_chroma)) >= 4.0 or float(np.percentile(vis_chroma, 85.0)) >= 7.0:
        return blended_f

    a_med = float(np.median(orig_lab[..., 1][vis]))
    b_med = float(np.median(orig_lab[..., 2][vis]))
    orig_f = orig_rgb.astype(np.float32) / np.float32(255.0)
    out_u8 = np.rint(np.clip(blended_f, 0.0, 1.0) * np.float32(255.0)).clip(0, 255).astype(np.uint8)
    out_lab = cv2.cvtColor(out_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    out_lab[..., 1][mask_bool] = np.float32(a_med) + np.float32(0.15) * (out_lab[..., 1][mask_bool] - np.float32(a_med))
    out_lab[..., 2][mask_bool] = np.float32(b_med) + np.float32(0.15) * (out_lab[..., 2][mask_bool] - np.float32(b_med))
    res_u8 = cv2.cvtColor(np.rint(out_lab).clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    res_f = res_u8.astype(np.float32) / np.float32(255.0)
    res_f[~mask_bool] = orig_f[~mask_bool]
    return res_f


def _blend_inside_mask(rgb, generated, removal, observed):
    """Strictly inside-mask harmonic Lab illumination transfer, 4-level Laplacian pyramid blending,
    and local skin grain/PSF matching.

    Zero pixels outside removal == 1 are modified; output is float32 in [0, 1].
    """
    mask_bool = removal != 0
    if not mask_bool.any():
        return generated
    import cv2

    mask_u8 = mask_bool.astype(np.uint8)
    orig_f = rgb.astype(np.float32) / np.float32(255.0)
    gen_f = np.clip(generated.astype(np.float32, copy=False), 0.0, 1.0)

    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    erode_3 = np.ones((3, 3), np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32)
    orig_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_u8 = np.rint(gen_f * np.float32(255.0)).clip(0, 255).astype(np.uint8)
    gen_lab = cv2.cvtColor(gen_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 1. Per-component spatially-varying screened-harmonic Lab illumination transfer.
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    shift_field = np.zeros((256, 256, 3), dtype=np.float32)
    out_med_map = np.zeros((256, 256, 3), dtype=np.float32)
    has_out_med = np.zeros((256, 256), dtype=bool)
    all_clean_outer = np.zeros((256, 256), dtype=bool)

    for idx in range(1, n_comp):
        x, y, w, h, area = (int(stats[idx, k]) for k in range(5))
        if area < 4:
            continue
        pad = 12
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(256, x + w + pad), min(256, y + h + pad)

        comp_roi = labels[y0:y1, x0:x1] == idx
        comp_u8 = comp_roi.astype(np.uint8)
        obs_roi = observed[y0:y1, x0:x1]
        mask_all_roi = mask_bool[y0:y1, x0:x1]

        outer_roi = (cv2.dilate(comp_u8, ring_kernel) > 0) & (~mask_all_roi) & obs_roi
        inner_roi = comp_roi & (cv2.erode(comp_u8, ring_kernel) == 0)
        if int(outer_roi.sum()) < 12 or int(inner_roi.sum()) < 4:
            continue

        orig_lab_roi = orig_lab[y0:y1, x0:x1]
        gen_lab_roi = gen_lab[y0:y1, x0:x1]

        # Filter extreme hair/shadow & non-skin chroma outliers without clipping natural side-lighting gradients.
        l_out = orig_lab_roi[..., 0][outer_roi]
        p05 = float(np.percentile(l_out, 5.0))
        min_l = 25.0 if p05 < 45.0 else 0.0
        ab_med = np.array([np.median(orig_lab_roi[..., 1][outer_roi]), np.median(orig_lab_roi[..., 2][outer_roi])], dtype=np.float32)
        ab_dist = np.hypot(orig_lab_roi[..., 1] - ab_med[0], orig_lab_roi[..., 2] - ab_med[1])
        ab_mad = float(np.median(ab_dist[outer_roi]))
        clean_outer = outer_roi & (orig_lab_roi[..., 0] >= min_l) & (ab_dist <= max(16.0, 2.8 * ab_mad + 6.0))
        if int(clean_outer.sum()) < 16:
            clean_outer = outer_roi
        med_lab = np.array([np.median(orig_lab_roi[..., c][clean_outer]) for c in range(3)], dtype=np.float32)

        all_clean_outer[y0:y1, x0:x1] |= clean_outer
        out_med_map[y0:y1, x0:x1][comp_roi] = med_lab
        has_out_med[y0:y1, x0:x1][comp_roi] = True

        # Estimate local outer skin Lab and local inner generated Lab separately (since gen == orig outside mask!).
        w_out = clean_outer.astype(np.float32)
        w_in = inner_roi.astype(np.float32)
        w_out_blur = cv2.GaussianBlur(w_out, (0, 0), sigmaX=4.5, sigmaY=4.5)
        w_in_blur = cv2.GaussianBlur(w_in, (0, 0), sigmaX=4.5, sigmaY=4.5)
        in_med_lab = np.array([np.median(gen_lab_roi[..., c][inner_roi]) for c in range(3)], dtype=np.float32)
        bnd_delta = np.zeros_like(orig_lab_roi)
        for c in range(3):
            out_num = cv2.GaussianBlur(orig_lab_roi[..., c] * w_out, (0, 0), sigmaX=4.5, sigmaY=4.5)
            in_num = cv2.GaussianBlur(gen_lab_roi[..., c] * w_in, (0, 0), sigmaX=4.5, sigmaY=4.5)
            out_local = np.where(w_out_blur > 1e-4, out_num / np.maximum(w_out_blur, np.float32(1e-4)), med_lab[c])
            in_local = np.where(w_in_blur > 1e-4, in_num / np.maximum(w_in_blur, np.float32(1e-4)), in_med_lab[c])
            bnd_delta[..., c] = out_local - in_local

        # Smooth tanh safety bounds (+-28 L, +-14 a/b) handle harsh CCTV casts without overshoot.
        bnd_delta[..., 0] = np.float32(28.0) * np.tanh(bnd_delta[..., 0] / np.float32(28.0))
        bnd_delta[..., 1] = np.float32(14.0) * np.tanh(bnd_delta[..., 1] / np.float32(14.0))
        bnd_delta[..., 2] = np.float32(14.0) * np.tanh(bnd_delta[..., 2] / np.float32(14.0))

        inner_edge = comp_roi & (~(cv2.erode(comp_u8, erode_3) > 0))
        solved = _solve_harmonic_lab_field(comp_roi, inner_edge, bnd_delta, screening=0.0025)
        shift_field[y0:y1, x0:x1][comp_roi] = solved[comp_roi]

    shifted_lab = gen_lab.copy()
    shifted_lab[..., 0] = np.clip(shifted_lab[..., 0] + shift_field[..., 0], 0.0, 255.0)
    shifted_lab[..., 1] = np.clip(shifted_lab[..., 1] + shift_field[..., 1], 0.0, 255.0)
    shifted_lab[..., 2] = np.clip(shifted_lab[..., 2] + shift_field[..., 2], 0.0, 255.0)
    shifted_u8 = np.rint(shifted_lab).clip(0, 255).astype(np.uint8)
    gen_adj_f = cv2.cvtColor(shifted_u8, cv2.COLOR_LAB2RGB).astype(np.float32) / np.float32(255.0)

    # 2. Covering-bleed guard: suppress edge feathering where original edge pixels are covering color (de > 20).
    bleed_guard = np.zeros((256, 256), dtype=bool)
    edge_band = mask_bool & (dist <= 3.0) & has_out_med
    if edge_band.any():
        de_orig = np.linalg.norm(orig_lab - out_med_map, axis=2)
        bleed_guard = edge_band & (de_orig > 20.0)

    # 3. 4-level Multi-Band Laplacian Pyramid blending strictly inside removal_mask.
    # Build lp_orig from an occluder-free canvas extrapolated from clean outside pixels so
    # bright blue/black occluder RGB inside mask_bool can NEVER leak into coarse pyramid levels L1/L2/G3!
    orig_safe = cv2.inpaint(rgb, mask_u8, 5, cv2.INPAINT_TELEA).astype(np.float32) / np.float32(255.0)
    orig_safe[~mask_bool] = orig_f[~mask_bool]

    alpha_high = np.clip((dist - np.float32(0.5)) / np.float32(1.5), 0.0, 1.0)
    alpha_mid = np.clip((dist - np.float32(0.5)) / np.float32(5.0), 0.0, 1.0)
    alpha_low = np.clip((dist - np.float32(0.5)) / np.float32(11.5), 0.0, 1.0)
    alpha_high = np.where(bleed_guard, np.float32(1.0), alpha_high)
    alpha_mid = np.where(bleed_guard & (dist <= 2.5), np.float32(1.0), alpha_mid)
    alpha_low = np.where(bleed_guard & (dist <= 2.5), np.float32(1.0), alpha_low)

    def _build_laplacian_pyr(img_f: np.ndarray, levels: int = 4) -> list[np.ndarray]:
        gp = [img_f]
        for _ in range(levels - 1):
            gp.append(cv2.pyrDown(gp[-1]))
        lp = []
        for lv in range(levels - 1):
            up = cv2.pyrUp(gp[lv + 1], dstsize=(gp[lv].shape[1], gp[lv].shape[0]))
            lp.append(gp[lv] - up)
        lp.append(gp[-1])
        return lp

    lp_orig = _build_laplacian_pyr(orig_safe, levels=4)
    lp_gen = _build_laplacian_pyr(gen_adj_f, levels=4)
    # Build per-level blend masks: tight on L0, mid on L1, wide on L2 & G3.
    a0 = alpha_high
    a1 = cv2.pyrDown(alpha_mid)
    a2 = cv2.pyrDown(cv2.pyrDown(alpha_low))
    a3 = cv2.pyrDown(a2)
    level_alphas = [a0, a1, a2, a3]

    blended_pyr = []
    for lv in range(4):
        al3 = np.clip(level_alphas[lv], 0.0, 1.0)[..., None]
        blended_pyr.append(lp_orig[lv] * (np.float32(1.0) - al3) + lp_gen[lv] * al3)

    blended = blended_pyr[-1]
    for lv in range(2, -1, -1):
        blended = cv2.pyrUp(blended, dstsize=(blended_pyr[lv].shape[1], blended_pyr[lv].shape[0])) + blended_pyr[lv]
    blended = np.clip(blended, 0.0, 1.0).astype(np.float32)

    # 4. Local skin PSF softness & zero-mean luminance-modulated sensor grain matching inside mask.
    interior_tex = mask_bool & (dist >= 2.0)
    if int(all_clean_outer.sum()) >= 64 and int(interior_tex.sum()) >= 64:
        orig_gray = cv2.cvtColor(orig_f, cv2.COLOR_RGB2GRAY)
        blend_gray = cv2.cvtColor(blended, cv2.COLOR_RGB2GRAY)
        lap_out = cv2.Laplacian(orig_gray, cv2.CV_32F)[all_clean_outer]
        lap_in = cv2.Laplacian(blend_gray, cv2.CV_32F)[interior_tex]
        sigma_out = float(np.median(np.abs(lap_out - np.median(lap_out))) * 1.4826 / np.sqrt(20.0))
        sigma_in = float(np.median(np.abs(lap_in - np.median(lap_in))) * 1.4826 / np.sqrt(20.0))

        sob_out = np.hypot(cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0), cv2.Sobel(orig_gray, cv2.CV_32F, 0, 1))[all_clean_outer]
        sob_in = np.hypot(cv2.Sobel(blend_gray, cv2.CV_32F, 1, 0), cv2.Sobel(blend_gray, cv2.CV_32F, 0, 1))[interior_tex]
        s_out = float(np.percentile(sob_out, 85.0))
        s_in = float(np.percentile(sob_in, 85.0))

        taper = np.clip((dist - np.float32(1.0)) / np.float32(3.0), 0.0, 1.0)
        if s_out > 0.006 and s_in > 1.35 * s_out:
            ratio_sq = max(1.0, (s_in / max(s_out, 1e-4)) ** 2)
            sigma_psf = min(1.2, 0.45 * float(np.sqrt(ratio_sq - 1.0)))
            if sigma_psf >= 0.30:
                soft = cv2.GaussianBlur(blended, (0, 0), sigmaX=sigma_psf, sigmaY=sigma_psf)
                t3 = taper[..., None]
                blended = blended * (np.float32(1.0) - t3) + soft * t3

        if sigma_out >= (1.5 / 255.0) and sigma_out > 1.25 * sigma_in:
            sigma_add = min(0.035, float(np.sqrt(max(0.0, sigma_out ** 2 - sigma_in ** 2))))
            seed_bytes = hashlib.sha256(rgb[::8, ::8].tobytes()).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
            raw_noise = rng.standard_normal((256, 256, 3), dtype=np.float32)
            # Light spatial correlation matches real camera/JPEG skin micro-grain.
            corr_noise = cv2.GaussianBlur(raw_noise, (3, 3), 0.45)
            corr_noise -= corr_noise.mean(axis=(0, 1), keepdims=True)
            std_n = float(corr_noise.std())
            if std_n > 1e-6:
                corr_noise /= np.float32(std_n)
            luma = np.clip(blend_gray, 0.0, 1.0)
            luma_mod = (np.float32(0.65) + np.float32(0.35) * luma) * taper
            grain = corr_noise * (np.float32(sigma_add) * luma_mod[..., None])
            blended = np.where(mask_bool[..., None], blended + grain, blended)

    blended = np.clip(blended, 0.0, 1.0).astype(np.float32)
    blended[~mask_bool] = orig_f[~mask_bool]
    blended = _apply_monochrome_guard(rgb, blended, removal, observed)
    return blended


def encode_output(rgb, removal, completed):
    need(isinstance(completed, np.ndarray) and completed.dtype == np.float32
         and completed.shape == rgb.shape and np.isfinite(completed).all(), "Invalid completion float output.")
    need(float(completed.min()) >= 0 and float(completed.max()) <= 1, "Completion output must lie in[0,1].")
    result = np.rint(completed * np.float32(255)).clip(0, 255).astype(np.uint8)
    result[removal == 0] = rgb[removal == 0]
    return result


def architecture_file_sha256(path):
    raw = assets.ordinary_file(path).read_bytes().replace(b"\r\n", b"\n")
    need(b"\r" not in raw, "Bare CR in completion source refused.")
    return hashlib.sha256(raw).hexdigest()


def verify_architecture_sources():
    for name, (filename, expected) in ARCHITECTURE_SOURCES.items():
        path = ROOT / "third_party" / "codeformer_completion" / filename
        need(architecture_file_sha256(path) == expected, "Completion architecture source changed: " + filename)
        module = sys.modules.get(name)
        need(module is None or Path(module.__file__).resolve() == path.resolve(), "Noncanonical completion architecture loaded.")


@dataclass(frozen=True)
class HookConfig:
    name: str
    w_in_32: float = 1.0
    w_in_64: float = 1.0
    w_in_128: float = 1.0
    adain_alpha: float = 0.0
    adain_mode: str = "matched_vis"
    feather_ksize: int = 3
    dilate_infer_px: int = 0
    force_no_align: bool = False


@contextmanager
def attach_r2_hooks(net, mask512, cfg: HookConfig):
    """Attach non-invasive forward hooks for Masked-AdaIN and Spatial-w SFT attenuation.

    Safely degrades when mock/toy networks lack encoder/generator/fuse_convs_dict and
    guarantees complete hook cleanup on exit even if forward raises.
    """
    handles = []
    use_adain = cfg.adain_alpha > 0.0
    use_spatial_w = (
        cfg.w_in_32 < 0.999 or cfg.w_in_64 < 0.999 or cfg.w_in_128 < 0.999
    )
    if not use_adain and not use_spatial_w:
        yield
        return

    from torch.nn import functional as F

    state = {"lq_feat": None}
    try:
        if use_adain:
            enc_blocks = getattr(getattr(net, "encoder", None), "blocks", None)
            gen_blocks = getattr(getattr(net, "generator", None), "blocks", None)
            if (
                enc_blocks is not None
                and gen_blocks is not None
                and len(enc_blocks) > 0
                and len(gen_blocks) > 0
                and hasattr(enc_blocks[-1], "register_forward_hook")
                and hasattr(gen_blocks[0], "register_forward_pre_hook")
            ):
                def _capture_lq(_mod, _inp, out):
                    state["lq_feat"] = out

                def _masked_adain_pre(_mod, args):
                    if not args:
                        return args
                    quant_feat = args[0]
                    lq_feat = state["lq_feat"]
                    if lq_feat is None or quant_feat.shape != lq_feat.shape:
                        return args
                    b, c, h, w = quant_feat.shape
                    if mask512.shape[-2] == 512 and mask512.shape[-1] == 512 and h > 0 and 512 % h == 0 and h == w:
                        m_small = F.avg_pool2d(mask512, kernel_size=512 // h)
                    else:
                        m_small = F.interpolate(mask512, size=(h, w), mode="area")
                    vis = (m_small < 0.25).float()
                    if float(vis.sum().item()) < 16.0:
                        vis = (m_small < 0.55).float()
                    if float(vis.sum().item()) < 12.0:
                        return args

                    vis_flat = vis.reshape(b, 1, -1)
                    denom = vis_flat.sum(dim=2, keepdim=True).clamp_min(1.0)
                    lq_flat = lq_feat.reshape(b, c, -1)
                    q_flat = quant_feat.reshape(b, c, -1)

                    s_mean = (lq_flat * vis_flat).sum(dim=2, keepdim=True) / denom
                    s_var = (((lq_flat - s_mean) ** 2) * vis_flat).sum(dim=2, keepdim=True) / denom
                    s_std = (s_var + 1e-5).sqrt()

                    if cfg.adain_mode == "matched_vis":
                        c_mean = (q_flat * vis_flat).sum(dim=2, keepdim=True) / denom
                        c_var = (((q_flat - c_mean) ** 2) * vis_flat).sum(dim=2, keepdim=True) / denom
                        c_std = (c_var + 1e-5).sqrt()
                    else:
                        c_mean = q_flat.mean(dim=2, keepdim=True)
                        c_std = (q_flat.var(dim=2, keepdim=True, unbiased=False) + 1e-5).sqrt()

                    ratio = (s_std / c_std).clamp(0.65, 1.50)
                    normed = ((q_flat - c_mean) * ratio + s_mean).reshape(b, c, h, w)
                    alpha = float(cfg.adain_alpha)
                    blended = quant_feat * (1.0 - alpha) + normed * alpha
                    return (blended,) + tuple(args[1:])

                handles.append(enc_blocks[-1].register_forward_hook(_capture_lq))
                handles.append(gen_blocks[0].register_forward_pre_hook(_masked_adain_pre))

        if use_spatial_w:
            fuse_dict = getattr(net, "fuse_convs_dict", None)
            if fuse_dict is not None:
                scale_map = {"32": cfg.w_in_32, "64": cfg.w_in_64, "128": cfg.w_in_128}
                for f_size, w_in in scale_map.items():
                    if w_in >= 0.999 or f_size not in fuse_dict:
                        continue
                    fuse_mod = fuse_dict[f_size]
                    if not hasattr(fuse_mod, "register_forward_hook"):
                        continue
                    s = int(f_size)
                    m_s = F.interpolate(mask512, size=(s, s), mode="bilinear", align_corners=False)
                    if cfg.feather_ksize >= 3:
                        pad = cfg.feather_ksize // 2
                        m_s = F.avg_pool2d(m_s, kernel_size=cfg.feather_ksize, stride=1, padding=pad)
                    w_spatial = (1.0 - m_s) * 1.0 + m_s * float(w_in)

                    def _make_sft_hook(w_map):
                        def _sft_hook(_mod, args, out):
                            if len(args) < 2:
                                return out
                            dec_feat = args[1]
                            w_eff = (
                                w_map
                                if w_map.shape[-2:] == out.shape[-2:]
                                else F.interpolate(w_map, size=out.shape[-2:], mode="bilinear", align_corners=False)
                            )
                            residual = out - dec_feat
                            return dec_feat + w_eff * residual
                        return _sft_hook

                    handles.append(fuse_mod.register_forward_hook(_make_sft_hook(w_spatial)))

        yield
    finally:
        for h in handles:
            h.remove()


class CompletionAdapter:
    """Owned wrapper with unchanged R2 adapter math; no shared restoration hooks."""
    def __init__(self, net):
        self.net = net.eval().requires_grad_(False)
        self.raw512 = None
        self.input512 = None

    def __call__(self, image, mask, forward_event=None, *, hook_cfg=None):
        import torch
        from torch.nn import functional as F
        need(image.ndim == 4 and image.shape[1:] == (3, 256, 256)
             and mask.shape == (image.shape[0], 1, 256, 256), "Expected NCHW256 RGB and N1HW mask.")
        need(torch.isfinite(image).all().item() and image.min().item() >= 0 and image.max().item() <= 1,
             "Expected finite RGB values in[0,1].")
        need(((mask == 0) | (mask == 1)).all().item(), "Expected a binary mask.")
        if (mask.mean((1, 2, 3)) >= .85).any().item():
            raise CompletionRefused("Too little visible face remains.")
        with torch.inference_mode():
            result = image.clone()
            active = mask.sum((1, 2, 3)) > 0
            if not active.any().item():
                return result
            visible = 1 - mask[active]
            support = F.interpolate(visible, size=(512, 512), mode="bilinear", align_corners=False)
            x = F.interpolate(image[active] * visible, size=(512, 512), mode="bilinear", align_corners=False)
            x = x / support.clamp_min(1e-8)
            m = F.interpolate(mask[active], size=(512, 512), mode="nearest")
            x = (x * (1 - m) + m) * 2 - 1
            self.input512 = x.detach().cpu().numpy().copy()
            if forward_event:
                forward_event("entered")
            if hook_cfg is not None:
                with attach_r2_hooks(self.net, m, hook_cfg):
                    raw = self.net(x, w=1, adain=False)[0]
            else:
                raw = self.net(x, w=1, adain=False)[0]
            if forward_event:
                forward_event("returned")
            need(raw.shape == x.shape and torch.isfinite(raw).all().item(), "Invalid R2 generator output.")
            self.raw512 = raw.detach().cpu().numpy().copy()
            generated = ((raw + 1) / 2).clamp(0, 1)
            generated = F.interpolate(generated, size=image.shape[-2:], mode="bilinear", align_corners=False)
            result[active] = image[active] * (1 - mask[active]) + generated * mask[active]
            return result


def load_frozen_cpu_completion():
    verify_architecture_sources()
    checkpoint = assets.asset_path("completion_r2")
    import torch
    from third_party.codeformer_completion.codeformer_arch import CodeFormer
    verify_architecture_sources()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    need(type(state) is dict and set(state) == {"params_ema"} and isinstance(state["params_ema"], dict),
         "Completion asset must contain only params_ema.")
    params = state["params_ema"]
    need(len(params) == 497, "Expected complete497-tensor R2 state.")
    logical = hashlib.sha256()
    for name, value in sorted(params.items()):
        need(isinstance(name, str) and isinstance(value, torch.Tensor) and value.device.type == "cpu"
             and value.dtype == torch.float32 and torch.isfinite(value).all().item(), "Invalid completion state tensor.")
        value = value.detach().contiguous()
        logical.update(name.encode()); logical.update(str(tuple(value.shape)).encode())
        logical.update(str(value.dtype).encode()); logical.update(value.numpy().tobytes())
    need(logical.hexdigest() == LOGICAL_SHA256, "Completion logical state differs.")
    net = CodeFormer(dim_embd=512, codebook_size=512, n_head=8, n_layers=9, connect_list=["32", "64", "128"])
    net.load_state_dict(params, strict=True)
    adapter = CompletionAdapter(net.cpu().float())
    need(not net.training and not any(p.requires_grad for p in net.parameters()), "Completion model must remain frozen.")
    assets.asset_path("completion_r2")
    return adapter


def _score_completion_candidate(rgb, out_rgb, removal, observed):
    """Evaluate 5-factor Q and optional eDifFIQA-T with bounded exception fallback."""
    try:
        import cv2 as _cv2
        catch_errs = (ImportError, OSError, RuntimeError, ValueError, _cv2.error)
    except ImportError:
        catch_errs = (ImportError, OSError, RuntimeError, ValueError)
    try:
        from .automatic_covering import evaluate_completion_quality
        q = evaluate_completion_quality(rgb, out_rgb, removal, observed)
    except catch_errs:
        q = None
    if not isinstance(q, dict) or "score_0_100" not in q:
        return None, None, None
    try:
        from .fiqa_scorer import score_face_crop_fiqa
        fiqa = score_face_crop_fiqa(out_rgb)
    except catch_errs:
        fiqa = None
    fiqa_val = round(float(fiqa), 4) if fiqa is not None else None
    return int(q["score_0_100"]), q, fiqa_val


def _run_completion_candidate(
    adapter,
    rgb,
    source_rgb_f,
    orig_removal,
    infer_mask,
    observed,
    align_mat,
    hook_cfg: HookConfig,
    counted_event,
    calls,
    *,
    inner_blend_rgb=None,
):
    """Run one frozen-R2 pass and composite strictly inside orig_removal against original rgb."""
    import torch

    orig_f = rgb.astype(np.float32) / np.float32(255.0)
    eff_mask = infer_mask
    if hook_cfg.dilate_infer_px > 0:
        import cv2
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * hook_cfg.dilate_infer_px + 1, 2 * hook_cfg.dilate_infer_px + 1),
        )
        dil = (cv2.dilate(infer_mask.astype(np.uint8), k) > 0) & observed
        if float(dil.mean()) < 0.80:
            eff_mask = dil.astype(np.uint8)

    use_mat = None if hook_cfg.force_no_align else align_mat
    used_alignment = False
    completed = None
    expected_calls = calls["returned"] + 1
    active_hook = (
        hook_cfg
        if (
            hook_cfg.adain_alpha > 0.0
            or hook_cfg.w_in_32 < 0.999
            or hook_cfg.w_in_64 < 0.999
            or hook_cfg.w_in_128 < 0.999
        )
        else None
    )

    if use_mat is not None:
        import cv2
        vis_f = (eff_mask == 0).astype(np.float32)
        warped_vis = cv2.warpAffine(
            vis_f, use_mat, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        warped_num = cv2.warpAffine(
            source_rgb_f * vis_f[..., None],
            use_mat,
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0),
        )
        vis_mean = (
            source_rgb_f[eff_mask == 0].mean(axis=0).astype(np.float32)
            if (eff_mask == 0).any()
            else np.full(3, 0.5, dtype=np.float32)
        )
        warped_rgb = np.where(
            warped_vis[..., None] > np.float32(1e-4),
            warped_num / np.maximum(warped_vis[..., None], np.float32(1e-4)),
            vis_mean[None, None, :],
        )
        warped_rgb = np.clip(warped_rgb, 0.0, 1.0).astype(np.float32)
        warped_mask = (
            cv2.warpAffine(
                eff_mask.astype(np.float32),
                use_mat,
                (256, 256),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            > 0.5
        ).astype(np.float32)
        if warped_mask.any() and float(warped_mask.mean()) < 0.85:
            image = torch.from_numpy(warped_rgb.copy()).permute(2, 0, 1)[None].float()
            tensor_mask = torch.from_numpy(warped_mask.copy())[None, None].float()
            result = adapter(image, tensor_mask, forward_event=counted_event, hook_cfg=active_hook)
            need(result.shape == image.shape and result.dtype == torch.float32 and torch.isfinite(result).all().item(),
                 "Invalid R2 completion result.")
            outside = (tensor_mask == 0).expand_as(image)
            need(torch.equal(result[outside], image[outside]), "R2 changed supplied canvas outside the reviewed mask.")
            need(calls == {"entered": expected_calls, "returned": expected_calls},
                 "Completion forward enter/return mismatch.")
            warped_out = result[0].permute(1, 2, 0).detach().cpu().numpy().copy()
            unwarped = cv2.warpAffine(
                warped_out,
                use_mat,
                (256, 256),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            completed = np.clip(unwarped, 0.0, 1.0).astype(np.float32)
            completed[eff_mask == 0] = source_rgb_f[eff_mask == 0]
            used_alignment = True

    if not used_alignment:
        image = torch.from_numpy(source_rgb_f.copy()).permute(2, 0, 1)[None].float()
        tensor_mask = torch.from_numpy(eff_mask.copy())[None, None].float()
        result = adapter(image, tensor_mask, forward_event=counted_event, hook_cfg=active_hook)
        need(result.shape == image.shape and result.dtype == torch.float32 and torch.isfinite(result).all().item(),
             "Invalid R2 completion result.")
        outside = (tensor_mask == 0).expand_as(image)
        need(torch.equal(result[outside], image[outside]), "R2 changed supplied canvas outside the reviewed mask.")
        need(calls == {"entered": expected_calls, "returned": expected_calls},
             "Completion forward enter/return mismatch.")
        completed = result[0].permute(1, 2, 0).detach().cpu().numpy().copy()

    # For progressive inner-core refinement, harmonize the refined core into Pass 2 first
    # so the inner boundary blends smoothly before compositing into the original reviewed mask.
    if inner_blend_rgb is not None:
        completed = _blend_inside_mask(inner_blend_rgb, completed, eff_mask, observed)
    # Always composite strictly inside ORIGINAL reviewed mask against ORIGINAL rgb.
    completed[orig_removal == 0] = orig_f[orig_removal == 0]
    completed = _blend_inside_mask(rgb, completed, orig_removal, observed)
    output = encode_output(rgb, orig_removal, completed)
    need(np.array_equal(output[orig_removal == 0], rgb[orig_removal == 0]), "Completion changed outside-mask RGB bytes.")
    need(np.array_equal(completed[orig_removal == 0], orig_f[orig_removal == 0]),
         "Completion changed outside-mask float pixels.")
    return {
        "config": hook_cfg.name,
        "output_rgb": output,
        "completed_rgb": completed,
        "raw512": adapter.raw512,
        "input512": adapter.input512,
        "aligned": used_alignment,
        "Q": None,
        "q_info": None,
        "fiqa": None,
    }


def complete(rgb, mask, prepared_bounds=None, *, forward_event=None, max_passes=1, q_threshold=78):
    """Return one plausible completion; no restoration, detection or extra mask margin."""
    started = time.monotonic()
    need(type(max_passes) is int and 1 <= max_passes <= 3, "max_passes must be an integer between 1 and 3.")
    need(type(q_threshold) is int and 0 <= q_threshold <= 100, "q_threshold must be an integer between 0 and 100.")
    removal, bounds, visibility = validate_arrays(rgb, mask, prepared_bounds)
    contract = model_contract()
    if not removal.any():
        return {"output_rgb": rgb.copy(), "completed_rgb": rgb.astype(np.float32) / np.float32(255),
                "raw512": None, "input512": None, "metadata": {**contract, "bypassed": True,
                "neural_calls": 0, "checkpoint_loaded": False, "prepared_bounds": bounds,
                "visibility": visibility, "seconds": time.monotonic() - started}}
    import torch
    torch.set_num_threads(4)
    adapter = load_frozen_cpu_completion()
    observed = np.zeros((256, 256), dtype=bool)
    observed[bounds[1]:bounds[3], bounds[0]:bounds[2]] = True

    alignment = _estimate_ffhq_alignment(rgb, removal, observed)
    align_mat = alignment.pop("matrix", None) if alignment.get("applied") else None

    calls = {"entered": 0, "returned": 0}

    def counted_event(event):
        calls[event] += 1
        need(calls[event] <= max_passes, f"{max_passes} completion forward(s) maximum.")
        if forward_event:
            forward_event(event)

    orig_f = rgb.astype(np.float32) / np.float32(255.0)
    c1 = _run_completion_candidate(
        adapter, rgb, orig_f, removal, removal, observed, align_mat,
        HookConfig("base_w1"), counted_event, calls,
    )
    if align_mat is not None and not c1["aligned"]:
        alignment["applied"] = False
    cands = [c1]

    if max_passes > 1:
        q1, q1_info, fiqa1 = _score_completion_candidate(rgb, c1["output_rgb"], removal, observed)
        c1["Q"], c1["q_info"], c1["fiqa"] = q1, q1_info, fiqa1

        if q1 is not None and q1 < q_threshold and max_passes >= 2:
            c2 = _run_completion_candidate(
                adapter, rgb, orig_f, removal, removal, observed, align_mat,
                HookConfig("d3_w1", dilate_infer_px=3), counted_event, calls,
            )
            q2, q2_info, fiqa2 = _score_completion_candidate(rgb, c2["output_rgb"], removal, observed)
            c2["Q"], c2["q_info"], c2["fiqa"] = q2, q2_info, fiqa2
            cands.append(c2)

            best_so_far = max(q1, q2 if q2 is not None else -1)
            if max_passes >= 3 and best_so_far < q_threshold:
                mask_pct = 100.0 * float(int(removal.sum())) / float(max(1, int(observed.sum())))
                c1_clr = int(((q1_info or {}).get("factors") or {}).get("occluder_clear", 100))
                c2_clr = int(((q2_info or {}).get("factors") or {}).get("occluder_clear", 100))
                c3 = None
                if mask_pct >= 15.0 and min(c1_clr, c2_clr) < 65:
                    import cv2
                    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    dil_mask = ((cv2.dilate(removal.astype(np.uint8), k7) > 0) & observed).astype(np.uint8)
                    dist = cv2.distanceTransform(dil_mask, cv2.DIST_L2, 5)
                    inner_core = (dist >= 5.0).astype(np.uint8)
                    if int(inner_core.sum()) >= 256:
                        c2_f = c2["output_rgb"].astype(np.float32) / np.float32(255.0)
                        c3 = _run_completion_candidate(
                            adapter, rgb, c2_f, removal, inner_core, observed, align_mat,
                            HookConfig("prog_from_c2"), counted_event, calls,
                            inner_blend_rgb=c2["output_rgb"],
                        )
                    if c3 is None:
                        c3 = _run_completion_candidate(
                            adapter, rgb, orig_f, removal, removal, observed, align_mat,
                            HookConfig("d6_w1", dilate_infer_px=6), counted_event, calls,
                        )
                elif mask_pct >= 18.0:
                    c3 = _run_completion_candidate(
                        adapter, rgb, orig_f, removal, removal, observed, align_mat,
                        HookConfig("d6_w1", dilate_infer_px=6), counted_event, calls,
                    )
                else:
                    c3 = _run_completion_candidate(
                        adapter, rgb, orig_f, removal, removal, observed, align_mat,
                        HookConfig(
                            "mild_sw07_d3",
                            w_in_32=0.85,
                            w_in_64=0.70,
                            w_in_128=0.55,
                            adain_alpha=0.25,
                            dilate_infer_px=3,
                        ),
                        counted_event,
                        calls,
                    )
                q3, q3_info, fiqa3 = _score_completion_candidate(rgb, c3["output_rgb"], removal, observed)
                c3["Q"], c3["q_info"], c3["fiqa"] = q3, q3_info, fiqa3
                cands.append(c3)

        scored = [c for c in cands if c["Q"] is not None]
        if scored:
            max_q = max(c["Q"] for c in scored)
            top_tier = [c for c in scored if c["Q"] >= max_q - 1]
            best = max(top_tier, key=lambda c: float(c["Q"]) + 5.0 * float(c["fiqa"] or 0.0))
        else:
            best = c1
    else:
        best = c1

    need(calls["entered"] == calls["returned"] and 1 <= calls["returned"] <= max_passes,
         "Inconsistent completion forward counts.")
    alignment["applied"] = bool(best["aligned"])
    output = best["output_rgb"]
    completed = best["completed_rgb"]
    need(np.array_equal(output[removal == 0], rgb[removal == 0]), "Completion changed outside-mask RGB bytes.")
    need(np.array_equal(completed[removal == 0], orig_f[removal == 0]), "Completion changed outside-mask float pixels.")
    adaptive_rerank = {
        "enabled": max_passes > 1,
        "max_passes": max_passes,
        "q_threshold": int(q_threshold),
        "passes": len(cands),
        "early_stopped": max_passes > 1 and len(cands) == 1,
        "chosen": best["config"],
        "pass1_Q": c1["Q"],
        "best_Q": best["Q"],
        "candidates": [{"config": c["config"], "Q": c["Q"], "fiqa": c["fiqa"]} for c in cands],
    }
    return {"output_rgb": output, "completed_rgb": completed, "raw512": best["raw512"],
            "input512": best["input512"], "metadata": {**contract, "bypassed": False,
            "neural_calls": int(calls["returned"]),
            "checkpoint_loaded": True, "prepared_bounds": bounds, "visibility": visibility,
            "ffhq_alignment": alignment,
            "adaptive_rerank": adaptive_rerank,
            "seconds": time.monotonic() - started, "exact_outside_reviewed_mask": True,
            "preservation_reference": "The exact supplied prepared/restored RGB256 canvas.",
            "png_conversion": "clamp[0,1], multiply255, np.rint, uint8; supplied bytes recopied outside reviewed support."}}

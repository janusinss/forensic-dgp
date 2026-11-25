"""Fast CPU automatic face-covering proposal refinement and research ONNX segmentation.

Combines frozen neural detector seed expansion, research BiSeNet 19-class semantic
face parsing (Yu et al., ECCV 2018; Lee et al., CVPR 2020) and XSeg occlusion-aware
segmentation (Perov et al., Pattern Recognition 2023), plus deterministic Lab/YCrCb
multi-family occlusion detection and visibility_check256 clamping.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import threading

import cv2
import numpy as np

from .completion_runtime import visibility_check256
from .managed_paths import ManagedPathError, data_path

ROOT = Path(__file__).resolve().parents[1]
_BISENET_FILENAME = "bisenet_resnet18.onnx"
_BISENET_BYTES = 53205364
_BISENET_SHA256 = "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
_XSEG_FILENAME = "xseg.onnx"
_XSEG_BYTES = 70327737
_XSEG_SHA256 = "0b57328efcb839d85973164b617ceee9dfe6cfcb2c82e8a033bba9f4f09b27e5"
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_ONNX_LOCK = threading.Lock()
_ONNX_CACHE: dict[str, object] = {}

_YY, _XX = np.mgrid[:256, :256]
_INNER_FACE = (((_XX - 127.5) / 76.0) ** 2 + ((_YY - 136.0) / 96.0) ** 2 <= 1.0)
_FACE_ELLIPSE = (((_XX - 127.5) / 83.0) ** 2 + ((_YY - 137.0) / 104.0) ** 2 <= 1.0)
_FACE_MARGIN_ELLIPSE = (((_XX - 127.5) / 93.0) ** 2 + ((_YY - 137.0) / 114.0) ** 2 <= 1.0)
_FOREHEAD_BAND = (_YY >= 48) & (_YY < 76) & (_XX >= 82) & (_XX < 174)
_CHEEK_BANDS = (_YY >= 110) & (_YY < 168) & (((_XX >= 58) & (_XX < 106)) | ((_XX >= 150) & (_XX < 198)))
_NOSE_BRIDGE = (_YY >= 84) & (_YY < 134) & (_XX >= 112) & (_XX < 144)
_SKIN_SAMPLE_ZONE = _FOREHEAD_BAND | _CHEEK_BANDS | _NOSE_BRIDGE
_EYE_ZONE = (_YY >= 68) & (_YY < 134) & (_XX >= 46) & (_XX < 210)
_INNER_EYE_BAND = (_YY >= 76) & (_YY < 134) & (_XX >= 60) & (_XX < 196)
_LOWER_ZONE = _INNER_FACE & (_YY >= 116) & (_YY < 232) & (_XX >= 50) & (_XX < 206)
_CENTER_ZONE = _INNER_FACE & (_YY >= 54) & (_YY < 220) & (_XX >= 54) & (_XX < 202)
_LIP_BAND = (_YY >= 168) & (_YY < 198) & (_XX >= 98) & (_XX < 158)
_CENTRAL_OBJ_BAND = (_YY >= 80) & (_YY < 176) & (_XX >= 72) & (_XX < 184)

_DILATE_3PX = np.ones((7, 7), np.uint8)
_OPEN_3PX = np.ones((3, 3), np.uint8)
_ERODE_2PX = np.ones((5, 5), np.uint8)
_ERODE_1PX = np.ones((3, 3), np.uint8)
_GLARE_BORDER = np.ones((13, 13), np.uint8)
_SEED_RADIUS = np.ones((41, 41), np.uint8)

_ELLIPSE_5PX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_ELLIPSE_7PX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_ELLIPSE_9PX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_ELLIPSE_11PX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def _validate_inputs(
    rgb: np.ndarray,
    probability: np.ndarray,
    base_proposed: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (256, 256, 3)):
        raise ValueError("Expected RGB256 uint8 array.")
    if not (
        isinstance(probability, np.ndarray)
        and probability.shape == (256, 256)
        and np.issubdtype(probability.dtype, np.floating)
        and np.isfinite(probability).all()
        and float(probability.min()) >= 0.0
        and float(probability.max()) <= 1.0
    ):
        raise ValueError("Expected finite float32 256x256 probability map in [0, 1].")
    if not (
        isinstance(base_proposed, np.ndarray)
        and base_proposed.shape == (256, 256)
        and base_proposed.dtype in (np.dtype("uint8"), np.dtype("bool"))
    ):
        raise ValueError("Expected uint8 or bool 256x256 base_proposed mask.")
    if base_proposed.dtype == np.dtype("uint8"):
        vals = set(np.unique(base_proposed).tolist())
        if not (vals.issubset({0, 1}) or vals.issubset({0, 255})):
            raise ValueError("Expected strictly binary 0/1 or 0/255 values in base_proposed.")
    if not (isinstance(observed, np.ndarray) and observed.shape == (256, 256) and observed.dtype == np.bool_):
        raise ValueError("Expected boolean 256x256 observed support.")
    return rgb, probability.astype(np.float32, copy=False), base_proposed > 0, observed


def _skin_mask(lab: np.ndarray, ycrcb: np.ndarray) -> np.ndarray:
    """Luminance-tolerant skin likelihood mask in YCrCb + Lab (including pale lit skin)."""
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]
    normal_skin = (
        (cr >= 134)
        & (cr <= 192)
        & (cb >= 68)
        & (cb <= 136)
        & ((cr - cb) >= 14)
        & (lab[..., 0] >= 42.0)
    )
    pale_skin = (
        (lab[..., 0] >= 165.0)
        & (cr >= 130)
        & (cr <= 192)
        & (cb >= 68)
        & (cb <= 142)
        & ((cr - cb) >= 8)
    )
    return normal_skin | pale_skin


def _reference_skin_lab(
    lab: np.ndarray,
    skin: np.ndarray,
    observed: np.ndarray,
    inner_face: np.ndarray,
    bface: np.ndarray | None = None,
    seg: np.ndarray | None = None,
    px: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate adaptive subject skin Lab color, supporting dark skin and avoiding occluders."""
    if bface is not None and seg is not None and px is not None:
        vis_skin_cls = (seg == 1) & observed & (px >= 0.65)
        warm_vis = vis_skin_cls & skin
        if int(warm_vis.sum()) >= 120:
            ref_mask = warm_vis
            return (
                float(np.median(lab[..., 0][ref_mask])),
                float(np.median(lab[..., 1][ref_mask])),
                float(np.median(lab[..., 2][ref_mask])),
            )
        if int(vis_skin_cls.sum()) >= 120:
            # Dark skin or monochrome lighting where YCrCb skin mask misses visible BiSeNet skin.
            ref_mask = vis_skin_cls
            return (
                float(np.median(lab[..., 0][ref_mask])),
                float(np.median(lab[..., 1][ref_mask])),
                float(np.median(lab[..., 2][ref_mask])),
            )

    ref_mask = _SKIN_SAMPLE_ZONE & observed & skin & (lab[..., 0] >= 80.0)
    if int(ref_mask.sum()) < 150:
        ref_mask = inner_face & skin & (lab[..., 0] >= 80.0)
    if int(ref_mask.sum()) < 150:
        ref_mask = inner_face & skin & (lab[..., 0] >= 35.0)
    if int(ref_mask.sum()) >= 150:
        l_vals = lab[..., 0][ref_mask]
        l_p40 = float(np.percentile(l_vals, 40.0))
        upper_lit = ref_mask & (lab[..., 0] >= l_p40)
        if int(upper_lit.sum()) >= 60:
            ref_mask = upper_lit
    elif int(ref_mask.sum()) == 0:
        forehead = _FOREHEAD_BAND & observed
        ref_mask = forehead if forehead.any() else (inner_face if inner_face.any() else observed)
    return (
        float(np.median(lab[..., 0][ref_mask])),
        float(np.median(lab[..., 1][ref_mask])),
        float(np.median(lab[..., 2][ref_mask])),
    )


def _expand_neural_seeds(
    lab: np.ndarray,
    probability: np.ndarray,
    inner_face: np.ndarray,
    observed: np.ndarray,
    px: np.ndarray | None = None,
    allowed_mask: np.ndarray | None = None,
    beard_suppress: np.ndarray | None = None,
    hairline_suppress: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Expand neural probability seeds with >=80px support and XSeg/BiSeNet FP gating."""
    domain = inner_face if allowed_mask is None else (inner_face & allowed_mask)
    prob_cand = (probability >= np.float32(0.08)) & domain
    if not prob_cand.any():
        return np.zeros((256, 256), dtype=bool), 0

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(prob_cand.astype(np.uint8), connectivity=8)
    seed_mask = np.zeros((256, 256), dtype=bool)
    grown_total = np.zeros((256, 256), dtype=bool)

    for idx in range(1, num_labels):
        comp = labels == idx
        comp_prob = probability[comp]
        strong_px = int((comp_prob >= 0.35).sum())
        if strong_px < 80:
            continue
        core = comp & (probability >= 0.35)
        ref_pts = core if core.any() else comp

        if beard_suppress is not None and float(beard_suppress[ref_pts].mean()) >= 0.55:
            continue
        if hairline_suppress is not None and float(hairline_suppress[ref_pts].mean()) >= 0.55:
            continue

        lab_spread = float(
            lab[..., 0][ref_pts].std() + lab[..., 1][ref_pts].std() + lab[..., 2][ref_pts].std()
        )
        uniform_patch = strong_px >= 150 and lab_spread < 4.0

        if px is not None and not uniform_patch:
            # Reject seeds sitting on XSeg-confirmed visible unoccluded skin.
            if float(px[ref_pts].mean()) >= 0.55:
                continue

        seed_mask |= comp
        l_s = float(np.median(lab[..., 0][ref_pts]))
        a_s = float(np.median(lab[..., 1][ref_pts]))
        b_s = float(np.median(lab[..., 2][ref_pts]))
        de_seed = np.sqrt(
            (lab[..., 0] - l_s) ** 2
            + (lab[..., 1] - a_s) ** 2
            + (lab[..., 2] - b_s) ** 2
        )
        dist_zone = cv2.dilate(comp.astype(np.uint8), _SEED_RADIUS) > 0
        if px is not None and not uniform_patch:
            grow_gate = (de_seed < 14.0) & dist_zone & (px < 0.55)
        elif uniform_patch:
            grow_gate = (de_seed < 6.0) & dist_zone
        else:
            grow_gate = (de_seed < 14.0) & dist_zone

        eligible = (comp | grow_gate) & domain
        if beard_suppress is not None:
            eligible &= ~beard_suppress
        if hairline_suppress is not None:
            eligible &= ~hairline_suppress

        n_grow, grow_labels, _, _ = cv2.connectedComponentsWithStats(eligible.astype(np.uint8), connectivity=8)
        for g_idx in range(1, n_grow):
            g_comp = grow_labels == g_idx
            if (g_comp & comp).any():
                grown_total |= g_comp

    seed_pixels = int(seed_mask.sum())
    if not grown_total.any():
        return grown_total, seed_pixels
    dilated = (cv2.dilate(grown_total.astype(np.uint8), _DILATE_3PX) > 0) & observed
    return dilated, seed_pixels


def _detect_eyewear_and_glare(
    lab: np.ndarray,
    skin: np.ndarray,
    observed: np.ndarray,
    l_ref: float,
    seg: np.ndarray | None = None,
    px: np.ndarray | None = None,
    both_eyes_visible: bool = False,
) -> np.ndarray:
    """Detect dark sunglasses lenses/frames and bright specular lens glare."""
    eye_zone = _EYE_ZONE & observed
    if seg is not None:
        eye_zone = eye_zone & (seg != 17)
    eye_mask = np.zeros((256, 256), dtype=bool)

    dark_cand = eye_zone & (lab[..., 0] < 82.0) & ((l_ref - lab[..., 0]) > 36.0)
    if dark_cand.any() and not both_eyes_visible:
        n_d, lbl_d, st_d, _ = cv2.connectedComponentsWithStats(dark_cand.astype(np.uint8), connectivity=8)
        for idx in range(1, n_d):
            x, y, w, h, area = (int(st_d[idx, k]) for k in range(5))
            if area < 320 or h < 10 or w < 18 or w < 0.75 * h:
                continue
            comp = lbl_d == idx
            if float(skin[comp].mean()) >= 0.35:
                continue
            if px is not None and float(px[comp].mean()) >= 0.62:
                continue
            # Reject scalp hair or dark background continuing directly above eye_zone.
            if y <= 70:
                y0 = max(0, y - 12)
                if y > y0 and float((lab[y0:y, x:x + w, 0] < 82.0).mean()) > 0.50:
                    continue
            if int((comp & _INNER_EYE_BAND).sum()) < 180:
                continue
            eye_mask |= comp

    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    glare_cand = (
        eye_zone
        & (~skin)
        & (lab[..., 0] >= 242.0)
        & ((lab[..., 0] - l_ref) > 42.0)
        & (chroma < 10.0)
    )
    if glare_cand.any():
        l_grad = np.hypot(cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0), cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1))
        n_g, lbl_g, st_g, cen_g = cv2.connectedComponentsWithStats(glare_cand.astype(np.uint8), connectivity=8)
        for idx in range(1, n_g):
            x, y, w, h, area = (int(st_g[idx, k]) for k in range(5))
            cx, cy = float(cen_g[idx][0]), float(cen_g[idx][1])
            if not (120 <= area <= 2200 and y >= 74 and w <= 3.5 * max(1, h) and 62.0 <= cx <= 194.0 and 84.0 <= cy <= 130.0):
                continue
            comp = lbl_g == idx
            comp_u8 = comp.astype(np.uint8)
            border = (cv2.dilate(comp_u8, _GLARE_BORDER) > 0) & (~comp) & eye_zone
            dark_border = bool(border.any()) and float((lab[..., 0][border] < 95.0).mean()) >= 0.10
            ring = (cv2.dilate(comp_u8, _ERODE_1PX) > 0) & (cv2.erode(comp_u8, _ERODE_1PX) == 0)
            sharp_edge = bool(ring.any()) and float(l_grad[ring].mean()) > 14.0
            if dark_border or sharp_edge:
                eye_mask |= comp

    return eye_mask


def _detect_masks_and_scarves(
    lab: np.ndarray,
    skin: np.ndarray,
    inner_face: np.ndarray,
    delta_e: np.ndarray,
    l_ref: float,
    allowed_lower: np.ndarray | None = None,
) -> np.ndarray:
    """Detect lower-face surgical masks, cloth masks, and scarves."""
    lower_zone = _LOWER_ZONE & inner_face
    if allowed_lower is not None:
        lower_zone = lower_zone & allowed_lower
    cand = lower_zone & (~skin) & ((delta_e > 22.0) | (np.abs(lab[..., 0] - l_ref) > 34.0))
    lower_mask = np.zeros((256, 256), dtype=bool)
    if not cand.any():
        return lower_mask

    n_m, lbl_m, st_m, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    for idx in range(1, n_m):
        if int(st_m[idx, cv2.CC_STAT_AREA]) < 1400:
            continue
        comp = lbl_m == idx
        if comp[135:205, 88:168].any() and int(comp[145:185, 100:156].sum()) >= 500:
            lower_mask |= comp
    return lower_mask


def _detect_foreign_objects(
    lab: np.ndarray,
    skin: np.ndarray,
    inner_face: np.ndarray,
    delta_e: np.ndarray,
    a_ref: float,
    b_ref: float,
    px: np.ndarray | None = None,
    seg: np.ndarray | None = None,
    beard_suppress: np.ndarray | None = None,
) -> np.ndarray:
    """Detect clear non-skin foreign objects and rectangular censor bars inside central face."""
    center_zone = _CENTER_ZONE & inner_face
    if seg is not None:
        center_zone = center_zone & (~np.isin(seg, (0, 7, 8, 9, 14, 15, 16, 17, 18)))
    chroma_diff = np.hypot(lab[..., 1] - a_ref, lab[..., 2] - b_ref)
    cand = center_zone & (~skin) & (delta_e > 32.0) & (chroma_diff > 18.0)
    obj_mask = np.zeros((256, 256), dtype=bool)
    if cand.any():
        n_o, lbl_o, st_o, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        for idx in range(1, n_o):
            x, y, w, h, area = (int(st_o[idx, k]) for k in range(5))
            if area < 600:
                continue
            comp = lbl_o == idx
            if area < 650 and (comp & _LIP_BAND).any() and float(lab[..., 1][comp].mean()) > a_ref + 4.0:
                continue
            # Exclude side hair and top scalp hair encroaching near the edge of center_zone.
            if (x <= 56 or x + w >= 200) and float(lab[..., 0][comp].mean()) < 60.0 and w < 65:
                continue
            if (x <= 56 and x + w >= 190 and h < 70) and y < 105:
                continue
            if y <= 58 and float((lab[max(0, y - 10):y, x:x + w, 0] < 85.0).mean()) > 0.50:
                continue
            if int((comp & _CENTRAL_OBJ_BAND).sum()) < 300:
                continue
            if beard_suppress is not None and float(beard_suppress[comp].mean()) >= 0.45:
                if float(chroma_diff[comp].mean()) < 28.0:
                    continue
            if px is not None and float(px[comp].mean()) >= 0.72 and float(chroma_diff[comp].mean()) < 36.0:
                continue
            obj_mask |= comp

    # Detect solid achromatic/colored rectangular censor bars (even without chroma_diff > 18).
    if px is not None:
        bar_cand = center_zone & (~skin) & (px < 0.35) & (delta_e > 26.0)
        if bar_cand.any():
            n_b, lbl_b, st_b, _ = cv2.connectedComponentsWithStats(bar_cand.astype(np.uint8), connectivity=8)
            for idx in range(1, n_b):
                x, y, w, h, area = (int(st_b[idx, k]) for k in range(5))
                if area < 400 or w < 45 or h < 8 or w < 2.2 * h:
                    continue
                if area / float(max(1, w * h)) < 0.85:
                    continue
                comp = lbl_b == idx
                if beard_suppress is not None and float(beard_suppress[comp].mean()) >= 0.50:
                    continue
                obj_mask |= comp

    return obj_mask


def _enforce_visibility(
    mask: np.ndarray,
    observed: np.ndarray,
    probability: np.ndarray,
    px: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Trim lowest-priority outer periphery until visibility_check256 passes."""
    empty_unavailable = np.where(observed, 0, 255).astype(np.uint8)
    if visibility_check256(empty_unavailable, observed=observed)["rejected"]:
        return mask, False

    def _is_rejected(m: np.ndarray) -> bool:
        unavailable = np.where(observed, np.where(m, 255, 0), 255).astype(np.uint8)
        return bool(visibility_check256(unavailable, observed=observed)["rejected"])

    if not _is_rejected(mask):
        return mask, False

    trimmed = mask.copy()
    dist_from_center = np.hypot(_XX - 127.5, _YY - 137.0).astype(np.float32)
    occ_confidence = np.maximum(probability, 1.0 - px) if px is not None else probability.astype(np.float32)
    outer_penalty = np.where((_YY > 224) | (~_FACE_ELLIPSE), np.float32(0.35), np.float32(0.0))
    priority = occ_confidence - np.float32(0.0025) * dist_from_center - outer_penalty

    # Stage 1: peel outer peripheral / lower-neck pixels outside core face ellipse first.
    outer_slice = trimmed & ((_YY > 226) | (~_FACE_ELLIPSE))
    if outer_slice.any():
        candidate = trimmed & (~outer_slice)
        if candidate.any():
            trimmed = candidate
            if not _is_rejected(trimmed):
                return trimmed, True

    # Stage 2: drop lowest-priority peripheral slices (15% per step) before resorting to uniform erosion.
    for _ in range(64):
        if not trimmed.any() or not _is_rejected(trimmed):
            break
        active_scores = priority[trimmed]
        cutoff = float(np.percentile(active_scores, 15.0))
        next_mask = trimmed & (priority > cutoff)
        if np.array_equal(next_mask, trimmed) or not next_mask.any():
            ys, xs = np.nonzero(trimmed)
            keep = max(0, len(ys) - max(1, len(ys) // 5))
            if keep == 0:
                trimmed[:] = False
                break
            order = np.argsort(active_scores)[-keep:]
            next_mask = np.zeros_like(trimmed)
            next_mask[ys[order], xs[order]] = True
        trimmed = next_mask

    if _is_rejected(trimmed):
        trimmed[:] = False
    return trimmed, True


def _is_ordinary_entry(entry: Path, *, directory: bool) -> bool:
    """Return True if path exists as an ordinary file/dir without symlink or reparse attributes."""
    try:
        info = entry.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
        return False
    return stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)


def _verify_onnx_file(path: Path, expected_bytes: int, expected_sha256: str) -> bool:
    """Verify an ordinary ONNX model file and ancestors match pinned size and SHA-256."""
    try:
        target = Path(os.path.abspath(path))
        if not _is_ordinary_entry(target, directory=False):
            return False
        stop_root = Path(os.path.abspath(ROOT.parent))
        for ancestor in target.parents:
            if not _is_ordinary_entry(ancestor, directory=True):
                return False
            if ancestor in (ROOT, stop_root):
                break
        before = target.stat()
        if before.st_size != expected_bytes:
            return False
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = target.stat()
        return (
            (before.st_size, before.st_mtime_ns, before.st_ino)
            == (after.st_size, after.st_mtime_ns, after.st_ino)
            and digest.hexdigest() == expected_sha256
        )
    except OSError:
        return False


def _resolve_occlusion_model_dir() -> Path | None:
    """Locate verified BiSeNet + XSeg ONNX assets across managed and repo roots."""
    candidates: list[Path] = []
    try:
        candidates.append(data_path(ROOT, "models", "occlusion"))
    except ManagedPathError:
        pass
    candidates.extend([ROOT / "models" / "occlusion", ROOT.parent / "models" / "occlusion"])
    seen: set[str] = set()
    for folder in candidates:
        key = os.path.normcase(os.path.abspath(folder))
        if key in seen:
            continue
        seen.add(key)
        if (
            _verify_onnx_file(folder / _BISENET_FILENAME, _BISENET_BYTES, _BISENET_SHA256)
            and _verify_onnx_file(folder / _XSEG_FILENAME, _XSEG_BYTES, _XSEG_SHA256)
        ):
            return folder
    return None


_BASE_ONNX_ERRORS: tuple[type[BaseException], ...] = (
    ImportError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    IndexError,
)


def _onnx_runtime_errors(ort_mod=None) -> tuple[type[BaseException], ...]:
    """Collect bounded ORT exceptions including pybind11 errors inheriting from Exception."""
    if ort_mod is None:
        cached = _ONNX_CACHE.get("ort_errors")
        if isinstance(cached, tuple) and cached:
            return cached
        try:
            import onnxruntime as ort_mod
        except _BASE_ONNX_ERRORS:
            return _BASE_ONNX_ERRORS

    errors: list[type[BaseException]] = list(_BASE_ONNX_ERRORS)
    capi = getattr(ort_mod, "capi", None)
    state = getattr(capi, "onnxruntime_pybind11_state", None)
    if state is None:
        try:
            from onnxruntime.capi import onnxruntime_pybind11_state as state
        except _BASE_ONNX_ERRORS:
            state = None
    if state is not None:
        for obj in vars(state).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, Exception)
                and obj is not Exception
                and obj not in errors
            ):
                errors.append(obj)
    result = tuple(errors)
    _ONNX_CACHE["ort_errors"] = result
    return result


def _load_onnx_sessions():
    """Lazily initialize and cache CPU ONNX Runtime sessions for BiSeNet and XSeg."""
    cached = _ONNX_CACHE.get("sessions")
    if cached is not None:
        return cached
    with _ONNX_LOCK:
        cached = _ONNX_CACHE.get("sessions")
        if cached is not None:
            return cached
        model_dir = _resolve_occlusion_model_dir()
        if model_dir is None:
            return None
        try:
            import onnxruntime as ort
        except _BASE_ONNX_ERRORS:
            return None
        ort_errors = _onnx_runtime_errors(ort)
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            bsess = ort.InferenceSession(str(model_dir / _BISENET_FILENAME), sess_options=opts, providers=["CPUExecutionProvider"])
            xsess = ort.InferenceSession(str(model_dir / _XSEG_FILENAME), sess_options=opts, providers=["CPUExecutionProvider"])
            xs_in = xsess.get_inputs()[0]
            bs_in = bsess.get_inputs()[0]
            bs_out = bsess.get_outputs()[0]
            xseg_nchw = len(xs_in.shape) == 4 and xs_in.shape[1] == 3
            sessions = (bsess, bs_in.name, bs_out.name, xsess, xs_in.name, xseg_nchw)
        except ort_errors:
            return None
        _ONNX_CACHE["sessions"] = sessions
        return sessions


def _run_onnx_models(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Run BiSeNet 19-class parsing and XSeg visible-face segmentation on RGB256."""
    sessions = _load_onnx_sessions()
    if sessions is None:
        return None
    bsess, bin_name, bout_name, xsess, xin_name, xseg_nchw = sessions
    ort_errors = _onnx_runtime_errors()

    try:
        img512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / np.float32(255.0)
        bx = ((img512 - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
        logits = bsess.run([bout_name], {bin_name: bx})[0][0]
        if not (isinstance(logits, np.ndarray) and logits.shape == (19, 512, 512)):
            return None
        logits = np.nan_to_num(logits.astype(np.float32, copy=False), nan=0.0, posinf=30.0, neginf=-30.0)
        exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
        denom = np.maximum(exp_logits.sum(axis=0, keepdims=True), np.float32(1e-8))
        prob512 = exp_logits / denom
        bprob = cv2.resize(prob512.transpose(1, 2, 0), (256, 256), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)

        x_img = rgb.astype(np.float32) / np.float32(255.0)
        x_in = x_img.transpose(2, 0, 1)[None] if xseg_nchw else x_img[None]
        px_raw = xsess.run(None, {xin_name: x_in})[0][0]
        px = px_raw[0] if xseg_nchw else px_raw[..., 0]
        if not (isinstance(px, np.ndarray) and px.shape == (256, 256)):
            return None
        px = np.nan_to_num(px.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(px, 0.0, 1.0, out=px)
        return bprob, px
    except ort_errors:
        return None


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes inside connected components of a binary mask."""
    if not mask.any():
        return mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = mask.astype(np.uint8).copy()
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled > 0


def _detect_onnx_eyewear(
    bprob: np.ndarray,
    seg: np.ndarray,
    px: np.ndarray,
    bface: np.ndarray,
    observed: np.ndarray,
    skin: np.ndarray,
    both_eyes_visible: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split hollow clear-glasses rims (tight 2px) from opaque/tinted sunglasses (5px)."""
    clear_rim_mask = np.zeros((256, 256), dtype=bool)
    opaque_eye_mask = np.zeros((256, 256), dtype=bool)
    clear_glasses_zone = np.zeros((256, 256), dtype=bool)
    eye_zone = _EYE_ZONE & observed

    if int(((seg == 6) & eye_zone).sum()) >= 200:
        g_full = ((seg == 6) | (bprob[6] >= 0.35)) & eye_zone
        g_inner = cv2.erode(g_full.astype(np.uint8), _ELLIPSE_5PX) > 0
        g_mean_px = float(px[g_full].mean())
        inner_vis_frac = float((px[g_inner] >= 0.72).mean()) if g_inner.any() else 0.0
        # Clear prescription glasses show high XSeg visibility inside the lenses.
        if g_mean_px >= 0.56 or (g_mean_px >= 0.38 and inner_vis_frac >= 0.30):
            g_rim = g_full & (~(cv2.erode(g_full.astype(np.uint8), _ERODE_1PX) > 0))
            clear_rim_mask |= ((g_full & (px < 0.70)) | g_rim) & observed
            clear_glasses_zone |= g_full
        else:
            # Tinted/opaque sunglasses (e.g., aviators, reflective, dark): mask full lenses+frames.
            opaque_eye_mask |= _fill_mask_holes(g_full) & observed

    # XSeg eye-band sunglasses / split-lens detector (works even when lower face is also masked).
    bface_px = int(bface.sum())
    feats_px = int((np.isin(seg, (2, 3, 4, 5, 6, 10)) & observed).sum())
    vis_anchor = int((bface & (px >= 0.55)).sum())
    up_fh = float(px[58:84, 70:186].mean())
    low_ch = float(px[132:188, 70:186].mean())
    both_bands_visible = up_fh >= 0.65 and low_ch >= 0.65
    face_with_forehead = feats_px >= 100 and vis_anchor >= 1500 and up_fh >= 0.45
    if not both_eyes_visible and not clear_glasses_zone.any() and (both_bands_visible or face_with_forehead):
        sg_band = (_YY >= 86) & (_YY < 128) & (_XX >= 56) & (_XX < 200) & observed
        sg_not_hair = ~np.isin(seg, (17, 18))
        if bface_px >= 2000:
            sg_not_hair = sg_not_hair & (seg != 0)
        low_px = int((sg_band & sg_not_hair & (px < 0.50)).sum())
        if low_px >= 900:
            sg_cand = (_YY >= 82) & (_YY < 132) & (_XX >= 52) & (_XX < 204) & observed & sg_not_hair & (px < 0.60)
            n_c, lbl_c, st_c, cen_c = cv2.connectedComponentsWithStats(sg_cand.astype(np.uint8), connectivity=8)
            wide_kept = np.zeros((256, 256), dtype=bool)
            left_blobs: list[tuple[int, float, np.ndarray]] = []
            right_blobs: list[tuple[int, float, np.ndarray]] = []
            for idx in range(1, n_c):
                x, y, w, h, area = (int(st_c[idx, k]) for k in range(5))
                if area < 220:
                    continue
                comp = lbl_c == idx
                # Sunglasses/censor bars are not pure skin-colored motion blur.
                if float(skin[comp].mean()) >= 0.55:
                    continue
                cx, cy = float(cen_c[idx][0]), float(cen_c[idx][1])
                if w >= 82 and area >= 1000 and 78.0 <= cx <= 178.0:
                    wide_kept |= comp
                elif area >= 260 and w >= 0.65 * h and 76.0 <= cx <= 122.0 and 86.0 <= cy <= 126.0:
                    left_blobs.append((area, cy, comp))
                elif area >= 260 and w >= 0.65 * h and 134.0 <= cx <= 180.0 and 86.0 <= cy <= 126.0:
                    right_blobs.append((area, cy, comp))
            if wide_kept.any():
                opaque_eye_mask |= _fill_mask_holes(wide_kept) & observed & _FACE_MARGIN_ELLIPSE
            elif left_blobs and right_blobs:
                l_best = max(left_blobs, key=lambda item: item[0])
                r_best = max(right_blobs, key=lambda item: item[0])
                if abs(l_best[1] - r_best[1]) <= 20.0:
                    pair = l_best[2] | r_best[2]
                    if float(px[pair].mean()) < 0.45:
                        opaque_eye_mask |= _fill_mask_holes(pair) & observed & _FACE_MARGIN_ELLIPSE

    return clear_rim_mask, opaque_eye_mask, clear_glasses_zone


def _recover_ear_loops(
    lower_occ: np.ndarray,
    px: np.ndarray,
    seg: np.ndarray,
    skin: np.ndarray,
    delta_e: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Trace thin cheek-to-ear mask straps/ear-loops contiguous with a detected lower-face mask."""
    if not lower_occ.any():
        return np.zeros((256, 256), dtype=bool)
    corridor = (_YY >= 104) & (_YY < 198) & ((_XX >= 28) & (_XX < 104) | (_XX >= 152) & (_XX < 228)) & observed
    strap_cand = corridor & (~np.isin(seg, (17, 14, 15, 16))) & ((px < 0.52) | ((~skin) & (delta_e > 22.0)))
    if not strap_cand.any():
        return np.zeros((256, 256), dtype=bool)
    # Use 3x3 open only so 2-6px ear loops survive (never 9x9).
    strap_clean = cv2.morphologyEx(strap_cand.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_3PX) > 0
    anchor = cv2.dilate(lower_occ.astype(np.uint8), _ELLIPSE_5PX) > 0
    n_s, lbl_s, st_s, _ = cv2.connectedComponentsWithStats(strap_clean.astype(np.uint8), connectivity=8)
    recovered = np.zeros((256, 256), dtype=bool)
    for idx in range(1, n_s):
        area = int(st_s[idx, cv2.CC_STAT_AREA])
        if not (25 <= area <= 1800):
            continue
        comp = lbl_s == idx
        if (comp & anchor).any():
            recovered |= comp
    return recovered


def _detect_onnx_face_hull_occlusions(
    rgb: np.ndarray,
    lab: np.ndarray,
    skin: np.ndarray,
    delta_e: np.ndarray,
    seg: np.ndarray,
    px: np.ndarray,
    bface: np.ndarray,
    observed: np.ndarray,
    *,
    lips_mask: np.ndarray,
    lips_visible: bool,
    lips_y_min: int,
    eyes_mask: np.ndarray,
    eyes_visible: bool,
    both_eyes_visible: bool,
    eyes_y_min: int,
    beard_guard: bool,
    beard_suppress: np.ndarray,
    hairline_suppress: np.ndarray,
    clear_glasses_zone: np.ndarray,
    is_eye_slit_face: bool,
) -> tuple[np.ndarray, bool]:
    """Unified upper/middle/lower face-hull XSeg occlusion detector + niqab/balaclava recovery."""
    occ_mask = np.zeros((256, 256), dtype=bool)
    has_lower_occ = False

    vis_face = bface & (px >= 0.60)
    vis_face_px = int(vis_face.sum())

    # Branch A: Eye-slit niqab / heavily wrapped scarf (only eyes/bridge visible in upper band).
    if is_eye_slit_face:
        vis_eye = bface & (px >= 0.60) & (_YY >= 55) & (_YY < 145) & (_XX >= 50) & (_XX < 206)
        ys_v, xs_v = np.nonzero(vis_eye)
        if ys_v.size >= 100:
            y_eye_bottom = int(np.percentile(ys_v, 95.0))
            cx_v = float(np.median(xs_v))
            rx_v = max(56.0, min(80.0, float(np.percentile(xs_v, 97.0) - np.percentile(xs_v, 3.0)) * 0.85))
            lower_oval = (
                (((_XX - cx_v) / rx_v) ** 2 + ((_YY - 148.0) / 96.0) ** 2 <= 1.0)
                & (_YY >= max(108, y_eye_bottom + 2))
                & (_YY < 236)
                & observed
            )
            niqab_cand = lower_oval & (px < 0.35) & np.isin(seg, (16, 18, 0, 1))
            if int(niqab_cand.sum()) >= 1500:
                closed = cv2.morphologyEx(niqab_cand.astype(np.uint8), cv2.MORPH_CLOSE, _ELLIPSE_11PX) > 0
                occ_mask |= _fill_mask_holes(closed) & lower_oval & (~vis_eye)
                has_lower_occ = True
        return occ_mask, has_lower_occ

    if vis_face_px < 1000:
        return occ_mask, False

    # Require recognizable facial features (brows/eyes/glasses/nose) so ultra-blurred blobs or
    # distant full-body heads never hallucinate face coverings from XSeg blur decay.
    facial_features = np.isin(seg, (2, 3, 4, 5, 6, 10)) & observed
    if int(facial_features.sum()) < 100:
        return occ_mask, False

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F)[bface].var()) if bface.any() else 0.0

    # Dynamic face coordinates from visible face pixels.
    ys_f, xs_f = np.nonzero(bface)
    bf_x_min, bf_x_max = int(xs_f.min()), int(xs_f.max())
    bf_y_max = int(ys_f.max())
    ys_u, xs_u = np.nonzero(vis_face)
    cx = float(np.median(xs_u)) if xs_u.size >= 50 else 127.5
    rx = max(54.0, min(82.0, float(np.percentile(xs_u, 97.0) - np.percentile(xs_u, 3.0)) * 0.56)) if xs_u.size >= 50 else 72.0
    dynamic_oval = (((_XX - cx) / rx) ** 2 + ((_YY - 140.0) / 104.0) ** 2 <= 1.0) & _FACE_ELLIPSE & observed

    # Build convex hull of bface to bridge BiSeNet holes caused by white N95 / black masks / hands.
    face_hull = dynamic_oval.copy()
    pts = np.column_stack((xs_f, ys_f)).astype(np.int32)
    if pts.shape[0] >= 20:
        hull_pts = cv2.convexHull(pts)
        hull_u8 = np.zeros((256, 256), dtype=np.uint8)
        cv2.fillConvexPoly(hull_u8, hull_pts, 1)
        face_hull = ((hull_u8 > 0) | dynamic_oval) & _FACE_ELLIPSE & observed

    bface_eroded = cv2.erode(bface.astype(np.uint8), _ERODE_2PX) > 0
    # Exclude hair/ears/neck/collar by default; allow seg 0/16/18 inside face hull when px < 0.35 & no lips.
    non_face_ext = np.isin(seg, (7, 8, 9, 14, 15, 17))
    # Never include seg==18 hat above visible eyes.
    above_eye_hat = (seg == 18) & (_YY < (eyes_y_min + 4 if eyes_visible else 92))
    outer_collar_corners = (_YY >= 200) & ((_XX < 78) | (_XX >= 178)) & (~bface)

    occ_interior = bface_eroded & (px < 0.42)
    if clear_glasses_zone.any():
        # Clear prescription glasses are handled tightly by clear_rim_mask (never 5px opaque dilation).
        occ_interior &= ~clear_glasses_zone
    if not lips_visible and (int(occ_interior.sum()) >= 200 or (int(bface.sum()) >= 8000 and bf_y_max >= 165)):
        # Recover white N95 cones (seg=0/18), black masks (seg=18), scarves/bandanas (seg=16/18) inside face hull.
        lower_hull = face_hull & (_YY >= 112) & (~non_face_ext) & (~above_eye_hat) & (~outer_collar_corners)
        occ_hull = lower_hull & (px < 0.35)
    else:
        occ_hull = np.zeros((256, 256), dtype=bool)

    raw_occ = (occ_interior | occ_hull) & (~hairline_suppress) & (~outer_collar_corners) & observed
    if raw_occ.any():
        # Light 3x3 morphological open preserves thin microphones, fingers, and straps.
        opened = cv2.morphologyEx(raw_occ.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_3PX) > 0
        closed = (cv2.morphologyEx(opened.astype(np.uint8), cv2.MORPH_CLOSE, _ELLIPSE_7PX) > 0) & (face_hull | bface) & (~non_face_ext) & (~outer_collar_corners)
        if clear_glasses_zone.any():
            closed &= ~clear_glasses_zone
        n_l, lbl_l, st_l, cen_l = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
        for idx in range(1, n_l):
            x, y, w, h, area = (int(st_l[idx, k]) for k in range(5))
            if area < 320:
                continue
            comp = lbl_l == idx
            mean_px = float(px[comp].mean())
            if mean_px >= 0.44:
                continue
            cx_c, cy_c = float(cen_l[idx][0]), float(cen_l[idx][1])
            # Reject thin vertical sideburn/hair strips clinging to left/right bface border.
            if w < 38 and h > 2.1 * max(1, w) and (x <= bf_x_min + 12 or x + w >= bf_x_max - 12):
                continue
            # Reject bottom-frame neck/collar edge strips below the chin.
            if y >= 218 or (cy_c >= 228.0 and (y + h) >= 252):
                continue
            # Reject forehead hairline / bangs / crown / bindi strictly above visible eyes (or above eye center).
            if eyes_visible and ((y + h) <= eyes_y_min + 4 or cy_c < eyes_y_min - 6):
                continue
            if cy_c < 80.0 and (y + h) <= 104:
                continue
            # Reject small blurry skin highlights/shadows on soft degraded inputs.
            if lap_var < 35.0 and float(skin[comp].mean()) >= 0.80 and (area < 600 or mean_px >= 0.20):
                continue
            # Reject beard/mustache regions wrapping around intact visible lips + nose.
            if beard_guard and float(beard_suppress[comp].mean()) >= 0.45:
                if int((comp & lips_mask & (px < 0.45)).sum()) < 40:
                    continue
            # Reject BiSeNet-vs-XSeg outer-perimeter skin mismatches on blurry/degraded unoccluded faces.
            if beard_guard and float(skin[comp].mean()) >= 0.50:
                if both_eyes_visible or y <= 35 or y >= 145:
                    continue
            occ_mask |= comp

    # Full lower-face mask body completion & ear-loop recovery when lower face is occluded.
    lower_part = occ_mask & (_YY >= 122) & _FACE_ELLIPSE
    if int(lower_part.sum()) >= 500:
        has_lower_occ = True
        if not lips_visible:
            # Absorb contiguous low-px nose-to-chin mask pixels inside face oval and fill concavities/holes.
            expand_domain = (
                face_hull
                & (_YY >= 106)
                & (_YY < 236)
                & (~non_face_ext)
                & (~above_eye_hat)
                & (~outer_collar_corners)
                & (~(eyes_mask & (px >= 0.55)))
                & (~clear_glasses_zone)
                & ((px < 0.48) | ((~skin) & (delta_e > 20.0) & (_YY >= 125) & (px < 0.65)))
            )
            seed_dil = cv2.dilate(lower_part.astype(np.uint8), _ELLIPSE_9PX) > 0
            n_e, lbl_e, _, _ = cv2.connectedComponentsWithStats((expand_domain | lower_part).astype(np.uint8), connectivity=8)
            completed_lower = lower_part.copy()
            for idx in range(1, n_e):
                e_comp = lbl_e == idx
                if (e_comp & seed_dil).any():
                    completed_lower |= e_comp
            completed_lower = cv2.morphologyEx(completed_lower.astype(np.uint8), cv2.MORPH_CLOSE, _ELLIPSE_11PX) > 0
            completed_lower = _fill_mask_holes(completed_lower) & face_hull & (~non_face_ext) & (~outer_collar_corners) & (~(eyes_mask & (px >= 0.55))) & (~clear_glasses_zone)
            occ_mask |= completed_lower
            occ_mask |= _recover_ear_loops(completed_lower, px, seg, skin, delta_e, observed)

    # Branch B: Tight achromatic gray/white balaclava (XSeg fooled by face-conforming gray knit, e.g., scarf_02).
    if not lips_visible and int(lips_mask.sum()) < 40 and eyes_visible:
        chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
        eye_skin = _EYE_ZONE & bface & (px >= 0.60) & skin
        is_color_photo = int((eye_skin & (chroma >= 9.0)).sum()) >= 120
        if is_color_photo:
            lower_gray = _LOWER_ZONE & observed & (~skin) & (chroma < 9.5) & (~non_face_ext)
            if int(lower_gray.sum()) >= 2800:
                eye_protect = cv2.dilate((eyes_mask | eye_skin).astype(np.uint8), _ELLIPSE_5PX) > 0
                cloth = _FACE_ELLIPSE & observed & (~skin) & (chroma < 10.5) & (~non_face_ext) & (~eye_protect)
                cloth = cv2.morphologyEx(cloth.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_3PX) > 0
                cloth = cv2.morphologyEx(cloth.astype(np.uint8), cv2.MORPH_CLOSE, _ELLIPSE_11PX) > 0
                n_g, lbl_g, st_g, _ = cv2.connectedComponentsWithStats(cloth.astype(np.uint8), connectivity=8)
                for idx in range(1, n_g):
                    if int(st_g[idx, cv2.CC_STAT_AREA]) >= 1200:
                        g_comp = lbl_g == idx
                        if (g_comp & lower_gray).sum() >= 600:
                            occ_mask |= _fill_mask_holes(g_comp) & _FACE_ELLIPSE & (~eye_protect)
                            has_lower_occ = True

    return occ_mask & observed, has_lower_occ


def _gate_base_mask(
    base_mask: np.ndarray,
    probability: np.ndarray,
    seg: np.ndarray,
    px: np.ndarray,
    bface: np.ndarray,
    observed: np.ndarray,
    *,
    lips_visible: bool,
    eyes_visible: bool,
    eyes_y_min: int,
    beard_guard: bool,
    beard_suppress: np.ndarray,
    hairline_suppress: np.ndarray,
) -> np.ndarray:
    """Gate raw epoch-2 segmenter base_mask per component with BiSeNet + XSeg."""
    if not base_mask.any():
        return base_mask
    # Preserve explicit full-canvas overcoverage clamps (unit test visibility clamping contract).
    if float(base_mask.mean()) >= 0.50:
        return base_mask

    # Strip hair, ears, neck, and non-face cloth/bg outside low-px face oval.
    hard_exterior = np.isin(seg, (7, 8, 9, 14, 15, 17))
    above_eye_hat = (seg == 18) & (_YY < (eyes_y_min + 4 if eyes_visible else 92))
    bg_or_cloth = np.isin(seg, (0, 16, 18))
    allow_oval_cloth = bg_or_cloth & _FACE_ELLIPSE & (px < 0.42) & (~above_eye_hat) & (not lips_visible)
    cleaned = base_mask & observed & (~hard_exterior) & ((~bg_or_cloth) | allow_oval_cloth)
    if not cleaned.any():
        return np.zeros((256, 256), dtype=bool)

    kept = np.zeros((256, 256), dtype=bool)
    n_c, lbl_c, st_c, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    for idx in range(1, n_c):
        area = int(st_c[idx, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        comp = lbl_c == idx
        is_glasses = float((seg[comp] == 6).mean()) >= 0.35
        if not is_glasses and float(px[comp].mean()) >= 0.62:
            continue
        prob_core = int((probability[comp] >= 0.50).sum())
        if not is_glasses and prob_core < 150 and float((px[comp] < 0.45).mean()) < 0.35:
            continue
        if beard_guard and not is_glasses and float(beard_suppress[comp].mean()) >= 0.45:
            continue
        if float(hairline_suppress[comp].mean()) >= 0.55 and not is_glasses:
            continue
        kept |= comp
    return kept


def _absorb_opaque_fringe(
    opaque_mask: np.ndarray,
    lab: np.ndarray,
    ref_skin_lab: tuple[float, float, float],
    observed: np.ndarray,
    px: np.ndarray | None = None,
    seg: np.ndarray | None = None,
    protect_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply 5px elliptical anti-bleed dilation + 4px adaptive Lab/XSeg fringe absorption."""
    if not opaque_mask.any():
        return opaque_mask
    dilated = (cv2.dilate(opaque_mask.astype(np.uint8), _ELLIPSE_11PX) > 0) & observed
    ring = (cv2.dilate(dilated.astype(np.uint8), _ELLIPSE_9PX) > 0) & (~dilated) & observed & _FACE_MARGIN_ELLIPSE
    if protect_mask is not None:
        ring &= ~protect_mask
    if seg is not None:
        ring &= ~np.isin(seg, (7, 8, 9, 14, 15, 17))
    if not ring.any():
        return dilated

    l_ref, a_ref, b_ref = ref_skin_lab
    l_m = float(np.median(lab[..., 0][opaque_mask]))
    a_m = float(np.median(lab[..., 1][opaque_mask]))
    b_m = float(np.median(lab[..., 2][opaque_mask]))
    de_skin = np.sqrt((lab[..., 0] - l_ref) ** 2 + (lab[..., 1] - a_ref) ** 2 + (lab[..., 2] - b_ref) ** 2)
    de_mask = np.sqrt((lab[..., 0] - l_m) ** 2 + (lab[..., 1] - a_m) ** 2 + (lab[..., 2] - b_m) ** 2)
    lab_fringe = de_mask < (de_skin - 6.0)
    if px is not None:
        fringe = ring & ((px < 0.52) | (lab_fringe & (px < 0.72)))
    else:
        fringe = ring & lab_fringe
    result = dilated | fringe
    if protect_mask is not None:
        result &= ~protect_mask
    return result & observed


def detect_face_covering(
    rgb: np.ndarray,
    probability: np.ndarray,
    base_proposed: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Refine neural covering proposals with research ONNX + Lab/YCrCb occlusion detection."""
    rgb, probability, base_mask, observed = _validate_inputs(rgb, probability, base_proposed, observed)
    base_mask = base_mask & observed
    if not observed.any():
        return np.zeros((256, 256), dtype=np.uint8), {
            "auto_detected_pixels": 0,
            "families_triggered": [],
            "visibility_trimmed": False,
            "neural_seed_pixels": 0,
            "onnx_used": False,
        }

    inner_face = _INNER_FACE & observed
    face_ellipse = _FACE_ELLIPSE & observed
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.int16)
    skin = _skin_mask(lab, ycrcb)

    onnx_out = _run_onnx_models(rgb) if int(observed.sum()) >= 4000 else None
    onnx_used = onnx_out is not None

    if onnx_out is not None:
        bprob, px = onnx_out
        seg = bprob.argmax(axis=0)
        bface = np.isin(seg, (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)) & observed
        left_eye_mask = (seg == 4) & observed
        right_eye_mask = (seg == 5) & observed
        eyes_mask = left_eye_mask | right_eye_mask
        nose_mask = (seg == 10) & observed
        lips_mask = np.isin(seg, (12, 13)) & observed

        xseg_centered_face = (
            int((px >= 0.50).sum()) >= 12000
            and float(px[58:84, 70:186].mean()) >= 0.65
            and float(px[132:188, 70:186].mean()) >= 0.65
        )
        has_face = int(bface.sum()) >= 2000 or xseg_centered_face
        vis_eye_band = bface & (px >= 0.60) & (_YY >= 55) & (_YY < 145) & (_XX >= 50) & (_XX < 206)
        is_eye_slit_face = False
        if not has_face and int(vis_eye_band.sum()) >= 350 and int(lips_mask.sum()) < 80:
            ys_v, xs_v = np.nonzero(vis_eye_band)
            y_v_max = int(np.percentile(ys_v, 95.0))
            cx_v = float(np.median(xs_v))
            below = (_YY >= max(108, y_v_max + 4)) & (_YY < 232) & (_XX >= max(44, int(cx_v - 74))) & (_XX < min(212, int(cx_v + 74))) & observed
            if below.any() and float((px[below] < 0.30).mean()) >= 0.65 and float(np.isin(seg[below], (16, 18)).mean()) >= 0.55:
                is_eye_slit_face = True

        # Non-face / full-body uncropped scene guard: never fire heuristics on shirts/placards.
        if not has_face and not is_eye_slit_face and not base_mask.any() and float(probability.max()) < 0.35:
            return np.zeros((256, 256), dtype=np.uint8), {
                "auto_detected_pixels": 0,
                "families_triggered": [],
                "visibility_trimmed": False,
                "neural_seed_pixels": 0,
                "onnx_used": True,
            }

        eyes_visible = (int(eyes_mask.sum()) >= 60) and (float(px[eyes_mask].mean()) >= 0.55)
        both_eyes_visible = (
            int(left_eye_mask.sum()) >= 25
            and int(right_eye_mask.sum()) >= 25
            and float(px[left_eye_mask].mean()) >= 0.55
            and float(px[right_eye_mask].mean()) >= 0.55
        )
        eyes_y_min = int(np.nonzero(eyes_mask)[0].min()) if eyes_visible else 92
        nose_visible = (int(nose_mask.sum()) >= 250) and (float(px[nose_mask].mean()) >= 0.45)
        lips_visible = (
            (int(lips_mask.sum()) >= 140 and float(px[lips_mask].mean()) >= 0.45)
            or (int(lips_mask.sum()) >= 900 and float(px[lips_mask].mean()) >= 0.35)
        )
        lips_y_min = int(np.nonzero(lips_mask)[0].min()) if lips_visible else 160
        beard_guard = lips_visible and nose_visible

        # Lower-face beard/mustache zone below visible nose/lips.
        beard_suppress = np.zeros((256, 256), dtype=bool)
        if beard_guard:
            beard_suppress = (_YY >= max(132, lips_y_min - 10)) & (seg != 6)
        # Forehead hairline / bangs / crown / bindi zone strictly above visible eyes.
        hairline_suppress = np.zeros((256, 256), dtype=bool)
        if eyes_visible:
            hairline_suppress = (_YY < max(60, eyes_y_min - 1)) & (seg != 6)

        if int(bface.sum()) >= 2000:
            collar_suppress = np.isin(seg, (0, 7, 8, 9, 14, 15, 16, 17, 18)) & (~bface)
        else:
            collar_suppress = np.isin(seg, (7, 8, 9, 14, 15, 16, 17, 18)) & (~bface)
        # Never suppress inner face oval low-px cloth when lips are occluded on an actual face crop.
        facial_features_px = int((np.isin(seg, (2, 3, 4, 5, 6, 10)) & observed).sum())
        bf_y_max_all = int(np.nonzero(bface)[0].max()) if bface.any() else 0
        if not lips_visible and (is_eye_slit_face or (facial_features_px >= 100 and bf_y_max_all >= 165)):
            above_eye_hat = (seg == 18) & (_YY < (eyes_y_min + 4 if eyes_visible else 92))
            oval_covering = np.isin(seg, (0, 16, 18)) & _FACE_ELLIPSE & (px < 0.45) & (~above_eye_hat)
            collar_suppress &= ~oval_covering
        # Always protect outer lower-left/right shirt collars and adjacent corner spill outside core chin.
        collar_corner_seed = np.isin(seg, (0, 14, 15, 16)) & (_YY >= 200) & ((_XX < 78) | (_XX >= 178))
        collar_suppress |= collar_corner_seed
        if collar_corner_seed.any():
            collar_suppress |= (
                (cv2.dilate(collar_corner_seed.astype(np.uint8), _ELLIPSE_5PX) > 0)
                & (_YY >= 200)
                & ((_XX < 78) | (_XX >= 178))
            )
    else:
        bprob = px = seg = bface = None
        eyes_mask = nose_mask = lips_mask = np.zeros((256, 256), dtype=bool)
        has_face = is_eye_slit_face = False
        eyes_visible = both_eyes_visible = nose_visible = lips_visible = beard_guard = False
        eyes_y_min, lips_y_min = 92, 160
        beard_suppress = np.zeros((256, 256), dtype=bool)
        hairline_suppress = np.zeros((256, 256), dtype=bool)
        collar_suppress = np.zeros((256, 256), dtype=bool)

    l_ref, a_ref, b_ref = _reference_skin_lab(lab, skin, observed, inner_face, bface=bface, seg=seg, px=px)
    # Adaptive dark-skin tolerance so dark-skinned subjects aren't treated as ~skin.
    if l_ref < 118.0:
        dark_de = np.sqrt((lab[..., 0] - l_ref) ** 2 + 1.4 * (lab[..., 1] - a_ref) ** 2 + 1.4 * (lab[..., 2] - b_ref) ** 2)
        skin = skin | ((lab[..., 0] >= max(20.0, l_ref - 40.0)) & (dark_de < 24.0))

    delta_e = np.sqrt(
        (lab[..., 0] - l_ref) ** 2
        + 1.4 * (lab[..., 1] - a_ref) ** 2
        + 1.4 * (lab[..., 2] - b_ref) ** 2
    )

    families_triggered: list[str] = []
    clear_rim_mask = np.zeros((256, 256), dtype=bool)
    opaque_mask = np.zeros((256, 256), dtype=bool)

    # 1. Gated base_mask from frozen segmenter
    if onnx_used:
        gated_base = _gate_base_mask(
            base_mask,
            probability,
            seg,
            px,
            bface,
            observed,
            lips_visible=lips_visible,
            eyes_visible=eyes_visible,
            eyes_y_min=eyes_y_min,
            beard_guard=beard_guard,
            beard_suppress=beard_suppress,
            hairline_suppress=hairline_suppress,
        )
    else:
        gated_base = base_mask.copy()

    # Preserve full-canvas clamp contract directly if base_mask covers >=50% of canvas.
    is_full_canvas_override = float(base_mask.mean()) >= 0.50
    if is_full_canvas_override:
        final_mask = gated_base.copy()
    else:
        opaque_mask |= gated_base

    # 2. Gated neural seed expansion
    seed_allowed = (~collar_suppress) if onnx_used else None
    grown_seeds, seed_pixels = _expand_neural_seeds(
        lab,
        probability,
        inner_face,
        observed,
        px=px,
        allowed_mask=seed_allowed,
        beard_suppress=beard_suppress if beard_guard else None,
        hairline_suppress=hairline_suppress if eyes_visible else None,
    )
    if seed_pixels > 0:
        families_triggered.append("neural_seed")
        if is_full_canvas_override:
            final_mask |= grown_seeds
        else:
            opaque_mask |= grown_seeds

    # 3. ONNX eyewear + unified face-hull occlusion detection
    onnx_clear_eye = np.zeros((256, 256), dtype=bool)
    onnx_opaque_eye = np.zeros((256, 256), dtype=bool)
    clear_glasses_zone = np.zeros((256, 256), dtype=bool)
    onnx_hull = np.zeros((256, 256), dtype=bool)
    onnx_has_lower = False
    if onnx_used and (has_face or is_eye_slit_face):
        onnx_clear_eye, onnx_opaque_eye, clear_glasses_zone = _detect_onnx_eyewear(
            bprob, seg, px, bface, observed, skin, both_eyes_visible=both_eyes_visible
        )
        onnx_hull, onnx_has_lower = _detect_onnx_face_hull_occlusions(
            rgb,
            lab,
            skin,
            delta_e,
            seg,
            px,
            bface,
            observed,
            lips_mask=lips_mask,
            lips_visible=lips_visible,
            lips_y_min=lips_y_min,
            eyes_mask=eyes_mask,
            eyes_visible=eyes_visible,
            both_eyes_visible=both_eyes_visible,
            eyes_y_min=eyes_y_min,
            beard_guard=beard_guard,
            beard_suppress=beard_suppress,
            hairline_suppress=hairline_suppress,
            clear_glasses_zone=clear_glasses_zone,
            is_eye_slit_face=is_eye_slit_face,
        )

    # 4. Deterministic Lab/YCrCb occlusion families (gated by ONNX when active)
    eye_mask = _detect_eyewear_and_glare(
        lab, skin, observed, l_ref, seg=seg, px=px, both_eyes_visible=both_eyes_visible
    )
    if clear_glasses_zone.any():
        eye_mask &= ~clear_glasses_zone
    if eye_mask.any() or onnx_clear_eye.any() or onnx_opaque_eye.any():
        families_triggered.append("eyewear_or_glare")

    if onnx_used and (beard_guard or (not is_eye_slit_face and int((np.isin(seg, (2, 3, 4, 5, 6, 10)) & observed).sum()) < 100)):
        lower_mask = np.zeros((256, 256), dtype=bool)
    else:
        allowed_lower = (~collar_suppress) if onnx_used else None
        lower_mask = _detect_masks_and_scarves(lab, skin, inner_face, delta_e, l_ref, allowed_lower=allowed_lower)
    if lower_mask.any() or (onnx_has_lower and not lips_visible):
        families_triggered.append("mask_or_scarf")

    obj_mask = _detect_foreign_objects(
        lab,
        skin,
        inner_face,
        delta_e,
        a_ref,
        b_ref,
        px=px,
        seg=seg,
        beard_suppress=beard_suppress if beard_guard else None,
    )
    if obj_mask.any():
        families_triggered.append("foreign_object")

    if onnx_hull.any():
        families_triggered.append("hand_or_object")

    # 5. Anti-ghosting split dilation: 2px tight on clear prescription rims vs 5px + fringe on opaque coverings
    if not is_full_canvas_override:
        clear_rim_mask |= onnx_clear_eye
        opaque_mask |= onnx_opaque_eye | onnx_hull | eye_mask | lower_mask | obj_mask
        if onnx_used and collar_suppress.any():
            opaque_mask &= ~collar_suppress

        # Protect visible eyes inside clear prescription glasses from opaque fringe expansion.
        protect_clear_eyes = np.zeros((256, 256), dtype=bool)
        if clear_glasses_zone.any() and onnx_used:
            protect_clear_eyes = clear_glasses_zone & (px >= 0.65) & (~clear_rim_mask)
        if is_eye_slit_face and onnx_used:
            protect_clear_eyes |= bface & (px >= 0.60) & (_YY >= 55) & (_YY < 145)
        if onnx_used and collar_suppress.any():
            protect_clear_eyes |= collar_suppress
        if beard_guard:
            protect_clear_eyes |= beard_suppress

        if clear_rim_mask.any():
            clear_dilated = (cv2.dilate(clear_rim_mask.astype(np.uint8), _ELLIPSE_5PX) > 0) & observed
            if protect_clear_eyes.any():
                # Keep 1px rim dilation inward while protecting clear visible eyes inside the lenses.
                clear_inner_1px = (cv2.dilate(clear_rim_mask.astype(np.uint8), _ERODE_1PX) > 0) & observed
                clear_dilated = (clear_dilated & (~protect_clear_eyes)) | clear_inner_1px
        else:
            clear_dilated = np.zeros((256, 256), dtype=bool)

        if opaque_mask.any():
            opaque_dilated = _absorb_opaque_fringe(
                opaque_mask,
                lab,
                (l_ref, a_ref, b_ref),
                observed,
                px=px,
                seg=seg,
                protect_mask=protect_clear_eyes,
            )
            if onnx_used and collar_suppress.any():
                opaque_dilated &= ~collar_suppress
        else:
            opaque_dilated = np.zeros((256, 256), dtype=bool)

        final_mask = (clear_dilated | opaque_dilated) & observed

    final_mask &= observed
    final_mask, visibility_trimmed = _enforce_visibility(final_mask, observed, probability, px=px)

    summary = {
        "auto_detected_pixels": int(final_mask.sum()),
        "families_triggered": families_triggered,
        "visibility_trimmed": bool(visibility_trimmed),
        "neural_seed_pixels": int(seed_pixels),
        "onnx_used": bool(onnx_used),
    }
    return final_mask.astype(np.uint8) * 255, summary

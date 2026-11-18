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
_OPEN_9PX = np.ones((9, 9), np.uint8)
_ERODE_2PX = np.ones((5, 5), np.uint8)
_ERODE_1PX = np.ones((3, 3), np.uint8)
_GLARE_BORDER = np.ones((13, 13), np.uint8)
_SEED_RADIUS = np.ones((41, 41), np.uint8)


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
) -> tuple[float, float, float]:
    """Estimate upper/mid-face reference skin Lab color excluding hair/dark shadows."""
    ref_mask = _SKIN_SAMPLE_ZONE & observed & skin & (lab[..., 0] >= 80.0)
    if int(ref_mask.sum()) < 150:
        ref_mask = inner_face & skin & (lab[..., 0] >= 80.0)
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
) -> tuple[np.ndarray, int]:
    """Expand sub-threshold neural probability components into Lab-similar neighbors."""
    prob_cand = (probability >= np.float32(0.08)) & inner_face
    if not prob_cand.any():
        return np.zeros((256, 256), dtype=bool), 0

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(prob_cand.astype(np.uint8), connectivity=8)
    seed_mask = np.zeros((256, 256), dtype=bool)
    grown_total = np.zeros((256, 256), dtype=bool)

    for idx in range(1, num_labels):
        comp = labels == idx
        comp_prob = probability[comp]
        if float(comp_prob.max()) < 0.30 and int((comp_prob >= 0.15).sum()) < 180:
            continue
        seed_mask |= comp
        core = comp & (probability >= 0.15)
        ref_pts = core if core.any() else comp
        l_s = float(np.median(lab[..., 0][ref_pts]))
        a_s = float(np.median(lab[..., 1][ref_pts]))
        b_s = float(np.median(lab[..., 2][ref_pts]))
        de_seed = np.sqrt(
            (lab[..., 0] - l_s) ** 2
            + (lab[..., 1] - a_s) ** 2
            + (lab[..., 2] - b_s) ** 2
        )
        dist_zone = cv2.dilate(comp.astype(np.uint8), _SEED_RADIUS) > 0
        eligible = (comp | ((de_seed < 14.0) & dist_zone)) & inner_face
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
) -> np.ndarray:
    """Detect dark sunglasses lenses/frames and bright specular lens glare."""
    eye_zone = _EYE_ZONE & observed
    eye_mask = np.zeros((256, 256), dtype=bool)

    dark_cand = eye_zone & (lab[..., 0] < 82.0) & ((l_ref - lab[..., 0]) > 36.0)
    if dark_cand.any():
        n_d, lbl_d, st_d, _ = cv2.connectedComponentsWithStats(dark_cand.astype(np.uint8), connectivity=8)
        for idx in range(1, n_d):
            x, y, w, h, area = (int(st_d[idx, k]) for k in range(5))
            if area < 320 or h < 12 or w < 18 or w < 0.75 * h:
                continue
            comp = lbl_d == idx
            if float(skin[comp].mean()) >= 0.35:
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
) -> np.ndarray:
    """Detect lower-face surgical masks, cloth masks, and scarves."""
    lower_zone = _LOWER_ZONE & inner_face
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
) -> np.ndarray:
    """Detect clear non-skin foreign objects inside the central face."""
    center_zone = _CENTER_ZONE & inner_face
    chroma_diff = np.hypot(lab[..., 1] - a_ref, lab[..., 2] - b_ref)
    cand = center_zone & (~skin) & (delta_e > 32.0) & (chroma_diff > 18.0)
    obj_mask = np.zeros((256, 256), dtype=bool)
    if not cand.any():
        return obj_mask

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
        obj_mask |= comp
    return obj_mask


def _enforce_visibility(
    mask: np.ndarray,
    observed: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Iteratively erode/trim mask until visibility_check256 passes or mask empties."""
    empty_unavailable = np.where(observed, 0, 255).astype(np.uint8)
    if visibility_check256(empty_unavailable)["rejected"]:
        return mask, False

    def _is_rejected(m: np.ndarray) -> bool:
        unavailable = np.where(observed, np.where(m, 255, 0), 255).astype(np.uint8)
        return bool(visibility_check256(unavailable)["rejected"])

    if not _is_rejected(mask):
        return mask, False

    trimmed = mask.copy()
    dist_from_center = np.hypot(_XX - 127.5, _YY - 137.0).astype(np.float32)
    # Prefer retaining high-probability and central pixels while peeling peripheral ones.
    priority = probability - 0.002 * dist_from_center

    for _ in range(64):
        if not trimmed.any() or not _is_rejected(trimmed):
            break
        eroded = (cv2.erode(trimmed.astype(np.uint8), _ERODE_2PX, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
        if eroded.any() and not np.array_equal(eroded, trimmed):
            trimmed = eroded
            continue
        # Fallback: drop lowest-priority peripheral slice (20% of remaining pixels).
        active_scores = priority[trimmed]
        cutoff = float(np.percentile(active_scores, 20.0))
        next_mask = trimmed & (priority > cutoff)
        if np.array_equal(next_mask, trimmed):
            ys, xs = np.nonzero(trimmed)
            keep = max(0, len(ys) - max(1, len(ys) // 4))
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


def _detect_onnx_eyewear(
    bprob: np.ndarray,
    seg: np.ndarray,
    px: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Extract hollow clear-glasses frames and opaque sunglasses bands."""
    eye_mask = np.zeros((256, 256), dtype=bool)
    eye_zone = _EYE_ZONE & observed
    if int(((seg == 6) & eye_zone).sum()) >= 200:
        g_full = ((seg == 6) | (bprob[6] >= 0.35)) & eye_zone
        g_rim = g_full & (~(cv2.erode(g_full.astype(np.uint8), _ERODE_1PX) > 0))
        frames = (g_full & (px < 0.72)) | g_rim
        eye_mask |= (cv2.dilate(frames.astype(np.uint8), _ERODE_1PX) > 0) & observed

    sg_band = (_YY >= 88) & (_YY < 126) & (_XX >= 66) & (_XX < 190) & observed
    low_px = int((sg_band & (px < 0.50)).sum())
    up_fh = float(px[58:84, 70:186].mean())
    low_ch = float(px[132:188, 70:186].mean())
    bridge_x = float(px[95:116, 112:144].mean())
    if up_fh >= 0.70 and low_ch >= 0.72 and low_px >= 1400 and bridge_x <= 0.65:
        n_c, _, st_c, _ = cv2.connectedComponentsWithStats((sg_band & (px < 0.65)).astype(np.uint8), connectivity=8)
        max_w = max((int(st_c[idx, cv2.CC_STAT_WIDTH]) for idx in range(1, n_c)), default=0)
        if max_w >= 85:
            sg_cand = (_YY >= 84) & (_YY < 130) & (_XX >= 56) & (_XX < 200) & observed & (px < 0.62)
            n_s, lbl_s, st_s, _ = cv2.connectedComponentsWithStats(sg_cand.astype(np.uint8), connectivity=8)
            kept = np.zeros((256, 256), dtype=bool)
            for idx in range(1, n_s):
                if int(st_s[idx, cv2.CC_STAT_AREA]) >= 150:
                    kept |= (lbl_s == idx)
            if kept.any():
                eye_mask |= (cv2.dilate(kept.astype(np.uint8), _ERODE_1PX) > 0) & observed & _FACE_ELLIPSE
    return eye_mask


def _detect_onnx_hand_or_object(
    bface: np.ndarray,
    px: np.ndarray,
    exterior_full: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Extract hands, microphones, and lower-face occluders via BiSeNet hull vs XSeg."""
    hand_mask = np.zeros((256, 256), dtype=bool)
    bface_eroded = cv2.erode(bface.astype(np.uint8), _DILATE_3PX) > 0
    seed_lower = bface_eroded & (_YY >= 130) & (_YY < 236) & (_XX >= 65) & (_XX < 195) & (px < 0.25)
    if (
        int(seed_lower.sum()) < 2000
        or float(px[70:145, 70:186].mean()) < 0.68
        or float(px[165:215, 88:168].mean()) > 0.42
    ):
        return hand_mask

    upper_vis = bface & (px >= 0.5) & (_YY < 165)
    ys_u, xs_u = np.nonzero(upper_vis)
    if xs_u.size >= 50:
        cx = float(np.median(xs_u))
        rx = max(48.0, float(np.percentile(xs_u, 97.0) - np.percentile(xs_u, 3.0)) * 0.52)
    else:
        cx, rx = 127.5, 62.0
    jaw_ellipse = (((_XX - cx) / rx) ** 2 + ((_YY - 142.0) / 112.0) ** 2 <= 1.0) & (_YY >= 120)

    face_domain = (bface_eroded | (jaw_ellipse & (~exterior_full))) & observed
    occ_cand = face_domain & (px < 0.50) & (_YY >= 125)
    occ_opened = cv2.morphologyEx(occ_cand.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_9PX) > 0
    occ_closed = (cv2.morphologyEx(occ_opened.astype(np.uint8), cv2.MORPH_CLOSE, _DILATE_3PX) > 0) & face_domain
    n_lbl, labels, _, _ = cv2.connectedComponentsWithStats(occ_closed.astype(np.uint8), connectivity=8)
    for idx in range(1, n_lbl):
        comp = labels == idx
        if int((comp & seed_lower).sum()) >= 500:
            hand_mask |= comp
    if hand_mask.any():
        hand_mask = (cv2.dilate(hand_mask.astype(np.uint8), _DILATE_3PX) > 0) & observed & (~exterior_full)
    return hand_mask


def _detect_onnx_occlusions(
    rgb: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    """Detect eyewear and hand/object occlusions while identifying non-face exterior."""
    if int(observed.sum()) < 4000:
        return None
    onnx_out = _run_onnx_models(rgb)
    if onnx_out is None:
        return None
    bprob, px = onnx_out
    seg = bprob.argmax(axis=0)
    bface = np.isin(seg, (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)) & observed
    if int(bface.sum()) < 2500 and int((px >= 0.5).sum()) < 4000:
        return None

    exterior_full = (
        np.isin(seg, (7, 8, 9, 14, 15, 16, 17, 18))
        | (bprob[16] >= 0.25)
        | (bprob[14] >= 0.25)
        | (bprob[17] >= 0.30)
    )
    collar_suppress = (
        np.isin(seg, (0, 7, 8, 9, 14, 15, 16, 17, 18))
        & (~((_YY < 210) & (_XX >= 75) & (_XX < 181)))
    ) & (~bface)
    eye_mask = _detect_onnx_eyewear(bprob, seg, px, observed)
    hand_mask = _detect_onnx_hand_or_object(bface, px, exterior_full, observed)
    return eye_mask, hand_mask, collar_suppress, int(bface.sum()) >= 2500


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

    families_triggered: list[str] = []
    final_mask = base_mask.copy()

    grown_seeds, seed_pixels = _expand_neural_seeds(lab, probability, inner_face, observed)
    if seed_pixels > 0:
        families_triggered.append("neural_seed")
        final_mask |= grown_seeds

    l_ref, a_ref, b_ref = _reference_skin_lab(lab, skin, observed, inner_face)
    delta_e = np.sqrt(
        (lab[..., 0] - l_ref) ** 2
        + 1.4 * (lab[..., 1] - a_ref) ** 2
        + 1.4 * (lab[..., 2] - b_ref) ** 2
    )

    onnx_result = _detect_onnx_occlusions(rgb, observed)
    onnx_used = onnx_result is not None
    if onnx_result is not None:
        onnx_eye, onnx_hand, collar_suppress, bface_active = onnx_result
        if bface_active and collar_suppress.any() and float(base_mask.mean()) < 0.50:
            final_mask &= ~collar_suppress
    else:
        onnx_eye = np.zeros((256, 256), dtype=bool)
        onnx_hand = np.zeros((256, 256), dtype=bool)
        collar_suppress = np.zeros((256, 256), dtype=bool)
        bface_active = False

    eye_mask = _detect_eyewear_and_glare(lab, skin, observed, l_ref)
    if eye_mask.any() or onnx_eye.any():
        families_triggered.append("eyewear_or_glare")

    lower_mask = _detect_masks_and_scarves(lab, skin, inner_face, delta_e, l_ref)
    if lower_mask.any():
        families_triggered.append("mask_or_scarf")

    obj_mask = _detect_foreign_objects(lab, skin, inner_face, delta_e, a_ref, b_ref)
    if obj_mask.any():
        families_triggered.append("foreign_object")

    if onnx_hand.any():
        families_triggered.append("hand_or_object")

    heuristic_mask = eye_mask | lower_mask | obj_mask
    if heuristic_mask.any():
        heuristic_dilated = (cv2.dilate(heuristic_mask.astype(np.uint8), _DILATE_3PX) > 0) & face_ellipse
        if bface_active and collar_suppress.any():
            heuristic_dilated &= ~collar_suppress
        final_mask |= heuristic_dilated

    final_mask |= onnx_eye | onnx_hand
    final_mask &= observed
    final_mask, visibility_trimmed = _enforce_visibility(final_mask, observed, probability)

    summary = {
        "auto_detected_pixels": int(final_mask.sum()),
        "families_triggered": families_triggered,
        "visibility_trimmed": bool(visibility_trimmed),
        "neural_seed_pixels": int(seed_pixels),
        "onnx_used": bool(onnx_used),
    }
    return final_mask.astype(np.uint8) * 255, summary

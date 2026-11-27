"""Multi-pass SCRFD-2.5GF CPU face detector with strict integrity verification and dual ORT/OpenCV-DNN backends."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat
import threading

import numpy as np

SCRFD_FILENAME = "det_2.5g.onnx"
SCRFD_BYTES = 3292009
SCRFD_SHA256 = "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9"
SCRFD_INPUT_SIZE = (640, 640)
SCRFD_STRIDES = (8, 16, 32)
SCRFD_FMC = 3
SCRFD_NUM_ANCHORS = 2
SCRFD_OUT_NAMES = ("446", "466", "486", "449", "469", "489", "452", "472", "492")
AUTO_CROP_MARGIN_RATIO = 0.32
DETECTOR_TAG = "scrfd_2.5g"

ROOT = Path(__file__).resolve().parents[1]

_BASE_RUNTIME_ERRORS: tuple[type[BaseException], ...] = (
    ImportError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    IndexError,
)

_LOCK = threading.RLock()
_CACHE: dict[str, object] = {}
_CENTER_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_GAMMA_055_LUT: np.ndarray | None = None


def reset_scrfd_cache() -> None:
    """Clear cached detector backend and error tuples (intended for tests)."""
    with _LOCK:
        _CACHE.clear()


def _import_managed_paths():
    """Import data_path/ManagedPathError safely across package, script, and runpy modes."""
    if __package__:
        try:
            from .managed_paths import ManagedPathError, data_path
            return data_path, ManagedPathError
        except ImportError:
            pass
    try:
        from implementation.managed_paths import ManagedPathError, data_path
        return data_path, ManagedPathError
    except ImportError:
        pass
    try:
        from managed_paths import ManagedPathError, data_path
        return data_path, ManagedPathError
    except ImportError:
        return None, RuntimeError


def _is_ordinary_entry(entry: Path, *, directory: bool) -> bool:
    """Return True if path exists as an ordinary file/dir without symlink or Windows reparse attributes."""
    try:
        info = entry.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
        return False
    return stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)


def _verify_onnx_file(path: Path, expected_bytes: int, expected_sha256: str) -> bool:
    """Verify ordinary file + ancestor lstat, exact byte length, streaming SHA-256, and TOCTOU stat."""
    try:
        target = Path(os.path.abspath(path))
        if not _is_ordinary_entry(target, directory=False):
            return False
        stop_root = Path(os.path.abspath(ROOT.parent))
        root_abs = Path(os.path.abspath(ROOT))
        for ancestor in target.parents:
            if not _is_ordinary_entry(ancestor, directory=True):
                return False
            if ancestor in (root_abs, stop_root):
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


def _resolve_scrfd_model_path() -> Path | None:
    """Resolve and verify det_2.5g.onnx across managed data_path, ROOT, and ROOT.parent."""
    candidates: list[Path] = []
    data_path_fn, managed_err = _import_managed_paths()
    if data_path_fn is not None:
        try:
            candidates.append(data_path_fn(ROOT, "models", "detection"))
        except managed_err:
            pass
    candidates.extend([ROOT / "models" / "detection", ROOT.parent / "models" / "detection"])
    seen: set[str] = set()
    for folder in candidates:
        key = os.path.normcase(os.path.abspath(folder))
        if key in seen:
            continue
        seen.add(key)
        candidate = folder / SCRFD_FILENAME
        if _verify_onnx_file(candidate, SCRFD_BYTES, SCRFD_SHA256):
            return candidate
    return None


def _runtime_errors(ort_mod=None, cv2_mod=None) -> tuple[type[BaseException], ...]:
    """Collect bounded runtime exceptions including ORT pybind11 errors and cv2.error."""
    if ort_mod is None and cv2_mod is None:
        cached = _CACHE.get("runtime_errors")
        if isinstance(cached, tuple) and cached:
            return cached

    errors: list[type[BaseException]] = list(_BASE_RUNTIME_ERRORS)

    if cv2_mod is None:
        try:
            import cv2 as cv2_mod
        except _BASE_RUNTIME_ERRORS:
            cv2_mod = None
    cv2_err = getattr(cv2_mod, "error", None) if cv2_mod is not None else None
    if isinstance(cv2_err, type) and issubclass(cv2_err, Exception) and cv2_err not in errors:
        errors.append(cv2_err)

    if ort_mod is None:
        try:
            import onnxruntime as ort_mod
        except _BASE_RUNTIME_ERRORS:
            ort_mod = None
    if ort_mod is not None:
        capi = getattr(ort_mod, "capi", None)
        state = getattr(capi, "onnxruntime_pybind11_state", None)
        if state is None:
            try:
                from onnxruntime.capi import onnxruntime_pybind11_state as state
            except _BASE_RUNTIME_ERRORS:
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
    if ort_mod is not None or cv2_mod is not None:
        _CACHE["runtime_errors"] = result
    return result


def _load_backend():
    """Lazily initialize CPU SCRFD backend: prefer ONNXRuntime (4T), fall back to cv2.dnn when ORT is absent."""
    cached = _CACHE.get("backend")
    if cached is not None:
        return cached
    with _LOCK:
        cached = _CACHE.get("backend")
        if cached is not None:
            return cached
        model_path = _resolve_scrfd_model_path()
        if model_path is None:
            return None

        try:
            import cv2
        except _BASE_RUNTIME_ERRORS:
            return None

        ort = None
        try:
            import onnxruntime as ort
        except ImportError:
            ort = None
        except _BASE_RUNTIME_ERRORS:
            return None

        errors = _runtime_errors(ort_mod=ort, cv2_mod=cv2)

        if ort is not None:
            try:
                so = ort.SessionOptions()
                so.intra_op_num_threads = 4
                so.inter_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    str(model_path),
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                in_name = sess.get_inputs()[0].name
                out_names = tuple(o.name for o in sess.get_outputs())
                if len(out_names) != 9:
                    return None
                backend = ("ort", sess, in_name, out_names)
            except errors:
                return None
            _CACHE["backend"] = backend
            return backend

        # Isolated capture venv fallback (opencv-python-headless + numpy only)
        try:
            net = cv2.dnn.readNetFromONNX(str(model_path))
            out_names = tuple(SCRFD_OUT_NAMES)
            backend = ("cv2_dnn", net, "input.1", out_names)
        except errors:
            return None
        _CACHE["backend"] = backend
        return backend


def _anchor_centers(h: int, w: int, stride: int) -> np.ndarray:
    key = (h, w, stride)
    centers = _CENTER_CACHE.get(key)
    if centers is None:
        grid = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        grid = (grid * float(stride)).reshape((-1, 2))
        if SCRFD_NUM_ANCHORS > 1:
            grid = np.stack([grid] * SCRFD_NUM_ANCHORS, axis=1).reshape((-1, 2))
        _CENTER_CACHE[key] = grid
        centers = grid
    return centers


def _nms(dets: np.ndarray, thresh: float = 0.35) -> list[int]:
    if dets.size == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def _padded_face_crop(box: list[int], img_w: int, img_h: int, max_side: int = 1600) -> list[int]:
    x, y, w, h = map(int, box)
    img_w, img_h = int(img_w), int(img_h)
    margin = int(max(w, h) * AUTO_CROP_MARGIN_RATIO + 0.5)
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(img_w, x + w + margin)
    bottom = min(img_h, y + h + margin)
    return [left, top, min(max_side, right - left), min(max_side, bottom - top)]


def _run_forward(backend, blob: np.ndarray):
    kind, model, in_name, out_names = backend
    errors = _runtime_errors()
    try:
        if kind == "ort":
            outs = model.run(list(out_names), {in_name: blob})
        else:
            with _LOCK:
                model.setInput(blob)
                outs = model.forward(list(out_names))
    except errors:
        return None
    if outs is None or len(outs) != 9:
        return None
    sanitized = []
    for arr in outs:
        a = np.asarray(arr, dtype=np.float32)
        sanitized.append(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
    return sanitized


def _detect_raw(
    backend,
    bgr: np.ndarray,
    *,
    score_thresh: float = 0.40,
    nms_thresh: float = 0.40,
) -> tuple[np.ndarray, np.ndarray] | None:
    import cv2

    img_h, img_w = bgr.shape[:2]
    in_w, in_h = SCRFD_INPUT_SIZE
    im_ratio = float(img_h) / max(1.0, float(img_w))
    model_ratio = float(in_h) / float(in_w)
    if im_ratio > model_ratio:
        new_h = in_h
        new_w = max(1, int(round(new_h / im_ratio)))
    else:
        new_w = in_w
        new_h = max(1, int(round(new_w * im_ratio)))
    det_scale = float(new_h) / float(img_h)
    interp = cv2.INTER_AREA if det_scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    det_img = np.zeros((in_h, in_w, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = resized

    blob = cv2.dnn.blobFromImage(
        det_img,
        1.0 / 128.0,
        (in_w, in_h),
        (127.5, 127.5, 127.5),
        swapRB=True,
    )
    outs = _run_forward(backend, blob)
    if outs is None:
        return None

    scores_list: list[np.ndarray] = []
    bboxes_list: list[np.ndarray] = []
    kpss_list: list[np.ndarray] = []
    for idx, stride in enumerate(SCRFD_STRIDES):
        fh, fw = in_h // stride, in_w // stride
        expected_n = fh * fw * SCRFD_NUM_ANCHORS
        scores = outs[idx].reshape(-1)
        bbox_raw = outs[idx + SCRFD_FMC].reshape(-1, 4)
        kps_raw = outs[idx + SCRFD_FMC * 2].reshape(-1, 10)
        if scores.shape[0] != expected_n or bbox_raw.shape[0] != expected_n or kps_raw.shape[0] != expected_n:
            return None

        bbox_preds = bbox_raw * float(stride)
        kps_preds = kps_raw * float(stride)
        centers = _anchor_centers(fh, fw, stride)
        pos_inds = np.where(scores >= score_thresh)[0]
        if pos_inds.size == 0:
            continue
        pos_scores = scores[pos_inds]
        pos_c = centers[pos_inds]
        pb = bbox_preds[pos_inds]
        x1 = pos_c[:, 0] - pb[:, 0]
        y1 = pos_c[:, 1] - pb[:, 1]
        x2 = pos_c[:, 0] + pb[:, 2]
        y2 = pos_c[:, 1] + pb[:, 3]
        bboxes = np.stack([x1, y1, x2, y2], axis=-1)
        scores_list.append(pos_scores[:, None])
        bboxes_list.append(bboxes)

        pk = kps_preds[pos_inds]
        kps = np.empty((pos_inds.size, 5, 2), dtype=np.float32)
        for k in range(5):
            kps[:, k, 0] = pos_c[:, 0] + pk[:, 2 * k]
            kps[:, k, 1] = pos_c[:, 1] + pk[:, 2 * k + 1]
        kpss_list.append(kps)

    if not scores_list:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

    all_scores = np.vstack(scores_list)
    all_bboxes = np.vstack(bboxes_list) / det_scale
    all_kpss = np.vstack(kpss_list) / det_scale
    dets = np.hstack([all_bboxes, all_scores]).astype(np.float32)
    keep = _nms(dets, thresh=nms_thresh)
    return dets[keep], all_kpss[keep]


def _detect_on_padded(
    backend,
    bgr: np.ndarray,
    *,
    pad_frac: float,
    score_thresh: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    import cv2

    h, w = bgr.shape[:2]
    ph = int(round(h * pad_frac))
    pw = int(round(w * pad_frac))
    padded = cv2.copyMakeBorder(bgr, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
    res = _detect_raw(backend, padded, score_thresh=score_thresh)
    if res is None:
        return None
    dets, kpss = res
    if len(dets) > 0:
        dets = dets.copy()
        dets[:, [0, 2]] -= pw
        dets[:, [1, 3]] -= ph
        kpss = kpss.copy()
        kpss[:, :, 0] -= pw
        kpss[:, :, 1] -= ph
    return dets, kpss


def _gamma_lut() -> np.ndarray:
    global _GAMMA_055_LUT
    if _GAMMA_055_LUT is None:
        _GAMMA_055_LUT = np.clip(
            np.rint(((np.arange(256, dtype=np.float32) / 255.0) ** 0.55) * 255.0),
            0,
            255,
        ).astype(np.uint8)
    return _GAMMA_055_LUT


def scan_scrfd_face_records(bgr_image: np.ndarray) -> list[dict] | None:
    """Multi-pass SCRFD-2.5GF face scan returning capture_worker-compatible dicts, or None on backend failure."""
    if bgr_image is None or getattr(bgr_image, "ndim", 0) != 3 or bgr_image.shape[2] != 3:
        return []
    img_h, img_w = bgr_image.shape[:2]
    if min(img_w, img_h) < 16:
        return []

    backend = _load_backend()
    if backend is None:
        return None

    try:
        import cv2
    except _BASE_RUNTIME_ERRORS:
        return None

    errors = _runtime_errors(cv2_mod=cv2)
    try:
        all_d: list[np.ndarray] = []
        all_k: list[np.ndarray] = []

        def _add(d: np.ndarray, k: np.ndarray, bias: float) -> None:
            if len(d) > 0:
                d2 = d.copy()
                d2[:, 4] = np.clip(d2[:, 4] + bias, 0.0, 1.0)
                all_d.append(d2)
                all_k.append(k)

        # Pass 0: native 640x640 (thresh=0.40)
        p0 = _detect_raw(backend, bgr_image, score_thresh=0.40)
        if p0 is None:
            return None
        d0, k0 = p0
        _add(d0, k0, 0.02)
        max_sc = float(d0[:, 4].max()) if len(d0) > 0 else 0.0

        # Pass 1a: 50% replicate-pad for tight close-up portraits
        if max_sc < 0.72:
            p1 = _detect_on_padded(backend, bgr_image, pad_frac=0.50, score_thresh=0.40)
            if p1 is None:
                return None
            d1, k1 = p1
            _add(d1, k1, 0.0)
            if len(d1) > 0:
                max_sc = max(max_sc, float(d1[:, 4].max()))

        # Pass 1b: mild Gaussian smooth + 50% replicate-pad for coarse mosaics
        if max_sc < 0.65:
            sigma = max(1.8, 0.012 * max(img_w, img_h))
            sm = cv2.GaussianBlur(bgr_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            p1s = _detect_on_padded(backend, sm, pad_frac=0.50, score_thresh=0.35)
            if p1s is None:
                return None
            d1s, k1s = p1s
            _add(d1s, k1s, -0.02)
            if len(d1s) > 0:
                max_sc = max(max_sc, float(d1s[:, 4].max()))

        # Pass 2a: gamma-0.55 dark lift for dark/backlit CCTV
        if max_sc < 0.55:
            lifted = cv2.LUT(bgr_image, _gamma_lut())
            p2 = _detect_raw(backend, lifted, score_thresh=0.30)
            if p2 is None:
                return None
            d2, k2 = p2
            _add(d2, k2, -0.04)
            if len(d2) > 0:
                max_sc = max(max_sc, float(d2[:, 4].max()))

        # Pass 2b: heavy smooth + 60% replicate-pad for extreme coarse block mosaics
        if max_sc < 0.45:
            sigma2 = max(3.5, 0.024 * max(img_w, img_h))
            sm2 = cv2.GaussianBlur(bgr_image, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
            p2s = _detect_on_padded(backend, sm2, pad_frac=0.60, score_thresh=0.28)
            if p2s is None:
                return None
            d2s, k2s = p2s
            _add(d2s, k2s, -0.05)

        if not all_d:
            return []

        dets = np.vstack(all_d)
        kpss = np.vstack(all_k)
        keep = _nms(dets, thresh=0.35)
        dets, kpss = dets[keep], kpss[keep]

        records: list[dict] = []
        for i in range(len(dets)):
            x1, y1, x2, y2, sc = [float(v) for v in dets[i]]
            if not all(map(math.isfinite, (x1, y1, x2, y2, sc))) or sc <= 0.0:
                continue
            left = max(0, int(math.floor(x1)))
            # Minimal 1.5% upper-forehead crown headroom without intruding into sharp hairline.
            forehead_pad = max(0.0, 0.015 * max(0.0, y2 - y1))
            top = max(0, int(math.floor(y1 - forehead_pad)))
            right = min(img_w, int(math.ceil(x2)))
            bottom = min(img_h, int(math.ceil(y2)))
            bw, bh = right - left, bottom - top
            if min(bw, bh) < 16:
                continue
            pts = kpss[i]
            landmarks = [[round(float(pts[k, 0]), 1), round(float(pts[k, 1]), 1)] for k in range(5)]
            re_pt, le_pt = landmarks[0], landmarks[1]
            roll_deg = round(math.degrees(math.atan2(le_pt[1] - re_pt[1], le_pt[0] - re_pt[0])), 2)
            if not math.isfinite(roll_deg):
                roll_deg = 0.0
            box = [left, top, bw, bh]
            crop = _padded_face_crop(box, img_w, img_h)
            records.append({
                "box": box,
                "crop": crop,
                "score": round(float(sc), 4),
                "landmarks": landmarks,
                "roll_deg": float(roll_deg),
                "detector": DETECTOR_TAG,
            })
            if len(records) == 100:
                break
        return records
    except errors:
        return None

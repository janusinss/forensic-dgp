"""eDifFIQA-T CPU face image quality assessment (FIQA) scorer with strict integrity checks and Laplacian fallback."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat
import threading
from typing import Sequence

import numpy as np

FIQA_FILENAME = "ediffiqa_tiny_jun2024.onnx"
FIQA_BYTES = 7272678
FIQA_SHA256 = "9426c899cc0f01665240cb7d9e7f98e18e24e456c178326c771a43da289bfc6a"
FIQA_INPUT_SIZE = (112, 112)

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


def reset_fiqa_cache() -> None:
    """Clear cached ORT session and error tuples (intended for tests)."""
    with _LOCK:
        _CACHE.clear()


def _import_managed_paths():
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


def _resolve_fiqa_model_path() -> Path | None:
    """Resolve and verify ediffiqa_tiny_jun2024.onnx across managed data_path, ROOT, and ROOT.parent."""
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
        candidate = folder / FIQA_FILENAME
        if _verify_onnx_file(candidate, FIQA_BYTES, FIQA_SHA256):
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


def _load_fiqa_session():
    """Thread-safe lazy CPU ONNXRuntime loader for eDifFIQA-T."""
    cached = _CACHE.get("session")
    if cached is not None:
        return cached
    with _LOCK:
        cached = _CACHE.get("session")
        if cached is not None:
            return cached
        model_path = _resolve_fiqa_model_path()
        if model_path is None:
            return None
        try:
            import onnxruntime as ort
        except _BASE_RUNTIME_ERRORS:
            return None
        errors = _runtime_errors(ort_mod=ort)
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
            out_name = sess.get_outputs()[0].name
            pair = (sess, in_name, out_name)
        except errors:
            return None
        _CACHE["session"] = pair
        return pair


def _validate_rgb_patch(rgb) -> np.ndarray | None:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    if rgb.shape[0] < 4 or rgb.shape[1] < 4:
        return None
    if rgb.dtype != np.uint8:
        arr = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return rgb


def _laplacian_variance(rgb: np.ndarray) -> float:
    try:
        import cv2
    except _BASE_RUNTIME_ERRORS:
        return 0.0
    errors = _runtime_errors(cv2_mod=cv2)
    try:
        resized = cv2.resize(
            rgb,
            FIQA_INPUT_SIZE,
            interpolation=cv2.INTER_AREA if min(rgb.shape[:2]) > 112 else cv2.INTER_LINEAR,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
        val = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        return val if math.isfinite(val) else 0.0
    except errors:
        return 0.0


def score_face_crops_fiqa(rgb_list: Sequence[np.ndarray]) -> list[float | None]:
    """Score a batch of RGB face crops in a single batched ORT call; returns None entries on failure."""
    items = list(rgb_list)
    if not items:
        return []
    session = _load_fiqa_session()
    if session is None:
        return [None] * len(items)

    try:
        import cv2
    except _BASE_RUNTIME_ERRORS:
        return [None] * len(items)

    sess, in_name, out_name = session
    errors = _runtime_errors(cv2_mod=cv2)

    valid_indices: list[int] = []
    tensors: list[np.ndarray] = []
    try:
        for idx, rgb in enumerate(items):
            patch = _validate_rgb_patch(rgb)
            if patch is None:
                continue
            interp = cv2.INTER_AREA if min(patch.shape[:2]) > 112 else cv2.INTER_LINEAR
            resized = cv2.resize(patch, FIQA_INPUT_SIZE, interpolation=interp)
            x = ((resized.astype(np.float32) / np.float32(255.0)) - np.float32(0.5)) / np.float32(0.5)
            tensors.append(np.moveaxis(x, -1, 0))
            valid_indices.append(idx)

        if not tensors:
            return [None] * len(items)

        batch = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        raw_out = sess.run([out_name], {in_name: batch})[0]
        arr = np.asarray(raw_out, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(valid_indices):
            return [None] * len(items)
        sanitized = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        results: list[float | None] = [None] * len(items)
        for out_i, orig_i in enumerate(valid_indices):
            val = float(sanitized[out_i])
            results[orig_i] = val if math.isfinite(val) else 0.0
        return results
    except errors:
        return [None] * len(items)


def score_face_crop_fiqa(rgb: np.ndarray) -> float | None:
    """Score a single RGB face crop using eDifFIQA-T, returning None when unavailable or invalid."""
    scores = score_face_crops_fiqa([rgb])
    return scores[0] if scores else None


def _extract_frame_patch(entry) -> tuple[dict, np.ndarray | None]:
    if isinstance(entry, dict):
        meta = dict(entry)
        patch = entry.get("patch")
        if patch is not None:
            return meta, _validate_rgb_patch(patch)
        frame = entry.get("frame")
        if frame is None:
            frame = entry.get("rgb")
        if frame is None:
            frame = entry.get("image")
        img = _validate_rgb_patch(frame)
        if img is None:
            return meta, None
        box = entry.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                vals = [float(v) for v in box]
                if not all(math.isfinite(v) for v in vals):
                    return meta, None
                bx, by, bw, bh = [int(v) for v in vals]
            except (TypeError, ValueError, OverflowError):
                return meta, None
            h, w = img.shape[:2]
            x0 = max(0, min(w - 1, bx))
            y0 = max(0, min(h - 1, by))
            x1 = max(x0 + 1, min(w, bx + max(1, bw)))
            y1 = max(y0 + 1, min(h, by + max(1, bh)))
            return meta, _validate_rgb_patch(img[y0:y1, x0:x1])
        return meta, img

    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        frame, box = entry[0], entry[1]
        return _extract_frame_patch({"frame": frame, "box": box})
    return {}, None


def rank_video_frames_fiqa(frames_with_boxes: Sequence) -> list[dict]:
    """Rank video frames for Lucky-Frame selection using eDifFIQA-T primary and Laplacian variance fallback."""
    entries = list(frames_with_boxes)
    if not entries:
        return []

    metas: list[dict] = []
    patches: list[np.ndarray | None] = []
    for item in entries:
        meta, patch = _extract_frame_patch(item)
        metas.append(meta)
        patches.append(patch)

    fiqa_scores = score_face_crops_fiqa(patches)
    use_fiqa = any(s is not None for s in fiqa_scores)

    ranked: list[dict] = []
    for idx, (meta, patch, fscore) in enumerate(zip(metas, patches, fiqa_scores)):
        lap = _laplacian_variance(patch) if patch is not None else 0.0
        if use_fiqa and fscore is not None:
            q_score = float(fscore)
            scorer = "ediffiqa_tiny"
        elif use_fiqa:
            q_score = -1e6
            scorer = "ediffiqa_tiny"
        else:
            q_score = float(lap)
            scorer = "laplacian_fallback"

        out_row = {
            **meta,
            "index": idx,
            "fiqa_score": None if fscore is None else round(float(fscore), 4),
            "laplacian_var": round(float(lap), 2),
            "quality_score": round(float(q_score), 4),
            "scorer": scorer,
        }
        ranked.append(out_row)

    ranked.sort(key=lambda r: (-float(r["quality_score"]), int(r["index"])))
    return ranked

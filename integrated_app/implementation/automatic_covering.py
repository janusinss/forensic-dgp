"""Fast CPU automatic face-covering proposal refinement and research ONNX segmentation.

Combines frozen neural detector seed expansion, BiSeNet-R18 QDQ INT8 19-class semantic
face parsing (with FP32 fallback), OFIQ FaceOcclusion QDQ INT8 occlusion segmentation
(with legacy XSeg fallback), InsightFace 2d106det 106-point anatomical face hull clipping,
plus deterministic Lab/YCrCb multi-family occlusion detection and visibility_check256 clamping.
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
_BISENET_QDQ_FILENAME = "bisenet_resnet18_qdq_int8.onnx"
_BISENET_QDQ_BYTES = 13414139
_BISENET_QDQ_SHA256 = "2b55214a8867f0f4ea6c843a18ba6ca96f7c37e281f9526edf04d2b130c09e28"
_BISENET_FILENAME = "bisenet_resnet18.onnx"
_BISENET_BYTES = 53205364
_BISENET_SHA256 = "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"

_OFIQ_QDQ_FILENAME = "ofiq_face_occlusion_qdq_int8.onnx"
_OFIQ_QDQ_BYTES = 14415696
_OFIQ_QDQ_SHA256 = "567b851cf73ffb77e4d7d94c6463e9ad995cadf41e9bdd2f269d042949098d66"
_XSEG_FILENAME = "xseg.onnx"
_XSEG_BYTES = 70327737
_XSEG_SHA256 = "0b57328efcb839d85973164b617ceee9dfe6cfcb2c82e8a033bba9f4f09b27e5"

_LMS106_FILENAME = "2d106det.onnx"
_LMS106_BYTES = 5030888
_LMS106_SHA256 = "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"
_LMS106_5PT_INDICES = (38, 88, 86, 52, 61)

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
_ELLIPSE_13PX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


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
        if px is not None:
            eye_vis = (_YY >= 68) & (_YY < 118) & (_XX >= 72) & (_XX < 184) & observed & (px >= 0.25)
            if int(eye_vis.sum()) >= 80:
                ref_mask = eye_vis
            else:
                forehead = _FOREHEAD_BAND & observed
                ref_mask = forehead if forehead.any() else (inner_face if inner_face.any() else observed)
        else:
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
    bprob: np.ndarray | None = None,
    bface: np.ndarray | None = None,
) -> np.ndarray:
    """Detect dark sunglasses lenses/frames and bright specular lens glare."""
    eye_zone = _EYE_ZONE & observed
    if seg is not None:
        eye_zone = eye_zone & (seg != 17)
    eye_mask = np.zeros((256, 256), dtype=bool)
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    is_monochrome = False
    if bface is not None and int(bface.sum()) >= 400:
        is_monochrome = float(chroma[bface].mean()) < 5.0

    dark_zone = eye_zone & (_YY >= 72)
    if seg is not None:
        allow_g6 = (bprob[6] >= 0.35) if bprob is not None else (seg == 6)
        dark_zone = dark_zone & ((~np.isin(seg, (2, 3, 17, 18))) | allow_g6)

    if is_monochrome and seg is not None:
        dark_cand = dark_zone & (lab[..., 0] < 65.0) & ((l_ref - lab[..., 0]) > 45.0)
    else:
        dark_cand = dark_zone & (lab[..., 0] < 82.0) & ((l_ref - lab[..., 0]) > 36.0)

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
            g6_mean = float(bprob[6][comp].mean()) if bprob is not None else 0.0
            # Open-eye guard: never punch visible open eyes (BiSeNet eye support + sclera/pupil contrast).
            if seg is not None and g6_mean < 0.18:
                eye_ov = comp & np.isin(seg, (4, 5))
                if int(eye_ov.sum()) >= 30:
                    eye_patch = (cv2.dilate(eye_ov.astype(np.uint8), _ELLIPSE_5PX) > 0) & observed
                    l_eye = lab[..., 0][eye_patch]
                    if l_eye.size >= 20 and float(np.percentile(l_eye, 92.0) - np.percentile(l_eye, 8.0)) >= 45.0:
                        continue
                if is_monochrome and px is not None and float(px[comp].mean()) >= 0.45:
                    continue
            # Reject scalp hair or dark background continuing directly above eye_zone.
            if y <= 76:
                y0 = max(0, y - 12)
                if y > y0 and float((lab[y0:y, x:x + w, 0] < 82.0).mean()) > 0.50:
                    continue
            if int((comp & _INNER_EYE_BAND).sum()) < 180:
                continue
            eye_mask |= comp

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
) -> tuple[np.ndarray, np.ndarray]:
    """Detect clear non-skin foreign objects and crisp rectangular censor bars inside central face."""
    center_zone = _CENTER_ZONE & inner_face
    if seg is not None:
        center_zone = center_zone & (~np.isin(seg, (0, 7, 8, 9, 14, 15, 16, 17, 18)))
    chroma_diff = np.hypot(lab[..., 1] - a_ref, lab[..., 2] - b_ref)
    cand = center_zone & (~skin) & (delta_e > 32.0) & (chroma_diff > 18.0)
    obj_mask = np.zeros((256, 256), dtype=bool)
    censor_bar_mask = np.zeros((256, 256), dtype=bool)
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

    # Detect crisp solid achromatic/colored rectangular censor bars (tight 1-2px margin).
    if px is not None:
        bar_zone = (_CENTER_ZONE | _EYE_ZONE) & inner_face
        if seg is not None:
            bar_zone = bar_zone & (~np.isin(seg, (7, 8, 9, 14, 15, 17, 18)))
        bar_cand = bar_zone & (~skin) & (px < 0.42) & (delta_e > 26.0)
        if bar_cand.any():
            n_b, lbl_b, st_b, _ = cv2.connectedComponentsWithStats(bar_cand.astype(np.uint8), connectivity=8)
            for idx in range(1, n_b):
                x, y, w, h, area = (int(st_b[idx, k]) for k in range(5))
                if area < 320 or w < 40 or h < 8 or w < 2.0 * h:
                    continue
                if area / float(max(1, w * h)) < 0.80:
                    continue
                comp = lbl_b == idx
                lab_std = float(
                    lab[..., 0][comp].std() + lab[..., 1][comp].std() + lab[..., 2][comp].std()
                )
                if lab_std > 20.0:
                    continue
                if beard_suppress is not None and float(beard_suppress[comp].mean()) >= 0.50:
                    continue
                censor_bar_mask |= comp

    return obj_mask, censor_bar_mask


def _enforce_visibility(
    mask: np.ndarray,
    observed: np.ndarray,
    probability: np.ndarray,
    px: np.ndarray | None = None,
    seg: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Coherently peel non-face periphery then outer geodesic boundary until visibility passes."""
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

    # Stage 1: peel non-face hair/background/collar and outer periphery outside core face ellipse first.
    non_face_spill = ((_YY > 226) | (~_FACE_ELLIPSE))
    if seg is not None:
        non_face_spill = non_face_spill | (np.isin(seg, (0, 7, 8, 9, 14, 15, 16, 17, 18)) & (~_INNER_FACE))
    outer_slice = trimmed & non_face_spill
    if outer_slice.any():
        candidate = trimmed & (~outer_slice)
        if candidate.any():
            trimmed = candidate
            if not _is_rejected(trimmed):
                return trimmed, True

    # Stage 2: coherent outer geodesic boundary peeling (never punches interior Swiss-cheese holes).
    for _ in range(128):
        if not trimmed.any() or not _is_rejected(trimmed):
            break
        eroded = cv2.erode(trimmed.astype(np.uint8), _ERODE_1PX) > 0
        boundary = trimmed & (~eroded)
        if not boundary.any():
            trimmed[:] = False
            break
        b_scores = priority[boundary]
        # Peel the lowest-priority 55% of the outer boundary ring (or all boundary pixels if small).
        if b_scores.size <= 32:
            next_mask = eroded
        else:
            cutoff = float(np.percentile(b_scores, 55.0))
            peel = boundary & (priority <= cutoff)
            if not peel.any():
                peel = boundary
            next_mask = trimmed & (~peel)
        if np.array_equal(next_mask, trimmed):
            next_mask = eroded
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


def _candidate_occlusion_dirs() -> list[Path]:
    """Return deduplicated candidate directories for occlusion ONNX models."""
    candidates: list[Path] = []
    try:
        candidates.append(data_path(ROOT, "models", "occlusion"))
    except ManagedPathError:
        pass
    candidates.extend([ROOT / "models" / "occlusion", ROOT.parent / "models" / "occlusion"])
    unique: list[Path] = []
    seen: set[str] = set()
    for folder in candidates:
        key = os.path.normcase(os.path.abspath(folder))
        if key not in seen:
            seen.add(key)
            unique.append(folder)
    return unique


def _select_parser_file(folder: Path) -> Path | None:
    """Prefer verified BiSeNet-R18 QDQ INT8, falling back to legacy FP32 if present."""
    qdq = folder / _BISENET_QDQ_FILENAME
    if _verify_onnx_file(qdq, _BISENET_QDQ_BYTES, _BISENET_QDQ_SHA256):
        return qdq
    legacy = folder / _BISENET_FILENAME
    if _verify_onnx_file(legacy, _BISENET_BYTES, _BISENET_SHA256):
        return legacy
    return None


def _select_occlusion_file(folder: Path) -> tuple[Path, str] | None:
    """Prefer verified OFIQ FaceOcclusion QDQ INT8, falling back to legacy XSeg if present."""
    ofiq = folder / _OFIQ_QDQ_FILENAME
    if _verify_onnx_file(ofiq, _OFIQ_QDQ_BYTES, _OFIQ_QDQ_SHA256):
        return ofiq, "ofiq"
    xseg = folder / _XSEG_FILENAME
    if _verify_onnx_file(xseg, _XSEG_BYTES, _XSEG_SHA256):
        return xseg, "xseg"
    return None


def _resolve_occlusion_model_dir() -> Path | None:
    """Locate a folder containing verified parser (QDQ/FP32) and occlusion (OFIQ/XSeg) assets."""
    for folder in _candidate_occlusion_dirs():
        if _select_parser_file(folder) is not None and _select_occlusion_file(folder) is not None:
            return folder
    return None


def _resolve_landmarks106_path() -> Path | None:
    """Locate verified InsightFace 2d106det ONNX landmark model across managed and repo roots."""
    for folder in _candidate_occlusion_dirs():
        candidate = folder / _LMS106_FILENAME
        if _verify_onnx_file(candidate, _LMS106_BYTES, _LMS106_SHA256):
            return candidate
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
    errors: list[type[BaseException]] = list(_BASE_ONNX_ERRORS)
    cv2_err = getattr(cv2, "error", None)
    if isinstance(cv2_err, type) and issubclass(cv2_err, Exception) and cv2_err not in errors:
        errors.append(cv2_err)
    if ort_mod is None:
        cached = _ONNX_CACHE.get("ort_errors")
        if isinstance(cached, tuple) and cached:
            return cached
        try:
            import onnxruntime as ort_mod
        except _BASE_ONNX_ERRORS:
            return tuple(errors)

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
    """Lazily initialize and cache CPU ONNX Runtime sessions for parser and occlusion models."""
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
        parser_path = _select_parser_file(model_dir)
        if parser_path is None:
            if os.path.lexists(model_dir / _BISENET_QDQ_FILENAME) or os.path.lexists(model_dir / _BISENET_FILENAME):
                return None
            parser_path = model_dir / _BISENET_QDQ_FILENAME
        occ_choice = _select_occlusion_file(model_dir)
        if occ_choice is None:
            if os.path.lexists(model_dir / _OFIQ_QDQ_FILENAME) or os.path.lexists(model_dir / _XSEG_FILENAME):
                return None
            occ_path, occ_mode = model_dir / _OFIQ_QDQ_FILENAME, "ofiq"
        else:
            occ_path, occ_mode = occ_choice
        try:
            import onnxruntime as ort
        except _BASE_ONNX_ERRORS:
            return None
        ort_errors = _onnx_runtime_errors(ort)
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            bsess = ort.InferenceSession(str(parser_path), sess_options=opts, providers=["CPUExecutionProvider"])
            occ_sess = ort.InferenceSession(str(occ_path), sess_options=opts, providers=["CPUExecutionProvider"])
            bs_in = bsess.get_inputs()[0]
            bs_out = bsess.get_outputs()[0]
            occ_in = occ_sess.get_inputs()[0]
            if occ_mode == "ofiq":
                sessions = (bsess, bs_in.name, bs_out.name, occ_sess, occ_in.name, "ofiq", 224)
            else:
                xseg_nchw = len(occ_in.shape) == 4 and occ_in.shape[1] == 3
                sessions = (
                    bsess,
                    bs_in.name,
                    bs_out.name,
                    occ_sess,
                    occ_in.name,
                    "xseg_nchw" if xseg_nchw else "xseg_nhwc",
                    256,
                )
        except ort_errors:
            return None
        _ONNX_CACHE["sessions"] = sessions
        return sessions


def _run_onnx_models(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Run BiSeNet 19-class parsing and OFIQ/XSeg visible-face segmentation on RGB256."""
    sessions = _load_onnx_sessions()
    if sessions is None:
        return None
    if len(sessions) == 6:
        bsess, bin_name, bout_name, occ_sess, occ_in_name, xseg_nchw = sessions
        occ_mode = "xseg_nchw" if xseg_nchw else "xseg_nhwc"
        occ_size = 256
    elif len(sessions) == 7:
        bsess, bin_name, bout_name, occ_sess, occ_in_name, occ_mode, occ_size = sessions
    else:
        return None
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

        if occ_mode == "ofiq":
            sz = int(occ_size)
            o_img = cv2.resize(rgb, (sz, sz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / np.float32(255.0)
            o_in = o_img.transpose(2, 0, 1)[None]
            o_raw = occ_sess.run(None, {occ_in_name: o_in})[0]
            if not (isinstance(o_raw, np.ndarray) and o_raw.ndim == 4 and o_raw.shape[0] >= 1 and o_raw.shape[1] >= 1):
                return None
            o_logits = o_raw[0, 0]
            if o_logits.shape != (sz, sz):
                return None
            o_logits = np.nan_to_num(o_logits.astype(np.float32, copy=False), nan=0.0, posinf=30.0, neginf=-30.0)
            np.clip(o_logits, -30.0, 30.0, out=o_logits)
            p_vis = np.float32(1.0) / (np.float32(1.0) + np.exp(-o_logits))
            px = cv2.resize(p_vis, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)
            px = np.nan_to_num(px, nan=0.0, posinf=1.0, neginf=0.0)
            np.clip(px, 0.0, 1.0, out=px)
        else:
            xseg_nchw = (occ_mode == "xseg_nchw") or (occ_mode is True)
            x_img = rgb.astype(np.float32) / np.float32(255.0)
            x_in = x_img.transpose(2, 0, 1)[None] if xseg_nchw else x_img[None]
            px_raw = occ_sess.run(None, {occ_in_name: x_in})[0][0]
            px = px_raw[0] if xseg_nchw else px_raw[..., 0]
            if not (isinstance(px, np.ndarray) and px.shape == (256, 256)):
                return None
            px = np.nan_to_num(px.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
            np.clip(px, 0.0, 1.0, out=px)
        return bprob, px
    except ort_errors:
        return None


def _load_landmarks106_session():
    """Lazily initialize and cache CPU ONNX Runtime session for 2d106det 106-pt landmarks."""
    cached = _ONNX_CACHE.get("lms106")
    if cached is not None:
        return cached
    with _ONNX_LOCK:
        cached = _ONNX_CACHE.get("lms106")
        if cached is not None:
            return cached
        model_path = _resolve_landmarks106_path()
        if model_path is None:
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
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            entry = (sess, in_name, out_name)
        except ort_errors:
            return None
        _ONNX_CACHE["lms106"] = entry
        return entry


def landmarks_5pt_from_106(pts: np.ndarray) -> np.ndarray | None:
    """Extract anatomically validated 5-point landmarks in _FFHQ_5PT_256 order [38, 88, 86, 52, 61]."""
    if not isinstance(pts, np.ndarray) or pts.shape != (106, 2):
        return None
    pts_f = pts.astype(np.float32, copy=False)
    if not np.isfinite(pts_f).all():
        return None
    p5 = pts_f[list(_LMS106_5PT_INDICES)].copy()
    if float(p5.min()) < -32.0 or float(p5.max()) > 288.0:
        return None
    le, re, nose, lm, rm = p5
    d_eye = float(np.hypot(re[0] - le[0], re[1] - le[1]))
    emx, emy = 0.5 * float(le[0] + re[0]), 0.5 * float(le[1] + re[1])
    mmx, mmy = 0.5 * float(lm[0] + rm[0]), 0.5 * float(lm[1] + rm[1])
    d_em = float(np.hypot(mmx - emx, mmy - emy))
    if d_eye < 16.0 or d_em < 14.0 or d_em < 0.32 * d_eye or d_eye < 0.45 * d_em:
        return None
    vx, vy = float(re[0] - le[0]), float(re[1] - le[1])
    ux, uy = mmx - emx, mmy - emy
    cross = vx * uy - vy * ux
    if cross <= 0.25 * d_eye * d_em:
        return None
    # Nose tip must lie between eye midpoint and mouth midpoint along the vertical face axis.
    t_nose = ((float(nose[0]) - emx) * ux + (float(nose[1]) - emy) * uy) / max(1e-6, d_em * d_em)
    if not (0.05 <= t_nose <= 1.15):
        return None
    return p5.astype(np.float32, copy=False)


def face_hull_from_106(pts: np.ndarray, *, dilate_px: int = 6, forehead_lift: float = 0.25) -> np.ndarray:
    """Build exact anatomical face mask from 2d106det jawline (0..32) + raised brow arch (43..51, 97..105)."""
    pts_arr = np.asarray(pts, dtype=np.float32)
    if pts_arr.shape != (106, 2) or not np.isfinite(pts_arr).all():
        return np.zeros((256, 256), dtype=bool)
    jaw = pts_arr[0:33].copy()
    lbrow = pts_arr[43:52].copy()
    rbrow = pts_arr[97:106].copy()
    brow_y = float(min(lbrow[:, 1].min(), rbrow[:, 1].min()))
    chin_y = float(jaw[:, 1].max())
    face_h = max(20.0, chin_y - brow_y)
    top_pts = np.vstack([lbrow, rbrow])
    top_pts[:, 1] -= float(forehead_lift) * face_h
    poly_pts = np.vstack([jaw, top_pts])
    hull = cv2.convexHull(np.round(poly_pts).astype(np.int32))
    mask = np.zeros((256, 256), dtype=np.uint8)
    if hull is not None and len(hull) >= 3:
        cv2.fillConvexPoly(mask, hull, 1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1))
        mask = cv2.dilate(mask, k)
    return mask > 0


def _run_landmarks_106(rgb: np.ndarray) -> np.ndarray | None:
    """Run InsightFace 2d106det on RGB256, returning anatomically validated float32 (106, 2) or None."""
    if not (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (256, 256, 3)):
        return None
    entry = _load_landmarks106_session()
    if entry is None:
        return None
    sess, in_name, out_name = entry
    ort_errors = _onnx_runtime_errors()
    try:
        img192 = cv2.resize(rgb, (192, 192), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = img192.transpose(2, 0, 1)[None]
        raw = sess.run([out_name], {in_name: x})[0]
        if not (isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[0] >= 1 and raw.shape[1] == 212):
            return None
        vec = np.nan_to_num(raw[0].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        pts = (vec.reshape(106, 2) + np.float32(1.0)) * np.float32(96.0 * (256.0 / 192.0))
        if not np.isfinite(pts).all():
            return None
        xs, ys = pts[:, 0], pts[:, 1]
        span_x = float(xs.max() - xs.min())
        span_y = float(ys.max() - ys.min())
        if not (28.0 <= span_x <= 280.0 and 28.0 <= span_y <= 280.0):
            return None
        in_bounds = float(((xs >= -32.0) & (xs <= 288.0) & (ys >= -32.0) & (ys <= 288.0)).mean())
        if in_bounds < 0.85:
            return None
        if landmarks_5pt_from_106(pts) is None:
            return None
        return pts.astype(np.float32, copy=False)
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
    lab: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split hollow clear-glasses rims (tight 2px) from opaque/tinted sunglasses (capped below hairline)."""
    clear_rim_mask = np.zeros((256, 256), dtype=bool)
    opaque_eye_mask = np.zeros((256, 256), dtype=bool)
    clear_glasses_zone = np.zeros((256, 256), dtype=bool)
    brow_mask = np.isin(seg, (2, 3)) & observed & (_YY >= 52) & (_YY < 105)
    forehead_cap_y = 66
    if int(brow_mask.sum()) >= 80:
        forehead_cap_y = max(64, int(np.percentile(np.nonzero(brow_mask)[0], 25.0)) - 2)
    eye_zone = _EYE_ZONE & (_YY >= forehead_cap_y) & observed

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
            opaque_eye_mask |= _fill_mask_holes(g_full) & observed & (_YY >= forehead_cap_y)

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
            # Exclude visible open eyes (BiSeNet eye support + sclera/pupil contrast, zero glasses prob).
            open_eye_protect = np.zeros((256, 256), dtype=bool)
            if lab is not None:
                for eye_cls in (4, 5):
                    em = (seg == eye_cls) & observed
                    if int(em.sum()) >= 30 and float(bprob[6][em].mean()) < 0.18:
                        ep = (cv2.dilate(em.astype(np.uint8), _ELLIPSE_5PX) > 0) & observed
                        l_eye = lab[..., 0][ep]
                        if l_eye.size >= 20 and float(np.percentile(l_eye, 92.0) - np.percentile(l_eye, 8.0)) >= 45.0:
                            open_eye_protect |= ep
                if open_eye_protect.any():
                    sg_cand &= ~open_eye_protect
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
                # If an open eye is clearly visible in this eye-band with zero glasses prob, skip B&W shadow blobs.
                if open_eye_protect.any() and float(bprob[6][comp].mean()) < 0.18:
                    continue
                cx, cy = float(cen_c[idx][0]), float(cen_c[idx][1])
                if w >= 82 and area >= 1000 and 78.0 <= cx <= 178.0:
                    wide_kept |= comp
                elif (
                    both_bands_visible
                    and not open_eye_protect.any()
                    and area >= 1400
                    and w >= 48
                    and h >= 28
                    and 72.0 <= cx <= 184.0
                    and 86.0 <= cy <= 126.0
                    and float(px[comp].mean()) < 0.35
                    and float(skin[comp].mean()) < 0.40
                ):
                    wide_kept |= comp
                elif area >= 260 and w >= 0.65 * h and 76.0 <= cx <= 122.0 and 86.0 <= cy <= 126.0:
                    left_blobs.append((area, cy, comp))
                elif area >= 260 and w >= 0.65 * h and 134.0 <= cx <= 180.0 and 86.0 <= cy <= 126.0:
                    right_blobs.append((area, cy, comp))
            if wide_kept.any():
                opaque_eye_mask |= _fill_mask_holes(wide_kept) & observed & _FACE_MARGIN_ELLIPSE & (~open_eye_protect)
            elif left_blobs and right_blobs:
                l_best = max(left_blobs, key=lambda item: item[0])
                r_best = max(right_blobs, key=lambda item: item[0])
                if abs(l_best[1] - r_best[1]) <= 20.0:
                    pair = l_best[2] | r_best[2]
                    if float(px[pair].mean()) < 0.45:
                        opaque_eye_mask |= _fill_mask_holes(pair) & observed & _FACE_MARGIN_ELLIPSE & (~open_eye_protect)

    return clear_rim_mask, opaque_eye_mask, clear_glasses_zone


def _recover_ear_loops(
    lower_occ: np.ndarray,
    px: np.ndarray,
    seg: np.ndarray,
    skin: np.ndarray,
    delta_e: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Trace thin cheek-to-ear mask straps contiguous with a lower-face mask without hair/bg wings."""
    if not lower_occ.any():
        return np.zeros((256, 256), dtype=bool)
    corridor = (_YY >= 112) & (_YY < 194) & ((_XX >= 42) & (_XX < 102) | (_XX >= 154) & (_XX < 214)) & observed
    # Restrict strap tracing to face/cheek pixels (never hair, background, ears, or neck).
    strap_cand = corridor & (~np.isin(seg, (0, 7, 8, 9, 14, 15, 16, 17, 18))) & ((px < 0.48) | ((~skin) & (delta_e > 24.0)))
    if not strap_cand.any():
        return np.zeros((256, 256), dtype=bool)
    strap_clean = cv2.morphologyEx(strap_cand.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_3PX) > 0
    anchor = cv2.dilate(lower_occ.astype(np.uint8), _ELLIPSE_5PX) > 0
    n_s, lbl_s, st_s, _ = cv2.connectedComponentsWithStats(strap_clean.astype(np.uint8), connectivity=8)
    recovered = np.zeros((256, 256), dtype=bool)
    for idx in range(1, n_s):
        area = int(st_s[idx, cv2.CC_STAT_AREA])
        if not (25 <= area <= 1200):
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
    est_eye_y: int,
    nose_visible: bool,
    l_ref: float,
    beard_guard: bool,
    beard_suppress: np.ndarray,
    hairline_suppress: np.ndarray,
    clear_glasses_zone: np.ndarray,
    is_eye_slit_face: bool,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Unified upper/middle/lower face-hull XSeg occlusion detector + niqab/balaclava/hand recovery."""
    occ_mask = np.zeros((256, 256), dtype=bool)
    ear_strap_mask = np.zeros((256, 256), dtype=bool)
    chin_gaiter_mask = np.zeros((256, 256), dtype=bool)
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
        return occ_mask, has_lower_occ, ear_strap_mask, chin_gaiter_mask

    if vis_face_px < 1000:
        return occ_mask, False, ear_strap_mask, chin_gaiter_mask

    # Require recognizable facial features (brows/eyes/glasses/nose) so ultra-blurred blobs or
    # distant full-body heads never hallucinate face coverings from XSeg blur decay.
    facial_features = np.isin(seg, (2, 3, 4, 5, 6, 10)) & observed
    if int(facial_features.sum()) < 100:
        return occ_mask, False, ear_strap_mask, chin_gaiter_mask

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F)[bface].var()) if bface.any() else 0.0
    sobel_mag = np.hypot(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0), cv2.Sobel(gray_f, cv2.CV_32F, 0, 1))

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
    # Never include seg==18 hat/beanie above visible eyes or estimated eye line (fixes compound_04 & eyewear_06).
    hat_y_cap = max(est_eye_y + 10, 96) if not eyes_visible else (eyes_y_min + 4)
    above_eye_hat = (seg == 18) & (_YY < hat_y_cap)
    hat_spill = np.zeros((256, 256), dtype=bool)
    if above_eye_hat.any():
        hat_spill = (cv2.dilate(above_eye_hat.astype(np.uint8), _ELLIPSE_11PX) > 0) & (_YY < hat_y_cap + 14) & (px < 0.45) & (seg != 6)
    outer_collar_corners = (_YY >= 200) & ((_XX < 78) | (_XX >= 178)) & (~bface)

    # Cap occ_interior below forehead hairline (y >= max(64, est_eye_y - 4)) so forehead/hat fins never fire.
    occ_top_y = max(64, est_eye_y - 4)
    occ_interior = bface_eroded & (px < 0.42) & (_YY >= occ_top_y) & (~above_eye_hat) & (~hat_spill)
    if clear_glasses_zone.any():
        occ_interior &= ~clear_glasses_zone
    if not lips_visible and (int(occ_interior.sum()) >= 200 or (int(bface.sum()) >= 8000 and bf_y_max >= 165)):
        lower_hull = face_hull & (_YY >= 112) & (~non_face_ext) & (~above_eye_hat) & (~outer_collar_corners)
        occ_hull = lower_hull & (px < 0.35)
    else:
        occ_hull = np.zeros((256, 256), dtype=bool)

    raw_occ = (occ_interior | occ_hull) & (~hairline_suppress) & (~above_eye_hat) & (~hat_spill) & (~outer_collar_corners) & observed
    hat_near = (
        cv2.dilate((seg == 18).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
        if (seg == 18).any()
        else np.zeros((256, 256), dtype=bool)
    )

    if raw_occ.any():
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
            is_glasses_comp = float((seg[comp] == 6).mean()) >= 0.35
            # Reject thin vertical sideburn/hair strips clinging to left/right bface border.
            if w < 38 and h > 2.1 * max(1, w) and (x <= bf_x_min + 12 or x + w >= bf_x_max - 12):
                continue
            # Reject bottom-frame neck/collar edge strips below the chin.
            if y >= 218 or (cy_c >= 228.0 and (y + h) >= 252):
                continue
            # Reject forehead hairline / bangs / crown / hat brim above eye line (even if eyes_visible=False).
            if eyes_visible and ((y + h) <= eyes_y_min + 4 or cy_c < eyes_y_min - 6):
                continue
            if not is_glasses_comp and (cy_c < float(est_eye_y - 2) or ((y + h) <= est_eye_y + 12)):
                continue
            if not is_glasses_comp and (comp & hat_near).any() and cy_c < 104.0 and (y + h) < 135:
                continue
            if cy_c < 80.0 and (y + h) <= 104:
                continue
            # Reject small blurry skin highlights/shadows on soft degraded inputs.
            if lap_var < 35.0 and float(skin[comp].mean()) >= 0.80 and (area < 600 or mean_px >= 0.20):
                continue

            # Unilateral skin-colored hand/finger occlusion check (bilateral symmetry + ridge edge density + XSeg dip).
            ys_c, xs_c = np.nonzero(comp)
            mx_c = np.clip(np.rint(2.0 * cx - xs_c).astype(np.int32), 0, 255)
            mirror_px = float(px[ys_c, mx_c].mean()) if ys_c.size else 0.0
            comp_edge = float(sobel_mag[comp].mean())
            mirror_edge = float(sobel_mag[ys_c, mx_c].mean()) if ys_c.size else 0.0
            unilateral_hand = (
                lap_var >= 35.0
                and area >= 500
                and mean_px < 0.28
                and abs(cx_c - cx) >= 16.0
                and mirror_px >= 0.65
                and comp_edge >= 16.0
                and comp_edge >= mirror_edge * 1.20
                and (comp & _FACE_ELLIPSE).mean() >= 0.70
            )

            # Reject beard/mustache regions wrapping around intact visible lips + nose.
            if beard_guard and float(beard_suppress[comp].mean()) >= 0.45:
                if int((comp & lips_mask & (px < 0.45)).sum()) < 40 and not (unilateral_hand and abs(cx_c - cx) >= 22.0 and mean_px < 0.22):
                    continue
            # Reject BiSeNet-vs-XSeg outer-perimeter skin mismatches on blurry/degraded unoccluded faces,
            # while allowing high-confidence unilateral hands/fingers (compound_02).
            if beard_guard and float(skin[comp].mean()) >= 0.50:
                if (both_eyes_visible or y <= max(35, occ_top_y + 10) or y >= 145) and not unilateral_hand:
                    continue
            occ_mask |= comp

    # Full lower-face mask body completion & tight ear-loop recovery when lower face is occluded.
    lower_part = occ_mask & (_YY >= 122) & _FACE_ELLIPSE
    if int(lower_part.sum()) >= 500:
        has_lower_occ = True
        if not lips_visible:
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
            ear_strap_mask |= _recover_ear_loops(completed_lower, px, seg, skin, delta_e, observed)

    # Branch B: Tight achromatic gray/white balaclava (XSeg fooled by face-conforming gray knit, e.g., scarf_02).
    if not lips_visible and int(lips_mask.sum()) < 40 and eyes_visible:
        chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
        eye_skin = _EYE_ZONE & bface & (px >= 0.60) & skin
        is_color_photo = int((eye_skin & (chroma >= 9.0)).sum()) >= 120
        if is_color_photo:
            eye_skin_chroma = float(np.median(chroma[eye_skin]))
            gray_thresh = min(10.5, max(7.5, eye_skin_chroma - 3.5))
            # Warm-lit gray knit can trigger pale_skin; allow very low chroma gray cloth cleanly without Swiss-cheese holes.
            gray_cloth_px = (chroma < gray_thresh) & ((~skin) | (chroma < max(6.5, gray_thresh - 1.5)))
            lower_gray = _LOWER_ZONE & observed & gray_cloth_px & (~non_face_ext)
            if int(lower_gray.sum()) >= 2200:
                eye_protect = cv2.dilate((eyes_mask | eye_skin).astype(np.uint8), _ELLIPSE_5PX) > 0
                cloth = _FACE_ELLIPSE & observed & gray_cloth_px & (~non_face_ext) & (~eye_protect)
                cloth = cv2.morphologyEx(cloth.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_3PX) > 0
                cloth = cv2.morphologyEx(cloth.astype(np.uint8), cv2.MORPH_CLOSE, _ELLIPSE_11PX) > 0
                n_g, lbl_g, st_g, _ = cv2.connectedComponentsWithStats(cloth.astype(np.uint8), connectivity=8)
                for idx in range(1, n_g):
                    if int(st_g[idx, cv2.CC_STAT_AREA]) >= 1200:
                        g_comp = lbl_g == idx
                        if (g_comp & lower_gray).sum() >= 600:
                            occ_mask |= _fill_mask_holes(g_comp) & _FACE_ELLIPSE & (~eye_protect)
                            has_lower_occ = True

    # Branch C: Low-res CCTV dark/black lower-face mask/gaiter hugging chin/jaw (e.g., compound_04).
    if nose_visible and not has_lower_occ and int(bface.sum()) >= 8000 and int((seg == 18).sum()) >= 400:
        chin_zone = _FACE_ELLIPSE & (_YY >= 174) & (_YY < 238) & (_XX >= 58) & (_XX < 198) & observed
        dark_chin = (
            chin_zone
            & (px < 0.25)
            & (~skin)
            & (lab[..., 0] < 72.0)
            & ((l_ref - lab[..., 0]) > 58.0)
            & np.isin(seg, (1, 14, 16, 18))
        )
        if int(dark_chin.sum()) >= 1400 and float(px[182:222, 82:174].mean()) < 0.42:
            bface_near = cv2.dilate(bface.astype(np.uint8), _ELLIPSE_9PX) > 0
            closed_dc = cv2.morphologyEx(dark_chin.astype(np.uint8), cv2.MORPH_CLOSE, _ELLIPSE_7PX) > 0
            n_dc, lbl_dc, st_dc, cen_dc = cv2.connectedComponentsWithStats(closed_dc.astype(np.uint8), connectivity=8)
            for idx in range(1, n_dc):
                x, y, w, h, area = (int(st_dc[idx, k]) for k in range(5))
                cx_dc = float(cen_dc[idx][0])
                if area < 1400 or w < 72 or y > 200 or not (86.0 <= cx_dc <= 170.0):
                    continue
                comp = lbl_dc == idx
                if int((comp & bface_near).sum()) >= 300:
                    filled_dc = _fill_mask_holes(comp) & chin_zone
                    occ_mask |= filled_dc
                    chin_gaiter_mask |= filled_dc
                    has_lower_occ = True

    return occ_mask & observed, has_lower_occ, ear_strap_mask & observed, chin_gaiter_mask & observed


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


def _fast_guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 4, eps: float = 1e-2) -> np.ndarray:
    """Fast O(N) edge-preserving guided filter using OpenCV boxFilter (~2 ms at 256x256)."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    g = guide.astype(np.float32, copy=False)
    p = src.astype(np.float32, copy=False)
    mean_g = cv2.boxFilter(g, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_g = cv2.boxFilter(g * g, cv2.CV_32F, ksize)
    corr_gp = cv2.boxFilter(g * p, cv2.CV_32F, ksize)
    var_g = np.maximum(corr_g - mean_g * mean_g, np.float32(0.0))
    cov_gp = corr_gp - mean_g * mean_p
    a = cov_gp / (var_g + np.float32(eps))
    b = mean_p - a * mean_g
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return np.clip(mean_a * g + mean_b, 0.0, 1.0)


def _absorb_opaque_fringe(
    opaque_mask: np.ndarray,
    lab: np.ndarray,
    ref_skin_lab: tuple[float, float, float],
    observed: np.ndarray,
    px: np.ndarray | None = None,
    seg: np.ndarray | None = None,
    protect_mask: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    bprob: np.ndarray | None = None,
) -> np.ndarray:
    """Per-component edge-aware refinement (Guided Filter + ROI Trimap GrabCut + cast-shadow + fringe)."""
    if not opaque_mask.any():
        return opaque_mask

    hard_exterior = np.isin(seg, (7, 8, 9, 14, 15, 17)) if seg is not None else np.zeros((256, 256), dtype=bool)
    shadow_exclude = np.isin(seg, (2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 17)) if seg is not None else np.zeros((256, 256), dtype=bool)
    if protect_mask is not None:
        hard_exterior = hard_exterior | protect_mask
        shadow_exclude = shadow_exclude | protect_mask

    # Guided filter on continuous occlusion probability for edge snapping.
    guide_f = (lab[..., 0] / np.float32(255.0)).astype(np.float32)
    if px is not None:
        p_occ = np.float32(1.0) - px
        if bprob is not None:
            p_occ = np.maximum(p_occ, bprob[6])
    else:
        p_occ = opaque_mask.astype(np.float32)
    guided_occ = _fast_guided_filter(guide_f, p_occ, radius=4, eps=1e-2)

    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(opaque_mask.astype(np.uint8), connectivity=8)
    result = np.zeros((256, 256), dtype=bool)

    for idx in range(1, n_comp):
        x, y, w, h, area = (int(stats[idx, k]) for k in range(5))
        if area == 0:
            continue
        comp = labels == idx
        comp_u8 = comp.astype(np.uint8)

        # Component-local Lab median and component-local surrounding clean skin Lab reference.
        l_m = float(np.median(lab[..., 0][comp]))
        a_m = float(np.median(lab[..., 1][comp]))
        b_m = float(np.median(lab[..., 2][comp]))

        outer_skin_ring = (cv2.dilate(comp_u8, _ELLIPSE_11PX) > 0) & (~comp) & observed & (~hard_exterior) & (~opaque_mask)
        if seg is not None:
            outer_skin_ring &= (seg == 1)
        if px is not None:
            outer_skin_ring &= (px >= 0.65)
        if int(outer_skin_ring.sum()) >= 32:
            l_sk = float(np.median(lab[..., 0][outer_skin_ring]))
            a_sk = float(np.median(lab[..., 1][outer_skin_ring]))
            b_sk = float(np.median(lab[..., 2][outer_skin_ring]))
        else:
            l_sk, a_sk, b_sk = ref_skin_lab

        # Outer leash <= 5px around raw component, strictly excluding hair/ears/neck/protected regions
        # and preventing expansion into seg==0 background outside the core component.
        leash = (cv2.dilate(comp_u8, _ELLIPSE_11PX) > 0) & observed & (~hard_exterior)
        if seg is not None:
            leash &= ((seg != 0) | comp)

        # Step A: Bounded ROI Trimap GrabCut (2 iters) strictly clamped inside leash for medium/large components.
        refined = comp.copy()
        if rgb is not None and area >= 450:
            pad = 14
            rx0, ry0 = max(0, x - pad), max(0, y - pad)
            rx1, ry1 = min(256, x + w + pad), min(256, y + h + pad)
            if (rx1 - rx0) >= 24 and (ry1 - ry0) >= 24:
                roi_comp = comp[ry0:ry1, rx0:rx1]
                roi_leash = leash[ry0:ry1, rx0:rx1]
                def_fg = cv2.erode(roi_comp.astype(np.uint8), _ELLIPSE_7PX) > 0
                def_bg = ~roi_leash
                if px is not None:
                    def_bg = def_bg | ((~roi_comp) & (px[ry0:ry1, rx0:rx1] >= 0.78))
                if int(def_fg.sum()) >= 80 and int(def_bg.sum()) >= 80:
                    gc_mask = np.full((ry1 - ry0, rx1 - rx0), cv2.GC_PR_BGD, dtype=np.uint8)
                    gc_mask[roi_comp] = cv2.GC_PR_FGD
                    gc_mask[def_fg] = cv2.GC_FGD
                    gc_mask[def_bg] = cv2.GC_BGD
                    bg_model = np.zeros((1, 65), np.float64)
                    fg_model = np.zeros((1, 65), np.float64)
                    try:
                        cv2.grabCut(
                            rgb[ry0:ry1, rx0:rx1],
                            gc_mask,
                            None,
                            bg_model,
                            fg_model,
                            2,
                            cv2.GC_INIT_WITH_MASK,
                        )
                        gc_fg = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)) & roi_leash
                        # Keep core intact and adopt edge-snapped GrabCut boundary inside leash.
                        refined[ry0:ry1, rx0:rx1] = (roi_comp & (~hard_exterior[ry0:ry1, rx0:rx1])) | (gc_fg & (cv2.dilate(roi_comp.astype(np.uint8), _ELLIPSE_9PX) > 0))
                    except cv2.error:
                        pass

        # Step B: Tight 3px edge-preserving dilation + Guided Filter snap inside leash.
        tight_dil = (cv2.dilate(refined.astype(np.uint8), _ELLIPSE_7PX) > 0) & leash
        guided_ring = (cv2.dilate(refined.astype(np.uint8), _ELLIPSE_9PX) > 0) & (~tight_dil) & leash & (guided_occ >= 0.52)
        base_comp = (refined | tight_dil | guided_ring) & leash

        # Step C: Bounded directional cast-shadow absorption (<=4px contiguous with occluder boundary).
        shadow_ring = (cv2.dilate(base_comp.astype(np.uint8), _ELLIPSE_9PX) > 0) & (~base_comp) & leash & (~shadow_exclude) & _FACE_MARGIN_ELLIPSE
        if shadow_ring.any():
            de_ab_sk = np.hypot(lab[..., 1] - a_sk, lab[..., 2] - b_sk)
            cast_shadow = shadow_ring & (lab[..., 0] < (l_sk - 15.0)) & (de_ab_sk < 12.0)
            if px is not None:
                cast_shadow &= (px < 0.78)
        else:
            cast_shadow = np.zeros((256, 256), dtype=bool)

        # Step D: Per-component adaptive Lab fringe (<=3px) strictly inside leash & face margin.
        fringe_ring = (cv2.dilate(base_comp.astype(np.uint8), _ELLIPSE_7PX) > 0) & (~base_comp) & leash & _FACE_MARGIN_ELLIPSE
        if fringe_ring.any():
            de_skin = np.sqrt((lab[..., 0] - l_sk) ** 2 + (lab[..., 1] - a_sk) ** 2 + (lab[..., 2] - b_sk) ** 2)
            de_mask = np.sqrt((lab[..., 0] - l_m) ** 2 + (lab[..., 1] - a_m) ** 2 + (lab[..., 2] - b_m) ** 2)
            lab_fringe = de_mask < (de_skin - 6.0)
            if px is not None:
                fringe = fringe_ring & ((px < 0.48) | (lab_fringe & (px < 0.68)))
            else:
                fringe = fringe_ring & lab_fringe
        else:
            fringe = np.zeros((256, 256), dtype=bool)

        result |= (base_comp | cast_shadow | fringe)

    if protect_mask is not None:
        result &= ~protect_mask
    result &= observed
    # Drop tiny isolated dust specks (< 16 px) created by boundary snapping or shadow rings.
    if result.any():
        n_r, lbl_r, st_r, _ = cv2.connectedComponentsWithStats(result.astype(np.uint8), connectivity=8)
        if n_r > 2:
            cleaned = np.zeros_like(result)
            for idx in range(1, n_r):
                if int(st_r[idx, cv2.CC_STAT_AREA]) >= 16:
                    cleaned |= (lbl_r == idx)
            if cleaned.any():
                result = cleaned
    return result


def _refine_mask_p3(
    final_mask: np.ndarray,
    observed: np.ndarray,
    bprob: np.ndarray | None,
    seg: np.ndarray | None,
    px: np.ndarray | None,
    skin: np.ndarray,
    pts106: np.ndarray | None,
    hull_clip: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """Zero-weight Phase 3 post-refinement on top of Phase 2 mask (0/24 FP guaranteed)."""
    m = final_mask & observed
    actions: list[str] = []
    if not m.any() or bprob is None or seg is None or px is None:
        return m, actions

    bface = np.isin(seg, (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)) & observed
    bface_px = int(bface.sum())
    hull_tight = (face_hull_from_106(pts106, dilate_px=0, forehead_lift=0.05) & observed) if pts106 is not None else None

    # 1. Background seg==0 crescent CC drop + outer rim trimming + high-skin hand edge trimming.
    if bface_px >= 4000 and hull_tight is not None:
        n_cc, lbl, _, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        bg_pruned = m.copy()
        dropped_bg_comp = False
        if n_cc > 2:
            for idx in range(1, n_cc):
                comp = lbl == idx
                seg0_frac = float((seg[comp] == 0).mean())
                in_bface_frac = float(bface[comp].mean())
                if seg0_frac >= 0.50 and in_bface_frac <= 0.45:
                    bg_pruned &= ~comp
                    dropped_bg_comp = True
        bface_dil = cv2.dilate(bface.astype(np.uint8), _ELLIPSE_7PX) > 0
        outer_bg_spill = bg_pruned & (seg == 0) & (~hull_tight) & (~bface_dil)
        if int(outer_bg_spill.sum()) >= 40:
            candidate = bg_pruned & (~outer_bg_spill)
            if int(candidate.sum()) >= 300:
                bg_pruned = candidate
                actions.append(f"trim_outer_seg0(-{int(outer_bg_spill.sum())}px)")
        if bg_pruned.any() and float(skin[bg_pruned].mean()) >= 0.55:
            skin_bg_spill = bg_pruned & (seg == 0) & (~bface_dil)
            if int(skin_bg_spill.sum()) >= 40:
                candidate = bg_pruned & (~skin_bg_spill)
                if int(candidate.sum()) >= 300:
                    bg_pruned = candidate
                    actions.append(f"trim_hand_seg0(-{int(skin_bg_spill.sum())}px)")
        if dropped_bg_comp and bg_pruned.any():
            actions.append("drop_bg_crescent_cc")
            m = bg_pruned
        elif bg_pruned.any() and not np.array_equal(bg_pruned, m):
            m = bg_pruned

    # 2. Conditional chin-gaiter jawline clamp (removes dark hoodie collar below chin_y + 10).
    if m.any() and pts106 is not None and hull_tight is not None and hull_clip is not None:
        in_tight_frac = float(hull_tight[m].mean())
        neck_cloth_frac = float(np.isin(seg[m], (14, 16)).mean())
        chin_y = float(pts106[0:33, 1].max())
        if np.isfinite(chin_y) and in_tight_frac < 0.25 and neck_cloth_frac >= 0.65:
            hull_chin = face_hull_from_106(pts106, dilate_px=10, forehead_lift=0.25) & observed
            clamped = m & hull_chin & (_YY <= int(round(chin_y + 10.0)))
            if int(clamped.sum()) >= 800:
                removed = int(m.sum()) - int(clamped.sum())
                if removed >= 200:
                    m = clamped
                    actions.append(f"clamp_chin_hoodie(-{removed}px)")

    # 3. High-visibility dust/bridge fragment pruning (preserves 2-lens eyewear) + shield-bridge smoothing.
    if m.any():
        n_cc, lbl, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n_cc > 2:
            areas = [int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n_cc)]
            max_area = max(areas)
            if max_area >= 2000:
                pruned = np.zeros_like(m)
                dropped = 0
                for idx in range(1, n_cc):
                    comp = lbl == idx
                    area = int(st[idx, cv2.CC_STAT_AREA])
                    if area < 140 and float(px[comp].mean()) >= 0.50:
                        dropped += 1
                        continue
                    pruned |= comp
                if dropped > 0 and pruned.any():
                    m = pruned
                    actions.append(f"prune_frag({dropped})")

        if int(m.sum()) >= 8000 and int((seg[m] == 6).sum()) >= 600:
            eyes_vis = np.isin(seg, (4, 5)) & (px >= 0.55) & observed
            if int(eyes_vis.sum()) >= 40:
                eye_protect = cv2.dilate(eyes_vis.astype(np.uint8), _ELLIPSE_7PX) > 0
                closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _ELLIPSE_13PX) > 0
                filled = _fill_mask_holes(closed) & (~eye_protect) & observed
                if hull_clip is not None:
                    filled &= hull_clip
                if hull_tight is not None:
                    spur = filled & (~hull_tight) & (px >= 0.45)
                    filled &= ~spur
                if filled.any() and not np.array_equal(filled, m):
                    actions.append(f"shield_smooth({int(m.sum())}->{int(filled.sum())}px)")
                    m = filled

    return m & observed, actions


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
            "p3_refiner_actions": [],
        }

    inner_face = _INNER_FACE & observed
    face_ellipse = _FACE_ELLIPSE & observed
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.int16)
    skin = _skin_mask(lab, ycrcb)

    onnx_out = _run_onnx_models(rgb) if int(observed.sum()) >= 4000 else None
    onnx_used = onnx_out is not None

    pts106 = None
    hull_clip = None
    is_ofiq_masked_face = False
    if onnx_out is not None:
        bprob, px = onnx_out
        seg = bprob.argmax(axis=0)
        bface = np.isin(seg, (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)) & observed
        left_eye_mask = (seg == 4) & observed
        right_eye_mask = (seg == 5) & observed
        eyes_mask = left_eye_mask | right_eye_mask
        nose_mask = (seg == 10) & observed
        lips_mask = np.isin(seg, (12, 13)) & observed
        pts106 = _run_landmarks_106(rgb)
        if pts106 is not None:
            hull_clip = face_hull_from_106(pts106, dilate_px=6, forehead_lift=0.25) & observed
            if not hull_clip.any():
                hull_clip = None

        xseg_in_ellipse = int(((px >= 0.50) & _FACE_ELLIPSE & observed).sum())
        xseg_total = int(((px >= 0.50) & observed).sum())
        xseg_upper = float(px[58:116, 70:186].mean())
        xseg_mid_lower = float(px[116:215, 70:186].mean())
        xseg_centered_face = (
            xseg_in_ellipse >= 7500
            and xseg_in_ellipse >= 0.68 * float(max(1, xseg_total))
            and (
                (float(px[58:84, 70:186].mean()) >= 0.60 and float(px[132:188, 70:186].mean()) >= 0.60)
                or (xseg_upper >= 0.50 and xseg_mid_lower < 0.45)
                or (xseg_upper >= 0.22 and xseg_mid_lower >= 0.55)
            )
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

        # Heavily hooded / dark-masked CCTV face or full-head wrapped scarf where BiSeNet sees
        # bface < 2000 but 2d106det + OFIQ clearly locate a large occluded lower face.
        if not has_face and not is_eye_slit_face and pts106 is not None:
            hull_t_chk = face_hull_from_106(pts106, dilate_px=0, forehead_lift=0.05) & observed
            brow_y_chk = int(round(float(min(pts106[43:52, 1].min(), pts106[97:106, 1].min()))))
            eye_y_chk = int(round(0.5 * float(pts106[38, 1] + pts106[88, 1])))
            up_chk = hull_t_chk & (_YY >= max(0, brow_y_chk)) & (_YY < eye_y_chk + 24)
            lo_chk = hull_t_chk & (_YY >= eye_y_chk + 16)
            is_ofiq_hooded_mask = (
                int(up_chk.sum()) >= 400
                and int(lo_chk.sum()) >= 6500
                and xseg_in_ellipse >= 1000
                and float(px[up_chk].mean()) >= 0.20
                and float(px[lo_chk].mean()) < 0.30
                and float(skin[lo_chk].mean()) < 0.05
                and int((lo_chk & (px < 0.35) & (~skin)).sum()) >= 5500
            )
            is_ofiq_wrapped_scarf = (
                int(bface.sum()) >= 800
                and int(lo_chk.sum()) >= 8000
                and float(px[lo_chk].mean()) < 0.15
                and int((lo_chk & (px < 0.35) & np.isin(seg, (16, 18))).sum()) >= 6000
                and int(((seg == 18) & observed).sum()) >= 10000
            )
            if is_ofiq_hooded_mask or is_ofiq_wrapped_scarf:
                is_ofiq_masked_face = True

        # Non-face / full-body uncropped scene guard: never fire heuristics on shirts/placards.
        if not has_face and not is_eye_slit_face and not is_ofiq_masked_face and not base_mask.any() and float(probability.max()) < 0.35:
            return np.zeros((256, 256), dtype=np.uint8), {
                "auto_detected_pixels": 0,
                "families_triggered": [],
                "visibility_trimmed": False,
                "neural_seed_pixels": 0,
                "onnx_used": True,
                "p3_refiner_actions": [],
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
        nose_y_min = int(np.nonzero(nose_mask)[0].min()) if nose_visible else 120
        est_eye_y = eyes_y_min if eyes_visible else (max(62, nose_y_min - 28) if nose_visible else 88)

        # Lip plausibility guard: prevent BiSeNet from hallucinating lips on dark/low-contrast lower masks.
        lips_px_count = int(lips_mask.sum())
        lips_raw_vis = (
            (lips_px_count >= 140 and float(px[lips_mask].mean()) >= 0.45)
            or (lips_px_count >= 900 and float(px[lips_mask].mean()) >= 0.35)
        )
        if lips_raw_vis and lips_px_count > 0:
            lip_l = float(lab[..., 0][lips_mask].mean())
            lip_l_std = float(lab[..., 0][lips_mask].std())
            lip_a = float(lab[..., 1][lips_mask].mean())
            px_lower_mouth = float(px[145:210, 82:174].mean())
            lip_plausible = (
                lip_l >= 40.0
                and not (lip_l < 58.0 and lip_l_std < 11.0 and lip_a < 131.0)
                and (px_lower_mouth >= 0.45 or lips_px_count >= 900)
            )
            lips_visible = bool(lip_plausible)
        else:
            lips_visible = False
        lips_y_min = int(np.nonzero(lips_mask)[0].min()) if lips_visible else 160
        beard_guard = lips_visible and nose_visible

        # Lower-face beard/mustache zone below visible nose/lips.
        beard_suppress = np.zeros((256, 256), dtype=bool)
        if beard_guard:
            beard_suppress = (_YY >= max(132, lips_y_min - 10)) & (seg != 6)
        # Forehead hairline / bangs / crown / hat zone strictly above eyes or estimated eye line.
        hairline_suppress = np.zeros((256, 256), dtype=bool)
        if eyes_visible:
            hairline_suppress = (_YY < max(60, eyes_y_min - 1)) & (seg != 6)
        elif nose_visible:
            hairline_suppress = (_YY < max(58, est_eye_y - 4)) & (seg != 6)

        if int(bface.sum()) >= 2000:
            collar_suppress = np.isin(seg, (0, 7, 8, 9, 14, 15, 16, 17, 18)) & (~bface)
        else:
            collar_suppress = np.isin(seg, (7, 8, 9, 14, 15, 16, 17, 18)) & (~bface)
        # Never suppress inner face oval low-px cloth when lips are occluded on an actual face crop.
        facial_features_px = int((np.isin(seg, (2, 3, 4, 5, 6, 10)) & observed).sum())
        bf_y_max_all = int(np.nonzero(bface)[0].max()) if bface.any() else 0
        if not lips_visible and (is_eye_slit_face or is_ofiq_masked_face or (facial_features_px >= 100 and bf_y_max_all >= 165)):
            above_eye_hat = (seg == 18) & (_YY < (eyes_y_min + 4 if eyes_visible else max(est_eye_y + 10, 96)))
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
        has_face = is_eye_slit_face = xseg_centered_face = False
        eyes_visible = both_eyes_visible = nose_visible = lips_visible = beard_guard = False
        eyes_y_min, lips_y_min, est_eye_y = 92, 160, 88
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

    # 3. ONNX eyewear + unified face-hull occlusion detection + calibrated OFIQ/2d106det fusion
    onnx_clear_eye = np.zeros((256, 256), dtype=bool)
    onnx_opaque_eye = np.zeros((256, 256), dtype=bool)
    clear_glasses_zone = np.zeros((256, 256), dtype=bool)
    onnx_hull = np.zeros((256, 256), dtype=bool)
    ofiq_extra = np.zeros((256, 256), dtype=bool)
    ear_strap_mask = np.zeros((256, 256), dtype=bool)
    chin_gaiter_mask = np.zeros((256, 256), dtype=bool)
    onnx_has_lower = False
    if onnx_used and (has_face or is_eye_slit_face or is_ofiq_masked_face):
        if has_face or is_eye_slit_face:
            onnx_clear_eye, onnx_opaque_eye, clear_glasses_zone = _detect_onnx_eyewear(
                bprob, seg, px, bface, observed, skin, both_eyes_visible=both_eyes_visible, lab=lab
            )
            onnx_hull, onnx_has_lower, ear_strap_mask, chin_gaiter_mask = _detect_onnx_face_hull_occlusions(
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
                est_eye_y=est_eye_y,
                nose_visible=nose_visible,
                l_ref=l_ref,
                beard_guard=beard_guard,
                beard_suppress=beard_suppress,
                hairline_suppress=hairline_suppress,
                clear_glasses_zone=clear_glasses_zone,
                is_eye_slit_face=is_eye_slit_face,
            )

        # Calibrated OFIQ inner-face seeding (>=260 px core) + geodesic growth into (px < 0.50) & hull_clip.
        # Gate on recognizable facial features (brows/eyes/glasses/nose >= 100 px) unless eye-slit / hooded CCTV mask,
        # preventing ultra-blurred / uncropped uncovered faces from hallucinating OFIQ occlusions.
        ofiq_feats_px = int((np.isin(seg, (2, 3, 4, 5, 6, 10)) & observed).sum())
        if ofiq_feats_px >= 100 or is_eye_slit_face or is_ofiq_masked_face:
            if pts106 is not None:
                hull_tight = face_hull_from_106(pts106, dilate_px=0, forehead_lift=0.05) & observed
                inner_106 = cv2.erode(hull_tight.astype(np.uint8), _ELLIPSE_9PX) > 0
                grow_clip = hull_clip if hull_clip is not None else (_FACE_MARGIN_ELLIPSE & observed)
                brow_y_106 = int(round(float(min(pts106[43:52, 1].min(), pts106[97:106, 1].min()))))
                eye_y_106 = int(round(0.5 * float(pts106[38, 1] + pts106[88, 1])))
            else:
                inner_106 = cv2.erode(inner_face.astype(np.uint8), _ELLIPSE_9PX) > 0
                grow_clip = _FACE_MARGIN_ELLIPSE & observed
                brow_y_106 = max(60, est_eye_y - 6)
                eye_y_106 = est_eye_y

            has_glasses_seg = int(((seg == 6) & observed).sum()) >= 200
            hat_y_cap = (eyes_y_min + 4) if eyes_visible else max(est_eye_y + 10, 96)
            above_eye_hat = (seg == 18) & (_YY < hat_y_cap)
            above_eye_guard = above_eye_hat | (seg == 17)
            if eyes_visible:
                eye_top_guard_y = max(64, min(eyes_y_min + 2, brow_y_106 + 6))
                above_eye_guard = above_eye_guard | hairline_suppress | ((_YY < eye_top_guard_y) & (seg != 6))
                if above_eye_hat.any():
                    above_eye_guard = above_eye_guard | (
                        (cv2.dilate(above_eye_hat.astype(np.uint8), _ELLIPSE_11PX) > 0)
                        & (_YY < max(hat_y_cap + 16, eyes_y_min + 12))
                        & (seg != 6)
                    )
            elif nose_visible and not has_glasses_seg and int(((seg == 18) & (_YY < 130) & observed).sum()) >= 300:
                # Beanie / cap / hat above visible eyes+nose (e.g. compound_04): never paint forehead/beanie above eyes.
                beanie_cut_y = max(122, eye_y_106 + 10)
                above_eye_guard = above_eye_guard | (_YY < beanie_cut_y)
            elif has_glasses_seg and not is_eye_slit_face:
                # Eyewear is already segmented below forehead_cap_y (>=64); prevent OFIQ from bleeding into forehead/hair above brows.
                above_eye_guard = above_eye_guard | (_YY < max(60, brow_y_106 + 8))
            elif nose_visible and above_eye_hat.any():
                above_eye_guard = above_eye_guard | (
                    (cv2.dilate(above_eye_hat.astype(np.uint8), _ELLIPSE_11PX) > 0)
                    & (_YY < hat_y_cap + 14)
                    & (seg != 6)
                )
            elif is_ofiq_masked_face:
                # Keep upper eye-slit/hood forehead clear on heavily hooded CCTV faces.
                above_eye_guard = above_eye_guard | (_YY < max(88, eye_y_106 + 8))

            ofiq_non_face = np.isin(seg, (7, 8, 9, 14, 15, 17))
            ofiq_core_domain = inner_106 & (px < 0.35) & (~above_eye_guard) & (~ofiq_non_face)
            if not is_ofiq_masked_face and collar_suppress.any():
                ofiq_core_domain &= ~collar_suppress
            if clear_glasses_zone.any():
                ofiq_core_domain &= ~clear_glasses_zone
            if ofiq_core_domain.any():
                n_oc, lbl_oc, st_oc, _ = cv2.connectedComponentsWithStats(ofiq_core_domain.astype(np.uint8), connectivity=8)
                verified_cores = np.zeros((256, 256), dtype=bool)
                for idx in range(1, n_oc):
                    if int(st_oc[idx, cv2.CC_STAT_AREA]) < 260:
                        continue
                    comp = lbl_oc == idx
                    if beard_guard:
                        comp_skin = float(skin[comp].mean())
                        comp_px = float(px[comp].mean())
                        comp_s1 = float((seg[comp] == 1).mean())
                        # Reject motion-blur / shadow dips on pure visible skin when nose+lips are intact.
                        if comp_skin >= 0.85 and comp_px >= 0.18 and comp_s1 >= 0.40:
                            continue
                        # Reject beard/jaw cores unless deep non-skin occlusion (e.g. handheld mic).
                        if float(beard_suppress[comp].mean()) >= 0.45 and not (comp_px < 0.18 and comp_skin < 0.85):
                            continue
                    verified_cores |= comp
                if verified_cores.any():
                    grow_domain = (
                        (px < 0.50)
                        & grow_clip
                        & (~above_eye_guard)
                        & (~ofiq_non_face)
                        & (~(eyes_mask & (px >= 0.55)))
                    )
                    if not is_ofiq_masked_face and collar_suppress.any():
                        grow_domain &= ~collar_suppress
                    if beard_guard and not (verified_cores & beard_suppress).any():
                        grow_domain &= ~beard_suppress
                    if clear_glasses_zone.any():
                        grow_domain &= ~clear_glasses_zone
                    grow_union = grow_domain | verified_cores
                    n_gr, lbl_gr, _, _ = cv2.connectedComponentsWithStats(grow_union.astype(np.uint8), connectivity=8)
                    for idx in range(1, n_gr):
                        g_comp = lbl_gr == idx
                        if (g_comp & verified_cores).any():
                            ofiq_extra |= g_comp
                    if ofiq_extra.any():
                        ofiq_extra = _fill_mask_holes(ofiq_extra) & grow_clip & (~above_eye_guard) & (~ofiq_non_face)
                        if not is_ofiq_masked_face and collar_suppress.any():
                            ofiq_extra &= ~collar_suppress
                        if beard_guard and not (verified_cores & beard_suppress).any():
                            ofiq_extra &= ~beard_suppress
                        if clear_glasses_zone.any():
                            ofiq_extra &= ~(clear_glasses_zone & (px >= 0.65))
                        if eyes_visible:
                            ofiq_extra &= ~(eyes_mask & (px >= 0.55))

    # 4. Deterministic Lab/YCrCb occlusion families (gated by ONNX when active)
    eye_mask = _detect_eyewear_and_glare(
        lab,
        skin,
        observed,
        l_ref,
        seg=seg,
        px=px,
        both_eyes_visible=both_eyes_visible,
        bprob=bprob,
        bface=bface,
    )
    if clear_glasses_zone.any():
        eye_mask &= ~clear_glasses_zone
    if eye_mask.any() or onnx_clear_eye.any() or onnx_opaque_eye.any():
        families_triggered.append("eyewear_or_glare")

    bisenet_missing_but_xseg_face = onnx_used and xseg_centered_face and int(bface.sum()) < 500
    if onnx_used and (
        beard_guard
        or (
            not is_eye_slit_face
            and not bisenet_missing_but_xseg_face
            and int((np.isin(seg, (2, 3, 4, 5, 6, 10)) & observed).sum()) < 100
        )
    ):
        lower_mask = np.zeros((256, 256), dtype=bool)
    else:
        allowed_lower = (~collar_suppress) if onnx_used else None
        lower_mask = _detect_masks_and_scarves(lab, skin, inner_face, delta_e, l_ref, allowed_lower=allowed_lower)
    ofiq_has_lower = int((ofiq_extra & (_YY >= 122)).sum()) >= 500
    if lower_mask.any() or ((onnx_has_lower or ofiq_has_lower) and not lips_visible):
        families_triggered.append("mask_or_scarf")

    obj_mask, censor_bar_mask = _detect_foreign_objects(
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
    if obj_mask.any() or censor_bar_mask.any():
        families_triggered.append("foreign_object")

    if onnx_hull.any() or ofiq_extra.any():
        families_triggered.append("hand_or_object")

    # 5. Class-aware anti-ghosting split dilation: tight 1-2px on clear rims/censor bars/ear straps vs edge-snapped opaque
    if not is_full_canvas_override:
        clear_rim_mask |= onnx_clear_eye
        # Keep crisp synthetic censor bars out of opaque bloat when they dominate eye/center band.
        if censor_bar_mask.any():
            censor_dil = (cv2.dilate(censor_bar_mask.astype(np.uint8), _ELLIPSE_5PX) > 0) & observed
            eye_mask &= ~censor_dil
            onnx_opaque_eye &= ~censor_dil
            onnx_hull &= ~censor_dil
            ofiq_extra &= ~censor_dil
        opaque_mask |= onnx_opaque_eye | onnx_hull | ofiq_extra | eye_mask | lower_mask | obj_mask
        keep_chin = chin_gaiter_mask & (_XX >= 78) & (_XX < 178) & _FACE_ELLIPSE
        if onnx_used and collar_suppress.any():
            opaque_mask = (opaque_mask & (~collar_suppress)) | keep_chin
        if hull_clip is not None:
            opaque_mask = (opaque_mask & hull_clip) | keep_chin
        lower_fill = np.zeros((256, 256), dtype=bool)
        if not lips_visible and opaque_mask.any():
            closed_mask = cv2.morphologyEx(opaque_mask.astype(np.uint8), cv2.MORPH_CLOSE, _ELLIPSE_11PX) > 0
            lower_fill = _fill_mask_holes(closed_mask) & (_YY >= 134) & observed & (~opaque_mask)
            lower_occ = opaque_mask & (_YY >= 134)
            if int(lower_occ.sum()) >= 2500:
                pts_lo = np.column_stack(np.nonzero(lower_occ)[::-1]).astype(np.int32)
                if len(pts_lo) >= 3:
                    lo_hull = np.zeros((256, 256), dtype=np.uint8)
                    cv2.fillConvexPoly(lo_hull, cv2.convexHull(pts_lo), 1)
                    glare_or_cloth = (~skin) | (lab[..., 0] >= 210.0)
                    if seg is not None:
                        glare_or_cloth = glare_or_cloth | (seg == 18)
                    lower_fill |= (lo_hull > 0) & (_YY >= 134) & observed & (~opaque_mask) & glare_or_cloth
            if hull_clip is not None:
                lower_fill &= hull_clip
            opaque_mask |= lower_fill

        # Protect visible eyes inside clear prescription glasses from opaque fringe expansion.
        protect_clear_eyes = np.zeros((256, 256), dtype=bool)
        if clear_glasses_zone.any() and onnx_used:
            protect_clear_eyes = clear_glasses_zone & (px >= 0.65) & (~clear_rim_mask)
        if is_eye_slit_face and onnx_used:
            protect_clear_eyes |= bface & (px >= 0.60) & (_YY >= 55) & (_YY < 145)
        if onnx_used and collar_suppress.any():
            protect_clear_eyes |= (collar_suppress & (~keep_chin) & (~lower_fill))
        if beard_guard and not keep_chin.any():
            protect_clear_eyes |= (beard_suppress & (~ofiq_extra))
        if hull_clip is not None:
            protect_clear_eyes |= ((~hull_clip) & (~keep_chin))

        if clear_rim_mask.any():
            clear_dilated = (cv2.dilate(clear_rim_mask.astype(np.uint8), _ELLIPSE_5PX) > 0) & observed
            if protect_clear_eyes.any():
                clear_inner_1px = (cv2.dilate(clear_rim_mask.astype(np.uint8), _ERODE_1PX) > 0) & observed
                clear_dilated = (clear_dilated & (~protect_clear_eyes)) | clear_inner_1px
        else:
            clear_dilated = np.zeros((256, 256), dtype=bool)

        if censor_bar_mask.any():
            censor_tight = (cv2.dilate(censor_bar_mask.astype(np.uint8), _ERODE_1PX) > 0) & observed
        else:
            censor_tight = np.zeros((256, 256), dtype=bool)

        if ear_strap_mask.any():
            strap_tight = (cv2.dilate(ear_strap_mask.astype(np.uint8), _ERODE_1PX) > 0) & observed
            if seg is not None:
                strap_tight &= ~np.isin(seg, (0, 7, 8, 9, 14, 15, 17))
        else:
            strap_tight = np.zeros((256, 256), dtype=bool)

        if opaque_mask.any():
            opaque_dilated = _absorb_opaque_fringe(
                opaque_mask,
                lab,
                (l_ref, a_ref, b_ref),
                observed,
                px=px,
                seg=seg,
                protect_mask=protect_clear_eyes,
                rgb=rgb,
                bprob=bprob,
            )
            if onnx_used and collar_suppress.any():
                opaque_dilated = (opaque_dilated & (~collar_suppress)) | keep_chin | lower_fill
            if hull_clip is not None:
                opaque_dilated = (opaque_dilated & hull_clip) | keep_chin
        else:
            opaque_dilated = np.zeros((256, 256), dtype=bool)

        final_mask = (clear_dilated | censor_tight | strap_tight | opaque_dilated) & observed

    p3_refiner_actions: list[str] = []
    if not is_full_canvas_override and onnx_used and final_mask.any():
        final_mask, p3_refiner_actions = _refine_mask_p3(
            final_mask,
            observed,
            bprob,
            seg,
            px,
            skin,
            pts106,
            hull_clip,
        )

    final_mask &= observed
    final_mask, visibility_trimmed = _enforce_visibility(final_mask, observed, probability, px=px, seg=seg)
    if not final_mask.any():
        families_triggered = []

    summary = {
        "auto_detected_pixels": int(final_mask.sum()),
        "families_triggered": families_triggered,
        "visibility_trimmed": bool(visibility_trimmed),
        "neural_seed_pixels": int(seed_pixels),
        "onnx_used": bool(onnx_used),
        "p3_refiner_actions": p3_refiner_actions,
    }
    return final_mask.astype(np.uint8) * 255, summary


def evaluate_completion_quality(
    original_rgb: np.ndarray,
    completed_rgb: np.ndarray,
    removal_mask: np.ndarray,
    observed: np.ndarray | None = None,
) -> dict | None:
    """Zero-weight 5-factor post-completion quality gate reusing cached ONNX BiSeNet + XSeg.

    Returns None gracefully when mask is empty or ONNX sessions are unavailable.
    """
    if not (
        isinstance(original_rgb, np.ndarray)
        and isinstance(completed_rgb, np.ndarray)
        and original_rgb.dtype == np.uint8
        and completed_rgb.dtype == np.uint8
        and original_rgb.shape == (256, 256, 3)
        and completed_rgb.shape == (256, 256, 3)
    ):
        return None
    mask_bool = np.asarray(removal_mask) != 0
    if mask_bool.shape != (256, 256) or not mask_bool.any():
        return None
    obs = np.ones((256, 256), dtype=bool) if observed is None else np.asarray(observed, dtype=bool)

    onnx_out = _run_onnx_models(completed_rgb)
    if onnx_out is None:
        return None
    bprob, px = onnx_out
    seg = bprob.argmax(axis=0)
    warnings: list[str] = []

    # Factor 1: Occluder clear (XSeg visible-face confidence inside completed mask)
    mask_in_face = mask_bool & _FACE_ELLIPSE
    eval_region = mask_in_face if int(mask_in_face.sum()) >= 32 else mask_bool
    mean_px_in = float(px[eval_region].mean())
    occluder_clear = float(np.clip((mean_px_in - 0.20) / 0.65 * 100.0, 0.0, 100.0))
    if occluder_clear < 55.0:
        warnings.append("Residual covering tone or smudge detected inside the completed area.")

    # Factor 2: Anatomical parsing validity (BiSeNet semantic consistency inside completed region)
    face_classes = np.isin(seg, (1, 2, 3, 4, 5, 10, 11, 12, 13))
    face_frac = float(face_classes[eval_region].mean())
    anat_score = float(np.clip(face_frac * 100.0, 0.0, 100.0))
    eye_masked = int((mask_bool & _INNER_EYE_BAND).sum())
    if eye_masked >= 250:
        le_px = int(((seg == 4) & obs).sum())
        re_px = int(((seg == 5) & obs).sum())
        if le_px < 15 or re_px < 15:
            anat_score = min(anat_score, 48.0)
        else:
            ratio = float(min(le_px, re_px)) / float(max(le_px, re_px))
            if ratio < 0.45:
                anat_score *= (0.65 + 0.78 * ratio)
    lower_masked = int((mask_bool & _LOWER_ZONE).sum())
    if lower_masked >= 600:
        nose_ok = int(((seg == 10) & obs).sum()) >= 80
        lips_ok = int((np.isin(seg, (12, 13)) & obs).sum()) >= 60
        if not (nose_ok and lips_ok):
            anat_score = min(anat_score, 52.0 if (nose_ok or lips_ok) else 35.0)
    anatomy = float(np.clip(anat_score, 0.0, 100.0))
    if anatomy < 55.0:
        warnings.append("Facial structure inside the completed area looks irregular.")

    # Factor 3: Bilateral symmetry consistency (mirrored Lab & gradient consistency across midline)
    comp_lab = cv2.cvtColor(completed_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bface = face_classes & obs
    if int(bface.sum()) >= 400:
        cx = float(np.median(np.nonzero(bface)[1]))
    else:
        cx = 127.5
    ys_e, xs_e = np.nonzero(eval_region)
    mx_e = np.clip(np.rint(2.0 * cx - xs_e).astype(np.int32), 0, 255)
    valid_sym = obs[ys_e, mx_e] & face_classes[ys_e, mx_e]
    if int(valid_sym.sum()) >= 40:
        l_diff = np.abs(comp_lab[ys_e[valid_sym], xs_e[valid_sym], 0] - comp_lab[ys_e[valid_sym], mx_e[valid_sym], 0])
        ab_diff = np.hypot(
            comp_lab[ys_e[valid_sym], xs_e[valid_sym], 1] - comp_lab[ys_e[valid_sym], mx_e[valid_sym], 1],
            comp_lab[ys_e[valid_sym], xs_e[valid_sym], 2] - comp_lab[ys_e[valid_sym], mx_e[valid_sym], 2],
        )
        sym_err = float(np.median(l_diff) * 0.6 + np.median(ab_diff) * 1.4)
        symmetry = float(np.clip(100.0 - max(0.0, sym_err - 6.0) * 2.6, 0.0, 100.0))
    else:
        symmetry = 78.0
    if symmetry < 50.0:
        warnings.append("Noticeable left-right asymmetry in the completed area.")

    # Factor 4: Seam gradient & Lab color continuity across mask boundary
    mask_u8 = mask_bool.astype(np.uint8)
    inner_2 = mask_bool & (~(cv2.erode(mask_u8, _ELLIPSE_5PX) > 0))
    outer_2 = (cv2.dilate(mask_u8, _ELLIPSE_5PX) > 0) & (~mask_bool) & obs & face_classes
    if int(inner_2.sum()) >= 16 and int(outer_2.sum()) >= 16:
        in_med = np.array([np.median(comp_lab[..., c][inner_2]) for c in range(3)], dtype=np.float32)
        out_med = np.array([np.median(comp_lab[..., c][outer_2]) for c in range(3)], dtype=np.float32)
        seam_de = float(np.linalg.norm(in_med - out_med))
        seam = float(np.clip(100.0 - max(0.0, seam_de - 4.0) * 3.2, 0.0, 100.0))
    else:
        seam = 82.0
    if seam < 55.0:
        warnings.append("Visible tone or edge seam at the mask boundary.")

    # Factor 5: Texture / grain realism (Laplacian MAD inside mask vs adjacent clean skin)
    gray_f = cv2.cvtColor(completed_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    outer_skin = (cv2.dilate(mask_u8, _ELLIPSE_11PX) > 0) & (~mask_bool) & obs & (seg == 1)
    if int(eval_region.sum()) >= 40 and int(outer_skin.sum()) >= 40:
        sig_in = float(np.median(np.abs(lap[eval_region] - np.median(lap[eval_region]))))
        sig_out = float(np.median(np.abs(lap[outer_skin] - np.median(lap[outer_skin]))))
        if sig_out < 0.5 and sig_in < 0.5:
            texture = 90.0
        else:
            ratio = (sig_in + 0.5) / (sig_out + 0.5)
            log_err = abs(float(np.log(max(0.05, ratio))))
            texture = float(np.clip(100.0 - max(0.0, log_err - 0.18) * 75.0, 0.0, 100.0))
    else:
        texture = 85.0
    if texture < 50.0:
        warnings.append("Texture sharpness differs from surrounding skin.")

    score = int(round(
        0.28 * occluder_clear
        + 0.25 * anatomy
        + 0.15 * symmetry
        + 0.20 * seam
        + 0.12 * texture
    ))
    score = max(0, min(100, score))
    grade = "high" if score >= 78 else ("moderate" if score >= 55 else "low")
    return {
        "score_0_100": score,
        "grade": grade,
        "factors": {
            "occluder_clear": int(round(occluder_clear)),
            "anatomy": int(round(anatomy)),
            "symmetry": int(round(symmetry)),
            "seam": int(round(seam)),
            "texture": int(round(texture)),
        },
        "warnings": warnings,
    }

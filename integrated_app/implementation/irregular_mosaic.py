"""Experimental detector for regions composed of small, nearly flat cells."""

import cv2
import numpy as np
from scipy.ndimage import (binary_closing, binary_fill_holes, find_objects, label,
                          uniform_filter)


def _cell_is_compact(cell_mask):
    """Reject thin curved arc fragments typical of gradient contour bands."""
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    if contour_area < 2.0:
        return True
    perimeter = cv2.arcLength(contour, True)
    if perimeter * perimeter > 48.0 * contour_area:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return hull_area <= 0.0 or contour_area >= 0.52 * hull_area


def _is_tonal_banding_region(region_rgb, local_mask):
    """Identify curved/quantized gradient banding regions by step concentration."""
    dh = np.abs(np.diff(region_rgb, axis=1)).max(-1)
    dv = np.abs(np.diff(region_rgb, axis=0)).max(-1)
    loc_h = local_mask[:, :-1] & local_mask[:, 1:] & (dh > 1)
    loc_v = local_mask[:-1, :] & local_mask[1:, :] & (dv > 1)
    steps = np.concatenate([dh[loc_h], dv[loc_v]])
    if steps.size < 24:
        return False
    med_step = float(np.median(steps))
    p90_step = float(np.percentile(steps, 90))
    if med_step > 16.0 or p90_step > 28.0:
        return False
    _, step_counts = np.unique(steps, return_counts=True)
    top3_frac = float(np.sort(step_counts)[-3:].sum() / step_counts.sum())
    dh_ch = (np.abs(np.diff(region_rgb, axis=1)) > 1).sum(-1)
    dv_ch = (np.abs(np.diff(region_rgb, axis=0)) > 1).sum(-1)
    one_ch_frac = float(np.concatenate([dh_ch[loc_h] == 1, dv_ch[loc_v] == 1]).mean())
    return (one_ch_frac >= 0.70 and top3_frac >= 0.55) or top3_frac >= 0.80


def _boundary_step_ratio(rgb, local_mask, present_local, tol):
    """Verify sharp cell-boundary gradient contrast vs flat cell interiors for JPEG-rippled polygons."""
    dh = np.abs(np.diff(rgb, axis=1)).max(-1).astype(np.float32)
    dv = np.abs(np.diff(rgb, axis=0)).max(-1).astype(np.float32)
    in_h = present_local[:, :-1] & present_local[:, 1:] & local_mask[:, :-1] & local_mask[:, 1:]
    in_v = present_local[:-1, :] & present_local[1:, :] & local_mask[:-1, :] & local_mask[1:, :]
    # Cell boundaries lie inside local_mask where at least one side exits a flat cell interior.
    bd_h = local_mask[:, :-1] & local_mask[:, 1:] & (~in_h) & (dh > tol)
    bd_v = local_mask[:-1, :] & local_mask[1:, :] & (~in_v) & (dv > tol)
    interior_vals = np.concatenate([dh[in_h], dv[in_v]])
    boundary_vals = np.concatenate([dh[bd_h], dv[bd_v]])
    if interior_vals.size < 24 or boundary_vals.size < 24:
        return 0.0, 0.0
    int_med = max(0.75, float(np.median(interior_vals)))
    bd_med = float(np.median(boundary_vals))
    return bd_med / int_med, bd_med


def _detect_irregular_pass(rgb, *, tol=1, max_area=200, min_cells=24, min_density=0.27, min_step_ratio=0.0):
    horizontal = np.abs(np.diff(rgb, axis=1)).max(-1)
    vertical = np.abs(np.diff(rgb, axis=0)).max(-1)
    flat = np.ones(rgb.shape[:2], dtype=bool)
    for variation, left, right in ((horizontal, flat[:, :-1], flat[:, 1:]),
                                    (vertical, flat[:-1, :], flat[1:, :])):
        left &= variation <= tol
        right &= variation <= tol
    components, count = label(flat)
    area = np.bincount(components.ravel(), minlength=count + 1)
    centers = np.zeros(flat.shape, dtype=np.float32)
    cell_ids = []
    for cell_id, bounds in enumerate(find_objects(components), 1):
        if not 6 <= area[cell_id] <= max_area or bounds is None:
            continue
        height, width = (axis.stop - axis.start for axis in bounds)
        if min(height, width) < 2 or max(height, width) > 3.5 * min(height, width):
            continue
        if area[cell_id] < .43 * height * width:
            continue
        if not _cell_is_compact((components[bounds] == cell_id).astype(np.uint8)):
            continue
        cell_ids.append(cell_id)
        y, x = (int((axis.start + axis.stop - 1) // 2) for axis in bounds)
        centers[y, x] = 1.
    if len(cell_ids) < min_cells:
        return np.zeros(flat.shape, dtype=bool)
    present = np.isin(components, cell_ids)
    density = uniform_filter(present.astype(np.float32), size=23, mode="constant")
    count_map = uniform_filter(centers, size=29, mode="constant") * 29**2
    supported = binary_fill_holes(binary_closing((density >= min_density) & (count_map >= 6), iterations=3))
    regions, _ = label(supported)
    output = np.zeros(flat.shape, dtype=bool)
    for index, bounds in enumerate(find_objects(regions), 1):
        local = regions[bounds] == index
        if local.sum() < max(200, flat.size * .008):
            continue
        region_rgb = rgb[bounds]
        samples = (region_rgb[local & present[bounds]] // 12)
        if not samples.size:
            continue
        _, colours = np.unique(samples, axis=0, return_counts=True)
        if len(colours) >= 8 and np.sort(colours)[-2:].sum() <= .60 * len(samples):
            if _is_tonal_banding_region(region_rgb, local):
                continue
            if min_step_ratio > 0.0:
                step_ratio, bd_med = _boundary_step_ratio(region_rgb, local, present[bounds], tol)
                if step_ratio < min_step_ratio or bd_med < 5.0:
                    continue
            output[bounds] |= local
    return output


def irregular_mosaic_mask(source):
    """Find concentrated, diverse bounded interiors without assuming a grid."""
    rgb = source.astype(np.int16)
    strict = _detect_irregular_pass(rgb, tol=1, max_area=200, min_cells=24, min_density=0.27, min_step_ratio=0.0)
    if strict.any():
        return strict
    # JPEG/PNG-recompressed Voronoi & low-poly mosaics exhibit +-3 LSB interior ripple;
    # require crisp cell-boundary gradient step ratio >= 3.8x to prevent natural-texture FPs.
    relaxed = _detect_irregular_pass(rgb, tol=3, max_area=550, min_cells=18, min_density=0.20, min_step_ratio=3.8)
    if relaxed.any():
        return relaxed
    if isinstance(source, np.ndarray) and source.dtype == np.uint8 and min(source.shape[:2]) >= 48:
        med = cv2.medianBlur(source, 3).astype(np.int16)
        return _detect_irregular_pass(med, tol=3, max_area=550, min_cells=18, min_density=0.20, min_step_ratio=3.8)
    return relaxed

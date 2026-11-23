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


def irregular_mosaic_mask(source):
    """Find concentrated, diverse bounded interiors without assuming a grid."""
    rgb = source.astype(np.int16)
    horizontal = np.abs(np.diff(rgb, axis=1)).max(-1)
    vertical = np.abs(np.diff(rgb, axis=0)).max(-1)
    flat = np.ones(rgb.shape[:2], dtype=bool)
    for variation, left, right in ((horizontal, flat[:, :-1], flat[:, 1:]),
                                    (vertical, flat[:-1, :], flat[1:, :])):
        left &= variation <= 1
        right &= variation <= 1
    components, count = label(flat)
    area = np.bincount(components.ravel(), minlength=count + 1)
    centers = np.zeros(flat.shape, dtype=np.float32)
    cell_ids = []
    for cell_id, bounds in enumerate(find_objects(components), 1):
        if not 6 <= area[cell_id] <= 200 or bounds is None:
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
    if len(cell_ids) < 24:
        return np.zeros(flat.shape, dtype=bool)
    present = np.isin(components, cell_ids)
    density = uniform_filter(present.astype(np.float32), size=23, mode="constant")
    count = uniform_filter(centers, size=29, mode="constant") * 29**2
    supported = binary_fill_holes(binary_closing((density >= .27) & (count >= 6), iterations=3))
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
            output[bounds] |= local
    return output

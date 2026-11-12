"""Experimental detector for regions composed of small, nearly flat cells."""

import numpy as np
from scipy.ndimage import (binary_closing, binary_fill_holes, find_objects, label,
                          uniform_filter)


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
        samples = (rgb[bounds][local & present[bounds]] // 12)
        if not samples.size:
            continue
        _, colours = np.unique(samples, axis=0, return_counts=True)
        if len(colours) >= 8 and np.sort(colours)[-2:].sum() <= .60 * len(samples):
            output[bounds] |= local
    return output

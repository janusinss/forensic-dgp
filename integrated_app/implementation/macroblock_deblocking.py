"""Conservative CPU detection and adaptive deblocking for 8x8/16x16 macroblocking."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

from .jpeg_deblocking import MAX_SIDE as FBCNN_MAX_SIDE, prepare_fbcnn_deblocking
from .preparation import png

MIN_CROP_SIDE = 32
FBCNN_MIN_SIDE = 96


def _bt601_luma(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB (H, W, 3) to float32 BT.601 luma."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return r * np.float32(0.299) + g * np.float32(0.587) + b * np.float32(0.114)


def _axis_diff_profile(diffs: np.ndarray, axis: int, mask_bool: np.ndarray | None = None) -> np.ndarray | None:
    """Average gradient profile along `axis`, optionally restricted to pixels inside `mask_bool`."""
    if mask_bool is None:
        return diffs.mean(axis=1 - axis)
    allowed = mask_bool[:-1, :] & mask_bool[1:, :] if axis == 0 else mask_bool[:, :-1] & mask_bool[:, 1:]
    counts = allowed.sum(axis=1 - axis)
    if int((counts > 0).sum()) < 24:
        return None
    sums = np.where(allowed, diffs, np.float32(0.0)).sum(axis=1 - axis)
    valid = counts > 0
    return np.where(valid, sums / np.maximum(counts, 1), np.nan)


def _line_excess_stats(profile: np.ndarray, period: int, offset: int) -> tuple[float, float, float] | None:
    """Compare each periodic boundary line against adjacent interior lines to reject isolated edges."""
    length = len(profile)
    idx = np.arange(length)
    pos = (idx + 1 + offset) % period
    boundary_idx = idx[(pos == 0) & np.isfinite(profile)]
    if boundary_idx.size < (1 if length < 2 * period else 2):
        return None
    half = period // 2
    excesses = []
    for k in boundary_idx:
        lo = max(0, int(k) - half + 1)
        hi = min(length, int(k) + half)
        win = idx[lo:hi]
        win_pos = (win + 1 + offset) % half
        interior_idx = win[(win_pos != 0) & np.isfinite(profile[win])]
        if interior_idx.size < 2:
            continue
        excesses.append(float(profile[k] - np.mean(profile[interior_idx])))
    if len(excesses) < (1 if length < 2 * period else 2):
        return None
    arr = np.asarray(excesses, dtype=np.float32)
    pos_frac = float((arr > 0.10).mean())
    med = float(np.median(arr))
    trimmed = float(np.sort(arr)[:-1].mean()) if arr.size >= 4 else med
    return pos_frac, med, trimmed


def _phase16_components(profile: np.ndarray, offset16: int) -> tuple[float, float, float, float] | None:
    """Return (seam16, odd8_excess, quarter4_excess, strict_interior) for phase `offset16`."""
    valid = np.isfinite(profile)
    if int(valid.sum()) < 24:
        return None
    pos = (np.arange(len(profile)) + 1 + offset16) % 16
    b16 = profile[(pos == 0) & valid]
    m16 = profile[(pos == 8) & valid]
    q16 = profile[((pos == 4) | (pos == 12)) & valid]
    i16 = profile[((pos % 4) != 0) & valid]
    if b16.size == 0 or i16.size == 0 or q16.size == 0:
        return None
    i_val = float(i16.mean())
    s16 = float(b16.mean()) - i_val
    s_odd8 = (float(m16.mean()) - i_val) if m16.size > 0 else 0.0
    s_q4 = float(q16.mean()) - i_val
    return s16, s_odd8, s_q4, i_val


def _classify_axis(profile: np.ndarray, origin_offset: int) -> dict | None:
    """Classify one axis profile as period 8, 16, or hierarchical (8+16), or None."""
    valid = np.isfinite(profile)
    if int(valid.sum()) < 24:
        return None
    orig16 = int(origin_offset) % 16
    orig8 = int(origin_offset) % 8

    stats16 = {}
    for off in range(16):
        comp = _phase16_components(profile, off)
        if comp is not None:
            stats16[off] = comp
    if not stats16:
        return None

    stats8 = {}
    pos_base = np.arange(len(profile)) + 1
    for off8 in range(8):
        pos8 = (pos_base + off8) % 8
        b8 = profile[(pos8 == 0) & valid]
        m8 = profile[(pos8 == 4) & valid]
        i8 = profile[((pos8 != 0) & (pos8 != 4)) & valid]
        if b8.size > 0 and m8.size > 0 and i8.size > 0:
            iv = float(i8.mean())
            stats8[off8] = (float(b8.mean()) - iv, float(m8.mean()) - iv, iv)

    # 1) Evaluate 8-px candidates (checking orig8 first)
    best_off8 = None
    if orig8 in stats8:
        s8, mid8, i8 = stats8[orig8]
        if s8 >= max(0.65, 0.28 * i8) and mid8 <= 0.60 * s8:
            best_off8 = orig8
    if best_off8 is None and stats8:
        cand8 = [(off, vals) for off, vals in stats8.items() if vals[0] > 0 and vals[1] <= 0.60 * vals[0]]
        if cand8:
            best_off8 = max(cand8, key=lambda item: item[1][0])[0]

    if best_off8 is not None:
        s8, mid8, i8 = stats8[best_off8]
        if s8 >= max(0.65, 0.28 * i8) and mid8 <= 0.60 * s8:
            c_even = stats16.get(best_off8)
            c_odd = stats16.get(best_off8 + 8)
            if c_even is not None and c_odd is not None:
                e_even = c_even[0]
                e_odd = c_odd[0]
                e_hi = max(e_even, e_odd)
                e_lo = min(e_even, e_odd)
                hi_off = best_off8 if e_even >= e_odd else best_off8 + 8
                if orig16 in (best_off8, best_off8 + 8) and stats16[orig16][0] >= 0.85 * e_hi:
                    hi_off = orig16
                i_strict = stats16[hi_off][3]
                q4 = stats16[hi_off][2]
                if e_hi >= max(0.65, 0.28 * i_strict) and q4 <= 0.60 * e_hi:
                    if e_lo < max(0.45, 0.20 * i_strict) or e_lo <= 0.35 * e_hi:
                        line = _line_excess_stats(profile, 16, hi_off)
                        if line is not None and line[0] >= 0.50 and line[1] > 0.12 and line[2] >= max(0.30, 0.16 * i_strict):
                            return {
                                'period': 16, 'hierarchical': False, 'offset': int(hi_off),
                                'seam': float(e_hi), 'midpoint': float(e_lo), 'interior': float(i_strict),
                                'seam8': float(s8), 'line_pos_frac': line[0],
                            }
                    elif (e_hi - e_lo) >= max(0.80, 0.45 * e_hi) and e_lo <= 0.52 * e_hi and q4 <= 0.60 * e_lo:
                        line16 = _line_excess_stats(profile, 16, hi_off)
                        line8 = _line_excess_stats(profile, 8, best_off8)
                        if (line16 is not None and line8 is not None and line16[0] >= 0.50 and line8[0] >= 0.50
                                and line8[1] > 0.12 and line8[2] >= max(0.30, 0.16 * i_strict)):
                            return {
                                'period': 16, 'hierarchical': True, 'offset': int(hi_off),
                                'seam': float(e_hi), 'midpoint': float(q4), 'interior': float(i_strict),
                                'seam8': float(s8), 'odd8': float(e_lo), 'line_pos_frac': line8[0],
                            }
            line8 = _line_excess_stats(profile, 8, best_off8)
            if line8 is not None and line8[0] >= 0.50 and line8[1] > 0.12 and line8[2] >= max(0.30, 0.16 * i8):
                return {
                    'period': 8, 'hierarchical': False, 'offset': int(best_off8),
                    'seam': float(s8), 'midpoint': float(mid8), 'interior': float(i8),
                    'seam8': float(s8), 'line_pos_frac': line8[0],
                }

    # 2) Pure 16-px check (e.g. if odd-8 average diluted s8 below threshold while s16 is strong)
    best_off16 = None
    if orig16 in stats16:
        s16, s_odd8, s_q4, i16 = stats16[orig16]
        if s16 >= max(0.65, 0.28 * i16) and s_q4 <= 0.60 * s16 and s_odd8 <= 0.45 * s16:
            best_off16 = orig16
    if best_off16 is None:
        cand16 = [
            (off, vals) for off, vals in stats16.items()
            if vals[0] > 0 and vals[2] <= 0.60 * vals[0] and vals[1] <= 0.45 * vals[0]
        ]
        if cand16:
            best_off16 = max(cand16, key=lambda item: item[1][0])[0]
    if best_off16 is not None:
        s16, s_odd8, s_q4, i16 = stats16[best_off16]
        if s16 >= max(0.65, 0.28 * i16) and s_q4 <= 0.60 * s16:
            line16 = _line_excess_stats(profile, 16, best_off16)
            if line16 is not None and line16[0] >= 0.50 and line16[1] > 0.12 and line16[2] >= max(0.30, 0.16 * i16):
                return {
                    'period': 16, 'hierarchical': False, 'offset': int(best_off16),
                    'seam': float(s16), 'midpoint': float(s_odd8), 'interior': float(i16),
                    'seam8': float(0.5 * (s16 + s_odd8)), 'line_pos_frac': line16[0],
                }
    return None


def _tile_scan_support(dy: np.ndarray, dx: np.ndarray, height: int, width: int, origin: tuple[int, int]) -> tuple[np.ndarray, dict | None]:
    """Fast vectorized 48x48 stride-16 scan to localize macroblocked regions."""
    tile, step = 48, 16
    k = tile // 16
    xs = np.arange(0, width - tile + 1, step)
    ys = np.arange(0, height - tile + 1, step)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((height, width), dtype=bool), None

    cs_y = np.pad(np.cumsum(dy.astype(np.float64), axis=1), ((0, 0), (1, 0)))
    hmean_y = (cs_y[:, xs + tile] - cs_y[:, xs]) / tile
    n16_y = (height - 1) // 16
    if n16_y < k:
        return np.zeros((height, width), dtype=bool), None
    hy_blocks = hmean_y[:n16_y * 16].reshape(n16_y, 16, len(xs))
    cs_hy = np.pad(np.cumsum(hy_blocks, axis=0), ((1, 0), (0, 0), (0, 0)))
    prof_y = (cs_hy[k:] - cs_hy[:-k]) / k  # (Ny_blk, 16, Nx)

    cs_x = np.pad(np.cumsum(dx.astype(np.float64), axis=0), ((1, 0), (0, 0)))
    vmean_x = (cs_x[ys + tile, :] - cs_x[ys, :]) / tile
    n16_x = (width - 1) // 16
    if n16_x < k:
        return np.zeros((height, width), dtype=bool), None
    vx_blocks = vmean_x[:, :n16_x * 16].reshape(len(ys), n16_x, 16)
    cs_vx = np.pad(np.cumsum(vx_blocks, axis=1), ((0, 0), (1, 0), (0, 0)))
    prof_x = (cs_vx[:, k:, :] - cs_vx[:, :-k, :]) / k  # (Ny, Nx_blk, 16)

    ny = min(prof_y.shape[0], len(ys))
    nx = min(prof_x.shape[1], len(xs))
    if ny <= 0 or nx <= 0:
        return np.zeros((height, width), dtype=bool), None

    py_grid = np.transpose(prof_y[:ny, :, :nx], (0, 2, 1))  # (ny, nx, 16)
    px_grid = prof_x[:ny, :nx, :]  # (ny, nx, 16)

    # Pre-filter candidate tiles where at least one 16-slot phase strongly exceeds tile mean
    def _candidate_mask(grid: np.ndarray) -> np.ndarray:
        mean_all = grid.mean(axis=2, keepdims=True)
        # Check either single-16 peak or 8-px paired peak
        peak16 = grid.max(axis=2) - mean_all[..., 0]
        grid8 = 0.5 * (grid[..., :8] + grid[..., 8:])
        peak8 = grid8.max(axis=2) - mean_all[..., 0]
        best = np.maximum(peak16, peak8)
        return best >= np.maximum(0.50, 0.22 * mean_all[..., 0])

    cand_iy, cand_ix = np.nonzero(_candidate_mask(py_grid) & _candidate_mask(px_grid))
    if cand_iy.size == 0:
        return np.zeros((height, width), dtype=bool), None

    ox0, oy0 = origin
    groups: dict[tuple[int, int, int], list[tuple[int, int, float, dict, dict]]] = {}
    for iy, ix in zip(cand_iy.tolist(), cand_ix.tolist()):
        y0, x0 = int(ys[iy]), int(xs[ix])
        ry = _classify_axis(dy[y0:y0 + tile - 1, x0:x0 + tile].mean(axis=1), oy0 + y0)
        if ry is None:
            continue
        rx = _classify_axis(dx[y0:y0 + tile, x0:x0 + tile - 1].mean(axis=0), ox0 + x0)
        if rx is None:
            continue
        dom_p = 16 if (ry['period'] == 16 or rx['period'] == 16) else 8
        # Normalize local tile phase back into image-global phase at (0, 0)
        gy_off = (ry['offset'] - y0) % ry['period']
        gx_off = (rx['offset'] - x0) % rx['period']
        key = (dom_p, gy_off % dom_p if dom_p == 8 else gy_off % ry['period'], gx_off % dom_p if dom_p == 8 else gx_off % rx['period'])
        score = min(ry['seam'], rx['seam'])
        groups.setdefault(key, []).append((y0, x0, score, ry, rx))

    if not groups:
        return np.zeros((height, width), dtype=bool), None

    best_key, best_tiles = max(groups.items(), key=lambda kv: (len(kv[1]), sum(t[2] for t in kv[1])))
    votes = np.zeros((height, width), dtype=np.int32)
    for y0, x0, _, _, _ in best_tiles:
        y1 = height if y0 + tile >= height - 15 else y0 + tile
        x1 = width if x0 + tile >= width - 15 else x0 + tile
        votes[y0:y1, x0:x1] += 1

    if len(best_tiles) < 3:
        return np.zeros((height, width), dtype=bool), None

    comps, count = label(votes >= 1)
    support = np.zeros((height, width), dtype=bool)
    for cid in range(1, count + 1):
        comp = comps == cid
        if int((comp & (votes >= 2)).sum()) >= 1024 and int(comp.sum()) >= 48 * 64:
            support |= comp

    if not support.any():
        return np.zeros((height, width), dtype=bool), None
    support = binary_fill_holes(support)
    sample_ry = best_tiles[0][3]
    sample_rx = best_tiles[0][4]
    meta = {
        'dominant_period': int(best_key[0]),
        'tile_count': len(best_tiles),
        'sample_ry': sample_ry,
        'sample_rx': sample_rx,
    }
    return support, meta


def _flat_cell_metrics(
    gray: np.ndarray,
    period: int,
    ox: int,
    oy: int,
    mask_bool: np.ndarray | None,
) -> tuple[float, float, float, int]:
    """Measure overall flat-cell fraction and constant-cell fraction across seam-carrying blocks."""
    height, width = gray.shape
    y0 = (-oy) % period
    x0 = (-ox) % period
    ny = (height - y0) // period
    nx = (width - x0) // period
    if ny <= 0 or nx <= 0:
        return 0.0, 0.0, 0.0, 0
    cells = gray[y0:y0 + ny * period, x0:x0 + nx * period].reshape(ny, period, nx, period).transpose(0, 2, 1, 3)
    dx_in = np.abs(np.diff(cells, axis=3))
    dy_in = np.abs(np.diff(cells, axis=2))
    mean_in = 0.5 * (dx_in.mean(axis=(2, 3)) + dy_in.mean(axis=(2, 3)))
    max_in = np.maximum(dx_in.max(axis=(2, 3)), dy_in.max(axis=(2, 3)))
    exact_zero = max_in <= 0.20
    near_flat = (mean_in <= 0.25) & (max_in <= 1.5)

    b_bot = np.zeros((ny, nx), dtype=np.float32)
    b_right = np.zeros((ny, nx), dtype=np.float32)
    if ny > 1:
        b_bot[:-1, :] = np.abs(cells[1:, :, 0, :] - cells[:-1, :, -1, :]).mean(axis=2)
        b_bot[-1, :] = b_bot[-2, :]
    if nx > 1:
        b_right[:, :-1] = np.abs(cells[:, 1:, :, 0] - cells[:, :-1, :, -1]).mean(axis=2)
        b_right[:, -1] = b_right[:, -2]
    seam_cell = 0.5 * (b_bot + b_right)

    if mask_bool is not None:
        m_cells = mask_bool[y0:y0 + ny * period, x0:x0 + nx * period].reshape(ny, period, nx, period).transpose(0, 2, 1, 3)
        valid = m_cells.mean(axis=(2, 3)) >= 0.5
        if not np.any(valid):
            valid = np.ones((ny, nx), dtype=bool)
    else:
        valid = np.ones((ny, nx), dtype=bool)

    excess_cells = valid & ((seam_cell - mean_in) >= 1.25) & (seam_cell >= 1.5 * np.maximum(mean_in, 0.25))
    excess_cnt = int(excess_cells.sum())
    ex_zero = float(exact_zero[excess_cells].mean()) if excess_cnt > 0 else 0.0
    ex_flat = float(near_flat[excess_cells].mean()) if excess_cnt > 0 else 0.0
    return float(near_flat[valid].mean()), ex_zero, ex_flat, excess_cnt


def _flat_cell_fraction(gray: np.ndarray, period: int, ox: int, oy: int, mask_bool: np.ndarray | None) -> float:
    """Measure fraction of aligned P x P blocks whose interiors are nearly constant (flat privacy mosaic)."""
    flat_frac, _, _, _ = _flat_cell_metrics(gray, period, ox, oy, mask_bool)
    return flat_frac


def _axis_2d_lag_corr(diffs: np.ndarray, axis: int, lag: int, mask_bool: np.ndarray | None) -> float | None:
    """Compute 2D Pearson autocorrelation of axial differences at a pixel lag, optionally restricted to a mask."""
    if lag <= 0:
        return None
    if mask_bool is not None:
        valid = (mask_bool[:-1, :] & mask_bool[1:, :]) if axis == 0 else (mask_bool[:, :-1] & mask_bool[:, 1:])
        ys, xs = np.where(valid)
        if ys.size < 64:
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        d = diffs[y0:y1, x0:x1]
        m = valid[y0:y1, x0:x1]
    else:
        d = diffs
        m = None
    if d.shape[axis] <= lag:
        return None
    if axis == 0:
        a = d[:-lag, :]
        b = d[lag:, :]
        if m is not None:
            pm = m[:-lag, :] & m[lag:, :]
            a = a[pm]
            b = b[pm]
        else:
            a = a.ravel()
            b = b.ravel()
    else:
        a = d[:, :-lag]
        b = d[:, lag:]
        if m is not None:
            pm = m[:, :-lag] & m[:, lag:]
            a = a[pm]
            b = b[pm]
        else:
            a = a.ravel()
            b = b.ravel()
    if a.size < 64:
        return None
    a = a.astype(np.float64, copy=False) - float(a.mean())
    b = b.astype(np.float64, copy=False) - float(b.mean())
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def _has_offgrid_lag_peak(diffs: np.ndarray, axis: int, period: int, mask_bool: np.ndarray | None) -> bool:
    """Return True if an off-grid neighbor lag exhibits stronger periodic correlation than target period."""
    c_target = _axis_2d_lag_corr(diffs, axis, period, mask_bool)
    for lag in (period - 1, period + 1):
        c_off = _axis_2d_lag_corr(diffs, axis, lag, mask_bool)
        if c_off is not None and c_off >= 0.20 and (c_target is None or c_off > c_target):
            return True
    for lag in (period - 2, period + 2):
        c_off = _axis_2d_lag_corr(diffs, axis, lag, mask_bool)
        if c_off is None or c_off < 0.20 or (c_target is not None and c_off <= c_target):
            continue
        c_lo = _axis_2d_lag_corr(diffs, axis, lag - 1, mask_bool)
        c_hi = _axis_2d_lag_corr(diffs, axis, lag + 1, mask_bool)
        if (c_lo is None or c_off > c_lo) and (c_hi is None or c_off > c_hi):
            return True
    return False


def _has_coherent_2d_peak(diffs: np.ndarray, axis: int, period: int, mask_bool: np.ndarray | None) -> bool:
    """Return True if target period is a strong (>= 0.20) local 2D lag correlation peak over p +/- 1."""
    c_target = _axis_2d_lag_corr(diffs, axis, period, mask_bool)
    if c_target is None or c_target < 0.20:
        return False
    c_prev = _axis_2d_lag_corr(diffs, axis, period - 1, mask_bool)
    c_next = _axis_2d_lag_corr(diffs, axis, period + 1, mask_bool)
    return (c_prev is None or c_target > c_prev) and (c_next is None or c_target > c_next)


def detect_macroblocking(source: np.ndarray, origin: tuple[int, int] = (0, 0)) -> dict | None:
    """Detect 8x8, 16x16, or hierarchical macroblocking and return region/grid metrics or None."""
    if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        return None
    height, width = source.shape[:2]
    if min(height, width) < MIN_CROP_SIDE:
        return None
    ox0, oy0 = (int(origin[0]), int(origin[1])) if origin is not None else (0, 0)
    gray = _bt601_luma(source)
    if float(gray.std()) < 1.5:
        return None

    dy = np.abs(np.diff(gray, axis=0))
    dx = np.abs(np.diff(gray, axis=1))

    support_mask = np.zeros((height, width), dtype=bool)
    if min(height, width) >= 112:
        support_mask, _ = _tile_scan_support(dy, dx, height, width, (ox0, oy0))

    has_support = bool(support_mask.any())
    support_frac = float(support_mask.mean()) if has_support else 0.0

    # Global classification first on whole frame, plus localized classification when regional (< 72%)
    prof_y_g = _axis_diff_profile(dy, 0, None)
    prof_x_g = _axis_diff_profile(dx, 1, None)
    ry_g = _classify_axis(prof_y_g, oy0) if prof_y_g is not None else None
    rx_g = _classify_axis(prof_x_g, ox0) if prof_x_g is not None else None

    eval_mask = None
    is_regional = False
    ry, rx = ry_g, rx_g

    if has_support and support_frac < 0.72:
        prof_y_l = _axis_diff_profile(dy, 0, support_mask)
        prof_x_l = _axis_diff_profile(dx, 1, support_mask)
        ry_l = _classify_axis(prof_y_l, oy0) if prof_y_l is not None else None
        rx_l = _classify_axis(prof_x_l, ox0) if prof_x_l is not None else None
        if ry_l is not None and rx_l is not None:
            glob_strong = (
                ry_g is not None and rx_g is not None
                and min(ry_g['seam'], rx_g['seam']) >= 0.65 * min(ry_l['seam'], rx_l['seam'])
            )
            if not glob_strong:
                eval_mask = support_mask
                is_regional = True
                ry, rx = ry_l, rx_l

    if ry is None or rx is None:
        return None

    py, px = int(ry['period']), int(rx['period'])
    if _has_offgrid_lag_peak(dy, 0, py, eval_mask) or _has_offgrid_lag_peak(dx, 1, px, eval_mask):
        return None
    if is_regional and float(support_mask.mean()) < 0.15:
        if min(ry['line_pos_frac'], rx['line_pos_frac']) < 0.70:
            return None
        if min(ry['seam'] / max(ry['interior'], 1e-4), rx['seam'] / max(rx['interior'], 1e-4)) < 0.65:
            return None
        if not (_has_coherent_2d_peak(dy, 0, py, eval_mask) and _has_coherent_2d_peak(dx, 1, px, eval_mask)):
            return None

    hierarchical = bool(ry['hierarchical'] or rx['hierarchical'] or py != px)
    periods = [8, 16] if hierarchical else [int(py)]
    primary_period = 16 if (16 in periods) else 8
    oy = int(ry['offset'])
    ox = int(rx['offset'])

    cell_p = 8 if 8 in periods else 16
    flat_frac, ex_zero, ex_flat, ex_cnt = _flat_cell_metrics(gray, cell_p, ox % cell_p, oy % cell_p, eval_mask)
    is_flat_mosaic = bool(
        (flat_frac >= 0.90 and max(ry['interior'], rx['interior']) <= 0.35)
        or (ex_cnt >= 12 and ex_zero >= 0.70 and ex_flat >= 0.72)
    )

    final_mask = support_mask if is_regional else np.ones((height, width), dtype=bool)
    return {
        'suspected': not is_flat_mosaic,
        'period': int(primary_period),
        'periods': periods,
        'axis_periods': [int(py), int(px)],
        'hierarchical': hierarchical,
        'phase': [ox, oy],
        'offset_x': ox,
        'offset_y': oy,
        'seam_excess': [round(float(ry['seam']), 4), round(float(rx['seam']), 4)],
        'midpoint_excess': [round(float(ry['midpoint']), 4), round(float(rx['midpoint']), 4)],
        'interior_gradient': [round(float(ry['interior']), 4), round(float(rx['interior']), 4)],
        'flat_cell_fraction': round(float(flat_frac), 4),
        'is_flat_privacy_mosaic': is_flat_mosaic,
        'regional': bool(is_regional),
        'mask_fraction': round(float(final_mask.mean()), 4),
        'mask': final_mask,
    }


def _hevc_deblock_axis0(
    arr: np.ndarray,
    luma: np.ndarray,
    period: int,
    offset: int,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Adaptive 4-tap HEVC-style 1D boundary deblocking across horizontal seams (axis=0)."""
    height = arr.shape[0]
    out = arr.copy()
    bs = np.arange(height - 1)
    bs = bs[(bs + 1 + offset) % period == 0]
    bs = bs[(bs >= 1) & (bs + 2 < height)]
    if bs.size == 0:
        return out

    idx_m2 = np.maximum(0, bs - 2)
    idx_p3 = np.minimum(height - 1, bs + 3)

    p0, q0 = out[bs], out[bs + 1]
    p1, q1 = out[bs - 1], out[bs + 2]
    p2, q2 = out[idx_m2], out[idx_p3]

    yp0, yq0 = luma[bs], luma[bs + 1]
    yp1, yq1 = luma[bs - 1], luma[bs + 2]
    yp2, yq2 = luma[idx_m2], luma[idx_p3]

    ystep = np.abs(yq0 - yp0)
    yadjp = np.abs(yp0 - yp1)
    yadjq = np.abs(yq0 - yq1)
    yadj1 = np.maximum(yadjp, yadjq)
    yadj2 = np.maximum(np.abs(yp1 - yp2), np.abs(yq1 - yq2))

    rgb_step = np.abs(q0 - p0).max(axis=-1)
    rgb_adj1 = np.maximum(np.abs(p0 - p1), np.abs(q0 - q1)).max(axis=-1)

    eff_step = np.maximum(ystep, np.float32(0.70) * rgb_step)
    eff_adj1 = np.maximum(yadj1, np.float32(0.70) * rgb_adj1)

    if weight is not None:
        w_edge = np.minimum(weight[bs], weight[bs + 1])
        in_mask = w_edge > 0.0
    else:
        w_edge = None
        in_mask = np.ones_like(ystep, dtype=bool)

    cand = in_mask & (eff_step > np.float32(1.15) * eff_adj1 + np.float32(0.5))
    excess = (eff_step - eff_adj1)[cand]
    if excess.size == 0:
        return out

    s50 = float(np.percentile(excess, 50))
    s75 = float(np.percentile(excess, 75))
    all75 = float(np.percentile(np.maximum(eff_step - eff_adj1, 0.0)[in_mask], 75))
    tc = float(np.clip(max(1.25 * s50, 0.85 * s75), 3.0, 14.0))
    edge_cap = float(min(36.0, max(24.0, 2.8 * all75 + 8.0)))

    active = (
        cand
        & (ystep > np.float32(1.05) * yadj2 + np.float32(0.25))
        & (eff_step < np.float32(edge_cap))
    )
    ydelta = (np.float32(9.0) * (yq0 - yp0) - np.float32(3.0) * (yq1 - yp1)) / np.float32(16.0)
    active &= np.abs(ydelta) < np.float32(8.0 * tc)

    ymean_adj = np.float32(0.5) * (yadjp + yadjq)
    damp = np.clip((ystep - ymean_adj) / np.maximum(np.float32(0.75) * ystep, np.float32(1e-4)), 0.0, 1.0)
    gate = active.astype(np.float32) * damp
    if w_edge is not None:
        gate = gate * w_edge

    raw_delta = (np.float32(9.0) * (q0 - p0) - np.float32(3.0) * (q1 - p1)) / np.float32(16.0)
    delta = np.clip(raw_delta, -tc, tc) * gate[..., None]

    out[bs] = p0 + delta
    out[bs + 1] = q0 - delta

    tc2 = np.float32(0.5 * tc)
    do_p1 = active & (np.abs(yp2 - yp0) < np.float32(1.2 * tc + 2.0)) & (yadjp < np.float32(0.5) * ystep)
    do_q1 = active & (np.abs(yq2 - yq0) < np.float32(1.2 * tc + 2.0)) & (yadjq < np.float32(0.5) * ystep)
    gate_p1 = do_p1.astype(np.float32) * damp * (w_edge if w_edge is not None else np.float32(1.0))
    gate_q1 = do_q1.astype(np.float32) * damp * (w_edge if w_edge is not None else np.float32(1.0))

    dp1 = np.clip(((p2 + p0) * np.float32(0.5) - p1 + delta) * np.float32(0.5), -tc2, tc2) * gate_p1[..., None]
    dq1 = np.clip(((q2 + q0) * np.float32(0.5) - q1 - delta) * np.float32(0.5), -tc2, tc2) * gate_q1[..., None]
    out[bs - 1] = p1 + dp1
    out[bs + 2] = q1 + dq1
    return out


def _normalize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Convert optional regional mask to (mask_bool, weight_float32, is_regional)."""
    if mask is None:
        return None, None, False
    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError('Mask shape must match image spatial dimensions.')
    if arr.dtype == bool:
        mask_bool = arr
        weight = arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.integer):
        mask_bool = arr > 0
        scale = np.float32(255.0) if arr.max() > 1 else np.float32(1.0)
        weight = np.clip(arr.astype(np.float32) / scale, 0.0, 1.0)
    else:
        mask_bool = arr > 0.1
        weight = np.clip(arr.astype(np.float32), 0.0, 1.0)
    if not np.any(mask_bool):
        return mask_bool, weight, True
    if bool(np.all(mask_bool)) and bool(np.all(weight >= 0.999)):
        return None, None, False
    return mask_bool, weight, True


def apply_hevc_deblocking(
    rgb: np.ndarray,
    period: int,
    offset_x: int,
    offset_y: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply deterministic CPU HEVC-style 4-tap adaptive 1D boundary deblocking."""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image array of shape (H, W, 3).')
    period = int(period)
    if period < 4 or min(rgb.shape[:2]) < 8:
        return rgb.copy()
    ox = int(offset_x) % period
    oy = int(offset_y) % period

    mask_bool, weight, is_regional = _normalize_mask(mask, rgb.shape[:2])
    if is_regional and not np.any(mask_bool):
        return rgb.copy()

    work = rgb.astype(np.float32)
    luma_y = _bt601_luma(rgb)
    work = _hevc_deblock_axis0(work, luma_y, period, oy, weight)
    luma_x = work[..., 0] * np.float32(0.299) + work[..., 1] * np.float32(0.587) + work[..., 2] * np.float32(0.114)
    work = _hevc_deblock_axis0(work.transpose(1, 0, 2), luma_x.T, period, ox, weight.T if weight is not None else None).transpose(1, 0, 2)

    cleaned = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    if is_regional:
        cleaned[~mask_bool] = rgb[~mask_bool]
    return cleaned


def _feather_mask(mask_bool: np.ndarray) -> np.ndarray:
    """Inward distance-transform feather inside `mask_bool` with zero weight outside."""
    dist = distance_transform_edt(mask_bool)
    feather = max(1.5, min(6.0, float(dist.max()) * 0.5))
    t = np.clip((dist - 0.5) / feather, 0.0, 1.0)
    alpha = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    alpha[~mask_bool] = 0.0
    return alpha


def _seam_and_detail_metrics(
    rgb: np.ndarray,
    period: int,
    ox: int,
    oy: int,
    mask_bool: np.ndarray | None,
) -> list[tuple[float, float]] | None:
    """Measure (seam_excess, non_seam_interior_gradient) on both axes outside 1-px seam lines."""
    gray = _bt601_luma(rgb)
    out = []
    for axis, offset in ((0, oy), (1, ox)):
        diffs = np.abs(np.diff(gray, axis=axis))
        positions = (np.arange(diffs.shape[axis]) + 1 + offset) % period
        if mask_bool is None:
            profile = diffs.mean(axis=1 - axis)
            b_vals = profile[positions == 0]
            i_vals = profile[positions != 0]
            if b_vals.size == 0 or i_vals.size == 0:
                return None
            boundary = float(b_vals.mean())
            interior = float(i_vals.mean())
        else:
            allowed = mask_bool[:-1, :] & mask_bool[1:, :] if axis == 0 else mask_bool[:, :-1] & mask_bool[:, 1:]
            phase_view = positions[:, None] if axis == 0 else positions[None, :]
            b_mask = allowed & (phase_view == 0)
            i_mask = allowed & (phase_view != 0)
            if int(b_mask.sum()) < 8 or int(i_mask.sum()) < 32:
                return None
            boundary = float(diffs[b_mask].mean())
            interior = float(diffs[i_mask].mean())
        out.append((boundary - interior, interior))
    return out


def _passes_macroblock_gate(
    source: np.ndarray,
    cleaned: np.ndarray,
    period: int,
    ox: int,
    oy: int,
    mask_bool: np.ndarray | None,
) -> tuple[bool, list[tuple[float, float]] | None, list[tuple[float, float]] | None, float]:
    """Verify >= 20% seam-excess drop and >= 90% interior detail retention outside 1-px seams."""
    if np.array_equal(cleaned, source):
        return False, None, None, 0.0
    if mask_bool is not None and not np.array_equal(cleaned[~mask_bool], source[~mask_bool]):
        return False, None, None, 0.0
    before = _seam_and_detail_metrics(source, period, ox, oy, mask_bool)
    after = _seam_and_detail_metrics(cleaned, period, ox, oy, mask_bool)
    if before is None or after is None:
        return False, before, after, 0.0

    active = [(b, a) for b, a in zip(before, after) if b[0] > 0.25]
    if len(active) < 2 and max(b[0] for b in before) <= 0.25:
        return False, before, after, 0.0
    for b_item, a_item in active:
        if abs(a_item[0]) > 0.80 * b_item[0]:
            return False, before, after, 0.0

    retentions = []
    for b_item, a_item in zip(before, after):
        if b_item[1] >= 0.50:
            ratio = a_item[1] / b_item[1]
            if ratio < 0.90:
                return False, before, after, ratio
            retentions.append(ratio)
    sum_before = before[0][1] + before[1][1]
    sum_after = after[0][1] + after[1][1]
    if sum_before >= 0.50 and (sum_after / sum_before) < 0.90:
        return False, before, after, sum_after / sum_before
    min_ret = min(retentions) if retentions else (sum_after / sum_before if sum_before >= 0.50 else 1.0)
    return True, before, after, float(min_ret)


def _padded_box_for_region(
    shape: tuple[int, int],
    mask_bool: np.ndarray,
    min_side: int = MIN_CROP_SIDE,
) -> tuple[int, int, int, int]:
    """Expand a boolean mask bounding box minimally to >= `min_side` within image bounds."""
    height, width = shape[:2]
    yy, xx = np.nonzero(mask_bool)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    if y1 - y0 < min_side:
        pad = min_side - (y1 - y0)
        y0 = max(0, y0 - pad // 2)
        y1 = min(height, y0 + min_side)
        y0 = max(0, y1 - min_side)
    if x1 - x0 < min_side:
        pad = min_side - (x1 - x0)
        x0 = max(0, x0 - pad // 2)
        x1 = min(width, x0 + min_side)
        x0 = max(0, x1 - min_side)
    return y0, y1, x0, x1


def _wang_bovik_dct_check(
    crop: np.ndarray,
    det: dict,
    eval_mask: np.ndarray | None,
) -> bool:
    """Verify Wang-Bovik periodic seam Z-score and non-flat intra-cell texture."""
    gray = _bt601_luma(crop)
    ox, oy = int(det['offset_x']), int(det['offset_y'])
    dy = np.abs(np.diff(gray, axis=0))
    dx = np.abs(np.diff(gray, axis=1))

    z_scores: list[float] = []
    for axis, diffs, off, period in (
        (0, dy, oy, int(det['axis_periods'][0])),
        (1, dx, ox, int(det['axis_periods'][1])),
    ):
        prof = _axis_diff_profile(diffs, axis, eval_mask)
        if prof is None:
            continue
        valid = np.isfinite(prof)
        pos = (np.arange(prof.size) + 1 + off) % period
        b = prof[(pos == 0) & valid]
        i = prof[(pos != 0) & valid]
        if b.size > 0 and i.size >= 4:
            mu_i = float(i.mean())
            sd_i = max(0.35, float(i.std()))
            z_scores.append((float(b.mean()) - mu_i) / sd_i)
    if len(z_scores) < 2 or min(z_scores) < 1.5:
        return False

    cell_p = 8 if 8 in det['periods'] else 16
    flat_frac, ex_zero, ex_flat, ex_cnt = _flat_cell_metrics(
        gray, cell_p, ox % cell_p, oy % cell_p, eval_mask
    )
    if (flat_frac >= 0.85 and max(det['interior_gradient']) <= 0.35) or (
        ex_cnt >= 12 and ex_zero >= 0.70 and ex_flat >= 0.72
    ):
        return False
    return True


def is_dct_macroblocking_patch(
    rgb: np.ndarray,
    region_mask: np.ndarray | None = None,
    origin: tuple[int, int] = (0, 0),
) -> bool:
    """Return True when `rgb` (optionally scoped to `region_mask`) shows genuine DCT macroblocking.

    Distinguishes genuine 8x8/16x16 DCT macroblocking from flat-cell privacy mosaics
    using Wang-Bovik periodic seam Z-scores, intra-cell flatness metrics, and
    deblocking gate verification without rejecting small regional DCT patches.
    """
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        return False
    height, width = rgb.shape[:2]
    if min(height, width) < MIN_CROP_SIDE:
        return False
    ox0, oy0 = (int(origin[0]), int(origin[1])) if origin is not None else (0, 0)

    mask_bool = None
    if region_mask is not None:
        try:
            m_bool, _, m_regional = _normalize_mask(region_mask, (height, width))
        except ValueError:
            return False
        if m_regional and not np.any(m_bool):
            return False
        mask_bool = m_bool

    if mask_bool is not None:
        y0, y1, x0, x1 = _padded_box_for_region((height, width), mask_bool)
        crop = rgb[y0:y1, x0:x1]
        sub_origin = (ox0 + x0, oy0 + y0)
        sub_mask = mask_bool[y0:y1, x0:x1]
    else:
        crop = rgb
        sub_origin = (ox0, oy0)
        sub_mask = None

    if min(crop.shape[:2]) < MIN_CROP_SIDE:
        return False

    det = detect_macroblocking(crop, origin=sub_origin)
    if det is not None and det.get('suspected') and not det.get('is_flat_privacy_mosaic'):
        eval_mask = det['mask'] if det.get('regional') else sub_mask
        if det.get('regional') and sub_mask is not None:
            overlap = det['mask'] & sub_mask
            if int(overlap.sum()) >= 256:
                eval_mask = overlap
        if _wang_bovik_dct_check(crop, det, eval_mask):
            return True
        cleaned, _, _ = prepare_macroblock_deblocking(crop, origin=sub_origin, mask=sub_mask, allow_fbcnn=True)
        if cleaned is not None:
            return True

    if mask_bool is not None:
        det_full = detect_macroblocking(rgb, origin=(ox0, oy0))
        if (
            det_full is not None
            and det_full.get('suspected')
            and not det_full.get('is_flat_privacy_mosaic')
            and float(det_full['mask'][mask_bool].mean()) >= 0.5
        ):
            eval_full = det_full['mask'] if det_full.get('regional') else None
            if _wang_bovik_dct_check(rgb, det_full, eval_full):
                return True
            cleaned_full, _, _ = prepare_macroblock_deblocking(rgb, origin=(ox0, oy0), allow_fbcnn=True)
            return cleaned_full is not None

    return False


def _apply_e2_detail_boost(
    source: np.ndarray,
    cleaned: np.ndarray,
    ox: int,
    oy: int,
    period: int,
    active_mask: np.ndarray | None,
) -> tuple[np.ndarray, bool, list[tuple[float, float]] | None, list[tuple[float, float]] | None, float]:
    """Blend FBCNN strongly near 8x8 block seams (+/-2 px) while retaining interior source detail."""
    height, width = source.shape[:2]
    ox8, oy8 = int(ox) % 8, int(oy) % 8
    yy = (np.arange(height) + oy8) % 8
    xx = (np.arange(width) + ox8) % 8
    dy_seam = np.minimum(np.minimum(yy, (8 - yy) % 8), np.abs(yy - 7))
    dx_seam = np.minimum(np.minimum(xx, (8 - xx) % 8), np.abs(xx - 7))
    seam_w = np.maximum(
        np.where(dy_seam <= 1, 0.90, np.where(dy_seam == 2, 0.50, 0.25))[:, None],
        np.where(dx_seam <= 1, 0.90, np.where(dx_seam == 2, 0.50, 0.25))[None, :],
    ).astype(np.float32)
    blended = np.clip(
        np.rint(
            source.astype(np.float32) * (np.float32(1.0) - seam_w[..., None])
            + cleaned.astype(np.float32) * seam_w[..., None]
        ),
        0,
        255,
    ).astype(np.uint8)
    if active_mask is not None:
        blended[~active_mask] = source[~active_mask]
    ok, before, after, min_ret = _passes_macroblock_gate(
        source, blended, period, ox % period, oy % period, active_mask
    )
    if not ok:
        return cleaned, False, None, None, 0.0
    return blended, True, before, after, min_ret


def prepare_macroblock_deblocking(
    source: np.ndarray,
    origin: tuple[int, int] = (0, 0),
    mask: np.ndarray | None = None,
    allow_fbcnn: bool = True,
    dc: int | None = None,
) -> tuple[np.ndarray | None, dict | None, dict]:
    """Detect and adaptively deblock 8x8/16x16 macroblocking across PNGs, EXIF JPEGs, frames, and regions."""
    if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        return None, None, {}
    height, width = source.shape[:2]
    if min(height, width) < MIN_CROP_SIDE:
        return None, None, {}

    caller_mask = None
    if mask is not None:
        try:
            m_bool, _, m_regional = _normalize_mask(mask, (height, width))
        except ValueError:
            return None, None, {}
        if m_regional and not np.any(m_bool):
            return None, None, {}
        if m_bool is not None:
            caller_mask = m_bool

    det = detect_macroblocking(source, origin=origin)
    # If caller supplied a small localized mask (< 64x64) inside a larger clean frame, also probe its bbox
    if det is None and caller_mask is not None:
        yy, xx = np.nonzero(caller_mask)
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        if min(y1 - y0, x1 - x0) >= MIN_CROP_SIDE:
            ox_b, oy_b = (int(origin[0]) + x0, int(origin[1]) + y0) if origin is not None else (x0, y0)
            sub_det = detect_macroblocking(source[y0:y1, x0:x1], origin=(ox_b, oy_b))
            if sub_det is not None:
                full_mask = np.zeros((height, width), dtype=bool)
                full_mask[y0:y1, x0:x1] = sub_det['mask']
                det = {
                    **sub_det,
                    'offset_x': (sub_det['offset_x'] - x0) % sub_det['period'],
                    'offset_y': (sub_det['offset_y'] - y0) % sub_det['period'],
                    'regional': True,
                    'mask': full_mask,
                }

    if det is None or not det['suspected'] or det['is_flat_privacy_mosaic']:
        return None, None, {}

    if caller_mask is not None and det['regional']:
        active_mask = caller_mask & det['mask']
        if int(active_mask.sum()) < 256:
            return None, None, {}
        is_regional = True
    elif caller_mask is not None:
        active_mask = caller_mask
        is_regional = not bool(active_mask.all())
    elif det['regional']:
        active_mask = det['mask']
        is_regional = True
    else:
        active_mask = None
        is_regional = False

    if active_mask is not None and bool(active_mask.all()):
        active_mask = None
        is_regional = False

    periods = det['periods']
    primary_p = int(det['period'])
    ox, oy = int(det['offset_x']), int(det['offset_y'])
    in_fbcnn_range = (FBCNN_MIN_SIDE <= min(height, width)) and (max(height, width) <= FBCNN_MAX_SIDE)
    feather_weight = _feather_mask(active_mask) if is_regional else None

    cleaned = None
    method = None
    fbcnn_report = None

    if periods == [8]:
        if allow_fbcnn and in_fbcnn_range:
            cand_fbcnn, fbcnn_report = prepare_fbcnn_deblocking(
                source,
                origin=(ox % 8, oy % 8),
                regional_mask=feather_weight if is_regional else None,
                dc=dc,
                require_seams=False,
            )
            if cand_fbcnn is not None:
                ok, before, after, min_ret = _passes_macroblock_gate(
                    source, cand_fbcnn, 8, ox % 8, oy % 8, active_mask,
                )
                if ok:
                    cleaned = cand_fbcnn
                    method = 'fbcnn_rgb'
        if cleaned is None:
            cand_hevc = apply_hevc_deblocking(source, 8, ox % 8, oy % 8, mask=active_mask)
            ok, before, after, min_ret = _passes_macroblock_gate(
                source, cand_hevc, 8, ox % 8, oy % 8, active_mask,
            )
            if ok:
                cleaned = cand_hevc
                method = 'hevc_8x8_fallback' if (allow_fbcnn and in_fbcnn_range) else 'hevc_8x8'

    elif periods == [16]:
        cand_hevc = apply_hevc_deblocking(source, 16, ox % 16, oy % 16, mask=active_mask)
        ok, before, after, min_ret = _passes_macroblock_gate(
            source, cand_hevc, 16, ox % 16, oy % 16, active_mask,
        )
        if ok:
            cleaned = cand_hevc
            method = 'hevc_16x16'

    else:
        # Hierarchical 8x8 + 16x16
        inter = None
        if allow_fbcnn and in_fbcnn_range:
            inter, fbcnn_report = prepare_fbcnn_deblocking(
                source,
                origin=(ox % 8, oy % 8),
                regional_mask=feather_weight if is_regional else None,
                dc=dc,
                require_seams=False,
            )
            if inter is not None:
                cand_combo = apply_hevc_deblocking(inter, 16, ox % 16, oy % 16, mask=active_mask)
                ok, before, after, min_ret = _passes_macroblock_gate(
                    source, cand_combo, 16, ox % 16, oy % 16, active_mask,
                )
                if ok:
                    cleaned = cand_combo
                    method = 'fbcnn_plus_hevc_16x16'
        if cleaned is None:
            step8 = apply_hevc_deblocking(source, 8, ox % 8, oy % 8, mask=active_mask)
            cand_hier = apply_hevc_deblocking(step8, 16, ox % 16, oy % 16, mask=active_mask)
            ok, before, after, min_ret = _passes_macroblock_gate(
                source, cand_hier, 16, ox % 16, oy % 16, active_mask,
            )
            if ok:
                cleaned = cand_hier
                method = 'hevc_hierarchical_8_16'

    if cleaned is None:
        return None, None, {}

    detail_boost = False
    if method in ('fbcnn_rgb', 'fbcnn_plus_hevc_16x16'):
        boosted, boost_ok, boost_before, boost_after, boost_ret = _apply_e2_detail_boost(
            source, cleaned, ox, oy, primary_p, active_mask
        )
        if boost_ok:
            cleaned = boosted
            before = boost_before
            after = boost_after
            min_ret = boost_ret
            detail_boost = True

    drops = [
        (b_item[0] - abs(a_item[0])) / b_item[0] if b_item[0] > 0 else 0.0
        for b_item, a_item in zip(before, after)
    ]
    changed = int(np.any(cleaned != source, axis=2).sum())
    if changed == 0:
        return None, None, {}

    evidence = {'macroblock-cleaned.png': png(cleaned)}
    if is_regional:
        mask_u8 = (active_mask.astype(np.uint8) * 255)
        evidence['macroblock-mask.png'] = png(mask_u8)

    report = {
        'policy': 'macroblock-adaptive-deblocking-v1',
        'method': str(method),
        'detail_boost': bool(detail_boost),
        'native_size': [int(width), int(height)],
        'period': int(primary_p),
        'periods': [int(p) for p in periods],
        'hierarchical': bool(det['hierarchical']),
        'grid_origin': [int(ox), int(oy)],
        'regional': bool(is_regional),
        'masked_native_pixels': int(active_mask.sum()) if is_regional else int(height * width),
        'changed_native_pixels': changed,
        'flat_cell_fraction': float(det['flat_cell_fraction']),
        'seam_excess_before': [round(float(item[0]), 4) for item in before],
        'seam_excess_after': [round(float(item[0]), 4) for item in after],
        'seam_drop_fraction': [round(float(d), 4) for d in drops],
        'interior_detail_before': [round(float(item[1]), 4) for item in before],
        'interior_detail_after': [round(float(item[1]), 4) for item in after],
        'interior_detail_retention': round(float(min_ret), 4),
    }
    if dc is not None:
        report['jpeg_luma_dc_quantization'] = int(dc)
    if fbcnn_report is not None and method in ('fbcnn_rgb', 'fbcnn_plus_hevc_16x16'):
        report['preparation_model'] = str(fbcnn_report['preparation_model'])
        report['preparation_model_sha256'] = str(fbcnn_report['preparation_model_sha256'])
        report['preparation_forward_returns'] = int(fbcnn_report['preparation_forward_returns'])
    return cleaned, report, evidence

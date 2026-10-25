"""Prepare the deterministic 56-identity x 6-profile CCTV TRAIN dataset + 48-case V2 monitor bank.

Pure NumPy/Pillow/SciPy data preparation; imports zero neural frameworks.
Consumes only the 64 root-admitted clean TRAIN FFHQ sources (`root_clean_target_review_v1.json`),
preserving 100% disjointness between:
- 56 optimization identities (336 paired training cases across 6 realistic CCTV profiles)
- 8 held-out monitor identities (48 V2 monitor cases across the same 6 CCTV profiles,
  alongside the unchanged 40 V1 monitors and 56 inherited old_quality cases).
"""
from datetime import datetime, timezone
import hashlib
import importlib.abc
import io
import json
import math
from pathlib import Path
import sys


class NoModelFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'tensorflow', 'jax', 'basicsr', 'facelib'}:
            raise ImportError('V2 data preparation forbids neural framework imports: ' + fullname)


sys.meta_path.insert(0, NoModelFramework())

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
TESTING_ROOT = ROOT.parent / 'Testing'
DATA_BASE = ROOT if (ROOT / 'outputs/codeformer_original_spatial_correction_v1_preparation').exists() else TESTING_ROOT
CLEAN_REVIEW = DATA_BASE / 'outputs/codeformer_original_spatial_correction_v1_preparation/data_audit_v1/root_clean_target_review_v1.json'
CLEAN_REVIEW_SHA = '2e06c37a59269a19582b4d3066af456ae6b330dcf8fd3f73b54f07366e9b804f'
OUT_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_preparation'

MONITOR_REFERENCES = (
    'tr_ffhq_08698', 'tr_ffhq_02147', 'tr_ffhq_05179', 'tr_ffhq_08949',
    'tr_ffhq_11039', 'tr_ffhq_11154', 'tr_ffhq_08064', 'tr_ffhq_08094',
)
ROLES_V2 = (
    'clear',
    'global_defocus_blur',
    'directional_motion_blur',
    'local_soft_surroundings',
    'compound_cctv_codec',
    'lowlight_sensor_noise',
)
ORDER_SEED = '20261025-conditioned-sft-v2-56id-cctv-order'


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode_png(arr_u8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, format='PNG')
    return buf.getvalue()


def make_motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Construct a normalized 2D sub-pixel anti-aliased linear motion blur kernel."""
    need(length >= 3 and length % 2 == 1, 'Odd motion kernel length >= 3 required')
    c = (length - 1) / 2.0
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    kernel = np.zeros((length, length), dtype=np.float64)
    samples = length * 8
    for i in range(samples):
        t = ((i + 0.5) / samples - 0.5) * (length - 1)
        x = c + t * dx
        y = c + t * dy
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for oy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            for ox, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                if 0 <= oy < length and 0 <= ox < length:
                    kernel[oy, ox] += wx * wy
    kernel /= kernel.sum()
    return kernel


def apply_motion_blur(rgb_u8: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    k2d = make_motion_kernel(length, angle_deg)
    f64 = rgb_u8.astype(np.float64)
    chans = [convolve(f64[:, :, ch], k2d, mode='reflect') for ch in range(3)]
    out = np.stack(chans, axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ('inputs', 'targets'):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    clean_raw = CLEAN_REVIEW.read_bytes()
    need(sha256_bytes(clean_raw) == CLEAN_REVIEW_SHA, 'Clean review manifest SHA256 mismatch')
    clean = json.loads(clean_raw)
    need(clean['root_personal_input_only_review_complete'] and len(clean['admitted_reference_ids']) == 64,
         'Exact 64 root-admitted clean TRAIN sources required')

    records = {row['reference_id']: row for row in clean['admitted_references']}
    adm_set = set(clean['admitted_reference_ids'])
    mon_set = set(MONITOR_REFERENCES)
    need(mon_set <= adm_set and len(mon_set) == 8, 'Exact 8 monitor references must be subset of 64 admitted')

    opt_refs = sorted(
        adm_set - mon_set,
        key=lambda r: sha256_bytes((ORDER_SEED + ':' + r).encode('ascii')),
    )
    need(len(opt_refs) == 56 and not (set(opt_refs) & mon_set), 'Exact 56 disjoint optimization identities required')
    all_refs = list(opt_refs) + list(MONITOR_REFERENCES)

    yy, xx = np.mgrid[:256, :256]
    radius = np.sqrt(((xx - 127.5) / 72.0) ** 2 + ((yy - 127.5) / 88.0) ** 2)
    frac = np.clip((1.0 - radius) / 0.20, 0.0, 1.0)
    ellipse_alpha = frac * frac * (3.0 - 2.0 * frac)

    defocus_sigmas = (1.8, 2.0, 2.3, 2.6)
    motion_specs = ((7, 15.0), (9, 45.0), (11, 0.0), (9, 135.0), (11, 75.0), (13, 25.0))
    codec_specs = ((1.1, 64, 68), (1.3, 56, 55), (1.4, 48, 48), (1.2, 64, 52))
    lowlight_specs = ((0.35, 0.006), (0.30, 0.007), (0.40, 0.005), (0.33, 0.0065))

    sources_meta = []
    opt_cases = []
    mon_v2_cases = []

    for idx, ref in enumerate(all_refs):
        is_opt = ref in set(opt_refs)
        cohort = 'optimize' if is_opt else 'monitor_v2'
        row = records[ref]
        need(row['historical_role'] == 'TRAIN' and row['official_category'] == 'training',
             'TRAIN-only photographic source required')
        src1024_path = DATA_BASE / row['physical_source']['path']
        targ256_src = DATA_BASE / 'outputs/cctv_dgp_hq_cohort_v2/targets256' / f'{ref}.png'
        targ_bytes = targ256_src.read_bytes()
        with Image.open(io.BytesIO(targ_bytes)) as im:
            target = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
        need(target.shape == (256, 256, 3), 'Exact 256x256 RGB target required')

        targ_dest = OUT_DIR / 'targets' / f'{ref}.png'
        targ_dest.write_bytes(targ_bytes)
        targ_pin = {
            'path': targ_dest.relative_to(ROOT).as_posix(),
            'bytes': len(targ_bytes),
            'sha256': sha256_bytes(targ_bytes),
        }
        sources_meta.append({
            'reference_id': ref,
            'cohort': cohort,
            'source1024_sha256': row['physical_source']['sha256'],
            'target': targ_pin,
            'target_RGB8_sha256': sha256_bytes(target.tobytes()),
        })

        # Deterministic per-identity realistic CCTV parameters
        h_idx = int(sha256_bytes(('v2-param:' + ref).encode('ascii'))[:8], 16)
        sigma_def = defocus_sigmas[h_idx % len(defocus_sigmas)] if is_opt else 2.2
        m_len, m_ang = motion_specs[h_idx % len(motion_specs)] if is_opt else (9, 30.0)
        c_sig, c_lr, c_q = codec_specs[h_idx % len(codec_specs)] if is_opt else (1.3, 52, 50)
        ll_exp, ll_std = lowlight_specs[h_idx % len(lowlight_specs)] if is_opt else (0.35, 0.006)

        blur_fn = lambda s: np.clip(
            np.rint(gaussian_filter(target.astype(np.float64), sigma=(s, s, 0), mode='reflect', truncate=3.0)),
            0, 255
        ).astype(np.uint8)

        obs_defocus = blur_fn(sigma_def)
        obs_motion = apply_motion_blur(target, m_len, m_ang)
        blur2p0 = blur_fn(2.0)
        obs_local = np.clip(
            np.rint(target.astype(np.float64) * (1.0 - ellipse_alpha[..., None]) + blur2p0.astype(np.float64) * ellipse_alpha[..., None]),
            0, 255
        ).astype(np.uint8)
        obs_local[ellipse_alpha == 0] = target[ellipse_alpha == 0]

        comp_pil = Image.fromarray(blur_fn(c_sig)).resize(
            (c_lr, c_lr), Image.Resampling.BILINEAR
        ).resize((256, 256), Image.Resampling.BILINEAR)
        jbuf = io.BytesIO()
        comp_pil.save(jbuf, format='JPEG', quality=c_q, subsampling=0, optimize=False, progressive=False)
        with Image.open(io.BytesIO(jbuf.getvalue())) as jim:
            obs_compound = np.asarray(jim.convert('RGB'), dtype=np.uint8).copy()

        vals = target.astype(np.float64) / 255.0
        lin = np.where(vals <= 0.04045, vals / 12.92, ((vals + 0.055) / 1.055) ** 2.4) * ll_exp
        enc = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * (lin ** (1.0 / 2.4)) - 0.055)
        n_seed = int(sha256_bytes(('20261025-v2-lowlight:' + ref).encode('ascii'))[:16], 16)
        noise = np.random.default_rng(n_seed).normal(0.0, ll_std, target.shape)
        obs_lowlight = np.floor(np.clip(enc + noise, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        role_arrays = (
            ('clear', target, targ_bytes, {'kind': 'clean'}),
            ('global_defocus_blur', obs_defocus, None, {'sigma': sigma_def}),
            ('directional_motion_blur', obs_motion, None, {'length': m_len, 'angle_deg': m_ang}),
            ('local_soft_surroundings', obs_local, None, {'sigma': 2.0, 'region': 'ellipse_smoothstep'}),
            ('compound_cctv_codec', obs_compound, None, {'sigma': c_sig, 'lr_px': c_lr, 'jpeg_q': c_q}),
            ('lowlight_sensor_noise', obs_lowlight, None, {'exposure': ll_exp, 'noise_std': ll_std, 'seed': n_seed}),
        )

        for role_name, arr_u8, preset_bytes, params in role_arrays:
            cid = f'{ref}__{role_name}'
            in_dest = OUT_DIR / 'inputs' / f'{cid}.png'
            raw_png = preset_bytes if preset_bytes is not None else encode_png(arr_u8)
            in_dest.write_bytes(raw_png)
            rec = {
                'id': cid,
                'reference_id': ref,
                'role': role_name,
                'cohort': cohort,
                'split': 'TRAIN',
                'optimization_permitted': is_opt,
                'input': {
                    'path': in_dest.relative_to(ROOT).as_posix(),
                    'bytes': len(raw_png),
                    'sha256': sha256_bytes(raw_png),
                },
                'target': targ_pin,
                'input_RGB8_sha256': sha256_bytes(arr_u8.tobytes()),
                'target_RGB8_sha256': sha256_bytes(target.tobytes()),
                'operator_params': params,
            }
            if is_opt:
                opt_cases.append(rec)
            else:
                mon_v2_cases.append(rec)

    need(len(opt_cases) == 56 * 6 == 336, 'Exact 336 optimization cases required')
    need(len(mon_v2_cases) == 8 * 6 == 48, 'Exact 48 V2 monitor cases required')

    manifest = {
        'complete': True,
        'status': 'CONDITIONED_SFT_V2_56ID_6ROLE_TRAIN_DATASET_READY',
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'clean_review_sha256': CLEAN_REVIEW_SHA,
        'order_seed': ORDER_SEED,
        'roles_v2': list(ROLES_V2),
        'optimization_reference_order': opt_refs,
        'monitor_reference_order': list(MONITOR_REFERENCES),
        'optimization_identities': 56,
        'monitor_identities': 8,
        'optimization_case_count': len(opt_cases),
        'monitor_v2_case_count': len(mon_v2_cases),
        'zero_monitor_overlap_verified': len(set(opt_refs) & mon_set) == 0,
        'sources': sources_meta,
        'optimization_cases': opt_cases,
        'monitor_v2_cases': mon_v2_cases,
    }
    manifest_path = OUT_DIR / 'v2_data_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({
        'complete': True,
        'optimization_identities': len(opt_refs),
        'optimization_cases': len(opt_cases),
        'monitor_v2_cases': len(mon_v2_cases),
        'manifest_sha256': sha256_bytes(manifest_path.read_bytes()),
    }))


if __name__ == '__main__':
    main()

"""Frozen inference-only V15 plan and arithmetic; no neural imports here."""
import hashlib
import json
from pathlib import Path, PurePosixPath

PLAN = 'generalization_protocol_v15.json'
FORMAT = 'dgp-direct-codes-fresh-image-generalization-v15'
PARENT_PIN = '06f056ea4b1b04e6d8c631e5e78df129345c78279230c7946f6b5deb2e226846'
CAPACITY_PIN = 'd26fb96ee86efeadbc9505f3c8ecf4e4ab4f6559cea7b33eac154204ee12337a'
MIXED_PIN = '6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
SOURCES = ['dataset/asian_faces', 'dataset/thumbnails128x128']
ARMS = ['retained_dgp_v2', 'starting_prior_none', 'trained_conditioner_none']
DESIGN = {'seed': 20261004, 'batch_size': 1, 'parity_cases': 50,
          'validation_references': 104, 'validation_cases': 520,
          'statistics': 'none', 'fidelity': 0.0, 'optimizer_updates': 0,
          'backward_calls': 0, 'runner_cap_seconds': 1200,
          'supervisor_cap_seconds': 1800, 'peak_vram_cap_bytes': 20 * 1024**3,
          'timing_at_case': 20, 'arms': ARMS, 'selection': 'None; diagnostic only.'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def safe(root, name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and
            '\\' not in name and ':' not in name, 'Unsafe asset path')
    target = (Path(root) / name).resolve()
    require(target.is_relative_to(Path(root).resolve()), 'Asset escapes root')
    return target


def validate_cohort(p):
    refs, cases = p['references'], p['cases']
    require(len(refs) == 104 and len({r['id'] for r in refs}) == 104 and
            all(r['role'] == 'validation' for r in refs), 'Require104 validation references')
    require({s: sum(r['source'] == s for r in refs) for s in SOURCES} ==
            {SOURCES[0]: 51, SOURCES[1]: 53}, 'Source cohort changed')
    require(len(cases) == 520 and len({c['id'] for c in cases}) == 520, 'Require520 distinct cases')
    lookup = {r['id']: r for r in refs}
    for c in cases:
        require(c['reference_id'] in lookup and c['source'] == lookup[c['reference_id']]['source'],
                'Case source/reference differs')
    for r in refs:
        require(sorted(c['profile'] for c in cases if c['reference_id'] == r['id']) == sorted(PROFILES),
                'Missing/repeated camera profile')
    training = p['training_identity_inventory']
    for key in ['id', 'source_sha256', 'target_rgb_sha256']:
        require(not ({r[key] for r in refs} & {r[key] for r in training}),
                'Own train/validation exact overlap: ' + key)
    require(not p['native_used'] and not p['native_reserved_used'] and
            not p['production_promoted'] and p['training'] is False, 'Wrong V15 scope')
    expected = [r['id'] for s in SOURCES for r in sorted(
        (r for r in refs if r['source'] == s), key=lambda r: r['id'])[:5]]
    require(p['preview_reference_ids'] == expected, 'Preview selection changed')


def verify(root, parent, expected_sha, *, include_weights=True):
    root, parent = Path(root), Path(parent)
    require(sha(root / PLAN) == expected_sha == (root / 'protocol.sha256').read_text().strip(),
            'V15 protocol differs')
    p = read(root / PLAN)
    require(p['format'] == FORMAT and p['design'] == DESIGN and
            p['parent_protocol_sha256'] == PARENT_PIN and
            p['capacity_results_sha256'] == CAPACITY_PIN and
            p['mixed_protocol_sha256'] == MIXED_PIN, 'Frozen V15 design/lineage differs')
    validate_cohort(p)
    for name, pin in p['assets_sha256'].items():
        require(sha(safe(root, name)) == pin, 'V15 source/asset changed: ' + name)
    require(sha(parent / 'face_code_fit_protocol_v12.json') == PARENT_PIN, 'Parent differs')
    pp = read(parent / 'face_code_fit_protocol_v12.json')
    require(pp['assets_sha256'] == p['parent_assets_sha256'], 'Parent manifest changed')
    for name, pin in p['parent_assets_sha256'].items():
        if include_weights or name not in pp['weights'].values():
            require(sha(safe(parent, name)) == pin, 'Parent source/asset changed: ' + name)
    return p


def rgb(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        require(im.mode == 'RGB' and im.size == (256, 256), 'Require exact RGB256')
        return np.asarray(im).copy()


def png(raw, camera, support):
    import numpy as np
    require(raw.dtype == np.float32 and raw.shape == (256, 256, 3) and
            np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1,
            'Invalid raw float restoration')
    out = np.floor(raw * 255).astype(np.uint8)
    out[~support] = camera[~support]
    return out


def metrics(actual, target, mask):
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity
    error = actual.astype(np.float32) / 255 - target.astype(np.float32) / 255
    mse = float(np.square(error[mask]).astype(np.float64).mean())
    _, smap = structural_similarity(target.astype(np.float32) / 255,
        actual.astype(np.float32) / 255, data_range=1, channel_axis=-1, win_size=7, full=True)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    require(mask.any() and inner.any(), 'Missing evaluation support')
    return {'MSE': mse, 'PSNR': float(-10 * np.log10(mse)) if mse else None,
            'perfect_match': mse == 0, 'SSIM': float(smap[inner].astype(np.float64).mean()),
            'MAE': float(np.abs(error[mask]).astype(np.float64).mean())}


def aggregate(rows):
    import numpy as np
    groups = {'clear': [r for r in rows if r['profile'] == 'clear'],
              'degraded': [r for r in rows if r['profile'] != 'clear']}
    for s in SOURCES:
        for profile in PROFILES:
            groups[s + '/' + profile] = [r for r in rows if r['source'] == s and r['profile'] == profile]
        groups[s + '/degraded'] = [r for r in rows if r['source'] == s and r['profile'] != 'clear']
    result = {}
    for name, items in groups.items():
        require(bool(items), 'Empty frozen group')
        mse = float(np.mean([r['MSE'] for r in items]))
        result[name] = {'cases': len(items), 'MSE': mse,
            'PSNR': float(-10 * np.log10(mse)) if mse else None,
            **{k: float(np.mean([r[k] for r in items])) for k in ['SSIM', 'MAE', 'ArcFace_observed_fixed']}}
    return result


def guard_report(summaries):
    candidate = summaries['trained_conditioner_none']; reports = {}
    for arm in ARMS[:2]:
        ref = summaries[arm]; failures = []
        for name, group in ref.items():
            c = candidate[name]
            if c['MSE'] > group['MSE'] + 1e-12: failures.append(name + ':MSE')
            for k in ['SSIM', 'ArcFace_observed_fixed']:
                if c[k] < group[k] - 1e-6: failures.append(name + ':' + k)
        gain = candidate['degraded']['PSNR'] - ref['degraded']['PSNR']
        reports[arm] = {'degraded_PSNR_gain_dB': gain, 'failed_groups_metrics': failures,
                       'strict_diagnostic_guard_passed': not failures and gain >= .1}
    return reports

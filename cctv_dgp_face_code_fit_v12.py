"""Frozen, training-only conditioning fit; verification has no neural forwards."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import random

FORMAT = 'cctv-dgp-face-code-fitting-v12'
SEED = 20261004
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
SOURCES = ['dataset/asian_faces', 'dataset/thumbnails128x128']
DESIGN = {'seed': SEED, 'epochs': 12, 'updates': 300, 'exposures': 600, 'batch_size': 2,
          'optimizer': 'Adam', 'learning_rate': 0.0001, 'betas': [.9, .999],
          'lambda_code_ce': .5, 'lambda_feature_mse': 1.0, 'gradient_clip_norm': 1.0,
          'snapshots': [0, 100, 300], 'fidelities': [0.0, 1.0],
          'trainer_cap_seconds': 600, 'supervisor_cap_seconds': 900,
          'peak_vram_cap_bytes': 20 * 1024**3, 'timing_update': 20,
          'use_amp': False, 'trainable_parameters': 455072,
          'checkpoint_selection': 'None; training-only capacity diagnostic. Do not create best.pth.'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')


def safe(root, name):
    item = PurePosixPath(name)
    require(bool(name) and not item.is_absolute() and '..' not in item.parts and
            '\\' not in name and ':' not in name, 'Unsafe asset path')
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Asset escapes root')
    return path


def validate_cohort(p):
    refs, cases = p['references'], p['cases']
    require(len(refs) == 10 and len({r['id'] for r in refs}) == 10 and
            all(r['role'] == 'train' for r in refs), 'Require ten distinct training references')
    require({r['source'] for r in refs} == set(SOURCES) and
            all(sum(r['source'] == s for r in refs) == 5 for s in SOURCES), 'Require five references/source')
    require(len(cases) == 50 and len({c['id'] for c in cases}) == 50, 'Require fifty distinct fitting cases')
    lookup = {r['id']: r for r in refs}
    for case in cases:
        require(case['reference_id'] in lookup and case['source'] == lookup[case['reference_id']]['source'], 'Case reference/source changed')
    for ref in refs:
        require(sorted(c['profile'] for c in cases if c['reference_id'] == ref['id']) == sorted(PROFILES), 'Missing or repeated profile')
    require(p['native_used'] is False and p['validation_used'] is False and
            p['native_reserved_used'] is False and p['production_promotion'] is False, 'Wrong experiment scope')


def schedule(p):
    validate_cohort(p)
    result = []
    for epoch in range(1, 13):
        rng = random.Random(SEED + epoch)
        per_source = []
        for source in SOURCES:
            rows = sorted(c['id'] for c in p['cases'] if c['source'] == source)
            rng.shuffle(rows); per_source.append(rows)
        result.extend({'epoch': epoch, 'case_ids': list(pair)} for pair in zip(*per_source))
    return result


def verify(root, expected_sha, *, include_weights=True):
    root = Path(root)
    require(sha(root / 'face_code_fit_protocol_v12.json') == expected_sha ==
            (root / 'face_code_fit_protocol_v12.sha256').read_text().strip(), 'Protocol fingerprint differs')
    p = read(root / 'face_code_fit_protocol_v12.json')
    require(p['format'] == FORMAT and p['design'] == DESIGN, 'Frozen design differs')
    validate_cohort(p)
    require(read(root / 'schedule_v12.json')['steps'] == schedule(p), 'Balanced schedule differs')
    for name, pin in p['assets_sha256'].items():
        if not include_weights and name in p['weights'].values():
            continue
        require(sha(safe(root, name)) == pin, 'Changed asset: ' + name)
    return p


def require_vm(root):
    from cctv_dgp_targets_v6 import require_vm as vm_only
    vm_only(root)

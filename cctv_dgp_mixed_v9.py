"""Frozen mixed-replay protocol helpers; verification does not construct models."""
import collections
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

MANIFEST_SHA = 'ec25bf9dc37b32b07fb426cf7875477e70181bf55398f6fdfe420bd4ca550f5d'
DESIGN_SHA = 'a7b365f97810b11aa35f2707a1ddeff8c609805394e82dac9198d16a9cd5e30b'
SCHEDULE_SHA = 'ce125d1a19d4880437758ad16b0ed9c6edfe7c63397b0ad37ab3e78c66e5a5eb'
DERIVATION_SHA = 'e4a1987964d83c955b9b27275568c77fa7730cf4ba119ba090de42b910981656'
START_STATE = 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe_path(root, name):
    assert isinstance(name, str) and name and '\\' not in name and ':' not in name
    path = PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts
    dest = Path(root) / name
    assert dest.resolve().is_relative_to(Path(root).resolve())
    return dest


def group(case):
    return case['source'] + '/' + case['profile']


def fixed_loss_weights(means):
    assert len(means) == 10 and all(math.isfinite(v) and v >= 0 for v in means.values())
    overall = math.fsum(means.values()) / 10
    assert overall > 0, 'Nonzero baseline training error required'
    raw = {k: min(4.0, max(0.25, overall / max(v, 1e-8))) for k, v in means.items()}
    scale = math.fsum(raw.values()) / 10
    weights = {k: v / scale for k, v in raw.items()}
    assert math.isclose(math.fsum(weights.values()) / 10, 1, rel_tol=1e-12)
    return weights


def verify(root, expected_protocol_sha=None, weights=True):
    root = Path(root)
    fingerprint = sha(root / 'mixed_protocol_v9.json')
    assert (root / 'mixed_protocol_v9.sha256').read_text(encoding='ascii').strip() == fingerprint
    if expected_protocol_sha is not None:
        assert fingerprint == expected_protocol_sha, 'Executable protocol differs'
    protocol = read(root / 'mixed_protocol_v9.json')
    assert protocol['format'] == 'cctv-dgp-mixed-replay-v9'
    for name, pin in {
        'cohort_assets_manifest_v9.json': MANIFEST_SHA,
        'lineage/mixed_design_v9.json': DESIGN_SHA,
        'training_schedule_v9.json': SCHEDULE_SHA,
        'lineage/mixed_assets_local_audit.json': DERIVATION_SHA,
    }.items():
        assert sha(root / name) == pin and protocol['assets_sha256'][name] == pin
    manifest = read(root / 'cohort_assets_manifest_v9.json')
    design = read(root / 'lineage/mixed_design_v9.json')
    schedule = read(root / 'training_schedule_v9.json')
    assert protocol['design'] == design
    for key in ('references', 'training_cases', 'validation_cases', 'weights', 'cache_assets',
                'validation_preview_case_ids', 'training_preview_case_ids', 'canonical_target_rgb_sha256'):
        assert protocol[key] == manifest[key], key
    assert all(protocol['assets_sha256'][n] == pin for n, pin in manifest['assets_sha256'].items())
    assert protocol['starting_state_hash'] == START_STATE == design['starting_state_hash']
    assert protocol['assets_sha256'][protocol['weights']['start']] == design['starting_checkpoint_sha256']
    assert design['selection'] == 'unchanged_strict_source_profile_png_guard_plus_0.1dB'
    assert (design['epochs'], design['batch_size'], design['steps_per_epoch'], design['total_optimizer_updates']) == (20, 10, 391, 7820)
    assert design['evaluation_epochs'] == [2, 5, 10, 20]
    assert (design['trainer_cap_seconds'], design['supervisor_cap_seconds']) == (2400, 3600)
    assert design['optimizer'] == {'type': 'fresh Adam', 'backbone_lr': 2e-5, 'head_lr': 1e-4,
        'weight_decay': 1e-5, 'gradient_clip_norm': 1, 'amp': False, 'ema': False,
        'normalization_running_statistics_frozen': True}
    assert protocol['training_location'] == 'forensic-dgp-thesis Linux NVIDIA L4 VM'
    assert not any(protocol[k] for k in ('native_used', 'native_reserved_used', 'production_promotion_permitted'))
    assert not design['identity_training_loss']
    cases = {c['id']: c for c in protocol['training_cases']}
    refs = {r['id']: r for r in protocol['references']}
    assert len(cases) == 3905 and len(refs) == 885
    assert len(protocol['validation_cases']) == 520
    assert collections.Counter(r['role'] for r in refs.values()) == {'train': 781, 'validation': 104}
    groups = {group(c) for c in cases.values()}
    assert len(groups) == 10
    assert schedule['design_sha256'] == DESIGN_SHA and schedule['manifest_sha256'] == MANIFEST_SHA
    assert set(schedule['epochs']) == {str(n) for n in range(1, 21)}
    for batches in schedule['epochs'].values():
        assert len(batches) == 391
        counts = collections.Counter(cid for batch in batches for cid in batch)
        assert set(counts) == set(cases) and sum(counts.values()) == 3910
        for batch in batches:
            assert len(batch) == len(set(batch)) == 10 and {group(cases[c]) for c in batch} == groups
        for cid, count in counts.items():
            assert count in ({1, 2} if cases[cid]['source'] == 'dataset/asian_faces' else {1})
    for name, pin in protocol['assets_sha256'].items():
        if weights or name not in protocol['cache_assets']:
            assert sha(safe_path(root, name)) == pin, 'Changed asset: ' + name
    return protocol

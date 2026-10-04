"""Audit exact camera inputs/roles and freeze bounded balanced batches; no models."""
import ast
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
MANIFEST_SHA = 'ec25bf9dc37b32b07fb426cf7875477e70181bf55398f6fdfe420bd4ca550f5d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    out = ROOT / 'outputs/cctv_dgp_mixed_assets_v9_local_audit.json'
    assert not out.exists() and not (BASE / 'training_schedule_v9.json').exists(), 'Preserve previous derivation/audit'
    assert sha(BASE / 'cohort_assets_manifest_v9.json') == MANIFEST_SHA
    manifest = read(BASE / 'cohort_assets_manifest_v9.json')
    design = read(BASE / 'lineage/mixed_design_v9.json')
    mixed = read(BASE / 'lineage/asian_mixed_cohort.json')
    v6 = read(BASE / 'lineage/targets_protocol_v6.json')
    parent = read(BASE / 'lineage/parent_protocol.json')
    approved = {r['reference_id']: r for r in read(BASE / 'lineage/hq_eligible_references.json')['references']}
    assert sha(BASE / 'lineage/mixed_design_v9.json') == manifest['design_sha256']
    assert sha(BASE / 'lineage/asian_mixed_cohort.json') == manifest['mixed_cohort_sha256']
    assert all(manifest[k] == 0 for k in ('local_model_forwards', 'local_backward_calls', 'local_optimizer_updates'))
    assert not any(manifest[k] for k in ('native_used', 'native_reserved_used', 'training_ready', 'execution_prepared', 'training_started'))
    for name, pin in manifest['assets_sha256'].items():
        path = BASE / name
        assert path.resolve().is_relative_to(BASE.resolve())
        assert sha(path) == pin, name
        if name.endswith('.py'):
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    refs = {r['id']: r for r in manifest['references']}
    current = {r['id']: r for r in v6['references']}
    old = {r['id']: r for r in parent['references']}
    assert len(refs) == len(manifest['references']) == 885
    train = [r for r in manifest['references'] if r['role'] == 'train']
    val = [r for r in manifest['references'] if r['role'] == 'validation']
    assert [r['id'] for r in train] == mixed['training_reference_ids']
    assert [r['id'] for r in val] == mixed['validation_reference_ids']
    assert manifest['validation_cases'] == v6['validation_cases']
    assert manifest['validation_preview_case_ids'] == v6['preview_case_ids']
    assert len(train) == 781 and len(val) == 104
    assert not set(r['id'] for r in train) & set(r['id'] for r in val)
    assert not set(r['source_sha256'] for r in train) & set(r['source_sha256'] for r in val)
    for ref in refs.values():
        assert ref == (current[ref['id']] if ref['id'] in current else old[ref['id']])
        with Image.open(BASE / ref['target']) as image:
            pixel_sha = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            assert image.size == (256, 256) and image.mode == 'RGB'
        assert pixel_sha == manifest['canonical_target_rgb_sha256'][ref['id']]
        if ref['source'] == 'dataset/thumbnails128x128':
            assert sha(BASE / ref['target']) == approved[ref['id']]['target_sha256']
        else:
            assert pixel_sha == ref['target_rgb_sha256']
    for case in manifest['validation_cases']:
        assert sha(BASE / case['input']) == v6['assets_sha256'][case['input']]
        assert refs[case['reference_id']]['role'] == 'validation'
    spec = importlib.util.spec_from_file_location('audit_camera_v9', BASE / 'scripts/cctv_camera_stress.py')
    camera = importlib.util.module_from_spec(spec); spec.loader.exec_module(camera)
    assert camera.PROFILES == manifest['evaluation_profiles'] == v6['evaluation_profiles']
    cases = {c['id']: c for c in manifest['training_cases']}
    assert len(cases) == len(manifest['training_cases']) == 3905
    for i, case in enumerate(cases.values()):
        assert time.monotonic() - started < 600, 'Offline verification exceeds600-second cap'
        ref = refs[case['reference_id']]
        assert ref['role'] == 'train' and case['source'] == ref['source']
        assert case['camera'] == next(p for p in camera.PROFILES if p['id'] == case['profile'])
        noise_seed = int.from_bytes(hashlib.sha256(('107:' + case['id']).encode()).digest()[:4], 'big')
        with Image.open(BASE / ref['target']) as image:
            target = np.asarray(image)
        with Image.open(BASE / case['input']) as image:
            assert image.mode == 'RGB' and image.size == (256, 256)
            actual = np.asarray(image)
        expected, details = camera.degrade(target, ref['bounds'], case['camera'], noise_seed)
        assert details == case['proxy_details']
        np.testing.assert_array_equal(actual, expected)
        observed = np.asarray(Image.open(BASE / ref['observed'])) > 0
        np.testing.assert_array_equal(actual[~observed], target[~observed])
        if case['profile'] == 'clear':
            np.testing.assert_array_equal(actual, target)
        if (i + 1) % 1000 == 0:
            print(f'exact training proxies checked {i + 1}/3905 elapsed={time.monotonic()-started:.0f}s', flush=True)
    groups = collections.defaultdict(list)
    for case in cases.values():
        groups[case['source'] + '/' + case['profile']].append(case['id'])
    groups = {key: sorted(value) for key, value in sorted(groups.items())}
    assert {k: len(v) for k, v in groups.items()} == manifest['training_source_profile_cases']
    assert len(groups) == 10 and set(map(len, groups.values())) == {390, 391}
    assert design['epochs'] == 20 and design['batch_size'] == 10 and design['steps_per_epoch'] == 391
    schedules = {}
    for epoch in range(1, 21):
        rng = np.random.default_rng(design['seed'] + epoch)
        cells = []
        for key, ids in groups.items():
            shuffled = [ids[i] for i in rng.permutation(len(ids))]
            cells.append((shuffled + shuffled[:391 - len(shuffled)]))
        batches = [[cell[step] for cell in cells] for step in range(391)]
        frequencies = collections.Counter(cid for batch in batches for cid in batch)
        assert set(frequencies) == set(cases)
        assert sum(frequencies.values()) == 3910
        for cid, count in frequencies.items():
            assert count in ({1, 2} if cases[cid]['source'] == 'dataset/asian_faces' else {1})
        for batch in batches:
            assert len(batch) == len(set(batch)) == 10
            assert sorted(cases[cid]['source'] + '/' + cases[cid]['profile'] for cid in batch) == sorted(groups)
        schedules[str(epoch)] = batches
    assert sum(map(len, schedules.values())) == design['total_optimizer_updates'] == 7820
    assert sum(len(b) for batches in schedules.values() for b in batches) == design['total_training_case_exposures'] == 78200
    schedule = {'version': 9, 'design_sha256': manifest['design_sha256'], 'manifest_sha256': MANIFEST_SHA,
                'source_profile_group_order': list(groups), 'epochs': schedules,
                'optimizer_updates': 7820, 'case_exposures': 78200, 'no_validation_cases': True,
                'numpy_version': np.__version__, 'execution_source_sha256': sha(Path(__file__)),
                'training_ready': False, 'interpretation': 'Fixed data order only; actual training executable and return auditor still required.'}
    write(BASE / 'training_schedule_v9.json', schedule)
    receipt = {'complete': True, 'version': 9, 'manifest_sha256': MANIFEST_SHA,
        'design_sha256': manifest['design_sha256'], 'schedule_sha256': sha(BASE / 'training_schedule_v9.json'),
        'pinned_assets_checked': len(manifest['assets_sha256']), 'target_pixel_hashes_checked': 885,
        'training_proxy_pixels_rebuilt': 3905, 'unchanged_validation_input_bytes_checked': 520,
        'balanced_batches_checked': 7820, 'case_exposures_checked': 78200,
        'original_roles_preserved': True, 'clear_training_inputs_exact_targets': True,
        'local_model_forwards': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'training_ready': False, 'training_started': False, 'native_reserved_used': False,
        'numpy': np.__version__, 'opencv': cv2.__version__, 'pillow': pillow_version,
        'execution_source_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started,
        'runtime_cap_seconds': 600,
        'limitation': 'Exact preparation/exposure verification does not establish model improvement or permit running an assets-only package. VM must consume these exact PNGs without regenerating camera pixels.'}
    write(out, receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()

"""Prepare exact mixed-cohort images locally; no model import/forward/training."""
import ast
import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_vm_bundle_v1'
V6 = ROOT / 'outputs/cctv_dgp_targets_vm_v6'
REVIEW = ROOT / 'outputs/cctv_dgp_asian_source_review_v9'
DESIGN = ROOT / 'outputs/cctv_dgp_mixed_design_v9.json'
OUT = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seed(case_id):
    return int.from_bytes(hashlib.sha256(('107:' + case_id).encode()).digest()[:4], 'big')


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve existing mixed-source preparation'
    assert shutil.disk_usage(ROOT).free > 2 * 1024 ** 3, 'Need2 GiB preparation headroom'
    assert sha(PARENT / 'protocol.json') == 'b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6'
    assert sha(V6 / 'targets_protocol_v6.json') == '0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c'
    assert sha(REVIEW / 'mixed_cohort.json') == '01b89afc21231c4159063fc9149a7892b8ed7529e471e2c11df79cf3d7219ac9'
    design, mixed = read(DESIGN), read(REVIEW / 'mixed_cohort.json')
    assert sha(REVIEW / 'local_integrity_audit.json') == design['source_integrity_audit_sha256']
    assert read(REVIEW / 'local_integrity_audit.json')['complete'] is True
    parent, v6 = read(PARENT / 'protocol.json'), read(V6 / 'targets_protocol_v6.json')
    refs = {r['id']: (V6, r) for r in v6['references']}
    for ref in parent['references']:
        if ref['id'] in mixed['training_reference_ids'] and ref['source'] == 'dataset/asian_faces':
            assert ref['role'] == 'train' and ref['id'] not in refs
            refs[ref['id']] = (PARENT, ref)
    ordered_ids = mixed['training_reference_ids'] + mixed['validation_reference_ids']
    assert len(refs) == len(set(ordered_ids)) == 885
    references = [copy.deepcopy(refs[rid][1]) for rid in ordered_ids]
    train = references[:781]
    assert all(r['role'] == 'train' for r in train)
    assert all(r['role'] == 'validation' for r in references[781:])
    assert dict(collections.Counter(r['source'] for r in train)) == design['training_sources']
    hq_approved = {r['reference_id']: r for r in read(V6 / 'lineage/eligible_references.json')['references']}
    OUT.mkdir()
    pins = {}
    target_pixels = {}

    def clock():
        assert time.monotonic() - started < 600, 'Preparation exceeds600-second finite budget'

    def copy_file(source, name, expected=None):
        clock()
        if expected is not None:
            assert sha(source) == expected, name
        dest = OUT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert sha(dest) == sha(source), 'Conflicting shared asset: ' + name
        else:
            shutil.copyfile(source, dest)
        pins[name] = sha(dest)

    for ref in references:
        source_root, original = refs[ref['id']]
        protocol = v6 if source_root == V6 else parent
        assert ref == original
        for key in ('target', 'observed'):
            copy_file(source_root / ref[key], ref[key], protocol['assets_sha256'][ref[key]])
        with Image.open(OUT / ref['target']) as image:
            assert image.size == (256, 256) and image.mode == 'RGB'
            target_pixels[ref['id']] = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            if ref['source'] == 'dataset/asian_faces':
                assert target_pixels[ref['id']] == ref['target_rgb_sha256']
            else:
                assert sha(OUT / ref['target']) == hq_approved[ref['id']]['target_sha256']
        with Image.open(OUT / ref['observed']) as image:
            assert image.size == (256, 256)
            assert set(np.unique(np.asarray(image))).issubset({0, 255})
    for case in v6['validation_cases']:
        copy_file(V6 / case['input'], case['input'], v6['assets_sha256'][case['input']])
    code = [name for name in v6['assets_sha256'] if name.startswith('models/') and name.endswith('.py')]
    code += ['cctv_dgp_pilot.py', 'cctv_dgp_frozen_norm.py', 'cctv_dgp_targets_v6.py', 'scripts/cctv_camera_stress.py']
    for name in code:
        ast.parse((V6 / name).read_text(encoding='utf-8'), feature_version=(3, 10))
        copy_file(V6 / name, name, v6['assets_sha256'][name])
    copies = {
        'lineage/parent_protocol.json': PARENT / 'protocol.json',
        'lineage/targets_protocol_v6.json': V6 / 'targets_protocol_v6.json',
        'lineage/mixed_design_v9.json': DESIGN,
        'lineage/hq_eligible_references.json': V6 / 'lineage/eligible_references.json',
        'lineage/training_coverage_v9_audit.json': ROOT / 'outputs/cctv_dgp_training_coverage_v9_audit.json',
        'scripts/prepare_cctv_dgp_mixed_assets_v9_r2.py': Path(__file__).resolve(),
    }
    for name in ('mixed_cohort.json', 'final_decisions.json', 'eligible_references.json', 'source_review_audit.json',
                 'local_integrity_audit.json', 'original_resolution_checks_v1.json', 'source_review_rubric.json'):
        copies['lineage/asian_' + name] = REVIEW / name
    for name, source in copies.items():
        copy_file(source, name)
    weights = {k: v6['weights'][k] for k in ('start', 'arcface')}
    cached = {name: v6['assets_sha256'][name] for name in weights.values()}
    # Use existing verified VM caches later; do not install or copy heavy runtimes locally.
    for name, pin in cached.items():
        assert sha(V6 / name) == pin
    camera_source = OUT / 'scripts/cctv_camera_stress.py'
    spec = importlib.util.spec_from_file_location('mixed_camera_pinned', camera_source)
    camera = importlib.util.module_from_spec(spec); spec.loader.exec_module(camera)
    assert camera.PROFILES == v6['evaluation_profiles']
    cases = []
    for i, ref in enumerate(train):
        target = np.asarray(Image.open(OUT / ref['target']).convert('RGB'))
        observed = np.asarray(Image.open(OUT / ref['observed'])) > 0
        for profile in camera.PROFILES:
            clock()
            cid = 'v9_' + ref['id'] + '_' + profile['id']
            noise_seed = seed(cid)
            rgb, details = camera.degrade(target, ref['bounds'], profile, noise_seed)
            assert rgb.shape == target.shape and rgb.dtype == np.uint8
            np.testing.assert_array_equal(rgb[~observed], target[~observed])
            if profile['id'] == 'clear':
                np.testing.assert_array_equal(rgb, target)
            name = 'data/mixed_training/' + cid + '.png'
            file = OUT / name; file.parent.mkdir(parents=True, exist_ok=True)
            assert not file.exists(); Image.fromarray(rgb).save(file)
            pins[name] = sha(file)
            cases.append({'id': cid, 'reference_id': ref['id'], 'source': ref['source'],
                          'profile': profile['id'], 'input': name, 'camera': profile, 'proxy_details': details})
        if (i + 1) % 100 == 0:
            print(f'prepared training sources {i + 1}/781; cases={len(cases)} elapsed={time.monotonic()-started:.0f}s', flush=True)
    assert len(cases) == len({c['id'] for c in cases}) == 3905
    counts = dict(sorted(collections.Counter(c['source'] + '/' + c['profile'] for c in cases).items()))
    assert set(counts.values()) == {390, 391} and len(counts) == 10
    previews = []
    for source in ('dataset/thumbnails128x128', 'dataset/asian_faces'):
        first = min(r['id'] for r in train if r['source'] == source)
        previews += [c for c in cases if c['reference_id'] == first]
    sheet = Image.new('RGB', (520, 2904), '#eeeeee'); draw = ImageDraw.Draw(sheet)
    draw.text((2, 3), 'training proxy input | reviewed source target; no model outputs', fill='black')
    by_id = {r['id']: r for r in references}
    for i, case in enumerate(previews):
        for j, file in enumerate([OUT / case['input'], OUT / by_id[case['reference_id']]['target']]):
            y = 24 + 288 * i
            draw.text((2 + j * 260, y + 2), case['id'], fill='black')
            with Image.open(file) as image:
                sheet.paste(image, (2 + j * 260, y + 28))
    preview = OUT / 'training_proxy_preview_10_rows.png'; sheet.save(preview)
    pins[preview.name] = sha(preview)
    manifest = {
        'version': 9, 'format': 'cctv-dgp-mixed-assets-v9', 'date': '2026-10-04',
        'design_sha256': sha(DESIGN), 'mixed_cohort_sha256': sha(REVIEW / 'mixed_cohort.json'),
        'canonical_target_rgb_sha256': target_pixels,
        'legacy_metadata_note': 'Immutable V6 HQ reference target_rgb_sha256 describes original thumbnail-derived pixels. Actual canonical HQ target pixels are separately pinned here; all files match the reviewed HQ ledger. Asian original pixel hashes remain exact.',
        'references': references, 'training_cases': cases, 'validation_cases': v6['validation_cases'],
        'validation_preview_case_ids': v6['preview_case_ids'], 'training_preview_case_ids': [c['id'] for c in previews],
        'evaluation_profiles': v6['evaluation_profiles'], 'training_source_profile_cases': counts,
        'weights': weights, 'cache_assets': cached, 'assets_sha256': pins,
        'validation_byte_copies': 520, 'target_mask_byte_copies': 1770,
        'historical_split_sha256': v6['historical_split_sha256'],
        'precomputed_exact_inputs_required': True, 'regenerate_camera_inputs_on_vm': False,
        'training_ready': False, 'execution_prepared': False, 'training_started': False,
        'local_model_forwards': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'native_used': False, 'native_reserved_used': False, 'production_promotion_permitted': False,
        'seconds': time.monotonic() - started, 'runtime_cap_seconds': 600,
        'next_required': 'Independent asset/sampler audit, VM-only executable trainer plus return auditor/supervisor, frozen executable protocol, exact-byte verified transfer. Do not launch this assets-only root.',
    }
    write(OUT / 'cohort_assets_manifest_v9.json', manifest)
    (OUT / 'cohort_assets_manifest_v9.sha256').write_text(sha(OUT / 'cohort_assets_manifest_v9.json') + '\n', encoding='ascii', newline='\n')
    receipt = {k: manifest[k] for k in ('version', 'training_source_profile_cases', 'validation_byte_copies', 'target_mask_byte_copies',
        'training_ready', 'execution_prepared', 'training_started', 'local_model_forwards', 'local_backward_calls',
        'local_optimizer_updates', 'native_reserved_used', 'seconds', 'runtime_cap_seconds')}
    receipt.update({'complete': True, 'training_references': 781, 'training_cases': 3905, 'validation_references': 104,
                    'manifest_sha256': sha(OUT / 'cohort_assets_manifest_v9.json'), 'prepared_assets': len(pins),
                    'execution_source_sha256': sha(Path(__file__))})
    write(ROOT / 'outputs/cctv_dgp_mixed_assets_v9_r2_preparation.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()

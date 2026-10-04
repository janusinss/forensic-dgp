"""Training coverage/integrity audit for the next design; no model operations."""
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run():
    started = time.monotonic()
    v6 = Path('outputs/cctv_dgp_targets_vm_v6')
    v1 = Path('outputs/cctv_dgp_vm_bundle_v1')
    parent_pin = 'b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6'
    assert sha(v1 / 'protocol.json') == sha(v6 / 'lineage/parent_protocol.json') == parent_pin
    assert sha(v6 / 'targets_protocol_v6.json') == '0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c'
    assert sha(v6 / 'lineage/eligible_references.json') == 'cadeedd2906fe169b7cc9741aac9fc29db8aedc1ba21d4438f97443490c1281e'
    parent = json.loads((v1 / 'protocol.json').read_text())
    current = json.loads((v6 / 'targets_protocol_v6.json').read_text())
    eligible = json.loads((v6 / 'lineage/eligible_references.json').read_text())
    ffhq = [r for r in current['references'] if r['role'] == 'train']
    asian = [r for r in parent['references'] if r['role'] == 'train' and r['source'] == 'dataset/asian_faces']
    validation = [r for r in current['references'] if r['role'] == 'validation']
    assert len(ffhq) == 391 and len(asian) == 451 and len(validation) == 104
    assert {r['id'] for r in ffhq} == {r['reference_id'] for r in eligible['references'] if r['original_role'] == 'train'}
    candidate = ffhq + asian
    assert len({r['id'] for r in candidate}) == 842
    assert not ({r['id'] for r in candidate} & {r['id'] for r in validation})
    collisions = sorted({r['source_sha256'] for r in candidate} & {r['source_sha256'] for r in validation})
    dimensions = []; grayscale = 0; target_pixel_hashes = 0; assets = set()
    for ref in asian:
        assert time.monotonic() - started <= 120
        for name in (ref['native'], ref['target'], ref['observed']):
            assert sha(v1 / name) == parent['assets_sha256'][name], name
            assets.add(name)
        assert sha(v1 / ref['native']) == ref['source_sha256']
        with Image.open(v1 / ref['native']) as image:
            assert list(image.size) == ref['native_size']; dimensions.append(list(image.size))
        with Image.open(v1 / ref['target']) as image:
            assert image.size == (256, 256) and image.mode == 'RGB'
            rgb = np.asarray(image)
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == ref['target_rgb_sha256']
            target_pixel_hashes += 1
            grayscale += int(float((rgb.max(2) - rgb.min(2) <= 2).mean()) >= 0.995)
        with Image.open(v1 / ref['observed']) as image:
            mask = np.asarray(image)
            assert mask.shape == (256, 256) and set(np.unique(mask)).issubset({0, 255})
    profile_counts = {}
    for epoch in ['1', '2']:
        rows = [c for c in current['training_epochs'][epoch]]
        rows += [c for c in parent['training_epochs'][epoch] if c['reference_id'] in {r['id'] for r in asian}]
        assert len(rows) == 842 and len({c['reference_id'] for c in rows}) == 842
        profile_counts[epoch] = {}
        for source in ['dataset/thumbnails128x128', 'dataset/asian_faces']:
            selected = [c for c in rows if c['source'] == source]
            profile_counts[epoch][source] = dict(sorted(collections.Counter(c['profile'] for c in selected).items()))
            for case in selected:
                package = v6 if source == 'dataset/thumbnails128x128' else v1
                protocol = current if source == 'dataset/thumbnails128x128' else parent
                assert sha(package / case['input']) == protocol['assets_sha256'][case['input']]
    result = {
        'complete': True, 'purpose': 'Next-design coverage audit; not an executable training protocol or model improvement',
        'parent_protocol_sha256': parent_pin, 'v6_protocol_sha256': sha(v6 / 'targets_protocol_v6.json'),
        'source_ledger_sha256': sha(v6 / 'lineage/eligible_references.json'),
        'v6_training_sources': {'dataset/thumbnails128x128': 391, 'dataset/asian_faces': 0},
        'proposed_training_sources': {'dataset/thumbnails128x128': 391, 'dataset/asian_faces': 451},
        'unchanged_validation_sources': {'dataset/thumbnails128x128': 53, 'dataset/asian_faces': 51},
        'training_validation_reference_ids_disjoint': True, 'exact_source_hash_collisions_train_validation': collisions,
        'asian_original_assets_checked': len(assets), 'asian_target_pixel_hashes_checked': target_pixel_hashes,
        'asian_native_sizes': dimensions, 'asian_minimum_edge_below256': sum(min(size) < 256 for size in dimensions),
        'asian_grayscale_like_targets': grayscale, 'two_epoch_source_profile_counts': profile_counts,
        'camera_inputs_checked': 1684, 'historical_roles_preserved': True,
        'seconds': time.monotonic() - started, 'runtime_cap_seconds': 120,
        'local_model_forwards': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'native_used': False, 'native_reserved_used': False, 'execution_prepared': False, 'training_started': False,
        'limitations': ['Exact reference/source separation does not establish subject identity disjointness or historical exposure.',
            'The HQ FFHQ source review does not cover the Asian training photos. Source-only covering/quality review is required before a new generator training freeze.',
            'Original Asian targets vary in native resolution; their 256-pixel canvas is not necessarily a genuine HQ clean target.',
            'V8 two-image fitting cannot identify task imbalance as the sole cause or qualify generalization.'],
    }
    path = Path('outputs/cctv_dgp_training_coverage_v9_audit.json')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print({k: v for k, v in result.items() if k not in ('asian_native_sizes', 'limitations')}, flush=True)


if __name__ == '__main__':
    run()

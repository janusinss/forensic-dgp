"""Independent raw/PNG/state readback and ten frozen initial-model replays."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_spatial_decoder_v39 import SpatialDGPCandidateV39, SpatialDecoder
from cctv_dgp_pilot import state_hash
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer

OUT = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixels(path, mode='RGB'):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode)).copy()


def main():
    start = time.monotonic(); assert not (OUT / 'independent_initial_audit.json').exists()
    p, r, e = [read(OUT / n) for n in ['plan.json', 'results.json', 'execution.json']]
    assert r['complete'] and r['plan_sha256'] == e['plan_sha256'] == sha(OUT / 'plan.json')
    for name, digest in p['sources_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, digest in r['artifacts_sha256'].items(): assert sha(OUT / name) == digest, name
    assert r['cases'] == 50 and len(p['cases']) == len(r['rows']) == 50
    assert p['gradient_calls'] == r['gradient_calls'] == r['optimizer_updates'] == 0
    assert not r['trained_checkpoint_created'] and not r['gradient_connectivity_established']
    seed = torch.load(OUT / 'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    declared = {row['name']: row for row in p['parameter_layout']}
    assert set(seed) == set(declared) and len(seed) == r['decoder_tensors'] == 57
    assert sum(v.numel() for v in seed.values()) == r['decoder_parameters'] == 17952
    for name, value in seed.items():
        assert value.dtype == torch.float32 and torch.isfinite(value).all() and list(value.shape) == declared[name]['shape']
    max_cache_error = 0.; compositions = 0
    for c, row in zip(p['cases'], r['rows'], strict=True):
        assert row['id'] == c['id'] and row['raw_maximum_error'] == row['PNG_maximum_byte_error'] == 0
        image = pixels(PARENT / c['input']); mask = pixels(PARENT / c['observed'], 'L') > 0
        a, b = [np.load(OUT / ('raw/' + c['id'] + '_' + k + '.npy'), allow_pickle=False) for k in ['original_raw', 'result']]
        assert a.dtype == b.dtype == np.float32 and a.shape == b.shape == (256, 256, 3) and np.array_equal(a, b)
        assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
        assert np.array_equal(a[~mask], image[~mask].astype(np.float32) / np.float32(255))
        for key in ['original_raw', 'result']:
            png = pixels(OUT / ('images/' + c['id'] + '_' + key + '.png'))
            expected = np.floor(a * np.float32(255)).astype(np.uint8); expected[~mask] = image[~mask]
            assert np.array_equal(png, expected); compositions += 1
        cache = np.load(PARENT / c['raw_dgp'], allow_pickle=False)
        err = float(np.abs(cache[mask] - a[mask]).max()); assert err <= 1e-5
        max_cache_error = max(max_cache_error, err)
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    model = SpatialDGPCandidateV39(original, seed)
    states = {'original': state_hash(original.net), 'decoder': state_hash(model.decoder), 'reference_decoder': state_hash(model.reference_decoder)}
    assert states == e['states_before'] == r['states_before_after']
    layout = []; offset = 0
    for name, value in model.decoder.named_parameters():
        layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()}); offset += value.numel()
    assert layout == p['parameter_layout']
    assert not ({v.data_ptr() for v in model.decoder.parameters()} & {v.data_ptr() for v in model.reference_decoder.parameters()})
    # Fixed metadata rule: first reference in each source, all five profiles.
    chosen = []
    for source in sorted({c['source'] for c in p['cases']}):
        ref = next(c['source_person_or_reference'] for c in p['cases'] if c['source'] == source)
        chosen += [c for c in p['cases'] if c['source_person_or_reference'] == ref]
    assert len(chosen) == 10
    with torch.inference_mode():
        for c in chosen:
            assert time.monotonic() - start <= 180
            image = pixels(PARENT / c['input']); mask = pixels(PARENT / c['observed'], 'L') > 0
            x = torch.from_numpy(image.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
            s = torch.from_numpy(mask)[None, None]
            raw = model(x, s)[0].permute(1, 2, 0).numpy().copy()
            cached = np.load(OUT / ('raw/' + c['id'] + '_result.npy'), allow_pickle=False)
            assert np.array_equal(raw, cached)
    assert states == {'original': state_hash(original.net), 'decoder': state_hash(model.decoder), 'reference_decoder': state_hash(model.reference_decoder)}
    assert all(not v.requires_grad and v.grad is None for v in model.parameters()) and all(not m.training for m in model.modules())
    assert r['model_forwards'] == {'original_DGP': 101, 'decoder': 51, 'reference_decoder': 51}
    assert r['partial_support_exact'] and r['invalid_inputs_rejected_before_model_calls'] == 5
    assert 'Linux VM' in r['local_learning_rejection']
    value = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'results_sha256': sha(OUT / 'results.json'),
             'sources_verified': len(p['sources_sha256']), 'artifacts_verified': len(r['artifacts_sha256']),
             'all50_raw_pairs_exact': True, 'all100_PNG_compositions_exact': compositions == 100,
             'seed_tensors': 57, 'seed_parameter_values': 17952, 'seed_is_untrained_asset': True,
             'all_original_and_seed_states_unchanged': True, 'first_reference_per_source_replay_cases': [c['id'] for c in chosen],
             'frozen_DGP_replay_forwards': 10, 'decoder_replay_forwards': 20,
             'historical_cache_raw_maximum_error': max_cache_error, 'historical_cache_tolerance': 1e-5,
             'local_gradient_calls': 0, 'optimizer_updates': 0, 'gradient_connectivity_established': False,
             'native_or_reserved_used': False, 'app_changed': False, 'restoration_qualified': False,
             'goal_complete': False, 'seconds': time.monotonic() - start, 'cap_seconds': 180}
    assert value['seconds'] <= 180
    with (OUT / 'independent_initial_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'cases': 50, 'CPU_replays': 10, 'seconds': value['seconds']}), flush=True)


if __name__ == '__main__':
    main()

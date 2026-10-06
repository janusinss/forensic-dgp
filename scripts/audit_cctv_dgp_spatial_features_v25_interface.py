"""Independent saved-array/source checks for the V25 untrained interface."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_spatial_features_v25_interface'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    result = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    p = json.loads((ROOT / 'outputs/cctv_dgp_degraded_detail_vm_v24/protocol.json').read_text(encoding='utf-8'))
    for name, expected in result['source_bindings_sha256'].items():
        path = (ROOT / name).resolve(); assert path.is_relative_to(ROOT) and sha(path) == expected
    for name, expected in result['artifacts_sha256'].items():
        assert sha(OUT / name) == expected
    assert result['counts'] == {'DGP_forwards': 4, 'head_forwards': 5}
    assert result['trainable_design_parameters'] == 53781
    assert result['backward_calls'] == result['optimizer_updates'] == 0 and not result['learned_capacity_or_quality_proven']
    feature_count = 0
    for row in result['rows']:
        case = next(c for c in p['cases'] if c['id'] == row['id'])
        base = np.load(OUT / (row['id'] + '_baseline.npy'), allow_pickle=False)
        initial = np.load(OUT / (row['id'] + '_initial.npy'), allow_pickle=False)
        assert base.dtype == np.float32 and base.shape == (256, 256, 3) and np.isfinite(base).all()
        assert np.array_equal(base, initial)
        with Image.open(ROOT / 'outputs/cctv_dgp_degraded_detail_vm_v24' / case['input']) as image:
            camera = np.asarray(image).copy()
        with Image.open(ROOT / 'outputs/cctv_dgp_degraded_detail_vm_v24' / case['observed']) as image:
            support = np.asarray(image).copy() > 0
        with Image.open(OUT / (row['id'] + '_initial.png')) as image:
            actual = np.asarray(image).copy()
        expected = np.where(support[..., None], np.floor(base * np.float32(255)), camera).astype(np.uint8)
        assert np.array_equal(expected, actual)
        with np.load(OUT / (row['id'] + '_features.npz'), allow_pickle=False) as cache:
            assert sorted(cache.files) == ['fpn0', 'fpn1', 'fpn2', 'fpn3', 'fpn4']
            for index, receipt in enumerate(row['features']):
                value = cache['fpn' + str(index)]
                assert value.dtype == np.float32 and [1] + list(value.shape) == receipt['shape'] and np.isfinite(value).all()
                assert hashlib.sha256(value.tobytes()).hexdigest() == receipt['sha256']
                assert not receipt['inference_tensor'] and not receipt['requires_grad']
                feature_count += 1
    u = np.arange(256, dtype=np.float32)[None, :]
    scene = (np.float32(.5) + np.float32(.15) * np.sin(np.float32(2 * np.pi) * u / np.float32(128))).repeat(256, axis=0)
    q = np.float32(.05) * np.tanh(np.float32(.1) * scene)
    support = np.zeros((256, 256), dtype=bool); support[32:224, 16:240] = True
    expected = np.where(support, np.float32(.5) + q - q[support].mean(dtype=np.float32), np.float32(.5))
    actual = np.load(OUT / 'disposable_smooth_raw.npy', allow_pickle=False)
    maximum = float(np.abs(actual - expected[..., None]).max())
    assert maximum <= 1e-7 and np.array_equal(actual[~support], np.full_like(actual[~support], .5))
    assert result['disposable_feature0_connection_maximum_response'] > 0 and feature_count == 10
    receipt = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'results_sha256': sha(OUT / 'results.json'), 'source_bindings_verified': len(result['source_bindings_sha256']),
        'artifact_bindings_verified': len(result['artifacts_sha256']), 'exact_baseline_raw_PNG_pairs': 2,
        'feature_arrays_verified': feature_count, 'independent_smooth_signal_maximum_difference': maximum,
        'scope_limit': 'Saved-array/arithmetic validation and source pins; no independent re-extraction of intermediate neural features or proof of CUDA gradients, fitting or useful restoration.',
        'neural_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'goal_complete': False,
        'seconds': time.monotonic() - started}
    with (OUT / 'independent_saved_interface_audit.json').open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

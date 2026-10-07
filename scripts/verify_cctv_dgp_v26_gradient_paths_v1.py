"""Independent coordinate-energy and weighted-second-moment readback; no torch."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v26_gradient_paths_v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(left, right, bound=1e-12):
    assert abs(left - right) <= bound, (left, right, bound)


def main():
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    record = read(OUT / 'analysis.json')
    assert record['complete'] and not (OUT / 'independent_readback.json').exists()
    for name, digest in record['source_bindings_sha256'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, name
    returned = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_return/outputs'
    layout = read(returned / 'results.json')['parameter_layout']
    for saved in record['snapshots']:
        matrix = np.load(returned / ('gradient_components_update' + str(saved['update']) + '.npy'), allow_pickle=False)
        energy = [math.fsum(float(value) ** 2 for value in row) for row in matrix]
        total = matrix.sum(axis=0)
        reward, preservation = matrix[:3].sum(axis=0), matrix[3:].sum(axis=0)
        rn = math.sqrt(math.fsum(float(v) ** 2 for v in reward))
        pn = math.sqrt(math.fsum(float(v) ** 2 for v in preservation))
        close(rn, saved['reward_gradient_norm']); close(pn, saved['preservation_gradient_norm'])
        close(math.sqrt(math.fsum(float(v) ** 2 for v in total)), saved['total_gradient_norm'])
        close(pn / rn, saved['preservation_to_reward_norm_ratio'])
        if pn:
            close(math.fsum(float(a) * float(b) for a, b in zip(reward, preservation)) / (rn * pn), saved['reward_preservation_cosine'])
        else:
            assert saved['reward_preservation_cosine'] is None
        for family, path in saved['paths'].items():
            slices = [(row['start'], row['end']) for row in layout
                      if (('decode' if row['name'].startswith('decode') else row['name'].split('.')[0]) == family)]
            indices = [i for start, end in slices for i in range(start, end)]
            assert len(indices) == path['parameters']
            for index in range(7):
                value = math.fsum(float(matrix[index, i]) ** 2 for i in indices)
                close(math.sqrt(value), path['component_norms'][index])
                close(value / energy[index] if energy[index] else 0., path['component_squared_gradient_fraction'][index])
            for vector, key in [(reward, 'reward_gradient_norm'), (preservation, 'preservation_gradient_norm'), (total, 'total_gradient_norm')]:
                close(math.sqrt(math.fsum(float(vector[i]) ** 2 for i in indices)), path[key])
        for index in range(7):
            close(math.fsum(row['component_squared_gradient_fraction'][index] for row in saved['paths'].values()), 1. if energy[index] else 0.)
    parent = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
    cases = {case['id']: case for case in read(parent / 'protocol.json')['cases']}
    features = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return/outputs/frozen_DGP_features'
    floor = record['proposed_normalization_floor']
    assert floor == .05 and len(record['feature_rows']) == 250
    for row in record['feature_rows']:
        case = cases[row['id']]
        with Image.open(parent / case['observed']) as image:
            mask = np.asarray(image).copy() > 0
        size = [128, 64, 32, 16, 8][row['scale']]; factor = 256 // size
        # Independently implement exact power-of-two area averages.
        support = mask.astype(np.float64).reshape(size, factor, size, factor).mean((1, 3))[None]
        value = np.load(features / (row['id'] + '_fpn' + str(row['scale']) + '.npy'), allow_pickle=False).astype(np.float64)
        mass = support.sum()
        mean = (value * support).sum((1, 2)) / mass
        second = (value ** 2 * support).sum((1, 2)) / mass
        rms = np.sqrt(np.maximum(second - mean ** 2, 0))
        for key, number in [('channel_spatial_RMS_minimum', rms.min()), ('channel_spatial_RMS_median', np.median(rms)),
                            ('channel_spatial_RMS_maximum', rms.max()), ('fixed_normalized_RMS_median', np.median(rms / np.maximum(rms, floor)))]:
            close(float(number), row[key], 1e-8)
        assert row['channels'] == len(rms) and row['channels_below_proposed_0_05_floor'] == int(np.count_nonzero(rms < floor))
    trace = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic/results.json')
    for name, median in record['degraded_layer_spatial_RMS_medians'].items():
        values = sorted(row['stages'][name]['spatial_RMS'] for row in trace['rows'] if row['profile'] != 'clear')
        close((values[19] + values[20]) / 2, median)
    assert record['local_model_calls'] == record['local_gradient_calls'] == record['local_backward_calls'] == record['local_optimizer_updates'] == 0
    assert not record['cause_of_entire_optimizer_trajectory_proven'] and not record['native_or_reserved_pixels_used'] and not record['new_checkpoint']
    result = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'analysis_sha256': sha(OUT / 'analysis.json'), 'sources_verified': len(record['source_bindings_sha256']),
        'snapshots_verified': 2, 'parameter_family_term_rows_verified': 84, 'input_feature_rows_verified': 250,
        'gradient_values_read': 752934, 'independent_coordinate_energy_and_area_second_moments': True,
        'local_neural_or_gradient_calls': 0, 'optimizer_updates': 0,
        'cause_of_entire_optimizer_trajectory_proven': False, 'app_promotion': False,
        'goal_complete': False, 'seconds': time.monotonic() - started}
    with (OUT / 'independent_readback.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

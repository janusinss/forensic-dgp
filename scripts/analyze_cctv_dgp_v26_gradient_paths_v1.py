"""Saved-array spatial-path analysis. No neural calls, local derivatives or fit."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v26_gradient_paths_v1'
RETURN = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_return'
PARENT = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
SAVED = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
FAMILIES = ['projections', 'decode', 'camera', 'fuse', 'tail', 'direct']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def family(name):
    return 'decode' if name.startswith('decode') else name.split('.')[0]


def main():
    import cv2
    import numpy as np
    from PIL import Image

    started = time.monotonic()
    assert not OUT.exists(), 'Preserve original analysis'
    audit_path = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['VM_optimizer_updates_verified'] == 0
    result = read(RETURN / 'outputs/results.json')
    assert result['component_gradient_calls'] == 140 and result['optimizer_updates'] == 0
    layout = result['parameter_layout']
    assert len(layout) == 26 and layout[-1]['end'] == 53781
    snapshots = []
    bindings = {}

    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)

    for path in [audit_path, RETURN / 'outputs/results.json', RETURN / 'protocol.json',
                 PARENT / 'protocol.json', SAVED / 'outputs/frozen_DGP_features.json',
                 ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic/results.json',
                 Path(__file__)]:
        bind(path)
    for update in [0, 50]:
        path = RETURN / 'outputs' / ('gradient_components_update' + str(update) + '.npy')
        bind(path)
        matrix = np.load(path, allow_pickle=False)
        assert matrix.dtype == np.float64 and matrix.shape == (7, 53781) and np.isfinite(matrix).all()
        reward, penalty = matrix[:3].sum(0), matrix[3:].sum(0)
        norms = np.linalg.norm(matrix, axis=1)
        reward_norm, penalty_norm = float(np.linalg.norm(reward)), float(np.linalg.norm(penalty))
        paths = {}
        for group in FAMILIES:
            indices = np.concatenate([np.arange(row['start'], row['end']) for row in layout if family(row['name']) == group])
            part = matrix[:, indices]
            energy = np.square(part).sum(1)
            paths[group] = {'parameters': len(indices), 'component_norms': np.sqrt(energy).tolist(),
                'component_squared_gradient_fraction': np.divide(energy, norms ** 2,
                    out=np.zeros(7), where=norms > 0).tolist(),
                'reward_gradient_norm': float(np.linalg.norm(part[:3].sum(0))),
                'preservation_gradient_norm': float(np.linalg.norm(part[3:].sum(0))),
                'total_gradient_norm': float(np.linalg.norm(part.sum(0)))}
        snapshots.append({'update': update, 'paths': paths,
            'reward_gradient_norm': reward_norm, 'preservation_gradient_norm': penalty_norm,
            'preservation_to_reward_norm_ratio': penalty_norm / reward_norm,
            'reward_preservation_cosine': float(reward @ penalty / (reward_norm * penalty_norm)) if penalty_norm else None,
            'total_gradient_norm': float(np.linalg.norm(reward + penalty))})
    cache = read(SAVED / 'outputs/frozen_DGP_features.json')
    protocol = read(PARENT / 'protocol.json')
    feature_rows = []
    for case in protocol['cases']:
        mask_path = PARENT / case['observed']
        bind(mask_path)
        with Image.open(mask_path) as image:
            mask = (np.asarray(image).copy() > 0).astype(np.float64)
        for index, size in enumerate([128, 64, 32, 16, 8]):
            name = case['id'] + '_fpn' + str(index) + '.npy'
            path = SAVED / 'outputs/frozen_DGP_features' / name
            assert sha(path) == cache['files_sha256'][name]
            bind(path)
            value = np.load(path, allow_pickle=False).astype(np.float64)
            weights = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)[None]
            mass = weights.sum()
            mean = (value * weights).sum((1, 2), keepdims=True) / mass
            rms = np.sqrt(((value - mean) ** 2 * weights).sum((1, 2)) / mass)
            feature_rows.append({'id': case['id'], 'profile': case['profile'], 'scale': index,
                'channel_spatial_RMS_minimum': float(rms.min()), 'channel_spatial_RMS_median': float(np.median(rms)),
                'channel_spatial_RMS_maximum': float(rms.max()),
                'channels_below_proposed_0_05_floor': int(np.count_nonzero(rms < .05)),
                'channels': len(rms), 'fixed_normalized_RMS_median': float(np.median(rms / np.maximum(rms, .05)))})
    layer_trace = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic/results.json')
    degraded = [row for row in layer_trace['rows'] if row['profile'] != 'clear']
    layer_medians = {stage: float(np.median([row['stages'][stage]['spatial_RMS'] for row in degraded]))
                     for stage in degraded[0]['stages']}
    record = {'complete': True, 'date': '2026-10-06', 'source_bindings_sha256': bindings,
        'scope': 'Saved matrices and input-only feature/layer magnitudes; parameter energy is not restoration quality or optimizer trajectory causality',
        'snapshots': snapshots, 'fixed_terms': result['terms'], 'feature_rows': feature_rows,
        'degraded_layer_spatial_RMS_medians': layer_medians,
        'proposed_normalization_floor': .05, 'normalization_is_per_input_channel_spatial_RMS': True,
        'hypothesis': 'Direct RGB dominates the saved landmark gradient; add zero-initialized direct multiscale feature readouts to shorten its spatial path, retaining the original decoder, corrected objective and every scientific stop.',
        'cause_of_entire_optimizer_trajectory_proven': False, 'local_model_calls': 0,
        'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'native_or_reserved_pixels_used': False, 'new_checkpoint': False, 'app_promotion': False,
        'goal_complete': False, 'seconds': time.monotonic() - started}
    OUT.mkdir()
    with (OUT / 'analysis.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'snapshots': snapshots,
        'feature_rows': len(feature_rows), 'sources': len(bindings), 'seconds': record['seconds']}, indent=2))


if __name__ == '__main__':
    main()

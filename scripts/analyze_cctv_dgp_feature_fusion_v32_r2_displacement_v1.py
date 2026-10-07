"""Measure the saved original-to-stopped parameter path; no model or optimizer."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_displacement_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def describe(before, delta):
    norm, old_norm = float(np.linalg.norm(delta)), float(np.linalg.norm(before))
    return {'parameters': len(delta), 'initial_weight_L2': old_norm,
            'displacement_L2': norm, 'displacement_RMS': float(np.sqrt(np.square(delta).mean())),
            'relative_weight_L2_change': norm / max(old_norm, 1e-30),
            'maximum_absolute_parameter_change': float(np.abs(delta).max())}


def main():
    started = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json'
    imported_path = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_import.json'
    audit, imported, p = map(read, [audit_path, imported_path, BUNDLE / 'protocol.json'])
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['necessary_capacity_pass']
    assert [s['update'] for s in audit['snapshots_audited']] == [0, 50]
    assert sha(BUNDLE / 'protocol.json') == audit['protocol_sha256']
    paths = [RETURNED / f'outputs/update{n}/dgp_candidate_v32.pth' for n in [0, 50]]
    matrix_path, summary_path = [RETURNED / 'outputs' / n for n in ['gradient_components.npy', 'gradient_summary.json']]
    for path in paths + [matrix_path, summary_path]:
        assert sha(path) == imported['files_sha256'][path.relative_to(RETURNED).as_posix()]
    initial, stopped = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    assert set(initial) == set(stopped)
    layout = p['parameter_layout']
    selected = {row['name'] for row in layout}
    assert len(selected) == 23 and layout[-1]['end'] == 978243
    for name in initial:
        assert initial[name].shape == stopped[name].shape and torch.isfinite(stopped[name]).all()
        if name not in selected:
            assert torch.equal(initial[name], stopped[name]), name
    before = np.concatenate([initial[row['name']].numpy().ravel().astype(np.float64) for row in layout])
    after = np.concatenate([stopped[row['name']].numpy().ravel().astype(np.float64) for row in layout])
    delta = after - before
    matrix = np.load(matrix_path, allow_pickle=False)
    summary = read(summary_path)
    assert matrix.shape == (7, 978243) and matrix.dtype == np.float64 and np.isfinite(matrix).all()
    assert summary['complete'] and summary['parameter_layout'] == layout and len(summary['terms']) == 7
    assert summary['gradient_array_sha256'] == sha(matrix_path)
    fusion = set(p['fusion_parameter_names'])
    per_tensor, partition_indices = [], {'fusion': [], 'decoder': []}
    for row in layout:
        a, b = row['start'], row['end']
        group = 'fusion' if row['name'] in fusion else 'decoder'
        partition_indices[group].extend(range(a, b))
        per_tensor.append({'name': row['name'], 'partition': group, **describe(before[a:b], delta[a:b]),
                           'initial_component_gradient_dot_displacement': dict(zip(summary['terms'], (matrix[:, a:b] @ delta[a:b]).tolist(), strict=True))})
    components = []
    delta_norm = float(np.linalg.norm(delta))
    for name, gradient in zip(summary['terms'], matrix, strict=True):
        norm = float(np.linalg.norm(gradient))
        dot = float(gradient @ delta)
        components.append({'name': name, 'initial_gradient_L2': norm,
                           'initial_gradient_dot_actual50_parameter_displacement': dot,
                           'cosine_with_negative_initial_gradient': -dot / (norm * delta_norm) if norm and delta_norm else None})
    total = matrix.sum(0)
    receipt = {'complete': True, 'analyzer_sha256': sha(Path(__file__)),
               'independent_return_audit_sha256': sha(audit_path), 'protocol_sha256': sha(BUNDLE / 'protocol.json'),
               'source_bindings_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in
                                        paths + [matrix_path, summary_path, audit_path, imported_path]},
               'all_selected': describe(before, delta), 'per_tensor': per_tensor,
               'partitions': {name: describe(before[index], delta[index]) for name, index in partition_indices.items()},
               'components': components, 'initial_total_gradient_dot_displacement': float(total @ delta),
               'cosine_with_negative_initial_total_gradient': -float(total @ delta) / max(float(np.linalg.norm(total)) * delta_norm, 1e-30),
               'scope': 'Actual finite50 parameter difference and original fixed50 TRAIN component gradients.',
               'limits': 'First-order quantities at the original point, not a finite loss prediction, optimizer trajectory reconstruction, causal isolation, or native evaluation.',
               'frozen_partition_unchanged': True, 'model_constructed': False, 'neural_calls': 0,
               'gradient_calls': 0, 'optimizer_updates': 0, 'local_training': False,
               'native_or_reserved_used': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    assert time.monotonic() - started < 60, 'Finite saved-array analysis budget'
    assert not OUT.exists()
    OUT.mkdir()
    with (OUT / 'analysis.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({'complete': True, 'partitions': receipt['partitions'], 'components': components,
                      'cosine_with_negative_initial_total_gradient': receipt['cosine_with_negative_initial_total_gradient'],
                      'model_constructed': False, 'optimizer_updates': 0, 'seconds': receipt['seconds']}, indent=2))


if __name__ == '__main__':
    main()

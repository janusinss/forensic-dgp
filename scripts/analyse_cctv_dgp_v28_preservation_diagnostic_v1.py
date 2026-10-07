"""Independent arithmetic on retained VM derivatives; no local derivative queries."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return/outputs'
OUT = ROOT / 'outputs/cctv_dgp_v28_preservation_diagnostic_v1_analysis'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    audit_path = ROOT / 'outputs/cctv_dgp_v28_preservation_diagnostic_v1_independent_audit_r1.json'
    audit = json.loads(audit_path.read_text())
    assert audit['complete'] and audit['diagnostic_complete'] and audit['saved_gradient_values_checked'] == 54848970
    assert audit['optimizer_updates'] == audit['local_gradient_calls'] == 0
    assert not OUT.exists()
    summary = json.loads((SOURCE / 'gradient_summary.json').read_text())
    result = json.loads((SOURCE / 'results.json').read_text())
    g = np.load(SOURCE / 'gradient_components.npy', allow_pickle=False)
    delta = np.load(SOURCE / 'parameter_displacement.npy', allow_pickle=False)
    original = g[:7].sum(0)
    improvement = g[:3].sum(0)
    preservation = g[3:7].sum(0)
    directional = -(g @ original)
    relative = directional / max(float(original @ original), 1e-300)
    cosine = float(improvement @ preservation) / (float(np.linalg.norm(improvement)) * float(np.linalg.norm(preservation)))
    parity = result['fresh_same_batch_replay_rows']
    assert len(parity) == 50 and all(r['final_PNG_exact'] for r in parity)
    measurements = {
        'complete': True, 'basis': 'Immutable final V28 state; whole50 mean of ten fixed five-case batches',
        'source_sha256': {str(path.relative_to(ROOT).as_posix()): sha(path) for path in [audit_path, SOURCE/'gradient_components.npy', SOURCE/'gradient_summary.json', SOURCE/'parameter_displacement.npy', SOURCE/'results.json']},
        'terms': summary['terms'], 'original_seven_term_objective': summary['original_seven_term_objective'],
        'original_objective_gradient_norm': float(np.linalg.norm(original)),
        'improvement_gradient_norm': float(np.linalg.norm(improvement)),
        'preservation_gradient_norm': float(np.linalg.norm(preservation)),
        'improvement_vs_preservation_gradient_cosine': cosine,
        'directional_derivative_along_negative_original_objective_gradient': directional.tolist(),
        'directional_derivative_per_unit_original_objective_descent': relative.tolist(),
        'component_dot_observed_original_to_final_displacement': (g @ delta).tolist(),
        'per_parameter_negative_objective_vs_mean_anchor_dot': {
            row['name']: float(-g[7, row['start']:row['end']] @ original[row['start']:row['end']])
            for row in summary['parameter_layout']},
        'all50_fresh_VM_parity_maxima': {key: max(r[key] for r in parity) for key in ['initial_raw_maximum_error', 'final_raw_maximum_error', 'final_vector_maximum_error', 'truth_vector_maximum_error']},
        'mean_anchor_increases_along_plain_SGD_direction_at_final_state': bool(directional[7] > 0),
        'clear_SSIM_and_ArcFace_hinges_decrease_along_same_direction': bool(directional[8] < 0 and directional[9] < 0),
        'interpretation': 'The original objective has no explicit RGB-mean anchor. At this measured final state, its plain negative-gradient direction increases the measured mean-shift penalty while decreasing clear SSIM/ArcFace hinge penalties. Preservation gradients are active; they were not absent. This supports testing one fixed mean-centering path during optimization, without changing loss weights.',
        'limits': 'A local directional derivative is not a finite-step guarantee, an AdamW trajectory reconstruction, a unique historical cause, or evidence of useful unseen CCTV restoration. Clipping can reintroduce output mean shift; every original delivered-PNG preservation and brightness gate remains required.',
        'VM_component_gradient_queries_already_audited': 100,
        'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'V28_failed_gates_retained': 3, 'mean_control_failed_gates_retained': 5,
        'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False,
    }
    OUT.mkdir()
    with (OUT/'results.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(measurements, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: measurements[k] for k in ['original_objective_gradient_norm', 'improvement_vs_preservation_gradient_cosine', 'directional_derivative_along_negative_original_objective_gradient', 'all50_fresh_VM_parity_maxima']}, indent=2))


if __name__ == '__main__':
    main()

"""Independent blockwise derivative replay and saved-gradient direction contrast."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1_return/outputs'
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_first_step_analysis'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    start = time.monotonic()
    assert not (OUT / 'independent_audit.json').exists()
    plan = read(OUT / 'plan.json')
    result = read(OUT / 'analysis.json')
    assert plan['complete'] and result['complete'] and len(result['arms']) == 12
    assert sha(ROOT / 'scripts/analyze_cctv_dgp_multiscale_first_step_v1.py') == plan['script_sha256']
    p = read(PACKET / 'protocol.json')
    grad = read(RETURNED / 'gradient_preflight.json')
    initial = torch.load(RETURNED / 'initial_candidate.pth', map_location='cpu', weights_only=True)
    matrix = np.load(RETURNED / 'initial_gradients.npy', mmap_mode='r', allow_pickle=False)
    assert sha(RETURNED / 'initial_gradients.npy') == plan['gradient_file_sha256']
    count = 0
    barriers = 0
    max_error = 0.
    unit_comparisons = []
    for arm, stored in zip(p['arms'], result['arms']):
        assert arm['id'] == stored['arm']
        folder = RETURNED / arm['id']
        state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=True)
        delta_parts = [(state['model'][row['name']].numpy().astype(np.float64)-initial[row['name']].numpy().astype(np.float64)).reshape(-1) for row in grad['layout']]
        delta = np.concatenate(delta_parts)
        assert np.isclose(np.linalg.norm(delta), stored['actual_parameter_delta_L2'], rtol=0, atol=1e-12)
        actual = []
        pure = []
        accumulated = np.load(folder / 'accumulated_gradient.npy', allow_pickle=False)
        descent = np.zeros(609219, dtype=np.float64)
        descent[:accumulated.size] = -accumulated.astype(np.float64)
        descent /= np.linalg.norm(descent)
        for batch in range(20):
            values = []
            for signal in range(14):
                pieces = []
                for row, difference in zip(grad['layout'], delta_parts):
                    product = matrix[batch,signal,row['start']:row['end']].astype(np.float64)*difference
                    pieces.append(float(np.sum(product, dtype=np.float64)))
                value = math.fsum(pieces)
                expected = stored['initial_reference_predictions'][batch]['terms'][p['signals'][signal]]
                error = abs(value-expected)
                max_error = max(max_error, error)
                assert error <= 1e-10, (arm['id'], batch, signal, error)
                assert (value>0) == (expected>0) and (value<0) == (expected<0)
                values.append(value)
                count += 1
            actual.append(values)
            pure.append(np.sum(matrix[batch].astype(np.float64)*descent[None,:],axis=1,dtype=np.float64))
        actual = np.asarray(actual)
        pure = np.asarray(pure)
        assert stored['clear_MSE_positive_initial_directions'] == int((actual[:,4]>0).sum())
        assert stored['clear_ArcFace_loss_positive_initial_directions'] == int((actual[:,6]>0).sum())
        step = read(folder / 'step_receipt.json')
        assert len(step['rows']) == 10 and all(row['barrier'] == 0 for row in step['rows'])
        barriers += len(step['rows'])
        if arm['lr'] == 1e-5:
            fitted = [begin//5 for begin in p['pools'][arm['pool']]]
            normalized_actual = actual/stored['actual_parameter_delta_L2']
            unit_comparisons.append(dict(partition=arm['partition'], pool=arm['pool'],
                unit_saved_Adam_change_training_slopes=dict(zip(p['signals'], normalized_actual[fitted].mean(0).tolist())),
                unit_negative_accumulated_gradient_training_slopes=dict(zip(p['signals'], pure[fitted].mean(0).tolist())),
                negative_gradient_clear_MSE_positive_references=int((pure[:,4]>0).sum()),
                negative_gradient_clear_ArcFace_positive_references=int((pure[:,6]>0).sum()),
                arithmetic_only_no_finite_SGD_output_or_optimizer_test=True,
                negative_gradient_excludes_weight_decay=True))
        del state
        assert time.monotonic()-start < 300
    assert count == 3360 and barriers == 120
    answer = dict(complete=True, checker_sha256=sha(Path(__file__)), analysis_sha256=sha(OUT/'analysis.json'),
                  scalar_predictions_replayed=3360, exact_signs_agree=True, maximum_absolute_replay_error=max_error,
                  zero_barriers_reverified=120, normalized_saved_direction_comparisons=unit_comparisons,
                  local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
                  mathematical_optimizer_calls=0, first_order_not_finite_step_quality=True,
                  model_qualification=False, goal_complete=False, seconds=time.monotonic()-start)
    with (OUT / 'independent_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(answer, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, scalar_predictions_replayed=3360, maximum_absolute_replay_error=max_error,
               local_neural_calls=0, local_optimizer_updates=0), flush=True)


if __name__ == '__main__':
    main()

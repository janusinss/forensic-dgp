"""Saved-gradient/actual-step algebra only; no neural calls or fitted weights."""
import hashlib
import json
from pathlib import Path
import sys
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


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    started = time.monotonic()
    assert not OUT.exists()
    audit_path = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['all_gate_decisions_and_failure_names_unchanged']
    p = read(PACKET / 'protocol.json')
    assert sha(PACKET / 'protocol.json') == audit['protocol_sha256']
    grad = read(RETURNED / 'gradient_preflight.json')
    matrix = np.load(RETURNED / 'initial_gradients.npy', mmap_mode='r', allow_pickle=False)
    assert matrix.shape == (20, 14, 609219) and matrix.dtype == np.float32
    initial = torch.load(RETURNED / 'initial_candidate.pth', map_location='cpu', weights_only=True)
    layout = grad['layout']
    names = [row['name'] for row in layout]
    theta = np.concatenate([initial[name].numpy().reshape(-1) for name in names]).astype(np.float64)
    base = read(RETURNED / 'baseline/metrics.json')
    OUT.mkdir()
    write(OUT / 'plan.json', dict(complete=True, protocol_sha256=audit['protocol_sha256'],
          independent_audit_sha256=sha(audit_path), gradient_file_sha256=grad['gradient_sha256'],
          calculation='stored float32 initial parameter derivatives dot actual saved parameter differences',
          all12_arms_all20_references_all14_signals=True, first_order_not_finite_or_epoch_forecast=True,
          scientific_thresholds_unchanged=True, script_sha256=sha(Path(__file__)),
          local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
          numerical_optimizer_calls=0, fitted_local_model_weights=False, goal_complete=False))
    entries = []
    for arm in p['arms']:
        folder = RETURNED / arm['id']
        state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=True)
        current = np.concatenate([state['model'][name].numpy().reshape(-1) for name in names]).astype(np.float64)
        delta = current-theta
        signed = np.empty((20, 14), dtype=np.float64)
        for batch in range(20):
            signed[batch] = np.einsum('nk,k->n', np.asarray(matrix[batch], dtype=np.float64), delta, optimize=False)
        step = read(folder / 'step_receipt.json')
        assert len(step['rows']) == 10 and all(row['barrier'] == 0 for row in step['rows'])
        assert state['completed_updates'] == 1
        metrics = read(folder / 'metrics.json')
        actual = metrics['groups']['raw']
        before = base['groups']['raw']
        training_indices = [begin//5 for begin in p['pools'][arm['pool']]]
        cohort_means = {}
        for source in sorted({case['source'] for case in p['cases']}):
            ii = [i for i in range(20) if p['cases'][i*5]['source'] == source]
            cohort_means[source] = dict(zip(p['signals'], signed[ii].mean(0).tolist()))
        entry = dict(arm=arm['id'], partition=arm['partition'], lr=arm['lr'], pool=arm['pool'],
                     actual_parameter_delta_L2=float(np.linalg.norm(delta)),
                     initial_training_term_predictions=dict(zip(p['signals'], signed[training_indices].mean(0).tolist())),
                     all_reference_term_predictions=dict(zip(p['signals'], signed.mean(0).tolist())),
                     source_initial_term_predictions=cohort_means,
                     initial_reference_predictions=[dict(reference=p['references'][i]['id'],
                        source=p['cases'][i*5]['source'], used_for_this_arm_fitting=i in training_indices,
                        terms=dict(zip(p['signals'], signed[i].tolist()))) for i in range(20)],
                     clear_MSE_positive_initial_directions=int((signed[:,4]>0).sum()),
                     clear_SSIM_loss_positive_initial_directions=int((signed[:,5]>0).sum()),
                     clear_ArcFace_loss_positive_initial_directions=int((signed[:,6]>0).sum()),
                     all_fitting_barriers_exactly_zero=True,
                     actual_clear_raw_MSE_before=before['clear']['MSE'],
                     actual_clear_raw_MSE_after=actual['clear']['MSE'],
                     clear_raw_MSE_multiplier=actual['clear']['MSE']/before['clear']['MSE'],
                     actual_clear_raw_SSIM_change=actual['clear']['SSIM']-before['clear']['SSIM'],
                     actual_clear_raw_ArcFace_change=actual['clear']['ArcFace_observed_fixed']-before['clear']['ArcFace_observed_fixed'],
                     actual_degraded_raw_feature_gain=1-actual['degraded']['landmark_high_frequency_MSE']/before['degraded']['landmark_high_frequency_MSE'])
        entries.append(entry)
        del state
        assert time.monotonic()-started < 300
    result = dict(complete=True, arms=entries, scalar_predictions=12*20*14,
                  training_barrier_values_verified=120, all_initial_preservation_barriers_inactive=True,
                  first_order_predictions_are_not_actual_finite_loss_changes=True,
                  training_SSIM_loss_and_evaluation_SSIM_are_distinct_implementations=True,
                  native_CCTV_has_no_clean_target_metrics=True, native_usefulness_not_established_by_this_algebra=True,
                  app_checkpoint_unchanged=True, local_neural_calls=0, local_gradient_queries=0,
                  local_optimizer_updates=0, numerical_optimizer_calls=0,
                  model_qualification=False, goal_complete=False, seconds=time.monotonic()-started)
    write(OUT / 'analysis.json', result)
    print(dict(complete=True, scalar_predictions=result['scalar_predictions'], inactive_preservation_barriers=120,
               local_neural_calls=0, local_optimizer_updates=0), flush=True)


if __name__ == '__main__':
    main()

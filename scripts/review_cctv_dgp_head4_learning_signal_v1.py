"""Saved-gradient arithmetic and paired TRAIN evidence review; no model calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_head4_learning_signal_review_v1'
GP = 'outputs/cctv_dgp_head4_reactivation_vm_v1/protocol.json'
GR = 'outputs/cctv_dgp_head4_reactivation_vm_v1_return/outputs/results.json'
GV = 'outputs/cctv_dgp_head4_reactivation_vm_v1_return/outputs/individual_gradients.npy'
CP = 'outputs/cctv_dgp_head4_capacity_vm_v1/protocol.json'
CR = 'outputs/cctv_dgp_head4_capacity_vm_v1_return/outputs/'
SOURCES = [GP, GR, GV, CP, CR + 'update0/training_state.pt',
           CR + 'update0/metrics.json', CR + 'update50/metrics.json', CR + 'early_gate.json',
           'outputs/cctv_dgp_head4_capacity_v1_independent_audit.json',
           'outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json',
           'outputs/cctv_dgp_full_training_coverage_v1/results.json',
           'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json',
           'outputs/cctv_dgp_original_head4_trace_v1/plan.json',
           'outputs/cctv_dgp_original_head4_trace_v1/results.json',
           'outputs/cctv_dgp_head4_training_design_review_v1/decay_direction_review.json',
           'scripts/cctv_dgp_head4_capacity_model_v1.py',
           'models/dgp_synthesizer.py', 'models/fpn_mobilenet.py',
           'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',
           'checkpoints/dgp_zamboanga_final.pth']
WEIGHTS = np.array([.2, 1., 1., 1.], np.float64)
NORMS = np.array([.022182490369457405, .30685945008702126,
                  .5253697948823282, .0017099954417771745], np.float64)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare():
    assert not OUT.exists(), 'Preserve prior work'
    OUT.mkdir()
    files = SOURCES + [Path(__file__).relative_to(ROOT).as_posix(),
                       'scripts/audit_cctv_dgp_head4_learning_signal_v1.py']
    write(OUT / 'plan.json', {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Averaged initial-gradient calibration does not cover individual reference-batch directions or finite training.',
        'source_sha256': {n: sha(ROOT / n) for n in files},
        'gradient_shape': [2, 20, 4, 147456], 'weights': WEIGHTS.tolist(), 'normalizers': NORMS.tolist(),
        'closed_form_initial_direction': 'clip(weighted normalized gradient, L2=1); add Adam coupled weight decay 1e-5 times initial parameters; negate g/(abs(g)+1e-8)',
        'direction_rates_for_linear_estimates_only': [1e-5, 1e-4, 1e-3],
        'direction_dot_ambiguity_atol': 1e-12,
        'worker_seconds': 180, 'gradient_queries': 0, 'optimizer_updates': 0,
        'neural_calls': 0, 'parameter_assignments': 0,
        'native_data_mode': 'Existing unpaired input-only metadata; no native target metrics or final identity pixels',
        'training_recipe_or_quality_gate_changed': False, 'goal_complete': False,
    })
    print({'prepared': True, 'plan_sha256': sha(OUT / 'plan.json')}, flush=True)


def run():
    start = time.monotonic()
    assert not (OUT / 'results.json').exists()
    plan = read(OUT / 'plan.json')
    for n, d in plan['source_sha256'].items():
        assert sha(ROOT / n) == d, n
    ca = read(ROOT / SOURCES[8]); ga = read(ROOT / SOURCES[9]); ia = read(ROOT / SOURCES[11])
    assert ca['complete'] and ca['full_raw_PNG_metric_records_verified'] == 7810 and not ca['early_capacity_pass']
    assert ga['complete'] and ga['saved_gradient_vectors_verified'] == 160 and not ga['model_qualification']
    assert ia['complete'] and ia['canonical_TRAIN_targets_verified'] == 781
    gp, cp, gr = read(ROOT / GP), read(ROOT / CP), read(ROOT / GR)
    assert all(c['role'] == 'train' for c in gp['cases'] + cp['cases'])
    assert cp['reconstruction_weights'] == WEIGHTS.tolist() and cp['normalizers'] == NORMS.tolist()
    vectors = np.load(ROOT / GV, mmap_mode='r', allow_pickle=False)
    assert vectors.shape == (2, 20, 4, 147456) and vectors.dtype == np.float64
    assert np.isfinite(vectors).all() and not np.count_nonzero(vectors[0])
    gradients = np.asarray(vectors[1]).copy() / NORMS[None, :, None]
    # Tensor loading only: no model construction, forward, autograd or optimizer.
    import torch
    state = torch.load(ROOT / (CR + 'update0/training_state.pt'), map_location='cpu', weights_only=True)
    assert state['completed_updates'] == 0 and state['original_checkpoint_sha256'] == cp['original_checkpoint_sha256']
    names = ['net.head4.block0.weight', 'net.head4.block1.weight', 'live_fusion4']
    assert state['optimizer_parameter_names'] == names
    theta = np.concatenate([state['model'][n].double().numpy().reshape(-1) for n in names])
    assert theta.shape == (147456,) and np.isfinite(theta).all()
    losses = np.empty((20, 4), np.float64)
    for row in gr['gradient_queries']:
        if row['model'] == 'repaired':
            losses[row['batch'], gp['components'].index(row['component'])] = row['loss']
    losses /= NORMS[None, :]
    labels = []
    for i in range(20):
        cases = gp['cases'][5 * i:5 * (i + 1)]
        assert len({c['source'] for c in cases}) == 1
        labels.append({'batch': i, 'cohort': gp['cohorts'][i // 10]['name'], 'source': cases[0]['source'],
                       'reference': cases[0]['source_person_or_reference'], 'case_ids': [c['id'] for c in cases]})
    scopes = sorted({(r['cohort'], r['source']) for r in labels})
    scope_ids = [[i for i, r in enumerate(labels) if (r['cohort'], r['source']) == scope] for scope in scopes]
    assert all(len(ids) == 5 for ids in scope_ids)
    directions, summaries = [], []
    for name, ids in [('macro_all20', list(range(20)))] + [(f'reference_batch_{i:02d}', [i]) for i in range(20)]:
        g = (gradients[ids].mean(0) * WEIGHTS[:, None]).sum(0)
        norm = float(np.linalg.norm(g)); g *= min(1., 1. / (norm + 1e-6))
        g += 1e-5 * theta
        direction = -g / (np.abs(g) + 1e-8)
        directions.append(direction)
        summaries.append({'name': name, 'saved_gradient_batches': ids, 'weighted_gradient_L2_before_clip': norm})
    # Full initial-state reference/component response matrix, not a trajectory.
    dots = (gradients.reshape(80, -1) @ np.stack(directions).T).reshape(20, 4, 21).transpose(2, 0, 1)
    group_dots = np.stack([dots[:, ids].mean(1) for ids in scope_ids], axis=1)
    write(OUT / 'directions.json', {
        'complete': True, 'labels': labels, 'components': gp['components'], 'directions': summaries,
        'reference_component_dots': dots.tolist(),
        'cohort_source_groups': [{'cohort': a, 'source': b, 'batches': ids} for (a, b), ids in zip(scopes, scope_ids)],
        'group_component_dots': group_dots.tolist(),
        'initial_linear_only': True, 'optimizer_updates': 0, 'parameter_assignments': 0,
    })
    # Independent previously measured input/target errors, joined by fixed TRAIN ID.
    coverage = read(ROOT / SOURCES[10]); measured = {r['id']: r for r in coverage['TRAIN_rows']}
    s0 = read(ROOT / (CR + 'update0/metrics.json')); s50 = read(ROOT / (CR + 'update50/metrics.json'))
    b0, b50 = {r['id']: r for r in s0['rows']}, {r['id']: r for r in s50['rows']}
    assert set(measured) == set(b0) == set(b50) == {c['id'] for c in cp['cases']}
    input_model = []
    for source, profile in sorted({(c['source'], c['profile']) for c in cp['cases']}):
        ids = [c['id'] for c in cp['cases'] if (c['source'], c['profile']) == (source, profile)]
        for stage in ['raw', 'png']:
            input_hf = np.array([measured[c]['input_landmark_HF_MSE_to_target'] for c in ids])
            initial = np.array([b0[c][stage]['landmark_high_frequency_MSE'] for c in ids])
            current = np.array([b50[c][stage]['landmark_high_frequency_MSE'] for c in ids])
            input_model.append({'source': source, 'profile': profile, 'stage': stage, 'cases': len(ids),
                                'input_HF_MSE_mean': float(input_hf.mean()), 'original_HF_MSE_mean': float(initial.mean()),
                                'update50_HF_MSE_mean': float(current.mean()),
                                'original_better_than_input_cases': int(np.sum(initial < input_hf)),
                                'update50_better_than_original_cases': int(np.sum(current < initial)),
                                'initial_vs_input_HF_gain': None if not input_hf.mean() else float(1 - initial.mean() / input_hf.mean())})
    trace = read(ROOT / SOURCES[13])
    shapes = {head: sorted({tuple(row['heads'][head]['shape']) for row in trace['rows']}) for head in ['1', '2', '3', '4']}
    assert shapes['4'] == [(64, 8, 8)] and shapes['1'] == [(64, 64, 64)]
    predictions = []
    for scope, ids, dot in zip(scopes, scope_ids, group_dots[0]):
        baseline = float(losses[ids, 3].mean())
        predictions.append({'cohort': scope[0], 'source': scope[1], 'initial_normalized_HF_loss': baseline,
                            'macro_initial_HF_slope': float(dot[3]),
                            'first_step_linear_structure_gain_percent': {str(lr): float(-100 * lr * dot[3] / baseline) for lr in plan['direction_rates_for_linear_estimates_only']}})
    result = {
        'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'directions_sha256': sha(OUT / 'directions.json'),
        'macro_all_four_components_descend_in_all_four_groups': bool(np.all(group_dots[0] < -1e-12)),
        'individual_directions_with_group_HF_ascent': int(np.sum(np.any(group_dots[1:, :, 3] > 1e-12, axis=1))),
        'individual_directions_with_any_group_component_ascent': int(np.sum(np.any(group_dots[1:] > 1e-12, axis=(1, 2)))),
        'individual_directions_with_own_batch_HF_ascent': int(sum(dots[i + 1, i, 3] > 1e-12 for i in range(20))),
        'individual_directions_with_other_reference_HF_ascent': int(sum(np.any(np.delete(dots[i + 1, :, 3], i) > 1e-12) for i in range(20))),
        'initial_linear_magnitude_estimates': predictions, 'full_TRAIN_input_vs_model_HF': input_model,
        'observed_original_head_shapes': {k: [list(v) for v in val] for k, val in shapes.items()},
        'native_unpaired_metadata_cases': len(coverage['native_input_rows']),
        'no_native_clean_reference_metrics': True, 'no_reserved_final_pixels': True,
        'input_HF_and_model_HF_same_support_kernel_but_input_has_float64_RGB_conversion': True,
        'unique_root_cause_or_longer_epoch_outcome_proven': False,
        'finite_step_predictions_or_preservation_certificates': False,
        'recipe_changed': False, 'new_model_checkpoint': False, 'model_qualification': False, 'app_promotion': False,
        'gradient_queries': 0, 'optimizer_updates': 0, 'neural_calls': 0, 'parameter_assignments': 0, 'goal_complete': False,
        'seconds': time.monotonic() - start,
    }
    assert result['seconds'] < plan['worker_seconds']
    for n, d in plan['source_sha256'].items():
        assert sha(ROOT / n) == d, n
    write(OUT / 'results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['full_TRAIN_input_vs_model_HF', 'initial_linear_magnitude_estimates']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()

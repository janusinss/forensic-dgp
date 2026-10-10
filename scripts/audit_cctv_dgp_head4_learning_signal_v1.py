"""Independent scalar-product audit of saved-gradient calibration evidence."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_head4_learning_signal_review_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    start = time.monotonic()
    receipt = OUT / 'independent_audit.json'
    assert not receipt.exists()
    plan, r, d = [read(OUT / n) for n in ['plan.json', 'results.json', 'directions.json']]
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert r['plan_sha256'] == sha(OUT / 'plan.json') and r['directions_sha256'] == sha(OUT / 'directions.json')
    assert r['complete'] and d['complete']
    for k in ['gradient_queries', 'optimizer_updates', 'neural_calls', 'parameter_assignments']:
        assert plan[k] == r[k] == 0
    assert not r['recipe_changed'] and not r['model_qualification'] and not r['goal_complete']
    gradient_path = ROOT / 'outputs/cctv_dgp_head4_reactivation_vm_v1_return/outputs/individual_gradients.npy'
    gradients = np.load(gradient_path, mmap_mode='r', allow_pickle=False)
    assert gradients.dtype == np.float64 and gradients.shape == (2, 20, 4, 147456)
    gp = read(ROOT / 'outputs/cctv_dgp_head4_reactivation_vm_v1/protocol.json')
    assert d['components'] == gp['components'] and len(d['labels']) == 20
    for i, label in enumerate(d['labels']):
        cases = gp['cases'][i * 5:(i + 1) * 5]
        assert label['case_ids'] == [c['id'] for c in cases]
        assert label['cohort'] == gp['cohorts'][i // 10]['name'] and label['source'] == cases[0]['source']
    # CPU tensor loading only; no neural calls, gradient queries or optimizer.
    import torch
    state = torch.load(ROOT / 'outputs/cctv_dgp_head4_capacity_vm_v1_return/outputs/update0/training_state.pt',
                       map_location='cpu', weights_only=True)
    names = ['net.head4.block0.weight', 'net.head4.block1.weight', 'live_fusion4']
    theta = np.concatenate([state['model'][n].double().numpy().ravel() for n in names])
    weights, norms = plan['weights'], plan['normalizers']
    stored = np.array(d['reference_component_dots'], np.float64)
    assert stored.shape == (21, 20, 4)
    fresh = np.empty_like(stored)
    max_error = 0.
    for direction_index in range(21):
        ids = list(range(20)) if direction_index == 0 else [direction_index - 1]
        assert d['directions'][direction_index]['saved_gradient_batches'] == ids
        combined = np.zeros(147456, np.float64)
        for index in ids:
            for component in range(4):
                combined += gradients[1, index, component] * (weights[component] / norms[component] / len(ids))
        norm = math.sqrt(float(np.sum(combined * combined, dtype=np.float64)))
        assert abs(norm - d['directions'][direction_index]['weighted_gradient_L2_before_clip']) < 1e-12
        combined *= min(1., 1. / (norm + 1e-6))
        adjusted = combined + theta * 1e-5
        direction = np.divide(-adjusted, np.abs(adjusted) + 1e-8)
        for target_batch in range(20):
            for component in range(4):
                value = float(np.sum((gradients[1, target_batch, component] / norms[component]) * direction, dtype=np.float64))
                fresh[direction_index, target_batch, component] = value
                max_error = max(max_error, abs(value - stored[direction_index, target_batch, component]))
    assert max_error < 1e-10
    eps = plan['direction_dot_ambiguity_atol']
    assert np.array_equal(fresh > eps, stored > eps) and np.array_equal(fresh < -eps, stored < -eps)
    groups = d['cohort_source_groups']
    fresh_groups = np.stack([fresh[:, g['batches']].mean(1) for g in groups], axis=1)
    recorded_groups = np.array(d['group_component_dots'], np.float64)
    assert np.allclose(fresh_groups, recorded_groups, atol=1e-10, rtol=0)
    assert np.array_equal(fresh_groups > eps, recorded_groups > eps)
    assert np.array_equal(fresh_groups < -eps, recorded_groups < -eps)
    assert r['macro_all_four_components_descend_in_all_four_groups'] == bool(np.all(fresh_groups[0] < -eps))
    assert r['individual_directions_with_group_HF_ascent'] == int(np.any(fresh_groups[1:, :, 3] > eps, axis=1).sum())
    assert r['individual_directions_with_any_group_component_ascent'] == int(np.any(fresh_groups[1:] > eps, axis=(1, 2)).sum())
    assert r['individual_directions_with_own_batch_HF_ascent'] == sum(fresh[i + 1, i, 3] > eps for i in range(20))
    assert r['individual_directions_with_other_reference_HF_ascent'] == sum(np.any(np.delete(fresh[i + 1, :, 3], i) > eps) for i in range(20))
    coverage = read(ROOT / 'outputs/cctv_dgp_full_training_coverage_v1/results.json')
    inputs = {v['id']: v for v in coverage['TRAIN_rows']}
    before = read(ROOT / 'outputs/cctv_dgp_head4_capacity_vm_v1_return/outputs/update0/metrics.json')
    after = read(ROOT / 'outputs/cctv_dgp_head4_capacity_vm_v1_return/outputs/update50/metrics.json')
    bm, am = {v['id']: v for v in before['rows']}, {v['id']: v for v in after['rows']}
    assert set(inputs) == set(bm) == set(am) and len(inputs) == 3905
    assert len(r['full_TRAIN_input_vs_model_HF']) == 20
    for group in r['full_TRAIN_input_vs_model_HF']:
        ids = [cid for cid, row in inputs.items() if row['source'] == group['source'] and row['profile'] == group['profile']]
        assert len(ids) == group['cases']
        stage = group['stage']
        values = [[inputs[c]['input_landmark_HF_MSE_to_target'] for c in ids],
                  [bm[c][stage]['landmark_high_frequency_MSE'] for c in ids],
                  [am[c][stage]['landmark_high_frequency_MSE'] for c in ids]]
        means = [math.fsum(v) / len(v) for v in values]
        for key, mean in zip(['input_HF_MSE_mean', 'original_HF_MSE_mean', 'update50_HF_MSE_mean'], means):
            assert abs(group[key] - mean) < 1e-14
        assert group['original_better_than_input_cases'] == sum(x < y for x, y in zip(values[1], values[0]))
        assert group['update50_better_than_original_cases'] == sum(x < y for x, y in zip(values[2], values[1]))
        if not means[0]:
            assert group['initial_vs_input_HF_gain'] is None
        else:
            assert abs(group['initial_vs_input_HF_gain'] - (1 - means[1] / means[0])) < 1e-12
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    result = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'results_sha256': sha(OUT / 'results.json'),
              'independent_dot_products_verified': int(fresh.size), 'maximum_dot_recomputation_error': max_error,
              'all_dot_sign_and_summary_decisions_unchanged': True, 'full_TRAIN_rows_joined': 3905,
              'input_model_source_profile_stage_aggregates_verified': 20,
              'gradient_queries': 0, 'optimizer_updates': 0, 'neural_calls': 0, 'parameter_assignments': 0,
              'quality_gate_changed': False, 'model_qualification': False, 'goal_complete': False,
              'seconds': time.monotonic() - start}
    with receipt.open('x', encoding='utf-8') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

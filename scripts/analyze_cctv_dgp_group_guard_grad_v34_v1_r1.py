"""Audit-bound group geometry from saved VM arrays; no local neural work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_return'
OLD = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_return'
BASIS = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return'
OUT = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    started = time.monotonic()
    import numpy as np
    from cctv_dgp_group_guard_cone_v35 import project_vectors

    audit_path = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['diagnostic_complete'] and not audit['failure_retained']
    assert audit['local_gradient_calls'] == audit['local_optimizer_updates'] == 0
    assert audit['pinned_baseline_CPU_replay']['outputs'] == 280
    p = read(RETURN / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == audit['protocol_sha256']
    previous_p = read(OLD / 'protocol.json')
    previous_import = read(ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_return_import.json')
    imported_path = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_return_import.json'
    imported = read(imported_path)
    assert imported['complete'] and imported['archive_sha256'] == audit['archive_sha256']
    helper = module('pinned_group_index_arithmetic', ROOT / 'scripts/audit_cctv_dgp_group_guard_grad_v34_return.py')
    assert not OUT.exists(), 'Preserve preceding analysis attempts'
    OUT.mkdir()
    bindings = {q.relative_to(ROOT).as_posix(): sha(q) for q in [audit_path, imported_path,
        ROOT / 'scripts/cctv_dgp_group_guard_cone_v35.py', Path(__file__)]}
    grouped, labels, summaries, restoration_rows, proposals = [], [], [], [], []
    theta = None
    for cohort in p['cohorts']:
        label = cohort['name']
        folder = RETURN / 'outputs' / label
        cases = cohort['cases']
        indices = helper.group_indices(cases)
        vectors = {key: np.zeros((3, 978243), np.float64) for key in indices}
        receipt = read(folder / 'receipt.json')
        assert len(receipt['batches']) == 10
        for batch_index, batch in enumerate(receipt['batches']):
            path = folder / f'batch{batch_index}_guard_gradients.npy'
            assert sha(path) == imported['files_sha256'][path.relative_to(RETURN).as_posix()] == batch['gradient_sha256']
            bindings[path.relative_to(ROOT).as_posix()] = sha(path)
            g = np.load(path, allow_pickle=False)
            assert g.shape == (15, 978243) and g.dtype == np.float32 and np.isfinite(g).all()
            for slot in range(5):
                index = 5 * batch_index + slot
                assert batch['rows'][slot]['id'] == cases[index]['id']
                for key, members in indices.items():
                    if index in members:
                        vectors[key] += g[3*slot:3*slot+3].astype(np.float64)
            del g
        matrix = np.concatenate([vectors[key] / len(indices[key]) for key in indices], axis=0)
        del vectors
        previous = next(s for s in audit['gradient_summaries'] if s['cohort'] == label)
        cohort_labels = [{'cohort': label, **value} for value in previous['labels']]
        assert [r['group'] for r in previous['labels']] == [key for key in indices for _ in range(3)]
        assert np.allclose(np.linalg.norm(matrix, axis=1), previous['group_gradient_norms'], rtol=2e-10, atol=1e-11)
        assert np.allclose(matrix @ matrix.T, previous['group_gradient_gram'], rtol=2e-10, atol=1e-11)
        before = np.load(OLD / f'outputs/state0_{label}/theta_before.npy', allow_pickle=False)
        after = np.load(OLD / f'outputs/state0_{label}/restoration/theta_after.npy', allow_pickle=False)
        for name in [f'outputs/state0_{label}/theta_before.npy', f'outputs/state0_{label}/restoration/theta_after.npy']:
            assert sha(OLD / name) == previous_import['files_sha256'][name]
        if theta is None:
            theta = before.copy()
        else:
            assert np.array_equal(theta, before)
        proposal = before.astype(np.float64) - after.astype(np.float64)
        predicted = -(matrix @ proposal)
        assert np.allclose(predicted, previous['predicted_changes_under_original_V33_restoration_proposal'], rtol=2e-10, atol=1e-11)
        summary = {'cohort': label, 'nonzero_group_guard_derivatives': int((np.linalg.norm(matrix, axis=1) > 0).sum()),
            'proposal_norm': float(np.linalg.norm(proposal)),
            'predicted_group_changes': [{'group': row['group'], 'metric': row['metric'], 'change': float(value)}
                for row, value in zip(cohort_labels, predicted)],
            'aggregate_cancellation': [{'metric': metric,
                'aggregate_predicted_change': float(predicted[next(i for i,r in enumerate(cohort_labels) if r['group'] == 'all' and r['metric'] == metric)]),
                'positive_subgroup_changes': int(sum(value > 0 for row,value in zip(cohort_labels,predicted)
                    if row['group'] != 'all' and row['metric'] == metric))} for metric in p['guard_metrics']]}
        gradients_path = BASIS / f'outputs/state0_{label}/gradient_components.npy'
        assert sha(gradients_path) == previous_p['basis_VM_sha256'][f'outputs/state0_{label}/gradient_components.npy']
        old_gradients = np.load(gradients_path, allow_pickle=False)
        assert old_gradients.shape == (7, 978243) and old_gradients.dtype == np.float64
        assert np.count_nonzero(old_gradients[3:]) == 0, 'Only original-state zero hinges may be omitted'
        restoration_rows.append(old_gradients[:3].copy())
        for path in [gradients_path, OLD / f'outputs/state0_{label}/theta_before.npy',
                OLD / f'outputs/state0_{label}/restoration/theta_after.npy']:
            bindings[path.relative_to(ROOT).as_posix()] = sha(path)
        proposals.append(proposal)
        grouped.append(matrix)
        labels.extend(cohort_labels)
        summaries.append(summary)
        print(json.dumps({'assembled_cohort': label, 'group_guards': 51}), flush=True)
    # One symmetric proposal is frozen here: the mean of the two already audited
    # fresh AdamW displacements. This is NOT AdamW applied to a mean gradient.
    all_rows = np.concatenate(grouped + restoration_rows, axis=0)
    del grouped
    labels.extend({'cohort': c['name'], 'group': 'existing_restoration_loss', 'metric': term}
        for c in p['cohorts'] for term in previous_p['terms'][:3])
    assert all_rows.shape == (108, 978243) and len(labels) == 108
    proposal = .5 * (proposals[0] + proposals[1])
    direction, proof = project_vectors(all_rows, proposal)
    tolerance = proof['KKT_tolerance']
    assert proof['constraint_count'] == 108 and (all_rows @ direction >= -tolerance * np.linalg.norm(all_rows, axis=1)).all()
    scales = []
    for scale in [1., .5, .25, .125]:
        actual_theta = (theta.astype(np.float64) - scale * direction).astype(np.float32)
        actual_displacement = theta.astype(np.float64) - actual_theta.astype(np.float64)
        changes = -(all_rows @ actual_displacement)
        scales.append({'scale': scale, 'float32_displacement_roundoff_norm': float(np.linalg.norm(actual_displacement-scale*direction)),
            'linear_guard_changes': changes[:102].tolist(), 'linear_existing_loss_changes': changes[102:].tolist(),
            'positive_float32_guard_derivatives': int((changes[:102] > 0).sum()),
            'finite_outputs_generated': False})
    np.save(OUT / 'projected_displacement.npy', direction, allow_pickle=False)
    np.save(OUT / 'mean_original_displacement.npy', proposal, allow_pickle=False)
    np.save(OUT / 'theta_before.npy', theta, allow_pickle=False)
    np.save(OUT / 'group_guard_matrix.npy', all_rows[:102], allow_pickle=False)
    write(OUT / 'projection.json', proof)
    result = {'complete': True, 'protocol_sha256': audit['protocol_sha256'],
        'independent_audit_sha256': sha(audit_path), 'cohort_summaries': summaries,
        'constraint_labels': labels, 'all108_nonzero_constraints_retained': True,
        'proposal_policy': 'Arithmetic mean of the two audited original-state V33 fresh AdamW displacements; not an AdamW step on the mean gradient',
        'joint_projection': proof, 'fixed_float32_scales': scales,
        'both_subsets_now_used_for_training_design': True,
        'unexposed_label_only_refers_to_first50_V32_updates': True,
        'paired_photographic_TRAIN_only': True, 'native_or_reserved_used': False,
        'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
        'finite_preservation_or_quality_implied': False, 'training_capacity_pass': False,
        'app_promotion': False, 'goal_complete': False, 'bindings_sha256': bindings,
        'saved_arrays_sha256': {q.name: sha(q) for q in sorted(OUT.glob('*.npy'))},
        'seconds': time.monotonic()-started}
    assert result['seconds'] < 300, 'Retain finite mathematical analysis limit'
    write(OUT / 'analysis.json', result)
    print(json.dumps({'complete': True, 'constraints': 108, 'projected_norm': proof['projected_norm'],
        'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()

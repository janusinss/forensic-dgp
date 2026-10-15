"""Interpret audited first-order directions; never construct or update a model."""
from datetime import datetime, timezone
import numpy as np
import torch

from import_cctv_dgp_bank_loss_alignment_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def summarize(rows, label):
    result = {}
    for state in ['initial', 'joint50']:
        selected = [r for r in rows if r['state'] == state]
        gram = np.array([r['groups'][label]['Gram'] for r in selected])
        dots = np.array([r['groups'][label]['Adam_direction_dot_components'] for r in selected])
        norms = np.sqrt(np.maximum(np.diagonal(gram, axis1=1, axis2=2), 0))
        direction_norm = np.sqrt([r['groups'][label]['direction_squared_norm'] for r in selected])
        cos = np.divide(dots, norms*direction_norm[:, None], out=np.zeros_like(dots), where=norms*direction_norm[:, None] > 0)
        guard_mse_cos = np.divide(gram[:, 6, 8], norms[:, 6]*norms[:, 8], out=np.zeros(20), where=norms[:, 6]*norms[:, 8] > 0)
        ratio = np.divide(norms[:, 6], norms[:, 3], out=np.zeros(20), where=norms[:, 3] > 0)
        result[state] = dict(reference_rows=20,
            scalar_zero_brightness_penalty_with_nonzero_gradient=sum(r['component_values'][6] == 0 and r['groups'][label]['Gram'][6][6] > 0 for r in selected),
            mean_weighted_component_values=np.mean([r['component_values'] for r in selected], axis=0).tolist(),
            median_component_gradient_norms=np.median(norms, axis=0).tolist(),
            median_brightness_gradient_norm_over_detail_gradient_norm=float(np.median(ratio)),
            median_guard_vs_degraded_MSE_gradient_cosine=float(np.median(guard_mse_cos)),
            nominal_single_reference_Adam_direction_objective_ascents=int((dots[:, :7].sum(1) > 0).sum()),
            nominal_single_reference_Adam_direction_detail_ascents=int((dots[:, 3] > 0).sum()),
            nominal_single_reference_Adam_direction_clear_MSE_ascents=int((dots[:, 7] > 0).sum()),
            nominal_single_reference_Adam_direction_degraded_MSE_ascents=int((dots[:, 8] > 0).sum()),
            median_nominal_Adam_detail_cosine=float(np.median(cos[:, 3])),
            negative_total_gradient_degraded_MSE_ascents=int((-gram[:, 8, :7].sum(1) > 0).sum()),
            negative_gradient_without_guard_degraded_MSE_ascents=int((-gram[:, 8, :6].sum(1) > 0).sum()),
            negative_total_gradient_clear_MSE_ascents=int((-gram[:, 7, :7].sum(1) > 0).sum()),
            negative_gradient_without_guard_clear_MSE_ascents=int((-gram[:, 7, :6].sum(1) > 0).sum()),
            counterfactual_gradient_sum_is_not_Adam_or_a_finite_optimizer_update=True)
    return result


def main():
    receipt = ANALYSIS / 'loss_direction_review.json'
    assert not receipt.exists()
    audit = read(ANALYSIS / 'independent_audit.json')
    assert audit['complete'] and audit['sampled_full_gradient_batches_checked'] == 4
    run_path = RETURN / 'outputs/loss_alignment_v1/run_results.json'
    run = read(run_path)
    assert sha(run_path) == audit['run_results_sha256']
    rows = run['rows']
    summaries = {label: summarize(rows, label) for label in rows[0]['groups']}
    assert summaries['all']['initial']['scalar_zero_brightness_penalty_with_nonzero_gradient'] == 20
    assert summaries['all']['initial']['nominal_single_reference_Adam_direction_degraded_MSE_ascents'] == 20
    # Existing sampled arrays permit analytical direction counterfactuals only.
    # At exact initial output parity, a strict inactive-at-zero hinge would have
    # zero guard derivative. No inference or new derivative is evaluated locally.
    torch.set_grad_enabled(False)
    initial = torch.load(PARENT / 'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    names = run['learning_names']
    counterfactuals = []
    for row in rows:
        if row['state'] != 'initial' or not row['sampled_gradient_file']:
            continue
        with np.load(RETURN / 'outputs/loss_alignment_v1' / row['sampled_gradient_file'], allow_pickle=False) as archive:
            gradients = [[archive[f'g{j:02d}_p{i:02d}'] for i in range(37)] for j in range(9)]
        total = [sum(g[i] for g in gradients[:6]) for i in range(37)]
        norm = float(np.sqrt(sum(np.square(g.astype(np.float64)).sum() for g in total)))
        clip = np.float32(min(1, 1/(norm+1e-6)))
        directions = []
        for i, g in enumerate(total):
            effective = g*clip+initial[names[i]].numpy()*np.float32(1e-5)
            m, v = np.float32(.1)*effective, np.float32(.001)*np.square(effective)
            directions.append(-(m/np.float32(.1))/(np.sqrt(v/np.float32(.001))+np.float32(1e-8)))
        dots = [sum(float((g[i].astype(np.float64)*directions[i].astype(np.float64)).sum()) for i in range(37)) for g in gradients]
        counterfactuals.append(dict(begin=row['begin'], source=row['source'],
            original_nominal_dots=row['groups']['all']['Adam_direction_dot_components'],
            hypothetical_initial_zero_guard_dots=dots,
            corrected_objective_dot=sum(dots[:6]), gradient_norm_without_guard=norm,
            zero_guard_scalar_preserved=True, no_parameter_displacement=True,
            not_an_actual_or_accumulated_batch_update=True))
    assert len(counterfactuals) == 2
    for name, expected in read(ANALYSIS / 'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT / name) == expected
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), reviewer_sha256=sha(__file__),
        independent_audit_sha256=sha(ANALYSIS / 'independent_audit.json'), run_results_sha256=sha(run_path),
        component_names=run['component_names'], group_summaries=summaries,
        two_sampled_initial_analytical_zero_guard_counterfactuals=counterfactuals,
        demonstrated_false_assumption='A zero brightness hinge value implies no brightness-loss gradient',
        primary_source='https://github.com/pytorch/pytorch/blob/v2.9.1/tools/autograd/derivatives.yaml#L407-L408',
        clamp_min_equality_gradient_is_one_in_recorded_Torch_version=True,
        proposed_next_change='Inactive-at-zero brightness hinge derivative, preserving scalar penalty and scientific thresholds',
        remaining_anchor_objective_conflicts_not_resolved_by_guard_boundary_evidence=True,
        no_guaranteed_quality_or_more_epoch_justification=True,
        all40_derivatives_not_freshly_replayed=True, four_saved_batches_arithmetic_audited=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        no_native_quality_or_completion_claim=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, zero_penalty_active_gradient_rows=20,
               nominal_degraded_MSE_ascents_initial=20, scalar_boundary_correction_VM_test_needed=True))


if __name__ == '__main__':
    main()

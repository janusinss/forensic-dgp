"""Review saved derivatives and loss scaling without a neural or optimizer call."""
import ast
from datetime import datetime, timezone
import numpy as np
import torch

from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def summary(rows, group):
    result = {}
    for state in ['initial', 'joint50']:
        selected = [r for r in rows if r['state'] == state]
        gram = np.asarray([r['groups'][group]['Gram'] for r in selected], dtype=np.float64)
        dots = np.asarray([r['groups'][group]['Adam_direction_dot_components'] for r in selected], dtype=np.float64)
        variants = {
            'retained': np.eye(9)[:7].sum(0),
            'omit_clear_anchor': np.eye(9)[:7].sum(0)-np.eye(9)[7],
            'omit_all_absolute_anchors': np.eye(9)[:7].sum(0)-np.eye(9)[4],
            'reconstruction_only': np.eye(9)[:4].sum(0),
        }
        counterfactual = {}
        for name, coefficients in variants.items():
            products = -np.einsum('bij,j->bi', gram, coefficients)
            counterfactual[name] = {label: int((products[:, j] > 0).sum())
                for j, label in [(3, 'detail_ascents'), (7, 'clear_MSE_ascents'), (8, 'degraded_MSE_ascents')]}
        anchor = -gram[:, :, 4]
        clear = -gram[:, :, 7]
        blur = -(gram[:, :, 4]-gram[:, :, 7])
        result[state] = dict(reference_rows=len(selected),
            zero_penalty_active_guard_rows=sum(r['component_values'][6] == 0 and r['groups'][group]['Gram'][6][6] > 0 for r in selected),
            nonzero_guard_gradient_rows=int(np.count_nonzero(gram[:, 6, 6])),
            nominal_single_reference_Adam_objective_ascents=int((dots[:, :7].sum(1) > 0).sum()),
            nominal_single_reference_Adam_detail_ascents=int((dots[:, 3] > 0).sum()),
            nominal_single_reference_Adam_clear_MSE_ascents=int((dots[:, 7] > 0).sum()),
            nominal_single_reference_Adam_degraded_MSE_ascents=int((dots[:, 8] > 0).sum()),
            negative_gradient_analytical_ablation_ascents=counterfactual,
            all_anchor_negative_gradient_degraded_MSE_ascents=int((anchor[:, 8] > 0).sum()),
            clear_anchor_negative_gradient_degraded_MSE_ascents=int((clear[:, 8] > 0).sum()),
            blur_anchor_negative_gradient_degraded_MSE_ascents=int((blur[:, 8] > 0).sum()),
            median_negative_component_gradient_degraded_MSE_products=np.median(-gram[:, 8, :7], axis=0).tolist(),
            plain_negative_gradient_ablation_is_not_Adam_or_a_finite_step=True)
    return result


def initial_adam(gradients, coefficients, parameters):
    total = [sum(np.float32(c)*gradients[j][i] for j, c in enumerate(coefficients) if c)
             for i in range(37)]
    norm = float(np.sqrt(sum(np.square(g.astype(np.float64)).sum() for g in total)))
    clip = np.float32(min(1, 1/(norm+1e-6)))
    direction = []
    for g, value in zip(total, parameters):
        effective = g*clip+value*np.float32(1e-5)
        moment, variance = np.float32(.1)*effective, np.float32(.001)*np.square(effective)
        direction.append(-(moment/np.float32(.1))/(np.sqrt(variance/np.float32(.001))+np.float32(1e-8)))
    products = [sum(float((a.astype(np.float64)*b.astype(np.float64)).sum())
                    for a, b in zip(values, direction)) for values in gradients]
    return dict(component_direction_products=products, gradient_norm=norm,
                coefficient_vector=coefficients.tolist(), no_parameter_displacement=True,
                not_an_actual_or_accumulated_batch_update=True)


def main():
    receipt = ANALYSIS / 'loss_direction_review.json'
    assert not receipt.exists()
    audit = read(ANALYSIS / 'independent_audit.json')
    assert audit['complete'] and audit['sampled_full_gradient_batches_checked'] == 4
    run_path = RETURN / 'outputs/loss_alignment_v1/run_results.json'
    assert sha(run_path) == audit['run_results_sha256']
    run = read(run_path)
    old_root = OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_return'
    old_analysis = OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_analysis'
    old_path = old_root / 'outputs/loss_alignment_v1/run_results.json'
    old_audit = read(old_analysis / 'independent_audit.json')
    assert old_audit['complete'] and sha(old_path) == old_audit['run_results_sha256']
    old_run = read(old_path)
    rows = run['rows']
    cross = []
    keep = [j for j in range(9) if j != 6]
    # ArcFace uses CUDA grid_sample; the active barrier also contains ArcFace.
    # Record cross-run variation rather than assert unproven derivative parity.
    pure = [0, 1, 3, 4, 7, 8]
    for before, after in zip(old_run['rows'], rows, strict=True):
        assert (before['state'], before['begin'], before['case_ids']) == (after['state'], after['begin'], after['case_ids'])
        assert before['component_values'] == after['component_values'], 'Scalar values changed'
        maximum, normalized, changed = 0., 0., 0
        for group in before['groups']:
            a = np.asarray(before['groups'][group]['Gram'])[np.ix_(keep, keep)]
            b = np.asarray(after['groups'][group]['Gram'])[np.ix_(keep, keep)]
            maximum = max(maximum, float(np.abs(a-b).max()))
            denominator = np.sqrt(np.outer(np.diag(a), np.diag(a)))
            scaled = np.divide(np.abs(a-b), denominator, out=np.zeros_like(a), where=denominator > 0)
            normalized = max(normalized, float(scaled.max()))
            changed += int(not np.array_equal(a, b))
            pa = np.asarray(before['groups'][group]['Gram'])[np.ix_(pure, pure)]
            pb = np.asarray(after['groups'][group]['Gram'])[np.ix_(pure, pure)]
            assert np.array_equal(pa, pb), 'Pure reconstruction/anchor Gram changed'
        cross.append(dict(state=after['state'], begin=after['begin'], scalar_values_exact=True,
            maximum_nonbrightness_Gram_absolute_difference=maximum,
            maximum_difference_over_component_norm_product=normalized,
            differing_nonbrightness_partition_matrices=changed,
            excluding_ArcFace_and_ArcFace_barrier_Gram_bit_exact=True))
    summaries = {group: summary(rows, group) for group in rows[0]['groups']}
    assert summaries['all']['initial']['zero_penalty_active_guard_rows'] == 0
    assert summaries['all']['initial']['nonzero_guard_gradient_rows'] == 0
    constants = ast.parse((PARENT / 'cctv_dgp_bank_comparison_v1_contract.py').read_text(encoding='utf-8'))
    assignments = {n.targets[0].id:n.value for n in constants.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    normalizers, anchors = [ast.literal_eval(assignments[name]) for name in ['NORMALIZERS', 'ANCHORS']]
    parent = read(PARENT / 'protocol.json')
    pixel_per_profile = parent['reconstruction_weights'][0]/(5*normalizers[0])
    scales = dict(weighted_MSE_per_profile_coefficient=pixel_per_profile,
        clear_absolute_anchor_coefficient=1/anchors[0], blur_absolute_anchor_coefficient=1/anchors[1],
        clear_anchor_to_weighted_MSE_per_profile_ratio=(1/anchors[0])/pixel_per_profile,
        blur_anchor_to_weighted_MSE_per_profile_ratio=(1/anchors[1])/pixel_per_profile)
    torch.set_grad_enabled(False)
    initial = torch.load(PARENT / 'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    parameters = [initial[name].numpy() for name in run['learning_names']]
    sampled, array_cross = [], []
    for row in rows:
        if not row['sampled_gradient_file']:
            continue
        with np.load(RETURN / 'outputs/loss_alignment_v1' / row['sampled_gradient_file'], allow_pickle=False) as archive, \
             np.load(old_root / 'outputs/loss_alignment_v1' / row['sampled_gradient_file'], allow_pickle=False) as prior:
            gradients = [[archive[f'g{j:02d}_p{i:02d}'] for i in range(37)] for j in range(9)]
            component_cross = []
            for j, values in enumerate(gradients):
                maximum, squared_error, squared_norm, changed = 0., 0., 0., 0
                for i, value in enumerate(values):
                    previous = prior[f'g{j:02d}_p{i:02d}']
                    delta = previous.astype(np.float64)-value
                    maximum = max(maximum, float(np.abs(delta).max()))
                    squared_error += float(np.square(delta).sum())
                    squared_norm += float(np.square(previous.astype(np.float64)).sum())
                    changed += int(np.count_nonzero(previous != value))
                    if j in pure:
                        assert np.array_equal(previous, value), 'Pure sampled gradient changed'
                component_cross.append(dict(component=j, different_values=changed, maximum_absolute_difference=maximum,
                    relative_L2_difference=float(np.sqrt(squared_error/squared_norm)) if squared_norm else None))
            array_cross.append(dict(state=row['state'], begin=row['begin'], components=component_cross))
        if row['state'] != 'initial':
            del gradients
            continue
        variants = {}
        for label, coefficients in [('retained', np.eye(9)[:7].sum(0)),
            ('omit_clear_anchor', np.eye(9)[:7].sum(0)-np.eye(9)[7]),
            ('omit_all_absolute_anchors', np.eye(9)[:7].sum(0)-np.eye(9)[4])]:
            variants[label] = initial_adam(gradients, coefficients, parameters)
        assert np.allclose(variants['retained']['component_direction_products'], row['groups']['all']['Adam_direction_dot_components'], rtol=2e-5, atol=2e-5)
        sampled.append(dict(begin=row['begin'], source=row['source'], reference=row['reference'], variants=variants))
    assert len(sampled) == 2 and len(array_cross) == 4
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), reviewer_sha256=sha(__file__),
        independent_audit_sha256=sha(ANALYSIS / 'independent_audit.json'), run_results_sha256=sha(run_path),
        prior_R1_run_results_sha256=sha(old_path), prior_R1_independent_audit_sha256=sha(old_analysis / 'independent_audit.json'),
        component_names=run['component_names'], cross_run_scalar_and_nonbrightness_Gram_checks=cross,
        four_sampled_cross_run_component_array_checks=array_cross,
        cross_run_ArcFace_gradient_variation_retained=True,
        earlier_cross_review_assumption_rejected='All unchanged losses must have bit-exact or 1e-9-close CUDA derivatives',
        CUDA_grid_sample_backward_variation_is_a_source_supported_inference_not_freshly_replayed=True,
        reproducibility_primary_sources=['https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.grid_sample.html',
            'https://docs.pytorch.org/docs/2.9/notes/randomness.html'],
        group_summaries=summaries, retained_scalar_loss_scales=scales,
        two_saved_initial_Adam_ablation_counterfactuals=sampled,
        inactive_guard_defect_corrected_on_manual_L4=True,
        next_cause_to_review='Unconditional clear/blur absolute anchors conflict with degraded reconstruction',
        removal_of_absolute_anchors_is_not_a_preservation_gate_change=True,
        removing_anchors_does_not_prove_clear_preservation=True,
        first_order_per_reference_evidence_not_an_actual_accumulated_update=True,
        no_failed_B50_resume_or_extra_epoch_qualification=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        new_model_quality_or_native_or_completion_claim=False, model_qualified=False, goal_complete=False))
    print(dict(complete=True, inactive_guard_defect_corrected=True,
        initial_nominal_degraded_MSE_ascents=summaries['all']['initial']['nominal_single_reference_Adam_degraded_MSE_ascents'],
        initial_anchor_gradient_degraded_MSE_ascents=summaries['all']['initial']['all_anchor_negative_gradient_degraded_MSE_ascents']))


if __name__ == '__main__':
    main()

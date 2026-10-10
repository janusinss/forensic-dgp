"""Manual-L4-only finite, recertified parameter learning; no automatic continuation."""
from pathlib import Path
import hashlib
import time
import numpy as np
from cctv_dgp_finite_guard_v1_r1_contract import BUDGETS, FRACTIONS, read, write, sha


def mechanics_accept(gates):
    """A microstep can be investigated without claiming that it meets the 1% gate."""
    return all(not gate['preservation_failures'] and gate['relative_feature_gain'] > 0
        and all(gain >= 0 for gain in gate['source_feature_gains'].values())
        and gate['brightness_gain_fraction'] <= .2 for gate in gates)


def fit(root, p, pin, original, candidate, identity, items, by_id, batch, evaluate,
        baseline, initial, progress, counts, frozen, clock, started, out,
        pixel_and_structure_losses, common_direction, state_hash, before, states, provenance):
    import torch
    from frozen_capacity_contract import capacity
    from cctv_dgp_finite_guard_v1_r1_candidate import require_vm
    require_vm(root, candidate.selected[0][1])
    candidate.enable_diagnostic_gradients(root)
    parameters = [value for _, value in candidate.selected]
    cohort_by_id = {cid: cohort['name'] for cohort in p['cohorts'] for cid in cohort['case_ids']}
    lookup = {(group['cohort'], group['source'], group['profile'], group['metric']): i
        for i, group in enumerate(p['group_losses'])}
    assert len(lookup) == 64 and len(cohort_by_id) == 100
    weight_norm = p['initial_decoder_weight_L2']
    current, current_receipt, current_file = initial.copy(), baseline, 'initial_parameters.npy'
    trials, steps = [], []
    reason = 'Three accepted changes reached; no automatic continuation'
    learning_start = time.monotonic()
    progress['accepted_states'] = []
    def retained_bytes():
        return sum(path.stat().st_size for path in root.rglob('*') if path.is_file()
            and path.relative_to(root).as_posix() not in p['assets_sha256'])
    for iteration in range(1, 4):
        clock()
        # Release the preceding state's large arrays before allocating another.
        if iteration > 1:
            del matrix, direction, query_rows
        step_dir = out / f'step_{iteration:02d}'
        step_dir.mkdir()
        current_state = state_hash(candidate.net)
        assert np.array_equal(current, candidate.vector())
        np.save(step_dir / 'current_parameters.npy', current, allow_pickle=False)
        matrix = np.zeros((64, 1996035), np.float64)
        query_rows = []
        gradient_start = time.monotonic()
        for begin in range(0, 100, 5):
            clock()
            assert time.monotonic() - gradient_start <= BUDGETS['gradient_seconds_per_state']
            ids = list(range(begin, begin + 5))
            b = batch(ids)
            prediction = candidate(b['x'], b['mask'], b['base'])
            losses = pixel_and_structure_losses(prediction, b['target'], b['mask'], b['feature'])
            embedding = identity.embedding(prediction, b['mask'], b['grid'])
            losses['ArcFace_loss'] = 1. - (embedding * b['truth']).sum(1)
            case = items[ids[0]]['case']
            cohort = cohort_by_id[case['id']]
            source = case['source']
            queries = []
            for slot, idx in enumerate(ids):
                c = items[idx]['case']
                assert cohort_by_id[c['id']] == cohort and c['source'] == source
                for metric in ['MSE', 'SSIM_loss', 'ArcFace_loss']:
                    queries.append((lookup[(cohort, source, c['profile'], metric)], losses[metric][slot]))
            queries.append((lookup[(cohort, source, 'degraded', 'landmark_structure')],
                losses['landmark_structure'][1:].mean()))
            assert len(queries) == 16
            for slot, (index, value) in enumerate(queries):
                clock()
                assert time.monotonic() - gradient_start <= BUDGETS['gradient_seconds_per_state']
                gradients = torch.autograd.grad(value, parameters, retain_graph=slot < 15,
                    allow_unused=True, materialize_grads=False)
                progress['gradient_queries'] += 1
                assert progress['gradient_queries'] <= 960
                full = np.zeros(1996035, np.float32)
                norms = []
                for desc, grad in zip(p['parameter_layout'], gradients):
                    part = np.zeros(desc['elements'], np.float32) if grad is None else grad.detach().reshape(-1).cpu().numpy().copy()
                    assert part.dtype == np.float32 and np.isfinite(part).all()
                    full[desc['start']:desc['end']] = part
                    norms.append({'name': desc['name'], 'graph_connected': grad is not None,
                        'L2': float(np.linalg.norm(part.astype(np.float64))),
                        'nonzero_values': int(np.count_nonzero(part))})
                matrix[index] += full.astype(np.float64) / 5.
                query_rows.append({'batch': begin // 5, 'group_index': index,
                    'group': p['group_losses'][index], 'value': float(value.detach()),
                    'case_ids': [items[idx]['case']['id'] for idx in ids],
                    'query_vector_sha256': hashlib.sha256(full.tobytes()).hexdigest(),
                    'query_L2': float(np.linalg.norm(full.astype(np.float64))), 'parameters': norms})
            del queries, gradients, prediction, losses, embedding
            frozen()
            print({'state': iteration, 'gradient_batch': begin // 5 + 1, 'of': 20,
                'gradient_queries': progress['gradient_queries'], 'accepted_changes': progress['optimizer_updates']}, flush=True)
        candidate.net.requires_grad_(False)
        gradient_seconds = time.monotonic() - gradient_start
        assert gradient_seconds <= BUDGETS['gradient_seconds_per_state']
        assert len(query_rows) == 320 and np.isfinite(matrix).all()
        assert retained_bytes() + matrix.nbytes + 64*1024**2 <= BUDGETS['return_uncompressed_bytes'], 'Conservative gradient-file storage stop'
        np.savez_compressed(step_dir / 'group_gradients.npz', gradients=matrix)
        write(step_dir / 'gradient_receipt.json', {'complete': True, 'groups': p['group_losses'],
            'rows': query_rows, 'gradient_queries': 320, 'aggregate_vectors': 64,
            'aggregation': 'Five equal reference contributions per cohort/source/profile group',
            'candidate_state': current_state, 'seconds': gradient_seconds,
            'individual_query_vectors_exported': False, 'local_autograd_replay_permitted': False})
        solver_start = time.monotonic()
        direction, certificate = common_direction(matrix, p['minimum_normalized_descent_cosine'])
        assert time.monotonic() - solver_start <= BUDGETS['solver_seconds_per_state']
        clock()
        write(step_dir / 'direction_certificate.json', certificate)
        step = {'iteration': iteration, 'current_vector_file': current_file,
            'current_state': current_state, 'common_direction_found': direction is not None,
            'proposal_variants': [], 'accepted_variant': None}
        if direction is None:
            reason = 'No certified direction at current state; stop without further changes'
            steps.append(step)
            write(step_dir / 'step_receipt.json', step)
            break
        np.save(step_dir / 'common_direction.npy', direction, allow_pickle=False)
        proposal_start = time.monotonic()
        accepted = False
        for fraction in FRACTIONS:
            clock()
            assert time.monotonic() - proposal_start <= BUDGETS['trial_seconds_per_state']
            label = f'proposal_s{iteration:02d}_' + format(fraction, '.0e').replace('-', 'm')
            proposal = (current.astype(np.float64) + fraction * weight_norm * direction).astype(np.float32)
            delta = proposal.astype(np.float64) - current.astype(np.float64)
            dots = matrix @ delta
            assert np.isfinite(proposal).all()
            assert retained_bytes() + p['prior_baseline_bytes']*1.5 + 40*1024**2 <= BUDGETS['return_uncompressed_bytes'], 'Proposal storage stop'
            candidate.assign_trial(proposal)
            np.save(out / (label + '_parameters.npy'), proposal, allow_pickle=False)
            receipt = evaluate(label, current_receipt['variant'])
            assert time.monotonic() - proposal_start <= BUDGETS['trial_seconds_per_state']
            comparisons = {co['name']: {stage: capacity(baseline['groups'][co['name']][stage],
                receipt['groups'][co['name']][stage], .01) for stage in ['raw', 'png']} for co in p['cohorts']}
            previous_comparisons = {co['name']: {stage: capacity(current_receipt['groups'][co['name']][stage],
                receipt['previous_anchor_groups'][co['name']][stage], .01) for stage in ['raw', 'png']} for co in p['cohorts']}
            gate_list = [gate for stages in comparisons.values() for gate in stages.values()] + [
                gate for stages in previous_comparisons.values() for gate in stages.values()]
            accepted = bool(np.all(dots < 0) and np.any(proposal != current) and mechanics_accept(gate_list))
            summary = {'variant': label, 'iteration': iteration, 'starting_vector_file': current_file,
                'relative_fraction': fraction, 'selected_weight_L2': weight_norm,
                'actual_displacement_L2': float(np.linalg.norm(delta)),
                'gradient_dot_actual_displacement': dots.tolist(),
                'all_actual_group_derivatives_negative': bool(np.all(dots < 0)),
                'candidate_state': receipt['candidate_state'], 'comparisons': comparisons,
                'previous_state_comparisons': previous_comparisons,
                'accepted_for_finite_mechanics_only': accepted,
                'model_qualification': False, 'new_epoch_or_full_capacity_pass': False}
            trials.append(summary)
            progress['trials_completed'].append(label)
            step['proposal_variants'].append(label)
            write(out / (label + '_decision.json'), summary)
            print({'proposal': label, 'accepted_for_mechanics': accepted,
                'original_one_percent_quality_gate_passes': [gate['pass'] for stages in comparisons.values() for gate in stages.values()]}, flush=True)
            if accepted:
                progress['optimizer_updates'] += 1
                progress['committed_trajectory_updates'] += 1
                progress['training_parameter_updates'] = progress['optimizer_updates']
                assert progress['optimizer_updates'] <= 3
                current, current_receipt = proposal.copy(), receipt
                current_file = label + '_parameters.npy'
                step['accepted_variant'] = label
                progress['accepted_states'].append({'iteration': iteration, 'variant': label,
                    'candidate_state': receipt['candidate_state'], 'quality_acceptance_not_implied': True})
                snapshot = {'model': {key: value.detach().cpu().clone() for key, value in candidate.net.state_dict().items()},
                    'torch_cpu_rng': torch.get_rng_state(), 'torch_cuda_rng_all': torch.cuda.get_rng_state_all(),
                    'optimizer': None, 'scheduler': None, 'accepted_parameter_changes': progress['optimizer_updates'],
                    'forward_policy': p['forward_policy'], 'protocol_sha256': pin,
                    'original_checkpoint_sha256': p['original_checkpoint_sha256'],
                    'resume_permitted': False, 'automatic_follow_on': False}
                torch.save(snapshot, out / f'trained_copy_step_{iteration:02d}.pth')
                progress['new_trained_checkpoint'] = True
                break
            candidate.assign_trial(current)
            frozen()
        step['seconds_for_proposals'] = time.monotonic() - proposal_start
        steps.append(step)
        write(step_dir / 'step_receipt.json', step)
        assert retained_bytes() <= BUDGETS['return_uncompressed_bytes']
        if not accepted:
            reason = 'All three finite proposals rejected; retain gates and stop'
            break
        candidate.enable_diagnostic_gradients(root)
        frozen()
    final_vector = current.copy()
    np.save(out / 'final_parameters.npy', final_vector, allow_pickle=False)
    candidate.net.requires_grad_(False)
    candidate.assign_trial(initial)
    frozen(True)
    clock()
    assert progress['gradient_queries'] == 320 * len(steps)
    assert counts == {'original': 20, 'candidate': 20 + 20 * len(steps) + 20 * len(trials),
        'recognizer': 40 + 20 * len(steps) + 20 * len(trials)}
    write(out / 'results.json', {'complete': True, 'protocol_sha256': pin, **progress,
        'training_update_method': 'Recomputed common descent with actual-output backtracking',
        'torch_optimizer_steps': 0, 'accepted_changes_are_actual_training': True,
        'trial_summaries': trials, 'steps': steps, 'stop_reason': reason,
        'final_vector_file': current_file, 'seconds': time.monotonic() - started,
        'learning_seconds': time.monotonic() - learning_start, 'states_before': before,
        'states_after_restore': states(), 'forward_counts': counts,
        'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(), 'original_provenance': provenance,
        'raw_and_PNG_cases': 100 * (1 + len(trials)), 'full_TRAIN_capacity_not_tested': True,
        'native_or_DEV_or_final_used': False, 'former_cross_cohort_used_for_fitting': True,
        'model_qualification': False, 'app_promotion': False, 'automatic_follow_on': False,
        'resume_permitted': False, 'goal_complete': False})

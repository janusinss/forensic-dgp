"""Generate a prospective checker from the already verified metric/replay checker."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def once(source, old, new):
    assert source.count(old) == 1, old[:140]
    return source.replace(old, new)


def main():
    target = ROOT/'scripts/audit_cctv_dgp_group_conflicts_v1_return.py'
    assert not target.exists()
    source = (ROOT/'scripts/audit_cctv_dgp_original_loss_balance_v1_return.py').read_text()
    source = source.replace('original_loss_balance_v1', 'group_conflicts_v1').replace('OriginalLossBalanceProbe', 'OriginalFeatureProbe')
    source = once(source, "result['gradient_queries'] == 0 and result['saved_parent_gradient_queries'] == 30", "result['gradient_queries'] == 160")
    source = once(source, "    assert result['forward_counts'] == {'original': 20, 'candidate': 200, 'recognizer': 220}",
        "    ntrial = len(result['trial_summaries']); assert ntrial in [0, 3]\n    assert result['forward_counts'] == {'original': 20, 'candidate': 30+20*ntrial, 'recognizer': 50+20*ntrial}")
    begin = source.index("    with np.load(BUNDLE/'evidence/aggregate_gradients.npz'")
    end = source.index('    rebuilt = {}; cases_by_id', begin)
    block = '''    with np.load(folder/'group_gradients.npz', allow_pickle=False) as saved:
        assert saved.files == ['gradients']; aggregate = saved['gradients'].copy()
    assert aggregate.dtype == np.float64 and aggregate.shape == (32, 1996035) and np.isfinite(aggregate).all()
    initial = np.load(folder/'initial_parameters.npy', allow_pickle=False)
    assert np.array_equal(initial, np.load(BUNDLE/'evidence/initial_parameters.npy', allow_pickle=False))
    assert initial.dtype == np.float32 and initial.shape == (1996035,) and np.isfinite(initial).all()
    decoder_mask = np.zeros(1996035, bool)
    for desc in p['parameter_layout']:
        if desc['partition'] == 'decoder_control': decoder_mask[desc['start']:desc['end']] = True
    weight_norm = p['initial_decoder_weight_L2']
    assert abs(float(np.linalg.norm(initial.astype(np.float64)[decoder_mask]))-weight_norm) <= 1e-10
    gradient = read(folder/'gradient_receipt.json')
    assert gradient['complete'] and gradient['groups'] == p['group_losses']
    assert gradient['seconds'] <= BUDGETS['gradient_seconds'] and gradient['gradient_queries'] == 160
    assert len(gradient['rows']) == 160 and gradient['aggregate_vectors'] == 32
    assert gradient['individual_query_vectors_exported'] is False and gradient['local_autograd_replay_permitted'] is False
    seen_queries = set(); query_counts = [0]*32
    for row in gradient['rows']:
        idx, batch_index = row['group_index'], row['batch']; assert 0 <= idx < 32 and 0 <= batch_index < 10
        assert (idx, batch_index) not in seen_queries; seen_queries.add((idx, batch_index)); query_counts[idx] += 1
        assert row['group'] == p['group_losses'][idx]
        assert row['case_ids'] == p['cohorts'][0]['case_ids'][5*batch_index:5*batch_index+5]
        assert set(row['case_ids']) & set(row['group']['case_ids'])
        assert len(row['query_vector_sha256']) == 64 and set(row['query_vector_sha256']) <= set('0123456789abcdef')
        assert math.isfinite(row['value']) and math.isfinite(row['query_L2']) and row['query_L2'] >= 0
        assert len(row['parameters']) == 158
        square_sum = 0.
        for desc, value in zip(p['parameter_layout'], row['parameters']):
            assert value['name'] == desc['name'] and isinstance(value['graph_connected'], bool)
            assert 0 <= value['nonzero_values'] <= desc['elements'] and math.isfinite(value['L2']) and value['L2'] >= 0
            assert (value['nonzero_values'] == 0) == (value['L2'] == 0)
            if not value['graph_connected']: assert value['nonzero_values'] == 0
            square_sum += value['L2']**2
        assert abs(math.sqrt(square_sum)-row['query_L2']) <= 1e-10
    assert query_counts == [5]*32
    parity = read(folder/'loss_surrogate_parity.json')
    assert parity['complete'] and parity['passed'] and parity['cases'] == 100 and parity['gradient_queries'] == 0
    assert [r['id'] for r in parity['rows']] == [c['id'] for c in p['cases']]
    maxima = {k: max(r['errors'][k] for r in parity['rows']) for k in ['MSE', 'SSIM', 'structure']}
    assert parity['maximum_errors'] == maxima
    assert maxima['MSE'] <= p['loss_surrogate_parity']['MSE_abs']
    assert maxima['SSIM'] <= p['loss_surrogate_parity']['SSIM_abs']
    assert maxima['structure'] <= p['loss_surrogate_parity']['landmark_structure_abs']
    certificate = read(folder/'direction_certificate.json'); assert certificate['complete'] and certificate['global_infeasibility_proven'] is False
    norms = np.linalg.norm(aggregate, axis=1)
    assert np.allclose(norms, certificate['gradient_norms'], rtol=1e-12, atol=1e-12)
    found = certificate['common_direction_found']; assert isinstance(found, bool)
    direction = None
    if np.all(norms > 0):
        unit = aggregate/norms[:, None]; gram = unit@unit.T
        assert np.allclose(gram, certificate['normalized_Gram'], rtol=0, atol=1e-12)
        weights = np.asarray(certificate['simplex_weights'], dtype=np.float64)
        assert weights.shape == (32,) and np.isfinite(weights).all() and (weights >= 0).all() and abs(weights.sum()-1.) <= 1e-12
        combined = weights@unit; combined_norm = float(np.linalg.norm(combined))
        assert abs(combined_norm-certificate['combined_gradient_norm']) <= 1e-12
        trial_direction = -combined/combined_norm if combined_norm > 0 else np.zeros(1996035, np.float64)
        cosines = -(unit@trial_direction)
        assert np.allclose(cosines, certificate['normalized_descent_cosines'], rtol=0, atol=1e-12)
        assert np.allclose(aggregate@trial_direction, certificate['raw_directional_derivatives'], rtol=1e-12, atol=1e-12)
        assert abs(float(cosines.min())-certificate['minimum_normalized_descent_cosine']) <= 1e-12
        assert certificate['minimum_descent_cosine_requirement'] == p['minimum_normalized_descent_cosine']
        expected_found = bool(certificate['dual_solver_success'] and combined_norm > 0 and cosines.min() >= 1e-7)
        assert found == expected_found and certificate['dual_solver_iterations'] <= 500 and certificate['seconds'] <= 120
        gap = 2.*(float(weights@gram@weights)-float(np.min(gram@weights)))
        assert abs(gap-certificate['duality_gap']) <= 1e-12
        if found:
            direction = np.load(folder/'common_direction.npy', allow_pickle=False)
            assert direction.dtype == np.float64 and direction.shape == (1996035,) and np.isfinite(direction).all()
            assert np.allclose(direction, trial_direction, rtol=0, atol=p['direction_reconstruction_arithmetic_atol'])
            assert abs(float(np.linalg.norm(direction))-1.) <= 1e-12
            assert np.all(-(unit@direction) >= 1e-7)
    else:
        assert not found and certificate['dual_solver_iterations'] == 0
    assert found == result['common_direction_found'] and ntrial == (3 if found else 0)
    if not found: assert not (folder/'common_direction.npy').exists()
    previous = read(folder/'previous_direction_group_predictions.json')
    assert previous['complete'] and previous['groups'] == p['group_losses']
    with np.load(BUNDLE/'evidence/previous_directions.npz', allow_pickle=False) as saved: old_direction = saved['balanced_2'].copy()
    old_cosines = -(aggregate@(old_direction/np.linalg.norm(old_direction)))/np.maximum(norms, 1e-300)
    assert np.allclose(old_cosines, previous['descent_cosines'], rtol=0, atol=1e-12)
    assert np.allclose(aggregate@old_direction, previous['raw_directional_derivatives'], rtol=1e-12, atol=1e-12)
    assert previous['negative_cosine_group_indices'] == np.flatnonzero(old_cosines < 0).tolist()
    labels = ['baseline']+(['common_'+format(f, '.0e').replace('-', 'm') for f in FRACTIONS] if found else [])
    assert result['trials_completed'] == labels[1:] and result['raw_and_PNG_cases'] == 100*len(labels)
'''
    source = source[:begin]+block+source[end:]
    begin = source.index("            assert summary['scope'] in SCOPES")
    end = source.index('            actual = np.load', begin)
    source = source[:begin]+'''            assert summary['scope'] == 'common' and direction is not None
            assert summary['scale_weight_partition'] == 'decoder_control'
            assert abs(summary['selected_weight_L2']-weight_norm) <= 1e-12 and summary['relative_fraction'] in FRACTIONS
            expected = (initial.astype(np.float64)+summary['relative_fraction']*weight_norm*direction).astype(np.float32)
'''+source[end:]
    source = once(source, "            for k in TERMS: assert abs(float(aggregate[k]@delta)-summary['gradient_dot_actual_displacement'][k]) <= 1e-12", '''            dots = aggregate@delta
            assert np.allclose(dots, summary['gradient_dot_actual_displacement'], rtol=1e-12, atol=1e-12)
            assert summary['all_actual_group_derivatives_negative'] == bool(np.all(dots < 0))
            assert abs(float(np.linalg.norm(delta))/weight_norm-summary['relative_displacement_actual']) <= 1e-12''')
    source = once(source, '    assert len(BRIGHTNESS_ARITHMETIC_ROWS) == 36', '    assert len(BRIGHTNESS_ARITHMETIC_ROWS) == 4*ntrial')
    source = source.replace("'all1000_raw_and_PNG_metrics_recomputed': True", "'all_saved_raw_and_PNG_metrics_recomputed': True, 'raw_and_PNG_cases': 100*len(labels)")
    source = source.replace("'audited_saved_gradients_and_all_nine_balanced_displacements_verified': True,", "'all32_saved_group_vectors_and_certificate_arithmetic_verified': True, 'conditional_displacements_verified': ntrial,\n        'individual_query_vector_aggregation_independently_recomputed': False, 'common_direction_found': found,")
    source = source.replace("'parent_audit_sha256': sha(BUNDLE/'evidence/parent_independent_audit_r2.json'),", "'parent_audit_sha256': sha(BUNDLE/'evidence/parent_independent_audit.json'),")
    source = source.replace("'all36_stored_row_comparisons_and_failure_decisions_exact': True,", "'all_stored_row_comparisons_and_failure_decisions_exact': True, 'comparisons_verified': 4*ntrial,")
    source = source.replace('Imported members changed after the retained R1 attempt', 'Imported members changed after hash-bound transfer')
    source = source.replace('Independent prospective saved-direction return audit; all scientific gates unchanged.', 'Prospective group-gradient arithmetic and fresh forward audit; no local autograd.')
    # Surrogate values are independently checked by forward arithmetic on saved outputs,
    # not by local gradient queries. This also checks the value reported for each VM query.
    insertion = '''    import torch
    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
    torch.set_num_threads(4)
    baseline_rows = {r['id']: r for r in read(folder/'baseline/metrics.json')['rows']}
    surrogate_values = {}
    with torch.no_grad():
        for c in p['cases']:
            cid = c['id']; mask = pixels(BUNDLE/c['observed'], 'L') > 0
            feature = feature_support(mask, c['landmarks5_canvas_xy'])
            raw = np.load(folder/'baseline'/(cid+'.npy'), allow_pickle=False); target = pixels(BUNDLE/c['target'])
            to_tensor = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None]
            losses = pixel_and_structure_losses(to_tensor(raw), to_tensor(target.astype(np.float32)/np.float32(255)),
                torch.from_numpy(mask.astype(np.float32))[None, None], torch.from_numpy(feature.astype(np.float32))[None, None])
            surrogate_values[cid] = {k: float(v[0]) for k, v in losses.items()}
            surrogate_values[cid]['ArcFace_loss'] = 1.-baseline_rows[cid]['raw']['ArcFace_observed_fixed']
            recorded = next(r['errors'] for r in parity['rows'] if r['id'] == cid)
            scientific = baseline_rows[cid]['raw']
            fresh_errors = {'MSE': abs(surrogate_values[cid]['MSE']-scientific['MSE']),
                'SSIM': abs(1.-surrogate_values[cid]['SSIM_loss']-scientific['SSIM']),
                'structure': abs(surrogate_values[cid]['landmark_structure']-scientific['landmark_high_frequency_MSE'])}
            assert abs(fresh_errors['MSE']-recorded['MSE']) <= 1e-12
            assert abs(fresh_errors['SSIM']-recorded['SSIM']) <= 1e-10
            assert abs(fresh_errors['structure']-recorded['structure']) <= 1e-10
    for row in gradient['rows']:
        group = row['group']; ids = [cid for cid in row['case_ids'] if cid in group['case_ids']]
        expected_value = float(np.mean([surrogate_values[cid][group['metric']] for cid in ids]))
        tolerance = 1e-6 if group['metric'] == 'ArcFace_loss' else 1e-10
        assert abs(expected_value-row['value']) <= tolerance

'''
    source = once(source, '    # Fresh saved-vector CPU inference only:', insertion+'    # Fresh saved-vector CPU inference only:')
    ast.parse(source, feature_version=(3, 10))
    target.write_text(source, encoding='utf-8', newline='\n')
    print({'prospective_checker_created': True, 'local_neural_or_gradient_calls': 0})


if __name__ == '__main__': main()

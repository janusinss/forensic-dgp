"""Audit saved loss, scope, controls and derivative arithmetic; no neural calls."""
from datetime import datetime, timezone
import hashlib
import numpy as np
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write


def check_vector(vector, record, directions):
    assert vector.dtype == np.float64 and vector.ndim == 1 and np.isfinite(vector).all()
    norm = float(np.linalg.norm(vector))
    assert np.isclose(norm,record['gradient_norm'],rtol=1e-10,atol=1e-12)
    for label,direction in directions.items():
        assert direction.shape == vector.shape and np.isfinite(direction).all()
        value = float(np.dot(vector,direction))
        assert np.isclose(value,record['saved_direction_dot'][label],rtol=1e-9,atol=1e-8)
    return norm


def baseline_value(group, rows):
    selected = [row['raw'] for row in rows if row['source']==group['source'] and
                (row['profile']!='clear' if group['profile']=='degraded' else row['profile']==group['profile'])]
    assert len(selected) == (40 if group['profile']=='degraded' else 10)
    metric = group['metric']
    if metric == 'MSE':
        return float(np.mean([r['MSE'] for r in selected]))
    if metric == 'SSIM_loss':
        return float(np.mean([1-r['SSIM'] for r in selected]))
    if metric == 'ArcFace_loss':
        return float(np.mean([1-r['ArcFace_observed_fixed'] for r in selected]))
    assert metric == 'feature_MSE'
    return float(np.mean([r['landmark_high_frequency_MSE'] for r in selected]))


def main():
    imported = read(ANALYSIS/'import.json')
    assert imported['complete'] and imported['protocol_sha256'] == PIN
    p = read(RETURN/'protocol.json')
    assert sha(RETURN/'protocol.json') == PIN
    manifest = read(RETURN/'export_manifest.json')
    for name,digest in manifest['files'].items():
        assert sha(RETURN/name) == digest, name
    assert all(sha(PARENT/n)==h for n,h in p['parent_required_sha256'].items())
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    out = RETURN/'outputs/group_gradients_v1_r1'
    if not (out/'run_results.json').exists():
        assert not manifest['complete_diagnostic']
        log=(RETURN/'review.log').read_text(encoding='utf-8')
        assert int((RETURN/'review_exit_code.txt').read_text().strip()) != 0
        if (out/'failure.json').exists():
            failure=read(out/'failure.json')
            assert manifest['failure_present'] and failure['optimizer_updates']==0
            if failure.get('stage')=='pre-model storage preflight':
                assert failure['available_free_bytes'] < failure['required_free_bytes']==p['minimum_free_bytes']==4294967296
                assert failure['neural_calls']==failure['gradient_queries']==failure['parameter_displacements']==0
            detail=dict(scope='structured retained diagnostic failure',failure=failure,failure_sha256=sha(out/'failure.json'))
        else:
            assert not out.exists() and not manifest['failure_present']
            detail=dict(scope='unstructured startup failure; no completed measurement',traceback=log,
                        no_neural_execution_claim_from_export_flag=True)
        write(ANALYSIS/'failure_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
            checker_sha256=sha(__file__),import_sha256=sha(ANALYSIS/'import.json'),**detail,
            review_log_sha256=sha(RETURN/'review.log'),independently_checked_all_return_file_and_parent_hashes=True,
            no_neural_reexecution_or_post_failure_RAM_state_hash_claimed=True,
            local_neural_or_optimizer_calls=0,no_training_qualification=True,model_qualified=False,goal_complete=False))
        print(dict(complete=True,retained_failure_audited=True,model_qualified=False))
        return
    run = read(out/'run_results.json')
    assert run['complete'] and manifest['complete_diagnostic'] and not manifest['failure_present']
    assert run['counts'] == dict(B=40,recognizer=20,original=0,gradient_queries=320,backwards=0,optimizer_updates=0,parameter_displacements=0)
    assert run['parameters'] == p['parameters'] == 3364643 and run['learning_names'] == p['learning_names']
    assert run['state_and_buffer_bytes_unchanged'] and run['original_and_recognizer_unchanged']
    assert not run['new_checkpoint_created'] and not run['model_qualified'] and run['more_epochs_not_qualified']
    assert run['model_state'] == p['initial_model_state']
    assert run['peak_VRAM'] <= p['peak_VRAM_bytes'] and run['peak_RSS'] <= p['peak_RSS_bytes'] and run['seconds'] < p['worker_seconds']
    assert run['native_pixels_used'] == run['final_identity_pixels'] == 0
    env = read(out/'environment.json')
    assert env['GPU']=='NVIDIA L4' and env['host'].split('.')[0]=='forensic-dgp-thesis' and env['manual_tmux']
    assert not env['optimizer_constructed'] and not env['failed_B50_state_loaded'] and not env['parameter_displacement_planned']
    assert int((RETURN/'review_exit_code.txt').read_text().strip()) == 0
    assert len(run['rows']) == 20 and len(run['groups']) == 32 and len(run['samples']) == 4
    sys_path = __import__('sys').path
    sys_path.insert(0,str(RETURN));sys_path.insert(1,str(PARENT))
    from cctv_dgp_bank_group_gradients_v1_r1_contract import query_key, projection_seconds
    from cctv_dgp_head4_lossless_v1 import unpack
    before_path = OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/finite_metrics.json'
    before_audit = read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/independent_audit.json')
    assert sha(before_path) == before_audit['finite_metrics_sha256']
    before = read(before_path)
    rows_by_key = {key:[] for key in [g['key'] for g in p['gradient_groups']]}
    shape_records, raw_receipts = [], []
    for block,row in zip(p['reference_blocks'],run['rows']):
        assert all(row[k] == block[k] for k in ['begin','source','reference','case_ids'])
        assert len(row['queries']) == 16 and row['raw_control_exact'] and row['input_NCHW_contiguous'] and row['target_NCHW_contiguous']
        assert row['maximum_gradient_forward_error'] <= 1e-6 and row['seconds'] < p['measurement_seconds']
        assert read(out/f"b{row['begin']//5:03d}.json") == row
        raw = np.load(out/row['raw_control_file'],allow_pickle=False)
        baseline,_ = unpack(PARENT/f"outputs/bank_comparison_v1/baseline/packs/b{row['begin']//5:03d}.npz",block['case_ids'])
        assert raw.shape == (5,256,256,3) and raw.dtype == np.float32 and np.array_equal(raw,baseline)
        assert sha(out/row['raw_control_file']) == row['raw_control_sha256']
        assert hashlib.sha256(raw.tobytes()).hexdigest() == row['exact_control_RGB_sha256']
        raw_receipts.append(dict(begin=block['begin'],cases=5,exact_control=True))
        for query,record in enumerate(row['queries']):
            key = query_key(block['source'],query)
            assert record['key'] == key and record['query'] == query and record['begin'] == block['begin'] and record['reference'] == block['reference']
            assert np.isfinite(record['value']) and np.isfinite(record['gradient_norm']) and record['gradient_norm'] > 0
            rows_by_key[key].append({k:v for k,v in record.items() if k!='key'})
    timing = read(out/'timing_projection.json')
    duration = [r['completed_batch_duration_seconds'] for r in run['rows'][:2]]
    assert timing['explicit_completed_batch_duration_seconds'] == duration and timing['measured_reference_batches']==2 and timing['safety_factor']==1.5
    assert timing['cap_seconds']==p['measurement_seconds']==600 and timing['projection_seconds']<=600
    assert np.isclose(timing['projection_seconds'],projection_seconds(duration),rtol=0,atol=1e-9)
    with np.load(RETURN/'directions.npz',allow_pickle=False) as archive:
        directions = {label:np.concatenate([archive[f'{label}_p{i:02d}'].reshape(-1) for i in range(37)]).astype(np.float64)
                      for label in ['ordinary','constrained']}
        shape_records = [archive[f'ordinary_p{i:02d}'].shape for i in range(37)]
    aggregates = []
    for expected,record in zip(p['gradient_groups'],run['groups']):
        assert all(record[k]==expected[k] for k in expected)
        assert record['mean_divisor']==10 and record['aggregate_dtype']=='float64' and record['raw_neural_gradient_dtype']=='float32'
        assert record['contribution_records'] == rows_by_key[record['key']] and len(rows_by_key[record['key']]) == 10
        assert len({r['reference'] for r in record['contribution_records']}) == 10
        vector = np.load(out/record['file'],allow_pickle=False)
        assert vector.shape == (p['parameters'],) and sha(out/record['file']) == record['sha256']
        norm = check_vector(vector,record,directions)
        assert norm > 0
        value = float(np.mean([r['value'] for r in record['contribution_records']]))
        assert np.isclose(value,record['value'],rtol=0,atol=1e-12)
        baseline_loss = baseline_value(record,before['baseline_rows'])
        assert np.isclose(value,baseline_loss,rtol=0,atol=1e-6), (record['key'],value,baseline_loss)
        for label in directions:
            mean_dot = float(np.mean([r['saved_direction_dot'][label] for r in record['contribution_records']]))
            assert np.isclose(mean_dot,record['saved_direction_dot'][label],rtol=1e-9,atol=1e-8)
        aggregates.append(dict(key=record['key'],sha256=record['sha256'],gradient_norm=norm,
            baseline_loss=baseline_loss,reported_loss=value,
            saved_direction_first_order_slopes=record['saved_direction_dot'],
            contribution_dot_means_match_aggregate=True,full_neural_gradient_reexecution=False))
        del vector
    checked_samples = []
    original_hinge = OUT/'cctv_dgp_bank_hinge_boundary_v1_return/outputs/loss_alignment_v1'
    direction_review = read(RETURN/'direction_review.json')
    for sample in run['samples']:
        assert sample['begin'] in p['sample_gradient_begins'] and sample['query'] in [0,2]
        assert sha(out/sample['file']) == sample['sha256']
        with np.load(out/sample['file'],allow_pickle=False) as archive:
            assert set(archive.files) == {f'p{i:02d}' for i in range(37)}
            assert all(archive[f'p{i:02d}'].dtype == np.float32 and archive[f'p{i:02d}'].shape == shape_records[i] for i in range(37))
            flat = np.concatenate([archive[f'p{i:02d}'].reshape(-1) for i in range(37)])
        assert hashlib.sha256(flat.tobytes()).hexdigest() == sample['gradient_vector_sha256']
        check_vector(flat.astype(np.float64),sample,directions)
        old_clear_check = None
        if sample['query'] == 0:
            binding = next(r for r in direction_review['sampled_gradient_bindings'] if r['begin']==sample['begin'])
            old_path = original_hinge/'sampled_gradients'/f"initial_b{sample['begin']//5:03d}.npz"
            assert sha(old_path) == binding['sha256']
            with np.load(old_path,allow_pickle=False) as archive:
                expected = np.concatenate([archive[f'g07_p{i:02d}'].reshape(-1) for i in range(37)]).astype(np.float64)
            expected *= read(PARENT/'protocol.json')['active_anchor_normalizers'][0]
            delta = float(np.abs(flat.astype(np.float64)-expected).max())
            assert np.allclose(flat,expected,rtol=1e-4,atol=1e-8), 'Raw clear-MSE derivative differs from retained normalized derivative'
            old_clear_check = dict(maximum_absolute_error=delta,rtol=1e-4,atol=1e-8,
                numerical_derivative_check_not_relaxed_quality_gate=True,old_measured_gradient_sha256=binding['sha256'])
        checked_samples.append(dict(file=sample['file'],sha256=sample['sha256'],key=sample['key'],
            saved_vector_sha_norm_and_direction_dots_checked=True,retained_clear_MSE_gradient_check=old_clear_check))
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/'independent_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=PIN,import_sha256=sha(ANALYSIS/'import.json'),run_results_sha256=sha(out/'run_results.json'),
        counts_recomputed=run['counts'],all100_raw_controls_match=True,query_scope_bindings_checked=320,
        all32_aggregate_vector_norm_dot_mean_and_loss_bindings_checked=True,
        aggregate_gradients=aggregates,sample_vector_checks=checked_samples,
        derivatives_measured_on_manual_VM_not_independently_reexecuted_for_all320=True,
        two_clear_MSE_sample_derivatives_compared_to_prior_measured_normalized_gradients=True,
        explicit_duration_timing_projection_checked=True,VM_seconds=run['seconds'],VM_peak_VRAM=run['peak_VRAM'],VM_peak_RSS=run['peak_RSS'],
        complete_measurement_not_a_finite_direction_or_training_result=True,
        native_or_final_or_new_PNG_visual_qualification=False,local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0,
        parameter_displacements=0,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,control_cases=100,gradient_groups=32,gradient_queries_measured=320,model_qualified=False))


if __name__ == '__main__':
    main()

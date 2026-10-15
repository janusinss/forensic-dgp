"""Prospective complete saved-stage PNG-gradient audit; no neural replay."""
from datetime import datetime, timezone
import hashlib
import math
import sys

import numpy as np
from PIL import Image

from import_cctv_dgp_png_group_gradients_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write


def check_vector(vector, record, directions):
    assert vector.dtype == np.float64 and vector.ndim == 1 and np.isfinite(vector).all()
    norm=float(np.linalg.norm(vector))
    assert np.isclose(norm,record['gradient_norm'],rtol=1e-10,atol=1e-12)
    for label,direction in directions.items():
        assert direction.shape==vector.shape and np.isfinite(direction).all()
        assert np.isclose(float(vector@direction),record['saved_direction_dot'][label],rtol=1e-9,atol=1e-8)
    return norm


def baseline_value(group,rows):
    selected=[row['png'] for row in rows if row['source']==group['source'] and
        (row['profile']!='clear' if group['profile']=='degraded' else row['profile']==group['profile'])]
    assert len(selected)==(40 if group['profile']=='degraded' else 10)
    metric=group['metric']
    key={'MSE':'MSE','SSIM_loss':'SSIM','ArcFace_loss':'ArcFace_observed_fixed','feature_MSE':'landmark_high_frequency_MSE'}[metric]
    return float(np.mean([1-row[key] if metric in ['SSIM_loss','ArcFace_loss'] else row[key] for row in selected]))


def image(path,mode='RGB'):
    with Image.open(path) as im:
        assert im.size==(256,256)
        return np.asarray(im.convert(mode)).copy()


def main():
    assert not (ANALYSIS/'independent_audit.json').exists() and not (ANALYSIS/'failure_audit.json').exists()
    imported=read(ANALYSIS/'import.json');assert imported['complete'] and imported['protocol_sha256']==PIN
    p=read(RETURN/'protocol.json');assert sha(RETURN/'protocol.json')==PIN and p['metric_stage']=='PNG'
    manifest=read(RETURN/'export_manifest.json')
    assert {path.relative_to(RETURN).as_posix() for path in RETURN.rglob('*') if path.is_file()}==set(manifest['files'])|{'export_manifest.json'}
    assert all(sha(RETURN/name)==digest for name,digest in manifest['files'].items())
    assert all(sha(PARENT/name)==digest for name,digest in p['parent_required_sha256'].items())
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256'];assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    assert not list(RETURN.rglob('*.pth'))
    out=RETURN/'outputs/png_group_gradients_v1_r1'
    if not (out/'run_results.json').exists():
        assert not manifest['complete_diagnostic']
        assert int((RETURN/'review_exit_code.txt').read_text().strip())!=0
        failure=read(out/'failure.json') if (out/'failure.json').exists() else None
        assert bool(failure)==manifest['failure_present']
        if failure and failure.get('stage')=='pre-model storage preflight':
            assert failure['available_free_bytes']<failure['required_free_bytes']==p['minimum_free_bytes']
            assert failure['neural_calls']==failure['gradient_queries']==failure['parameter_displacements']==0
        write(ANALYSIS/'failure_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
            checker_sha256=sha(__file__),import_sha256=sha(ANALYSIS/'import.json'),failure=failure,
            review_log_sha256=sha(RETURN/'review.log'),retained_immutable_failure=True,
            no_execution_or_quality_counts_inferred_from_export=True,new_neural_or_gradient_calls=0,
            model_qualified=False,goal_complete=False))
        print(dict(complete=True,retained_failure_audited=True,model_qualified=False));return
    run=read(out/'run_results.json');env=read(out/'environment.json')
    counts=dict(B=40,recognizer=20,original=0,gradient_queries=320,backwards=0,optimizer_updates=0,parameter_displacements=0)
    assert run['complete'] and manifest['complete_diagnostic'] and not manifest['failure_present']
    assert run['counts']==counts and run['metric_stage']=='PNG' and run['straight_through_backward_is_approximate']
    assert run['all100_delivered_forward_pixels_exact'] and run['state_and_buffer_bytes_unchanged']
    assert run['original_and_recognizer_unchanged'] and run['model_state']==p['initial_model_state']
    assert not run['new_checkpoint_created'] and not run['model_qualified'] and run['more_epochs_not_qualified']
    assert run['parameters']==p['parameters']==3364643 and run['learning_names']==p['learning_names']
    assert 0<run['seconds']<p['worker_seconds'] and 0<run['peak_VRAM']<=p['peak_VRAM_bytes']
    assert 0<run['peak_RSS']<=p['peak_RSS_bytes'] and run['native_pixels_used']==run['final_identity_pixels']==0
    assert env['GPU']=='NVIDIA L4' and env['host'].split('.')[0]=='forensic-dgp-thesis' and env['manual_tmux']
    assert env['metric_stage']=='PNG' and env['exact_delivered_pixels_forward'] and env['straight_through_backward_is_approximate']
    assert not any(env[key] for key in ['optimizer_constructed','failed_B50_state_loaded','parameter_displacement_planned'])
    assert int((RETURN/'review_exit_code.txt').read_text().strip())==0
    assert len(run['rows'])==20 and len(run['groups'])==32 and len(run['samples'])==4
    sys.path.insert(0,str(RETURN));sys.path.insert(1,str(PARENT))
    from cctv_dgp_png_group_gradients_v1_contract import scopes,query_key,projection_seconds
    from cctv_dgp_head4_lossless_v1 import unpack
    parent=read(PARENT/'protocol.json');blocks,groups=scopes(parent['cases'],p['case_ids'])
    assert p['reference_blocks']==blocks and p['gradient_groups']==groups
    baselines=[row for row in read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')['rows'] if row['id'] in p['case_ids']]
    assert len(baselines)==100 and {row['id'] for row in baselines}==set(p['case_ids'])
    metrics={row['id']:row for row in baselines}
    contributions={group['key']:[] for group in groups};control_records=[]
    for block,row in zip(blocks,run['rows']):
        assert all(row[key]==block[key] for key in ['begin','source','reference','case_ids'])
        assert len(row['queries'])==16 and row['raw_control_exact'] and row['maximum_gradient_forward_error']==0
        assert row['input_NCHW_contiguous'] and row['target_NCHW_contiguous']
        assert row['exact_RGB32_and_feature64_PNG_forward'] and row['straight_through_backward_is_approximate']
        assert read(out/f"b{block['begin']//5:03d}.json")==row
        raw=np.load(out/row['raw_control_file'],allow_pickle=False)
        png=np.load(out/row['PNG_control_file'],allow_pickle=False)
        vectors=np.load(out/row['PNG_embeddings_file'],allow_pickle=False)
        assert raw.shape==(5,256,256,3) and raw.dtype==np.float32
        assert png.shape==raw.shape and png.dtype==np.uint8
        assert vectors.dtype==np.float32 and vectors.shape==(15,512) and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors,axis=1),1,rtol=0,atol=1e-5)
        before,emb=unpack(PARENT/f"outputs/bank_comparison_v1/baseline/packs/b{block['begin']//5:03d}.npz",block['case_ids'])
        np.testing.assert_array_equal(raw,before)
        assert np.allclose(vectors[:5],vectors[5:10],rtol=1e-6,atol=1e-6)
        assert np.allclose(vectors[5:10],emb[:,1],rtol=1e-6,atol=1e-6)
        assert np.allclose(vectors[10:15],emb[:,2],rtol=1e-6,atol=1e-6)
        assert sha(out/row['raw_control_file'])==row['raw_control_sha256']
        assert sha(out/row['PNG_control_file'])==row['PNG_control_sha256']
        assert sha(out/row['PNG_embeddings_file'])==row['PNG_embeddings_sha256']
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['exact_control_RGB_sha256']
        assert hashlib.sha256(png.tobytes()).hexdigest()==row['actual_PNG_RGB_sha256']
        for index,case in enumerate(block['case_ids']):
            np.testing.assert_array_equal(png[index],image(PARENT/'outputs/bank_comparison_v1/baseline/previews'/(case+'.png')))
        expected=[value for case in block['case_ids'] for value in
            [metrics[case]['png']['MSE'],1-metrics[case]['png']['SSIM'],1-metrics[case]['png']['ArcFace_observed_fixed']]]
        expected+=[float(np.mean([metrics[case]['png']['landmark_high_frequency_MSE'] for case in block['case_ids'][1:]]))]
        assert row['expected_initial_PNG_loss_values']==expected
        assert row['forward_loss_numeric_tolerances']==p['forward_loss_numeric_tolerances']
        for query,record in enumerate(row['queries']):
            key=query_key(block['source'],query)
            assert record['key']==key and record['begin']==block['begin'] and record['query']==query and record['reference']==block['reference']
            metric=['MSE','SSIM_loss','ArcFace_loss'][query%3] if query<15 else 'feature_MSE'
            assert abs(record['value']-expected[query])<=p['forward_loss_numeric_tolerances'][metric]
            assert math.isfinite(record['gradient_norm']) and record['gradient_norm']>0
            if query<15 and query%3==2:
                i=query//3
                measured=1-float(np.dot(vectors[i].astype(np.float64),vectors[10+i].astype(np.float64)))
                assert abs(record['value']-measured)<=1e-12
            contributions[key].append({key:value for key,value in record.items() if key!='key'})
        control_records.append(dict(begin=block['begin'],exact_raw_and_PNG_cases=5,
            exact_raw_sha256=row['raw_control_sha256'],exact_PNG_sha256=row['PNG_control_sha256']))
    timing=read(out/'timing_projection.json');durations=[row['completed_batch_duration_seconds'] for row in run['rows'][:2]]
    assert timing['explicit_completed_batch_duration_seconds']==durations and timing['measured_reference_batches']==2
    assert timing['safety_factor']==1.5 and timing['cap_seconds']==p['measurement_seconds']==600
    assert timing['projection_seconds']<=600 and np.isclose(timing['projection_seconds'],projection_seconds(durations),rtol=0,atol=1e-9)
    with np.load(RETURN/'directions.npz',allow_pickle=False) as archive:
        assert set(archive.files)=={f'balanced_p{i:02d}' for i in range(37)}
        directions={'balanced':np.concatenate([archive[f'balanced_p{i:02d}'].reshape(-1) for i in range(37)]).astype(np.float64)}
        shapes=[archive[f'balanced_p{i:02d}'].shape for i in range(37)]
    aggregate_records=[]
    for expected,record in zip(groups,run['groups']):
        assert all(record[key]==value for key,value in expected.items())
        assert record['mean_divisor']==10 and record['aggregate_dtype']=='float64' and record['raw_neural_gradient_dtype']=='float32'
        assert record['contribution_records']==contributions[record['key']]
        assert len(contributions[record['key']])==len({r['reference'] for r in contributions[record['key']]})==10
        vector=np.load(out/record['file'],allow_pickle=False)
        assert vector.shape==(p['parameters'],) and sha(out/record['file'])==record['sha256']
        assert check_vector(vector,record,directions)>0
        assert np.isclose(record['value'],np.mean([r['value'] for r in record['contribution_records']]),rtol=0,atol=1e-12)
        baseline=baseline_value(record,baselines)
        assert abs(record['value']-baseline)<=p['forward_loss_numeric_tolerances'][record['metric']]
        mean_dot=float(np.mean([r['saved_direction_dot']['balanced'] for r in record['contribution_records']]))
        assert np.isclose(mean_dot,record['saved_direction_dot']['balanced'],rtol=1e-9,atol=1e-8)
        aggregate_records.append(dict(key=record['key'],sha256=record['sha256'],gradient_norm=record['gradient_norm'],
            scientific_PNG_baseline_loss=baseline,surrogate_gradient_forward_loss=record['value'],
            balanced_raw_direction_dot=record['saved_direction_dot']['balanced'],
            full_neural_gradient_reexecution=False,backward_is_surrogate=True))
    for sample in run['samples']:
        assert sample['begin'] in p['sample_gradient_begins'] and sample['query'] in p['sample_queries']
        assert sha(out/sample['file'])==sample['sha256']
        with np.load(out/sample['file'],allow_pickle=False) as archive:
            assert set(archive.files)=={f'p{i:02d}' for i in range(37)}
            assert all(archive[f'p{i:02d}'].dtype==np.float32 and archive[f'p{i:02d}'].shape==shapes[i] for i in range(37))
            flat=np.concatenate([archive[f'p{i:02d}'].reshape(-1) for i in range(37)])
        assert hashlib.sha256(flat.tobytes()).hexdigest()==sample['gradient_vector_sha256']
        check_vector(flat.astype(np.float64),sample,directions)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(ANALYSIS/'independent_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=PIN,import_sha256=sha(ANALYSIS/'import.json'),run_results_sha256=sha(out/'run_results.json'),
        counts_recomputed=counts,exact_raw_controls=100,exact_delivered_PNG_controls=100,query_scope_bindings_checked=320,
        scientific_PNG_forward_value_checks=320,aggregate_gradient_groups_checked=32,sampled_full_gradient_vectors=4,
        aggregate_gradients=aggregate_records,control_records=control_records,explicit_timing_projection_checked=True,
        VM_seconds=run['seconds'],VM_peak_VRAM=run['peak_VRAM'],VM_peak_RSS=run['peak_RSS'],
        derivatives_are_declared_surrogates_not_true_PNG_derivatives=True,
        returned_derivatives_not_independently_reexecuted_neurally=True,
        no_new_finite_model_displacements_or_images_or_checkpoint=True,
        local_neural_gradient_optimizer_calls=0,parameter_displacements=0,
        raw_PNG_preservation_structure_and_all_native_final_completion_requirements_remain=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,PNG_controls=100,surrogate_gradient_groups=32,model_qualified=False))


if __name__=='__main__':main()

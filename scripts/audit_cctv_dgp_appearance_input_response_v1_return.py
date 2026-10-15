"""Replay saved RGB-input gradients and compare response boundaries; no neural calls."""
from datetime import datetime,timezone
import hashlib
import sys
import cv2
import numpy as np

from import_cctv_dgp_appearance_input_response_v1_return import ROOT,OUT,RETURN,ANALYSIS,SOURCE,BANK,PIN,read,sha,write
from cctv_dgp_conditioning_response_v1_contract import response_receipt
from cctv_dgp_cross_run_gradient_report_v1 import report as parity_report


def fixed_grid(matrix):
    matrix=np.asarray(matrix,dtype=np.float64)
    assert matrix.shape==(2,3) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix[:,:2]))>=1e-8
    inverse=cv2.invertAffineTransform(matrix);yy,xx=np.meshgrid(np.arange(112),np.arange(112),indexing='ij')
    x=inverse[0,0]*xx+inverse[0,1]*yy+inverse[0,2];y=inverse[1,0]*xx+inverse[1,1]*yy+inverse[1,2]
    return np.stack((2*(x+.5)/256-1,2*(y+.5)/256-1),axis=-1).astype(np.float32)


def audit_batch(folder,p,old,block):
    from PIL import Image
    parent=SOURCE/'outputs/conditioning_response_v1';stem=f"b{block['begin']//5:03d}"
    record=read(folder/(stem+'.json'));assert record['begin']==block['begin'] and record['case_ids']==block['case_ids']
    gradient_file=folder/(stem+'_RGB_gradients.npy');embedding_file=folder/(stem+'_embeddings.npz');grid_file=folder/(stem+'_grid.npy')
    assert sha(gradient_file)==record['gradients_sha256'] and sha(embedding_file)==record['embeddings_sha256'] and sha(grid_file)==record['grid_sha256']
    before=np.load(parent/'initial'/(stem+'.npy'),allow_pickle=False)
    assert sha(parent/'initial'/(stem+'.npy'))==record['initial_RAW_sha256']
    g=np.load(gradient_file,allow_pickle=False);assert g.dtype==np.float32 and g.shape==(5,256,256,3) and np.isfinite(g).all()
    with np.load(embedding_file,allow_pickle=False) as f:
        assert f.files==['raw'];v=f['raw'];assert v.dtype==np.float32 and v.shape==(15,512) and np.isfinite(v).all()
    with np.load(parent/'initial'/(stem+'_embeddings.npz'),allow_pickle=False) as f:expected=f['raw']
    np.testing.assert_allclose(v,expected,rtol=1e-6,atol=1e-6)
    assert record['original_forward_embedding_rtol']==record['original_forward_embedding_atol']==1e-6
    assert record['maximum_initial_embedding_difference']==float(np.max(np.abs(v-expected)))
    references={r['id']:r for r in old['references']};cases=old['cases'][block['begin']:block['begin']+5]
    grid=np.load(grid_file,allow_pickle=False);assert grid.dtype==np.float32
    np.testing.assert_array_equal(grid,np.stack([fixed_grid(references[c['source_person_or_reference']]['matrix112']) for c in cases]))
    initial={r['id']:r for r in read(parent/'initial/snapshot.json')['rows']}
    snapshots={label:{r['id']:r for r in read(parent/label/'snapshot.json')['rows']} for label in p['signed_labels']}
    changes={label:np.load(parent/label/(stem+'.npy'),allow_pickle=False) for label in p['signed_labels']}
    assert len(record['rows'])==5
    for i,(row,c) in enumerate(zip(record['rows'],cases)):
        assert {k:row[k] for k in ['id','source','profile']}=={k:c[k] for k in ['id','source','profile']}
        assert row['reference']==c['source_person_or_reference'] and row['begin']==block['begin'] and row['case_index']==i
        assert row['cross_case_gradient_maximum']==0 and row['scalar_alignment_bound']==1e-6
        assert row['gradient_HWC_bytes_sha256']==hashlib.sha256(g[i].tobytes()).hexdigest()
        assert np.isclose(np.linalg.norm(g[i].astype(np.float64)),row['gradient_norm'],rtol=1e-10,atol=1e-12)
        fresh=1-float(v[i].astype(np.float64)@v[10+i].astype(np.float64))
        assert row['fresh_double_loss']==fresh and abs(row['derivative_scalar']-fresh)<=1e-13
        assert row['scientific_cached_loss']==1-initial[c['id']]['raw']['ArcFace_observed_fixed']
        assert abs(row['derivative_scalar']-row['scientific_cached_loss'])<=1e-6
        with Image.open(BANK/references[c['source_person_or_reference']]['observed']) as mask:observed=np.asarray(mask.convert('L'))>0
        assert [r['label'] for r in row['projections']]==p['signed_labels']
        for trial in row['projections']:
            after=changes[trial['label']]
            assert after.dtype==np.float32 and after.shape==before.shape and np.isfinite(after).all()
            assert np.array_equal(after[i][~observed],before[i][~observed])
            assert sha(parent/trial['label']/(stem+'.npy'))==trial['saved_RAW_sha256']
            # Independently express the complete actual displacement in double precision.
            predicted=float(np.dot(g[i].astype(np.float64).reshape(-1),(after[i].astype(np.float64)-before[i].astype(np.float64)).reshape(-1)))
            assert np.isclose(predicted,trial['input_projection'],rtol=1e-10,atol=1e-13)
            assert trial['actual_scientific_loss']==1-snapshots[trial['label']][c['id']]['raw']['ArcFace_observed_fixed']
    return record,g


def main():
    imported=read(ANALYSIS/'import.json');p=read(RETURN/'protocol.json');old=read(BANK/'protocol.json')
    assert imported['complete'] and imported['protocol_sha256']==PIN==sha(RETURN/'protocol.json')
    manifest=read(RETURN/'export_manifest.json')
    assert all(sha(RETURN/n)==h for n,h in manifest['files'].items())
    for folder,bindings in [(SOURCE,p['response_required_sha256']),(BANK,p['bank_required_sha256'])]:assert all(sha(folder/n)==h for n,h in bindings.items())
    out=RETURN/'outputs/appearance_input_response_v1';run=read(out/'run_results.json') if (out/'run_results.json').exists() else None
    failure=read(out/'failure.json') if (out/'failure.json').exists() else None
    assert manifest['complete_diagnostic']==bool(run) and manifest['failure_present']==bool(failure)
    assert manifest['optimizer_updates']==manifest['accepted_updates']==0 and not manifest['complete_pilot']
    exitcode=int((RETURN/'review_exit_code.txt').read_text().strip());assert (exitcode==0)==bool(run)
    if not (out/'environment.json').exists():
        assert not run
        if failure and failure.get('stage')=='pre-model storage preflight':
            assert failure['available_free_bytes']<failure['required_free_bytes']==p['minimum_free_bytes']
            assert all(v==0 for v in failure['counts'].values())
        write(ANALYSIS/'failure_audit.json',dict(complete=True,scope='Pre-model failure only',failure=failure,exit_code=exitcode,
            local_neural_or_training_calls=0,model_qualified=False,goal_complete=False));return
    env=read(out/'environment.json')
    assert env['GPU']=='NVIDIA L4' and env['host'].split('.')[0]=='forensic-dgp-thesis' and env['manual_tmux']
    assert env['torch']==p['expected_torch'] and env['recognizer_state']==p['recognizer_state']
    assert not env['restorer_or_prior_constructed'] and not env['optimizer_constructed'] and env['model_parameters_requiring_grad']==0
    assert all(env['precision'][k] is False for k in ['matmul_tf32','cudnn_tf32','cudnn_benchmark']) and env['precision']['cudnn_deterministic']
    batches={};repeats=[];parity=[]
    for repeat in range(2):
        folder=out/f'repeat{repeat:02d}'
        if not folder.exists():continue
        rows=[]
        for block in p['reference_blocks']:
            stem=f"b{block['begin']//5:03d}"
            if not (folder/(stem+'.json')).exists():continue
            record,g=audit_batch(folder,p,old,block);rows.extend(record['rows']);batches[(repeat,block['begin'])]=g
            if repeat and (0,block['begin']) in batches:
                for i,row in enumerate(record['rows']):parity.append(dict(id=row['id'],**parity_report(g[i].reshape(-1),batches[(0,block['begin'])][i].reshape(-1))))
        if (folder/'repeat.json').exists():
            r=read(folder/'repeat.json');assert r['rows']==rows and len(rows)==100 and r['repeat']==repeat
            assert r['recognizer_state']==p['recognizer_state'] and r['buffer_state']==env['buffer_state']
            assert r['reference_batch_order']==[b['begin'] for b in p['reference_blocks']] and r['seconds']<=p['round_seconds']
            timing=read(folder/'timing_projection.json')
            assert timing['cap_seconds']==180 and timing['projected_seconds']<=180
            assert np.isclose(timing['projected_seconds'],max(timing['first_two_batch_seconds'])*20*1.5,rtol=0,atol=1e-9)
            repeats.append(r)
    comparisons=[]
    if len(repeats)==2:
        source=SOURCE/'outputs/conditioning_response_v1';initial=read(source/'initial/snapshot.json')
        signed={label:read(source/label/'snapshot.json') for label in p['signed_labels']}
        old_responses=read(RETURN/'evidence/parameter_response_audit.json')['central_response_comparisons']
        for group in p['appearance_groups']:
            sets=[[r for r in record['rows'] if r['source']==group['source'] and r['profile']==group['profile']] for record in repeats]
            assert all(len(rows)==10 for rows in sets)
            values=[float(np.mean([r['derivative_scalar'] for r in rows])) for rows in sets]
            before=1-initial['groups']['raw'][group['source']+'/'+group['profile']]['ArcFace_observed_fixed']
            spread=max(abs(values[0]-values[1]),abs(values[0]-before),abs(values[1]-before))
            for pair in range(3):
                minus,plus=f'minus{pair:02d}',f'plus{pair:02d}'
                prior=next(r for r in old_responses if r['pair']==pair and r['key']=='RAW/'+group['key'])
                losses={label:1-signed[label]['groups']['raw'][group['source']+'/'+group['profile']]['ArcFace_observed_fixed'] for label in [plus,minus]}
                for repeat,rows in enumerate(sets):
                    projections={label:float(np.mean([next(v['input_projection'] for v in r['projections'] if v['label']==label) for r in rows])) for label in [plus,minus]}
                    receipt=response_receipt(before,losses[plus],losses[minus],prior['magnitude'],projections[plus],projections[minus],spread)
                    comparisons.append(dict(repeat=repeat,pair=pair,key='RAW/'+group['key'],
                        RGB_input_response=receipt,retained_parameter_response=prior,
                        diagnostic_boundary_localization_only=True,scientific_quality_gate_relaxed=False))
    if run:
        assert not failure and len(repeats)==2 and len(batches)==40 and len(parity)==100 and len(comparisons)==60
        assert run['counts']==dict(restorer=0,recognizer=40,gradient_queries=200,neural_parameter_displacements=0,optimizer_updates=0,accepted_updates=0)
        assert run['frozen_initial_and_final_state']==p['recognizer_state'] and run['buffer_state']==env['buffer_state']
        with np.load(out/'initial_RNG.npz',allow_pickle=False) as a,np.load(out/'restored_RNG.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[n],b[n]) for n in a.files)
        assert run['Torch_RNG_restored'] and run['seconds']<p['measurement_seconds']
        assert run['peak_VRAM']<=p['peak_VRAM_bytes'] and run['peak_RSS']<=p['peak_RSS_bytes']
        assert run['full_epochs']==run['native_pixels_used']==run['final_identity_pixels']==0 and not run['model_qualified']
    else:
        assert failure
        if failure.get('fresh_frozen_recognizer_state') is not None:assert failure['fresh_frozen_recognizer_state']==p['recognizer_state']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256'];assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/('independent_audit.json' if run else 'failure_audit.json'),dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),import_sha256=sha(ANALYSIS/'import.json'),protocol_sha256=PIN,full_diagnostic_returned=bool(run),failure=failure,
        completed_gradient_batches=len(batches),completed_repeats=len(repeats),per_case_gradient_parity_original_bounds=parity,
        all_original_parity_checks_pass=all(r['original_parity_check_pass'] for r in parity) if parity else None,
        RGB_input_and_parameter_boundary_comparisons=comparisons,
        original_quality_thresholds_and_diagnostic_bounds_unchanged=True,
        exact_input_gradient_shapes_hashes_scores_affines_and_actual_saved_pixel_projections_replayed=True,
        cross_case_zero_gradients_are_VM_reported_and_not_neurally_recomputed_locally=True,
        neural_derivatives_and_embeddings_not_locally_reexecuted=True,
        no_new_face_images_or_visual_sheets_generated=True,local_neural_or_training_calls=0,
        optimizer_updates=0,accepted_updates=0,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,diagnostic_completed=bool(run),gradient_batches=len(batches),boundary_comparisons=len(comparisons),model_qualified=False))


if __name__=='__main__':main()

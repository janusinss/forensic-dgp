"""Recompute saved outputs, finite-response arithmetic and repeated-vector facts."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from import_cctv_dgp_conditioning_response_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write
from audit_cctv_dgp_raw_conditioning_learning_v1_r1_return import schema, state_with_component
from audit_cctv_dgp_joint_raw_png_forward_v1_return import image, support, metrics, delivered, grouped, state_hash
from cctv_dgp_raw_conditioning_learning_v1_r1_contract import decision
from raw_conditioning_decision_receipt_check_v1_r1 import validate_receipt
from cctv_dgp_cross_run_gradient_report_v1 import report as cross_run_report
from cctv_dgp_conditioning_response_v1_contract import response_receipt


def audit_snapshot(folder,p,old,virtual,names,edges):
    snapshot=read(folder/'snapshot.json');refs={r['id']:r for r in old['references']}
    rows=[];cells={}
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    stored={r['id']:r for r in snapshot['rows']};assert set(stored)==set(p['case_ids'])
    exact=folder.name in ['initial','rollback'];assert snapshot['exact_controls']==exact
    for block in p['reference_blocks']:
        cs=old['cases'][block['begin']:block['begin']+5];stem=f"b{block['begin']//5:03d}"
        raw=np.load(folder/(stem+'.npy'),allow_pickle=False)
        assert raw.dtype==np.float32 and raw.shape==(5,256,256,3) and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
        baseline,expected=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/{stem}.npz',block['case_ids'])
        with np.load(folder/(stem+'_embeddings.npz'),allow_pickle=False) as data:
            assert set(data.files)=={'raw','png'};emb={k:data[k] for k in data.files}
        proof=next(r for r in snapshot['blocks'] if r['begin']==block['begin'])
        assert sha(folder/(stem+'.npy'))==proof['raw_sha256'] and sha(folder/(stem+'_embeddings.npz'))==proof['embeddings_sha256']
        for s,j in [('raw',0),('png',1)]:
            assert emb[s].dtype==np.float32 and emb[s].shape==(15,512) and np.isfinite(emb[s]).all()
            np.testing.assert_allclose(emb[s][5:10],expected[:,j],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(emb[s][10:15],expected[:,2],rtol=1e-6,atol=1e-6)
        for i,c in enumerate(cs):
            ref=refs[c['source_person_or_reference']];camera,target,mask=image(PARENT/c['input']),image(PARENT/ref['target']),image(PARENT/ref['observed'],'L')>0
            patch=support(mask,c['landmarks5_canvas_xy']);png=image(folder/'png'/(c['id']+'.png'))
            assert np.array_equal(delivered(raw[i],camera,mask),png)
            assert np.array_equal(raw[i][~mask],(camera.astype(np.float32)/np.float32(255))[~mask]) and np.array_equal(png[~mask],camera[~mask])
            if exact:assert np.array_equal(raw[i],baseline[i])
            shift=(raw[i]-baseline[i])[mask].astype(np.float64).mean(0)
            mean=np.clip(baseline[i].astype(np.float64)+shift,0,1).astype(np.float32);mean[~mask]=(camera.astype(np.float32)/np.float32(255))[~mask]
            mean_png=delivered(mean,camera,mask)
            row=dict(id=c['id'],source=c['source'],profile=c['profile'],reference=c['source_person_or_reference'])
            assert all(row[k]==stored[c['id']][k] for k in row)
            fresh={}
            for s,actual,mean_actual,j in [('raw',raw[i],mean,0),('png',png.astype(np.float32)/np.float32(255),mean_png.astype(np.float32)/np.float32(255),1)]:
                v=emb[s];fresh[s]=float(v[i]@v[10+i]);assert fresh[s]==stored[c['id']]['fresh_recognizer_scores'][s]
                sv,tv=(expected[i,j],expected[i,2]) if exact else (v[i],v[10+i])
                row[s]=metrics(actual,target,mask,patch,sv,tv,mean_actual,prediction8=png if s=='png' else None)
                for key,value in row[s].items():
                    other=stored[c['id']][s][key]
                    if value is None or isinstance(value,bool):assert value==other
                    else:assert np.isclose(value,other,rtol=0,atol=1e-13),(folder.name,c['id'],s,key)
            row['fresh_recognizer_scores']=fresh;rows.append(row);cells[c['id']]=png
    groups={s:grouped(rows,s) for s in ['raw','png']}
    for s,values in groups.items():
        for k,r in values.items():
            for m,v in r.items():assert np.isclose(v,snapshot['groups'][s][k][m],rtol=0,atol=1e-13)
    flat=np.load(folder.parent/'initial_conditioning.npy' if folder.name=='initial' else folder/'conditioning.npy',allow_pickle=False)
    assert state_with_component(virtual,names,edges,flat)==snapshot['model_state']
    return snapshot,groups,cells


def loss(groups,g):
    row=groups['raw'][g['source']+'/'+g['profile']]
    field={'MSE':'MSE','SSIM_loss':'SSIM','ArcFace_loss':'ArcFace_observed_fixed','feature_MSE':'landmark_high_frequency_MSE'}[g['metric']]
    v=row[field];return 1-v if g['metric'] in ['SSIM_loss','ArcFace_loss'] else v


def main():
    imported=read(ANALYSIS/'import.json');p=read(RETURN/'protocol.json');old=read(PARENT/'protocol.json')
    assert imported['complete'] and imported['protocol_sha256']==PIN==sha(RETURN/'protocol.json')
    manifest=read(RETURN/'export_manifest.json');assert not manifest['complete_pilot'] and manifest['accepted_updates']==manifest['optimizer_updates']==0
    assert all(sha(RETURN/n)==h for n,h in manifest['files'].items())
    assert all(sha(PARENT/n)==h for n,h in p['parent_required_sha256'].items())
    out=RETURN/'outputs/conditioning_response_v1';run=read(out/'run_results.json') if (out/'run_results.json').exists() else None
    failure=read(out/'failure.json') if (out/'failure.json').exists() else None
    assert manifest['complete_diagnostic']==bool(run) and manifest['failure_present']==bool(failure)
    if not (out/'environment.json').exists():
        assert failure and not run and int((RETURN/'review_exit_code.txt').read_text().strip())!=0
        if failure.get('stage')=='pre-model storage preflight':
            assert failure['available_free_bytes']<failure['required_free_bytes']==p['minimum_free_bytes']
            assert all(v==0 for v in failure['counts'].values())
        write(ANALYSIS/'failure_audit.json',dict(complete=True,scope='Pre-model failure only',failure=failure,model_qualified=False,goal_complete=False));return
    env=read(out/'environment.json');assert env['GPU']=='NVIDIA L4' and env['host'].split('.')[0]=='forensic-dgp-thesis' and env['manual_tmux']
    assert env['torch']==p['expected_torch'] and not env['optimizer_constructed'] and not env['new_direction_solved']
    assert not env['precision']['matmul_tf32'] and not env['precision']['cudnn_tf32'] and not env['precision']['cudnn_benchmark'] and env['precision']['cudnn_deterministic']
    virtual=schema(RETURN);names=p['conditioning_names'];edges=np.cumsum([0]+[int(np.prod(p['conditioning_shapes'][n])) for n in names])
    assert state_hash(virtual)==p['initial_model_state']==env['initial_model_state']
    assert state_hash({n:v for n,v in virtual.items() if n not in names})==env['frozen_state']
    initial=np.load(out/'initial_conditioning.npy',allow_pickle=False);expected=np.concatenate([virtual[n].numpy().reshape(-1) for n in names])
    np.testing.assert_array_equal(initial,expected)
    direction=np.load(RETURN/'saved_direction.npy',allow_pickle=False);assert sha(RETURN/'saved_direction.npy')==p['saved_direction_sha256']
    assert direction.dtype==np.float32 and direction.shape==initial.shape and np.isfinite(direction).all()
    snapshots={};cells={}
    for label in ['initial']+[r['label'] for r in p['step_plan']]+['rollback']:
        folder=out/label
        if (folder/'snapshot.json').exists():
            s,g,c=audit_snapshot(folder,p,old,virtual,names,edges);snapshots[label]=dict(snapshot=s,groups=g);cells[label]=c
    previous=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_return/outputs/raw_conditioning_learning_v1_r1/round00/gradients.json')
    previous_base=OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_return/outputs/raw_conditioning_learning_v1_r1'
    repeats=[];stored_vectors=[]
    for repeat in range(2):
        folder=out/f'repeat{repeat:02d}'
        if not (folder/'gradients.json').exists():continue
        record=read(folder/'gradients.json');assert record['repeat']==repeat and record['snapshot']=='initial' and record['model_state']==p['initial_model_state']
        assert record['reference_batch_order']==[b['begin'] for b in p['reference_blocks']]
        assert len(record['forward_repeatability'])==20 and all(r['exact_initial_forward'] and r['maximum_absolute_difference']==0 and r['raw_bytes_sha256']==r['baseline_bytes_sha256'] for r in record['forward_repeatability'])
        assert len(record['groups'])==32
        vectors=[];parity=[]
        for i,(row,g) in enumerate(zip(record['groups'],p['raw_gradient_groups'])):
            assert row['key']=='RAW/'+g['key'] and row['metric_stage']=='RAW' and all(row[k]==g[k] for k in g if k!='key')
            path=out/row['file'];assert sha(path)==row['sha256'];v=np.load(path,allow_pickle=False,mmap_mode='r')
            assert v.dtype==np.float64 and v.shape==direction.shape and np.isfinite(v).all()
            assert np.isclose(np.linalg.norm(v),row['gradient_norm'],rtol=1e-10,atol=1e-12)
            assert np.isclose(float(v@direction.astype(np.float64)),row['existing_direction_dot'],rtol=1e-10,atol=1e-12)
            c=row['contributions'];assert len(c)==10
            blocks=[b for b in p['reference_blocks'] if b['source']==g['source']]
            assert [r['begin'] for r in c]==[b['begin'] for b in blocks] and [r['reference'] for r in c]==[b['reference'] for b in blocks]
            assert np.isclose(np.mean([r['value'] for r in c]),row['value'],rtol=0,atol=1e-12)
            assert np.isclose(np.mean([r['existing_direction_dot'] for r in c]),row['existing_direction_dot'],rtol=1e-9,atol=1e-12)
            profiles=['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
            query=15 if g['metric']=='feature_MSE' else profiles.index(g['profile'])*3+['MSE','SSIM_loss','ArcFace_loss'].index(g['metric'])
            for r,b in zip(c,blocks):
                assert r['query']==query and r['key']==row['key'] and r['gradient_norm']>0
                assert r['numeric_alignment_bound']==(1e-6 if query<15 and query%3 else 1e-13)
                assert abs(r['value']-r['scientific_forward_loss'])<=r['numeric_alignment_bound']
                rows={r['id']:r for r in snapshots['initial']['snapshot']['rows']}
                if query<15:
                    score=rows[b['case_ids'][query//3]]['raw'];forward=[score['MSE'],1-score['SSIM'],1-score['ArcFace_observed_fixed']][query%3]
                else:forward=float(np.mean([rows[cid]['raw']['landmark_high_frequency_MSE'] for cid in b['case_ids'][1:]]))
                assert r['scientific_forward_loss']==forward
                if 'sample_file' in r:
                    sample=out/r['sample_file'];assert sha(sample)==r['sample_sha256'];a=np.load(sample,allow_pickle=False)
                    assert a.dtype==np.float32 and a.shape==direction.shape and np.isfinite(a).all()
                    assert hashlib.sha256(a.tobytes()).hexdigest()==r['gradient_vector_sha256']
                    assert np.isclose(np.linalg.norm(a.astype(np.float64)),r['gradient_norm'],rtol=1e-10)
                    assert np.isclose(float(a.astype(np.float64)@direction.astype(np.float64)),r['existing_direction_dot'],rtol=1e-10,atol=1e-12)
            prior_row=previous['groups'][i];assert prior_row['key']==row['key'] and sha(previous_base/prior_row['file'])==prior_row['sha256']
            prior=np.load(previous_base/prior_row['file'],allow_pickle=False,mmap_mode='r')
            parity.append(dict(key=row['key'],comparison='previous stopped R1',**cross_run_report(v,prior)))
            if repeat:
                parity.append(dict(key=row['key'],comparison='repeat00 versus repeat01',**cross_run_report(v,stored_vectors[0][i])))
            vectors.append(v)
        assert len(vectors)==32
        timing=read(folder/'timing_projection.json');assert timing['cap_seconds']==600 and timing['projection_seconds']<=600
        assert np.isclose(timing['projection_seconds'],max(timing['completed_batch_seconds'])*20*1.5,rtol=0,atol=1e-9)
        repeats.append(dict(repeat=repeat,original_unchanged_parity_bounds=parity,all_cross_run_checks_passed=all(r['original_parity_check_pass'] for r in parity)))
        stored_vectors.append(vectors)
    responses=[];receipt_checks=[];all37norm=float(np.sqrt(sum(float(np.square(virtual[n].numpy().astype(np.float64)).sum()) for n in p['all37_learning_names'])))
    assert np.isclose(all37norm,env['initial_all37_norm'],rtol=1e-12)
    for step in p['step_plan']:
        folder=out/step['label']
        if not (folder/'response.json').exists():continue
        record=read(folder/'response.json');assert record['nominal_rate']==step['rate'] and record['actual_float32_rate']==float(np.float32(step['rate']))
        assert sha(folder/'conditioning.npy')==record['conditioning_sha256'] and sha(folder/'snapshot.json')==record['snapshot_sha256']
        v=np.load(folder/'conditioning.npy',allow_pickle=False);np.testing.assert_array_equal(v,initial+np.float32(step['rate'])*direction)
        distance=float(np.linalg.norm(v.astype(np.float64)-initial.astype(np.float64))/all37norm)
        assert 0<distance<=.001 and np.isclose(distance,record['cumulative_relative_parameter_distance'],rtol=1e-10)
        checked=decision(snapshots['initial']['groups'],snapshots['initial']['groups'],snapshots[step['label']]['groups'])
        receipt_checks.append(validate_receipt(checked,record['scientific_decision_information_only'],(snapshots['initial']['snapshot']['groups'],snapshots['initial']['snapshot']['groups'],snapshots[step['label']]['snapshot']['groups'])))
        assert record['accepted_update'] is False and record['candidate_selection'] is False and record['diagnostic_not_qualification']
        responses.append(record)
    central=[]
    if len(stored_vectors)==2:
        records=read(out/'repeat00/gradients.json')['groups'];records1=read(out/'repeat01/gradients.json')['groups']
        for pair in range(3):
            minus,plus=f'minus{pair:02d}',f'plus{pair:02d}'
            if minus not in snapshots or plus not in snapshots:continue
            dm=np.load(out/minus/'conditioning.npy',allow_pickle=False).astype(np.float64)-initial.astype(np.float64)
            dp=np.load(out/plus/'conditioning.npy',allow_pickle=False).astype(np.float64)-initial.astype(np.float64)
            magnitude=abs(p['step_plan'][2*pair]['rate'])
            for i,g in enumerate(records):
                before=loss(snapshots['initial']['groups'],g);a,b=loss(snapshots[plus]['groups'],g),loss(snapshots[minus]['groups'],g)
                # Preserve cached scientific controls. The fresh initial scalar
                # mismatch is included in the resolution floor, never hidden.
                spread=max(abs(records[i]['value']-records1[i]['value']),abs(records[i]['value']-before),abs(records1[i]['value']-before))
                pp=float(stored_vectors[0][i]@dp);pm=float(stored_vectors[0][i]@dm)
                central.append(dict(pair=pair,key=g['key'],metric=g['metric'],SSIM_backward_explicitly_approximate=g['metric']=='SSIM_loss',
                    **response_receipt(before,a,b,magnitude,pp,pm,spread)))
    if run:
        assert not failure and int((RETURN/'review_exit_code.txt').read_text().strip())==0 and len(responses)==6 and len(repeats)==2
        assert run['counts']==dict(B=200,recognizer=360,gradient_queries=640,temporary_displacements=6,optimizer_updates=0,accepted_updates=0)
        assert len(snapshots)==8 and len(central)==96
        assert run['initial_and_restored_state']==snapshots['rollback']['snapshot']['model_state']==p['initial_model_state']
        np.testing.assert_array_equal(initial,np.load(out/'rollback/conditioning.npy',allow_pickle=False))
        assert snapshots['initial']['groups']==snapshots['rollback']['groups']
        with np.load(out/'initial_RNG.npz',allow_pickle=False) as a,np.load(out/'restored_RNG.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[n],b[n]) for n in a.files)
        assert run['RNG_restored'] and run['original_and_recognizer_unchanged'] and run['initial_and_rollback_outputs_exact']
        assert run['seconds']<p['measurement_seconds'] and run['peak_VRAM']<=p['peak_VRAM_bytes'] and run['peak_RSS']<=p['peak_RSS_bytes']
        assert run['full_epochs']==run['native_pixels_used']==run['final_identity_pixels']==0 and not run['model_qualified']
    else:
        assert failure and int((RETURN/'review_exit_code.txt').read_text().strip())!=0
        if failure.get('fresh_restored_state_hash') is not None:assert failure['fresh_restored_state_hash']==p['initial_model_state'] and failure['rollback_error'] is None
    gallery=ANALYSIS/'sheets';gallery.mkdir();visual=[]
    for block in p['reference_blocks']:
        labels=list(cells);sheet=Image.new('RGB',(256*(2+len(labels)),5*280),(20,20,20));draw=ImageDraw.Draw(sheet)
        for row,cid in enumerate(block['case_ids']):
            c=next(c for c in old['cases'] if c['id']==cid);r=next(r for r in old['references'] if r['id']==c['source_person_or_reference'])
            columns=[('input',image(PARENT/c['input'])),('paired TRAIN target',image(PARENT/r['target']))]+[(n,cells[n][cid]) for n in labels]
            for col,(name,png) in enumerate(columns):sheet.paste(Image.fromarray(png),(256*col,280*row+24));draw.text((256*col+3,280*row+4),name,fill='white')
        path=gallery/f"b{block['begin']//5:03d}.png";sheet.save(path)
        visual.append(dict(file=path.relative_to(ROOT).as_posix(),sha256=sha(path),cells=5*(2+len(labels)),direct_visual_review_pending=True))
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256'];assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/('independent_audit.json' if run else 'failure_audit.json'),dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),protocol_sha256=PIN,import_sha256=sha(ANALYSIS/'import.json'),full_diagnostic_returned=bool(run),failure=failure,
        completed_snapshots=list(snapshots),repeat_vector_checks=repeats,central_response_comparisons=central,exact_decision_receipt_checks=receipt_checks,
        original_parity_bounds_not_relaxed=True,underlying_scientific_metric_replay_bound_unchanged=1e-13,
        diagnostic_five_percent_central_comparison_is_not_image_preservation_or_model_qualification=True,
        actual_raw_PNG_state_protected_pixel_and_saved_embedding_replays=True,neural_gradients_and_embeddings_not_locally_reexecuted=True,
        accepted_updates=0,optimizer_updates=0,all_planned_visual_sheets=visual,all_planned_sheets_actually_reviewed=False,
        local_neural_gradient_or_training_calls=0,full_TRAIN_native_DEV_final_and_covering_scope_pending=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,diagnostic_completed=bool(run),snapshots=len(snapshots),sheets_to_review=len(visual),accepted_updates=0,model_qualified=False))


if __name__=='__main__':main()

"""Saved-output, derivative arithmetic and component-state audit; no models."""
from datetime import datetime,timezone
import hashlib
import sys
import numpy as np
from PIL import Image,ImageDraw
import torch

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT,OUT,RETURN,ANALYSIS,PARENT,PIN,read,sha,write
from audit_cctv_dgp_joint_raw_png_forward_v1_return import image,support,metrics,delivered,grouped,state_hash
from cctv_dgp_conditioning_learning_v1_r4_contract import decision,bounded_rate
from conditioning_decision_receipt_check_v1 import validate_receipt


def schema(packet):
    torch.set_grad_enabled(False)
    state=torch.load(PARENT/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
    virtual={'live.'+n:v for n,v in state.items()}
    with np.load(packet/'readouts.npz',allow_pickle=False) as arrays:
        for size in [16,32,64,128,256]:
            for term in ['weight','bias']:
                v=torch.from_numpy(arrays[f'f{size}_{term}'])
                virtual[f'live.fusions.{size}.{term}']=v
                virtual[f'anchor_fusions.{size}.{term}']=v
    for n,v in state.items():
        if n.startswith('style_conditioner.'):
            virtual['anchor_style_conditioner.'+n.removeprefix('style_conditioner.')]=v
        if n.startswith('spatial_conditioners.'):
            virtual['anchor_spatial_conditioners.'+n.removeprefix('spatial_conditioners.')]=v
    return virtual


def state_with_component(virtual,names,edges,flat):
    # Hash a byte schema; never construct or displace a neural model.
    values=dict(virtual)
    for i,n in enumerate(names):values[n]=torch.from_numpy(flat[edges[i]:edges[i+1]].reshape(tuple(values[n].shape)))
    return state_hash(values)


def audit_gradient_round(folder,p,initial_round=False):
    record=read(folder/'gradients.json');keys=[stage+'/'+g['key'] for stage in ['RAW','PNG'] for g in p['raw_gradient_groups']]
    expected_groups=[dict(g,key=stage+'/'+g['key'],metric_stage=stage) for stage in ['RAW','PNG'] for g in p['raw_gradient_groups']]
    assert [r['key'] for r in record['groups']]==keys
    vectors=[];norms=[];losses=[]
    raw_origin=OUT/'cctv_dgp_live_conditioning_v1_r1_return/outputs/live_conditioning_v1_r1'
    origin=read(raw_origin/'run_results.json')
    origin_p=read(OUT/'cctv_dgp_live_conditioning_v1_r1_return/protocol.json')
    origin_edges=np.cumsum([0]+[int(np.prod(origin_p['selected_parameter_shapes'][n])) for n in origin_p['learning_names']])
    for index,row in enumerate(record['groups']):
        assert all(row[k]==expected_groups[index][k] for k in expected_groups[index])
        path=RETURN/'outputs/conditioning_learning_v1_r4'/row['file']
        assert sha(path)==row['sha256']
        v=np.load(path,allow_pickle=False,mmap_mode='r')
        assert v.dtype==np.float64 and v.shape==(p['conditioning_parameters'],) and np.isfinite(v).all()
        norm=float(np.linalg.norm(v));assert norm>0 and np.isclose(norm,row['gradient_norm'],rtol=1e-10,atol=1e-12)
        c=row['contributions'];assert len(c)==10 and len({r['reference'] for r in c})==10
        assert all(r['key']==row['key'] and np.isfinite(r['value']) and r['gradient_norm']>0 for r in c)
        selected_blocks=[b for b in p['reference_blocks'] if b['source']==row['source']]
        assert [v['begin'] for v in c]==[b['begin'] for b in selected_blocks]
        assert [v['reference'] for v in c]==[b['reference'] for b in selected_blocks]
        profiles=['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
        wanted=15 if row['metric']=='feature_MSE' else profiles.index(row['profile'])*3+['MSE','SSIM_loss','ArcFace_loss'].index(row['metric'])
        assert all(v['query']==wanted for v in c)
        assert np.isclose(np.mean([r['value'] for r in c]),row['value'],rtol=0,atol=1e-12)
        for item in c:
            assert abs(item['value']-item['scientific_forward_loss'])<=item['numeric_alignment_bound']
            if 'sample_file' in item:
                sample=RETURN/'outputs/conditioning_learning_v1_r4'/item['sample_file']
                assert sha(sample)==item['sample_sha256']
                a=np.load(sample,allow_pickle=False)
                assert a.dtype==np.float32 and a.shape==(p['conditioning_parameters'],) and np.isfinite(a).all()
                assert hashlib.sha256(a.tobytes()).hexdigest()==item['gradient_vector_sha256']
                assert np.isclose(np.linalg.norm(a.astype(np.float64)),item['gradient_norm'],rtol=1e-10)
        if initial_round and row['metric_stage']=='RAW':
            old=origin['groups'][index]
            assert row['key']=='RAW/'+old['key'] and sha(raw_origin/old['file'])==old['sha256']
            full=np.load(raw_origin/old['file'],allow_pickle=False,mmap_mode='r')
            expected=np.concatenate([full[origin_edges[i]:origin_edges[i+1]] for i,n in enumerate(origin_p['learning_names']) if n in p['conditioning_names']])
            np.testing.assert_allclose(v,expected,rtol=1e-4,atol=1e-8)
            assert np.isclose(np.linalg.norm(v),np.linalg.norm(expected),rtol=1e-4,atol=1e-10)
        vectors.append(v);norms.append(norm);losses.append(row['value'])
    norms=np.asarray(norms);losses=np.asarray(losses);cert=record['certificate']
    direction=np.load(folder/'direction.npy',allow_pickle=False)
    assert sha(folder/'direction.npy')==record['direction_sha256'] and direction.dtype==np.float32 and direction.shape==(p['conditioning_parameters'],)
    dual=np.asarray(cert['dual_multipliers']);gram=np.zeros((64,64));dots=np.zeros(64)
    for begin in range(0,p['conditioning_parameters'],32768):
        block=np.vstack([v[begin:begin+32768]/norm for v,norm in zip(vectors,norms)])
        gram+=np.einsum('ij,kj->ik',block,block,optimize=True)
        reconstructed=np.einsum('i,ij->j',-dual,block,optimize=True).astype(np.float32)
        np.testing.assert_allclose(reconstructed,direction[begin:begin+32768],rtol=1e-6,atol=1e-8)
        dots+=np.einsum('ij,j->i',block*norms[:,None],direction[begin:begin+32768].astype(np.float64),optimize=True)
    np.testing.assert_allclose(gram,cert['normalized_Gram'],rtol=0,atol=1e-10)
    fractions=np.array([1. if r['metric']=='feature_MSE' else .1 for r in record['groups']])
    target=fractions*losses/norms;slack=gram@dual-target
    assert dual.min()>=0 and slack.min()>-1e-9 and np.max(abs(slack*dual))<1e-9
    assert np.linalg.eigvalsh(gram)[0]>cert['rank_bound'] and cert['all64_loss_dimensions_retained'] and cert['no_ridge_or_loss_removal']
    np.testing.assert_allclose(dots,cert['direction_dots'],rtol=1e-8,atol=1e-10)
    assert np.all(-dots/losses>=fractions-1e-6)
    timing=read(folder/'timing_projection.json')
    assert timing['cap_seconds']==600 and timing['projection_seconds']<=600
    assert np.isclose(timing['projection_seconds'],max(timing['completed_batch_seconds'])*20*1.5,rtol=0,atol=1e-9)
    return dict(round=record['round'],gradient_groups=64,full_vector_KKT_and_float32_dots_checked=True,
        initial_RAW_conditioning_derivatives_match_prior_audited_manual_measurement=initial_round,
        neural_mean_contributions_not_independently_reexecuted=True)


def audit_snapshot(folder,p,old,virtual,names,edges):
    snapshot=read(folder/'snapshot.json');refs={r['id']:r for r in old['references']}
    rows=[];cells={}
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    stored={r['id']:r for r in snapshot['rows']};assert set(stored)==set(p['case_ids'])
    for block in p['reference_blocks']:
        begin=block['begin'];cs=old['cases'][begin:begin+5];stem=f'b{begin//5:03d}'
        raw=np.load(folder/(stem+'.npy'),allow_pickle=False)
        assert raw.dtype==np.float32 and raw.shape==(5,256,256,3) and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
        baseline,expected=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/{stem}.npz',block['case_ids'])
        with np.load(folder/(stem+'_embeddings.npz'),allow_pickle=False) as archive:
            assert set(archive.files)=={'raw','png'};emb={k:archive[k] for k in archive.files}
        proof=next(r for r in snapshot['blocks'] if r['begin']==begin)
        assert sha(folder/(stem+'.npy'))==proof['raw_sha256'] and sha(folder/(stem+'_embeddings.npz'))==proof['embeddings_sha256']
        for stage,index in [('raw',0),('png',1)]:
            assert emb[stage].dtype==np.float32 and emb[stage].shape==(15,512) and np.isfinite(emb[stage]).all()
            np.testing.assert_allclose(emb[stage][5:10],expected[:,index],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(emb[stage][10:15],expected[:,2],rtol=1e-6,atol=1e-6)
        for i,c in enumerate(cs):
            ref=refs[c['source_person_or_reference']]
            camera,target,mask=image(PARENT/c['input']),image(PARENT/ref['target']),image(PARENT/ref['observed'],'L')>0
            patch=support(mask,c['landmarks5_canvas_xy']);png=image(folder/'png'/(c['id']+'.png'))
            assert np.array_equal(delivered(raw[i],camera,mask),png)
            assert np.array_equal(raw[i][~mask],(camera.astype(np.float32)/np.float32(255))[~mask]) and np.array_equal(png[~mask],camera[~mask])
            if folder.name=='initial':assert np.array_equal(raw[i],baseline[i])
            shift=(raw[i]-baseline[i])[mask].astype(np.float64).mean(0)
            mean=np.clip(baseline[i].astype(np.float64)+shift,0,1).astype(np.float32);mean[~mask]=(camera.astype(np.float32)/np.float32(255))[~mask]
            mean_png=delivered(mean,camera,mask)
            row=dict(id=c['id'],source=c['source'],profile=c['profile'],reference=c['source_person_or_reference'])
            assert all(row[k]==stored[c['id']][k] for k in row)
            for stage,actual,mean_actual in [('raw',raw[i],mean),('png',png.astype(np.float32)/np.float32(255),mean_png.astype(np.float32)/np.float32(255))]:
                v=emb[stage]
                j=0 if stage=='raw' else 1
                score_vector=expected[i,j] if folder.name=='initial' else v[i]
                target_vector=expected[i,2] if folder.name=='initial' else v[10+i]
                row[stage]=metrics(actual,target,mask,patch,score_vector,target_vector,mean_actual,prediction8=png if stage=='png' else None)
                for key,value in row[stage].items():
                    other=stored[c['id']][stage][key]
                    if value is None or isinstance(value,bool):assert value==other
                    else:assert np.isclose(value,other,rtol=0,atol=1e-13),(folder.name,c['id'],stage,key)
            rows.append(row);cells[c['id']]=png
    group={stage:grouped(rows,stage) for stage in ['raw','png']}
    for stage,groups in group.items():
        for key,values in groups.items():
            for metric,value in values.items():assert np.isclose(value,snapshot['groups'][stage][key][metric],rtol=0,atol=1e-13)
    if folder.name=='initial':flat=np.load(folder.parent/'initial_conditioning.npy',allow_pickle=False)
    else:flat=np.load(folder/'conditioning.npy',allow_pickle=False)
    assert state_with_component(virtual,names,edges,flat)==snapshot['model_state']
    return snapshot,group,cells


def main():
    imported=read(ANALYSIS/'import.json');assert imported['complete'] and imported['protocol_sha256']==PIN
    p,old=read(RETURN/'protocol.json'),read(PARENT/'protocol.json')
    manifest=read(RETURN/'export_manifest.json')
    assert all(sha(RETURN/n)==h for n,h in manifest['files'].items())
    assert all(sha(PARENT/n)==h for n,h in p['parent_required_sha256'].items())
    out=RETURN/'outputs/conditioning_learning_v1_r4';failure=read(out/'failure.json') if (out/'failure.json').exists() else None
    assert manifest['failure_present']==bool(failure)
    if not (out/'environment.json').exists():
        if failure and failure.get('stage')=='pre-model storage preflight':
            assert failure['available_free_bytes']<failure['required_free_bytes']==p['minimum_free_bytes']
            assert failure['neural_calls']==failure['gradient_queries']==failure['accepted_updates']==failure['parameter_proposals']==0
        assert not manifest['complete_pilot'] and int((RETURN/'review_exit_code.txt').read_text().strip())!=0
        write(ANALYSIS/'failure_audit.json',dict(complete=True,checker_sha256=sha(__file__),failure=failure,
            scope='Startup failure only; no complete model-state/output claim',model_qualified=False,goal_complete=False))
        return
    env=read(out/'environment.json');assert env['GPU']=='NVIDIA L4' and env['host'].split('.')[0]=='forensic-dgp-thesis' and env['manual_tmux']
    virtual=schema(RETURN);names=p['conditioning_names'];edges=np.cumsum([0]+[int(np.prod(p['conditioning_shapes'][n])) for n in names])
    assert state_hash(virtual)==p['initial_model_state']==env['initial_model_state']
    assert state_hash({n:v for n,v in virtual.items() if n not in names})==env['frozen_state']
    initial=np.load(out/'initial_conditioning.npy',allow_pickle=False)
    expected=np.concatenate([virtual[n].numpy().reshape(-1) for n in names])
    np.testing.assert_array_equal(initial,expected)
    all37norm=float(np.sqrt(sum(float(np.square(virtual[n].numpy().astype(np.float64)).sum()) for n in p['all37_learning_names'])))
    assert np.isclose(all37norm,env['initial_all37_norm'],rtol=1e-12)
    snapshots={};cells={}
    for folder in [out/'initial']+sorted(f for f in out.glob('r??_trial??') if (f/'snapshot.json').exists() and (f/'conditioning.npy').exists()):
        if not (folder/'snapshot.json').exists():continue
        snapshot,groups,images=audit_snapshot(folder,p,old,virtual,names,edges)
        snapshots[folder.name]=dict(snapshot=snapshot,groups=groups);cells[folder.name]=images
    gradient_audits=[]
    for folder in sorted(out.glob('round??')):
        if (folder/'gradients.json').exists():
            meta=read(folder/'gradients.json')
            assert meta['snapshot'] in snapshots and meta['model_state']==snapshots[meta['snapshot']]['snapshot']['model_state']
            gradient_audits.append(audit_gradient_round(folder,p,folder.name=='round00'))
    accepted=initial.copy();current='initial';accepted_trials=[];trial_count=0
    receipt_checks=[]
    for path in sorted(out.glob('r??_trial??/decision.json')):
        trial=read(path);label=trial['label'];trial_count+=1
        assert trial['before_snapshot']==current and sha(out/label/'snapshot.json')==trial['snapshot_sha256']
        assert sha(out/label/'conditioning.npy')==trial['conditioning_sha256']
        direction_path=out/f"round{trial['round']:02d}/direction.npy"
        assert sha(direction_path)==trial['round_direction_sha256']
        direction=np.load(direction_path,allow_pickle=False)
        ceiling=bounded_rate(initial,accepted,direction,all37norm)
        assert np.isclose(ceiling,trial['rate_ceiling'],rtol=1e-10) and trial['attempt'] in [0,1,2]
        rate=np.float32(ceiling*[1.,.5,.25][trial['attempt']]);assert float(rate)==trial['rate']
        actual=np.load(out/label/'conditioning.npy',allow_pickle=False)
        np.testing.assert_array_equal(actual,accepted+rate*direction)
        distance=float(np.linalg.norm(actual.astype(np.float64)-initial.astype(np.float64))/all37norm)
        assert distance<=.001 and np.isclose(distance,trial['cumulative_relative_parameter_distance'],rtol=1e-10)
        checked=decision(snapshots['initial']['groups'],snapshots[current]['groups'],snapshots[label]['groups'])
        verified=validate_receipt(checked,trial['decision'],(
            snapshots['initial']['snapshot']['groups'],snapshots[current]['snapshot']['groups'],
            snapshots[label]['snapshot']['groups']))
        receipt_checks.append(dict(trial=label,**verified))
        if checked['accept_step']:
            accepted=actual.copy();current=label;accepted_trials.append(label)
    assert trial_count<=6 and len(accepted_trials)<=2
    for path in sorted(out.glob('accepted??.json')):
        c=read(path);i=c['accepted_update'];assert 1<=i<=len(accepted_trials) and c['trial']==accepted_trials[i-1]
        checkpoint=path.with_suffix('.npz');assert sha(checkpoint)==c['checkpoint_sha256']
        with np.load(checkpoint,allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['conditioning'],np.load(out/c['trial']/'conditioning.npy',allow_pickle=False))
            assert arrays['torch_CPU_RNG'].dtype==np.uint8 and any(n.startswith('torch_CUDA_RNG_') for n in arrays.files)
        assert c['model_state']==snapshots[c['trial']]['snapshot']['model_state'] and c['frozen_state']==env['frozen_state']
    run=read(out/'run_results.json') if (out/'run_results.json').exists() else None
    if run:
        assert manifest['complete_pilot'] and not failure and int((RETURN/'review_exit_code.txt').read_text().strip())==0
        assert run['counts']==dict(B=20+20*len(gradient_audits)+20*trial_count,recognizer=40+40*len(gradient_audits)+40*trial_count,
            gradient_queries=640*len(gradient_audits),backwards=0,accepted_updates=len(accepted_trials),parameter_proposals=trial_count)
        assert run['counts']['accepted_updates']==manifest['accepted_updates'] and 0<len(accepted_trials)<=2
        assert run['final_snapshot']==current and run['final_model_state']==snapshots[current]['snapshot']['model_state']
        assert decision(snapshots['initial']['groups'],snapshots['initial']['groups'],snapshots[current]['groups'])['one_percent_checkpoint_pass']
        assert run['frozen_state_unchanged'] and run['buffers_unchanged'] and run['original_and_recognizer_unchanged']
        assert run['full_epochs']==run['native_pixels_used']==run['final_identity_pixels']==0
        assert run['seconds']<p['worker_seconds'] and run['peak_VRAM']<=p['peak_VRAM_bytes'] and run['peak_RSS']<=p['peak_RSS_bytes']
    else:assert not manifest['complete_pilot'] and int((RETURN/'review_exit_code.txt').read_text().strip())!=0
    if (out/'loss_alignment_failure.json').exists():
        alignment=read(out/'loss_alignment_failure.json')
        assert failure and failure['message']=='Scientific/surrogate loss alignment differs'
        assert alignment['metric_stage'] in ['RAW','PNG'] and 0<=alignment['query']<16
        assert abs(alignment['actual_forward_value']-alignment['expected_scientific_value'])==alignment['absolute_difference']
        assert alignment['absolute_difference']>alignment['alignment_bound']
        block=next(b for b in p['reference_blocks'] if b['begin']==alignment['begin'])
        assert block['reference']==alignment['reference'] and alignment['snapshot'] in snapshots
        expected_stage=alignment['metric_stage'].lower();query=alignment['query']
        rows={r['id']:r for r in snapshots[alignment['snapshot']]['snapshot']['rows']}
        if query<15:
            score=rows[block['case_ids'][query//3]][expected_stage]
            expected=[score['MSE'],1-score['SSIM'],1-score['ArcFace_observed_fixed']][query%3]
        else:expected=float(np.mean([rows[cid][expected_stage]['landmark_high_frequency_MSE'] for cid in block['case_ids'][1:]]))
        assert alignment['expected_scientific_value']==expected
        assert alignment['raw_helper_sha256']==sha(RETURN/'conditioning_raw_losses.py')
        assert alignment['PNG_helper_sha256']==sha(RETURN/'conditioning_losses.py')
        assert alignment['candidate_state']==snapshots[alignment['snapshot']]['snapshot']['model_state']
        write(ANALYSIS/'alignment_scalar_audit.json',dict(complete=True,reported_alignment=alignment,
            scientific_snapshot_value_independently_checked=True,actual_neural_loss_not_reexecuted_locally=True))
    # Every complete candidate snapshot gets a sheet for every reference,
    # including rejected trials. Viewing these images is a later human/agent
    # task, never inferred from successful hash checks.
    gallery=ANALYSIS/'sheets';gallery.mkdir()
    visual=[]
    for block in p['reference_blocks']:
        labels=list(cells);sheet=Image.new('RGB',(256*(2+len(labels)),5*280),(20,20,20));draw=ImageDraw.Draw(sheet)
        for row,cid in enumerate(block['case_ids']):
            c=next(c for c in old['cases'] if c['id']==cid);ref=next(r for r in old['references'] if r['id']==c['source_person_or_reference'])
            columns=[('input',image(PARENT/c['input'])),('paired TRAIN target',image(PARENT/ref['target']))]+[(label,cells[label][cid]) for label in labels]
            for col,(label,png) in enumerate(columns):
                sheet.paste(Image.fromarray(png),(256*col,280*row+24));draw.text((256*col+3,280*row+4),label,fill='white')
        path=gallery/f"b{block['begin']//5:03d}.png";sheet.save(path)
        visual.append(dict(file=path.relative_to(ROOT).as_posix(),sha256=sha(path),cells=5*(2+len(labels)),direct_visual_review_pending=True))
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/('independent_audit.json' if run else 'failure_audit.json'),dict(complete=True,
        UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),protocol_sha256=PIN,
        import_sha256=sha(ANALYSIS/'import.json'),full_pilot_returned=bool(run),failure=failure,
        original_checker_sha256='30f8785e8c519624760d6c41ffcf79afeee2a643b0409e5f9b876c4eea780e6e',
        original_checker_stop_sha256=sha(ANALYSIS/'original_checker_stop.json'),
        receipt_helper_sha256=sha(ROOT/'scripts/conditioning_decision_receipt_check_v1.py'),
        exact_saved_receipt_and_independent_gate_checks=receipt_checks,
        underlying_scientific_metric_replay_bound_unchanged=1e-13,
        completed_snapshots_audited=list(snapshots),gradient_audits=gradient_audits,completed_trials=trial_count,
        accepted_trials_verified=accepted_trials,actual_update_count_reported=manifest['accepted_updates'],
        all_complete_RAW_PNG_rows_masks_delivery_and_saved_embeddings_checked=True,
        selected_component_states_reconstructed_without_neural_models=True,all_planned_visual_sheets=visual,
        all_planned_visual_sheets_actually_reviewed=False,derivatives_and_embeddings_not_neurally_reexecuted_locally=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        full_TRAIN_native_DEV_independent_final_and_covering_scope_pending=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,pilot_completed=bool(run),accepted_trials_verified=len(accepted_trials),sheets_to_review=len(visual),model_qualified=False))


if __name__=='__main__':main()

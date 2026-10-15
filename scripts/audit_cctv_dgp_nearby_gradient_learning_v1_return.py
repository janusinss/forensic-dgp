"""Audit saved nearby-state training without constructing or running neural models."""
from datetime import datetime, timezone
import hashlib
import numpy as np
import torch
from PIL import Image, ImageDraw

from import_cctv_dgp_nearby_gradient_learning_v1_return import ANALYSIS, OUT, PARENT, PIN, RESPONSE, RETURN, ROOT, read, sha, write
from audit_cctv_dgp_raw_conditioning_learning_v1_r1_return import schema, state_with_component
from cctv_dgp_nearby_snapshot_audit_v1 import audit_snapshot
from audit_cctv_dgp_joint_raw_png_forward_v1_return import state_hash, image
from cctv_dgp_nearby_gradient_certificate_v1 import audit_certificate
from cctv_dgp_raw_conditioning_learning_v1_r1_contract import decision, bounded_rate
from raw_conditioning_decision_receipt_check_v1_r1 import validate_receipt
from cctv_dgp_RGB_gradient_parity_v1 import report as parity_report


def expected_query(row):
    profiles=['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
    return 15 if row['metric']=='feature_MSE' else profiles.index(row['profile'])*3+['MSE','SSIM_loss','ArcFace_loss'].index(row['metric'])


def retained_initial_gradient(index,key):
    origin=RESPONSE/'outputs/conditioning_response_v1'
    previous=read(origin/'repeat00/gradients.json')['groups'][index]
    assert previous['key']==key
    path=origin/previous['file']
    assert path.resolve().is_relative_to(origin.resolve()) and sha(path)==previous['sha256']
    vector=np.load(path,allow_pickle=False,mmap_mode='r')
    assert vector.dtype==np.float64 and vector.shape==(2580864,) and np.isfinite(vector).all()
    return previous,vector


def audit_site(out, site, p):
    folder=out/('gradients_'+site);record=read(folder/'gradients.json')
    assert record['site']==record['snapshot']==site and len(record['groups'])==32
    assert record['model_state']==read(out/site/'snapshot.json')['model_state']
    rows={r['id']:r for r in read(out/site/'snapshot.json')['rows']}
    vectors=[];parity=[]
    for i,(row,expected) in enumerate(zip(record['groups'],p['raw_gradient_groups'])):
        assert row['site']==site and row['key']=='RAW/'+expected['key']
        assert all(row[k]==expected[k] for k in ['source','profile','metric'])
        path=out/row['file'];assert path.resolve().is_relative_to(out.resolve()) and sha(path)==row['sha256']
        v=np.load(path,allow_pickle=False,mmap_mode='r')
        assert v.shape==(p['conditioning_parameters'],) and v.dtype==np.float64 and np.isfinite(v).all()
        assert np.isclose(np.linalg.norm(v),row['gradient_norm'],rtol=1e-10,atol=1e-12) and row['gradient_norm']>0
        contributions=row['contributions'];blocks=[b for b in p['reference_blocks'] if b['source']==row['source']]
        assert len(contributions)==len(blocks)==10
        assert [c['begin'] for c in contributions]==[b['begin'] for b in blocks]
        assert [c['reference'] for c in contributions]==[b['reference'] for b in blocks]
        for c,b in zip(contributions,blocks):
            query=c['query'];assert c['key']==row['key'] and query==expected_query(row)
            if query<15:
                actual=rows[b['case_ids'][query//3]]['raw']
                value=[actual['MSE'],1-actual['SSIM'],1-actual['ArcFace_observed_fixed']][query%3]
                bound=1e-6 if query%3 else 1e-13
            else:
                value=float(np.mean([rows[n]['raw']['landmark_high_frequency_MSE'] for n in b['case_ids'][1:]]));bound=1e-13
            assert c['scientific_forward_loss']==value and c['numeric_alignment_bound']==bound
            assert abs(c['value']-value)<=bound and c['gradient_norm']>0
            if 'sample_file' in c:
                a=np.load(out/c['sample_file'],allow_pickle=False)
                assert sha(out/c['sample_file'])==c['sample_sha256'] and a.dtype==np.float32 and a.shape==v.shape
                assert hashlib.sha256(a.tobytes()).hexdigest()==c['gradient_vector_sha256'] and np.isfinite(a).all()
                assert np.isclose(np.linalg.norm(a.astype(np.float64)),c['gradient_norm'],rtol=1e-10)
        assert np.isclose(np.mean([c['value'] for c in contributions]),row['value'],rtol=0,atol=1e-12)
        if site=='initial':
            previous,old=retained_initial_gradient(i,row['key'])
            parity.append(dict(key=row['key'],**parity_report(v,old)))
        vectors.append(v)
    timing=read(folder/'timing_projection.json')
    assert timing['cap_seconds']==600 and timing['projection_seconds']<=600 and record['seconds']<600
    assert np.isclose(timing['projection_seconds'],max(timing['completed_batch_seconds'])*20*1.5,rtol=0,atol=1e-9)
    return record,vectors,parity


def main():
    torch.set_grad_enabled(False)
    imported=read(ANALYSIS/'import.json');p=read(RETURN/'protocol.json');old=read(PARENT/'protocol.json')
    assert imported['complete'] and imported['protocol_sha256']==PIN==sha(RETURN/'protocol.json')
    manifest=read(RETURN/'export_manifest.json')
    for folder,bindings in [(RETURN,manifest['files']),(PARENT,p['parent_required_sha256']),(RESPONSE,p['response_required_sha256'])]:
        assert all(sha(folder/n)==h for n,h in bindings.items())
    out=RETURN/'outputs/nearby_gradient_learning_v1'
    failure=read(out/'failure.json') if (out/'failure.json').exists() else None
    run=read(out/'run_results.json') if (out/'run_results.json').exists() else None
    assert manifest['failure_present']==bool(failure) and manifest['complete_pilot']==bool(run)
    exit_code=int((RETURN/'review_exit_code.txt').read_text().strip());assert (exit_code==0)==bool(run)
    if not (out/'environment.json').exists():
        if failure and failure.get('stage')=='pre-model storage preflight':
            assert failure['available_free_bytes']<failure['required_free_bytes']==9*1024**3
            assert failure['neural_calls']==failure['gradient_queries']==failure['accepted_updates']==failure['parameter_proposals']==0
        write(ANALYSIS/'failure_audit.json',dict(complete=True,scope='Startup failure only',failure=failure,exit_code=exit_code,
            model_qualified=False,goal_complete=False,local_neural_or_training_calls=0));return
    env=read(out/'environment.json')
    assert env['GPU']=='NVIDIA L4' and env['host'].split('.')[0]=='forensic-dgp-thesis' and env['manual_tmux']
    assert env['torch']==p['expected_torch'] and env['expected_torch']==p['expected_torch']
    assert env['torch_optimizer_constructed'] is False and env['stateless_gradient_descent_is_actual_training']
    virtual=schema(RETURN);names=p['conditioning_names'];edges=np.cumsum([0]+[int(np.prod(p['conditioning_shapes'][n])) for n in names])
    assert state_hash(virtual)==p['initial_model_state']==env['initial_model_state']
    assert state_hash({n:v for n,v in virtual.items() if n not in names})==env['frozen_state']
    initial=np.load(out/'initial_conditioning.npy',allow_pickle=False)
    np.testing.assert_array_equal(initial,np.concatenate([virtual[n].numpy().reshape(-1) for n in names]))
    norm=float(np.sqrt(sum(float(np.square(virtual[n].numpy().astype(np.float64)).sum()) for n in p['all37_learning_names'])))
    assert np.isclose(norm,env['initial_all37_norm'],rtol=1e-12)
    snapshots={};cells={}
    labels=p['sampling_sites']+[f'trial{i:02d}' for i in range(3)]+['rollback']
    for label in labels:
        folder=out/label
        if not (folder/'snapshot.json').exists() or (label!='initial' and not (folder/'conditioning.npy').exists()):continue
        snapshot,groups,png=audit_snapshot(folder,p,old,virtual,names,edges)
        assert snapshot['exact_controls']==(label in ['initial','rollback'])
        snapshots[label]=dict(snapshot=snapshot,groups=groups);cells[label]=png
        if label in ['plus02','minus02']:
            np.testing.assert_array_equal(np.load(folder/'conditioning.npy',allow_pickle=False),np.load(RESPONSE/'outputs/conditioning_response_v1'/label/'conditioning.npy',allow_pickle=False))
        if label in p['sampling_sites']:
            for block in p['reference_blocks']:
                stem=f"b{block['begin']//5:03d}.npy"
                np.testing.assert_array_equal(np.load(folder/stem,allow_pickle=False),np.load(RESPONSE/'outputs/conditioning_response_v1'/label/stem,allow_pickle=False))
    records=[];vectors=[];parity=[]
    for site in p['sampling_sites']:
        if (out/('gradients_'+site)/'gradients.json').exists():
            record,v,checks=audit_site(out,site,p);records.append(record);vectors.extend(v);parity.extend(checks)
    certificate=None
    if (out/'direction_certificate.json').exists():
        direction_record=read(out/'direction_certificate.json');assert direction_record['complete']
        assert direction_record['groups']==[r for site in records for r in site['groups']]
        assert sha(out/'direction.npy')==direction_record['direction_sha256']
        direction=np.load(out/'direction.npy',allow_pickle=False)
        certificate=audit_certificate(vectors,direction_record['groups'],direction,direction_record['certificate'])
    trials=[];accepted=[]
    for path in sorted(out.glob('trial??/decision.json')):
        t=read(path);label=t['label'];assert t['attempt']==len(trials) and t['before_snapshot']=='initial' and certificate
        assert sha(out/label/'snapshot.json')==t['snapshot_sha256'] and sha(out/label/'conditioning.npy')==t['conditioning_sha256']
        assert sha(out/'direction.npy')==t['direction_sha256']
        ceiling=bounded_rate(initial,initial,direction,norm);assert np.isclose(ceiling,t['rate_ceiling'],rtol=1e-10)
        rate=np.float32(ceiling*p['backtrack_factors'][t['attempt']]);assert float(rate)==t['rate']
        actual=np.load(out/label/'conditioning.npy',allow_pickle=False);np.testing.assert_array_equal(actual,initial+rate*direction)
        distance=float(np.linalg.norm(actual.astype(np.float64)-initial.astype(np.float64))/norm)
        assert distance<=.001 and np.isclose(distance,t['cumulative_relative_parameter_distance'],rtol=1e-10)
        checked=decision(snapshots['initial']['groups'],snapshots['initial']['groups'],snapshots[label]['groups'])
        validate_receipt(checked,t['decision'],(snapshots['initial']['snapshot']['groups'],snapshots['initial']['snapshot']['groups'],snapshots[label]['snapshot']['groups']))
        if checked['accept_step']:accepted.append(label)
        trials.append(t)
    assert len(trials)<=3 and len(accepted)<=1 and (not accepted or accepted[0]==trials[-1]['label'])
    assert manifest['accepted_updates']==len(accepted)
    if accepted:
        c=read(out/'accepted01.json');assert c['trial']==accepted[0] and c['accepted_update']==1 and c['protocol_sha256']==PIN
        assert sha(out/'accepted01.npz')==c['checkpoint_sha256'] and sha(out/'accepted_component_full_state.pth')==c['full_state_sha256']
        with np.load(out/'accepted01.npz',allow_pickle=False) as a:
            np.testing.assert_array_equal(a['conditioning'],np.load(out/accepted[0]/'conditioning.npy',allow_pickle=False))
            assert a['torch_CPU_RNG'].dtype==np.uint8 and a['torch_CUDA_RNG_0'].dtype==np.uint8
        state=torch.load(out/'accepted_component_full_state.pth',map_location='cpu',weights_only=True)
        assert state_hash(state)==c['model_state']==snapshots[accepted[0]]['snapshot']['model_state']
        assert state_hash({n:v for n,v in state.items() if n not in names})==env['frozen_state'];del state
    if (out/'rollback.json').exists():
        r=read(out/'rollback.json');assert r['complete'] and r['fresh_state']==p['initial_model_state']
        assert r['frozen_state']==env['frozen_state'] and r['buffer_state']==env['buffer_state']
        assert r['original_state']==env['original_state'] and r['recognizer_state']==env['recognizer_state']
        assert r['RNG_restored'] and r['rollback_outputs_exact']
        assert snapshots['rollback']['groups']==snapshots['initial']['groups']
        with np.load(out/'initial_RNG.npz',allow_pickle=False) as a,np.load(out/'restored_RNG.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[n],b[n]) for n in a.files)
    if (out/'failure_restoration.json').exists():
        r=read(out/'failure_restoration.json');assert r['complete'] and r['fresh_state']==r['expected_initial']==p['initial_model_state']
        assert r['frozen_state']==env['frozen_state'] and r['buffer_state']==env['buffer_state']
    if run:
        assert len(records)==3 and certificate and len(accepted)==1 and not failure
        assert run['counts']==dict(B=140+20*len(trials),recognizer=220+40*len(trials),gradient_queries=960,backwards=0,
            accepted_updates=1,parameter_proposals=len(trials),derivative_sampling_displacements=2)
        assert run['sampling_sites']==records and run['trials']==trials
        assert run['final_snapshot']==accepted[0] and run['fresh_restored_model_state']==p['initial_model_state']
        assert run['final_model_state']==snapshots[accepted[0]]['snapshot']['model_state']
        assert decision(snapshots['initial']['groups'],snapshots['initial']['groups'],snapshots[accepted[0]]['groups'])['one_percent_checkpoint_pass']
        assert run['full_epochs']==run['native_pixels_used']==run['final_identity_pixels']==0
        assert run['seconds']<1350 and run['peak_VRAM']<=p['peak_VRAM_bytes'] and run['peak_RSS']<=p['peak_RSS_bytes']
    if failure and 'counts' in failure:
        assert failure['counts']['accepted_updates']==len(accepted) and failure['counts']['parameter_proposals']<=3
        assert failure['counts']['gradient_queries']<=960
    gallery=ANALYSIS/'sheets';gallery.mkdir();visual=[]
    for block in p['reference_blocks']:
        sheet=Image.new('RGB',(256*(2+len(cells)),1400),'#181818');draw=ImageDraw.Draw(sheet)
        for row,cid in enumerate(block['case_ids']):
            c=next(c for c in old['cases'] if c['id']==cid);ref=next(r for r in old['references'] if r['id']==c['source_person_or_reference'])
            columns=[('input',image(PARENT/c['input'])),('paired TRAIN target',image(PARENT/ref['target']))]+[(label,images[cid]) for label,images in cells.items()]
            for column,(label,rgb) in enumerate(columns):
                sheet.paste(Image.fromarray(rgb),(column*256,row*280+24));draw.text((column*256+2,row*280+4),label,fill='white')
        path=gallery/f"reference_{block['begin']//5:03d}.png";sheet.save(path)
        visual.append(dict(file=path.relative_to(ANALYSIS).as_posix(),sha256=sha(path),case_ids=block['case_ids'],columns=['input','paired TRAIN target']+list(cells),directly_viewed=False))
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256'];assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/('independent_audit.json' if run else 'failure_audit.json'),dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),protocol_sha256=PIN,completed_pilot=bool(run),failure=failure,
        snapshot_labels=list(snapshots),complete_gradient_sites=[r['site'] for r in records],certificate_replay=certificate,
        original_initial_gradient_parity_reports=parity,all_original_initial_parity_checks_pass=all(r['original_parity_check_pass'] for r in parity) if parity else None,
        old_parity_failures_not_erased_or_bounds_relaxed=True,accepted_trials=accepted,trials_replayed=len(trials),
        visual_sheets=visual,all20_sheets_require_direct_review=True,original_scientific_RAW_PNG_gate_branches_exact=True,
        neural_derivatives_not_locally_reexecuted=True,frozen_state_and_rollback_receipts_distinguished_from_replayed_arrays=True,
        local_neural_or_training_calls=0,full_TRAIN_native_DEV_and_final_review_pending=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,accepted_updates=len(accepted),gradient_sites=len(records),sheets=20,model_qualified=False))


if __name__=='__main__':main()

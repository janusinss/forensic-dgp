"""Finite active-original-decoder capacity pilot; called only after VM preflight."""
from pathlib import Path
import hashlib
import json
import time


def frozen_partition(net, selected):
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        if name in selected:
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def train(root, parent, p, pin, original, candidate, identity, items, progress,
          started, clock, normalizers, namespace, canonical_tensor, write, sha, recheck_parent):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import convolve1d
    import torch
    from cctv_dgp_pilot import exported_pixel_metrics, state_hash
    from cctv_dgp_batchmatched_identity_v26 import objective_terms
    out = root/'outputs'
    selected = [(name,value) for name,value in candidate.net.named_parameters() if value.requires_grad]
    assert len(selected)==12 and sum(value.numel() for _,value in selected)==498627
    names = {name for name,_ in selected};parameters=[value for _,value in selected]
    frozen_before = frozen_partition(candidate.net,names)
    assert frozen_before==frozen_partition(original.net,names)
    original_state=state_hash(original.net);identity_state=state_hash(identity)
    assert original_state==p['original_DGP_state'] and identity_state==p['frozen_recognizer_state']
    progress.update({'backwards':0,'optimizer_constructed':False,'new_checkpoint_created':False})
    schedule=json.loads((root/'schedule.json').read_text(encoding='utf-8'))['batches']
    assert len(schedule)==800 and all(len(row)==5 and len(set(row))==5 and all(0<=i<50 for i in row) for row in schedule)
    for begin in range(0,800,10):assert sorted(i for row in schedule[begin:begin+10] for i in row)==list(range(50))
    keys=['x','base','target','mask','feature','interior','valid7','grid','truth','degraded_weight','clear_weight']
    def batch(ids):return {key:torch.cat([items[i][key] for i in ids]) for key in keys}
    positions=np.arange(-6,7,dtype=np.float64);kernel=np.exp(-.5*(positions/2)**2);kernel/=kernel.sum()
    def detail_metric(png,target,support):
        luma=np.array([.299,.587,.114])
        value=(png.astype(np.float64)/255*luma).sum(2)
        truth=(target.astype(np.float64)/255*luma).sum(2)
        high=lambda a:a-convolve1d(convolve1d(a,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')
        return float(np.square(high(value)-high(truth))[support].mean())
    snapshots=[];step_times=[];fit_started=None
    def assert_frozen():
        recheck_parent()
        assert frozen_partition(candidate.net,names)==frozen_before,'Frozen encoder/head4/normalization changed'
        assert state_hash(original.net)==original_state and state_hash(identity)==identity_state
        assert all(not value.requires_grad and value.grad is None for value in original.parameters())
        assert all(not value.requires_grad and value.grad is None for value in identity.parameters())
        assert all(value.grad is None for name,value in candidate.net.named_parameters() if name not in names)
        assert sha(parent/'weights/dgp_v2.pth')==p['original_checkpoint_sha256']
    def snapshot(update):
        clock();folder=out/('update'+str(update));folder.mkdir();rows=[]
        torch.save(candidate.net.state_dict(),folder/'dgp_candidate_v28.pth')
        progress['new_checkpoint_created']=True
        assert_frozen()
        with torch.no_grad():
            for begin in range(0,50,5):
                clock();group=items[begin:begin+5];b=batch(list(range(begin,begin+5)))
                prediction=candidate(b['x'],b['mask'])
                if update==0:assert torch.equal(prediction,b['base']),'Actual app-context initial batch parity changed'
                raw=prediction.permute(0,2,3,1).cpu().numpy().copy()
                delivered=np.stack([np.where(item['mask8'][...,None],np.floor(a*np.float32(255)),item['camera']).astype(np.uint8) for item,a in zip(group,raw)])
                # Follow the app's CPU NumPy float32 division before transfer.
                delivered_tensor=torch.cat([canonical_tensor(a,'cuda') for a in delivered])
                vectors=identity.embedding(delivered_tensor,b['mask'],b['grid']).cpu().numpy().copy()
                for item,a,png,vector in zip(group,raw,delivered,vectors):
                    c=item['case'];cid=c['id'];truth=item['truth'][0].cpu().numpy().copy()
                    np.save(folder/(cid+'.npy'),a,allow_pickle=False);Image.fromarray(png).save(folder/(cid+'.png'))
                    np.save(folder/(cid+'_embedding.npy'),vector,allow_pickle=False)
                    if update==0:np.save(folder/(cid+'_target_embedding.npy'),truth,allow_pickle=False)
                    metric=exported_pixel_metrics(png,item['target8'],item['mask8'])
                    metric['ArcFace_observed_fixed']=float(vector@truth)
                    feature=item['feature'][0,0].cpu().numpy()>0
                    metric['landmark_high_frequency_MSE']=detail_metric(png,item['target8'],feature)
                    baseline=item['base'][0].permute(1,2,0).cpu().numpy()
                    shift=(a-baseline)[item['mask8']].astype(np.float64).mean(0)
                    mean_only=np.clip(baseline.astype(np.float64)+shift,0,1).astype(np.float32)
                    mean_png=np.where(item['mask8'][...,None],np.floor(mean_only*np.float32(255)),item['camera']).astype(np.uint8)
                    metric['constant_mean_shift_only_MSE']=exported_pixel_metrics(mean_png,item['target8'],item['mask8'])['MSE']
                    if update==0:
                        with Image.open(out/'initial_baseline'/(cid+'.png')) as im:initial=np.asarray(im.convert('RGB'))
                        assert np.array_equal(png,initial)
                    rows.append({'id':cid,'source':c['source'],'profile':c['profile'],'metrics':metric,'postclip_mean_RGB_shift':shift.tolist()})
        groups={}
        for key in sorted({'all','clear','degraded'}|{c['source']+'/'+k for c in p['case_rows'] for k in ['all','clear','degraded',c['profile']]}):
            selected_rows=[row for row in rows if key in {'all','clear' if row['profile']=='clear' else 'degraded',row['source']+'/all',row['source']+('/clear' if row['profile']=='clear' else '/degraded'),row['source']+'/'+row['profile']}]
            assert selected_rows
            groups[key]={'cases':len(selected_rows),**{key:float(np.mean([row['metrics'][key] for row in selected_rows])) for key in ['MSE','SSIM','ArcFace_observed_fixed','landmark_high_frequency_MSE','constant_mean_shift_only_MSE']}}
        assert len(groups)==17
        receipt={'update':update,'seconds':time.monotonic()-started,'rows':rows,'groups':groups,
                 'candidate_DGP_state':state_hash(candidate.net),'candidate_checkpoint_sha256':sha(folder/'dgp_candidate_v28.pth'),
                 'frozen_partition_sha256':frozen_before,'snapshot_batch_size':5,
                 'normalization':'Actual app NumPy float32 division before device transfer'}
        write(folder/'metrics.json',receipt);snapshots.append(receipt)
        print(json.dumps({'snapshot':update,'training_outputs':50,'feature_error':groups['degraded']['landmark_high_frequency_MSE']}),flush=True)
        return receipt
    def execution(terminal):
        write(out/'execution_receipt.json',{'complete':True,'protocol_sha256':pin,'terminal':terminal,**progress,
              'worker_seconds':time.monotonic()-started,'fit_seconds':None if fit_started is None else time.monotonic()-fit_started,
              'fit_cap_seconds':1500,'worker_cap_seconds':1800,'step_times_seconds':step_times,
              'optimizer':'AdamW selected12 original decoder tensors','learning_rate':.00003,'weight_decay':.01,
              'gradient_clip_norm':1,'frozen_partition_before_after':frozen_before,
              'original_DGP_state':state_hash(original.net),'recognizer_state':state_hash(identity),
              'candidate_DGP_state':state_hash(candidate.net),'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),
              'VRAM_cap_bytes':20*1024**3,'app_promotion':False,'goal_complete':False})
    try:
        baseline=snapshot(0)
        # Initial selected12 differentiation and exact preservation proof already
        # passed in the actual app context; no optimizer existed during that proof.
        optimizer=torch.optim.AdamW(parameters,lr=.00003,weight_decay=.01)
        progress['optimizer_constructed']=True;fit_started=time.monotonic()
        for update,ids in enumerate(schedule,1):
            clock();assert time.monotonic()-fit_started<1500,'V28 fitting cap1500 seconds'
            step=time.monotonic();optimizer.zero_grad(set_to_none=True);b=batch(ids)
            pred=candidate(b['x'],b['mask'])
            terms=objective_terms(b,pred,identity,namespace['mean'],namespace['feature_errors'],namespace['ssim'],normalizers)
            objective=sum(terms.values()).mean();assert torch.isfinite(objective)
            objective.backward();progress['backwards']+=1
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters)
            assert all(v.grad is None for name,v in candidate.net.named_parameters() if name not in names)
            torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step()
            progress['optimizer_updates']=update;progress['epochs']=update//10
            torch.cuda.synchronize();step_times.append(time.monotonic()-step)
            if update==20:
                elapsed=time.monotonic()-fit_started
                projected=elapsed+(800-20)*float(np.mean(step_times[1:]))*1.25+120
                write(out/'timing_update20.json',{'updates':20,'seconds':elapsed,'steady_sample_seconds':step_times[1:].copy(),
                      'remaining_updates':780,'safety_factor':1.25,'overhead_seconds':120,'projected_seconds':projected,'cap_seconds':1500})
                assert projected<=1500,'Training timing projection exceeds cap'
            if update in [50,400,800]:
                current=snapshot(update)
                if update==50:
                    gain=1-current['groups']['degraded']['landmark_high_frequency_MSE']/baseline['groups']['degraded']['landmark_high_frequency_MSE']
                    write(out/'early_structure_stop.json',{'update':50,'relative_feature_error_gain':gain,'minimum':.01,'pass':gain>=.01})
                    assert gain>=.01,'No one-percent early structural gain; retain stop'
            if update==1 or update%50==0:print(f'V28 update {update}/800 objective={float(objective.detach()):.6f}',flush=True)
        assert time.monotonic()-fit_started<=1500
        assert_frozen();current=snapshots[-1];failures=[]
        for group,base in baseline['groups'].items():
            actual=current['groups'][group];assert actual['cases']==base['cases']
            for metric in ['MSE','SSIM','ArcFace_observed_fixed']:
                bad=actual[metric]>base[metric]+1e-12 if metric=='MSE' else actual[metric]<base[metric]-1e-6
                if bad:failures.append({'group':group,'metric':metric,'baseline':base[metric],'candidate':actual[metric]})
        gain=1-current['groups']['degraded']['landmark_high_frequency_MSE']/baseline['groups']['degraded']['landmark_high_frequency_MSE']
        source_gains={src:1-current['groups'][src+'/degraded']['landmark_high_frequency_MSE']/baseline['groups'][src+'/degraded']['landmark_high_frequency_MSE'] for src in sorted({c['source'] for c in p['case_rows']})}
        base=baseline['groups']['degraded'];actual=current['groups']['degraded']
        photometric_fraction=max(0,base['MSE']-actual['constant_mean_shift_only_MSE'])/max(base['MSE']-actual['MSE'],1e-12)
        if photometric_fraction>.2:failures.append({'group':'degraded','metric':'brightness_gain_fraction','candidate':photometric_fraction,'maximum':.2})
        execution('completed800')
        result={'complete':True,'protocol_sha256':pin,'seconds':time.monotonic()-started,'updates':progress['optimizer_updates'],
                **progress,'preservation_failures':failures,'degraded_feature_MSE_relative_gain':gain,'source_feature_gains':source_gains,
                'brightness_gain_fraction':photometric_fraction,'necessary_capacity_pass':not failures and gain>=.1 and all(v>=0 for v in source_gains.values()),
                'selection':'Final800 only; no checkpoint selected for the app','trained_parameters':498627,'trained_tensors':12,
                'frozen_encoder_head4_and_all_buffers_unchanged':True,'historical_R2_all14_failure_retained':True,
                'candidate_DGP_state':state_hash(candidate.net),'original_DGP_checkpoint_unchanged':True,
                'native_or_reserved_used':False,'visual_review_pending':True,'independent_audit_pending':True,
                'app_promotion':False,'goal_complete':False}
        write(out/'results.json',result);print(json.dumps(result),flush=True)
    except BaseException:
        torch.save(candidate.net.state_dict(),out/'stopped_dgp_candidate_v28.pth')
        progress['new_checkpoint_created']=True
        execution('failed_partial')
        raise

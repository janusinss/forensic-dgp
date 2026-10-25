"""Finite paired U/S feature-only training on the user's manual Linux L4 VM.

Import is inert. run() verifies the sealed packet before guarded Torch import.
All actual prefix/adapter/full/backward/optimizer operations reserve durable
ledger entries. Quality stops end one arm; infrastructure failures end the run.
"""
import hashlib
import io
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from . import packet as p
from . import model as m

O='outputs/spatial_objectives_v1'
SITES=('32','64','128','256')
UPDATES=(0,6,12,24)
MOMENTS=('wanted_abs','final_abs','wanted_AC_abs','final_AC_abs','wanted_sq','correction_sq',
         'final_sq','wanted_spatial_sq','correction_spatial_sq','spatial_dot','observed_sq','target_sq')


class GuardStop(Exception):
    """Only this explicitly measured quality event permits the other arm."""


def _asset(root, item):
    path=p.regular(root/item['path']);actual=p.pin(root,path)
    p.require(actual==item,'Exact content-addressed asset changed: '+item['path'])
    return path


def _rgb(root, item, expected, np, Image):
    with Image.open(_asset(root,item)) as image:
        p.require(image.mode=='RGB' and image.size==(256,256),'Exact RGB PNG256 required')
        rgb=np.asarray(image).copy()
    p.require(rgb.dtype==np.uint8 and hashlib.sha256(rgb.tobytes()).hexdigest()==expected,'Decoded RGB8 pixels differ')
    return rgb


def _input(root, row, torch, np, Image):
    rgb=_rgb(root,row['input'],row['input_RGB8_sha256'],np,Image)
    values=np.ascontiguousarray(rgb.astype(np.float32)/np.float32(255))
    return torch.from_numpy(values.transpose(2,0,1).copy())[None].to('cuda'),rgb


def _feature(root, item, site, np, torch, resources):
    resources.check('feature file read')
    array=np.load(_asset(root,item),mmap_mode='r',allow_pickle=False)
    p.require(array.shape==(1,m.CHANNELS[site],int(site),int(site)) and array.dtype==np.float32 and
              np.isfinite(array).all(),'Exact finite observed/clean feature schema')
    return torch.from_numpy(array.copy()).to('cuda')


def _features(root, items, np, torch, resources):
    p.require(set(items)==set(SITES),'Exact four-site feature cache')
    return {site:_feature(root,items[site],site,np,torch,resources) for site in SITES}


def _rng(torch, np):
    state=np.random.get_state()
    return {'CPU_RNG':torch.get_rng_state().clone(),'CUDA_RNG':torch.cuda.get_rng_state(0).clone(),
        'NumPy_RNG':{'algorithm':state[0],'state':torch.from_numpy(state[1].astype(np.int64)),
            'position':int(state[2]),'has_gauss':int(state[3]),'cached_gaussian':float(state[4])},
        'python_RNG':random.getstate()}


def _restore_rng(saved, torch, np):
    torch.set_rng_state(saved['CPU_RNG'].clone());torch.cuda.set_rng_state(saved['CUDA_RNG'].clone(),0)
    n=saved['NumPy_RNG'];np.random.set_state((n['algorithm'],n['state'].numpy().astype(np.uint32),
        n['position'],n['has_gauss'],n['cached_gaussian']));random.setstate(saved['python_RNG'])


def _rng_pin(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def _fingerprints(state, torch):
    answer={}
    for name,value in sorted(state.items()):
        answer[name]={'shape':list(value.shape),'dtype':str(value.dtype),
            'canonical_tensor_sha256':m.tensor_state_digest({name:value}),
            'raw_bytes_sha256':hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
    return answer


def _checkpoint(writer, candidate, optimizer, arm, update, protocol_sha, ledger, torch, np):
    state=m.adapter_state(candidate)
    p.require(m.frozen_state_digest(candidate)==m.R2_DIGEST,'Base515 changed before checkpoint')
    adam=optimizer.state_dict();steps={str(k):float(v['step'].detach().cpu().item()) for k,v in adam['state'].items()}
    p.require((update==0 and not steps) or (len(steps)==24 and all(value==update for value in steps.values())),
              'Exact committed per-parameter Adam step counters required')
    saved={'format':'codeformer-spatial-objective-adapter-state-v1','protocol_sha256':protocol_sha,
        'arm':arm,'completed_updates':update,'adapter_state':state,'adapter_sha256':m.tensor_state_digest(state),
        'adapter_tensor_fingerprints':_fingerprints(state,torch),'base515_sha256':m.R2_DIGEST,
        'optimizer':adam,'optimizer_type':'fresh Adam per arm','Adam_step_counters':steps,'sampler_committed_rows':update*3,
        'counts':dict(ledger.counts),'returns':dict(ledger.returns),'pending':dict(ledger.pending),
        **_rng(torch,np),'model_qualified':False,'no_base515_tensors_serialized':True,
        'gradient_buffers_excluded':True,'automatic_resume_authorized':False,
        'expected_S_null_biases':arm=='S','all24_tensor_change_required':False}
    p.require(not ledger.pending,'Checkpoint only between completed operations')
    return writer.checkpoint(O+'/checkpoints/'+arm+f'/u{update:04d}.pt',saved,torch)


def _cosine(dot,a2,b2):
    return float(dot/math.sqrt(a2*b2)) if a2>0 and b2>0 else None


def _statistics(observed, used, target, scale, np, resources):
    """Full spatial FP64 arithmetic, eight channels at a time; retain8x8 pools."""
    x=observed[0].detach().cpu().numpy();z=used[0].detach().cpu().numpy()
    t=target[0].detach().cpu().numpy();channels,h,w=x.shape
    means=np.empty((3,channels),dtype=np.float64);moments=np.empty((12,channels),dtype=np.float64)
    pooled=np.empty((3,channels,8,8),dtype=np.float64)
    def pool(value):return value.reshape(len(value),8,h//8,8,w//8).mean(axis=(2,4))
    for start in range(0,channels,8):
        resources.check('feature statistics channel chunk');end=min(start+8,channels)
        a=x[start:end].astype(np.float64);b=t[start:end].astype(np.float64)
        wanted=b-a;correction=z[start:end].astype(np.float64)-a;final=correction-wanted
        means[0,start:end]=a.mean((1,2));means[1,start:end]=wanted.mean((1,2));means[2,start:end]=correction.mean((1,2))
        wa=wanted-means[1,start:end,None,None];ca=correction-means[2,start:end,None,None];fa=ca-wa
        ops=(lambda:np.abs(wanted),lambda:np.abs(final),lambda:np.abs(wa),lambda:np.abs(fa),
            lambda:np.square(wanted),lambda:np.square(correction),lambda:np.square(final),
            lambda:np.square(wa),lambda:np.square(ca),lambda:ca*wa,lambda:np.square(a),lambda:np.square(b))
        for index,op in enumerate(ops):moments[index,start:end]=op().mean(axis=(1,2))
        pooled[0,start:end]=pool(a);pooled[1,start:end]=pool(wanted);pooled[2,start:end]=pool(correction)
    values={key:float(moments[i].mean()) for i,key in enumerate(MOMENTS)}
    wdc=float(np.square(means[1]).mean());cdc=float(np.square(means[2]).mean());ddot=float((means[1]*means[2]).mean())
    ini,fin=values['wanted_abs']/scale,values['final_abs']/scale
    ac0,ac1=values['wanted_AC_abs']/scale,values['final_AC_abs']/scale
    result={'initial_normalized_L1':ini,'final_normalized_L1':fin,'initial_centered_normalized_L1':ac0,
        'final_centered_normalized_L1':ac1,'raw_relative_gain':1-fin/ini if ini>0 else None,
        'centered_relative_gain':1-ac1/ac0 if ac0>0 else None,
        'correction_RMS_over_clean_RMS':math.sqrt(values['correction_sq'])/scale,
        'wanted_RMS_over_clean_RMS':math.sqrt(values['wanted_sq'])/scale,
        'final_RMS_over_clean_RMS':math.sqrt(values['final_sq'])/scale,
        'correction_DC_energy':cdc,'correction_spatial_energy':values['correction_spatial_sq'],
        'wanted_DC_energy':wdc,'wanted_spatial_energy':values['wanted_spatial_sq'],
        'correction_DC_energy_fraction':cdc/values['correction_sq'] if values['correction_sq']>0 else None,
        'direction_alignment':_cosine(values['spatial_dot']+ddot,values['correction_sq'],values['wanted_sq']),
        'DC_direction_alignment':_cosine(ddot,cdc,wdc),
        'spatial_direction_alignment':_cosine(values['spatial_dot'],values['correction_spatial_sq'],values['wanted_spatial_sq'])}
    return means,moments,pooled,result


def _stats_snapshot(root, protocol, writer, ledger, resources, candidate, cache, arm, update, torch, np):
    adapter_sha=m.tensor_state_digest(m.adapter_state(candidate))
    rows=[]
    for row in protocol['train_rows']:
        rid=row['reference_id'];observed=_features(root,cache[row['id']],np,torch,resources)
        clean=_features(root,protocol['clean_features'][rid]['features'],np,torch,resources)
        with torch.no_grad():
            used=observed if update==0 else m.adapter_features(candidate,observed,ledger,'adapter_stats',
                {'arm':arm,'update':update,'id':row['id']})
        arrays={};site_metrics={}
        for site in SITES:
            means,moments,pooled,metrics=_statistics(observed[site],used[site],clean[site],
                protocol['clean_features'][rid]['scales'][site],np,resources)
            arrays[site+'_means']=means;arrays[site+'_moments']=moments;arrays[site+'_pooled']=pooled
            site_metrics[site]=metrics
        stem=O+'/shared_features0/'+row['id'] if update==0 else O+'/stats/'+arm+f'/u{update:04d}/'+row['id']
        bundle=writer.npz(stem+'.npz',arrays)
        result={'id':row['id'],'reference_id':rid,'arm':arm,'update':update,
            'scales':protocol['clean_features'][rid]['scales'],'site_metrics':site_metrics,'arrays':bundle,
            'adapter_state_sha256':adapter_sha,'base515_sha256':m.R2_DIGEST,'source_input':row['input'],
            'clean_targets':protocol['clean_features'][rid]['features'],'initial_identity':update==0,
            'moment_order':list(MOMENTS),'means_pooled_order':['observed','wanted','correction'],
            'audit_limit':'Source-captured full spatial sufficient-statistic consistency; pooled8 is not full activation replay.',
            'model_qualified':False,'calibration_not_subtracted':True}
        writer.json(stem+'.json',result);rows.append(result)
        del observed,used,clean,arrays
    return rows


def _metrics(raw, png, target, np):
    fields=('mae','mse','psnr','ssim','luminance_bias','signed_luminance_bias','coarse_rgb_mae','adjacent_rgb_error')
    if target is None:return {v:dict.fromkeys(fields) for v in ('raw','native_png')}
    from skimage.metrics import structural_similarity
    truth=target.astype(np.float64);answer={}
    for name,value in (('raw',raw),('native_png',png.astype(np.float32)/np.float32(255))):
        value=value.astype(np.float64);residual=value-truth;mse=float(np.square(residual).mean())
        signed=float((residual@np.asarray([.2126,.7152,.0722])).mean())
        pool=lambda a:a.reshape(16,16,16,16,3).mean(axis=(1,3),dtype=np.float64)
        answer[name]={'mae':float(np.abs(residual).mean()),'mse':mse,'psnr':None if mse==0 else -10*math.log10(mse),
            'ssim':float(structural_similarity(value,truth,data_range=1.,channel_axis=2)),
            'luminance_bias':abs(signed),'signed_luminance_bias':signed,
            'coarse_rgb_mae':float(np.abs(pool(value)-pool(truth)).mean()),
            'adjacent_rgb_error':float(.5*(np.abs(residual[:,1:]-residual[:,:-1]).mean()+
                np.abs(residual[1:]-residual[:-1]).mean()))}
    return answer


def _guard_gate(current, initial):
    """Exact old24 repeated TRAIN thresholds; no tolerance or changed gate."""
    p.require(len(current)==len(initial)==24 and set(current)==set(initial),'Same24 guard rows required')
    answer={'pass':True,'additional_TRAIN_stop_only':True,'model_qualified':False,
        'arithmetic':'Python float64; exact <=; no added tolerance','groups':{}}
    for role in ('clear','dark','filter'):
        ids=sorted(k for k,r in current.items() if r['role']==role);p.require(len(ids)==8,'Eight same-role guards')
        group={'pass':True,'cases':ids,'views':{}}
        for view in ('raw','native_png'):
            group['views'][view]={}
            for field in (('mae','adjacent_rgb_error') if role=='filter' else ('behavior_distance',)):
                def value(rows,k):
                    return rows[k]['behavior_distance'][view] if field=='behavior_distance' else rows[k]['metrics'][view][field]
                start={k:float(value(initial,k)) for k in ids};now={k:float(value(current,k)) for k in ids}
                if role=='filter':
                    deltas={k:now[k]-start[k] for k in ids};limits=dict.fromkeys(ids,.03)
                    mean=sum(deltas.values())/8;mean_limit=.01
                else:
                    deltas=now;limits={k:max(.03,start[k]) for k in ids};mean=sum(now.values())/8
                    mean_limit=max(.01,sum(start.values())/8)
                p.require(all(math.isfinite(v) and v>=0 for v in (*start.values(),*now.values())),'Finite guard distances')
                bad=[k for k in ids if deltas[k]>limits[k]];passed=not bad and mean<=mean_limit
                group['views'][view][field]={'pass':passed,'current':now,'initial':start,'case_limits':limits,
                    'failed_case_ids':bad,'mean_measurement':mean,'mean_limit':mean_limit,
                    'measurement':'increase from initial' if role=='filter' else 'distance from frozen teacher'}
                group['pass']=group['pass'] and passed
        answer['groups'][role]=group;answer['pass']=answer['pass'] and group['pass']
    return answer


def _full_output(root, protocol, writer, ledger, resources, candidate, row, arm, update, phase, torch, np, Image):
    resources.check('before full forward');value,_=_input(root,row,torch,np,Image)
    context={'arm':arm,'update':update,'id':row['id'],'phase':phase}
    token=ledger.reserve('full_forward',**context)
    with torch.no_grad():
        record=m.forward_record(candidate,value,ledger=ledger,context={**context,'parent_full_reservation':token})
    ledger.returned(token);resources.check('after full forward')
    p.require(m.readout_fingerprints(record)==row['upstream'],'Frozen observed upstream logits/LQ/tokens/AdaIN changed')
    raw=np.ascontiguousarray(record.delivered256.detach().cpu().numpy()[0].transpose(1,2,0))
    png=np.ascontiguousarray(m.png_tensor(record.delivered256).cpu().numpy()[0].transpose(1,2,0))
    target=None
    if row.get('target') is not None:
        target=_rgb(root,row['target'],row['target_RGB8_sha256'],np,Image).astype(np.float32)/np.float32(255)
    stem=O+'/parity/'+row['id'] if phase=='parity' else O+'/'+phase+'s/'+arm+f'/u{update:04d}/'+row['id']
    # Strong plural is a fixed namespace exception to the guard plural.
    if phase=='strong':stem=O+'/strong/'+arm+f'/u{update:04d}/'+row['id']
    outputs={'raw':writer.array(stem+'.npy',raw),'png':writer.png(stem+'.png',png)}
    result={'id':row['id'],'reference_id':row['reference_id'],'profile':row['profile'],'role':row.get('role'),
        'split':'TRAIN','arm':arm,'update':update,'phase':phase,'input':row['input'],'target':row.get('target'),
        'outputs':outputs,'upstream':row['upstream'],'metrics':_metrics(raw,png,target,np),
        'clipping':m.clipping_statistics(record),'RAW_PNG_separate':True,'model_qualified':False}
    if row.get('behavior_teacher') is not None:
        teacher=row['behavior_teacher'];traw=np.load(_asset(root,teacher['raw']),allow_pickle=False)
        with Image.open(_asset(root,teacher['png'])) as image:tpng=np.asarray(image).copy()
        result['behavior_distance']={'raw':float(np.abs(raw.astype(np.float64)-traw.astype(np.float64)).mean()),
            'native_png':float(np.abs((png.astype(np.float32)/np.float32(255)).astype(np.float64)-
                (tpng.astype(np.float32)/np.float32(255)).astype(np.float64)).mean())}
        result['behavior_teacher']=teacher
    if phase=='parity':
        reference=np.load(_asset(root,row['outputs']['raw']),allow_pickle=False)
        with Image.open(_asset(root,row['outputs']['png'])) as image:rp=np.asarray(image).copy()
        p.require(np.array_equal(raw,reference) and np.array_equal(png,rp),'Exact initial cached CUDA RAW/PNG parity failed')
        result['initial_cached_RAW_PNG_upstream_exact']=True
    writer.json(stem+'.json',result)
    return result


def _guard_and_strong(root, protocol, writer, ledger, resources, candidate, arm, update, torch, np, Image):
    bank={}
    for row in protocol['guards']:
        bank[row['id']]=_full_output(root,protocol,writer,ledger,resources,candidate,row,arm,update,'guard',torch,np,Image)
    initial={row['id']:row for row in protocol['guards']}
    gate=_guard_gate(bank,initial)
    stem=O+'/guards/'+arm+f'/u{update:04d}/'
    writer.json(stem+'bank.json',{'complete':True,'arm':arm,'update':update,'cases':list(bank.values()),'model_qualified':False})
    writer.json(stem+'gates.json',gate)
    if not gate['pass']:raise GuardStop('Unchanged24 TRAIN guard failure; stop branch before strong probes or more updates')
    probes=[]
    for row in protocol['strong']:
        probes.append(_full_output(root,protocol,writer,ledger,resources,candidate,row,arm,update,'strong',torch,np,Image))
    writer.json(O+'/strong/'+arm+f'/u{update:04d}/bank.json',{'complete':True,'arm':arm,'update':update,
        'cases':probes,'initial_R2_baseline_reused':True,'model_qualified':False})
    return gate


def _train_update(root, protocol, writer, ledger, resources, candidate, optimizer, cache, arm, update, torch, np):
    optimizer.zero_grad(set_to_none=True);lr=m.learning_rate(update)
    for group in optimizer.param_groups:group['lr']=lr
    slots=[s for s in protocol['sampler'] if s['update']==update]
    p.require(len(slots)==3 and [s['slot'] for s in slots]==[1,2,3],'Exact three fixed sampler slots')
    rows={row['id']:row for row in protocol['train_rows']}
    for slot in slots:
        row=rows[slot['id']];context={'arm':arm,'update':update,'slot':slot['slot'],'id':row['id']}
        resources.check('training microbatch')
        observed=_features(root,cache[row['id']],np,torch,resources)
        entry=protocol['clean_features'][row['reference_id']]
        clean=_features(root,entry['features'],np,torch,resources)
        with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False):
            used=m.adapter_features(candidate,observed,ledger,'adapter_train',context)
            divided,terms,scalar=m.feature_objective(arm,used,clean,entry['scales'])
            actual={'event':'train_microbatch',**context,'LR':lr,'scalar':float(scalar.detach().item()),
                'divided_scalar':float(divided.detach().item()),'site_losses':{s:float(v.detach().item()) for s,v in terms.items()},
                'adapter_state_before_update_sha256':m.tensor_state_digest(m.adapter_state(candidate)),
                'observed_cache':cache[row['id']],'clean_targets':entry['features'],'scales':entry['scales']}
            token=ledger.reserve('backward',**context);divided.backward();torch.cuda.synchronize();ledger.returned(token)
        writer.append(O+'/events.jsonl',actual)
        del used,divided,scalar,terms,observed,clean
    named=dict(candidate.named_parameters());names=sorted(m.expected_adapter_shapes())
    grads=[named[name].grad for name in names];summary=m.summarize_gradients(names,grads,np=np)
    m.validate_gradient_summary(summary,allow_all_zero=True)
    p.require(any(summary['sites'][s]['up_l2']>0 for s in SITES),'No finite final-projection gradient in any site')
    before=m.adapter_state(candidate)
    norm=torch.nn.utils.clip_grad_norm_([named[n] for n in names],max_norm=1.,error_if_nonfinite=True)
    token=ledger.reserve('optimizer',arm=arm,update=update);optimizer.step();torch.cuda.synchronize();ledger.returned(token)
    p.require(all(torch.isfinite(named[n]).all().item() for n in names) and m.frozen_state_digest(candidate)==m.R2_DIGEST,
              'Nonfinite adapter or frozen515 state change')
    changes=m.summarize_gradients(names,[named[n].detach()-before[n].to('cuda') for n in names],np=np)
    state=m.adapter_state(candidate)
    writer.append(O+'/events.jsonl',{'event':'optimizer_commit','arm':arm,'update':update,'LR':lr,
        'preclip_norm':float(norm.detach().item()),'gradient_clip_norm':1.,'gradients':summary,'changes':changes,
        'adapter_before_sha256':m.tensor_state_digest(before),'adapter_after_sha256':m.tensor_state_digest(state),
        'all515_frozen_exact':True,'S_expected_null_up_biases':arm=='S','all24_tensors_changed_required':False})
    optimizer.zero_grad(set_to_none=True);resources.check('after optimizer commit')


def run(root, protocol, expected_sha):
    root=Path(root).absolute();m.manual_runtime_guard(root,expected_sha)
    verified=p.verify(root,expected_sha)
    p.require(verified==protocol,'Caller supplied protocol differs from exact static verification')
    p.require(not (root/O).exists(),'Preserve earlier/partial run; no unchanged relaunch')
    p.require(shutil.disk_usage(root).free>=p.LIMITS['initial_free_bytes'],'Initial6GiB physical free space required')
    writer=p.Writer(root,protocol);ledger=p.Ledger(writer);resources=p.Resources(root,writer)
    writer.json(O+'/started.json',{'complete':True,'protocol_sha256':expected_sha,'training_unrun_at_this_receipt':True,
        'model_qualified':False,'caller_checked_static_packet':True})
    torch=None;branches=[]
    try:
        p.require('torch' not in sys.modules,'Fresh worker must configure deterministic CUDA before import')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        resources.check('before Torch import and before any model call')
        torch=m.activate_runtime(root,expected_sha);resources.torch=torch
        import numpy as np
        from PIL import Image
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        random.seed(m.INITIALIZATION_SEED);np.random.seed(m.INITIALIZATION_SEED)
        torch.manual_seed(m.INITIALIZATION_SEED);torch.cuda.manual_seed_all(m.INITIALIZATION_SEED)
        p.require(torch.__version__=='2.9.1+cu129','Exact retained CUDA runtime required for cached parity')
        p.require(torch.cuda.get_device_name(0)=='NVIDIA L4','Exact single L4 required')
        resources.check('activated guarded runtime')
        writer.json(O+'/preflight.json',{'complete':True,'protocol_sha256':expected_sha,
            'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),'max_calls':protocol['max_calls'],
            'numpy':np.__version__,'seed':m.INITIALIZATION_SEED,'TF32':False,'deterministic_algorithms':True,
            'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'actual_neural_optimizer_calls_before_initialization':0,'actual_static_verification_complete':True,
            'model_qualified':False})
        with resources.stage('initialization',300):
            r2,r2_receipt=m.load_frozen_baseline('r2')
            candidate,census,initial_receipt=m.make_candidate(r2);del r2;torch.cuda.empty_cache()
            initial=m.adapter_state(candidate);initial_digest=m.tensor_state_digest(initial)
            p.require(initial_digest==protocol['initial_adapter_sha256']==m.INITIAL_ADAPTER_DIGEST and
                m.frozen_state_digest(candidate)==protocol['base515_sha256']==m.R2_DIGEST,'Exact declared identity/R2 state required')
            initial_rng=_rng(torch,np)
            writer.json(O+'/initial.json',{'complete':True,'r2':r2_receipt,'candidate':initial_receipt,'census':census,
                'adapter_sha256':initial_digest,'base515_sha256':m.R2_DIGEST,'adapter_tensor_fingerprints':_fingerprints(initial,torch),
                'CPU_RNG_sha256':_rng_pin(initial_rng['CPU_RNG']),'CUDA_RNG_sha256':_rng_pin(initial_rng['CUDA_RNG']),
                'two_arm_reset_required':True,'model_qualified':False})
        guards={row['id']:row for row in protocol['guards']}
        p.require(len(protocol['parity_ids'])==4,'Exactly four parity witnesses')
        parity_started=time.monotonic()
        with resources.stage('parity',300):
            for identifier in protocol['parity_ids']:
                _full_output(root,protocol,writer,ledger,resources,candidate,guards[identifier],'shared',0,'parity',torch,np,Image)
        parity_elapsed=time.monotonic()-parity_started
        resources.project('all172_full_output_forwards',parity_elapsed,4,172,900)
        cache={};clock=time.monotonic()
        with resources.stage('observed_feature_cache',600):
            for index,row in enumerate(protocol['train_rows'],1):
                resources.check('before cache prefix');value,_=_input(root,row,torch,np,Image)
                token=ledger.reserve('prefix',id=row['id'])
                with torch.no_grad():features=m.prefix_features(candidate,value)
                ledger.returned(token);cache[row['id']]={}
                for site in SITES:
                    values=np.ascontiguousarray(features[site].cpu().numpy())
                    cache[row['id']][site]=writer.array('cache/observed/'+row['id']+'/encoder'+site+'.npy',values)
                del features,value
                resources.project('observed_feature_cache',time.monotonic()-clock,index,10,600)
        shared=_stats_snapshot(root,protocol,writer,ledger,resources,candidate,cache,'shared',0,torch,np)
        p.require(m.tensor_state_digest(m.adapter_state(candidate))==initial_digest,'Shared readout changed identity')
        optimization_elapsed=0.;guard_elapsed=parity_elapsed
        for arm in ('U','S'):
            m.reset_adapters(candidate,initial);_restore_rng(initial_rng,torch,np)
            optimizer=m.make_optimizer(candidate)
            p.require(not optimizer.state,'New empty Adam required per arm')
            reset={'initial_adapter_sha256':m.tensor_state_digest(m.adapter_state(candidate)),
                'CPU_RNG_sha256':_rng_pin(torch.get_rng_state()),'CUDA_RNG_sha256':_rng_pin(torch.cuda.get_rng_state(0)),
                'fresh_Adam_empty':True}
            p.require(reset['CPU_RNG_sha256']==_rng_pin(initial_rng['CPU_RNG']) and
                reset['CUDA_RNG_sha256']==_rng_pin(initial_rng['CUDA_RNG']),'Exact shared initial RNG reset differs')
            writer.append(O+'/events.jsonl',{'event':'branch_reset','arm':arm,**reset,
                'same72_sampler_rows':True,'base515_sha256':m.frozen_state_digest(candidate)})
            branch={'arm':arm,'completed_updates':0,'terminal_status':'running','guard_failed_update':None,
                'reset':reset,'start_counts':dict(ledger.counts),'feature_statistics':[],'last_checkpoint':None}
            branch['last_checkpoint']=_checkpoint(writer,candidate,optimizer,arm,0,expected_sha,ledger,torch,np)
            for update in range(1,25):
                clock=time.monotonic()
                with resources.stage('one_optimization_update',max(1,int(900-optimization_elapsed))):
                    _train_update(root,protocol,writer,ledger,resources,candidate,optimizer,cache,arm,update,torch,np)
                optimization_elapsed+=time.monotonic()-clock;branch['completed_updates']=update
                resources.check('branch update committed')
                p.require(optimization_elapsed<=900,'Finite total900s optimization cap')
                if arm=='U' and update==1:resources.project('both_arm_optimization',optimization_elapsed,1,48,900)
                if update in (6,12,24):
                    branch['last_checkpoint']=_checkpoint(writer,candidate,optimizer,arm,update,expected_sha,ledger,torch,np)
                    branch['feature_statistics'].append({'update':update,'rows':_stats_snapshot(root,protocol,writer,ledger,
                        resources,candidate,cache,arm,update,torch,np)})
                    clock=time.monotonic()
                    try:
                        with resources.stage('guard_and_strong_outputs',max(1,int(900-guard_elapsed))):
                            _guard_and_strong(root,protocol,writer,ledger,resources,candidate,arm,update,torch,np,Image)
                    except GuardStop as stop:
                        branch.update(terminal_status='guard_stop',guard_failed_update=update,error=str(stop));break
                    finally:
                        guard_elapsed+=time.monotonic()-clock
                        # A quality-stop break must not bypass the infrastructure
                        # time bound or permit the other arm after a time breach.
                        p.require(guard_elapsed<=900,'Finite total900s guard/probe cap')
                print({'arm':arm,'update':update,'maximum':24},flush=True)
            if branch['terminal_status']=='running':branch['terminal_status']='complete'
            branch['end_counts']=dict(ledger.counts);branch['returns']=dict(ledger.returns)
            branch['automatic_candidate_selection']=False;branch['model_qualified']=False
            p.require(not ledger.pending and m.frozen_state_digest(candidate)==m.R2_DIGEST,'Pending call or frozen base drift at branch end')
            writer.json(O+'/branches/'+arm+'.json',branch);branches.append(branch)
            optimizer.zero_grad(set_to_none=True);del optimizer;torch.cuda.empty_cache()
        p.require(not ledger.pending,'No unresolved operations at terminal result')
        writer.json(O+'/result.json',{'complete':True,'protocol_sha256':expected_sha,'branches':branches,
            'shared_initial_feature_stats':shared,'counts':dict(ledger.counts),'returns':dict(ledger.returns),
            'pending':dict(ledger.pending),'base515_sha256':m.frozen_state_digest(candidate),
            'observed_VM_only_cache':cache,'optimization_seconds':optimization_elapsed,'guard_probe_seconds':guard_elapsed,
            'guard_probe_seconds_include_initial4_parity':True,'parity_seconds':parity_elapsed,
            'no_DEV_FINAL_native_optimized':True,'no_automatic_model_selection':True,'model_qualified':False,
            'face_visual_review_pending':True,'full478_development_gates_not_replaced':True,'training_updates_maximum':48})
    except BaseException as error:
        writer.json(O+'/failure.json',{'complete':True,'protocol_sha256':expected_sha,'error':str(error),
            'traceback':traceback.format_exc(),'branches_completed':branches,'counts':dict(ledger.counts),
            'returns':dict(ledger.returns),'pending':dict(ledger.pending),'model_qualified':False,
            'latest_current_adapter_optimizer_state_not_serialized_on_infrastructure_failure':True,
            'only_prior_completed_fixed_checkpoints_are_retained':True,
            'partial_evidence_preserved':True,'automatic_resume_authorized':False})
        raise

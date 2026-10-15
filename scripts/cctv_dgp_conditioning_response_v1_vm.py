"""Manual L4 repeatability/finite response measurement, not an optimizer pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

NAME='cctv_dgp_conditioning_response_v1_vm'
STEM='cctv-dgp-conditioning-response-v1'
STAGE='outputs/conditioning_response_v1'
PARENT_NAME='cctv_dgp_bank_photometric_guard_v1_vm'
PARENT_PIN='f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root,parent,pin):
    assert sha(root/'protocol.json')==pin
    p=read(root/'protocol.json');old=read(parent/'protocol.json')
    assert p['format']=='finite-conditioning-response-v1' and p['name']==NAME
    assert sha(parent/'protocol.json')==PARENT_PIN==p['parent_protocol_sha256']
    assert p['maximum_gradient_queries']==640 and p['model_forward_cap']==200 and p['recognizer_forward_cap']==360
    assert p['optimizer_updates']==p['accepted_updates']==p['full_epochs']==0 and p['temporary_displacements']==6
    assert p['gradient_repeats']==2 and len(p['raw_gradient_groups'])==32 and len(p['reference_blocks'])==20
    assert len(p['case_ids'])==p['paired_TRAIN_cases']==100 and len(set(p['case_ids']))==100
    assert p['conditioning_parameters']==2580864 and len(p['conditioning_names'])==12
    assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert p['native_pixels_used']==p['final_identity_pixels']==0
    assert not p['app_promotion'] and not p['automatic_follow_on'] and not p['failed_run_resume_allowed']
    assert not p['new_direction_solve'] and not p['optimizer_constructed']
    assert p['output_byte_cap']==int(3.5*1024**3) and p['minimum_free_bytes']==9*1024**3
    assert p['disk_reserve_bytes']==1024**3 and p['peak_VRAM_bytes']==20*1024**3 and p['peak_RSS_bytes']==12*1024**3
    assert p['worker_seconds']==1200 and p['measurement_seconds']==1170 and p['export_seconds']==450
    assert p['gradient_round_seconds']==600 and p['maximum_run_relative_parameter_distance']==.001
    assert len(p['step_plan'])==6 and [r['label'] for r in p['step_plan']]==['minus00','plus00','minus01','plus01','minus02','plus02']
    for i,r in enumerate(p['step_plan']):
        assert r['rate']==p['base_signed_magnitude']*(10.**(-(i//2)))*(-1 if i%2==0 else 1)
    for folder,bindings in [(root,p['assets_sha256']),(parent,p['parent_required_sha256'])]:
        for name,digest in bindings.items():
            path=folder/name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder)
            assert sha(path)==digest,name
    sys.path.insert(0,str(root))
    from conditioning_scopes import scopes
    blocks,groups=scopes(old['cases'],p['case_ids'])
    assert blocks==p['reference_blocks'] and groups==p['raw_gradient_groups']
    assert all(c['role']=='train' for b in blocks for c in old['cases'][b['begin']:b['begin']+5])
    return p,old


def require_vm(root,parent,idle=True):
    assert platform.system()=='Linux','Manual Linux L4 diagnostic only'
    assert platform.node().split('.')[0]=='forensic-dgp-thesis'
    assert root==(Path.home()/'forensic-dgp'/NAME).resolve() and parent==(Path.home()/'forensic-dgp'/PARENT_NAME).resolve()
    assert Path.cwd().resolve()==root and os.environ.get('TMUX'),'Human manual tmux required'
    assert sys.prefix==str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=3) as response:assert response.read().decode().rstrip().endswith('/g2-standard-4')
    if idle:
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
        assert not r.stdout.strip(),'Competing GPU workload; preserve it'


def diagnose(root,parent,p,old,pin):
    require_vm(root,parent)
    assert not (root/'outputs').exists(),'Immutable used diagnostic; do not rerun'
    out=root/STAGE
    available=shutil.disk_usage(root).free
    if available<p['minimum_free_bytes']:
        out.mkdir(parents=True)
        write(out/'failure.json',dict(stage='pre-model storage preflight',available_free_bytes=available,required_free_bytes=p['minimum_free_bytes'],
            counts=dict(B=0,recognizer=0,gradient_queries=0,temporary_displacements=0,optimizer_updates=0,accepted_updates=0)))
        raise AssertionError('Need9GiB free after installation; retain storage stop')
    sys.path.insert(0,str(root));sys.path.insert(1,str(parent))
    import resource
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from conditioning_model import AnchoredBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash,buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_lossless_v1 import unpack
    from conditioning_scopes import batch_values,query_key,fixed_reference_matrices
    from conditioning_losses import delivered_surrogates
    from conditioning_raw_losses import pixel_and_structure_losses as raw_losses
    from frozen_raw_metrics import deliver,mean_only,pixel_metrics,detail_float,review_groups
    from frozen_capacity_contract import feature_support,exported_pixel_metrics,detail_metric
    from conditioning_contract import decision
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L4'
    assert str(torch.__version__)==p['expected_torch']
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(20261013);torch.cuda.manual_seed_all(20261013)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();out.mkdir(parents=True)
    counts=dict(B=0,recognizer=0,gradient_queries=0,temporary_displacements=0,optimizer_updates=0,accepted_updates=0)
    progress=dict(stage='initialize',counts=counts)
    candidate=None;params=None;initial=None;initial_RNG=None;frozen_before=None
    def stop(_signum,_frame):raise TimeoutError('Finite diagnostic stop; preserve evidence')
    signal.signal(signal.SIGTERM,stop)
    def clock():
        torch.cuda.synchronize()
        assert time.monotonic()-start<p['worker_seconds'] and time.monotonic()-start<p['measurement_seconds'],'Measurement time stop'
        assert shutil.disk_usage(root).free>=p['disk_reserve_bytes'],'Disk reserve stop'
        assert torch.cuda.max_memory_allocated()<=p['peak_VRAM_bytes'],'VRAM stop'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=p['peak_RSS_bytes'],'RSS stop'
        assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<=p['output_byte_cap'],'Output cap stop'
        assert counts['B']<=p['model_forward_cap'] and counts['recognizer']<=p['recognizer_forward_cap']
        assert counts['gradient_queries']<=p['maximum_gradient_queries'] and counts['temporary_displacements']<=6
        assert counts['optimizer_updates']==counts['accepted_updates']==0
    def pixels(path,mode='RGB'):
        with Image.open(path) as im:
            assert im.size==(256,256)
            return np.asarray(im.convert(mode)).copy()
    def tensor(a):return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous().cuda()
    def save_array(path,value):
        with Path(path).open('xb') as stream:np.save(stream,value,allow_pickle=False)
    def flat_values():return np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in params]).copy()
    def load_values(vector):
        assert vector.dtype==np.float32 and vector.shape==(p['conditioning_parameters'],) and np.isfinite(vector).all()
        with torch.no_grad():
            for i,v in enumerate(params):v.copy_(torch.from_numpy(vector[edges[i]:edges[i+1]].reshape(tuple(v.shape))).to(v.device))
        assert state_hash({n:v for n,v in candidate.state_dict().items() if n not in names})==frozen_before
    def rng_values():return dict(torch_CPU_RNG=torch.get_rng_state().numpy(),**{f'torch_CUDA_RNG_{i}':v.cpu().numpy() for i,v in enumerate(torch.cuda.get_rng_state_all())})
    def restore_rng(values):
        torch.set_rng_state(torch.from_numpy(values['torch_CPU_RNG']))
        torch.cuda.set_rng_state_all([torch.from_numpy(values[f'torch_CUDA_RNG_{i}']) for i in range(torch.cuda.device_count())])
    try:
        probe=np.tile(np.arange(256,dtype=np.uint8)[None,None,None,:],(1,3,256,1))
        a=torch.from_numpy(probe.astype(np.float32)/np.float32(255)).cuda()
        rgb,feature=delivered_surrogates(a,torch.from_numpy(probe).cuda())
        assert np.array_equal(rgb.cpu().numpy(),probe.astype(np.float32)/np.float32(255))
        assert np.array_equal(feature.cpu().numpy(),probe.astype(np.float64)/255.)
        write(out/'canonical_pixels.json',dict(complete=True,all256_byte_levels=True,model_calls=0,gradient_queries=0));del a,rgb,feature
        original,_=load_frozen_dgp_restorer(parent/'weights/dgp_v2.pth',expected_sha256=old['original_checkpoint_sha256'],device='cuda')
        identity=FixedObservedIdentity(parent/'weights/w600k_r50.onnx','cuda')
        prior=load_prior(parent/'prior_source',parent/'bank_implementation',old['standalone_prior_sha256']).cuda()
        live=ConditionedBankDGP(original.net,prior,torch.from_numpy(np.load(parent/'initializers/mean_style.npy',allow_pickle=False)).cuda()).cuda()
        live.load_state_dict(torch.load(parent/'initializers/initial_B.pth',map_location='cpu',weights_only=True),strict=True)
        candidate=AnchoredBankDGP(live,root/'readouts.npz',root).cuda();candidate.enable_vm_learning(parent)
        names=p['conditioning_names'];all37=candidate.learning_parameters()
        all37norm=float(np.sqrt(sum(float(v.detach().double().square().sum()) for v in all37)))
        for name,v in candidate.named_parameters():v.requires_grad_(name in names)
        assert [n for n,v in candidate.named_parameters() if v.requires_grad]==names
        params=[dict(candidate.named_parameters())[n] for n in names];edges=np.cumsum([0]+[v.numel() for v in params])
        assert edges[-1]==p['conditioning_parameters']
        initial=flat_values();direction=np.load(root/'saved_direction.npy',allow_pickle=False)
        assert direction.dtype==np.float32 and direction.shape==initial.shape and np.isfinite(direction).all()
        assert state_hash(candidate)==p['initial_model_state']
        frozen_before=state_hash({n:v for n,v in candidate.state_dict().items() if n not in names});buffers=buffer_hash(candidate)
        assert state_hash(original.net)==old['original_state'] and state_hash(identity)==old['recognizer_state']
        original_before=state_hash(original.net);identity_before=state_hash(identity)
        initial_RNG=rng_values();np.savez_compressed(out/'initial_RNG.npz',**initial_RNG)
        candidate.register_forward_hook(lambda *_:counts.__setitem__('B',counts['B']+1))
        identity.encoder.register_forward_hook(lambda *_:counts.__setitem__('recognizer',counts['recognizer']+1))
        save_array(out/'initial_conditioning.npy',initial)
        write(out/'environment.json',dict(GPU=torch.cuda.get_device_name(0),host=platform.node(),torch=torch.__version__,CUDA=torch.version.cuda,
            manual_tmux=True,cudnn_version=torch.backends.cudnn.version(),GPU_capability=list(torch.cuda.get_device_capability(0)),
            method='Repeat32 RAW gradients twice; six fixed signed changes of the existing direction; restore initial state',
            optimizer_constructed=False,new_direction_solved=False,conditioning_names=names,parameters=int(edges[-1]),initial_all37_norm=all37norm,
            initial_model_state=p['initial_model_state'],frozen_state=frozen_before,buffer_state=buffers,original_state=original_before,
            recognizer_state=identity_before,full_epochs=0,precision=dict(parameter='float32',MSE_square='float32',MSE_reduction='float64',
            SSIM_forward='canonical float32 scikit image',SSIM_backward='declared float64 surrogate',feature='float64',
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic),seed=20261013))
        refs={r['id']:r for r in old['references']};blocks=[]
        for block in p['reference_blocks']:
            cs=old['cases'][block['begin']:block['begin']+5]
            x8=np.stack([pixels(parent/c['input']) for c in cs]);target8=np.stack([pixels(parent/refs[c['source_person_or_reference']]['target']) for c in cs])
            mask8=np.stack([pixels(parent/refs[c['source_person_or_reference']]['observed'],'L')>0 for c in cs])
            support8=np.stack([feature_support(m,c['landmarks5_canvas_xy']) for m,c in zip(mask8,cs)])
            baseline,expected=unpack(parent/f"outputs/bank_comparison_v1/baseline/packs/b{block['begin']//5:03d}.npz",block['case_ids'])
            blocks.append((block,cs,x8,target8,mask8,support8,baseline,expected))
        def evaluate(label,exact_controls=False):
            folder=out/label;folder.mkdir(exist_ok=True);(folder/'png').mkdir()
            rows=[];saved=[]
            with torch.no_grad():
                for block,cs,x8,target8,mask8,support8,baseline,expected in blocks:
                    clock();stem=f"b{block['begin']//5:03d}";x,target=tensor(x8),tensor(target8)
                    mask=torch.from_numpy(mask8.astype(np.float32)).cuda()[:,None]
                    grids=torch.from_numpy(np.stack([grid112(v) for v in fixed_reference_matrices(cs,refs)])).cuda()
                    pred=torch.where(mask.bool(),candidate(x),x);raw=pred.permute(0,2,3,1).cpu().numpy().copy()
                    assert np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
                    assert np.array_equal(raw[~mask8],(x8.astype(np.float32)/np.float32(255))[~mask8])
                    if exact_controls:assert np.array_equal(raw,baseline),'Initial or restored DGP parity differs'
                    png=np.stack([deliver(a,c,m) for a,c,m in zip(raw,x8,mask8)]);zero_png=np.stack([deliver(a,c,m) for a,c,m in zip(baseline,x8,mask8)])
                    base=torch.from_numpy(baseline).permute(0,3,1,2).contiguous().cuda()
                    ve=identity.embedding(torch.cat([pred,base,target]),torch.cat([mask]*3),torch.cat([grids]*3)).cpu().numpy().copy()
                    pe=identity.embedding(torch.cat([tensor(png),tensor(zero_png),target]),torch.cat([mask]*3),torch.cat([grids]*3)).cpu().numpy().copy()
                    for e,j in [(ve,0),(pe,1)]:
                        assert np.allclose(e[5:10],expected[:,j],rtol=1e-6,atol=1e-6)
                        assert np.allclose(e[10:15],expected[:,2],rtol=1e-6,atol=1e-6)
                    save_array(folder/(stem+'.npy'),raw);np.savez_compressed(folder/(stem+'_embeddings.npz'),raw=ve,png=pe)
                    saved.append(dict(begin=block['begin'],raw_sha256=sha(folder/(stem+'.npy')),embeddings_sha256=sha(folder/(stem+'_embeddings.npz'))))
                    for i,c in enumerate(cs):
                        Image.fromarray(png[i]).save(folder/'png'/(c['id']+'.png'))
                        mean_raw,mean_png,_=mean_only(raw[i],baseline[i],x8[i],mask8[i])
                        row=dict(id=c['id'],source=c['source'],profile=c['profile'],reference=c['source_person_or_reference'],
                            fresh_recognizer_scores=dict(raw=float(ve[i]@ve[10+i]),png=float(pe[i]@pe[10+i])))
                        for stage,actual,emb in [('raw',raw[i],ve),('png',png[i],pe)]:
                            m=pixel_metrics(actual,target8[i],mask8[i]) if stage=='raw' else exported_pixel_metrics(actual,target8[i],mask8[i])
                            m['landmark_high_frequency_MSE']=detail_float(actual,target8[i],support8[i]) if stage=='raw' else detail_metric(actual,target8[i],support8[i])
                            m['ArcFace_observed_fixed']=float(expected[i,0 if stage=='raw' else 1]@expected[i,2]) if exact_controls else float(emb[i]@emb[10+i])
                            m['constant_mean_shift_only_MSE']=(pixel_metrics(mean_raw,target8[i],mask8[i]) if stage=='raw' else exported_pixel_metrics(mean_png,target8[i],mask8[i]))['MSE']
                            row[stage]=m
                        rows.append(row)
                    del pred,x,target,mask,grids,base
            snapshot=dict(label=label,exact_controls=exact_controls,rows=rows,groups={s:review_groups(rows,s) for s in ['raw','png']},
                blocks=saved,model_state=state_hash(candidate),seconds=time.monotonic()-start)
            write(folder/'snapshot.json',snapshot);print(dict(snapshot=label,cases=100,seconds=snapshot['seconds']),flush=True)
            return snapshot
        base_snapshot=evaluate('initial',True);snapshot_rows={r['id']:r for r in base_snapshot['rows']}
        keys=['RAW/'+g['key'] for g in p['raw_gradient_groups']]
        for repeat in range(2):
            progress.update(stage='repeat_gradient',repeat=repeat);folder=out/f'repeat{repeat:02d}';folder.mkdir();(folder/'aggregates').mkdir();(folder/'samples').mkdir()
            np.savez_compressed(folder/'RNG_before.npz',**rng_values())
            aggregates={key:np.zeros(edges[-1],np.float64) for key in keys};contributions={key:[] for key in keys}
            durations=[];round_start=time.monotonic();forward_proofs=[]
            for serial,(block,cs,x8,target8,mask8,support8,baseline,expected) in enumerate(blocks):
                clock();batch_start=time.monotonic();x,target=tensor(x8),tensor(target8)
                mask=torch.from_numpy(mask8.astype(np.float32)).cuda()[:,None];patch=torch.from_numpy(support8.astype(np.float32)).cuda()[:,None]
                grids=torch.from_numpy(np.stack([grid112(v) for v in fixed_reference_matrices(cs,refs)])).cuda()
                pred=torch.where(mask.bool(),candidate(x),x);raw=pred.detach().permute(0,2,3,1).cpu().numpy().copy()
                same=np.array_equal(raw,baseline)
                forward_proofs.append(dict(begin=block['begin'],exact_initial_forward=same,maximum_absolute_difference=float(np.max(np.abs(raw-baseline))),
                    raw_bytes_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),baseline_bytes_sha256=hashlib.sha256(baseline.tobytes()).hexdigest()))
                if not same:
                    save_array(folder/f"forward_b{block['begin']//5:03d}.npy",raw)
                    write(folder/'forward_failure.json',forward_proofs[-1])
                assert same,'Repeat gradient forward differs from original initial output; retain stop'
                terms=raw_losses(pred,target,mask,patch);base=torch.from_numpy(baseline).permute(0,3,1,2).contiguous().cuda()
                vectors=identity.embedding(torch.cat([pred,base,target]),torch.cat([mask]*3),torch.cat([grids]*3))
                assert np.allclose(vectors[5:10].detach().cpu().numpy(),expected[:,0],rtol=1e-6,atol=1e-6)
                assert np.allclose(vectors[10:15].detach().cpu().numpy(),expected[:,2],rtol=1e-6,atol=1e-6)
                arc=1-(vectors[:5].double()*vectors[10:15].detach().double()).sum(1);values=batch_values(terms,arc)
                for query,value in enumerate(values):
                    clock();key='RAW/'+query_key(block['source'],query)
                    if query<15:
                        row=snapshot_rows[cs[query//3]['id']]['raw'];expected_loss=[row['MSE'],1-row['SSIM'],1-row['ArcFace_observed_fixed']][query%3]
                        tolerance=1e-6 if query%3 else 1e-13
                    else:expected_loss=float(np.mean([snapshot_rows[c['id']]['raw']['landmark_high_frequency_MSE'] for c in cs[1:]]));tolerance=1e-13
                    scalar=float(value.detach())
                    if abs(scalar-expected_loss)>tolerance:
                        write(out/'loss_alignment_failure.json',dict(repeat=repeat,begin=block['begin'],reference=block['reference'],metric_stage='RAW',query=query,
                            actual_forward_value=scalar,expected_scientific_value=expected_loss,alignment_bound=tolerance,absolute_difference=abs(scalar-expected_loss)))
                    assert abs(scalar-expected_loss)<=tolerance,'Scientific/surrogate loss alignment differs'
                    grads=torch.autograd.grad(value,params,retain_graph=query!=15,allow_unused=False);counts['gradient_queries']+=1
                    flat=np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in grads]).copy()
                    assert flat.dtype==np.float32 and np.isfinite(flat).all()
                    norm=float(np.linalg.norm(flat.astype(np.float64)));assert norm>0
                    aggregates[key]+=flat.astype(np.float64)/10
                    contribution=dict(begin=block['begin'],reference=block['reference'],query=query,key=key,value=scalar,scientific_forward_loss=expected_loss,
                        numeric_alignment_bound=tolerance,gradient_vector_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),gradient_norm=norm,
                        existing_direction_dot=float(flat.astype(np.float64)@direction.astype(np.float64)))
                    if block['begin'] in p['sample_gradient_begins'] and query==0:
                        sample=folder/'samples'/f"RAW_b{block['begin']//5:03d}.npy";save_array(sample,flat)
                        contribution.update(sample_file=sample.relative_to(out).as_posix(),sample_sha256=sha(sample))
                    contributions[key].append(contribution);del grads,flat
                del vectors,base,arc,values,terms,pred,x,target,mask,patch,grids
                clock();durations.append(time.monotonic()-batch_start)
                if serial==1:
                    projection=max(durations[:2])*20*1.5
                    write(folder/'timing_projection.json',dict(completed_batch_seconds=durations[:2],projection_seconds=projection,cap_seconds=600,safety_factor=1.5))
                    assert projection<=600,'Gradient round projection exceeds600s'
                assert time.monotonic()-round_start<=600,'Gradient round time stop'
                print(dict(repeat=repeat,reference_batches=serial+1,of=20,gradient_queries=counts['gradient_queries']),flush=True)
            groups=[]
            for i,g in enumerate(p['raw_gradient_groups']):
                key='RAW/'+g['key'];v=aggregates[key];file=folder/'aggregates'/f'g{i:02d}.npy';save_array(file,v)
                groups.append(dict(g,key=key,metric_stage='RAW',file=file.relative_to(out).as_posix(),sha256=sha(file),gradient_norm=float(np.linalg.norm(v)),
                    value=float(np.mean([r['value'] for r in contributions[key]])),contributions=contributions[key],
                    existing_direction_dot=float(v@direction.astype(np.float64))))
            np.savez_compressed(folder/'RNG_after.npz',**rng_values())
            write(folder/'gradients.json',dict(repeat=repeat,snapshot='initial',model_state=state_hash(candidate),groups=groups,
                forward_repeatability=forward_proofs,reference_batch_order=[b['begin'] for b in p['reference_blocks']],seconds=time.monotonic()-round_start))
            del aggregates,contributions
        checks=[]
        for step in p['step_plan']:
            clock();progress.update(stage='signed_response',label=step['label']);load_values(initial)
            rate=np.float32(step['rate']);proposal=initial+rate*direction
            distance=float(np.linalg.norm(proposal.astype(np.float64)-initial.astype(np.float64))/all37norm)
            assert 0<distance<=.001,'Signed diagnostic displacement leaves retained trust region'
            load_values(proposal);counts['temporary_displacements']+=1
            folder=out/step['label'];folder.mkdir();save_array(folder/'conditioning.npy',flat_values())
            snapshot=evaluate(step['label']);gates=decision(base_snapshot['groups'],base_snapshot['groups'],snapshot['groups'])
            record=dict(label=step['label'],nominal_rate=step['rate'],actual_float32_rate=float(rate),
                conditioning_sha256=sha(folder/'conditioning.npy'),snapshot_sha256=sha(folder/'snapshot.json'),
                cumulative_relative_parameter_distance=distance,scientific_decision_information_only=gates,
                accepted_update=False,candidate_selection=False,diagnostic_not_qualification=True)
            write(folder/'response.json',record);checks.append(record)
            print(dict(response=step['label'],raw_gain=gates['stages']['raw']['relative_feature_gain'],PNG_gain=gates['stages']['png']['relative_feature_gain'],accepted_updates=0),flush=True)
        progress.update(stage='restore_initial');load_values(initial);restore_rng(initial_RNG)
        assert np.array_equal(flat_values(),initial) and state_hash(candidate)==p['initial_model_state']
        assert buffer_hash(candidate)==buffers and state_hash(original.net)==original_before and state_hash(identity)==identity_before
        rollback=evaluate('rollback',True);save_array(out/'rollback/conditioning.npy',flat_values())
        assert rollback['groups']==base_snapshot['groups']
        # Persist the restored starting RNG state after the final inference too.
        restore_rng(initial_RNG);np.savez_compressed(out/'restored_RNG.npz',**rng_values())
        clock();assert counts==dict(B=200,recognizer=360,gradient_queries=640,temporary_displacements=6,optimizer_updates=0,accepted_updates=0)
        write(out/'run_results.json',dict(complete_diagnostic=True,protocol_sha256=pin,counts=counts,seconds=time.monotonic()-start,
            peak_VRAM=torch.cuda.max_memory_allocated(),peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            initial_and_restored_state=p['initial_model_state'],frozen_state=frozen_before,buffer_state=buffers,
            original_and_recognizer_unchanged=True,all_initial_gradient_forwards_exact=True,initial_and_rollback_outputs_exact=True,
            RNG_restored=True,response_labels=[r['label'] for r in checks],existing_saved_direction_sha256=sha(root/'saved_direction.npy'),
            new_direction_or_optimizer=False,full_epochs=0,native_pixels_used=0,final_identity_pixels=0,
            derivative_reproducibility_and_response_conclusions_require_independent_audit=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    except BaseException as error:
        restore_error=None;restored_state=None;rng_restored=False
        if candidate is not None and initial is not None:
            try:
                load_values(initial)
                if initial_RNG is not None:restore_rng(initial_RNG);rng_restored=True
                restored_state=state_hash(candidate)
                assert restored_state==p['initial_model_state']
            except BaseException as restoration_error:restore_error=repr(restoration_error)
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),progress=progress,counts=counts,
            seconds=time.monotonic()-start,rollback_error=restore_error,fresh_restored_state_hash=restored_state,RNG_restored=rng_restored,
            accepted_updates=0,model_qualified=False,more_epochs_qualified=False))
        raise


def export(root,parent,p,pin):
    require_vm(root,parent,idle=False);start=time.monotonic()
    manifest=root/'export_manifest.json';assert not manifest.exists()
    files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    payload=sum(f.stat().st_size for f in files)
    assert payload<=p['output_byte_cap']+64*1024**2 and shutil.disk_usage(root).free>=payload+p['disk_reserve_bytes']+1024**2
    result=root/STAGE/'run_results.json';failure=root/STAGE/'failure.json'
    write(manifest,dict(protocol_sha256=pin,parent_protocol_sha256=PARENT_PIN,parent_required_sha256=p['parent_required_sha256'],
        files={f.relative_to(root).as_posix():sha(f) for f in files},accepted_updates=0,optimizer_updates=0,
        complete_diagnostic=result.exists(),complete_pilot=False,failure_present=failure.exists(),training_success_not_implied=True))
    archive=Path.home()/(STEM+'-results.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in files+[manifest]:
            assert time.monotonic()-start<p['export_seconds'],'Export time stop'
            assert shutil.disk_usage(root).free>=p['disk_reserve_bytes']+file.stat().st_size,'Export reserve stop'
            tar.add(file,arcname=NAME+'/'+file.relative_to(root).as_posix(),recursive=False)
            assert archive.stat().st_size<=p['output_byte_cap']+64*1024**2
    digest=sha(archive);assert time.monotonic()-start<p['export_seconds'] and shutil.disk_usage(root).free>=p['disk_reserve_bytes']
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    record=dict(complete=True,archive_sha256=digest,bytes=archive.stat().st_size,seconds=time.monotonic()-start,
        accepted_updates=0,optimizer_updates=0,training_success_not_implied=True)
    write(Path.home()/(STEM+'-export.json'),record);print(record,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--parent-root',required=True,type=Path)
    parser.add_argument('--protocol-sha',required=True);action=parser.add_mutually_exclusive_group(required=True)
    for flag in ['verify-transfer','diagnose','export']:action.add_argument('--'+flag,action='store_true')
    a=parser.parse_args();root,parent=a.root.resolve(),a.parent_root.resolve()
    if a.diagnose:require_vm(root,parent)
    p,old=verify(root,parent,a.protocol_sha)
    if a.verify_transfer:print(dict(complete=True,packet_assets=len(p['assets_sha256']),parent_assets=len(p['parent_required_sha256']),
        cases=p['paired_TRAIN_cases'],maximum_gradient_queries=p['maximum_gradient_queries'],temporary_displacements=p['temporary_displacements'],
        optimizer_updates=p['optimizer_updates'],actual_neural_calls=0,available_free_bytes=shutil.disk_usage(root).free,minimum_free_bytes=p['minimum_free_bytes']),flush=True)
    elif a.diagnose:diagnose(root,parent,p,old,a.protocol_sha)
    else:export(root,parent,p,a.protocol_sha)


if __name__=='__main__':main()

"""Human-manual L4 finite learning: fresh derivatives and checked acceptance."""
import argparse
from datetime import datetime,timezone
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

NAME='cctv_dgp_nearby_gradient_learning_v1_vm'
STEM='cctv-dgp-nearby-gradient-learning-v1'
PARENT_NAME='cctv_dgp_bank_photometric_guard_v1_vm'
PARENT_PIN='f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
STAGE='outputs/nearby_gradient_learning_v1'
RESPONSE_NAME='cctv_dgp_conditioning_response_v1_vm'
RESPONSE_PIN='e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root,parent,pin):
    assert sha(root/'protocol.json')==pin and sha(parent/'protocol.json')==PARENT_PIN
    p,old=read(root/'protocol.json'),read(parent/'protocol.json')
    assert p['name']==NAME and p['parent_protocol_sha256']==PARENT_PIN
    assert p['case_ids']==old['preview_case_ids'] and p['paired_TRAIN_cases']==100
    assert p['maximum_accepted_updates']==1 and p['maximum_trials_per_update']==3
    assert p['maximum_gradient_queries']==960 and p['model_forward_cap']==200 and p['recognizer_forward_cap']==340
    assert p['RAW_groups_per_round']==32 and p['PNG_groups_per_round']==0
    assert p['optimizer_constraints']=='96 nearby-state RAW constraints; actual original RAW/PNG gates'
    assert p['sampling_sites']==['initial','plus02','minus02'] and p['direction_solves']==1
    assert p['conditioning_parameters']==2580864 and len(p['conditioning_names'])==12
    assert p['worker_seconds']==1500 and p['measurement_seconds']==1350 and p['gradient_round_seconds']==600
    assert p['external_seconds']==1530 and p['export_seconds']==420 and p['export_external_seconds']==450
    assert p['kill_grace_seconds']==30 and p['minimum_free_bytes']==9*1024**3
    assert p['disk_reserve_bytes']==1024**3 and p['output_byte_cap']==3584*1024**2
    assert p['peak_VRAM_bytes']==20*1024**3 and p['peak_RSS_bytes']==12*1024**3
    assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert p['maximum_run_relative_parameter_distance']==.001 and p['maximum_rate']==.02
    assert p['backtrack_factors']==[1.,.5,.25]
    assert p['native_pixels_used']==p['final_identity_pixels']==p['full_epochs']==0
    assert not p['app_promotion'] and not p['automatic_follow_on'] and not p['failed_run_resume_allowed']
    response=parent.parent/RESPONSE_NAME
    assert sha(response/'protocol.json')==RESPONSE_PIN==p['response_protocol_sha256']
    for folder,bindings in [(root,p['assets_sha256']),(parent,p['parent_required_sha256']),(response,p['response_required_sha256'])]:
        for name,digest in bindings.items():
            path=folder/name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder)
            assert sha(path)==digest,name
    sys.path.insert(0,str(root))
    from conditioning_scopes import scopes
    blocks,groups=scopes(old['cases'],p['case_ids'])
    assert blocks==p['reference_blocks'] and groups==p['raw_gradient_groups']
    assert all(n.startswith(('live.style_conditioner.','live.spatial_conditioners.')) for n in p['conditioning_names'])
    assert read(root/'activation_audit.json')['complete'] and read(root/'subspace_audit.json')['complete']
    return p,old


def require_vm(root,parent,check_idle=True):
    assert platform.system()=='Linux','Manual Linux L4 nearby-state training only'
    assert platform.node().split('.')[0]=='forensic-dgp-thesis'
    assert root==(Path.home()/'forensic-dgp'/NAME).resolve() and parent==(Path.home()/'forensic-dgp'/PARENT_NAME).resolve()
    assert Path.cwd().resolve()==root and os.environ.get('TMUX'),'Human manual tmux required'
    assert sys.prefix==str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    if check_idle:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
        assert not result.stdout.strip(),'Competing GPU workload; preserve it'


def run(root,parent,p,old,pin):
    require_vm(root,parent)
    out=root/STAGE
    assert not (root/'outputs').exists(),'Immutable used pilot; no unchanged rerun'
    available=shutil.disk_usage(root).free
    if available<p['minimum_free_bytes']:
        out.mkdir(parents=True)
        write(out/'failure.json',dict(stage='pre-model storage preflight',available_free_bytes=available,
            required_free_bytes=p['minimum_free_bytes'],neural_calls=0,gradient_queries=0,accepted_updates=0,parameter_proposals=0))
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
    from conditioning_contract import solve_direction,bounded_rate,decision
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L4'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(20261013);torch.cuda.manual_seed_all(20261013)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();out.mkdir(parents=True)
    counts=dict(B=0,recognizer=0,gradient_queries=0,backwards=0,accepted_updates=0,parameter_proposals=0,derivative_sampling_displacements=0)
    progress=dict(stage='initialize',counts=counts)
    candidate=None;accepted=None;params=None;frozen_before=None;initial_RNG=None;initial=None
    def stop(_signal,_frame):raise TimeoutError('Finite manual learning stop; retain current evidence')
    signal.signal(signal.SIGTERM,stop)
    def clock():
        torch.cuda.synchronize()
        assert time.monotonic()-start<p['worker_seconds'],'Worker time stop'
        assert time.monotonic()-start<p['measurement_seconds'],'Learning measurement stop'
        assert shutil.disk_usage(root).free>=p['disk_reserve_bytes'],'Disk reserve stop'
        assert torch.cuda.max_memory_allocated()<=p['peak_VRAM_bytes'],'VRAM stop'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=p['peak_RSS_bytes'],'RSS stop'
        assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<=p['output_byte_cap'],'Output cap stop'
        assert counts['B']<=p['model_forward_cap'] and counts['recognizer']<=p['recognizer_forward_cap']
        assert counts['gradient_queries']<=p['maximum_gradient_queries'] and counts['accepted_updates']<=1 and counts['parameter_proposals']<=3
        assert counts['derivative_sampling_displacements']<=2
    def pixels(path,mode='RGB'):
        with Image.open(path) as im:
            assert im.size==(256,256)
            return np.asarray(im.convert(mode)).copy()
    def tensor(a):return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous().cuda()
    def save_array(path,value):
        with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
    def flat_values():return np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in params]).copy()
    def load_values(vector):
        assert vector.dtype==np.float32 and vector.shape==(p['conditioning_parameters'],) and np.isfinite(vector).all()
        with torch.no_grad():
            for i,param in enumerate(params):
                param.copy_(torch.from_numpy(vector[edges[i]:edges[i+1]].reshape(tuple(param.shape))).to(param.device))
        assert state_hash({n:v for n,v in candidate.state_dict().items() if n not in names})==frozen_before
    try:
        # Check every canonical byte before constructing any model.
        probe=np.tile(np.arange(256,dtype=np.uint8)[None,None,None,:],(1,3,256,1))
        a=torch.from_numpy(probe.astype(np.float32)/np.float32(255)).cuda()
        rgb,feature=delivered_surrogates(a,torch.from_numpy(probe).cuda())
        assert np.array_equal(rgb.cpu().numpy(),probe.astype(np.float32)/np.float32(255))
        assert np.array_equal(feature.cpu().numpy(),probe.astype(np.float64)/255.)
        write(out/'canonical_pixels.json',dict(complete=True,all256_byte_levels=True,model_calls=0,gradient_queries=0))
        del a,rgb,feature
        original,_=load_frozen_dgp_restorer(parent/'weights/dgp_v2.pth',expected_sha256=old['original_checkpoint_sha256'],device='cuda')
        identity=FixedObservedIdentity(parent/'weights/w600k_r50.onnx','cuda')
        prior=load_prior(parent/'prior_source',parent/'bank_implementation',old['standalone_prior_sha256']).cuda()
        live=ConditionedBankDGP(original.net,prior,torch.from_numpy(np.load(parent/'initializers/mean_style.npy',allow_pickle=False)).cuda()).cuda()
        live.load_state_dict(torch.load(parent/'initializers/initial_B.pth',map_location='cpu',weights_only=True),strict=True)
        candidate=AnchoredBankDGP(live,root/'readouts.npz',root).cuda()
        candidate.enable_vm_learning(parent)
        names=p['conditioning_names'];all37=candidate.learning_parameters()
        initial_all37_norm=float(np.sqrt(sum(float(v.detach().double().square().sum()) for v in all37)))
        for name,param in candidate.named_parameters():param.requires_grad_(name in names)
        assert [n for n,v in candidate.named_parameters() if v.requires_grad]==names
        params=[dict(candidate.named_parameters())[n] for n in names]
        edges=np.cumsum([0]+[v.numel() for v in params])
        assert edges[-1]==p['conditioning_parameters']
        initial=flat_values();accepted=initial.copy()
        assert state_hash(candidate)==p['initial_model_state']
        frozen_before=state_hash({n:v for n,v in candidate.state_dict().items() if n not in names})
        buffers=buffer_hash(candidate)
        assert state_hash(original.net)==old['original_state'] and state_hash(identity)==old['recognizer_state']
        original_before=state_hash(original.net);identity_before=state_hash(identity)
        candidate.register_forward_hook(lambda *_:counts.__setitem__('B',counts['B']+1))
        identity.encoder.register_forward_hook(lambda *_:counts.__setitem__('recognizer',counts['recognizer']+1))
        save_array(out/'initial_conditioning.npy',initial)
        write(out/'environment.json',dict(GPU=torch.cuda.get_device_name(0),host=platform.node(),torch=torch.__version__,CUDA=torch.version.cuda,
            manual_tmux=True,method='Three-site96 RAW constrained descent with actual original RAW/PNG acceptance',
            torch_optimizer_constructed=False,stateless_gradient_descent_is_actual_training=True,
            conditioning_names=names,parameters=int(edges[-1]),initial_all37_norm=initial_all37_norm,
            initial_model_state=p['initial_model_state'],frozen_state=frozen_before,buffer_state=buffers,
            original_state=original_before,recognizer_state=identity_before,full_epochs=0,
            expected_torch=p['expected_torch'],precision=dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True)))
        assert str(torch.__version__)==p['expected_torch']
        def rng_values():return dict(torch_CPU_RNG=torch.get_rng_state().numpy(),**{f'torch_CUDA_RNG_{i}':s.cpu().numpy() for i,s in enumerate(torch.cuda.get_rng_state_all())})
        def restore_rng(v):
            torch.set_rng_state(torch.from_numpy(v['torch_CPU_RNG']))
            torch.cuda.set_rng_state_all([torch.from_numpy(v[f'torch_CUDA_RNG_{i}']) for i in range(torch.cuda.device_count())])
        initial_RNG=rng_values();np.savez_compressed(out/'initial_RNG.npz',**initial_RNG)
        refs={r['id']:r for r in old['references']};blocks=[]
        for block in p['reference_blocks']:
            cs=old['cases'][block['begin']:block['begin']+5]
            x8=np.stack([pixels(parent/c['input']) for c in cs])
            target8=np.stack([pixels(parent/refs[c['source_person_or_reference']]['target']) for c in cs])
            mask8=np.stack([pixels(parent/refs[c['source_person_or_reference']]['observed'],'L')>0 for c in cs])
            support8=np.stack([feature_support(m,c['landmarks5_canvas_xy']) for m,c in zip(mask8,cs)])
            baseline,expected=unpack(parent/f"outputs/bank_comparison_v1/baseline/packs/b{block['begin']//5:03d}.npz",block['case_ids'])
            blocks.append((block,cs,x8,target8,mask8,support8,baseline,expected))
        def evaluate(label,exact_controls=False):
            folder=out/label;folder.mkdir(exist_ok=True);(folder/'png').mkdir()
            rows=[];saved=[]
            with torch.no_grad():
                for block,cs,x8,target8,mask8,support8,baseline,expected in blocks:
                    clock();begin=block['begin'];stem=f'b{begin//5:03d}'
                    x,target=tensor(x8),tensor(target8)
                    mask=torch.from_numpy(mask8.astype(np.float32)).cuda()[:,None]
                    grids=torch.from_numpy(np.stack([grid112(v) for v in fixed_reference_matrices(cs,refs)])).cuda()
                    pred=torch.where(mask.bool(),candidate(x),x)
                    raw=pred.permute(0,2,3,1).cpu().numpy().copy()
                    assert np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
                    assert np.array_equal(raw[~mask8],(x8.astype(np.float32)/np.float32(255))[~mask8])
                    if exact_controls:assert np.array_equal(raw,baseline),'Initial DGP parity differs'
                    if label in p['sampling_sites']:
                        expected_raw=np.load(parent.parent/RESPONSE_NAME/STAGE.replace('nearby_gradient_learning_v1','conditioning_response_v1')/label/(stem+'.npy'),allow_pickle=False)
                        assert np.array_equal(raw,expected_raw),'Sampling-state RAW forward differs from retained measurement'
                    png=np.stack([deliver(a,c,m) for a,c,m in zip(raw,x8,mask8)])
                    zero_png=np.stack([deliver(a,c,m) for a,c,m in zip(baseline,x8,mask8)])
                    base=torch.from_numpy(baseline).permute(0,3,1,2).contiguous().cuda()
                    ve=identity.embedding(torch.cat([pred,base,target]),torch.cat([mask]*3),torch.cat([grids]*3)).cpu().numpy().copy()
                    pe=identity.embedding(torch.cat([tensor(png),tensor(zero_png),target]),torch.cat([mask]*3),torch.cat([grids]*3)).cpu().numpy().copy()
                    for e,j in [(ve,0),(pe,1)]:
                        assert np.allclose(e[5:10],expected[:,j],rtol=1e-6,atol=1e-6)
                        assert np.allclose(e[10:15],expected[:,2],rtol=1e-6,atol=1e-6)
                    save_array(folder/(stem+'.npy'),raw)
                    np.savez_compressed(folder/(stem+'_embeddings.npz'),raw=ve,png=pe)
                    saved.append(dict(begin=begin,raw_sha256=sha(folder/(stem+'.npy')),embeddings_sha256=sha(folder/(stem+'_embeddings.npz'))))
                    for i,c in enumerate(cs):
                        Image.fromarray(png[i]).save(folder/'png'/(c['id']+'.png'))
                        mean_raw,mean_png,_=mean_only(raw[i],baseline[i],x8[i],mask8[i])
                        row=dict(id=c['id'],source=c['source'],profile=c['profile'],reference=c['source_person_or_reference'])
                        for stage,actual,emb in [('raw',raw[i],ve),('png',png[i],pe)]:
                            m=pixel_metrics(actual,target8[i],mask8[i]) if stage=='raw' else exported_pixel_metrics(actual,target8[i],mask8[i])
                            m['landmark_high_frequency_MSE']=detail_float(actual,target8[i],support8[i]) if stage=='raw' else detail_metric(actual,target8[i],support8[i])
                            if exact_controls:
                                j=0 if stage=='raw' else 1
                                m['ArcFace_observed_fixed']=float(expected[i,j]@expected[i,2])
                            else:
                                m['ArcFace_observed_fixed']=float(emb[i]@emb[10+i])
                            m['constant_mean_shift_only_MSE']=(pixel_metrics(mean_raw,target8[i],mask8[i]) if stage=='raw' else exported_pixel_metrics(mean_png,target8[i],mask8[i]))['MSE']
                            row[stage]=m
                        rows.append(row)
                    del pred,x,target,mask,grids,base
            grouped={stage:review_groups(rows,stage) for stage in ['raw','png']}
            snapshot=dict(label=label,rows=rows,groups=grouped,blocks=saved,model_state=state_hash(candidate),exact_controls=exact_controls,seconds=time.monotonic()-start)
            write(folder/'snapshot.json',snapshot)
            print(dict(snapshot=label,cases=100,seconds=snapshot['seconds']),flush=True)
            return snapshot
        base_snapshot=evaluate('initial',True);current_snapshot=base_snapshot
        response=parent.parent/RESPONSE_NAME/'outputs/conditioning_response_v1'
        raw_groups=p['raw_gradient_groups']
        all_groups=[dict(g,key='RAW/'+g['key'],metric_stage='RAW') for g in raw_groups]
        sampling_records=[];trial_records=[];qualified=False
        for site in p['sampling_sites']:
            clock();progress.update(stage='nearby_group_gradients',site=site)
            sample=initial if site=='initial' else np.load(response/site/'conditioning.npy',allow_pickle=False)
            if site!='initial':
                load_values(sample);counts['derivative_sampling_displacements']+=1
                current_snapshot=evaluate(site)
            else:current_snapshot=base_snapshot
            save_array(out/site/'conditioning.npy',sample)
            folder=out/('gradients_'+site);folder.mkdir();(folder/'aggregates').mkdir();(folder/'samples').mkdir()
            grad_start=time.monotonic()
            aggregates={g['key']:np.zeros(int(edges[-1]),np.float64) for g in all_groups}
            contributions={g['key']:[] for g in all_groups};durations=[]
            snapshot_rows={r['id']:r for r in current_snapshot['rows']}
            round_state=state_hash(candidate)
            for serial,(block,cs,x8,target8,mask8,support8,baseline,expected) in enumerate(blocks):
                clock();batch_start=time.monotonic()
                x,target=tensor(x8),tensor(target8)
                mask=torch.from_numpy(mask8.astype(np.float32)).cuda()[:,None]
                support=torch.from_numpy(support8.astype(np.float32)).cuda()[:,None]
                grids=torch.from_numpy(np.stack([grid112(v) for v in fixed_reference_matrices(cs,refs)])).cuda()
                pred=torch.where(mask.bool(),candidate(x),x)
                raw=pred.detach().permute(0,2,3,1).cpu().numpy().copy()
                stored=np.load(out/current_snapshot['label']/f"b{block['begin']//5:03d}.npy",allow_pickle=False)
                assert np.array_equal(raw,stored),'Gradient forward differs from actual evaluated snapshot'
                stages=[('RAW',pred,raw_losses(pred,target,mask,support))]
                for stage,actual,terms in stages:
                    base=torch.from_numpy(baseline).permute(0,3,1,2).contiguous().cuda() if stage=='RAW' else tensor(np.stack([deliver(a,c,m) for a,c,m in zip(baseline,x8,mask8)]))
                    vectors=identity.embedding(torch.cat([actual,base,target]),torch.cat([mask]*3),torch.cat([grids]*3))
                    j=0 if stage=='RAW' else 1
                    assert np.allclose(vectors[5:10].detach().cpu().numpy(),expected[:,j],rtol=1e-6,atol=1e-6)
                    assert np.allclose(vectors[10:15].detach().cpu().numpy(),expected[:,2],rtol=1e-6,atol=1e-6)
                    arc=1-(vectors[:5].double()*vectors[10:15].detach().double()).sum(1)
                    values=batch_values(terms,arc)
                    for query,value in enumerate(values):
                        clock();key=stage+'/'+query_key(block['source'],query)
                        if query<15:
                            row=snapshot_rows[cs[query//3]['id']][stage.lower()]
                            expected_loss=[row['MSE'],1-row['SSIM'],1-row['ArcFace_observed_fixed']][query%3]
                            tolerance=1e-6 if query%3 else 1e-13
                        else:
                            expected_loss=float(np.mean([snapshot_rows[c['id']][stage.lower()]['landmark_high_frequency_MSE'] for c in cs[1:]]))
                            tolerance=1e-13
                        if abs(float(value.detach())-expected_loss)>tolerance:
                            write(out/'loss_alignment_failure.json',dict(site=site,
                                begin=block['begin'],reference=block['reference'],metric_stage=stage,
                                query=query,key=key,actual_forward_value=float(value.detach()),
                                expected_scientific_value=expected_loss,alignment_bound=tolerance,
                                absolute_difference=abs(float(value.detach())-expected_loss),
                                snapshot=current_snapshot['label'],candidate_state=round_state,
                                raw_helper_sha256=sha(root/'conditioning_raw_losses.py'),
                                PNG_helper_sha256=sha(root/'conditioning_losses.py')))
                        assert abs(float(value.detach())-expected_loss)<=tolerance,'Scientific/surrogate loss alignment differs'
                        grads=torch.autograd.grad(value,params,retain_graph=query!=15,allow_unused=False)
                        counts['gradient_queries']+=1
                        flat=np.concatenate([g.detach().cpu().numpy().reshape(-1) for g in grads]).copy()
                        assert flat.dtype==np.float32 and np.isfinite(flat).all()
                        aggregates[key]+=flat.astype(np.float64)/10
                        contribution=dict(begin=block['begin'],reference=block['reference'],query=query,key=key,
                            value=float(value.detach()),scientific_forward_loss=expected_loss,numeric_alignment_bound=tolerance,
                            gradient_vector_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),gradient_norm=float(np.linalg.norm(flat.astype(np.float64))))
                        contributions[key].append(contribution)
                        if block['begin'] in p['sample_gradient_begins'] and query==0:
                            sample=folder/'samples'/f"{stage}_b{block['begin']//5:03d}.npy"
                            save_array(sample,flat);contribution['sample_file']=sample.relative_to(out).as_posix();contribution['sample_sha256']=sha(sample)
                        del grads,flat
                    del vectors,base,arc,values
                del pred,stages,terms,actual,x,target,mask,support,grids
                torch.cuda.synchronize();durations.append(time.monotonic()-batch_start)
                if serial==1:
                    projection=max(durations[:2])*20*1.5
                    write(folder/'timing_projection.json',dict(completed_batch_seconds=durations[:2],projection_seconds=projection,cap_seconds=600,safety_factor=1.5))
                    assert projection<=600,'Gradient round projection exceeds600s'
                assert time.monotonic()-grad_start<600,'Gradient round stop'
                print(dict(site=site,reference_batches=serial+1,of=20,gradient_queries=counts['gradient_queries']),flush=True)
            records=[]
            for i,g in enumerate(all_groups):
                key=g['key'];path=folder/'aggregates'/f'g{i:02d}.npy';save_array(path,aggregates[key])
                assert len(contributions[key])==10
                records.append(dict(g,site=site,file=path.relative_to(out).as_posix(),sha256=sha(path),
                    value=float(np.mean([r['value'] for r in contributions[key]])),
                    gradient_norm=float(np.linalg.norm(aggregates[key])),contributions=contributions[key]))
            assert state_hash(candidate)==round_state and buffer_hash(candidate)==buffers
            site_record=dict(site=site,model_state=round_state,snapshot=site,groups=records,seconds=time.monotonic()-grad_start)
            write(folder/'gradients.json',site_record);sampling_records.append(site_record)
            del aggregates,contributions
        load_values(initial);assert state_hash(candidate)==p['initial_model_state'];current_snapshot=base_snapshot
        vectors=[np.load(out/r['file'],allow_pickle=False,mmap_mode='r') for site in sampling_records for r in site['groups']]
        rows=[r for site in sampling_records for r in site['groups']]
        direction,certificate=solve_direction(vectors,rows);del vectors
        save_array(out/'direction.npy',direction)
        write(out/'direction_certificate.json',dict(complete=True,groups=rows,certificate=certificate,direction_sha256=sha(out/'direction.npy')))
        ceiling=bounded_rate(initial,initial,direction,initial_all37_norm)
        for attempt,factor in enumerate(p['backtrack_factors']):
            clock();label=f'trial{attempt:02d}';rate=np.float32(ceiling*factor)
            proposal=initial+rate*direction
            relative=float(np.linalg.norm(proposal.astype(np.float64)-initial.astype(np.float64))/initial_all37_norm)
            assert relative<=.001,'Rounded cumulative trust-region distance exceeds retained limit'
            load_values(proposal);counts['parameter_proposals']+=1
            progress.update(stage='actual_proposal_acceptance',trial=label)
            (out/label).mkdir();save_array(out/label/'conditioning.npy',proposal)
            snapshot=evaluate(label)
            checked=decision(base_snapshot['groups'],base_snapshot['groups'],snapshot['groups'])
            trial=dict(label=label,attempt=attempt,rate=float(rate),rate_ceiling=ceiling,
                before_snapshot='initial',direction_sha256=sha(out/'direction.npy'),snapshot_sha256=sha(out/label/'snapshot.json'),
                conditioning_sha256=sha(out/label/'conditioning.npy'),cumulative_relative_parameter_distance=relative,decision=checked)
            write(out/label/'decision.json',trial);trial_records.append(trial)
            print(dict(trial=label,accepted=checked['accept_step'],one_percent=checked['one_percent_checkpoint_pass'],
                raw_gain=checked['stages']['raw']['relative_feature_gain'],PNG_gain=checked['stages']['png']['relative_feature_gain']),flush=True)
            if checked['accept_step']:
                accepted=proposal.copy();current_snapshot=snapshot;counts['accepted_updates']+=1
                full_state={n:v.detach().cpu() for n,v in candidate.state_dict().items()}
                torch.save(full_state,out/'accepted_component_full_state.pth');del full_state
                np.savez_compressed(out/'accepted01.npz',conditioning=accepted,**rng_values())
                write(out/'accepted01.json',dict(model_state=state_hash(candidate),full_state_sha256=sha(out/'accepted_component_full_state.pth'),
                    checkpoint_sha256=sha(out/'accepted01.npz'),conditioning_names=names,accepted_update=1,trial=label,
                    frozen_state=frozen_before,buffer_state=buffers,original_state=original_before,recognizer_state=identity_before,
                    stateless_optimizer='Three-site96 constrained descent; no momentum/scheduler',protocol_sha256=pin,
                    exact_continuation_needs_code_environment_data_and_rng_bindings=True,research_component_not_app_model=True))
                qualified=checked['one_percent_checkpoint_pass'];break
            load_values(initial);assert state_hash(candidate)==p['initial_model_state']
        load_values(initial);restore_rng(initial_RNG)
        rollback=evaluate('rollback',True);save_array(out/'rollback/conditioning.npy',initial)
        assert rollback['groups']==base_snapshot['groups'] and state_hash(candidate)==p['initial_model_state']
        restore_rng(initial_RNG);np.savez_compressed(out/'restored_RNG.npz',**rng_values())
        write(out/'rollback.json',dict(complete=True,initial_model_state=p['initial_model_state'],fresh_state=state_hash(candidate),
            frozen_state=state_hash({n:v for n,v in candidate.state_dict().items() if n not in names}),buffer_state=buffer_hash(candidate),
            original_state=state_hash(original.net),recognizer_state=state_hash(identity),RNG_restored=True,
            rollback_outputs_exact=True,accepted_state_preserved_separately=counts['accepted_updates']>0))
        final=decision(base_snapshot['groups'],base_snapshot['groups'],current_snapshot['groups'])
        assert counts['accepted_updates']==1,'No actual raw/PNG preserving nearby-state proposal; retain stop'
        assert qualified and final['one_percent_checkpoint_pass'],'One-update budget ended below retained1% structure/preservation gate'
        assert state_hash({n:v for n,v in candidate.state_dict().items() if n not in names})==frozen_before
        assert buffer_hash(candidate)==buffers and state_hash(original.net)==original_before and state_hash(identity)==identity_before
        assert all(v.grad is None for v in candidate.parameters()) and all(not m.training for m in candidate.modules())
        clock();verify(root,parent,pin);clock()
        write(out/'run_results.json',dict(complete=True,counts=counts,sampling_sites=sampling_records,direction_certificate_sha256=sha(out/'direction_certificate.json'),trials=trial_records,
            final_snapshot=current_snapshot['label'],final_checkpoint_gate=final,final_model_state=current_snapshot['model_state'],fresh_restored_model_state=state_hash(candidate),RNG_restored=True,
            initial_model_state=p['initial_model_state'],frozen_state_unchanged=True,buffers_unchanged=True,
            original_and_recognizer_unchanged=True,full_epochs=0,accepted_updates_are_actual_training=True,
            raw_PNG_independent_audit_and_all_visual_sheets_pending=True,
            full_TRAIN_and_native_DEV_qualification_pending=True,model_qualified=False,more_epochs_qualified=False,
            native_pixels_used=0,final_identity_pixels=0,peak_VRAM=torch.cuda.max_memory_allocated(),
            peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-start))
    except BaseException as error:
        restore_error=None
        if candidate is not None and accepted is not None and params is not None and frozen_before is not None:
            try:
                load_values(initial)
                if initial_RNG is not None:restore_rng(initial_RNG)
                write(out/'failure_restoration.json',dict(complete=True,fresh_state=state_hash(candidate),expected_initial=p['initial_model_state'],
                    frozen_state=state_hash({n:v for n,v in candidate.state_dict().items() if n not in names}),buffer_state=buffer_hash(candidate),RNG_restored=initial_RNG is not None))
                assert state_hash(candidate)==p['initial_model_state']
            except BaseException as failure:restore_error=repr(failure)
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            progress=progress,counts=counts,seconds=time.monotonic()-start,
            rollback_to_initial_component_error=restore_error,
            fresh_failure_restoration_receipt=(out/'failure_restoration.json').exists(),model_qualified=False,more_epochs_qualified=False))
        raise


def export(root,parent,p,pin):
    require_vm(root,parent,check_idle=False)
    started=time.monotonic();manifest=root/'export_manifest.json';assert not manifest.exists()
    files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    payload=sum(f.stat().st_size for f in files)
    assert payload<=p['output_byte_cap']+64*1024**2 and shutil.disk_usage(root).free>=payload+p['disk_reserve_bytes']+1024**2
    result=root/STAGE/'run_results.json';failure=root/STAGE/'failure.json'
    # A stopped learned run can contain accepted research states. Never hardcode
    # zero updates from the archive exporter.
    count=read(result if result.exists() else failure).get('counts',{}).get('accepted_updates',0) if result.exists() or failure.exists() else None
    write(manifest,dict(protocol_sha256=pin,parent_protocol_sha256=PARENT_PIN,parent_required_sha256=p['parent_required_sha256'],
        files={f.relative_to(root).as_posix():sha(f) for f in files},accepted_updates=count,
        complete_pilot=result.exists(),failure_present=failure.exists(),training_success_not_implied=True))
    archive=Path.home()/(STEM+'-results.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in files+[manifest]:
            assert time.monotonic()-started<p['export_seconds'],'Export time stop'
            assert shutil.disk_usage(root).free>=p['disk_reserve_bytes']+file.stat().st_size,'Export reserve stop'
            tar.add(file,arcname=NAME+'/'+file.relative_to(root).as_posix(),recursive=False)
            assert archive.stat().st_size<=p['output_byte_cap']+64*1024**2
    digest=sha(archive);assert time.monotonic()-started<p['export_seconds'] and shutil.disk_usage(root).free>=p['disk_reserve_bytes']
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    record=dict(complete=True,archive_sha256=digest,bytes=archive.stat().st_size,seconds=time.monotonic()-started,
                accepted_updates=count,training_success_not_implied=True)
    write(Path.home()/(STEM+'-export.json'),record);print(record,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--parent-root',required=True,type=Path);parser.add_argument('--protocol-sha',required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    for flag in ['verify-transfer','run','export']:action.add_argument('--'+flag,action='store_true')
    a=parser.parse_args();root,parent=a.root.resolve(),a.parent_root.resolve()
    if a.run:require_vm(root,parent)
    p,old=verify(root,parent,a.protocol_sha)
    if a.verify_transfer:print(dict(complete=True,packet_assets=len(p['assets_sha256']),parent_assets=len(p['parent_required_sha256']),
        cases=p['paired_TRAIN_cases'],maximum_accepted_updates=p['maximum_accepted_updates'],maximum_gradient_queries=p['maximum_gradient_queries'],actual_neural_calls=0,
        available_free_bytes=shutil.disk_usage(root).free,minimum_free_bytes=p['minimum_free_bytes']),flush=True)
    elif a.run:run(root,parent,p,old,a.protocol_sha)
    else:export(root,parent,p,a.protocol_sha)


if __name__=='__main__':main()

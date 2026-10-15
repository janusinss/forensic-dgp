"""Human-manual derivative-only decoder repair; never call historical training."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from cctv_dgp_decoder_hinge_repair_v1_contract import (
    NAME,STEM,STAGE,SOURCE_NAME,PARENT_NAME,REPLAY_NAME,LEGACY_HELPER_SHA,
    CORRECTED_HELPER_SHA,PROBES,UPDATES,read,write,sha,array_sha,
    verify_transfer,require_manual,continuation_eligible,
)


def machine_guard(root,idle=False):
    require_manual(root)
    request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    if idle:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
        assert not result.stdout.strip(),'Competing GPU workload; preserve it'


@contextmanager
def source_cwd(source):
    before=Path.cwd()
    assert source==(Path.home()/'forensic-dgp'/SOURCE_NAME).resolve()
    os.chdir(source)
    try:yield
    finally:os.chdir(before)


def file_module(path,name,expected):
    assert sha(path)==expected
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert Path(module.__file__).resolve()==Path(path).resolve()
    return module


def run(root,source,parent,replay,p,old,data,pin):
    machine_guard(root,idle=True)
    assert not (root/'outputs').exists(),'Immutable used run; retain all evidence'
    out=root/STAGE;out.mkdir(parents=True)
    start=time.monotonic();torch=None
    counts=dict(optimizer_updates=0,gradient_queries=0,backwards=0,
        original_batch_calls=0,prior_feature_batch_calls=0,decoder_batch_calls=0,recognizer_batch_calls=0)
    progress=dict(stage='resource preflight',updates=0)
    model=optimizer=shadow=None;snapshot_labels=[];fit_outcomes=[]
    def clock():
        assert time.monotonic()-start<=p['budgets']['worker_seconds'],'Finite worker stop'
        assert shutil.disk_usage(root).free>=p['budgets']['disk_reserve_bytes'],'Disk reserve stop'
        assert sum(v.stat().st_size for v in out.rglob('*') if v.is_file())<=p['budgets']['output_byte_cap'],'Output cap stop'
        for k,v in counts.items():
            assert v<=p['budgets'].get('maximum_'+k,p['budgets'].get(k)),k+' count stop'
        if torch is not None:
            import resource
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_allocated()<=p['budgets']['peak_VRAM_bytes'],'VRAM stop'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=p['budgets']['peak_RSS_bytes'],'RSS stop'
    def interrupted(_signal,_frame):raise RuntimeError('Finite supervisor stop; retain partial evidence')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        free=shutil.disk_usage(root).free
        write(out/'storage_start.json',dict(free_bytes=free,MemAvailable_bytes=mem['MemAvailable'],neural_calls=0,optimizer_updates=0))
        assert free>=p['budgets']['minimum_free_bytes'] and mem['MemAvailable']>=p['budgets']['available_RAM_before_import_bytes'],'Need8GiB disk and available RAM'
        assert source==(root.parent/SOURCE_NAME).resolve() and parent==(root.parent/PARENT_NAME).resolve() and replay==(root.parent/REPLAY_NAME).resolve()
        with source_cwd(source):
            spec=importlib.util.spec_from_file_location('retained_decoder_initial_loader',source/'scripts/cctv_dgp_spatial_readout_v1_vm.py')
            loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
            np,torch,original,identity,provider,provenance,loading=loader.load_dependencies(source,parent,old)
            torch.set_grad_enabled(False);torch.cuda.reset_peak_memory_stats()
            from PIL import Image
            from cctv_dgp_pilot import grid112,state_hash,buffer_hash
            from cctv_dgp_head4_lossless_v1 import unpack
            from cctv_dgp_spatial_readout_v1_model import build_model
            from cctv_dgp_spatial_readout_v1_contract import stage_gate,project_original_ball
            from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
            from frozen_capacity_contract import feature_support,groups,exported_pixel_metrics,detail_metric
            from frozen_raw_metrics import deliver,mean_only,pixel_metrics,detail_float
            legacy=file_module(parent/'cctv_dgp_bank_photometric_guard_v1_loss.py','declared_legacy_decoder_hinge',LEGACY_HELPER_SHA)
            corrected=file_module(root/'cctv_dgp_bank_hinge_boundary_v1_loss.py','declared_corrected_decoder_hinge',CORRECTED_HELPER_SHA)
            assert all(not v.requires_grad for net in [original.net,identity,provider] for v in net.parameters())
            write(out/'environment.json',dict(host=os.uname().nodename,GPU=torch.cuda.get_device_name(0),torch=torch.__version__,CUDA=torch.version.cuda,python=sys.version,
                packages={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','Pillow','scipy','scikit-image','onnx','onnx2torch']},
                original_state=state_hash(original.net),recognizer_state=state_hash(identity),provider_state=state_hash(provider),DGP_provenance=provenance,
                historical_training_called=False,legacy_helper_path=str(Path(legacy.__file__).resolve()),legacy_helper_sha256=sha(legacy.__file__),
                corrected_helper_path=str(Path(corrected.__file__).resolve()),corrected_helper_sha256=sha(corrected.__file__),learning_rate=.0001))
            refs,cases={r['id']:r for r in data['references']},{c['id']:c for c in data['cases']}
            def pixels(path,mode='RGB'):
                with Image.open(path) as image:
                    assert image.size==(256,256);return np.asarray(image.convert(mode)).copy()
            def tensor(values):return torch.from_numpy(np.stack(values).astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous().cuda()
            def gpu(b):return {k:v.cuda() for k,v in b['CPU'].items()}|dict(maps=[v.cuda() for v in b['maps']],features={k:v.cuda() for k,v in b['features'].items()})
            caches=[];cache_rows=[];began=time.monotonic()
            source_cache=read(source/'outputs/spatial_readout_v1/cache_receipt.json')['rows']
            for index,block in enumerate(p['reference_blocks']):
                progress.update(stage='fresh exact feature cache',reference=block['reference'])
                cs=[cases[c] for c in block['case_ids']];ref=refs[block['reference']]
                camera=[pixels(parent/c['input']) for c in cs];targets=[pixels(parent/ref['target']) for _ in cs]
                masks=[pixels(parent/ref['observed'],'L')>0 for _ in cs];support=[feature_support(m,c['landmarks5_canvas_xy']) for m,c in zip(masks,cs)]
                values=dict(x=tensor(camera),target=tensor(targets),mask=torch.from_numpy(np.stack(masks).astype(np.float32))[:,None].cuda(),
                    feature=torch.from_numpy(np.stack(support).astype(np.float32))[:,None].cuda(),grid=torch.from_numpy(np.stack([grid112(ref['matrix112'])]*5)).cuda())
                counts['original_batch_calls']+=1
                baseline=torch.where(values['mask'].bool(),original(values['x']),values['x'])
                maps=provider.current.observe(values['x']);counts['prior_feature_batch_calls']+=1;features=provider.conditioned_features(maps)
                retained,embedding=unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz",block['case_ids'])
                raw=baseline.cpu().permute(0,2,3,1).numpy().copy();assert np.array_equal(raw,retained)
                values.update(baseline=baseline,target_vector=torch.from_numpy(embedding[:,2].copy()).cuda())
                values['old_arc']=1-(torch.from_numpy(embedding[:,0].copy()).cuda().double()*values['target_vector'].double()).sum(1)
                b=dict(cases=cs,camera=camera,target8=targets,mask8=masks,support8=support,baseline=raw,retained_embeddings=embedding,
                    target_vectors=embedding[:,2].copy(),CPU={k:v.detach().cpu().contiguous() for k,v in values.items()},
                    old_terms={k:v.detach().cpu() for k,v in pixel_and_structure_losses(baseline,values['target'],values['mask'],values['feature']).items()},
                    maps=[v.detach().cpu().contiguous() for v in maps],features={k:v.detach().cpu().contiguous() for k,v in features.items()})
                proof=dict(reference=block['reference'],source=block['source'],maps=[array_sha(v.numpy()) for v in b['maps']],features={k:array_sha(v.numpy()) for k,v in b['features'].items()})
                assert all(proof[k]==source_cache[index][k] for k in proof),'Fresh source feature parity differs'
                caches.append(b);cache_rows.append(proof);del values,baseline,maps,features,retained,embedding,raw
                clock();elapsed=time.monotonic()-began
                if index==1:
                    projected=elapsed*10*1.5;write(out/'cache_projection.json',dict(references=2,seconds=elapsed,projected_seconds=projected,cap_seconds=240));assert projected<=240
                assert elapsed<=p['budgets']['cache_seconds'];print(json.dumps(dict(cache=index+1,of=20)),flush=True)
            write(out/'cache_receipt.json',dict(complete=True,rows=cache_rows,all_source_feature_hashes_exact=True,seconds=time.monotonic()-began))
            initial=np.load(source/'outputs/spatial_readout_v1/A_initial_vector.npy',allow_pickle=False)
            with np.load(source/'outputs/spatial_readout_v1/trajectory_A/update_001.npz',allow_pickle=False) as saved:legacy_first=saved['actual'].copy()
            initial_state=torch.load(source/'outputs/spatial_readout_v1/state_A0/full_state.pth',map_location='cpu',weights_only=True)
            names=old['trainable_parameter_names']['A'];layout=read(source/'outputs/spatial_readout_v1/A_parameter_layout.json')
            model=build_model(source,parent,provider,source/'readouts.npz','A');model.load_state_dict(initial_state['model'],strict=True);model.eval().requires_grad_(False)
            assert model.learning_names()==names and len(names)==15
            fixed=state_hash({k:v for k,v in model.state_dict().items() if k not in names});buffers=buffer_hash(model)
            original_receipt=read(source/'outputs/spatial_readout_v1/state_A0/receipt.json')
            assert state_hash(model)==original_receipt['model_state'] and buffers==original_receipt['buffer_state']
            def flat():return np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in model.learning_parameters()]).astype(np.float32)
            def assign(vector):
                assert vector.dtype==np.float32 and vector.shape==(609219,) and np.isfinite(vector).all()
                assert float(np.linalg.norm(vector.astype(np.float64)-initial.astype(np.float64)))<=p['common_radius_L2']
                with torch.no_grad():
                    for row,value in zip(layout,model.learning_parameters()):value.copy_(torch.from_numpy(vector[row['begin']:row['end']].reshape(row['shape'])).cuda())
                assert np.array_equal(flat(),vector) and buffer_hash(model)==buffers
                assert state_hash({k:v for k,v in model.state_dict().items() if k not in names})==fixed
            def save_state(key):
                folder=out/('state_'+key);folder.mkdir()
                torch.save(dict(model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
                    shadow=[] if shadow is None else [v.detach().cpu().clone() for v in shadow],
                    CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),updates=progress['updates'],protocol_sha256=pin,counts=dict(counts),
                    failed_recipe_resume_allowed=False,exact_future_resume_guaranteed=False),folder/'full_state.pth')
                write(folder/'receipt.json',dict(model_state=state_hash(model),buffer_state=buffer_hash(model),full_state_sha256=sha(folder/'full_state.pth'),
                    updates=progress['updates'],optimizer_present=optimizer is not None,model_qualified=False))
            assert np.array_equal(flat(),initial)
            def snapshot(key,baseline_only=False,initial_metric=False,control=None):
                stamp=time.monotonic();folder=out/key;folder.mkdir();(folder/'packs').mkdir();(folder/'png').mkdir();rows=[];component_rows=[]
                if not baseline_only:
                    vector=flat();np.save(out/(key+'_vector.npy'),vector,allow_pickle=False)
                    write(out/(key+'_state.json'),dict(model_state=state_hash(model),fixed_state=fixed,buffer_state=buffers,
                        vector_sha256=array_sha(vector),installed_vector_exact=True,optimizer_updates=counts['optimizer_updates'],gradients_enabled=False))
                with torch.no_grad():
                    for index,b in enumerate(caches):
                        g=gpu(b)
                        if baseline_only:pred=g['baseline']
                        else:
                            counts['decoder_batch_calls']+=1;pred=torch.where(g['mask'].bool(),model.forward_cached(g['x'],g['maps'],g['features']),g['x'])
                        raw=pred.cpu().permute(0,2,3,1).numpy().copy();png=np.stack([deliver(a,c,m) for a,c,m in zip(raw,b['camera'],b['mask8'])])
                        if control is not None:
                            with np.load(control/'packs'/f'r{index:02d}.npz',allow_pickle=False) as saved:assert np.array_equal(raw,saved['raw']),'Fresh RAW control differs'
                            assert all(np.array_equal(v,pixels(control/'png'/(c['id']+'.png'))) for v,c in zip(png,b['cases'])),'Fresh PNG control differs'
                        zero_png=[deliver(a,c,m) for a,c,m in zip(b['baseline'],b['camera'],b['mask8'])]
                        counts['recognizer_batch_calls']+=2
                        er=identity.embedding(torch.cat([pred,g['baseline'],g['target']]),torch.cat([g['mask']]*3),torch.cat([g['grid']]*3)).cpu().numpy()
                        ep=identity.embedding(torch.cat([tensor(list(png)),tensor(zero_png),g['target']]),torch.cat([g['mask']]*3),torch.cat([g['grid']]*3)).cpu().numpy()
                        for values,slot in [(er,0),(ep,1)]:
                            np.testing.assert_allclose(values[5:10],b['retained_embeddings'][:,slot],rtol=1e-6,atol=1e-6)
                            np.testing.assert_allclose(values[10:15],b['target_vectors'],rtol=1e-6,atol=1e-6)
                        terms=pixel_and_structure_losses(pred,g['target'],g['mask'],g['feature'])
                        av=g['old_arc'] if initial_metric else 1-(torch.from_numpy(er[:5]).cuda().double()*g['target_vector'].double()).sum(1)
                        ns=old['normalizers'];old_terms={k:v.cuda() for k,v in b['old_terms'].items()}
                        averages=[terms['MSE'].mean(),terms['SSIM_loss'].mean(),av.mean(),terms['landmark_structure'][1:].mean()]
                        components=[float(w*v/n) for w,v,n in zip(old['reconstruction_weights'],averages,ns)]
                        barriers=[float(5*torch.relu(terms['MSE']-old_terms['MSE']-1e-12).mean()/ns[0]),
                            float(5*torch.relu(terms['SSIM_loss']-old_terms['SSIM_loss']-1e-6).mean()/ns[1]),
                            float(5*torch.relu(av-g['old_arc']-1e-6).mean()/ns[2])]
                        anchors=[float(terms['MSE'][0]/old['active_anchor_normalizers'][0]),float(terms['MSE'][1]/old['active_anchor_normalizers'][1])]
                        photo,details=corrected.brightness_guard(pred,g['baseline'],g['target'],g['mask'],ns[0],10.,.2)
                        old_photo,old_details=legacy.brightness_guard(pred,g['baseline'],g['target'],g['mask'],ns[0],10.,.2)
                        assert torch.equal(photo,old_photo) and details==old_details
                        component_rows.append(dict(reference=p['reference_blocks'][index]['reference'],source=p['reference_blocks'][index]['source'],
                            objective=sum(components)+sum(barriers)+sum(anchors)+float(photo),
                            normalized_weighted_RGB_SSIM_recognizer_structure=components,regression_barriers_RGB_SSIM_recognizer=barriers,clear_blur_RGB_anchors=anchors,
                            photometric_guard=float(photo),per_image_photometric=details,raw_RGB_MSE=terms['MSE'].tolist(),raw_SSIM_loss=terms['SSIM_loss'].tolist(),
                            raw_recognizer_loss=av.tolist(),raw_structure_loss=terms['landmark_structure'].tolist()))
                        np.savez_compressed(folder/'packs'/f'r{index:02d}.npz',ids=np.asarray([c['id'] for c in b['cases']]),raw=raw,embeddings_raw=er,embeddings_png=ep,target_embeddings=b['target_vectors'])
                        for i,c in enumerate(b['cases']):
                            mask,target=b['mask8'][i],b['target8'][i]
                            assert np.array_equal(raw[i][~mask],b['camera'][i][~mask].astype(np.float32)/np.float32(255))
                            assert np.array_equal(png[i][~mask],b['camera'][i][~mask])
                            mean,mean_png,_=mean_only(raw[i],b['baseline'][i],b['camera'][i],mask)
                            rm,pm=pixel_metrics(raw[i],target,mask),exported_pixel_metrics(png[i],target,mask)
                            for metric,vectors,slot in [(rm,er,0),(pm,ep,1)]:
                                v=b['retained_embeddings'][i,slot] if initial_metric else vectors[i]
                                t=b['target_vectors'][i] if initial_metric else vectors[10+i]
                                metric['ArcFace_observed_fixed']=float(v@t)
                            rm.update(landmark_high_frequency_MSE=detail_float(raw[i],target,b['support8'][i]),constant_mean_shift_only_MSE=pixel_metrics(mean,target,mask)['MSE'])
                            pm.update(landmark_high_frequency_MSE=detail_metric(png[i],target,b['support8'][i]),constant_mean_shift_only_MSE=exported_pixel_metrics(mean_png,target,mask)['MSE'])
                            Image.fromarray(png[i]).save(folder/'png'/(c['id']+'.png'));rows.append(dict(case=c['id'],source=c['source'],profile=c['profile'],raw=rm,png=pm))
                        del g,pred,raw,png,er,ep,zero_png,terms,av,old_terms,photo,old_photo;clock();assert time.monotonic()-stamp<=p['budgets']['snapshot_seconds']
                record=dict(rows=rows,groups={mode:groups([dict(source=r['source'],profile=r['profile'],metrics=r[mode]) for r in rows]) for mode in ['raw','png']},seconds=time.monotonic()-stamp)
                write(folder/'metrics.json',record)
                write(folder/'objective_components.json',dict(rows=component_rows,diagnostic_embedding_batch15_not_fresh_training_batch5=True,
                    initial_cached_embedding_controls=initial_metric,gradient_or_model_quality_claim=False))
                snapshot_labels.append(key);return record
            baseline=snapshot('baseline',True,True,source/'outputs/spatial_readout_v1/baseline')
            snapshot('initial',initial_metric=True,control=source/'outputs/spatial_readout_v1/baseline')
            assign(legacy_first);snapshot('legacy_A1',control=replay/'outputs/spatial_path_replay_v1/A_u001');assign(initial)
            # Pixel-leaf derivatives isolate the helper convention without any
            # candidate parameter change or extra neural forward.
            boundary=out/'guard_boundary';boundary.mkdir();boundary_rows=[];preflight_start=time.monotonic()
            for index,b in enumerate(caches):
                g=gpu(b)
                with torch.enable_grad():
                    leaf=g['baseline'].detach().clone().requires_grad_(True)
                    old_pen,old_meta=legacy.brightness_guard(leaf,g['baseline'],g['target'],g['mask'],old['normalizers'][0],10.,.2)
                    new_pen,new_meta=corrected.brightness_guard(leaf,g['baseline'],g['target'],g['mask'],old['normalizers'][0],10.,.2)
                    assert torch.equal(old_pen.detach(),new_pen.detach()) and old_meta==new_meta and float(new_pen.detach())==0.
                    old_grad=torch.autograd.grad(old_pen,leaf)[0];counts['gradient_queries']+=1
                    new_grad=torch.autograd.grad(new_pen,leaf)[0];counts['gradient_queries']+=1
                    assert torch.isfinite(old_grad).all() and torch.isfinite(new_grad).all() and torch.count_nonzero(new_grad).item()==0
                    ga,gb=old_grad.cpu().numpy(),new_grad.cpu().numpy()
                ids=[c['id'] for c in b['cases']]
                np.savez_compressed(boundary/f'r{index:02d}.npz',ids=np.asarray(ids),legacy_gradient=ga,corrected_gradient=gb)
                boundary_rows.append(dict(reference=p['reference_blocks'][index]['reference'],case_ids=ids,legacy_scalar=float(old_pen.detach()),corrected_scalar=float(new_pen.detach()),
                    legacy_metadata=old_meta,corrected_metadata=new_meta,legacy_gradient_L2=float(np.linalg.norm(ga.astype(np.float64))),corrected_gradient_L2=0.,
                    legacy_gradient_sha256=array_sha(ga),corrected_gradient_sha256=array_sha(gb)))
                del g,leaf,old_pen,new_pen,old_grad,new_grad,ga,gb;clock();assert time.monotonic()-preflight_start<=p['budgets']['preflight_seconds']
            write(boundary/'receipt.json',dict(complete=True,rows=boundary_rows,all20_scalar_metadata_exact=True,all20_corrected_zero_gradient=True,gradient_queries=40,optimizer_updates=0))
            def objective(pred,g,b):
                terms=pixel_and_structure_losses(pred,g['target'],g['mask'],g['feature'])
                counts['recognizer_batch_calls']+=1
                arc=1-(identity.embedding(pred,g['mask'],g['grid']).double()*g['target_vector'].double()).sum(1)
                old_terms={k:v.cuda() for k,v in b['old_terms'].items()}
                raw=[terms['MSE'].mean(),terms['SSIM_loss'].mean(),arc.mean(),terms['landmark_structure'][1:].mean()]
                normalizers=old['normalizers']
                barrier=5*(torch.relu(terms['MSE']-old_terms['MSE']-1e-12).mean()/normalizers[0]
                    +torch.relu(terms['SSIM_loss']-old_terms['SSIM_loss']-1e-6).mean()/normalizers[1]
                    +torch.relu(arc-g['old_arc']-1e-6).mean()/normalizers[2])
                anchors=terms['MSE'][0]/old['active_anchor_normalizers'][0]+terms['MSE'][1]/old['active_anchor_normalizers'][1]
                photo,detail=corrected.brightness_guard(pred,g['baseline'],g['target'],g['mask'],normalizers[0],10.,.2)
                value=sum(w*v/n for w,v,n in zip(old['reconstruction_weights'],raw,normalizers))+barrier+anchors+photo
                assert torch.isfinite(value)
                return value,dict(objective=float(value.detach()),raw_terms=[float(v.detach()) for v in raw],barrier=float(barrier.detach()),anchors=float(anchors.detach()),**detail)
            assign(initial);torch.manual_seed(20261014);torch.cuda.manual_seed_all(20261014)
            model.enable_learning();selected=model.learning_parameters();preflight=[];sums=np.zeros(15,np.float64)
            with torch.enable_grad():
                for index in p['schedule'][:2]:
                    t=time.monotonic();b=caches[index];g=gpu(b);counts['decoder_batch_calls']+=1
                    pred=torch.where(g['mask'].bool(),model.forward_cached(g['x'],g['maps'],g['features']),g['x']);loss,terms=objective(pred,g,b)
                    derivatives=torch.autograd.grad(loss,selected,allow_unused=False);counts['gradient_queries']+=1
                    assert all(torch.isfinite(v).all() for v in derivatives)
                    norms=[float(v.detach().double().norm()) for v in derivatives];sums+=norms
                    torch.cuda.synchronize();preflight.append(dict(reference=p['reference_blocks'][index]['reference'],names=names,norms=norms,terms=terms,seconds=time.monotonic()-t))
                    del g,pred,loss,derivatives;clock();assert time.monotonic()-preflight_start<=p['budgets']['preflight_seconds']
            assert (sums>0).all();projection=max(r['seconds'] for r in preflight)*50*1.75+50*.25
            write(out/'preflight.json',dict(rows=preflight,all15_finite_nonzero=True,summed_norms=sums.tolist(),fit_projected_seconds=projection,cap_seconds=600,optimizer_updates=0,
                combined_boundary_and_parameter_preflight_seconds=time.monotonic()-preflight_start,combined_preflight_cap_seconds=p['budgets']['preflight_seconds']))
            assert projection<=600
            projected=time.monotonic()-start+projection+8*baseline['seconds']*1.75+90
            write(out/'worker_projection.json',dict(projected_seconds=projected,cap_seconds=2400,before_optimizer=True));assert projected<=2400
            # Uncompressed arrays, 50 full journals, seven complete states,
            # 40 pixel-gradient arrays, selected vectors and metadata allowance.
            estimated=(1100*(256*256*3*4+256*256*3)+3*50*609219*4
                +7*((source/'outputs/spatial_readout_v1/state_A0/full_state.pth').stat().st_size+3*609219*4+32*1024**2)
                +20*2*5*3*256*256*4+10*609219*4+128*1024**2)
            write(out/'output_projection.json',dict(projected_bytes=estimated,cap_bytes=p['budgets']['output_byte_cap']))
            assert estimated<=p['budgets']['output_byte_cap'] and shutil.disk_usage(root).free>=2*estimated+p['budgets']['disk_reserve_bytes']
            shadow=[torch.nn.Parameter(v.detach().clone(),requires_grad=False) for v in selected]
            optimizer=torch.optim.Adam(shadow,lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5,foreach=False,fused=False)
            save_state('initial');journal=out/'trajectory';journal.mkdir();training_seconds=0.;trace=[];stop_reason='maximum_updates';training_stopped=False
            def outcome(key,metrics):
                gates={mode:stage_gate(baseline['groups'][mode],metrics['groups'][mode]) for mode in ['raw','png']}
                row=dict(snapshot=key,updates=progress['updates'],gates=gates,continuation_eligible=continuation_eligible(gates),model_qualified=False,new_incumbent_selection=False)
                write(out/(key+'_outcome.json'),row);return row
            for update,index in enumerate(p['schedule'],1):
                progress.update(stage='corrected derivative finite fit',updates=update-1);clock();t=time.monotonic();b=caches[index];g=gpu(b)
                with torch.enable_grad():
                    optimizer.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);counts['decoder_batch_calls']+=1
                    pred=torch.where(g['mask'].bool(),model.forward_cached(g['x'],g['maps'],g['features']),g['x']);loss,terms=objective(pred,g,b)
                    loss.backward();counts['backwards']+=1
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in selected)
                    norm=float(torch.nn.utils.clip_grad_norm_(selected,1.).detach())
                    clipped=np.concatenate([v.grad.detach().cpu().numpy().reshape(-1) for v in selected]).astype(np.float32)
                    for q,v in zip(shadow,selected):q.grad=v.grad.detach().clone()
                    optimizer.step();counts['optimizer_updates']+=1
                proposal=np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in shadow]).astype(np.float32)
                actual,bound=project_original_ball(initial,proposal,p['common_radius_L2']);assign(actual)
                with torch.no_grad():
                    for q,v in zip(shadow,selected):q.copy_(v)
                progress['updates']=update
                np.savez_compressed(journal/f'update_{update:03d}.npz',clipped_gradient=clipped,unconstrained=proposal,actual=actual)
                row=dict(update=update,cache_index=index,reference=p['reference_blocks'][index]['reference'],gradient_norm_before_clip=norm,
                    clip_factor=min(1.,1./(norm+1e-6)),projection=bound,**terms);trace.append(row)
                with (journal/'trace.jsonl').open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                torch.cuda.synchronize();training_seconds+=time.monotonic()-t
                assert training_seconds<=p['budgets']['training_seconds'];del g,pred,loss,proposal,clipped
                if update in UPDATES:
                    key=f'fit_u{update:03d}';metrics=snapshot(key);checked=outcome(key,metrics);fit_outcomes.append(checked);save_state(key)
                    if update==1:
                        first=actual.copy();first_state=state_hash(model)
                        probe_cpu=torch.get_rng_state().clone();probe_cuda=[v.clone() for v in torch.cuda.get_rng_state_all()];probes_complete=False
                        try:
                            for probe,amplitude in PROBES:
                                value=(initial.astype(np.float64)+amplitude*(first.astype(np.float64)-initial.astype(np.float64))).astype(np.float32)
                                assign(value);probe_metrics=snapshot(probe);outcome(probe,probe_metrics)
                            probes_complete=True
                        finally:
                            cuda_after=torch.cuda.get_rng_state_all()
                            unchanged=torch.equal(torch.get_rng_state(),probe_cpu) and len(cuda_after)==len(probe_cuda) and all(torch.equal(a,b) for a,b in zip(cuda_after,probe_cuda))
                            assign(first);torch.set_rng_state(probe_cpu);torch.cuda.set_rng_state_all(probe_cuda)
                            assert state_hash(model)==first_state and torch.equal(torch.get_rng_state(),probe_cpu)
                            restored_cuda=torch.cuda.get_rng_state_all()
                            assert len(restored_cuda)==len(probe_cuda) and all(torch.equal(a,b) for a,b in zip(restored_cuda,probe_cuda))
                            assert all(np.array_equal(q.detach().cpu().numpy(),v.detach().cpu().numpy()) for q,v in zip(shadow,selected))
                            write(out/'probe_restoration.json',dict(complete=probes_complete,CPU_rng_sha256=array_sha(probe_cpu.numpy()),
                                CUDA_rng_sha256=[array_sha(v.numpy()) for v in probe_cuda],RNG_unchanged_before_restore=unchanged,
                                CPU_CUDA_rng_restored_exact=True,corrected_first_model_restored_exact=True,optimizer_shadow_unchanged=True,model_state=first_state))
                        assert unchanged,'Diagnostic probes unexpectedly consumed RNG; retain stop'
                    print(json.dumps(dict(snapshot=key,RAW_gain_percent=100*checked['gates']['raw']['global_structure_gain'],PNG_gain_percent=100*checked['gates']['png']['global_structure_gain'],
                        RAW_failures=len(checked['gates']['raw']['preservation_failures']),PNG_failures=len(checked['gates']['png']['preservation_failures']),continuation_eligible=checked['continuation_eligible'])),flush=True)
                    if not checked['continuation_eligible']:
                        stop_reason='quality_stop';training_stopped=True;write(out/'quality_stop.json',dict(snapshot=key,outcome=checked,failed_recipe_resume_allowed=False,automatic_extra_updates=0));break
                del actual;clock()
            write(out/'fit_trace.json',dict(rows=trace,updates=progress['updates'],training_seconds_excluding_snapshots=training_seconds,cap_seconds=600))
            assert state_hash(original.net)==old['original_state'] and state_hash(identity)==old['recognizer_state'] and state_hash(provider)==old['feature_provider_initial_state']
            assert state_hash({k:v for k,v in model.state_dict().items() if k not in names})==fixed and buffer_hash(model)==buffers
            endpoint=fit_outcomes[-1];numerical_endpoint=progress['updates']==50 and all(v['pass'] for v in endpoint['gates'].values())
            write(out/'run_results.json',dict(complete=True,counts=counts,snapshot_labels=snapshot_labels,training_updates=progress['updates'],training_stopped=training_stopped,
                stop_reason=stop_reason,fit_outcomes=fit_outcomes,numerical_endpoint_pass=numerical_endpoint,probe_admission_used=False,
                original_provider_recognizer_unchanged=True,derivative_boundary_is_only_intended_objective_change=True,learning_rate=.0001,learning_rate_changed=False,
                automatic_extra_epochs=False,noautomaticresume=True,failed_recipe_resume_allowed=False,full_corpus_epochs=0,model_qualified=False,new_incumbent_selection=False,seconds=time.monotonic()-start))
        require_manual(root)
        return 0 if numerical_endpoint else 2
    except BaseException as exc:
        # Model/optimizer/RNG state is retained if initialization reached them.
        save_error=None
        if model is not None and 'save_state' in locals():
            try:save_state(f"technical_stop_u{progress['updates']:03d}")
            except BaseException as stopped:save_error=repr(stopped)
        write(out/'failure.json',dict(complete=False,error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),progress=progress,counts=counts,
            snapshot_labels=snapshot_labels,failure_state_save_error=save_error,model_qualified=False,failed_recipe_resume_allowed=False,seconds=time.monotonic()-start))
        raise


def export(root,p,pin):
    require_manual(root);start=time.monotonic();manifest=root/'result_manifest.json'
    assert not manifest.exists(),'Retain prior export'
    archive=Path.home()/(STEM+'-results.tar.gz');sidecar=Path(str(archive)+'.sha256');receipt=Path.home()/(STEM+'-export.json')
    partial=Path(str(archive)+'.partial')
    assert all(not v.exists() for v in [archive,partial,sidecar,receipt])
    files=[root/name for name in p['assets_sha256']]+[root/'protocol.json']
    files += sorted(v for v in (root/STAGE).rglob('*') if v.is_file())
    files += [root/n for n in ['run.log','run_exit.json','saved_audit.log','saved_audit_exit.json'] if (root/n).is_file()]
    assert len(files)==len(set(files)) and all(not v.is_symlink() and v.resolve().is_relative_to(root) for v in files)
    total=sum(v.stat().st_size for v in files)
    assert total<=p['budgets']['output_byte_cap']+16*1024**2 and shutil.disk_usage(root).free>=total+p['budgets']['disk_reserve_bytes']
    result_path=root/STAGE/'run_results.json';failure_path=root/STAGE/'failure.json'
    result=read(result_path) if result_path.is_file() else read(failure_path)
    optimizer_updates=result['counts']['optimizer_updates']
    bindings={}
    for path in files:
        bindings[path.relative_to(root).as_posix()]=dict(bytes=path.stat().st_size,sha256=sha(path))
        assert time.monotonic()-start<=p['budgets']['export_seconds']
    write(manifest,dict(protocol_sha256=pin,files=bindings,complete=True,training_success_not_implied=True,
        model_quality_not_implied=True,model_qualified=False,optimizer_updates=optimizer_updates))
    with tarfile.open(partial,'x:gz',compresslevel=1) as stream:
        for path in [manifest]+files:
            assert time.monotonic()-start<=p['budgets']['export_seconds'] and shutil.disk_usage(root).free>=p['budgets']['disk_reserve_bytes']
            stream.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
    digest=sha(partial)
    assert time.monotonic()-start<=p['budgets']['export_seconds'] and shutil.disk_usage(root).free>=p['budgets']['disk_reserve_bytes']
    assert not archive.exists();partial.rename(archive)
    with sidecar.open('x',encoding='utf-8',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    write(receipt,dict(complete=True,protocol_sha256=pin,archive_sha256=digest,bytes=archive.stat().st_size,seconds=time.monotonic()-start,
        training_success_not_implied=True,model_quality_not_implied=True,optimizer_updates=optimizer_updates,model_qualified=False,
        run_results_present=result_path.is_file(),failure_present=failure_path.is_file()))
    print(json.dumps(read(receipt)),flush=True)


def child(root,source,parent,replay,pin,flag,seconds,log_name):
    import selectors
    args=[sys.executable,'-B','-u',str(root/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py'),'--root',str(root),'--source-root',str(source),
        '--parent-root',str(parent),'--replay-root',str(replay),'--protocol-sha',pin,flag]
    start=time.monotonic();timeout=False;killed=False
    with (root/log_name).open('xb') as log,selectors.DefaultSelector() as selector:
        process=subprocess.Popen(args,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
        print(json.dumps(dict(flag=flag,child_pid=process.pid,log=log_name,external_cap_seconds=seconds)),flush=True)
        selector.register(process.stdout,selectors.EVENT_READ)
        while selector.get_map() or process.poll() is None:
            elapsed=time.monotonic()-start
            if elapsed>seconds and not timeout and process.poll() is None:
                timeout=True
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if elapsed>seconds+30 and process.poll() is None and not killed:
                killed=True
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            for key,_ in selector.select(.2):
                value=os.read(key.fileobj.fileno(),65536)
                if value:log.write(value);log.flush();sys.stdout.buffer.write(value);sys.stdout.buffer.flush()
                else:selector.unregister(key.fileobj)
        code=process.wait()
    return dict(flag=flag,exit_code=code,timeout=timeout,kill_sent=killed,seconds=time.monotonic()-start,child_pid=process.pid,maximum_seconds=seconds,kill_grace_seconds=30)


def supervise(root,source,parent,replay,p,old,data,pin):
    machine_guard(root,idle=True)
    reserved=['outputs','result_manifest.json','run.log','run_exit.json','saved_audit.log','saved_audit_exit.json','export.log','supervisor_receipt.json']
    reserved += [str(Path.home()/(STEM+'-results.tar.gz'+suffix)) for suffix in ['', '.partial','.sha256']]+[str(Path.home()/(STEM+'-export.json'))]
    assert all(not (root/name).exists() for name in reserved),'Used run/export; preserve existing files'
    worked=child(root,source,parent,replay,pin,'--run',p['budgets']['external_seconds'],'run.log');write(root/'run_exit.json',worked)
    checked=None
    if (root/STAGE/'run_results.json').is_file() or (root/STAGE/'failure.json').is_file():
        checked=child(root,source,parent,replay,pin,'--audit-saved',p['budgets']['saved_audit_external_seconds'],'saved_audit.log');write(root/'saved_audit_exit.json',checked)
    exported=child(root,source,parent,replay,pin,'--export',p['budgets']['export_external_seconds'],'export.log')
    write(root/'supervisor_receipt.json',dict(complete=True,run=worked,saved_audit=checked,export=exported,model_qualified=False,automatic_follow_on=False))
    return 0 if worked['exit_code']==0 and checked and checked['exit_code']==0 and exported['exit_code']==0 else 2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    for name in ['source-root','parent-root','replay-root']:parser.add_argument('--'+name,type=Path)
    parser.add_argument('--protocol-sha',required=True)
    flags=parser.add_mutually_exclusive_group(required=True)
    for name in ['verify-transfer','run','audit-saved','export','supervise']:flags.add_argument('--'+name,action='store_true')
    a=parser.parse_args();root=a.root.resolve()
    source=a.source_root.resolve() if a.source_root else root.parent/SOURCE_NAME
    parent=a.parent_root.resolve() if a.parent_root else root.parent/PARENT_NAME
    replay=a.replay_root.resolve() if a.replay_root else root.parent/REPLAY_NAME
    if not a.verify_transfer:require_manual(root)
    p,old,data=verify_transfer(root,source,parent,a.protocol_sha,replay=replay)
    if a.verify_transfer:
        print(json.dumps(dict(complete=True,packet_assets=len(p['assets_sha256']),source_assets=len(p['source_required_sha256']),parent_assets=len(p['parent_required_sha256']),
            replay_assets=len(p['replay_required_sha256']),optimizer_updates=0,gradient_queries=0,neural_calls=0)))
    elif a.run:raise SystemExit(run(root,source,parent,replay,p,old,data,a.protocol_sha))
    elif a.export:export(root,p,a.protocol_sha)
    elif a.audit_saved:
        from audit_cctv_dgp_decoder_hinge_repair_v1_return import audit_saved
        audit_saved(root,source,parent,replay,p,old,data,root/STAGE/'independent_saved_audit_vm.json')
    else:raise SystemExit(supervise(root,source,parent,replay,p,old,data,a.protocol_sha))


if __name__=='__main__':main()

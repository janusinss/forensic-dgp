"""Manual L4 diagnostic of a frozen recognizer's RGB-input response; no optimizer."""
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

NAME='cctv_dgp_appearance_input_response_v1_vm'
STEM='cctv-dgp-appearance-input-response-v1'
STAGE='outputs/appearance_input_response_v1'
RESPONSE_NAME='cctv_dgp_conditioning_response_v1_vm'
BANK_NAME='cctv_dgp_bank_photometric_guard_v1_vm'
RESPONSE_PIN='e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44'
BANK_PIN='f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
RECOGNIZER_STATE='9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root,response,bank,pin):
    assert sha(root/'protocol.json')==pin
    p=read(root/'protocol.json');source=read(response/'protocol.json');old=read(bank/'protocol.json')
    assert p['format']=='frozen-appearance-input-response-v1' and p['name']==NAME
    assert sha(response/'protocol.json')==RESPONSE_PIN==p['response_protocol_sha256']
    assert sha(bank/'protocol.json')==BANK_PIN==p['bank_protocol_sha256']
    assert p['case_ids']==source['case_ids'] and p['reference_blocks']==source['reference_blocks']
    assert len(p['case_ids'])==100 and len(set(p['case_ids']))==100 and len(p['reference_blocks'])==20
    assert p['scientific_thresholds']==source['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert p['recognizer_state']==RECOGNIZER_STATE and p['expected_torch']=='2.9.1+cu129'
    assert p['gradient_repeats']==2 and p['maximum_gradient_queries']==200 and p['recognizer_forward_cap']==40
    assert p['restorer_forward_cap']==p['optimizer_updates']==p['neural_parameter_displacements']==p['accepted_updates']==p['full_epochs']==0
    assert p['worker_seconds']==450 and p['measurement_seconds']==420 and p['round_seconds']==180 and p['export_seconds']==180
    assert p['minimum_free_bytes']==3*1024**3 and p['disk_reserve_bytes']==1024**3 and p['output_byte_cap']==512*1024**2
    assert p['peak_VRAM_bytes']==12*1024**3 and p['peak_RSS_bytes']==8*1024**3
    assert p['native_pixels_used']==p['final_identity_pixels']==0 and not p['app_promotion'] and not p['automatic_follow_on']
    assert p['signed_labels']==[r['label'] for r in source['step_plan']]
    for folder,bindings in [(root,p['assets_sha256']),(response,p['response_required_sha256']),(bank,p['bank_required_sha256'])]:
        for name,h in bindings.items():
            path=folder/name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder)
            assert sha(path)==h,name
    sys.path.insert(0,str(root))
    from input_scopes import scopes
    blocks,_=scopes(old['cases'],p['case_ids']);assert blocks==p['reference_blocks']
    assert all(c['role']=='train' for b in blocks for c in old['cases'][b['begin']:b['begin']+5])
    run=read(response/'outputs/conditioning_response_v1/run_results.json')
    assert run['complete_diagnostic'] and run['counts']['accepted_updates']==run['counts']['optimizer_updates']==0
    return p,source,old


def require_vm(root,response,bank,idle=True):
    assert platform.system()=='Linux','Manual Linux L4 input diagnostic only'
    assert platform.node().split('.')[0]=='forensic-dgp-thesis'
    for path,name in [(root,NAME),(response,RESPONSE_NAME),(bank,BANK_NAME)]:
        assert path==(Path.home()/'forensic-dgp'/name).resolve()
    assert Path.cwd().resolve()==root and os.environ.get('TMUX'),'Human manual tmux required'
    assert sys.prefix==str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=3) as response_file:assert response_file.read().decode().rstrip().endswith('/g2-standard-4')
    if idle:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
        assert not result.stdout.strip(),'Competing GPU workload; preserve it'


def measure(root,response,bank,p,source,old,pin):
    require_vm(root,response,bank)
    assert not (root/'outputs').exists(),'Used diagnostic; retain it and do not rerun'
    out=root/STAGE;out.mkdir(parents=True);start=time.monotonic()
    counts=dict(restorer=0,recognizer=0,gradient_queries=0,neural_parameter_displacements=0,optimizer_updates=0,accepted_updates=0)
    free=shutil.disk_usage(root).free
    if free<p['minimum_free_bytes']:
        write(out/'failure.json',dict(stage='pre-model storage preflight',available_free_bytes=free,required_free_bytes=p['minimum_free_bytes'],counts=counts))
        raise AssertionError('Need3GiB free after installation; retain stop')
    identity=None;initial_rng=None;progress=dict(stage='initialize');torch=None
    try:
        import resource
        import numpy as np
        from PIL import Image
        import torch
        from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash,buffer_hash
        from input_scopes import fixed_reference_matrices
        from input_projection_contract import input_projection
        assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L4'
        assert str(torch.__version__)==p['expected_torch']
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.manual_seed(20261014);torch.cuda.manual_seed_all(20261014)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        def stop(_signum,_frame):raise TimeoutError('Finite image-input diagnostic stop')
        signal.signal(signal.SIGTERM,stop)
        def rng():return dict(torch_CPU_RNG=torch.get_rng_state().numpy(),**{f'torch_CUDA_RNG_{i}':v.cpu().numpy() for i,v in enumerate(torch.cuda.get_rng_state_all())})
        def restore_rng(values):
            torch.set_rng_state(torch.from_numpy(values['torch_CPU_RNG']))
            torch.cuda.set_rng_state_all([torch.from_numpy(values[f'torch_CUDA_RNG_{i}']) for i in range(torch.cuda.device_count())])
        def save_array(path,value):
            with Path(path).open('xb') as stream:np.save(stream,value,allow_pickle=False)
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic()-start<p['measurement_seconds'] and time.monotonic()-start<p['worker_seconds'],'Measurement time stop'
            assert shutil.disk_usage(root).free>=p['disk_reserve_bytes'],'Disk reserve stop'
            assert torch.cuda.max_memory_allocated()<=p['peak_VRAM_bytes'],'VRAM stop'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=p['peak_RSS_bytes'],'RSS stop'
            assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<=p['output_byte_cap'],'Output cap stop'
            assert counts['recognizer']<=40 and counts['gradient_queries']<=200
            assert all(counts[k]==0 for k in ['restorer','neural_parameter_displacements','optimizer_updates','accepted_updates'])
        def pixels(path,mode='RGB'):
            with Image.open(path) as image:
                assert image.size==(256,256);return np.asarray(image.convert(mode)).copy()
        def tensor(a):return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous().cuda()
        identity=FixedObservedIdentity(bank/'weights/w600k_r50.onnx','cuda')
        frozen=state_hash(identity);buffers=buffer_hash(identity)
        assert frozen==RECOGNIZER_STATE and all(not m.training for m in identity.modules())
        identity.encoder.register_forward_hook(lambda *_:counts.__setitem__('recognizer',counts['recognizer']+1))
        def frozen_check():
            assert all(not v.requires_grad and v.grad is None for v in identity.parameters())
            assert all(not m.training for m in identity.modules())
            assert state_hash(identity)==frozen and buffer_hash(identity)==buffers
        initial_rng=rng();np.savez_compressed(out/'initial_RNG.npz',**initial_rng)
        write(out/'environment.json',dict(host=platform.node(),GPU=torch.cuda.get_device_name(0),torch=str(torch.__version__),CUDA=torch.version.cuda,
            manual_tmux=True,recognizer_state=frozen,buffer_state=buffers,seed=20261014,
            precision=dict(RGB='canonical saved float32',gradient='float32',derivative_scalar='double cosine dot',projection='float64 saved displacement',
                scientific_metric='unchanged canonical float32 score',matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True),
            derivative_target='RGB leaf only; frozen recognizer parameters',restorer_or_prior_constructed=False,optimizer_constructed=False,
            model_parameters_requiring_grad=0,python=sys.version))
        parent=response/'outputs/conditioning_response_v1';initial=read(parent/'initial/snapshot.json')
        rows={r['id']:r for r in initial['rows']};refs={r['id']:r for r in old['references']}
        plans={label:read(parent/label/'snapshot.json') for label in p['signed_labels']}
        plan_rows={label:{r['id']:r for r in snapshot['rows']} for label,snapshot in plans.items()}
        for repeat in range(2):
            progress.update(stage='RGB-input derivatives',repeat=repeat);folder=out/f'repeat{repeat:02d}';folder.mkdir()
            np.savez_compressed(folder/'RNG_before.npz',**rng());round_start=time.monotonic();durations=[];all_rows=[]
            for serial,block in enumerate(p['reference_blocks']):
                clock();batch_start=time.monotonic();stem=f"b{block['begin']//5:03d}"
                progress.update(begin=block['begin'],case_index=None)
                cases=old['cases'][block['begin']:block['begin']+5]
                before=np.load(parent/'initial'/(stem+'.npy'),allow_pickle=False)
                assert before.dtype==np.float32 and before.shape==(5,256,256,3) and np.isfinite(before).all() and before.min()>=0 and before.max()<=1
                target8=np.stack([pixels(bank/refs[c['source_person_or_reference']]['target']) for c in cases])
                mask8=np.stack([pixels(bank/refs[c['source_person_or_reference']]['observed'],'L')>0 for c in cases])
                fixed=np.stack([grid112(m) for m in fixed_reference_matrices(cases,refs)])
                rgb=torch.from_numpy(before).permute(0,3,1,2).contiguous().cuda().requires_grad_(True)
                assert rgb.is_leaf and np.array_equal(rgb.detach().permute(0,2,3,1).cpu().numpy(),before)
                target=tensor(target8);mask=torch.from_numpy(mask8.astype(np.float32)).cuda()[:,None];grids=torch.from_numpy(fixed).cuda()
                vectors=identity.embedding(torch.cat([rgb,rgb.detach(),target]),torch.cat([mask]*3),torch.cat([grids]*3))
                saved_vectors=vectors.detach().cpu().numpy().copy()
                with np.load(parent/'initial'/(stem+'_embeddings.npz'),allow_pickle=False) as prior:expected=prior['raw']
                assert np.allclose(saved_vectors,expected,rtol=1e-6,atol=1e-6),'Frozen initial embeddings exceed retained bound'
                np.savez_compressed(folder/(stem+'_embeddings.npz'),raw=saved_vectors)
                save_array(folder/(stem+'_grid.npy'),fixed)
                losses=1-(vectors[:5].double()*vectors[10:15].detach().double()).sum(1)
                gradients=[];batch_rows=[]
                changes={label:np.load(parent/label/(stem+'.npy'),allow_pickle=False) for label in p['signed_labels']}
                for i,c in enumerate(cases):
                    clock();loss=losses[i]
                    progress.update(case_index=i,case_id=c['id'])
                    g_all=torch.autograd.grad(loss,rgb,retain_graph=i<4,allow_unused=False)[0];counts['gradient_queries']+=1
                    whole=g_all.detach().permute(0,2,3,1).cpu().numpy().copy()
                    assert whole.dtype==np.float32 and whole.shape==before.shape and np.isfinite(whole).all()
                    cross=float(np.max(np.abs(np.delete(whole,i,axis=0))));assert cross==0,'Unexpected cross-case dependence; retain stop'
                    g=whole[i].copy();gradients.append(g)
                    value=float(loss.detach());scientific_loss=1-rows[c['id']]['raw']['ArcFace_observed_fixed']
                    fresh=1-float(saved_vectors[i].astype(np.float64)@saved_vectors[10+i].astype(np.float64))
                    assert abs(value-fresh)<=1e-13 and abs(value-scientific_loss)<=1e-6
                    projections=[]
                    for label,after in changes.items():
                        assert after.dtype==np.float32 and after.shape==before.shape and np.array_equal(after[i][~mask8[i]],before[i][~mask8[i]])
                        projections.append(dict(label=label,saved_RAW_sha256=sha(parent/label/(stem+'.npy')),
                            input_projection=input_projection(g,before[i],after[i]),
                            actual_scientific_loss=1-plan_rows[label][c['id']]['raw']['ArcFace_observed_fixed']))
                    row=dict(id=c['id'],source=c['source'],profile=c['profile'],reference=c['source_person_or_reference'],begin=block['begin'],case_index=i,
                        derivative_scalar=value,fresh_double_loss=fresh,scientific_cached_loss=scientific_loss,scalar_alignment_bound=1e-6,
                        gradient_norm=float(np.linalg.norm(g.astype(np.float64))),gradient_HWC_bytes_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                        cross_case_gradient_maximum=cross,projections=projections)
                    batch_rows.append(row);all_rows.append(row);del g_all,whole,g
                gradient_path=folder/(stem+'_RGB_gradients.npy');save_array(gradient_path,np.stack(gradients))
                write(folder/(stem+'.json'),dict(begin=block['begin'],case_ids=block['case_ids'],rows=batch_rows,
                    initial_RAW_sha256=sha(parent/'initial'/(stem+'.npy')),gradients_sha256=sha(gradient_path),
                    embeddings_sha256=sha(folder/(stem+'_embeddings.npz')),grid_sha256=sha(folder/(stem+'_grid.npy')),
                    maximum_initial_embedding_difference=float(np.max(np.abs(saved_vectors-expected))),
                    original_forward_embedding_rtol=1e-6,original_forward_embedding_atol=1e-6))
                assert rgb.grad is None;del rgb,target,mask,grids,vectors,losses,gradients,changes
                frozen_check();clock();durations.append(time.monotonic()-batch_start)
                if serial==1:
                    projection=max(durations[:2])*20*1.5
                    write(folder/'timing_projection.json',dict(first_two_batch_seconds=durations[:2],projected_seconds=projection,cap_seconds=180))
                    assert projection<=180,'Input gradient round projection exceeds180s'
                assert time.monotonic()-round_start<=180,'Input gradient round time stop'
                print(json.dumps(dict(repeat=repeat,batches=serial+1,of=20,input_gradient_queries=counts['gradient_queries'])),flush=True)
            np.savez_compressed(folder/'RNG_after.npz',**rng())
            write(folder/'repeat.json',dict(repeat=repeat,rows=all_rows,recognizer_state=state_hash(identity),buffer_state=buffer_hash(identity),
                reference_batch_order=[b['begin'] for b in p['reference_blocks']],seconds=time.monotonic()-round_start))
        restore_rng(initial_rng);np.savez_compressed(out/'restored_RNG.npz',**rng());frozen_check();clock()
        assert counts==dict(restorer=0,recognizer=40,gradient_queries=200,neural_parameter_displacements=0,optimizer_updates=0,accepted_updates=0)
        write(out/'run_results.json',dict(complete_diagnostic=True,protocol_sha256=pin,counts=counts,seconds=time.monotonic()-start,
            peak_VRAM=torch.cuda.max_memory_allocated(),peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            frozen_initial_and_final_state=frozen,buffer_state=buffers,Torch_RNG_restored=True,
            derivative_localization_conclusions_require_independent_audit=True,full_epochs=0,native_pixels_used=0,final_identity_pixels=0,
            model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    except BaseException as error:
        state=None;rollback_error=None
        try:
            if identity is not None:state=state_hash(identity);assert state==RECOGNIZER_STATE
            if initial_rng is not None:restore_rng(initial_rng)
        except BaseException as rollback:rollback_error=repr(rollback)
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),progress=progress,counts=counts,
            seconds=time.monotonic()-start,fresh_frozen_recognizer_state=state,RNG_restore_error=rollback_error,
            optimizer_updates=0,accepted_updates=0,model_qualified=False))
        raise


def export(root,response,bank,p,pin):
    require_vm(root,response,bank,idle=False);start=time.monotonic();manifest=root/'export_manifest.json'
    assert not manifest.exists();files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    payload=sum(f.stat().st_size for f in files)
    assert payload<=p['output_byte_cap']+64*1024**2 and shutil.disk_usage(root).free>=payload+p['disk_reserve_bytes']+1024**2
    write(manifest,dict(protocol_sha256=pin,response_protocol_sha256=RESPONSE_PIN,bank_protocol_sha256=BANK_PIN,
        files={f.relative_to(root).as_posix():sha(f) for f in files},complete_diagnostic=(root/STAGE/'run_results.json').exists(),
        failure_present=(root/STAGE/'failure.json').exists(),optimizer_updates=0,accepted_updates=0,training_success_not_implied=True,complete_pilot=False))
    archive=Path.home()/(STEM+'-results.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in files+[manifest]:
            assert time.monotonic()-start<p['export_seconds'],'Export time stop'
            assert shutil.disk_usage(root).free>=p['disk_reserve_bytes']+file.stat().st_size,'Export reserve stop'
            tar.add(file,arcname=NAME+'/'+file.relative_to(root).as_posix(),recursive=False)
            assert archive.stat().st_size<=p['output_byte_cap']+64*1024**2
    digest=sha(archive);assert time.monotonic()-start<p['export_seconds'] and shutil.disk_usage(root).free>=p['disk_reserve_bytes']
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    receipt=dict(complete=True,archive_sha256=digest,bytes=archive.stat().st_size,seconds=time.monotonic()-start,optimizer_updates=0,accepted_updates=0,training_success_not_implied=True)
    write(Path.home()/(STEM+'-export.json'),receipt);print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--response-root',type=Path,required=True);parser.add_argument('--bank-root',type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True);action=parser.add_mutually_exclusive_group(required=True)
    for flag in ['verify-transfer','measure','export']:action.add_argument('--'+flag,action='store_true')
    a=parser.parse_args();root,response,bank=[v.resolve() for v in [a.root,a.response_root,a.bank_root]]
    if a.measure:require_vm(root,response,bank)
    p,source,old=verify(root,response,bank,a.protocol_sha)
    if a.verify_transfer:print(json.dumps(dict(complete=True,packet_assets=len(p['assets_sha256']),response_assets=len(p['response_required_sha256']),
        bank_assets=len(p['bank_required_sha256']),cases=100,maximum_gradient_queries=200,maximum_recognizer_forwards=40,
        optimizer_updates=0,neural_parameter_displacements=0,actual_neural_calls=0,available_free_bytes=shutil.disk_usage(root).free,minimum_free_bytes=p['minimum_free_bytes'])),flush=True)
    elif a.measure:measure(root,response,bank,p,source,old,a.protocol_sha)
    else:export(root,response,bank,p,a.protocol_sha)


if __name__=='__main__':main()

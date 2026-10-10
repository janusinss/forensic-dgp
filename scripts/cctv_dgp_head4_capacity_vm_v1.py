"""Manual L4 50-update capacity stage, all TRAIN raw/PNG records retained."""
import argparse
import ast
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cctv_dgp_head4_capacity_contract_v1 import NAME,STEM,STATE,BUDGETS,WEIGHTS,NORMALIZERS,read,write,sha,verify


def scope(root):
    assert sys.platform=='linux' and platform.node().split('.')[0]=='forensic-dgp-thesis','Actual training requires the manual existing Linux VM'
    assert root==(Path.home()/'forensic-dgp'/NAME).resolve()


def run(root,p,pin):
    scope(root);assert os.environ.get('TMUX'),'Manual tmux required'
    assert not (root/'outputs').exists(),'Preserve every previous run; no repeat or failed resume'
    out=root/'outputs';out.mkdir();start=time.monotonic()
    candidate=optimizer=scheduler=save_state=None
    progress={'optimizer_updates':0,'backwards':0,'gradient_queries':0,'completed_epochs':0,'epoch_fraction':0.,'optimizer_constructed':False}
    counts={'original':0,'candidate':0,'recognizer':0}
    try:
        assert shutil.disk_usage(root).free>=BUDGETS['minimum_free_disk_bytes'],'Need14GiB free after install; stop before neural calls'
        tree=ast.parse((root/'frozen_definitions.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_vm')
        ns={'sys':sys,'platform':platform,'Path':Path,'os':os,'subprocess':subprocess,'urllib':__import__('urllib')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-hardware-idle>','exec'),ns);ns['require_vm'](root,idle=True)
        import hashlib
        import importlib.metadata
        import numpy as np
        import torch
        from PIL import Image
        from cctv_dgp_head4_capacity_model_v1 import Head4CapacityDGP
        from cctv_dgp_head4_lossless_v1 import pack,unpack
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash,buffer_hash
        from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
        from frozen_capacity_contract import feature_support,capacity,exported_pixel_metrics,detail_metric
        from frozen_raw_metrics import pixel_metrics,detail_float,deliver,mean_only,review_groups
        random.seed(501050);np.random.seed(501050);torch.manual_seed(501050);torch.cuda.manual_seed_all(501050)
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
        original,_=load_frozen_dgp_restorer(root/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cuda')
        candidate=Head4CapacityDGP(original.net).cuda().eval();candidate.enable_vm_learning(root)
        identity=FixedObservedIdentity(root/'weights/w600k_r50.onnx','cuda')
        assert state_hash(original.net)==STATE and state_hash(candidate)==p['initial_candidate_state']
        assert state_hash(identity)==p['recognizer_state']
        frozen_names=set(candidate.learning_names());parameters=candidate.learning_parameters()
        def frozen_hash():return state_hash({n:v for n,v in candidate.state_dict().items() if n not in frozen_names})
        before={'original':STATE,'recognizer':state_hash(identity),'frozen':frozen_hash(),'buffers':buffer_hash(candidate)}
        for model,label in [(original.net,'original'),(candidate,'candidate'),(identity.encoder,'recognizer')]:
            model.register_forward_hook(lambda *_a,label=label:counts.__setitem__(label,counts[label]+1))
        def frozen():
            assert state_hash(original.net)==before['original'] and state_hash(identity)==before['recognizer']
            assert frozen_hash()==before['frozen'] and buffer_hash(candidate)==before['buffers']
            assert all(v.grad is None and not v.requires_grad for m in [original,identity] for v in m.parameters())
            assert all(v.grad is None and not v.requires_grad for n,v in candidate.named_parameters() if n not in frozen_names)
            assert all(not m.training for model in [original,candidate,identity] for m in model.modules())
        def retained_bytes():return sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic()-start<BUDGETS['worker_seconds'],'Worker2400s stop'
            assert shutil.disk_usage(root).free>=BUDGETS['disk_reserve_bytes'],'Disk1GiB reserve stop'
            assert torch.cuda.max_memory_allocated()<=BUDGETS['peak_vram_bytes'],'VRAM20GiB stop'
            assert retained_bytes()<=BUDGETS['return_uncompressed_bytes'],'Output6GiB stop'
            for label in counts:assert counts[label]<=BUDGETS[label+'_forward_calls'],'Forward cap '+label
        def pixels(path,mode='RGB'):
            with Image.open(path) as im:
                assert im.size==(256,256);return np.asarray(im.convert(mode)).copy()
        def rgb(a):return torch.from_numpy(a.astype(np.float32)/np.float32(255)).cuda().permute(2,0,1)[None]
        refs={r['id']:r for r in p['references']};cases=p['cases'];by_id={c['id']:i for i,c in enumerate(cases)}
        def load_batch(ids):
            selected=[cases[i] for i in ids];cameras=[];targets=[];masks=[];features=[];grids=[]
            for c in selected:
                ref=refs[c['source_person_or_reference']];camera=pixels(root/c['input']);target=pixels(root/ref['target']);mask=pixels(root/ref['observed'],'L')>0
                cameras.append(camera);targets.append(target);masks.append(mask);features.append(feature_support(mask,c['landmarks5_canvas_xy']));grids.append(grid112(ref['matrix112']))
            return {'cases':selected,'camera':cameras,'target8':targets,'mask8':masks,'feature8':features,
                'x':torch.cat([rgb(a) for a in cameras]),'target':torch.cat([rgb(a) for a in targets]),
                'mask':torch.from_numpy(np.stack(masks).astype(np.float32)).cuda()[:,None],
                'feature':torch.from_numpy(np.stack(features).astype(np.float32)).cuda()[:,None],
                'grid':torch.from_numpy(np.stack(grids)).cuda()}
        write(out/'environment.json',{'python':sys.version,'GPU':torch.cuda.get_device_name(0),'host':platform.node(),
            'CUDA':torch.version.cuda,'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','Pillow','scipy','scikit-image','onnx','onnx2torch']},
            'AMP':False,'TF32':False,'normalization':'unchanged stored evaluation statistics',
            'deterministic_cudnn':True,'bitwise_future_resume_guaranteed':False})
        def snapshot(update):
            stamp=time.monotonic();folder=out/('update'+str(update));folder.mkdir();(folder/'packs').mkdir();(folder/'previews').mkdir();rows=[]
            frozen();torch.save({n:v.detach().cpu().clone() for n,v in candidate.state_dict().items()},folder/'candidate.pth')
            with torch.no_grad():
                for begin in range(0,3905,5):
                    clock();assert time.monotonic()-stamp<BUDGETS['snapshot_seconds'],'Snapshot900s stop'
                    b=load_batch(list(range(begin,begin+5)));ids=[c['id'] for c in b['cases']]
                    pred=torch.where(b['mask'].bool(),candidate(b['x']),b['x'])
                    a=pred.permute(0,2,3,1).cpu().numpy().copy()
                    if update==0:
                        zero=torch.where(b['mask'].bool(),original.net(b['x']),b['x']);assert torch.equal(pred,zero),'Full TRAIN initializer parity failed'
                        base=a
                    else:base,_=unpack(out/'update0/packs'/f'b{begin//5:04d}.npz',ids)
                    pngs=[deliver(raw,camera,mask) for raw,camera,mask in zip(a,b['camera'],b['mask8'])]
                    # Same batch-size15 context at every snapshot; raw/PNG/truth.
                    vectors=identity.embedding(torch.cat([pred,torch.cat([rgb(q) for q in pngs]),b['target']]),
                        torch.cat([b['mask']]*3),torch.cat([b['grid']]*3)).cpu().numpy().copy()
                    em=np.stack([vectors[:5],vectors[5:10],vectors[10:15]],axis=1)
                    pack(folder/'packs'/f'b{begin//5:04d}.npz',a,ids,em,baseline=None if update==0 else base)
                    for i,c in enumerate(b['cases']):
                        cid=c['id'];raw=a[i];png=pngs[i];target=b['target8'][i];mask=b['mask8'][i];feature=b['feature8'][i]
                        mraw,mpng,shift=mean_only(raw,base[i],b['camera'][i],mask)
                        rm=pixel_metrics(raw,target,mask);pm=exported_pixel_metrics(png,target,mask)
                        rm.update({'ArcFace_observed_fixed':float(em[i,0]@em[i,2]),'landmark_high_frequency_MSE':detail_float(raw,target,feature),
                            'constant_mean_shift_only_MSE':pixel_metrics(mraw,target,mask)['MSE']})
                        pm.update({'ArcFace_observed_fixed':float(em[i,1]@em[i,2]),'landmark_high_frequency_MSE':detail_metric(png,target,feature),
                            'constant_mean_shift_only_MSE':exported_pixel_metrics(mpng,target,mask)['MSE']})
                        rows.append({'id':cid,'source':c['source'],'profile':c['profile'],'raw':rm,'png':pm,'postclip_mean_RGB_shift':shift.tolist(),
                            'raw_RGB_float32_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'PNG_RGB_sha256':hashlib.sha256(png.tobytes()).hexdigest()})
                        if cid in p['preview_case_ids']:Image.fromarray(png).save(folder/'previews'/(cid+'.png'))
                    if begin%250==0:print({'head4_snapshot':update,'cases':begin+5,'of':3905},flush=True)
            groups={stage:review_groups(rows,stage) for stage in ['raw','png']}
            result={'complete':True,'update':update,'rows':rows,'groups':groups,'candidate_state':state_hash(candidate),
                'checkpoint_sha256':sha(folder/'candidate.pth'),'duration_seconds':time.monotonic()-stamp,
                'full_TRAIN_initial_parity_cases':3905 if update==0 else None,'raw_codec_lossless':True,'model_qualification':False}
            write(folder/'metrics.json',result);frozen();clock();return result
        baseline=snapshot(0)
        initial_bytes=retained_bytes();projection=initial_bytes*2*1.25+64*1024**2
        write(out/'storage_projection.json',{'initial_bytes':initial_bytes,'snapshots':2,'factor':1.25,'projected_bytes':projection,
            'cap_bytes':BUDGETS['return_uncompressed_bytes'],'exact_raw_storage':True})
        assert projection<=BUDGETS['return_uncompressed_bytes'],'Measured two-snapshot storage projection exceeds6GiB; no optimizer'
        assert shutil.disk_usage(root).free>=2*projection-initial_bytes+BUDGETS['disk_reserve_bytes'],'Cannot reserve output and export; no optimizer'
        def training_terms(b,pred,base):
            t=pixel_and_structure_losses(pred,b['target'],b['mask'],b['feature'])
            with torch.no_grad():bt=pixel_and_structure_losses(base,b['target'],b['mask'],b['feature'])
            vectors=identity.embedding(torch.cat([pred,base.detach(),b['target']]),torch.cat([b['mask']]*3),torch.cat([b['grid']]*3))
            truth=vectors[10:15].detach().double()
            arc=1-(vectors[:5].double()*truth).sum(1);barc=1-(vectors[5:10].detach().double()*truth).sum(1)
            components=[t['MSE'].mean()/NORMALIZERS[0],t['SSIM_loss'].mean()/NORMALIZERS[1],arc.mean()/NORMALIZERS[2],t['landmark_structure'][1:].mean()/NORMALIZERS[3]]
            guards=5*(torch.relu(t['MSE']-bt['MSE']-1e-12).mean()/NORMALIZERS[0]+
                torch.relu(t['SSIM_loss']-bt['SSIM_loss']-1e-6).mean()/NORMALIZERS[1]+torch.relu(arc-barc-1e-6).mean()/NORMALIZERS[2])
            return components,guards
        def get_base(ids):
            assert len(ids)==5 and ids==list(range(ids[0],ids[0]+5)) and ids[0]%5==0
            cs=[cases[i]['id'] for i in ids];a,_=unpack(out/'update0/packs'/f'b{ids[0]//5:04d}.npz',cs)
            return torch.from_numpy(a).cuda().permute(0,3,1,2)
        grad_records=[]
        for ids in p['preflight_batches']:
            clock();b=load_batch(ids);base=get_base(ids);pred=torch.where(b['mask'].bool(),candidate(b['x']),b['x'])
            assert torch.equal(pred,base),'Gradient-context initial parity changed'
            components,guards=training_terms(b,pred,base)
            for component in [0,2,3]:
                gg=torch.autograd.grad(components[component],parameters,retain_graph=True,allow_unused=False);progress['gradient_queries']+=1
                norms=[float(g.detach().double().norm()) for g in gg]
                assert all(torch.isfinite(g).all() for g in gg) and all(n>0 for n in norms),'Disconnected or nonfinite trainable piece'
                grad_records.append({'case_ids':[cases[i]['id'] for i in ids],'component':component,'part_gradient_L2':norms})
            del pred,components,guards,gg
        write(out/'gradient_preflight.json',{'complete':True,'queries':6,'rows':grad_records,'optimizer_updates':0})
        frozen();assert progress['gradient_queries']==6
        optimizer=torch.optim.Adam(parameters,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=781,gamma=1.)
        progress['optimizer_constructed']=True
        def save(update,path,gate_status):
            numpy_rng=np.random.get_state()
            torch.save({'format':'head4-capacity-portable-full-state-v1','protocol_sha256':pin,
                'model':{n:v.detach().cpu().clone() for n,v in candidate.state_dict().items()},'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'python_rng':random.getstate(),
                'numpy_rng':(numpy_rng[0],torch.from_numpy(numpy_rng[1].astype(np.int64)),*numpy_rng[2:]),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'completed_updates':update,
                'completed_epochs':0,'epoch_fraction':update/781,'next_schedule_index':update,
                'original_checkpoint_sha256':p['original_checkpoint_sha256'],'gate_status':gate_status,
                'failed_gate_must_not_resume':True,'automatic_resume':False,'bitwise_resume_guaranteed':False,
                'optimizer_parameter_names':candidate.learning_names(),'schedule_sha256':sha(root/'protocol.json')},path)
        save_state=save;save(0,out/'update0/training_state.pt','not_run')
        fit_start=time.monotonic()
        with (out/'training_steps.jsonl').open('x',encoding='utf-8') as log:
            for update,ids in enumerate(p['schedule'],1):
                clock();assert time.monotonic()-fit_start<BUDGETS['fit_seconds'],'Fit300s stop'
                b=load_batch(ids);base=get_base(ids);optimizer.zero_grad(set_to_none=True)
                pred=torch.where(b['mask'].bool(),candidate(b['x']),b['x']);components,guards=training_terms(b,pred,base)
                objective=sum(w*term for w,term in zip(WEIGHTS,components))+guards
                assert torch.isfinite(objective);objective.backward();progress['backwards']+=1
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters)
                frozen();gradnorm=float(torch.nn.utils.clip_grad_norm_(parameters,1.));optimizer.step();scheduler.step()
                progress['optimizer_updates']=update;progress['epoch_fraction']=update/781
                log.write(__import__('json').dumps({'update':update,'case_ids':[cases[i]['id'] for i in ids],
                    'objective':float(objective.detach()),'normalized_components':[float(v.detach()) for v in components],
                    'regression_guard':float(guards.detach()),'gradient_L2_before_clip':gradnorm})+'\n');log.flush()
                if update==1 or update%10==0:print({'head4_update':update,'of':50,'objective':float(objective.detach())},flush=True)
                del pred,components,guards,objective
        optimizer.zero_grad(set_to_none=True);current=snapshot(50)
        gate={stage:capacity(baseline['groups'][stage],current['groups'][stage],.01) for stage in ['raw','png']}
        passed=all(g['pass'] for g in gate.values());write(out/'early_gate.json',{'update':50,'comparisons':gate,'pass':passed})
        save(50,out/'update50/training_state.pt','early_capacity_pass_pending_independent_review' if passed else 'failed_gate')
        frozen();clock()
        result={'complete':True,'protocol_sha256':pin,'progress':progress,'forward_counts':counts,'gate_pass':passed,
            'gates':gate,'initial_full_TRAIN_parity_cases':3905,'trainable_tensors':3,'trainable_elements':147456,
            'original_state':STATE,'frozen_state_hash':before['frozen'],'frozen_buffers_hash':before['buffers'],
            'recognizer_state':before['recognizer'],'candidate_state':state_hash(candidate),'original_checkpoint_unchanged':True,
            'optimizer_updates':50,'completed_epochs':0,'epoch_fraction':50/781,'stage50_only':True,
            'final800_ten_percent_not_tested':True,'additional_epochs_1_2_5_not_run':True,
            'native_DEV_or_reserved_final_used':False,'app_promotion':False,'model_qualification':False,
            'visual_review_pending':True,'independent_audit_pending':True,'goal_complete':False,'seconds':time.monotonic()-start}
        write(out/'results.json',result)
        assert passed,'Structure/preservation requirement failed at50; retain stop, checkpoint and full optimizer state'
        print({'complete':True,'updates':50,'early_capacity_pass':True,'qualified_model':False,'seconds':time.monotonic()-start},flush=True)
    except BaseException:
        if save_state is not None:
            try:save_state(progress['optimizer_updates'],out/'stopped_training_state.pt','failed_or_partial_stop')
            except Exception as e:write(out/'state_export_failure.json',{'error':repr(e)})
        write(out/'failure.json',{'complete':False,'traceback':traceback.format_exc(),'progress':progress,'forward_counts':counts,
            'training_success_not_implied':True,'seconds':time.monotonic()-start});raise


def export(root,p,pin):
    scope(root);stamp=time.monotonic();destination=Path.home()/(STEM+'-results.tar.gz');partial=Path(str(destination)+'.partial')
    assert not destination.exists() and not partial.exists(),'Preserve exports'
    files=[f for f in (root/'outputs').rglob('*') if f.is_file()]
    files+=[root/n for n in ['protocol.json','trainer.log','trainer_exit_code.txt','supervisor_receipt.json'] if (root/n).exists()]
    files+=[root/n for n in p['assets_sha256'] if n.endswith('.py')]
    total=sum(f.stat().st_size for f in files)
    assert total<=BUDGETS['return_uncompressed_bytes'] and shutil.disk_usage(root).free>=total+BUDGETS['disk_reserve_bytes']
    write(root/'export_manifest.json',{'complete':True,'protocol_sha256':pin,'files_sha256':{f.relative_to(root).as_posix():sha(f) for f in files},
        'uncompressed_bytes':total,'training_success_not_implied':True})
    files.append(root/'export_manifest.json')
    with tarfile.open(partial,'x:gz',compresslevel=1) as tar:
        for f in sorted(files):
            assert time.monotonic()-stamp<BUDGETS['export_seconds'];assert not f.is_symlink() and f.resolve().is_relative_to(root)
            tar.add(f,arcname=NAME+'_return/'+f.relative_to(root).as_posix(),recursive=False)
    assert time.monotonic()-stamp<BUDGETS['export_seconds'];partial.rename(destination);digest=sha(destination)
    with Path(str(destination)+'.sha256').open('x',encoding='ascii') as f:f.write(digest+'  '+destination.name+'\n')
    result_path=root/'outputs/results.json';failure_path=root/'outputs/failure.json'
    updates=read(result_path)['optimizer_updates'] if result_path.exists() else read(failure_path)['progress']['optimizer_updates']
    receipt={'complete':True,'archive_sha256':digest,'bytes':destination.stat().st_size,'seconds':time.monotonic()-stamp,
        'optimizer_updates':updates,'run_results_present':result_path.exists(),'failure_present':failure_path.exists(),'training_success_not_implied':True}
    write(Path.home()/(STEM+'-export.json'),receipt);print(receipt,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    m=parser.add_mutually_exclusive_group(required=True)
    for mode in ['verify-transfer','run','export']:m.add_argument('--'+mode,action='store_true')
    a=parser.parse_args();root=a.root.resolve();p=verify(root,a.protocol_sha)
    if a.verify_transfer:print({'complete':True,'assets':len(p['assets_sha256']),'neural_or_training_calls':0});return
    scope(root)
    if a.export:export(root,p,a.protocol_sha);return
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Worker2400s stop')));signal.alarm(2400)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External deadline stop')))
    run(root,p,a.protocol_sha)


if __name__=='__main__':main()

"""Manual L4 matched gradient ablation. Zero optimizer steps or weight fitting."""
import argparse
import ast
import copy
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cctv_dgp_head4_reactivation_contract_v1 import NAME,STEM,BUDGETS,PARTS,COMPONENTS,STATE,read,write,sha,verified_assets


def scope(root):
    assert sys.platform=='linux' and platform.node().split('.')[0]=='forensic-dgp-thesis','Existing Linux VM only; no local gradients'
    assert root==(Path.home()/'forensic-dgp'/NAME).resolve(),'New isolated packet root only'


def hardware_idle(root):
    tree=ast.parse((root/'frozen_definitions.py').read_text())
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_vm')
    ns={'sys':sys,'platform':platform,'Path':Path,'os':os,'subprocess':subprocess,'urllib':__import__('urllib')}
    exec(compile(ast.Module(body=[guard],type_ignores=[]),'<pinned-L4-idle-guard>','exec'),ns)
    ns['require_vm'](root,idle=True)


def run(root,p,pin):
    scope(root); assert os.environ.get('TMUX'),'Manual tmux launch required'
    assert not (root/'outputs').exists(),'Preserve every result/partial failure; no rerun or resume'
    start=time.monotonic(); out=root/'outputs'; out.mkdir()
    progress={'gradient_queries':0,'optimizer_updates':0,'epochs':0,'optimizer_constructed':False,
        'model_parameter_updates':0,'completed_batches':0}
    counts={'original':0,'repaired':0,'recognizer':0}; gradients=None
    try:
        assert shutil.disk_usage(root).free>=BUDGETS['minimum_free_disk_bytes'],'Need3GiB free after install'
        hardware_idle(root)
        import importlib.metadata
        import numpy as np
        import torch
        from PIL import Image
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_head4_reactivation_v1 import ReactivatedDGP
        from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash,buffer_hash
        from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
        from frozen_capacity_contract import feature_support
        from frozen_raw_metrics import deliver
        torch.set_num_threads(4); torch.manual_seed(101010); np.random.seed(101010)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        source,_=load_frozen_dgp_restorer(root/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cuda')
        original=copy.deepcopy(source.net).eval().requires_grad_(False)
        repaired=ReactivatedDGP(source.net).cuda().eval()
        repaired.enable_vm_gradients(root)
        selected_names=['head4.block0.weight','head4.block1.weight','smooth.0.weight']
        for model in [original,repaired.net]:
            for name,value in model.named_parameters(): value.requires_grad_(name in selected_names)
        params={label:[dict(model.named_parameters())[name] for name in selected_names]
            for label,model in [('original',original),('repaired',repaired.net)]}
        identity=FixedObservedIdentity(root/'weights/w600k_r50.onnx','cuda')
        def states():
            return {'source':state_hash(source.net),'original':state_hash(original),'repaired':state_hash(repaired),
                'source_buffers':buffer_hash(source.net),'original_buffers':buffer_hash(original),
                'repaired_buffers':buffer_hash(repaired),'recognizer':state_hash(identity)}
        before=states()
        assert before['source']==before['original']==STATE
        assert before['repaired']==p['initial_repaired_state']
        assert before['recognizer']==p['recognizer_state']
        for model,label in [(original,'original'),(repaired,'repaired'),(identity.encoder,'recognizer')]:
            model.register_forward_hook(lambda *_a,label=label:counts.__setitem__(label,counts[label]+1))
        pre={}
        for model,label in [(original,'original'),(repaired.net,'repaired')]:
            model.head4.block1.register_forward_hook(lambda _m,_i,y,label=label:pre.__setitem__(label,y.detach()))
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic()-start<=BUDGETS['worker_seconds'],'Worker time stop'
            assert torch.cuda.max_memory_allocated()<=BUDGETS['peak_vram_bytes'],'VRAM20GiB stop'
            assert shutil.disk_usage(root).free>=BUDGETS['disk_reserve_bytes'],'Disk reserve stop'
            for key in counts: assert counts[key]<=BUDGETS[key+'_forward_calls'],'Forward cap '+key
            assert progress['gradient_queries']<=BUDGETS['gradient_queries']
        def pixels(path,mode='RGB'):
            with Image.open(path) as image:
                assert image.size==(256,256)
                return np.asarray(image.convert(mode)).copy()
        def rgb(a):return torch.from_numpy(a.astype(np.float32)/np.float32(255)).cuda().permute(2,0,1)[None]
        refs={r['id']:r for r in p['references']}; records=[]; query_rows=[]
        write(out/'environment.json',{'python':sys.version,'GPU':torch.cuda.get_device_name(0),
            'host':platform.node(),'CUDA':torch.version.cuda,'machine_type':'g2-standard-4',
            'packages':{k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','Pillow','scipy','scikit-image','onnx','onnx2torch']},
            'AMP':False,'TF32':False,'seed':101010,'normalization':'unchanged stored evaluation statistics'})
        gradients=np.lib.format.open_memmap(out/'individual_gradients.npy',mode='w+',dtype=np.float64,shape=tuple(p['gradient_shape']))
        gradients[:]=np.nan; gradients.flush()
        raw_dir=out/'initial_outputs'; raw_dir.mkdir()
        for batch_index,begin in enumerate(range(0,100,5)):
            clock(); cases=p['cases'][begin:begin+5]; cameras=[]; targets=[]; supports=[]; masks=[]; grids=[]
            for c in cases:
                r=refs[c['source_person_or_reference']]; camera=pixels(root/c['input']); target=pixels(root/r['target'])
                mask=pixels(root/r['observed'],'L')>0
                cameras.append(camera); targets.append(target); masks.append(mask)
                supports.append(feature_support(mask,c['landmarks5_canvas_xy'])); grids.append(grid112(r['matrix112']))
            x=torch.cat([rgb(a) for a in cameras]); target=torch.cat([rgb(a) for a in targets])
            mask=torch.from_numpy(np.stack(masks).astype(np.float32)).cuda()[:,None]
            support=torch.from_numpy(np.stack(supports).astype(np.float32)).cuda()[:,None]
            grid=torch.from_numpy(np.stack(grids)).cuda()
            with torch.no_grad(): truth=identity.embedding(target,mask,grid).detach().clone()
            output_values=[]; embeddings=[]; losses=[]
            for label,model in [('original',original),('repaired',repaired)]:
                output=torch.where(mask.bool(),model(x),x)
                embedding=identity.embedding(output,mask,grid)
                terms=pixel_and_structure_losses(output,target,mask,support)
                components=[terms['MSE'].mean(),terms['SSIM_loss'].mean(),
                    (1-(embedding.double()*truth.double()).sum(1)).mean(),terms['landmark_structure'][1:].mean()]
                output_values.append(output.detach().permute(0,2,3,1).cpu().numpy().copy())
                embeddings.append(embedding.detach().cpu().numpy().copy()); losses.append([float(v.detach()) for v in components])
                model_index=0 if label=='original' else 1
                for component_index,loss in enumerate(components):
                    clock()
                    values=torch.autograd.grad(loss,params[label],retain_graph=component_index<3,allow_unused=False)
                    progress['gradient_queries']+=1
                    pieces=[values[0],values[1],values[2][:,:64]]
                    arrays=[v.detach().double().cpu().numpy().reshape(-1).copy() for v in pieces]
                    assert all(np.isfinite(a).all() for a in arrays),'Nonfinite connected gradient'
                    vector=np.concatenate(arrays); assert vector.shape==(147456,)
                    gradients[model_index,batch_index,component_index]=vector
                    query_rows.append({'model':label,'batch':batch_index,'component':COMPONENTS[component_index],
                        'case_ids':[c['id'] for c in cases],'loss':losses[-1][component_index],
                        'part_L2':[float(np.linalg.norm(a)) for a in arrays],
                        'part_nonzero_elements':[int(np.count_nonzero(a)) for a in arrays]})
                assert all(v.grad is None for v in model.parameters())
                del components,terms,embedding,output,values,pieces
            gradients.flush()
            assert np.array_equal(output_values[0],output_values[1]),'Repaired initialization changes CUDA output'
            assert np.array_equal(embeddings[0],embeddings[1]),'Repaired initialization changes CUDA recognition'
            assert losses[0]==losses[1],'Matched initial losses differ'
            for i,c in enumerate(cases):
                raw=output_values[1][i]; png=deliver(raw,cameras[i],masks[i]); cid=c['id']
                np.save(raw_dir/(cid+'.npy'),raw,allow_pickle=False)
                Image.fromarray(png).save(raw_dir/(cid+'.png'))
                np.save(raw_dir/(cid+'_embeddings.npy'),np.stack([embeddings[0][i],embeddings[1][i],truth[i].detach().cpu().numpy()]),allow_pickle=False)
                records.append({'id':cid,'source':c['source'],'profile':c['profile'],
                    'original_repaired_raw_max_abs':0.,'original_repaired_embedding_max_abs':0.,
                    'original_head4_zero':bool((pre['original'][i]==0).all()),
                    'repaired_head4_positive_activations':int((pre['repaired'][i]>0).sum()),
                    'raw_sha256':sha(raw_dir/(cid+'.npy')),'png_sha256':sha(raw_dir/(cid+'.png'))})
            progress['completed_batches']+=1
            print({'head4_batch':batch_index+1,'of':20,'gradient_queries':progress['gradient_queries'],
                'original_part_norms':query_rows[-8]['part_L2'],'repaired_part_norms':query_rows[-4]['part_L2'],
                'seconds':time.monotonic()-start},flush=True)
        summaries=[]; split=np.cumsum([0]+[n for _,n in PARTS])
        for model_index,label in enumerate(['original','repaired']):
            for cohort_index,co in enumerate(p['cohorts']):
                for part_index,(name,_) in enumerate(PARTS):
                    # Improvement components MSE, ArcFace and degraded structure.
                    a=np.asarray(gradients[model_index,cohort_index*10:(cohort_index+1)*10,[0,2,3],split[part_index]:split[part_index+1]])
                    summaries.append({'model':label,'cohort':co['name'],'part':name,
                        'improvement_gradient_L2':float(np.linalg.norm(a.reshape(-1))),
                        'nonzero_elements':int(np.count_nonzero(a))})
        route_pass=all(r['improvement_gradient_L2']>0 for r in summaries if r['model']=='repaired')
        after=states(); assert after==before
        assert progress['gradient_queries']==160 and progress['completed_batches']==20
        assert all(v.grad is None for m in [source,original,repaired,identity] for v in m.parameters())
        assert all(not m.training for m in [source,original,repaired,identity] for m in m.modules())
        for name,digest in p['assets_sha256'].items(): assert sha(root/name)==digest,name
        clock()
        result={'complete':True,'protocol_sha256':pin,'initial_cases':records,'gradient_queries':query_rows,
            'connected_gradient_summaries':summaries,'connected_improvement_route_pass':route_pass,
            'states_before':before,'states_after':after,'forward_counts':counts,'progress':progress,
            'initial_output_parity_cases':100,'maximum_original_repaired_raw_error':0.,
            'original_zero_head4_cases':sum(r['original_head4_zero'] for r in records),
            'repaired_positive_head4_cases':sum(r['repaired_head4_positive_activations']>0 for r in records),
            'all_individual_gradient_vectors_saved':True,'optimizer':None,'scheduler':None,
            'optimizer_updates':0,'epochs':0,'new_trained_checkpoint':False,
            'capacity_or_quality_requirements_tested':False,'native_DEV_or_reserved_final_used':False,
            'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start}
        write(out/'results.json',result)
        assert route_pass,'Connected branch still has a zero improvement gradient; retain diagnostic stop'
        print({'complete':True,'connected_route_pass':True,'gradient_queries':160,'optimizer_updates':0,
            'quality_not_qualified':True,'seconds':time.monotonic()-start},flush=True)
    except BaseException:
        if gradients is not None: gradients.flush()
        if not (out/'failure.json').exists():
            write(out/'failure.json',{'complete':False,'traceback':traceback.format_exc(),'progress':progress,
                'forward_counts':counts,'optimizer_updates':0,'training_success_not_implied':True,
                'seconds':time.monotonic()-start})
        raise


def export(root,p,pin):
    scope(root); start=time.monotonic()
    destination=Path.home()/(STEM+'-results.tar.gz')
    partial=Path(str(destination)+'.partial')
    assert not destination.exists() and not partial.exists(),'Preserve prior/partial export'
    assert (root/'supervisor_receipt.json').exists()
    files=[f for f in (root/'outputs').rglob('*') if f.is_file()]
    files+=[root/name for name in ['protocol.json','diagnostic.log','diagnostic_exit_code.txt','supervisor_receipt.json'] if (root/name).exists()]
    # Scientific sources are small; large input/model assets remain pinned in protocol.
    files += [root/name for name in p['assets_sha256'] if name.endswith('.py') or name.endswith('.sh')]
    total=sum(f.stat().st_size for f in files)
    assert total<=BUDGETS['return_uncompressed_bytes'],'Return768MiB stop'
    assert shutil.disk_usage(root).free>=total+BUDGETS['disk_reserve_bytes'],'Archive space stop'
    write(root/'export_manifest.json',{'complete':True,'protocol_sha256':pin,
        'files_sha256':{f.relative_to(root).as_posix():sha(f) for f in files},'uncompressed_bytes':total,
        'gradient_queries_not_locally_replayed':True,'training_success_not_implied':True})
    files.append(root/'export_manifest.json')
    with tarfile.open(partial,'x:gz',compresslevel=1) as tar:
        for file in sorted(files):
            assert time.monotonic()-start<BUDGETS['export_seconds'],'Export time stop'
            assert not file.is_symlink() and file.resolve().is_relative_to(root)
            tar.add(file,arcname=NAME+'_return/'+file.relative_to(root).as_posix(),recursive=False)
    assert time.monotonic()-start<BUDGETS['export_seconds']
    partial.rename(destination); digest=sha(destination)
    with Path(str(destination)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(digest+'  '+destination.name+'\n')
    receipt={'complete':True,'archive_sha256':digest,'bytes':destination.stat().st_size,
        'seconds':time.monotonic()-start,'optimizer_updates':0,'training_success_not_implied':True,
        'run_results_present':(root/'outputs/results.json').exists(),'failure_present':(root/'outputs/failure.json').exists()}
    write(Path.home()/(STEM+'-export.json'),receipt); print(receipt,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ['verify-transfer','run','export','record-supervision']:modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--elapsed',type=float);parser.add_argument('--exit-code',type=int)
    a=parser.parse_args();root=a.root.resolve();p=verified_assets(root,a.protocol_sha)
    if a.verify_transfer:
        print({'complete':True,'assets':len(p['assets_sha256']),'optimizer_updates':0,'neural_or_gradient_calls':0},flush=True);return
    scope(root)
    if a.record_supervision:
        assert a.elapsed is not None and 0<=a.elapsed and a.exit_code is not None
        write(root/'supervisor_receipt.json',{'complete':True,'protocol_sha256':a.protocol_sha,
            'seconds':a.elapsed,'external_cap_seconds':630,'kill_grace_seconds':30,
            'within_external_bound':a.elapsed<=665,'diagnostic_exit_code':a.exit_code,'optimizer_updates':0});return
    if a.export:export(root,p,a.protocol_sha);return
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Worker600s stop')));signal.alarm(600)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External deadline stop')))
    run(root,p,a.protocol_sha)


if __name__=='__main__':main()

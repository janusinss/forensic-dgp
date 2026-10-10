"""Manual L4 finite inference of saved proposals: zero optimizers or gradients."""
import argparse
import ast
from pathlib import Path
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from types import MethodType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_actual_step_review_v1_contract import NAME, STEM, RETURN_PREFIX, PROPOSALS, ROLES, BUDGETS, sha, read, write, vm_scope, verify, role_ids, output_prefix, allowed_return_names


def timeout_signal(*_):
    raise TimeoutError('Finite actual-step review stop limit reached')


def review(root,p,pin):
    vm_scope(root);out=root/'outputs'
    assert not out.exists(), 'Retain every earlier/partial review; no resume or automatic repeat'
    assert shutil.disk_usage(root).free >= BUDGETS['minimum_free_bytes'], 'Need6GiB free after installation; no deletion is performed'
    out.mkdir();start=time.monotonic();candidate=None;seed=None;completed=[];bytes_written=0
    progress={'optimizer_updates':0,'gradient_queries':0,'backward_calls':0,'optimizer_constructed':False,
              'new_trained_checkpoint':False,'parameter_proposals_loaded':0,'forward_slots':0,'completed_conditions':0}
    signal.signal(signal.SIGALRM,timeout_signal);signal.alarm(BUDGETS['worker_seconds'])
    try:
        # The original idle/machine guard is compiled alone, before model imports.
        tree=ast.parse((root/'frozen_definitions.py').read_text(encoding='utf-8'))
        guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_vm')
        import urllib.request
        env={'sys':sys,'platform':platform,'Path':Path,'os':os,'subprocess':subprocess,'urllib':__import__('urllib')}
        exec(compile(ast.Module(body=[guard],type_ignores=[]),'<unchanged-existing-VM-guard>','exec'),env)
        env['require_vm'](root,idle=True)
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        from cctv_dgp_actual_step_review_v1_metrics import parameter_state_hash, pixel_metrics, detail_float, deliver, mean_only, review_groups, finite_comparison
        from cctv_dgp_spatial_fit_v40_contract import erode,feature_support,exported_pixel_metrics,detail_metric
        from cctv_dgp_spatial_decoder_v41 import SpatialDGPCandidateV41
        from cctv_dgp_pilot import FixedObservedIdentity,state_hash,grid112
        from cctv_dgp_batchmatched_identity_v26 import objective_terms
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        torch.set_num_threads(4);torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        original,provenance=load_frozen_dgp_restorer(root/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cuda')
        seed=torch.load(root/'untrained_initial_decoder.pth',map_location='cpu',weights_only=True)
        candidate=SpatialDGPCandidateV41(original,seed).eval().requires_grad_(False)
        identity=FixedObservedIdentity(root/'weights/w600k_r50.onnx','cuda').eval().requires_grad_(False)
        parameters=list(candidate.decoder.parameters())
        assert len(parameters)==57 and sum(v.numel() for v in parameters)==17952
        assert [n for n,_ in candidate.decoder.named_parameters()]==[r['name'] for r in p['parameter_layout']]
        counts={'original_DGP':0,'decoder':0,'reference_decoder':0,'recognizer':0}
        for model,key in [(original.net,'original_DGP'),(candidate.decoder,'decoder'),(candidate.reference_decoder,'reference_decoder'),(identity.encoder,'recognizer')]:
            model.register_forward_hook(lambda *_args,key=key:counts.__setitem__(key,counts[key]+1))
        def states():
            return {'original':state_hash(original.net),'decoder':state_hash(candidate.decoder),
                    'reference_decoder':state_hash(candidate.reference_decoder),'recognizer':state_hash(identity)}
        initial=states();assert initial==p['initial_states']
        def frozen():
            value=states()
            for k in ['original','reference_decoder','recognizer']:assert value[k]==initial[k],k
            assert not torch.is_grad_enabled()
            for model in [candidate,identity]:assert all(not v.requires_grad and v.grad is None for v in model.parameters())
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic()-start <= BUDGETS['worker_seconds']
            assert torch.cuda.max_memory_allocated() <= BUDGETS['maximum_allocated_VRAM_bytes']
        def account(path):
            nonlocal bytes_written
            bytes_written+=path.stat().st_size
            assert bytes_written <= BUDGETS['maximum_output_bytes']-16*1024**2, 'Return-storage limit; retain partial review'
        def pixels(path,mode='RGB'):
            with Image.open(path) as image:
                assert image.size==(256,256)
                return np.asarray(image.convert(mode)).copy()
        def canonical(a):
            return torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2,0,1)[None].cuda()
        t=lambda a:torch.from_numpy(np.asarray(a).copy()).cuda()
        ns={'torch':torch,'F':F};names={'blur','high','mean','feature_errors','ssim'}
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in functions}==names
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<unchanged-filters-only>','exec'),ns)
        z=torch.arange(-6,7,dtype=torch.float32);kernel=torch.exp(-.5*(z/2).square());kernel/=kernel.sum()
        head=SimpleNamespace(kernel=kernel.cuda(),reflect_indices=torch.cat((torch.arange(5,-1,-1),torch.arange(256),torch.arange(255,249,-1))).cuda())
        head.blur=MethodType(ns['blur'],head);head.high=MethodType(ns['high'],head);ns['head']=head
        normalizers=tuple(torch.tensor(p['normalizers'][k],dtype=torch.float32,device='cuda') for k in ['feature','interior'])
        shared={};cache_start=time.monotonic()
        for ref in p['references']:
            clock();assert time.monotonic()-cache_start <= BUDGETS['cache_seconds']
            target=pixels(root/ref['target']);mask=pixels(root/ref['observed'],'L')>0
            item={'target8':target,'mask8':mask,'target':canonical(target),'mask':t(mask.astype(np.float32))[None,None],
                  'feature':t(feature_support(mask,ref['landmarks5'][0]).astype(np.float32))[None,None],
                  'interior':t(erode(mask,6).astype(np.float32))[None,None],'valid7':t(erode(mask,3).astype(np.float32))[None,None],
                  'grid':t(grid112(ref['matrix112']))[None]}
            item['truth']=identity.embedding(item['target'],item['mask'],item['grid']).detach().clone()
            shared[ref['id']]=item
        items={}
        for c in p['cases']:
            camera=pixels(root/c['input']);item={**shared[c['reference_id']],'case':c,'camera':camera,'x':canonical(camera)}
            item['degraded_weight']=item['x'].new_tensor([0. if c['profile']=='clear' else 1.25])
            item['clear_weight']=item['x'].new_tensor([1. if c['profile']=='clear' else 0.]);items[c['id']]=item
        keys=['x','base','target','mask','feature','interior','valid7','grid','truth','degraded_weight','clear_weight']
        for begin in range(0,145,5):
            clock();assert time.monotonic()-cache_start <= BUDGETS['cache_seconds']
            chosen=[items[c['id']] for c in p['cases'][begin:begin+5]]
            x=torch.cat([i['x'] for i in chosen]);mask=torch.cat([i['mask'] for i in chosen])
            base=torch.where(mask.bool(),original(x),x).detach().clone()
            parity=candidate(x,mask);assert torch.equal(parity,base), 'Exact initial DGP parity required'
            for slot,item in enumerate(chosen):item['base']=base[slot:slot+1].clone()
        frozen();clock();assert time.monotonic()-cache_start <= BUDGETS['cache_seconds']
        write(out/'preflight.json',{'complete':True,'states':initial,'DGP_provenance':provenance,'initial_exact_case_parity':145,
              'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'grad_enabled':False,'optimizer_updates':0,
              'gradient_queries':0,'new_trained_checkpoint':False,'source_forward_counts':counts.copy()})
        write(out/'cache_receipt.json',{'complete':True,'cases':145,'references':29,'seconds':time.monotonic()-cache_start,
              'cap_seconds':BUDGETS['cache_seconds'],'optimizer_updates':0,'original_reference_and_recognizer_frozen':True})
        def batch(ids):return {key:torch.cat([items[cid][key] for cid in ids]) for key in keys}
        def assign(values):
            for parameter,layout in zip(parameters,p['parameter_layout']):
                parameter.copy_(torch.from_numpy(values[layout['start']:layout['end']].copy()).to(parameter.device).reshape(layout['shape']))
            assert state_hash(candidate.decoder)==parameter_state_hash(values,p['parameter_layout'])
            assert all(not v.requires_grad and v.grad is None for v in parameters)
            progress['parameter_proposals_loaded']+=1
        review_start=time.monotonic()
        for probe_index,probe in enumerate(p['probes']):
            with np.load(root/probe['proposal_arrays'],allow_pickle=False) as data:values=data['values'].copy()
            zero_raw={};zero_condition=None
            for proposal_index,proposal in enumerate(PROPOSALS):
                clock();assert time.monotonic()-review_start <= BUDGETS['review_seconds']
                assign(values[proposal_index]);folder=root/output_prefix(probe['update'],proposal);folder.mkdir(parents=True)
                rows_by_role={}
                for role in ROLES:
                    role_folder=folder/role;role_folder.mkdir();rows=[];ids=role_ids(p,probe,role)
                    for begin in range(0,len(ids),5):
                        clock();assert time.monotonic()-review_start <= BUDGETS['review_seconds']
                        chosen=ids[begin:begin+5];b=batch(chosen);pred=candidate(b['x'],b['mask'])
                        assert not pred.requires_grad and pred.grad_fn is None
                        captures=[]
                        handle=identity.encoder.register_forward_hook(lambda _m,_a,result:captures.append(result.detach()))
                        try:terms=objective_terms(b,pred,identity,ns['mean'],ns['feature_errors'],ns['ssim'],normalizers)
                        finally:handle.remove()
                        assert list(terms)==p['terms'] and len(captures)==1 and captures[0].shape==(2*len(chosen),512)
                        encoded=F.normalize(captures[0].float(),dim=1).cpu().numpy().copy()
                        reference_vectors=encoded[:len(chosen)];raw_vectors=encoded[len(chosen):]
                        term_values=np.stack([v.cpu().numpy() for v in terms.values()],axis=1).astype(np.float32)
                        assert term_values.shape==(len(chosen),7) and np.isfinite(term_values).all()
                        raw=pred.permute(0,2,3,1).cpu().numpy().copy()
                        cameras=[items[cid]['camera'] for cid in chosen];masks=[items[cid]['mask8'] for cid in chosen]
                        pngs=[deliver(a,camera,mask) for a,camera,mask in zip(raw,cameras,masks)]
                        vectors=identity.embedding(torch.cat([canonical(a) for a in pngs]),b['mask'],b['grid']).cpu().numpy().copy()
                        truths=b['truth'].cpu().numpy().copy()
                        for slot,cid in enumerate(chosen):
                            item=items[cid];c=item['case'];a=raw[slot];mask=item['mask8'];camera=item['camera'];target=item['target8'];png=pngs[slot]
                            assert np.array_equal(a[~mask],camera[~mask].astype(np.float32)/np.float32(255))
                            key=(role,cid)
                            if proposal=='zero':zero_raw[key]=a.copy()
                            mean_raw,mean_png,shift=mean_only(a,zero_raw[key],camera,mask)
                            support=feature_support(mask,c['landmarks5_canvas_xy'])
                            raw_value=pixel_metrics(a,target,mask)
                            raw_value.update({'ArcFace_observed_fixed':float(raw_vectors[slot]@truths[slot]),
                                'landmark_high_frequency_MSE':detail_float(a,target,support),
                                'constant_mean_shift_only_MSE':pixel_metrics(mean_raw,target,mask)['MSE']})
                            png_value=exported_pixel_metrics(png,target,mask)
                            png_value.update({'ArcFace_observed_fixed':float(vectors[slot]@truths[slot]),
                                'landmark_high_frequency_MSE':detail_metric(png,target,support),
                                'constant_mean_shift_only_MSE':exported_pixel_metrics(mean_png,target,mask)['MSE']})
                            path=role_folder/(cid+'.npz')
                            extra={'original_rgb':item['base'][0].permute(1,2,0).cpu().numpy().copy()} if proposal=='zero' else {}
                            with path.open('xb') as stream:np.savez_compressed(stream,rgb=a,raw_vector=raw_vectors[slot],
                                PNG_vector=vectors[slot],truth=truths[slot],raw_reference_vector=reference_vectors[slot],terms=term_values[slot],**extra)
                            account(path)
                            for suffix,image in [('.png',png),('_mean_only.png',mean_png)]:
                                path=role_folder/(cid+suffix);Image.fromarray(image).save(path);account(path)
                            rows.append({'id':cid,'source':c['source'],'profile':c['profile'],'raw':raw_value,'PNG':png_value,
                                         'objective_terms':term_values[slot].astype(np.float64).tolist(),
                                         'postclip_mean_RGB_shift':shift.tolist(),
                                         'raw_float32_sha256':__import__('hashlib').sha256(a.tobytes()).hexdigest()})
                        progress['forward_slots']+=len(chosen)
                        assert progress['forward_slots'] <= BUDGETS['maximum_forward_slots']
                    rows_by_role[role]=rows
                groups_by_role={role:{stage:review_groups(rows_by_role[role],stage) for stage in ['raw','PNG']} for role in ROLES}
                comparisons=None
                if zero_condition is not None:
                    comparisons={role:{stage:finite_comparison(zero_condition['groups'][role][stage],groups_by_role[role][stage])
                                 for stage in ['raw','PNG']} for role in ROLES}
                condition={'complete':True,'update':probe['update'],'proposal':proposal,'rows':rows_by_role,'groups':groups_by_role,
                    'decoder_state':state_hash(candidate.decoder),'parameter_vector_sha256':probe['proposal_flat_float32_sha256'][proposal_index],
                    'objective_term_means':{role:np.mean([r['objective_terms'] for r in rows],axis=0).tolist() for role,rows in rows_by_role.items()},
                    'comparison_to_same_before_state':comparisons,'first_order_proposal_arithmetic':probe['arithmetic'],
                    'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'app_promotion':False}
                write(folder/'metrics.json',condition);account(folder/'metrics.json')
                if proposal=='zero':zero_condition=condition
                completed.append({'update':probe['update'],'proposal':proposal,'metrics_sha256':sha(folder/'metrics.json'),
                                  'comparison_to_same_before_state':comparisons})
                progress['completed_conditions']+=1;frozen();clock()
                assert time.monotonic()-review_start <= BUDGETS['review_seconds']
                print({'actual_step_probe':probe['update'],'proposal':proposal,'forward_slots':progress['forward_slots'],
                       'of':3150,'optimizer_updates':0,'seconds':time.monotonic()-start},flush=True)
            if probe_index==0:
                elapsed=time.monotonic()-review_start
                projected=elapsed*10*BUDGETS['timing_safety_factor']
                write(out/'timing_projection.json',{'completed_probe_states':1,'completed_forward_slots':315,
                    'seconds':elapsed,'projected_review_seconds':projected,'cap_seconds':BUDGETS['review_seconds'],
                    'safety_factor':BUDGETS['timing_safety_factor'],'remaining_probe_states':9})
                assert projected <= BUDGETS['review_seconds'], 'Measured finite-review timing exceeds1500s; retain stop'
        assert progress['forward_slots']==3150 and progress['completed_conditions']==30
        frozen();candidate.decoder.load_state_dict(seed,strict=True);assert states()==initial
        write(out/'results.json',{'complete':True,'protocol_sha256':pin,'progress':progress,'conditions':completed,
              'final_restored_states':states(),'original_initial_reference_and_recognizer_unchanged':True,
              'source_forward_counts':counts,'allocated_peak_VRAM_bytes':torch.cuda.max_memory_allocated(),
              'seconds':time.monotonic()-start,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,
              'native_or_DEV_or_reserved_final_used':False,'app_promotion':False,'full_TRAIN_capacity_pass':False,
              'diagnostic_not_model_qualification':True})
        print({'complete':True,'forward_slots':progress['forward_slots'],'optimizer_updates':0,'seconds':time.monotonic()-start},flush=True)
    except BaseException as error:
        original_traceback=traceback.format_exc();restoration={'attempted':False,'decoder_restored':False}
        if candidate is not None and seed is not None:
            restoration['attempted']=True
            try:
                candidate.decoder.load_state_dict(seed,strict=True)
                restoration['decoder_restored']=state_hash(candidate.decoder)==p['initial_states']['decoder']
            except BaseException as restore_error:restoration['error']=str(restore_error)
        failure={'complete':False,'error':str(error),'type':type(error).__name__,'traceback':original_traceback,'restoration':restoration,
                 'progress':progress,'completed_conditions':completed,'seconds':time.monotonic()-start,
                 'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'app_promotion':False}
        if not (out/'failure.json').exists():write(out/'failure.json',failure)
        raise
    finally:signal.alarm(0)


def export(root,p,pin):
    vm_scope(root);started=time.monotonic();signal.signal(signal.SIGALRM,timeout_signal);signal.alarm(BUDGETS['export_seconds'])
    try:
        archive=Path.home()/(STEM+'-results.tar.gz');checksum=Path(str(archive)+'.sha256');receipt=Path.home()/(STEM+'-export.json')
        assert all(not q.exists() for q in [archive,checksum,receipt,root/'export_manifest.json']), 'Retain every previous export'
        assert (root/'outputs').is_dir() and ((root/'outputs/results.json').exists() or (root/'outputs/failure.json').exists())
        allowed=allowed_return_names(p)-{'export_manifest.json'};files={};total=0
        for name in sorted(allowed):
            path=root/name
            if path.is_file():
                assert not path.is_symlink() and path.resolve().is_relative_to(root)
                files[name]=sha(path);total+=path.stat().st_size
                assert time.monotonic()-started <= BUDGETS['export_seconds']
        assert total <= BUDGETS['maximum_output_bytes']-8*1024**2
        write(root/'export_manifest.json',{'complete':True,'protocol_sha256':pin,'files_sha256':files,'optimizer_updates':0,
              'gradient_queries':0,'new_trained_checkpoint':False,'diagnostic_completed':(root/'outputs/results.json').exists()})
        with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
            for name in sorted(set(files)|{'export_manifest.json'}):
                assert time.monotonic()-started <= BUDGETS['export_seconds']
                tar.add(root/name,arcname=RETURN_PREFIX+'/'+name,recursive=False)
        digest=sha(archive)
        with checksum.open('x',encoding='utf-8',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
        value={'complete':True,'archive_sha256':digest,'bytes':archive.stat().st_size,'seconds':time.monotonic()-started,
               'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'training_success_not_implied':True,
               'run_results_present':(root/'outputs/results.json').exists(),'failure_present':(root/'outputs/failure.json').exists()}
        write(receipt,value);print(value,flush=True)
    finally:signal.alarm(0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    flags=parser.add_mutually_exclusive_group(required=True);flags.add_argument('--verify-transfer',action='store_true');flags.add_argument('--review',action='store_true');flags.add_argument('--export',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if not args.verify_transfer:vm_scope(root)
    p=verify(root,args.protocol_sha)
    if args.verify_transfer:print({'complete':True,'cases':145,'proposals':30,'optimizer_updates':0,'gradient_queries':0,'neural_calls':0});return
    export(root,p,args.protocol_sha) if args.export else review(root,p,args.protocol_sha)


if __name__=='__main__':main()

"""Finite manual-L4 saved-state gradient diagnosis. No optimizer or weight update."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import signal
import shutil
import sys
import tarfile
import time
import traceback


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root,pin):
    assert sha(root/'protocol.json')==pin,'Diagnostic protocol changed'
    p=read(root/'protocol.json');assert p['format']=='own-DGP-V26-saved-state-gradient-diagnostic-v1'
    assert p['snapshots']==[0,50] and p['optimizer_updates']==0 and p['head_batches']==20 and p['component_gradient_calls']==140
    for name,digest in p['assets_sha256'].items():
        path=(root/name).resolve();assert path.is_relative_to(root) and sha(path)==digest,name
    return p


def parent_check(parent,p):
    assert sha(parent/'protocol.json')==p['closed_V26_protocol_sha256'],'Original V26 protocol changed'
    base=read(parent/'protocol.json');assert base['format']=='dgp-spatial-batchmatched-identity-capacity-v26'
    for name,digest in base['assets_sha256'].items():
        path=(parent/name).resolve();assert path.is_relative_to(parent) and sha(path)==digest,'Original asset changed: '+name
    for name,digest in p['closed_V26_inputs_sha256'].items():
        path=(parent/name).resolve();assert path.is_relative_to(parent) and sha(path)==digest,'Original saved input changed: '+name
    failure=read(parent/'outputs/failure.json');early=read(parent/'outputs/early_structure_stop.json')
    assert failure['updates']==50 and failure['backwards']==51 and failure['cause']=='No one-percent early structural gain; retain stop'
    assert early['pass'] is False and not (parent/'outputs/results.json').exists(),'Retain the V26 failure'
    proof=read(parent/p['identity_proof_file'])
    assert proof['complete'] and proof['cases']==50 and proof['gradient_calls']==20
    assert proof['batchmatched_component_value']==proof['batchmatched_component_gradient_norm']==0
    assert len(proof['rows'])==10 and all(r['all26_matched_gradient_tensors_exactly_zero'] and r['exact_reference_prediction_cosines'] for r in proof['rows'])
    cache=read(parent/'outputs/frozen_DGP_features.json')
    assert len(cache['files_sha256'])==250 and cache['optimizer_constructed'] is False
    for name,digest in cache['files_sha256'].items():assert sha(parent/'outputs/frozen_DGP_features'/name)==digest,name
    return base,cache


def guarded_namespace(parent):
    """Compile only fixed guard/functions/class AST from verified local V26 assets."""
    import os
    import platform
    import subprocess
    import urllib.request
    source=ast.parse((parent/'scripts/cctv_dgp_batchmatched_identity_v26_vm.py').read_text())
    guard=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='require_vm')
    namespace={'sys':sys,'platform':platform,'Path':Path,'urllib':__import__('urllib'),'os':os,'subprocess':subprocess}
    exec(compile(ast.Module(body=[guard],type_ignores=[]),'<pinned-V26-existing-L4-guard>','exec'),namespace)
    return namespace


def run(root,parent,p,pin):
    assert not (root/'outputs').exists(),'Preserve prior/partial diagnostic'
    # The host test runs before any model import, gradient call or result directory.
    assert sys.platform=='linux', 'Existing Linux VM only'
    assert root.is_relative_to((Path.home()/'forensic-dgp').resolve()) and parent.is_relative_to((Path.home()/'forensic-dgp').resolve()),'~/forensic-dgp only'
    start=time.monotonic();progress={'head_batches':0,'component_gradient_calls':0,'recognizer_forwards':0,'optimizer_updates':0}
    out=root/'outputs';out.mkdir()
    try:
        assert shutil.disk_usage(root).free>=1024**3,'Diagnostic requires1GiB free disk'
        base,cache=parent_check(parent,p)
        guard=guarded_namespace(parent)['require_vm'];guard(root,idle=True)
        before_parent={name:sha(parent/name) for name in p['closed_V26_inputs_sha256']}
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        torch.set_num_threads(4);torch.manual_seed(20261005);torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic()-start<420,'Diagnostic worker cap420 seconds'
            assert torch.cuda.max_memory_allocated()<=20*1024**3,'Diagnostic allocated VRAM cap20GiB'
        sys.path.insert(0,str(parent))
        from PIL import Image
        from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash
        namespace={'torch':torch,'nn':nn,'F':F}
        headtree=ast.parse((parent/'cctv_dgp_spatial_features_v25.py').read_text())
        cls=next(n for n in headtree.body if isinstance(n,ast.ClassDef) and n.name=='SpatialFeatureHead')
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'<pinned-V26-head-only>','exec'),namespace)
        head=namespace['SpatialFeatureHead']().cuda().eval().requires_grad_(True)
        assert sum(v.numel() for v in head.parameters())==53781
        identity=FixedObservedIdentity(parent/'weights/w600k_r50.onnx','cuda')
        def count_identity(*_):progress['recognizer_forwards']+=1
        identity.encoder.register_forward_hook(count_identity);identity_before=state_hash(identity)
        assert identity_before==p['frozen_recognizer_state']
        def rgb(name):
            with Image.open(parent/name) as im:a=np.asarray(im).copy()
            assert a.shape==(256,256,3) and a.dtype==np.uint8
            return a
        def tensor(a):return torch.from_numpy(np.asarray(a).copy()).cuda()
        def image(a):return tensor(a).permute(2,0,1)[None]
        refs={r['id']:r for r in base['references']};items=[]
        for case in base['cases']:
            clock();camera,target=rgb(case['input']),rgb(case['target'])
            with Image.open(parent/case['observed']) as im:mask=np.asarray(im).copy()>0
            raw=np.load(parent/case['raw_dgp'],allow_pickle=False)
            assert raw.shape==camera.shape and raw.dtype==np.float32 and np.isfinite(raw).all()
            # Independent square erosion; match the fixed observed supports.
            def erode(radius):
                size=2*radius+1;padded=np.pad(mask.astype(np.int64),radius)
                summed=np.pad(padded.cumsum(0).cumsum(1),((1,0),(1,0)))
                return summed[size:,size:]-summed[:-size,size:]-summed[size:,:-size]+summed[:-size,:-size]==size*size
            interior=erode(6);feature=np.zeros((256,256),bool)
            for point in case['landmarks5_canvas_xy']:
                xx,yy=np.floor(point).astype(int);feature[max(0,yy-12):min(256,yy+12),max(0,xx-12):min(256,xx+12)]=True
            feature&=interior;assert feature.any()
            maps=[]
            for index,(channels,size) in enumerate(zip([64,128,128,128,128],[128,64,32,16,8])):
                a=np.load(parent/'outputs/frozen_DGP_features'/(case['id']+'_fpn'+str(index)+'.npy'),allow_pickle=False)
                assert a.shape==(channels,size,size) and a.dtype==np.float32 and np.isfinite(a).all()
                maps.append(tensor(a)[None]);assert not maps[-1].requires_grad and not torch.is_inference(maps[-1])
            items.append({'id':case['id'],'x':image(camera.astype(np.float32)/np.float32(255)),'base':image(raw),
                'target':image(target.astype(np.float32)/np.float32(255)),'mask':tensor(mask.astype(np.float32))[None,None],
                'feature':tensor(feature.astype(np.float32))[None,None],'interior':tensor(interior.astype(np.float32))[None,None],
                'valid7':tensor(erode(3).astype(np.float32))[None,None],
                'grid':tensor(grid112(refs[case['source_person_or_reference']]['matrix112']))[None],
                'degraded_weight':tensor(np.array([0. if case['profile']=='clear' else 1.25],np.float32)),
                'clear_weight':tensor(np.array([1. if case['profile']=='clear' else 0.],np.float32)),'fpn':tuple(maps)})
        with torch.no_grad():
            for item in items:
                item['truth']=identity.embedding(item['target'],item['mask'],item['grid'])
        assert progress['recognizer_forwards']==50
        cohort=read(parent/'outputs/cohort_loss_setup.json')
        normalizers=(tensor(np.array(cohort['feature_normalizer'],np.float32)),tensor(np.array(cohort['interior_normalizer'],np.float32)))
        namespace.update({'head':head,'identity':identity,'normalizers':normalizers})
        source=ast.parse((parent/'scripts/cctv_dgp_batchmatched_identity_v26_vm.py').read_text())
        originalrun=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        functions=[next(n for n in originalrun.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['mean','feature_errors','ssim']]
        helper=ast.parse((parent/'cctv_dgp_degraded_objective_v24.py').read_text())
        functions += [next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='assemble_terms')]
        corrected=ast.parse((parent/'cctv_dgp_batchmatched_identity_v26.py').read_text())
        functions += [next(n for n in corrected.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['batchmatched_scores','objective_terms']]
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned-V26-loss-functions-only>','exec'),namespace)
        keys=['x','base','mask','target','feature','interior','valid7','grid','truth','degraded_weight','clear_weight']
        def batch(ids):
            b={k:torch.cat([items[i][k] for i in ids]) for k in keys}
            b['fpn']=tuple(torch.cat([items[i]['fpn'][scale] for i in ids]) for scale in range(5));return b
        named=list(head.named_parameters());parameters=[v for _,v in named]
        layout=[];offset=0
        for name,value in named:
            layout.append({'name':name,'shape':list(value.shape),'start':offset,'end':offset+value.numel()});offset+=value.numel()
        assert offset==53781
        term_names=p['terms'];rows=[]
        for update in [0,50]:
            clock();state=torch.load(parent/'outputs'/('update'+str(update))/'head.pth',map_location='cpu',weights_only=True)
            head.load_state_dict(state,strict=True);initial=state_hash(head)
            assert initial==p['head_states'][str(update)]
            if update==0:assert all(torch.count_nonzero(state[k])==0 for k in ['tail.weight','tail.bias','direct.weight','direct.bias'])
            gradients=torch.zeros((7,53781),dtype=torch.float64,device='cuda');values=[0.]*7;raw_errors=[];batch_rows=[]
            for begin in range(0,50,5):
                clock();b=batch(list(range(begin,begin+5)))
                pred=head(b['x'],b['base'],b['mask'],b['fpn']);progress['head_batches']+=1
                for slot,index in enumerate(range(begin,begin+5)):
                    saved=np.load(parent/'outputs'/('update'+str(update))/(items[index]['id']+'.npy'),allow_pickle=False)
                    error=float(np.abs(pred[slot].detach().cpu().permute(1,2,0).numpy()-saved).max())
                    assert error<=2e-6,'Saved-state diagnostic batch raw parity failed: '+items[index]['id'];raw_errors.append(error)
                terms=namespace['objective_terms'](b,pred,identity,namespace['mean'],namespace['feature_errors'],namespace['ssim'],normalizers)
                assert list(terms)==term_names
                batch_values=[];batch_norms=[]
                for i,name in enumerate(term_names):
                    clock();scalar=terms[name].mean()/10
                    values[i]+=float(scalar.detach())
                    pieces=torch.autograd.grad(scalar,parameters,retain_graph=i<6,create_graph=False,allow_unused=False)
                    progress['component_gradient_calls']+=1
                    assert len(pieces)==len(parameters) and all(torch.isfinite(g).all() for g in pieces)
                    if update==0 and name=='ArcFace_regression':
                        assert float(scalar.detach())==0 and all(torch.count_nonzero(g)==0 for g in pieces),'Initial corrected identity value/all26 gradients must stay exactly zero'
                    component=torch.cat([g.detach().reshape(-1).double() for g in pieces])
                    gradients[i]+=component
                    batch_values.append(float(scalar.detach()));batch_norms.append(float(torch.linalg.vector_norm(component)))
                batch_rows.append({'ids':[items[i]['id'] for i in range(begin,begin+5)],
                    'cohort_weighted_component_values':batch_values,'cohort_weighted_component_gradient_norms':batch_norms})
            array=gradients.cpu().numpy().copy();np.save(out/('gradient_components_update'+str(update)+'.npy'),array,allow_pickle=False)
            norms=np.linalg.norm(array,axis=1);gram=array@array.T
            denominator=norms[:,None]*norms[None,:]
            cosine=np.divide(gram,denominator,out=np.zeros_like(gram),where=denominator>0)
            total=array.sum(0);partitions={}
            for parameter in layout:
                block=array[:,parameter['start']:parameter['end']]
                partitions[parameter['name']]={'component_norms':np.linalg.norm(block,axis=1).tolist(),'total_norm':float(np.linalg.norm(block.sum(0)))}
            rows.append({'update':update,'head_state_before_after':initial,'objective':sum(values),'component_values':values,
                'component_norms':norms.tolist(),'component_gram':gram.tolist(),'component_cosines':cosine.tolist(),
                'total_gradient_norm':float(np.linalg.norm(total)),'per_parameter_gradients':partitions,
                'batches':batch_rows,'maximum_saved_raw_difference':max(raw_errors),
                'gradient_array_sha256':sha(out/('gradient_components_update'+str(update)+'.npy'))})
            assert state_hash(head)==initial and state_hash(identity)==identity_before
            assert all(v.grad is None for v in parameters) and all(not v.requires_grad and v.grad is None for v in identity.parameters())
            assert all(not value.requires_grad and value.grad is None for item in items for value in item['fpn'])
            print(json.dumps({'snapshot':update,'objective':sum(values),'component_norms':norms.tolist(),'seconds':time.monotonic()-start}),flush=True)
        assert progress=={'head_batches':20,'component_gradient_calls':140,'recognizer_forwards':70,'optimizer_updates':0}
        assert all(sha(parent/name)==digest for name,digest in before_parent.items()),'Original V26 evidence changed'
        parent_check(parent,p)
        clock()
        write(out/'results.json',{'complete':True,'protocol_sha256':pin,'seconds':time.monotonic()-start,
            'scope':'Fixed saved-state loss gradients, no optimization trajectory reconstruction or quality acceptance',
            'terms':term_names,'parameter_layout':layout,'snapshots':rows,**progress,'DGP_forwards':0,
            'frozen_recognizer_state':identity_before,'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),
            'original_evidence_unchanged':True,'independent_audit_pending':True,'new_checkpoint_created':False,
            'automatic_follow_on':False,'app_promotion':False,'goal_complete':False})
    except BaseException as exc:
        write(out/'failure.json',{'complete':False,'protocol_sha256':pin,'seconds':time.monotonic()-start,**progress,
            'cause':str(exc),'traceback':traceback.format_exc(),'resume_permitted':False,'new_checkpoint_created':False,'app_promotion':False})
        raise


def export(root,p,pin):
    assert sys.platform=='linux' and root.is_relative_to((Path.home()/'forensic-dgp').resolve())
    start=time.monotonic();name='cctv-dgp-v26-gradient-diagnostic-v1-results.tar.gz';destination=Path.home()/name
    assert not destination.exists() and not Path(str(destination)+'.sha256').exists(),'Preserve prior diagnostic export'
    files=[root/'protocol.json',root/'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py',root/'scripts/run_gradient.sh']
    files += [f for f in (root/'outputs').rglob('*') if f.is_file()] if (root/'outputs').exists() else []
    files += [root/n for n in ['trainer.log','trainer_exit_code.txt','supervisor_receipt.json'] if (root/n).exists()]
    write(root/'export_manifest.json',{'complete':True,'protocol_sha256':pin,'files_sha256':{f.relative_to(root).as_posix():sha(f) for f in files},'quality_acceptance_not_implied':True})
    files.append(root/'export_manifest.json')
    with tarfile.open(destination,'x:gz',compresslevel=3) as tar:
        for f in files:
            assert time.monotonic()-start<30,'Diagnostic export cap30 seconds'
            tar.add(f,arcname='cctv_dgp_v26_gradient_diagnostic_v1_return/'+f.relative_to(root).as_posix(),recursive=False)
    digest=sha(destination)
    with Path(str(destination)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+name+'\n')
    receipt={'complete':True,'archive_sha256':digest,'bytes':destination.stat().st_size,'seconds':time.monotonic()-start,
        'training_success_not_implied':True,'optimizer_updates':0,'run_results_present':(root/'outputs/results.json').exists(),'failure_present':(root/'outputs/failure.json').exists()}
    write(Path.home()/'cctv-dgp-v26-gradient-diagnostic-v1-export.json',receipt);print(json.dumps(receipt))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--protocol-sha',required=True)
    parser.add_argument('--parent',type=Path,default=Path.home()/'forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26')
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ['verify-transfer','run','export','record-supervision']:modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--supervisor-start',type=float);parser.add_argument('--supervisor-end',type=float);parser.add_argument('--trainer-exit',type=int)
    a=parser.parse_args();root=a.root.resolve();p=verify(root,a.protocol_sha)
    if a.verify_transfer:print(json.dumps({'complete':True,'assets':len(p['assets_sha256']),'optimizer_updates':0,'neural_or_gradient_calls':0}));return
    if a.record_supervision:
        assert sys.platform=='linux' and root.is_relative_to((Path.home()/'forensic-dgp').resolve())
        assert all(v is not None and math.isfinite(v) for v in [a.supervisor_start,a.supervisor_end]) and a.supervisor_end>=a.supervisor_start and a.trainer_exit is not None
        elapsed=a.supervisor_end-a.supervisor_start
        write(root/'supervisor_receipt.json',{'complete':True,'protocol_sha256':a.protocol_sha,'seconds':elapsed,'cap_seconds':480,'kill_grace_seconds':30,
            'within_external_bound':elapsed<=510,'trainer_exit_code':a.trainer_exit,'optimizer_updates':0,'quality_acceptance_not_implied':True});return
    if a.export:export(root,p,a.protocol_sha);return
    if sys.platform=='linux':
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Diagnostic worker cap420 seconds')));signal.alarm(420)
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Diagnostic external supervisor deadline')))
    run(root,a.parent.resolve(),p,a.protocol_sha)


if __name__=='__main__':main()

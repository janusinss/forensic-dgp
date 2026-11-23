"""Manual-L4 finite CodeFormer completion pilot; source-only until packet approval.

--verify-transfer and --help never import Torch. --run owns an1800-second child.
No checkpoint is overwritten and no historical pilot is automatically resumed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

VM_NAME="cctv_completion_codeformer_finetune_v1_vm"
WEIGHT_SHA="b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
KINDS=("lower_face","eye_band","image_left_lens","cheek_object")
CONTRACT={"cycles":32,"microbatches_per_route":4,"batch_size":1,"optimizer_steps_per_cycle":2,
          "optimizer_steps_max":64,"backward_calls_max":256,"forward_slots_max":600,"teacher_references":40,
          "fit_references":32,"monitor_references":8,"fit_cases":128,"monitor_cases":32,"development_cases":32,
          "snapshots":[0,8,16,32],"worker_seconds":1800,"cooperative_stop_seconds":1680,"output_bytes":1536*1024**2,"minimum_free_bytes":4*1024**3,
          "stop_free_bytes":2*1024**3,"CUDA_allocated_bytes_cap":20*1024**3,
          "seed":240,"precision":"FP32_no_autocast_no_TF32","codebook_size":512,"fusion_sizes":["32","64","128"],
          "Adam":{"lr_code":1e-5,"lr_fusion":1e-5,"betas":[0.9,0.99],"eps":1e-8,"weight_decay":0.,"clip_L2":1.},
          "code_loss":"CE_all256_clean_nearest_codes + MSE_clean_HQ512_encoder_latent",
          "fusion_loss":"unclamped_mask_RGB512_L1 + inner_boundary_RGB512_L1_kernel7",
          "selection_min_macro_PNG_hole_gain":0.01,"selection_seam":"nonworse_raw_and_PNG_macro_and_each_geometry",
          "selection_hole":"nonworse_raw_and_PNG_each_geometry","catastrophic_regression_fraction":0.10,
          "teacher_limitation":"same original frozen inpainting encoder/codebook, not independent clean prior",
          "original_VAL_used":False,"reserved_FINAL_used":False,"automatic_launch":False}
SOURCE_MODULES={"third_party":"third_party/__init__.py","third_party.codeformer":"third_party/codeformer/__init__.py",
                "third_party.codeformer.registry":"third_party/codeformer/registry.py","third_party.codeformer.vqgan_arch":"third_party/codeformer/vqgan_arch.py",
                "third_party.codeformer.codeformer_arch":"third_party/codeformer/codeformer_arch.py","completion":"completion.py","pretrained_completion":"pretrained_completion.py"}

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
def member(root,relative):
    need(isinstance(relative,str) and relative and '\\' not in relative,'Invalid relative packet member')
    name=Path(relative);need(not name.is_absolute() and '..' not in name.parts,'Packet traversal')
    path=root/name
    for part in [path,*path.parents]:
        need(not part.is_symlink(),'Symlink member refused')
        if part==root:break
    need(path.resolve().is_relative_to(root),'Member escaped packet')
    return path

def training_schedule(fit_cases):
    by={(r['reference_id'],r['mask_kind']):r['id'] for r in fit_cases}
    refs=sorted({r['reference_id'] for r in fit_cases});random.Random(CONTRACT['seed']).shuffle(refs)
    return [[by[(refs[(cycle+8*j)%32],kind)] for j,kind in enumerate(KINDS)] for cycle in range(32)]

def verify(root,expected):
    root=root.resolve(strict=True);need(sha(root/'protocol.json')==expected,'Protocol SHA differs')
    p=read(root/'protocol.json');need(p['format']=='completion-codeformer-finetune-v1' and p['contract']==CONTRACT,'Pilot contract differs')
    need(len(p['references'])==40 and len(p['fit_cases'])==128 and len(p['monitor_cases'])==32 and len(p['development_cases'])==32,'Bounded data count differs')
    refs={r['id']:r for r in p['references']};need(len(refs)==40,'Duplicate reference ID')
    for row in p['references']+p['fit_cases']+p['monitor_cases']+p['development_cases']:
        need(isinstance(row['id'],str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',row['id']),'Unsafe output identifier')
    fit={r['id'] for r in p['references'] if r['role']=='fit'};monitor={r['id'] for r in p['references'] if r['role']=='monitor'}
    need(len(fit)==32 and len(monitor)==8 and not fit&monitor,'FIT/MONITOR overlap')
    need(all(r.get('original_role')=='train' and r.get('legacy_completion_split')=='train' for r in refs.values()),'Only audited original TRAIN-role sources allowed')
    for role,rows,selected in [('fit',p['fit_cases'],fit),('monitor',p['monitor_cases'],monitor)]:
        need({(r['reference_id'],r['mask_kind']) for r in rows}=={(ref,kind) for ref in selected for kind in KINDS},role+' mask geometry coverage differs')
        need(len({r['id'] for r in rows})==len(rows),'Duplicate case ID')
    files={r['path']:r for r in p['files']};need(len(files)==len(p['files']),'Duplicate pinned member')
    for row in files.values():
        file=member(root,row['path']);need(file.is_file() and file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],'Packet drift: '+row['path'])
    for relative in SOURCE_MODULES.values():need(relative in files,'Unpinned imported source')
    need('scripts/cctv_completion_codeformer_finetune_v1_vm.py' in files,'Unpinned worker')
    for row in refs.values():
        need(row['clean256'] in files and row['clean512'] in files,'Unpinned clean target')
    for row in p['fit_cases']+p['monitor_cases']:need(row['mask'] in files,'Unpinned mask')
    for row in p['development_cases']:need(row['mask'] in files and row['input256'] in files,'Unpinned fixed public development input/mask')
    fit_hash={files[refs[i]['clean256']]['sha256'] for i in fit};monitor_hash={files[refs[i]['clean256']]['sha256'] for i in monitor}
    need(not fit_hash&monitor_hash,'FIT/MONITOR duplicate clean256 byte content')
    checkpoint=(root/p['checkpoint']['path']).resolve(strict=True)
    need(checkpoint.name=='codeformer_inpainting.pth' and not checkpoint.is_symlink(),'Original inpainting checkpoint required')
    if not checkpoint.is_relative_to(root):need(checkpoint.is_relative_to((Path.home()/'forensic-dgp').resolve()),'Shared weights must stay inside forensic-dgp')
    need(p['checkpoint']['bytes']==370786611 and p['checkpoint']['sha256']==WEIGHT_SHA and checkpoint.stat().st_size==370786611 and sha(checkpoint)==WEIGHT_SHA,'Original weight bytes/SHA differ')
    if 'schedule' in p:need(p['schedule']==training_schedule(p['fit_cases']),'Deterministic microbatch schedule changed')
    return p,checkpoint

def require_vm(root):
    need(os.name=='posix' and root.resolve()==(Path.home()/'forensic-dgp'/VM_NAME).resolve(),'Only named L4 VM pilot root permitted')
    req=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(req,timeout=4) as response:machine=response.read(4096).decode().strip()
    need(machine.rsplit('/',1)[-1]=='g2-standard-4','Expected g2-standard-4')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
    need(len(gpu)==1 and 'L4' in gpu[0],'Expected one NVIDIA L4')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
    need(not active,'Competing GPU task; do not launch')
    return {'machine_type':machine,'GPU':gpu[0]}

def worker(root,p,checkpoint,expected):
    hardware=require_vm(root)
    def cooperative_termination(signum,frame):raise KeyboardInterrupt('Supervisor requested termination; retain current partial state')
    prior_termination=signal.signal(signal.SIGTERM,cooperative_termination)
    import shutil
    need(shutil.disk_usage(root).free>=CONTRACT['minimum_free_bytes'],'Need4GiB free before pilot')
    out=root/'outputs';need(not out.exists(),'Never overwrite/resume this finite historical pilot');out.mkdir()
    started=time.monotonic();counts={'forward_slots':0,'backward_calls':0,'code_optimizer_steps':0,'fusion_optimizer_steps':0,'empty_bypasses':0}
    typed={'teacher_encoder':0,'teacher_quantizer':0,'parity':0,'fit_code':0,'fit_RGB':0,'monitor':0,'public_development':0}
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    need(torch.cuda.is_available(),'CUDA unavailable')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);random.seed(240);np.random.seed(240);torch.manual_seed(240);torch.cuda.manual_seed_all(240)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(root))
    for module,relative in SOURCE_MODULES.items():
        loaded=importlib.import_module(module);need(Path(loaded.__file__).resolve()==(root/relative).resolve(),'Imported source escaped packet: '+module)
    CodeFormer=importlib.import_module('third_party.codeformer.codeformer_arch').CodeFormer
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(isinstance(loaded,dict) and isinstance(loaded.get('params_ema'),dict),'Expected original params_ema')
    net=CodeFormer(dim_embd=512,codebook_size=512,n_head=8,n_layers=9,connect_list=['32','64','128'])
    net.load_state_dict(loaded['params_ema'],strict=True);del loaded
    net=net.to('cuda',dtype=torch.float32).eval().requires_grad_(False)
    def guard(*,terminal_save=False):
        limit=1800 if terminal_save else CONTRACT['cooperative_stop_seconds']
        need(time.monotonic()-started<limit,'Finite pilot stop; reserve120seconds for terminal state')
        need(counts['forward_slots']<=600 and counts['backward_calls']<=256,'Finite neural/backward bound')
        need(counts['code_optimizer_steps']<=32 and counts['fusion_optimizer_steps']<=32,'Finite optimizer bound')
        need(torch.cuda.memory_allocated()<=CONTRACT['CUDA_allocated_bytes_cap'],'CUDA memory cap')
        need(shutil.disk_usage(root).free>=CONTRACT['stop_free_bytes'],'Reserve free disk stop')
        need(sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<=CONTRACT['output_bytes'],'Output byte cap')
    def forward(kind,call):
        guard();counts['forward_slots']+=1;typed[kind]+=1;guard();return call()
    def digest_state(prefixes=None):
        digest=hashlib.sha256()
        for name,value in sorted(net.state_dict().items()):
            if prefixes and not any(name.startswith(prefix) for prefix in prefixes):continue
            array=value.detach().cpu().contiguous();need(torch.isfinite(array).all().item(),'Nonfinite model state')
            digest.update(name.encode());digest.update(str(tuple(array.shape)).encode());digest.update(str(array.dtype).encode());digest.update(array.numpy().tobytes())
        return digest.hexdigest()
    base_state=digest_state();frozen_state=digest_state(('quantize.','generator.'))
    def image(path,size):
        with Image.open(root/path) as im:
            need(im.format=='PNG' and im.size==(size,size) and im.mode=='RGB','Exact prepared RGB target/input geometry differs')
            a=np.array(im,copy=True)
        return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32)/255
    def mask(path):
        with Image.open(root/path) as im:
            need(im.format=='PNG' and im.size==(256,256) and im.mode=='L','Exact L256 binary mask required')
            a=np.array(im,copy=True);need(np.isin(a,[0,255]).all(),'Mask not binary')
        return torch.from_numpy(a).to('cuda',torch.float32)[None,None]/255
    def app_input(rgb,m):
        visible=1-m;support=F.interpolate(visible,size=(512,512),mode='bilinear',align_corners=False)
        x=F.interpolate(rgb*visible,size=(512,512),mode='bilinear',align_corners=False)/support.clamp_min(1e-8)
        m512=F.interpolate(m,size=(512,512),mode='nearest')
        return (x*(1-m512)+m512)*2-1,m512
    def delivered(rgb,m,raw):
        generated=F.interpolate(((raw+1)/2).clamp(0,1),size=(256,256),mode='bilinear',align_corners=False)
        return rgb*(1-m)+generated*m
    def save_pixels(path,value):
        a=np.rint(value.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        Image.fromarray(a).save(path);return a
    def boundary(m):return (m-(1-F.max_pool2d(1-m,7,1,3))).clamp(0,1)
    references={r['id']:r for r in p['references']};clean={};teacher={}
    fit_cases={r['id']:r for r in p['fit_cases']};schedule=training_schedule(p['fit_cases'])
    code_groups={'encoder':list(net.encoder.parameters()),'position':[net.position_emb],'feature':list(net.feat_emb.parameters()),'transformer':list(net.ft_layers.parameters()),'prediction':list(net.idx_pred_layer.parameters())}
    fusion_groups={size:list(net.fuse_convs_dict[size].parameters()) for size in ('32','64','128')}
    code_parameters=[param for group in code_groups.values() for param in group]
    fusion_parameters=[param for group in fusion_groups.values() for param in group]
    parameter_names={id(parameter):name for name,parameter in net.named_parameters()}
    optimizer_parameter_names={'code':[parameter_names[id(parameter)] for parameter in code_parameters],'fusion':[parameter_names[id(parameter)] for parameter in fusion_parameters]}
    code_optimizer=torch.optim.Adam(code_parameters,lr=1e-5,betas=(.9,.99),eps=1e-8,weight_decay=0.)
    fusion_optimizer=torch.optim.Adam(fusion_parameters,lr=1e-5,betas=(.9,.99),eps=1e-8,weight_decay=0.)
    def active(groups):
        net.requires_grad_(False)
        for parameters in groups.values():
            for parameter in parameters:parameter.requires_grad_(True)
    def gradient_receipt(groups):
        receipt={}
        for name,parameters in groups.items():
            need(all(param.grad is not None for param in parameters),'Disconnected intended group '+name)
            need(all(torch.isfinite(param.grad).all().item() for param in parameters),'Nonfinite gradients '+name)
            norm=math.sqrt(sum(float(param.grad.detach().double().square().sum().item()) for param in parameters))
            receipt[name]={'L2':norm,'tensors':len(parameters),'finite':True,'nonzero':norm>0}
        return receipt
    snapshots=[];terminal_cycle=0;stop_reason=None;parity=[];fit_log=[]
    progress={'cycle':0,'route':'initialization','microbatch':0,'phase':'before_first_update'}
    model_forwards={'calls':0}
    def model_hook(module,args):model_forwards['calls']+=1
    forward_hook=net.register_forward_pre_hook(model_hook)
    def save_state(cycle,reason):
        need(digest_state(('quantize.','generator.'))==frozen_state,'Frozen generator/codebook changed')
        trainable={name:value.detach().cpu() for name,value in net.state_dict().items() if not name.startswith(('quantize.','generator.'))}
        snapshot={'format':'completion-codeformer-trainable-delta-resume-v1','base_checkpoint_sha256':WEIGHT_SHA,'protocol_sha256':expected,'cycle':cycle,'progress':dict(progress),'resume_progress_requires_new_manual_audit':True,'optimizer_parameter_names':optimizer_parameter_names,'completed_paired_cycles':terminal_cycle,'trainable_state':trainable,'frozen_state_sha256':frozen_state,'logical_state_sha256':digest_state(),'code_optimizer':code_optimizer.state_dict(),'fusion_optimizer':fusion_optimizer.state_dict(),'torch_CPU_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state_all(),'python_random_state':random.getstate(),'numpy_random_state':{'algorithm':np.random.get_state()[0],'keys':np.random.get_state()[1].tolist(),'position':np.random.get_state()[2],'has_gauss':np.random.get_state()[3],'cached_gaussian':np.random.get_state()[4]},'sampler_schedule':schedule,'next_cycle':terminal_cycle+1,'next_clean_sampler_index':terminal_cycle*4,'counts':dict(counts),'contract':CONTRACT,'reason':reason,'partial_accumulated_gradients':{name:parameter.grad.detach().cpu() for name,parameter in net.named_parameters() if parameter.grad is not None},'automatic_resume_permitted':False,'model_qualified':False}
        path=out/'terminal_training_state.pth';need(not path.exists(),'Terminal state already exists')
        partial=out/'terminal_training_state.pth.partial'
        with partial.open('xb') as stream:
            torch.save(snapshot,stream);stream.flush();os.fsync(stream.fileno())
        guard(terminal_save=True)
        need(not path.exists(),'Committed terminal state already exists')
        os.replace(partial,path)
        return {'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path),'cycle':cycle,'logical_state_sha256':snapshot['logical_state_sha256']}
    def assess(rows):
        macro={key:sum(row[key] for row in rows)/len(rows) for key in ('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')}
        geometry={kind:{key:sum(row[key] for row in rows if row['mask_kind']==kind)/sum(row['mask_kind']==kind for row in rows) for key in macro} for kind in KINDS}
        return {'macro':macro,'geometry':geometry}
    def evaluate(cycle,rows,kind):
        folder=out/('u%04d'%cycle)/kind;folder.mkdir(parents=True,exist_ok=False);net.requires_grad_(False);records=[]
        with torch.no_grad():
            for case in rows:
                if kind=='monitor':rgb=clean[case['reference_id']][0];target=rgb
                else:rgb=image(case['input256'],256);target=None
                m=mask(case['mask']);guard()
                if not m.any().item():
                    counts['empty_bypasses']+=1;out_tensor=rgb.clone();raw=None
                else:
                    x,m512=app_input(rgb,m);raw,_,_=forward(kind if kind=='monitor' else 'public_development',lambda:net(x,w=1,detach_16=True,adain=False));out_tensor=delivered(rgb,m,raw)
                need(torch.isfinite(out_tensor).all().item() and torch.equal(out_tensor*(1-m),rgb*(1-m)),'Visible pixel preservation failure')
                array=save_pixels(folder/(case['id']+'.png'),out_tensor)
                observed=np.rint(rgb[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                support=m[0,0].cpu().numpy().astype(bool)
                need(np.array_equal(array[~support],observed[~support]),'PNG visible pixels changed')
                record={'id':case['id'],'mask_kind':case.get('mask_kind'),'family':case.get('family'),'condition':case.get('condition'),'PNG':str((folder/(case['id']+'.png')).relative_to(out)),'PNG_sha256':sha(folder/(case['id']+'.png')),'outside_exact':True,'empty':not bool(support.any())}
                if target is not None and support.any():
                    b=boundary(m);need(b.sum().item()>0,'Empty boundary support')
                    target_array=np.rint(target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                    difference=abs(out_tensor-target)
                    record.update({'raw_hole_L1':float((difference*m).sum().item()/(m.sum().item()*3)),'raw_boundary_L1':float((difference*b).sum().item()/(b.sum().item()*3)), 'PNG_hole_L1':float(abs(array.astype(np.float64)-target_array)[support].mean()/255),'PNG_boundary_L1':float(abs(array.astype(np.float64)-target_array)[b[0,0].cpu().numpy().astype(bool)].mean()/255)})
                np.savez_compressed(folder/(case['id']+'.npz'),input256=rgb.detach().cpu().numpy(),mask256=m.detach().cpu().numpy(),result256=out_tensor.detach().cpu().numpy(),raw512=raw.detach().cpu().numpy() if raw is not None else np.empty((0,),np.float32))
                records.append(record);guard()
        summary={'cycle':cycle,'kind':kind,'records':records,'counts':dict(counts),'model_qualified':False}
        if kind=='monitor':summary['metrics']=assess([r for r in records if not r['empty']])
        write(folder/'results.json',summary);return summary
    def empty_controls(cycle):
        bypass=importlib.import_module('pretrained_completion').CodeFormerCompletion(net)
        before_calls=model_forwards['calls']
        for ref,(rgb,_) in clean.items():
            if references[ref]['role'] not in ('fit','monitor'):continue
            out_rgb=bypass(rgb,torch.zeros_like(rgb[:,:1]));need(torch.equal(rgb,out_rgb),'Empty control mismatch');counts['empty_bypasses']+=1
        need(model_forwards['calls']==before_calls,'Empty control unexpectedly called neural model')
        write(out/('empty_controls_u%04d.json'%cycle),{'references':sorted(clean),'exact_bypasses':40,'model_calls':0,'clear_glasses_and_uncovered_control':'No removal means exact source bypass; this is functional preservation, not identity quality'})
    def compare(summary,baseline):
        now=summary['metrics'];old=baseline['metrics'];gain=(old['macro']['PNG_hole_L1']-now['macro']['PNG_hole_L1'])/max(old['macro']['PNG_hole_L1'],1e-12)
        catastrophic=any(now['macro'][key]>old['macro'][key]*1.10 for key in old['macro']) or any(now['geometry'][kind][key]>old['geometry'][kind][key]*1.10 for kind in KINDS for key in old['macro'])
        nonworse=all(now['macro'][key]<=old['macro'][key]+1e-12 for key in old['macro']) and all(now['geometry'][kind][key]<=old['geometry'][kind][key]+1e-12 for kind in KINDS for key in old['macro'])
        return {'prospective_numeric_screen_pass':bool(gain>=.01 and nonworse),'PNG_macro_hole_gain':gain,'all_raw_PNG_hole_boundary_macro_geometry_nonworse':nonworse,'catastrophic_regression_stop':catastrophic,'one_percent_scope':'Prospective delivered-pixel signal threshold for this pilot, not a universal completion quality or identity criterion','visual_review_pending':True,'model_qualified':False}
    try:
        for ref,row in references.items():
            c256=image(row['clean256'],256);c512=image(row['clean512'],512);clean[ref]=(c256,c512)
            with torch.no_grad():
                latent=forward('teacher_encoder',lambda:net.encoder(c512*2-1));_,_,stats=forward('teacher_quantizer',lambda:net.quantize(latent))
            need(tuple(latent.shape)==(1,256,16,16),'Teacher latent shape')
            codes=stats['min_encoding_indices'].reshape(1,256).long();teacher[ref]=(latent.detach(),codes.detach())
            guard()
        wrapper=importlib.import_module('pretrained_completion').CodeFormerCompletion(net)
        parity_cases=[r for r in p['fit_cases'] if r['reference_id'] in sorted(references)[:2] and r['mask_kind'] in ('lower_face','eye_band')]
        need(len(parity_cases)==4,'Four fresh baseline parity cases required')
        with torch.no_grad():
            for case in parity_cases:
                rgb=clean[case['reference_id']][0];m=mask(case['mask']);x,_=app_input(rgb,m)
                raw,_,_=forward('parity',lambda:net(x,w=1,detach_16=True,adain=False));own=delivered(rgb,m,raw)
                reference=forward('parity',lambda:wrapper(rgb,m));error=float((own-reference).abs().max().item())
                need(error<=1e-6,'Fresh frozen app pipeline baseline parity failed');parity.append({'id':case['id'],'max_raw256_error':error})
        del wrapper
        need(digest_state()==base_state,'Initial teacher/parity changed original model state')
        write(out/'initial_state.json',{'original_logical_state_sha256':base_state,'frozen_generator_codebook_sha256':frozen_state,'checkpoint_sha256':WEIGHT_SHA,'fresh_app_parity':parity,'teacher_limitation':CONTRACT['teacher_limitation'],'hardware':hardware,'torch':torch.__version__,'counts':counts})
        baseline=evaluate(0,p['monitor_cases'],'monitor');evaluate(0,p['development_cases'],'public_development');empty_controls(0)
        monitors=[baseline]
        for cycle,case_ids in enumerate(schedule,1):
            progress.update(cycle=cycle,route='code',microbatch=0,microbatches_backward_complete=0,phase='accumulating')
            guard();active(code_groups);code_optimizer.zero_grad(set_to_none=True);code_losses=[]
            for micro,ident in enumerate(case_ids,1):
                progress['microbatch']=micro
                case=fit_cases[ident];rgb=clean[case['reference_id']][0];m=mask(case['mask']);x,_=app_input(rgb,m)
                logits,lq=forward('fit_code',lambda:net(x,code_only=True));latent,codes=teacher[case['reference_id']]
                ce=F.cross_entropy(logits.reshape(-1,512),codes.reshape(-1));latent_loss=F.mse_loss(lq,latent);loss=(ce+latent_loss)/4
                need(torch.isfinite(loss).item(),'Nonfinite code objective');counts['backward_calls']+=1;progress['phase']='backward_started';loss.backward();progress.update(phase='accumulating',microbatches_backward_complete=micro);code_losses.append({'id':ident,'CE':float(ce.detach().item()),'latent_MSE':float(latent_loss.detach().item())})
                del logits,lq,ce,latent_loss,loss;guard()
            code_gradient=gradient_receipt(code_groups);code_clip=float(torch.nn.utils.clip_grad_norm_(code_parameters,1.,error_if_nonfinite=True).item());progress['phase']='optimizer_step_started';code_optimizer.step();counts['code_optimizer_steps']+=1;code_optimizer.zero_grad(set_to_none=True)
            need(all(torch.isfinite(parameter).all().item() for parameter in code_parameters),'Nonfinite updated code state')
            progress.update(route='fusion',microbatch=0,microbatches_backward_complete=0,phase='accumulating')
            active(fusion_groups);fusion_optimizer.zero_grad(set_to_none=True);fusion_losses=[]
            for micro,ident in enumerate(case_ids,1):
                progress['microbatch']=micro
                case=fit_cases[ident];rgb,target=clean[case['reference_id']];m=mask(case['mask']);x,m512=app_input(rgb,m)
                raw,_,_=forward('fit_RGB',lambda:net(x,w=1,detach_16=True,adain=False));prediction=(raw+1)/2;b=boundary(m512)
                rgb_loss=(abs(prediction-target)*m512).sum()/(m512.sum()*3);seam=(abs(prediction-target)*b).sum()/(b.sum()*3);loss=(rgb_loss+seam)/4
                need(torch.isfinite(loss).item(),'Nonfinite fusion objective');counts['backward_calls']+=1;progress['phase']='backward_started';loss.backward();progress.update(phase='accumulating',microbatches_backward_complete=micro);fusion_losses.append({'id':ident,'mask_RGB512_L1':float(rgb_loss.detach().item()),'inner_boundary_RGB512_L1':float(seam.detach().item())})
                del raw,prediction,rgb_loss,seam,loss;guard()
            fusion_gradient=gradient_receipt(fusion_groups);fusion_clip=float(torch.nn.utils.clip_grad_norm_(fusion_parameters,1.,error_if_nonfinite=True).item());progress['phase']='optimizer_step_started';fusion_optimizer.step();counts['fusion_optimizer_steps']+=1;fusion_optimizer.zero_grad(set_to_none=True)
            need(all(torch.isfinite(parameter).all().item() for parameter in fusion_parameters),'Nonfinite updated fusion state')
            terminal_cycle=cycle;progress.update(phase='paired_cycle_complete',microbatch=4)
            row={'cycle':cycle,'microbatch_ids':case_ids,'code_losses':code_losses,'fusion_losses':fusion_losses,'code_gradient_groups':code_gradient,'fusion_gradient_groups':fusion_gradient,'code_preclip_L2':code_clip,'fusion_preclip_L2':fusion_clip,'counts':dict(counts),'seconds':time.monotonic()-started}
            fit_log.append(row);write(out/('training_cycle_%04d.json'%cycle),row)
            if cycle in (8,16,32):
                need(digest_state(('quantize.','generator.'))==frozen_state,'Frozen generator/codebook changed')
                snapshots.append({'cycle':cycle,'logical_state_sha256':digest_state(),'state_saved':False,'recoverable_checkpoint':False,'scope':'Images/metrics only. Not selectable or resumable; terminal state only is saved.'})
                summary=evaluate(cycle,p['monitor_cases'],'monitor');monitors.append(summary);screen=compare(summary,baseline);write(out/('screen_u%04d.json'%cycle),screen);empty_controls(cycle)
                if screen['catastrophic_regression_stop']:
                    stop_reason='Prospective10percent monitor macro/geometry hole-or-boundary regression';break
            print(json.dumps({'cycle':cycle,'maximum':32,'counts':counts,'seconds':time.monotonic()-started}),flush=True)
        evaluate(terminal_cycle,p['development_cases'],'public_development')
        final_screen=compare(monitors[-1],baseline)
        need(digest_state(('quantize.','generator.'))==frozen_state and sha(checkpoint)==WEIGHT_SHA,'Frozen/original assets changed')
        terminal_state=save_state(terminal_cycle,'completed_finite_pilot' if not stop_reason else 'quality_stopped_terminal')
        final={'format':'completion-codeformer-finetune-results-v1','complete':True,'protocol_sha256':expected,'cycles_completed':terminal_cycle,'stop_reason':stop_reason,'numeric_screen':final_screen,'metric_snapshots':snapshots,'terminal_training_state':terminal_state,'earlier_snapshots_recoverable':False,'initial_state_sha256':base_state,'final_logical_state_sha256':digest_state(),'frozen_state_unchanged':True,'original_checkpoint_unchanged':True,'counts':counts,'typed_forward_slots':typed,'actual_whole_model_forward_calls':model_forwards['calls'],'seconds':time.monotonic()-started,'source_selection':'Adaptive TRAIN fit/monitor only; not final independent evaluation','teacher_limitation':CONTRACT['teacher_limitation'],'visual_review_pending':True,'user_acceptance_pending':True,'model_qualified':False,'automatic_app_promotion':False}
        write(out/'results.json',final);guard();return 0 if final_screen['prospective_numeric_screen_pass'] and terminal_cycle==32 and not stop_reason else 2
    except BaseException as exc:
        if not (out/'terminal_training_state.pth').exists():
            try:snapshots.append(save_state(terminal_cycle,'failure_retained_current_partial_state'))
            except BaseException as save_error:
                if not (out/'terminal_save_failure.json').exists():write(out/'terminal_save_failure.json',{'error':repr(save_error),'progress':dict(progress),'counts':dict(counts),'note':'State may not be recoverable; never auto-resume'})
        if not (out/'failure.json').exists():write(out/'failure.json',{'complete':False,'exception':repr(exc),'traceback':traceback.format_exc(),'last_full_cycle':terminal_cycle,'partial_progress':dict(progress),'counts':counts,'typed_forward_slots':typed,'snapshots':snapshots,'seconds':time.monotonic()-started,'model_qualified':False})
        raise
    finally:
        forward_hook.remove()
        signal.signal(signal.SIGTERM,prior_termination)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-transfer',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--worker',action='store_true')
    a=parser.parse_args();root=a.root.resolve(strict=True);p,checkpoint=verify(root,a.protocol_sha)
    if a.verify_transfer:print(json.dumps({'complete':True,'files':len(p['files']),'fit_cases':128,'monitor_cases':32,'optimizer_updates':0,'neural_calls':0}));return 0
    if a.worker:return worker(root,p,checkpoint,a.protocol_sha)
    hardware=require_vm(root);need(not (root/'outputs').exists() and not (root/'supervisor.json').exists(),'Pilot already attempted; preserve evidence')
    command=[sys.executable,'-B','-u',str(root/'scripts/cctv_completion_codeformer_finetune_v1_vm.py'),'--root',str(root),'--protocol-sha',a.protocol_sha,'--worker']
    started=time.monotonic();timeout=False;interruption=None;code=1;child=None;shutdown_mode=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Supervisor termination signal '+str(signum))
    prior_termination=signal.signal(signal.SIGTERM,interrupted)
    def kill_owned(sig):
        if child is None or child.poll() is not None:return
        try:
            need(os.getpgid(child.pid)==child.pid,'Owned process group differs; refuse unrelated termination')
            os.killpg(child.pid,sig)
        except ProcessLookupError:pass
    try:
        with (root/'trainer.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                timeout=True;shutdown_mode='hard_cap_SIGKILL';kill_owned(signal.SIGKILL);child.wait();code=124
    except BaseException as exc:
        interruption=repr(exc);shutdown_mode='interruption_SIGTERM_then_bounded_wait'
        if child is not None and child.poll() is None:
            kill_owned(signal.SIGTERM)
            remaining=max(0.,min(15.,1800-(time.monotonic()-started)))
            try:child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                shutdown_mode='interruption_grace_exhausted_SIGKILL';kill_owned(signal.SIGKILL);child.wait()
        code=130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        if child is not None and child.poll() is None:
            kill_owned(signal.SIGKILL);child.wait()
        signal.signal(signal.SIGTERM,prior_termination)
        write(root/'supervisor.json',{'complete':True,'worker_exit':code,'actual_child_exit':None if child is None else child.returncode,'child_pid':None if child is None else child.pid,'timeout':timeout,'interruption':interruption,'shutdown_mode':shutdown_mode,'external_cap_seconds':1800,'seconds':time.monotonic()-started,'hardware':hardware,'terminal_state_present':(root/'outputs/terminal_training_state.pth').is_file(),'partial_state_recoverability':'Requires returned-state audit. Hard kill may prevent terminal save; no automatic resume.','training_success_not_implied':True})
    print(json.dumps({'worker_exit':code,'timeout':timeout,'interruption':interruption,'log':'trainer.log','model_qualified':False}));return code
if __name__=='__main__':raise SystemExit(main())

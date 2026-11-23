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
import selectors
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.completion_codeformer_coverage_v2.contracts import (CONTRACT,VM_NAME,ORIGINAL_SHA,PARENT_SHA,PARENT_PROTOCOL_SHA,PARENT_LOGICAL,ORIGINAL_LOGICAL,KINDS,METRICS,OutputBudget,aggregate,compare as compare_metrics)
WEIGHT_SHA=ORIGINAL_SHA
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

def verify(root,expected):
    root=root.resolve(strict=True);need((root/'protocol.json').stat().st_size<=1024**2 and sha(root/'protocol.json')==expected,'Protocol size/SHA differs')
    p=read(root/'protocol.json');need(p['format']=='completion-codeformer-coverage-v2' and p['contract']==CONTRACT,'Coverage-v2 contract differs')
    files={r['path']:r for r in p['files']};need(len(files)==len(p['files']),'Duplicate pinned file')
    for row in files.values():
        path=member(root,row['path']);need(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Packet drift: '+row['path'])
    for name in list(SOURCE_MODULES.values())+['scripts/cctv_completion_codeformer_coverage_v2_vm.py','research/completion_codeformer_coverage_v2/contracts.py']:
        need(name in files,'Imported/worker source is not pinned: '+name)
    for name in (p['data_manifest'],p['original_parent_protocol'],p['original_monitor_summary'],*p['evidence'].values()):need(name in files,'Unpinned evidence')
    need(sha(member(root,p['original_parent_protocol']))==PARENT_PROTOCOL_SHA,'Original R1 protocol binding differs')
    evidence_expected={'design_review':'9566e77f7fb33b4116f744d043ccf5af1ff133222fb69d6ad0b047db324a4682','data_admission':'7c072ec5a916468275e4e5ba3e3124e9b472030014f1e7afed803f5beca3d63a','root_visual_review':'8488e9354a87ce186931c2770a9ce6a8de0da06b92baddba2c6d593aea54db72','parent_state_audit':'e920c3ddd9b8a85b50fd6f324bde8760c13c6df805f6be61d1e4393472d59b06'}
    for key,digest in evidence_expected.items():need(sha(member(root,p['evidence'][key]))==digest,'Independent source/data/parent admission differs: '+key)
    refs={r['id']:r for r in p['references']};need(len(refs)==40 and len(p['teacher_targets'])==80,'Bounded source/teacher count differs')
    need(all(isinstance(ref,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',ref) for ref in refs),'Unsafe reference ID')
    fit={r['id'] for r in refs.values() if r['role']=='fit'};monitor={r['id'] for r in refs.values() if r['role']=='monitor'}
    need(len(fit)==32 and len(monitor)==8 and not fit&monitor,'FIT/MONITOR split differs')
    need(all(r.get('original_role')=='train' and r.get('legacy_completion_split')=='train' for r in refs.values()),'Only original TRAIN sources admitted')
    teachers={r['id']:r for r in p['teacher_targets']};need(set(teachers)=={ref+'::'+style for ref in refs for style in ('RGB','GRAY')},'Exact80RGB/GRAY targets required')
    for row in teachers.values():
        need(row['id']==row['reference_id']+'::'+row['style'],'Teacher key differs')
        need(row.get('cache_id',row['reference_id']+'_'+row['style'].lower())==row['reference_id']+'_'+row['style'].lower(),'Teacher cache filename is not portable/canonical')
        need(row['clean256'] in files and row['clean512'] in files,'Teacher target unpinned')
    counts={'fit_cases':256,'monitor_cases':32,'challenge_cases':16,'development_cases':32}
    for name,count in counts.items():
        rows=p[name];need(len(rows)==count and len({r['id'] for r in rows})==count,'Case count/uniqueness differs: '+name)
        for row in rows:
            need(isinstance(row['id'],str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',row['id']),'Unsafe case ID')
            need(row['input256'] in files and row['mask'] in files,'Input/mask is unpinned')
            if name!='development_cases':
                need(row['teacher_key'] in teachers and row['clean256']==teachers[row['teacher_key']]['clean256'] and row['clean512']==teachers[row['teacher_key']]['clean512'],'Case target/teacher mismatch')
                need(row['reference_id'] in (fit if name=='fit_cases' else monitor),'Case crossed source split')
    cases={r['id']:r for r in p['fit_cases']};schedule=p['schedule'];need(len(schedule)==64 and all(len(row)==4 for row in schedule),'64four-case cycles required')
    flat=[]
    for row in schedule:
        need(len(set(row))==4 and set(row)<=cases.keys(),'Unsafe/duplicate schedule IDs')
        selected=[cases[i] for i in row];need(len({r['reference_id'] for r in selected})==4,'Eachcycle needs4distinct sources')
        need(sum(r['role']=='fit_anchor' for r in selected)==2 and sum(r['role']=='fit_augmented' for r in selected)==2,'Eachcycle needs2anchors+2augmented')
        flat.extend(row)
    need(len(set(flat))==256 and set(flat)==set(cases),'Each256newcase must occur once perroute')
    need(sum(r['role']=='fit_anchor' for r in cases.values())==128,'Original128anchors required')
    conditions={name:sum(r['condition']==name for r in cases.values()) for name in ('strict_gray_clear','RGB_mild_degraded','strict_gray_mild_degraded','RGB_clear_irregular')}
    need(set(conditions.values())=={32},'Augmentation conditions unbalanced')
    challenge=p['challenge_cases'];mon_order=sorted(monitor)
    for i,ref in enumerate(mon_order):
        rows=[r for r in challenge if r['reference_id']==ref];need(len(rows)==2,'Challenge source count differs')
        expected={('strict_gray_mild_degraded',KINDS[i%4]),('RGB_irregular_mild_degraded',KINDS[(i+2)%4])}
        need({(r['condition'],r['mask_kind']) for r in rows}==expected,'Challenge frozen rotation differs')
    original=read(member(root,p['original_parent_protocol']));need(fit=={r['id'] for r in original['references'] if r['role']=='fit'} and monitor=={r['id'] for r in original['references'] if r['role']=='monitor'},'Original source split changed')
    orig_mon=read(member(root,p['original_monitor_summary']));need({r['id'] for r in orig_mon['records']}=={r['id'] for r in p['monitor_cases']},'Original baseline monitor membership differs')
    def external(row,name,want_sha,want_bytes):
        raw=root/row['path'];need(not raw.is_symlink(),'Linked external file refused');path=raw.resolve(strict=True)
        need(path.is_relative_to((Path.home()/'forensic-dgp').resolve()) and path.name==name,'Shared evidence escaped forensic-dgp')
        need(row['sha256']==want_sha and row['bytes']==want_bytes and path.is_file() and path.stat().st_size==want_bytes and sha(path)==want_sha,'External evidence size/SHA differs')
        return path
    checkpoint=external(p['checkpoint'],'codeformer_inpainting.pth',ORIGINAL_SHA,370786611)
    parent=external(p['parent'],'terminal_training_state.pth',PARENT_SHA,701952165)
    need(p['parent']['protocol_sha256']==PARENT_PROTOCOL_SHA and p['parent']['logical_sha256']==PARENT_LOGICAL,'FullR1resume parent differs')
    return p,checkpoint,parent

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

def worker(root,p,checkpoint,parent_path,expected):
    hardware=require_vm(root)
    def cooperative_termination(signum,frame):raise KeyboardInterrupt('Supervisor requested termination; retain current partial state')
    prior_termination=signal.signal(signal.SIGTERM,cooperative_termination)
    import shutil
    need(shutil.disk_usage(root).free>=CONTRACT['minimum_free_bytes'],'Need7GiB free before pilot')
    out=root/'outputs';need(not out.exists(),'Never overwrite/resume this finite historical pilot');out.mkdir()
    started=time.monotonic();counts={'forward_slots':0,'backward_calls':0,'code_optimizer_steps':0,'fusion_optimizer_steps':0,'empty_bypasses':0}
    typed={'teacher_encoder':0,'teacher_quantizer':0,'parity':0,'fit_code':0,'fit_RGB':0,'monitor':0,'challenge':0,'public_development':0}
    budget=OutputBudget(out,started);budget.metadata_external=[root/'trainer.log',root/'protocol.json']
    def write(path,value,*,emergency=False):budget.json(path,value,emergency=emergency)
    def save_npz(path,**arrays):
        with budget.open(path) as stream:np.savez_compressed(stream,**arrays)
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
        budget.guard(terminal=terminal_save)
        need(counts['forward_slots']<=1000 and counts['backward_calls']<=512,'Finite typed/backward cap')
        need(counts['code_optimizer_steps']<=64 and counts['fusion_optimizer_steps']<=64,'Finite phase optimizer cap')
        need(torch.cuda.memory_allocated()<=CONTRACT['CUDA_allocated_bytes_cap'],'CUDA memory cap')
    def forward(kind,call):
        need(not budget.terminal_committed,'No neural calls after terminal commit')
        guard();counts['forward_slots']+=1;typed[kind]+=1;guard();return call()
    def digest_state(prefixes=None):
        digest=hashlib.sha256()
        for name,value in sorted(net.state_dict().items()):
            if prefixes and not any(name.startswith(prefix) for prefix in prefixes):continue
            array=value.detach().cpu().contiguous();need(torch.isfinite(array).all().item(),'Nonfinite model state')
            digest.update(name.encode());digest.update(str(tuple(array.shape)).encode());digest.update(str(array.dtype).encode());digest.update(array.numpy().tobytes())
        return digest.hexdigest()
    original_teacher_state=digest_state();base_state=original_teacher_state;frozen_state=digest_state(('quantize.','generator.'))
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
        # Match the admitted app's active advanced indexing, including layout.
        active=m.sum((1,2,3))>0
        need(rgb.shape[0]==1 and bool(active.all().item()),'Use one active reviewed case; caller bypasses empty masks')
        visible=1-m[active]
        support=F.interpolate(visible,size=(512,512),mode='bilinear',align_corners=False)
        x=F.interpolate(rgb[active]*visible,size=(512,512),mode='bilinear',align_corners=False)
        x=x/support.clamp_min(1e-8)
        m512=F.interpolate(m[active],size=(512,512),mode='nearest')
        return (x*(1-m512)+m512)*2-1,m512
    def delivered(rgb,m,raw):
        result=rgb.clone()
        active=m.sum((1,2,3))>0
        if not active.any():return result
        generated=((raw+1)/2).clamp(0,1)
        generated=F.interpolate(generated,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        result[active]=rgb[active]*(1-m[active])+generated*m[active]
        return result
    def save_pixels(path,value):
        a=np.rint(value.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        with budget.open(path) as stream:Image.fromarray(a).save(stream,format="PNG")
        return a
    def boundary(m):return (m-(1-F.max_pool2d(1-m,7,1,3))).clamp(0,1)
    references={r['id']:r for r in p['references']};clean={};teacher={}
    fit_cases={r['id']:r for r in p['fit_cases']};schedule=p['schedule']
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
    snapshots=[];terminal_cycle=0;stop_reason=None;parity=[];fit_log=[];parent_optimizer_restored=False;parent_RNG_restored=False;parent_metadata={};resume_rng=None
    progress={'cycle':0,'route':'initialization','microbatch':0,'phase':'before_first_update'}
    model_forwards={'calls':0}
    def model_hook(module,args):model_forwards['calls']+=1
    forward_hook=net.register_forward_pre_hook(model_hook)
    def save_state(cycle,reason):
        need(digest_state(('quantize.','generator.'))==frozen_state,'Frozen generator/codebook changed')
        trainable={name:value.detach().cpu() for name,value in net.state_dict().items() if not name.startswith(('quantize.','generator.'))}
        snapshot={'format':'completion-codeformer-coverage-v2-trainable-resume-v1','parent_training_state_sha256':PARENT_SHA,'parent_protocol_sha256':PARENT_PROTOCOL_SHA,'parent_initial_logical_sha256':PARENT_LOGICAL,'parent_optimizer_restored':parent_optimizer_restored,'parent_RNG_restored':parent_RNG_restored,'parent_metadata':parent_metadata,'phase_completed_paired_cycles':terminal_cycle,'cumulative_completed_paired_cycles':32+terminal_cycle,'cumulative_counts':{'backward_calls':256+counts['backward_calls'],'code_optimizer_steps':32+counts['code_optimizer_steps'],'fusion_optimizer_steps':32+counts['fusion_optimizer_steps'],'optimizer_steps_total':64+counts['code_optimizer_steps']+counts['fusion_optimizer_steps'],'forward_slots':528+counts['forward_slots'],'actual_whole_model_forward_calls':448+model_forwards['calls']},'phase_name':'coverage_v2','base_checkpoint_sha256':WEIGHT_SHA,'protocol_sha256':expected,'cycle':cycle,'progress':dict(progress),'resume_progress_requires_new_manual_audit':True,'optimizer_parameter_names':optimizer_parameter_names,'completed_paired_cycles':32+terminal_cycle,'trainable_state':trainable,'frozen_state_sha256':frozen_state,'logical_state_sha256':digest_state(),'code_optimizer':code_optimizer.state_dict(),'fusion_optimizer':fusion_optimizer.state_dict(),'torch_CPU_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state_all(),'python_random_state':random.getstate(),'numpy_random_state':{'algorithm':np.random.get_state()[0],'keys':np.random.get_state()[1].tolist(),'position':np.random.get_state()[2],'has_gauss':np.random.get_state()[3],'cached_gaussian':np.random.get_state()[4]},'sampler_schedule':schedule,'next_cycle':terminal_cycle+1,'next_clean_sampler_index':terminal_cycle*4,'counts':dict(counts),'contract':CONTRACT,'reason':reason,'partial_accumulated_gradients':{name:parameter.grad.detach().cpu() for name,parameter in net.named_parameters() if parameter.grad is not None},'automatic_resume_permitted':False,'model_qualified':False}
        path=out/'terminal_training_state.pth';need(not path.exists(),'Terminal state already exists')
        partial=out/'terminal_training_state.pth.partial'
        with budget.open(partial,terminal=True) as stream:
            torch.save(snapshot,stream);stream.flush();os.fsync(stream.stream.fileno())
        guard(terminal_save=True)
        need(not path.exists(),'Committed terminal state already exists')
        os.replace(partial,path);budget.rename(partial,path)
        return {'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path),'cycle':cycle,'logical_state_sha256':snapshot['logical_state_sha256']}
    def evaluate(cycle,rows,kind):
        folder=out/('u%04d'%cycle)/kind;folder.mkdir(parents=True,exist_ok=False);net.requires_grad_(False);records=[]
        with torch.no_grad():
            for case in rows:
                rgb=image(case['input256'],256);target=image(case['clean256'],256) if kind!='public_development' else None
                m=mask(case['mask']);guard()
                if not m.any().item():
                    counts['empty_bypasses']+=1;out_tensor=rgb.clone();raw=None
                else:
                    x,m512=app_input(rgb,m);raw,_,_=forward(kind,lambda:net(x,w=1,detach_16=True,adain=False));out_tensor=delivered(rgb,m,raw)
                need(torch.isfinite(out_tensor).all().item() and torch.equal(out_tensor*(1-m),rgb*(1-m)),'Visible raw preservation failed')
                array=save_pixels(folder/(case['id']+'.png'),out_tensor)
                observed=np.rint(rgb[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8);support=m[0,0].cpu().numpy().astype(bool)
                need(np.array_equal(array[~support],observed[~support]),'PNG visible preservation failed')
                record={'id':case['id'],'mask_kind':case.get('mask_kind'),'family':case.get('family'),'condition':case.get('condition'),'style':case.get('style'),'PNG':str((folder/(case['id']+'.png')).relative_to(out)),'PNG_sha256':sha(folder/(case['id']+'.png')),'outside_exact':True,'empty':not bool(support.any())}
                if target is not None and support.any():
                    b=boundary(m);need(b.sum().item()>0,'Empty boundary');target_array=np.rint(target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8);difference=abs(out_tensor-target)
                    record.update({'raw_hole_L1':float((difference*m).sum().item()/(m.sum().item()*3)),'raw_boundary_L1':float((difference*b).sum().item()/(b.sum().item()*3)),'PNG_hole_L1':float(abs(array.astype(np.float64)-target_array)[support].mean()/255),'PNG_boundary_L1':float(abs(array.astype(np.float64)-target_array)[b[0,0].cpu().numpy().astype(bool)].mean()/255)})
                    if case.get('style')=='GRAY':
                        raw_rgb=out_tensor[0].permute(1,2,0).cpu().numpy();record['GRAY_raw_mean_channel_spread']=float(np.ptp(raw_rgb,axis=-1)[support].mean());record['GRAY_PNG_mean_channel_spread']=float(np.ptp(array.astype(np.int16),axis=-1)[support].mean());record['GRAY_chroma_diagnostic_not_selection_gate']=True
                save_npz(folder/(case['id']+'.npz'),input256=rgb.detach().cpu().numpy(),mask256=m.detach().cpu().numpy(),result256=out_tensor.detach().cpu().numpy(),raw512=raw.detach().cpu().numpy() if raw is not None else np.empty((0,),np.float32))
                records.append(record);guard()
        summary={'cycle':cycle,'kind':kind,'records':records,'counts':dict(counts),'model_qualified':False}
        if kind!='public_development':summary['metrics']=aggregate([r for r in records if not r['empty']],condition=kind=='challenge')
        write(folder/'results.json',summary);return summary
    def empty_controls(cycle):
        bypass=importlib.import_module('pretrained_completion').CodeFormerCompletion(net)
        before_calls=model_forwards['calls']
        for ref in references:
            rgb=clean[ref+'::RGB'][0]
            out_rgb=bypass(rgb,torch.zeros_like(rgb[:,:1]));need(torch.equal(rgb,out_rgb),'Empty control mismatch');counts['empty_bypasses']+=1
        need(model_forwards['calls']==before_calls,'Empty control unexpectedly called neural model')
        write(out/('empty_controls_u%04d.json'%cycle),{'references':sorted(references),'exact_bypasses':40,'model_calls':0,'clear_glasses_and_uncovered_control':'No removal means exact source bypass; this is functional preservation, not identity quality'})
    def compare(summary,baseline,*,conditions=False):return compare_metrics(summary,baseline,conditions=conditions)
    try:
        need(original_teacher_state==ORIGINAL_LOGICAL,'Original teacher logical state differs')
        teacher_index=[]
        for row in p['teacher_targets']:
            c256=image(row['clean256'],256);c512=image(row['clean512'],512);clean[row['id']]=(c256,c512)
            if row['style']=='GRAY':need(torch.equal(c256[:,0],c256[:,1]) and torch.equal(c256[:,1],c256[:,2]) and torch.equal(c512[:,0],c512[:,1]) and torch.equal(c512[:,1],c512[:,2]),'GRAY targets not exactgray')
            with torch.no_grad():
                latent=forward('teacher_encoder',lambda:net.encoder(c512*2-1));_,_,stats=forward('teacher_quantizer',lambda:net.quantize(latent))
            need(tuple(latent.shape)==(1,256,16,16),'Teacher latent shape')
            codes=stats['min_encoding_indices'].reshape(1,256).long()
            need(latent.dtype==torch.float32 and torch.isfinite(latent).all().item() and bool(((codes>=0)&(codes<512)).all().item()),'Invalidteacher latent/codes')
            teacher[row['id']]=(latent.detach(),codes.detach())
            label_path=out/'teacher_labels'/(row.get('cache_id',row['reference_id']+'_'+row['style'].lower())+'.npz')
            save_npz(label_path,prequant_latent=latent.detach().cpu().numpy(),nearest_codes=codes.detach().cpu().numpy())
            teacher_index.append({'id':row['id'],'path':label_path.relative_to(out).as_posix(),'bytes':label_path.stat().st_size,'sha256':sha(label_path),'clean512':row['clean512'],'clean512_sha256':sha(root/row['clean512']),'original_teacher_weights_sha256':ORIGINAL_SHA,'teacher_original_logical_sha256':original_teacher_state})
            guard()
        write(out/'teacher_labels/index.json',{'complete':True,'labels':teacher_index,'sets':80,'component_calls':160,'definition':'originalprequantencoderlatent andnearestcodes, notofficialquantizedHQVQGANteacher','counter':dict(counts)})
        need(digest_state()==ORIGINAL_LOGICAL,'Teacher cache mutated original state')
        parent=torch.load(parent_path,map_location='cpu',weights_only=True)
        need(parent['protocol_sha256']==PARENT_PROTOCOL_SHA and parent['base_checkpoint_sha256']==ORIGINAL_SHA and parent['logical_state_sha256']==PARENT_LOGICAL,'Parent fingerprint differs')
        need(parent['cycle']==parent['completed_paired_cycles']==32 and parent['partial_accumulated_gradients']=={} and parent['reason']=='completed_finite_pilot','Only audited completedR1parent allowed')
        need(parent['optimizer_parameter_names']==optimizer_parameter_names,'Named optimizer order differs')
        need(parent['counts']=={'forward_slots':528,'backward_calls':256,'code_optimizer_steps':32,'fusion_optimizer_steps':32,'empty_bypasses':168},'Parent actualtraining counters differ')
        parent_protocol=read(root/p['original_parent_protocol']);need(parent['contract']==parent_protocol['contract'] and parent['sampler_schedule']==parent_protocol['schedule'],'Parent contract/sampler differs')
        original_state=net.state_dict();expected_names={n for n in original_state if not n.startswith(('quantize.','generator.'))}
        need(set(parent['trainable_state'])==expected_names and len(expected_names)==332,'Parent trainable partition differs')
        net.load_state_dict({**original_state,**parent['trainable_state']},strict=True)
        need(digest_state()==PARENT_LOGICAL and digest_state(('quantize.','generator.'))==frozen_state==parent['frozen_state_sha256'],'R1 reconstruction/frozen prior differs')
        optimizer_admission={}
        for route,optimizer,params in [('code',code_optimizer,code_parameters),('fusion',fusion_optimizer,fusion_parameters)]:
            state=parent[route+'_optimizer'];need(len(state['param_groups'])==1 and len(state['state'])==len(params),'Parent Adam map differs')
            group=state['param_groups'][0];need(group['lr']==1e-5 and tuple(group['betas'])==(.9,.99) and group['eps']==1e-8 and group['weight_decay']==0,'Parent Adam hyperparameters differ')
            optimizer.load_state_dict(state)
            for index,param in enumerate(params):
                restored=optimizer.state[param];prior=state['state'][group['params'][index]]
                need(float(restored['step'].item())==32 and set(restored)==set(prior),'Parent Adam step/schema differs')
                for key in ('exp_avg','exp_avg_sq'):
                    need(restored[key].shape==param.shape and restored[key].dtype==prior[key].dtype==param.dtype==torch.float32 and restored[key].device==param.device and torch.isfinite(restored[key]).all().item() and torch.equal(restored[key].cpu(),prior[key].cpu()),'Adam moments not restored exactly')
            optimizer_admission[route]={'parameter_tensors':len(params),'step_before':32,'named_order_exact':True,'moments_exact':True,'lr':group['lr']}
        parent_optimizer_restored=True
        resume_rng={'CPU':parent['torch_CPU_RNG'].clone(),'CUDA':[x.clone() for x in parent['CUDA_RNG']],'Python':parent['python_random_state'],'NumPy':parent['numpy_random_state']}
        parent_metadata={'terminal_sha256':PARENT_SHA,'protocol_sha256':PARENT_PROTOCOL_SHA,'logical_state_sha256':PARENT_LOGICAL,'counts':parent['counts'],'sampler_schedule':parent['sampler_schedule'],'completed_cycles':32,'optimizer_admission':optimizer_admission}
        del parent,original_state,state,prior
        base_state=digest_state();progress.update(phase='R1weights_Adam_restored_before_parity_RNG_pending')
        wrapper=importlib.import_module('pretrained_completion').CodeFormerCompletion(net)
        parity_cases=[r for r in p['fit_cases'] if r['role']=='fit_anchor' and r['reference_id'] in sorted(references)[:2] and r['mask_kind'] in ('lower_face','eye_band')]
        need(len(parity_cases)==4,'Four fresh baseline parity cases required')
        parity_folder=out/'fresh_app_parity';parity_folder.mkdir()
        with torch.no_grad():
            for case in parity_cases:
                rgb=image(case['input256'],256);m=mask(case['mask']);x,_=app_input(rgb,m)
                captures=[]
                def capture_input(module,args):
                    value=args[0]
                    captures.append({'input':value.detach().cpu().numpy().copy(),'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),'device':str(value.device),'NCHW_contiguous':value.is_contiguous(),'channels_last_contiguous':value.is_contiguous(memory_format=torch.channels_last)})
                def capture_output(module,args,output):
                    captures[-1].update(raw=output[0].detach().cpu().numpy().copy(),codes=output[1].argmax(dim=2).detach().cpu().numpy().copy())
                input_hook=net.register_forward_pre_hook(capture_input)
                output_hook=net.register_forward_hook(capture_output)
                try:
                    raw,_,_=forward('parity',lambda:net(x,w=1,detach_16=True,adain=False));own=delivered(rgb,m,raw)
                    reference=forward('parity',lambda:wrapper(rgb,m))
                finally:
                    input_hook.remove();output_hook.remove()
                need(len(captures)==2 and all('raw' in row for row in captures),'Capture both fresh parity calls')
                own_capture,app_capture=captures
                error=float((own-reference).abs().max().item())
                receipt={'id':case['id'],'max_raw256_error':error,'max_model_input512_error':float(np.max(np.abs(own_capture['input']-app_capture['input']))),'input512_exact':bool(np.array_equal(own_capture['input'],app_capture['input'])),'input_stride_exact':own_capture['stride']==app_capture['stride'],'model_raw512_max_error':float(np.max(np.abs(own_capture['raw']-app_capture['raw']))),'selected_codes_exact':bool(np.array_equal(own_capture['codes'],app_capture['codes'])),'own_input_layout':{k:v for k,v in own_capture.items() if k not in ('input','raw','codes')},'app_input_layout':{k:v for k,v in app_capture.items() if k not in ('input','raw','codes')},'tolerance_raw256':1e-6,'counts':dict(counts),'optimizer_steps_before_parity':counts['code_optimizer_steps']+counts['fusion_optimizer_steps']}
                save_npz(parity_folder/(case['id']+'.npz'),input256=rgb.detach().cpu().numpy(),mask256=m.detach().cpu().numpy(),own_input512=own_capture['input'],app_input512=app_capture['input'],own_raw512=own_capture['raw'],app_raw512=app_capture['raw'],own_codes=own_capture['codes'],app_codes=app_capture['codes'],own_result256=own.detach().cpu().numpy(),app_result256=reference.detach().cpu().numpy())
                write(parity_folder/(case['id']+'.json'),receipt);parity.append(receipt)
                # Retain all four paired calls before aggregate parity assertions.
        write(parity_folder/'summary.json',{'complete':True,'cases':parity,'all_input_values_and_layout_exact':all(row['input512_exact'] and row['input_stride_exact'] for row in parity),'max_raw256_error':max(row['max_raw256_error'] for row in parity),'tolerance_raw256':1e-6,'counts':dict(counts),'training_updates':0})
        need(all(row['input512_exact'] and row['input_stride_exact'] for row in parity),'Fresh app input value/layout parity failed')
        need(all(row['max_raw256_error']<=1e-6 for row in parity),'Fresh frozen app pipeline baseline parity failed')
        del wrapper
        need(digest_state()==base_state,'Initial teacher/parity changed original model state')
        write(out/'initial_state.json',{'original_teacher_logical_state_sha256':original_teacher_state,'initial_R1_logical_state_sha256':base_state,'parent_training_state_sha256':PARENT_SHA,'parent_optimizer_admission':optimizer_admission,'frozen_generator_codebook_sha256':frozen_state,'checkpoint_sha256':WEIGHT_SHA,'fresh_app_parity':parity,'teacher_limitation':CONTRACT['teacher_limitation'],'hardware':hardware,'torch':torch.__version__,'counts':counts})
        baseline=evaluate(0,p['monitor_cases'],'monitor');challenge_baseline=evaluate(0,p['challenge_cases'],'challenge');evaluate(0,p['development_cases'],'public_development');empty_controls(0)
        torch.set_rng_state(resume_rng['CPU']);torch.cuda.set_rng_state_all(resume_rng['CUDA']);random.setstate(resume_rng['Python'])
        nr=resume_rng['NumPy'];np.random.set_state((nr['algorithm'],np.asarray(nr['keys'],dtype=np.uint32),nr['position'],nr['has_gauss'],nr['cached_gaussian']))
        restored_cuda=torch.cuda.get_rng_state_all()
        need(torch.equal(torch.get_rng_state(),resume_rng['CPU']) and len(restored_cuda)==len(resume_rng['CUDA'])==1 and all(torch.equal(a,b) for a,b in zip(restored_cuda,resume_rng['CUDA'])) and random.getstate()==resume_rng['Python'],'ParentRNG restoration differs')
        actual_nr=np.random.get_state()
        need(actual_nr[0]==nr['algorithm'] and np.array_equal(actual_nr[1],np.asarray(nr['keys'],dtype=np.uint32)) and actual_nr[2]==nr['position'] and actual_nr[3]==nr['has_gauss'] and actual_nr[4]==nr['cached_gaussian'],'FullNumPyRNG restoration differs')
        def json_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        def bytes_digest(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        expected_rng_hashes={'CPU':bytes_digest(resume_rng['CPU']),'CUDA':[bytes_digest(value) for value in resume_rng['CUDA']],'Python':json_digest(resume_rng['Python']),'NumPy':json_digest(nr)}
        actual_nr_dict={'algorithm':actual_nr[0],'keys':actual_nr[1].tolist(),'position':actual_nr[2],'has_gauss':actual_nr[3],'cached_gaussian':actual_nr[4]}
        restored_rng_hashes={'CPU':bytes_digest(torch.get_rng_state()),'CUDA':[bytes_digest(value) for value in restored_cuda],'Python':json_digest(random.getstate()),'NumPy':json_digest(actual_nr_dict)}
        need(expected_rng_hashes==restored_rng_hashes,'Parent RNG boundary hashes differ')
        parent_RNG_restored=True;parent_metadata['RNG_boundary_hashes']={'expected':expected_rng_hashes,'restored':restored_rng_hashes}
        write(out/'RNG_transition.json',{'complete':True,'parent_terminal_sha256':PARENT_SHA,'parent_RNG_restored':True,'expected_hashes':expected_rng_hashes,'restored_hashes':restored_rng_hashes,'new_sampler_prematerialized':True,'newphase_cycles':64,'global_rng_not_consumed_for_sampler':True})
        monitors=[baseline]
        for cycle,case_ids in enumerate(schedule,1):
            progress.update(cycle=cycle,route='code',microbatch=0,microbatches_backward_complete=0,phase='accumulating')
            guard();active(code_groups);code_optimizer.zero_grad(set_to_none=True);code_losses=[]
            for micro,ident in enumerate(case_ids,1):
                progress['microbatch']=micro
                case=fit_cases[ident];rgb=image(case['input256'],256);m=mask(case['mask']);x,_=app_input(rgb,m)
                logits,lq=forward('fit_code',lambda:net(x,code_only=True));latent,codes=teacher[case['teacher_key']]
                ce=F.cross_entropy(logits.reshape(-1,512),codes.reshape(-1));latent_loss=F.mse_loss(lq,latent);loss=(ce+latent_loss)/4
                need(torch.isfinite(loss).item(),'Nonfinite code objective');counts['backward_calls']+=1;progress['phase']='backward_started';loss.backward();progress.update(phase='accumulating',microbatches_backward_complete=micro);code_losses.append({'id':ident,'CE':float(ce.detach().item()),'latent_MSE':float(latent_loss.detach().item())})
                del logits,lq,ce,latent_loss,loss;guard()
            code_gradient=gradient_receipt(code_groups);code_clip=float(torch.nn.utils.clip_grad_norm_(code_parameters,1.,error_if_nonfinite=True).item());progress['phase']='optimizer_step_started';code_optimizer.step();counts['code_optimizer_steps']+=1;code_optimizer.zero_grad(set_to_none=True)
            need(all(torch.isfinite(parameter).all().item() for parameter in code_parameters),'Nonfinite updated code state')
            progress.update(route='fusion',microbatch=0,microbatches_backward_complete=0,phase='accumulating')
            active(fusion_groups);fusion_optimizer.zero_grad(set_to_none=True);fusion_losses=[]
            for micro,ident in enumerate(case_ids,1):
                progress['microbatch']=micro
                case=fit_cases[ident];rgb=image(case['input256'],256);target=clean[case['teacher_key']][1];m=mask(case['mask']);x,m512=app_input(rgb,m)
                raw,_,_=forward('fit_RGB',lambda:net(x,w=1,detach_16=True,adain=False));prediction=(raw+1)/2;b=boundary(m512)
                rgb_loss=(abs(prediction-target)*m512).sum()/(m512.sum()*3);seam=(abs(prediction-target)*b).sum()/(b.sum()*3);loss=(rgb_loss+seam)/4
                need(torch.isfinite(loss).item(),'Nonfinite fusion objective');counts['backward_calls']+=1;progress['phase']='backward_started';loss.backward();progress.update(phase='accumulating',microbatches_backward_complete=micro);fusion_losses.append({'id':ident,'mask_RGB512_L1':float(rgb_loss.detach().item()),'inner_boundary_RGB512_L1':float(seam.detach().item())})
                del raw,prediction,rgb_loss,seam,loss;guard()
            fusion_gradient=gradient_receipt(fusion_groups);fusion_clip=float(torch.nn.utils.clip_grad_norm_(fusion_parameters,1.,error_if_nonfinite=True).item());progress['phase']='optimizer_step_started';fusion_optimizer.step();counts['fusion_optimizer_steps']+=1;fusion_optimizer.zero_grad(set_to_none=True)
            need(all(torch.isfinite(parameter).all().item() for parameter in fusion_parameters),'Nonfinite updated fusion state')
            terminal_cycle=cycle;progress.update(phase='paired_cycle_complete',microbatch=4)
            row={'cycle':cycle,'microbatch_ids':case_ids,'code_losses':code_losses,'fusion_losses':fusion_losses,'code_gradient_groups':code_gradient,'fusion_gradient_groups':fusion_gradient,'code_preclip_L2':code_clip,'fusion_preclip_L2':fusion_clip,'counts':dict(counts),'seconds':time.monotonic()-started}
            fit_log.append(row);write(out/('training_cycle_%04d.json'%cycle),row)
            if cycle in (16,32,64):
                need(digest_state(('quantize.','generator.'))==frozen_state,'Frozen generator/codebook changed')
                snapshots.append({'cycle':cycle,'logical_state_sha256':digest_state(),'state_saved':False,'recoverable_checkpoint':False,'scope':'Images/metrics only. Not selectable or resumable; terminal state only is saved.'})
                summary=evaluate(cycle,p['monitor_cases'],'monitor');monitors.append(summary);screen=compare(summary,baseline);write(out/('screen_u%04d.json'%cycle),screen);empty_controls(cycle)
                if screen['catastrophic_regression_stop']:
                    stop_reason='Prospective10percent monitor macro/geometry hole-or-boundary regression';break
            print(json.dumps({'cycle':cycle,'maximum':64,'counts':counts,'seconds':time.monotonic()-started}),flush=True)
        challenge_final=evaluate(terminal_cycle,p['challenge_cases'],'challenge');evaluate(terminal_cycle,p['development_cases'],'public_development')
        clear_screen=compare(monitors[-1],baseline);challenge_screen=compare(challenge_final,challenge_baseline,conditions=True)
        cumulative_original=compare(monitors[-1],read(root/p['original_monitor_summary']))
        final_screen={'prospective_numeric_screen_pass':bool(clear_screen['prospective_numeric_screen_pass'] and challenge_screen['prospective_numeric_screen_pass']),'clear_incremental_vs_R1':clear_screen,'challenge_incremental_vs_R1':challenge_screen,'clear_cumulative_vs_original':cumulative_original,'cohorts_never_pooled':True,'model_qualified':False}
        need(digest_state(('quantize.','generator.'))==frozen_state and sha(checkpoint)==ORIGINAL_SHA and sha(parent_path)==PARENT_SHA,'Original or parent assets changed')
        if terminal_cycle==64 and not stop_reason:
            need(counts=={'forward_slots':896,'backward_calls':512,'code_optimizer_steps':64,'fusion_optimizer_steps':64,'empty_bypasses':168},'Completedrun actualcounts differ from plan')
            need(model_forwards['calls']==736,'Completedwholemodel count differs')
        terminal_state=save_state(terminal_cycle,'completed_finite_pilot' if not stop_reason else 'quality_stopped_terminal')
        final={'format':'completion-codeformer-coverage-v2-results','complete':True,'protocol_sha256':expected,'cycles_completed_newphase':terminal_cycle,'cycles_completed_cumulative':32+terminal_cycle,'stop_reason':stop_reason,'numeric_screen':final_screen,'metric_snapshots':snapshots,'terminal_training_state':terminal_state,'earlier_snapshots_recoverable':False,'initial_R1_logical_state_sha256':base_state,'original_teacher_logical_state_sha256':original_teacher_state,'final_logical_state_sha256':digest_state(),'frozen_state_unchanged':True,'original_checkpoint_unchanged':True,'parent_terminal_unchanged':True,'counts_newphase':counts,'counts_cumulative':{'backward_calls':256+counts['backward_calls'],'code_optimizer_steps':32+counts['code_optimizer_steps'],'fusion_optimizer_steps':32+counts['fusion_optimizer_steps'],'optimizer_steps_total':64+counts['code_optimizer_steps']+counts['fusion_optimizer_steps'],'forward_slots':528+counts['forward_slots'],'actual_whole_model_forward_calls':448+model_forwards['calls']},'typed_forward_slots':typed,'actual_whole_model_forward_calls':model_forwards['calls'],'seconds':time.monotonic()-started,'source_selection':'Adaptive originalTRAIN FIT/MONITOR; no independent identityFINAL','teacher_limitation':CONTRACT['teacher_limitation'],'visual_review_pending':True,'user_acceptance_pending':True,'model_qualified':False,'automatic_app_promotion':False,'output_budget_bytes':{category:budget.total(category) for category in ('terminal','teacher','metadata','array','PNG')}}
        write(out/'results.json',final,emergency=True);guard(terminal_save=True)
        return 0 if final_screen['prospective_numeric_screen_pass'] and terminal_cycle==64 and not stop_reason else 2
    except BaseException as exc:
        if not (out/'terminal_training_state.pth').exists():
            try:snapshots.append(save_state(terminal_cycle,'failure_retained_current_partial_state'))
            except BaseException as save_error:
                if not (out/'terminal_save_failure.json').exists():write(out/'terminal_save_failure.json',{'error':repr(save_error),'progress':dict(progress),'counts':dict(counts),'note':'Failedpartial retained; state may not recover, neverauto-resume'},emergency=True)
        if not (out/'failure.json').exists():write(out/'failure.json',{'complete':False,'exception':repr(exc),'traceback':traceback.format_exc(),'last_full_cycle':terminal_cycle,'partial_progress':dict(progress),'counts':counts,'typed_forward_slots':typed,'snapshots':snapshots,'seconds':time.monotonic()-started,'model_qualified':False,'parent_optimizer_restored':parent_optimizer_restored,'parent_RNG_restored':parent_RNG_restored},emergency=True)
        raise
    finally:
        forward_hook.remove()
        signal.signal(signal.SIGTERM,prior_termination)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-transfer',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--worker',action='store_true')
    a=parser.parse_args();root=a.root.resolve(strict=True);p,checkpoint,parent=verify(root,a.protocol_sha)
    if a.verify_transfer:
        print(json.dumps({'complete':True,'files':len(p['files']),'fit_cases':256,'monitor_cases':32,'challenge_cases':16,'teacher_target_sets':80,'optimizer_updates':0,'neural_calls':0}));return 0
    if a.worker:return worker(root,p,checkpoint,parent,a.protocol_sha)
    hardware=require_vm(root);need(not(root/'outputs').exists() and not(root/'supervisor.json').exists(),'Pilot already attempted; preserve all evidence')
    command=[sys.executable,'-B','-u',str(root/'scripts/cctv_completion_codeformer_coverage_v2_vm.py'),'--root',str(root),'--protocol-sha',a.protocol_sha,'--worker']
    started=time.monotonic();code=1;child=None;timeout=False;interruption=None;shutdown_mode=None;captured=0;log_cap_hit=False
    def interrupted(signum,frame):raise KeyboardInterrupt('Supervisor signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,interrupted)
    def kill_owned(sig):
        if child is None or child.poll() is not None:return
        try:
            need(os.getpgid(child.pid)==child.pid,'Owned process group differs; refuse unrelated signal');os.killpg(child.pid,sig)
        except ProcessLookupError:pass
    def stop_owned(grace):
        if child is not None and child.poll() is None:
            kill_owned(signal.SIGTERM)
            try:child.wait(timeout=max(0,min(grace,1800-(time.monotonic()-started))))
            except subprocess.TimeoutExpired:kill_owned(signal.SIGKILL);child.wait()
    try:
        with (root/'trainer.log').open('xb') as log:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
            print(json.dumps({'child_pid':child.pid,'log':'trainer.log','external_cap_seconds':1800}),flush=True)
            selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
            try:
                while selector.get_map():
                    if time.monotonic()-started>=1800:
                        timeout=True;shutdown_mode='hard_cap_SIGKILL';kill_owned(signal.SIGKILL);child.wait();code=124;break
                    for key,_ in selector.select(timeout=.1):
                        chunk=os.read(key.fileobj.fileno(),16384)
                        if not chunk:selector.unregister(key.fileobj);continue
                        remaining=CONTRACT['trainer_log_bytes']-captured
                        if len(chunk)>remaining:
                            if remaining:log.write(chunk[:remaining]);captured+=remaining
                            log_cap_hit=True;shutdown_mode='stdout_bytecap_SIGTERM';stop_owned(15);code=125;break
                        log.write(chunk);log.flush();captured+=len(chunk)
                    if log_cap_hit:break
                if not timeout and not log_cap_hit:code=child.wait(timeout=max(0,1800-(time.monotonic()-started)))
            finally:selector.close();child.stdout.close()
    except subprocess.TimeoutExpired:
        timeout=True;shutdown_mode='hard_cap_SIGKILL';kill_owned(signal.SIGKILL)
        if child is not None:child.wait()
        code=124
    except BaseException as exc:
        interruption=repr(exc);shutdown_mode='interruption_SIGTERM_then_bounded_wait';stop_owned(15);code=130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        if child is not None and child.poll() is None:kill_owned(signal.SIGKILL);child.wait()
        signal.signal(signal.SIGTERM,previous)
        receipt={'complete':True,'worker_exit':code,'actual_child_exit':None if child is None else child.returncode,'child_pid':None if child is None else child.pid,'timeout':timeout,'interruption':interruption,'shutdown_mode':shutdown_mode,'external_cap_seconds':1800,'seconds':time.monotonic()-started,'hardware':hardware,'terminal_state_present':(root/'outputs/terminal_training_state.pth').is_file(),'terminal_partial_present':(root/'outputs/terminal_training_state.pth.partial').is_file(),'stdout_capture_bytes':captured,'stdout_cap_bytes':CONTRACT['trainer_log_bytes'],'stdout_cap_reached':log_cap_hit,'stdout_prefix_retained_if_capped':True,'state_recoverability':'Pendingindependentreturnaudit;hardkillmaypreventsave;neverautomaticresume','training_success_not_implied':True}
        body=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode();need(len(body)<=1024*1024,'Supervisor receipt bytecap')
        with (root/'supervisor.json').open('xb') as stream:stream.write(body)
    print(json.dumps({'worker_exit':code,'timeout':timeout,'stdout_cap_reached':log_cap_hit,'log':'trainer.log','model_qualified':False}));return code
if __name__=='__main__':raise SystemExit(main())

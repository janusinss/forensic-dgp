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
from research.completion_codeformer_delivered_objective_v4.contracts import (CONTRACT,VM_NAME,ORIGINAL_SHA,PARENT_SHA,PARENT_PROTOCOL_SHA,PARENT_LOGICAL,ORIGINAL_LOGICAL,KINDS,METRICS,aggregate,compare as compare_metrics)
from research.completion_codeformer_delivered_objective_v4.optimized_budget import OutputBudget
from research.completion_codeformer_delivered_objective_v4.delivered_loss import delivered_float,delivered_hole_loss,clipping_diagnostic,OBJECTIVE_ID
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
def read(path):
    def reject(value):raise ValueError('Nonfinite JSON '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)
def write(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
def member(root,relative):
    need(isinstance(relative,str) and relative and '\\' not in relative and ':' not in relative,'Invalid relative packet member')
    name=Path(relative);need(not name.is_absolute() and '..' not in name.parts,'Packet traversal')
    path=root/name
    for part in [path,*path.parents]:
        need(not part.is_symlink(),'Symlink member refused')
        if part==root:break
    need(path.resolve().is_relative_to(root),'Member escaped packet')
    return path

def verify(root,expected):
    root=root.resolve(strict=True);need((root/'protocol.json').stat().st_size<=1024**2 and sha(root/'protocol.json')==expected,'Protocol size/SHA differs')
    p=read(root/'protocol.json');need(p['format']=='completion-codeformer-delivered-objective-v4' and p['contract']==CONTRACT,'Fusion-only-v3 contract differs')
    files={r['path']:r for r in p['files']};need(len(files)==len(p['files']),'Duplicate pinned file')
    for row in files.values():
        path=member(root,row['path']);need(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Packet drift: '+row['path'])
    for name in list(SOURCE_MODULES.values())+['scripts/cctv_completion_codeformer_delivered_objective_v4_vm.py','research/completion_codeformer_delivered_objective_v4/contracts.py','research/completion_codeformer_delivered_objective_v4/optimized_budget.py','research/completion_codeformer_delivered_objective_v4/delivered_loss.py']:
        need(name in files,'Imported/worker source is not pinned: '+name)
    for name in (p['data_manifest'],p['original_parent_protocol'],p['original_monitor_summary'],*p['evidence'].values()):need(name in files,'Unpinned evidence')
    need(sha(member(root,p['original_parent_protocol']))==PARENT_PROTOCOL_SHA,'Original R1 protocol binding differs')
    evidence_expected={'design_review':'9566e77f7fb33b4116f744d043ccf5af1ff133222fb69d6ad0b047db324a4682','data_admission':'7c072ec5a916468275e4e5ba3e3124e9b472030014f1e7afed803f5beca3d63a','root_visual_review':'8488e9354a87ce186931c2770a9ce6a8de0da06b92baddba2c6d593aea54db72','parent_state_audit':'e920c3ddd9b8a85b50fd6f324bde8760c13c6df805f6be61d1e4393472d59b06','route_design_review':'c565e8aa1362378b45c948463fe6018562858ff217f7f66b375161b6117ca289','writer_admission':'07eb9a55e2aa5dd737629e1695a15f8499833183d88d92987a2e45ea29249541','prior_route_review':'f6ec0010289d426a3c651733a9567c5f53ea5c778ab24bdff110f1fe75ad66dd','objective_diagnosis':'89396f99bb21cd57743e3549e201ae8a12f0a514cff0baf63d74421923144a51','FIT256_targets':'4dd1c4b8088fc20104937d8b9efebb4aca0c9b6292d3bf2a54973fc2e5921741','objective_independent_review':'c3fc9c2235517b088e963ef8756656b17c52c2bd229e20bdd56f9c0595f88dfc','design_admission':'888ae1368f62b7e150e152a6b16fc46aa47f1fb0fd8efd1a3e385db597190aea'}
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
    historical=p['historical_control']
    need(historical['protocol'] in files and sha(member(root,historical['protocol']))=='c838f4472cf87c138b2d518ef4f543bfcec3e8149ea7a5411b863397bf66859d','Historical joint protocol differs')
    for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
        rows=historical['initial_rows'][kind]
        need(len(rows)==len(p[key]) and {r['id'] for r in rows}=={r['id'] for r in p[key]},'Historical initial cohort differs')
        for row in rows:
            need(row['result256'] in files and row['PNG'] in files,'Unpinned historical initial array/PNG')
        need(historical['initial'][kind] in files and historical['joint_terminal'][kind] in files,'Unpinned historical summaries')
        for stage in ('initial','joint_terminal'):
            saved=read(member(root,historical[stage][kind]));need({r['id'] for r in saved['records']}=={r['id'] for r in p[key]},'Historical summary membership differs')
    for kind,rows in historical['joint_rows'].items():
        need(kind in ('monitor','challenge','public_development'),'Unexpected historical control kind')
        need(all(row['PNG'] in files for row in rows),'Unpinned historical joint PNG')
        key={'monitor':'monitor_cases','challenge':'challenge_cases','public_development':'development_cases'}[kind]
        need(len(rows)==len(p[key]) and {r['id'] for r in rows}=={r['id'] for r in p[key]},'Historical joint PNG membership differs')
    need(historical['joint_terminal']['public_development'] in files,'Unpinned historical public summary')
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
    typed={'parity':0,'fit_RGB':0,'monitor':0,'challenge':0,'public_development':0}
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
        need(counts['forward_slots']<=600 and counts['backward_calls']<=256,'Finite typed/backward cap')
        need(counts['code_optimizer_steps']==0 and counts['fusion_optimizer_steps']<=64,'Finite phase optimizer cap')
        need(torch.cuda.memory_allocated()<=CONTRACT['CUDA_allocated_bytes_cap'],'CUDA memory cap')
    def forward(kind,call):
        need(not budget.terminal_committed,'No neural calls after terminal commit')
        need(time.monotonic()-started<CONTRACT['cooperative_stop_seconds'],'No neural calls after cooperative work deadline')
        guard();counts['forward_slots']+=1;typed[kind]+=1;guard();return call()
    def digest_state(prefixes=None,*,check_finite=True):
        digest=hashlib.sha256()
        for name,value in sorted(net.state_dict().items()):
            if prefixes and not any(name.startswith(prefix) for prefix in prefixes):continue
            array=value.detach().cpu().contiguous()
            if check_finite:need(torch.isfinite(array).all().item(),'Nonfinite model state')
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
        return delivered_float(raw,rgb,m,interpolate=F.interpolate)
    def save_pixels(path,value):
        a=np.rint(value.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        with budget.open(path) as stream:Image.fromarray(a).save(stream,format="PNG")
        return a
    def boundary(m):return (m-(1-F.max_pool2d(1-m,7,1,3))).clamp(0,1)
    references={r['id']:r for r in p['references']};clean={}
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
    snapshots=[];terminal_cycle=0;stop_reason=None;parity=[];fit_log=[];parent_optimizer_restored=False;parent_RNG_restored=False;parent_metadata={};resume_rng=None;frozen_all_state=None;code_optimizer_baseline=None;phase_events=[]
    progress={'cycle':0,'route':'initialization','microbatch':0,'phase':'before_first_update'}
    model_forwards={'calls':0}
    def model_hook(module,args):model_forwards['calls']+=1
    forward_hook=net.register_forward_pre_hook(model_hook)
    def tree_digest(value,*,check_finite=True):
        digest=hashlib.sha256()
        def visit(v):
            if isinstance(v,torch.Tensor):
                a=v.detach().cpu().contiguous()
                if check_finite:need(torch.isfinite(a).all().item(),'Nonfinite recovery tensor')
                digest.update(str(a.dtype).encode());digest.update(str(tuple(a.shape)).encode());digest.update(a.numpy().tobytes())
            elif isinstance(v,dict):
                for key in sorted(v,key=lambda x:(str(type(x)),str(x))):digest.update(repr(key).encode());visit(v[key])
            elif isinstance(v,(list,tuple)):
                digest.update(type(v).__name__.encode())
                for x in v:visit(x)
            else:digest.update(repr(v).encode())
        visit(value);return digest.hexdigest()
    def frozen_digest(*,check_finite=True):
        return tree_digest({name:value for name,value in net.state_dict().items() if not name.startswith('fuse_convs_dict.')},check_finite=check_finite)
    def phase_start(name,*,emergency=False):
        value={'phase':name,'start_seconds':time.monotonic()-started,'complete':False}
        phase_events.append(value);write(out/('phase_%03d_%s_start.json'%(len(phase_events),name)),value,emergency=emergency)
        return value
    def phase_end(value,**details):
        value.update(end_seconds=time.monotonic()-started,complete=True,**details);value['seconds']=value['end_seconds']-value['start_seconds']
        write(out/('phase_%03d_%s_end.json'%(phase_events.index(value)+1,value['phase'])),value,emergency=True)
    def cpu_copy(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu_copy(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
        return value
    def tensor_bytes(value):
        if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(tensor_bytes(v) for v in value)
        return 0
    def save_state(cycle,reason):
        actual_prior=digest_state(('quantize.','generator.'),check_finite=False)
        actual_frozen=frozen_digest(check_finite=False)
        actual_code_Adam=tree_digest(code_optimizer.state_dict(),check_finite=False)
        current_logical=digest_state(check_finite=False)
        retention_failures=[]
        if actual_prior!=frozen_state:retention_failures.append('original generator/codebook state changed')
        if frozen_all_state is not None and actual_frozen!=frozen_all_state:retention_failures.append('443frozen state changed')
        if code_optimizer_baseline is not None and actual_code_Adam!=code_optimizer_baseline:retention_failures.append('frozen code Adam changed')
        if not all(torch.isfinite(v).all().item() for v in net.state_dict().values()):retention_failures.append('nonfinite current model state')
        assembly=phase_start('terminal_CPU_copy_and_assembly',emergency=True)
        delta={name:value for name,value in net.state_dict().items() if not name.startswith(('quantize.','generator.'))}
        pending={'trainable_state':delta,'failed_frozen_prior_state':({n:v for n,v in net.state_dict().items() if n.startswith(('quantize.','generator.'))} if actual_prior!=frozen_state else {}),'code_optimizer':code_optimizer.state_dict(),'fusion_optimizer':fusion_optimizer.state_dict(),'partial_accumulated_gradients':{name:parameter.grad for name,parameter in net.named_parameters() if parameter.grad is not None}}
        planned_tensor_bytes=tensor_bytes(pending)
        need(planned_tensor_bytes<CONTRACT['terminal_bytes'],'Planned terminal tensor payload exceeds bound')
        write(out/'pre_save_receipt.json',{'complete':False,'reason':reason,'cycle':cycle,'counts':dict(counts),'logical_state_sha256':current_logical,'frozen443_sha256':frozen_all_state,'code_Adam_sha256':code_optimizer_baseline,'planned_tensor_payload_bytes':planned_tensor_bytes,'free_disk_bytes':shutil.disk_usage(root).free,'remaining_hard_seconds':max(0.,1800-(time.monotonic()-started)),'ledger_totals':{k:budget.total(k) for k in ('terminal','teacher','metadata','PNG','array')},'serialization_started':False,'retained_failure_gates':retention_failures},emergency=True)
        snapshot=cpu_copy(pending)
        snapshot.update({'format':'completion-codeformer-delivered-objective-v4-trainable-resume-v1','parent_training_state_sha256':PARENT_SHA,'parent_protocol_sha256':PARENT_PROTOCOL_SHA,'parent_initial_logical_sha256':PARENT_LOGICAL,'parent_optimizer_restored':parent_optimizer_restored,'parent_RNG_restored':parent_RNG_restored,'parent_metadata':parent_metadata,'phase_completed_paired_cycles':terminal_cycle,'cumulative_completed_paired_cycles':32+terminal_cycle,'cumulative_counts':{'backward_calls':256+counts['backward_calls'],'code_optimizer_steps':32,'fusion_optimizer_steps':32+counts['fusion_optimizer_steps'],'optimizer_steps_total':64+counts['fusion_optimizer_steps'],'forward_slots':528+counts['forward_slots'],'actual_whole_model_forward_calls':448+model_forwards['calls']},'phase_name':'delivered_objective_v4','base_checkpoint_sha256':WEIGHT_SHA,'protocol_sha256':expected,'cycle':cycle,'progress':dict(progress),'resume_progress_requires_new_manual_audit':True,'optimizer_parameter_names':optimizer_parameter_names,'completed_paired_cycles':32+terminal_cycle,'frozen_state_sha256':frozen_state,'frozen443_sha256':frozen_all_state,'code_optimizer_unchanged_sha256':code_optimizer_baseline,'logical_state_sha256':current_logical,'actual_frozen443_sha256':actual_frozen,'actual_prior_sha256':actual_prior,'actual_code_Adam_sha256':actual_code_Adam,'retained_failure_gates':retention_failures,'state_is_qualified':False,'torch_CPU_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state_all(),'python_random_state':random.getstate(),'numpy_random_state':{'algorithm':np.random.get_state()[0],'keys':np.random.get_state()[1].tolist(),'position':np.random.get_state()[2],'has_gauss':np.random.get_state()[3],'cached_gaussian':np.random.get_state()[4]},'sampler_schedule':schedule,'next_cycle':terminal_cycle+1,'next_clean_sampler_index':terminal_cycle*4,'counts':dict(counts),'contract':CONTRACT,'reason':reason,'automatic_resume_permitted':False,'model_qualified':False})
        phase_end(assembly,tensor_payload_bytes=planned_tensor_bytes)
        path=out/'terminal_training_state.pth';need(not path.exists(),'Terminal state already exists')
        partial=out/'terminal_training_state.pth.partial'
        serial=phase_start('terminal_serialization',emergency=True);before_callbacks=budget.check_counts['write_checks']
        with budget.open(partial,terminal=True) as stream:
            torch.save(snapshot,stream)
            serialize_end=time.monotonic()-started
            flush_start=time.monotonic();stream.flush();os.fsync(stream.stream.fileno());flush_seconds=time.monotonic()-flush_start
        phase_end(serial,serialized_bytes=partial.stat().st_size,write_callbacks=budget.check_counts['write_checks']-before_callbacks,serializer_finished_seconds=serialize_end,flush_fsync_seconds=flush_seconds)
        guard(terminal_save=True);commit=phase_start('terminal_hash_and_atomic_commit',emergency=True)
        digest=sha(partial);need(not path.exists(),'Committed terminal state already exists')
        os.link(partial,path);need(path.stat().st_ino==partial.stat().st_ino,'Terminal exclusive publication identity differs')
        # Remove only the verified successful alias; failed .partial files never reach this branch.
        partial.unlink();budget.rename(partial,path);budget.assert_ledger_consistent()
        phase_end(commit,bytes=path.stat().st_size,sha256=digest,ledger_total=budget.total())
        return {'path':path.name,'bytes':path.stat().st_size,'sha256':digest,'cycle':cycle,'logical_state_sha256':current_logical,'frozen443_sha256':frozen_all_state,'phase_timings':phase_events,'retained_failure_gates':retention_failures}
    def evaluate(cycle,rows,kind):
        evaluation_phase=phase_start('evaluate_%s_u%04d'%(kind,cycle))
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
        write(folder/'results.json',summary);phase_end(evaluation_phase,cases=len(records),counts=dict(counts));return summary
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
        write(out/'initialization_timing.json',{'complete':True,'phase':'imports_original_model_initialization_and_optimizer_construction','start_seconds':0.,'end_seconds':time.monotonic()-started,'typed_neural_calls':0,'optimizer_steps':0})
        need(original_teacher_state==ORIGINAL_LOGICAL,'Original base logical state differs')
        target_phase=phase_start('styled_target_loading_no_neural_teacher')
        for row in p['teacher_targets']:
            c256=image(row['clean256'],256);c512=image(row['clean512'],512);clean[row['id']]=(c256,c512)
            if row['style']=='GRAY':need(torch.equal(c256[:,0],c256[:,1]) and torch.equal(c256[:,1],c256[:,2]) and torch.equal(c512[:,0],c512[:,1]) and torch.equal(c512[:,1],c512[:,2]),'GRAY targets not exactgray')
        phase_end(target_phase,target_sets=len(clean),teacher_neural_calls=0)
        parent_phase=phase_start('parent_state_load')
        parent=torch.load(parent_path,map_location='cpu',weights_only=True)
        phase_end(parent_phase,parent_bytes=parent_path.stat().st_size)
        reconstruction_phase=phase_start('R1_state_reconstruction')
        need(parent['protocol_sha256']==PARENT_PROTOCOL_SHA and parent['base_checkpoint_sha256']==ORIGINAL_SHA and parent['logical_state_sha256']==PARENT_LOGICAL,'Parent fingerprint differs')
        need(parent['cycle']==parent['completed_paired_cycles']==32 and parent['partial_accumulated_gradients']=={} and parent['reason']=='completed_finite_pilot','Only audited completedR1parent allowed')
        need(parent['optimizer_parameter_names']==optimizer_parameter_names,'Named optimizer order differs')
        need(parent['counts']=={'forward_slots':528,'backward_calls':256,'code_optimizer_steps':32,'fusion_optimizer_steps':32,'empty_bypasses':168},'Parent actualtraining counters differ')
        parent_protocol=read(root/p['original_parent_protocol']);need(parent['contract']==parent_protocol['contract'] and parent['sampler_schedule']==parent_protocol['schedule'],'Parent contract/sampler differs')
        original_state=net.state_dict();expected_names={n for n in original_state if not n.startswith(('quantize.','generator.'))}
        need(set(parent['trainable_state'])==expected_names and len(expected_names)==332,'Parent trainable partition differs')
        net.load_state_dict({**original_state,**parent['trainable_state']},strict=True)
        need(digest_state()==PARENT_LOGICAL and digest_state(('quantize.','generator.'))==frozen_state==parent['frozen_state_sha256'],'R1 reconstruction/frozen prior differs')
        phase_end(reconstruction_phase,logical_state_sha256=digest_state())
        optimizer_phase=phase_start('named_Adam_restore')
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
        need(len(code_parameters)==278 and len(fusion_parameters)==54 and len(net.state_dict())==497,'Expected443frozen/54trainable partition')
        frozen_all_state=frozen_digest();code_optimizer_baseline=tree_digest(code_optimizer.state_dict())
        need(sum(not name.startswith('fuse_convs_dict.') for name in net.state_dict())==443,'Frozen state tensor count differs')
        parent_optimizer_restored=True
        phase_end(optimizer_phase,frozen_tensors=443,trainable_fusion_tensors=54,code_Adam_digest=code_optimizer_baseline,frozen443_digest=frozen_all_state)
        resume_rng={'CPU':parent['torch_CPU_RNG'].clone(),'CUDA':[x.clone() for x in parent['CUDA_RNG']],'Python':parent['python_random_state'],'NumPy':parent['numpy_random_state']}
        parent_metadata={'terminal_sha256':PARENT_SHA,'protocol_sha256':PARENT_PROTOCOL_SHA,'logical_state_sha256':PARENT_LOGICAL,'counts':parent['counts'],'sampler_schedule':parent['sampler_schedule'],'completed_cycles':32,'optimizer_admission':optimizer_admission}
        del parent,original_state,state,prior
        base_state=digest_state();progress.update(phase='R1weights_Adam_restored_before_parity_RNG_pending')
        parity_phase=phase_start('fresh_R1_app_parity')
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
        del wrapper;phase_end(parity_phase,pairs=len(parity),max_error=max(r['max_raw256_error'] for r in parity))
        need(digest_state()==base_state,'Initial teacher/parity changed original model state')
        write(out/'initial_state.json',{'original_base_logical_state_sha256':original_teacher_state,'initial_R1_logical_state_sha256':base_state,'parent_training_state_sha256':PARENT_SHA,'parent_optimizer_admission':optimizer_admission,'frozen_generator_codebook_sha256':frozen_state,'checkpoint_sha256':WEIGHT_SHA,'fresh_app_parity':parity,'teacher_neural_calls':0,'hardware':hardware,'torch':torch.__version__,'counts':counts})
        baseline=evaluate(0,p['monitor_cases'],'monitor');challenge_baseline=evaluate(0,p['challenge_cases'],'challenge');evaluate(0,p['development_cases'],'public_development');empty_controls(0)
        compatibility=phase_start('historical_R1_48case_compatibility')
        compatibility_rows=[]
        for kind,summary in [('monitor',baseline),('challenge',challenge_baseline)]:
            historical={r['id']:r for r in p['historical_control']['initial_rows'][kind]}
            for record in summary['records']:
                row=historical[record['id']]
                with np.load(root/row['result256'],allow_pickle=False) as archive:
                    need(set(archive.files)=={'result256'},'Compact historical array schema differs');prior=archive['result256']
                need(prior.dtype==np.float32 and prior.shape==(1,3,256,256) and np.isfinite(prior).all(),'Compact historical values invalid')
                with np.load(out/Path(record['PNG']).with_suffix('.npz'),allow_pickle=False) as archive:current=archive['result256']
                error=float(np.max(np.abs(current-prior)));png_exact=sha(out/record['PNG'])==sha(root/row['PNG'])
                compatibility_rows.append({'kind':kind,'id':record['id'],'max_raw256_error':error,'PNG_bytes_exact':png_exact,'historical_result256_sha256':sha(root/row['result256']),'historical_PNG_sha256':sha(root/row['PNG'])})
        write(out/'historical_R1_compatibility.json',{'complete':True,'cases':compatibility_rows,'all48_before_update':len(compatibility_rows)==48,'raw_tolerance':1e-6,'updates_before':counts['code_optimizer_steps']+counts['fusion_optimizer_steps'],'pass':len(compatibility_rows)==48 and all(r['max_raw256_error']<=1e-6 and r['PNG_bytes_exact'] for r in compatibility_rows)})
        phase_end(compatibility,cases=len(compatibility_rows))
        need(len(compatibility_rows)==48 and all(r['max_raw256_error']<=1e-6 and r['PNG_bytes_exact'] for r in compatibility_rows),'Historical R1 baseline incompatible; retain stop before updates')
        RNG_phase=phase_start('parent_RNG_optimization_boundary')
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
        phase_end(RNG_phase,hashes=restored_rng_hashes)
        write(out/'objective_contract.json',{'objective':OBJECTIVE_ID,'loss_source':'exact pinned clean256, existing clean[teacher_key][0]','delivery':'clamp raw512 RGB then bilinear align_corners=False to256, active-index compose source','boundary_loss_added':False,'all_boundary_acceptance_gates_retained':True,'microbatch_divisor_once':4,'PNG_rounding_or_STE_in_training':False,'clipping_metadata_is_detached_diagnostic_only':True,'R1_incumbent_and_failed_V2_V3_preserved':True})
        monitors=[baseline]
        for cycle,case_ids in enumerate(schedule,1):
            progress.update(cycle=cycle,route='fusion',microbatch=0,microbatches_backward_complete=0,phase='accumulating')
            guard()
            progress.update(route='fusion',microbatch=0,microbatches_backward_complete=0,phase='accumulating')
            active(fusion_groups);fusion_optimizer.zero_grad(set_to_none=True);fusion_losses=[]
            for micro,ident in enumerate(case_ids,1):
                progress['microbatch']=micro
                case=fit_cases[ident];rgb=image(case['input256'],256);target=clean[case['teacher_key']][0];m=mask(case['mask']);x,m512=app_input(rgb,m)
                if case['style']=='GRAY':need(torch.equal(rgb[:,0],rgb[:,1]) and torch.equal(rgb[:,1],rgb[:,2]) and torch.equal(target[:,0],target[:,1]) and torch.equal(target[:,1],target[:,2]),'FIT gray input/256target style mismatch')
                raw,_,_=forward('fit_RGB',lambda:net(x,w=1,detach_16=True,adain=False))
                prediction,rgb_loss,loss=delivered_hole_loss(raw,rgb,m,target,interpolate=F.interpolate,microbatches=4)
                clipping=clipping_diagnostic(raw,m512)
                need(torch.isfinite(loss).item(),'Nonfinite delivered256 hole objective');counts['backward_calls']+=1;progress['phase']='backward_started';loss.backward();progress.update(phase='accumulating',microbatches_backward_complete=micro)
                fusion_losses.append({'id':ident,'style':case['style'],'clean256':case['clean256'],'teacher_key':case['teacher_key'],'objective':OBJECTIVE_ID,'mask_RGB256_L1':float(rgb_loss.detach().item()),'scaled_microbatch_loss':float(loss.detach().item()),'microbatch_divisor':4,'separate_boundary_loss_added':False,'PNG_rounding_or_STE_in_training':False,'clipping':clipping})
                del raw,prediction,rgb_loss,loss,clipping;guard()
            fusion_gradient=gradient_receipt(fusion_groups);fusion_clip=float(torch.nn.utils.clip_grad_norm_(fusion_parameters,1.,error_if_nonfinite=True).item());progress['phase']='optimizer_step_started';fusion_optimizer.step();counts['fusion_optimizer_steps']+=1;fusion_optimizer.zero_grad(set_to_none=True)
            need(all(torch.isfinite(parameter).all().item() for parameter in fusion_parameters),'Nonfinite updated fusion state')
            need(all(parameter.grad is None for parameter in code_parameters),'Frozen code route accumulated a gradient')
            terminal_cycle=cycle;progress.update(phase='fusion_cycle_complete',microbatch=4)
            row={'cycle':cycle,'microbatch_ids':case_ids,'code_losses':[],'code_route_frozen':True,'fusion_losses':fusion_losses,'code_gradient_groups':{},'fusion_gradient_groups':fusion_gradient,'code_preclip_L2':None,'fusion_preclip_L2':fusion_clip,'fusion_gradient_clipping_applied':fusion_clip>1.,'objective':OBJECTIVE_ID,'counts':dict(counts),'seconds':time.monotonic()-started}
            fit_log.append(row);write(out/('training_cycle_%04d.json'%cycle),row)
            if cycle in (16,32,64):
                need(digest_state(('quantize.','generator.'))==frozen_state and frozen_digest()==frozen_all_state and tree_digest(code_optimizer.state_dict())==code_optimizer_baseline,'Frozen443 or code Adam changed')
                snapshots.append({'cycle':cycle,'logical_state_sha256':digest_state(),'state_saved':False,'recoverable_checkpoint':False,'scope':'Images/metrics only. Not selectable or resumable; terminal state only is saved.'})
                summary=evaluate(cycle,p['monitor_cases'],'monitor');monitors.append(summary);screen=compare(summary,baseline);write(out/('screen_u%04d.json'%cycle),screen);empty_controls(cycle)
                if screen['catastrophic_regression_stop']:
                    stop_reason='Prospective10percent monitor macro/geometry hole-or-boundary regression';break
            print(json.dumps({'cycle':cycle,'maximum':64,'counts':counts,'seconds':time.monotonic()-started}),flush=True)
        challenge_final=evaluate(terminal_cycle,p['challenge_cases'],'challenge');evaluate(terminal_cycle,p['development_cases'],'public_development')
        clear_screen=compare(monitors[-1],baseline);challenge_screen=compare(challenge_final,challenge_baseline,conditions=True)
        cumulative_original=compare(monitors[-1],read(root/p['original_monitor_summary']))
        historical_joint={kind:compare(summary,read(root/p['historical_control']['joint_terminal'][kind]),conditions=kind=='challenge') for kind,summary in [('monitor',monitors[-1]),('challenge',challenge_final)]}
        historical_joint['comparison_scope']={'new_actual_terminal_cycle':terminal_cycle,'historical_joint_cycle':64,'same_case_exposure':terminal_cycle==64,'historical_control_is_not_a_new_simultaneous_run':True,'quality_gates_still_vs_R1':True}
        final_screen={'prospective_numeric_screen_pass':bool(clear_screen['prospective_numeric_screen_pass'] and challenge_screen['prospective_numeric_screen_pass']),'clear_incremental_vs_R1':clear_screen,'challenge_incremental_vs_R1':challenge_screen,'clear_cumulative_vs_original':cumulative_original,'historical_joint_V2_diagnostic_only':historical_joint,'cohorts_never_pooled':True,'model_qualified':False}
        need(digest_state(('quantize.','generator.'))==frozen_state and frozen_digest()==frozen_all_state and tree_digest(code_optimizer.state_dict())==code_optimizer_baseline and sha(checkpoint)==ORIGINAL_SHA and sha(parent_path)==PARENT_SHA,'Frozen443/codeAdam/original/parent changed')
        if terminal_cycle==64 and not stop_reason:
            need(counts=={'forward_slots':480,'backward_calls':256,'code_optimizer_steps':0,'fusion_optimizer_steps':64,'empty_bypasses':168},'Completedrun actualcounts differ from plan')
            need(model_forwards['calls']==480,'Completedwholemodel count differs')
        for route,optimizer,target_step in [('code',code_optimizer,32),('fusion',fusion_optimizer,32+counts['fusion_optimizer_steps'])]:
            need(all(float(row['step'].item())==target_step and torch.isfinite(row['exp_avg']).all().item() and torch.isfinite(row['exp_avg_sq']).all().item() for row in optimizer.state_dict()['state'].values()),'Final named Adam step/moments differ: '+route)
        terminal_state=save_state(terminal_cycle,'completed_finite_pilot' if not stop_reason else 'quality_stopped_terminal')
        final={'format':'completion-codeformer-delivered-objective-v4-results','complete':True,'protocol_sha256':expected,'cycles_completed_newphase':terminal_cycle,'cycles_completed_cumulative':32+terminal_cycle,'stop_reason':stop_reason,'numeric_screen':final_screen,'metric_snapshots':snapshots,'terminal_training_state':terminal_state,'earlier_snapshots_recoverable':False,'initial_R1_logical_state_sha256':base_state,'original_base_logical_state_sha256':original_teacher_state,'final_logical_state_sha256':digest_state(),'frozen_state_unchanged':True,'frozen443_sha256':frozen_all_state,'code_Adam_unchanged_sha256':code_optimizer_baseline,'original_checkpoint_unchanged':True,'parent_terminal_unchanged':True,'counts_newphase':counts,'counts_cumulative':{'backward_calls':256+counts['backward_calls'],'code_optimizer_steps':32,'fusion_optimizer_steps':32+counts['fusion_optimizer_steps'],'optimizer_steps_total':64+counts['fusion_optimizer_steps'],'forward_slots':528+counts['forward_slots'],'actual_whole_model_forward_calls':448+model_forwards['calls']},'typed_forward_slots':typed,'actual_whole_model_forward_calls':model_forwards['calls'],'seconds':time.monotonic()-started,'source_selection':'Adaptive originalTRAIN FIT/MONITOR; no independent identityFINAL','teacher_neural_calls':0,'objective':OBJECTIVE_ID,'boundary_loss_added':False,'PNG_rounding_or_STE_in_training':False,'visual_review_pending':True,'user_acceptance_pending':True,'model_qualified':False,'automatic_app_promotion':False,'output_budget_bytes':{category:budget.total(category) for category in ('terminal','teacher','metadata','array','PNG')}}
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
        print(json.dumps({'complete':True,'files':len(p['files']),'fit_cases':256,'monitor_cases':32,'challenge_cases':16,'styled_target_sets':80,'teacher_neural_calls':0,'optimizer_updates':0,'neural_calls':0}));return 0
    if a.worker:return worker(root,p,checkpoint,parent,a.protocol_sha)
    hardware=require_vm(root);need(not(root/'outputs').exists() and not(root/'supervisor.json').exists(),'Pilot already attempted; preserve all evidence')
    command=[sys.executable,'-B','-u',str(root/'scripts/cctv_completion_codeformer_delivered_objective_v4_vm.py'),'--root',str(root),'--protocol-sha',a.protocol_sha,'--worker']
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

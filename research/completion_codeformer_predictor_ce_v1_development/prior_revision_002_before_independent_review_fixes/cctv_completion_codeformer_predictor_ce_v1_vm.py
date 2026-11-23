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
from research.completion_codeformer_predictor_ce_v1.contracts import (CONTRACT,VM_NAME,ORIGINAL_SHA,PARENT_SHA,PARENT_PROTOCOL_SHA,PARENT_LOGICAL,ORIGINAL_LOGICAL,KINDS,METRICS,aggregate,compare as compare_metrics)
from research.completion_codeformer_predictor_ce_v1.optimized_budget import OutputBudget
from research.completion_codeformer_predictor_ce_v1.app_delivery import delivered_float
from research.completion_codeformer_predictor_ce_v1.contracts import OBJECTIVE_ID,SUBSET_SHA,predictor_name,TYPED_CAPS,gray_style,compare_gray
from research.completion_codeformer_predictor_ce_v1.optimizer_topology import project,merge,ordered_ids
DESIGN_SHA="UNBOUND_PREPARATORY_DESIGN_NOT_RELEASED"
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
    root=root.resolve(strict=True)
    need(len(DESIGN_SHA)==64,'PREPARATORY SOURCE ONLY: formal design/source admission not bound; no execution')
    need((root/'protocol.json').stat().st_size<=1024**2 and sha(root/'protocol.json')==expected,'Protocol size/SHA differs')
    p=read(root/'protocol.json');need(p['format']=='completion-codeformer-predictor-ce-v1' and p['contract']==CONTRACT,'Predictor CE contract differs')
    files={r['path']:r for r in p['files']};need(len(files)==len(p['files']),'Duplicate pinned file')
    for row in files.values():
        path=member(root,row['path']);need(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Pinned packet drift '+row['path'])
    required=list(SOURCE_MODULES.values())+['scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py']+[f'research/completion_codeformer_predictor_ce_v1/{n}.py' for n in ('contracts','optimized_budget','app_delivery','optimizer_topology')]
    for name in required:need(name in files,'Unpinned imported source '+name)
    for name in [p['balanced_subset'],p['original_parent_protocol'],*p['evidence'].values()]:need(name in files,'Unpinned design/data evidence')
    need(sha(member(root,p['evidence']['root_design']))==DESIGN_SHA,'Formal finite design differs')
    need(sha(member(root,p['balanced_subset']))==SUBSET_SHA,'Independent balanced subset differs')
    subset=read(member(root,p['balanced_subset']));need(subset['complete'] and subset['subset_cases']==128 and subset['source_full_pool_cases']==256 and p['fit_cases']==subset['cases_in_reference_role_order'] and p['schedule']==subset['schedule'],'Use exact independent balanced128 schedule/cases')
    need(sha(member(root,p['original_parent_protocol']))==PARENT_PROTOCOL_SHA,'Original R1 source binding')
    refs={r['id']:r for r in p['references']};need(len(refs)==40,'Exactly40 original TRAIN sources')
    fit={r['id'] for r in refs.values() if r['role']=='fit'};monitor={r['id'] for r in refs.values() if r['role']=='monitor'}
    need(len(fit)==32 and len(monitor)==8 and not fit&monitor and all(r['original_role']=='train' and r['legacy_completion_split']=='train' for r in refs.values()),'Exact original32FIT8monitor split')
    need(len(p['teacher_targets'])==80 and len(p['cached_labels'])==80,'Reuse all80 retained same-codebook labels only')
    teachers={r['id']:r for r in p['teacher_targets']};labels={r['teacher_key']:r for r in p['cached_labels']}
    need(set(teachers)==set(labels)=={ref+'::'+style for ref in refs for style in ('RGB','GRAY')},'Exact64FIT16monitor labels')
    for key,row in labels.items():
        target=teachers[key];need(row['path'] in files and row['sha256']==files[row['path']]['sha256'] and row['bytes']==files[row['path']]['bytes'] and row['clean512']==target['clean512'] and row['clean512_sha256']==files[target['clean512']]['sha256'] and row['teacher_original_logical_sha256']==ORIGINAL_LOGICAL,'Exact original label/source binding')
        need(target['clean256'] in files and target['clean512'] in files,'Style matched target pins')
    need(len(p['parity_cases'])==4,'Four explicit unchanged fresh parity cases')
    for key,count,roles in [('fit_cases',128,fit),('monitor_cases',32,monitor),('challenge_cases',16,monitor)]:
        rows=p[key];need(len(rows)==count and len({r['id'] for r in rows})==count,'Exact case count/order')
        for row in rows:
            need(re.fullmatch(r'[A-Za-z0-9_-]{1,100}',row['id']) and row['reference_id'] in roles and row['teacher_key'] in labels,'Case split/key safe')
            for field in ('input256','clean256','clean512','mask'):need(row[field] in files,'Unpinned fixed case asset')
    for row in p['parity_cases']+p['development_cases']:
        need(re.fullmatch(r'[A-Za-z0-9_-]{1,100}',row['id']) and row['input256'] in files and row['mask'] in files,'Pinned parity/public case')
    need(len(p['development_cases'])==32 and len({r['id'] for r in p['development_cases']})==32,'Public32 fixed case set')
    need(len(p['schedule'])==32 and all(len(v)==4 for v in p['schedule']) and set(sum(p['schedule'],[]))=={r['id'] for r in p['fit_cases']} and len(set(sum(p['schedule'],[])))==128,'128unique FIT cases exactlyonce')
    for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
        rows=p['historical_control']['initial_rows'][kind];need([r['id'] for r in rows]==[r['id'] for r in p[key]],'Historical R1 case order exact')
        for row in rows:need(row['result256'] in files and row['PNG'] in files,'Pinned historical R1 controls')
    def external(row,name,want_sha,want_bytes):
        raw=root/row['path'];need(not raw.is_symlink(),'Linked external file refused');path=raw.resolve(strict=True)
        need(path.is_relative_to((Path.home()/'forensic-dgp').resolve()) and path.name==name and path.is_file() and row['sha256']==want_sha and row['bytes']==want_bytes and path.stat().st_size==want_bytes and sha(path)==want_sha,'External original/R1 evidence differs');return path
    checkpoint=external(p['checkpoint'],'codeformer_inpainting.pth',ORIGINAL_SHA,370786611);parent=external(p['parent'],'terminal_training_state.pth',PARENT_SHA,701952165)
    need(p['parent']['protocol_sha256']==PARENT_PROTOCOL_SHA and p['parent']['logical_sha256']==PARENT_LOGICAL,'Only full audited R1 parent')
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
    need(shutil.disk_usage(root).free>=CONTRACT['minimum_free_bytes'],'Need8GiB free before pilot')
    out=root/'outputs';need(not out.exists(),'Never overwrite/resume this finite historical pilot');out.mkdir()
    started=time.monotonic();counts={'forward_slots':0,'backward_calls':0,'predictor_optimizer_steps':0,'empty_bypasses':0}
    typed={kind:0 for kind in TYPED_CAPS}
    budget=OutputBudget(out,started);budget.metadata_external=[root/'trainer.log',root/'protocol.json']
    def write(path,value,*,emergency=False):budget.json(path,value,emergency=emergency)
    saved_npz_count=0
    def save_npz(path,**arrays):
        nonlocal saved_npz_count
        need(saved_npz_count<388,'Exact maximum388 saved numeric arrays')
        relative=path.relative_to(out)
        if relative.parts[0]=='FIT_logits':expected_keys={'logits','labels','case_index','cycle','microbatch'}
        elif relative.parts[0]=='fresh_app_parity':expected_keys={'input256','mask256','own_input512','app_input512','own_raw512','app_raw512','own_codes','app_codes','own_result256','app_result256'}
        else:expected_keys={'input256','mask256','result256','raw512','predicted_codes'}
        need(set(arrays)==expected_keys and len(arrays)<=10,'Exact numeric schema; no silent output expansion')
        for key,value in arrays.items():
            need(isinstance(value,np.ndarray) and value.dtype in (np.float32,np.int64) and not value.dtype.hasobject,'Only exact numeric arrays')
            if key in ('case_index','cycle','microbatch'):need(value.dtype==np.int64 and value.shape==(),'Exact scalar FIT metadata')
            elif key in ('labels','predicted_codes','own_codes','app_codes'):need(value.dtype==np.int64 and value.shape in ((1,256),(0,)) and (value.size==0 or value.min()>=0 and value.max()<512),'Exact selected-code geometry')
            elif key=='logits':need(value.dtype==np.float32 and value.shape==(1,256,512) and np.isfinite(value).all(),'Finite saved FITlogits')
            else:
                shape=(1,1,256,256) if key=='mask256' else (1,3,512,512) if key.endswith('512') else (1,3,256,256)
                need(value.dtype==np.float32 and (value.shape==shape or key=='raw512' and value.shape==(0,)) and np.isfinite(value).all(),'Finite exact raw/input/mask/result geometry')
        with budget.open(path) as stream:np.savez_compressed(stream,**arrays)
        saved_npz_count+=1
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
        need(counts['forward_slots']<=420 and counts['backward_calls']<=128 and counts['predictor_optimizer_steps']<=32 and counts['empty_bypasses']<=168,'Finite phase caps')
        need(all(typed[k]<=TYPED_CAPS[k] for k in typed),'Per-type finite neural caps')
        need(torch.cuda.memory_allocated()<=CONTRACT['CUDA_allocated_bytes_cap'],'CUDA memory cap')
    def forward(kind,call):
        need(not budget.terminal_committed,'No neural calls after terminal commit')
        need(time.monotonic()-started<CONTRACT['cooperative_stop_seconds'],'No neural calls after cooperative deadline')
        guard();need(kind in typed and counts['forward_slots']+1<=420 and typed[kind]+1<=TYPED_CAPS[kind],'Refuse neural dispatch beyond typed/hard cap')
        need(model_forwards['calls']==counts['forward_slots'],'Actual prior dispatch count exact')
        counts['forward_slots']+=1;typed[kind]+=1;value=call()
        need(model_forwards['calls']==counts['forward_slots'],'Actual forward hook count exact');guard();return value
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
    references={r['id']:r for r in p['references']};clean={};teacher={}
    fit_cases={r['id']:r for r in p['fit_cases']};schedule=p['schedule']
    all_parameters=dict(net.named_parameters());predictor_names=[name for name in all_parameters if predictor_name(name)]
    predictor_parameters=[all_parameters[n] for n in predictor_names]
    predictor_groups={'position':[net.position_emb],'feature':list(net.feat_emb.parameters()),'transformer':list(net.ft_layers.parameters()),'prediction':list(net.idx_pred_layer.parameters())}
    original_optimizer_names={'code':[name for name in all_parameters if name.startswith('encoder.')]+predictor_names,'fusion':[name for name in all_parameters if name.startswith('fuse_convs_dict.')]}
    need(len(predictor_names)==114 and len(original_optimizer_names['code'])==278 and len(original_optimizer_names['fusion'])==54,'Exact runtime predictor/original topology')
    need(sum(not predictor_name(name) for name in net.state_dict())==383 and len(net.state_dict())==497,'Exact383 frozen tensor partition')
    predictor_optimizer=torch.optim.Adam(predictor_parameters,lr=1e-5,betas=(.9,.99),eps=1e-8,weight_decay=0.)
    retained_code_optimizer=None;retained_fusion_optimizer=None;retained_encoder_Adam_digest=None;retained_fusion_Adam_digest=None
    def active_predictor():
        net.requires_grad_(False)
        for name in predictor_names:all_parameters[name].requires_grad_(True)
        need({n for n,v in net.named_parameters() if v.requires_grad}==set(predictor_names),'Optimizer requires_grad allowlist exact')
    def gradient_receipt(groups):
        receipt={}
        for name,parameters in groups.items():
            need(all(param.grad is not None for param in parameters),'Disconnected intended group '+name)
            need(all(torch.isfinite(param.grad).all().item() for param in parameters),'Nonfinite gradients '+name)
            norm=math.sqrt(sum(float(param.grad.detach().double().square().sum().item()) for param in parameters))
            receipt[name]={'L2':norm,'tensors':len(parameters),'finite':True,'nonzero':norm>0}
        return receipt
    snapshots=[];terminal_cycle=0;stop_reason=None;parity=[];fit_log=[];parent_optimizer_restored=False;parent_RNG_restored=False;parent_metadata={};resume_rng=None;frozen_all_state=None;predictor_optimizer_baseline=None;phase_events=[];terminal_attempted=False
    progress={'cycle':0,'route':'initialization','microbatch':0,'phase':'before_first_update'}
    model_forwards={'calls':0,'generator_calls':0,'code_only_calls':0}
    def model_hook(module,args,kwargs):
        model_forwards['calls']+=1;model_forwards['code_only_calls' if kwargs.get('code_only',False) else 'generator_calls']+=1
    forward_hook=net.register_forward_pre_hook(model_hook,with_kwargs=True)
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
        return tree_digest({name:value for name,value in net.state_dict().items() if not predictor_name(name)},check_finite=check_finite)
    def state_receipt():
        tensors={}
        for name,value in sorted(net.state_dict().items()):
            cpu=value.detach().cpu().contiguous();tensors[name]={'shape':list(cpu.shape),'dtype':str(cpu.dtype),'sha256':hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),'finite':bool(torch.isfinite(cpu).all().item()),'frozen_in_newphase':not predictor_name(name)}
        return {'logical_sha256':digest_state(check_finite=False),'tensors':tensors,'finite':all(v['finite'] for v in tensors.values()),'state_tensors':len(tensors),'frozen_tensors':sum(v['frozen_in_newphase'] for v in tensors.values())}
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
        nonlocal terminal_attempted
        need(not terminal_attempted,'Only one bounded terminal save attempt; retain any partial evidence');terminal_attempted=True
        actual_prior=digest_state(('quantize.','generator.'),check_finite=False);actual_frozen=frozen_digest(check_finite=False);current_logical=digest_state(check_finite=False)
        if not (out/'actual_state_at_save.json').exists():write(out/'actual_state_at_save.json',state_receipt(),emergency=True)
        retention_failures=[]
        if actual_prior!=frozen_state:retention_failures.append('original generator/codebook changed')
        if frozen_all_state is not None and actual_frozen!=frozen_all_state:retention_failures.append('383frozen tensors changed')
        if not all(torch.isfinite(v).all().item() for v in net.state_dict().values()):retention_failures.append('nonfinite current model')
        need(retained_code_optimizer is not None and retained_fusion_optimizer is not None,'Parent optimizer evidence unavailable: do not invent recovery')
        assembly=phase_start('terminal_CPU_copy_and_assembly',emergency=True)
        merged=merge(retained_code_optimizer,original_optimizer_names['code'],predictor_optimizer.state_dict(),predictor_names,clone=cpu_copy)
        encoder_ids=ordered_ids(retained_code_optimizer,original_optimizer_names['code'])
        actual_encoder_Adam=tree_digest({n:merged['state'][encoder_ids[n]] for n in original_optimizer_names['code'] if n.startswith('encoder.')},check_finite=False)
        actual_fusion_Adam=tree_digest(retained_fusion_optimizer,check_finite=False)
        if actual_encoder_Adam!=retained_encoder_Adam_digest:retention_failures.append('inactive encoder Adam changed')
        if actual_fusion_Adam!=retained_fusion_Adam_digest:retention_failures.append('inactive fusion Adam changed')
        delta={n:v for n,v in net.state_dict().items() if not n.startswith(('quantize.','generator.'))}
        pending={'trainable_state':delta,'failed_frozen_prior_state':({n:v for n,v in net.state_dict().items() if n.startswith(('quantize.','generator.'))} if actual_prior!=frozen_state else {}),'code_optimizer':merged,'fusion_optimizer':retained_fusion_optimizer,'partial_accumulated_gradients':{n:v.grad for n,v in net.named_parameters() if v.grad is not None}}
        planned_tensor_bytes=tensor_bytes(pending);need(planned_tensor_bytes<CONTRACT['terminal_bytes'],'Planned complete recovery payload exceeds bound')
        write(out/'pre_save_receipt.json',{'complete':False,'reason':reason,'cycle':cycle,'counts':dict(counts),'logical_state_sha256':current_logical,'frozen383_sha256':frozen_all_state,'planned_tensor_payload_bytes':planned_tensor_bytes,'remaining_hard_seconds':max(0.,1800-(time.monotonic()-started)),'serialization_started':False,'retained_failure_gates':retention_failures},emergency=True)
        snapshot=cpu_copy(pending)
        snapshot.update({'format':'completion-codeformer-predictor-ce-v1-trainable-resume-v1','parent_training_state_sha256':PARENT_SHA,'parent_protocol_sha256':PARENT_PROTOCOL_SHA,'parent_initial_logical_sha256':PARENT_LOGICAL,'parent_optimizer_restored':parent_optimizer_restored,'parent_RNG_restored':parent_RNG_restored,'parent_metadata':parent_metadata,'phase_completed_predictor_steps':terminal_cycle,'phase_name':'predictor_ce_v1','base_checkpoint_sha256':WEIGHT_SHA,'protocol_sha256':expected,'cycle':cycle,'progress':dict(progress),'resume_progress_requires_new_manual_audit':True,'optimizer_parameter_names':original_optimizer_names,'active_optimizer_parameter_names':predictor_names,'active_optimizer_slots_stored_only_in_full_code_optimizer':True,'full_code_optimizer_is_historical_mixed_step_container_not_active278_optimizer':True,'expected_mixed_steps':{'inactive_encoder164':32,'active_predictor114':32+counts['predictor_optimizer_steps'],'inactive_fusion54':32},'completed_paired_cycles':32,'completed_predictor_steps_newphase':terminal_cycle,'frozen_state_sha256':frozen_state,'frozen383_sha256':frozen_all_state,'logical_state_sha256':current_logical,'actual_frozen383_sha256':actual_frozen,'actual_prior_sha256':actual_prior,'inactive_encoder_Adam_sha256':actual_encoder_Adam,'inactive_fusion_Adam_sha256':actual_fusion_Adam,'retained_failure_gates':retention_failures,'state_is_qualified':False,'torch_CPU_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state_all(),'python_random_state':random.getstate(),'numpy_random_state':{'algorithm':np.random.get_state()[0],'keys':np.random.get_state()[1].tolist(),'position':np.random.get_state()[2],'has_gauss':np.random.get_state()[3],'cached_gaussian':np.random.get_state()[4]},'sampler_schedule':schedule,'next_cycle':terminal_cycle+1,'next_clean_sampler_index':terminal_cycle*4,'counts':dict(counts),'contract':CONTRACT,'reason':reason,'automatic_resume_permitted':False,'model_qualified':False})
        phase_end(assembly,tensor_payload_bytes=planned_tensor_bytes)
        path=out/'terminal_training_state.pth';need(not path.exists(),'Terminal state already exists');partial=out/'terminal_training_state.pth.partial'
        serial=phase_start('terminal_serialization',emergency=True);before_callbacks=budget.check_counts['write_checks']
        with budget.open(partial,terminal=True) as stream:
            torch.save(snapshot,stream);serialize_end=time.monotonic()-started;flush_start=time.monotonic();stream.flush();os.fsync(stream.stream.fileno());flush_seconds=time.monotonic()-flush_start
        phase_end(serial,serialized_bytes=partial.stat().st_size,write_callbacks=budget.check_counts['write_checks']-before_callbacks,serializer_finished_seconds=serialize_end,flush_fsync_seconds=flush_seconds)
        guard(terminal_save=True);commit=phase_start('terminal_hash_and_atomic_commit',emergency=True);digest=sha(partial);need(not path.exists(),'Committed state already exists')
        os.link(partial,path);need(path.stat().st_ino==partial.stat().st_ino,'Exclusive terminal publication identity differs');partial.unlink();budget.rename(partial,path);budget.assert_ledger_consistent();phase_end(commit,bytes=path.stat().st_size,sha256=digest,ledger_total=budget.total())
        return {'path':path.name,'bytes':path.stat().st_size,'sha256':digest,'cycle':cycle,'logical_state_sha256':current_logical,'frozen383_sha256':frozen_all_state,'retained_failure_gates':retention_failures}
    def evaluate(cycle,rows,kind):
        evaluation_phase=phase_start('evaluate_%s_u%04d'%(kind,cycle))
        folder=out/('u%04d'%cycle)/kind;folder.mkdir(parents=True,exist_ok=False);net.requires_grad_(False);records=[]
        with torch.no_grad():
            for case in rows:
                rgb=image(case['input256'],256);target=image(case['clean256'],256) if kind!='public_development' else None
                m=mask(case['mask']);guard()
                if not m.any().item():
                    counts['empty_bypasses']+=1;out_tensor=rgb.clone();raw=None;predicted=None
                else:
                    x,m512=app_input(rgb,m);raw,logits,_=forward(kind,lambda:net(x,w=1,detach_16=True,adain=False));_,indices=torch.topk(F.softmax(logits,dim=2),1,dim=2);predicted=indices.squeeze(-1);out_tensor=delivered(rgb,m,raw)
                need(torch.isfinite(out_tensor).all().item() and torch.equal(out_tensor*(1-m),rgb*(1-m)),'Visible raw preservation failed')
                array=save_pixels(folder/(case['id']+'.png'),out_tensor)
                observed=np.rint(rgb[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8);support=m[0,0].cpu().numpy().astype(bool)
                need(np.array_equal(array[~support],observed[~support]),'PNG visible preservation failed')
                record={'id':case['id'],'mask_kind':case.get('mask_kind'),'family':case.get('family'),'condition':case.get('condition'),'style':case.get('style'),'PNG':str((folder/(case['id']+'.png')).relative_to(out)),'PNG_sha256':sha(folder/(case['id']+'.png')),'outside_exact':True,'empty':not bool(support.any())}
                if target is not None and support.any():
                    record['same_codebook_clean_context_agreement_diagnostic']=float((predicted==teacher[case['teacher_key']]).float().mean().item())
                    b=boundary(m);need(b.sum().item()>0,'Empty boundary');target_array=np.rint(target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8);difference=abs(out_tensor-target)
                    record.update({'raw_hole_L1':float((difference*m).sum().item()/(m.sum().item()*3)),'raw_boundary_L1':float((difference*b).sum().item()/(b.sum().item()*3)),'PNG_hole_L1':float(abs(array.astype(np.float64)-target_array)[support].mean()/255),'PNG_boundary_L1':float(abs(array.astype(np.float64)-target_array)[b[0,0].cpu().numpy().astype(bool)].mean()/255)})
                    if case.get('style')=='GRAY':
                        raw_rgb=out_tensor[0].permute(1,2,0).cpu().numpy();record['GRAY_raw_mean_channel_spread']=float(np.ptp(raw_rgb,axis=-1)[support].mean());record['GRAY_PNG_mean_channel_spread']=float(np.ptp(array.astype(np.int16),axis=-1)[support].mean());record['GRAY_chroma_diagnostic_not_selection_gate']=False;record['GRAY_final32_style_gate_required']=True
                save_npz(folder/(case['id']+'.npz'),input256=rgb.detach().cpu().numpy(),mask256=m.detach().cpu().numpy(),result256=out_tensor.detach().cpu().numpy(),raw512=raw.detach().cpu().numpy() if raw is not None else np.empty((0,),np.float32),predicted_codes=predicted.cpu().numpy() if predicted is not None else np.empty((0,),np.int64))
                records.append(record);guard()
        summary={'cycle':cycle,'kind':kind,'records':records,'counts':dict(counts),'model_qualified':False}
        if kind!='public_development':summary['metrics']=aggregate([r for r in records if not r['empty']],condition=kind=='challenge')
        if kind=='challenge':summary['GRAY_style']=gray_style(summary)
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
        write(out/'initialization_timing.json',{'complete':True,'phase':'imports_original_model_initialization_and_one_predictor_optimizer_construction','start_seconds':0.,'end_seconds':time.monotonic()-started,'typed_neural_calls':0,'optimizer_steps':0})
        need(original_teacher_state==ORIGINAL_LOGICAL,'Original base logical state differs')
        target_phase=phase_start('styled_target_loading_no_neural_teacher')
        for row in p['teacher_targets']:
            c256=image(row['clean256'],256);c512=image(row['clean512'],512);clean[row['id']]=(c256,c512)
            if row['style']=='GRAY':need(torch.equal(c256[:,0],c256[:,1]) and torch.equal(c256[:,1],c256[:,2]) and torch.equal(c512[:,0],c512[:,1]) and torch.equal(c512[:,1],c512[:,2]),'GRAY targets not exactgray')
        phase_end(target_phase,target_sets=len(clean),teacher_neural_calls=0)
        import zipfile
        for label in p['cached_labels']:
            labelpath=root/label['path']
            with zipfile.ZipFile(labelpath) as archive:
                need(len(archive.infolist())==2 and set(archive.namelist())=={'prequant_latent.npy','nearest_codes.npy'} and all(not v.is_dir() and not v.flag_bits&1 and 0<v.file_size<1024**2 and v.compress_type in (0,8) for v in archive.infolist()) and archive.testzip() is None,'Retained teacher numeric schema/CRC')
            with np.load(labelpath,allow_pickle=False,max_header_size=512) as archive:
                codes=archive['nearest_codes'];latent=archive['prequant_latent'];need(codes.dtype==np.int64 and codes.shape==(1,256) and codes.min()>=0 and codes.max()<512,'Existing same-codebook labels exact geometry/range');need(latent.dtype==np.float32 and latent.shape==(1,256,16,16) and np.isfinite(latent).all(),'Unused retained latent finite')
                teacher[label['teacher_key']]=torch.from_numpy(codes.copy()).to('cuda')
        need(len(teacher)==80,'Exact64FIT16monitor label cache')
        write(out/'cached_label_receipt.json',{'complete':True,'sets':80,'FIT_sets':64,'monitor_diagnostic_only_sets':16,'teacher_neural_calls':0,'new_labels_generated':False,'unused_prequant_latents_not_in_objective':True,'source_pins':p['cached_labels']})
        parent_phase=phase_start('parent_state_load')
        parent=torch.load(parent_path,map_location='cpu',weights_only=True)
        phase_end(parent_phase,parent_bytes=parent_path.stat().st_size)
        reconstruction_phase=phase_start('R1_state_reconstruction')
        need(parent['protocol_sha256']==PARENT_PROTOCOL_SHA and parent['base_checkpoint_sha256']==ORIGINAL_SHA and parent['logical_state_sha256']==PARENT_LOGICAL,'Parent fingerprint differs')
        need(parent['cycle']==parent['completed_paired_cycles']==32 and parent['partial_accumulated_gradients']=={} and parent['reason']=='completed_finite_pilot','Only audited completedR1parent allowed')
        need(parent['optimizer_parameter_names']==original_optimizer_names,'Named historical optimizer order differs')
        need(parent['counts']=={'forward_slots':528,'backward_calls':256,'code_optimizer_steps':32,'fusion_optimizer_steps':32,'empty_bypasses':168},'Parent actualtraining counters differ')
        parent_protocol=read(root/p['original_parent_protocol']);need(parent['contract']==parent_protocol['contract'] and parent['sampler_schedule']==parent_protocol['schedule'],'Parent contract/sampler differs')
        original_state=net.state_dict();expected_names={n for n in original_state if not n.startswith(('quantize.','generator.'))}
        need(set(parent['trainable_state'])==expected_names and len(expected_names)==332,'Parent trainable partition differs')
        net.load_state_dict({**original_state,**parent['trainable_state']},strict=True)
        need(digest_state()==PARENT_LOGICAL and digest_state(('quantize.','generator.'))==frozen_state==parent['frozen_state_sha256'],'R1 reconstruction/frozen prior differs')
        phase_end(reconstruction_phase,logical_state_sha256=digest_state())
        optimizer_phase=phase_start('named_predictor_subset_Adam_restore')
        need(parent['optimizer_parameter_names']==original_optimizer_names,'Original278code54fusion named order exact')
        retained_code_optimizer=cpu_copy(parent['code_optimizer']);retained_fusion_optimizer=cpu_copy(parent['fusion_optimizer'])
        for route,state in [('code',retained_code_optimizer),('fusion',retained_fusion_optimizer)]:
            ids=ordered_ids(state,original_optimizer_names[route]);group=state['param_groups'][0]
            need(group['lr']==1e-5 and tuple(group['betas'])==(.9,.99) and group['eps']==1e-8 and group['weight_decay']==0,'Parent exact Adam hyperparameters')
            for name,index in ids.items():
                slot=state['state'][index];parameter=all_parameters[name]
                need(set(slot)=={'step','exp_avg','exp_avg_sq'} and float(slot['step'].item())==32,'Parent32step/slot schema')
                for key in ('exp_avg','exp_avg_sq'):need(slot[key].shape==parameter.shape and slot[key].dtype==parameter.dtype==torch.float32 and torch.isfinite(slot[key]).all().item(),'Parent finite matching Adam slots')
        projected=project(retained_code_optimizer,original_optimizer_names['code'],predictor_names,clone=cpu_copy)
        predictor_optimizer.load_state_dict(projected)
        for name,parameter in zip(predictor_names,predictor_parameters):
            slot=predictor_optimizer.state[parameter];original=retained_code_optimizer['state'][ordered_ids(retained_code_optimizer,original_optimizer_names['code'])[name]]
            need(float(slot['step'].item())==32 and set(slot)==set(original),'Active114 copiedstep/schema')
            for key in ('exp_avg','exp_avg_sq'):need(slot[key].device==parameter.device and torch.equal(slot[key].cpu(),original[key]),'Copied114moments bitexactbyname')
        codeids=ordered_ids(retained_code_optimizer,original_optimizer_names['code'])
        retained_encoder_Adam_digest=tree_digest({n:retained_code_optimizer['state'][codeids[n]] for n in original_optimizer_names['code'] if n.startswith('encoder.')})
        retained_fusion_Adam_digest=tree_digest(retained_fusion_optimizer)
        frozen_all_state=frozen_digest();predictor_optimizer_baseline=tree_digest(predictor_optimizer.state_dict());parent_optimizer_restored=True
        optimizer_admission={'active_predictor114':{'step_before':32,'named_parent_projection_exact':True,'moments_exact':True},'inactive_encoder164':{'step':32,'sha256':retained_encoder_Adam_digest,'optimizer_constructed':False},'inactive_fusion54':{'step':32,'sha256':retained_fusion_Adam_digest,'optimizer_constructed':False}}
        phase_end(optimizer_phase,frozen_tensors=383,trainable_predictor_tensors=114,inactive_encoder_Adam=retained_encoder_Adam_digest,inactive_fusion_Adam=retained_fusion_Adam_digest)
        resume_rng={'CPU':parent['torch_CPU_RNG'].clone(),'CUDA':[x.clone() for x in parent['CUDA_RNG']],'Python':parent['python_random_state'],'NumPy':parent['numpy_random_state']}
        parent_metadata={'terminal_sha256':PARENT_SHA,'protocol_sha256':PARENT_PROTOCOL_SHA,'logical_state_sha256':PARENT_LOGICAL,'counts':parent['counts'],'sampler_schedule':parent['sampler_schedule'],'completed_cycles':32,'optimizer_admission':optimizer_admission}
        del parent,original_state,projected
        base_state=digest_state();write(out/'R1_state_before.json',state_receipt());progress.update(phase='R1weights_Adam_restored_before_parity_RNG_pending')
        parity_phase=phase_start('fresh_R1_app_parity')
        wrapper=importlib.import_module('pretrained_completion').CodeFormerCompletion(net)
        parity_cases=p['parity_cases']
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
                    captures[-1].update(raw=output[0].detach().cpu().numpy().copy(),codes=torch.topk(F.softmax(output[1],dim=2),1,dim=2).indices.squeeze(-1).detach().cpu().numpy().copy())
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
                receipt={'id':case['id'],'max_raw256_error':error,'max_model_input512_error':float(np.max(np.abs(own_capture['input']-app_capture['input']))),'input512_exact':bool(np.array_equal(own_capture['input'],app_capture['input'])),'input_stride_exact':own_capture['stride']==app_capture['stride'],'model_raw512_max_error':float(np.max(np.abs(own_capture['raw']-app_capture['raw']))),'selected_codes_exact':bool(np.array_equal(own_capture['codes'],app_capture['codes'])),'own_input_layout':{k:v for k,v in own_capture.items() if k not in ('input','raw','codes')},'app_input_layout':{k:v for k,v in app_capture.items() if k not in ('input','raw','codes')},'tolerance_raw256':1e-6,'counts':dict(counts),'optimizer_steps_before_parity':counts['predictor_optimizer_steps']}
                save_npz(parity_folder/(case['id']+'.npz'),input256=rgb.detach().cpu().numpy(),mask256=m.detach().cpu().numpy(),own_input512=own_capture['input'],app_input512=app_capture['input'],own_raw512=own_capture['raw'],app_raw512=app_capture['raw'],own_codes=own_capture['codes'],app_codes=app_capture['codes'],own_result256=own.detach().cpu().numpy(),app_result256=reference.detach().cpu().numpy())
                write(parity_folder/(case['id']+'.json'),receipt);parity.append(receipt)
                # Retain all four paired calls before aggregate parity assertions.
        write(parity_folder/'summary.json',{'complete':True,'cases':parity,'all_input_values_and_layout_exact':all(row['input512_exact'] and row['input_stride_exact'] for row in parity),'max_raw256_error':max(row['max_raw256_error'] for row in parity),'tolerance_raw256':1e-6,'counts':dict(counts),'training_updates':0})
        need(all(row['input512_exact'] and row['input_stride_exact'] for row in parity),'Fresh app input value/layout parity failed')
        need(all(row['max_raw256_error']<=1e-6 and row['selected_codes_exact'] for row in parity),'Fresh frozen app input/code pipeline parity failed')
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
        write(out/'historical_R1_compatibility.json',{'complete':True,'cases':compatibility_rows,'all48_before_update':len(compatibility_rows)==48,'raw_tolerance':1e-6,'updates_before':counts['predictor_optimizer_steps'],'pass':len(compatibility_rows)==48 and all(r['max_raw256_error']<=1e-6 and r['PNG_bytes_exact'] for r in compatibility_rows)})
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
        write(out/'RNG_transition.json',{'complete':True,'parent_terminal_sha256':PARENT_SHA,'parent_RNG_restored':True,'expected_hashes':expected_rng_hashes,'restored_hashes':restored_rng_hashes,'new_sampler_prematerialized':True,'newphase_cycles':32,'global_rng_not_consumed_for_sampler':True})
        phase_end(RNG_phase,hashes=restored_rng_hashes)
        write(out/'objective_contract.json',{'objective':OBJECTIVE_ID,'training_path':'covered input -> fixed R1 encoder -> active114 predictor -> logits CE','label_source':'existing original inpainting nearest-code indices only;64FITlabelsets,16monitorlabels neveroptimized','teacher_neural_calls':0,'mean_CE_indices':256,'codebook_size':512,'microbatch_divisor_once':4,'latent_MSE_added':False,'image_loss_added':False,'fusion_w_during_evaluation':1,'hard_top1_used_in_ordinary_evaluation':True,'oracle_inference':False,'all_old_quality_gates_retained':True,'FIT_logits_saved_for_independent_CE_replay':True})
        monitors=[baseline];challenges=[challenge_baseline];fit_saved=0
        fit_order={case['id']:index for index,case in enumerate(p['fit_cases'])}
        for cycle,case_ids in enumerate(schedule,1):
            progress.update(cycle=cycle,route='predictor',microbatch=0,microbatches_backward_complete=0,phase='accumulating')
            guard();active_predictor();predictor_optimizer.zero_grad(set_to_none=True);losses=[]
            for micro,ident in enumerate(case_ids,1):
                progress['microbatch']=micro;case=fit_cases[ident];need(references[case['reference_id']]['role']=='fit','No monitor labels optimized')
                rgb=image(case['input256'],256);m=mask(case['mask']);x,_=app_input(rgb,m)
                logits,lq=forward('fit_code',lambda:net(x,code_only=True));labels=teacher[case['teacher_key']]
                need(logits.shape==(1,256,512) and logits.dtype==torch.float32 and torch.isfinite(logits).all().item(),'Finite full FIT logits required')
                ce=F.cross_entropy(logits.reshape(-1,512),labels.reshape(-1),reduction='mean');loss=ce/4
                need(torch.isfinite(loss).item(),'Nonfinite CE objective')
                logpath=out/'FIT_logits'/('cycle_%04d_micro%d_%s.npz'%(cycle,micro,ident))
                save_npz(logpath,logits=logits.detach().cpu().numpy(),labels=labels.cpu().numpy(),case_index=np.asarray(fit_order[ident],dtype=np.int64),cycle=np.asarray(cycle,dtype=np.int64),microbatch=np.asarray(micro,dtype=np.int64));fit_saved+=1
                need(counts['backward_calls']+1<=128,'Refuse backward beyond finite128 cap');counts['backward_calls']+=1;progress['phase']='backward_started';loss.backward();progress.update(phase='accumulating',microbatches_backward_complete=micro)
                _,chosen=torch.topk(F.softmax(logits.detach(),dim=2),1,dim=2)
                losses.append({'id':ident,'reference_id':case['reference_id'],'teacher_key':case['teacher_key'],'style':case['style'],'CE_mean_all256':float(ce.detach().item()),'scaled_microbatch_loss':float(loss.detach().item()),'microbatch_divisor':4,'FIT_logits':str(logpath.relative_to(out)),'FIT_logits_sha256':sha(logpath),'clean_context_code_agreement_diagnostic':float((chosen.squeeze(-1)==labels).float().mean().item())})
                del logits,lq,ce,loss,labels;guard()
            gradients=gradient_receipt(predictor_groups)
            need(all(v.grad is None for n,v in net.named_parameters() if not predictor_name(n)),'Frozen383 parameters must have no gradient')
            clip=float(torch.nn.utils.clip_grad_norm_(predictor_parameters,1.,error_if_nonfinite=True).item());need(counts['predictor_optimizer_steps']+1<=32,'Refuse optimizer beyond32steps');progress['phase']='optimizer_step_started';predictor_optimizer.step();counts['predictor_optimizer_steps']+=1;predictor_optimizer.zero_grad(set_to_none=True)
            need(all(torch.isfinite(v).all().item() for v in predictor_parameters),'Nonfinite updated predictor');need(frozen_digest()==frozen_all_state,'383frozen tensors changed')
            terminal_cycle=cycle;progress.update(phase='predictor_cycle_complete',microbatch=4)
            row={'cycle':cycle,'microbatch_ids':case_ids,'predictor_losses':losses,'predictor_gradient_groups':gradients,'predictor_preclip_L2':clip,'gradient_clip_applied':clip>1,'objective':OBJECTIVE_ID,'counts':dict(counts),'seconds':time.monotonic()-started};fit_log.append(row);write(out/('training_cycle_%04d.json'%cycle),row)
            if cycle in (8,16,32):
                need(digest_state(('quantize.','generator.'))==frozen_state and frozen_digest()==frozen_all_state,'Frozen prior/383 invariant')
                snapshots.append({'cycle':cycle,'logical_state_sha256':digest_state(),'state_saved':False,'recoverable_checkpoint':False,'scope':'Images/metrics only; not selectable or resumable. Only terminal state saved.'})
                summary=evaluate(cycle,p['monitor_cases'],'monitor');challenge=evaluate(cycle,p['challenge_cases'],'challenge');monitors.append(summary);challenges.append(challenge)
                screen={'monitor':compare(summary,baseline),'challenge':compare(challenge,challenge_baseline,conditions=True),'early_selection_forbidden':cycle in (8,16),'only_catastrophic_preservation_stop_before32':True,'GRAY_style':compare_gray(challenge,challenge_baseline),'GRAY_gate_required_only_at32':True};write(out/('screen_u%04d.json'%cycle),screen);empty_controls(cycle)
                if any(screen[k]['catastrophic_regression_stop'] for k in ('monitor','challenge')):stop_reason='Prospective10percent monitor/challenge macro/geometry/condition hole-or-boundary regression';break
            print(json.dumps({'cycle':cycle,'maximum':32,'counts':counts,'seconds':time.monotonic()-started}),flush=True)
        if terminal_cycle not in (8,16,32):
            monitors.append(evaluate(terminal_cycle,p['monitor_cases'],'monitor'));challenges.append(evaluate(terminal_cycle,p['challenge_cases'],'challenge'))
        evaluate(terminal_cycle,p['development_cases'],'public_development')
        clear_screen=compare(monitors[-1],baseline);challenge_screen=compare(challenges[-1],challenge_baseline,conditions=True)
        gray_screen=compare_gray(challenges[-1],challenge_baseline)
        final_screen={'prospective_numeric_screen_pass':bool(clear_screen['prospective_numeric_screen_pass'] and challenge_screen['prospective_numeric_screen_pass'] and gray_screen['all_GRAY_macro_geometry_chroma_nonworse']),'clear_incremental_vs_R1':clear_screen,'challenge_incremental_vs_R1':challenge_screen,'GRAY_style_incremental_vs_R1':gray_screen,'GRAY_final32_required':True,'cohorts_never_pooled':True,'model_qualified':False}
        need(digest_state(('quantize.','generator.'))==frozen_state and frozen_digest()==frozen_all_state and sha(checkpoint)==ORIGINAL_SHA and sha(parent_path)==PARENT_SHA,'Frozen383/original/parent drift')
        if terminal_cycle==32 and not stop_reason:
            need(counts=={'forward_slots':384,'backward_calls':128,'predictor_optimizer_steps':32,'empty_bypasses':168} and typed==TYPED_CAPS,'Exact completed finite typed counts')
            need(model_forwards=={'calls':384,'generator_calls':256,'code_only_calls':128} and fit_saved==128 and saved_npz_count==388,'Actual model-hook/generator/code-only/FITlogit counts')
        need(all(float(v['step'].item())==32+counts['predictor_optimizer_steps'] for v in predictor_optimizer.state.values()),'All114active Adam steps exact')
        terminal_state=save_state(terminal_cycle,'completed_finite_pilot' if not stop_reason else 'quality_stopped_terminal')
        final={'format':'completion-codeformer-predictor-ce-v1-results','complete':True,'protocol_sha256':expected,'predictor_steps_completed_newphase':terminal_cycle,'historical_R1_paired_cycles':32,'stop_reason':stop_reason,'numeric_screen':final_screen,'metric_snapshots':snapshots,'terminal_training_state':terminal_state,'earlier_snapshots_recoverable':False,'initial_R1_logical_state_sha256':base_state,'original_base_logical_state_sha256':original_teacher_state,'final_logical_state_sha256':digest_state(),'frozen383_unchanged':True,'frozen383_sha256':frozen_all_state,'inactive_encoder_Adam_sha256':retained_encoder_Adam_digest,'inactive_fusion_Adam_sha256':retained_fusion_Adam_digest,'original_checkpoint_unchanged':True,'parent_terminal_unchanged':True,'counts_newphase':counts,'typed_forward_slots':typed,'actual_model_forward_counts':model_forwards,'FIT_logits_saved':fit_saved,'seconds':time.monotonic()-started,'source_selection':'Adaptive originalTRAIN FIT/MONITOR; no independent identityFINAL','teacher_neural_calls':0,'objective':OBJECTIVE_ID,'latent_MSE_added':False,'fusion_or_encoder_training':False,'oracle_inference':False,'visual_review_pending':True,'user_acceptance_pending':True,'model_qualified':False,'automatic_app_promotion':False,'output_budget_bytes':{category:budget.total(category) for category in ('terminal','teacher','metadata','array','PNG')}}
        write(out/'results.json',final,emergency=True);guard(terminal_save=True)
        return 0 if final_screen['prospective_numeric_screen_pass'] and terminal_cycle==32 and not stop_reason else 2
    except BaseException as exc:
        if not terminal_attempted and not (out/'terminal_training_state.pth').exists():
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
        print(json.dumps({'complete':True,'files':len(p['files']),'fit_cases':128,'monitor_cases':32,'challenge_cases':16,'styled_target_sets':80,'teacher_neural_calls':0,'optimizer_updates':0,'neural_calls':0}));return 0
    if a.worker:return worker(root,p,checkpoint,parent,a.protocol_sha)
    hardware=require_vm(root);need(not(root/'outputs').exists() and not(root/'supervisor.json').exists(),'Pilot already attempted; preserve all evidence')
    command=[sys.executable,'-B','-u',str(root/'scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py'),'--root',str(root),'--protocol-sha',a.protocol_sha,'--worker']
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

"""Manual L4 zero-optimizer SAME-codebook representation diagnostic only.

No oracle result is deployable or available from a real covered input alone.
Transfer-only mode imports only standard library and owned pure helpers.
"""
from __future__ import annotations
import argparse,hashlib,importlib,io,json,os,random,re,selectors,signal,subprocess,sys,time,traceback,urllib.request,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.completion_codeformer_same_codebook_probe_v1.contracts import CONTRACT,ARMS,VM_NAME,ORIGINAL_SHA,PARENT_SHA,PARENT_PROTOCOL_SHA,PARENT_LOGICAL,ORIGINAL_LOGICAL,aggregate,compare
from research.completion_codeformer_same_codebook_probe_v1.optimized_budget import OutputBudget
from research.completion_codeformer_same_codebook_probe_v1.probe_forward import probe_forward
from research.completion_codeformer_same_codebook_probe_v1.app_delivery import delivered_float
DESIGN_SHA='2ca6fafbaf2a3008459dace2d83bfc0cecb6594ae6e33adc7ac4aaef5bd0effb'
SOURCE_MODULES={'third_party':'third_party/__init__.py','third_party.codeformer':'third_party/codeformer/__init__.py','third_party.codeformer.registry':'third_party/codeformer/registry.py','third_party.codeformer.vqgan_arch':'third_party/codeformer/vqgan_arch.py','third_party.codeformer.codeformer_arch':'third_party/codeformer/codeformer_arch.py','completion':'completion.py','pretrained_completion':'pretrained_completion.py'}

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(path):
    def reject(x):raise ValueError('Nonfinite JSON '+x)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)
def write(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def member(root,name):
    need(isinstance(name,str) and name and ':' not in name and '\\' not in name,'Unsafe member')
    rel=PurePosixPath(name);need(not rel.is_absolute() and '..' not in rel.parts and str(rel)==name,'Noncanonical member')
    path=root.joinpath(*rel.parts)
    for part in (path,*path.parents):
        need(not part.is_symlink(),'Linked packet member')
        if part==root:break
    need(path.resolve().is_relative_to(root),'Packet member escaped root');return path

def verify(root,expected):
    root=root.resolve(strict=True);need((root/'protocol.json').stat().st_size<=1024**2 and sha(root/'protocol.json')==expected,'Exact protocol SHA/size required')
    p=read(root/'protocol.json');need(p['format']=='completion-codeformer-same-codebook-probe-v1' and p['contract']==CONTRACT,'SAME-codebook zero-optimizer contract differs')
    files={r['path']:r for r in p['files']};need(len(files)==len(p['files']),'Duplicate source member')
    for row in files.values():
        path=member(root,row['path']);need(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Packet source/asset drift '+row['path'])
    for name in list(SOURCE_MODULES.values())+['scripts/cctv_completion_codeformer_same_codebook_probe_v1_vm.py',*[f'research/completion_codeformer_same_codebook_probe_v1/{x}.py' for x in ('contracts','optimized_budget','probe_forward','app_delivery')]]:need(name in files,'Imported source unpinned '+name)
    need(sha(member(root,p['evidence']['route_design_review']))==DESIGN_SHA,'Frozen scientific/resource design differs')
    design=read(member(root,p['evidence']['route_design_review']))
    need(sha(member(root,p['evidence']['resource_admission']))=='4c9b2a1da70e508e0482f06235f079ab513dc3910c99073c7c678a67886cef86','Resource admission differs')
    need(sha(member(root,p['evidence']['oracle_label_index']))=='0567a9ad54e6397390341a8dfc1ffa8ec50ceb9159da299605fee021ec678a69','Original16label index source binding differs')
    need(sha(member(root,p['original_parent_protocol']))==PARENT_PROTOCOL_SHA,'Original R1 protocol differs')
    need(sha(member(root,'third_party/codeformer/codeformer_arch.py'))=='ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb','Exact original forward/prior source required')
    refs={r['id']:r for r in p['references']};need(len(refs)==8 and all(r['role']=='monitor' and r['original_role']=='train' and r['legacy_completion_split']=='train' for r in refs.values()),'Only8adaptive originalTRAIN monitor references')
    cases=p['monitor_cases']+p['challenge_cases'];need(len(p['monitor_cases'])==32 and len(p['challenge_cases'])==16 and len({r['id'] for r in cases})==48,'Exact32clear16challenge cases required')
    need([r['id'] for r in cases]==design['data']['case_ids_in_fixed_protocol_order'],'Frozen48case order differs')
    for row in cases:
        need(row['reference_id'] in refs and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',row['id']) and row['teacher_key']==row['reference_id']+'::'+row['style'],'Case/source/style metadata differs')
        for field in ('input256','clean256','clean512','mask'):need(row[field] in files,'Case asset unpinned')
    need(p['arms']==ARMS,'Fourarm w/oracle labels differ')
    labels={r['teacher_key']:r for r in p['oracle_labels']};need(len(labels)==16 and set(labels)=={ref+'::'+style for ref in refs for style in ('RGB','GRAY')},'Exactly16 originalSAMEcodebook label sets')
    declared={r['teacher_key']:r for r in design['data']['reuse_only_original16_exact_clean_context_teacher_labels']}
    for key,row in labels.items():
        need(row['path'] in files and row['sha256']==declared[key]['original_label']['sha256'] and row['bytes']==declared[key]['original_label']['bytes'] and row['clean512_sha256']==declared[key]['clean512_sha256'] and row['teacher_original_logical_sha256']==ORIGINAL_LOGICAL and row['oracle_unavailable_clean_information'],'Oracle label provenance/codebook differs')
        need(sha(member(root,row['path']))==row['sha256'] and sha(member(root,row['clean512']))==row['clean512_sha256'],'Oracle clean512/label binding differs')
    need(len(p['parity_cases'])==4 and len({r['id'] for r in p['parity_cases']})==4,'Four fresh app parity cases required')
    for row in p['parity_cases']:
        for field in ('input256','mask'):need(row[field] in files,'Parity source unpinned')
    for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
        rows=p['historical_control']['initial_rows'][kind];need(len(rows)==len(p[key]) and {r['id'] for r in rows}=={r['id'] for r in p[key]},'Historical R1 cohort differs')
        for row in rows:need(row['result256'] in files and row['PNG'] in files,'Historical R1 asset unpinned')
    def external(row,name,want_sha,want_bytes):
        raw=root/row['path'];need(not raw.is_symlink(),'Linked external file refused');path=raw.resolve(strict=True)
        need(path.is_relative_to((Path.home()/'forensic-dgp').resolve()) and path.name==name and path.is_file() and row['sha256']==want_sha and row['bytes']==want_bytes and path.stat().st_size==want_bytes and sha(path)==want_sha,'External original/R1 evidence differs');return path
    checkpoint=external(p['checkpoint'],'codeformer_inpainting.pth',ORIGINAL_SHA,370786611);parent=external(p['parent'],'terminal_training_state.pth',PARENT_SHA,701952165)
    need(p['parent']['protocol_sha256']==PARENT_PROTOCOL_SHA and p['parent']['logical_sha256']==PARENT_LOGICAL,'Only full audited R1 parent allowed')
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
    def terminate(signum,frame):raise KeyboardInterrupt('Owned supervisor termination; retain partial diagnostic evidence')
    previous=signal.signal(signal.SIGTERM,terminate)
    import shutil
    need(shutil.disk_usage(root).free>=CONTRACT['minimum_free_bytes'],'Need7GiB free before diagnostic')
    out=root/'outputs';need(not out.exists(),'Preserve any previous diagnostic attempt');out.mkdir()
    started=time.monotonic();budget=OutputBudget(out,started);budget.metadata_external=[root/'trainer.log',root/'protocol.json']
    counts={'forward_slots':0,'actual_whole_model_forward_calls':0,'backward_calls':0,'gradient_queries':0,'optimizer_objects':0,'optimizer_steps':0,'teacher_neural_calls':0}
    typed={'parity':0,'A':0,'B':0,'C':0,'D':0};progress={'phase':'initialization'};before=None;net=None;original_call_hook=None
    def record(path,value,*,emergency=False):budget.json(path,value,emergency=emergency)
    saved_npz_count=0
    def save_npz(path,**arrays):
        nonlocal saved_npz_count
        parity='ACTUAL_fresh_app_parity' in path.parts
        expected_keys={'input256','mask256','own_input512','app_input512','own_raw512','app_raw512','own_codes','app_codes','own_result256','app_result256'} if parity else {'input256','mask256','result256','raw512','predicted_codes','used_codes','arm_code','oracle_unavailable_clean_information'}
        need(set(arrays)==expected_keys and len(arrays)<=10 and saved_npz_count<196,'Bounded exact diagnostic NPZ schema/count')
        need(len(path.relative_to(root).as_posix().encode('utf-8'))<=240,'Bounded output path')
        for name,value in arrays.items():
            need(isinstance(value,np.ndarray) and not value.dtype.hasobject and value.dtype in (np.float32,np.int64,np.uint8),'Numeric-only exact diagnostic arrays')
            header=io.BytesIO();np.lib.format.write_array_header_1_0(header,np.lib.format.header_data_from_array_1_0(value));need(header.tell()<=512,'NPY header bound')
            if name.endswith('codes'):need(value.dtype==np.int64 and value.shape==(1,256),'Saved256indices int64 geometry')
            elif name in ('arm_code','oracle_unavailable_clean_information'):need(value.dtype==np.uint8 and value.shape==(),'Scalar arm/oracle flags')
            else:
                shape=(1,1,256,256) if name=='mask256' else (1,3,512,512) if name.endswith('512') else (1,3,256,256)
                need(value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all(),'Finite exact diagnostic image shape')
        with budget.open(path) as f:np.savez_compressed(f,**arrays)
        saved_npz_count+=1
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    need(torch.cuda.is_available(),'CUDA unavailable')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);random.seed(240);np.random.seed(240);torch.manual_seed(240);torch.cuda.manual_seed_all(240)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def guard():
        budget.guard();need(counts['forward_slots']<=220 and counts['actual_whole_model_forward_calls']==counts['forward_slots'],'Finite200planned/220hard call accounting')
        need(counts['backward_calls']==counts['gradient_queries']==counts['optimizer_objects']==counts['optimizer_steps']==counts['teacher_neural_calls']==0,'Zerooptimizer/gradient/teacher diagnostic only')
        need(torch.cuda.memory_allocated()<=CONTRACT['CUDA_allocated_bytes_cap'],'CUDA20GiB cap')
    def call(kind,fn):
        guard();need(counts['forward_slots']+1<=CONTRACT['forward_slots_max'],'Hard forward cap before dispatch');counts['forward_slots']+=1;typed[kind]+=1
        value=fn();guard();return value
    def state():
        hashes={};h=hashlib.sha256();all_finite=True
        for name,value in sorted(net.state_dict().items()):
            a=value.detach().cpu().contiguous();finite=bool(torch.isfinite(a).all().item());all_finite=all_finite and finite;body=a.numpy().tobytes()
            h.update(name.encode());h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode());h.update(body)
            hashes[name]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(body).hexdigest(),'finite':finite}
        need(len(hashes)==497,'Exact497frozen tensors');return {'logical_sha256':h.hexdigest(),'all_tensors_finite':all_finite,'tensors':hashes,'all_parameters_frozen':all(not v.requires_grad for v in net.parameters()),'all_parameter_gradients_none':all(v.grad is None for v in net.parameters()),'all_modules_eval':all(not module.training for module in net.modules())}
    def image(name):
        with Image.open(root/name) as im:
            need(im.format=='PNG' and im.mode=='RGB' and im.size==(256,256),'Exact RGB256 required');a=np.array(im,copy=True)
        return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32)/255
    def mask(name):
        with Image.open(root/name) as im:
            need(im.format=='PNG' and im.mode=='L' and im.size==(256,256),'Exact L256 mask required');a=np.array(im,copy=True);need(np.isin(a,[0,255]).all(),'Binary mask required')
        return torch.from_numpy(a).to('cuda',torch.float32)[None,None]/255
    def app_input(rgb,m):
        active=m.sum((1,2,3))>0;need(rgb.shape[0]==1 and bool(active.all().item()),'One active reviewed crop')
        visible=1-m[active];support=F.interpolate(visible,size=(512,512),mode='bilinear',align_corners=False)
        x=F.interpolate(rgb[active]*visible,size=(512,512),mode='bilinear',align_corners=False)/support.clamp_min(1e-8)
        m512=F.interpolate(m[active],size=(512,512),mode='nearest');return (x*(1-m512)+m512)*2-1
    def boundary(m):return (m-(1-F.max_pool2d(1-m,7,1,3))).clamp(0,1)
    def deliver(rgb,m,raw):return delivered_float(raw,rgb,m,interpolate=F.interpolate)
    def save_png(path,result):
        a=np.rint(result[0].detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        with budget.open(path) as f:Image.fromarray(a).save(f,format='PNG')
        return a
    try:
        for module,relative in SOURCE_MODULES.items():
            loaded=importlib.import_module(module);need(Path(loaded.__file__).resolve()==(root/relative).resolve(),'Imported source escaped packet '+module)
        architecture=importlib.import_module('third_party.codeformer.codeformer_arch');CodeFormer=architecture.CodeFormer
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=True);need(isinstance(loaded.get('params_ema'),dict),'Original params_ema required')
        net=CodeFormer(dim_embd=512,codebook_size=512,n_head=8,n_layers=9,connect_list=['32','64','128']);net.load_state_dict(loaded['params_ema'],strict=True);del loaded
        net=net.to('cuda',dtype=torch.float32).eval().requires_grad_(False);original_state=state();need(original_state['all_tensors_finite'] and original_state['logical_sha256']==ORIGINAL_LOGICAL,'Original logical state differs')
        def count_original_call(module,args):counts['actual_whole_model_forward_calls']+=1
        original_call_hook=net.register_forward_pre_hook(count_original_call)
        parent=torch.load(parent_path,map_location='cpu',weights_only=True)
        need(parent['logical_state_sha256']==PARENT_LOGICAL and parent['protocol_sha256']==PARENT_PROTOCOL_SHA and parent['base_checkpoint_sha256']==ORIGINAL_SHA and parent['cycle']==parent['completed_paired_cycles']==32 and parent['reason']=='completed_finite_pilot' and parent['partial_accumulated_gradients']=={},'Exact completeR1parent required')
        full=net.state_dict();names={n for n in full if not n.startswith(('quantize.','generator.'))};need(set(parent['trainable_state'])==names and len(names)==332,'FullR1delta required');net.load_state_dict({**full,**parent['trainable_state']},strict=True);del parent,full
        before=state();need(before['all_tensors_finite'] and before['logical_sha256']==PARENT_LOGICAL and before['all_parameters_frozen'] and before['all_parameter_gradients_none'] and before['all_modules_eval'],'Exact frozen/eval R1 state required')
        prior_names=[n for n in before['tensors'] if n.startswith(('quantize.','generator.'))];need(len(prior_names)==165 and all(before['tensors'][n]==original_state['tensors'][n] for n in prior_names),'Original165prior tensors changed during R1 reconstruction')
        record(out/'state_before.json',before);guard();progress['phase']='reuse_existing16_SAMEcodebook_labels'
        oracle={}
        for row in p['oracle_labels']:
            path=root/row['path']
            with zipfile.ZipFile(path) as z:
                need(set(z.namelist())=={'prequant_latent.npy','nearest_codes.npy'} and z.testzip() is None,'Original teacher NPZ schema/CRC differs')
                need(all(not i.flag_bits&1 and i.file_size<1024**2 and len(i.filename)<80 for i in z.infolist()),'Original label bounded members')
            with np.load(path,allow_pickle=False) as z:
                codes=z['nearest_codes'];latent=z['prequant_latent'];need(codes.dtype==np.int64 and codes.shape==(1,256) and codes.min()>=0 and codes.max()<512,'Original samecodebook256indices required');need(latent.dtype==np.float32 and latent.shape==(1,256,16,16) and np.isfinite(latent).all(),'Retained unused original latent differs')
                oracle[row['teacher_key']]=torch.from_numpy(codes.copy()).to('cuda').unsqueeze(-1)
        record(out/'oracle_label_receipt.json',{'sets':16,'teacher_calls':0,'existing_same_codebook_labels_only':True,'unused_prequant_latents_not_passed_to_model':True,'oracle_unavailable_clean_information':True,'source_pins':p['oracle_labels']})
        def owned(x,w,indices=None):
            counts['actual_whole_model_forward_calls']+=1
            return probe_forward(net,x,w=w,detach_16=True,code_only=False,adain=False,F=F,torch=torch,adaptive_instance_normalization=architecture.adaptive_instance_normalization,oracle_indices=indices)
        progress['phase']='four_fresh_owned_vs_original_app_parity';parity=[];folder=out/'ACTUAL_fresh_app_parity';folder.mkdir()
        wrapper=importlib.import_module('pretrained_completion').CodeFormerCompletion(net)
        with torch.no_grad():
            for case in p['parity_cases']:
                rgb=image(case['input256']);m=mask(case['mask']);x=app_input(rgb,m);capture=[]
                def grab_input(module,args):capture.append({'input':args[0].detach().cpu().numpy().copy(),'stride':list(args[0].stride())})
                def grab_output(module,args,result):
                    _,indices=torch.topk(F.softmax(result[1],dim=2),1,dim=2);capture[-1].update(raw=result[0].detach().cpu().numpy().copy(),codes=indices.squeeze(-1).detach().cpu().numpy().copy())
                raw,logits,lq,predicted,used=call('parity',lambda:owned(x,1));own=deliver(rgb,m,raw)
                ih=net.register_forward_pre_hook(grab_input);oh=net.register_forward_hook(grab_output)
                try:reference=call('parity',lambda:wrapper(rgb,m))
                finally:ih.remove();oh.remove()
                need(len(capture)==1,'One original app call captured');app=capture[0];err=float((own-reference).abs().max().item());input_exact=np.array_equal(x.detach().cpu().numpy(),app['input']);code_exact=np.array_equal(predicted.squeeze(-1).cpu().numpy(),app['codes'])
                save_npz(folder/(case['id']+'.npz'),input256=rgb.cpu().numpy(),mask256=m.cpu().numpy(),own_input512=x.cpu().numpy(),app_input512=app['input'],own_raw512=raw.cpu().numpy(),app_raw512=app['raw'],own_codes=predicted.squeeze(-1).cpu().numpy(),app_codes=app['codes'],own_result256=own.cpu().numpy(),app_result256=reference.cpu().numpy())
                row={'id':case['id'],'max_raw256_error':err,'model_raw512_max_error':float(np.max(np.abs(raw.cpu().numpy()-app['raw']))),'input512_exact':input_exact,'input_stride_exact':list(x.stride())==app['stride'],'predicted_codes_exact':code_exact,'used_equals_predicted':bool(torch.equal(used,predicted))};record(folder/(case['id']+'.json'),row);parity.append(row);guard()
        record(folder/'summary.json',{'complete':True,'cases':parity,'all4_saved_before_gate':True,'tolerance':1e-6,'counts':dict(counts)})
        need(len(parity)==4 and all(r['max_raw256_error']<=1e-6 and r['input512_exact'] and r['input_stride_exact'] and r['predicted_codes_exact'] and r['used_equals_predicted'] for r in parity),'Fresh ordinary path parity failed; no oracle allowed');del wrapper
        all_summaries={};baseline_predictions={};compatibility=[]
        for arm in ARMS:
            progress.update(phase='four_arm_saved_diagnostic',arm=arm['id'],oracle=arm['oracle'])
            for kind,cases in [('monitor',p['monitor_cases']),('challenge',p['challenge_cases'])]:
                folder=out/arm['filename_tag']/kind;folder.mkdir(parents=True);records=[]
                with torch.no_grad():
                    for case in cases:
                        progress['case']=case['id'];rgb=image(case['input256']);m=mask(case['mask']);target=image(case['clean256']);x=app_input(rgb,m);indices=oracle[case['teacher_key']] if arm['oracle'] else None
                        raw,logits,lq,predicted,used=call(arm['id'],lambda:owned(x,arm['w'],indices));result=deliver(rgb,m,raw);support=m[0,0].cpu().numpy().astype(bool);tag=case['id']+('__ORACLE_unavailable_clean_information' if arm['oracle'] else '')
                        need(torch.equal(result*(1-m),rgb*(1-m)),'Visible float preservation failed');array=save_png(folder/(tag+'.png'),result);source=np.rint(rgb[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8);need(np.array_equal(array[~support],source[~support]),'Visible PNG preservation failed')
                        predicted_np=predicted.squeeze(-1).cpu().numpy();used_np=used.squeeze(-1).cpu().numpy()
                        if arm['id']=='A':baseline_predictions[case['id']]=predicted_np.copy()
                        else:need(np.array_equal(predicted_np,baseline_predictions[case['id']]),'Observed encoder/predictor changed across arms')
                        if arm['oracle']:need(torch.equal(used,indices),'Oracle selected indices not exact')
                        else:need(torch.equal(used,predicted),'Actual arm changed selected predictions')
                        difference=abs(result-target);b=boundary(m);target_array=np.rint(target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                        metrics={'raw_hole_L1':float((difference*m).sum().item()/(m.sum().item()*3)),'raw_boundary_L1':float((difference*b).sum().item()/(b.sum().item()*3)),'PNG_hole_L1':float(abs(array.astype(np.float64)-target_array)[support].mean()/255),'PNG_boundary_L1':float(abs(array.astype(np.float64)-target_array)[b[0,0].cpu().numpy().astype(bool)].mean()/255)}
                        save_npz(folder/(tag+'.npz'),input256=rgb.cpu().numpy(),mask256=m.cpu().numpy(),result256=result.cpu().numpy(),raw512=raw.cpu().numpy(),predicted_codes=predicted_np,used_codes=used_np,arm_code=np.asarray(ord(arm['id']),dtype=np.uint8),oracle_unavailable_clean_information=np.asarray(arm['oracle'],dtype=np.uint8))
                        row={'id':case['id'],'kind':kind,'arm':arm['id'],'w':arm['w'],'oracle_unavailable_clean_information':arm['oracle'],'not_deployable':arm['id']!='A','mask_kind':case['mask_kind'],'style':case['style'],'condition':case['condition'],'teacher_key':case['teacher_key'],'PNG':str((folder/(tag+'.png')).relative_to(out)),'PNG_sha256':sha(folder/(tag+'.png')),'predicted_used_code_agreement':float((predicted==used).float().mean().item()),'outside_exact':True,**metrics};records.append(row)
                        if arm['id']=='A':
                            old=next(r for r in p['historical_control']['initial_rows'][kind] if r['id']==case['id'])
                            with np.load(root/old['result256'],allow_pickle=False) as z:historical=z['result256'];need(historical.dtype==np.float32 and historical.shape==(1,3,256,256) and np.isfinite(historical).all(),'Historical R1 array schema');error=float(np.max(np.abs(result.cpu().numpy()-historical)))
                            compatibility.append({'id':case['id'],'kind':kind,'raw256_max_error':error,'PNG_bytes_exact':sha(folder/(tag+'.png'))==sha(root/old['PNG'])})
                        guard()
                summary={'arm':arm['id'],'w':arm['w'],'oracle_unavailable_clean_information':arm['oracle'],'kind':kind,'records':records,'metrics':aggregate(records,condition=kind=='challenge'),'model_qualified':False,'counts':dict(counts)};record(folder/'results.json',summary);all_summaries[(arm['id'],kind)]=summary
            if arm['id']=='A':
                record(out/'ACTUAL_R1_48case_compatibility.json',{'complete':True,'cases':compatibility,'before_any_oracle':True,'counts':dict(counts),'pass':len(compatibility)==48 and all(r['raw256_max_error']<=1e-6 and r['PNG_bytes_exact'] for r in compatibility)})
                need(len(compatibility)==48 and all(r['raw256_max_error']<=1e-6 and r['PNG_bytes_exact'] for r in compatibility),'All48actual R1 historical compatibility failed; no subsequent controls/oracle')
            record(out/('arm_'+arm['id']+'_complete.json'),{'arm':arm,'counts':dict(counts),'complete':True,'oracle_unavailable_clean_information':arm['oracle']})
        after=state();need(after['all_tensors_finite'] and after==before,'Frozen497model state, flags or gradients changed');record(out/'state_after.json',after)
        comparisons={arm['id']:{kind:compare(all_summaries[(arm['id'],kind)],all_summaries[('A',kind)],conditions=kind=='challenge') for kind in ('monitor','challenge')} for arm in ARMS[1:]}
        need(saved_npz_count==196,'All192outputs plus4parity arrays required')
        need(counts=={'forward_slots':200,'actual_whole_model_forward_calls':200,'backward_calls':0,'gradient_queries':0,'optimizer_objects':0,'optimizer_steps':0,'teacher_neural_calls':0} and typed=={'parity':8,'A':48,'B':48,'C':48,'D':48},'Exact finite actual diagnostic counts differ')
        need(sha(checkpoint)==ORIGINAL_SHA and sha(parent_path)==PARENT_SHA,'Original or parent checkpoint changed');budget.assert_ledger_consistent()
        final={'format':'completion-codeformer-same-codebook-probe-v1-results','complete':True,'protocol_sha256':expected,'counts':counts,'typed_forward_slots':typed,'outputs_saved':192,'oracle_outputs_explicitly_labeled':96,'all497_state_before_after_identical':True,'R1_logical_sha256':PARENT_LOGICAL,'ordinary_app_parity':parity,'actual_R1_historical_compatibility':compatibility,'diagnostic_comparisons_vs_A':comparisons,'oracle_not_deployable_or_identity_recovery':True,'same_codebook_target_not_mathematical_upper_bound':True,'diagnostic_only':True,'new_model_created':False,'checkpoint_saved':False,'optimizer_steps':0,'backward_calls':0,'visual_review_pending':True,'model_qualified':False,'automatic_app_promotion':False,'seconds':time.monotonic()-started,'output_budget_bytes':budget.total()}
        record(out/'results.json',final,emergency=True);return 0
    except BaseException as exc:
        if net is not None and before is not None:
            try:
                actual=state();record(out/'state_on_failure.json',{'actual':actual,'equals_R1_before':actual==before,'counts':dict(counts)},emergency=True)
            except BaseException as state_error:record(out/'state_receipt_failure.json',{'error':repr(state_error),'counts':dict(counts)},emergency=True)
        record(out/'failure.json',{'complete':False,'exception':repr(exc),'traceback':traceback.format_exc(),'progress':progress,'counts':counts,'typed_forward_slots':typed,'partial_outputs_retained':True,'no_learned_checkpoint_expected':True,'model_qualified':False},emergency=True);raise
    finally:
        if original_call_hook is not None:original_call_hook.remove()
        signal.signal(signal.SIGTERM,previous)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-transfer',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--worker',action='store_true')
    a=parser.parse_args();root=a.root.resolve(strict=True);p,checkpoint,parent=verify(root,a.protocol_sha)
    if a.verify_transfer:
        print(json.dumps({'complete':True,'files':len(p['files']),'monitor_cases':32,'challenge_cases':16,'oracle_label_sets':16,'diagnostic_outputs_planned':192,'teacher_neural_calls':0,'optimizer_updates':0,'optimizer_objects':0,'neural_calls':0}));return 0
    if a.worker:return worker(root,p,checkpoint,parent,a.protocol_sha)
    hardware=require_vm(root);need(not(root/'outputs').exists() and not(root/'supervisor.json').exists(),'Pilot already attempted; preserve all evidence')
    command=[sys.executable,'-B','-u',str(root/'scripts/cctv_completion_codeformer_same_codebook_probe_v1_vm.py'),'--root',str(root),'--protocol-sha',a.protocol_sha,'--worker']
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
            try:child.wait(timeout=max(0,min(grace,900-(time.monotonic()-started))))
            except subprocess.TimeoutExpired:kill_owned(signal.SIGKILL);child.wait()
    try:
        with (root/'trainer.log').open('xb') as log:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
            print(json.dumps({'child_pid':child.pid,'log':'trainer.log','external_cap_seconds':900}),flush=True)
            selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
            try:
                while selector.get_map():
                    if time.monotonic()-started>=900:
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
                if not timeout and not log_cap_hit:code=child.wait(timeout=max(0,900-(time.monotonic()-started)))
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
        receipt={'complete':True,'worker_exit':code,'actual_child_exit':None if child is None else child.returncode,'child_pid':None if child is None else child.pid,'timeout':timeout,'interruption':interruption,'shutdown_mode':shutdown_mode,'external_cap_seconds':900,'seconds':time.monotonic()-started,'hardware':hardware,'no_learned_checkpoint':not any((root/'outputs').glob('*.pth*')),'stdout_capture_bytes':captured,'stdout_cap_bytes':CONTRACT['trainer_log_bytes'],'stdout_cap_reached':log_cap_hit,'stdout_prefix_retained_if_capped':True,'state_recoverability':'Zero optimizer diagnostic: no learned checkpoint; retain partial arrays and state-invariance receipt; no automatic resume','training_success_not_implied':True}
        body=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode();need(len(body)<=1024*1024,'Supervisor receipt bytecap')
        with (root/'supervisor.json').open('xb') as stream:stream.write(body)
    print(json.dumps({'worker_exit':code,'timeout':timeout,'stdout_cap_reached':log_cap_hit,'log':'trainer.log','model_qualified':False}));return code
if __name__=='__main__':raise SystemExit(main())

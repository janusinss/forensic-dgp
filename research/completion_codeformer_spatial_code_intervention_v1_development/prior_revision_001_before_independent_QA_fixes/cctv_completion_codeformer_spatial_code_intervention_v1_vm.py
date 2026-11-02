"""Manual L4 zero-update fixed spatial-code intervention diagnostic only.

No oracle result is deployable or available from a real covered input alone.
Transfer-only mode imports only standard library and owned pure helpers.
"""
from __future__ import annotations
import argparse,hashlib,importlib,io,json,os,random,re,selectors,signal,subprocess,sys,time,traceback,urllib.request,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.completion_codeformer_spatial_code_intervention_v1.contracts import CONTRACT,ARMS,EXTERNAL_ARMS,TYPED_CAPS,PROBE_PROTOCOL_SHA,output_paths,VM_NAME,ORIGINAL_SHA,PARENT_SHA,PARENT_PROTOCOL_SHA,PARENT_LOGICAL,ORIGINAL_LOGICAL,aggregate,compare
from research.completion_codeformer_spatial_code_intervention_v1.optimized_budget import OutputBudget
from research.completion_codeformer_spatial_code_intervention_v1.probe_forward import probe_forward,geometric_touched_grid
from research.completion_codeformer_spatial_code_intervention_v1.app_delivery import delivered_float
DESIGN_SHA='1983f9fed9bede2078ffc6a1af0dd6ca08fe6ec532e5c3de9c4ce41d4db2c657'
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

def external_base(root,p):
    named=Path.home()/'forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs'
    need(p['external_controls']['root']=='../cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs','Exact external old-probe root spelling')
    need((root/p['external_controls']['root']).resolve(strict=True)==named.resolve(strict=True),'External controls must be named old-probe outputs')
    need(not any(v.is_symlink() for v in (named,*named.parents)),'External control root links refused')
    return named.resolve(strict=True)

def verify(root,expected):
    root=root.resolve(strict=True);protocol=root/'protocol.json'
    need(protocol.stat().st_size<=1024**2 and sha(protocol)==expected,'Exact protocol SHA/size')
    p=read(protocol);need(p['format']=='completion-codeformer-spatial-code-intervention-v1' and p['contract']==CONTRACT,'Spatial code intervention contract differs')
    need(p.get('manual_execution_only') is True and p.get('automatic_resume_or_promotion') is False,'Manual-only diagnostic required')
    files={r['path']:r for r in p['files']};need(len(files)==len(p['files']) and files,'Unique nonempty source manifest')
    for row in files.values():
        path=member(root,row['path']);need(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Packet source/asset drift '+row['path'])
    required=list(SOURCE_MODULES.values())+['scripts/cctv_completion_codeformer_spatial_code_intervention_v1_vm.py']+[f'research/completion_codeformer_spatial_code_intervention_v1/{name}.py' for name in ('contracts','optimized_budget','probe_forward','app_delivery')]
    need(all(name in files for name in required),'Every imported runtime source pinned')
    designpath=member(root,p['evidence']['root_design']);need(DESIGN_SHA!='UNBOUND_ROOT_DESIGN' and sha(designpath)==DESIGN_SHA,'Root scientific/resource design binding')
    design=read(designpath)
    need(sha(member(root,p['evidence']['formal_design_admission']))=='4f6ea7134e9c8628efb7deae4ca5dc137b35dc9f1d0471c796c8aa03692e63b5','Independent formal design admission')
    need(sha(member(root,p['evidence']['pre_release_clarifications']))=='98c3521e5e7e0043479862923d03cc21c9b6b7c2e5371a0ffcd292d919ca1d78','Prospective interpretation/resource clarification')
    need(sha(member(root,p['evidence']['original_probe_protocol']))==PROBE_PROTOCOL_SHA,'Original probe source protocol differs')
    need(sha(member(root,p['evidence']['oracle_label_index']))=='0567a9ad54e6397390341a8dfc1ffa8ec50ceb9159da299605fee021ec678a69','Original same-codebook label index differs')
    need(sha(member(root,p['original_parent_protocol']))==PARENT_PROTOCOL_SHA,'Original complete R1 protocol differs')
    need(sha(member(root,'third_party/codeformer/codeformer_arch.py'))=='ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb','Exact original architecture')
    refs={r['id']:r for r in p['references']};need(len(refs)==8 and all(r['role']=='monitor' and r['original_role']=='train' and r['legacy_completion_split']=='train' for r in refs.values()),'Eight adaptive originalTRAIN monitor references')
    cases=p['monitor_cases']+p['challenge_cases'];need(len(p['monitor_cases'])==32 and len(p['challenge_cases'])==16 and len({r['id'] for r in cases})==48,'Fixed32clear16challenge cases')
    need([r['id'] for r in cases]==design['data']['case_ids_in_fixed_protocol_order'],'Exact48 root-design case order')
    for row in cases:
        need(row['reference_id'] in refs and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',row['id']) and row['teacher_key']==row['reference_id']+'::'+row['style'],'Case ID/split/style binding')
        for field in ('input256','clean256','clean512','mask'):need(row[field] in files,'Case asset unpinned')
    need(p['arms']==ARMS,'Two fixed H/V arms only')
    labels={r['teacher_key']:r for r in p['oracle_labels']};declared={r['teacher_key']:r for r in design['data']['original16labels_exact_only']}
    need(len(labels)==16 and set(labels)==set(declared)=={ref+'::'+style for ref in refs for style in ('RGB','GRAY')},'Exact16 labels with no FIT target additions')
    for key,row in labels.items():
        old=declared[key];need(row['path'] in files and row['sha256']==old['sha256'] and row['bytes']==old['bytes'] and row['clean512_sha256']==old['clean512_sha256'] and row['teacher_original_logical_sha256']==ORIGINAL_LOGICAL and row['oracle_unavailable_clean_information'] is True,'Exact original same-codebook target provenance')
        need(sha(member(root,row['path']))==row['sha256'] and sha(member(root,row['clean512']))==row['clean512_sha256'],'Oracle clean512/label exactbinding')
    need(len(p['parity_cases'])==4 and len({r['id'] for r in p['parity_cases']})==4,'Four app parity cases')
    for row in p['parity_cases']:
        for field in ('input256','mask'):need(row[field] in files,'Parity input pin')
    need([r['id'] for r in p['bridge_cases']]==design['data']['bridge_case_ids'],'Four prospectively chosen C bridge IDs')
    case_by_id={r['id']:r for r in cases};need(len(p['bridge_cases'])==4 and all(row==case_by_id[row['id']] for row in p['bridge_cases']),'Bridge rows are exact fixed cases')
    need(p['export']['allowed_output_paths']==output_paths(p) and len(output_paths(p))<395,'Closed prospective output namespace/member bound')
    ext=p['external_controls'];base=external_base(root,p)
    need(ext['source_protocol_sha256']==PROBE_PROTOCOL_SHA and set(ext['arms'])=={'A','C'},'External A/C provenance')
    external_files={r['path']:r for r in ext['files']};need(len(external_files)==len(ext['files'])==196,'Exact196 external reused control files')
    for row in external_files.values():
        path=member(base,row['path']);need(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'External control source drift '+row['path'])
    required_controls=set()
    for arm,folder in EXTERNAL_ARMS.items():
        need(set(ext['arms'][arm])=={'monitor','challenge'},'Only two paired control cohorts')
        for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
            reportname=folder+'/'+kind+'/results.json';need(reportname in external_files,'External summary pin');summary=read(member(base,reportname));need(summary==ext['arms'][arm][kind],'Control summary exactbytes/semantic join')
            need(summary['arm']==arm and summary['w']==1 and summary['kind']==kind and summary['oracle_unavailable_clean_information']==(arm=='C'),'Reused control semantics')
            need([r['id'] for r in summary['records']]==[r['id'] for r in p[key]],'Control exact ordered cohort')
            required_controls.add(reportname)
            for row in summary['records']:
                relative=row['PNG'];need(relative.startswith(folder+'/'+kind+'/') and relative in external_files and row['PNG_sha256']==external_files[relative]['sha256'],'Pinned control PNG binding')
                array=str(PurePosixPath(relative).with_suffix('.npz'));need(array in external_files,'Pinned control NPZ binding');required_controls.update((relative,array))
    need(set(external_files)==required_controls,'No omitted or extra external controls')
    def weight(row,name,want_sha,want_bytes):
        raw=root/row['path'];need(not any(v.is_symlink() for v in (raw,*raw.parents)),'Linked original/R1 weight path');path=raw.resolve(strict=True)
        need(path.is_relative_to((Path.home()/'forensic-dgp').resolve()) and path.name==name and path.is_file() and row['sha256']==want_sha and row['bytes']==want_bytes and path.stat().st_size==want_bytes and sha(path)==want_sha,'Exact original/R1 byte pin');return path
    checkpoint=weight(p['checkpoint'],'codeformer_inpainting.pth',ORIGINAL_SHA,370786611);parent=weight(p['parent'],'terminal_training_state.pth',PARENT_SHA,701952165)
    need(p['parent']['protocol_sha256']==PARENT_PROTOCOL_SHA and p['parent']['logical_sha256']==PARENT_LOGICAL,'Only full R1 parent, never failed predictor CE state')
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
    need(shutil.disk_usage(root).free>=CONTRACT['minimum_free_bytes'],'Need6GiB free before diagnostic')
    out=root/'outputs';need(not out.exists(),'Preserve any previous diagnostic attempt');out.mkdir()
    started=time.monotonic();budget=OutputBudget(out,started);budget.metadata_external=[root/'trainer.log',root/'protocol.json'];budget.allowed_names={name[len('outputs/'):] for name in output_paths(p)}
    counts={'forward_slots':0,'actual_whole_model_forward_calls':0,'backward_calls':0,'gradient_queries':0,'optimizer_objects':0,'optimizer_steps':0,'teacher_neural_calls':0}
    typed={key:0 for key in TYPED_CAPS};progress={'phase':'initialization'};before=None;net=None;original_call_hook=None;generator_start_hook=None;generator_end_hook=None;original_quantize=None;original_quantize_had_instance_attr=False;witness={'quantizer':0,'generator_starts':0,'generator_completions':0}
    def record(path,value,*,emergency=False):budget.json(path,value,emergency=emergency)
    saved_npz_count=0
    def save_npz(path,**arrays):
        nonlocal saved_npz_count
        parity='fresh_app_parity' in path.parts
        expected_keys={'input256','mask256','own_input512','app_input512','own_raw512','app_raw512','own_codes','app_codes','own_result256','app_result256'} if parity else {'input256','mask256','result256','raw512','predicted_codes','used_codes','grid_mask','arm_code','oracle_unavailable_clean_information'}
        need(set(arrays)==expected_keys and len(arrays)<=10 and saved_npz_count<104,'Bounded exact diagnostic NPZ schema/count')
        need(len(path.relative_to(root).as_posix().encode('utf-8'))<=240,'Bounded output path')
        for name,value in arrays.items():
            need(isinstance(value,np.ndarray) and not value.dtype.hasobject and value.dtype in (np.float32,np.int64,np.uint8),'Numeric-only exact diagnostic arrays')
            header=io.BytesIO();np.lib.format.write_array_header_1_0(header,np.lib.format.header_data_from_array_1_0(value));need(header.tell()<=512,'NPY header bound')
            if name.endswith('codes'):need(value.dtype==np.int64 and value.shape==(1,256) and value.min()>=0 and value.max()<512,'Saved256indices int64 geometry')
            elif name=='grid_mask':need(value.dtype==np.uint8 and value.shape==(1,256) and np.isin(value,[0,1]).all(),'Exact binary cell selector uint8[1,256]')
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
        budget.guard();need(counts['forward_slots']<=120 and counts['actual_whole_model_forward_calls']==counts['forward_slots'],'Finite108planned/120hard call accounting')
        need(counts['backward_calls']==counts['gradient_queries']==counts['optimizer_objects']==counts['optimizer_steps']==counts['teacher_neural_calls']==0,'Zerooptimizer/gradient/teacher diagnostic only')
        need(torch.cuda.memory_allocated()<=CONTRACT['CUDA_allocated_bytes_cap'],'CUDA20GiB cap')
    def call(kind,fn):
        guard();need(kind in TYPED_CAPS and typed[kind]+1<=TYPED_CAPS[kind],'Typed forward cap before dispatch')
        need(counts['forward_slots']+1<=CONTRACT['forward_slots_max'],'Hard forward cap before dispatch')
        need(not torch.is_grad_enabled() and all(not v.requires_grad and v.grad is None for v in net.parameters()) and all(not m.training for m in net.modules()),'Frozen/eval/no-grad before neural dispatch')
        counts['forward_slots']+=1;typed[kind]+=1
        value=fn()
        need(all(witness[key]==counts['forward_slots'] for key in witness),'Actual quantizer/generator witnesses must join every completed dispatch')
        guard();return value
    def state():
        hashes={};h=hashlib.sha256();all_finite=True
        for name,value in sorted(net.state_dict().items()):
            a=value.detach().cpu().contiguous();finite=bool(torch.isfinite(a).all().item());all_finite=all_finite and finite;body=a.numpy().tobytes()
            h.update(name.encode());h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode());h.update(body)
            hashes[name]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(body).hexdigest(),'finite':finite}
        return {'logical_sha256':h.hexdigest(),'all_tensors_finite':all_finite,'state_tensors':len(hashes),'tensors':hashes,'all_parameters_frozen':all(not v.requires_grad for v in net.parameters()),'all_parameter_gradients_none':all(v.grad is None for v in net.parameters()),'all_modules_eval':all(not module.training for module in net.modules())}
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
    failure_state_attempted=False
    def retain_failed_actual_state(actual):
        nonlocal failure_state_attempted
        need(not failure_state_attempted,'Exactly one unexpected-mutation state save attempt');failure_state_attempted=True
        partial=out/'failed_actual_model_state.pth.partial';committed=out/'failed_actual_model_state.pth';capture_error=None
        try:
            budget.guard(terminal=True,emergency=True)
            values=net.state_dict();raw_bytes=sum(v.numel()*v.element_size() for v in values.values())
            need(raw_bytes<=CONTRACT['terminal_bytes']-2*1024**2,'Actual state exceeds bounded failure reservation; do not invent recovery')
            actual_values={name:value.detach().cpu().contiguous().clone() for name,value in values.items()}
            snapshot={'format':'completion-codeformer-spatial-code-intervention-v1-failed-actual-state-v1','state_dict':actual_values,'actual_state_hash_receipt':actual,'expected_frozen_R1_logical':PARENT_LOGICAL,'original_checkpoint_sha256':ORIGINAL_SHA,'parent_checkpoint_sha256':PARENT_SHA,'protocol_sha256':expected,'counts':dict(counts),'typed_forward_slots':dict(typed),'progress':dict(progress),'actual_path_witness':dict(witness),'torch_CPU_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state_all(),'python_random_state':random.getstate(),'numpy_random_state':{'algorithm':np.random.get_state()[0],'keys':np.random.get_state()[1].tolist(),'position':np.random.get_state()[2],'has_gauss':np.random.get_state()[3],'cached_gaussian':np.random.get_state()[4]},'optimizer_objects':0,'optimizer_updates':0,'failure_evidence_not_qualified':True,'automatic_resume_permitted':False,'model_qualified':False}
            with budget.open(partial,terminal=True,emergency=True) as stream:torch.save(snapshot,stream)
            budget.guard(terminal=True,emergency=True)
            with zipfile.ZipFile(partial) as archive:
                infos=archive.infolist();need(0<len(infos)<=1024 and len({v.filename for v in infos})==len(infos),'Closed failure ZIP members')
                need(len([v for v in infos if v.filename.endswith('/data.pkl')])==1,'One failure tensor pickle')
                for info in infos:
                    need(not info.is_dir() and not info.flag_bits&1 and info.compress_type in (0,8) and not PurePosixPath(info.filename).is_absolute() and '..' not in PurePosixPath(info.filename).parts and '\\' not in info.filename,'Failure ZIP regular canonical members')
                    if info.filename.endswith('/data.pkl'):need(info.file_size<=1024**2,'Failure pickle metadata1MiBcap')
                    with archive.open(info) as source:
                        for chunk in iter(lambda:source.read(1024**2),b''):budget.guard(terminal=True,emergency=True)
            digest=sha(partial);need(partial.stat().st_size<=CONTRACT['terminal_bytes'],'Failure state384MiBcap');budget.guard(terminal=True,emergency=True)
            need(not committed.exists(),'Never overwrite failure state');os.link(partial,committed);need(partial.stat().st_ino==committed.stat().st_ino,'Exclusive same-inode failure commit');partial.unlink();budget.rename(partial,committed)
            del snapshot,actual_values
        except BaseException as exc:capture_error=repr(exc)
        pins=[]
        for path in (partial,committed):
            if path.is_file():pins.append({'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)})
        record(out/'failure_state_receipt.json',{'save_attempted_once':True,'reason':'Unexpected actual model state/hash/finite/frozen/eval/gradient invariant differs from before','complete_committed_state':capture_error is None and committed.is_file(),'capture_error':capture_error,'actual_state_logical_sha256':actual['logical_sha256'],'payloads':pins,'actual_nonfinite_flags_preserved':True,'partial_retained_not_usable':partial.exists(),'same_inode_commit_alias_may_remain_if_interrupted':True,'no_retry_or_automatic_resume':True,'model_qualified':False},emergency=True)

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
        before=state();need(before['state_tensors']==497 and before['all_tensors_finite'] and before['logical_sha256']==PARENT_LOGICAL and before['all_parameters_frozen'] and before['all_parameter_gradients_none'] and before['all_modules_eval'],'Exact frozen/eval R1 state required')
        prior_names=[n for n in before['tensors'] if n.startswith(('quantize.','generator.'))];need(len(prior_names)==165 and all(before['tensors'][n]==original_state['tensors'][n] for n in prior_names),'Original165prior tensors changed during R1 reconstruction')
        record(out/'state_before.json',before);guard();progress['phase']='reuse_existing16_SAMEcodebook_labels'
        oracle={}
        for row in p['oracle_labels']:
            path=root/row['path']
            with zipfile.ZipFile(path) as z:
                need(len(z.infolist())==2 and set(z.namelist())=={'prequant_latent.npy','nearest_codes.npy'} and z.testzip() is None,'Original teacher NPZ schema/CRC differs')
                need(all(not i.flag_bits&1 and i.file_size<1024**2 and len(i.filename)<80 for i in z.infolist()),'Original label bounded members')
            with np.load(path,allow_pickle=False,max_header_size=512) as z:
                codes=z['nearest_codes'];latent=z['prequant_latent'];need(codes.dtype==np.int64 and codes.shape==(1,256) and codes.min()>=0 and codes.max()<512,'Original samecodebook256indices required');need(latent.dtype==np.float32 and latent.shape==(1,256,16,16) and np.isfinite(latent).all(),'Retained unused original latent differs')
                oracle[row['teacher_key']]=torch.from_numpy(codes.copy()).to('cuda').unsqueeze(-1)
        record(out/'oracle_label_receipt.json',{'sets':16,'teacher_calls':0,'existing_same_codebook_labels_only':True,'unused_prequant_latents_not_passed_to_model':True,'oracle_unavailable_clean_information':True,'source_pins':p['oracle_labels']})
        def owned(x,w,indices=None,grid=None):
            counts['actual_whole_model_forward_calls']+=1
            return probe_forward(net,x,w=w,detach_16=True,code_only=False,adain=False,F=F,torch=torch,adaptive_instance_normalization=architecture.adaptive_instance_normalization,oracle_indices=indices,grid_mask=grid)
        # Actual witness counts from numerical subpaths, not just dispatch wrappers.
        witness={'quantizer':0,'generator_starts':0,'generator_completions':0}
        original_quantize_had_instance_attr='get_codebook_feat' in net.quantize.__dict__
        original_quantize=net.quantize.get_codebook_feat
        def counted_quantize(*args,**kwargs):
            witness['quantizer']+=1
            return original_quantize(*args,**kwargs)
        net.quantize.get_codebook_feat=counted_quantize
        def generator_start(module,args):witness['generator_starts']+=1
        def generator_done(module,args,result):witness['generator_completions']+=1
        generator_start_hook=net.generator.blocks[0].register_forward_pre_hook(generator_start)
        generator_end_hook=net.generator.blocks[-1].register_forward_hook(generator_done)
        progress['phase']='four_fresh_owned_vs_original_app_parity';parity=[];folder=out/'fresh_app_parity';folder.mkdir()
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
        need(len(parity)==4 and all(r['max_raw256_error']<=1e-6 and r['model_raw512_max_error']<=1e-6 and r['input512_exact'] and r['input_stride_exact'] and r['predicted_codes_exact'] and r['used_equals_predicted'] for r in parity),'Fresh ordinary path parity failed; no oracle allowed');del wrapper
        external=external_base(root,p);external_files={r['path']:r for r in p['external_controls']['files']}
        controls=p['external_controls']['arms'];baseline_predictions={};all_summaries={};control_compatibility=[]
        def source_file(relative):
            path=member(external,relative);row=external_files[relative]
            need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Reused control bytes changed')
            return path
        def load_control(arm,kind,ident):
            row=next(r for r in controls[arm][kind]['records'] if r['id']==ident)
            pngpath=source_file(row['PNG']);arrayname=str(PurePosixPath(row['PNG']).with_suffix('.npz'));arraypath=source_file(arrayname)
            with zipfile.ZipFile(arraypath) as z:
                need(len(z.infolist())==8 and set(z.namelist())=={key+'.npy' for key in ('input256','mask256','result256','raw512','predicted_codes','used_codes','arm_code','oracle_unavailable_clean_information')} and z.testzip() is None,'Reused numeric-only control schema/CRC')
                need(all(not v.is_dir() and not v.flag_bits&1 and v.file_size<=4*1024**2 and v.compress_type in (0,8) for v in z.infolist()),'Reused control ZIP member bound')
            with np.load(arraypath,allow_pickle=False,max_header_size=512) as z:arrays={key:z[key].copy() for key in z.files}
            for key,value in arrays.items():
                if key.endswith('codes'):need(value.dtype==np.int64 and value.shape==(1,256) and value.min()>=0 and value.max()<512,'Control codes exactrange')
                elif key in ('arm_code','oracle_unavailable_clean_information'):need(value.dtype==np.uint8 and value.shape==(),'Control scalarflags')
                else:need(value.dtype==np.float32 and value.shape==((1,1,256,256) if key=='mask256' else (1,3,512,512) if key=='raw512' else (1,3,256,256)) and np.isfinite(value).all(),'Reused control finite exact geometry')
            need(int(arrays['arm_code'])==ord(arm) and bool(arrays['oracle_unavailable_clean_information'])==(arm=='C'),'Control numeric armflags')
            need(np.isin(arrays['mask256'],[0,1]).all(),'Control mask binary')
            if arm=='A':need(np.array_equal(arrays['used_codes'],arrays['predicted_codes']),'Actual A codes ordinary')
            else:need(np.array_equal(arrays['used_codes'],oracle[next(c['teacher_key'] for c in p['monitor_cases']+p['challenge_cases'] if c['id']==ident)].squeeze(-1).cpu().numpy()),'C exact old samecodebook oracle labels')
            with Image.open(pngpath) as im:need(im.mode=='RGB' and im.size==(256,256),'Control PNG geometry');pixels=np.array(im,copy=True)
            need(np.array_equal(np.rint(arrays['result256'][0].transpose(1,2,0).clip(0,1)*255).astype(np.uint8),pixels),'Reused raw256/PNG exact')
            need(sha(arraypath)==external_files[arrayname]['sha256'] and sha(pngpath)==external_files[row['PNG']]['sha256'],'Reused controls stable after read')
            return arrays,pixels,row
        # Rehydrate only retained reports and arrays. No A/C model recomputation.
        with torch.no_grad():
            for arm in ('A','C'):
                for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
                    records=[]
                    for case in p[key]:
                        arrays,pixels,old=load_control(arm,kind,case['id']);rgb=image(case['input256']);m=mask(case['mask'])
                        need(np.array_equal(arrays['input256'],rgb.cpu().numpy()) and np.array_equal(arrays['mask256'],m.cpu().numpy()),'Reused A/C originalinput/reviewedmask exact')
                        if arm=='A':baseline_predictions[case['id']]=arrays['predicted_codes'].copy()
                        else:need(np.array_equal(arrays['predicted_codes'],baseline_predictions[case['id']]),'Reused A/C predictedcodes same observedsource')
                        support=arrays['mask256'][0,0].astype(bool);row=dict(old)
                        if case['style']=='GRAY':
                            value=arrays['result256'][0].transpose(1,2,0);row['GRAY_raw_mean_channel_spread']=float(np.ptp(value,axis=-1)[support].mean());row['GRAY_PNG_mean_channel_spread']=float(np.ptp(pixels.astype(np.int16),axis=-1)[support].mean())
                        records.append(row);control_compatibility.append({'arm':arm,'kind':kind,'id':case['id'],'source_input_mask_exact':True,'PNG_exact':True,'external_NPZ':str(PurePosixPath(row['PNG']).with_suffix('.npz'))});guard()
                    all_summaries[(arm,kind)]={'arm':arm,'w':1,'oracle_unavailable_clean_information':arm=='C','kind':kind,'records':records,'metrics':aggregate(records,condition=kind=='challenge'),'reused_exact_historical_controls':True,'new_model_calls':0,'model_qualified':False}
        record(out/'reused_control_binding.json',{'complete':True,'external_root':str(external),'source_protocol_sha256':PROBE_PROTOCOL_SHA,'files':p['external_controls']['files'],'A_C_reused_without_recompute':True,'new_neural_calls':0,'copied_payloads':0,'diagnostic_only':True})
        record(out/'external_control_compatibility.json',{'complete':True,'records':control_compatibility,'all96_sources_and_masks_exact':len(control_compatibility)==96,'before_H_V':True,'new_neural_calls':0})
        # Bridge the C intervention path on the four prospectively declared cases.
        folder=out/'external_bridge';folder.mkdir();bridges=[]
        with torch.no_grad():
            for case in p['bridge_cases']:
                kind='monitor' if case['id'] in {v['id'] for v in p['monitor_cases']} else 'challenge'
                old,pixels,_=load_control('C',kind,case['id']);rgb=image(case['input256']);m=mask(case['mask']);x=app_input(rgb,m)
                raw,logits,lq,predicted,used=call('bridge',lambda:owned(x,1,oracle[case['teacher_key']]))
                result=deliver(rgb,m,raw);array=np.rint(result[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                old_rgb=torch.from_numpy(old['input256']).to('cuda');old_mask=torch.from_numpy(old['mask256']).to('cuda');old_input=app_input(old_rgb,old_mask)
                row={'id':case['id'],'kind':kind,'input256_exact':np.array_equal(rgb.cpu().numpy(),old['input256']),'mask256_exact':np.array_equal(m.cpu().numpy(),old['mask256']),'input512_exact_from_exact_source_and_pinned_rule':bool(torch.equal(x,old_input)),'old_input512_not_stored_reconstructed_only':True,'raw512_max_error':float(np.max(np.abs(raw.cpu().numpy()-old['raw512']))),'result256_max_error':float(np.max(np.abs(result.cpu().numpy()-old['result256']))),'predicted_codes_exact':np.array_equal(predicted.squeeze(-1).cpu().numpy(),old['predicted_codes']),'used_codes_exact':np.array_equal(used.squeeze(-1).cpu().numpy(),old['used_codes']),'PNG_pixels_exact':np.array_equal(array,pixels),'oracle_unavailable_clean_information':True,'not_deployable':True}
                save_npz(folder/(case['id']+'.npz'),input256=rgb.cpu().numpy(),mask256=m.cpu().numpy(),result256=result.cpu().numpy(),raw512=raw.cpu().numpy(),predicted_codes=predicted.squeeze(-1).cpu().numpy(),used_codes=used.squeeze(-1).cpu().numpy(),grid_mask=np.ones((1,256),np.uint8),arm_code=np.asarray(ord('C'),np.uint8),oracle_unavailable_clean_information=np.asarray(1,np.uint8))
                record(folder/(case['id']+'.json'),row);bridges.append(row);guard()
        record(folder/'summary.json',{'complete':True,'cases':bridges,'all4_saved_before_gate':True,'before_H_V':True,'counts':dict(counts),'raw_tolerance':1e-6})
        need(len(bridges)==4 and all(r['input256_exact'] and r['mask256_exact'] and r['input512_exact_from_exact_source_and_pinned_rule'] and r['raw512_max_error']<=1e-6 and r['result256_max_error']<=1e-6 and r['predicted_codes_exact'] and r['used_codes_exact'] and r['PNG_pixels_exact'] for r in bridges),'Fresh C bridge failed; do not run H/V')
        for arm in ARMS:
            progress.update(phase='fixed_H_V_spatial_interventions',arm=arm['id'],oracle=True)
            for kind,cases in [('monitor',p['monitor_cases']),('challenge',p['challenge_cases'])]:
                folder=out/arm['filename_tag']/kind;folder.mkdir(parents=True);records=[]
                with torch.no_grad():
                    for case in cases:
                        progress['case']=case['id'];rgb=image(case['input256']);m=mask(case['mask']);target=image(case['clean256']);x=app_input(rgb,m)
                        touched=geometric_touched_grid(m.cpu().numpy(),np=np);selected=touched if arm['id']=='H' else ~touched
                        need(selected.any() and (~selected).any(),'Both reviewedregion and visiblecontext cells must exist')
                        grid=torch.from_numpy(selected.copy()).to('cuda').unsqueeze(-1);indices=oracle[case['teacher_key']]
                        raw,logits,lq,predicted,used=call(arm['id'],lambda:owned(x,1,indices,grid));result=deliver(rgb,m,raw);support=m[0,0].cpu().numpy().astype(bool);tag=case['id']+'__ORACLE_unavailable_clean_information'
                        predicted_np=predicted.squeeze(-1).cpu().numpy();used_np=used.squeeze(-1).cpu().numpy();need(np.array_equal(predicted_np,baseline_predictions[case['id']]),'Observed predictor differs from reused A')
                        expected_used=np.where(selected,indices.squeeze(-1).cpu().numpy(),predicted_np);need(np.array_equal(used_np,expected_used),'Only declared geometric region receives exact oraclecodes')
                        need(torch.equal(result*(1-m),rgb*(1-m)),'Visible float preservation failed');array=save_png(folder/(tag+'.png'),result);source=np.rint(rgb[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8);need(np.array_equal(array[~support],source[~support]),'Visible PNG preservation failed')
                        difference=abs(result-target);b=boundary(m);target_array=np.rint(target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                        metrics={'raw_hole_L1':float((difference*m).sum().item()/(m.sum().item()*3)),'raw_boundary_L1':float((difference*b).sum().item()/(b.sum().item()*3)),'PNG_hole_L1':float(abs(array.astype(np.float64)-target_array)[support].mean()/255),'PNG_boundary_L1':float(abs(array.astype(np.float64)-target_array)[b[0,0].cpu().numpy().astype(bool)].mean()/255)}
                        save_npz(folder/(tag+'.npz'),input256=rgb.cpu().numpy(),mask256=m.cpu().numpy(),result256=result.cpu().numpy(),raw512=raw.cpu().numpy(),predicted_codes=predicted_np,used_codes=used_np,grid_mask=selected.astype(np.uint8),arm_code=np.asarray(ord(arm['id']),np.uint8),oracle_unavailable_clean_information=np.asarray(1,np.uint8))
                        row={'id':case['id'],'kind':kind,'arm':arm['id'],'w':1,'oracle_unavailable_clean_information':True,'not_deployable':True,'mask_kind':case['mask_kind'],'style':case['style'],'condition':case['condition'],'teacher_key':case['teacher_key'],'PNG':str((folder/(tag+'.png')).relative_to(out)),'PNG_sha256':sha(folder/(tag+'.png')),'predicted_codes_equal_reused_A':True,'grid_rule':CONTRACT['grid_rule'],'selected_grid_cells':int(selected.sum()),'changed_selected_cells':int((used_np!=predicted_np).sum()),'outside_exact':True,**metrics}
                        if case['style']=='GRAY':
                            value=result[0].permute(1,2,0).cpu().numpy();row['GRAY_raw_mean_channel_spread']=float(np.ptp(value,axis=-1)[support].mean());row['GRAY_PNG_mean_channel_spread']=float(np.ptp(array.astype(np.int16),axis=-1)[support].mean())
                        records.append(row);guard()
                summary={'arm':arm['id'],'w':1,'oracle_unavailable_clean_information':True,'not_deployable':True,'kind':kind,'records':records,'metrics':aggregate(records,condition=kind=='challenge'),'model_qualified':False,'counts':dict(counts)};record(folder/'results.json',summary);all_summaries[(arm['id'],kind)]=summary
            record(out/('arm_'+arm['id']+'_complete.json'),{'arm':arm,'counts':dict(counts),'actual_path_witness':dict(witness),'complete':True,'oracle_unavailable_clean_information':True})

        after=state();record(out/'state_after.json',after)
        need(after['state_tensors']==497 and after['all_tensors_finite'] and after==before,'Frozen497 model state, flags or gradients changed')
        comparisons={arm['id']:{control:{kind:compare(all_summaries[(arm['id'],kind)],all_summaries[(control,kind)],conditions=kind=='challenge') for kind in ('monitor','challenge')} for control in ('A','C')} for arm in ARMS}
        gray={}
        for arm in ('A','C','H','V'):
            rows=[r for r in all_summaries[(arm,'challenge')]['records'] if r['style']=='GRAY'];need(len(rows)==8,'All8fixedGRAYchallenge diagnostics')
            def style_mean(items):return {key:sum(r[key] for r in items)/len(items) for key in ('GRAY_raw_mean_channel_spread','GRAY_PNG_mean_channel_spread')}
            gray[arm]={'cases':rows,'macro':style_mean(rows),'geometry':{kind:style_mean([r for r in rows if r['mask_kind']==kind]) for kind in ('lower_face','eye_band','image_left_lens','cheek_object')},'not_a_deployable_model':True}
        need(saved_npz_count==104,'Exactly96 H/V outputs plus4parity4bridge arrays')
        need(counts=={'forward_slots':108,'actual_whole_model_forward_calls':108,'backward_calls':0,'gradient_queries':0,'optimizer_objects':0,'optimizer_steps':0,'teacher_neural_calls':0} and typed==TYPED_CAPS,'Exact finite actual diagnostic counts')
        need(witness=={'quantizer':108,'generator_starts':108,'generator_completions':108},'Actual quantizer and generator numerical path must witness all108 calls')
        need(sha(checkpoint)==ORIGINAL_SHA and sha(parent_path)==PARENT_SHA,'Original or R1 parent checkpoint changed')
        for relative in external_files:source_file(relative)
        budget.assert_ledger_consistent()
        final={'format':'completion-codeformer-spatial-code-intervention-v1-results','complete':True,'protocol_sha256':expected,'counts':counts,'typed_forward_slots':typed,'actual_path_witness':witness,'outputs_saved':96,'new_NPZ_saved':104,'reused_A_C_outputs':96,'reused_control_files':196,'all497_state_before_after_identical':True,'R1_logical_sha256':PARENT_LOGICAL,'ordinary_app_parity':parity,'fresh_C_control_bridge':bridges,'all_external_sources_hash_unchanged':True,'diagnostic_comparisons_vs_reused_A_C':comparisons,'GRAY_style_diagnostics':gray,'oracle_not_deployable_or_identity_recovery':True,'geometric_proxy_not_receptive_field':True,'same_codebook_target_not_mathematical_upper_bound':True,'quality_stop_or_candidate_selection':False,'diagnostic_only':True,'new_model_created':False,'checkpoint_saved':False,'optimizer_steps':0,'backward_calls':0,'visual_review_pending':True,'model_qualified':False,'automatic_app_promotion':False,'seconds':time.monotonic()-started,'output_budget_bytes':budget.total(),'output_categories':{key:budget.total(key) for key in ('terminal','teacher','metadata','array','PNG')}}
        record(out/'results.json',final,emergency=True);return 0

    except BaseException as exc:
        if net is not None and before is not None:
            try:
                actual=state();record(out/'state_on_failure.json',{'actual':actual,'equals_R1_before':actual==before,'counts':dict(counts)},emergency=True)
                if actual!=before:retain_failed_actual_state(actual)
            except BaseException as state_error:record(out/'state_receipt_failure.json',{'error':repr(state_error),'counts':dict(counts)},emergency=True)
        record(out/'failure.json',{'complete':False,'exception':repr(exc),'traceback':traceback.format_exc(),'progress':progress,'counts':counts,'typed_forward_slots':typed,'partial_outputs_retained':True,'no_learned_checkpoint_expected':True,'model_qualified':False},emergency=True);raise
    finally:
        if original_call_hook is not None:original_call_hook.remove()
        if generator_start_hook is not None:generator_start_hook.remove()
        if generator_end_hook is not None:generator_end_hook.remove()
        if original_quantize is not None:
            if original_quantize_had_instance_attr:net.quantize.get_codebook_feat=original_quantize
            else:delattr(net.quantize,'get_codebook_feat')
        signal.signal(signal.SIGTERM,previous)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-transfer',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--worker',action='store_true')
    a=parser.parse_args();root=a.root.resolve(strict=True);p,checkpoint,parent=verify(root,a.protocol_sha)
    if a.verify_transfer:
        print(json.dumps({'complete':True,'files':len(p['files']),'monitor_cases':32,'challenge_cases':16,'oracle_label_sets':16,'external_controls_files':len(p['external_controls']['files']),'diagnostic_outputs_planned':96,'total_model_calls_planned':108,'teacher_neural_calls':0,'optimizer_updates':0,'optimizer_objects':0,'neural_calls':0}));return 0
    if a.worker:return worker(root,p,checkpoint,parent,a.protocol_sha)
    hardware=require_vm(root);need(not(root/'outputs').exists() and not(root/'supervisor.json').exists(),'Pilot already attempted; preserve all evidence')
    command=[sys.executable,'-B','-u',str(root/'scripts/cctv_completion_codeformer_spatial_code_intervention_v1_vm.py'),'--root',str(root),'--protocol-sha',a.protocol_sha,'--worker']
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
        receipt={'complete':True,'worker_exit':code,'actual_child_exit':None if child is None else child.returncode,'child_pid':None if child is None else child.pid,'timeout':timeout,'interruption':interruption,'shutdown_mode':shutdown_mode,'external_cap_seconds':900,'seconds':time.monotonic()-started,'hardware':hardware,'zero_update_diagnostic':True,'unexpected_mutation_failure_state_present':(root/'outputs'/'failed_actual_model_state.pth').is_file(),'failure_state_partial_present':(root/'outputs'/'failed_actual_model_state.pth.partial').is_file(),'stdout_capture_bytes':captured,'stdout_cap_bytes':CONTRACT['trainer_log_bytes'],'stdout_cap_reached':log_cap_hit,'stdout_prefix_retained_if_capped':True,'state_recoverability':'Zero optimizer; preserve actual failed497state onlyif unexpectedmutation. Failedpartials not usable; no automaticresume.','training_success_not_implied':True}
        body=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode();need(len(body)<=1024*1024,'Supervisor receipt bytecap')
        with (root/'supervisor.json').open('xb') as stream:stream.write(body)
    print(json.dumps({'worker_exit':code,'timeout':timeout,'stdout_cap_reached':log_cap_hit,'log':'trainer.log','model_qualified':False}));return code
if __name__=='__main__':raise SystemExit(main())

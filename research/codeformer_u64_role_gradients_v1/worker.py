"""Inert finite manual-L4 role-gradient worker; import uses only stdlib.

The explicit guarded run measures gradients at one exact trained-u64 state.
No optimizer, update, parameter application, routing or inference change exists.
The historical parent stage is an intentional read-only import dependency.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings

OUTPUT = 'outputs/u64_role_gradients_v1'
PACKAGE = 'research/codeformer_u64_role_gradients_v1'
PARENT_PACKAGE = 'research/codeformer_u64_cleananchors_continuation_v1'
PARENT_PROTOCOL = (279879, '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419')
PARENT_SOURCES = {
    PARENT_PACKAGE+'/model.py': (21294, '6a67468fd6b94e938efe8548205282c048635fcc63e8e4efd8389a2910129829'),
    PARENT_PACKAGE+'/objective.py': (8205, 'cffafed3feae10ce91890cfd68ee5de96c8db8edad00041420615cd071f2b7ad'),
    'research/codeformer_u96_postfusion_v1/model.py': (23240, 'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8'),
    'research/codeformer_u96_spatial_feasibility_v1/worker.py': (72086, 'fae271100642604ae0682ebf52ec163e768fa6882eced87e0c34e587980126b6'),
}
CATEGORIES = {'parity_forwards':4, 'repeat_forwards':4, 'diagnostic_forwards':64}
OPERATIONS = {'forward':('full_forward_attempts','full_forward_returns',72),
              'gradient_query':('gradient_query_attempts','gradient_query_returns',192)}
COUNTER_KEYS = ('full_forward_attempts','full_forward_returns','gradient_query_attempts','gradient_query_returns',*CATEGORIES)
PHASE_CATEGORY = {'historical_u64_parity':'parity_forwards',
                  'deterministic_same_u64_repeat':'repeat_forwards',
                  'paired_TRAIN_gradient_diagnostic':'diagnostic_forwards'}
QUERIES = ('objective_div4','gate_mae','gate_adjacent')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def supervisor_context(sha, *, environ, parent_pid):
    need(type(sha) is str and re.fullmatch('[0-9a-f]{64}',sha)
         and environ.get('FORENSIC_U64_ROLE_GRADIENTS_SUPERVISED')=='1'
         and environ.get('FORENSIC_U64_ROLE_GRADIENTS_PROTOCOL')==sha
         and environ.get('FORENSIC_U64_ROLE_GRADIENTS_SUPERVISOR_PID')==str(parent_pid),
         'Exact finite supervisor seal and actual parent PID required')


def install_parent_import_paths(parent,research_package):
    """Support both regular packages and iterable namespace-package paths."""
    value=str(Path(parent)/'research')
    research_package.__path__=[value,*[str(p) for p in research_package.__path__ if str(p)!=value]]
    sys.path.insert(0,str(parent))


class OperationLedger:
    def __init__(self,persist):
        self.persist=persist;self.counters=dict.fromkeys(COUNTER_KEYS,0);self.pending={};self.sequence=0

    def reserve(self,kind,**metadata):
        need(kind in OPERATIONS and not self.pending and not set(metadata)&{'kind','token','counters'},
             'Known nonoverlapping operation and nonreserved metadata required')
        attempted,_,cap=OPERATIONS[kind]
        need(self.counters[attempted]<cap,'Finite operation cap exhausted')
        if kind=='forward':
            category=metadata.get('category')
            need(category in CATEGORIES and self.counters[category]<CATEGORIES[category],'Finite forward category required')
            self.counters[category]+=1
        self.counters[attempted]+=1;self.sequence+=1
        self.pending[self.sequence]={'kind':kind,**metadata}
        self.persist({'kind':kind+'_reserved','token':self.sequence,'counters':dict(self.counters),**metadata})
        return self.sequence

    def returned(self,token,**metadata):
        need(type(token) is int and token in self.pending and not set(metadata)&{'kind','token','counters'},
             'One known unmatched operation required')
        pending=self.pending[token];_,returned,_=OPERATIONS[pending['kind']]
        count=dict(self.counters);count[returned]+=1
        self.persist({**pending,**metadata,'kind':pending['kind']+'_returned','token':token,'counters':count})
        self.counters=count;del self.pending[token]

    def snapshot(self):
        return {'counters':dict(self.counters),'unknown_reservations':[{'token':k,**v} for k,v in sorted(self.pending.items())],
                'operation_outcome_unknown':bool(self.pending)}


def run(root,parent,protocol,expected_sha):
    from . import runtime,contract
    root,parent=Path(root).absolute(),Path(parent).absolute()
    began=time.monotonic();store=ledger=candidate=torch=np=None;helper=linux_resource=None
    phase='static_guards';failed_phase=None;active_case=active_slot=active_query=None
    initial=None;caller_rng=None;last_complete_state=None;old_term=None;ready=False;full_states_attempted=0

    def event(row):
        store.append('events.jsonl',{'UTC':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds':time.monotonic()-began,'phase':phase,**row})

    def status(label,complete=False):
        if store is not None:
            snap=ledger.snapshot() if ledger else {'counters':dict.fromkeys(COUNTER_KEYS,0),'unknown_reservations':[],
                                                 'operation_outcome_unknown':False}
            store.status({'complete':complete,'protocol_sha256':expected_sha,'status':'FINITE_INITIAL_U64_ROLE_GRADIENTS_NOT_QUALIFIED',
                'label':label,'phase':failed_phase or phase,'active_case':active_case,'active_slot':active_slot,
                'active_query':active_query,'elapsed_seconds':time.monotonic()-began,
                'last_complete_state':last_complete_state,'model_qualified':False,**snap})

    def checked(base,row):
        need(type(row) is dict and set(row)=={'path','bytes','sha256'},'Exact file binding required')
        name=row['path'];need(type(name) is str and name and '\\' not in name and ':' not in name
            and all(p not in ('','.','..') for p in name.split('/')),'Confined binding path required')
        path=runtime.regular(base/name)
        need(runtime.pin(base,path)==row,'Bound input differs from its exact bytes')
        return path

    def resource(label):
        elapsed=time.monotonic()-began;free=shutil.disk_usage(root).free
        rss=helper.linux_current_rss_bytes(Path('/proc/self/statm').read_text(),int(os.sysconf('SC_PAGE_SIZE')))
        peak=int(linux_resource.getrusage(linux_resource.RUSAGE_SELF).ru_maxrss)*1024
        cuda=peak_cuda=0
        if torch is not None and torch.cuda.is_initialized():
            cuda=int(torch.cuda.memory_reserved(0));peak_cuda=int(torch.cuda.max_memory_reserved(0))
        row={'label':label,'elapsed_seconds':elapsed,'free_disk_bytes':free,'current_RSS_bytes':rss,'peak_RSS_bytes':peak,
             'CUDA_reserved_bytes':cuda,'CUDA_peak_reserved_bytes':peak_cuda,'output_bytes':store.bytes if store else 0}
        if ready:store.append('resources.jsonl',row)
        limits=contract.LIMITS
        need(elapsed<limits['worker_seconds'],'Finite900s scientific stop')
        need(free>=limits['continuing_free_disk_bytes'],'Continuing4GiB disk stop')
        need(max(rss,peak)<=limits['maximum_RSS_bytes'],'RSS10GiB stop')
        need(max(cuda,peak_cuda)<=limits['maximum_CUDA_reserved_bytes'],'CUDA18GiB stop')
        need(store is None or store.bytes<=limits['maximum_output_bytes'],'Output2GiB stop')

    @contextmanager
    def timed(seconds,name):
        need(seconds>0 and signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'Fresh positive timer required')
        old=signal.getsignal(signal.SIGALRM)
        def expired(signum,frame):raise TimeoutError('Finite stop: '+name)
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
        try:yield
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

    @contextmanager
    def bounded_phase(name):
        nonlocal phase,failed_phase
        previous=phase;phase=name;start=time.monotonic();resource('before')
        try:
            with timed(min(180.0,contract.LIMITS['worker_seconds']-(time.monotonic()-began)),name):
                yield;resource('after')
                event({'kind':'phase_returned','phase_seconds':time.monotonic()-start,'counters':dict(ledger.counters)})
        except BaseException:failed_phase=name;raise
        finally:phase=previous

    try:
        p=runtime.verify_static(root,parent,expected_sha)
        need(p==protocol,'Exact verified diagnostic protocol required')
        need(Path(__file__).resolve()==root/PACKAGE/'worker.py','Own staged worker origin required')
        need(Path(contract.__file__).resolve()==root/PACKAGE/'contract.py','Own staged contract origin required')
        facts=runtime.require_vm(root,parent)
        need(shutil.disk_usage(root).free>=contract.LIMITS['initial_free_disk_bytes'],
             'Initial8GiB free disk admission required before Torch')
        supervisor_context(expected_sha,environ=os.environ,parent_pid=os.getppid())
        need(Path.cwd().resolve()==parent.resolve() and not (root/OUTPUT).exists(),
             'Intentional exact read-only parent execution cwd and fresh output required')
        parent_protocol=(parent/'protocol.json').read_bytes()
        need((len(parent_protocol),hashlib.sha256(parent_protocol).hexdigest())==PARENT_PROTOCOL,'Exact retained parent protocol required')
        oldp=json.loads(parent_protocol)
        for name,fixed in PARENT_SOURCES.items():
            path=runtime.regular(parent/name)
            need((path.stat().st_size,runtime.sha256(path))==fixed
                 and oldp['files'][name]=={'bytes':fixed[0],'sha256':fixed[1]},'Exact parent source before import required')
        # Only the verified parent historical packages are resolved through this
        # read-only path. The new runtime/contract/worker remain the new stage.
        import research
        install_parent_import_paths(parent,research)
        helper=importlib.import_module('research.codeformer_u96_spatial_feasibility_v1.worker')
        model=importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.model')
        objective=importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.objective')
        for module,name in [(helper,'research/codeformer_u96_spatial_feasibility_v1/worker.py'),
                            (model,PARENT_PACKAGE+'/model.py'),(objective,PARENT_PACKAGE+'/objective.py')]:
            need(Path(module.__file__).resolve()==parent/name,'Verified parent import origin required')
        import resource as linux_resource
        standard_bytes=checked(parent,oldp['parent_protocol']).read_bytes()
        selection=contract.select_cases(parent_protocol,standard_bytes);plan=contract.operation_plan(selection)
        need(selection==p['selection'] and plan==p['plan'] and len(plan)==72,'Exact sealed72 call plan required')
        rows={r['id']:r for r in oldp['cases']+oldp['guards']}
        witnesses={r['id']:r for r in oldp['u64_parity']}
        store=helper.DurableStore(root,root/OUTPUT,contract.LIMITS['maximum_output_bytes'],runtime.pin)
        store.json('started.json',{'complete':False,'protocol_sha256':expected_sha,'host':facts,
                                 'parent_read_only':True,'model_qualified':False})
        store.write('events.jsonl',b'');store.write('resources.jsonl',b'');ledger=OperationLedger(event);ready=True
        old_term=signal.getsignal(signal.SIGTERM)
        def stopped(signum,frame):raise TimeoutError('External finite SIGTERM; preserve actual prefix and unknown outcomes')
        signal.signal(signal.SIGTERM,stopped)
        store.json('operation_plan.json',{'complete':True,'calls':plan,'selection':selection,'model_qualified':False})
        event({'kind':'static_VM_supervisor_guards_passed_before_Torch','Torch_imported':False,'counters':dict(ledger.counters)})
        with bounded_phase('idle_GPU_before_framework'):
            probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                                 text=True,capture_output=True,timeout=10,check=True)
            need(not probe.stdout.strip(),'Competing GPU workload; no model construction')
        with bounded_phase('dependencies_and_FP32_policy'):
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            need(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8',':16:8'),'Deterministic cuBLAS config required')
            import torch as torch_module
            import numpy as numpy_module
            from PIL import Image
            from torch.nn import functional as F
            torch,np=torch_module,numpy_module
            need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA L4',
                 'Exactly one actual NVIDIA L4 required')
            torch.cuda.set_device(0);device=torch.device('cuda:0')
            need(not torch.is_inference_mode_enabled() and torch.get_default_dtype()==torch.float32,'Ordinary FP32 autograd required')
            caller_rng={'Python':random.getstate(),'NumPy':np.random.get_state(),'CPU':torch.get_rng_state().cpu().clone(),
                        'CUDA':[v.cpu().clone() for v in torch.cuda.get_rng_state_all()]}
            random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.use_deterministic_algorithms(True,warn_only=False);torch.cuda.reset_peak_memory_stats(0)

        def arrays(state):return {n:v.detach().cpu().contiguous().numpy().copy() for n,v in state.items()}
        def save_array(name,value):
            a=value.detach().cpu().contiguous().numpy().copy() if isinstance(value,torch.Tensor) else np.asarray(value)
            need(a.dtype.kind in ('f','i','u') and np.isfinite(a).all(),'Finite numeric array required')
            stream=io.BytesIO();np.save(stream,a,allow_pickle=False);return store.write(name,stream.getvalue())
        def tensor_record(value):
            a=value.detach().cpu().contiguous().numpy()
            return {'shape':list(a.shape),'dtype':str(a.dtype),'bytes_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
        def rgb(row):
            with Image.open(checked(parent,row)) as image:
                need(image.format=='PNG' and image.mode=='RGB' and image.size==(256,256)
                     and getattr(image,'n_frames',1)==1,'Exact RGB256 input required')
                return np.asarray(image).copy()
        def raw_file(row):
            with checked(parent,row).open('rb') as stream:a=np.load(stream,allow_pickle=False)
            need(a.dtype==np.float32 and a.shape==(256,256,3) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1,
                 'Exact delivered FP32 RAW witness required')
            return a
        def rng_record():
            ns=np.random.get_state()
            return {'Python':repr(random.getstate()),'NumPy':[ns[0],ns[1].tobytes().hex(),int(ns[2]),int(ns[3]),float(ns[4])],
                    'CPU':torch.get_rng_state().cpu().numpy().tobytes().hex(),
                    'CUDA':[v.cpu().numpy().tobytes().hex() for v in torch.cuda.get_rng_state_all()]}

        with bounded_phase('exact_trained_u64_initialization_no_forward'):
            r2_path,u96_path=(checked(parent,oldp['model'][k]) for k in ('r2','u96'))
            r2=torch.load(r2_path,map_location='cpu',weights_only=True);u96=torch.load(u96_path,map_location='cpu',weights_only=True)
            need(type(r2) is dict and set(r2)=={'params_ema'} and type(u96) is dict and set(u96)=={'format','adapter_state'}
                 and u96['format']=='codeformer-spatial-adapter-inference-v1','Exact original initializer schemas required')
            need(helper.canonical_arrays(arrays(r2['params_ema']),np=np)[0]==model.R2_CANONICAL_SHA256
                 and helper.canonical_arrays(arrays(u96['adapter_state']),np=np)[0]==model.U96_CANONICAL_SHA256,
                 'Exact original R2/U96 canonical values required')
            from third_party.codeformer.codeformer_arch import CodeFormer
            for name,digest in model.ARCHITECTURE_SOURCES.items():
                module=sys.modules.get(name[:-3].replace('/','.'))
                need(module is not None and Path(module.__file__).resolve()==parent/name
                     and runtime.sha256(parent/name)==digest,'Exact parent architecture import required')
            before=rng_record()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261023)
                base=CodeFormer(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=list(model.SITES))
                base.load_state_dict(r2['params_ema'],strict=True);base=base.float().eval().requires_grad_(False).to(device)
            need(before==rng_record(),'Base construction changed caller RNG')
            candidate=model.build_restorer(base,u96['adapter_state'],r2_artifact=r2_path,u96_artifact=u96_path,
                head_artifact=checked(parent,oldp['u64_head']),manual_vm_root=parent,device='cuda:0',
                acknowledgement=model.MANUAL_ACKNOWLEDGEMENT,prototype_source_sha256=oldp['files'][PARENT_PACKAGE+'/model.py']['sha256']).eval()
            need(before==rng_record(),'Exact trained-u64 initialization changed RNG')
            named=dict(candidate.named_parameters());head_items=[(n,named[n]) for n in sorted(model.expected_head_shapes())]
            need(len(named)==575 and len(head_items)==36 and sum(v.numel() for _,v in head_items)==168960,
                 'Exact575/36/168960 model census required')
            need(all(v.requires_grad==n.startswith('heads.') and v.grad is None for n,v in named.items()),
                 'Only36 heads require gradients and every grad field must be None')
            initial=arrays(named);run_rng=rng_record()
            need(helper.canonical_arrays({n:initial[n] for n,_ in head_items},np=np)[0]==contract.HEAD_CANONICAL_SHA256,
                 'All36 initial trained-u64 heads must match retained canonical values')
            del r2,u96
            store.json('head_schema.json',{'complete':True,'head_shapes':{n:list(v.shape) for n,v in head_items},
                                          'total_tensors':575,'frozen_tensors':539,'model_qualified':False})

        def assert_unchanged():
            actual=arrays(dict(candidate.named_parameters()))
            need(set(actual)==set(initial) and all(actual[n].dtype==initial[n].dtype and
                 actual[n].shape==initial[n].shape and actual[n].tobytes()==initial[n].tobytes() for n in initial),
                 'Every one of575 parameter values must remain bitwise unchanged')
            need(all(v.grad is None for v in candidate.parameters()),'Autograd queries must not populate .grad fields')
            return actual

        def full_state(label):
            nonlocal last_complete_state,full_states_attempted
            need(full_states_attempted<2,'At most two ordinary full575 state payload attempts')
            full_states_attempted+=1
            actual=assert_unchanged();payload,description=helper.npz_state_payload(actual,np=np)
            need(len(payload)<=384*1024**2,'Full575 state384MiB cap')
            file=store.write('full_states/'+label+'.npz',payload)
            path=checked(root,file)
            with path.open('rb') as stream:
                with np.load(stream,allow_pickle=False) as decoded:
                    need(set(decoded.files)==set(actual) and all(decoded[n].dtype==actual[n].dtype and
                         decoded[n].shape==actual[n].shape and decoded[n].tobytes()==actual[n].tobytes() for n in actual),
                         'Durable full575 NPZ readback differs')
            last_complete_state=store.json('full_states/'+label+'.json',{'complete':True,'state':file,'description':description,
                'canonical_sha256':helper.canonical_arrays(actual,np=np)[0],'all575_exact_initial_values':True,
                'all539_frozen_exact':True,'gradient_fields_all_None':True,'model_qualified':False})
            return last_complete_state

        with bounded_phase('initial_full575_value_witness'):full_state('initial')

        @contextmanager
        def query_warning_scope():
            previous=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
            need(previous and not warn,'Strict deterministic forward policy required before query')
            captured=[]
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always');torch.use_deterministic_algorithms(True,warn_only=True)
                    yield
            finally:
                torch.use_deterministic_algorithms(previous,warn_only=warn)
                actual=[{'category':w.category.__name__,'message':str(w.message)} for w in captured]
                unexpected=[w for w in actual if ('deterministic' in w['message'].lower() or 'nondeterminism' in w['message'].lower())
                            and re.match(r'^upsample_bilinear2d_backward_out_cuda\b',w['message']) is None]
                event({'kind':'gradient_warning_scope','warnings':actual,'unexpected':unexpected,'strict_flags_restored':True,
                       'bitwise_CUDA_gradient_repeatability_claimed':False,'counters':dict(ledger.counters)})
                need(not unexpected,'Unexpected nondeterministic gradient operation')

        parity_records={};diagnostic_records=[]
        for call in plan:
            active_slot=call['slot'];active_case=call['id'];active_query=None;row=rows[call['id']]
            with bounded_phase('call_'+str(active_slot).zfill(4)):
                observed=rgb(row['input']);source=torch.from_numpy(observed.transpose(2,0,1).copy()).to(device=device,dtype=torch.float32)[None]/255.
                sample=F.interpolate(source,(512,512),mode='bilinear',align_corners=False)*2-1
                target=None if row.get('target') is None else torch.from_numpy(rgb(row['target']).transpose(2,0,1).copy()).to(device=device,dtype=torch.float32)[None]/255.
                category=PHASE_CATEGORY[call['phase']];resource('before_forward')
                token=ledger.reserve('forward',id=active_case,plan_slot=active_slot,category=category)
                status('forward_reserved')
                with torch.set_grad_enabled(bool(call['gradient_queries'])),torch.autocast(device_type='cuda',enabled=False):
                    triple=candidate(sample);torch.cuda.synchronize(0)
                    ledger.returned(token,synchronized_return=True);status('forward_returned')
                    valid=isinstance(triple,tuple) and len(triple)==3 and all(isinstance(v,torch.Tensor) and v.dtype==torch.float32
                         and v.device==device and tuple(v.shape)==shape and torch.isfinite(v).all().item()
                         for v,shape in zip(triple,((1,3,512,512),(1,256,1024),(1,256,16,16))))
                    if not valid:
                        retained=[]
                        for index,value in enumerate(triple if isinstance(triple,(tuple,list)) else (triple,)):
                            if isinstance(value,torch.Tensor):
                                metadata={'index':index,'shape':list(value.shape),'dtype':str(value.dtype)}
                                if value.numel()*value.element_size()<=8*1024**2:
                                    bits=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().copy()
                                    metadata['raw_bits']=save_array('invalid_returns/'+str(active_slot)+'/'+str(index)+'.npy',bits)
                                else:metadata['raw_bits_omitted_due8MiB_cap']=True
                                retained.append(metadata)
                        store.json('invalid_returns/'+str(active_slot)+'/record.json',{'accepted_as_valid':False,'values':retained})
                        raise ValueError('Invalid full return retained within cap; do not retry')
                    need(candidate.last_base_calls==1 and tuple(candidate.last_forward_sites)==tuple(model.SITES),
                         'One complete four-site full forward required')
                    native=(triple[0]+1)/2
                    unclipped=F.interpolate(native,(256,256),mode='bilinear',align_corners=False)
                    delivered=F.interpolate(native.clamp(0,1),(256,256),mode='bilinear',align_corners=False)
                    raw=delivered.detach().cpu().numpy()[0].transpose(1,2,0).copy()
                    quantized=(np.clip(raw,0,1)*255).astype(np.uint8);image=io.BytesIO();Image.fromarray(quantized).save(image,format='PNG')
                    need(len(image.getvalue())<=256*1024,'Per-image PNG256KiB cap')
                    stem='calls/'+str(active_slot).zfill(4)
                    outputs={'raw':save_array(stem+'/raw.npy',raw),'unclipped':save_array(stem+'/unclipped.npy',unclipped),
                             'png':store.write(stem+'/result.png',image.getvalue())}
                    stock={name:{**tensor_record(v),'file':save_array(stem+'/'+name+'.npy',v)} for name,v in zip(('native','logits','LQ'),triple)}
                    with Image.open(checked(root,outputs['png'])) as saved:
                        need(saved.mode=='RGB' and saved.size==(256,256) and np.array_equal(np.asarray(saved),quantized),
                             'Durable PNG pixel readback differs')
                    record={'complete':True,'call':call,'id':active_case,'reference_id':row['reference_id'],'role':row['role'],
                        'split':row['split'],'input':row['input'],'target':row.get('target'),'outputs':outputs,'stock':stock,
                        'clipping':{'native_below_minus1':int((triple[0]<-1).sum().item()),'native_above1':int((triple[0]>1).sum().item()),
                                    'unclipped_below0':int((unclipped<0).sum().item()),'unclipped_above1':int((unclipped>1).sum().item())},
                        'PNG_gradient_claimed':False,'model_qualified':False}
                    record_pin=store.json(stem+'/record.json',record)
                    if call['phase']=='historical_u64_parity':
                        witness=witnesses[active_case]
                        need(np.array_equal(raw,raw_file(witness['raw'])) and np.array_equal(quantized,rgb(witness['png'])),
                             'Exact saved historical trained-u64 RAW/PNG parity failed')
                        parity_records[active_case]=record
                    elif call['phase']=='deterministic_same_u64_repeat':
                        previous=parity_records[active_case]
                        for field in ('raw','unclipped'):
                            need({k:outputs[field][k] for k in ('bytes','sha256')}=={k:previous['outputs'][field][k] for k in ('bytes','sha256')},
                                 'Same-candidate RAW/unclipped repeat differs')
                        with Image.open(checked(root,previous['outputs']['png'])) as saved:
                            need(np.array_equal(quantized,np.asarray(saved)),'Same-candidate PNG repeat differs')
                        for name in stock:
                            need({k:v for k,v in stock[name].items() if k!='file'}=={k:v for k,v in previous['stock'][name].items() if k!='file'},
                                 'Same-candidate native/logits/LQ repeat differs')
                    else:
                        need(row['cohort']=='optimize' and row['optimization_permitted'] is True and row['split']=='TRAIN'
                             and target is not None and tuple(call['gradient_queries'])==QUERIES,'Exact admitted optimization queries required')
                        loss,terms=objective.connected_objective(delivered,target,row['role'],torch=torch)
                        residual=delivered-target
                        if call['metric_region']=='central128':residual=residual[:,:,64:192,64:192]
                        else:need(call['metric_region']=='whole','Declared gate region required')
                        mae=residual.abs().mean()
                        adjacent=.5*((residual[:,:,:,1:]-residual[:,:,:,:-1]).abs().mean()+
                                      (residual[:,:,1:,:]-residual[:,:,:-1,:]).abs().mean())
                        values={'objective_div4':loss/4,'gate_mae':mae,'gate_adjacent':adjacent}
                        gradients={}
                        for index,name in enumerate(QUERIES):
                            active_query=name;resource('before_gradient_query')
                            token=ledger.reserve('gradient_query',id=active_case,plan_slot=active_slot,query=name)
                            status('gradient_query_reserved')
                            with query_warning_scope():
                                result=torch.autograd.grad(values[name],tuple(v for _,v in head_items),retain_graph=index<2,
                                    create_graph=False,allow_unused=True,materialize_grads=False)
                                torch.cuda.synchronize(0)
                                ledger.returned(token,synchronized_return=True);status('gradient_query_returned')
                            actual={};unused=[];zero=[]
                            for (key,parameter),gradient in zip(head_items,result):
                                if gradient is None:unused.append(key);gradient=torch.zeros_like(parameter)
                                need(gradient.dtype==torch.float32 and gradient.shape==parameter.shape and torch.isfinite(gradient).all().item(),
                                     'Every gradient must be finite FP32 or explicitly unused zero')
                                if not torch.count_nonzero(gradient).item():zero.append(key)
                                actual[key]=gradient.detach().cpu().contiguous().numpy().copy()
                            payload,description=helper.npz_state_payload(actual,np=np)
                            need(len(payload)<=1024**2,'One36-head gradient NPZ1MiB cap')
                            gradient_pin=store.write(stem+'/gradients_'+name+'.npz',payload)
                            with checked(root,gradient_pin).open('rb') as stream:
                                with np.load(stream,allow_pickle=False) as decoded:
                                    need(set(decoded.files)==set(actual) and all(decoded[n].dtype==actual[n].dtype and
                                         decoded[n].shape==actual[n].shape and decoded[n].tobytes()==actual[n].tobytes() for n in actual),
                                         'Durable gradient NPZ value readback differs')
                            gradients[name]={'file':gradient_pin,'description':description,'unused_names_explicit_zero':unused,
                                'all_zero_names':zero,'scalar':float(values[name].detach().item()),
                                'canonical_sha256':helper.canonical_arrays(actual,np=np)[0]}
                            need(all(v.grad is None for v in candidate.parameters()),'autograd.grad changed a parameter .grad field')
                            del actual,result,payload
                            resource('after_gradient_query')
                        scalar={'complete':True,'call':call,'output_record':record_pin,'gradients':gradients,
                            'objective':float(loss.detach().item()),'objective_terms':{k:float(v.detach().item()) for k,v in terms.items()},
                            'objective_divided_exactly_once':True,'metric_region':call['metric_region'],
                            'RAW_differentiable_gate_metrics_only':True,'PNG_gradient_claimed':False,'model_qualified':False}
                        diagnostic_records.append(store.json(stem+'/gradients.json',scalar))
                active_query=None
                del sample,target,source,triple,native,unclipped,delivered,raw,quantized
                if call['gradient_queries']:del loss,terms,residual,mae,adjacent,values
                resource('after_forward_and_queries')

        with bounded_phase('closing_full575_value_witness_and_integrity'):
            closing=full_state('closing')
            need(rng_record()==run_rng,'Fixed diagnostic changed run RNG')
            snap=ledger.snapshot()
            need(not snap['operation_outcome_unknown'] and ledger.counters=={
                'full_forward_attempts':72,'full_forward_returns':72,'gradient_query_attempts':192,'gradient_query_returns':192,
                'parity_forwards':4,'repeat_forwards':4,'diagnostic_forwards':64},'Complete known72/192 diagnostic prefix required')
            store.audit_namespace()
            store.json('result.json',{'complete':True,'status':'ROLE_GRADIENTS_MEASURED_ZERO_UPDATES_NOT_QUALIFIED',
                'protocol_sha256':expected_sha,'closing_state':closing,'diagnostic_records':diagnostic_records,
                'all575_parameters_bitwise_unchanged':True,'all539_frozen_exact':True,'gradient_fields_all_None':True,
                'optimizer_constructions':0,'optimizer_steps':0,'parameter_applications':0,
                'PNG_gradient_claimed':False,'model_qualified':False,**snap})
            event({'kind':'run_result_saved','counters':dict(ledger.counters)});status('diagnostic_complete',complete=True)
        return 0
    except BaseException as error:
        if store is not None:
            failure={'complete':False,'protocol_sha256':expected_sha,'phase':failed_phase or phase,'active_case':active_case,
                'active_slot':active_slot,'active_query':active_query,'error_type':type(error).__name__,'error':str(error)[:4000],
                'traceback':traceback.format_exc()[-20000:],'last_complete_state':last_complete_state,
                'partial_files':sorted(store.partial_files),'optimizer_constructions':0,'optimizer_steps':0,
                'parameter_applications':0,'model_qualified':False,**(ledger.snapshot() if ledger else {})}
            if candidate is not None and initial is not None and full_states_attempted<2:
                budget=min(30.0,max(0.0,contract.LIMITS['external_seconds']-(time.monotonic()-began)-10.0))
                if budget>0:
                    try:
                        with timed(budget,'best_effort_failed_closing_value_witness'):
                            failure['failed_closing_state']=full_state('failed_closing')
                    except BaseException as capture_error:
                        failure['failed_closing_capture_error']=type(capture_error).__name__+': '+str(capture_error)[:1000]
            failure['partial_files']=sorted(store.partial_files)
            try:
                with timed(5.0,'failure_receipt'):
                    store.json('failure.json',failure,emergency=True);status('retained_failure')
            except BaseException as failed:
                print(json.dumps({'failure_receipt_incomplete':type(failed).__name__,'partial_outcome_retained':True}),flush=True)
        traceback.print_exc();return 1
    finally:
        if caller_rng is not None:
            try:
                with timed(5.0,'caller_RNG_restoration'):
                    random.setstate(caller_rng['Python']);np.random.set_state(caller_rng['NumPy'])
                    torch.set_rng_state(caller_rng['CPU']);torch.cuda.set_rng_state_all(caller_rng['CUDA'])
                    torch.cuda.synchronize(0)
            except BaseException as error:
                raise RuntimeError('Caller RNG restoration failed; diagnostic cannot report success') from error
        if old_term is not None:signal.signal(signal.SIGTERM,old_term)

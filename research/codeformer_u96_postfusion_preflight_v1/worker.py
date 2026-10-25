"""Source-only worker for one manually launched L4 zero-optimizer diagnostic.

Importing this module constructs no model and imports no Torch or NumPy.
Only run() may cross that dependency boundary, after exact packet/VM guards.
No optimizer, updates, perturbation, automatic retry or runtime entrypoint exists.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
import warnings


OUTPUT_RELATIVE = 'outputs/u96_postfusion_preflight_v1'
SITES = ('32','64','128','256')
CHANNELS = {'32':256,'64':256,'128':128,'256':128}
R2_FILE_SHA = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
U96_FILE_SHA = '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'
R2_STATE_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_STATE_SHA = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'


def need(condition,message):
    if not condition:raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def finite_json(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')


def byte_sha(value):return hashlib.sha256(value).hexdigest()


def run(root,protocol,expected_sha):
    """Execute only within a verified, fresh manual tmux L4 stage.

    Failure evidence is retained even when guards fail before Torch import.
    A preexisting output directory is immutable and is never taken over.
    """
    from . import packet

    root=Path(root).absolute();out=root/OUTPUT_RELATIVE
    need(not out.exists() and not out.is_symlink(),
         'Prior output directory is immutable; no resume/retry or overwrite')
    # Nothing has been claimed or created yet. Verify/require_vm must precede
    # started.json because require_vm rejects a prior started run.
    start=time.monotonic();owned=False;torch=None;np=None;process=None;p=None
    phase='packet_and_VM_guards';failed_phase=None;events=None;resources=None
    counters={'full_forward_attempts':0,'full_forward_returns':0,
              'reference_forwards':0,'candidate_parity_forwards':0,
              'candidate_gradient_forwards':0,'gradient_queries':0,
              'gradient_returns':0,'optimizer_constructions':0,'optimizer_updates':0,
              'frozen_U96_adapter_site_reservations':0,'frozen_U96_adapter_site_returns':0,
              'new_head_site_reservations':0,'new_head_site_returns':0}
    limits=dict(packet.LIMITS)
    reference=None;candidate=None;initial_states=None;final_states=None
    gpu_guard=None;factory_rng=None;run_rng_before=None
    model_construction_attempted=False
    resource_module=None
    gradient_warning_history=[]

    def claim_output():
        nonlocal owned
        need(not out.exists() and not out.is_symlink(),'Output ownership conflict; preserve existing run')
        root_abs=packet.directory(root)
        need(root_abs==root.resolve() and out.parent==root/'outputs','Output scope differs')
        if out.parent.exists():packet.directory(out.parent)
        else:out.parent.mkdir()
        out.mkdir(exist_ok=False);owned=True

    def safe_output(path):
        path=Path(path).absolute()
        need(owned and path.is_relative_to(out),'Worker writes only below its owned output directory')
        packet.directory(out)
        current=path.parent
        missing=[]
        while not current.exists():missing.append(current);current=current.parent
        packet.directory(current)
        for directory in reversed(missing):directory.mkdir(exist_ok=False)
        if path.exists() or path.is_symlink():
            packet.regular(path)
        return path

    def output_size():
        total=0
        if not owned:return 0
        for item in out.rglob('*'):
            if item.is_dir():packet.directory(item)
            else:total+=packet.regular(item).stat().st_size
        return total

    def write_bytes(path,payload,*,emergency=False):
        path=safe_output(path)
        # Keep64KiB inside the declared cap for a durable failure record.
        ceiling=limits['maximum_output_bytes']-(0 if emergency else 65536)
        need(output_size()+len(payload)<=ceiling,'Bounded128MiB output cap exceeded')
        with path.open('xb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        return packet.pin(root,path)

    def write_json(path,value,*,emergency=False):
        return write_bytes(path,finite_json(value),emergency=emergency)

    def append_json(path,value):
        payload=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
        path=safe_output(path)
        need(path.exists(),'Append requires an exclusively created log')
        need(output_size()+len(payload)<=limits['maximum_output_bytes']-65536,
             'Output cap before durable log append')
        with path.open('ab') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())

    def event(kind,**fields):
        if events is not None:append_json(events,{'UTC':utc(),'seconds':time.monotonic()-start,
            'phase':phase,'kind':kind,'counters':dict(counters),**fields})

    def resource(label):
        elapsed=time.monotonic()-start
        need(elapsed<limits['internal_seconds'],'Internal1200s stop')
        free=__import__('shutil').disk_usage(root).free
        if process is not None:
            rss=int(process.memory_info().rss)
            current_rss_source='psutil.Process.memory_info().rss'
        else:
            # Before numerical dependencies import, Linux /proc supplies the
            # same current-resident measure without importing psutil early.
            rss=int(Path('/proc/self/statm').read_text().split()[1])*int(os.sysconf('SC_PAGE_SIZE'))
            current_rss_source='Linux /proc/self/statm resident pages times SC_PAGE_SIZE'
        # Linux ru_maxrss is KiB and retains transient process peaks between
        # current-RSS samples. The worker may reach this only after require_vm.
        peak_rss=int(resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss)*1024 if resource_module is not None else 0
        reserved=0;allocated=0;peak_reserved=0
        if torch is not None and torch.cuda.is_initialized():
            reserved=int(torch.cuda.memory_reserved(0));allocated=int(torch.cuda.memory_allocated(0))
            peak_reserved=int(torch.cuda.max_memory_reserved(0))
        size=output_size()
        row={'UTC':utc(),'seconds':elapsed,'phase':phase,'label':label,
             'free_disk_bytes':free,'current_RSS_bytes':rss,'peak_RSS_bytes':peak_rss,
             'current_RSS_measurement':current_rss_source,
             'peak_RSS_measurement':'Linux resource.getrusage(RUSAGE_SELF).ru_maxrss KiB converted to bytes',
             'CUDA_reserved_bytes':reserved,
             'CUDA_allocated_bytes':allocated,'CUDA_peak_reserved_bytes':peak_reserved,
             'output_bytes':size}
        if resources is not None:append_json(resources,row)
        need(free>=limits['continuing_disk_reserve'],'Continuing4GiB disk reserve stop')
        need(rss<=limits['maximum_RSS_bytes'] and peak_rss<=limits['maximum_RSS_bytes'],
             'Current or Linux peak RSS10GiB stop')
        need(reserved<=limits['maximum_CUDA_reserved_bytes'] and
             peak_reserved<=limits['maximum_CUDA_reserved_bytes'],'CUDA reserved18GiB stop')
        need(size<=limits['maximum_output_bytes'],'Output128MiB stop')
        return row

    @contextmanager
    def bounded_phase(name):
        nonlocal phase,failed_phase
        old_phase=phase;phase=name;began=time.monotonic()
        resource('before');event('phase_before')
        remaining=limits['internal_seconds']-(time.monotonic()-start)
        need(remaining>0,'Internal budget exhausted before phase')
        timeout=min(float(limits['per_phase_seconds']),remaining)
        previous=signal.getsignal(signal.SIGALRM)
        def expired(signum,frame):raise TimeoutError('Finite phase stop: '+name)
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,timeout)
        try:
            yield
            resource('after');event('phase_after',phase_seconds=time.monotonic()-began)
        except BaseException:
            # The outer receipt runs after finally restores phase. Preserve the
            # actual failing phase independently, without changing the failure.
            failed_phase=name
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
            phase=old_phase

    def checked(row):
        need(type(row) is dict and set(row)=={'path','bytes','sha256'},'Exact direct file pin required')
        path=packet.regular(root/row['path'])
        need(packet.pin(root,path)==row,'Bound source/model/input file changed')
        return path

    try:
        p=packet.verify(root,expected_sha)
        need(p==protocol,'Supplied protocol object must equal exact verified protocol')
        facts=packet.require_vm(root,p)
        import resource as linux_resource
        resource_module=linux_resource
        need(p['limits']==packet.LIMITS,'Worker requires exact packet resource contract')
        need(all(p['witnesses'][index]['role']=='clear' and
                 p['witnesses'][index+1]['role']=='local_gaussian_1p2' and
                 p['witnesses'][index]['reference_id']==p['witnesses'][index+1]['reference_id'] and
                 p['witnesses'][index]['reference_id'].startswith('tr_')
                 for index in range(0,8,2)),
             'Four fixed TRAIN reference pairs must remain clear then localGaussian1.2')
        claim_output()
        write_json(out/'started.json',{'UTC':utc(),'protocol_sha256':expected_sha,
            'host':facts,'stage':p['stage_name'],'status':'STARTED_NOT_TRAINED',
            'training_optimizer_updates':0,'model_qualified':False})
        events=out/'events.jsonl';resources=out/'resources.jsonl'
        write_bytes(events,b'');write_bytes(resources,b'')
        event('verified_before_Torch_import',Torch_imported=False)
        # Do not initialize CUDA while another compute workload owns any GPU.
        with bounded_phase('GPU_idle_and_device_guard_before_Torch'):
            apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
                '--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
            names=subprocess.run(['nvidia-smi','--query-gpu=name,uuid',
                '--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
            need(not apps.stdout.strip(),'Another GPU compute application exists; do not compete')
            name_lines=[line.strip() for line in names.stdout.splitlines() if line.strip()]
            need(len(name_lines)==1 and name_lines[0].split(',',1)[0].strip()=='NVIDIA L4',
                 'Exactly one GPU with exact NVIDIA L4 name required')
            gpu_guard={'compute_apps_before_Torch':apps.stdout,'GPU_inventory_before_Torch':name_lines}
            event('GPU_guard_pass',**gpu_guard)

        with bounded_phase('guarded_dependencies_and_CUDA_configuration'):
            # Deterministic CUDA matmul requires this before CUDA context setup.
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            need(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8',':16:8'),
                 'Known deterministic cuBLAS workspace configuration required')
            import torch as torch_module
            import numpy as numpy_module
            import psutil
            from PIL import Image
            from torch.nn import functional as F
            torch=torch_module;np=numpy_module;process=psutil.Process(os.getpid())
            need(torch.cuda.is_available() and torch.cuda.device_count()==1,'One available CUDA GPU required')
            need(torch.cuda.get_device_name(0)=='NVIDIA L4','CUDA device name must equal NVIDIA L4 exactly')
            torch.cuda.set_device(0);device=torch.device('cuda',0)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.use_deterministic_algorithms(True,warn_only=False)
            torch.cuda.reset_peak_memory_stats(0)
            need(not torch.is_inference_mode_enabled(),'Worker requires ordinary autograd-capable runtime')

        def canonical(state):
            digest=hashlib.sha256()
            for name,value in sorted(state.items()):
                value=value.detach().cpu().contiguous()
                header=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode('utf-8')
                digest.update(len(header).to_bytes(8,'big'));digest.update(header)
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        def tensor_record(value):
            need(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and
                 torch.isfinite(value).all().item(),'Finite FP32 returned tensor required')
            array=value.detach().cpu().contiguous().numpy()
            return {'shape':list(array.shape),'dtype':str(array.dtype),'bytes':int(array.nbytes),
                    'RGB_or_tensor_bytes_sha256':byte_sha(array.tobytes()),
                    'minimum':float(array.min()),'maximum':float(array.max()),
                    'negative_zero_count':int(np.count_nonzero((array==0)&np.signbit(array)))}

        def rng_snapshot():
            cpu=torch.random.get_rng_state().cpu().numpy().tobytes()
            cuda=[value.cpu().numpy().tobytes() for value in torch.cuda.get_rng_state_all()]
            return {'CPU_sha256':byte_sha(cpu),'CUDA_sha256':[byte_sha(x) for x in cuda],
                    'CPU_bytes':len(cpu),'CUDA_bytes':[len(x) for x in cuda]}

        def assert_clean(net):
            need(all(not mod._forward_pre_hooks and not mod._forward_hooks for mod in net.modules()),
                 'Temporary hooks leaked from completed model call')

        def adapter_state_of(net):return {'adapters.'+name:value for name,value in net.adapters.state_dict().items()}

        def states_of(net,include_heads):
            base=net.base.state_dict();adapters=adapter_state_of(net)
            need(len(base)==515 and len(adapters)==24,'Incumbent515+24 census mismatch')
            need(canonical(base)==R2_STATE_SHA and canonical(adapters)==U96_STATE_SHA,
                 'Frozen incumbent tensors changed')
            frozen=list(net.base.parameters())+list(net.adapters.parameters())
            need(len(frozen)==539 and all(not v.requires_grad and v.grad is None for v in frozen),
                 'All539 frozen gradients must remain None')
            need(all(not mod.training for mod in net.base.modules()) and
                 all(not mod.training for mod in net.adapters.modules()),'Frozen incumbent modules must stay eval')
            result={'R2_canonical515_sha256':R2_STATE_SHA,'U96_canonical24_sha256':U96_STATE_SHA,
                    'frozen_tensors':539,'frozen_grad_fields_None':539,
                    'full_state_sha256':canonical(net.state_dict()),'full_state_tensors':len(net.state_dict())}
            if include_heads:
                heads={'heads.'+name:value for name,value in net.heads.state_dict().items()}
                need(len(heads)==36 and sum(x.numel() for x in heads.values())==168960,'Head36/168960 census mismatch')
                need(len(net.state_dict())==575,'Candidate575 state tensors required')
                result.update({'heads_canonical_sha256':canonical(heads),'head_tensors':36,'head_parameters':168960})
            else:need(len(net.state_dict())==539,'Independent reference539 state tensors required')
            assert_clean(net);return result

        def save_array(path,value,*,allow_nonfinite_failure_evidence=False):
            array=value.detach().cpu().contiguous().numpy().copy() if isinstance(value,torch.Tensor) else np.asarray(value)
            need(array.dtype==np.float32 and (allow_nonfinite_failure_evidence or np.isfinite(array).all()),
                 'Saved arrays must be float32; nonfinite values only retained as explicit failed-gradient evidence')
            stream=io.BytesIO();np.save(stream,array,allow_pickle=False)
            return write_bytes(path,stream.getvalue())

        def save_delivery_array(path,value):
            need(value.shape==(1,3,256,256),'Expected singleton NCHW delivery before saved HWC conversion')
            return save_array(path,value[0].permute(1,2,0))

        def save_head_npz(path,state):
            need(len(state)==36,'Only36 untrained head arrays may enter portable NPZ')
            arrays={};schema=[]
            for name,value in sorted(state.items()):
                array=value.detach().cpu().contiguous().numpy().copy()
                need(array.dtype==np.float32 and np.isfinite(array).all(),
                     'Finite FP32 untrained head NPZ arrays required')
                arrays[name]=array
                schema.append({'name':name,'shape':list(array.shape),'dtype':str(array.dtype),
                               'bytes_sha256':byte_sha(array.tobytes())})
            stream=io.BytesIO();np.savez(stream,**arrays)
            file_pin=write_bytes(path,stream.getvalue())
            # Check the durable format immediately without pickle semantics.
            with np.load(path,allow_pickle=False) as readback:
                need(set(readback.files)==set(arrays) and all(np.array_equal(readback[name],value)
                     for name,value in arrays.items()),'Saved36-head NPZ readback differs')
            return {'file':file_pin,'arrays':schema,'canonical_sha256':canonical(state)}

        def save_png(path,raw):
            array=raw.detach().cpu().contiguous().numpy().copy()
            need(array.shape==(1,3,256,256) and array.dtype==np.float32 and
                 np.isfinite(array).all() and array.min()>=0 and array.max()<=1,
                 'Finite clipped singleton delivery required')
            image=np.floor(array[0].transpose(1,2,0)*np.float32(255)).astype(np.uint8)
            stream=io.BytesIO();Image.fromarray(image,mode='RGB').save(stream,format='PNG')
            return write_bytes(path,stream.getvalue())

        def deliver(native):
            # Ordinary connected operations: do not detach before gradient loss.
            unclipped=F.interpolate((native.float()+1)/2,(256,256),mode='bilinear',align_corners=False)
            raw=F.interpolate(((native.float()+1)/2).clamp(0,1),(256,256),mode='bilinear',align_corners=False)
            return raw,unclipped

        with bounded_phase('load_exact_components_and_construct_independent_reference'):
            r2_path=checked(p['model']['r2']);u96_path=checked(p['model']['u96'])
            need(packet.sha256(r2_path)==R2_FILE_SHA and packet.sha256(u96_path)==U96_FILE_SHA,
                 'Fixed incumbent artifact hashes differ')
            saved_r2=torch.load(r2_path,map_location='cpu',weights_only=True,mmap=True)
            saved_u96=torch.load(u96_path,map_location='cpu',weights_only=True,mmap=True)
            need(type(saved_r2) is dict and set(saved_r2)=={'params_ema'},'Exact inference-only R2 format required')
            need(type(saved_u96) is dict and set(saved_u96)=={'format','adapter_state'} and
                 saved_u96['format']=='codeformer-spatial-adapter-inference-v1','Exact inference-only U96 format required')
            base_state=saved_r2['params_ema'];adapter_state=saved_u96['adapter_state']
            need(type(base_state) is dict and len(base_state)==515 and canonical(base_state)==R2_STATE_SHA,
                 'Canonical R2 initializer mismatch')
            need(type(adapter_state) is dict and len(adapter_state)==24 and canonical(adapter_state)==U96_STATE_SHA,
                 'Canonical U96 initializer mismatch')
            # Imports are now permitted, but validate their exact module origins
            # before constructing either reference or candidate.
            from third_party.codeformer.codeformer_arch import CodeFormer
            from research.codeformer_u96_postfusion_v1 import model as prototype
            need(Path(prototype.__file__).resolve()==
                 (root/'research/codeformer_u96_postfusion_v1/model.py').resolve(),
                 'Candidate factory must come from the exact staged prototype module')
            for name,binding in prototype.ARCHITECTURE_SOURCES.items():
                loaded=sys.modules.get(name);need(loaded is not None and loaded.__file__,'Missing architecture module')
                path=Path(loaded.__file__).resolve();need(path==(root/binding['path']).resolve() and
                    packet.sha256(path)==binding['sha256'],'Exact staged architecture origin/hash required')
            need(prototype.R2_CANONICAL_SHA256==R2_STATE_SHA and prototype.U96_CANONICAL_SHA256==U96_STATE_SHA,
                 'Prototype exact complete-U96 initializer differs')
            need(set(adapter_state)==set(prototype.expected_adapter_shapes()),'Adapter schema differs')
            need(all(v.dtype==torch.float32 and not v.requires_grad and tuple(v.shape)==prototype.expected_adapter_shapes()[k]
                 and torch.isfinite(v).all().item() for k,v in adapter_state.items()),'Adapter finite schema mismatch')

            class ReferenceAdapter(torch.nn.Module):
                def __init__(self,channels):
                    super().__init__();options={'device':'cpu','dtype':torch.float32}
                    self.down=torch.nn.Conv2d(channels,32,1,bias=True,**options)
                    self.depthwise=torch.nn.Conv2d(32,32,3,padding=1,groups=32,bias=True,**options)
                    self.up=torch.nn.Conv2d(32,channels,1,bias=True,**options)
                    self.activation=torch.nn.SiLU(inplace=False)
                def forward(self,observed):
                    return observed+self.up(self.activation(self.depthwise(self.activation(self.down(observed)))))

            class IndependentU96Reference(torch.nn.Module):
                def __init__(self,base):
                    super().__init__();self.base=base
                    self.adapters=torch.nn.ModuleDict({site:ReferenceAdapter(CHANNELS[site]) for site in SITES})
                    self.adapters.load_state_dict({k.removeprefix('adapters.'):v for k,v in adapter_state.items()},strict=True)
                    self.last_forward_sites=();self.last_base_calls=0
                def forward(self,sample):
                    need(not torch.is_grad_enabled(),'Independent reference must use no_grad')
                    assert_clean(self);handles=[];sites=[];self.last_forward_sites=();self.last_base_calls=0
                    def hook(site):
                        def adapt(module,args):
                            need(len(args)==3 and args[2]==1. and len(sites)<4 and site==SITES[len(sites)],
                                 'One ordered independent U96 adaptation per site required')
                            observed,decoder,w=args;shape=(1,CHANNELS[site],int(site),int(site))
                            need(observed.shape==decoder.shape==shape and observed.dtype==torch.float32 and
                                 not observed.requires_grad,'Independent U96 observed feature shape differs')
                            used=self.adapters[site](observed)
                            need(used.shape==shape and torch.isfinite(used).all().item(),'Independent U96 adapter invalid')
                            sites.append(site);return used,decoder,w
                        return adapt
                    try:
                        for site in SITES:handles.append(self.base.fuse_convs_dict[site].register_forward_pre_hook(hook(site)))
                        self.last_base_calls=1
                        result=self.base(sample,w=1.,detach_16=True,adain=True)
                        need(tuple(sites)==SITES,'All independent U96 sites must execute')
                        return result
                    finally:
                        for handle in reversed(handles):handle.remove()
                        self.last_forward_sites=tuple(sites)

            def base_factory():
                nonlocal model_construction_attempted
                model_construction_attempted=True
                base=CodeFormer(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=list(SITES))
                base.load_state_dict(base_state,strict=True)
                return base.float().eval().requires_grad_(False).to(device)

            rng_before_reference=rng_snapshot()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261021)
                reference=IndependentU96Reference(base_factory()).float().eval().requires_grad_(False).to(device)
            need(rng_snapshot()==rng_before_reference,'Independent construction changed caller CPU/CUDA RNG')
            states_of(reference,False)

        with bounded_phase('construct_untrained_candidate_preserving_factory_RNG'):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261021)
                candidate_base=base_factory()
            before_factory=rng_snapshot()
            candidate=prototype.build_restorer(candidate_base,adapter_state,
                r2_artifact=r2_path,u96_artifact=u96_path).eval()
            after_factory=rng_snapshot();need(before_factory==after_factory,'Prototype factory changed CPU/CUDA RNG')
            factory_rng={'before':before_factory,'after':after_factory,'exact':True,
                         'reference_factory_CPU_CUDA_preserved':True}
            initial_states={'reference':states_of(reference,False),'candidate':states_of(candidate,True)}
            prototype_check=candidate.validate_invariants(check_frozen_digests=True)
            head_state={'heads.'+name:value.detach().cpu().clone() for name,value in candidate.heads.state_dict().items()}
            need(len(head_state)==36,'Save36 untrained head tensors only')
            stream=io.BytesIO();torch.save(head_state,stream);initial_head_pin=write_bytes(out/'initial_heads.pt',stream.getvalue())
            initial_head_npz=save_head_npz(out/'initial_heads.npz',head_state)
            need(all(torch.count_nonzero(head.output.weight).item()==0 and torch.count_nonzero(head.output.bias).item()==0
                     for head in candidate.heads.values()),'Zero final head projections required')
            run_rng_before=rng_snapshot()
            write_json(out/'preflight.json',{'complete':True,'status':'PRE_FORWARD_INTEGRITY_ONLY',
                'protocol_sha256':expected_sha,'host':facts,'gpu_guard':gpu_guard,
                'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,'CUDA':torch.version.cuda,
                'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],
                'initial_states':initial_states,'prototype_invariants':prototype_check,
                'factory_RNG':factory_rng,'initial_heads':initial_head_pin,
                'initial_heads_NPZ':initial_head_npz,
                'limits':limits,'display_processing':'none','optimizer_constructions':0,
                'optimizer_updates':0,'model_qualified':False,'actual_training_permitted':False})

        def input_values(row):
            need(re.fullmatch(r'[A-Za-z0-9_\-]{1,120}',row['id']) is not None,'Safe unique witness ID required')
            arrays=[]
            for field in ('input','target'):
                path=checked(row[field])
                with Image.open(path) as image:
                    need(image.mode=='RGB' and image.size==(256,256) and getattr(image,'n_frames',1)==1,
                         'All witnesses must be single RGB256 full-support inputs')
                    array=np.asarray(image).copy()
                need(byte_sha(array.tobytes())==row[field+'_RGB8_sha256'],'Witness RGB bytes differ')
                arrays.append(torch.from_numpy(array.transpose(2,0,1).copy()).to(device=device,dtype=torch.float32)[None]/255.)
            sample=F.interpolate(arrays[0],(512,512),mode='bilinear',align_corners=False)*2-1
            need(sample.shape==(1,3,512,512) and not sample.requires_grad,'Fixed prepared512 geometry required')
            return sample,arrays[1]

        def full_call(net,sample,kind):
            need(torch.are_deterministic_algorithms_enabled() and
                 not torch.is_deterministic_algorithms_warn_only_enabled(),
                 'Every full forward must retain strict deterministic mode')
            need(kind in ('reference_forwards','candidate_parity_forwards','candidate_gradient_forwards'),'Unknown call reservation')
            need(counters['full_forward_attempts']<limits['full_forward_attempts'] and counters[kind]<limits[kind],
                 'Finite full-forward reservation exhausted')
            resource('before_reserved_forward');counters['full_forward_attempts']+=1;counters[kind]+=1
            counters['frozen_U96_adapter_site_reservations']+=4
            if kind!='reference_forwards':counters['new_head_site_reservations']+=4
            event('forward_reserved',forward_kind=kind)
            with torch.autocast(device_type='cuda',enabled=False):result=net(sample)
            torch.cuda.synchronize(0)
            need(isinstance(result,tuple) and len(result)==3,'Expected native/logits/LQ stock tuple')
            for value,shape in zip(result,[(1,3,512,512),(1,256,1024),(1,256,16,16)]):
                need(isinstance(value,torch.Tensor) and tuple(value.shape)==shape and value.dtype==torch.float32 and
                     value.device==device and torch.isfinite(value).all().item(),'Exact finite stock output shapes required')
            need(tuple(net.last_forward_sites)==SITES and net.last_base_calls==1,'Four-site lifecycle or base-call count differs')
            assert_clean(net)
            counters['full_forward_returns']+=1;counters['frozen_U96_adapter_site_returns']+=4
            if kind!='reference_forwards':counters['new_head_site_returns']+=4
            event('forward_returned',forward_kind=kind,sites=list(net.last_forward_sites),
                  module_site_counts_are_not_extra_full_forwards=True)
            resource('after_reserved_forward');return result

        def exact_compare(left,right):
            need(left.shape==right.shape and left.dtype==right.dtype==torch.float32,'Comparable FP32 tensors required')
            l=left.detach().cpu().contiguous().numpy();r=right.detach().cpu().contiguous().numpy()
            equal=bool(np.array_equal(l,r));bits=bool(l.tobytes()==r.tobytes())
            result={'exact_value_equal':equal,'exact_bits_equal':bits,
                    'maximum_absolute_error':float(np.max(np.abs(l.astype(np.float64)-r.astype(np.float64)))),
                    'unequal_values':int(np.count_nonzero(l!=r)),
                    'signed_zero_mismatches':int(np.count_nonzero((l==0)&(r==0)&(np.signbit(l)!=np.signbit(r)))),
                    'reference':tensor_record(left),'candidate':tensor_record(right)}
            return result

        parity_receipts=[]
        for row in p['witnesses']:
            with bounded_phase('parity_'+row['id']):
                sample,target=input_values(row);folder=out/'parity'/row['id']
                before_rng=rng_snapshot()
                with torch.no_grad():
                    baseline=full_call(reference,sample,'reference_forwards')
                    tested=full_call(candidate,sample,'candidate_parity_forwards')
                    baseline_raw,baseline_unclipped=deliver(baseline[0])
                    candidate_raw,candidate_unclipped=deliver(tested[0])
                pairs={name:exact_compare(left,right) for name,left,right in
                       [('native512',baseline[0],tested[0]),('logits',baseline[1],tested[1]),
                        ('LQ',baseline[2],tested[2]),('raw',baseline_raw,candidate_raw),
                        ('unclipped',baseline_unclipped,candidate_unclipped)]}
                saved={}
                for arm,triple,raw,unclipped in [('reference',baseline,baseline_raw,baseline_unclipped),
                                                ('candidate',tested,candidate_raw,candidate_unclipped)]:
                    saved[arm]={}
                    for label,value in [('native512',triple[0]),('logits',triple[1]),('LQ',triple[2])]:
                        saved[arm][label]=save_array(folder/(arm+'-'+label+'.npy'),value)
                    for label,value in [('raw',raw),('unclipped',unclipped)]:
                        saved[arm][label]=save_delivery_array(folder/(arm+'-'+label+'.npy'),value)
                    saved[arm]['PNG']=save_png(folder/(arm+'.png'),raw)
                png_equal=(folder/'reference.png').read_bytes()==(folder/'candidate.png').read_bytes()
                after_rng=rng_snapshot()
                receipt={'complete':True,'id':row['id'],'reference_id':row['reference_id'],'role':row['role'],
                    'comparisons':pairs,'PNG_bytes_exact':png_equal,'saved':saved,
                    'saved_delivery_layout':'float32 HWC256x256x3; full stock tensors retain original singleton shapes',
                    'RNG_before':before_rng,'RNG_after':after_rng,'RNG_exact':before_rng==after_rng,
                    'counters':dict(counters),'input':row['input'],'target':row['target'],
                    'same_device_reference_not_CPU_L4_parity':True,'model_qualified':False}
                receipt_pin=write_json(folder/'parity.json',receipt);parity_receipts.append(receipt_pin)
                need(all(value['exact_value_equal'] for value in pairs.values()) and png_equal,
                     'Initial complete-U96 exact value/native/RAW/PNG parity failed; retain stop')
                need(before_rng==after_rng,'Parity forwards changed RNG')
                del baseline,tested,baseline_raw,baseline_unclipped,candidate_raw,candidate_unclipped,sample,target

        def regional_objective(raw,target):
            residual=raw-target;inside=torch.zeros((256,256),dtype=torch.bool,device=device)
            inside[64:192,64:192]=True;terms={}
            for name,region in [('central128',inside),('outside',~inside)]:
                terms[name+'_image_L1']=residual.abs()[0,:,region].mean()
                horizontal=region[:,:-1]&region[:,1:];vertical=region[:-1,:]&region[1:,:]
                dh=(residual[:,:,:,1:]-residual[:,:,:,:-1]).abs()
                dv=(residual[:,:,1:,:]-residual[:,:,:-1,:]).abs()
                denominator=(horizontal.sum()+vertical.sum())*3
                need(int(denominator.item())>0,'Regional adjacent edge count must be positive')
                terms[name+'_adjacent_residual_L1']=(dh[0,:,horizontal].sum()+dv[0,:,vertical].sum())/denominator
            image=.5*(terms['central128_image_L1']+terms['outside_image_L1'])
            adjacent=.5*(terms['central128_adjacent_residual_L1']+terms['outside_adjacent_residual_L1'])
            objective=image+.25*adjacent
            return objective,{**terms,'equal_mean_image_L1':image,'equal_mean_adjacent_residual_L1':adjacent}

        gradient_receipts=[]
        @contextmanager
        def gradient_warn_only_scope(witness_id):
            """Only this gradient query may warn for CUDA bilinear backward.

            Retain strict mode for all forward/parity operations. This does not
            certify repeatable gradients; save actual arrays and warnings, and
            restore both global deterministic flags even after query failure.
            """
            before={'enabled':torch.are_deterministic_algorithms_enabled(),
                    'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}
            need(before=={'enabled':True,'warn_only':False},'Strict deterministic mode required before gradient query')
            entry={'id':witness_id,'scope':'one autograd.grad query and CUDA synchronization only',
                   'before':before,'during':{'enabled':True,'warn_only':True},
                   'warnings':[],'CUDA_bilinear_backward_may_be_nondeterministic':True,
                   'warn_only_applies_to_all_operations_inside_this_query_scope':True,
                   'deterministic_gradient_replay_claim':False}
            gradient_warning_history.append(entry)
            caught=[]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    torch.use_deterministic_algorithms(True,warn_only=True)
                    yield entry
                finally:
                    torch.use_deterministic_algorithms(before['enabled'],warn_only=before['warn_only'])
                    entry['after']={'enabled':torch.are_deterministic_algorithms_enabled(),
                                    'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}
                    entry['restored_exact']=entry['after']==before
                    entry['warnings']=[{'category':w.category.__name__,'message':str(w.message)[:12000],
                                        'filename':str(w.filename),'lineno':int(w.lineno)} for w in caught]
                    entry['expected_nondeterministic_operator']='upsample_bilinear2d_backward_out_cuda'
                    entry['unexpected_nondeterminism_warnings']=[w for w in entry['warnings']
                        if ('deterministic' in w['message'].lower() or 'nondeterminism' in w['message'].lower()) and
                           re.match(r'^upsample_bilinear2d_backward_out_cuda\b',w['message']) is None]
                    event('gradient_determinism_scope',**entry)
                    need(entry['restored_exact'],'Restore strict deterministic mode after gradient query')
        head_items=[(name,value) for name,value in candidate.named_parameters() if name.startswith('heads.')]
        need(len(head_items)==36 and all(value.requires_grad for _,value in head_items),'Exactly36 trainable head query inputs required')
        for row in p['witnesses']:
            if not row['gradient_witness']:continue
            with bounded_phase('gradient_'+row['id']):
                sample,target=input_values(row);folder=out/'gradient'/row['id'];before_rng=rng_snapshot()
                with torch.enable_grad():
                    tested=full_call(candidate,sample,'candidate_gradient_forwards')
                    raw,unclipped=deliver(tested[0]);need(raw.requires_grad and raw.grad_fn is not None,
                        'Connected clamp-resize image delivery required for gradient objective')
                    objective,terms=regional_objective(raw,target)
                    need(objective.requires_grad and torch.isfinite(objective).item(),'Finite connected scalar objective required')
                    saved={'raw':save_delivery_array(folder/'candidate-raw.npy',raw),
                           'PNG':save_png(folder/'candidate.png',raw)}
                    need(counters['gradient_queries']<limits['gradient_queries'],'Finite4 gradient query cap exhausted')
                    resource('before_reserved_gradient');counters['gradient_queries']+=1;event('gradient_reserved',id=row['id'])
                    with gradient_warn_only_scope(row['id']) as gradient_determinism:
                        gradients=torch.autograd.grad(objective,tuple(value for _,value in head_items),
                                                      retain_graph=False,create_graph=False,allow_unused=False)
                        torch.cuda.synchronize(0)
                    counters['gradient_returns']+=1
                    resource('after_reserved_gradient');event('gradient_returned',id=row['id'])
                tensor_rows=[]
                for (name,value),gradient in zip(head_items,gradients):
                    need(gradient is not None and gradient.shape==value.shape,
                         'Every requested head gradient must be present with the declared shape')
                    flat=gradient.detach().double()
                    is_finite=bool(torch.isfinite(gradient).all().item())
                    gradient_file=save_array(folder/'gradients'/(name+'.npy'),gradient,
                        allow_nonfinite_failure_evidence=True)
                    tensor_rows.append({'name':name,'shape':list(gradient.shape),
                        'L2_norm':float(torch.linalg.vector_norm(flat).item()) if is_finite else None,
                        'maximum_absolute':float(flat.abs().max().item()) if is_finite else None,
                        'nonzero_count':int(torch.count_nonzero(flat).item()),'finite':is_finite,
                        'nonfinite_count':int((~torch.isfinite(gradient)).sum().item()),
                        'array':gradient_file,'array_dtype':'float32','array_shape':list(gradient.shape)})
                finite_gate=all(r['finite'] for r in tensor_rows)
                projections={site:sum((r['L2_norm'] or 0.)**2 for r in tensor_rows
                    if r['name'] in ('heads.'+site+'.output.weight','heads.'+site+'.output.bias'))**.5 for site in SITES}
                projection_gate=finite_gate and all(value>0 for value in projections.values())
                need(all(value.grad is None for value in candidate.parameters()),
                     'autograd.grad must leave parameter grad fields None; no backward accumulation')
                need(all(not value.requires_grad and value.grad is None for value in
                         list(candidate.base.parameters())+list(candidate.adapters.parameters())),
                     'All539 frozen incumbent gradient fields must stay None')
                after_rng=rng_snapshot();need(before_rng==after_rng,'Gradient witness changed RNG')
                clipping={'native_channels_below_minus1':int((tested[0].detach() < -1).sum().item()),
                          'native_channels_above1':int((tested[0].detach() > 1).sum().item()),
                          'native_channel_count':int(tested[0].numel()),
                          'unclipped256':tensor_record(unclipped),'clipped_RAW256':tensor_record(raw)}
                receipt={'complete':True,'id':row['id'],'reference_id':row['reference_id'],'role':row['role'],
                    'objective':float(objective.detach().item()),
                    'terms':{key:float(value.detach().item()) for key,value in terms.items()},
                    'formula':'equal_mean central/outside imageL1 +0.25 equal_mean regional adjacent-residualL1',
                    'known_synthetic_TRAIN_supervision_not_deployment_mask':True,
                    'gradients':tensor_rows,'output_projection_L2_by_site':projections,
                    'all_joint_gradients_finite_gate_pass':finite_gate,
                    'output_projection_gradient_gate_pass':projection_gate,
                    'zero_earlier_layer_gradients_allowed_at_zero_output_initialization':True,
                    'connected_clamp_resize_delivery':True,'clipping':clipping,
                    'gradient_determinism':gradient_determinism,
                    'gradient_arrays_saved':36,'gradient_norms_independently_recomputable_from_saved_arrays':True,
                    'saved':saved,'frozen_grad_fields_None':539,'all_parameter_grad_fields_None':575,
                    'sites':list(candidate.last_forward_sites),'counters':dict(counters),
                    'RNG_before':before_rng,'RNG_after':after_rng,'RNG_exact':True,
                    'optimizer_updates':0,'weight_perturbations':0,'model_qualified':False}
                gradient_receipts.append(write_json(folder/'gradient.json',receipt))
                need(finite_gate,'Every requested head gradient must be finite; retain saved failure arrays')
                need(not gradient_determinism['unexpected_nondeterminism_warnings'],
                     'Unexpected nondeterministic gradient operator warning; retain arrays and stop')
                need(projection_gate,
                     'Each of four final output projections needs a nonzero finite image-gradient witness')
                del tested,raw,unclipped,objective,terms,gradients,gradient,flat,sample,target

        with bounded_phase('final_frozen_state_RNG_and_source_readback'):
            need(counters['full_forward_attempts']==counters['full_forward_returns']==20 and
                 counters['reference_forwards']==8 and counters['candidate_parity_forwards']==8 and
                 counters['candidate_gradient_forwards']==4 and
                 counters['gradient_queries']==counters['gradient_returns']==4,
                 'Exact20 full forwards/4 gradient queries required')
            need(counters['frozen_U96_adapter_site_reservations']==counters['frozen_U96_adapter_site_returns']==80 and
                 counters['new_head_site_reservations']==counters['new_head_site_returns']==48,
                 'Expected80 frozen adapter and48 new-head site invocations; not extra full model calls')
            candidate.validate_invariants(check_frozen_digests=True)
            final_states={'reference':states_of(reference,False),'candidate':states_of(candidate,True)}
            need(final_states==initial_states,'Any weight/state change is forbidden in zero-optimizer diagnostic')
            final_head_state={'heads.'+name:value.detach().cpu().clone() for name,value in candidate.heads.state_dict().items()}
            need(set(final_head_state)==set(head_state) and all(torch.equal(head_state[k],final_head_state[k])
                 for k in head_state),'All36 untrained head tensors must stay exact')
            stream=io.BytesIO();torch.save(final_head_state,stream);final_head_pin=write_bytes(out/'final_heads.pt',stream.getvalue())
            final_head_npz=save_head_npz(out/'final_heads.npz',final_head_state)
            final_rng=rng_snapshot();need(final_rng==run_rng_before,'Whole preflight changed caller RNG after construction')
            for name,binding in p['files'].items():
                current=packet.pin(root,root/name)
                need(current['bytes']==binding['bytes'] and current['sha256']==binding['sha256'],
                     'Static source/model/data changed during diagnostic: '+name)
            invariant_pin=write_json(out/'state_invariants.json',{'complete':True,'unchanged':True,
                'before':initial_states,'after':final_states,'initial_heads':initial_head_pin,'final_heads':final_head_pin,
                'initial_heads_NPZ':initial_head_npz,'final_heads_NPZ':final_head_npz,
                'factory_RNG':factory_rng,'run_RNG_before':run_rng_before,'run_RNG_after':final_rng,
                'gradient_determinism_scopes':gradient_warning_history,
                'all539_incumbent_tensors_frozen_and_exact':True,'all36_untrained_heads_exact':True,
                'all575_parameter_grad_fields_None':True,'static_sources_models_data_exact':True,
                'counters':dict(counters),'training_optimizer_updates':0,'model_qualified':False})
            event('all_integrity_parity_gradient_checks_passed',model_qualified=False,training_permitted=False)

        result={'complete':True,'diagnostic_pass':True,'status':'ZERO_OPTIMIZER_PREFLIGHT_PASSED_NOT_TRAINED',
            'protocol_sha256':expected_sha,'parity_cases':8,'gradient_witnesses':4,
            'parity_receipts':parity_receipts,'gradient_receipts':gradient_receipts,
            'state_invariants':invariant_pin,'counters':dict(counters),
            'elapsed_seconds':time.monotonic()-start,'last_resources':resource('before_result'),
            'gradient_determinism_scopes':gradient_warning_history,
            'strict_forward_parity_preserved':True,'CUDA_bilinear_gradient_determinism_claim':False,
            'model_qualified':False,'useful_restoration_proven':False,'automatic_training_authorized':False,
            'actual_optimizer_steps_or_weight_changes':0,'DEV_FINAL_native_evidence':False,
            'remaining':'Loss/LR/sampler/actual-step preservation and readable DEV/native usefulness remain separate manual-pilot requirements.'}
        result_pin=write_json(out/'result.json',result)
        print(json.dumps({'complete':True,'diagnostic_pass':True,'result':result_pin,
            'counters':counters,'training_optimizer_updates':0,'model_qualified':False}),flush=True)
        return result
    except BaseException as error:
        # A prior directory is never claimed, altered or used as failure output.
        # For an early packet/VM guard failure, a fresh owned directory contains
        # only failure evidence, with Torch/model calls still zero.
        if not owned and not out.exists() and not out.is_symlink():
            try:claim_output()
            except BaseException:pass
        if owned:
            failure={'complete':False,'UTC':utc(),'phase':failed_phase or phase,
                'restored_outer_phase':phase,'error_type':type(error).__name__,
                'error':str(error)[:4000],'traceback':traceback.format_exc()[-24000:],
                'protocol_sha256':expected_sha,'counters':dict(counters),
                'elapsed_seconds':time.monotonic()-start,'Torch_imported':'torch' in sys.modules,
                'model_construction_attempted':model_construction_attempted,
                'model_constructed':reference is not None or candidate is not None,
                'gradient_determinism_scope_summaries':[{'id':entry['id'],
                    'restored_exact':entry.get('restored_exact',False),
                    'warning_count':len(entry.get('warnings',[])),
                    'unexpected_nondeterminism_warning_count':len(entry.get('unexpected_nondeterminism_warnings',[]))}
                    for entry in gradient_warning_history],
                'full_gradient_warning_records':'events.jsonl and saved gradient/{id}/gradient.json when available',
                'training_optimizer_updates':0,'model_qualified':False,
                'preserve_partial_evidence_no_automatic_retry':True}
            try:
                failure['partial_output_bytes']=output_size()
                write_json(out/'failure.json',failure,emergency=True)
            except BaseException as save_error:
                print('Failure evidence write also failed: '+repr(save_error),flush=True)
        raise

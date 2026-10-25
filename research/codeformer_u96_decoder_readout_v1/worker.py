"""UNRUN manual-only existing-U96-decoder-RGB-readout B64 worker source.

Importing this module constructs no model and imports no Torch or NumPy.
Only run() may cross that dependency boundary, after exact packet/VM guards.
Only run() constructs the reserved Adam and commits at most64 updates.
No automatic retry or import-time runtime entrypoint exists.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import random
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


OUTPUT_RELATIVE = 'outputs/u96_decoder_readout_v1'
SITES = ('32','64','128','256')
CHANNELS = {'32':256,'64':256,'128':128,'256':128}
R2_FILE_SHA = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
U96_FILE_SHA = '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'
R2_STATE_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_STATE_SHA = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'


LEARNING = {'format': 'u96-existing-decoder-readout-clean-B64-v1', 'max_updates': 64, 'microbatches_per_update': 4, 'slot_roles': ['local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35'], 'sampler_offsets': [0, 4, 8, 12], 'snapshot_updates': [0, 1, 4, 16, 32, 64], 'warmup_updates': 4, 'gradient_clip_max_norm': 1.0, 'optimizer': {'type': 'Adam', 'lr': 5e-05, 'betas': [0.9, 0.99], 'eps': 1e-08, 'weight_decay': 0.0, 'amsgrad': False, 'foreach': False, 'fused': False, 'capturable': False, 'differentiable': False, 'maximize': False}, 'loss': {'image': 1.0, 'adjacent': 0.25, 'local_inside': 0.5, 'local_outside': 0.5}, 'local_half_open_roi': [64, 64, 192, 192], 'luminance_weights': [0.2126, 0.7152, 0.0722], 'illumination_block_size': 16, 'trainable_names': ['base.generator.blocks.24.weight', 'base.generator.blocks.24.bias'], 'trainable_parameters': 1731, 'frozen_tensors': 537, 'early_trend_update': 16}
LIMITS = {
    'full_forward_attempts':712,'full_forward_returns':712,
    'reference_forwards':120,'candidate_snapshot_forwards':336,
    'candidate_training_forwards':256,'backward_attempts':256,'backward_returns':256,
    'optimizer_constructions':1,'controlled_hook_exception_probes':1,
    'optimizer_update_attempts':64,'optimizer_updates':64,
    'internal_seconds':1800,'external_seconds':1860,'per_phase_seconds':180,
    'initial_free_disk_bytes':8*1024**3,'continuing_disk_reserve':4*1024**3,
    'maximum_RSS_bytes':10*1024**3,'maximum_CUDA_reserved_bytes':18*1024**3,
    'maximum_output_bytes':2*1024**3,
}

def validate_learning_contract(p):
    learning=p['learning'];fixed={k:v for k,v in learning.items() if k not in ('optimization_reference_order','monitor_reference_order')}
    need(fixed==LEARNING,'Exact prospective learning policy required')
    opt=learning['optimization_reference_order'];monitor=learning['monitor_reference_order']
    need(len(opt)==len(set(opt))==16 and len(monitor)==len(set(monitor))==8 and not set(opt)&set(monitor),'Disjoint16 optimization and8 TRAIN-monitor photos required')
    need(len(p['cases'])==96 and len({r['id'] for r in p['cases']})==96 and len(p['guards'])==24,'Exact96 newcases and24 legacy guards required')
    preflight={'tr_ffhq_00084','tr_ffhq_00178','tr_ffhq_00323','tr_ffhq_00616'}
    need(not (set(opt)|set(monitor))&preflight,'Preflight-only sources remain reserve/readout')
    need(not {r['reference_id'] for r in p['guards']}&(set(opt)|set(monitor)),'Legacy guard photos excluded from new cohorts')
    roles=set(LEARNING['slot_roles'])
    for cohort,refs,permit in (('optimize',opt,True),('monitor',monitor,False)):
        for ref in refs:
            rows=[r for r in p['cases'] if r['reference_id']==ref]
            need(len(rows)==4 and {r['role'] for r in rows}==roles,'Each photograph requires exactly4 roles')
            need(all(r['cohort']==cohort and r['split']=='TRAIN' and r['optimization_permitted'] is permit for r in rows),'New optimization admission/cohort differs')
    need(len({r['reference_id'] for r in p['cases']})==24,'Exactly24 new TRAIN photos')


def early_trend_gate(bank,baseline,update):
    """Additional feasibility rule only; unchanged legacy/terminal gates remain."""
    need(type(update) is int and update in (0,1,4,16,32,64),'Declared snapshot update required')
    report={'pass':True,'evaluated':update==16,'update':update,'groups':{},
            'no_threshold_relaxation':True,'TRAIN_only_not_qualification':True}
    if update!=16:return report
    for role,region in (('global_gaussian_1p2','whole'),('local_gaussian_1p2','central128')):
        ids=sorted(key for key,row in bank.items() if row['role']==role)
        need(len(ids)==8,'Eight adaptive TRAIN-monitor photographs per blur role')
        group={'pass':True,'views':{}}
        for view in ('raw','native_png'):
            initial=ordered_float_sum(baseline[key]['regional_metrics'][view][region]['adjacent_rgb_error'] for key in ids)/8
            current=ordered_float_sum(bank[key]['regional_metrics'][view][region]['adjacent_rgb_error'] for key in ids)/8
            need(math.isfinite(initial) and math.isfinite(current) and initial>0 and current>=0,
                 'Finite nonzero initial adjacent monitor means required')
            passed=current<initial
            group['views'][view]={'pass':passed,'initial_mean':initial,'current_mean':current,'region':region}
            group['pass']=group['pass'] and passed
        report['groups'][role]=group
    report['pass']=any(group['pass'] for group in report['groups'].values())
    return report


def schedule_indices(update):
    """Pure retained four-role cyclic sampler; all64 cases appear four times."""
    need(type(update) is int and 1<=update<=64,'Exactly one of64 finite updates required')
    return tuple((update-1+offset)%16 for offset in LEARNING['sampler_offsets'])


class QualityStop(ValueError):
    """A retained scientific stop; never permission to retry or add epochs."""


def need(condition,message):
    if not condition:raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def finite_json(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')


def byte_sha(value):return hashlib.sha256(value).hexdigest()

def ordered_float_sum(values):
    """Explicit scalar order matches sealed B1 CPython3.10, including on3.13."""
    total=0.0
    for value in values:total=total+float(value)
    return total


def linux_current_rss_bytes(statm_text,page_size):
    """Pure Linux statm resident-page parser; no optional dependency required."""
    need(type(statm_text) is str and type(page_size) is int and page_size>0,
         'Linux statm text and positive page size required')
    fields=statm_text.split()
    need(len(fields)>=2 and fields[1].isdigit(),'Linux statm resident pages field required')
    return int(fields[1])*page_size


def run(root,protocol,expected_sha):
    """Train only within a verified, fresh manual tmux L4 stage.

    Failure evidence is retained even when guards fail before Torch import.
    A preexisting output directory is immutable and is never taken over.
    """
    from . import packet

    root=Path(root).absolute();out=root/OUTPUT_RELATIVE
    need(not out.exists() and not out.is_symlink(),
         'Prior output directory is immutable; no resume/retry or overwrite')
    # Nothing has been claimed or created yet. Verify/require_vm must precede
    # started.json because require_vm rejects a prior started run.
    start=time.monotonic();owned=False;torch=None;np=None;p=None
    phase='packet_and_VM_guards';failed_phase=None;events=None;resources=None
    counters={'full_forward_attempts':0,'full_forward_returns':0,
              'reference_forwards':0,'candidate_snapshot_forwards':0,
              'candidate_training_forwards':0,'gradient_queries':0,
              'backward_attempts':0,'backward_returns':0,'optimizer_constructions':0,
              'optimizer_construction_returns':0,'optimizer_update_attempts':0,'optimizer_updates':0,
              'controlled_hook_exception_probes':0,
              'frozen_U96_adapter_site_reservations':0,'frozen_U96_adapter_site_returns':0,
              'existing_RGB_readout_reservations':0,'existing_RGB_readout_returns':0}
    limits=dict(LIMITS)
    reference=None;candidate=None;initial_states=None;final_states=None
    gpu_guard=None;factory_rng=None;run_rng_before=None
    model_construction_attempted=False
    resource_module=None;caller_rng_backup=None;initial_frozen_r2_sha=None
    backward_warning_history=[]
    optimizer=None;last_checkpoint=None;active_update=0;active_slot=None
    sampler_pin=None;readout_items=[]
    previous_termination_handler=None
    owned_bytes=0;owned_file_sizes={}

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

    def output_size(force_scan=False):
        """Own durable-write ledger, fully rechecked at every saved boundary."""
        if not force_scan:return owned_bytes
        total=0
        if not owned:return 0
        observed={}
        for item in out.rglob('*'):
            if item.is_dir():packet.directory(item)
            else:
                size=packet.regular(item).stat().st_size;total+=size
                observed[item.relative_to(out).as_posix()]=size
        need(observed==owned_file_sizes and total==owned_bytes,
             'Full owned-output namespace/byte readback differs from durable-write ledger')
        return total

    def write_bytes(path,payload,*,emergency=False):
        nonlocal owned_bytes
        path=safe_output(path)
        # Keep128KiB inside the declared cap for a durable failure record.
        ceiling=limits['maximum_output_bytes']-(0 if emergency else 131072)
        need(output_size()+len(payload)<=ceiling,'Bounded2GiB output cap exceeded')
        with path.open('xb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        owned_bytes+=len(payload);owned_file_sizes[path.relative_to(out).as_posix()]=len(payload)
        return packet.pin(root,path)

    def write_json(path,value,*,emergency=False):
        return write_bytes(path,finite_json(value),emergency=emergency)

    def append_json(path,value):
        nonlocal owned_bytes
        payload=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
        path=safe_output(path)
        need(path.exists(),'Append requires an exclusively created log')
        key=path.relative_to(out).as_posix()
        need(path.stat().st_size==owned_file_sizes[key],'Owned log changed outside durable writer')
        need(output_size()+len(payload)<=limits['maximum_output_bytes']-131072,
             'Output cap before durable log append')
        with path.open('ab') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        owned_bytes+=len(payload);owned_file_sizes[key]+=len(payload)

    def event(kind,**fields):
        if events is not None:append_json(events,{'UTC':utc(),'seconds':time.monotonic()-start,
            'phase':phase,'kind':kind,'counters':dict(counters),**fields})
        if owned:write_status(kind)

    def write_status(label):
        nonlocal owned_bytes
        path=safe_output(out/'status.json');temporary=safe_output(out/'status.pending.json')
        need(not temporary.exists(),'Preserve interrupted status write; no automatic retry')
        payload=finite_json({'complete':False,'phase':phase,'label':label,'UTC':utc(),
            'protocol_sha256':expected_sha,'counters':dict(counters),'active_update':active_update,
            'active_slot':active_slot,'last_committed_checkpoint':last_checkpoint,
            'optimizer_outcome_unknown':counters['optimizer_update_attempts']>counters['optimizer_updates'],
            'model_qualified':False,'manual_training_only':True})
        need(output_size()+len(payload)<=limits['maximum_output_bytes']-131072,'Status output cap')
        with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        key=path.relative_to(out).as_posix();owned_bytes+=len(payload)-owned_file_sizes.get(key,0)
        owned_file_sizes[key]=len(payload)

    def resource(label):
        elapsed=time.monotonic()-start
        need(elapsed<limits['internal_seconds'],'Internal1800s stop')
        free=__import__('shutil').disk_usage(root).free
        rss=linux_current_rss_bytes(Path('/proc/self/statm').read_text(),int(os.sysconf('SC_PAGE_SIZE')))
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
             'output_bytes':size,
             'output_measurement':'durable-write ledger; full namespace/bytes rechecked at every snapshot/checkpoint/final'}
        if resources is not None:append_json(resources,row)
        need(free>=limits['continuing_disk_reserve'],'Continuing4GiB disk reserve stop')
        need(rss<=limits['maximum_RSS_bytes'] and peak_rss<=limits['maximum_RSS_bytes'],
             'Current or Linux peak RSS10GiB stop')
        need(reserved<=limits['maximum_CUDA_reserved_bytes'] and
             peak_reserved<=limits['maximum_CUDA_reserved_bytes'],'CUDA reserved18GiB stop')
        need(size<=limits['maximum_output_bytes'],'Output2GiB stop')
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
        p=packet.verify_static(root,expected_sha)
        need(p==protocol,'Supplied protocol object must equal exact verified protocol')
        active_path=Path(__file__).resolve()
        need(active_path==(root/'research/codeformer_u96_decoder_readout_v1/worker.py').resolve(),
             'Active worker must come from this exact staged root')
        active_pin=packet.pin(root,active_path)
        expected_worker=p['files']['research/codeformer_u96_decoder_readout_v1/worker.py']
        need(active_pin['bytes']==expected_worker['bytes'] and active_pin['sha256']==expected_worker['sha256'],
             'Active loaded worker source differs from sealed source')
        facts=packet.require_vm(root)
        import resource as linux_resource
        resource_module=linux_resource
        need(p['limits']==packet.LIMITS==LIMITS,'Worker requires exact packet resource contract')
        validate_learning_contract(p)
        checked(p['inherited_DEV_evidence'])
        claim_output()
        write_json(out/'started.json',{'UTC':utc(),'protocol_sha256':expected_sha,
            'host':facts,'stage':p['stage_name'],'status':'MANUAL_LEARNING_STARTED_NOT_QUALIFIED',
            'training_optimizer_updates':0,'model_qualified':False})
        write_json(out/'inherited_DEV_failures.json',{'complete':True,
            'evidence':p['inherited_DEV_evidence'],'U96_inherited_DEV_failures_remain_binding':True,
            'not_repaired_by_this_TRAIN_screen':True,'model_qualified':False})
        events=out/'events.jsonl';resources=out/'resources.jsonl'
        write_bytes(events,b'');write_bytes(resources,b'')
        previous_termination_handler=signal.getsignal(signal.SIGTERM)
        def external_stop(signum,frame):
            raise TimeoutError('External SIGTERM stop; retain last committed state and operation status')
        signal.signal(signal.SIGTERM,external_stop)
        event('verified_before_Torch_import',Torch_imported=False)
        with bounded_phase('numerical_dependency_discoverability_before_Torch'):
            available={name:importlib.util.find_spec(name) is not None for name in ('torch','numpy','PIL')}
            event('numerical_dependency_discoverability',available=available,
                  modules_imported_by_this_discoverability_check=False)
            need(all(available.values()),'Required numerical/image dependency missing before model construction')
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
            from PIL import Image
            from torch.nn import functional as F
            torch=torch_module;np=numpy_module
            need(torch.cuda.is_available() and torch.cuda.device_count()==1,'One available CUDA GPU required')
            need(torch.cuda.get_device_name(0)=='NVIDIA L4','CUDA device name must equal NVIDIA L4 exactly')
            torch.cuda.set_device(0);device=torch.device('cuda',0)
            caller_rng_backup={'Python':random.getstate(),'NumPy':np.random.get_state(),
                'CPU':torch.get_rng_state().cpu().clone(),
                'CUDA':[value.cpu().clone() for value in torch.cuda.get_rng_state_all()]}
            random.seed(20261023);np.random.seed(20261023)
            torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
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

        def cpu_tree(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
            if isinstance(value,dict):return {key:cpu_tree(item) for key,item in value.items()}
            if isinstance(value,list):return [cpu_tree(item) for item in value]
            if isinstance(value,tuple):return tuple(cpu_tree(item) for item in value)
            need(value is None or type(value) in (str,int,float,bool),
                 'Primitive or tensor portable state fields only')
            return value

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

        def states_of(net,is_candidate):
            base=net.base.state_dict();adapters=adapter_state_of(net)
            need(len(base)==515 and len(adapters)==24 and len(net.state_dict())==539,
                 'Exact complete515+24/539 existing U96 tensor census required')
            need(canonical(adapters)==U96_STATE_SHA,'Frozen24 U96 adapter state changed')
            if is_candidate:
                invariant=net.validate_invariants(check_frozen_digests=True)
                fixed={name:value for name,value in base.items()
                       if 'base.'+name not in LEARNING['trainable_names']}
                need(len(fixed)==513 and canonical(fixed)==initial_frozen_r2_sha,
                     'All513 non-readout base tensors must remain exact')
                selected={name:value for name,value in net.named_parameters()
                          if name in LEARNING['trainable_names']}
                need(set(selected)==set(LEARNING['trainable_names']) and
                     sum(value.numel() for value in selected.values())==1731 and
                     all(value.requires_grad for value in selected.values()),
                     'Only two existing RGB readout tensors/1731 scalars may train')
                frozen=[value for value in net.parameters() if not value.requires_grad]
                need(len(frozen)==537 and all(value.grad is None for value in frozen),
                     'All537 frozen tensor gradient fields must stay None')
                readout_sha=canonical(selected)
            else:
                need(canonical(base)==R2_STATE_SHA,'Independent complete R2 reference changed')
                frozen=list(net.parameters());readout_sha=None
                need(len(frozen)==539 and all(not value.requires_grad and value.grad is None for value in frozen),
                     'All539 independent reference tensors stay frozen')
            need(all(not module.training for module in net.modules()),'Every module remains eval')
            result={'R2_initial_canonical515_sha256':R2_STATE_SHA,
                'R2_current_canonical515_sha256':canonical(base),'U96_canonical24_sha256':U96_STATE_SHA,
                'R2_except_readout_canonical513_sha256':initial_frozen_r2_sha,
                'frozen_tensors':537 if is_candidate else 539,
                'frozen_grad_fields_None':len(frozen),'full_state_sha256':canonical(net.state_dict()),
                'full_state_tensors':539,'readout_canonical_sha256':readout_sha,
                'trainable_tensors':2 if is_candidate else 0,'trainable_parameters':1731 if is_candidate else 0}
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

        def save_readout_npz(path,state):
            need(set(state)==set(LEARNING['trainable_names']),'Only two declared RGB readout arrays may enter NPZ')
            arrays={};schema=[]
            for name,value in sorted(state.items()):
                array=value.detach().cpu().contiguous().numpy().copy()
                need(array.dtype==np.float32 and tuple(array.shape)==prototype.expected_readout_shapes()[name] and np.isfinite(array).all(),
                     'Finite FP32 existing readout NPZ arrays required')
                arrays[name]=array
                schema.append({'name':name,'shape':list(array.shape),'dtype':str(array.dtype),
                               'bytes_sha256':byte_sha(array.tobytes())})
            stream=io.BytesIO();np.savez(stream,**arrays)
            file_pin=write_bytes(path,stream.getvalue())
            # Check the durable format immediately without pickle semantics.
            with np.load(path,allow_pickle=False) as readback:
                need(set(readback.files)==set(arrays) and all(np.array_equal(readback[name],value)
                     for name,value in arrays.items()),'Savedtwo-readout NPZ readback differs')
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
            raw=(prototype.connected_rgb256(native,torch=torch,manual_vm_root=root) if native.requires_grad else
                 F.interpolate(((native.float()+1)/2).clamp(0,1),(256,256),mode='bilinear',align_corners=False))
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
            initial_frozen_r2_sha=canonical({name:value for name,value in base_state.items()
                if 'base.'+name not in LEARNING['trainable_names']})
            # Imports are now permitted, but validate their exact module origins
            # before constructing either reference or candidate.
            from third_party.codeformer.codeformer_arch import CodeFormer
            from research.codeformer_u96_decoder_readout_v1 import model as prototype
            from research.codeformer_u96_decoder_readout_v1 import objective as clean_objective
            for relative,module in (
                ('research/codeformer_u96_decoder_readout_v1/model.py',prototype),
                ('research/codeformer_u96_decoder_readout_v1/objective.py',clean_objective)):
                active=Path(module.__file__).resolve()
                need(active==(root/relative).resolve(),'Active model/objective source origin differs')
                active_pin=packet.pin(root,active);sealed=p['files'][relative]
                need(active_pin['bytes']==sealed['bytes'] and active_pin['sha256']==sealed['sha256'],
                     'Active model/objective source bytes differ')
            need(Path(prototype.__file__).resolve()==
                 (root/'research/codeformer_u96_decoder_readout_v1/model.py').resolve(),
                 'Candidate factory must come from the exact staged prototype module')
            for name,(relative,source_sha) in prototype.ARCHITECTURE_SOURCES.items():
                loaded=sys.modules.get(name);need(loaded is not None and loaded.__file__,'Missing architecture module')
                path=Path(loaded.__file__).resolve();need(path==(root/relative).resolve() and
                    packet.sha256(path)==source_sha,'Exact staged architecture origin/hash required')
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
                r2_artifact=r2_path,u96_artifact=u96_path,manual_vm_root=root,
                prototype_source_sha256=p['files']['research/codeformer_u96_decoder_readout_v1/model.py']['sha256'],
                device='cuda:0',acknowledgement=prototype.MANUAL_ACKNOWLEDGEMENT).eval()
            after_factory=rng_snapshot();need(before_factory==after_factory,'Prototype factory changed CPU/CUDA RNG')
            factory_rng={'before':before_factory,'after':after_factory,'exact':True,
                         'reference_factory_CPU_CUDA_preserved':True}
            initial_states={'reference':states_of(reference,False),'candidate':states_of(candidate,True)}
            prototype_check=candidate.validate_invariants(check_frozen_digests=True)
            readout_state={name:value.detach().cpu().clone() for name,value in candidate.named_parameters()
                        if name in LEARNING['trainable_names']}
            need(set(readout_state)==set(prototype.expected_readout_shapes()),'Save only existing RGB readout')
            stream=io.BytesIO();torch.save(readout_state,stream)
            initial_readout_pin=write_bytes(out/'initial_readout.pt',stream.getvalue())
            initial_readout_npz=save_readout_npz(out/'initial_readout.npz',readout_state)
            run_rng_before=rng_snapshot()
            write_json(out/'preflight.json',{'complete':True,'status':'PRE_FORWARD_INTEGRITY_ONLY',
                'protocol_sha256':expected_sha,'host':facts,'gpu_guard':gpu_guard,
                'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,'CUDA':torch.version.cuda,
                'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'seed':20261023,
                'initial_states':initial_states,'prototype_invariants':prototype_check,
                'factory_RNG':factory_rng,'initial_readouts':initial_readout_pin,
                'initial_readouts_NPZ':initial_readout_npz,
                'limits':limits,'display_processing':'none','optimizer_constructions':0,
                'optimizer_updates':0,'model_qualified':False,'actual_training_permitted':True,'manual_launch_only':True})

        def input_values(row):
            need(re.fullmatch(r'[A-Za-z0-9_\-]{1,120}',row['id']) is not None,'Safe unique witness ID required')
            arrays=[]
            for field in ('input','target'):
                if row.get(field) is None:
                    need(field=='target' and row.get('role')=='dark','Only legacy dark guard targets may be null')
                    arrays.append(None);continue
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
            need(kind in ('reference_forwards','candidate_snapshot_forwards','candidate_training_forwards'),'Unknown call reservation')
            need(counters['full_forward_attempts']<limits['full_forward_attempts'] and counters[kind]<limits[kind],
                 'Finite full-forward reservation exhausted')
            resource('before_reserved_forward');counters['full_forward_attempts']+=1;counters[kind]+=1
            counters['frozen_U96_adapter_site_reservations']+=4
            if kind!='reference_forwards':counters['existing_RGB_readout_reservations']+=1
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
            if kind!='reference_forwards':
                need(net.last_readout_calls==1,'Exactly one existing final RGB readout required')
                counters['existing_RGB_readout_returns']+=1
            event('forward_returned',forward_kind=kind,sites=list(net.last_forward_sites),
                  module_site_counts_are_not_extra_full_forwards=True)
            resource('after_reserved_forward');return result

        def load_saved_raw(binding):
            with checked(binding).open('rb') as stream:value=np.load(stream,allow_pickle=False)
            need(value.shape==(256,256,3) and value.dtype==np.float32 and np.isfinite(value).all(),
                 'Finite saved HWC256 RAW required')
            return value

        def png_rgb(binding):
            with Image.open(checked(binding)) as image:
                need(image.mode=='RGB' and image.size==(256,256),'RGB256 output PNG required')
                return np.asarray(image).copy()

        def target_rgb(row):
            if row.get('target') is None:return None
            value=png_rgb(row['target'])
            need(byte_sha(value.tobytes())==row['target_RGB8_sha256'],'Target RGB hash differs')
            return value.astype(np.float32)/np.float32(255)

        def region_statistics(value,target,region):
            residual=value.astype(np.float64)-target.astype(np.float64)
            h=region[:,:-1]&region[:,1:];v=region[:-1,:]&region[1:,:]
            # Preserve the exact historical whole-image mean-of-two-means
            # arithmetic used by the old24 guard bank.
            adjacent=(float(.5*(np.abs(residual[:,1:]-residual[:,:-1]).mean()+
                np.abs(residual[1:]-residual[:-1]).mean())) if region.all() else
                float((np.abs(np.diff(residual,axis=1))[h].sum()+
                    np.abs(np.diff(residual,axis=0))[v].sum())/(int(h.sum()+v.sum())*3)))
            return {'mae':float(np.abs(residual[region]).mean()),
                'adjacent_rgb_error':adjacent}

        def regions():
            whole=np.ones((256,256),dtype=np.bool_);inside=np.zeros_like(whole)
            inside[64:192,64:192]=True
            return {'whole':whole,'central128':inside,'outside':~inside}

        def illumination(value):
            y=value.astype(np.float64)@np.asarray(LEARNING['luminance_weights'],dtype=np.float64)
            return y.reshape(16,16,16,16).mean(axis=(1,3))

        def saved_record(row,triple,stem,kind,baseline_record=None):
            raw,unclipped=deliver(triple[0])
            pins={'raw':save_delivery_array(stem/'raw.npy',raw),'png':save_png(stem/'output.png',raw)}
            if kind!='training' or active_update==1:
                pins['unclipped']=save_delivery_array(stem/'unclipped.npy',unclipped)
            stock={label:tensor_record(value) for label,value in zip(('native512','logits','LQ'),triple)}
            if row['id'] in first_slot_ids and (kind=='reference' or kind=='training' and active_update==1):
                pins['stock']={label:save_array(stem/(label+'.npy'),value)
                    for label,value in zip(('native512','logits','LQ'),triple)}
            a=raw.detach().cpu().numpy()[0].transpose(1,2,0).copy()
            q=png_rgb(pins['png']).astype(np.float32)/np.float32(255);target=target_rgb(row)
            statistics={view:({name:region_statistics(x,target,r) for name,r in regions().items()}
                             if target is not None else None)
                        for view,x in (('raw',a),('native_png',q))}
            record={'complete':True,'id':row['id'],'reference_id':row['reference_id'],'role':row['role'],
                'cohort':row.get('cohort','legacy_guard'),'split':'TRAIN','kind':kind,
                'update':active_update,'slot':active_slot,'input':row['input'],'target':row.get('target'),
                'outputs':pins,'stock':stock,'regional_metrics':statistics,
                'clipping':{'native_below_minus1':int((triple[0].detach() < -1).sum().item()),
                    'native_above1':int((triple[0].detach() > 1).sum().item()),
                    'native_values':int(triple[0].numel()),'unclipped256':tensor_record(unclipped)},
                'counters':dict(counters),'model_qualified':False}
            if row.get('behavior_teacher') is not None:
                teacher=row['behavior_teacher'];traw=load_saved_raw(teacher['raw'])
                tpng=png_rgb(teacher['png']).astype(np.float32)/np.float32(255)
                record['behavior_distance']={'raw':float(np.abs(a.astype(np.float64)-traw.astype(np.float64)).mean()),
                    'native_png':float(np.abs(q.astype(np.float64)-tpng.astype(np.float64)).mean())}
            if baseline_record is not None:
                braw=load_saved_raw(baseline_record['outputs']['raw'])
                bpng=png_rgb(baseline_record['outputs']['png']).astype(np.float32)/np.float32(255)
                record['U96_behavior_distance']={view:float(np.abs(x.astype(np.float64)-b.astype(np.float64)).mean())
                    for view,x,b in (('raw',a,braw),('native_png',q,bpng))}
                record['U96_illumination_distance']={view:float(np.abs(illumination(x)-illumination(b)).mean())
                    for view,x,b in (('raw',a,braw),('native_png',q,bpng))}
                need(stock['logits']==baseline_record['stock']['logits'] and stock['LQ']==baseline_record['stock']['LQ'],
                     'Frozen upstream logits/LQ changed')
            record['metrics']=({view:stats['whole'] for view,stats in statistics.items()} if target is not None else
                {view:{'mae':None,'adjacent_rgb_error':None} for view in ('raw','native_png')})
            record['receipt']=write_json(stem/'record.json',record)
            return record,raw,unclipped

        all_rows=p['cases']+p['guards'];by_id={row['id']:row for row in all_rows}
        need(len(by_id)==120,'All120 reference inputs must have unique IDs')
        opt_order=p['learning']['optimization_reference_order'];monitor_order=p['learning']['monitor_reference_order']
        def slot_rows(update):
            result=[]
            for role,index in zip(LEARNING['slot_roles'],schedule_indices(update)):
                ref=opt_order[index]
                row=next(r for r in p['cases'] if r['reference_id']==ref and r['role']==role)
                need(row['cohort']=='optimize' and row['optimization_permitted'] is True,'Only newly admitted optimization cases')
                result.append(row)
            need(len({r['reference_id'] for r in result})==4,'Four distinct photographs per update')
            return result
        sampler=[{'update':u,'slots':[r['id'] for r in slot_rows(u)]} for u in range(1,65)]
        sampler_pin=write_json(out/'sampler.json',{'complete':True,'updates':sampler,'role_order':LEARNING['slot_roles'],
            'reference_order':opt_order,'exposures':256,'full_64_rolecase_passes':4,'dynamic_augmentation':False})
        first_slot_ids={r['id'] for r in slot_rows(1)}
        with bounded_phase('controlled_owned_hook_exception_without_CodeFormer_forward'):
            need(counters['controlled_hook_exception_probes']<limits['controlled_hook_exception_probes'],
                 'Exactly one explicit non-neural-base lifecycle probe')
            before_probe_rng=rng_snapshot();before_probe_state=states_of(candidate,True)
            sample,target=input_values(slot_rows(1)[0])
            had_forward='forward' in candidate.base.__dict__
            prior_instance_forward=candidate.base.__dict__.get('forward')
            prior_bound_forward=candidate.base.forward;stub_calls=[]
            class ControlledHookProbe(RuntimeError):pass
            def raising_stub(*args,**kwargs):
                stub_calls.append(1)
                raise ControlledHookProbe('Controlled stub before original CodeFormer.forward')
            counters['controlled_hook_exception_probes']+=1
            event('controlled_hook_probe_reserved',original_CodeFormer_forward_calls=0)
            caught_expected=False
            try:
                candidate.base.forward=raising_stub
                with torch.enable_grad():candidate(sample)
            except ControlledHookProbe:
                caught_expected=True
            finally:
                if had_forward:candidate.base.forward=prior_instance_forward
                else:delattr(candidate.base,'forward')
            restored=candidate.base.forward
            need(caught_expected and len(stub_calls)==1 and
                 getattr(restored,'__self__',None) is getattr(prior_bound_forward,'__self__',None) and
                 getattr(restored,'__func__',None) is getattr(prior_bound_forward,'__func__',None),
                 'Controlled stub exception and exact original callable restoration required')
            assert_clean(candidate)
            need(states_of(candidate,True)==before_probe_state and rng_snapshot()==before_probe_rng,
                 'Controlled exception changed model state or caller RNG')
            write_json(out/'controlled_hook_exception.json',{'complete':True,
                'candidate_wrapper_attempts':1,'base_stub_calls':1,'original_CodeFormer_forward_calls':0,
                'owned_hooks_removed':True,'exact_callable_restored':True,'all539_states_and_RNG_exact':True,
                'separate_from_712_real_CodeFormer_forwards':True,'counters':dict(counters)})
            event('controlled_hook_probe_returned',original_CodeFormer_forward_calls=0)
            del sample,target
        baseline={}
        for row in all_rows:
            with bounded_phase('reference_'+row['id']):
                sample,target=input_values(row)
                with torch.no_grad():
                    triple=full_call(reference,sample,'reference_forwards')
                    record,raw,unclipped=saved_record(row,triple,out/'reference'/row['id'],'reference')
                baseline[row['id']]=record
                del sample,target,triple,record,raw,unclipped
            if len(baseline)%16==0:print(json.dumps({'U96_reference_cases':len(baseline),'maximum':120}),flush=True)
        write_json(out/'reference_bank.json',{'complete':True,'cases':list(baseline.values()),
            'full_U96_initializer':True,'model_qualified':False})
        snapshot_receipts=[]

        def quality_need(ok,message):
            if not ok:raise QualityStop(message)

        def legacy_gate(bank):
            initial={r['id']:r for r in p['guards']};report={'pass':True,'thresholds_reset_to_U96':False,'groups':{}}
            for role in ('clear','dark','filter'):
                ids=sorted(k for k,r in bank.items() if r['role']==role);need(len(ids)==8,'Eight historical same-role guards')
                group={'pass':True,'views':{}}
                for view in ('raw','native_png'):
                    group['views'][view]={}
                    for field in (('mae','adjacent_rgb_error') if role=='filter' else ('behavior_distance',)):
                        def metric(rows,k):
                            return float(rows[k]['behavior_distance'][view] if field=='behavior_distance' else rows[k]['metrics'][view][field])
                        start={k:metric(initial,k) for k in ids};now={k:metric(bank,k) for k in ids}
                        need(all(math.isfinite(v) and v>=0 for v in (*start.values(),*now.values())),'Finite historical metrics')
                        if role=='filter':
                            values={k:now[k]-start[k] for k in ids};case_limits=dict.fromkeys(ids,.03);mean_limit=.01
                        else:
                            values=now;case_limits={k:max(.03,start[k]) for k in ids};mean_limit=max(.01,ordered_float_sum(start.values())/8)
                        mean=ordered_float_sum(values.values())/8;bad=[k for k in ids if values[k]>case_limits[k]]
                        passed=not bad and mean<=mean_limit
                        group['views'][view][field]={'pass':passed,'initial':start,'current':now,
                            'measurements':values,'case_limits':case_limits,'mean_limit':mean_limit,'mean':mean,'failed_ids':bad}
                        group['pass']=group['pass'] and passed
                report['groups'][role]=group;report['pass']=report['pass'] and group['pass']
            return report

        def monitor_gate(bank,update):
            report={'pass':True,'TRAIN_monitor_only_not_DEV':True,'endpoint_evaluated':update==64,'groups':{}}
            for role in LEARNING['slot_roles']:
                ids=sorted(k for k,r in bank.items() if r['role']==role);need(len(ids)==8,'Eight monitor photos per role')
                group={'pass':True,'views':{}}
                for view in ('raw','native_png'):
                    checks={}
                    for region in (('whole','central128','outside') if role=='local_gaussian_1p2' else ('whole',)):
                        for field in ('mae','adjacent_rgb_error'):
                            start={k:baseline[k]['regional_metrics'][view][region][field] for k in ids}
                            now={k:bank[k]['regional_metrics'][view][region][field] for k in ids}
                            delta={k:now[k]-start[k] for k in ids};mean=ordered_float_sum(delta.values())/8
                            checks[region+'/'+field]={'pass':max(delta.values())<=.03 and mean<=.01,
                                'initial':start,'current':now,'deltas':delta,'mean':mean,'mean_limit':.01,'worst_limit':.03}
                    if role in ('clear','lowlight_linear_0p35'):
                        field='U96_behavior_distance' if role=='clear' else 'U96_illumination_distance'
                        distances={k:bank[k][field][view] for k in ids}
                        checks[field]={'pass':ordered_float_sum(distances.values())/8<=.01 and max(distances.values())<=.03,
                            'values':distances,'mean_limit':.01,'worst_limit':.03}
                    if update==64 and role in ('local_gaussian_1p2','global_gaussian_1p2'):
                        region='central128' if role=='local_gaussian_1p2' else 'whole'
                        start=ordered_float_sum(baseline[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids)/8
                        now=ordered_float_sum(bank[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids)/8
                        need(start>0,'Nonzero initial monitored blur adjacent error required')
                        gain=(start-now)/start
                        checks['new_1_percent_TRAIN_blur_endpoint']={'pass':gain>=.01,'gain_fraction':gain,
                            'initial_mean':start,'current_mean':now,'region':region,
                            'not_historical_DGP_feature_gate_or_DEV_qualification':True}
                    if update==64 and role=='lowlight_linear_0p35':
                        for field in ('mae','adjacent_rgb_error'):
                            start=ordered_float_sum(baseline[k]['regional_metrics'][view]['whole'][field] for k in ids)/8
                            now=ordered_float_sum(bank[k]['regional_metrics'][view]['whole'][field] for k in ids)/8
                            checks['endpoint_dark_'+field]={'pass':now<=start,'initial_mean':start,'current_mean':now}
                    group['views'][view]=checks;group['pass']=group['pass'] and all(c['pass'] for c in checks.values())
                report['groups'][role]=group;report['pass']=report['pass'] and group['pass']
            return report

        def snapshot(update):
            nonlocal failed_phase
            bank={}
            selected=[r for r in p['cases'] if r['cohort']=='monitor']+p['guards']
            need(len(selected)==56,'Exactly56 monitoring and historical guard readouts')
            for row in selected:
                with bounded_phase('snapshot_'+str(update)+'_'+row['id']):
                    sample,target=input_values(row)
                    with torch.enable_grad():
                        triple=full_call(candidate,sample,'candidate_snapshot_forwards')
                        record,raw,unclipped=saved_record(row,triple,out/'snapshots'/f'u{update:04d}'/row['id'],
                            'snapshot',baseline[row['id']])
                    if update==0:
                        need(record['stock']==baseline[row['id']]['stock'],
                             'Initial native512/logits/LQ exact same-device parity failed')
                        need(np.array_equal(load_saved_raw(record['outputs']['raw']),load_saved_raw(baseline[row['id']]['outputs']['raw'])) and
                             checked(record['outputs']['png']).read_bytes()==checked(baseline[row['id']]['outputs']['png']).read_bytes(),
                             'Initial same-device complete-U96 RAW/PNG parity failed')
                    bank[row['id']]=record
                    del sample,target,triple,record,raw,unclipped
            legacy=legacy_gate({r['id']:bank[r['id']] for r in p['guards']})
            monitor=monitor_gate({r['id']:bank[r['id']] for r in p['cases'] if r['cohort']=='monitor'},update)
            trend=early_trend_gate(bank,baseline,update)
            receipt={'complete':True,'update':update,'legacy':legacy,'monitor':monitor,'early_trend':trend,'cases':list(bank.values()),
                     'counters':dict(counters),'model_qualified':False}
            snapshot_receipts.append(write_json(out/'snapshots'/f'u{update:04d}'/'bank_and_gates.json',receipt))
            output_size(force_scan=True)
            print(json.dumps({'snapshot':update,'legacy_pass':legacy['pass'],'TRAIN_monitor_pass':monitor['pass'],
                'optimizer_updates':counters['optimizer_updates'],'model_qualified':False}),flush=True)
            if not (legacy['pass'] and monitor['pass']):
                failed_phase='snapshot_u'+str(update)+'_quality_gates'
            quality_need(legacy['pass'] and monitor['pass'] and trend['pass'],
                'Unchanged legacy/monitor gate or additional update16 trend stop')
            return receipt

        def objective_for(row,raw,target):
            need(row['split']=='TRAIN' and row['cohort']=='optimize' and
                 row['optimization_permitted'] is True and target is not None,
                 'Only admitted paired photographic clean TRAIN targets enter objective')
            return clean_objective.connected_clean_objective(raw,target,row['role'],torch=torch)

        @contextmanager
        def backward_warn_only_scope(update,slot,id_):
            before={'enabled':torch.are_deterministic_algorithms_enabled(),
                    'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}
            need(before=={'enabled':True,'warn_only':False},'Strict mode before backward required')
            record={'update':update,'slot':slot,'id':id_,'before':before,'during':{'enabled':True,'warn_only':True},
                    'scope':'one reserved backward plus CUDA synchronize; all operations inside this scope',
                    'repeatable_CUDA_gradient_claim':False}
            backward_warning_history.append(record)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    torch.use_deterministic_algorithms(True,warn_only=True);yield record
                finally:
                    torch.use_deterministic_algorithms(before['enabled'],warn_only=before['warn_only'])
                    record['after']={'enabled':torch.are_deterministic_algorithms_enabled(),
                        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}
                    record['warnings']=[{'category':w.category.__name__,'message':str(w.message)[:12000],
                        'filename':str(w.filename),'lineno':int(w.lineno)} for w in caught]
                    record['unexpected_nondeterminism_warnings']=[w for w in record['warnings']
                        if ('deterministic' in w['message'].lower() or 'nondeterminism' in w['message'].lower()) and
                        re.match(r'^upsample_bilinear2d_backward_out_cuda\b',w['message']) is None]
                    record['restored_exact']=record['after']==before
                    event('backward_determinism_scope',**record)
                    need(record['restored_exact'],'Strict deterministic flags must restore after backward')

        readout_items=sorted((name,value) for name,value in candidate.named_parameters() if value.requires_grad)
        need([name for name,_ in readout_items]==sorted(LEARNING['trainable_names']) and
             sum(value.numel() for _,value in readout_items)==1731,'Exactly two existing RGB tensors/1731 scalars')

        def checkpoint(update):
            nonlocal last_checkpoint
            folder=out/'checkpoints'/f'u{update:04d}'
            readout={name:value.detach().cpu().clone() for name,value in readout_items}
            readout_pin=save_readout_npz(folder/'readout.npz',readout)
            numpy_state=np.random.get_state()
            resume={'format':'complete-U96-existing-RGB-readout-clean-resume-v1','protocol_sha256':expected_sha,
                'update':update,'pending':False,'next_update':update+1 if update<64 else None,
                'finite_terminal_update':64,'sampler':sampler_pin,
                'sampler_committed_exposures':update*4,'sampler_role_order':LEARNING['slot_roles'],
                'sampler_reference_order':opt_order,'readout_state':readout,'readout_canonical_sha256':canonical(readout),
                'optimizer':cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
                'optimizer_parameter_names':[name for name,_ in readout_items],
                'CPU_RNG':torch.get_rng_state().cpu().clone(),
                'CUDA_RNG':[value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
                'Python_random_state':random.getstate(),
                'NumPy_random_state':{'algorithm':numpy_state[0],
                    'keys':torch.from_numpy(numpy_state[1].copy()),'position':int(numpy_state[2]),
                    'has_gauss':int(numpy_state[3]),'cached_gaussian':float(numpy_state[4])},
                'R2_canonical_sha256':R2_STATE_SHA,'U96_adapter_canonical_sha256':U96_STATE_SHA,
                'all537_frozen_exact':True,'learning':p['learning'],'counters':dict(counters),
                'no_automatic_resume_or_cross_device_bitwise_claim':True,'model_qualified':False}
            stream=io.BytesIO();torch.save(resume,stream);resume_pin=write_bytes(folder/'resume.pt',stream.getvalue())
            receipt={'complete':True,'update':update,'resume':resume_pin,'readout':readout_pin,
                'readout_canonical_sha256':canonical(readout),'counters':dict(counters),'model_qualified':False}
            last_checkpoint=write_json(folder/'checkpoint.json',receipt);write_status('checkpoint_committed')
            output_size(force_scan=True)
            return receipt

        snapshot(0)
        with bounded_phase('fresh_Adam_construction'):
            need(counters['optimizer_constructions']<limits['optimizer_constructions'],'Single fresh Adam only')
            counters['optimizer_constructions']+=1;event('optimizer_construction_reserved')
            options=dict(LEARNING['optimizer']);options.pop('type');options['betas']=tuple(options['betas'])
            optimizer=torch.optim.Adam([value for _,value in readout_items],**options)
            need(not optimizer.state,'New objective requires fresh empty Adam state')
            counters['optimizer_construction_returns']+=1;event('optimizer_construction_returned')
            checkpoint(0)

        for update in range(1,65):
            active_update=update;active_slot=None
            with bounded_phase('optimizer_update_'+str(update)):
                optimizer.zero_grad(set_to_none=True)
                lr=LEARNING['optimizer']['lr']*min(1.,update/LEARNING['warmup_updates'])
                for group in optimizer.param_groups:group['lr']=lr
                micro_receipts=[]
                for slot,row in enumerate(slot_rows(update),1):
                    active_slot=slot;resource('before_training_microbatch')
                    sample,target=input_values(row)
                    with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False):
                        triple=full_call(candidate,sample,'candidate_training_forwards')
                        stem=out/'training'/f'u{update:04d}'/f'slot{slot}'
                        record,raw,unclipped=saved_record(row,triple,stem,'training',baseline[row['id']])
                        if update==1:
                            need(record['stock']==baseline[row['id']]['stock'],
                                 'First reserved graphs must match complete initial stock tuple')
                            need(np.array_equal(load_saved_raw(record['outputs']['raw']),load_saved_raw(baseline[row['id']]['outputs']['raw'])) and
                                checked(record['outputs']['png']).read_bytes()==checked(baseline[row['id']]['outputs']['png']).read_bytes(),
                                'All4 first grad-enabled slots must equal initial U96 RAW/PNG before first commit')
                        objective,terms=objective_for(row,raw,target);divided=objective/4.
                        need(objective.requires_grad and torch.isfinite(objective).item(),'Finite connected role objective required')
                        scalar={'complete':True,'update':update,'slot':slot,'id':row['id'],'role':row['role'],
                            'LR':lr,'objective':float(objective.detach().item()),'divided_objective':float(divided.detach().item()),
                            'terms':{k:float(v.detach().item()) for k,v in terms.items()},'output_record':record['receipt'],
                            'readout_state_before_commit_sha256':canonical({n:v for n,v in readout_items}),
                            'comparison_reference_not_optimization_target':baseline[row['id']]['receipt'],
                            'photographic_optimization_target':row['target'],
                            'every_reported_loss_term_connected':all(value.requires_grad for value in terms.values()),
                            'model_qualified':False}
                        need(scalar['every_reported_loss_term_connected'],
                             'Every RGB/adjacent role term must retain its ordinary graph')
                        micro_receipts.append(write_json(stem/'objective.json',scalar))
                        need(counters['backward_attempts']<limits['backward_attempts'],'Finite256 backward reservation cap')
                        counters['backward_attempts']+=1;event('backward_reserved',id=row['id'],update=update,slot=slot)
                        with backward_warn_only_scope(update,slot,row['id']) as warning_record:
                            divided.backward();torch.cuda.synchronize(0)
                        counters['backward_returns']+=1;event('backward_returned',id=row['id'],update=update,slot=slot)
                    need(not warning_record['unexpected_nondeterminism_warnings'],'Unexpected nondeterministic backward operator; retain stop')
                    need(all(value.grad is None for value in candidate.parameters() if not value.requires_grad),
                         'All537 frozen gradient fields must stay None')
                    del sample,target,triple,record,raw,unclipped,objective,divided,terms
                active_slot=None
                gradients=[value.grad for _,value in readout_items]
                need(all(g is not None and g.dtype==torch.float32 and torch.isfinite(g).all().item() for g in gradients),
                     'Both joint-loss accumulated readout gradients finite required')
                need(any(torch.count_nonzero(g).item()>0 for g in gradients),'All-zero optimizer update refused')
                before_readout={name:value.detach().cpu().clone() for name,value in readout_items}
                preclip=torch.nn.utils.clip_grad_norm_([value for _,value in readout_items],max_norm=1.,error_if_nonfinite=True)
                postclip={name:value.grad.detach().cpu().clone() for name,value in readout_items}
                gradient_pin=save_readout_npz(out/'updates'/f'u{update:04d}'/'postclip_gradients.npz',postclip)
                need(counters['optimizer_update_attempts']<limits['optimizer_update_attempts'],'Finite64 commit reservation cap')
                counters['optimizer_update_attempts']+=1;event('optimizer_reserved',update=update)
                optimizer.step();torch.cuda.synchronize(0)
                counters['optimizer_updates']+=1;event('optimizer_returned',update=update)
                need(all(torch.isfinite(value).all().item() for _,value in readout_items),'Nonfinite readout after optimizer')
                optimizer.zero_grad(set_to_none=True)
                candidate.validate_invariants(check_frozen_digests=True)
                states_of(reference,False)
                need(all(value.grad is None for value in candidate.parameters()),'All539 grad fields None after commit clearing')
                current={name:value.detach().cpu().clone() for name,value in readout_items}
                need(any(not torch.equal(before_readout[name],current[name]) for name in current),'Adam commit changed no existing readout tensor')
                receipt={'complete':True,'update':update,'LR':lr,'microbatch_receipts':micro_receipts,
                    'postclip_gradients':gradient_pin,'preclip_norm':float(preclip.detach().item()),
                    'clip_max_norm':1.,'clip_coefficient':min(1.,1./(float(preclip.detach().item())+1e-6)),
                    'readout_before_sha256':canonical(before_readout),'readout_after_sha256':canonical(current),
                    'all537_frozen_exact':True,'all539_grad_fields_None':True,'counters':dict(counters)}
                write_json(out/'updates'/f'u{update:04d}'/'commit.json',receipt)
                checkpoint(update);event('update_fully_saved',update=update)
                print(json.dumps({'B64_update':update,'maximum':64,'LR':lr,'optimizer_updates':counters['optimizer_updates']}),flush=True)
            if update in LEARNING['snapshot_updates']:snapshot(update)

        active_slot=None
        with bounded_phase('final_integrity_and_result'):
            expected={'full_forward_attempts':712,'full_forward_returns':712,'reference_forwards':120,
                'candidate_snapshot_forwards':336,'candidate_training_forwards':256,'gradient_queries':0,
                'backward_attempts':256,'backward_returns':256,'optimizer_constructions':1,
                'optimizer_construction_returns':1,'optimizer_update_attempts':64,'optimizer_updates':64,
                'controlled_hook_exception_probes':1,
                'frozen_U96_adapter_site_reservations':2848,'frozen_U96_adapter_site_returns':2848,
                'existing_RGB_readout_reservations':592,'existing_RGB_readout_returns':592}
            need(counters==expected,'Exact finite successful call/commit ledger required')
            final_states={'reference':states_of(reference,False),'candidate':states_of(candidate,True)}
            need(final_states['reference']==initial_states['reference'],'Reference U96 changed')
            need(rng_snapshot()==run_rng_before,'Deterministic fixed sampler/forward training changed caller RNG')
            for name,binding in p['files'].items():
                current=packet.pin(root,root/name)
                need(current['bytes']==binding['bytes'] and current['sha256']==binding['sha256'],'Static source/model/data changed: '+name)
            invariant_pin=write_json(out/'state_invariants.json',{'complete':True,'before':initial_states,'after':final_states,
                'all537_nonreadout_tensors_frozen_exact':True,'only_two_existing_readout_tensors_trainable':True,
                'initial_readouts':initial_readout_pin,'initial_readouts_NPZ':initial_readout_npz,'last_checkpoint':last_checkpoint,
                'caller_RNG_before':run_rng_before,'caller_RNG_after':rng_snapshot(),'counters':dict(counters),
                'CUDA_backward_bitwise_repeatability_not_claimed':True,'model_qualified':False})
            output_size(force_scan=True)
            result={'complete':True,'status':'FINITE_B64_LEARNING_SCREEN_PASSED_NOT_QUALIFIED',
                'protocol_sha256':expected_sha,'counters':dict(counters),'sampler':sampler_pin,
                'snapshot_receipts':snapshot_receipts,'last_checkpoint':last_checkpoint,'state_invariants':invariant_pin,
                'elapsed_seconds':time.monotonic()-start,'last_resources':resource('before_result'),
                'optimizer_updates':64,'reference_photos':{'optimization':16,'TRAIN_monitor':8,'legacy_guards':8},
                'one_percent_endpoint_is_new_TRAIN_monitor_not_historical_DGP_or_DEV':True,
                'full_DEV_native_FINAL_user_app_qualification_pending':True,'model_qualified':False,
                'automatic_training_or_resume_authorized':False}
            result_pin=write_json(out/'result.json',result);write_status('finite_B64_screen_complete')
            print(json.dumps({'complete':True,'result':result_pin,'optimizer_updates':64,'model_qualified':False}),flush=True)
            return result
    except BaseException as error:
        if not owned and not out.exists() and not out.is_symlink():
            try:claim_output()
            except BaseException:pass
        if owned:
            cleanup={}
            for name,net in (('reference',reference),('candidate',candidate)):
                if net is not None:
                    try:assert_clean(net);cleanup[name]='clean'
                    except BaseException as cleanup_error:cleanup[name]=repr(cleanup_error)
            failure={'complete':False,'status':'QUALITY_STOP' if isinstance(error,QualityStop) else 'INFRASTRUCTURE_OR_INTEGRITY_STOP',
                'UTC':utc(),'phase':failed_phase or phase,'restored_outer_phase':phase,
                'error_type':type(error).__name__,'error':str(error)[:4000],'traceback':traceback.format_exc()[-18000:],
                'protocol_sha256':expected_sha,'counters':dict(counters),'active_update':active_update,'active_slot':active_slot,
                'last_committed_checkpoint':last_checkpoint,'elapsed_seconds':time.monotonic()-start,
                'Torch_imported':'torch' in sys.modules,'model_construction_attempted':model_construction_attempted,
                'model_constructed':reference is not None or candidate is not None,'hook_cleanup':cleanup,
                'optimizer_outcome_unknown':counters['optimizer_update_attempts']>counters['optimizer_updates'],
                'backward_scopes_completed_or_partial':len(backward_warning_history),
                'last_backward_scope_summaries':[{'update':e['update'],'slot':e['slot'],'id':e['id'],
                    'restored_exact':e.get('restored_exact',False),'warning_count':len(e.get('warnings',[]))}
                    for e in backward_warning_history[-16:]],'model_qualified':False,
                'preserve_all_partial_state_no_automatic_retry':True}
            if candidate is not None and torch is not None:
                try:
                    named=dict(candidate.named_parameters())
                    selected={name:named[name] for name in LEARNING['trainable_names']}
                    numpy_state=np.random.get_state()
                    partial={'format':'stopped-existing-RGB-readout-state-v1',
                        'protocol_sha256':expected_sha,'pending':True,
                        'active_update':active_update,'active_slot':active_slot,
                        'readout_state':cpu_tree(selected),
                        'partial_gradients':cpu_tree({name:value.grad for name,value in selected.items()}),
                        'optimizer':cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
                        'optimizer_parameter_names':sorted(selected),'counters':dict(counters),
                        'sampler':sampler_pin,'sampler_completed_backwards':counters['backward_returns'],
                        'CPU_RNG':torch.get_rng_state().cpu().clone(),
                        'CUDA_RNG':[value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
                        'Python_random_state':random.getstate(),
                        'NumPy_random_state':{'algorithm':numpy_state[0],
                            'keys':torch.from_numpy(numpy_state[1].copy()),'position':int(numpy_state[2]),
                            'has_gauss':int(numpy_state[3]),'cached_gaussian':float(numpy_state[4])},
                        'optimizer_outcome_unknown':failure['optimizer_outcome_unknown'],
                        'nonfinite_values_permitted_only_as_failed_evidence':True,
                        'automatic_resume_or_retry_permitted':False,'model_qualified':False}
                    stream=io.BytesIO();torch.save(partial,stream)
                    failure['retained_partial_state']=write_bytes(out/'failed_state.pt',stream.getvalue(),emergency=True)
                except BaseException as partial_error:
                    failure['partial_state_capture_error']=repr(partial_error)
            try:write_status('failed')
            except BaseException as status_error:
                failure['status_write_error']=repr(status_error)
            try:write_json(out/'failure.json',failure,emergency=True)
            except BaseException as save_error:print('Failure evidence write also failed: '+repr(save_error),flush=True)
        if isinstance(error,QualityStop):raise SystemExit(2) from error
        raise
    finally:
        exception_in_flight=sys.exc_info()[0] is not None
        if previous_termination_handler is not None:
            signal.signal(signal.SIGTERM,previous_termination_handler)
        if caller_rng_backup is not None and torch is not None and np is not None:
            try:
                random.setstate(caller_rng_backup['Python'])
                np.random.set_state(caller_rng_backup['NumPy'])
                torch.set_rng_state(caller_rng_backup['CPU'])
                torch.cuda.set_rng_state_all(caller_rng_backup['CUDA'])
                restored_cuda=torch.cuda.get_rng_state_all();restored_numpy=np.random.get_state()
                restored=(torch.equal(torch.get_rng_state(),caller_rng_backup['CPU']) and
                    len(restored_cuda)==len(caller_rng_backup['CUDA']) and
                    all(torch.equal(actual,expected) for actual,expected in
                        zip(restored_cuda,caller_rng_backup['CUDA'])) and
                    random.getstate()==caller_rng_backup['Python'] and
                    restored_numpy[0]==caller_rng_backup['NumPy'][0] and
                    np.array_equal(restored_numpy[1],caller_rng_backup['NumPy'][1]) and
                    restored_numpy[2:]==caller_rng_backup['NumPy'][2:])
                need(restored,'Outer caller RNG restoration failed')
                if owned:write_json(out/'caller_RNG_restored.json',{'complete':True,
                    'CPU_CUDA_Python_NumPy_restored':True,'new_neural_or_gradient_calls':0,
                    'training_checkpoint_RNG_retained_separately':True},emergency=True)
            except BaseException as restore_error:
                print('Caller RNG restoration or its receipt failed: '+repr(restore_error),flush=True)
                if owned:
                    try:write_json(out/'caller_RNG_restoration_failure.json',{'complete':False,
                        'error':repr(restore_error),'model_qualified':False},emergency=True)
                    except BaseException:pass
                if not exception_in_flight:
                    raise RuntimeError('Caller RNG restoration/receipt failed; retain stopped evidence') from restore_error


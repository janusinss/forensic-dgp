"""Independent bounded saved-state audit of the manual U96 decoder readout return.

Only saved arrays and primitive records are read. Torch is used solely with
weights_only=True/map_location=cpu for bounded detached checkpoints; no model,
optimizer, forward, backward, step, training, VM or CUDA operation is created.
The transport/strict parser patterns derive from the retained postfusion LR
saved-state audit, with an independent two-tensor schema and recurrence checks.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_SHA='a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
ARCHIVE_SHA='73c8f4bcca5814cb845058cbc7d553ad3ea1f29e97967449cad66ac848fed2c4'
STAGE='cctv_codeformer_u96_decoder_readout_v1_vm'
OUTPUT='outputs/u96_decoder_readout_v1'
MAX_ARCHIVE=2*1024**3
MAX_PAYLOAD=2*1024**3
MAX_FILE=8*1024**2
MAX_RETAINED=96*1024**2
MAX_FILES=6000
MAX_SECONDS=240
MAX_EXPANDED=MAX_PAYLOAD+16*1024**2
R2_FILE_SHA='d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
R2_FILE_BYTES=376609465
R2_STATE='a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_STATE='fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
ROLES=['local_gaussian_1p2','global_gaussian_1p2','clear','lowlight_linear_0p35']
SNAPSHOTS=[0,1,4,16,32,64]
# These are arithmetic-only CPU/CUDA FP32 operation-order envelopes, never
# image-quality/preservation tolerances. FP32 moment lerp/addcmul may use FMA;
# the parameter allowance is below one FP32 epsilon at unit magnitude. Each
# saved parameter is checked against both saved-moment and fully recurrent
# NumPy predictions, so a bad moment cannot be hidden by a matching weight.
TOLERANCES={'first_moment':{'atol':2e-9,'rtol':2e-5},
            'second_moment':{'atol':2e-12,'rtol':2e-5},
            'parameter_update':{'atol':6e-8,'rtol':3e-6},
            'fully_recurrent_parameter':{'atol':8e-7,'rtol':3e-5}}
CHECKS=0


def need(ok,message):
    global CHECKS
    CHECKS+=1
    if not ok:raise ValueError(message)


def shape_schema():
    return {'base.generator.blocks.24.bias':(3,),
            'base.generator.blocks.24.weight':(3,64,3,3)}


def sha(b):return hashlib.sha256(b).hexdigest()


def strict_json(b):
    def pairs(items):
        d={}
        for k,v in items:
            need(k not in d,'Duplicate JSON key');d[k]=v
        return d
    def constant(v):raise ValueError('Nonfinite JSON '+v)
    value=json.loads(b.decode('utf-8'),object_pairs_hook=pairs,parse_constant=constant)
    def finite(v):
        if isinstance(v,float):need(math.isfinite(v),'Nonfinite exponent-overflow JSON')
        elif isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,list):
            for x in v:finite(x)
    finite(value);return value


def regular(path,maximum=MAX_ARCHIVE):
    path=Path(path).absolute()
    for parent in path.parents:
        info=parent.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info,'st_file_attributes',0)&1024,'Ordinary input parent')
    info=path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_nlink==1
         and not getattr(info,'st_file_attributes',0)&1024 and info.st_size<=maximum,'Bounded ordinary input file')
    return path


def canonical(state):
    h=hashlib.sha256()
    for name,a in sorted(state.items()):
        header=json.dumps([name,'torch.float32',list(a.shape)],separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'big'));h.update(header);h.update(a.tobytes(order='C'))
    return h.hexdigest()


def zip_guard(raw,kind):
    need(type(raw)is bytes and 0<len(raw)<4*1024**2,'Bounded saved ZIP bytes')
    with zipfile.ZipFile(io.BytesIO(raw))as z:
        info=z.infolist();names=z.namelist()
        need(len(names)==len(set(names)) and 0<len(names)<=300,'Unique bounded ZIP members')
        need(sum(x.file_size for x in info)<=4*1024**2,'Bounded expanded ZIP')
        for x in info:
            p=PurePosixPath(x.filename)
            need(not x.flag_bits&1 and not p.is_absolute() and '..'not in p.parts
                 and '\\'not in x.filename and ':'not in x.filename,'Safe unencrypted ZIP names')
            mode=x.external_attr>>16
            need(not stat.S_ISLNK(mode),'No ZIP symlink')
        if kind=='npz':
            need(len(info)==2 and all('/'not in n and n.endswith('.npy')for n in names)
                 and set(n[:-4]for n in names)==set(shape_schema()),'Exact2 readout NPZ keys')
            need(sum(x.file_size for x in info)<1024**2,'Dense bounded NPZ')
        elif kind=='pt':
            prefixes={PurePosixPath(n).parts[0]for n in names}
            need(len(prefixes)==1 and any(n.endswith('/data.pkl')for n in names),'One Torch ZIP root/data.pkl')
        else:raise ValueError('Unknown ZIP kind')


def read_npz(raw,np):
    zip_guard(raw,'npz')
    # Check declared shapes/dtypes before np.load may allocate their arrays.
    with zipfile.ZipFile(io.BytesIO(raw))as archive:
        for member in archive.infolist():
            with archive.open(member)as stream:
                version=np.lib.format.read_magic(stream)
                need(version in((1,0),(2,0)),'Supported bounded NPY version')
                reader=(np.lib.format.read_array_header_1_0 if version==(1,0)
                        else np.lib.format.read_array_header_2_0)
                shape,fortran,dtype=reader(stream,max_header_size=4096)
                name=member.filename[:-4]
                need(shape==shape_schema()[name]and dtype==np.dtype('float32')
                     and not dtype.hasobject and not fortran,'Exact NPY declared schema before allocation')
                need(member.file_size==stream.tell()+math.prod(shape)*4,'Exact NPY payload length')
    with np.load(io.BytesIO(raw),allow_pickle=False)as z:
        result={}
        for name,shape in shape_schema().items():
            a=z[name]
            need(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite exact FP32 head/gradient '+name)
            result[name]=np.ascontiguousarray(a).copy()
        return result


def torch_cpu_decode(raw,torch):
    zip_guard(raw,'pt')
    value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    def walk(x):
        if isinstance(x,torch.Tensor):need(x.device.type=='cpu'and not x.requires_grad,'Detached CPU checkpoint tensor')
        elif isinstance(x,dict):
            for k,v in x.items():need(type(k)in(str,int),'Primitive dict key');walk(v)
        elif isinstance(x,(tuple,list)):
            for v in x:walk(v)
        else:need(x is None or type(x)in(str,int,float,bool),'Primitive/Tensor checkpoint tree')
    walk(value);return value


def finite_array(t,shape,np,torch):
    need(isinstance(t,torch.Tensor)and t.device.type=='cpu' and tuple(t.shape)==shape and t.dtype==torch.float32 and not t.requires_grad,'Exact detached CPU FP32 tensor required before allocation')
    a=t.detach().contiguous().numpy()
    need(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite saved moment/head schema')
    return a


class HashReader:
    def __init__(self,raw):self.raw=raw;self.sha=hashlib.sha256();self.bytes=0
    def read(self,n=-1):
        b=self.raw.read(n);self.bytes+=len(b);self.sha.update(b)
        need(self.bytes<=MAX_ARCHIVE,'Archive byte cap');return b
    def readable(self):return True


class ExpandedReader:
    """Cap actual inflated bytes, including tar padding and trailing gzip data."""
    def __init__(self,raw):self.raw=raw;self.bytes=0;self.tail=b''
    def read(self,n=-1):
        # All callers are bounded; refusing read-all avoids a hidden bomb.
        need(0<=n<=1024**2,'Bounded expanded read request')
        b=self.raw.read(n);self.bytes+=len(b);self.tail=(self.tail+b)[-32768:]
        need(self.bytes<=MAX_EXPANDED,'Actual expanded gzip cap');return b
    def readable(self):return True


def read_archive(path,expected_sha,stage,output):
    members={};jsons={};blobs={};invalid={};total=retained=0;began=time.monotonic()
    before=regular(path).stat()
    with path.open('rb')as f:
        raw=HashReader(f)
        with gzip.GzipFile(fileobj=raw,mode='rb')as gz:
            expanded=ExpandedReader(gz)
            with tarfile.open(fileobj=expanded,mode='r|')as tar:
                folded=set()
                for member in tar:
                    need(time.monotonic()-began<MAX_SECONDS,'Bounded archive read time')
                    parts=PurePosixPath(member.name).parts
                    need(member.type in(tarfile.REGTYPE,tarfile.AREGTYPE)and member.sparse is None
                         and len(parts)>1 and parts[0]==stage and
                         not PurePosixPath(member.name).is_absolute()and'..'not in parts
                         and member.name==PurePosixPath(member.name).as_posix()
                         and '\\'not in member.name and ':'not in member.name,'Regular canonical scoped TAR member')
                    rel=PurePosixPath(*parts[1:]).as_posix()
                    need(rel not in members and rel.casefold()not in folded,'Duplicate/case-colliding member')
                    folded.add(rel.casefold());total+=member.size
                    need(0<=member.size<=MAX_FILE and total<=MAX_PAYLOAD and len(members)<MAX_FILES,'Bounded TAR payload')
                    keep=(rel.endswith('.json')or re.fullmatch(output+r'/checkpoints/u\d{4}/(?:readout\.npz|resume\.pt)',rel)is not None
                          or re.fullmatch(output+r'/updates/u\d{4}/postclip_gradients\.npz',rel)is not None or rel in (output+'/initial_readout.npz',output+'/initial_readout.pt',output+'/failed_state.pt',output+'/events.jsonl'))
                    stream=tar.extractfile(member);need(stream is not None,'Regular TAR file stream')
                    h=hashlib.sha256();parts_b=[];n=0
                    for b in iter(lambda:stream.read(1024**2),b''):
                        n+=len(b);h.update(b)
                        if keep:parts_b.append(b)
                    need(n==member.size,'Exact member size');members[rel]={'bytes':n,'sha256':h.hexdigest()}
                    if keep:
                        b=b''.join(parts_b);retained+=len(b);need(retained<=MAX_RETAINED,'Bounded retained-state bytes')
                        if rel.endswith('.json'):
                            try:jsons[rel]=strict_json(b)
                            except(ValueError,UnicodeError,TypeError)as error:
                                if not rel.startswith(output+'/'):raise
                                invalid[rel]=str(error)[:300]
                        else:blobs[rel]=b
                archive_end=tar.offset
            # The stream reader may have buffered tar padding. Validate its
            # retained tail from the final file boundary, then every later byte.
            need(0<=expanded.bytes-archive_end<=len(expanded.tail),'Bounded buffered tar suffix')
            suffix=expanded.tail[len(expanded.tail)-(expanded.bytes-archive_end):]if expanded.bytes>archive_end else b''
            need(not any(suffix),'Nonzero data after final TAR member')
            while True:
                b=expanded.read(1024**2)
                if not b:break
                need(not any(b),'Nonzero trailing expanded TAR data')
                need(time.monotonic()-began<MAX_SECONDS,'Bounded gzip drain')
        while raw.read(1024**2):pass
    after=regular(path).stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Archive changed while reading')
    need(raw.sha.hexdigest()==expected_sha and raw.bytes==before.st_size,'Exact required archive SHA256')
    return members,jsons,blobs,invalid,{'bytes':raw.bytes,'sha256':raw.sha.hexdigest(),
        'payload_files':len(members),'payload_bytes':total,'retained_bytes':retained,
        'expanded_bytes':expanded.bytes,'gzip_footer_CRC_drained':True,'zero_TAR_suffix_verified':True}


def pin_check(pin,members):
    need(type(pin)is dict and set(pin)=={'path','bytes','sha256'},'Exact direct pin')
    need(pin['path']in members and members[pin['path']]=={'bytes':pin['bytes'],'sha256':pin['sha256']},'Payload pin mismatch '+str(pin.get('path')))


def integer(value,low,high,label):
    need(type(value)is int and low<=value<=high,label);return value


def outcome(status,pending,failure,result,supervisor):
    """Classify recorded outcome without inventing a completed update."""
    if status is None:
        count=attempts=None;unknown=True
    else:
        c=status['counters'];count=integer(c['optimizer_updates'],0,64,'Confirmed commits')
        attempts=integer(c['optimizer_update_attempts'],count,64,'Optimizer attempts')
        need(type(status.get('optimizer_outcome_unknown'))is bool,'Unknown-outcome bool')
        unknown=status['optimizer_outcome_unknown']or pending or attempts>count
    need(not(failure is not None and result is not None),'Failure/result conflict')
    if result is not None:
        need(count==attempts==64 and not unknown and result.get('complete')is True
             and result.get('model_qualified')is False,'Result cannot manufacture terminal success')
        label='FINITE_SCREEN_PASSED_NOT_QUALIFIED'
    elif unknown:label='INTERRUPTED_OR_UNRECORDED_OUTCOME'
    elif failure is not None:
        need(failure.get('status')in('QUALITY_STOP','INFRASTRUCTURE_OR_INTEGRITY_STOP'),'Known failure status')
        label=failure['status']
    elif supervisor is not None and supervisor.get('timeout')is True:label='EXTERNALLY_STOPPED_WITH_LAST_DURABLE_COUNTS'
    else:label='INCOMPLETE_RETURN_NO_RESULT'
    return {'scientific_status':label,'confirmed_optimizer_updates':count,
        'optimizer_attempts':attempts,'optimizer_outcome_unknown':unknown,
        'interrupted_status_write_present':pending,'result_present':result is not None,
        'failure_present':failure is not None,'model_qualified':False}


def state_array_pin(binding,state,members):
    pin_check(binding['file'],members)
    need(binding['canonical_sha256']==canonical(state),'Portable canonical mismatch')
    rows=binding['arrays'];need(type(rows)is list and len(rows)==2,'2 portable schema rows')
    need(len({r['name']for r in rows})==2 and {r['name']for r in rows}==set(state),'Unique portable row names')
    for row in rows:
        a=state[row['name']]
        need(row['shape']==list(a.shape)and row['dtype']=='float32'
             and row['bytes_sha256']==sha(a.tobytes()),'Portable schema/content hash')


def check_sampler(p,sampler):
    cases={r['id']:r for r in p['cases']}
    need(len(cases)==96,'Exact96 admitted cases')
    order=p['learning']['optimization_reference_order']
    need(len(order)==len(set(order))==16,'Frozen16 optimizer photo order')
    coverage={r['id']:0 for r in p['cases']if r['cohort']=='optimize'}
    need(len(coverage)==64 and sampler['exposures']==256 and len(sampler['updates'])==64,'Finite64-case/256-exposure sampler')
    for u,item in enumerate(sampler['updates'],1):
        expected=[]
        for role,offset in zip(ROLES,[0,4,8,12]):
            rows=[r for r in cases.values()if r['reference_id']==order[(u-1+offset)%16]and r['role']==role]
            need(len(rows)==1,'One rolecase per optimizer photograph');expected.append(rows[0]['id'])
        need(item=={'update':u,'slots':expected},'Frozen sampler update')
        need(len({cases[x]['reference_id']for x in expected})==4,'Distinct photographs per update')
        for id_ in expected:
            need(cases[id_]['cohort']=='optimize'and cases[id_]['optimization_permitted']is True
                 and cases[id_]['split']=='TRAIN','Explicit TRAIN optimization admission')
            coverage[id_]+=1
    need(set(coverage.values())=={4},'Four exposures per optimizer rolecase')
    return cases,coverage



def expected_counts(update, *, after_snapshot=False):
    """Reconstruct fixed call accounting from ordering, not VM counters."""
    need(type(update)is int and 0<=update<=64,'Finite counted update')
    snapshots=1+sum(0<s<update for s in SNAPSHOTS)
    if after_snapshot and update>0 and update in SNAPSHOTS:snapshots+=1
    candidate=56*snapshots+4*update
    full=120+candidate
    return {'full_forward_attempts':full,'full_forward_returns':full,
        'reference_forwards':120,'candidate_snapshot_forwards':56*snapshots,
        'candidate_training_forwards':4*update,'gradient_queries':0,
        'backward_attempts':4*update,'backward_returns':4*update,
        'optimizer_constructions':1,'optimizer_construction_returns':1,
        'optimizer_update_attempts':update,'optimizer_updates':update,
        'controlled_hook_exception_probes':1,
        'frozen_U96_adapter_site_reservations':4*full,
        'frozen_U96_adapter_site_returns':4*full,
        'existing_RGB_readout_reservations':candidate,
        'existing_RGB_readout_returns':candidate}


def validate_rng(resume,np,torch):
    cpu=resume['CPU_RNG'];cuda=resume['CUDA_RNG'];nr=resume['NumPy_random_state'];pr=resume['Python_random_state']
    need(isinstance(cpu,torch.Tensor)and cpu.dtype==torch.uint8 and tuple(cpu.shape)==(5056,),
         'Exact saved5056-byte CPU RNG')
    need(type(cuda)is list and len(cuda)==1 and isinstance(cuda[0],torch.Tensor)
         and cuda[0].dtype==torch.uint8 and tuple(cuda[0].shape)==(16,),'Exact single16-byte CUDA RNG decoded on CPU')
    need(type(nr)is dict and set(nr)=={'algorithm','keys','position','has_gauss','cached_gaussian'},'NumPy RNG field census')
    need(nr['algorithm']=='MT19937'and isinstance(nr['keys'],torch.Tensor)and nr['keys'].dtype==torch.uint32
         and tuple(nr['keys'].shape)==(624,)and type(nr['position'])is int and 0<=nr['position']<=624
         and type(nr['has_gauss'])is int and nr['has_gauss']in(0,1)
         and type(nr['cached_gaussian'])is float and math.isfinite(nr['cached_gaussian']),'Portable NumPy RNG')
    need(type(pr)is tuple and len(pr)==3 and pr[0]==3 and type(pr[1])is tuple and len(pr[1])==625
         and all(type(x)is int and 0<=x<=2**32-1 for x in pr[1][:-1]) and 0<=pr[1][-1]<=624
         and(pr[2]is None or type(pr[2])is float and math.isfinite(pr[2])),'Portable Python RNG')
    return {'CPU':sha(cpu.numpy().tobytes()),'CPU_bytes':cpu.numel(),
        'CUDA':[sha(cuda[0].numpy().tobytes())],'CUDA_bytes':[cuda[0].numel()],
        'NumPy_keys':sha(nr['keys'].numpy().tobytes()),'NumPy_position':nr['position'],
        'NumPy_has_gauss':nr['has_gauss'],'NumPy_cached_gaussian':nr['cached_gaussian'],
        'Python':sha(repr(pr).encode())}


def validate_optimizer(opt,u,names,np,torch):
    need(type(opt)is dict and set(opt)=={'state','param_groups'}and len(opt['param_groups'])==1,'One saved Adam group')
    g=opt['param_groups'][0];lr=5e-5*min(1.,u/4) if u else 5e-5
    policy={'betas':(.9,.99),'eps':1e-8,'weight_decay':0.,'amsgrad':False,
        'foreach':False,'fused':False,'capturable':False,'differentiable':False,'maximize':False}
    need(g['params']==list(range(2))and g['lr']==lr,'Exact two-parameter order and warmup LR')
    for name,value in policy.items():
        actual=tuple(g[name]) if name=='betas' else g[name]
        need(actual==value and (type(actual)is bool if type(value)is bool else True),'Saved Adam policy '+name)
    need(set(opt['state'])==(set() if u==0 else {0,1}),'Fresh at0, two states thereafter')
    states={}
    for i,name in enumerate(names):
        if u==0:
            states[name]={'m':np.zeros(shape_schema()[name],np.float32),'v':np.zeros(shape_schema()[name],np.float32)}
            continue
        row=opt['state'][i]
        need(set(row)=={'step','exp_avg','exp_avg_sq'},'Only expected Adam moment fields')
        step=row['step'];need(isinstance(step,torch.Tensor)and tuple(step.shape)==()
             and step.dtype==torch.float32 and float(step.item())==u,'Exact saved CPU scalar Adam step')
        m=finite_array(row['exp_avg'],shape_schema()[name],np,torch)
        v=finite_array(row['exp_avg_sq'],shape_schema()[name],np,torch)
        need((v>=0).all(),'Nonnegative second moment')
        states[name]={'m':m.copy(),'v':v.copy()}
    return states,lr


def validate_resume(resume,u,readout,p,sampler_pin,names,np,torch):
    need(type(resume)is dict and resume['format']=='complete-U96-existing-RGB-readout-clean-resume-v1'
         and resume['protocol_sha256']==PROTOCOL_SHA,'Exact resume format/protocol')
    need(resume['update']==u and resume['pending']is False and resume['next_update']==(u+1 if u<64 else None)
         and resume['finite_terminal_update']==64,'Finite checkpoint cursor')
    need(resume['sampler']==sampler_pin and resume['sampler_committed_exposures']==u*4
         and resume['sampler_role_order']==ROLES
         and resume['sampler_reference_order']==p['learning']['optimization_reference_order'],'Exact sampler/cursor')
    need(resume['learning']==p['learning'] and resume['all537_frozen_exact']is True
         and resume['R2_canonical_sha256']==R2_STATE and resume['U96_adapter_canonical_sha256']==U96_STATE,
         'Resume declares537 frozen lineage; values are not returned')
    need(resume['optimizer_parameter_names']==names and set(resume['readout_state'])==set(names),'Exact two readout names')
    for name,t in resume['readout_state'].items():
        need(np.array_equal(finite_array(t,shape_schema()[name],np,torch),readout[name]),'PT/NPZ exact readout '+name)
    need(resume['readout_canonical_sha256']==canonical(readout),'Resume readout canonical')
    need(resume['model_qualified']is False and resume['no_automatic_resume_or_cross_device_bitwise_claim']is True,
         'No automatic resume/qualification claim')
    rng=validate_rng(resume,np,torch)
    moments,lr=validate_optimizer(resume['optimizer'],u,names,np,torch)
    return moments,rng,lr


def adam_predict(parameter,gradient,previous_m,previous_v,update,lr,np,*,actual_m=None,actual_v=None):
    """Saved-array arithmetic only; not an optimizer constructor or operation."""
    m=(previous_m+(gradient-previous_m)*np.float32(.1)).astype(np.float32)
    v=(previous_v*np.float32(.99)+(gradient*gradient)*np.float32(.01)).astype(np.float32)
    cm=m if actual_m is None else actual_m;cv=v if actual_v is None else actual_v
    denom=(np.sqrt(cv,dtype=np.float32)/np.float32(math.sqrt(1-.99**update))+np.float32(1e-8)).astype(np.float32)
    parameter=(parameter-np.float32(lr/(1-.9**update))*(cm/denom)).astype(np.float32)
    return m,v,parameter


def comparison(actual,predicted,np,kind):
    need(actual.shape==predicted.shape and np.isfinite(actual).all() and np.isfinite(predicted).all(),
         'Finite matching arithmetic comparison')
    error=np.abs(actual.astype(np.float64)-predicted.astype(np.float64));tol=TOLERANCES[kind]
    envelope=tol['atol']+tol['rtol']*np.abs(actual.astype(np.float64))
    return float(error.max()),int(np.count_nonzero(error>envelope))


def tensor_tree_equal(a,b,torch):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if type(a)is dict:return type(b)is dict and set(a)==set(b)and all(tensor_tree_equal(a[k],b[k],torch)for k in a)
    if type(a)in(list,tuple):return type(a)is type(b)and len(a)==len(b)and all(tensor_tree_equal(x,y,torch)for x,y in zip(a,b))
    return type(a)is type(b)and a==b


def initial_from_retained_r2(path,np,torch):
    """Pinned trusted checkpoint decode only; constructs no CodeFormer model."""
    path=regular(path,R2_FILE_BYTES);before=path.stat();h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    need(before.st_size==R2_FILE_BYTES and h.hexdigest()==R2_FILE_SHA,'Exact retained original R2 artifact before decode')
    with zipfile.ZipFile(path)as z:
        rows=z.infolist();names=[x.filename for x in rows]
        need(0<len(rows)<1000 and len(set(names))==len(rows)
             and sum(x.file_size for x in rows)<=384*1024**2,'Bounded515-tensor R2 checkpoint ZIP')
        for x in rows:
            q=PurePosixPath(x.filename)
            need(not q.is_absolute() and '..'not in q.parts and ':'not in x.filename and '\\'not in x.filename
                 and not x.flag_bits&1 and not stat.S_ISLNK(x.external_attr>>16),'Ordinary R2 ZIP entry')
        need(len({PurePosixPath(n).parts[0]for n in names})==1,'One R2 ZIP root')
        metadata=[x for x in rows if x.filename.endswith('/data.pkl')]
        need(len(metadata)==1 and metadata[0].file_size<256*1024,'Bounded R2 metadata')
    value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    need(type(value)is dict and 'params_ema'in value and type(value['params_ema'])is dict,'Retained R2 tensor mapping')
    base=value['params_ema'];need(len(base)==515,'Exact515 R2 tensor count')
    state={}
    for name,t in base.items():
        need(type(name)is str and isinstance(t,torch.Tensor)and t.dtype==torch.float32
             and t.device.type=='cpu'and not t.requires_grad and t.is_contiguous(),'Detached dense FP32 R2 tensor')
        a=t.numpy();need(np.isfinite(a).all(),'Finite retained R2 tensor');state[name]=a
    need(canonical(state)==R2_STATE,'Exact retained515 canonical initializer')
    selected={name:state[name.removeprefix('base.')].copy()for name in shape_schema()}
    need(all(a.shape==shape_schema()[name]for name,a in selected.items()),'Exact original RGB readout dimensions')
    fixed={name:a for name,a in state.items()if 'base.'+name not in selected}
    fixed_sha=canonical(fixed)
    after=regular(path,R2_FILE_BYTES).stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
         'Retained R2 unchanged during decode')
    return selected,{'path':str(path),'bytes':R2_FILE_BYTES,'sha256':R2_FILE_SHA,
        'canonical515_sha256':R2_STATE,'canonical513_except_readout_sha256':fixed_sha,
        'readout_canonical_sha256':canonical(selected),'Torch_decode_only_no_model':True}


def scalar_objective(role,terms):
    expected={'image_inside','image_outside','adjacent_inside','adjacent_outside'}if role==ROLES[0]else{'image','adjacent'}
    need(set(terms)==expected and all(type(v)is float and math.isfinite(v)and v>=0 for v in terms.values()),
         'Exact finite all-clean target objective terms')
    return (.5*(terms['image_inside']+terms['image_outside'])+.125*(terms['adjacent_inside']+terms['adjacent_outside'])
            if role==ROLES[0]else terms['image']+.25*terms['adjacent'])


def audit(archive,archive_sha,export_path,r2_path):
    global CHECKS
    CHECKS=0;began=time.monotonic()
    need(archive_sha==ARCHIVE_SHA,'This audit binds one exact actual returned archive')
    members,jsons,blobs,invalid,transport=read_archive(archive,archive_sha,STAGE,OUTPUT)
    need(not invalid,'No malformed returned runtime JSON')
    need(members['protocol.json']['sha256']==PROTOCOL_SHA,'Exact sealed protocol')
    p=jsons['protocol.json'];need(p['stage_name']==STAGE,'Exact decoder-readout namespace')
    learning=p['learning']
    need(learning['format']=='u96-existing-decoder-readout-clean-B64-v1' and learning['max_updates']==64
         and learning['microbatches_per_update']==4 and learning['slot_roles']==ROLES
         and learning['sampler_offsets']==[0,4,8,12]and learning['snapshot_updates']==SNAPSHOTS
         and learning['trainable_names']==['base.generator.blocks.24.weight','base.generator.blocks.24.bias']
         and learning['trainable_parameters']==1731 and learning['frozen_tensors']==537,'Sealed two-tensor finite contract')
    need(learning['optimizer']=={'type':'Adam','lr':5e-5,'betas':[.9,.99],'eps':1e-8,'weight_decay':0.,
        'amsgrad':False,'foreach':False,'fused':False,'capturable':False,'differentiable':False,'maximize':False}
        and learning['warmup_updates']==4 and learning['gradient_clip_max_norm']==1.,'Fixed fresh Adam contract')
    need(learning['loss']=={'image':1.,'adjacent':.25,'local_inside':.5,'local_outside':.5}
         and learning['early_trend_update']==16,'Fixed all-clean target role objective')
    static=0;omitted=[]
    for name,pin in p['files'].items():
        if name.startswith('weights/'):
            need(name not in members,'Original checkpoint omitted from return');omitted.append(name);continue
        need(name in members and members[name]==pin,'Exact sealed static binding '+name);static+=1
    need(set(omitted)=={p['model']['r2']['path'],p['model']['u96']['path']},'Only two complete-U96 initializer files omitted')
    export=strict_json(regular(export_path,1024**2).read_bytes())
    need(export['complete']is True and export['archive_sha256']==archive_sha and export['bytes']==transport['bytes']
         and export['payload_files']==transport['payload_files']and export['payload_bytes']==transport['payload_bytes']
         and export['protocol_sha256']==PROTOCOL_SHA,'Sidecar binds actual payload')
    status=jsons.get(OUTPUT+'/status.json');pending=OUTPUT+'/status.pending.json'in members
    failure=jsons.get(OUTPUT+'/failure.json');result=jsons.get(OUTPUT+'/result.json');supervisor=jsons.get('supervisor.json')
    actual=outcome(status,pending,failure,result,supervisor)
    need(actual['scientific_status']=='QUALITY_STOP' and actual['confirmed_optimizer_updates']==64
         and actual['optimizer_attempts']==64 and actual['optimizer_outcome_unknown']is False,
         'Actual completed64-commit quality stop retained as failure')
    need(failure['error_type']=='QualityStop' and failure['phase']=='snapshot_u64_quality_gates'
         and failure['active_update']==64 and failure['active_slot']is None and result is None,
         'Retained terminal64 snapshot stop; no success result')
    need(supervisor['worker_exit']==2 and supervisor['timeout']is False,'Supervisor quality-stop exit2, no timeout')
    for owner in(status,failure,supervisor,export):
        need(owner['protocol_sha256']==PROTOCOL_SHA and owner['optimizer_outcome_unknown']is False,'Consistent outcome protocol/unknown')
        if 'counters'in owner:need(owner['counters']==expected_counts(64,after_snapshot=True),'Exact final712/256/64 counter receipts')
        for key,val in(('optimizer_updates',64),('optimizer_update_attempts',64)):
            if key in owner:need(owner[key]==val,'Consistent final durable count')
        if owner.get('last_committed_checkpoint')is not None:pin_check(owner['last_committed_checkpoint'],members)
    pin_check(export['status_receipt'],members)
    need(export['failure_present']is True and export['result_present']is False
         and export['interrupted_status_write_present']is False,'Truthful transport outcome fields')
    need(failure['backward_scopes_completed_or_partial']==256 and failure['preserve_all_partial_state_no_automatic_retry']is True,
         'Failure keeps completed backward count and no-retry rule')
    pattern=re.compile(re.escape(OUTPUT)+r'/checkpoints/u(\d{4})/checkpoint\.json')
    prefix=sorted(int(m.group(1))for path in members if(m:=pattern.fullmatch(path)))
    need(prefix==list(range(65)),'All65 contiguous fully saved checkpoints required for actual64 commits')
    checkpoint_files=[path for path in members if path.startswith(OUTPUT+'/checkpoints/')]
    need(len(checkpoint_files)==195,'Exactly65 times3 complete checkpoint members')
    gradient_files=[path for path in members if re.fullmatch(re.escape(OUTPUT)+r'/updates/u\d{4}/postclip_gradients\.npz',path)]
    commit_files=[path for path in members if re.fullmatch(re.escape(OUTPUT)+r'/updates/u\d{4}/commit\.json',path)]
    need(len(gradient_files)==len(commit_files)==64,'Exact64 gradient sets/commits')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    need(not torch.cuda.is_initialized(),'No CUDA initialization')
    names=sorted(shape_schema());need(sum(math.prod(s)for s in shape_schema().values())==1731,'Independent2/1731 schema arithmetic')
    initial,r2_binding=initial_from_retained_r2(r2_path,np,torch)
    preflight=jsons[OUTPUT+'/preflight.json'];need(preflight['protocol_sha256']==PROTOCOL_SHA
         and preflight['limits']==p['limits'] and preflight['optimizer_constructions']==0 and preflight['optimizer_updates']==0,
         'Pre-forward receipt binds finite protocol')
    pin_check(preflight['initial_readouts'],members)
    initial_npz=read_npz(blobs[preflight['initial_readouts_NPZ']['file']['path']],np)
    state_array_pin(preflight['initial_readouts_NPZ'],initial_npz,members)
    initial_pt=torch_cpu_decode(blobs[preflight['initial_readouts']['path']],torch)
    need(set(initial_pt)==set(names),'Initial portable PT exact two tensor keys')
    for name in names:
        need(np.array_equal(initial[name],initial_npz[name])and np.array_equal(initial[name],finite_array(initial_pt[name],shape_schema()[name],np,torch)),
             'Exact original R2 / initial NPZ / initial PT readout '+name)
    for label,claimed in preflight['initial_states'].items():
        need(claimed['R2_current_canonical515_sha256']==R2_STATE and claimed['U96_canonical24_sha256']==U96_STATE
             and claimed['R2_except_readout_canonical513_sha256']==r2_binding['canonical513_except_readout_sha256'],
             'Initial whole-model declaration associations '+label)
    sampler=jsons[OUTPUT+'/sampler.json'];cases,coverage=check_sampler(p,sampler)
    need(sampler['reference_order']==learning['optimization_reference_order'] and sampler['role_order']==ROLES
         and sampler['full_64_rolecase_passes']==4 and sampler['dynamic_augmentation']is False,'Fixed no-augmentation sampler metadata')
    sampler_pin={'path':OUTPUT+'/sampler.json',**members[OUTPUT+'/sampler.json']}
    states={};moments={};rngs=[];trajectory=[];last_resume=None
    worst={k:0. for k in TOLERANCES};bad={k:0 for k in TOLERANCES}
    recurrent={name:{'p':initial[name].copy(),'m':np.zeros_like(initial[name]),'v':np.zeros_like(initial[name])}for name in names}
    for u in prefix:
        need(time.monotonic()-began<MAX_SECONDS,'Bounded whole saved-state audit')
        folder=OUTPUT+'/checkpoints/u'+str(u).zfill(4);cp=jsons[folder+'/checkpoint.json']
        need(cp['complete']is True and cp['update']==u and cp['model_qualified']is False,'Checkpoint cursor, not qualification')
        pin_check(cp['resume'],members);pin_check(cp['readout']['file'],members)
        state=read_npz(blobs[cp['readout']['file']['path']],np);state_array_pin(cp['readout'],state,members)
        need(cp['readout_canonical_sha256']==canonical(state),'Checkpoint canonical readout')
        resume=torch_cpu_decode(blobs[cp['resume']['path']],torch)
        opt,rng,lr=validate_resume(resume,u,state,p,sampler_pin,names,np,torch)
        need(cp['counters']==resume['counters']==expected_counts(u),'Exact independently counted percheckpoint ledger')
        states[u]=state;moments[u]=opt;rngs.append(rng);last_resume=resume
        if u==0:
            need(all(np.array_equal(initial[n],state[n])for n in names),'Actual checkpoint0 exactly original R2 RGB readout')
            continue
        c=jsons[OUTPUT+'/updates/u'+str(u).zfill(4)+'/commit.json']
        need(c['complete']is True and c['update']==u and c['LR']==lr and c['readout_before_sha256']==canonical(states[u-1])
             and c['readout_after_sha256']==canonical(state)and c['counters']==expected_counts(u),'Actual linked committed trajectory')
        need(c['all537_frozen_exact']is True and c['all539_grad_fields_None']is True,'VM frozen/grad assertions retained as declarations')
        grads=read_npz(blobs[c['postclip_gradients']['file']['path']],np);state_array_pin(c['postclip_gradients'],grads,members)
        need(len(c['microbatch_receipts'])==4,'Four microbatch scalar witnesses per commit')
        for slot,binding in enumerate(c['microbatch_receipts'],1):
            pin_check(binding,members);w=jsons[binding['path']];id_=sampler['updates'][u-1]['slots'][slot-1]
            need(w['complete']is True and w['id']==id_ and w['update']==u and w['slot']==slot and w['role']==ROLES[slot-1]
                 and w['LR']==lr and w['readout_state_before_commit_sha256']==canonical(states[u-1]),'Microbatch sampler/state binding')
            need(w['photographic_optimization_target']==cases[id_]['target'] and w['every_reported_loss_term_connected']is True,
                 'Same clean photographic target for every role; connectivity remains runtime assertion')
            pin_check(w['photographic_optimization_target'],members);pin_check(w['output_record'],members)
            pin_check(w['comparison_reference_not_optimization_target'],members)
            expected_objective=scalar_objective(w['role'],w['terms'])
            need(w['divided_objective']==w['objective']/4 and abs(w['objective']-expected_objective)<=8e-8+2e-7*abs(expected_objective),
                 'Independent objective-term scalar assembly within FP32 arithmetic only')
        norm=float(np.sqrt(sum(float(np.square(g.astype(np.float64)).sum())for g in grads.values())))
        preclip=c['preclip_norm'];coefficient=min(1.,1./(preclip+1e-6))
        need(type(preclip)is float and math.isfinite(preclip)and preclip>0 and c['clip_max_norm']==1.
             and c['clip_coefficient']==coefficient and norm<=1.00001
             and abs(norm-preclip*coefficient)<=2e-7+preclip*coefficient*2e-5,'Saved postclip gradient norm / declared coefficient')
        changed=sum(not np.array_equal(states[u-1][n],state[n])for n in names)
        need(changed>0 and any(np.count_nonzero(g)for g in grads.values()),'Committed readout changed with finite nonzero saved gradients')
        errors={k:0. for k in TOLERANCES}
        for name in names:
            cm=opt[name]['m'];cv=opt[name]['v'];previous=moments[u-1][name]
            pm,pv,prediction=adam_predict(states[u-1][name],grads[name],previous['m'],previous['v'],u,lr,np,actual_m=cm,actual_v=cv)
            rm,rv,rp=adam_predict(recurrent[name]['p'],grads[name],recurrent[name]['m'],recurrent[name]['v'],u,lr,np)
            recurrent[name]={'m':rm,'v':rv,'p':rp}
            for kind,a,b in zip(TOLERANCES,(cm,cv,state[name],state[name]),(pm,pv,prediction,rp)):
                delta,count=comparison(a,b,np,kind);errors[kind]=max(errors[kind],delta)
                worst[kind]=max(worst[kind],delta);bad[kind]+=count
        trajectory.append({'update':u,'LR':lr,'changed_readout_tensors':changed,
            'readout_canonical_sha256':canonical(state),'preclip_norm':preclip,'postclip_norm_float64':norm,
            'clip_coefficient':coefficient,'Adam_step':u,'recurrence_max_abs':errors,
            'actual_parameter_step_L2':float(math.sqrt(sum(float(np.square(state[n].astype(np.float64)-states[u-1][n]).sum())for n in names))),
            'nonzero_gradient_tensors':sum(bool(np.count_nonzero(g))for g in grads.values())})
    need(all(v==0 for v in bad.values()),'Independent saved Adam arithmetic must match declared envelopes')
    need(all(rng==rngs[0]for rng in rngs),'All65 actual saved CPU/CUDA/Python/NumPy RNG states remain exact')
    initial_factory=preflight['factory_RNG']
    need(initial_factory['before']==initial_factory['after'] and initial_factory['exact']is True,'Declared factory RNG preservation')
    need(initial_factory['before']['CPU_sha256']==rngs[0]['CPU']and initial_factory['before']['CUDA_sha256']==rngs[0]['CUDA'],
         'Factory digest matches actual saved RNG bytes')
    restored=jsons[OUTPUT+'/caller_RNG_restored.json']
    need(restored['complete']is True and restored['CPU_CUDA_Python_NumPy_restored']is True
         and restored['new_neural_or_gradient_calls']==0,'Outer restoration is a retained VM assertion')
    pin_check(failure['retained_partial_state'],members)
    partial=torch_cpu_decode(blobs[failure['retained_partial_state']['path']],torch)
    need(partial['format']=='stopped-existing-RGB-readout-state-v1' and partial['protocol_sha256']==PROTOCOL_SHA
         and partial['pending']is True and partial['active_update']==64 and partial['active_slot']is None
         and partial['optimizer_outcome_unknown']is False and partial['automatic_resume_or_retry_permitted']is False
         and partial['model_qualified']is False,'Retained stopped payload, no unrecorded optimizer outcome')
    need(partial['counters']==expected_counts(64,after_snapshot=True) and partial['sampler']==sampler_pin
         and partial['sampler_completed_backwards']==256 and partial['optimizer_parameter_names']==names,
         'Partial state exact terminal counter/sampler association')
    need(set(partial['readout_state'])==set(partial['partial_gradients'])==set(names)
         and all(value is None for value in partial['partial_gradients'].values()),'No outstanding gradients after64 committed steps')
    for name in names:
        need(np.array_equal(finite_array(partial['readout_state'][name],shape_schema()[name],np,torch),states[64][name]),
             'Failed-state readout exactly equals last committed checkpoint')
    need(tensor_tree_equal(partial['optimizer'],last_resume['optimizer'],torch),'Failed-state Adam exactly equals checkpoint64')
    need(validate_rng(partial,np,torch)==rngs[-1],'Failed-state RNG exactly equals last committed checkpoint')
    need(not torch.cuda.is_initialized(),'No CUDA used by independent saved-state audit')
    final_l2={n:float(np.linalg.norm((states[64][n].astype(np.float64)-states[0][n]).ravel()))for n in names}
    return {'complete':True,'mode':'U96_EXISTING_DECODER_READOUT_REAL_RETURN_SAVED_STATE_ONLY',
        'checks_passed':CHECKS,'artifact_outcome':actual,'archive':{'path':str(archive),**transport},
        'protocol_sha256':PROTOCOL_SHA,'sealed_nonweight_static_files_verified':static,
        'original_weight_files_intentionally_omitted':omitted,'retained_R2_independent_decode':r2_binding,
        'verified_checkpoint_ids':prefix,'checkpoint_sets_verified':65,'postclip_gradient_sets_verified':64,
        'portable_readout_and_gradient_arrays_verified':2*(65+64),'readout_trainable_tensors':2,'readout_parameters':1731,
        'safe_weights_only_CPU_return_decodes':67,'safe_weights_only_CPU_original_R2_decodes':1,
        'actual_committed_prefix_verified':64,'committed_trajectory_fully_covered':True,
        'verified_initial_readout_canonical':canonical(initial),'verified_final_readout_canonical':canonical(states[64]),
        'verified_fresh_Adam_state_at0':True,'declared_maximum_LR':5e-5,'warmup_LRs':[1.25e-5,2.5e-5,3.75e-5,5e-5],
        'sampler_optimization_photographs':16,'sampler_monitor_photographs':8,'sampler_optimization_rolecases':len(coverage),
        'sampler_exposures_per_rolecase':4,'verified_committed_exposures':256,
        'final_vs_initial_L2_by_tensor':final_l2,'final_vs_initial_L2_total':math.sqrt(sum(v*v for v in final_l2.values())),
        'trajectory':trajectory,'Adam_arithmetic':{'tolerances':TOLERANCES,'worst_absolute_errors':worst,
            'values_over_tolerance':bad,'per_step_parameters_use_actual_saved_current_moments':True,
            'additional_fully_recurrent_parameters_use_only_saved_initial_state_and_gradients':True,
            'tolerance_scope':'Arithmetic FP32 CPU/CUDA lerp/addcmul/addcdiv rounding only; not quality gate thresholds.',
            'fully_recurrent_envelope':'64-step accumulation allowance; stricter individual-step/moment checks also apply.',
            'no_quality_or_preservation_threshold_relaxation':True},
        'RNG':{'checkpoint_count':65,'initial':rngs[0],'final':rngs[-1],'all_saved_checkpoint_RNG_exact':True,'failed_state_RNG_exact':True},
        'failed_state_readout_optimizer_exactly_matches_commit64':True,
        'frozen537_evidence_scope':'Original515 R2 initializer independently CPU-decoded and hashed; readout2 matched exactly. Returned whole frozen537 tensor values are absent; every resume/commit frozen assertion and source binding is checked, not a post-training rehash of omitted tensor values.',
        'recorded_quality_stop_preserved':True,'quality_gate_recalculation_scope':'Not performed by this state auditor; independent saved-pixel gate audit required.',
        'no_model_or_quality_qualification':True,'actual_local_model_constructions':0,'actual_local_forward_calls':0,
        'actual_local_backward_calls':0,'actual_local_optimizer_constructions':0,'actual_local_optimizer_steps':0,
        'VM_connections':0,'Torch_used_only_for_weights_only_CPU_checkpoint_decode':True,'CUDA_initialized':False,
        'elapsed_seconds':time.monotonic()-began}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--archive-sha',required=True)
    parser.add_argument('--export-receipt',type=Path,required=True)
    parser.add_argument('--r2-checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.absolute()
    need(not out.exists()and not out.is_symlink()and out.is_relative_to(ROOT/'outputs'),'Exclusive in-workspace output receipt')
    out.parent.mkdir(parents=True,exist_ok=True)
    for parent in out.parents:
        s=parent.lstat();need(stat.S_ISDIR(s.st_mode)and not stat.S_ISLNK(s.st_mode)
            and not getattr(s,'st_file_attributes',0)&1024,'Ordinary output ancestry')
    report=audit(args.archive.absolute(),args.archive_sha,args.export_receipt.absolute(),args.r2_checkpoint.absolute())
    raw=Path(__file__).read_bytes();report['auditor']={'path':str(Path(__file__).absolute()),'bytes':len(raw),'sha256':sha(raw)}
    with out.open('x',encoding='utf-8',newline='\n')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':True,'receipt':str(out),'scientific_status':report['artifact_outcome']['scientific_status'],
        'verified_commits':report['actual_committed_prefix_verified'],'checkpoints':65,'gradient_sets':64,
        'checks_passed':report['checks_passed'],'model_qualified':False}),flush=True)


if __name__=='__main__':main()

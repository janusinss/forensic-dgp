"""Bounded saved-state auditor. Imports never execute models or numerical code.

Torch, if needed, only decodes bounded CPU weights_only resume payloads.
No model/optimizer construction, forward, backward, step, VM or resume operation.
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
PROTOCOL_SHA='7bf251dd257d79dfcd8e4bd07fe7674badf4b2f30c15fac10cc048b9a36e778a'
B1_PROTOCOL_SHA='11e3286c64d9de282e46d3fa95ae5c792a402c72a681fba4040d816ae5944903'
B1_ARCHIVE_SHA='5f99df00c2eff645d3bc957606766753d3308aaa4e0e22da57ac28edeb8d858a'
MAX_ARCHIVE=1536*1024**2
MAX_PAYLOAD=1536*1024**2
MAX_FILE=128*1024**2
MAX_RETAINED=320*1024**2
MAX_FILES=6000
MAX_SECONDS=240
MAX_EXPANDED=MAX_PAYLOAD+16*1024**2
R2_STATE='a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_STATE='fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
INITIAL_HEADS='2e7826c07d31298eecf07d7f4d53286aa86247c00157dea9005242f23902726c'
ROLES=['local_gaussian_1p2','global_gaussian_1p2','clear','lowlight_linear_0p35']
ANCHOR={'name':'SmoothL1','beta':.01,'reduction':'mean',
        'scope':['clear_RGB','clear_adjacent_residual','dark_block16_luminance']}
TOLERANCES={'first_moment':{'atol':2e-9,'rtol':2e-5},
            'second_moment':{'atol':2e-12,'rtol':2e-5},
            'parameter_update':{'atol':6e-8,'rtol':3e-6}}


def need(ok,message):
    if not ok:raise ValueError(message)


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


def shape_schema():
    out={}
    for site,c in [('32',256),('64',256),('128',128),('256',128)]:
        shapes={'encoder.weight':(32,c,1,1),'encoder.bias':(32,),
                'decoder.weight':(32,c,1,1),'fused.weight':(32,c,1,1),
                'coarse.weight':(32,256,1,1),'spatial.weight':(32,32,3,3),
                'spatial.bias':(32,),'output.weight':(c,32,1,1),'output.bias':(c,)}
        for name,shape in shapes.items():out['heads.'+site+'.'+name]=shape
    need(len(out)==36 and sum(math.prod(s)for s in out.values())==168960,'Independent shape arithmetic')
    return out


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
            need(len(info)==36 and all('/'not in n and n.endswith('.npy')for n in names)
                 and set(n[:-4]for n in names)==set(shape_schema()),'Exact36 NPZ keys')
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
    need(isinstance(t,torch.Tensor)and t.device.type=='cpu','CPU Tensor required')
    a=t.detach().contiguous().numpy()
    need(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite saved moment/head schema')
    return a


def adam_formula(previous_head,gradient,previous_m,previous_v,current_m,current_v,update,lr,np):
    """Pure NumPy recurrence, not an optimizer constructor or step."""
    mp=(previous_m+(gradient-previous_m)*np.float32(.1)).astype(np.float32)
    vp=(previous_v*np.float32(.99)+(gradient*gradient)*np.float32(.01)).astype(np.float32)
    denom=(np.sqrt(current_v,dtype=np.float32)/np.float32(math.sqrt(1-.99**update))+np.float32(1e-8)).astype(np.float32)
    param=(previous_head-np.float32(lr/(1-.9**update))*(current_m/denom)).astype(np.float32)
    return mp,vp,param


def comparison(actual,predicted,np,kind):
    delta=np.abs(actual.astype(np.float64)-predicted.astype(np.float64))
    t=TOLERANCES[kind];allowed=t['atol']+t['rtol']*np.abs(actual.astype(np.float64))
    return float(delta.max()),int(np.count_nonzero(delta>allowed))


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
                    keep=(rel.endswith('.json')or re.fullmatch(output+r'/checkpoints/u\d{4}/(?:heads\.npz|resume\.pt)',rel)is not None
                          or re.fullmatch(output+r'/updates/u\d{4}/postclip_gradients\.npz',rel)is not None)
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
    rows=binding['arrays'];need(type(rows)is list and len(rows)==36,'36 portable schema rows')
    need(len({r['name']for r in rows})==36 and {r['name']for r in rows}==set(state),'Unique portable row names')
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


def validate_resume(resume,u,heads,p,sampler_pin,names,np,torch,mode):
    fmt='complete-U96-postfusion-learning-resume-v1'if mode=='B1_FIXTURE'else'complete-U96-postfusion-smoothanchors-resume-v1'
    need(type(resume)is dict and resume['format']==fmt and resume['protocol_sha256']==p['_verified_sha'],'Resume format/protocol')
    need(resume['update']==u and resume['pending']is False
         and resume['next_update']==(u+1 if u<64 else None)and resume['finite_terminal_update']==64,'Finite resume cursor')
    need(resume['sampler']==sampler_pin and resume['sampler_committed_exposures']==u*4
         and resume['sampler_role_order']==ROLES
         and resume['sampler_reference_order']==p['learning']['optimization_reference_order'],'Resume sampler binding/cursor')
    need(resume['learning']==p['learning']and resume['all539_frozen_exact']is True
         and resume['R2_canonical_sha256']==R2_STATE and resume['U96_adapter_canonical_sha256']==U96_STATE,'Resume frozen lineage')
    need(resume['optimizer_parameter_names']==names and set(resume['head_state'])==set(names),'Stable36 parameter order')
    for n,t in resume['head_state'].items():
        need(np.array_equal(finite_array(t,shape_schema()[n],np,torch),heads[n]),'PT/NPZ head mismatch')
    need(resume['heads_canonical_sha256']==canonical(heads),'Resume canonical')
    cpu=resume['CPU_RNG'];cuda=resume['CUDA_RNG'];nr=resume['NumPy_random_state'];pr=resume['Python_random_state']
    need(isinstance(cpu,torch.Tensor)and cpu.dtype==torch.uint8 and cpu.ndim==1 and cpu.numel()>0
         and type(cuda)is list and len(cuda)==1 and cuda[0].dtype==torch.uint8 and cuda[0].ndim==1,'Portable CPU/CUDA RNG')
    need(nr['algorithm']=='MT19937'and nr['keys'].dtype==torch.uint32 and tuple(nr['keys'].shape)==(624,)
         and type(nr['position'])is int and 0<=nr['position']<=624
         and nr['has_gauss']in(0,1)and math.isfinite(nr['cached_gaussian']),'Portable NumPy RNG')
    need(type(pr)is tuple and len(pr)==3 and pr[0]==3 and type(pr[1])is tuple
         and len(pr[1])==625,'Portable Python RNG')
    rng={'CPU':sha(cpu.numpy().tobytes()),'CUDA':sha(cuda[0].numpy().tobytes()),
         'NumPy_keys':sha(nr['keys'].numpy().tobytes()),'NumPy_position':nr['position'],
         'NumPy_has_gauss':nr['has_gauss'],'NumPy_cached_gaussian':nr['cached_gaussian'],
         'Python':sha(repr(pr).encode())}
    opt=resume['optimizer']
    need(type(opt)is dict and set(opt)=={'state','param_groups'}and len(opt['param_groups'])==1,'One saved Adam group')
    g=opt['param_groups'][0];lr=5e-5*min(1.,u/4)if u else 5e-5
    need(g['params']==list(range(36))and tuple(g['betas'])==(.9,.99)and g['eps']==1e-8
         and g['weight_decay']==0 and g['amsgrad']is False and g['foreach']is False
         and g['fused']is False and g['capturable']is False and g['differentiable']is False
         and g['maximize']is False and g['lr']==lr,'Exact saved Adam policy')
    need(set(opt['state'])==(set()if u==0 else set(range(36))),'Saved Adam state census')
    if u:
        for i,n in enumerate(names):
            row=opt['state'][i];need(set(row)=={'step','exp_avg','exp_avg_sq'},'Adam moment fields')
            step=row['step'];need(isinstance(step,torch.Tensor)and step.device.type=='cpu'
                 and step.numel()==1 and float(step.item())==u,'Saved Adam step')
            finite_array(row['exp_avg'],shape_schema()[n],np,torch)
            v=finite_array(row['exp_avg_sq'],shape_schema()[n],np,torch);need((v>=0).all(),'Nonnegative second moment')
    return opt,rng,lr


def audit(archive,archive_sha,export_path,mode='SMOOTHANCHORS_RETURN'):
    began=time.monotonic()
    fixture=mode=='B1_FIXTURE'
    stage='cctv_codeformer_u96_postfusion_learning_v1_vm'if fixture else'cctv_codeformer_u96_postfusion_smoothanchors_v1_vm'
    output='outputs/u96_postfusion_learning_v1'if fixture else'outputs/u96_postfusion_smoothanchors_v1'
    protocol_sha=B1_PROTOCOL_SHA if fixture else PROTOCOL_SHA
    if fixture:need(archive_sha==B1_ARCHIVE_SHA,'Fixture mode only accepts exact historical B1 archive')
    members,jsons,blobs,invalid,transport=read_archive(archive,archive_sha,stage,output)
    need('protocol.json'in jsons and members['protocol.json']['sha256']==protocol_sha,'Exact sealed protocol')
    p=jsons['protocol.json'];need(p['stage_name']==stage,'Sealed stage')
    need(p['learning']['slot_roles']==ROLES and p['learning']['max_updates']==64
         and p['learning']['initial_heads_canonical_sha256']==INITIAL_HEADS,'Fixed learning/census')
    if not fixture:need(p['learning']['anchor_criterion']==ANCHOR,'Exact SmoothL1 anchor policy')
    static=0;omitted=[]
    for name,binding in p['files'].items():
        if name.startswith('weights/'):
            need(name not in members,'Return should omit original incumbent weights');omitted.append(name);continue
        need(name in members and members[name]==binding,'All sealed static source/data pins '+name);static+=1
    need(set(omitted)=={p['model']['r2']['path'],p['model']['u96']['path']},'Only exact two incumbent files omitted')
    export=strict_json(regular(export_path,1024**2).read_bytes())
    need(export.get('complete')is True and export['archive_sha256']==archive_sha
         and export['bytes']==transport['bytes']and export['protocol_sha256']==protocol_sha
         and export['payload_files']==transport['payload_files']and export['payload_bytes']==transport['payload_bytes'],'Export binds actual archive/payload')
    status=jsons.get(output+'/status.json');pending=output+'/status.pending.json'in members
    failure=jsons.get(output+'/failure.json');result=jsons.get(output+'/result.json');supervisor=jsons.get('supervisor.json')
    state=outcome(status,pending,failure,result,supervisor)
    need(export['optimizer_updates']==state['confirmed_optimizer_updates']
         and export['confirmed_optimizer_updates']==state['confirmed_optimizer_updates']
         and export['optimizer_update_attempts']==state['optimizer_attempts']
         and export['optimizer_outcome_unknown']==state['optimizer_outcome_unknown']
         and export['interrupted_status_write_present']==pending
         and export['result_present']==(output+'/result.json'in members)
         and export['failure_present']==(output+'/failure.json'in members),'Export durable truth')
    need(export['status_receipt_present']==(status is not None),'Export status presence')
    if status:
        need(status['protocol_sha256']==protocol_sha,'Status protocol')
        pin_check(export['status_receipt'],members)
    for row in(failure,result):
        if row is not None:
            need(row['protocol_sha256']==protocol_sha,'Outcome protocol')
            if status is not None:need(row['counters']==status['counters'],'Final failure/result/status counts')
    p['_verified_sha']=protocol_sha
    checkpoint_pattern=re.compile(re.escape(output)+r'/checkpoints/u(\d{4})/checkpoint\.json')
    checkpoint_ids=sorted(int(m.group(1))for name in members if(m:=checkpoint_pattern.fullmatch(name))and name in jsons)
    need(all(0<=u<=64 for u in checkpoint_ids),'Bounded checkpoint update')
    prefix=[]
    for u in range(65):
        if u not in checkpoint_ids:break
        prefix.append(u)
    unattached=[u for u in checkpoint_ids if u not in prefix]
    count=state['confirmed_optimizer_updates']
    need(not prefix or count is None or prefix[-1]<=count,'Checkpoint cannot exceed durable committed count')
    if state['scientific_status']=='FINITE_SCREEN_PASSED_NOT_QUALIFIED':
        need(prefix==list(range(65))and not invalid,'Success requires all65 complete checkpoints/no malformed runtimeJSON')
    # Numerical dependencies are imported only after bounded transport/source pins.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    need(not torch.cuda.is_initialized(),'Audit must not initialize CUDA')
    names=sorted(shape_schema());heads={};opts={};rngs=[];trajectory=[];gradient_sets=0
    worst={k:0. for k in TOLERANCES};bad={k:0 for k in TOLERANCES};cases={};coverage={}
    sampler=jsons.get(output+'/sampler.json')
    if prefix:
        need(sampler is not None,'Committed states require frozen sampler')
        cases,coverage=check_sampler(p,sampler)
    sampler_pin={'path':output+'/sampler.json',**members[output+'/sampler.json']}if sampler is not None else None
    if sampler_pin:pin_check(sampler_pin,members)
    final_names=[n for n in names if'.output.'in n];need(len(final_names)==8,'Eight final projections')
    for u in prefix:
        need(time.monotonic()-began<MAX_SECONDS,'Bounded saved-state audit time')
        folder=output+'/checkpoints/u'+str(u).zfill(4);cp=jsons[folder+'/checkpoint.json']
        pin_check(cp['resume'],members);state_array=cp['heads'];pin_check(state_array['file'],members)
        heads[u]=read_npz(blobs[state_array['file']['path']],np)
        state_array_pin(state_array,heads[u],members)
        need(cp['update']==u and cp['head_canonical_sha256']==canonical(heads[u]),'Checkpoint canonical/update')
        resume=torch_cpu_decode(blobs[cp['resume']['path']],torch)
        opt,rng,lr=validate_resume(resume,u,heads[u],p,sampler_pin,names,np,torch,mode);opts[u]=opt;rngs.append(rng)
        need(resume['counters']==cp['counters']and cp['counters']['optimizer_updates']==u
             and cp['counters']['optimizer_update_attempts']==u,'Checkpoint committed counters')
        if u==0:
            need(canonical(heads[u])==INITIAL_HEADS,'Exact full-U96 fresh head initializer')
            need(all(not np.count_nonzero(heads[0][n])for n in final_names)
                 and any(np.count_nonzero(heads[0][n])for n in names if n not in final_names),'Zero final projections only')
            continue
        commit_path=output+'/updates/u'+str(u).zfill(4)+'/commit.json'
        need(commit_path in jsons,'Complete saved checkpoint requires prior commit receipt')
        c=jsons[commit_path]
        need(c['update']==u and c['LR']==lr and c['head_before_sha256']==canonical(heads[u-1])
             and c['head_after_sha256']==canonical(heads[u]),'Commit before/after actual trajectory')
        need(c['all539_frozen_exact']is True and c['all575_grad_fields_None']is True
             and c['counters']['optimizer_updates']==u and c['counters']['optimizer_update_attempts']==u,'Commit invariant/count association')
        binding=c['postclip_gradients'];pin_check(binding['file'],members)
        grads=read_npz(blobs[binding['file']['path']],np);state_array_pin(binding,grads,members);gradient_sets+=1
        need(len(c['microbatch_receipts'])==4,'Four objective witnesses')
        for slot,binding in enumerate(c['microbatch_receipts'],1):
            pin_check(binding,members);w=jsons[binding['path']];id_=sampler['updates'][u-1]['slots'][slot-1]
            need(w['id']==id_ and w['update']==u and w['slot']==slot and w['role']==cases[id_]['role']
                 and w['LR']==lr and w['head_state_before_commit_sha256']==canonical(heads[u-1])
                 and w['divided_objective']==w['objective']/4,'Actual microbatch scalar/sampler/state association')
            pin_check(w['output_record'],members);pin_check(w['teacher_reference'],members)
            if not fixture:
                required=(['U96_image_anchor_L1','U96_adjacent_anchor_L1','U96_image_anchor_SmoothL1','U96_adjacent_anchor_SmoothL1']
                    if w['role']=='clear'else['U96_block16_luminance_anchor_L1','U96_block16_luminance_anchor_SmoothL1']
                    if w['role']=='lowlight_linear_0p35'else[])
                need(all(k in w['terms']for k in required),'OldL1/newSmoothL1 diagnostics retained')
        norm=float(np.sqrt(sum(float(np.square(g.astype(np.float64)).sum())for g in grads.values())))
        recorded=c['preclip_norm'];coef=min(1.,1./(recorded+1e-6))
        need(recorded>0 and c['clip_max_norm']==1. and c['clip_coefficient']==coef and norm<=1.00001
             and abs(norm-recorded*coef)<=2e-7+recorded*coef*2e-5,'Actual gradient norm/clip receipt')
        changed=sum(not np.array_equal(heads[u-1][n],heads[u][n])for n in names)
        need(changed>0,'Confirmed optimizer changed a head tensor')
        errors={k:0. for k in TOLERANCES}
        for i,n in enumerate(names):
            current=opt['state'][i];previous=opts[u-1]['state'].get(i)
            m=finite_array(current['exp_avg'],shape_schema()[n],np,torch)
            v=finite_array(current['exp_avg_sq'],shape_schema()[n],np,torch)
            pm=finite_array(previous['exp_avg'],shape_schema()[n],np,torch)if previous else np.zeros_like(grads[n])
            pv=finite_array(previous['exp_avg_sq'],shape_schema()[n],np,torch)if previous else np.zeros_like(grads[n])
            predictions=adam_formula(heads[u-1][n],grads[n],pm,pv,m,v,u,lr,np)
            for kind,a,b in zip(TOLERANCES,(m,v,heads[u][n]),predictions):
                delta,failed=comparison(a,b,np,kind);errors[kind]=max(errors[kind],delta)
                worst[kind]=max(worst[kind],delta);bad[kind]+=failed
        trajectory.append({'update':u,'LR':lr,'changed_tensors':changed,
            'nonzero_final_projection_gradient_tensors':sum(bool(np.count_nonzero(grads[n]))for n in final_names),
            'preclip_norm':recorded,'postclip_norm_float64':norm,'clip_coefficient':coef,
            'head_canonical_sha256':canonical(heads[u]),'Adam_step':u,'recurrence_max_abs':errors})
    need(all(v==0 for v in bad.values()),'Saved Adam recurrence outside explicit CPU/CUDA FP32 audit tolerance')
    last=prefix[-1]if prefix else None
    for owner in(status,failure,result):
        if owner is None:continue
        pin=owner.get('last_committed_checkpoint')or owner.get('last_checkpoint')
        if pin is not None:
            pin_check(pin,members)
            need(last is not None and pin['path']==output+'/checkpoints/u'+str(last).zfill(4)+'/checkpoint.json','Last fully saved checkpoint binding')
    if export.get('last_committed_checkpoint')is not None:
        pin_check(export['last_committed_checkpoint'],members)
        need(last is not None and export['last_committed_checkpoint']['path']==output+'/checkpoints/u'+str(last).zfill(4)+'/checkpoint.json','Export last checkpoint')
    incomplete=[name for name in members if name.startswith(output+'/checkpoints/')
                and (m:=re.match(re.escape(output)+r'/checkpoints/u(\d{4})/',name))and int(m.group(1))not in prefix]
    gap=(count-last)if count is not None and last is not None else None
    covered=(count is not None and not state['optimizer_outcome_unknown']and not invalid
             and not unattached and not incomplete and((count==0 and last in(None,0))or last==count))
    need(not torch.cuda.is_initialized(),'No CUDA use in saved-state audit')
    delta=({n:float(np.linalg.norm((heads[last][n]-heads[0][n]).astype(np.float64)))for n in names}if prefix else None)
    return {'complete':True,'mode':'FIXTURE_B1_ONLY_NOT_A_NEW_RESULT'if fixture else'SMOOTHANCHORS_REAL_SAVED_RETURN',
        'saved_state_audit_ran':True,'audited_complete_checkpoint_prefix_pass':True,
        'durable_committed_trajectory_fully_covered':covered,'artifact_outcome':state,
        'archive':{'path':str(archive),**transport},'protocol_sha256':protocol_sha,
        'sealed_nonweight_static_files_verified':static,'original_weight_files_intentionally_omitted':omitted,
        'verified_checkpoint_ids':prefix,'unattached_checkpoint_ids':unattached,
        'malformed_runtime_JSON':invalid,'incomplete_checkpoint_members':incomplete,
        'last_verified_saved_checkpoint':last,'confirmed_commits_beyond_verified_checkpoint':gap,
        'head_checkpoint_sets_verified':len(prefix),'postclip_gradient_sets_verified':gradient_sets,
        'portable_head_gradient_arrays_verified':36*(len(prefix)+gradient_sets),'head_parameters':168960,
        'safe_weights_only_CPU_resume_decodes':len(prefix),'sampler_optimization_rolecases':len(coverage),
        'sampler_full_plan_exposures_per_rolecase':4 if coverage else None,
        'verified_committed_training_exposures':4*last if last is not None else 0,
        'final_vs_initial_head_L2_by_tensor':delta,
        'final_vs_initial_head_L2_total':float(math.sqrt(sum(v*v for v in delta.values())))if delta else None,
        'changed_head_tensors_last_vs_initial':sum(v>0 for v in delta.values())if delta else 0,
        'trajectory':trajectory,'Adam_formula_reconstruction':{'tolerances':TOLERANCES,
            'worst_absolute_errors':worst,'values_over_tolerance':bad,
            'parameter_formula_uses_actual_saved_current_moments':True,
            'no_quality_threshold_relaxation':True,'cross_device_bitwise_gradient_repeatability_not_claimed':True},
        'RNG':{'validated_checkpoint_count':len(rngs),'initial':rngs[0]if rngs else None,
            'last':rngs[-1]if rngs else None,'unchanged_in_verified_prefix':all(x==rngs[0]for x in rngs)if rngs else None},
        'frozen539_evidence_scope':'Exact sealed initializer pins and every resume/commit canonical association checked; omitted original539 tensor values cannot be rehashed from this return.',
        'no_model_or_quality_qualification':True,'actual_local_model_constructions':0,
        'actual_local_forward_calls':0,'actual_local_backward_calls':0,
        'actual_local_optimizer_constructions':0,'actual_local_optimizer_steps':0,'VM_connections':0,
        'Torch_used_only_for_weights_only_CPU_checkpoint_decode':True,'CUDA_initialized':False,
        'elapsed_seconds':time.monotonic()-began}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--archive-sha',required=True)
    parser.add_argument('--export-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True,help='New JSON output; never overwrite')
    parser.add_argument('--fixture-b1',action='store_true',help='Explicit exact historical B1 replay; never a SmoothL1 result')
    args=parser.parse_args()
    need(re.fullmatch('[0-9a-f]{64}',args.archive_sha)is not None,'Required lowercase SHA256')
    out=args.out.absolute();need(not out.exists()and not out.is_symlink(),'Exclusive output receipt')
    need(out.is_relative_to(ROOT/'outputs'),'Audit receipt stays inside project outputs')
    out.parent.mkdir(parents=True,exist_ok=True)
    for parent in(out.parent,*out.parents):
        info=parent.lstat()
        need(stat.S_ISDIR(info.st_mode)and not stat.S_ISLNK(info.st_mode)
             and not getattr(info,'st_file_attributes',0)&1024,'Ordinary audit-output parent')
    regular(args.archive);regular(args.export_receipt,1024**2)
    report=audit(args.archive.absolute(),args.archive_sha,args.export_receipt.absolute(),
                 'B1_FIXTURE'if args.fixture_b1 else'SMOOTHANCHORS_RETURN')
    raw=Path(__file__).read_bytes();report['auditor']={'path':str(Path(__file__).absolute()),'bytes':len(raw),'sha256':sha(raw)}
    with out.open('x',encoding='utf-8',newline='\n')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':True,'receipt':str(out),'mode':report['mode'],
        'scientific_status':report['artifact_outcome']['scientific_status'],
        'last_verified_saved_checkpoint':report['last_verified_saved_checkpoint'],
        'durable_committed_trajectory_fully_covered':report['durable_committed_trajectory_fully_covered'],
        'model_qualified':False}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

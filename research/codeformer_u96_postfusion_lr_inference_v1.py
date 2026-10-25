"""Frozen CPU-only terminal-LR inference source; inert until explicit loading.

This module does not invoke the trainable prototype factory. Its narrow copy
preserves the audited numerical architecture while requiring all575 tensors
frozen/eval and caller no_grad. Source review is not CPU numerical parity,
DEV usefulness, identity recovery or model qualification.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import sys
import threading
import weakref
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SITES=('32','64','128','256')
CHANNELS={'32':256,'64':256,'128':128,'256':128}
WIDTH=32
R2_FILE={'bytes':376609465,'sha256':'d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'}
U96_FILE={'bytes':212827,'sha256':'958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'}
HEAD_FILE={'bytes':685818,'sha256':'77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'}
R2_STATE='a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_STATE='fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
HEAD_STATE='6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
PROTOCOL_SHA='8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9'
ARCHIVE_SHA='7e5208d2ff69713c834c21fb466020ae54d7ee68b84795e2c044ffc11847041e'
PROTOTYPE={'path':'research/codeformer_u96_postfusion_v1/model.py','bytes':23240,
           'sha256':'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8'}
ARCHITECTURE_SOURCES={
 'third_party.codeformer.codeformer_arch':{'path':'third_party/codeformer/codeformer_arch.py',
  'sha256':'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb'},
 'third_party.codeformer.vqgan_arch':{'path':'third_party/codeformer/vqgan_arch.py',
  'sha256':'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf'},
 'third_party.codeformer.registry':{'path':'third_party/codeformer/registry.py',
  'sha256':'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'}}
_OWNER_ATTRIBUTE='_forensic_codeformer_exclusive_owner'


def need(ok,message):
    if not ok:raise ValueError(message)


def ordinary_file(path,maximum=512*1024**2):
    path=Path(path).absolute()
    need(path.is_relative_to(ROOT),'Artifact/source must stay inside this project')
    for parent in path.parents:
        info=parent.lstat()
        need(stat.S_ISDIR(info.st_mode)and not stat.S_ISLNK(info.st_mode)
             and not getattr(info,'st_file_attributes',0)&1024,'Ordinary source/artifact parent')
    info=path.lstat()
    need(stat.S_ISREG(info.st_mode)and not stat.S_ISLNK(info.st_mode)and info.st_nlink==1
         and not getattr(info,'st_file_attributes',0)&1024 and 0<info.st_size<=maximum,
         'Bounded unlinked ordinary source/artifact required')
    return path


def file_pin(path,maximum=512*1024**2):
    path=ordinary_file(path,maximum);before=path.stat();h=hashlib.sha256()
    with path.open('rb')as stream:
        for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
    after=path.stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
         'Source/artifact changed during hash read')
    return {'bytes':before.st_size,'sha256':h.hexdigest()}


def adapter_shapes():
    return {'adapters.'+site+'.'+name:shape for site,c in CHANNELS.items()
        for name,shape in{'down.weight':(32,c,1,1),'down.bias':(32,),
            'depthwise.weight':(32,1,3,3),'depthwise.bias':(32,),
            'up.weight':(c,32,1,1),'up.bias':(c,)}.items()}


def head_shapes():
    out={}
    for site,c in CHANNELS.items():
        for name,shape in{'encoder.weight':(32,c,1,1),'encoder.bias':(32,),
            'decoder.weight':(32,c,1,1),'fused.weight':(32,c,1,1),
            'coarse.weight':(32,256,1,1),'spatial.weight':(32,32,3,3),
            'spatial.bias':(32,),'output.weight':(c,32,1,1),'output.bias':(c,)}.items():
            out['heads.'+site+'.'+name]=shape
    need(len(out)==36 and sum(math.prod(s)for s in out.values())==168960,'Exact36/168960 head schema')
    return out


def inspect_head_npz(raw):
    """Pure NumPy schema parser, not a model loader or terminal-value assertion."""
    need(type(raw)is bytes and 0<len(raw)<1024**2,'Bounded head NPZ bytes')
    with zipfile.ZipFile(io.BytesIO(raw))as archive:
        items=archive.infolist();names=archive.namelist()
        need(len(items)==len(set(names))==36 and set(n[:-4]for n in names)==set(head_shapes())
             and all('/'not in n and'\\'not in n and':'not in n and n.endswith('.npy')for n in names),
             'Exact36 unique flat head array keys')
        need(all(not x.flag_bits&1 and not stat.S_ISLNK(x.external_attr>>16)for x in items)
             and sum(x.file_size for x in items)<1024**2,'Ordinary bounded expanded NPZ')
        import numpy as np
        for member in items:
            with archive.open(member)as stream:
                version=np.lib.format.read_magic(stream);need(version in((1,0),(2,0)),'Supported bounded NPY version')
                reader=np.lib.format.read_array_header_1_0 if version==(1,0)else np.lib.format.read_array_header_2_0
                shape,fortran,dtype=reader(stream,max_header_size=4096)
                need(shape==head_shapes()[member.filename[:-4]]and dtype==np.dtype('float32')
                     and not dtype.hasobject and not fortran,'Exact NPY head schema before allocation')
                need(member.file_size==stream.tell()+math.prod(shape)*4,'Exact NPY data size')
        with np.load(io.BytesIO(raw),allow_pickle=False)as z:
            result={}
            for name,shape in head_shapes().items():
                a=z[name];need(a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all(),'Finite exact head values')
                result[name]=np.ascontiguousarray(a).copy()
            return result


def numpy_state_sha(state):
    """Canonical FP32 array digest matching the sealed tensor-state algorithm."""
    h=hashlib.sha256()
    for name,a in sorted(state.items()):
        header=json.dumps([name,'torch.float32',list(a.shape)],separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'big'));h.update(header);h.update(a.tobytes(order='C'))
    return h.hexdigest()


def source_contract():
    """Read-only static map; no construction, extraction or inference."""
    return {'format':'complete-U96-terminal-LR-frozen-CPU-inference-source-v1',
        'R2_file':dict(R2_FILE),'U96_file':dict(U96_FILE),'head_NPZ_file':dict(HEAD_FILE),
        'canonical_states':{'R2_515':R2_STATE,'U96_24':U96_STATE,'terminal_heads_36':HEAD_STATE},
        'protocol_sha256':PROTOCOL_SHA,'archive_sha256':ARCHIVE_SHA,
        'frozen_tensors':575,'sites':list(SITES),'head_parameters':168960,
        'policy':'CPU FP32 singleton512, w1, detach16, AdaIN, caller torch.no_grad required',
        'base_policy':'newly constructed internally; no caller/shared base accepted',
        'no_training_entrypoint':True,'model_qualified':False,'runtime_and_DEV_review_pending':True}


def load_frozen_cpu_restorer(r2_artifact,u96_artifact,head_artifact,*,inference_source_sha256):
    """Construct a distinct frozen CPU restorer only after source/runtime review.

    Each artifact is exact fixed bytes; the terminal head is the unchanged NPZ
    from the audited u64 checkpoint, not a resume.pt or a newly exported pickle.
    No current app object/base can be supplied or mutated by this API.
    """
    need(re.fullmatch('[0-9a-f]{64}',str(inference_source_sha256))is not None,'Reviewed inference source SHA required')
    own=Path(__file__).absolute()
    need(own.resolve()==(ROOT/'research/codeformer_u96_postfusion_lr_inference_v1.py').resolve()
         and file_pin(own)['sha256']==inference_source_sha256,'Exact reviewed inference source origin/bytes')
    need(file_pin(ROOT/PROTOTYPE['path'])=={k:PROTOTYPE[k]for k in('bytes','sha256')},'Immutable prototype source differs')
    paths=[ordinary_file(p)for p in(r2_artifact,u96_artifact,head_artifact)]
    for path,expected in zip(paths,(R2_FILE,U96_FILE,HEAD_FILE)):
        need(file_pin(path)==expected,'Exact inference artifact bytes required')
    for binding in ARCHITECTURE_SOURCES.values():
        need(file_pin(ROOT/binding['path'])['sha256']==binding['sha256'],'Retained architecture bytes differ')
    arrays=inspect_head_npz(paths[2].read_bytes())
    need(numpy_state_sha(arrays)==HEAD_STATE,'Only exact audited terminal LR head values accepted')
    import torch
    from torch.nn import functional as F
    from third_party.codeformer.codeformer_arch import CodeFormer
    need(torch.get_default_dtype()==torch.float32,'FP32 default dtype required before CPU model construction')
    for name,binding in ARCHITECTURE_SOURCES.items():
        loaded=sys.modules.get(name)
        need(loaded is not None and getattr(loaded,'__file__',None)
             and Path(loaded.__file__).resolve()==(ROOT/binding['path']).resolve()
             and file_pin(loaded.__file__)['sha256']==binding['sha256'],'Exact loaded architecture origin')
    saved_r2=torch.load(paths[0],map_location='cpu',weights_only=True,mmap=True)
    saved_u96=torch.load(paths[1],map_location='cpu',weights_only=True,mmap=True)
    need(type(saved_r2)is dict and set(saved_r2)=={'params_ema'},'Exact R2 inference-only state format')
    need(type(saved_u96)is dict and set(saved_u96)=={'format','adapter_state'}
         and saved_u96['format']=='codeformer-spatial-adapter-inference-v1','Exact U96 adapter inference format')
    r2=saved_r2['params_ema'];u96=saved_u96['adapter_state']
    need(isinstance(r2,dict)and len(r2)==515 and type(u96)is dict and set(u96)==set(adapter_shapes()),'Exact515/24 state census')

    def state_sha(state):
        need(isinstance(state,dict)and all(isinstance(v,torch.Tensor)and v.dtype==torch.float32
             and v.device.type=='cpu'and not v.requires_grad and v.grad is None
             and torch.isfinite(v).all().item()for v in state.values()),'Detached finite CPU FP32 state')
        return numpy_state_sha({n:v.detach().cpu().contiguous().numpy()for n,v in state.items()})

    need(state_sha(r2)==R2_STATE and state_sha(u96)==U96_STATE
         and all(tuple(v.shape)==adapter_shapes()[n]for n,v in u96.items()),'Exact incumbent canonical values/schema')

    class ResidualSkipAdapter(torch.nn.Module):
        def __init__(self,channels):
            super().__init__();options={'device':'cpu','dtype':torch.float32}
            self.down=torch.nn.Conv2d(channels,32,1,bias=True,**options)
            self.depthwise=torch.nn.Conv2d(32,32,3,padding=1,groups=32,bias=True,**options)
            self.up=torch.nn.Conv2d(32,channels,1,bias=True,**options)
            self.activation=torch.nn.SiLU(inplace=False)
        def forward(self,observed):
            return observed+self.up(self.activation(self.depthwise(self.activation(self.down(observed)))))

    class SpatialHead(torch.nn.Module):
        def __init__(self,channels):
            super().__init__();options={'device':'cpu','dtype':torch.float32}
            self.encoder=torch.nn.Conv2d(channels,WIDTH,1,bias=True,**options)
            self.decoder=torch.nn.Conv2d(channels,WIDTH,1,bias=False,**options)
            self.fused=torch.nn.Conv2d(channels,WIDTH,1,bias=False,**options)
            self.coarse=torch.nn.Conv2d(256,WIDTH,1,bias=False,**options)
            self.spatial=torch.nn.Conv2d(WIDTH,WIDTH,3,padding=1,bias=True,**options)
            self.output=torch.nn.Conv2d(WIDTH,channels,1,bias=True,**options)
            self.activation=torch.nn.GELU()
        def forward(self,used,decoder,fused,coarse):
            context=self.coarse(coarse.detach())
            context=F.interpolate(context,size=used.shape[-2:],mode='bilinear',align_corners=False)
            mixed=(self.encoder(used.detach())+self.decoder(decoder)+self.fused(fused)+context)/2.
            logits=self.output(self.activation(self.spatial(self.activation(mixed))))
            return fused+torch.tanh(logits)*(fused-decoder)

    class FrozenCPUPostFusionRestorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # No supplied base and no assign=True: own new tensor storages.
            self.base=CodeFormer(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=list(SITES))
            self.base.load_state_dict(r2,strict=True)
            self.adapters=torch.nn.ModuleDict({site:ResidualSkipAdapter(CHANNELS[site])for site in SITES})
            self.heads=torch.nn.ModuleDict({site:SpatialHead(CHANNELS[site])for site in SITES})
            self.adapters.load_state_dict({n.removeprefix('adapters.'):v for n,v in u96.items()},strict=True)
            self.heads.load_state_dict({n.removeprefix('heads.'):torch.from_numpy(v.copy())for n,v in arrays.items()},strict=True)
            self._forward_lock=threading.Lock();self.last_forward_sites=();self.last_base_calls=0
            self.eval();self.requires_grad_(False)
            setattr(self.base,_OWNER_ATTRIBUTE,weakref.ref(self))

        def train(self,mode=True):
            need(mode is False,'Training mode is unavailable in this frozen inference adapter')
            return super().train(False)

        def validate_invariants(self):
            named=dict(self.named_parameters());state=self.state_dict()
            need(len(named)==len(state)==575 and len(self.base.state_dict())==515
                 and self.base.connect_list==list(SITES)and set(self.base.fuse_convs_dict)==set(SITES)
                 and self.base.fuse_generator_block=={'16':6,'32':9,'64':12,'128':15,'256':18,'512':21},
                 'Exact575 tensors/four-site architecture')
            need(all(not m.training for m in self.modules())
                 and all(v.device.type=='cpu'and v.dtype==torch.float32 and not v.requires_grad
                         and v.grad is None and torch.isfinite(v).all().item()for v in named.values()),
                 'All575 tensors must stay finite frozen/eval CPU FP32 with grad=None')
            adapters={'adapters.'+n:v for n,v in self.adapters.state_dict().items()}
            heads={'heads.'+n:v for n,v in self.heads.state_dict().items()}
            need(set(adapters)==set(adapter_shapes())and set(heads)==set(head_shapes())
                 and all(tuple(v.shape)==adapter_shapes()[n]for n,v in adapters.items())
                 and all(tuple(v.shape)==head_shapes()[n]for n,v in heads.items()),'Exact24/36 tensor schema')
            need(state_sha(self.base.state_dict())==R2_STATE and state_sha(adapters)==U96_STATE
                 and state_sha(heads)==HEAD_STATE,'Frozen canonical values changed; refuse mutation')
            owner=getattr(self.base,_OWNER_ATTRIBUTE,None)
            need(isinstance(owner,weakref.ReferenceType)and owner()is self,'Exclusive owned new base required')
            return {'pass':True,'frozen_tensors':575,'head_canonical_sha256':HEAD_STATE,
                    'no_shared_caller_base':True,'runtime_integrity_is_not_model_qualification':True}

        def forward(self,sample,*,w=1.,detach_16=True,adain=True):
            need(not torch.is_grad_enabled()and not torch.is_inference_mode_enabled(),
                 'Caller torch.no_grad required; inference_mode/gradient runtime refused')
            need(w==1. and detach_16 is True and adain is True,'Fixed w1/detach16/AdaIN-on policy')
            need(isinstance(sample,torch.Tensor)and tuple(sample.shape)==(1,3,512,512)
                 and sample.dtype==torch.float32 and sample.device.type=='cpu'and not sample.requires_grad
                 and torch.isfinite(sample).all().item()and sample.min().item()>=-1. and sample.max().item()<=1.,
                 'Finite singleton CPU FP32 prepared512[-1,1] input')
            need(self._forward_lock.acquire(blocking=False),'Nested/concurrent frozen forward refused')
            handles=[];pending={};completed=[];coarse=None;base_calls=0
            thread_id=threading.get_ident();self.last_forward_sites=();self.last_base_calls=0
            def guard_thread():
                need(threading.get_ident()==thread_id,'Foreign-thread/nested base/SFT call refused')
            def base_pre_hook(module,args):
                nonlocal base_calls
                guard_thread()
                need(module is self.base and base_calls==0 and len(args)==1 and args[0]is sample,
                     'Exactly one explicit base call required')
                base_calls+=1
            def pre_hook(site):
                def adapt_and_capture(module,args):
                    nonlocal coarse
                    guard_thread()
                    need(module is self.base.fuse_convs_dict[site]and base_calls==1 and len(args)==3
                         and args[2]==1. and len(completed)<4 and site==SITES[len(completed)]and not pending,
                         'One ordered U96 pre-SFT adaptation')
                    encoder,decoder,_=args;shape=(1,CHANNELS[site],int(site),int(site))
                    need(encoder.shape==decoder.shape==shape and encoder.dtype==decoder.dtype==torch.float32
                         and encoder.device.type==decoder.device.type=='cpu'and not encoder.requires_grad
                         and not decoder.requires_grad and torch.isfinite(encoder).all().item()
                         and torch.isfinite(decoder).all().item(),'Finite no-grad CPU SFT inputs')
                    used=self.adapters[site](encoder.detach())
                    need(used.shape==shape and used.dtype==torch.float32 and used.device.type=='cpu'
                         and not used.requires_grad and torch.isfinite(used).all().item(),'Finite frozen U96 feature')
                    if site=='32':
                        need(coarse is None,'Capture used coarse32 exactly once');coarse=used.detach()
                    need(coarse is not None and coarse.shape==(1,256,32,32),'Used coarse32 required')
                    pending[site]=(used,decoder);return used,decoder,args[2]
                return adapt_and_capture
            def post_hook(site):
                def correct(module,args,fused):
                    guard_thread()
                    need(module is self.base.fuse_convs_dict[site]and set(pending)=={site}
                         and len(completed)<4 and site==SITES[len(completed)],'Ordered post-SFT capture')
                    used,decoder=pending.pop(site)
                    need(len(args)==3 and args[0]is used and args[1]is decoder and args[2]==1.
                         and isinstance(fused,torch.Tensor)and fused.shape==used.shape
                         and fused.dtype==torch.float32 and fused.device.type=='cpu'
                         and not fused.requires_grad and torch.isfinite(fused).all().item(),'Finite stock fused result')
                    corrected=self.heads[site](used,decoder,fused,coarse)
                    need(corrected.shape==fused.shape and corrected.dtype==torch.float32
                         and corrected.device.type=='cpu'and not corrected.requires_grad
                         and torch.isfinite(corrected).all().item(),'Finite same-geometry corrected result')
                    completed.append(site);return corrected
                return correct
            try:
                self.validate_invariants()
                need(all(not m._forward_pre_hooks and not m._forward_hooks for m in self.modules()),'Clean owned hook namespace')
                handles.append(self.base.register_forward_pre_hook(base_pre_hook))
                for site in SITES:
                    module=self.base.fuse_convs_dict[site]
                    handles.append(module.register_forward_pre_hook(pre_hook(site)))
                    handles.append(module.register_forward_hook(post_hook(site)))
                with torch.autocast(device_type='cpu',enabled=False):
                    result=self.base(sample,w=w,detach_16=detach_16,adain=adain)
                need(base_calls==1 and tuple(completed)==SITES and not pending,'One complete ordered four-site call')
                need(isinstance(result,tuple)and len(result)==3,'Stock native/logits/LQ tuple')
                for value,shape in zip(result,((1,3,512,512),(1,256,1024),(1,256,16,16))):
                    need(isinstance(value,torch.Tensor)and tuple(value.shape)==shape and value.dtype==torch.float32
                         and value.device.type=='cpu'and not value.requires_grad and torch.isfinite(value).all().item(),
                         'Finite stock output schema')
                self.validate_invariants()
                return result
            finally:
                try:
                    for handle in reversed(handles):handle.remove()
                    pending.clear();coarse=None
                    self.last_forward_sites=tuple(completed);self.last_base_calls=base_calls
                    need(all(not m._forward_pre_hooks and not m._forward_hooks for m in self.modules()),'Hooks must clean up on every exit')
                finally:self._forward_lock.release()

    # Conv initialization only seeds CPU and restores caller RNG afterward.
    # No CUDA discovery/random state or device transfer is performed.
    with torch.no_grad(),torch.device('cpu'),torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261021)
        net=FrozenCPUPostFusionRestorer()
    net.validate_invariants()
    # Fresh load_state_dict copies own storages instead of reusing mapped inputs.
    need(all(v.untyped_storage().data_ptr()!=r2[n].untyped_storage().data_ptr()
             for n,v in net.base.state_dict().items()),'New base may not share mapped artifact storages')
    for path,expected in zip(paths,(R2_FILE,U96_FILE,HEAD_FILE)):
        need(file_pin(path)==expected,'Inference artifact changed during loading')
    net.provenance={**source_contract(),'inference_source_sha256':inference_source_sha256,
        'all575_frozen':True,'source_terminal_update':64,'CPU_runtime_PARITY_pending':True,
        'broader_DEV_FINAL_user_acceptance_pending':True,'main_app_default_changed':False}
    return net

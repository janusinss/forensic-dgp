"""Inert source-only U96 post-fusion prototype; no Torch import at module load.

The explicit factory requires the complete exact R2 base plus U96 adapter state.
It constructs an UNTRAINED candidate only in a future guarded manual runtime.
There is no optimizer, data loader, automatic execution or training entry point.
Local source checks do not establish numerical parity, gradients or usefulness.
"""
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
import threading
import weakref


SITES = ('32', '64', '128', '256')
CHANNELS = {'32':256, '64':256, '128':128, '256':128}
WIDTH = 32
INITIALIZATION_SEED = 20261021
ROOT = Path(__file__).resolve().parents[2]
R2_FILE_SHA256 = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
R2_CANONICAL_SHA256 = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_FILE_SHA256 = '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'
U96_CANONICAL_SHA256 = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
BASE_TENSORS = 515
ADAPTER_TENSORS = 24
ADAPTER_PARAMETERS = 51328
FROZEN_TENSORS = 539
HEAD_TENSORS = 36
HEAD_PARAMETERS = 168960
TOTAL_TENSORS = 575
ARCHITECTURE_SOURCES = {
    'third_party.codeformer.codeformer_arch':{
        'path':'third_party/codeformer/codeformer_arch.py',
        'sha256':'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb'},
    'third_party.codeformer.vqgan_arch':{
        'path':'third_party/codeformer/vqgan_arch.py',
        'sha256':'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf'},
    'third_party.codeformer.registry':{
        'path':'third_party/codeformer/registry.py',
        'sha256':'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'},
}
_OWNERS = weakref.WeakKeyDictionary()
_OWNERS_LOCK = threading.Lock()
_OWNER_ATTRIBUTE = '_forensic_codeformer_exclusive_owner'


def require(condition,message):
    if not condition:raise ValueError(message)


def expected_adapter_shapes():
    """Exact retained inference schema; this arithmetic imports no Torch."""
    result={}
    for site,channels in CHANNELS.items():
        for name,shape in {
            'down.weight':(32,channels,1,1),'down.bias':(32,),
            'depthwise.weight':(32,1,3,3),'depthwise.bias':(32,),
            'up.weight':(channels,32,1,1),'up.bias':(channels,),
        }.items():result['adapters.'+site+'.'+name]=shape
    require(len(result)==ADAPTER_TENSORS and
            sum(math.prod(shape) for shape in result.values())==ADAPTER_PARAMETERS,
            'Exact24-tensor/51328-parameter U96 schema differs')
    return result


def expected_head_shapes():
    """Prospective exact36-tensor spatial correction schema, without Torch."""
    result={}
    for site in SITES:
        channels=CHANNELS[site]
        shapes={
            'encoder.weight':(WIDTH,channels,1,1),'encoder.bias':(WIDTH,),
            'decoder.weight':(WIDTH,channels,1,1),'fused.weight':(WIDTH,channels,1,1),
            'coarse.weight':(WIDTH,256,1,1),'spatial.weight':(WIDTH,WIDTH,3,3),
            'spatial.bias':(WIDTH,),'output.weight':(channels,WIDTH,1,1),
            'output.bias':(channels,),
        }
        require(sum(math.prod(shape) for shape in shapes.values())==129*channels+17472,
                'Per-head prospective parameter arithmetic differs')
        for name,shape in shapes.items():result['heads.'+site+'.'+name]=shape
    require(len(result)==HEAD_TENSORS and
            sum(math.prod(shape) for shape in result.values())==HEAD_PARAMETERS,
            'Exact36-tensor/168960-parameter head schema differs')
    return result


def architecture_contract():
    """Static contract only; no runtime/model-quality certification."""
    expected_adapter_shapes();expected_head_shapes()
    return {
        'format':'CodeFormer-complete-U96-postfusion-prototype-v1',
        'status':'INERT_UNRUN_NOT_TRAINED_NOT_A_SEALED_PACKET',
        'initializer':{'R2_file_sha256':R2_FILE_SHA256,'R2_canonical515_sha256':R2_CANONICAL_SHA256,
                       'U96_file_sha256':U96_FILE_SHA256,'U96_canonical24_sha256':U96_CANONICAL_SHA256},
        'base_tensors':BASE_TENSORS,'U96_adapter_tensors':ADAPTER_TENSORS,
        'U96_adapter_parameters':ADAPTER_PARAMETERS,'frozen_tensors':FROZEN_TENSORS,
        'head_tensors':HEAD_TENSORS,'head_parameters':HEAD_PARAMETERS,
        'total_tensors':TOTAL_TENSORS,'sites':list(SITES),'width':WIDTH,
        'U96_pre_SFT':'e_used=e+up(SiLU(depthwise(SiLU(down(e))))) with exact frozen24 U96 tensors',
        'stock':'z=stock_SFT(e_used,d,w=1)',
        'head':'h=output(GELU(spatial(GELU((E(e_used)+D(d)+Z(z)+resize(C(coarse32_used)))/2))))',
        'post_SFT':'z_new=z+tanh(h)*(z-d)',
        'multiplier':'q=1+tanh(h), mathematically0<q<2; FP32 may round endpoints',
        'zero_head':'output.weight/output.bias exactly0; numerical U96 parity is still UNMEASURED',
        'frozen_observed_encoder_and_coarse_detached':True,
        'decoder_and_fused_graph_not_detached':True,
        'frozen_R2_and_U96_eval':True,'postfusion_heads_trainable_only':True,
        'same_four_U96_pre_hooks_then_postfusion_heads_in_one_owned_forward':True,
        'factory_CPU_RNG_preserved':True,
        'cannot_repair_frozen_wrong_code_predictions':True,
        'componentwise_correction_zero_where_stock_SFT_residual_is_zero':True,
        'quality_or_parity_or_gradient_or_training_verified':False,
        'quality_gates_changed':False,'manual_finite_L4_runtime_required_for_actual_training':True,
    }


def build_restorer(r2_base,u96_adapter_state,*,r2_artifact,u96_artifact):
    """Explicit factory for an UNTRAINED candidate; never called by local checks.

    Future manual L4 code must guard reservations, resource limits, exact initial
    U96 RAW/PNG/token parity, actual gradients, hook failures and quality stops.
    Artifact hashes verify the declared initializer files; canonical hashes also
    bind the actual caller-supplied tensors, independently of path declarations.
    """
    import torch
    from torch.nn import functional as F

    def file_digest(path):
        path=Path(path).absolute()
        require(path.is_file() and not path.is_symlink(),'Regular explicit initializer file required')
        info=path.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&1024,
                'Reparse initializer file refused')
        require(0<info.st_size<1024**3,'Bounded initializer artifact required')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                chunk=stream.read(1024*1024)
                if not chunk:break
                digest.update(chunk)
        return digest.hexdigest()

    require(isinstance(r2_base,torch.nn.Module) and
            type(r2_base).__module__=='third_party.codeformer.codeformer_arch' and
            type(r2_base).__name__=='CodeFormer','Exact supplied raw R2 CodeFormer class required')
    for name,expected in ARCHITECTURE_SOURCES.items():
        module=sys.modules.get(name)
        require(module is not None and getattr(module,'__file__',None),
                'Retained architecture module must already be loaded: '+name)
        path=Path(module.__file__).absolute()
        require(path.resolve()==(ROOT/expected['path']).resolve() and
                file_digest(path)==expected['sha256'],
                'Exact retained module origin and source bytes required: '+name)
    require(file_digest(r2_artifact)==R2_FILE_SHA256 and
            file_digest(u96_artifact)==U96_FILE_SHA256,'Exact separate R2/U96 initializer artifacts required')
    require(r2_base.connect_list==list(SITES) and set(r2_base.fuse_convs_dict)==set(SITES) and
            r2_base.fuse_generator_block=={'16':6,'32':9,'64':12,'128':15,'256':18,'512':21},
            'Exact retained four-site SFT CodeFormer architecture required')
    require(all(not module._forward_pre_hooks and not module._forward_hooks
                for module in r2_base.modules()),'Base must have a clean hook namespace')
    initial=r2_base.state_dict()
    require(len(initial)==len(dict(r2_base.named_parameters()))==BASE_TENSORS,
            'Exactly515 supplied R2 tensors required')
    device=next(r2_base.parameters()).device
    require(device.type in ('cpu','cuda') and all(
        isinstance(value,torch.Tensor) and value.dtype==torch.float32 and value.device==device and
        torch.isfinite(value).all().item() for value in initial.values()),
        'Finite colocated FP32 R2 tensors required')
    require(all(value.grad is None for value in r2_base.parameters()),'No retained R2 gradient fields allowed')
    adapter_shapes=expected_adapter_shapes()
    require(type(u96_adapter_state) is dict and set(u96_adapter_state)==set(adapter_shapes),
            'Exact explicit24-key U96 adapter-state dictionary required')
    require(all(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and
                tuple(value.shape)==adapter_shapes[name] and value.device.type in ('cpu','cuda') and
                not value.requires_grad and value.grad is None and torch.isfinite(value).all().item()
                for name,value in u96_adapter_state.items()),'Finite detached FP32 U96 state schema required')

    def state_digest(state):
        digest=hashlib.sha256()
        for name,value in sorted(state.items()):
            value=value.detach().cpu().contiguous()
            header=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode('utf-8')
            digest.update(len(header).to_bytes(8,'big'));digest.update(header)
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    require(state_digest(initial)==R2_CANONICAL_SHA256,'Exact canonical515 R2 content required')
    require(state_digest(u96_adapter_state)==U96_CANONICAL_SHA256,
            'Exact canonical24 U96 content required; S96 or partial state is not accepted')

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
            torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)

        def forward(self,used,decoder,fused,coarse):
            # Only observed frozen features detach. Preserve decoder/fused autograd.
            context=self.coarse(coarse.detach())
            context=F.interpolate(context,size=used.shape[-2:],mode='bilinear',align_corners=False)
            mixed=(self.encoder(used.detach())+self.decoder(decoder)+self.fused(fused)+context)/2.
            logits=self.output(self.activation(self.spatial(self.activation(mixed))))
            return fused+torch.tanh(logits)*(fused-decoder)

    class U96PostFusionRestorer(torch.nn.Module):
        def __init__(self,frozen_base):
            super().__init__();self.base=frozen_base
            self.adapters=torch.nn.ModuleDict({site:ResidualSkipAdapter(CHANNELS[site]) for site in SITES})
            self.heads=torch.nn.ModuleDict({site:SpatialHead(CHANNELS[site]) for site in SITES})
            self.adapters.load_state_dict({name.removeprefix('adapters.'):value
                                           for name,value in u96_adapter_state.items()},strict=True)
            self.adapters.to(device=device,dtype=torch.float32).eval().requires_grad_(False)
            self.heads.to(device=device,dtype=torch.float32)
            self.base.eval().requires_grad_(False)
            self._forward_lock=threading.Lock();self.last_forward_sites=();self.last_base_calls=0

        def train(self,mode=True):
            super().train(mode);self.base.eval();self.adapters.eval();return self

        def validate_invariants(self,*,check_frozen_digests=False):
            require(type(check_frozen_digests) is bool,'Explicit frozen digest-check boolean required')
            named=dict(self.named_parameters());head_schema=expected_head_shapes()
            heads={name:value for name,value in named.items() if name.startswith('heads.')}
            adapters={name:value for name,value in named.items() if name.startswith('adapters.')}
            require(len(named)==len(self.state_dict())==TOTAL_TENSORS and
                    len(self.base.state_dict())==BASE_TENSORS and set(heads)==set(head_schema) and
                    set(adapters)==set(adapter_shapes) and
                    sum(v.numel() for v in heads.values())==HEAD_PARAMETERS and
                    sum(v.numel() for v in adapters.values())==ADAPTER_PARAMETERS,
                    'Actual575 total/539 frozen/36 head/168960 trainable census differs')
            require(all(tuple(heads[name].shape)==shape and heads[name].requires_grad and
                        heads[name].dtype==torch.float32 and heads[name].device==device and
                        torch.isfinite(heads[name]).all().item() for name,shape in head_schema.items()),
                    'Exact trainable FP32 post-fusion head schema required')
            frozen=list(self.base.parameters())+list(self.adapters.parameters())
            require(len(frozen)==FROZEN_TENSORS and all(not value.requires_grad and value.grad is None and
                        value.dtype==torch.float32 and value.device==device for value in frozen) and
                    all(not module.training for module in self.base.modules()) and
                    all(not module.training for module in self.adapters.modules()) and
                    all(tuple(adapters[name].shape)==shape for name,shape in adapter_shapes.items()),
                    'All539 R2/U96 tensors must stay frozen/eval FP32 with grad=None')
            digests=None
            if check_frozen_digests:
                digests={'R2':state_digest(self.base.state_dict()),
                         'U96':state_digest({'adapters.'+name:value
                                             for name,value in self.adapters.state_dict().items()})}
                require(digests=={'R2':R2_CANONICAL_SHA256,'U96':U96_CANONICAL_SHA256},
                        'Frozen initializer changed; preserve failure, never rebind silently')
            return {'pass':True,'frozen_tensors':FROZEN_TENSORS,'head_tensors':HEAD_TENSORS,
                    'total_tensors':TOTAL_TENSORS,'head_parameters':HEAD_PARAMETERS,
                    'frozen_digests_checked':check_frozen_digests,'digests':digests,
                    'runtime_integrity_is_not_parity_gradient_or_quality':True}

        def forward(self,sample,*,w=1.,detach_16=True,adain=True):
            require(w==1. and detach_16 is True and adain is True,
                    'Fixed w1/detach16/AdaIN-on policy required')
            require(isinstance(sample,torch.Tensor) and sample.shape==(1,3,512,512) and
                    sample.dtype==torch.float32 and sample.device==device and not sample.requires_grad and
                    torch.isfinite(sample).all().item() and sample.min().item()>=-1. and sample.max().item()<=1.,
                    'Finite singleton FP32 prepared512[-1,1] input required')
            require(not torch.is_inference_mode_enabled(),
                    'Autograd-capable ordinary runtime required; inference_mode refused')
            self.validate_invariants(check_frozen_digests=False)
            require(self._forward_lock.acquire(blocking=False),'Nested/concurrent candidate forward refused')
            handles=[];pending={};completed=[];coarse=None;base_calls=0
            thread_id=threading.get_ident();self.last_forward_sites=();self.last_base_calls=0

            def guard_thread():
                require(threading.get_ident()==thread_id,'Concurrent/foreign-thread base or SFT call refused')

            def base_pre_hook(module,args):
                nonlocal base_calls
                guard_thread()
                require(module is self.base and base_calls==0 and len(args)==1 and args[0] is sample,
                        'One explicit nonnested base call required')
                base_calls+=1

            def pre_hook(site):
                def adapt_and_capture(module,args):
                    nonlocal coarse
                    guard_thread()
                    require(module is self.base.fuse_convs_dict[site] and base_calls==1 and
                            len(args)==3 and args[2]==1. and len(completed)<len(SITES) and
                            site==SITES[len(completed)] and not pending,
                            'Exactly one ordered U96 pre-SFT adaptation required')
                    encoder,decoder,_=args;shape=(1,CHANNELS[site],int(site),int(site))
                    require(encoder.shape==decoder.shape==shape and
                            encoder.dtype==decoder.dtype==torch.float32 and
                            encoder.device==decoder.device==device and not encoder.requires_grad and
                            torch.isfinite(encoder).all().item() and torch.isfinite(decoder).all().item(),
                            'Exact detached encoder and ordinary connected decoder required')
                    used=self.adapters[site](encoder.detach())
                    require(used.shape==shape and used.dtype==torch.float32 and used.device==device and
                            not used.requires_grad and torch.isfinite(used).all().item(),
                            'Finite exact-schema frozen U96 encoder-feature result required')
                    if site=='32':
                        require(coarse is None,'Capture used coarse32 exactly once')
                        coarse=used.detach()
                    require(coarse is not None and coarse.shape==(1,256,32,32),
                            'First-site used U96 coarse32 required')
                    pending[site]=(used,decoder)
                    return used,decoder,args[2]
                return adapt_and_capture

            def post_hook(site):
                def correct(module,args,fused):
                    guard_thread()
                    require(module is self.base.fuse_convs_dict[site] and set(pending)=={site} and
                            len(completed)<len(SITES) and site==SITES[len(completed)],
                            'Matching ordered U96 SFT capture required')
                    used,decoder=pending.pop(site)
                    require(len(args)==3 and args[0] is used and args[1] is decoder and args[2]==1. and
                            isinstance(fused,torch.Tensor) and fused.shape==used.shape and
                            fused.dtype==torch.float32 and fused.device==device and torch.isfinite(fused).all().item(),
                            'Finite stock fused result from the captured U96 SFT arguments required')
                    corrected=self.heads[site](used,decoder,fused,coarse)
                    require(corrected.shape==fused.shape and corrected.dtype==torch.float32 and
                            corrected.device==device and torch.isfinite(corrected).all().item(),
                            'Finite unchanged-geometry post-fusion correction required')
                    completed.append(site);return corrected
                return correct

            try:
                require(all(not module._forward_pre_hooks and not module._forward_hooks
                            for module in self.modules()),'Clean candidate hook namespace required')
                handles.append(self.base.register_forward_pre_hook(base_pre_hook))
                for site in SITES:
                    module=self.base.fuse_convs_dict[site]
                    handles.append(module.register_forward_pre_hook(pre_hook(site)))
                    handles.append(module.register_forward_hook(post_hook(site)))
                # Do not wrap the base in no_grad: downstream decoder and fused
                # paths must remain connected to earlier post-fusion heads.
                with torch.autocast(device_type=device.type,enabled=False):
                    result=self.base(sample,w=w,detach_16=detach_16,adain=adain)
                require(base_calls==1 and tuple(completed)==SITES and not pending,
                        'All four U96 adaptations/corrections must execute in one complete base call')
                require(isinstance(result,tuple) and len(result)==3 and
                        result[0].shape==(1,3,512,512) and result[0].dtype==torch.float32 and
                        result[0].device==device and
                        torch.isfinite(result[0]).all().item(),
                        'Finite native512 plus unchanged stock logits/LQ return structure required')
                return result
            finally:
                try:
                    for handle in reversed(handles):handle.remove()
                    pending.clear();coarse=None
                    self.last_forward_sites=tuple(completed);self.last_base_calls=base_calls
                finally:self._forward_lock.release()

    with _OWNERS_LOCK:
        owner=_OWNERS.get(r2_base)
        shared_owner=getattr(r2_base,_OWNER_ATTRIBUTE,None)
        require((owner is None or owner() is None) and
                (shared_owner is None or isinstance(shared_owner,weakref.ReferenceType) and shared_owner() is None),
                'R2 base already has a live exclusive wrapper owner')
        # Conv construction is CPU-only. Restore CPU RNG exactly; moving the
        # constructed tensors consumes no CUDA random draws.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(INITIALIZATION_SEED)
            candidate=U96PostFusionRestorer(r2_base)
        candidate.validate_invariants(check_frozen_digests=True)
        require(all(torch.count_nonzero(head.output.weight).item()==0 and
                    torch.count_nonzero(head.output.bias).item()==0 for head in candidate.heads.values()),
                'Exact zero final projections required; parity still needs VM measurement')
        _OWNERS[r2_base]=weakref.ref(candidate)
        setattr(r2_base,_OWNER_ATTRIBUTE,weakref.ref(candidate))
    return candidate

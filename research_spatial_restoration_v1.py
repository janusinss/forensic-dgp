"""Opt-in frozen CPU inference for independently reviewed spatial adapters.

Status, input refusal and page rendering never import Torch. The guarded manual
VM training helper is never imported or modified. No model is usable until an
exact inference-export manifest and all of its review files are bound below.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
import zipfile

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

import research_restoration_web as base
from research_filter_restoration_web_v1 import review_input, decode_crop, prepare_crop, delivered_views, read_receipt

ROOT=Path(__file__).resolve().parent
MODEL_DIR=ROOT/'outputs/codeformer_spatial_continuation_v1_models'
MANIFEST_NAME='model_exports_v1.json'
# Root binds the actual reviewed export after physical return/state/pixel and
# visual verification. A prepared path or completed training is not a binding.
MANIFEST_BINDING={'path':'outputs/codeformer_spatial_continuation_v1_models/model_exports_v1.json',
    'bytes':4445,'sha256':'f0b2f5b25c43b7702867833e28ed6a28fe88e6e994b5597698820f3950ad190b'}
SITES=('32','64','128','256')
CHANNELS={'32':256,'64':256,'128':128,'256':128}
BASE_SHA='a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
ARMS=('original','r2','spatial_u','spatial_s')
PREFERRED_ARM='spatial_u'
LIMIT=10*1024**2
QUALITY_STATUS=('U96 is the user-selected development restorer. R2 remains a permanent comparison and frozen base. '
    'Preservation limitations and independent final review remain unresolved.')
router=APIRouter(prefix='/restoration-spatial-test',tags=['Experimental spatial restoration'])


def _binding(item):
    if (type(item) is not dict or set(item)!={'path','bytes','sha256'} or
        type(item['path']) is not str or type(item['bytes']) is not int or item['bytes']<=0 or
        not re.fullmatch('[0-9a-f]{64}',str(item['sha256']))):
        raise ValueError('Exact physical export/review pin required.')
    relative=Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts or '\\' in item['path'] or ':' in item['path']:
        raise ValueError('Research proof must stay within this workspace.')
    path=ROOT/relative
    if path.stat().st_size!=item['bytes'] or base.sha(path)!=item['sha256']:
        raise ValueError('The bound experimental export or review changed.')
    return path


def manifest():
    if MANIFEST_BINDING is None:
        raise ValueError('The new model is awaiting completed return, state, pixel and visual review.')
    path=_binding(MANIFEST_BINDING)
    if path!=MODEL_DIR/MANIFEST_NAME:
        raise ValueError('Exact experimental model manifest path required.')
    value=read_receipt(path,2*1024**2)
    if (type(value) is not dict or value.get('complete') is not True or
        value.get('format')!='codeformer-spatial-adapters-research-export-v1' or
        value.get('model_qualified') is not False or value.get('app_default_changed') is not False or
        value.get('neural_gradient_optimizer_VM_calls')!=0 or
        value.get('original515_base_values_unchanged') is not True or
        value.get('adapter_tensors')!=24 or value.get('adapter_scalar_parameters')!=51328 or
        value.get('base_tensor_sha256')!=BASE_SHA or
        set(value.get('arms',{}))!={'spatial_u','spatial_s'} or
        set(value.get('review_pins',{}))!={'import','state','pixels','visual'}):
        raise ValueError('Experimental inference-export contract differs from the reviewed state.')
    for key,item in value['review_pins'].items():
        proof=read_receipt(_binding(item),8*1024**2)
        if proof.get('complete') is not True:
            raise ValueError('Completed independent review required: '+key)
        if key in ('state','pixels') and not (proof.get('audit_pass') is True or proof.get('pass') is True):
            raise ValueError('Saved-state and pixel integrity audits must pass.')
        if key=='visual' and proof.get('root_personal_review_complete') is not True:
            raise ValueError('Root review of every planned actual visual sheet is required.')
    if not re.fullmatch('[0-9a-f]{64}',str(value.get('protocol_sha256',''))):
        raise ValueError('Exact source protocol fingerprint required.')
    for arm,row in value['arms'].items():
        if (row.get('source_arm')!=('U' if arm=='spatial_u' else 'S') or
            type(row.get('absolute_updates')) is not int or not 24<=row['absolute_updates']<=96 or
            row.get('additional_continuation_updates')!=row['absolute_updates']-24 or
            type(row.get('label')) is not str or not row['label'] or
            row.get('model_qualified') is not False or
            row.get('terminal_status') not in ('complete','guard_stop') or
            type(row.get('quality_status')) is not str or not row['quality_status'] or
            any(not re.fullmatch('[0-9a-f]{64}',str(row.get(k,''))) for k in
                ('adapter_tensor_sha256','source_checkpoint_sha256'))):
            raise ValueError('Exact committed arm/checkpoint/quality lineage required.')
        item=row.get('export')
        if type(item) is not dict or item.get('bytes',0)>1024**2:
            raise ValueError('Bounded adapter-only inference artifact required.')
        artifact=_binding(item)
        if not artifact.is_relative_to(MODEL_DIR):
            raise ValueError('Adapter export must remain in its bound model directory.')
    if type(value.get('quality_status')) is not str or not value['quality_status']:
        raise ValueError('Explicit experimental quality status required.')
    return value


def adapter_shapes():
    return {'adapters.'+site+'.'+name:shape for site,c in CHANNELS.items()
        for name,shape in {'down.weight':(32,c,1,1),'down.bias':(32,),
            'depthwise.weight':(32,1,3,3),'depthwise.bias':(32,),
            'up.weight':(c,32,1,1),'up.bias':(c,)}.items()}


def inference_class(torch):
    """Inference-only copy of the exact four-adapter architecture.

    The two candidates share an immutable frozen R2 base. The engine lock makes
    temporary SFT hooks serial. No optimizer or VM training helper is involved.
    """
    class ResidualSkipAdapter(torch.nn.Module):
        def __init__(self,c):
            super().__init__()
            self.down=torch.nn.Conv2d(c,32,1,bias=True)
            self.depthwise=torch.nn.Conv2d(32,32,3,padding=1,groups=32,bias=True)
            self.up=torch.nn.Conv2d(32,c,1,bias=True)
            self.activation=torch.nn.SiLU(inplace=False)
        def forward(self,x):
            return x+self.up(self.activation(self.depthwise(self.activation(self.down(x)))))
    class FrozenSpatialRestorer(torch.nn.Module):
        def __init__(self,frozen_base):
            super().__init__();self.base=frozen_base
            self.adapters=torch.nn.ModuleDict({s:ResidualSkipAdapter(CHANNELS[s]) for s in SITES})
            self.last_forward_sites=()
        def train(self,mode=True):return super().train(False)
        def forward(self,sample,*,w=1.,detach_16=True,adain=True):
            if (torch.is_grad_enabled() or w!=1. or detach_16 is not True or adain is not True or
                sample.shape!=(1,3,512,512) or sample.dtype!=torch.float32 or sample.device.type!='cpu' or
                any(v.requires_grad or v.grad is not None for v in self.parameters())):
                raise ValueError('Frozen CPU FP32 singleton w1/AdaIN inference required.')
            if any(m._forward_hooks or m._forward_pre_hooks for m in self.base.modules()):
                raise ValueError('Clean stock hook namespace required.')
            sites=[];handles=[];self.last_forward_sites=()
            def hook(site):
                def apply(module,args):
                    if (len(args)!=3 or args[2]!=1. or site in sites or
                        args[0].shape!=(1,CHANNELS[site],int(site),int(site)) or args[0].requires_grad or
                        args[0].dtype!=torch.float32 or args[0].device.type!='cpu' or not torch.isfinite(args[0]).all().item()):
                        raise ValueError('Exact detached observed encoder SFT argument required.')
                    used=self.adapters[site](args[0]);sites.append(site)
                    if used.shape!=args[0].shape or not torch.isfinite(used).all().item():
                        raise ValueError('Invalid spatial adapter output.')
                    return used,args[1],args[2]
                return apply
            try:
                for site in SITES:handles.append(self.base.fuse_convs_dict[site].register_forward_pre_hook(hook(site)))
                result=self.base(sample,w=w,detach_16=detach_16,adain=adain)
            finally:
                for h in handles:h.remove()
                self.last_forward_sites=tuple(sites)
            if self.last_forward_sites!=SITES:raise ValueError('All four spatial adapter sites must execute in order.')
            return result
    return FrozenSpatialRestorer


class Comparison(base.Comparison):
    def __init__(self):
        super().__init__();self.counter_lock=threading.Lock()
        self.counters={'neural_forward_attempts':0,'neural_forward_returns':0,'neural_forward_successes':0,
            'operator_review_refusals':0,'completed_comparisons':0}
        self.arm_counters={a:{'attempts':0,'returns':0,'successes':0} for a in ARMS}
    def counted(self,name,arm=None):
        with self.counter_lock:
            self.counters[name]+=1
            if arm:self.arm_counters[arm][name.removeprefix('neural_forward_')]+=1
    def measured_status(self):
        with self.counter_lock:return {**self.counters,'per_arm':{a:dict(v) for a,v in self.arm_counters.items()}}
    def load(self,arm,torch):
        if arm in ('original','r2'):return super().load(arm,torch)
        reviewed=manifest();row=reviewed['arms'][arm];path=_binding(row['export'])
        if arm in self.models:
            if self.pins[arm]['weights_sha256']!=row['export']['sha256']:
                raise ValueError('Model changed after loading. Restart this experimental server.')
            return self.models[arm],self.pins[arm]
        frozen,r2_metadata=super().load('r2',torch)
        if base.state_sha(frozen.state_dict(),torch)!=BASE_SHA:
            raise ValueError('Exact frozen R2 base required.')
        saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if type(saved) is not dict or set(saved)!={'format','adapter_state'} or saved['format']!='codeformer-spatial-adapter-inference-v1':
            raise ValueError('Exact optimizer/RNG-free adapter inference artifact required.')
        state=saved['adapter_state'];shapes=adapter_shapes()
        if (type(state) is not dict or set(state)!=set(shapes) or any(not isinstance(v,torch.Tensor) or
            v.dtype!=torch.float32 or v.device.type!='cpu' or tuple(v.shape)!=shapes[k] or
            not torch.isfinite(v).all().item() for k,v in state.items()) or
            base.state_sha(state,torch)!=row['adapter_tensor_sha256']):
            raise ValueError('Audited24 adapter tensor shape or value mismatch.')
        with torch.random.fork_rng(devices=[]):
            net=inference_class(torch)(frozen).float().eval().requires_grad_(False)
            net.adapters.load_state_dict({k.removeprefix('adapters.'):v for k,v in state.items()},strict=True)
        if len(net.state_dict())!=539 or any(v.requires_grad or v.grad is not None for v in net.parameters()) or any(m.training for m in net.modules()):
            raise ValueError('Exact frozen eval539-state model required.')
        if base.state_sha(net.base.state_dict(),torch)!=BASE_SHA or base.sha(path)!=row['export']['sha256']:
            raise ValueError('Frozen base or bound export changed during loading.')
        metadata={'arm':arm,'label':row['label'],'weights_sha256':row['export']['sha256'],
            'adapter_tensor_sha256':row['adapter_tensor_sha256'],'base_tensor_sha256':BASE_SHA,
            'source_checkpoint_sha256':row['source_checkpoint_sha256'],'source_training_updates':row['absolute_updates'],
            'additional_continuation_updates':row['additional_continuation_updates'],
            'source_terminal_status':row['terminal_status'],'protocol_sha256':reviewed['protocol_sha256'],
            'quality_status':row['quality_status'],'model_qualified':False,'main_app_default_changed':False,
            'architecture':'CodeFormer by S-Lab; our four learned encoder-feature residual adapters',
            'license':'S-Lab License1.0; third_party/codeformer/LICENSE','adapter_parameters':51328,
            'all515_R2_tensors_frozen':True,'shared_frozen_R2_base_serial_inference':True,
            'R2_weights_sha256':r2_metadata['weights_sha256'],'internal_resolution':512,'output_resolution':256,
            'fidelity':1.,'adain':True,'alignment':'none','display_processing':'none'}
        self.models[arm],self.pins[arm]=net,metadata
        return net,metadata
    def generate(self,contents,input_review,selection):
        try:review_input(input_review)
        except ValueError:
            self.counted('operator_review_refusals');raise
        if selection not in ('all',*ARMS):raise ValueError('Choose all four models or one listed model.')
        rgb=decode_crop(contents);canvas,observed,_,geometry=prepare_crop(rgb)
        arms=ARMS if selection=='all' else (selection,)
        reviewed=manifest() if any(a.startswith('spatial_') for a in arms) else None
        if 'original' in arms and base.sha(base.ORIGINAL,allow_benchmark_hardlinks=True)!=base.ORIGINAL_SHA:
            raise ValueError('Untouched benchmark fingerprint differs.')
        if any(a!='original' for a in arms):self.candidate_file()
        if not self.lock.acquire(blocking=False):raise RuntimeError('A comparison is already running. Wait for it to finish.')
        started=time.monotonic()
        try:
            import torch
            from torch.nn import functional as F
            torch.set_num_threads(4)
            x=torch.from_numpy((canvas.astype(np.float32)/np.float32(255)).copy()).permute(2,0,1)[None]
            outputs={}
            for arm in arms:
                net,provenance=self.load(arm,torch)
                with torch.no_grad():
                    sample=F.interpolate(x,(512,512),mode='bilinear',align_corners=False)*2-1
                    self.counted('neural_forward_attempts',arm)
                    native=net(sample,w=1.,detach_16=True,adain=True)[0]
                    self.counted('neural_forward_returns',arm)
                    if native.shape!=(1,3,512,512) or not torch.isfinite(native).all().item():raise ValueError('Invalid restoration output.')
                    if arm.startswith('spatial_') and (net.last_forward_sites!=SITES or
                        any(m._forward_hooks or m._forward_pre_hooks for m in net.base.modules()) or
                        base.state_sha(net.base.state_dict(),torch)!=BASE_SHA):
                        raise ValueError('Spatial hooks or shared frozen R2 state were not preserved.')
                    self.counted('neural_forward_successes',arm)
                    raw=F.interpolate(((native.float()+1)/2).clamp(0,1),(256,256),mode='bilinear',align_corners=False)[0].permute(1,2,0).contiguous().numpy().copy()
                raw,pixels=delivered_views(raw,canvas,observed);encoded=base.png(pixels)
                outputs[arm]={'raw':raw,'PNG':encoded,'metadata':dict(provenance),
                    'raw_rgb_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                    'PNG_RGB8_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                    'PNG_file_sha256':hashlib.sha256(encoded).hexdigest()}
            metadata={'policy':'research-CodeFormer-spatial-comparison-v1','input_review':input_review,
                'selection':selection,'geometry':geometry,'uploaded_bytes_sha256':hashlib.sha256(contents).hexdigest(),
                'device':'cpu','inference_only':True,'no_completion_or_covering_removal':True,
                'operator_input_review_is_not_a_validated_automatic_threshold':True,
                'quality_status':QUALITY_STATUS,'preferred_development_model':PREFERRED_ARM,
                'source_export_quality_status':reviewed['quality_status'] if reviewed else None,
                'model_qualified':False,'main_app_default_changed':False,'RAW_PNG_separate':True,
                'RAW_float_definition':'Native RGB clipped to[0,1], bilinear-resized to256; delivery floats before RGB8 floor quantization, not unclipped network activations.',
                'outside_observed_support_RAW_and_PNG_match_input_exact':True,
                'native_input_has_no_aligned_clean_reference_in_this_flow':True,'seconds':time.monotonic()-started,
                'outputs':{a:{k:v[k] for k in ('metadata','raw_rgb_sha256','PNG_RGB8_sha256','PNG_file_sha256')} for a,v in outputs.items()}}
            bundle=io.BytesIO()
            with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('original-upload.png',base.png(rgb));z.writestr('resize-control.png',base.png(canvas))
                z.writestr('observed-support.png',base.png(observed.astype(np.uint8)*255))
                for arm,row in outputs.items():
                    z.writestr(arm+'.png',row['PNG']);data=io.BytesIO();np.save(data,row['raw'],allow_pickle=False)
                    z.writestr(arm+'-raw.npy',data.getvalue())
                z.writestr('processing.json',json.dumps(metadata,indent=2,allow_nan=False)+'\n')
            self.counted('completed_comparisons')
            return {'original':base.data_url(base.png(canvas)),'outputs':{a:base.data_url(v['PNG']) for a,v in outputs.items()},
                'metadata':metadata,'bundle':base.data_url(bundle.getvalue(),'application/zip')}
        finally:self.lock.release()


engine=Comparison()


@router.get('',response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT/'templates/research_spatial_restoration_v1.html').read_text(encoding='utf-8'),headers={'Cache-Control':'no-store'})


@router.get('/status')
def status():
    result={'candidate_qualified':False,'main_app_default_changed':False,'manual_input_review_required':True,
        'review_pins_configured':MANIFEST_BINDING is not None,'route':'/restoration-spatial-test',
        'quality_status':QUALITY_STATUS,'preferred_development_model':PREFERRED_ARM,
        'development_preference_is_not_final_qualification':True,
        'models':{'original':{'label':'Untouched CodeFormer','available':True},
            'r2':{'label':'Our trained R2 · experimental','available':True}},**engine.measured_status()}
    try:
        reviewed=manifest();result['source_export_quality_status']=reviewed['quality_status']
        for arm,row in reviewed['arms'].items():
            result['models'][arm]={'label':row['label']+(' · preferred' if arm==PREFERRED_ARM else ''),
                'preferred':arm==PREFERRED_ARM,'available':True,'absolute_updates':row['absolute_updates'],
                'terminal_status':row['terminal_status'],'quality_status':row['quality_status']}
    except (ValueError,OSError) as error:
        result['binding_status']=str(error)
        for arm in ('spatial_u','spatial_s'):result['models'][arm]={'label':arm.upper()+' · awaiting verification','available':False}
    return result


@router.post('/compare')
async def compare(file:UploadFile=File(...),input_review:str=Form(''),selection:str=Form(PREFERRED_ARM)):
    try:return await run_in_threadpool(engine.generate,await file.read(LIMIT+1),input_review,selection)
    except ValueError as error:raise HTTPException(422,str(error)) from error
    except (FileNotFoundError,OSError) as error:raise HTTPException(503,'The verified experimental spatial export is unavailable.') from error
    except RuntimeError as error:raise HTTPException(409,str(error)) from error

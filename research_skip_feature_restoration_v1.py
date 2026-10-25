"""Opt-in local skip-feature comparison bound to the audited stopped55 export.

No VM runtime helper or historical training launcher is imported. Input refusal,
status and pages never load Torch. Model loading requires exact physical state,
pixel, visual and inference-export evidence. Existing app defaults stay.
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
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
import research_restoration_web as base
from research_filter_restoration_web_v1 import review_input, decode_crop, prepare_crop, delivered_views, read_receipt

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'outputs/codeformer_skip_feature_adapter_v1_model'
MODEL_NAME = 'codeformer_skip_feature_adapter_v1_u55.pth'
PROTOCOL_SHA = '0d358ab29e0344c7243b6312cb1d04ca6cc0e0a1de270bca7e6b4040dd566c90'
BASE_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
CHECKPOINT_SOURCE = 'outputs/skip_feature_adapter_v1/snapshot_0055/checkpoint.pt'
# Exact actual export after independent state/pixel audit and all69 root reviews.
EXPECTED_EXPORT = {
    'format':'codeformer-skip-feature-adapter-research-model-export-v1',
    'export_bytes':376824649,
    'export_sha256':'1a086299da7c263bbdda896f63be6e912a5ba76678f07ae76217898b8e941704',
    'model_tensor_sha256':'45f3e1d39f4abb19d1adadff45e6c2d841df90becc0c5c97c381a8e5b7502762',
    'adapter_tensor_sha256':'3950cfddd938fcff85850ad846a936133efa3ae17ca405a2c63417870d42acf5',
    'archive_sha256':'f445f5ffe3a97215046cb8e265eece2dd2d36961ac0b607948a52a29a5809c76',
    'source_checkpoint_sha256':'f947643b80b6871dc507a1f7cb1f873f56317cc1fbd14bb2407f09ebc231dcbb',
    'source_checkpoint_bytes':377485699,
    'reviewed_sheet_count':69,
    'quality_stop_retained':'New/worsened preservation or parent/R2 dark retention requirement failed; retain stop',
    'original_training_optimizer_RNG_archive_retained':True,
    'optimizer_RNG_excluded_from_inference_payload':True,
    'serialized_keys':['format','model'],
    'returned_code_imports':0,
    'source_sha256':'9d27c37358c3a5a59b4fb4ab00211944d07dcf179131fcbc914c59945e102b2a',
    'reviewed_visual_manifest':{
        'path':'outputs/codeformer_skip_feature_adapter_return_review_v1/pixels_v1/visual_manifest.json',
        'bytes':48433,'sha256':'98579c3c020ebfc7e19faa2a8e9851a05eca9f8f8965575ac9a74ef933c44361'}
}
REVIEW_BINDINGS = {
    'import':{'path':'outputs/codeformer_skip_feature_adapter_return_review_v1/return_import_v1.json',
        'bytes':1898450,'sha256':'a5070c80cb61233abde4c77f761b242633b5cd4093997c65bdc2dbfcdc0910b0'},
    'state':{'path':'outputs/codeformer_skip_feature_adapter_return_review_v1/state_v1/state_audit_v1.json',
        'bytes':418438,'sha256':'3e9d4f684fbe6ff2fcebe212268406edec65b699749910c9ac6d4d5796481434'},
    'pixels':{'path':'outputs/codeformer_skip_feature_adapter_return_review_v1/pixels_v1/audit.json',
        'bytes':1886277,'sha256':'0ce9e216ffe3a73fd7c3a42df33ce3e4cf28cd5b43e875b73898325c157f38ab'},
    'visual':{'path':'outputs/codeformer_skip_feature_adapter_return_review_v1/root_visual_review_v1.json',
        'bytes':27894,'sha256':'b922921517ab3389f4ef95c30631964418e9c9d7f38a74743bc72f4a87997749'}
}
SITES = ('32', '64', '128', '256')
CHANNELS = {'32':256, '64':256, '128':128, '256':128}
LIMIT = 10 * 1024**2
ARMS = ('original', 'r2', 'skip55')
QUALITY_STATUS = ('Experimental comparison only. Training stopped at 55 updates because R2 dark-image retention failed. '
                  'The new model still fails preservation checks against untouched CodeFormer; this model is not accepted.')
QUALITY_CHECKS = {'completed_training_updates':55, 'training_stopped_on_quality':True,
                  'preservation_pass':False, 'parent_dark_retention_pass':True,
                  'R2_dark_retention_pass':False, 'additional_interim_stop_pass':True}
router = APIRouter(prefix='/restoration-skip-test', tags=['Experimental skip-feature restoration'])


def adapter_shapes():
    shapes = {}
    for site in SITES:
        channels = CHANNELS[site]
        for name, shape in {'down.weight':(32,channels,1,1), 'down.bias':(32,),
            'depthwise.weight':(32,1,3,3), 'depthwise.bias':(32,),
            'up.weight':(channels,32,1,1), 'up.bias':(channels,)}.items():
            shapes['adapters.' + site + '.' + name] = shape
    return shapes


def validate_export_receipt(receipt):
    if EXPECTED_EXPORT is None or REVIEW_BINDINGS is None:
        raise ValueError('The new research model is awaiting completed independent state, pixel and visual review.')
    if (type(receipt) is not dict or receipt.get('complete') is not True or
        receipt.get('protocol_sha256') != PROTOCOL_SHA or receipt.get('source_checkpoint_member') != CHECKPOINT_SOURCE or
        receipt.get('source_training_updates') != 55 or receipt.get('base_tensor_sha256') != BASE_SHA or
        receipt.get('model_qualified') is not False or receipt.get('app_default_changed') is not False or
        receipt.get('neural_gradient_optimizer_VM_calls') != 0 or
        any(receipt.get(key) is not QUALITY_CHECKS[key] for key in
            ('preservation_pass','parent_dark_retention_pass','R2_dark_retention_pass','additional_interim_stop_pass')) or
        receipt.get('all539_tensor_values_exact') is not True or receipt.get('original515_base_values_unchanged') is not True or
        receipt.get('new24_adapter_tensors') != 24 or receipt.get('new_trainable_scalar_parameters') != 51328 or
        receipt.get('model_path') != str(MODEL_DIR / MODEL_NAME)):
        raise ValueError('The local research export differs from its audited committed state.')
    for key, value in EXPECTED_EXPORT.items():
        if receipt.get(key) != value:
            raise ValueError('Exact export proof differs: ' + key)
    if set(REVIEW_BINDINGS) != {'import','state','pixels','visual'}:
        raise ValueError('Four exact returned review bindings are required.')
    for key, expected in REVIEW_BINDINGS.items():
        proof = receipt.get('review_pins',{}).get(key)
        if type(proof) is not dict or proof != expected:
            raise ValueError('Exact independent review binding differs: ' + key)
    for key in ('export_sha256','model_tensor_sha256','adapter_tensor_sha256','archive_sha256','source_checkpoint_sha256'):
        if not isinstance(receipt.get(key),str) or not re.fullmatch('[0-9a-f]{64}',receipt[key]):
            raise ValueError('Missing exact model/export lineage fingerprint.')
    if type(receipt.get('export_bytes')) is not int or not 0 < receipt['export_bytes'] <= 400*1024**2:
        raise ValueError('Invalid inference-only export size.')
    return receipt


def skip_file():
    receipt = validate_export_receipt(read_receipt(MODEL_DIR/'model_export_v1.json'))
    for key, proof in REVIEW_BINDINGS.items():
        relative = Path(proof['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative).startswith(('\\','/')):
            raise ValueError('Review proof path must stay within the workspace.')
        path = ROOT/relative
        if path.stat().st_size != proof['bytes'] or base.sha(path) != proof['sha256']:
            raise ValueError('The closed returned review changed: ' + key)
        value = read_receipt(path,8*1024**2)
        if value.get('complete') is not True:
            raise ValueError('Completed returned review required.')
        if key in ('state','pixels') and value.get('audit_pass') is not True:
            raise ValueError('Independent saved-state and pixel audits must pass.')
        if key == 'visual' and value.get('root_personal_review_complete') is not True:
            raise ValueError('Root review of every planned visual sheet is required.')
    path = MODEL_DIR/MODEL_NAME
    if path.stat().st_size != receipt['export_bytes'] or base.sha(path) != receipt['export_sha256']:
        raise ValueError('Inference-only model bytes differ from the audited export.')
    return path, receipt


def inference_class(torch):
    """Construct types only after usable-input review and physical proof checks."""
    from third_party.codeformer.codeformer_arch import CodeFormer
    class ResidualSkipAdapter(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.down = torch.nn.Conv2d(channels,32,1,bias=True)
            self.depthwise = torch.nn.Conv2d(32,32,3,padding=1,groups=32,bias=True)
            self.up = torch.nn.Conv2d(32,channels,1,bias=True)
            self.activation = torch.nn.SiLU(inplace=False)
        def forward(self,value):
            return value+self.up(self.activation(self.depthwise(self.activation(self.down(value)))))
    class FrozenSkipRestorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = CodeFormer(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=list(SITES))
            self.adapters = torch.nn.ModuleDict({site:ResidualSkipAdapter(CHANNELS[site]) for site in SITES})
            self.last_forward_sites = ()
        def train(self, mode=True):
            return super().train(False)
        def forward(self, sample, *, w=1., detach_16=True, adain=True):
            if (torch.is_grad_enabled() or w != 1. or detach_16 is not True or adain is not True or
                sample.shape != (1,3,512,512) or sample.dtype != torch.float32 or sample.device.type != 'cpu'):
                raise ValueError('Frozen CPU FP32 singleton inference at w1/AdaIN is required.')
            if any(part._forward_hooks or part._forward_pre_hooks for part in self.base.modules()):
                raise ValueError('Clean stock hook namespace required.')
            sites = []; handles = []; self.last_forward_sites = ()
            def hook(site):
                def apply(module,args):
                    if (len(args) != 3 or args[2] != 1. or site in sites or
                        args[0].shape != (1,CHANNELS[site],int(site),int(site)) or
                        args[0].dtype != torch.float32 or args[0].device.type != 'cpu' or args[0].requires_grad or
                        not torch.isfinite(args[0]).all().item()):
                        raise ValueError('Exact detached observed SFT encoder argument required.')
                    used = self.adapters[site](args[0]); sites.append(site)
                    if used.shape != args[0].shape or not torch.isfinite(used).all().item():
                        raise ValueError('Invalid finite skip-adapter result.')
                    return used,args[1],args[2]
                return apply
            try:
                for site in SITES:
                    handles.append(self.base.fuse_convs_dict[site].register_forward_pre_hook(hook(site)))
                result = self.base(sample,w=w,detach_16=detach_16,adain=adain)
            finally:
                for handle in handles:
                    handle.remove()
                self.last_forward_sites = tuple(sites)
            if self.last_forward_sites != SITES:
                raise ValueError('All four audited skip-adapter sites must execute in order.')
            return result
    return FrozenSkipRestorer


class Comparison(base.Comparison):
    def __init__(self):
        super().__init__(); self.counter_lock = threading.Lock()
        self.counters = {'neural_forward_attempts':0,'neural_forward_returns':0,'neural_forward_successes':0,
                         'operator_review_refusals':0,'completed_comparisons':0}
        self.arm_counters = {arm:{'attempts':0,'returns':0,'successes':0} for arm in ARMS}
    def counted(self,name,arm=None):
        with self.counter_lock:
            self.counters[name] += 1
            if arm is not None:
                self.arm_counters[arm][name.removeprefix('neural_forward_')] += 1
    def measured_status(self):
        with self.counter_lock:
            return {**self.counters,'per_arm':{key:dict(value) for key,value in self.arm_counters.items()}}
    def load(self,arm,torch):
        if arm in ('original','r2'):
            return super().load(arm,torch)
        if arm != 'skip55':
            raise ValueError('Unknown experimental skip-feature model.')
        path,receipt = skip_file()
        if arm in self.models:
            if self.pins[arm]['weights_sha256'] != receipt['export_sha256']:
                raise ValueError('Model changed after loading. Restart this research server.')
            return self.models[arm],self.pins[arm]
        for name,digest in base.ARCHITECTURE_PINS.items():
            if base.sha(ROOT/'third_party/codeformer'/name) != digest:
                raise ValueError('Audited official CodeFormer architecture changed.')
        saved = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if (type(saved) is not dict or set(saved) != {'format','model'} or
            saved['format'] != 'codeformer-skip-feature-adapter-inference-v1' or type(saved['model']) is not dict):
            raise ValueError('Exact inference-only539 tensor export required.')
        state = saved['model']; shapes = adapter_shapes()
        if (len(state) != 539 or any(not isinstance(value,torch.Tensor) or value.device.type != 'cpu' or
            value.dtype != torch.float32 or not torch.isfinite(value).all().item() for value in state.values())):
            raise ValueError('Finite CPU FP32 539-state required.')
        frozen = {name.removeprefix('base.'):value for name,value in state.items() if name.startswith('base.')}
        adapters = {name:value for name,value in state.items() if name.startswith('adapters.')}
        if (len(frozen) != 515 or set(adapters) != set(shapes) or any(tuple(adapters[name].shape) != shapes[name] for name in adapters) or
            base.state_sha(state,torch) != receipt['model_tensor_sha256'] or base.state_sha(frozen,torch) != BASE_SHA or
            base.state_sha(adapters,torch) != receipt['adapter_tensor_sha256']):
            raise ValueError('Audited539/base515/adapter24 tensor schema or values differ.')
        with torch.random.fork_rng(devices=[]):
            net = inference_class(torch)().float().eval().requires_grad_(False)
            net.load_state_dict(state,strict=True)
        if (len(net.state_dict()) != 539 or any(value.requires_grad or value.grad is not None for value in net.parameters()) or
            any(part.training for part in net.modules()) or base.sha(path) != receipt['export_sha256']):
            raise ValueError('Frozen eval inference export validation failed.')
        metadata = {'arm':arm,'label':'New skip-feature update55 (experimental)',
            'weights_sha256':receipt['export_sha256'],'model_tensor_sha256':receipt['model_tensor_sha256'],
            'base_tensor_sha256':BASE_SHA,'adapter_tensor_sha256':receipt['adapter_tensor_sha256'],
            'source_checkpoint_sha256':receipt['source_checkpoint_sha256'],'source_training_updates':55,
            'protocol_sha256':PROTOCOL_SHA,'architecture':'CodeFormer by S-Lab; our four learned encoder-feature adapters',
            'license':'S-Lab License1.0; third_party/codeformer/LICENSE','new_trainable_parameters':51328,
            'all515_original_R2_tensors_frozen':True,'internal_resolution':512,'output_resolution':256,
            'fidelity':1.,'adain':True,'alignment':'none','display_processing':'none',
            'model_qualified':False,'quality_status':QUALITY_STATUS,'quality_checks':dict(QUALITY_CHECKS),
            'main_app_default_changed':False}
        self.models[arm],self.pins[arm] = net,metadata
        return net,metadata

    def generate(self, contents, input_review, selection):
        try:
            review_input(input_review)  # Must occur before Torch or a neural module import.
        except ValueError:
            self.counted('operator_review_refusals')
            raise
        if selection not in ('all', *ARMS):
            raise ValueError('Choose all three models, untouched CodeFormer, R2 or the new skip adapter.')
        rgb = decode_crop(contents); canvas, observed, _, geometry = prepare_crop(rgb)
        arms = ARMS if selection == 'all' else (selection,)
        # Verify requested exports before loading Torch or a model instance.
        if 'skip55' in arms:
            skip_file()
        if 'r2' in arms:
            self.candidate_file()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('A comparison is already running. Wait for it to finish.')
        started = time.monotonic()
        try:
            import torch
            from torch.nn import functional as F
            torch.set_num_threads(4)
            x = torch.from_numpy((canvas.astype(np.float32) / np.float32(255)).copy()).permute(2,0,1)[None]
            outputs = {}
            for arm in arms:
                net, provenance = self.load(arm, torch)
                with torch.no_grad():
                    sample = F.interpolate(x, (512,512), mode='bilinear', align_corners=False) * 2 - 1
                    self.counted('neural_forward_attempts', arm)
                    native = net(sample, w=1., detach_16=True, adain=True)[0]
                    self.counted('neural_forward_returns', arm)
                    if arm == 'skip55' and net.last_forward_sites != SITES:
                        raise ValueError('Exactly four audited skip-adapter sites must execute.')
                    if native.shape != (1,3,512,512) or not torch.isfinite(native).all().item():
                        raise ValueError('Invalid restoration output.')
                    self.counted('neural_forward_successes', arm)
                    raw = F.interpolate(((native.float()+1)/2).clamp(0,1), (256,256), mode='bilinear',
                        align_corners=False)[0].permute(1,2,0).contiguous().numpy().copy()
                # Both exported views preserve exact prepared pixels off support.
                raw, pixels = delivered_views(raw, canvas, observed)
                encoded = base.png(pixels)
                outputs[arm] = {'raw': raw, 'pixels': pixels, 'PNG': encoded, 'metadata': dict(provenance),
                    'raw_rgb_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'PNG_RGB8_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                    'PNG_file_sha256': hashlib.sha256(encoded).hexdigest()}
            metadata = {'policy': 'research-CodeFormer-skip-feature-comparison-v1', 'input_review': input_review,
                'selection': selection, 'geometry': geometry, 'uploaded_bytes_sha256': hashlib.sha256(contents).hexdigest(),
                'device': 'cpu', 'inference_only': True, 'no_completion_or_covering_removal': True,
                'operator_input_review_is_not_a_validated_automatic_threshold': True,
                'quality_status': QUALITY_STATUS, 'quality_checks':dict(QUALITY_CHECKS),
                'model_qualified': False, 'main_app_default_changed': False,
                'outside_observed_support_RAW_and_PNG_match_input_exact': True, 'RAW_PNG_separate': True,
                'native_input_has_no_aligned_clean_reference_in_this_flow': True,
                'seconds': time.monotonic()-started,
                'outputs': {k:{n:v[n] for n in ('metadata','raw_rgb_sha256','PNG_RGB8_sha256','PNG_file_sha256')} for k,v in outputs.items()}}
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('original-upload.png', base.png(rgb)); archive.writestr('resize-control.png', base.png(canvas))
                archive.writestr('observed-support.png', base.png(observed.astype(np.uint8)*255))
                for arm, value in outputs.items():
                    archive.writestr(arm+'.png', value['PNG'])
                    data=io.BytesIO(); np.save(data,value['raw'],allow_pickle=False); archive.writestr(arm+'-raw.npy',data.getvalue())
                archive.writestr('processing.json',json.dumps(metadata,indent=2,allow_nan=False)+'\n')
            self.counted('completed_comparisons')
            return {'original':base.data_url(base.png(canvas)),
                'outputs':{k:base.data_url(v['PNG']) for k,v in outputs.items()},
                'metadata':metadata,'bundle':base.data_url(bundle.getvalue(),'application/zip')}
        finally:
            self.lock.release()


engine = Comparison()


@router.get('',response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT/'templates/research_skip_restoration_v1.html').read_text(encoding='utf-8'),
                        headers={'Cache-Control':'no-store'})


@router.get('/status')
def status():
    return {'candidate_export_present':(MODEL_DIR/'model_export_v1.json').is_file(),
        'review_pins_configured':EXPECTED_EXPORT is not None and REVIEW_BINDINGS is not None,
        'candidate_qualified':False,'main_app_default_changed':False,'source_training_updates':55,
        'manual_input_review_required':True,'quality_status':QUALITY_STATUS,'quality_checks':dict(QUALITY_CHECKS),
        'route':'/restoration-skip-test',
        **engine.measured_status()}


@router.post('/compare')
async def compare(file:UploadFile=File(...),input_review:str=Form(''),selection:str=Form('all')):
    try:
        return await run_in_threadpool(engine.generate,await file.read(LIMIT+1),input_review,selection)
    except ValueError as error:
        raise HTTPException(422,str(error)) from error
    except (FileNotFoundError,OSError) as error:
        raise HTTPException(503,'The verified experimental skip-feature export is unavailable.') from error
    except RuntimeError as error:
        raise HTTPException(409,str(error)) from error

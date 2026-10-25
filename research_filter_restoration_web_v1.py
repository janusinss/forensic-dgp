"""Unused opt-in CodeFormer fusion50 research comparison; no training/default switch.

Import/status/page and operator refusal use only image/HTTP utilities.  Torch is
loaded only after a usable-input confirmation and exact model lineage checks.
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
from PIL import Image, ImageOps
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

# Existing inert utilities and the untouched benchmark/R2 loader are reused.
# This module imports no DGP workflow, constructor or Torch during refusal.
import research_restoration_web as base

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'outputs/codeformer_filter_fusion_training_v1_model'
MODEL_NAME = 'codeformer_filter_fusion_v1_u50.pth'
PROTOCOL_SHA = 'c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949'
ARCHIVE_SHA = '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
CHECKPOINT_PATH = 'outputs/filter_fusion_training_v1/snapshot_0050/checkpoint.pt'
CHECKPOINT_SHA = '8ddc036762508e5c65c012d09e8f6a0f7f8e339c85d15e278f96958d98e4d869'
STATE_SHA = 'eff91cc6832bdf7ee27cd9f749a8f64a64a2351ab1443b14971b543ae2d86f35'
FROZEN443_SHA = '1d2f005f527078a721dc502256d06d3be99dc3b3a6b7258456ec9af141b26e67'
STATE_AUDIT_SHA = '38a0d6c6666fbc6cffcca0274cb7d5409ff0f4a403a11deda7db1b251b7026fa'
REVIEW_DIR = ROOT / 'outputs/codeformer_filter_fusion_return_review_20261019_v1'
REVIEW_PINS = {
    'import': ('return_import_v1.json', 'de5aae1f82e98f08c689950451c41b13d83a18b4e9dd0a23cec34cf9a2e51cf5'),
    'state': ('state_v1/state_audit_v1.json', STATE_AUDIT_SHA),
    'pixels': ('pixels_v1/audit.json', 'ec0cba576913e6580a07b2dee120d0c2dc9741ee927c68eb2b0dae9ae83747fa')}
LIMIT = 10 * 1024**2
ARMS = ('original', 'r2', 'fusion50')
QUALITY_STOP = '50-update quality stop: preservation and dark-retention checks failed; experimental comparison only.'
router = APIRouter(prefix='/restoration-filter-test', tags=['Experimental filter restoration'])


def review_input(review):
    if review == 'needs_clearer':
        raise ValueError('Facial structure is insufficient. Upload a clearer face crop.')
    if review == 'out_of_scope':
        raise ValueError('Upload one already cropped frontal or mildly turned face.')
    if review != 'usable':
        raise ValueError('Review the input first: one frontal or mildly turned face with readable visible features.')


def decode_crop(contents):
    if not contents or len(contents) > LIMIT:
        raise ValueError('Upload a PNG or JPEG smaller than 10 MB.')
    try:
        with Image.open(io.BytesIO(contents)) as source:
            if source.format not in ('PNG', 'JPEG'):
                raise ValueError('Use a PNG or JPEG image.')
            if max(source.size) > 4096 or source.width * source.height > 16_000_000:
                raise ValueError('Use a face crop no larger than 4096 pixels per side.')
            return np.array(ImageOps.exif_transpose(source).convert('RGB'))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Cannot read this image. Use a PNG or JPEG face crop.') from exc


def prepare_crop(rgb):
    """Exact legacy neutral128 center-pad/bilinear256/nearest-support geometry."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected nonempty uint8 RGB face pixels')
    h, w = rgb.shape[:2]
    if max(h, w) > 4096 or h*w > 16_000_000:
        raise ValueError('Face crop exceeds the image bounds')
    side = max(h, w); top, left = (side-h)//2, (side-w)//2
    canvas = Image.new('RGB', (side, side), (128, 128, 128)); canvas.paste(Image.fromarray(rgb), (left, top))
    support = Image.new('L', (side, side), 0); support.paste(255, (left, top, left+w, top+h))
    target = np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR)).copy()
    observed = np.asarray(support.resize((256, 256), Image.Resampling.NEAREST)).copy() > 0
    mask = np.zeros((256, 256), dtype=bool)
    yy, xx = np.nonzero(observed)
    if len(xx) == 0:
        raise ValueError('Upload a face crop with readable facial width and height.')
    geometry = {'native_size': [w,h], 'native_center_offset': [left,top],
        'observed_bounds_256': [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
        'output_size': [256,256], 'policy': 'dgp256-observed-canvas-v1',
        'input_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'canvas_rgb_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
        'observed_sha256': hashlib.sha256(observed.tobytes()).hexdigest(),
        'removal_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
    return target, observed, mask, geometry


def read_receipt(path, cap=1024**2):
    if path.stat().st_size > cap:
        raise ValueError('Research export receipt exceeds its metadata limit.')
    base.sha(path)
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('Research export receipt has duplicate fields.')
            result[k] = v
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite export metadata.')))


def validate_export_receipt(receipt):
    """Pure, strict receipt validation; physical files are checked separately."""
    if not isinstance(receipt, dict):
        raise ValueError('Research export must be a metadata object.')
    if (receipt.get('complete') is not True or receipt.get('model_qualified') is not False or
        receipt.get('protocol_sha256') != PROTOCOL_SHA or receipt.get('archive_sha256') != ARCHIVE_SHA or
        receipt.get('source_checkpoint') != CHECKPOINT_PATH or receipt.get('source_checkpoint_sha256') != CHECKPOINT_SHA or
        receipt.get('source_checkpoint_bytes') != 461955077 or receipt.get('model_tensor_sha256') != STATE_SHA or
        receipt.get('source_training_updates') != 50 or receipt.get('all515_tensor_values_exact') is not True or
        receipt.get('app_default_changed') is not False or
        not isinstance(receipt.get('review_pins'), dict) or
        receipt.get('preservation_pass') is not False or receipt.get('R2_dark_retention_pass') is not False or
        receipt.get('parent_dark_retention_pass') is not True or receipt.get('additional_interim_stop_pass') is not False or
        receipt.get('params_ema_key_is_loader_compatibility_not_new_EMA') is not True or
        receipt.get('original_training_optimizer_RNG_archive_retained') is not True or
        receipt.get('neural_gradient_optimizer_VM_calls') != 0):
        raise ValueError('The new model export differs from the audited stopped state.')
    for name, (_, digest) in REVIEW_PINS.items():
        proof = receipt['review_pins'].get(name)
        if not isinstance(proof, dict) or proof.get('sha256') != digest or type(proof.get('bytes')) is not int or proof['bytes'] <= 0:
            raise ValueError('The independent return review binding differs.')
    visual = receipt['review_pins'].get('visual')
    if not isinstance(visual, dict) or type(visual.get('bytes')) is not int or visual['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', str(visual.get('sha256', ''))):
        raise ValueError('The root visual review binding is missing.')
    if not isinstance(receipt.get('export_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', receipt['export_sha256']):
        raise ValueError('Invalid research model fingerprint.')
    if type(receipt.get('export_bytes')) is not int or not 0 < receipt['export_bytes'] <= 400 * 1024**2:
        raise ValueError('Invalid research model size.')
    if receipt.get('model_path') != str(MODEL_DIR / MODEL_NAME):
        raise ValueError('The research model path differs from the exact exported selection.')
    return receipt


def fusion_file():
    receipt = validate_export_receipt(read_receipt(MODEL_DIR / 'model_export_v1.json'))
    for name, (relative, _) in REVIEW_PINS.items():
        proof = REVIEW_DIR / relative
        if proof.stat().st_size != receipt['review_pins'][name]['bytes'] or base.sha(proof) != receipt['review_pins'][name]['sha256']:
            raise ValueError('The completed independent return review changed.')
    visual_path = REVIEW_DIR / 'root_visual_review_v1.json'
    visual = read_receipt(visual_path, 8 * 1024**2)
    if visual_path.stat().st_size != receipt['review_pins']['visual']['bytes'] or base.sha(visual_path) != receipt['review_pins']['visual']['sha256']:
        raise ValueError('The completed root visual review changed.')
    if visual.get('complete') is not True or visual.get('root_personal_review_complete') is not True or visual.get('reviewed_sheets') != 63 or visual.get('model_qualified') is not False or visual.get('archive_sha256') != ARCHIVE_SHA or visual.get('protocol_sha256') != PROTOCOL_SHA or visual.get('pixel_audit_sha256') != REVIEW_PINS['pixels'][1]:
        raise ValueError('The complete root review of the stopped state is required.')
    path = MODEL_DIR / MODEL_NAME
    if path.stat().st_size != receipt['export_bytes'] or base.sha(path) != receipt['export_sha256']:
        raise ValueError('The exported research model fingerprint differs.')
    return path, receipt


def delivered_views(raw, canvas, observed):
    """One explicit support composition for both downloadable output forms."""
    if (not isinstance(raw, np.ndarray) or raw.shape != (256, 256, 3) or raw.dtype != np.float32 or
        not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1 or
        not isinstance(canvas, np.ndarray) or canvas.shape != raw.shape or canvas.dtype != np.uint8 or
        not isinstance(observed, np.ndarray) or observed.shape != (256, 256) or observed.dtype != bool):
        raise ValueError('Invalid restoration pixels or observed support.')
    raw = raw.copy()
    raw[~observed] = canvas[~observed].astype(np.float32) / np.float32(255)
    pixels = np.floor(raw * np.float32(255)).astype(np.uint8)
    pixels[~observed] = canvas[~observed]
    return raw, pixels


class Comparison(base.Comparison):
    def __init__(self):
        super().__init__()
        self.counter_lock = threading.Lock()
        self.counters = {'neural_forward_attempts': 0, 'neural_forward_returns': 0,
                         'neural_forward_successes': 0, 'operator_review_refusals': 0,
                         'completed_comparisons': 0}
        self.arm_counters = {arm: {'attempts': 0, 'returns': 0, 'successes': 0} for arm in ARMS}

    def counted(self, name, arm=None):
        with self.counter_lock:
            self.counters[name] += 1
            if arm is not None:
                self.arm_counters[arm][name.removeprefix('neural_forward_')] += 1

    def measured_status(self):
        with self.counter_lock:
            return {**self.counters, 'per_arm': {arm: dict(value) for arm, value in self.arm_counters.items()}}

    def load(self, arm, torch):
        if arm in ('original', 'r2'):
            return super().load(arm, torch)
        if arm != 'fusion50':
            raise ValueError('Unknown experimental restoration model.')
        path, receipt = fusion_file(); digest = receipt['export_sha256']
        if arm in self.models:
            if self.pins[arm]['weights_sha256'] != digest:
                raise ValueError('Model changed after loading. Restart this comparison server.')
            return self.models[arm], self.pins[arm]
        for name, fingerprint in base.ARCHITECTURE_PINS.items():
            if base.sha(ROOT / 'third_party/codeformer' / name) != fingerprint:
                raise ValueError('The audited CodeFormer architecture differs.')
        from third_party.codeformer.codeformer_arch import CodeFormer
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if set(saved) != {'params_ema'} or len(saved['params_ema']) != 515:
            raise ValueError('Expected exactly 515 inference-only CodeFormer tensors.')
        state = saved['params_ema']
        if any(not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or v.dtype != torch.float32 or
               not torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('Invalid finite FP32 restoration state.')
        frozen = {n:v for n,v in state.items() if not n.startswith('fuse_convs_dict.')}
        if base.state_sha(state, torch) != STATE_SHA or len(frozen) != 443 or base.state_sha(frozen, torch) != FROZEN443_SHA:
            raise ValueError('The new model state or frozen tensors differ from the audited values.')
        net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32','64','128','256'])
        net.load_state_dict(state, strict=True); net.float().eval().requires_grad_(False)
        if base.sha(path) != digest:
            raise ValueError('Research model changed while loading.')
        metadata = {'arm': arm, 'label': 'Filter-fusion update50 (experimental)', 'weights_sha256': digest,
            'model_tensor_sha256': STATE_SHA, 'source_checkpoint_sha256': CHECKPOINT_SHA,
            'frozen443_tensor_sha256': FROZEN443_SHA, 'protocol_sha256': PROTOCOL_SHA,
            'source_training_updates': 50, 'initializer': 'Our retained R2 adaptation of pretrained CodeFormer',
            'architecture': 'CodeFormer by S-Lab; our fusion-only fine-tuned adaptation',
            'license': 'S-Lab License 1.0; third_party/codeformer/LICENSE',
            'internal_resolution': 512, 'output_resolution': 256, 'fidelity': 1., 'adain': True,
            'alignment': 'none', 'display_processing': 'none', 'model_qualified': False,
            'quality_stop_retained': QUALITY_STOP, 'main_app_default_changed': False,
            'params_ema_key_is_loader_compatibility_not_new_EMA': True}
        self.models[arm], self.pins[arm] = net, metadata
        return net, metadata

    def generate(self, contents, input_review, selection):
        try:
            review_input(input_review)  # Must occur before Torch or a neural module import.
        except ValueError:
            self.counted('operator_review_refusals')
            raise
        if selection not in ('all', *ARMS):
            raise ValueError('Choose all models, untouched CodeFormer, R2 or filter-fusion update50.')
        rgb = decode_crop(contents); canvas, observed, _, geometry = prepare_crop(rgb)
        arms = ARMS if selection == 'all' else (selection,)
        # Verify requested exports before loading Torch or a model instance.
        if 'fusion50' in arms:
            fusion_file()
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
            metadata = {'policy': 'research-CodeFormer-filter-fusion-comparison-v1', 'input_review': input_review,
                'selection': selection, 'geometry': geometry, 'uploaded_bytes_sha256': hashlib.sha256(contents).hexdigest(),
                'device': 'cpu', 'inference_only': True, 'no_completion_or_covering_removal': True,
                'operator_input_review_is_not_a_validated_automatic_threshold': True,
                'quality_status': QUALITY_STOP, 'model_qualified': False, 'main_app_default_changed': False,
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


@router.get('', response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT/'templates/research_filter_restoration_v1.html').read_text(encoding='utf-8'),
                        headers={'Cache-Control':'no-store'})


@router.get('/status')
def status():
    return {'candidate_export_present':(MODEL_DIR/'model_export_v1.json').is_file(),
        'candidate_qualified':False,'main_app_default_changed':False,'source_training_updates':50,
        'manual_input_review_required':True,'quality_status':QUALITY_STOP,'route':'/restoration-filter-test',
        **engine.measured_status()}


@router.post('/compare')
async def compare(file: UploadFile=File(...), input_review: str=Form(''), selection: str=Form('all')):
    try:
        return await run_in_threadpool(engine.generate,await file.read(LIMIT+1),input_review,selection)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    except (FileNotFoundError,OSError) as exc:
        raise HTTPException(503,'The verified research export is unavailable.') from exc
    except RuntimeError as exc:
        raise HTTPException(409,str(exc)) from exc

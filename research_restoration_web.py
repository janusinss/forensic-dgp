"""Opt-in local comparison of untouched CodeFormer and the actual R2 state.

This route never trains, completes covered regions, selects an app default or
downloads weights. A research candidate remains unqualified even when it loads.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import stat
import threading
import time
import zipfile

import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_model'
FINAL_CHECKPOINT_SHA = 'ec115accce587307d005b4e1e0128991f89fb4095b1f70d95b9e065663a8fa93'
FINAL_STATE_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
PROTOCOL_SHA = '82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a'
ORIGINAL = ROOT / 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth'
ORIGINAL_SHA = '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
ARCHITECTURE_PINS = {
    'codeformer_arch.py': 'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb',
    'vqgan_arch.py': 'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf',
    'registry.py': 'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'}
LIMIT = 10 * 1024**2
router = APIRouter(prefix='/restoration-test', tags=['Experimental restoration comparison'])


def sha(path, allow_benchmark_hardlinks=False):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            value = part.lstat()
            if stat.S_ISLNK(value.st_mode) or getattr(value, 'st_file_attributes', 0) & 1024:
                raise ValueError('Linked research-model paths are not accepted.')
    value = path.stat()
    allowed_link = allow_benchmark_hardlinks and path == ORIGINAL.absolute()
    if not stat.S_ISREG(value.st_mode) or (value.st_nlink != 1 and not allowed_link):
        raise ValueError('An ordinary research-model file is required.')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def data_url(payload, mime='image/png'):
    return 'data:' + mime + ';base64,' + base64.b64encode(payload).decode('ascii')


def state_sha(state, torch):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Comparison:
    def __init__(self):
        self.lock = threading.Lock()
        self.models = {}
        self.pins = {}

    def candidate_file(self):
        receipt_path = MODEL_DIR / 'model_export_v1.json'
        sha(receipt_path)  # Enforce the same ordinary-file/no-link policy as weights.
        if receipt_path.stat().st_size > 1024**2:
            raise ValueError('Invalid model export receipt.')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        # The extraction audit owns exact tensor/file lineage, not model quality.
        if (receipt.get('complete') is not True or receipt.get('model_qualified') is not False or
                receipt.get('source_checkpoint_sha256') != FINAL_CHECKPOINT_SHA or
                receipt.get('model_tensor_sha256') != FINAL_STATE_SHA or
                receipt.get('protocol_sha256') != PROTOCOL_SHA):
            raise ValueError('The exported R2 model lineage differs from the reviewed selection.')
        path = MODEL_DIR / 'codeformer_role_preserving_r2.pth'
        if path.stat().st_size != receipt['export_bytes'] or sha(path) != receipt['export_sha256']:
            raise ValueError('The exported R2 model fingerprint differs.')
        return path, receipt

    def load(self, arm, torch):
        if arm == 'r2':
            path, receipt = self.candidate_file()
            digest = receipt['export_sha256']
            expected_state = FINAL_STATE_SHA
        else:
            path, digest, expected_state = ORIGINAL, ORIGINAL_SHA, None
            if sha(path, allow_benchmark_hardlinks=True) != digest:
                raise ValueError('The untouched CodeFormer benchmark fingerprint differs.')
        if arm in self.models:
            if self.pins[arm]['weights_sha256'] != digest:
                raise ValueError('Model changed after loading. Restart the comparison server.')
            return self.models[arm], self.pins[arm]
        for name, fingerprint in ARCHITECTURE_PINS.items():
            if sha(ROOT / 'third_party/codeformer' / name) != fingerprint:
                raise ValueError('The audited CodeFormer architecture source differs.')
        from third_party.codeformer.codeformer_arch import CodeFormer
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if set(saved) != {'params_ema'} or len(saved['params_ema']) != 515:
            raise ValueError('Expected exactly 515 inference-only CodeFormer tensors.')
        state = saved['params_ema']
        if any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32 or
               not torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('Invalid restoration state.')
        tensor_digest = state_sha(state, torch)
        if expected_state is not None and tensor_digest != expected_state:
            raise ValueError('R2 model tensor values differ from the actual trained checkpoint.')
        net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                         connect_list=['32', '64', '128', '256'])
        net.load_state_dict(state, strict=True)
        net.float().eval().requires_grad_(False)
        if sha(path, allow_benchmark_hardlinks=arm == 'original') != digest:
            raise ValueError('Model changed while loading.')
        metadata = {'arm': arm, 'label': 'Our R2 · 50 new updates · experimental' if arm == 'r2' else 'Untouched CodeFormer',
            'weights_sha256': digest, 'model_tensor_sha256': tensor_digest,
            'model_qualified': False, 'main_app_default_changed': False,
            'source_checkpoint_sha256': FINAL_CHECKPOINT_SHA if arm == 'r2' else None,
            'training_updates': 50 if arm == 'r2' else None,
            'architecture': 'CodeFormer by S-Lab; our fine-tuned adaptation for R2',
            'license': 'S-Lab License 1.0; third_party/codeformer/LICENSE',
            'internal_resolution': 512, 'output_resolution': 256,
            'fidelity': 1., 'adain': True, 'alignment': 'none', 'display_processing': 'none'}
        self.models[arm], self.pins[arm] = net, metadata
        return net, metadata

    def generate(self, contents, input_review, selection):
        # An operator must judge the observed input, never a generated face.
        from dgp_face_workflow_v3 import decode_crop, review_input
        from dgp_face_restoration import prepare_crop
        review_input(input_review)
        if selection not in ('both', 'original', 'r2'):
            raise ValueError('Choose both models, untouched CodeFormer or R2.')
        rgb = decode_crop(contents)
        canvas, observed, _, geometry = prepare_crop(rgb)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('A comparison is already running. Wait for it to finish.')
        started = time.monotonic()
        try:
            import torch
            from torch.nn import functional as F
            torch.set_num_threads(4)
            x = torch.from_numpy((canvas.astype(np.float32) / np.float32(255)).copy()).permute(2, 0, 1)[None]
            outputs = {}
            arms = ('original', 'r2') if selection == 'both' else (selection,)
            for arm in arms:
                net, provenance = self.load(arm, torch)
                with torch.no_grad():
                    sample = F.interpolate(x, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
                    native = net(sample, w=1., detach_16=True, adain=True)[0]
                    if native.shape != (1, 3, 512, 512) or not torch.isfinite(native).all().item():
                        raise ValueError('Invalid restoration output.')
                    raw = F.interpolate(((native.float() + 1) / 2).clamp(0, 1),
                        (256, 256), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).contiguous().numpy().copy()
                pixels = np.floor(raw * np.float32(255)).astype(np.uint8)
                pixels[~observed] = canvas[~observed]
                outputs[arm] = {'raw': raw, 'pixels': pixels, 'metadata': dict(provenance),
                    'raw_rgb_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'PNG_RGB8_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}
            metadata = {'policy': 'research-CodeFormer-R2-comparison-v1', 'input_review': input_review,
                'selection': selection, 'geometry': geometry, 'device': 'cpu', 'inference_only': True,
                'no_completion_or_covering_removal': True, 'operator_input_review_is_not_a_validated_automatic_threshold': True,
                'quality_status': 'R2 failed clear/normal preservation; experimental comparison only',
                'model_qualified': False, 'main_app_default_changed': False, 'seconds': time.monotonic() - started,
                'outputs': {k: {n: v[n] for n in ('metadata', 'raw_rgb_sha256', 'PNG_RGB8_sha256')} for k, v in outputs.items()}}
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('original-upload.png', png(rgb))
                archive.writestr('resize-control.png', png(canvas))
                archive.writestr('observed-support.png', png(observed.astype(np.uint8) * 255))
                for arm, value in outputs.items():
                    archive.writestr(arm + '.png', png(value['pixels']))
                    data = io.BytesIO(); np.save(data, value['raw'], allow_pickle=False)
                    archive.writestr(arm + '-raw.npy', data.getvalue())
                archive.writestr('processing.json', json.dumps(metadata, indent=2, allow_nan=False) + '\n')
            return {'original': data_url(png(canvas)), 'outputs': {k: data_url(png(v['pixels'])) for k, v in outputs.items()},
                'metadata': metadata, 'bundle': data_url(bundle.getvalue(), 'application/zip')}
        finally:
            self.lock.release()


engine = Comparison()


@router.get('', response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT / 'templates/research_restoration.html').read_text(encoding='utf-8'),
                        headers={'Cache-Control': 'no-store'})


@router.get('/status')
def status():
    return {'candidate_export_present': (MODEL_DIR / 'model_export_v1.json').is_file(),
            'candidate_qualified': False, 'main_app_default_changed': False}


@router.post('/compare')
async def compare(file: UploadFile = File(...), input_review: str = Form(''), selection: str = Form('both')):
    try:
        return await run_in_threadpool(engine.generate, await file.read(LIMIT + 1), input_review, selection)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(503, 'The verified research export is unavailable.') from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

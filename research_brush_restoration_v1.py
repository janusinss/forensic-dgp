"""Optional reviewed Protect/Apply preparation with selected frozen restoration.

Separate research route; existing app and historical rectangle route are intact.
Inspect/preview/refusals are non-neural. Two forwards per comparison, one bypass.
"""
import base64
import io
import json
from pathlib import Path
import threading
import time
import zipfile

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

import research_postfusion_restoration_v1 as P
from research import protected_surroundings_brush_v1 as preparation

ROOT = P.ROOT
ROUTE = '/restoration-brush-test'
SOURCES = ('research_brush_restoration_v1.py', 'serve_codeformer_brush_research_v1.py',
           'research/protected_surroundings_brush_v1.py', 'templates/research_brush_restoration_v1.html',
           'static/research_brush_restoration_v1.js', 'static/research_brush_restoration_v1.css')
router = APIRouter(prefix=ROUTE, tags=['Reviewed Protect and Apply research'])
_sources = None


def need(ok, message):
    if not ok: raise ValueError(message)


def source_check():
    need(_sources is not None, 'Start this separate page with its reviewed source manifest.')
    P.bound(_sources['manifest'])
    for row in _sources['sources']:
        P.bound(row)
    return _sources


def configure_sources(path, expected_sha):
    global _sources
    need(_sources is None, 'Fresh server required for a changed source manifest.')
    record = P.pin(path, 1024 ** 2)
    need(record['sha256'] == expected_sha, 'Exact brush source-manifest SHA required.')
    value = P.strict_json(P.bound(record).read_bytes())
    need(value['complete'] and value['purpose'] == 'manual-Protect-Apply-research-page-v1'
         and {r['path'] for r in value['sources']} == set(SOURCES), 'Exact six-source brush manifest required.')
    review = P.strict_json(P.bound(value['source_review']).read_bytes())
    need(review['complete'] and review['source_review_pass']
         and set(SOURCES) <= {r['path'] for r in review['sources']}, 'Independent current brush source review required.')
    for row in review['sources']: P.bound(row)
    _sources = {**value, 'manifest': record}
    try: return source_check()
    except BaseException:
        _sources = None
        raise


def zip_bytes(payloads):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in payloads.items(): archive.writestr(name, raw)
    need(output.tell() <= P.BUNDLE_LIMIT, 'Bundle exceeds64MiB; use a smaller native crop.')
    return output.getvalue()


def preview(contents, protect, apply, radius):
    sources = source_check()
    response, payloads = preparation.build_preview(contents, protect, apply, nominal_radius=radius)
    # Bind deployment sources too; the pure pixel module's identity remains available.
    pure_id = response['preview_id']
    binding = {'pure_preview_id': pure_id, 'source_manifest': sources['manifest']}
    response['preview_id'] = preparation.sha(preparation.json_bytes(binding))
    response['metadata'].update({'pure_preview_id': pure_id, 'preview_id': response['preview_id'],
                                 'reviewed_page_sources': sources['manifest']})
    payloads['processing.json'] = preparation.json_bytes(response['metadata'])
    response['bundle'] = P.base.data_url(zip_bytes(payloads), 'application/zip')
    return response, payloads


class SelectedRestorer(P.Comparison):
    """Own CPU model instances; same retained loader, tensor math and invariants."""
    def release_models(self):
        # Only BrushFlow calls this, while its request lock is still held.
        # These are owned transient objects, never the existing app's engine.
        self.models.clear()
        self.incumbent_loader.models.clear()
        import gc
        gc.collect()

    def run_one(self, contents, selection):
        need(selection in P.ARMS, 'Choose retained U96 or experimental u64.')
        source_check()
        rgb = P.decode_crop(contents)
        canvas, observed, _, geometry = P.prepare_crop(rgb)
        reviewed = P.manifest()
        if not self.models:
            import psutil
            if psutil.virtual_memory().available < 5 * 1024 ** 3 // 2:
                raise RuntimeError('Local memory is busy. Retry after the current image task or audit finishes.')
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(4)
        models = self.load(reviewed, torch)
        need(torch.get_default_dtype() == torch.float32, 'CPU FP32 required.')
        x = torch.from_numpy((canvas.astype(np.float32) / np.float32(255)).copy()).permute(2, 0, 1)[None]
        with torch.no_grad(), torch.device('cpu'), torch.autocast(device_type='cpu', enabled=False):
            sample = F.interpolate(x, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            self.invariants(torch)
            self.count('neural_forward_attempts')
            native = models[selection](sample, w=1., detach_16=True, adain=True)[0]
            self.count('neural_forward_returns')
            need(tuple(native.shape) == (1, 3, 512, 512) and native.dtype == torch.float32
                 and native.device.type == 'cpu' and not native.requires_grad and torch.isfinite(native).all().item(),
                 'Finite frozen singleton output required.')
            self.invariants(torch)
            need(models[selection].last_forward_sites == P.spatial.SITES, 'All four ordered feature sites required.')
            raw = F.interpolate(((native + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
            raw = raw[0].permute(1, 2, 0).contiguous().numpy().copy()
            raw, pixels = P.delivered_views(raw, canvas, observed)
        stream = io.BytesIO(); np.save(stream, raw, allow_pickle=False)
        return {'PNG': P.base.png(pixels), 'raw': stream.getvalue(),
                'prepared': P.base.png(canvas), 'support': P.base.png(observed.astype(np.uint8) * 255),
                'metadata': {'selection': selection, 'label': P.LABELS[selection], 'geometry': geometry,
                             'uploaded_bytes_sha256': preparation.sha(contents), 'models': reviewed['models'],
                             'raw_RGB_sha256': preparation.sha(raw.tobytes()), 'PNG_RGB_sha256': preparation.sha(pixels.tobytes()),
                             'CPU_FP32': True, 'w': 1., 'AdaIN': True, 'detach16': True,
                             'internal_resolution': 512, 'delivery_resolution': 256, 'display_processing': 'none',
                             'model_qualified': False, 'main_app_default_changed': False}}


class BrushFlow:
    def __init__(self, restorer=None):
        self.restorer = restorer if restorer is not None else SelectedRestorer()
        self.lock = threading.Lock()

    def compare(self, contents, protect, apply, radius, preview_id, confirmed, input_review, selection, bypass=False):
        try: P.review_input(input_review)
        except ValueError:
            self.restorer.count('operator_review_refusals')
            raise
        need(selection in P.ARMS, 'Choose retained U96 or experimental u64.')
        source_check()
        if bypass:
            decoded = preparation.decode_original(contents)
            displayed, payloads = None, {'uploaded-original.bin': contents,
                                         'original-decoded-rgb.png': preparation.png(decoded)}
        else:
            need(confirmed is True, 'Confirm the current Protect and Apply preview before restoration.')
            displayed, payloads = preview(contents, protect, apply, radius)
            need(type(preview_id) is str and preview_id == displayed['preview_id'],
                 'The preview is stale. Preview and confirm the current image and both masks again.')
            need(displayed['metadata']['no_effect'] or displayed['metadata']['protected_native_pixels'] > 0,
                 'Protect the face and existing blurred edges before adding surroundings blur, or bypass preparation.')
        if not self.lock.acquire(blocking=False): raise RuntimeError('A restoration is already running. Wait for it to finish.')
        try:
            before = self.restorer.measured_status(); started = time.monotonic()
            original = self.restorer.run_one(contents, selection)
            result = original if bypass else self.restorer.run_one(payloads['derived-native.png'], selection)
            after = self.restorer.measured_status()
            maximum = 1 if bypass else 2
            need(after['neural_forward_attempts'] - before['neural_forward_attempts'] == maximum
                 and after['neural_forward_returns'] - before['neural_forward_returns'] == maximum,
                 'Exact selected-model forward budget required.')
            need(original['metadata']['uploaded_bytes_sha256'] == preparation.sha(contents), 'Original output binding mismatch.')
            if not bypass:
                need(result['metadata']['uploaded_bytes_sha256'] == preparation.sha(payloads['derived-native.png']), 'Prepared output binding mismatch.')
            metadata = {'policy': 'optional-reviewed-Protect-Apply-selected-restoration-v1',
                        'mode': 'original_bypass' if bypass else 'manual_assisted_Protect_Apply',
                        'input_review': input_review, 'selection': selection, 'preview_id': None if bypass else displayed['preview_id'],
                        'confirmed': False if bypass else confirmed, 'preparation': None if bypass else displayed['metadata'],
                        'original': original['metadata'], 'prepared_result': result['metadata'],
                        'source_manifest': source_check()['manifest'], 'parent_test_manifest': dict(P._configured),
                        'neural_forward_attempts': maximum, 'neural_forward_returns': maximum,
                        'seconds': time.monotonic() - started, 'model_qualified': False, 'main_app_default_changed': False,
                        'automatic_preparation_or_detection': False, 'training_optimizer_VM_calls': 0,
                        'native_input_has_no_aligned_clean_reference_in_this_flow': True,
                        'protection_scope': 'preprocessing only; generated pixels and appearance may change',
                        'no_completion_or_covering_removal': True, 'quality_status': P.QUALITY_STATUS}
            for key, item in [('original', original), ('prepared', result)]:
                payloads[key + '-restoration.png'] = item['PNG']
                payloads[key + '-restoration-raw.npy'] = item['raw']
            payloads['resize-control.png'] = original['prepared']
            payloads['observed-support-256.png'] = original['support']
            payloads['restoration.json'] = preparation.json_bytes(metadata)
            bundle = zip_bytes(payloads)
            self.restorer.count('completed_comparisons')
            return {'original_input': P.base.data_url(original['prepared']),
                    'prepared_input': P.base.data_url(result['prepared']),
                    'original_result': P.base.data_url(original['PNG']), 'prepared_result': P.base.data_url(result['PNG']),
                    'metadata': metadata, 'bundle': P.base.data_url(bundle, 'application/zip')}
        finally:
            try:
                release = getattr(self.restorer, 'release_models', None)
                if release is not None: release()
            finally: self.lock.release()


engine = BrushFlow()


@router.get('', response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT / SOURCES[3]).read_text(encoding='utf-8'), headers={'Cache-Control': 'no-store'})


@router.get('/status')
def status():
    source_check(); P.manifest()
    return {'available': True, 'maximum_compare_forwards': 2, 'maximum_bypass_forwards': 1,
            'preparation_neural_calls': 0, 'model_qualified': False, 'main_app_default_changed': False,
            'quality_status': P.QUALITY_STATUS, **engine.restorer.measured_status()}


@router.post('/inspect')
async def inspect(file: UploadFile = File(...)):
    try:
        source_check(); rgb = preparation.decode_original(await file.read(preparation.MAX_UPLOAD_BYTES + 1))
        return {'width': rgb.shape[1], 'height': rgb.shape[0], 'source': preparation.url(preparation.png(rgb)), 'neural_calls': 0}
    except ValueError as error: raise HTTPException(422, str(error)) from error


@router.post('/preview')
async def preview_route(file: UploadFile = File(...), protect_mask: UploadFile = File(...), apply_mask: UploadFile = File(...), radius: float = Form(6.)):
    try:
        return (await run_in_threadpool(preview, await file.read(preparation.MAX_UPLOAD_BYTES + 1),
                await protect_mask.read(preparation.MAX_MASK_BYTES + 1), await apply_mask.read(preparation.MAX_MASK_BYTES + 1), radius))[0]
    except ValueError as error: raise HTTPException(422, str(error)) from error


@router.post('/compare')
async def compare_route(file: UploadFile = File(...), protect_mask: UploadFile = File(...), apply_mask: UploadFile = File(...),
                        radius: float = Form(6.), preview_id: str = Form(''), confirmed: bool = Form(False),
                        input_review: str = Form(''), selection: str = Form('incumbent_U96')):
    try:
        return await run_in_threadpool(engine.compare, await file.read(preparation.MAX_UPLOAD_BYTES + 1),
                await protect_mask.read(preparation.MAX_MASK_BYTES + 1), await apply_mask.read(preparation.MAX_MASK_BYTES + 1),
                radius, preview_id, confirmed, input_review, selection)
    except ValueError as error: raise HTTPException(422, str(error)) from error
    except OSError as error: raise HTTPException(503, 'A reviewed source or model file is unavailable.') from error
    except RuntimeError as error: raise HTTPException(409, str(error)) from error


@router.post('/bypass')
async def bypass_route(file: UploadFile = File(...), input_review: str = Form(''), selection: str = Form('incumbent_U96')):
    try:
        return await run_in_threadpool(engine.compare, await file.read(preparation.MAX_UPLOAD_BYTES + 1),
                                      b'', b'', 0., '', False, input_review, selection, True)
    except ValueError as error: raise HTTPException(422, str(error)) from error
    except OSError as error: raise HTTPException(503, 'A reviewed source or model file is unavailable.') from error
    except RuntimeError as error: raise HTTPException(409, str(error)) from error

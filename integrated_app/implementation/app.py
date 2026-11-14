"""Loopback restoration application. Models live only in disposable workers."""
from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from typing import Annotated

import numpy as np
from PIL import Image, ImageFilter
import psutil
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .preparation import MAX_MASK_BYTES, MAX_UPLOAD_BYTES, build_preview, decode_original, png
from .quality import quality_signals
from .runtime import model_contract, prepare_crop
from .setup_models import ROOT, verify_models
from .managed_paths import data_path

app = FastAPI(title="Forensic face restoration — IMPLEMENTATION")
app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')
LOCK = threading.Lock()
STATE = {'neural_forward_attempts': 0, 'neural_forward_returns': 0,
         'active_worker': False, 'uncertain': False, 'last_worker_pid': None}
ARMS = {'v2': 'conditioned_sft_v2_p2_u0224'}
DEFAULT_MODEL = json.loads((ROOT / 'implementation/models.json').read_bytes())['default_model']
REQUEST_LIMIT = 360
MAX_BUNDLE = 64 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def url(value, mime='image/png'):
    return 'data:' + mime + ';base64,' + base64.b64encode(value).decode('ascii')


def bundle(payloads):
    need(sum(map(len, payloads.values())) <= MAX_BUNDLE, 'Evidence bundle exceeds 64 MiB.')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)
    need(stream.tell() <= MAX_BUNDLE, 'Compressed evidence bundle exceeds 64 MiB.')
    return stream.getvalue()


def review_input(value):
    if value == 'needs_clearer':
        raise ValueError('Facial structure is insufficient. Upload a clearer or less-covered face crop.')
    if value == 'out_of_scope':
        raise ValueError('Upload one already cropped frontal or mildly turned face.')
    need(value in ('usable', 'picture_upload'), 'Choose a picture before continuing.')


def picture_native(contents, rectangle=None):
    image = decode_original(contents)
    if rectangle is None:
        return image
    need(isinstance(rectangle, list) and len(rectangle) == 4 and all(type(n) is int for n in rectangle),
         'Choose a valid picture crop.')
    x, y, width, height = rectangle
    need(x >= 0 and y >= 0 and min(width, height) >= 16 and
         x + width <= image.shape[1] and y + height <= image.shape[0], 'Picture crop is outside the allowed area.')
    return image[y:y + height, x:x + width].copy()


@lru_cache(maxsize=1)
def _frozen_frame_pixelation(verified_frame):
    """Share the most recent verified frame analysis across its queued crops."""
    from .automatic_preparation import prepare_source
    from .capture_api import _image

    source = np.asarray(_image(verified_frame))
    return source, prepare_source(source)


def picture_rectangle(value):
    if not value:
        return None
    need(len(value) <= 64, 'Choose a valid picture crop.')
    rectangle = json.loads(value)
    need(isinstance(rectangle, list), 'Choose a valid picture crop.')
    return rectangle


def prepared_preview(selected, original, protect, apply, radius, rectangle, *, exact_marks, match_blur):
    response, payloads = build_preview(selected, protect, apply, nominal_radius=radius, exact_marks=exact_marks,
                                       match_blur=match_blur, full_picture=original if rectangle is not None else None,
                                       rectangle=rectangle)
    if rectangle is not None:
        crop_preview_id = response['preview_id']
        original_hash = digest(original)
        response['preview_id'] = digest(canonical([crop_preview_id, original_hash, rectangle]))
        payloads['uploaded-original.bin'] = original
        payloads['picture-crop.json'] = canonical({'rectangle': rectangle, 'uploaded_bytes_sha256': original_hash})
        payloads['crop-input.png'] = selected
        metadata = response['metadata']
        metadata.update(preview_id=response['preview_id'], picture_crop=rectangle,
                        original_upload_sha256=original_hash, crop_input_sha256=digest(selected),
                        crop_preview_id=crop_preview_id)
        for name in ('uploaded-original.bin', 'picture-crop.json', 'crop-input.png'):
            raw = payloads[name]
            metadata['payload_pins'][name] = {'bytes': len(raw), 'sha256': digest(raw)}
        payloads['processing.json'] = canonical(metadata)
    return response, payloads


def invoke(function, *args):
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, 'A request is already running. Wait for it to finish.')
    try:
        if STATE['uncertain']:
            raise HTTPException(409, 'The previous worker stopped unexpectedly. Inspect scratch/implementation_requests and restart start.bat before retrying.')
        return function(*args)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except (RuntimeError, OSError) as error:
        raise HTTPException(503, str(error)) from error
    finally:
        LOCK.release()


def one_forward(contents, model):
    need(model in ARMS, 'Only the V2 (p2_u0224) model is supported.')
    if psutil.virtual_memory().available < 2684354560:
        raise RuntimeError('Restoration needs at least 2.5 GiB available RAM. Close unused applications and retry.')
    verify_models()
    request_dir = data_path(ROOT, 'scratch', 'implementation_requests') / uuid.uuid4().hex
    request_dir.mkdir(parents=True)
    image_path = request_dir / 'input.bin'
    image_path.write_bytes(contents)
    job = {'input_path': str(image_path), 'model': model, 'output_dir': str(request_dir / 'output')}
    request_file, result_file = request_dir / 'request.json', request_dir / 'result.json'
    request_file.write_bytes(canonical(job))
    process = None
    STATE['active_worker'] = True
    try:
        with (request_dir / 'worker.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, '-B', '-m', 'implementation.model_worker',
                                        '--request', str(request_file), '--result', str(result_file)],
                                       cwd=ROOT, stdout=log, stderr=log,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            STATE['last_worker_pid'] = process.pid
            started = time.monotonic()
            while process.poll() is None:
                if time.monotonic() - started > REQUEST_LIMIT:
                    raise RuntimeError('Restoration exceeded the 360-second limit. Evidence was retained; no automatic retry.')
                try:
                    if psutil.Process(process.pid).memory_info().rss > 6 * 1024**3:
                        raise RuntimeError('Restoration exceeded the 6 GiB worker memory limit.')
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.15)
        if process.returncode != 0 or not result_file.is_file():
            raise RuntimeError('The restoration worker failed. Evidence is retained in scratch/implementation_requests; restart the application after reviewing the error.')
        need(result_file.stat().st_size < 65536, 'Worker result metadata is too large.')
        result = json.loads(result_file.read_bytes())
        output = request_dir / 'output'
        need(Path(result['raw_path']).resolve() == (output / 'raw.npy').resolve() and
             Path(result['png_path']).resolve() == (output / 'restoration.png').resolve(), 'Worker output paths differ.')
        status = json.loads((output / 'status.json').read_bytes())
        need(status['phase'] == 'complete' and status['forward_attempts'] == status['forward_returns'] == 1,
             'One known completed forward is required.')
        raw_bytes, png_bytes = (output / 'raw.npy').read_bytes(), (output / 'restoration.png').read_bytes()
        need(len(raw_bytes) < 1024**2 and len(png_bytes) < 1024**2, 'Worker output exceeds bounds.')
        raw = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
        with Image.open(io.BytesIO(png_bytes)) as image:
            need(image.mode == 'RGB' and image.size == (256, 256), 'Expected RGB256 output.')
            pixels = np.array(image)
        canvas, observed, _, geometry = prepare_crop(decode_original(contents))
        need(raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
             and 0 <= raw.min() <= raw.max() <= 1, 'Invalid RAW output.')
        expected = np.floor(raw * np.float32(255)).astype(np.uint8)
        expected[~observed] = canvas[~observed]
        metadata = result['metadata']
        need(np.array_equal(expected, pixels) and np.array_equal(raw[~observed], canvas[~observed].astype(np.float32) / np.float32(255)),
             'Delivered RAW/PNG or padding preservation differs.')
        need(metadata['uploaded_bytes_sha256'] == digest(contents) and metadata['geometry'] == geometry
             and metadata['raw_rgb_sha256'] == digest(raw.tobytes())
             and metadata['PNG_RGB8_sha256'] == digest(pixels.tobytes())
             and metadata['PNG_file_sha256'] == digest(png_bytes), 'Worker delivery hashes differ.')
        return raw_bytes, png_bytes, metadata
    except BaseException:
        STATE['uncertain'] = True
        raise
    finally:
        cleanup_error = None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            STATE['uncertain'] = True
            cleanup_error = error
        finally:
            status_file = request_dir / 'output/status.json'
            if status_file.is_file() and status_file.stat().st_size < 16384:
                try:
                    known = json.loads(status_file.read_bytes())
                    attempts, returns = known['forward_attempts'], known['forward_returns']
                    if type(attempts) is int and type(returns) is int and 0 <= returns <= attempts <= 1:
                        STATE['neural_forward_attempts'] += attempts
                        STATE['neural_forward_returns'] += returns
                    else:
                        STATE['uncertain'] = True
                except (ValueError, KeyError, OSError):
                    STATE['uncertain'] = True
            STATE['active_worker'] = process is not None and process.poll() is None
        if cleanup_error is not None:
            raise RuntimeError('The owned worker could not be confirmed stopped. Do not retry; inspect its retained request and close the owned process before restarting.') from cleanup_error


@app.get('/health')
def health():
    return {'status': 'ok', 'default_model': DEFAULT_MODEL, 'model_loaded_in_server': False,
            'model_qualified': False, 'busy': LOCK.locked(), 'uncertain': STATE['uncertain']}


_STAGE_REASONS = {
    'noise_denoise': 'suspected_additive_or_sensor_noise',
    'cast_dehaze': 'suspected_color_cast_and_haze',
    'color_cast_balance': 'suspected_color_cast_imbalance',
    'haze_dehaze': 'suspected_atmospheric_haze',
    'spatial_comb': 'suspected_spatial_interlacing_comb',
    'macroblock_fbcnn': 'genuine_8x8_macroblocking_fbcnn',
    'macroblock_hevc': 'genuine_macroblocking_hevc_deblocking',
    'edge_deringing': 'suspected_edge_ringing_halos',
    'gradient_debanding': 'suspected_quantized_gradient_banding',
    'tangent_antialiasing': 'suspected_diagonal_staircase_aliasing',
    'chroma_debleeding': 'suspected_luma_chroma_color_bleeding',
}


def _jpeg_luma_dc(contents):
    if not contents or not contents.startswith(b'\xff\xd8'):
        return None
    with Image.open(io.BytesIO(contents)) as source:
        if source.format != 'JPEG':
            return None
        luma = source.quantization.get(0, [])
        return int(luma[0]) if len(luma) == 64 else None


def _face_mask_for_native(native_shape, face, origin):
    height, width = native_shape[:2]
    fx, fy, fw, fh = face
    ox, oy = origin
    x0 = max(0, int(round(fx - ox)))
    y0 = max(0, int(round(fy - oy)))
    x1 = min(width, int(round(fx + fw - ox)))
    y1 = min(height, int(round(fy + fh - oy)))
    if x1 <= x0 or y1 <= y0 or min(height, width) < 32:
        return None
    if y1 - y0 < 32:
        pad = 32 - (y1 - y0)
        y0 = max(0, y0 - pad // 2)
        y1 = min(height, y0 + 32)
        y0 = max(0, y1 - 32)
    if x1 - x0 < 32:
        pad = 32 - (x1 - x0)
        x0 = max(0, x0 - pad // 2)
        x1 = min(width, x0 + 32)
        x0 = max(0, x1 - 32)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _package_artifact_stages(current_rgb, native, applied_stages, stage_reports, stage_evidence, payloads):
    selected = png(current_rgb)
    prepared, _, _, _ = prepare_crop(current_rgb)
    changed_pixels = int(np.any(current_rgb != native, axis=2).sum())
    selected_sha = digest(selected)
    if len(applied_stages) == 1 and applied_stages[0] == 'jpeg_deblocking':
        jpeg_report = stage_reports['jpeg_deblocking']
        preparation = {**jpeg_report, 'applied_stages': ['jpeg_deblocking'],
                       'crop_changed_native_pixels': changed_pixels, 'derived_native_sha256': selected_sha}
        auto_decision = {'route': 'jpeg_deblocking', 'reason': 'strong_native_jpeg_grid',
                         'preparation_model': jpeg_report['preparation_model'],
                         'applied_stages': ['jpeg_deblocking']}
    elif len(applied_stages) == 1 and applied_stages[0] == 'scanline_cleanup':
        scanline_report = stage_reports['scanline_cleanup']
        preparation = {**scanline_report, 'applied_stages': ['scanline_cleanup'],
                       'crop_changed_native_pixels': changed_pixels, 'derived_native_sha256': selected_sha}
        auto_decision = {'route': 'scanline_cleanup', 'reason': 'strong_horizontal_scanline_pattern',
                         'spacing_native_pixels': scanline_report['spacing_native_pixels'],
                         'applied_stages': ['scanline_cleanup']}
    elif len(applied_stages) == 1:
        stage = applied_stages[0]
        report = stage_reports[stage]
        preparation = {**report, 'route': stage, 'applied_stages': list(applied_stages),
                       'stage_reports': stage_reports, 'crop_changed_native_pixels': changed_pixels,
                       'derived_native_sha256': selected_sha}
        auto_decision = {'route': stage, 'reason': _STAGE_REASONS[stage],
                         'applied_stages': list(applied_stages), 'stage_reports': stage_reports}
        if stage == 'spatial_comb' and 'spacing_native_pixels' in report:
            auto_decision['spacing_native_pixels'] = report['spacing_native_pixels']
        for key in ('preparation_model', 'preparation_model_sha256', 'preparation_forward_returns'):
            if key in report:
                auto_decision[key] = report[key]
    else:
        preparation = {'policy': 'compound-artifact-preparation-v1', 'route': 'compound_artifact_preparation',
                       'stages': list(applied_stages), 'applied_stages': list(applied_stages),
                       'stage_reports': stage_reports, 'crop_changed_native_pixels': changed_pixels,
                       'derived_native_sha256': selected_sha}
        auto_decision = {'route': 'compound_artifact_preparation',
                         'reason': 'multiple_degradation_stages_applied',
                         'stages': list(applied_stages), 'applied_stages': list(applied_stages),
                         'stage_reports': stage_reports}
        for comb_stage in ('scanline_cleanup', 'spatial_comb'):
            if comb_stage in stage_reports and 'spacing_native_pixels' in stage_reports[comb_stage]:
                spacing = stage_reports[comb_stage]['spacing_native_pixels']
                preparation['spacing_native_pixels'] = spacing
                auto_decision['spacing_native_pixels'] = spacing
                break
        for model_stage in ('macroblock_fbcnn', 'jpeg_deblocking'):
            if model_stage in stage_reports:
                for key in ('preparation_model', 'preparation_model_sha256', 'preparation_forward_returns'):
                    if key in stage_reports[model_stage]:
                        preparation[key] = stage_reports[model_stage][key]
                        auto_decision[key] = stage_reports[model_stage][key]
    payloads.update(stage_evidence)
    payloads.update({'derived-native.png': selected, 'processing.json': canonical(preparation)})
    macroblock_report = stage_reports.get('macroblock_fbcnn') or stage_reports.get('macroblock_hevc')
    return selected, prepared, preparation, auto_decision, macroblock_report


@app.get('/', response_class=HTMLResponse)
@app.get('/restoration-workflow', response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT / 'templates/implementation.html').read_text(encoding='utf8'),
                        headers={'Cache-Control': 'no-store'})


@app.get('/favicon.ico')
def favicon():
    return Response(status_code=204)


@app.get('/face/restoration/status')
def status():
    available = not LOCK.locked() and not STATE['uncertain']
    return {'available': available, 'maximum_restore_forwards': 1, 'maximum_off_forwards': 0,
            'automatic_preparation': False, 'default_model': DEFAULT_MODEL,
            'controller': {'selection': ARMS[DEFAULT_MODEL], 'worker': dict(STATE)}}


@app.post('/face/restoration/inspect')
async def inspect(file: UploadFile = File(...), scan_faces: Annotated[bool, Form()] = False):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    def execute():
        image = decode_original(contents)
        height, width = image.shape[:2]
        result = {'width': width, 'height': height, 'source': url(png(image)),
                  'neural_calls': 0, 'input_review_required': False}
        if scan_faces:
            from . import capture_api
            use_details = capture_api.detect_faces is capture_api._DEFAULT_DETECT_FACES
            records = []
            faces = []
            unavailable = False
            if capture_api.DETECTION_LOCK.acquire(blocking=False):
                try:
                    pil_image = Image.fromarray(image)
                    if use_details:
                        records = capture_api.detect_face_details(pil_image)
                    else:
                        faces = capture_api.detect_faces(pil_image)
                except RuntimeError:
                    unavailable = True
                finally:
                    capture_api.DETECTION_LOCK.release()
            else:
                unavailable = True
            if use_details:
                sorted_recs = sorted(records, key=lambda rec: (rec['box'][0], rec['box'][1]))
                result['face_crops'] = [rec['crop'] for rec in sorted_recs]
                result['face_boxes'] = [rec['box'] for rec in sorted_recs]
            else:
                sorted_faces = sorted(faces, key=lambda box: (box[0], box[1]))
                result['face_crops'] = []
                for x, y, w, h in sorted_faces:
                    margin = int(max(w, h) * .20 + .5)
                    left, top = max(0, x - margin), max(0, y - margin)
                    right, bottom = min(width, x + w + margin), min(height, y + h + margin)
                    result['face_crops'].append([left, top, right - left, bottom - top])
                result['face_boxes'] = sorted_faces
            result['face_scan_unavailable'] = unavailable
        return result
    return await run_in_threadpool(invoke, execute)


@app.post('/face/restoration/preview')
async def make_preview(file: UploadFile = File(...), protect_mask: UploadFile = File(...),
                       apply_mask: UploadFile = File(...), radius: float = Form(1), input_review: str = Form(''),
                       picture_crop: Annotated[str, Form()] = '', match_blur: Annotated[bool, Form()] = False):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    protect, apply = await protect_mask.read(MAX_MASK_BYTES + 1), await apply_mask.read(MAX_MASK_BYTES + 1)
    def execute():
        review_input(input_review)
        need(not picture_crop or input_review == 'picture_upload', 'A picture crop requires a picture upload.')
        rectangle = picture_rectangle(picture_crop)
        selected = png(picture_native(contents, rectangle)) if rectangle is not None else contents
        response, payloads = prepared_preview(selected, contents, protect, apply, radius, rectangle,
                                               exact_marks=input_review == 'picture_upload', match_blur=match_blur)
        return {**response, 'bundle': url(bundle(payloads), 'application/zip'), 'neural_calls': 0}
    return await run_in_threadpool(invoke, execute)


@app.post('/face/restoration/restore')
async def restore(file: UploadFile = File(...), input_review: str = Form(''),
                  restoration: str = Form('auto'), preparation_mode: str = Form('bypass'),
                  protect_mask: UploadFile | None = File(None), apply_mask: UploadFile | None = File(None),
                  radius: float = Form(1), preview_id: str = Form(''), confirmed: bool = Form(False),
                  include_bundle: bool = Form(True), model: str = Form(DEFAULT_MODEL),
                  capture_id: Annotated[str, Form()] = '', request: Request = None,
                  picture_crop: Annotated[str, Form()] = '', match_blur: Annotated[bool, Form()] = False,
                  include_smoothed_frame: Annotated[bool, Form()] = True):
    video_selection = input_review == 'video_capture'
    owner = None
    if video_selection or capture_id:
        from .capture_api import _guard
        if request is None:
            raise HTTPException(403, 'Capture session token required.')
        owner = _guard(request)
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    protect = await protect_mask.read(MAX_MASK_BYTES + 1) if protect_mask else None
    apply = await apply_mask.read(MAX_MASK_BYTES + 1) if apply_mask else None
    def execute():
        started = time.monotonic()
        automatic = preparation_mode == 'automatic'
        need(not automatic or (restoration == 'auto' and input_review in ('picture_upload', 'video_capture')),
             'Automatic preparation requires Auto and a picture or saved video crop.')
        if video_selection:
            need(bool(capture_id) and ((preparation_mode == 'bypass' and restoration in ('on', 'off')) or automatic),
                 'Video restoration requires a saved capture and a supported restoration setting.')
        else:
            review_input(input_review)
        need(model in ARMS and restoration in ('auto', 'on', 'off'), 'Choose a valid model and Auto, On or Off.')
        need(preparation_mode in ('manual', 'bypass', 'automatic'), 'Choose a valid preparation mode.')
        need(not picture_crop or (input_review == 'picture_upload' and not capture_id),
             'A picture crop requires a picture upload.')
        rectangle = picture_rectangle(picture_crop)
        native = picture_native(contents, rectangle)
        need(np.ptp(native.reshape(-1, 3).astype(np.int16), axis=0).max() > 0,
             'No usable structure is visible. Upload a clearer face crop.')
        original, observed, _, geometry = prepare_crop(native)
        signals = quality_signals(original, (~observed).astype(np.uint8), native_rgb=native)
        if restoration == 'auto' and not automatic and signals['blur_variance'] is None:
            raise ValueError('Too little area for Auto. Choose On or Off, or upload a clearer crop.')
        applied = automatic or restoration == 'on' or (restoration == 'auto' and signals['suggest_restoration'])
        payloads = {'uploaded-original.bin': contents, 'original-native.png': png(native),
                    'original-256.png': png(original)}
        if capture_id:
            from .capture_api import evidence_for_crop
            payloads.update(evidence_for_crop(contents, capture_id, owner=owner))
        selected, prepared, preparation = png(native) if rectangle is not None else contents, original, None
        if rectangle is not None:
            payloads['picture-crop.json'] = canonical({'rectangle': rectangle, 'uploaded_bytes_sha256': digest(contents)})
        auto_decision = picture_regions = frame_regions = smoothed_frame_png = None
        scanline_report = cleaned_full_image = None
        deblocked = jpeg_report = macroblock_report = None
        softened_detection = None
        tiny_report = None
        if automatic:
            need(protect is None and apply is None and not preview_id and not confirmed,
                 'Auto selects its own preparation. Remove manual preparation fields.')
            from .automatic_preparation import (
                prepare as prepare_automatically,
                prepare_composables,
                prepare_macroblocking,
                prepare_noise,
                prepare_photometric,
                prepare_source,
                pixelation_signal,
                tiny_crop_pixelation,
            )
            from .capture_api import detect_faces
            selected_rectangle = rectangle
            detection_method = 'original_image'
            if capture_id:
                source_rgb, (smoothed_image, regional_report, region_evidence) = _frozen_frame_pixelation(payloads['capture/frame.png'])
                selected_rectangle = json.loads(payloads['capture/metadata.json'])['rectangle']
                if regional_report is not None:
                    regional_report = {**regional_report, 'frame_sha256': digest(payloads['capture/frame.png'])}
            else:
                source_rgb = decode_original(contents)
                smoothed_image, regional_report, region_evidence = prepare_source(source_rgb)
                if 'scanline-cleaned.png' not in region_evidence:
                    from .jpeg_deblocking import prepare_jpeg
                    from .mixed_jpeg import prepare_mixed_jpeg
                    mixed = prepare_mixed_jpeg(contents, source_rgb, rectangle)
                    if mixed is not None:
                        smoothed_image, regional_report, region_evidence = mixed
                    else:
                        deblocked, jpeg_report = prepare_jpeg(contents, native, rectangle)
                payloads.update(region_evidence)

            if 'scanline-cleaned.png' in region_evidence:
                scanline_report = regional_report
                cleaned_full_image = region_evidence['scanline-cleaned.png']
                if capture_id and 'scanline-mosaic-mask.png' in region_evidence:
                    payloads['capture/frame-scanline-mosaic-mask.png'] = region_evidence['scanline-mosaic-mask.png']
            elif 'softened-mosaic-mask.png' in region_evidence:
                mask_contents = region_evidence['softened-mosaic-mask.png']
                with Image.open(io.BytesIO(mask_contents)) as mask_image:
                    if selected_rectangle is not None:
                        x, y, width, height = selected_rectangle
                        mask_image = mask_image.crop((x, y, x + width, y + height))
                    intersects = mask_image.getbbox() is not None
                softened_detection = {**regional_report, 'crop_intersects': intersects}
                if capture_id:
                    payloads['capture/frame-softened-mosaic-mask.png'] = mask_contents
            elif regional_report is not None:
                if capture_id:
                    frame_regions = regional_report
                    smoothed_frame_png = region_evidence['full-image-smoothed.png']
                    payloads['capture/frame-pixelation-mask.png'] = region_evidence['full-image-pixelation-mask.png']
                    if 'irregular-region-mask.png' in region_evidence:
                        payloads['capture/frame-irregular-region-mask.png'] = region_evidence['irregular-region-mask.png']
                else:
                    picture_regions = regional_report

            region_input = deblocked
            if region_input is None and smoothed_image is not None:
                region_input = smoothed_image
                if selected_rectangle is not None:
                    x, y, width, height = selected_rectangle
                    region_input = smoothed_image[y:y + height, x:x + width]
                if np.array_equal(region_input, native):
                    region_input = None
            if region_input is None and scanline_report is None:
                region_input, tiny_report, tiny_evidence = tiny_crop_pixelation(native)
                if tiny_report is not None:
                    payloads['tiny-crop-pixelation-mask.png'] = tiny_evidence['full-image-pixelation-mask.png']

            is_privacy_mosaic_region = (
                region_input is not None
                and (
                    tiny_report is not None
                    or (deblocked is None and scanline_report is None)
                    or (scanline_report is not None and 'mosaic' in scanline_report)
                )
            )
            if is_privacy_mosaic_region:
                selected = png(region_input)
                prepared, _, _, _ = prepare_crop(region_input)
                preparation = {**(tiny_report or regional_report),
                               'crop_changed_native_pixels': int(np.any(region_input != native, axis=2).sum()),
                               'derived_native_sha256': digest(selected)}
                if scanline_report is not None:
                    auto_decision = {'route': 'scanline_mosaic_coordinated',
                                     'reason': 'strong_horizontal_scanline_pattern',
                                     'spacing_native_pixels': scanline_report['spacing_native_pixels']}
                elif tiny_report is not None:
                    auto_decision = {'route': 'tiny_mosaic_gaussian', 'reason': 'strong_grid_in_tiny_crop',
                                     'tiny_grid_period_native': tiny_report['tiny_grid_period_native'],
                                     'Gaussian_radius_at256': tiny_report['Gaussian_radius_at256']}
                elif regional_report.get('mixed_jpeg_mosaic'):
                    auto_decision = {'route': 'jpeg_mosaic_coordinated', 'reason': 'independent_mosaic_and_jpeg_evidence',
                                     'mosaic_grid_period_native': regional_report['mosaic_grid_period_native'],
                                     'outside_deblock_fraction': regional_report['outside_deblock_fraction']}
                elif regional_report.get('coordinated_mosaic_regions'):
                    auto_decision = {'route': 'mosaic_regions_coordinated', 'reason': 'spatially_distinct_mosaic_regions'}
                elif pattern := regional_report.get('geometric_mosaic_pattern'):
                    auto_decision = {'route': f'{pattern}_mosaic_gaussian', 'reason': f'suspected_{pattern}_mosaic',
                                     'Gaussian_radius_at256': regional_report['Gaussian_radius_at256']}
                elif 'rotation_correction_degrees' in regional_report:
                    auto_decision = {'route': 'rotated_mosaic_gaussian', 'reason': 'suspected_rotated_mosaic',
                                     'rotation_correction_degrees': regional_report['rotation_correction_degrees'],
                                     'Gaussian_radius_at256': regional_report['Gaussian_radius_at256']}
                else:
                    auto_decision = {'route': 'frame_region_gaussian' if capture_id else 'picture_region_gaussian',
                                     'reason': 'suspected_pixelation_in_model_input',
                                     'Gaussian_radius_at256': regional_report['Gaussian_radius_at256'],
                                     'strength_basis': regional_report.get('strength_selection', {}).get('basis')}
                payloads.update({'derived-native.png': selected, 'processing.json': canonical(preparation)})
            elif softened_detection is not None and softened_detection['crop_intersects']:
                auto_decision = {'route': 'direct', 'reason': 'suspected_softened_mosaic'}
            else:
                applied_stages = []
                stage_reports = {}
                stage_evidence = {}
                current_rgb = region_input.copy() if region_input is not None else native.copy()
                if jpeg_report is not None:
                    applied_stages.append('jpeg_deblocking')
                    stage_reports['jpeg_deblocking'] = jpeg_report
                elif scanline_report is not None and region_input is not None:
                    applied_stages.append('scanline_cleanup')
                    stage_reports['scanline_cleanup'] = scanline_report

                noise_out, noise_report, noise_evidence = prepare_noise(current_rgb)
                if noise_out is not None and noise_report is not None:
                    current_rgb = noise_out
                    applied_stages.append('noise_denoise')
                    stage_reports['noise_denoise'] = noise_report
                    stage_evidence.update(noise_evidence)

                cd_out, cd_route, cd_report, cd_evidence = prepare_photometric(current_rgb)
                if cd_out is not None and cd_route is not None and cd_report is not None:
                    current_rgb = cd_out
                    applied_stages.append(cd_route)
                    stage_reports[cd_route] = cd_report
                    stage_evidence.update(cd_evidence)

                crop_origin = (int(selected_rectangle[0]), int(selected_rectangle[1])) if selected_rectangle is not None else (0, 0)
                macroblock_dc = None if capture_id else _jpeg_luma_dc(contents)
                if jpeg_report is None:
                    mb_out, mb_stage, mb_report, mb_evidence = prepare_macroblocking(
                        current_rgb, origin=crop_origin, allow_fbcnn=True, dc=macroblock_dc
                    )
                    if mb_out is not None and mb_stage is not None:
                        current_rgb = mb_out
                        applied_stages.append(mb_stage)
                        stage_reports[mb_stage] = mb_report
                        stage_evidence.update(mb_evidence)

                comp_out, comp_stages, comp_reports, comp_evidence = prepare_composables(
                    current_rgb, allow_spatial_comb=(scanline_report is None)
                )
                if comp_out is not None and comp_stages:
                    current_rgb = comp_out
                    applied_stages.extend(comp_stages)
                    stage_reports.update(comp_reports)
                    stage_evidence.update(comp_evidence)

                if scanline_report is None and 'spatial_comb' in stage_reports:
                    scanline_report = stage_reports['spatial_comb']

                if applied_stages:
                    selected, prepared, preparation, auto_decision, macroblock_report = _package_artifact_stages(
                        current_rgb, native, applied_stages, stage_reports, stage_evidence, payloads
                    )
                else:
                    def in_selection(boxes):
                        if selected_rectangle is None:
                            return boxes
                        x, y, width, height = selected_rectangle
                        return [box for box in boxes if x <= box[0] + box[2] / 2 < x + width
                                and y <= box[1] + box[3] / 2 < y + height]

                    scan_image = Image.fromarray(source_rgb)
                    try:
                        faces = in_selection(detect_faces(scan_image))
                        if not faces:
                            # Retry detection on copies. Preparation still requires
                            # a block-pattern signal on the untouched source pixels.
                            scale = max(scan_image.size) / 256
                            for scan_radius in (min(2., max(.5, scale * .9)), min(12., max(2.5, scale * 3.3))):
                                smoothed = scan_image.filter(ImageFilter.GaussianBlur(scan_radius))
                                matches = in_selection(detect_faces(smoothed))
                                if len(matches) > 1:
                                    break
                                if matches and pixelation_signal(source_rgb, matches[0])['suspected']:
                                    faces = matches
                                    detection_method = 'smoothed_detection_copy_with_block_signal'
                                    break
                    except RuntimeError:
                        faces = []
                    face = faces[0] if len(faces) == 1 else None
                    if face is None:
                        auto_decision = {'route': 'direct', 'reason': 'face_detection_uncertain'}
                    else:
                        face_mask = _face_mask_for_native(native.shape, face, crop_origin) if jpeg_report is None else None
                        if face_mask is not None:
                            mb_out, mb_stage, mb_report, mb_evidence = prepare_macroblocking(
                                native, origin=crop_origin, mask=face_mask, allow_fbcnn=True, dc=macroblock_dc
                            )
                        else:
                            mb_out, mb_stage, mb_report, mb_evidence = None, None, None, {}
                        if mb_out is not None and mb_stage is not None:
                            applied_stages = [mb_stage]
                            stage_reports = {mb_stage: mb_report}
                            stage_evidence = dict(mb_evidence)
                            comp_out, comp_stages, comp_reports, comp_evidence = prepare_composables(
                                mb_out, allow_spatial_comb=(scanline_report is None)
                            )
                            final_rgb = mb_out
                            if comp_out is not None and comp_stages:
                                final_rgb = comp_out
                                applied_stages.extend(comp_stages)
                                stage_reports.update(comp_reports)
                                stage_evidence.update(comp_evidence)
                            if scanline_report is None and 'spatial_comb' in stage_reports:
                                scanline_report = stage_reports['spatial_comb']
                            selected, prepared, preparation, auto_decision, macroblock_report = _package_artifact_stages(
                                final_rgb, native, applied_stages, stage_reports, stage_evidence, payloads
                            )
                        else:
                            candidate, preparation, auto_evidence, auto_decision = prepare_automatically(native, source_rgb, face, selected_rectangle)
                            auto_decision['face_detection'] = detection_method
                            payloads.update(auto_evidence)
                            if candidate is not None:
                                selected = candidate
                                prepared, _, _, _ = prepare_crop(decode_original(selected))
            payloads['automatic-decision.json'] = canonical(auto_decision)
        elif preparation_mode == 'bypass':
            need(protect is None and apply is None and not preview_id and not confirmed,
                 'Bypass uses the original. Remove manual preparation fields.')
        elif applied:
            need(protect is not None and apply is not None and confirmed,
                 'Preview and confirm the selected areas first.')
            response, prepared_payloads = prepared_preview(selected, contents, protect, apply, radius, rectangle,
                                                           exact_marks=input_review == 'picture_upload', match_blur=match_blur)
            need(preview_id == response['preview_id'], 'The preview is stale. Preview and confirm the current areas again.')
            payloads.update(prepared_payloads)
            selected, preparation = prepared_payloads['derived-native.png'], response['metadata']
            prepared, _, _, _ = prepare_crop(decode_original(selected))
        raw_bytes, output, worker = None, png(original), None
        if applied:
            raw_bytes, output, worker = one_forward(selected, model)
        output_signals = quality_signals(decode_original(output), (~observed).astype(np.uint8)) if applied else signals
        mask_u8 = (~observed).astype(np.uint8)
        before_diag = {name: signals[name] for name in ('blur_variance', 'noise_sigma_255')}
        noise_native = signals.get('noise_sigma_native')
        if isinstance(noise_native, (int, float)) and not isinstance(noise_native, bool) and math.isfinite(noise_native):
            before_diag['noise_sigma_native'] = noise_native
        diagnostics = {'before': before_diag}
        if applied and preparation is not None and prepared is not None:
            prepared_signals = quality_signals(prepared, mask_u8)
            diagnostics['prepared'] = {
                name: prepared_signals[name] for name in ('blur_variance', 'noise_sigma_255')
            }
        diagnostics['after'] = {
            name: output_signals[name] for name in ('blur_variance', 'noise_sigma_255')
        }
        metadata = {'selection': ARMS[model], 'model': model, 'restoration_requested': restoration,
                    'input_review': input_review,
                    'restoration_applied': bool(applied), 'preparation_applied': preparation is not None,
                    'preparation': preparation, 'preparation_requested': preparation_mode,
                    'preparation_bypass_reason': ('restoration_off' if restoration == 'off' else
                         'Auto_did_not_suggest_restoration' if not applied else
                         'operator_bypass' if preparation_mode == 'bypass' else None),
                    'quality_signals': signals, 'output_diagnostics': diagnostics,
                    'Auto_assessed_original_canvas_before_preparation': bool(applied and automatic),
                    'neural_forward_attempts': int(applied), 'neural_forward_returns': int(applied),
                    'seconds': time.monotonic() - started, 'uploaded_bytes_sha256': digest(contents),
                    'selected_input_sha256': digest(selected), 'result_file_sha256': digest(output),
                    'geometry': geometry, 'worker_delivery': worker, 'model_qualified': False,
                    'protection_scope': 'Preprocessing input only; generated appearance may change.',
                    'training_optimizer_VM_calls': 0, 'capture_id': capture_id or None}
        if rectangle is not None:
            metadata['picture_crop'] = rectangle
        if auto_decision is not None:
            metadata['automatic_preparation'] = auto_decision
        if picture_regions is not None:
            metadata['picture_pixelation'] = picture_regions
        if frame_regions is not None:
            metadata['frame_pixelation'] = frame_regions
        if scanline_report is not None:
            metadata['scanline_cleanup'] = scanline_report
        if jpeg_report is not None:
            metadata['jpeg_deblocking'] = jpeg_report
        if macroblock_report is not None:
            metadata['macroblock_deblocking'] = macroblock_report
        if softened_detection is not None:
            metadata['softened_mosaic_detection'] = softened_detection
        payloads.update({'prepared-256.png': png(prepared), 'result.png': output,
                         'restoration.json': canonical(metadata)})
        if raw_bytes is not None:
            payloads['restoration-raw-float32.npy'] = raw_bytes
        result = {'original_input': url(png(original)), 'prepared_input': url(png(prepared)),
                  'result': url(output), 'metadata': metadata}
        if picture_regions is not None:
            result['smoothed_full_picture'] = url(payloads['full-image-smoothed.png'])
        if smoothed_frame_png is not None and include_smoothed_frame:
            result['smoothed_full_frame'] = url(smoothed_frame_png)
        if cleaned_full_image is not None and (not capture_id or include_smoothed_frame):
            result['scanline_cleaned_full_image'] = url(cleaned_full_image)
        if include_bundle:
            result['bundle'] = url(bundle(payloads), 'application/zip')
        return result
    return await run_in_threadpool(invoke, execute)

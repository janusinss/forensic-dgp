"""Loopback restoration application. Models live only in disposable workers."""
from __future__ import annotations

import base64
import binascii
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

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import psutil
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .preparation import MAX_MASK_BYTES, MAX_UPLOAD_BYTES, build_preview, decode_original, png
from .quality import quality_signals
from .runtime import MAX_IMAGE_SIDE, model_contract, prepare_crop
from .setup_models import ROOT, verify_models
from .managed_paths import data_path
from .updater import (
    UpdateUnavailableError,
    UpdateValidationError,
    check_for_updates,
    get_current_version,
    install_update,
)

app = FastAPI(title="Forensic face restoration — IMPLEMENTATION")
app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')
LOCK = threading.Lock()
STATE = {'neural_forward_attempts': 0, 'neural_forward_returns': 0,
         'active_worker': False, 'uncertain': False, 'last_worker_pid': None}
ARMS = {'v3': 'preconditioned_v34_codeformer'}
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


def picture_native_with_source_size(contents, rectangle=None):
    image = decode_original(contents)
    source_size = [int(image.shape[1]), int(image.shape[0])]
    if rectangle is None:
        return image, source_size
    need(isinstance(rectangle, list) and len(rectangle) == 4 and all(type(n) is int for n in rectangle),
         'Choose a valid picture crop.')
    x, y, width, height = rectangle
    need(x >= 0 and y >= 0 and min(width, height) >= 16 and
         x + width <= source_size[0] and y + height <= source_size[1], 'Picture crop is outside the allowed area.')
    return image[y:y + height, x:x + width].copy(), source_size


def picture_native(contents, rectangle=None):
    native, _ = picture_native_with_source_size(contents, rectangle)
    return native


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
    need(model in ARMS, 'Only the V3.4 (preconditioned_v34_codeformer) model is supported.')
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


@app.get('/app/version')
def app_version():
    return {'version': get_current_version()}


@app.post('/app/update/check')
async def app_update_check():
    def execute():
        try:
            return check_for_updates()
        except UpdateValidationError as error:
            raise HTTPException(400, str(error)) from error
        except UpdateUnavailableError as error:
            raise HTTPException(502, str(error)) from error
    return await run_in_threadpool(execute)


@app.post('/app/update/install')
async def app_update_install():
    def execute():
        try:
            return install_update()
        except UpdateValidationError as error:
            raise HTTPException(400, str(error)) from error
        except UpdateUnavailableError as error:
            raise HTTPException(502, str(error)) from error
    return await run_in_threadpool(execute)


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


def _recommend_face_index(image, records, width, height):
    """Score sorted face records to pick the primary foreground subject (favoring degraded faces on ties)."""
    if not records:
        return 0
    if len(records) == 1:
        return 0
    from .automatic_preparation import _hf_deficit_stats, pixelation_signal

    best_idx = 0
    best_val = -1.0
    img_h, img_w = image.shape[:2]
    for idx, rec in enumerate(records):
        bx, by, bw, bh = [int(v) for v in rec['box']]
        score = max(0.05, float(rec.get('score', 0.75)))
        area = max(1.0, float(bw * bh))
        cx = (bx + bw * 0.5) / max(1.0, float(width))
        cy = (by + bh * 0.5) / max(1.0, float(height))
        dist = math.hypot(cx - 0.5, cy - 0.45)
        centrality = max(0.35, 1.15 - 0.95 * dist)
        val = (score ** 1.5) * (area ** 0.35) * centrality
        if 0 <= bx and 0 <= by and bx + bw <= img_w and by + bh <= img_h and min(bw, bh) >= 24:
            crop_box = rec.get('crop')
            degraded_bonus = False
            if pixelation_signal(image, (bx, by, bw, bh))['suspected']:
                degraded_bonus = True
            elif isinstance(crop_box, (list, tuple)) and len(crop_box) == 4:
                rx, ry, rw, rh = [int(v) for v in crop_box]
                if (
                    0 <= rx <= bx
                    and 0 <= ry <= by
                    and bx + bw <= rx + rw <= img_w
                    and by + bh <= ry + rh <= img_h
                    and rw >= bw + 12
                    and rh >= bh + 12
                ):
                    hf = _hf_deficit_stats(image[ry:ry + rh, rx:rx + rw], [bx - rx, by - ry, bw, bh])
                    if hf is not None and hf.get('clear'):
                        degraded_bonus = True
            if degraded_bonus:
                val *= 1.35
        if val > best_val:
            best_val = val
            best_idx = idx
    return int(best_idx)


def _fallback_inspect_face_box(image, width, height):
    """Fallback ROI when YuNet finds 0 faces on extreme privacy blur or localized mosaic."""
    from .automatic_preparation import detect_blur_island_face_box, picture_pixelation

    island = detect_blur_island_face_box(image)
    if island is not None:
        return [int(v) for v in island]
    _, report, _ = picture_pixelation(image)
    if report is not None:
        bounds = report.get('region_bounds') or []
        if len(bounds) == 1:
            bx, by, bw, bh = [int(v) for v in bounds[0]]
            aspect = bw / max(1.0, float(bh))
            if 0.45 <= aspect <= 1.80 and (bw * bh) <= 0.45 * width * height and min(bw, bh) >= 24:
                return [bx, by, bw, bh]
    return None


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
            from .capture_worker import AUTO_CROP_MARGIN_RATIO, _padded_face_crop
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
                if not records and not unavailable:
                    fb_box = _fallback_inspect_face_box(image, width, height)
                    if fb_box is not None:
                        records = [{
                            'box': fb_box,
                            'crop': _padded_face_crop(fb_box, width, height),
                            'score': 0.5,
                        }]
                sorted_recs = sorted(records, key=lambda rec: (rec['box'][0], rec['box'][1]))
                result['face_crops'] = [rec['crop'] for rec in sorted_recs]
                result['face_boxes'] = [rec['box'] for rec in sorted_recs]
                result['recommended_index'] = _recommend_face_index(image, sorted_recs, width, height)
            else:
                sorted_faces = sorted(faces, key=lambda box: (box[0], box[1]))
                result['face_crops'] = []
                pseudo_recs = []
                for x, y, w, h in sorted_faces:
                    margin = int(max(w, h) * AUTO_CROP_MARGIN_RATIO + .5)
                    left, top = max(0, x - margin), max(0, y - margin)
                    right, bottom = min(width, x + w + margin), min(height, y + h + margin)
                    crop_rect = [left, top, right - left, bottom - top]
                    result['face_crops'].append(crop_rect)
                    pseudo_recs.append({'box': [x, y, w, h], 'crop': crop_rect, 'score': 0.75})
                result['face_boxes'] = sorted_faces
                result['recommended_index'] = _recommend_face_index(image, pseudo_recs, width, height)
            result['face_scan_unavailable'] = unavailable
        return result
    return await run_in_threadpool(invoke, execute)


def _decode_reference(contents, max_side=MAX_IMAGE_SIDE, max_pixels=MAX_IMAGE_SIDE * MAX_IMAGE_SIDE):
    need(type(contents) is bytes and 0 < len(contents) <= MAX_UPLOAD_BYTES,
         'Choose a PNG or JPEG reference under 10 MB.')
    try:
        with Image.open(io.BytesIO(contents)) as source:
            need(source.format in ('PNG', 'JPEG') and getattr(source, 'n_frames', 1) == 1,
                 'Choose a valid PNG or JPEG reference image.')
            size_msg = (
                'Use a reference image no larger than 1600 pixels per side.'
                if max_side == MAX_IMAGE_SIDE
                else f'Use a reference image no larger than {max_side} pixels per side.'
            )
            need(0 < source.width <= max_side and 0 < source.height <= max_side
                 and source.width * source.height <= max_pixels,
                 size_msg)
            has_alpha = 'A' in source.mode or 'transparency' in source.info
            transposed = ImageOps.exif_transpose(source)
            if has_alpha:
                alpha_min, _ = transposed.convert('RGBA').getchannel('A').getextrema()
                need(alpha_min == 255, 'The reference must be opaque.')
            return np.asarray(transposed.convert('RGB')).copy()
    except ValueError:
        raise
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('Choose a valid PNG or JPEG reference image.') from error


def _valid_reference_crop(picture_crop, source_size):
    if not (isinstance(picture_crop, list) and len(picture_crop) == 4
            and all(type(n) is int for n in picture_crop)):
        return None
    if not (isinstance(source_size, list) and len(source_size) == 2
            and all(type(n) is int and n > 0 for n in source_size)):
        return None
    cx, cy, cw, ch = picture_crop
    sw, sh = source_size
    if cx < 0 or cy < 0 or cw <= 0 or ch <= 0 or cx + cw > sw or cy + ch > sh:
        return None
    return (cx, cy, cw, ch), (sw, sh)


def _parse_reference_geometry(geometry):
    if isinstance(geometry, str):
        need(0 < len(geometry) <= 4096, 'Invalid reference geometry.')
        try:
            geo = json.loads(geometry)
        except ValueError as error:
            raise ValueError('Invalid reference geometry.') from error
    elif isinstance(geometry, dict):
        geo = geometry
    else:
        raise ValueError('Invalid reference geometry.')
    need(isinstance(geo, dict), 'Invalid reference geometry.')
    bounds = geo.get('observed_bounds_256')
    need(isinstance(bounds, list) and len(bounds) == 4
         and all(type(n) is int and 0 <= n <= 256 for n in bounds)
         and bounds[0] < bounds[2] and bounds[1] < bounds[3],
         'The observed crop bounds could not be verified.')
    native_size = geo.get('native_size')
    if not (isinstance(native_size, list) and len(native_size) == 2
            and all(type(n) is int and n > 0 for n in native_size)):
        native_size = None
    crop_rect = _valid_reference_crop(geo.get('picture_crop'), geo.get('source_size'))
    return bounds, native_size, crop_rect


def _has_uniform_gray_padding(rgb, bounds):
    left, top, right, bottom = bounds
    margin = 2
    yy, xx = np.indices((256, 256))
    outside_bounds = (xx < left) | (xx >= right) | (yy < top) | (yy >= bottom)
    outside_margin = (
        (xx < left - margin) | (xx >= right + margin)
        | (yy < top - margin) | (yy >= bottom + margin)
    )
    mask = outside_bounds & outside_margin
    if not mask.any():
        mask = outside_bounds
    if not mask.any():
        return True
    region = rgb[mask].astype(np.int16)
    mismatched = np.any(np.abs(region - 128) > 16, axis=1)
    return float(mismatched.mean()) <= 0.02


def _decode_original_256(original_input):
    if not isinstance(original_input, str) or not original_input.startswith('data:image/png;base64,'):
        return None
    try:
        raw_orig = base64.b64decode(original_input.split(',', 1)[1], validate=True)
        orig_256 = _decode_reference(raw_orig)
    except (ValueError, TypeError, binascii.Error):
        return None
    return orig_256 if orig_256.shape == (256, 256, 3) else None


def _match_template_best(haystack_gray, needle_gray):
    if np.ptp(haystack_gray) == 0 or np.ptp(needle_gray) == 0:
        return -1.0, (0, 0)
    import cv2
    res = cv2.matchTemplate(haystack_gray, needle_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if not math.isfinite(max_val):
        return -1.0, (0, 0)
    return float(max_val), max_loc


def prepare_reference(contents, geometry, original_input=''):
    """Align an uploaded ground-truth reference to the exact 256x256 restoration canvas."""
    bounds, native_size, crop_rect = _parse_reference_geometry(geometry)
    orig_256 = _decode_original_256(original_input)
    has_extra_context = crop_rect is not None or orig_256 is not None
    rgb = _decode_reference(
        contents,
        max_side=4096 if has_extra_context else MAX_IMAGE_SIDE,
        max_pixels=8388608 if has_extra_context else MAX_IMAGE_SIDE * MAX_IMAGE_SIDE,
    )
    nat_h, nat_w = rgb.shape[:2]
    is_padded = bounds != [0, 0, 256, 256]
    left, top, right, bottom = bounds
    obs_w, obs_h = right - left, bottom - top
    nw, nh = tuple(native_size) if native_size is not None else (obs_w, obs_h)
    ref_geo = None
    target_256 = None
    source_mode = None

    tmpl_native = None
    if orig_256 is not None:
        tmpl_native = np.asarray(
            Image.fromarray(orig_256[top:bottom, left:right]).resize((nw, nh), Image.Resampling.BILINEAR)
        )

    exact_native = native_size is not None and (nat_w, nat_h) == tuple(native_size)
    ref_aspect = nat_w / nat_h
    obs_aspect = obs_w / obs_h
    diff_obs = abs(ref_aspect - obs_aspect) / max(ref_aspect, obs_aspect)
    if native_size is not None:
        nat_aspect = native_size[0] / native_size[1]
        diff_native = abs(ref_aspect - nat_aspect) / max(ref_aspect, nat_aspect)
    else:
        diff_native = diff_obs
    aspect_ok = exact_native or diff_native <= 0.05 or diff_obs <= 0.05

    def _from_full_crop():
        cx, cy, cw, ch = crop_rect[0]
        return prepare_crop(rgb[cy:cy + ch, cx:cx + cw].copy())

    if (nat_w, nat_h) == (256, 256):
        if is_padded and _has_uniform_gray_padding(rgb, bounds):
            target_256, source_mode = rgb, 'padded_256'
        elif crop_rect is not None and crop_rect[1] == (256, 256):
            target_256, _, _, ref_geo = _from_full_crop()
            source_mode = 'full_cropped'
        elif is_padded:
            raise ValueError(
                'The 256 × 256 reference is not padded to match this crop. '
                'Upload the clean unpadded crop or a 256 × 256 padded frame.'
            )
        else:
            target_256, source_mode = rgb, 'padded_256'
    elif crop_rect is not None and (nat_w, nat_h) == crop_rect[1]:
        target_256, _, _, ref_geo = _from_full_crop()
        source_mode = 'full_cropped'
    elif (orig_256 is not None and crop_rect is not None
          and abs((nat_w / nat_h) - (crop_rect[1][0] / crop_rect[1][1]))
          / max(nat_w / nat_h, crop_rect[1][0] / crop_rect[1][1]) <= 0.03):
        (cx, cy, cw, ch), (sw, sh) = crop_rect
        scaled_rgb = np.asarray(Image.fromarray(rgb).resize((sw, sh), Image.Resampling.BILINEAR))
        target_256, _, _, ref_geo = prepare_crop(scaled_rgb[cy:cy + ch, cx:cx + cw].copy())
        source_mode = 'full_cropped'

    if target_256 is None and tmpl_native is not None and np.ptp(rgb) > 0 and np.ptp(tmpl_native) > 0:
        import cv2
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        tmpl_gray = cv2.cvtColor(tmpl_native, cv2.COLOR_RGB2GRAY)

        if nat_w >= nw and nat_h >= nh and (nat_w, nat_h) != (nw, nh):
            max_val, (mx, my) = _match_template_best(rgb_gray, tmpl_gray)
            if max_val >= 0.55:
                target_256, _, _, _ = prepare_crop(rgb[my:my + nh, mx:mx + nw].copy())
                source_mode = 'template_cropped'
                ref_geo = None

        if target_256 is None and nat_w <= nw and nat_h <= nh and (nat_w, nat_h) != (nw, nh):
            max_val, (mx, my) = _match_template_best(tmpl_gray, rgb_gray)
            if max_val >= 0.55:
                sub_l = max(left, min(right - 1, left + int(round(mx * obs_w / nw))))
                sub_t = max(top, min(bottom - 1, top + int(round(my * obs_h / nh))))
                sub_r = max(sub_l + 1, min(right, left + int(round((mx + nat_w) * obs_w / nw))))
                sub_b = max(sub_t + 1, min(bottom, top + int(round((my + nat_h) * obs_h / nh))))
                target_256 = np.full((256, 256, 3), 128, dtype=np.uint8)
                target_256[sub_t:sub_b, sub_l:sub_r] = np.asarray(
                    Image.fromarray(rgb).resize((sub_r - sub_l, sub_b - sub_t), Image.Resampling.BILINEAR)
                )
                return {
                    'reference_256': url(png(target_256)),
                    'observed_bounds_256': bounds,
                    'eval_bounds_256': [sub_l, sub_t, sub_r, sub_b],
                    'source_mode': 'subcrop_aligned',
                }

        if target_256 is None and not aspect_ok:
            best_val = -1.0
            best_match = None
            for scale in np.linspace(0.3, 1.8, 13):
                sw_c = int(round(nw * float(scale)))
                sh_c = int(round(nh * float(scale)))
                if 16 <= sw_c <= nat_w and 16 <= sh_c <= nat_h:
                    scaled_tmpl = np.asarray(
                        Image.fromarray(tmpl_gray).resize((sw_c, sh_c), Image.Resampling.BILINEAR)
                    )
                    max_val, (mx, my) = _match_template_best(rgb_gray, scaled_tmpl)
                    if max_val > best_val:
                        best_val = max_val
                        best_match = (mx, my, sw_c, sh_c)
            if best_val >= 0.65 and best_match is not None:
                mx, my, sw_c, sh_c = best_match
                matched_crop = np.asarray(
                    Image.fromarray(rgb[my:my + sh_c, mx:mx + sw_c]).resize((nw, nh), Image.Resampling.BILINEAR)
                )
                target_256, _, _, _ = prepare_crop(matched_crop)
                source_mode = 'template_cropped'
                ref_geo = None

    if target_256 is None:
        need(aspect_ok,
             'Reference aspect ratio does not match the input crop. '
             'Use the full original photo, a clean crop matching the input, or a 256 × 256 padded frame.')
        target_256, _, _, ref_geo = prepare_crop(rgb)
        source_mode = 'native_crop'
        if orig_256 is not None and ref_geo['observed_bounds_256'] != bounds:
            target_256 = np.full((256, 256, 3), 128, dtype=np.uint8)
            target_256[top:bottom, left:right] = np.asarray(
                Image.fromarray(rgb).resize((obs_w, obs_h), Image.Resampling.BILINEAR)
            )
            ref_geo = None

    if ref_geo is not None:
        need(ref_geo['observed_bounds_256'] == bounds,
             'Reference aspect ratio does not match the input crop. '
             'Use the full original photo, a clean crop matching the input, or a 256 × 256 padded frame.')
    return {
        'reference_256': url(png(target_256)),
        'observed_bounds_256': bounds,
        'source_mode': source_mode,
    }


@app.post('/face/restoration/reference')
async def reference(file: UploadFile = File(...), geometry: str = Form(...),
                    original_input: Annotated[str, Form()] = ''):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    def execute():
        try:
            return prepare_reference(contents, geometry, original_input=original_input)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    return await run_in_threadpool(execute)


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
        native, source_size = picture_native_with_source_size(contents, rectangle)
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
        alignment_info = None
        if automatic:
            need(protect is None and apply is None and not preview_id and not confirmed,
                 'Auto selects its own preparation. Remove manual preparation fields.')
            from . import capture_api
            from .capture_worker import _box_iou, _padded_face_crop
            from .automatic_preparation import (
                _is_genuine_dct_macroblocking,
                detect_blur_island_face_box,
                is_background_grid_with_blurred_face,
                prepare as prepare_automatically,
                prepare_composables,
                prepare_macroblocking,
                prepare_mosaics,
                prepare_noise,
                prepare_photometric,
                prepare_source,
                pixelation_signal,
                tiny_crop_pixelation,
            )
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

            def _in_selection(items):
                if selected_rectangle is None:
                    return items
                sx, sy, sw, sh = selected_rectangle
                matched = []
                for item in items:
                    box = item['box'] if isinstance(item, dict) else item
                    cx = box[0] + box[2] / 2.0
                    cy = box[1] + box[3] / 2.0
                    if sx <= cx < sx + sw and sy <= cy < sy + sh:
                        matched.append(item)
                        continue
                    ix1 = max(sx, box[0])
                    iy1 = max(sy, box[1])
                    ix2 = min(sx + sw, box[0] + box[2])
                    iy2 = min(sy + sh, box[1] + box[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter >= 0.40 * max(1, box[2] * box[3]):
                        matched.append(item)
                return matched

            def _pick_target_face(items):
                if not items:
                    return None
                if len(items) == 1:
                    return items[0]
                if selected_rectangle is None:
                    return None
                sx, sy, sw, sh = selected_rectangle
                scx, scy = sx + sw * 0.5, sy + sh * 0.5
                diag = max(1.0, math.hypot(sw, sh))
                src_h, src_w = source_rgb.shape[:2]

                def _rank(item):
                    box = item['box'] if isinstance(item, dict) else item
                    ix1 = max(sx, box[0])
                    iy1 = max(sy, box[1])
                    ix2 = min(sx + sw, box[0] + box[2])
                    iy2 = min(sy + sh, box[1] + box[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    overlap = inter / max(1.0, float(box[2] * box[3]))
                    padded = item['crop'] if isinstance(item, dict) and 'crop' in item else _padded_face_crop(box, src_w, src_h)
                    crop_iou = _box_iou(padded, selected_rectangle)
                    dist = math.hypot((box[0] + box[2] * 0.5) - scx, (box[1] + box[3] * 0.5) - scy) / diag
                    centrality = max(0.0, 1.0 - dist)
                    score = float(item.get('score', 0.75)) if isinstance(item, dict) else 0.75
                    return (crop_iou * 2.0 + overlap + centrality * 1.2 + score * 0.35, box[2] * box[3])

                return max(items, key=_rank)

            _face_cache = {'done': False, 'face': None, 'method': 'original_image', 'record': None}

            def _detect_single_face():
                if _face_cache['done']:
                    return _face_cache['face'], _face_cache['method']
                scan_image = Image.fromarray(source_rgb)
                method = 'original_image'
                use_details = capture_api.detect_faces is capture_api._DEFAULT_DETECT_FACES
                picked_rec = None
                try:
                    with capture_api.DETECTION_LOCK:
                        if use_details:
                            items = _in_selection(capture_api.detect_face_details(scan_image))
                        else:
                            items = _in_selection(capture_api.detect_faces(scan_image))
                        if not items:
                            # Retry detection on copies. Preparation still requires
                            # a block-pattern signal on the untouched source pixels.
                            scale = max(scan_image.size) / 256
                            for scan_radius in (min(2., max(.5, scale * .9)), min(12., max(2.5, scale * 3.3))):
                                smoothed = scan_image.filter(ImageFilter.GaussianBlur(scan_radius))
                                matches = _in_selection(
                                    capture_api.detect_face_details(smoothed) if use_details
                                    else capture_api.detect_faces(smoothed)
                                )
                                if len(matches) > 1 and selected_rectangle is None:
                                    break
                                chosen = _pick_target_face(matches)
                                if chosen is not None:
                                    chosen_box = chosen['box'] if isinstance(chosen, dict) else chosen
                                    if pixelation_signal(source_rgb, chosen_box)['suspected']:
                                        items = [chosen]
                                        method = 'smoothed_detection_copy_with_block_signal'
                                        break
                except RuntimeError:
                    items = []
                chosen = _pick_target_face(items)
                if isinstance(chosen, dict):
                    detected = chosen['box']
                    picked_rec = chosen
                else:
                    detected = chosen
                if detected is None:
                    island = detect_blur_island_face_box(source_rgb, selected_rectangle)
                    if island is not None:
                        detected = island
                        method = 'blur_island_fallback'
                _face_cache.update(done=True, face=detected, method=method, record=picked_rec)
                return detected, method

            def _apply_upright_roll_if_needed(rgb_arr):
                nonlocal alignment_info
                rec = _face_cache.get('record')
                if not isinstance(rec, dict) or float(rec.get('score', 0.0)) < 0.65:
                    return rgb_arr
                roll_deg = rec.get('roll_deg')
                if roll_deg is None or not math.isfinite(float(roll_deg)):
                    return rgb_arr
                roll_val = float(roll_deg)
                if abs(roll_val) < 15.0 or abs(roll_val) > 165.0:
                    return rgb_arr
                import cv2

                h_a, w_a = rgb_arr.shape[:2]
                mat = cv2.getRotationMatrix2D((w_a * 0.5, h_a * 0.5), roll_val, 1.0)
                rotated = cv2.warpAffine(
                    rgb_arr,
                    mat,
                    (w_a, h_a),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                alignment_info = {
                    'applied': True,
                    'roll_deg': round(roll_val, 2),
                    'interpolation': 'INTER_CUBIC',
                    'border_mode': 'BORDER_REPLICATE',
                }
                return rotated

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
            # Crop-scoped mosaic fallback when a small face mosaic inside a large scene
            # (e.g. Voronoi/low-poly or compact face blocks) was below full-frame scale.
            if (
                region_input is None
                and not capture_id
                and selected_rectangle is not None
                and scanline_report is None
                and softened_detection is None
                and min(native.shape[:2]) >= 48
                and max(source_rgb.shape[:2]) > 1.5 * max(native.shape[:2])
            ):
                crop_sm, crop_rep, crop_ev = prepare_mosaics(native)
                if crop_sm is not None and crop_rep is not None and not np.array_equal(crop_sm, native):
                    rx, ry, rw, rh = selected_rectangle
                    full_sm = source_rgb.copy()
                    full_sm[ry:ry + rh, rx:rx + rw] = crop_sm
                    full_mask = np.zeros(source_rgb.shape[:2], dtype=np.uint8)
                    with Image.open(io.BytesIO(crop_ev['full-image-pixelation-mask.png'])) as cm_im:
                        full_mask[ry:ry + rh, rx:rx + rw] = np.asarray(cm_im.convert('L'))
                    full_sm_png = png(full_sm)
                    full_mask_png = png(full_mask)
                    region_input = crop_sm
                    regional_report = {
                        **crop_rep,
                        'native_size': [int(source_rgb.shape[1]), int(source_rgb.shape[0])],
                        'crop_scoped_mosaic': True,
                        'full_image_sha256': digest(full_sm_png),
                        'mask_sha256': digest(full_mask_png),
                    }
                    picture_regions = regional_report
                    region_evidence = {
                        'full-image-smoothed.png': full_sm_png,
                        'full-image-pixelation-mask.png': full_mask_png,
                    }
                    payloads.update(region_evidence)

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
            if (
                is_privacy_mosaic_region
                and tiny_report is None
                and scanline_report is None
                and regional_report is not None
                and not regional_report.get('degraded_patch_conditioning')
                and not regional_report.get('mixed_jpeg_mosaic')
                and not regional_report.get('coordinated_mosaic_regions')
                and not regional_report.get('geometric_mosaic_pattern')
                and not regional_report.get('crop_scoped_mosaic')
                and 'full-image-pixelation-mask.png' in region_evidence
            ):
                with Image.open(io.BytesIO(region_evidence['full-image-pixelation-mask.png'])) as m_im:
                    mosaic_mask = np.asarray(m_im) > 0
                face_candidate, _ = _detect_single_face()
                if face_candidate is not None and is_background_grid_with_blurred_face(source_rgb, mosaic_mask, face_candidate):
                    is_privacy_mosaic_region = False
                    region_input = None
                    smoothed_image = None
                    picture_regions = None
                    frame_regions = None
                    smoothed_frame_png = None
                    for key in ('full-image-smoothed.png', 'full-image-pixelation-mask.png',
                                'capture/frame-pixelation-mask.png', 'capture/frame-irregular-region-mask.png',
                                'irregular-region-mask.png'):
                        payloads.pop(key, None)

            if is_privacy_mosaic_region:
                extra_stages = []
                extra_reports = {}
                mosaic_out = region_input.copy()
                if (
                    tiny_report is None
                    and scanline_report is None
                    and regional_report is not None
                    and not regional_report.get('degraded_patch_conditioning')
                    and not regional_report.get('mixed_jpeg_mosaic')
                ):
                    # Allow noise/cast cleanup to compose after privacy mosaic smoothing
                    # while keeping smoothed mosaic pixels protected from luma noise blur.
                    crop_mosaic_mask = None
                    if 'full-image-pixelation-mask.png' in region_evidence:
                        with Image.open(io.BytesIO(region_evidence['full-image-pixelation-mask.png'])) as cm_im:
                            full_m = np.asarray(cm_im) > 0
                        if selected_rectangle is not None:
                            rx, ry, rw, rh = selected_rectangle
                            crop_mosaic_mask = full_m[ry:ry + rh, rx:rx + rw]
                        elif full_m.shape == mosaic_out.shape[:2]:
                            crop_mosaic_mask = full_m
                    if crop_mosaic_mask is not None and int((~crop_mosaic_mask).sum()) >= 64:
                        n_out, n_rep, n_ev = prepare_noise(mosaic_out)
                        if n_out is not None and n_rep is not None:
                            # Preserve the smoothed mosaic region exact; denoise exterior noise only.
                            merged_n = n_out.copy()
                            merged_n[crop_mosaic_mask] = mosaic_out[crop_mosaic_mask]
                            if not np.array_equal(merged_n, mosaic_out):
                                mosaic_out = merged_n
                                extra_stages.append('noise_denoise')
                                extra_reports['noise_denoise'] = n_rep
                                payloads.update(n_ev)
                    cd_out, cd_route, cd_rep, cd_ev = prepare_photometric(mosaic_out)
                    if cd_out is not None and cd_route == 'color_cast_balance' and cd_rep is not None:
                        mosaic_out = cd_out
                        extra_stages.append(cd_route)
                        extra_reports[cd_route] = cd_rep
                        payloads.update(cd_ev)

                selected = png(mosaic_out)
                prepared, _, _, _ = prepare_crop(mosaic_out)
                preparation = {**(tiny_report or regional_report),
                               'crop_changed_native_pixels': int(np.any(mosaic_out != native, axis=2).sum()),
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
                if extra_stages:
                    preparation['applied_stages'] = [auto_decision['route']] + extra_stages
                    preparation['stage_reports'] = extra_reports
                    auto_decision['applied_stages'] = [auto_decision['route']] + extra_stages
                    auto_decision['stage_reports'] = extra_reports
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

                face, detection_method = _detect_single_face()

                if not applied_stages and face is not None:
                    face_mask = _face_mask_for_native(native.shape, face, crop_origin) if jpeg_report is None else None
                    if face_mask is not None:
                        mb_out, mb_stage, mb_report, mb_evidence = prepare_macroblocking(
                            current_rgb, origin=crop_origin, mask=face_mask, allow_fbcnn=True, dc=macroblock_dc
                        )
                    else:
                        mb_out, mb_stage, mb_report, mb_evidence = None, None, None, {}
                    if mb_out is not None and mb_stage is not None:
                        current_rgb = mb_out
                        applied_stages = [mb_stage]
                        stage_reports = {mb_stage: mb_report}
                        stage_evidence = dict(mb_evidence)
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

                allow_auto_blur = face is not None and (
                    not applied_stages
                    or detection_method != 'blur_island_fallback'
                    or selected_rectangle is not None
                )
                if allow_auto_blur:
                    if applied_stages:
                        if selected_rectangle is None:
                            blur_source = current_rgb
                        else:
                            bx, by, bw, bh = selected_rectangle
                            blur_source = source_rgb.copy()
                            blur_source[by:by + bh, bx:bx + bw] = current_rgb
                    else:
                        blur_source = source_rgb
                    candidate, blur_prep, auto_evidence, blur_decision = prepare_automatically(
                        current_rgb, blur_source, face, selected_rectangle
                    )
                    blur_route = blur_decision.get('route')
                    if candidate is not None and (not applied_stages or blur_route in ('protect_auto', 'face_gaussian')):
                        if applied_stages:
                            composed_stages = list(applied_stages) + [blur_route]
                            stage_reports[blur_route] = blur_decision
                            payloads.update(stage_evidence)
                            payloads.update(auto_evidence)
                            cand_rgb = decode_original(candidate)
                            aligned_rgb = _apply_upright_roll_if_needed(cand_rgb)
                            selected = png(aligned_rgb) if alignment_info is not None else candidate
                            selected_sha = digest(selected)
                            if alignment_info is not None:
                                payloads['derived-native.png'] = selected
                            prepared, _, _, _ = prepare_crop(aligned_rgb)
                            preparation = {
                                **blur_prep,
                                'applied_stages': composed_stages,
                                'stage_reports': stage_reports,
                                'crop_changed_native_pixels': int(np.any(aligned_rgb != native, axis=2).sum()),
                                'derived_native_sha256': selected_sha,
                            }
                            if alignment_info is not None and isinstance(preparation.get('payload_pins'), dict):
                                preparation['payload_pins'] = {
                                    **preparation['payload_pins'],
                                    'derived-native.png': {'bytes': len(selected), 'sha256': selected_sha},
                                }
                            auto_decision = {
                                **blur_decision,
                                'route': blur_route,
                                'face_detection': detection_method,
                                'applied_stages': composed_stages,
                                'composed_stages': composed_stages,
                                'stage_reports': stage_reports,
                            }
                            payloads['processing.json'] = canonical(preparation)
                            macroblock_report = stage_reports.get('macroblock_fbcnn') or stage_reports.get('macroblock_hevc')
                        else:
                            blur_decision['face_detection'] = detection_method
                            auto_decision = blur_decision
                            preparation = dict(blur_prep) if isinstance(blur_prep, dict) else blur_prep
                            payloads.update(auto_evidence)
                            cand_rgb = decode_original(candidate)
                            aligned_rgb = _apply_upright_roll_if_needed(cand_rgb)
                            selected = png(aligned_rgb) if alignment_info is not None else candidate
                            if alignment_info is not None:
                                selected_sha = digest(selected)
                                payloads['derived-native.png'] = selected
                                if isinstance(preparation, dict):
                                    preparation['derived_native_sha256'] = selected_sha
                                    preparation['crop_changed_native_pixels'] = int(np.any(aligned_rgb != native, axis=2).sum())
                                    if isinstance(preparation.get('payload_pins'), dict):
                                        preparation['payload_pins'] = {
                                            **preparation['payload_pins'],
                                            'derived-native.png': {'bytes': len(selected), 'sha256': selected_sha},
                                        }
                                    payloads['processing.json'] = canonical(preparation)
                            prepared, _, _, _ = prepare_crop(aligned_rgb)
                    elif applied_stages:
                        aligned_rgb = _apply_upright_roll_if_needed(current_rgb)
                        selected, prepared, preparation, auto_decision, macroblock_report = _package_artifact_stages(
                            aligned_rgb, native, applied_stages, stage_reports, stage_evidence, payloads
                        )
                    else:
                        blur_decision['face_detection'] = detection_method
                        auto_decision = blur_decision
                        payloads.update(auto_evidence)
                        aligned_rgb = _apply_upright_roll_if_needed(current_rgb)
                        if alignment_info is not None:
                            selected = png(aligned_rgb)
                            prepared, _, _, _ = prepare_crop(aligned_rgb)
                elif applied_stages:
                    aligned_rgb = _apply_upright_roll_if_needed(current_rgb)
                    selected, prepared, preparation, auto_decision, macroblock_report = _package_artifact_stages(
                        aligned_rgb, native, applied_stages, stage_reports, stage_evidence, payloads
                    )
                else:
                    # Crop-level pixelation_signal fallback when face detection is uncertain.
                    ch_c, cw_c = current_rgb.shape[:2]
                    crop_pix = (
                        pixelation_signal(current_rgb, (0, 0, cw_c, ch_c))
                        if min(ch_c, cw_c) >= 24 else {'suspected': False}
                    )
                    if crop_pix.get('suspected') and not _is_genuine_dct_macroblocking(current_rgb, origin=crop_origin):
                        cand, blur_prep, auto_evidence, blur_decision = prepare_automatically(
                            current_rgb, current_rgb, [0, 0, cw_c, ch_c], None
                        )
                        if cand is not None:
                            blur_decision['face_detection'] = 'crop_pixelation_fallback'
                            auto_decision = blur_decision
                            preparation = blur_prep
                            payloads.update(auto_evidence)
                            selected = cand
                            prepared, _, _, _ = prepare_crop(decode_original(cand))
                        else:
                            auto_decision = {'route': 'direct', 'reason': 'face_detection_uncertain'}
                    else:
                        auto_decision = {'route': 'direct', 'reason': 'face_detection_uncertain'}
            if alignment_info is not None and auto_decision is not None:
                auto_decision['alignment'] = alignment_info
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
        metadata['source_size'] = source_size
        if capture_id and 'capture/metadata.json' in payloads and 'capture/frame.png' in payloads:
            cap_meta = json.loads(payloads['capture/metadata.json'])
            if isinstance(cap_meta.get('rectangle'), list) and len(cap_meta['rectangle']) == 4:
                metadata['capture_crop'] = cap_meta['rectangle']
            if (
                'source_time' in cap_meta
                and isinstance(cap_meta['source_time'], (int, float))
                and not isinstance(cap_meta['source_time'], bool)
                and math.isfinite(cap_meta['source_time'])
            ):
                metadata['source_time'] = float(cap_meta['source_time'])
            with Image.open(io.BytesIO(payloads['capture/frame.png'])) as frame_im:
                metadata['frame_size'] = [int(frame_im.width), int(frame_im.height)]
        if auto_decision is not None:
            metadata['automatic_preparation'] = auto_decision
        if alignment_info is not None:
            metadata['alignment'] = alignment_info
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


from .completion_api import register as _register_completion
from .capture_api import register as _register_capture

_register_completion(app)
_register_capture(app)

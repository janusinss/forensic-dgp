"""Loopback restoration application. Models live only in disposable workers."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import numpy as np
from PIL import Image
import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .preparation import MAX_MASK_BYTES, MAX_UPLOAD_BYTES, build_preview, decode_original, png
from .quality import quality_signals
from .runtime import model_contract, prepare_crop
from .setup_models import ROOT, verify_models

app = FastAPI(title="Forensic face restoration — IMPLEMENTATION")
app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')
LOCK = threading.Lock()
STATE = {'neural_forward_attempts': 0, 'neural_forward_returns': 0,
         'active_worker': False, 'uncertain': False, 'last_worker_pid': None}
ARMS = {'u96': 'incumbent_U96', 'u64': 'postfusion_LR_u64'}
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
    need(value == 'usable', 'Inspect the original and confirm readable visible facial features first.')


def preview(contents, protect, apply, radius):
    return build_preview(contents, protect, apply, nominal_radius=radius)


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
    need(model in ARMS, 'Choose U96 or experimental u64.')
    if psutil.virtual_memory().available < 2684354560:
        raise RuntimeError('Restoration needs at least 2.5 GiB available RAM. Close unused applications and retry.')
    verify_models()
    request_dir = ROOT / 'scratch/implementation_requests' / uuid.uuid4().hex
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
async def inspect(file: UploadFile = File(...)):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    def execute():
        image = decode_original(contents)
        return {'width': image.shape[1], 'height': image.shape[0], 'source': url(png(image)),
                'neural_calls': 0, 'input_review_required': True}
    return await run_in_threadpool(invoke, execute)


@app.post('/face/restoration/preview')
async def make_preview(file: UploadFile = File(...), protect_mask: UploadFile = File(...),
                       apply_mask: UploadFile = File(...), radius: float = Form(1), input_review: str = Form('')):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    protect, apply = await protect_mask.read(MAX_MASK_BYTES + 1), await apply_mask.read(MAX_MASK_BYTES + 1)
    def execute():
        review_input(input_review)
        response, payloads = preview(contents, protect, apply, radius)
        return {**response, 'bundle': url(bundle(payloads), 'application/zip'), 'neural_calls': 0}
    return await run_in_threadpool(invoke, execute)


@app.post('/face/restoration/restore')
async def restore(file: UploadFile = File(...), input_review: str = Form(''),
                  restoration: str = Form('auto'), preparation_mode: str = Form('bypass'),
                  protect_mask: UploadFile | None = File(None), apply_mask: UploadFile | None = File(None),
                  radius: float = Form(1), preview_id: str = Form(''), confirmed: bool = Form(False),
                  include_bundle: bool = Form(True), model: str = Form(DEFAULT_MODEL)):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    protect = await protect_mask.read(MAX_MASK_BYTES + 1) if protect_mask else None
    apply = await apply_mask.read(MAX_MASK_BYTES + 1) if apply_mask else None
    def execute():
        started = time.monotonic()
        review_input(input_review)
        need(model in ARMS and restoration in ('auto', 'on', 'off'), 'Choose a valid model and Auto, On or Off.')
        need(preparation_mode in ('manual', 'bypass'), 'Choose bypass or reviewed manual preparation.')
        native = decode_original(contents)
        need(np.ptp(native.reshape(-1, 3).astype(np.int16), axis=0).max() > 0,
             'No usable structure is visible. Upload a clearer face crop.')
        original, observed, _, geometry = prepare_crop(native)
        signals = quality_signals(original, (~observed).astype(np.uint8))
        if restoration == 'auto' and signals['blur_variance'] is None:
            raise ValueError('Too little area for Auto. Choose On or Off, or upload a clearer crop.')
        applied = restoration == 'on' or (restoration == 'auto' and signals['suggest_restoration'])
        payloads = {'uploaded-original.bin': contents, 'original-native.png': png(native),
                    'original-256.png': png(original)}
        selected, prepared, preparation = contents, original, None
        if preparation_mode == 'bypass':
            need(protect is None and apply is None and not preview_id and not confirmed,
                 'Bypass uses the original. Remove manual preparation fields.')
        elif applied:
            need(protect is not None and apply is not None and confirmed,
                 'Preview and confirm both Protect and Apply areas first.')
            response, prepared_payloads = preview(contents, protect, apply, radius)
            need(preview_id == response['preview_id'], 'The preview is stale. Preview and confirm the current areas again.')
            need(response['metadata']['no_effect'] or response['metadata']['protected_native_pixels'] > 0,
                 'Protect the existing facial blur and soft edges, or bypass preparation.')
            payloads.update(prepared_payloads)
            selected, preparation = prepared_payloads['derived-native.png'], response['metadata']
            prepared, _, _, _ = prepare_crop(decode_original(selected))
        raw_bytes, output, worker = None, png(original), None
        if applied:
            raw_bytes, output, worker = one_forward(selected, model)
        metadata = {'selection': ARMS[model], 'model': model, 'restoration_requested': restoration,
                    'restoration_applied': bool(applied), 'preparation_applied': preparation is not None,
                    'preparation': preparation, 'preparation_requested': preparation_mode,
                    'preparation_bypass_reason': ('restoration_off' if restoration == 'off' else
                         'Auto_did_not_suggest_restoration' if not applied else
                         'operator_bypass' if preparation_mode == 'bypass' else None),
                    'quality_signals': signals, 'Auto_assessed_original_canvas_before_preparation': True,
                    'neural_forward_attempts': int(applied), 'neural_forward_returns': int(applied),
                    'seconds': time.monotonic() - started, 'uploaded_bytes_sha256': digest(contents),
                    'selected_input_sha256': digest(selected), 'result_file_sha256': digest(output),
                    'geometry': geometry, 'worker_delivery': worker, 'model_qualified': False,
                    'protection_scope': 'Preprocessing input only; generated appearance may change.',
                    'training_optimizer_VM_calls': 0}
        payloads.update({'prepared-256.png': png(prepared), 'result.png': output,
                         'restoration.json': canonical(metadata)})
        if raw_bytes is not None:
            payloads['restoration-raw-float32.npy'] = raw_bytes
        result = {'original_input': url(png(original)), 'prepared_input': url(png(prepared)),
                  'result': url(output), 'metadata': metadata}
        if include_bundle:
            result['bundle'] = url(bundle(payloads), 'application/zip')
        return result
    return await run_in_threadpool(invoke, execute)

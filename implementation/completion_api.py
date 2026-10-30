"""Additive reviewed completion. The existing restoration endpoint is unchanged.

The passive receipt middleware observes successful restoration responses without
changing their bytes. Completion accepts only one of those exact canvases. It
never calls restoration, changes its model, or restores an R1-generated result.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image
import psutil
from fastapi import File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from . import app as restoration
from .completion_process_guard import OwnedWorker
from .setup_models import ROOT
from .managed_paths import data_path

STORE = data_path(ROOT, 'scratch', 'completion_requests')
MAX_PNG = 1024**2
MAX_RECEIPT = 96 * 1024**2
REGISTRY_LOCK = threading.RLock()
ADMISSIONS = {}
SESSIONS = {}
STATE = {'uncertain': False, 'active_worker': False, 'detector_calls': 0, 'completion_calls': 0}
HEX = re.compile(r'^[0-9a-f]{32}$')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def metadata_key(value):
    """Exact JSON numeric meaning across browser1 and Python1.0 serialization.

    No tolerance or rounding: only an exactly integral finite float becomes an
    integer. Saved restoration JSON and scientific/model values are untouched.
    """
    def normalized(item):
        if type(item) is float and item.is_integer():
            return int(item)
        if type(item) is dict:
            return {key: normalized(value) for key,value in item.items()}
        if type(item) is list:
            return [normalized(value) for value in item]
        return item
    return sha(canonical(normalized(value)))


def save(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def read_png(raw, mode):
    need(0 < len(raw) <= MAX_PNG, 'Use a PNG smaller than 1 MiB.')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            need(image.format == 'PNG' and image.size == (256, 256) and image.mode == mode,
                 'Use an exact RGB256 image or opaque black-and-white L256 mask.')
            array = np.array(image)
    except (OSError, SyntaxError) as exc:
        raise ValueError('Cannot read this PNG.') from exc
    if mode == 'L':
        need(np.isin(array, (0, 255)).all(), 'Removal masks must contain only black and white.')
    return array


def from_url(value, prefix, maximum):
    need(isinstance(value, str) and value.startswith(prefix) and len(value) <= maximum * 4 // 3 + 128,
         'The restoration response contains an invalid artifact.')
    try:
        raw = base64.b64decode(value[len(prefix):], validate=True)
    except ValueError as exc:
        raise ValueError('Invalid restoration artifact encoding.') from exc
    need(0 < len(raw) <= maximum, 'Restoration artifact exceeds bounds.')
    return raw


def admit_restoration(body):
    """Record actual successful server output, not a client-asserted model label."""
    result = json.loads(body)
    metadata = result['metadata']
    need(metadata['model'] == 'v2' and metadata['selection'] == 'conditioned_sft_v2_p2_u0224',
         'Unexpected restoration model.')
    key = metadata_key(metadata)
    images = {name: from_url(result[field], 'data:image/png;base64,', MAX_PNG)
              for name, field in [('original-256.png', 'original_input'),
                                  ('selected-input-256.png', 'prepared_input'),
                                  ('restoration-result.png', 'result')]}
    for value in images.values():
        read_png(value, 'RGB')
    need(sha(images['restoration-result.png']) == metadata['result_file_sha256'],
         'Restoration result fingerprint differs.')
    files = {**images, 'restoration.json': canonical(metadata)}
    if 'bundle' in result:
        files['restoration-evidence.zip'] = from_url(result['bundle'], 'data:application/zip;base64,', 64 * 1024**2)
    with REGISTRY_LOCK:
        if key in ADMISSIONS:
            return
        need(len(ADMISSIONS) < 100, 'Completion review is full; restart the app to review more outputs.')
        path = STORE / 'admissions' / key
        path.mkdir(parents=True, exist_ok=False)
        for name, value in files.items():
            save(path / name, value)
        ADMISSIONS[key] = {'path': path, 'metadata': metadata,
                           'pins': {name: sha(raw) for name, raw in files.items()}}


class RestorationReceiptMiddleware:
    """Pass every ASGI message through unchanged; retain successful output only."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/face/restoration/restore' or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        parts, count, status = [], 0, None

        async def observe(message):
            nonlocal count, status
            if message['type'] == 'http.response.start':
                status = message['status']
            elif message['type'] == 'http.response.body' and status == 200:
                raw = message.get('body', b'')
                count += len(raw)
                if count <= MAX_RECEIPT:
                    parts.append(raw)
                else:
                    parts.clear()
                if not message.get('more_body', False) and count <= MAX_RECEIPT:
                    try:
                        admit_restoration(b''.join(parts))
                    except Exception as exc:
                        # Restoration still succeeds exactly. Completion reports its
                        # independent receipt error rather than mutating that result.
                        STATE['admission_error'] = str(exc)[:1000]
                        try:
                            STORE.mkdir(parents=True, exist_ok=True)
                            (STORE / 'admission-error.json').write_bytes(canonical({'error': str(exc)[:1000]}))
                        except OSError:
                            pass  # Reported by completion status; restoration is unchanged.
            await send(message)
        await self.app(scope, receive, observe)


class BoundedResponseMiddleware:
    """Deliver identical bytes in bounded writes on Windows loopback transports.

    Model results, JSON, Content-Length and all other headers stay unchanged.
    This also avoids repeating a finished inference after a large body reset.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        async def bounded_send(message):
            body = message.get('body', b'')
            if message['type'] != 'http.response.body' or len(body) <= 65536:
                return await send(message)
            for offset in range(0, len(body), 65536):
                piece = body[offset:offset+65536]
                await send({**message, 'body': piece,
                            'more_body': offset+len(piece) < len(body) or message.get('more_body', False)})
        await self.app(scope, receive, bounded_send)


def invoke(function, *args):
    if not restoration.LOCK.acquire(blocking=False):
        raise HTTPException(409, 'A request is already running. Wait for it to finish.')
    try:
        need(not STATE['uncertain'], 'The completion worker needs review. Restart the app after inspecting retained evidence.')
        need(not restoration.STATE['uncertain'], 'Inspect the previous restoration worker error before continuing.')
        return function(*args)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        restoration.LOCK.release()


def session(ident):
    need(isinstance(ident, str) and HEX.fullmatch(ident) and ident in SESSIONS,
         'Review a current restoration result before completing a covering.')
    record = SESSIONS[ident]
    for name, expected in record['pins'].items():
        need(sha((record['path'] / name).read_bytes()) == expected, 'The reviewed source changed; start a new review.')
    return record


def prepare(original, image, metadata_raw):
    need(0 < len(original) <= restoration.MAX_UPLOAD_BYTES and 0 < len(metadata_raw) <= 128 * 1024,
         'Input or restoration record exceeds bounds.')
    metadata = json.loads(metadata_raw)
    key = metadata_key(metadata)
    need(key in ADMISSIONS, 'This restoration result was not produced in the current app session. Restore or choose Off again.')
    admitted = ADMISSIONS[key]
    need(sha(original) == metadata['uploaded_bytes_sha256'] and sha(image) == metadata['result_file_sha256'],
         'The original or restoration result changed. Start a new review.')
    for name, expected in admitted['pins'].items():
        need(sha((admitted['path'] / name).read_bytes()) == expected, 'Retained restoration evidence changed.')
    rgb = read_png(image, 'RGB')
    bounds = metadata['geometry']['observed_bounds_256']
    need(isinstance(bounds, list) and len(bounds) == 4 and all(type(v) is int for v in bounds), 'Invalid observed support.')
    left, top, right, bottom = bounds
    need(0 <= left < right <= 256 and 0 <= top < bottom <= 256, 'Invalid observed support.')
    native = restoration.decode_original(original)
    original256, observed, _, geometry = restoration.prepare_crop(native)
    need(bounds == geometry['observed_bounds_256'] and
         np.array_equal(read_png((admitted['path']/'original-256.png').read_bytes(), 'RGB'), original256),
         'The original geometry differs from the restoration record.')
    # Bilinear resizing can blend prepared edge pixels into the padding fringe.
    # Restoration preserves the selected input there, which may differ from the original.
    selected256 = read_png((admitted['path']/'selected-input-256.png').read_bytes(), 'RGB')
    need(np.array_equal(rgb[~observed], selected256[~observed]), 'Restoration padding changed.')
    need(len(SESSIONS) < 100, 'Completion review is full; restart the app to review more outputs.')
    ident = uuid.uuid4().hex
    path = STORE / 'sessions' / ident
    path.mkdir(parents=True, exist_ok=False)
    files = {name: (admitted['path']/name).read_bytes() for name in admitted['pins']}
    files.update({'uploaded-original.bin': original, 'input.png': image, 'observed.png': png(observed.astype(np.uint8)*255)})
    for name, value in files.items():
        save(path/name, value)
    SESSIONS[ident] = {'path': path, 'pins': {name: sha(value) for name, value in files.items()},
                      'bounds': bounds, 'restoration': metadata, 'proposals': {}, 'review': None}
    save(path/'session.json', canonical({'session_id': ident, 'input_sha256': sha(image), 'bounds': bounds,
                                       'restoration_metadata_sha256': key}))
    return {'session_id': ident, 'input_png': restoration.url(image), 'bounds': bounds,
            'completion_source': 'Exact reviewed restoration result; completion does not rerun restoration.'}


def run_worker(record, kind, mask=None):
    need(kind in ('detector', 'completion'), 'Unknown stage.')
    if psutil.virtual_memory().available < 2684354560:
        raise RuntimeError('This stage needs at least 2.5 GiB available RAM. Close unused applications and retry.')
    request_root = record['path'] / (kind + '-' + uuid.uuid4().hex)
    request_root.mkdir()
    save(request_root/'input.png', (record['path']/'input.png').read_bytes())
    request = {'input_path': str(request_root/'input.png'), 'output_dir': str(request_root/'output')}
    if kind == 'detector':
        request['observed_path'] = str(record['path']/'observed.png')
        module = 'implementation.covering_detector_worker'
    else:
        mask_path = request_root/'mask.png'
        save(mask_path, mask)
        request.update(mask_path=str(mask_path), prepared_bounds=record['bounds'])
        module = 'implementation.completion_worker'
    request_path, response_path = request_root/'request.json', request_root/'result.json'
    save(request_path, canonical(request))
    started = time.monotonic()
    STATE['active_worker'] = True
    process, supervisor = None, None
    try:
        with (request_root/'worker.log').open('xb') as log:
            with OwnedWorker([sys.executable, '-B', '-m', module, '--request', str(request_path),
                              '--result', str(response_path)], cwd=ROOT, stdout=log, stderr=log) as process:
                deadline = time.monotonic() + 120
                try:
                    while True:
                        try:
                            process.wait_tree(timeout=min(.25, max(.001, deadline-time.monotonic())))
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() >= deadline:
                                raise RuntimeError('The stage exceeded 120 seconds. Evidence retained; no automatic retry.')
                            if process.memory_usage().get('peak_job_private_bytes', 0) > 6 * 1024**3:
                                raise RuntimeError('The stage exceeded the 6 GiB owned-worker memory limit.')
                finally:
                    if process.poll() is None or process.active_processes():
                        process.kill()
                        process.wait_tree(timeout=15)
                    supervisor = {'seconds': time.monotonic()-started, 'returncode': process.poll(),
                                  'active_owned_processes': process.active_processes(), 'memory': process.memory_usage()}
                    save(request_root/'supervisor.json', canonical(supervisor))
        need(supervisor['active_owned_processes'] == 0, 'Owned worker is still active.')
        if supervisor['returncode'] != 0 or not response_path.is_file():
            status_path = request_root/'output/status.json'
            state = json.loads(status_path.read_bytes()) if status_path.is_file() else {}
            if state.get('status') == 'refused' or state.get('error_type') in ('ValueError', 'CompletionRefused'):
                raise ValueError(state.get('error', 'The reviewed input was refused.'))
            raise RuntimeError('The '+kind+' worker failed. Evidence is retained; no automatic retry.')
        need(response_path.stat().st_size <= 65536, 'Worker response is too large.')
        result = json.loads(response_path.read_bytes())
        status = json.loads((request_root/'output/status.json').read_bytes())
        expected = 1 if kind == 'detector' else int(bool(read_png(mask, 'L').any()))
        attempts_key, returns_key = (('forward_attempts', 'forward_returns') if kind == 'detector'
                                     else ('neural_attempts', 'neural_returns'))
        need(status.get('phase') == 'complete' and status.get(attempts_key) == expected
             and status.get(returns_key) == expected, 'Worker forward receipt differs.')
        STATE['detector_calls' if kind == 'detector' else 'completion_calls'] += expected
        return request_root/'output', result, supervisor
    except BaseException as exc:
        if supervisor is None or supervisor.get('active_owned_processes') != 0:
            STATE['uncertain'] = True
        save(request_root/'failure.json', canonical({'error': str(exc)[:1000], 'supervisor': supervisor}))
        raise
    finally:
        STATE['active_worker'] = False


def propose(ident):
    record = session(ident)
    output, result, supervisor = run_worker(record, 'detector')
    names = {'probability_path': 'detector-probability.npy', 'raw_mask_path': 'detector-raw-mask.png',
             'proposed_mask_path': 'detector-proposed-mask.png'}
    for key, name in names.items():
        need(Path(result[key]).resolve() == (output/name).resolve(), 'Detector output path differs.')
    probability = np.load(io.BytesIO((output/names['probability_path']).read_bytes()), allow_pickle=False)
    raw = read_png((output/names['raw_mask_path']).read_bytes(), 'L')
    proposed_raw = (output/names['proposed_mask_path']).read_bytes()
    proposed = read_png(proposed_raw, 'L')
    from .covering_detector import proposal_from_probability
    observed = read_png((record['path']/'observed.png').read_bytes(), 'L') > 0
    expected = proposal_from_probability(probability, observed)
    need(np.array_equal(raw, expected[0]) and np.array_equal(proposed, expected[1]), 'Detector proposal replay differs.')
    proposal_id = uuid.uuid4().hex
    record['proposals'][proposal_id] = {'mask': proposed.copy(), 'path': output, 'metadata': result['metadata'],
                                        'sha256': sha(proposed_raw), 'supervisor': supervisor}
    record['review'] = None
    return {'proposal_id': proposal_id, 'proposal_png': restoration.url(proposed_raw),
            'message': ('No covering was proposed. This does not establish an uncovered face; paint any covered area.'
                        if not proposed.any() else 'Correct the proposed area before confirming. Clear glasses and visible features must stay protected.'),
            'metadata': result['metadata']}


def review(ident, mask_raw, proposal_id=''):
    record = session(ident)
    mask = read_png(mask_raw, 'L')
    from .completion_runtime import visibility_check256
    observed = read_png((record['path']/'observed.png').read_bytes(), 'L') > 0
    need(not mask[~observed].any(), 'Removal must not include neutral padding.')
    visibility = visibility_check256(np.where(observed, mask, 255))
    need(not visibility['rejected'], 'Too little visible face remains. Upload a clearer or less-covered crop.')
    need(not proposal_id or proposal_id in record['proposals'], 'The proposal is stale. Review this mask again.')
    review_id = uuid.uuid4().hex
    record['review'] = {'id': review_id, 'mask_sha256': sha(mask_raw), 'proposal_id': proposal_id}
    save(record['path']/('review-'+review_id+'.json'), canonical({**record['review'], 'visibility': visibility}))
    return {'review_id': review_id, 'mask_png': restoration.url(mask_raw), 'visibility': visibility}


def generate(ident, mask_raw, review_id, mask_confirmed, visible_confirmed, proposal_id=''):
    record = session(ident)
    need(mask_confirmed is True and visible_confirmed is True, 'Confirm the removal area and sufficient visible facial structure.')
    approved = record['review']
    need(approved and approved['id'] == review_id and approved['mask_sha256'] == sha(mask_raw)
         and approved['proposal_id'] == proposal_id, 'The removal mask or input changed. Review and confirm again.')
    mask = read_png(mask_raw, 'L')
    proposal = record['proposals'].get(proposal_id)
    source = ('automatic_reviewed' if np.array_equal(mask, proposal['mask']) else 'assisted_corrected') if proposal else 'assisted_manual'
    output, result, supervisor = run_worker(record, 'completion', mask_raw)
    for field, filename in [('png_path', 'completion.png'), ('raw_path', 'raw.npy'), ('stages_path', 'stages.npz')]:
        need(Path(result[field]).resolve() == (output/filename).resolve(), 'Completion output path differs.')
    output_raw = (output/'completion.png').read_bytes()
    final = read_png(output_raw, 'RGB')
    original = read_png((record['path']/'input.png').read_bytes(), 'RGB')
    raw = np.load(io.BytesIO((output/'raw.npy').read_bytes()), allow_pickle=False)
    need(raw.dtype == np.float32 and raw.shape == (256,256,3) and np.isfinite(raw).all()
         and 0 <= raw.min() <= raw.max() <= 1, 'Invalid RAW completion.')
    replay = np.rint(raw*np.float32(255)).clip(0,255).astype(np.uint8)
    replay[mask == 0] = original[mask == 0]
    need(np.array_equal(final, replay) and np.array_equal(final[mask == 0], original[mask == 0]),
         'Completion changed pixels outside the reviewed removal area or PNG replay differs.')
    need(np.array_equal(raw[mask == 0], original.astype(np.float32)[mask == 0]/np.float32(255)), 'RAW visible pixels changed.')
    with np.load(output/'stages.npz', allow_pickle=False) as stages:
        need(set(stages.files) == {'input_rgb','removal_mask','completed_rgb','output_rgb'} and
             np.array_equal(stages['input_rgb'], original) and np.array_equal(stages['removal_mask'], mask//255)
             and np.array_equal(stages['completed_rgb'], raw) and np.array_equal(stages['output_rgb'], final), 'Saved completion stages differ.')
    receipt = json.loads((output/'receipt.json').read_bytes())
    need(receipt == result['metadata'], 'Completion receipt differs from the worker response.')
    from .completion_runtime import model_contract
    contract = model_contract()
    need(all(receipt.get(key) == value for key,value in contract.items()), 'Completion model or processing provenance differs.')
    for name,item in receipt['output_pins'].items():
        need(name in ('raw.npy','raw512.npy','input512.npy','stages.npz','completion.png') and
             item['filename'] == name, 'Unexpected completion evidence member.')
        value = (output/name).read_bytes()
        need(len(value) == item['bytes'] and sha(value) == item['sha256'], 'Completion evidence fingerprint differs.')
    need(receipt['source_pins']['input']['sha256'] == record['pins']['input.png'] and
         receipt['source_pins']['mask']['sha256'] == sha(mask_raw), 'Worker input bindings differ.')
    metadata = {**result['metadata'], 'source': source, 'proposal_id': proposal_id or None,
                'review_id': review_id, 'restoration': record['restoration'], 'supervisor': supervisor,
                'seconds': supervisor['seconds'], 'neural_calls': int(bool(mask.any())),
                'outside_reviewed_support_exact_to': 'restoration-result.png',
                'exact_hidden_identity_claimed': False, 'model_qualified': False}
    save(output/'combined-workflow.json', canonical(metadata))
    payloads = {name: (record['path']/name).read_bytes() for name in record['pins']}
    payloads.update({'reviewed-removal-mask.png': mask_raw, 'completion-result.png': output_raw,
                     'completion-stages.npz': (output/'stages.npz').read_bytes(), 'combined-workflow.json': canonical(metadata),
                     'completion-receipt.json': (output/'receipt.json').read_bytes(), 'completion-supervisor.json': canonical(supervisor)})
    if proposal:
        for name in ('detector-probability.npy','detector-raw-mask.png','detector-proposed-mask.png','receipt.json'):
            payloads['proposal/'+name] = (proposal['path']/name).read_bytes()
    payloads['README.txt'] = ("Original upload, restoration and completion are separate stages. R1 changes only the reviewed removal area of restoration-result.png. "
                               "Restoration may change observed appearance. Hidden pixels are one plausible estimate, not recovered identity. "
                               "Automatic proposals can miss coverings and require review. source records automatic_reviewed versus assisted correction. "
                               "No training or external inference API is used.\n").encode()
    payloads['manifest.json'] = canonical({'sha256': {name: sha(value) for name,value in payloads.items()}})
    bundle = restoration.bundle(payloads)
    save(output/'evidence.zip', bundle)
    record['review'] = None
    return {'result_png': restoration.url(output_raw), 'mask_png': restoration.url(mask_raw),
            'bundle': restoration.url(bundle, 'application/zip'), 'metadata': metadata, 'neural_calls': int(bool(mask.any()))}


def register(app):
    app.add_middleware(RestorationReceiptMiddleware)
    app.add_middleware(BoundedResponseMiddleware)

    @app.get('/face/completion/status')
    def status():
        from .completion_assets import verify_assets
        try:
            verify_assets()
            available, error = True, None
        except (OSError, ValueError, RuntimeError) as exc:
            available, error = False, str(exc)
        return {'available': available, 'error': error, 'model': 'completion_r1', 'state': dict(STATE),
                'automatic_proposal_requires_review': True, 'model_qualified': False, 'local_cpu': True}

    @app.post('/face/completion/prepare')
    async def prepare_route(file: UploadFile = File(...), original: UploadFile = File(...), restoration_metadata: str = Form(...)):
        image = await file.read(MAX_PNG+1)
        contents = await original.read(restoration.MAX_UPLOAD_BYTES+1)
        return await run_in_threadpool(invoke, prepare, contents, image, restoration_metadata.encode())

    @app.post('/face/completion/propose')
    async def propose_route(session_id: str = Form(...)):
        return await run_in_threadpool(invoke, propose, session_id)

    @app.post('/face/completion/review')
    async def review_route(session_id: str = Form(...), mask: UploadFile = File(...), proposal_id: str = Form('')):
        raw = await mask.read(MAX_PNG+1)
        return await run_in_threadpool(invoke, review, session_id, raw, proposal_id)

    @app.post('/face/completion/generate')
    async def generate_route(session_id: str = Form(...), mask: UploadFile = File(...), review_id: str = Form(''),
                             mask_confirmed: bool = Form(False), visible_confirmed: bool = Form(False), proposal_id: str = Form('')):
        raw = await mask.read(MAX_PNG+1)
        return await run_in_threadpool(invoke, generate, session_id, raw, review_id, mask_confirmed, visible_confirmed, proposal_id)

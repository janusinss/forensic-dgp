"""Assisted surroundings-only input experiment beside the unchanged normal route.

Preview, refusal and mask editing use image operations only. Actual restoration
uses the existing frozen comparison engine; this module imports no training code.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, __version__ as PILLOW
from scipy.ndimage import distance_transform_edt
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

import research_spatial_restoration_v1 as spatial
from research_filter_restoration_web_v1 import decode_crop, prepare_crop, review_input

ROOT = Path(__file__).resolve().parent
LIMIT = 10 * 1024**2
MAX_SIDE = 1600
MAX_PIXELS = 2_560_000
EXAMPLE_SHA = 'ca59a8cd5602777fd7493a090cb545e71ffe3c132bd68a76f1f1f8d24cb9af3a'
POLYGON = [(72,58),(386,58),(394,428),(364,483),(97,490),(55,439),(36,284),(36,201),(61,167)]
ALGORITHM = {
    'version': 'assisted-surroundings-preview-v1', 'Gaussian_nominal_radius_at256': 6.0,
    'outward_guard_native_pixels': 8.0, 'outside_feather_native_pixels': 16.0,
    'distance': 'Euclidean EDT', 'feather': 'smoothstep(distance/16) outside guarded protection only',
    'blend': 'float64 RGB, float32 alpha, np.rint ties-to-even RGB8', 'Pillow': PILLOW,
}
router = APIRouter(prefix='/restoration-context-test', tags=['Assisted surroundings experiment'])
pair_lock = threading.Lock()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _png(array):
    stream = io.BytesIO(); Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def _url(data, mime='image/png'):
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')


def _blob(value):
    head, payload = value.split(',', 1)
    if not head.startswith('data:') or not head.endswith(';base64'):
        raise ValueError('Invalid upstream comparison payload.')
    return base64.b64decode(payload, validate=True)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _fingerprints():
    names = ('research_surroundings_restoration_v1.py', 'research_spatial_restoration_v1.py',
             'research_filter_restoration_web_v1.py', 'research_restoration_web.py')
    result = {}
    for name in names:
        data = (ROOT/name).read_bytes()
        result[name] = {'bytes': len(data), 'sha256': _sha(data)}
    return result


def _install_startup_pair_serialization():
    """Upgrade only an unused per-process engine; never hot-swap a running lock."""
    if getattr(spatial.engine, '_assisted_surroundings_reentrant_lock', False):
        return
    original_lock = spatial.engine.lock
    if not original_lock.acquire(blocking=False):
        raise RuntimeError('Load the surroundings router only during fresh server startup.')
    try:
        status = spatial.engine.measured_status()
        if spatial.engine.models or any(status[key] for key in (
                'neural_forward_attempts', 'neural_forward_returns',
                'neural_forward_successes', 'completed_comparisons')):
            raise RuntimeError('Load the surroundings router before any model is loaded or called.')
        spatial.engine.lock = threading.RLock()
        spatial.engine._assisted_surroundings_reentrant_lock = True
    finally:
        original_lock.release()


_install_startup_pair_serialization()
SERIALIZATION = ('Startup-only unused-engine RLock; own nonblocking pair lock plus shared '
                 'reentrant engine lock held across both same-thread generate calls. '
                 'Other-thread normal comparisons retain nonblocking busy refusal.')


def _decode(contents):
    rgb = decode_crop(contents)
    height, width = rgb.shape[:2]
    if max(width, height) > MAX_SIDE or width*height > MAX_PIXELS:
        raise ValueError('Upload a smaller face crop: this preview supports at most 1600 pixels per side.')
    return rgb


def _protection(value, width, height, source_sha):
    if type(value) is not str or not value or len(value) > 2048:
        raise ValueError('Choose and review one protection rectangle.')
    def unique(pairs):
        row = {}
        for key, item in pairs:
            if key in row: raise ValueError('Protection fields must be unique.')
            row[key] = item
        return row
    try:
        row = json.loads(value, object_pairs_hook=unique,
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Finite protection values required.')))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('Protection must be a valid rectangle object.') from error
    mask = Image.new('L', (width,height), 0); draw = ImageDraw.Draw(mask)
    if type(row) is dict and set(row) == {'preset'} and row['preset'] == 'reviewed-wild5':
        if source_sha != EXAMPLE_SHA or (width,height) != (455,523):
            raise ValueError('The reviewed example protection is valid only for the unchanged wild5 image.')
        draw.polygon(POLYGON, fill=255); draw.rectangle((70,55,388,488), fill=255)
        spec = {'preset': 'reviewed-wild5'}
    elif type(row) is dict and set(row) == {'rectangle'}:
        rect = row['rectangle']
        if type(rect) is not list or len(rect) != 4 or any(type(x) is not int for x in rect):
            raise ValueError('Use four native whole-pixel rectangle coordinates.')
        left,top,right,bottom = rect
        if not (0 <= left <= right < width and 0 <= top <= bottom < height):
            raise ValueError('Protection rectangle must stay inside the uploaded crop.')
        if right-left+1 < 16 or bottom-top+1 < 16 or (right-left+1)*(bottom-top+1) < max(256,width*height*.01):
            raise ValueError('Protection is too small. Cover the entire face and existing blur patch.')
        draw.rectangle(tuple(rect), fill=255); spec = {'rectangle': list(rect)}
    else:
        raise ValueError('Use one protection rectangle or the reviewed example preset.')
    union = np.asarray(mask) > 0
    protected = distance_transform_edt(~union) <= ALGORITHM['outward_guard_native_pixels']
    if np.count_nonzero(~protected) < 16:
        raise ValueError('The protection leaves no useful surroundings. Choose a smaller reviewed region.')
    return protected, spec


def make_preview(contents, protection):
    """Recompute deterministic native pixels; nothing is persisted on the server."""
    rgb = _decode(contents); height,width = rgb.shape[:2]; source_sha = _sha(contents)
    protected,spec = _protection(protection, width, height, source_sha)
    distance = distance_transform_edt(~protected)
    t = np.clip(distance/ALGORITHM['outside_feather_native_pixels'], 0, 1)
    alpha = (t*t*(3-2*t)).astype(np.float32)
    if np.any(alpha[protected]): raise ValueError('The protected region received blur weight.')
    native_radius = ALGORITHM['Gaussian_nominal_radius_at256']*max(width,height)/256
    blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=native_radius)))
    mixed = np.rint(rgb.astype(np.float64)*(1-alpha[...,None].astype(np.float64)) +
                    blurred.astype(np.float64)*alpha[...,None].astype(np.float64)).clip(0,255).astype(np.uint8)
    if not np.array_equal(mixed[protected], rgb[protected]):
        raise ValueError('Protected native face pixels changed during preprocessing.')
    original256,support,_,original_geometry = prepare_crop(rgb)
    derived256,derived_support,_,derived_geometry = prepare_crop(mixed)
    if not np.array_equal(support,derived_support): raise ValueError('Observed crop support changed.')
    side = max(width,height); left,top = (side-width)//2,(side-height)//2
    mask = (protected*255).astype(np.uint8)
    mask_canvas = Image.new('L',(side,side),0); mask_canvas.paste(Image.fromarray(mask),(left,top))
    interior = np.asarray(mask_canvas.resize((256,256),Image.Resampling.BILINEAR)) == 255
    if not np.any(interior) or not np.array_equal(original256[interior],derived256[interior]):
        raise ValueError('Conservative prepared face pixels were not preserved.')
    colors = np.empty_like(rgb); colors[:] = (210,70,70)
    colors[(alpha > 0)&(alpha < 1)] = (245,180,45); colors[protected] = (45,185,90)
    overlay = np.rint(rgb.astype(np.float64)*.60 + colors.astype(np.float64)*.40).astype(np.uint8)
    native_original = _png(rgb); native_derived = _png(mixed)
    mask_png = _png(mask); alpha_stream = io.BytesIO(); np.save(alpha_stream,alpha,allow_pickle=False)
    fingerprint = _fingerprints()
    identity = {'source_sha256':source_sha,'protection':spec,'algorithm':ALGORITHM,'source_fingerprints':fingerprint}
    preview_id = _sha(_json(identity))
    metadata = {'policy':'assisted-surroundings-preview-v1','preview_id':preview_id,'source_sha256':source_sha,
        'uploaded_original_bytes':len(contents),'original_RGB_sha256':_sha(rgb.tobytes()),
        'derived_RGB_sha256':_sha(mixed.tobytes()),'derived_PNG_sha256':_sha(native_derived),
        'source_size':[width,height],'protection':spec,'algorithm':dict(ALGORITHM),
        'Gaussian_native_radius':native_radius,'protected_native_pixels':int(protected.sum()),
        'protected_prepared_interior_pixels':int(interior.sum()),
        'protected_native_RGB_values_exact':True,'protected_prepared_interior_RGB_exact':True,
        'protection_mask_RGB8_sha256':_sha(mask.tobytes()),'alpha_float32_sha256':_sha(alpha.tobytes()),
        'native_original_RGB_PNG_sha256':_sha(native_original),
        'original_geometry':original_geometry,'derived_geometry':derived_geometry,
        'manual_assisted_protection_not_automatic_blur_detection':True,
        'no_completion_or_covering_removal':True,'blur_matching_is_approximate_unknown_original_kernel':True,
        'protected_pixel_guarantee_scope':'Input preprocessing only, not generated model output.',
        'native_input_has_no_aligned_clean_reference_in_this_flow':True,
        'source_fingerprints':fingerprint,'serialization_policy':SERIALIZATION,
        'model_qualified':False,'main_app_default_changed':False,'training_optimizer_VM_calls':0,
        'preprocessing_neural_calls':0,'server_preview_cache':False}
    payloads = {'uploaded-original.bin':contents,'original-decoded-rgb.png':native_original,
        'protection-native.png':mask_png,'blur-weight-float32.npy':alpha_stream.getvalue(),
        'derived-native.png':native_derived,'prepared-original-256.png':_png(original256),
        'prepared-derived-256.png':_png(derived256),'observed-support-256.png':_png((support*255).astype(np.uint8)),
        'protected-interior-256.png':_png((interior*255).astype(np.uint8))}
    metadata['bundle_payloads'] = {name:{'bytes':len(data),'sha256':_sha(data)} for name,data in payloads.items()}
    output = io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in payloads.items(): z.writestr(name,data)
        z.writestr('preprocessing.json',json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    response = {'preview_id':preview_id,'original':_url(native_original),'prepared_input':_url(native_derived),
        'protection_overlay':_url(_png(overlay)),'width':width,'height':height,'metadata':metadata,
        'bundle':_url(output.getvalue(),'application/zip')}
    return response, native_derived, original256, derived256


def compare_pair(contents, protection, preview_id, confirmed, input_review, selection):
    review_input(input_review)
    if confirmed is not True: raise ValueError('Confirm the displayed protected area before restoration.')
    if selection not in ('original','r2','spatial_u'): raise ValueError('Choose U96, retained R2 or untouched CodeFormer.')
    preview,derived,original256,derived256 = make_preview(contents, protection)
    if type(preview_id) is not str or preview_id != preview['preview_id']:
        raise ValueError('The preview is stale. Preview and confirm the current image and protection again.')
    if not pair_lock.acquire(blocking=False):
        raise RuntimeError('A surroundings comparison is already running. Wait for it to finish.')
    shared_held = False
    try:
        if not spatial.engine.lock.acquire(blocking=False):
            raise RuntimeError('A comparison is already running. Wait for it to finish.')
        shared_held = True; before = spatial.engine.measured_status(); started = time.monotonic()
        original_result = spatial.engine.generate(contents,input_review,selection)
        prepared_result = spatial.engine.generate(derived,input_review,selection)
        after = spatial.engine.measured_status()
        for result,expected_sha in ((original_result,_sha(contents)),(prepared_result,_sha(derived))):
            if result['metadata']['uploaded_bytes_sha256'] != expected_sha or set(result['outputs']) != {selection}:
                raise ValueError('Upstream restoration does not match the reviewed input.')
        original_zip = _blob(original_result['bundle']); prepared_zip = _blob(prepared_result['bundle'])
        preprocessing_zip = _blob(preview['bundle'])
        metadata = {'policy':'assisted-surroundings-original-versus-prepared-v1','preview_id':preview_id,
            'selection':selection,'seconds':time.monotonic()-started,'confirmed_protection':True,
            'preprocessing':preview['metadata'],'original_result':original_result['metadata'],
            'prepared_result':prepared_result['metadata'],'server_before':before,'server_after':after,
            'own_planned_neural_calls':2,'no_automatic_retry':True,'serialization_policy':SERIALIZATION,
            'protected_pixel_guarantee_scope':'Preprocessing input only; generated facial appearance may change.',
            'model_qualified':False,'main_app_default_changed':False,'training_optimizer_VM_calls':0,
            'bundles':{'original-comparison.zip':{'bytes':len(original_zip),'sha256':_sha(original_zip)},
                'prepared-comparison.zip':{'bytes':len(prepared_zip),'sha256':_sha(prepared_zip)},
                'preprocessing.zip':{'bytes':len(preprocessing_zip),'sha256':_sha(preprocessing_zip)}}}
        output = io.BytesIO()
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('original-comparison.zip',original_zip); z.writestr('prepared-comparison.zip',prepared_zip)
            z.writestr('preprocessing.zip',preprocessing_zip)
            z.writestr('comparison.json',json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        return {'original_input':_url(_png(original256)),'prepared_input':_url(_png(derived256)),
            'original_result':original_result['outputs'][selection],'prepared_result':prepared_result['outputs'][selection],
            'metadata':metadata,'bundle':_url(output.getvalue(),'application/zip')}
    finally:
        if shared_held: spatial.engine.lock.release()
        pair_lock.release()


@router.get('',response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT/'templates/research_surroundings_restoration_v1.html').read_text(encoding='utf-8'),
                        headers={'Cache-Control':'no-store'})


@router.get('/status')
def status():
    value = spatial.status()
    return {'route':'/restoration-context-test','models':{a:value['models'][a] for a in ('spatial_u','r2','original')},
        'preferred_development_model':spatial.PREFERRED_ARM,
        'quality_status':value['quality_status'],'model_qualified':False,'main_app_default_changed':False,
        'manual_protection_review_required':True,'automatic_boundary_detection':False,
        'preprocessing_optional_and_experimental':True,'max_native_side':MAX_SIDE,'max_native_pixels':MAX_PIXELS,
        'algorithm':ALGORITHM,'serialization_policy':SERIALIZATION,**spatial.engine.measured_status()}


@router.get('/example')
def example():
    data = (ROOT/'test_images/wild5.png').read_bytes()
    if _sha(data) != EXAMPLE_SHA: raise HTTPException(503,'The reviewed example changed.')
    return Response(data,media_type='image/png',headers={
        'Cache-Control':'no-store','Content-Disposition':'inline; filename="wild5.png"'})


@router.post('/inspect')
async def inspect(file: UploadFile = File(...)):
    try:
        contents = await file.read(LIMIT+1); rgb = await run_in_threadpool(_decode, contents)
        height,width = rgb.shape[:2]
        return {'source':_url(_png(rgb)),'width':width,'height':height,'source_sha256':_sha(contents)}
    except ValueError as error: raise HTTPException(422,str(error)) from error


@router.post('/preview')
async def preview(file: UploadFile = File(...), protection: str = Form('')):
    try:
        result,_,_,_ = await run_in_threadpool(make_preview, await file.read(LIMIT+1), protection)
        return result
    except ValueError as error: raise HTTPException(422,str(error)) from error


@router.post('/compare')
async def compare(file: UploadFile = File(...), protection: str = Form(''), preview_id: str = Form(''),
                  confirmed: bool = Form(False), input_review: str = Form(''), selection: str = Form(spatial.PREFERRED_ARM)):
    try:
        return await run_in_threadpool(compare_pair,await file.read(LIMIT+1),protection,preview_id,
                                       confirmed,input_review,selection)
    except ValueError as error: raise HTTPException(422,str(error)) from error
    except RuntimeError as error: raise HTTPException(409,str(error)) from error
    except (OSError,FileNotFoundError) as error:
        raise HTTPException(503,'The verified experimental model or source is unavailable.') from error

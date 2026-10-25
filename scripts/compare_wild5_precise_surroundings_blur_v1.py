"""Controlled surroundings-only preprocessing experiment; no training or VM.

The user requested the sharp surroundings to approximate the already blurred
face. Pixels in the declared face polygon are copied exactly, not regenerated.
One fixed Gaussian strength; no per-model tuning or parameter search.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, __version__ as PILLOW
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT = ROOT / 'outputs/codeformer_wild5_precise_surroundings_blur_v1'
SOURCE_SHA = 'ca59a8cd5602777fd7493a090cb545e71ffe3c132bd68a76f1f1f8d24cb9af3a'
TRANSPORT_SHA = '3e5e69b8a58c47f5395eed0953d022c695224f8349f634af6188b8c478111ac2'
POLYGON = [(72,58),(386,58),(394,428),(364,483),(97,490),(55,439),(36,284),(36,201),(61,167)]
NOMINAL_RADIUS_256 = 6.0
FEATHER_NATIVE = 16.0
PATCH_RECTANGLE = (70,55,388,488)
MANUAL_EXISTING_PATCH = (75,62,383,478)
OUTWARD_GUARD_NATIVE = 8.0


def digest(b): return hashlib.sha256(b).hexdigest()


def pin(p):
    b=p.read_bytes()
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':digest(b)}


def dump(p,value): p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def checked(binding):
    p=ROOT/binding['path'];b=p.read_bytes()
    assert len(b)==binding['bytes'] and digest(b)==binding['sha256']
    return b


def preparation():
    assert not OUT.exists()
    OUT.mkdir()
    source=ROOT/'test_images/wild5.png';data=source.read_bytes();assert digest(data)==SOURCE_SHA
    with Image.open(io.BytesIO(data)) as im:
        assert im.format=='PNG' and im.mode=='RGBA' and im.size==(455,523)
        original=ImageOps.exif_transpose(im).convert('RGB')
    rgb=np.asarray(original).copy()
    mask=Image.new('L',original.size,0)
    draw=ImageDraw.Draw(mask);draw.polygon(POLYGON,fill=255);draw.rectangle(PATCH_RECTANGLE,fill=255)
    union=np.asarray(mask)>0
    protect=distance_transform_edt(~union)<=OUTWARD_GUARD_NATIVE
    mask=Image.fromarray((protect*255).astype(np.uint8))
    patch_mask=Image.new('L',original.size,0);ImageDraw.Draw(patch_mask).rectangle(MANUAL_EXISTING_PATCH,fill=255)
    patch=np.asarray(patch_mask)>0
    assert protect[patch].all()
    distance=distance_transform_edt(~protect)
    t=np.clip(distance/FEATHER_NATIVE,0,1)
    alpha=(t*t*(3-2*t)).astype(np.float32);assert (alpha[protect]==0).all()
    radius=NOMINAL_RADIUS_256*max(original.size)/256
    blurred=np.asarray(original.filter(ImageFilter.GaussianBlur(radius=radius)))
    mixed=np.rint(rgb.astype(np.float64)*(1-alpha[...,None].astype(np.float64))+
                  blurred.astype(np.float64)*alpha[...,None].astype(np.float64)).clip(0,255).astype(np.uint8)
    assert np.array_equal(mixed[protect],rgb[protect])
    assert np.array_equal(mixed[patch],rgb[patch])
    Image.fromarray(mixed).save(OUT/'wild5_surroundings_blurred.png')
    mask.save(OUT/'protected_face_region.png')
    Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(OUT/'surroundings_blur_weight.png')
    np.save(OUT/'surroundings_blur_weight-float32.npy',alpha,allow_pickle=False)
    # Match the existing app's neutral canvas; no model or Torch is imported.
    import research_filter_restoration_web_v1 as processing
    prepared,support,_,geometry=processing.prepare_crop(mixed)
    prepared_original,original_support,_,_=processing.prepare_crop(rgb)
    assert np.array_equal(support,original_support)
    side=max(original.size);left=(side-original.width)//2;top=(side-original.height)//2
    mask_canvas=Image.new('L',(side,side),0);mask_canvas.paste(mask,(left,top))
    # Conservative complete interpolation-support interior, not a face detector.
    interior=np.asarray(mask_canvas.resize((256,256),Image.Resampling.BILINEAR))==255
    assert np.array_equal(prepared[interior],prepared_original[interior])
    Image.fromarray(prepared).save(OUT/'prepared-input.png')
    Image.fromarray((interior*255).astype(np.uint8)).save(OUT/'protected_face_interior256.png')
    result={'complete':True,'status':'PREPARED_INPUT_ONLY_NOT_INFERRED','source':pin(source),
        'source_mode':'RGBA','decode_policy':'EXIF transpose then RGB, matching app; no alpha composite',
        'source_size':[455,523],'protected_face_polygon_xy':POLYGON,
        'protected_union_rectangle_xyxy_inclusive':list(PATCH_RECTANGLE),
        'manually_estimated_existing_blur_patch_xyxy_inclusive':list(MANUAL_EXISTING_PATCH),
        'outward_guard_native_pixels':OUTWARD_GUARD_NATIVE,
        'full_manual_existing_patch_protected':True,
        'full_manual_existing_patch_RGB_exact':True,
        'manual_existing_patch_pixels':int(patch.sum()),
        'boundary_estimate_is_manual_not_automatic_ground_truth':True,
        'assisted_preprocessing_not_covering_completion_mask':True,
        'protected_native_pixels':int(protect.sum()),'protected_native_RGB_values_exact':True,
        'protected_prepared_interior_pixels':int(interior.sum()),'protected_prepared_interior_RGB_exact':True,
        'whole_source_face_information_added':False,'blur':{'implementation':'Pillow ImageFilter.GaussianBlur',
            'Pillow':PILLOW,'nominal_radius_at256':NOMINAL_RADIUS_256,'native_radius_pixels':radius,
            'matching_status':'Visual approximation only; original face blur kernel unknown and not estimated.',
            'feather_native_pixels':FEATHER_NATIVE,'feather_policy':'Outside protected polygon only; smoothstep(distance/16)',
            'rounding':'float64 blend with float32 alpha, np.rint ties-to-even toRGB8',
            'strength_fixed_before_inference':True,'parameter_search':False},
        'prepared_geometry':geometry,'derived_input':pin(OUT/'wild5_surroundings_blurred.png'),
        'protected_mask':pin(OUT/'protected_face_region.png'),
        'alpha_float':pin(OUT/'surroundings_blur_weight-float32.npy'),
        'prepared_input':pin(OUT/'prepared-input.png'),
        'prepared_protected_interior':pin(OUT/'protected_face_interior256.png'),
        'training_optimizer_VM_calls':0,'neural_calls':0,'model_qualified':False,
        'input_is_unpaired_stress_not_automatic_usable_admission':True}
    dump(OUT/'input_preparation.json',result)
    assert 'torch' not in sys.modules
    print(json.dumps({'prepared':True,'protected_face_exact':True,'radius_native':radius,
                      'new_neural_calls':0,'input':result['derived_input']}))


def run(expected_sha):
    prep=OUT/'input_preparation.json';assert digest(prep.read_bytes())==expected_sha
    prepared=json.loads(prep.read_bytes());contents=checked(prepared['derived_input'])
    assert prepared['protected_native_RGB_values_exact'] and prepared['protected_prepared_interior_RGB_exact']
    assert not (OUT/'experiment.json').exists() and not (OUT/'transport').exists()
    worker=ROOT/'scripts/compare_codeformer_spatial_dev_local_v1_r2.py';assert pin(worker)['sha256']==TRANSPORT_SHA
    spec=importlib.util.spec_from_file_location('saved_transport',worker)
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    transport.OUT=OUT;transport.RUN_STARTED=time.monotonic()
    prior=ROOT/'outputs/codeformer_wild5_whole_blur_v1_r1'
    baseline=json.loads((prior/'experiment.json').read_bytes());assert baseline['complete']
    assert baseline['source']['sha256']==SOURCE_SHA and len(baseline['variants'])==2
    for variant in baseline['variants']:
        for binding in variant['files']:checked(binding)
    result={'complete':False,'scope':'One user-requested surroundings-only approximate blur, unpaired stress test.',
        'input_preparation':pin(prep),'worker':pin(Path(__file__)),'transport_worker':pin(worker),
        'models':list(transport.ARMS),'new_planned_requests_and_forwards':4,
        'baseline_and_prior_whole_blur_reused':{'experiment':pin(prior/'experiment.json'),
            'original_folder':(prior/'original_input').relative_to(ROOT).as_posix(),
            'whole_blur_folder':(prior/'whole_gaussian_1p2').relative_to(ROOT).as_posix(),
            'reuse_note':'Exact saved outputs reused; no extra baseline forwards.'},
        'automatic_input_threshold_validated':False,'model_qualified':False,'main_app_default_changed':False,
        'training_optimizer_VM_calls':0,'face_information_added_by_preprocessing':False}
    dump(OUT/'experiment.json',result)
    url=json.loads((ROOT/'scratch/codeformer_spatial_research_server_desktop_v1.json').read_bytes())['url']
    start=time.monotonic()
    try:
        result['server_before']=transport.request(url+'/status')
        assert result['server_before']['neural_forward_attempts']==result['server_before']['neural_forward_returns']
        folder=OUT/'surroundings';folder.mkdir()
        response=transport.post(url,contents)
        metadata=response['metadata'];archive=transport.blob(response['bundle'])
        (folder/'comparison.zip').write_bytes(archive);dump(folder/'metadata.json',metadata)
        assert metadata['geometry']==prepared['prepared_geometry']
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert z.testzip() is None and len(z.namelist())==12
            canvas=transport.rgb(z.read('resize-control.png'))
            assert np.array_equal(canvas,transport.rgb(checked(prepared['prepared_input'])))
            with Image.open(io.BytesIO(z.read('observed-support.png'))) as im:support=np.asarray(im)>0
            (folder/'prepared-input.png').write_bytes(z.read('resize-control.png'))
            (folder/'observed-support.png').write_bytes(z.read('observed-support.png'))
            for arm in transport.ARMS:
                raw=np.load(io.BytesIO(z.read(arm+'-raw.npy')),allow_pickle=False)
                pixels=transport.rgb(z.read(arm+'.png'))
                assert raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all()
                assert raw.min()>=0 and raw.max()<=1
                assert np.array_equal(np.floor(raw*np.float32(255)).astype(np.uint8),pixels)
                assert np.array_equal(pixels[~support],canvas[~support])
                assert digest(raw.tobytes())==metadata['outputs'][arm]['raw_rgb_sha256']
                assert digest(pixels.tobytes())==metadata['outputs'][arm]['PNG_RGB8_sha256']
                assert digest(z.read(arm+'.png'))==metadata['outputs'][arm]['PNG_file_sha256']
                assert transport.blob(response['outputs'][arm])==z.read(arm+'.png')
                np.save(folder/(arm+'-raw.npy'),raw,allow_pickle=False)
                (folder/(arm+'.png')).write_bytes(z.read(arm+'.png'))
        result['response_metadata']=metadata
        result['files']=[pin(p) for p in sorted(folder.iterdir())]
        result['server_after']=transport.request(url+'/status')
        result['server_success_delta']=result['server_after']['neural_forward_successes']-result['server_before']['neural_forward_successes']
        assert result['server_success_delta']>=4
        result['other_server_activity_observed']=result['server_success_delta']!=4
        result['complete']=True
    except BaseException as error:
        result['failure']={'type':type(error).__name__,'message':str(error),'no_automatic_retry':True}
        raise
    finally:
        result['seconds']=time.monotonic()-start;dump(OUT/'experiment.json',result)
    print(json.dumps({'complete':True,'new_forwards':4,'seconds':result['seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run-prepared-sha')
    args=parser.parse_args();assert bool(args.prepare)!=bool(args.run_prepared_sha)
    preparation() if args.prepare else run(args.run_prepared_sha)

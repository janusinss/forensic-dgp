"""User-requested whole-image blur at the prior surroundings test's strength.

One fixed transform, four local inference calls, exact retained baseline reuse.
No actual training, VM, app defaults, or parameter search.
"""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageFilter, ImageOps, __version__ as PILLOW

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/codeformer_wild5_whole_strong_blur_v1'
ARMS=('original','r2','spatial_u','spatial_s')


def digest(b):return hashlib.sha256(b).hexdigest()


def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':digest(b)}


def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    assert not OUT.exists();OUT.mkdir()
    source=ROOT/'test_images/wild5.png';data=source.read_bytes()
    assert digest(data)=='ca59a8cd5602777fd7493a090cb545e71ffe3c132bd68a76f1f1f8d24cb9af3a'
    with Image.open(io.BytesIO(data)) as im:
        assert im.format=='PNG' and im.mode=='RGBA' and im.size==(455,523)
        image=ImageOps.exif_transpose(im).convert('RGB')
    prior=ROOT/'outputs/codeformer_wild5_surroundings_blur_v1_r1'
    prepared=json.loads((prior/'input_preparation.json').read_bytes())
    assert prepared['source']['sha256']==digest(data)
    assert prepared['blur']['nominal_radius_at256']==6
    radius=prepared['blur']['native_radius_pixels'];assert radius==12.2578125
    blurred=image.filter(ImageFilter.GaussianBlur(radius=radius))
    input_path=OUT/'wild5_whole_gaussian_6p0.png';blurred.save(input_path)
    worker=ROOT/'scripts/compare_codeformer_spatial_dev_local_v1_r2.py'
    assert pin(worker)['sha256']=='3e5e69b8a58c47f5395eed0953d022c695224f8349f634af6188b8c478111ac2'
    spec=importlib.util.spec_from_file_location('preserved_transport',worker)
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    transport.OUT=OUT;transport.RUN_STARTED=time.monotonic()
    prior_exp=json.loads((prior/'experiment.json').read_bytes());assert prior_exp['complete']
    for entry in prior_exp['files']:
        actual=pin(ROOT/entry['path']);assert actual==entry
    original=ROOT/'outputs/codeformer_wild5_whole_blur_v1_r1/original_input'
    first=json.loads((original.parent/'experiment.json').read_bytes());assert first['complete']
    baseline=next(v for v in first['variants'] if v['id']=='original_input')
    for entry in baseline['files']:assert pin(ROOT/entry['path'])==entry
    record={'complete':False,'user_request':'Blur the WHOLE image using the same stronger blur as the face/surroundings test.',
        'source':pin(source),'derived_input':pin(input_path),'worker':pin(Path(__file__)),
        'transport_worker':pin(worker),'paired_clean_reference':None,'evidence':'unpaired_user_stress',
        'blur':{'implementation':'Pillow ImageFilter.GaussianBlur','Pillow':PILLOW,
            'native_radius_pixels':radius,'nominal_radius_at256':6,
            'scope':'Entire RGB image, including the already-blurred face, before padding/resizing.',
            'strength_matches_prior_surroundings_transform':True,
            'equal_final_face_and_background_blur_claimed':False,
            'limitation':'The face was already blurred; this adds blur there too. Its original kernel is unknown.',
            'parameter_search':False},
        'models':list(ARMS),'new_planned_requests_and_forwards':4,'maximum_launch_checkpoint_seconds':600,
        'baseline_reuse':{'original_experiment':pin(original.parent/'experiment.json'),
            'original_folder':original.relative_to(ROOT).as_posix(),
            'surroundings_experiment':pin(prior/'experiment.json'),
            'surroundings_folder':(prior/'surroundings').relative_to(ROOT).as_posix(),
            'reused_outputs':8,'new_baseline_calls':0},
        'training_optimizer_VM_calls':0,'model_qualified':False,'main_app_default_changed':False,
        'input_is_explicit_user_stress_override_not_automatic_usable_admission':True}
    dump(OUT/'experiment.json',record)
    url=json.loads((ROOT/'scratch/codeformer_spatial_research_server_surroundings_v1.json').read_bytes())['url']
    start=time.monotonic()
    try:
        record['server_before']=transport.request(url+'/status')
        assert record['server_before']['neural_forward_attempts']==record['server_before']['neural_forward_returns']
        assert time.monotonic()-start<600
        folder=OUT/'whole';folder.mkdir()
        response=transport.post(url,input_path.read_bytes())
        metadata=response['metadata'];archive=transport.blob(response['bundle'])
        (folder/'comparison.zip').write_bytes(archive);dump(folder/'metadata.json',metadata)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert z.testzip() is None and len(z.namelist())==12
            canvas=transport.rgb(z.read('resize-control.png'))
            from research_filter_restoration_web_v1 import prepare_crop
            expected,support,_,geometry=prepare_crop(np.asarray(blurred).copy())
            assert np.array_equal(canvas,expected) and metadata['geometry']==geometry
            with Image.open(io.BytesIO(z.read('observed-support.png'))) as im:
                assert np.array_equal(np.asarray(im)>0,support)
            (folder/'prepared-input.png').write_bytes(z.read('resize-control.png'))
            (folder/'observed-support.png').write_bytes(z.read('observed-support.png'))
            for arm in ARMS:
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
        record['response_metadata']=metadata
        record['files']=[pin(p) for p in sorted(folder.iterdir())]
        record['server_after']=transport.request(url+'/status')
        record['server_success_delta']=record['server_after']['neural_forward_successes']-record['server_before']['neural_forward_successes']
        assert record['server_success_delta']>=4
        record['other_server_activity_observed']=record['server_success_delta']!=4
        record['complete']=True
    except BaseException as error:
        record['failure']={'type':type(error).__name__,'message':str(error),'no_retry':True};raise
    finally:
        record['seconds']=time.monotonic()-start;dump(OUT/'experiment.json',record)
        assert 'torch' not in sys.modules
    print(json.dumps({'complete':True,'new_forwards':4,'seconds':record['seconds']}))


if __name__=='__main__':main()

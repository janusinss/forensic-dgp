"""User-requested whole-image Gaussian-blur stress test; no training or VM."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps, ImageFilter, __version__ as PILLOW

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_wild5_whole_blur_v1_r1'


def pin(path):
    b=path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    assert not OUT.exists();OUT.mkdir()
    worker=ROOT/'scripts/compare_codeformer_spatial_dev_local_v1_r2.py'
    assert pin(worker)['sha256']=='3e5e69b8a58c47f5395eed0953d022c695224f8349f634af6188b8c478111ac2'
    spec=importlib.util.spec_from_file_location('fixed_transport',worker)
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    transport.OUT=OUT;transport.RUN_STARTED=time.monotonic()
    source=ROOT/'test_images/wild5.png';contents=source.read_bytes()
    with Image.open(io.BytesIO(contents)) as im:
        assert im.format in ('PNG','JPEG')
        source_format,source_mode=im.format,im.mode
        image=ImageOps.exif_transpose(im).convert('RGB')
    sigma_256=1.2
    sigma_native=sigma_256*max(image.size)/256
    blurred=image.filter(ImageFilter.GaussianBlur(radius=sigma_native))
    path=OUT/'wild5_whole_gaussian_1p2.png';blurred.save(path)
    arms=transport.ARMS
    manifest={'complete':False,'scope':'User-requested unpaired stress test; unknown source, no clean target, no accuracy metrics.',
        'user_request':'using wild5 blur the whole picture then try the model',
        'source':pin(source),'source_format':source_format,'source_mode':source_mode,'decode_policy':'EXIF transpose, convertRGB exactly as existing app; no alpha compositing', 'source_size':list(image.size),'transport_worker':pin(worker),
        'models':list(arms),'maximum_requests_and_neural_forwards':8,'training_optimizer_VM_calls':0,
        'global_blur':{'library':'Pillow ImageFilter.GaussianBlur','Pillow':PILLOW,'sigma_effective_at256':sigma_256,
                       'radius_native_pixels':sigma_native,'policy':'Entire original RGB crop blurred before identical neutral padding/resizing; no local retouching.',
                       'no_parameter_search':True,'derived_input':pin(path)},
        'input_review':'Explicit user-requested stress override; wild5 is heavily blurred, usable anatomy remains uncertain.',
        'model_qualified':False,'main_app_default_changed':False,'variants':[]}
    write(OUT/'experiment.json',manifest)
    url=json.loads((ROOT/'scratch/codeformer_spatial_research_server_desktop_v1.json').read_bytes())['url']
    start=time.monotonic()
    try:
        manifest['server_before']=transport.request(url+'/status')
        for name,data in [('original_input',contents),('whole_gaussian_1p2',path.read_bytes())]:
            assert time.monotonic()-start<600
            folder=OUT/name;folder.mkdir()
            response=transport.post(url,data)
            metadata=response['metadata'];archive=transport.blob(response['bundle'])
            (folder/'comparison.zip').write_bytes(archive);write(folder/'metadata.json',metadata)
            with zipfile.ZipFile(io.BytesIO(archive)) as z:
                assert z.testzip() is None and len(z.namelist())==12
                prepared=transport.rgb(z.read('resize-control.png'))
                with Image.open(io.BytesIO(z.read('observed-support.png'))) as im:support=np.asarray(im)>0
                (folder/'prepared-input.png').write_bytes(z.read('resize-control.png'))
                (folder/'observed-support.png').write_bytes(z.read('observed-support.png'))
                for arm in arms:
                    raw=np.load(io.BytesIO(z.read(arm+'-raw.npy')),allow_pickle=False)
                    pixels=transport.rgb(z.read(arm+'.png'))
                    assert raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all()
                    assert raw.min()>=0 and raw.max()<=1
                    assert np.array_equal(np.floor(raw*np.float32(255)).astype(np.uint8),pixels)
                    assert np.array_equal(pixels[~support],prepared[~support])
                    assert hashlib.sha256(raw.tobytes()).hexdigest()==metadata['outputs'][arm]['raw_rgb_sha256']
                    assert hashlib.sha256(pixels.tobytes()).hexdigest()==metadata['outputs'][arm]['PNG_RGB8_sha256']
                    assert hashlib.sha256(z.read(arm+'.png')).hexdigest()==metadata['outputs'][arm]['PNG_file_sha256']
                    assert transport.blob(response['outputs'][arm])==z.read(arm+'.png')
                    np.save(folder/(arm+'-raw.npy'),raw,allow_pickle=False)
                    (folder/(arm+'.png')).write_bytes(z.read(arm+'.png'))
            manifest['variants'].append({'id':name,'uploaded_bytes_sha256':hashlib.sha256(data).hexdigest(),
                'seconds':metadata['seconds'],'files':[pin(p) for p in sorted(folder.iterdir())]})
            write(OUT/'experiment.json',manifest)
            print(json.dumps({'complete_variant':name,'forwards':4,'seconds':time.monotonic()-start}),flush=True)
        manifest['server_after']=transport.request(url+'/status')
        manifest['server_success_delta']=manifest['server_after']['neural_forward_successes']-manifest['server_before']['neural_forward_successes']
        assert manifest['server_success_delta']>=8
        manifest['other_server_calls_observed']=manifest['server_success_delta']!=8
        manifest['complete']=True
    except BaseException as error:
        manifest['failure']={'type':type(error).__name__,'message':str(error),'no_retry':True}
        raise
    finally:
        manifest['seconds']=time.monotonic()-start
        write(OUT/'experiment.json',manifest)


if __name__=='__main__':main()

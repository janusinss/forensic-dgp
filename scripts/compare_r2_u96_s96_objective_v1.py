"""Complete a frozen three-model DEV comparison. No training, retries or VM calls."""
import argparse
import base64
from collections import Counter
import hashlib
import http.client
import io
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit
import zipfile

import numpy as np
from PIL import Image
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_filter_restoration_web_v1 import decode_crop, prepare_crop

OUT = ROOT / 'outputs/codeformer_r2_u96_s96_objective_v1'
COHORT = ROOT / 'outputs/codeformer_spatial_dev_review_v1/cohort_manifest_v1.json'
COHORT_SHA = 'e42afd973305e42422bcb273dc419ab608bbc1437ab332e62d6eb006e0fe9307'
OLD = ROOT / 'outputs/codeformer_spatial_dev_review_v1/run_v1_r2/comparison.json'
OLD_SHA = '6457a7293c3954e9c8430ae72d2fe32b1ccb43e446b5947e7d77cc203b6feb62'
ARMS = ('r2', 'spatial_u', 'spatial_s')
MODEL_SHA = {
    'r2': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896',
    'spatial_u': '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0',
    'spatial_s': '7e06314b4c467f3bbd9bc117a5ff73fd33ab2dd76809ef2c99daff315fd76d71',
}
LIMITS = {'wall_seconds': 2400, 'HTTP_seconds': 150, 'disk_reserve_bytes': 4*1024**3,
          'RAM_reserve_bytes': 384*1024**2, 'output_bytes': 2*1024**3,
          'new_forwards': 174, 'reused_outputs': 30}


def sha(data): return hashlib.sha256(data).hexdigest()


def pin(p):
    p = Path(p); data = p.read_bytes()
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)}


def checked(binding):
    p = ROOT / binding['path']
    assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
    data = p.read_bytes()
    assert len(data) == binding['bytes'] and sha(data) == binding['sha256'], binding['path']
    return data


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def blob(url): return base64.b64decode(url.split(',', 1)[1], validate=True)


def rgb(data):
    with Image.open(io.BytesIO(data)) as im:
        assert im.mode == 'RGB'
        return np.asarray(im).copy()


def request(url, body=None, headers=None, folder=None):
    target = urlsplit(url)
    assert target.scheme == 'http' and target.hostname == '127.0.0.1'
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=LIMITS['HTTP_seconds'])
    sink = None
    try:
        connection.request('POST' if body is not None else 'GET', target.path, body=body,
                           headers={**(headers or {}), 'Connection': 'keep-alive'})
        response = connection.getresponse()
        assert response.status == 200, 'HTTP '+str(response.status)+': '+response.read(2048).decode(errors='replace')
        length = int(response.getheader('content-length'))
        assert 0 < length <= 10*1024**2
        if folder:
            dump(folder/'response-headers.json', dict(response.headers))
            sink = (folder/'response.body').open('xb')
        parts = []; total = 0
        while total < length:
            part = response.read1(min(65536, length-total))
            if not part: break
            parts.append(part); total += len(part)
            if sink: sink.write(part); sink.flush()
        assert total == length, 'Incomplete response; do not retry'
        return json.loads(b''.join(parts))
    finally:
        if sink: sink.close()
        connection.close()


def preflight():
    assert pin(COHORT)['sha256'] == COHORT_SHA and pin(OLD)['sha256'] == OLD_SHA
    manifest = json.loads(COHORT.read_bytes()); prior = json.loads(OLD.read_bytes())
    assert len(manifest['cases']) == 68 and len(prior['cases']) == 10 and not prior['complete']
    assert manifest['FINAL_identity_or_pixel_review'] is False
    assert Counter(r['cohort'] for r in manifest['cases']) == {
        'filter_DEV': 36, 'native_ChokePoint': 24, 'native_QMUL-SurvFace': 8}
    assert [r['id'] for r in prior['cases']] == [r['id'] for r in manifest['cases'][:10]]
    assert all(r['split'] in ('DEV','native_DEV') for r in manifest['cases'])
    for row in manifest['cases']:
        for key in ('input', 'target', 'support', 'upload_source'):
            if row.get(key): checked(row[key])
        original = checked(row['upload_source'])
        canvas, support, _, geometry = prepare_crop(decode_crop(original))
        assert geometry == row['geometry'] and np.array_equal(canvas, rgb(checked(row['input'])))
        with Image.open(io.BytesIO(checked(row['support']))) as image:
            assert np.array_equal(support, np.asarray(image)>0)
    for saved in prior['cases']:
        for binding in saved['files']: checked(binding)
    assert 'torch' not in sys.modules
    return manifest, prior


def plan():
    manifest, prior = preflight()
    assert not OUT.exists(); OUT.mkdir()
    value = {'complete': True, 'status': 'FROZEN_BEFORE_NEW_INFERENCE',
        'requested_models': list(ARMS), 'cohort': pin(COHORT), 'prior_partial_run': pin(OLD),
        'worker': pin(Path(__file__)), 'input_cases': [r['id'] for r in manifest['cases']],
        'reused_case_ids': [r['id'] for r in prior['cases']], 'limits': LIMITS,
        'primary_paired_measure': 'Delivered PNG whole-observed-image MAE to the retained clean target; descriptive not identity/structure qualification.',
        'secondary_paired_measures': ['SSIM uniform7x7 RGB sample covariance', 'PSNR',
            'adjacent residual-gradient MAE', 'mean-centered residual MAE', 'signed/absolute luminance bias'],
        'required_strata': ['readable vs borderline', 'clear vs degraded profiles',
            'each reference photograph', 'appearance-flagged08025 separately', 'each native capture source separately'],
        'native_policy': 'No clean reference: no target PSNR/SSIM, no accuracy ranking from sharpness or no-reference scores. Review structure/appearance against input.',
        'reuse_policy': 'Exact saved ten-case three-arm outputs only; no automatic resume or changed historical records.',
        'input_only_refusal_controls': 'All40 excluded from neural work; selected borderline/QMUL cases retain explicit diagnostic-only status.',
        'evaluation_subset_not_full406_or478': True, 'independent_FINAL_review': False,
        'repeated_DEV_exposure': True, 'pretraining_or_subject_overlap_unknown': True,
        'preprocessing': 'Off; exact original prepared inputs, w1/AdaIN, no display changes.',
        'model_default_or_user_preference_change': False, 'training_optimizer_VM_calls': 0}
    dump(OUT/'comparison_plan_v1.json', value)
    print(json.dumps({'planned_cases':68, 'new_forwards':174, 'reused_outputs':30,
                      'plan':pin(OUT/'comparison_plan_v1.json')}))


def verify_payload(arm, archive, metadata, row, upload, canvas, support):
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        expected = {'original-upload.png','resize-control.png','observed-support.png',
                    arm+'.png',arm+'-raw.npy','processing.json'}
        assert set(z.namelist()) == expected and len(z.namelist()) == 6 and z.testzip() is None
        assert json.loads(z.read('processing.json')) == metadata
        assert np.array_equal(rgb(z.read('resize-control.png')), canvas)
        assert np.array_equal(rgb(z.read('original-upload.png')), decode_crop(upload))
        with Image.open(io.BytesIO(z.read('observed-support.png'))) as image:
            assert np.array_equal(np.asarray(image)>0, support)
        raw = np.load(io.BytesIO(z.read(arm+'-raw.npy')), allow_pickle=False)
        image = rgb(z.read(arm+'.png'))
        assert raw.shape == image.shape == (256,256,3) and raw.dtype == np.float32
        assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
        assert np.array_equal(np.floor(raw*np.float32(255)).astype(np.uint8), image)
        assert np.array_equal(raw[~support], (canvas.astype(np.float32)/np.float32(255))[~support])
        assert np.array_equal(image[~support], canvas[~support])
        declaration = metadata['outputs'][arm]; lineage = declaration['metadata']
        assert sha(raw.tobytes()) == declaration['raw_rgb_sha256']
        assert sha(image.tobytes()) == declaration['PNG_RGB8_sha256']
        assert sha(z.read(arm+'.png')) == declaration['PNG_file_sha256']
        assert lineage['weights_sha256'] == MODEL_SHA[arm]
        assert lineage['arm'] == arm and lineage['fidelity'] == 1 and lineage['adain'] is True
        assert metadata['geometry'] == row['geometry'] and metadata['uploaded_bytes_sha256'] == sha(upload)
        assert metadata['model_qualified'] is False and metadata['main_app_default_changed'] is False
        if arm != 'r2':
            assert lineage['base_tensor_sha256'] == 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
        return raw, z.read(arm+'.png')


def run(server_receipt, expected_plan):
    manifest, prior = preflight()
    assert pin(OUT/'comparison_plan_v1.json')['sha256'] == expected_plan
    assert not (OUT/'comparison.json').exists()
    server_path = ROOT/server_receipt
    server = json.loads(server_path.read_bytes()); url = server['url']
    assert url.startswith('http://127.0.0.1:') and url.endswith('/restoration-spatial-test')
    start = time.monotonic(); before = request(url+'/status')
    assert before['neural_forward_attempts'] == before['neural_forward_returns'] == before['neural_forward_successes'] == 0
    assert all(before['models'][a]['available'] for a in ARMS)
    result = {'complete': False, 'plan':pin(OUT/'comparison_plan_v1.json'), 'server_receipt':pin(server_path),
        'server_before':before,'cases':[],'new_forwards':0,'reused_outputs':0,'training_optimizer_VM_calls':0,
        'model_qualified':False,'main_app_default_changed':False,'no_automatic_retry':True}
    prior_by_id = {row['id']:row for row in prior['cases']}
    try:
        for index,row in enumerate(manifest['cases']):
            upload = checked(row['upload_source']); canvas = rgb(checked(row['input']))
            with Image.open(io.BytesIO(checked(row['support']))) as image: support=np.asarray(image)>0
            case = {'id':row['id'],'cohort':row['cohort'],'reference_id':row['reference_id'],
                    'profile':row['profile'],'input_review':row['input_review'],
                    'appearance_flag':row.get('appearance_flag',False),'outputs':{},'native_target_metrics':None}
            if row['id'] in prior_by_id:
                saved = prior_by_id[row['id']]
                source_folder = OLD.parent/row['id']
                for arm in ARMS:
                    folder = OLD.parent/'transport'/sha(upload)/arm
                    metadata = json.loads((folder/'metadata.json').read_bytes())
                    verify_payload(arm,(folder/'app-comparison.zip').read_bytes(),metadata,row,upload,canvas,support)
                    case['outputs'][arm] = {'origin':'prior_saved_exact_reuse',
                        'PNG':pin(source_folder/(arm+'.png')),'RAW':pin(source_folder/(arm+'-raw.npy')),
                        'ZIP':pin(folder/'app-comparison.zip'),'metadata':pin(folder/'metadata.json'),
                        'body':pin(folder/'response.body'),'headers':pin(folder/'response-headers.json')}
                    result['reused_outputs'] += 1
            else:
                for arm in ARMS:
                    assert time.monotonic()-start < LIMITS['wall_seconds']
                    assert shutil.disk_usage(ROOT).free >= LIMITS['disk_reserve_bytes']
                    assert psutil.virtual_memory().available >= LIMITS['RAM_reserve_bytes']
                    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < LIMITS['output_bytes']
                    folder=OUT/'cases'/row['id']/arm; folder.mkdir(parents=True)
                    boundary='FrozenThreeModelObjectiveBoundaryV1'
                    prefix=''.join('--'+boundary+'\r\nContent-Disposition: form-data; name="'+key+'"\r\n\r\n'+value+'\r\n'
                                   for key,value in [('selection',arm),('input_review','usable')]).encode()
                    body=prefix+('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="frozen.png"\r\nContent-Type: image/png\r\n\r\n').encode()+upload+('\r\n--'+boundary+'--\r\n').encode()
                    response=request(url+'/compare',body,{'Content-Type':'multipart/form-data; boundary='+boundary},folder)
                    assert set(response['outputs']) == {arm}
                    archive=blob(response['bundle']); metadata=response['metadata']
                    raw,image=verify_payload(arm,archive,metadata,row,upload,canvas,support)
                    assert blob(response['outputs'][arm]) == image
                    (folder/'app-comparison.zip').write_bytes(archive);dump(folder/'metadata.json',metadata)
                    np.save(folder/'raw.npy',raw,allow_pickle=False);(folder/'result.png').write_bytes(image)
                    case['outputs'][arm]={'origin':'new_local_frozen_inference','PNG':pin(folder/'result.png'),
                        'RAW':pin(folder/'raw.npy'),'ZIP':pin(folder/'app-comparison.zip'),
                        'metadata':pin(folder/'metadata.json'),'body':pin(folder/'response.body'),
                        'headers':pin(folder/'response-headers.json')}
                    result['new_forwards'] += 1
            result['cases'].append(case);result['seconds']=time.monotonic()-start
            dump(OUT/'comparison.json',result)
            print(json.dumps({'completed':index+1,'of':68,'id':row['id'],
                'new_forwards':result['new_forwards'],'reused':result['reused_outputs'],
                'seconds':round(result['seconds'],1)}),flush=True)
        result['server_after']=request(url+'/status')
        assert result['new_forwards']==174 and result['reused_outputs']==30
        assert all(result['server_after'][key]-before[key]==174 for key in
                   ('neural_forward_attempts','neural_forward_returns','neural_forward_successes'))
        assert result['server_after']['per_arm']['original']['attempts']==0
        result['complete']=True
    except BaseException as error:
        result['failure']={'type':type(error).__name__,'message':str(error),'no_retry':True,
                           'HTTP_failure_is_not_proof_inference_stopped':True}
        raise
    finally:
        result['seconds']=time.monotonic()-start;dump(OUT/'comparison.json',result)
        assert 'torch' not in sys.modules


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--server-receipt');parser.add_argument('--plan-sha');args=parser.parse_args()
    assert args.prepare != bool(args.server_receipt)
    plan() if args.prepare else run(args.server_receipt,args.plan_sha)

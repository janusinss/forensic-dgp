"""Fixed development comparison through the already loaded local inference server.

R1: four smaller single-arm HTTP requests; preserve each response before parsing.
No Torch import, training, optimizer, checkpoint mutation, VM call or app change.
The cohort must be independently frozen before launch. No automatic resume.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter

import numpy as np
from PIL import Image
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_codeformer_filter_comparison_v1 import residual_metrics

ARMS = ('original', 'r2', 'spatial_u', 'spatial_s')
COHORT = ROOT / 'outputs/codeformer_spatial_dev_review_v1/cohort_manifest_v1.json'
SERVER = ROOT / 'scratch/codeformer_spatial_research_server_desktop_v1.json'
OUT = ROOT / 'outputs/codeformer_spatial_dev_review_v1/run_v1_r1'
RUN_STARTED = None
EXPECTED_BUNDLE = '1fd8feba1fb8f7475b3022d8e906da59d7a979525d1b0a10e2e200603674b72b'
EXPECTED_APP = 'bafc0ec5ef912d2354418623d963e06eebea1289eeb831ab4dd5b6d7ad6bd208'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    p = Path(path)
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
            'sha256': digest(p.read_bytes())}


def verified(binding):
    relative = Path(binding['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = ROOT / relative
    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
    value = path.read_bytes()
    assert len(value) == binding['bytes'] and digest(value) == binding['sha256'], relative
    return value


def rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        assert image.mode == 'RGB'
        return np.asarray(image).copy()


def png(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format='PNG')
    return output.getvalue()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def request(url, data=None, headers=None, trace=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=180) as response:
        assert response.status == 200
        chunks = []; count = 0
        sink = (trace/'response.body').open('xb') if trace else None
        if trace: dump(trace/'response-headers.json', dict(response.headers))
        try:
            while True:
                chunk = response.read1(65536)
                if not chunk: break
                count += len(chunk)
                assert count <= 25 * 1024**2
                chunks.append(chunk)
                if sink: sink.write(chunk); sink.flush()
        finally:
            if sink: sink.close()
        return json.loads(b''.join(chunks))


def post_single(url, content, arm, trace):
    boundary = 'SpatialDevFrozenComparisonV1Boundary'
    parts = []
    for key, value in [('input_review', 'usable'), ('selection', arm)]:
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n').encode())
    parts.extend([(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="frozen-input.png"\r\nContent-Type: image/png\r\n\r\n').encode(),
                  content, f'\r\n--{boundary}--\r\n'.encode()])
    return request(url + '/compare', b''.join(parts), {'Content-Type': 'multipart/form-data; boundary=' + boundary}, trace=trace)


def post(url, content):
    transport = OUT/'transport'/digest(content)
    transport.mkdir(parents=True)
    responses = {}; raw_zips = {}; source_pins = []
    for arm in ARMS:
        assert RUN_STARTED is not None and time.monotonic()-RUN_STARTED < 3000
        assert shutil.disk_usage(ROOT).free >= 4 * 1024**3
        assert psutil.virtual_memory().available >= 256 * 1024**2
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < 2 * 1024**3
        trace = transport/arm; trace.mkdir()
        response = post_single(url, content, arm, trace)
        assert set(response['outputs']) == {arm}
        assert response['metadata']['selection'] == arm
        assert response['metadata']['uploaded_bytes_sha256'] == digest(content)
        archive = blob(response['bundle'])
        (trace/'app-comparison.zip').write_bytes(archive)
        dump(trace/'metadata.json', response['metadata'])
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 6
            assert set(z.namelist()) == {'original-upload.png','resize-control.png','observed-support.png',
                                        arm+'.png',arm+'-raw.npy','processing.json'}
            assert json.loads(z.read('processing.json')) == response['metadata']
            raw_zips[arm] = {name:z.read(name) for name in z.namelist()}
        responses[arm] = response
        source_pins.append({'arm':arm,'response_body':pin(trace/'response.body'),
                            'headers':pin(trace/'response-headers.json'),'ZIP':pin(trace/'app-comparison.zip'),
                            'metadata':pin(trace/'metadata.json')})
    metadata = dict(responses['original']['metadata'])
    assert all(r['metadata']['geometry'] == metadata['geometry'] for r in responses.values())
    metadata['selection'] = 'client_merge_of_four_single_arm_requests'
    metadata['transport'] = {'version':'single-arm-incremental-body-r1','source_responses':source_pins,
                             'each_original_server_ZIP_retained':True,'no_retry':True}
    metadata['seconds'] = sum(r['metadata']['seconds'] for r in responses.values())
    metadata['outputs'] = {arm:responses[arm]['metadata']['outputs'][arm] for arm in ARMS}
    output = io.BytesIO()
    common = ('original-upload.png','resize-control.png','observed-support.png')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in common:
            assert all(raw_zips[arm][name] == raw_zips['original'][name] for arm in ARMS)
            z.writestr(name,raw_zips['original'][name])
        for arm in ARMS:
            for name in (arm+'.png',arm+'-raw.npy'):z.writestr(name,raw_zips[arm][name])
        z.writestr('processing.json',json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return {'original':responses['original']['original'],
            'outputs':{arm:responses[arm]['outputs'][arm] for arm in ARMS},'metadata':metadata,
            'bundle':'data:application/zip;base64,'+base64.b64encode(output.getvalue()).decode('ascii')}


def blob(value):
    header, data = value.split(',', 1)
    assert header.startswith('data:') and header.endswith(';base64')
    return base64.b64decode(data, validate=True)


def main():
    global RUN_STARTED
    args = argparse.ArgumentParser()
    args.add_argument('--cohort-sha', required=True)
    args.add_argument('--verify-only', action='store_true')
    a = args.parse_args()
    assert digest(COHORT.read_bytes()) == a.cohort_sha
    cohort = json.loads(COHORT.read_bytes())
    assert cohort['complete'] is True and len(cohort['cases']) == 68
    assert cohort['FINAL_identity_or_pixel_review'] is False
    assert Counter((r['cohort'], r['split'], r['target'] is None) for r in cohort['cases']) == {
        ('filter_DEV', 'DEV', False): 36, ('native_QMUL-SurvFace', 'native_DEV', True): 8,
        ('native_ChokePoint', 'native_DEV', True): 24}
    assert len({r['id'] for r in cohort['cases']}) == 68
    assert all(__import__('re').fullmatch('[A-Za-z0-9_]+', r['id']) for r in cohort['cases'])
    assert sum(r.get('appearance_flag', False) for r in cohort['cases']) == 9
    assert digest((ROOT / 'research_spatial_restoration_v1.py').read_bytes()) == EXPECTED_APP
    assert digest((ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation/evaluation_bundle_v1.json').read_bytes()) == EXPECTED_BUNDLE
    for row in cohort['cases']:
        assert row['split'] in ('DEV', 'native_DEV') and row['expected_neural_forwards'] == 4
        assert row['inference_only_diagnostic_not_user_admission'] is (row['cohort'] != 'native_ChokePoint')
        for key in ('input', 'target', 'support', 'upload_source'):
            if row.get(key) is not None:
                verified(row[key])
    assert 'torch' not in sys.modules
    model_manifest = ROOT / 'outputs/codeformer_spatial_continuation_v1_models/model_exports_v1.json'
    assert digest(model_manifest.read_bytes()) == 'f0b2f5b25c43b7702867833e28ed6a28fe88e6e994b5597698820f3950ad190b'
    expected_models = json.loads(model_manifest.read_bytes())
    if a.verify_only:
        print(json.dumps({'complete': True, 'cases': 68, 'planned_forwards': 272,
                          'neural_training_VM_calls': 0, 'HTTP_calls': 0}))
        return
    assert not OUT.exists(), 'Fresh run required; never automatically resume/retry inference.'
    OUT.mkdir()
    url = json.loads(SERVER.read_bytes())['url']
    assert url.startswith('http://127.0.0.1:') and url.endswith('/restoration-spatial-test')
    start = time.monotonic()
    RUN_STARTED = start
    record = {'complete': False, 'cohort': pin(COHORT), 'worker': pin(Path(__file__)),
              'app_source': pin(ROOT / 'research_spatial_restoration_v1.py'),
              'server_receipt': pin(SERVER), 'url': url, 'cases': [], 'model_qualified': False,
              'actual_training_optimizer_VM_calls': 0, 'no_automatic_retry_or_resume': True,
              'limits': {'launch_checkpoint_seconds': 3000, 'individual_HTTP_timeout_seconds': 180,
                         'minimum_disk_bytes': 4 * 1024**3, 'minimum_available_RAM_bytes': 256 * 1024**2,
                         'maximum_output_bytes': 2 * 1024**3},
              'scope': 'Fixed DEV subset, not full historical406, FINAL or restoration acceptance.',
              'transport_version': 'single-arm-incremental-body-r1',
              'original_all_arm_run_retained': 'outputs/codeformer_spatial_dev_review_v1/run_v1/comparison.json',
              'four_completed_but_undelivered_prior_forwards_not_counted_as_own_results': True,
              'expected_successful_requests': 272, 'individual_original_API_ZIPs_preserved': True,
              'borderline_filter_inputs_are_explicit_diagnostic_overrides': True,
              'unknown_subject_or_pretraining_overlap_not_excluded': True}
    try:
        record['server_before'] = request(url + '/status')
        assert all(record['server_before']['models'][arm]['available'] for arm in ARMS)
        for index, row in enumerate(cohort['cases']):
            assert time.monotonic() - start < 3000, 'Stop before new input at elapsed limit.'
            assert shutil.disk_usage(ROOT).free >= 4 * 1024**3
            assert psutil.virtual_memory().available >= 256 * 1024**2
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < 2 * 1024**3
            folder = OUT / row['id']
            folder.mkdir()
            contents = verified(row.get('upload_source') or row['input'])
            response = post(url, contents)  # Never retry timeout/409 or ambiguous response.
            metadata = response['metadata']
            assert set(response['outputs']) == set(ARMS)
            assert metadata['uploaded_bytes_sha256'] == digest(contents)
            assert metadata['geometry'] == row['geometry'] and metadata['model_qualified'] is False
            for arm in ARMS:
                provenance = metadata['outputs'][arm]['metadata']
                assert provenance['arm'] == arm and provenance['fidelity'] == 1 and provenance['adain'] is True
                assert provenance['internal_resolution'] == 512 and provenance['output_resolution'] == 256
                if arm == 'original':
                    assert provenance['weights_sha256'] == '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
                elif arm == 'r2':
                    assert provenance['weights_sha256'] == 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
                    assert provenance['model_tensor_sha256'] == expected_models['base_tensor_sha256']
                else:
                    expected = expected_models['arms'][arm]
                    assert provenance['weights_sha256'] == expected['export']['sha256']
                    assert provenance['adapter_tensor_sha256'] == expected['adapter_tensor_sha256']
                    assert provenance['source_checkpoint_sha256'] == expected['source_checkpoint_sha256']
                    assert provenance['base_tensor_sha256'] == expected_models['base_tensor_sha256']
            archive_bytes = blob(response['bundle'])
            (folder / 'app-comparison.zip').write_bytes(archive_bytes)
            dump(folder / 'app-response-metadata.json', metadata)
            canvas = rgb(verified(row['input']))
            assert canvas.shape == (256, 256, 3)
            support = np.ones((256, 256), dtype=bool)
            if row['support']:
                with Image.open(io.BytesIO(verified(row['support']))) as im:
                    support = np.asarray(im) > 0
            scores = {}
            target = rgb(verified(row['target'])).astype(np.float32) / np.float32(255) if row['target'] else None
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
                assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 12
                assert np.array_equal(rgb(z.read('resize-control.png')), canvas)
                with Image.open(io.BytesIO(z.read('observed-support.png'))) as im:
                    assert np.array_equal(np.asarray(im) > 0, support), 'Original geometry/support differs.'
                processing = json.loads(z.read('processing.json'))
                assert processing == metadata
                if target is not None:
                    scores['resize_control'] = residual_metrics(canvas.astype(np.float32)/np.float32(255), target, support)
                rows = {}
                for arm in ARMS:
                    raw = np.load(io.BytesIO(z.read(arm + '-raw.npy')), allow_pickle=False)
                    assert raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
                    assert raw.min() >= 0 and raw.max() <= 1
                    image = rgb(z.read(arm + '.png'))
                    assert digest(raw.tobytes()) == metadata['outputs'][arm]['raw_rgb_sha256']
                    assert digest(image.tobytes()) == metadata['outputs'][arm]['PNG_RGB8_sha256']
                    assert digest(z.read(arm + '.png')) == metadata['outputs'][arm]['PNG_file_sha256']
                    assert np.array_equal(np.floor(raw * np.float32(255)).astype(np.uint8), image)
                    assert np.array_equal(image[~support], canvas[~support])
                    assert np.array_equal(raw[~support], (canvas.astype(np.float32)/np.float32(255))[~support])
                    assert blob(response['outputs'][arm]) == z.read(arm + '.png')
                    np.save(folder / (arm + '-raw.npy'), raw, allow_pickle=False)
                    (folder / (arm + '.png')).write_bytes(z.read(arm + '.png'))
                    rows[arm] = {'RAW': raw, 'PNG': image.astype(np.float32)/np.float32(255)}
                    scores[arm] = {'RAW': residual_metrics(raw, target, support) if target is not None else None,
                                   'PNG': residual_metrics(rows[arm]['PNG'], target, support) if target is not None else None}
                for arm in ARMS:
                    scores[arm]['diagnostic_distance_to_R2'] = {
                        kind: float(np.abs(rows[arm][kind][support].astype(np.float64)-rows['r2'][kind][support].astype(np.float64)).mean())
                        for kind in ('RAW', 'PNG')}
            result = {'id': row['id'], 'source': row['source'], 'profile': row['profile'],
                      'evidence': row['evidence'], 'split': row['split'], 'reference_id': row.get('reference_id'),
                      'input_review_retained': row.get('input_review', row.get('sourcebindings', {}).get('input_review')),
                      'appearance_flag': row.get('appearance_flag', False),
                      'inference_only_diagnostic_not_user_admission': row['inference_only_diagnostic_not_user_admission'],
                      'target_metrics': scores,
                      'native_PSNR_SSIM': None, 'API_seconds': metadata['seconds'],
                      'files': [pin(p) for p in sorted(folder.iterdir())], 'no_optimization': True}
            record['cases'].append(result)
            record['seconds'] = time.monotonic()-start
            dump(OUT / 'comparison.json', record)
            print(json.dumps({'completed': index+1, 'of': 68, 'id': row['id'],
                              'seconds': round(record['seconds'], 2), 'model_qualified': False}), flush=True)
        record['server_after'] = request(url + '/status')
        record['successful_forwards_in_own_responses'] = 4 * len(record['cases'])
        record['observed_server_counter_delta'] = {
            key: record['server_after'][key] - record['server_before'][key]
            for key in ('neural_forward_attempts', 'neural_forward_returns', 'neural_forward_successes', 'completed_comparisons')}
        assert all(record['observed_server_counter_delta'][key] >= 272
                   for key in ('neural_forward_attempts', 'neural_forward_returns', 'neural_forward_successes'))
        record['per_arm_server_success_deltas'] = {arm: record['server_after']['per_arm'][arm]['successes']-
            record['server_before']['per_arm'][arm]['successes'] for arm in ARMS}
        assert all(value >= 68 for value in record['per_arm_server_success_deltas'].values())
        record['other_server_activity_observed'] = record['observed_server_counter_delta'] != {
            'neural_forward_attempts': 272, 'neural_forward_returns': 272,
            'neural_forward_successes': 272, 'completed_comparisons': 272}
        record['complete'] = True
    except BaseException as error:
        record['failure'] = {'type': type(error).__name__, 'message': str(error),
                             'no_retry': True, 'HTTP_timeout_is_not_proof_server_forward_stopped': True}
        raise
    finally:
        record['seconds'] = time.monotonic()-start
        record['output_bytes'] = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        dump(OUT / 'comparison.json', record)
        assert 'torch' not in sys.modules


if __name__ == '__main__':
    main()

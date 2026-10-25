"""Freeze or run one finite local paired context study; no training or retries.

--freeze imports only image/web declaration helpers. --run is an HTTP client to
an explicitly supplied fresh dedicated loopback server; it never loads Torch.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
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

OUT = ROOT / 'outputs/codeformer_u96_context_study_v1'
CANDIDATE = OUT / 'candidate_manifest_v1.json'
REVIEW = OUT / 'root_input_review_v1.json'
CANDIDATE_SHA = 'faec2e7f00960ba76ccc4751ba0dfdcda14a2c6d3e6f83b2cb55e27bf7bdeede'
REVIEW_SHA = 'a1ee7e6b62920e408c16b3b7a962bbc75074bc57f906a022d684bb7fd6e0ba50'
PLAN = OUT / 'comparison_plan_v1.json'
ARMS = ('original', 'r2', 'spatial_u')
ROLES = ('clear', 'global_gaussian_2p4', 'local_gaussian_3p0')
MODEL_SHA = {
    'original': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7',
    'r2': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896',
    'spatial_u': '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0',
}
LIMITS = {'wall_seconds': 1800, 'HTTP_seconds': 150,
          'disk_reserve_bytes': 4*1024**3, 'RAM_reserve_bytes': 384*1024**2,
          'output_bytes': 1024**3, 'new_forwards': 96,
          'per_arm': {'original': 24, 'r2': 24, 'spatial_u': 48}}


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()


def safe_path(relative):
    assert type(relative) is str and '\\' not in relative and ':' not in relative
    parts = PurePosixPath(relative)
    assert not parts.is_absolute() and parts.as_posix() == relative
    assert all(p not in ('', '.', '..') for p in parts.parts)
    path = ROOT / relative
    assert path.resolve().is_relative_to(ROOT)
    for item in (path, *path.parents):
        if item == ROOT.parent: break
        if item.exists() or item.is_symlink():
            info = item.lstat()
            assert not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024
    return path


def pin(path):
    path = Path(path)
    safe_path(path.relative_to(ROOT).as_posix())
    assert path.is_file()
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest}


def checked(row):
    assert type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
    path = safe_path(row['path']); data = path.read_bytes()
    assert len(data) == row['bytes'] and sha(data) == row['sha256'], row['path']
    return data


def document(path): return json.loads(Path(path).read_bytes())


def dump(path, value):
    path = Path(path); temp = path.with_name(path.name+'.writing')
    assert not temp.exists()
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    temp.replace(path)


def blob(value):
    head, encoded = value.split(',', 1)
    assert head.startswith('data:') and head.endswith(';base64')
    return base64.b64decode(encoded, validate=True)


def rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        assert image.mode == 'RGB' and getattr(image, 'n_frames', 1) == 1
        return np.asarray(image).copy()


def image_array(data):
    with Image.open(io.BytesIO(data)) as image: return np.asarray(image).copy()


def write_exclusive(path, data):
    with Path(path).open('xb') as stream: stream.write(data)


def source_review():
    assert pin(CANDIDATE)['sha256'] == CANDIDATE_SHA
    assert pin(REVIEW)['sha256'] == REVIEW_SHA
    candidate, review = document(CANDIDATE), document(REVIEW)
    assert candidate['complete'] and review['complete']
    assert review['selection_done_before_any_study_outputs']
    assert review['automatic_detection_claim'] is False and review['assisted_known_synthetic_footprint']
    assert review['native_or_reserved_FINAL_inputs_used'] is False
    selected = review['selected_reference_ids']
    assert len(selected) == len(set(selected)) == 8
    reviews = {r['reference_id']: r for r in review['references']}
    cases = {(r['reference_id'], r['role']): r for r in candidate['cases']}
    assert len(cases) == 36 and candidate['reference_count'] == 12
    sources = {r['reference_id']: r for r in candidate['sources']}
    result = []
    for reference in selected:
        row = reviews[reference]
        assert row['selected'] and row['historical_split'] == 'DEV'
        assert row['optimization_permitted'] is False
        rectangle = row['protection_rectangle_inclusive']
        assert len(rectangle) == 4 and all(type(n) is int for n in rectangle)
        assert rectangle[0] <= 64 and rectangle[1] <= 64 and rectangle[2] >= 191 and rectangle[3] >= 191
        for role in ROLES:
            source = cases[(reference, role)]
            assert source['split'] == 'DEV' and source['optimization_permitted'] is False
            assert source['paired_clean_target'] and row['input_readability'][role] == 'usable'
            assert rgb(checked(source['input'])).shape == rgb(checked(source['target'])).shape == (256, 256, 3)
            result.append((source, row))
        for name in ('source1024', 'target256_original', 'target256_saved'):
            checked(sources[reference][name])
    return candidate, review, result


def freeze():
    assert not PLAN.exists() and not (OUT/'prepared').exists()
    started = time.monotonic()
    candidate, review, selected = source_review()
    import research_restoration_web as base
    import research_spatial_restoration_v1 as spatial
    import research_surroundings_restoration_v1 as surroundings
    export = spatial.manifest()
    r2path, _ = spatial.engine.candidate_file()
    assert base.sha(base.ORIGINAL, allow_benchmark_hardlinks=True) == MODEL_SHA['original']
    assert pin(r2path)['sha256'] == MODEL_SHA['r2']
    upath = safe_path(export['arms']['spatial_u']['export']['path'])
    assert pin(upath)['sha256'] == MODEL_SHA['spatial_u']
    model_bindings = {name: pin(path) for name, path in {
        'original': base.ORIGINAL, 'r2': r2path, 'spatial_u': upath}.items()}
    sources = [Path(__file__), ROOT/'research_spatial_restoration_v1.py',
               ROOT/'research_surroundings_restoration_v1.py', ROOT/'research_filter_restoration_web_v1.py',
               ROOT/'research_restoration_web.py', ROOT/'serve_codeformer_spatial_research_comparison_v1.py']
    sources += [ROOT/'third_party/codeformer'/name for name in base.ARCHITECTURE_PINS]
    receipts = [ROOT/'outputs/codeformer_role_preserving_training_v1_r2_model/model_export_v1.json',
                ROOT/'outputs/codeformer_spatial_continuation_v1_models/model_exports_v1.json']
    receipts += [safe_path(r['path']) for r in export['review_pins'].values()]
    case_rows = []
    for source, review_row in selected:
        folder = OUT/'prepared'/source['id']; folder.mkdir(parents=True)
        upload = checked(source['input'])
        protection = json.dumps({'rectangle': review_row['protection_rectangle_inclusive']}, separators=(',', ':'))
        preview, derived, original256, derived256 = surroundings.make_preview(upload, protection)
        metadata = preview['metadata']; archive = blob(preview['bundle'])
        write_exclusive(folder/'preprocessing.zip', archive)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 10
            assert json.loads(z.read('preprocessing.json')) == metadata
            assert z.read('uploaded-original.bin') == upload
            for name in z.namelist():
                assert '/' not in name and '\\' not in name and name not in ('.', '..')
                write_exclusive(folder/name, z.read(name))
        assert (folder/'derived-native.png').read_bytes() == derived
        original = decode_crop(upload); protected = image_array((folder/'protection-native.png').read_bytes()) > 0
        prepared = rgb(derived)
        assert protected[64:192, 64:192].all()
        assert np.array_equal(original[protected], prepared[protected])
        assert np.array_equal(original[64:192, 64:192], prepared[64:192, 64:192])
        alpha = np.load(folder/'blur-weight-float32.npy', allow_pickle=False)
        assert alpha.dtype == np.float32 and alpha.shape == (256, 256)
        assert np.isfinite(alpha).all() and np.all(alpha[protected] == 0)
        assert np.all((0 <= alpha) & (alpha <= 1))
        support = image_array((folder/'observed-support-256.png').read_bytes()) > 0
        interior = image_array((folder/'protected-interior-256.png').read_bytes()) > 0
        assert support.all() and np.array_equal(original256[interior], derived256[interior])
        assert metadata['protected_native_RGB_values_exact'] and metadata['protected_prepared_interior_RGB_exact']
        row = {k: source[k] for k in ('id', 'reference_id', 'role', 'split', 'input', 'target')}
        row.update({'appearance_flag': review_row['appearance_flag'], 'input_review': 'usable',
            'optimization_permitted': False, 'derived': pin(folder/'derived-native.png'),
            'protection': pin(folder/'protection-native.png'),
            'protected_interior': pin(folder/'protected-interior-256.png'),
            'alpha': pin(folder/'blur-weight-float32.npy'),
            'support': pin(folder/'observed-support-256.png'),
            'preprocessing_metadata': pin(folder/'preprocessing.json'),
            'preprocessing_bundle': pin(folder/'preprocessing.zip'),
            'protection_rectangle_inclusive': review_row['protection_rectangle_inclusive'],
            'geometry': metadata['original_geometry'], 'derived_geometry': metadata['derived_geometry'],
            'derived_input_is_forced_negative_control': source['role'] != 'local_gaussian_3p0',
            'preparation_method': 'assisted known-footprint reviewed rectangle, not automatic detection',
            'prepared_files': [pin(p) for p in sorted(folder.iterdir()) if p.is_file()]})
        case_rows.append(row)
    assert len(case_rows) == 24 and Counter(r['role'] for r in case_rows) == {r: 8 for r in ROLES}
    assert 'torch' not in sys.modules
    plan = {'complete': True, 'status': 'FROZEN_BEFORE_ANY_STUDY_MODEL_OUTPUTS',
        'created_UTC': utc(), 'candidate_manifest': pin(CANDIDATE), 'root_input_review': pin(REVIEW),
        'worker': pin(Path(__file__)), 'source_pins': [pin(p) for p in sources],
        'model_files': model_bindings, 'model_manifest_and_reviews': [pin(p) for p in receipts],
        'models_original_inputs': list(ARMS), 'models_prepared_inputs': ['spatial_u'],
        'cases': case_rows, 'selected_reference_ids': review['selected_reference_ids'],
        'limits': LIMITS, 'roles': list(ROLES), 'preprocessing_algorithm': dict(surroundings.ALGORITHM),
        'fixed_Pillow_nominal_radius6_is_not_matched_to_scipy_local_sigma3': True,
        'clear_global_preparation_is_forced_harm_control_not_recommended_automatic_action': True,
        'assisted_known_synthetic_footprint_not_automatic_detection': True,
        'paired_scope': 'Eight previously exposed DEV photographs; unknown subject/pretraining overlap; no FINAL.',
        'primary_preprocessing_comparison': 'spatial_u original versus prepared, same frozen model/settings; face ROI and context separate',
        'original_input_model_comparison': 'original versus r2 versus spatial_u, same exact uploaded input',
        'RAW_note': 'Clipped/resized delivery float32 before PNG floor quantization; not unclipped network activations',
        'input_protection_is_not_unchanged_generated_face': True,
        'neural_training_gradient_optimizer_VM_calls_during_freeze': 0,
        'automatic_resume_or_retry': False, 'model_qualified': False, 'main_app_default_changed': False,
        'seconds': time.monotonic()-started}
    dump(PLAN, plan)
    print(json.dumps({'complete': True, 'plan': pin(PLAN), 'cases': 24, 'planned_forwards': 96,
                      'neural_or_training_calls': 0}), flush=True)


def request(url, *, body=None, headers=None, folder=None, deadline):
    target = urlsplit(url)
    assert target.scheme == 'http' and target.hostname == '127.0.0.1'
    assert target.port and target.port != 61815 and not target.query and not target.fragment
    remaining = deadline-time.monotonic(); assert remaining > 0, 'Study wall stop reached before HTTP'
    connection = http.client.HTTPConnection(target.hostname, target.port,
        timeout=min(LIMITS['HTTP_seconds'], remaining))
    sink = None
    started = time.monotonic()
    try:
        connection.request('POST' if body is not None else 'GET', target.path, body=body,
                           headers={**(headers or {}), 'Connection': 'close'})
        response = connection.getresponse()
        lengths = response.headers.get_all('Content-Length')
        assert lengths and len(lengths) == 1
        length = int(lengths[0]); assert 0 < length <= 10*1024**2
        if folder:
            dump(folder/'response-headers.json', dict(response.headers))
            dump(folder/'transport.json', {'requested_UTC': utc(), 'url': url, 'HTTP_status': response.status,
                'declared_content_length': length, 'no_retry': True})
            sink = (folder/'response.body').open('xb')
        parts = []; count = 0
        while count < length:
            remaining = deadline-time.monotonic()
            assert remaining > 0, 'Study wall stop reached while reading HTTP'
            if connection.sock:
                connection.sock.settimeout(min(LIMITS['HTTP_seconds'], remaining))
            part = response.read1(min(65536, length-count))
            if not part: break
            parts.append(part); count += len(part)
            if sink: sink.write(part); sink.flush()
        assert count == length, 'Incomplete Content-Length response; no retry; server state unknown'
        raw = b''.join(parts)
        if folder:
            dump(folder/'transport_read_complete.json', {'complete': True, 'completed_UTC': utc(),
                'bytes_read': count, 'response_sha256': sha(raw), 'seconds': time.monotonic()-started})
        assert response.status == 200, 'HTTP '+str(response.status)+': '+raw[:2048].decode(errors='replace')
        return json.loads(raw)
    finally:
        if sink: sink.close()
        connection.close()


def verify_payload(arm, archive, metadata, upload, expected_geometry):
    canvas, support, _, geometry = prepare_crop(decode_crop(upload))
    assert geometry == expected_geometry
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names = {'original-upload.png', 'resize-control.png', 'observed-support.png',
                 arm+'.png', arm+'-raw.npy', 'processing.json'}
        assert set(z.namelist()) == names and len(z.namelist()) == 6 and z.testzip() is None
        assert json.loads(z.read('processing.json')) == metadata
        assert np.array_equal(rgb(z.read('original-upload.png')), decode_crop(upload))
        assert np.array_equal(rgb(z.read('resize-control.png')), canvas)
        assert np.array_equal(image_array(z.read('observed-support.png')) > 0, support)
        raw = np.load(io.BytesIO(z.read(arm+'-raw.npy')), allow_pickle=False)
        pixels = rgb(z.read(arm+'.png'))
        assert raw.shape == pixels.shape == (256, 256, 3) and raw.dtype == np.float32
        assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
        quantized = np.floor(raw*np.float32(255)).astype(np.uint8); quantized[~support] = canvas[~support]
        assert np.array_equal(quantized, pixels)
        assert np.array_equal(raw[~support], (canvas.astype(np.float32)/np.float32(255))[~support])
        assert np.array_equal(pixels[~support], canvas[~support])
        assert metadata['geometry'] == geometry and metadata['uploaded_bytes_sha256'] == sha(upload)
        assert metadata['selection'] == arm and set(metadata['outputs']) == {arm}
        assert metadata['model_qualified'] is False and metadata['main_app_default_changed'] is False
        assert metadata['device'] == 'cpu' and metadata['inference_only'] is True
        assert metadata['RAW_PNG_separate'] and metadata['no_completion_or_covering_removal']
        row = metadata['outputs'][arm]; lineage = row['metadata']
        assert sha(raw.tobytes()) == row['raw_rgb_sha256']
        assert sha(pixels.tobytes()) == row['PNG_RGB8_sha256']
        assert sha(z.read(arm+'.png')) == row['PNG_file_sha256']
        assert lineage['arm'] == arm and lineage['weights_sha256'] == MODEL_SHA[arm]
        assert lineage['fidelity'] == 1 and lineage['adain'] is True
        assert lineage['internal_resolution'] == 512 and lineage['output_resolution'] == 256
        assert lineage['alignment'] == lineage['display_processing'] == 'none'
        if arm == 'spatial_u':
            assert lineage['base_tensor_sha256'] == 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
            assert lineage['R2_weights_sha256'] == MODEL_SHA['r2'] and lineage['source_training_updates'] == 96
        else:
            expected = ('6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'
                        if arm == 'original' else 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f')
            assert lineage['model_tensor_sha256'] == expected
        return raw, z.read(arm+'.png')


def validate_plan(expected_sha):
    assert pin(PLAN)['sha256'] == expected_sha
    plan = document(PLAN)
    assert plan['complete'] and plan['limits'] == LIMITS and len(plan['cases']) == 24
    assert plan['worker'] == pin(Path(__file__))
    for row in [plan['candidate_manifest'], plan['root_input_review'], *plan['source_pins'],
                *plan['model_files'].values(), *plan['model_manifest_and_reviews']]: checked(row)
    for case in plan['cases']:
        for row in [case['input'], case['target'], *case['prepared_files']]: checked(row)
        assert case['optimization_permitted'] is False and case['input_review'] == 'usable'
    assert Counter(c['role'] for c in plan['cases']) == {role: 8 for role in ROLES}
    assert 'torch' not in sys.modules
    return plan


def resources(start):
    assert time.monotonic()-start < LIMITS['wall_seconds'], 'Study wall limit reached'
    assert shutil.disk_usage(ROOT).free >= LIMITS['disk_reserve_bytes'], 'Disk reserve reached'
    assert psutil.virtual_memory().available >= LIMITS['RAM_reserve_bytes'], 'RAM reserve reached'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < LIMITS['output_bytes'], 'Output cap reached'


def run(server_receipt, expected_plan):
    assert not (OUT/'comparison.json').exists() and not (OUT/'results').exists(), 'No automatic resume/retry'
    plan = validate_plan(expected_plan)
    receipt_path = safe_path(server_receipt)
    assert receipt_path.is_relative_to(ROOT/'scratch')
    server = document(receipt_path); url = server['url']
    assert url.endswith('/restoration-spatial-test') and server['host'] == '127.0.0.1'
    assert type(server['pid']) is int and server['pid'] > 0 and server['port'] != 61815
    start = time.monotonic(); deadline = start+LIMITS['wall_seconds']
    result = {'complete': False, 'started_UTC': utc(), 'plan': pin(PLAN), 'worker': pin(Path(__file__)),
        'server_receipt': pin(receipt_path), 'cases': [], 'new_forwards': 0,
        'submitted_HTTP_calls': 0, 'training_optimizer_VM_calls': 0, 'model_qualified': False,
        'main_app_default_changed': False, 'automatic_retry_or_resume': False}
    dump(OUT/'comparison.json', result)
    try:
        before = request(url+'/status', deadline=deadline)
        result['server_before'] = before
        assert all(before[k] == 0 for k in ('neural_forward_attempts', 'neural_forward_returns',
            'neural_forward_successes', 'completed_comparisons'))
        assert all(before['models'][arm]['available'] for arm in ARMS)
        assert all(all(n == 0 for n in counters.values()) for counters in before['per_arm'].values())
        for index, row in enumerate(plan['cases']):
            case = {k: row[k] for k in ('id', 'reference_id', 'role', 'appearance_flag')}
            case.update({'complete': False, 'outputs': {}}); result['cases'].append(case)
            for state, arm in [('original', a) for a in ARMS]+[('prepared', 'spatial_u')]:
                resources(start)
                upload = checked(row['input'] if state == 'original' else row['derived'])
                geometry = row['geometry'] if state == 'original' else row['derived_geometry']
                key = state+':'+arm
                folder = OUT/'results'/row['id']/(state+'_'+arm); folder.mkdir(parents=True)
                boundary = 'FrozenU96ContextStudyBoundaryV1'
                prefix = ''.join('--'+boundary+'\r\nContent-Disposition: form-data; name="'+k+'"\r\n\r\n'+v+'\r\n'
                                 for k,v in [('selection', arm), ('input_review', 'usable')]).encode()
                body = prefix+('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="frozen.png"\r\nContent-Type: image/png\r\n\r\n').encode()+upload+('\r\n--'+boundary+'--\r\n').encode()
                result['active_call'] = {'id': row['id'], 'key': key, 'submitted_UTC': utc(),
                    'folder': folder.relative_to(ROOT).as_posix(), 'uploaded_sha256': sha(upload)}
                result['submitted_HTTP_calls'] += 1; dump(OUT/'comparison.json', result)
                response = request(url+'/compare', body=body,
                    headers={'Content-Type': 'multipart/form-data; boundary='+boundary}, folder=folder, deadline=deadline)
                assert set(response['outputs']) == {arm}
                archive = blob(response['bundle']); metadata = response['metadata']
                raw, pixels = verify_payload(arm, archive, metadata, upload, geometry)
                assert blob(response['outputs'][arm]) == pixels
                write_exclusive(folder/'app-comparison.zip', archive); dump(folder/'metadata.json', metadata)
                with (folder/'raw.npy').open('xb') as stream: np.save(stream, raw, allow_pickle=False)
                write_exclusive(folder/'result.png', pixels)
                case['outputs'][key] = {'PNG': pin(folder/'result.png'), 'RAW': pin(folder/'raw.npy'),
                    'ZIP': pin(folder/'app-comparison.zip'), 'metadata': pin(folder/'metadata.json'),
                    'body': pin(folder/'response.body'), 'headers': pin(folder/'response-headers.json')}
                result['new_forwards'] += 1; result.pop('active_call')
                result['seconds'] = time.monotonic()-start; dump(OUT/'comparison.json', result)
            case['complete'] = True; dump(OUT/'comparison.json', result)
            print(json.dumps({'completed': index+1, 'of': 24, 'id': row['id'],
                'new_forwards': result['new_forwards'], 'seconds': round(time.monotonic()-start, 1)}), flush=True)
        resources(start)
        after = request(url+'/status', deadline=deadline); result['server_after'] = after
        assert result['new_forwards'] == result['submitted_HTTP_calls'] == 96
        assert all(after[k]-before[k] == 96 for k in ('neural_forward_attempts', 'neural_forward_returns',
            'neural_forward_successes', 'completed_comparisons'))
        for arm, count in LIMITS['per_arm'].items():
            assert all(after['per_arm'][arm][k]-before['per_arm'][arm][k] == count for k in ('attempts','returns','successes'))
        assert after['per_arm']['spatial_s'] == before['per_arm']['spatial_s']
        result['complete'] = True; result['completed_UTC'] = utc()
    except BaseException as error:
        result['failure'] = {'UTC': utc(), 'type': type(error).__name__, 'message': str(error),
            'no_retry': True, 'HTTP_failure_is_not_proof_inference_stopped': True,
            'do_not_restart_server_solely_from_client_timeout': True}
        raise
    finally:
        result['seconds'] = time.monotonic()-start; dump(OUT/'comparison.json', result)
        assert 'torch' not in sys.modules


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--freeze', action='store_true'); modes.add_argument('--run', action='store_true')
    parser.add_argument('--server-receipt'); parser.add_argument('--plan-sha')
    args = parser.parse_args()
    if args.freeze:
        assert not args.server_receipt and not args.plan_sha
        freeze()
    else:
        assert args.server_receipt and args.plan_sha
        run(args.server_receipt, args.plan_sha)


if __name__ == '__main__': main()

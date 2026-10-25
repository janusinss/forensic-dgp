"""Independent saved-payload audit only; no neural, HTTP, browser or VM calls."""
from pathlib import Path
import base64
import hashlib
import io
import json
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_wild5_surroundings_blur_v1_r1'
ARMS = ('original', 'r2', 'spatial_u', 'spatial_s')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)}


def checked(binding):
    rel = Path(binding['path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    path = ROOT / rel
    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
    data = path.read_bytes()
    assert len(data) == binding['bytes'] and sha(data) == binding['sha256']
    return data


def rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        assert image.mode == 'RGB'
        return np.array(image)


def blob(value):
    prefix, value = value.split(',', 1)
    assert prefix.startswith('data:') and prefix.endswith(';base64')
    return base64.b64decode(value, validate=True)


def zipped(data, names):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(names)
        assert set(z.namelist()) == set(names)
        return {name: z.read(name) for name in names}


def main():
    assert not (OUT / 'independent_saved_payload_audit_v1.json').exists()
    experiment_path = OUT / 'experiment.json'
    experiment = json.loads(experiment_path.read_bytes())
    assert experiment['complete'] is True and 'failure' not in experiment
    assert experiment['models'] == list(ARMS)
    assert experiment['new_planned_requests_and_forwards'] == 4
    assert experiment['model_qualified'] is False and experiment['main_app_default_changed'] is False
    assert experiment['training_optimizer_VM_calls'] == 0
    assert experiment['face_information_added_by_preprocessing'] is False
    prep = json.loads(checked(experiment['input_preparation']))
    prep_review = json.loads((OUT / 'independent_input_preparation_review_v1.json').read_bytes())
    assert prep_review['audit_pass'] is True
    assert prep_review['input_preparation'] == experiment['input_preparation']
    assert prep_review['worker'] == experiment['worker']
    checked(experiment['worker']); checked(experiment['transport_worker'])
    source_data = checked(prep['source'])
    with Image.open(io.BytesIO(source_data)) as im:
        assert im.size == (455, 523) and im.mode == 'RGBA'
        source = ImageOps.exif_transpose(im).convert('RGB')
    source_rgb = np.array(source)
    protected_image = Image.new('L', source.size, 0)
    poly = [(72,58),(386,58),(394,428),(364,483),(97,490),(55,439),(36,284),(36,201),(61,167)]
    assert prep['protected_face_polygon_xy'] == [list(x) for x in poly]
    ImageDraw.Draw(protected_image).polygon(poly, fill=255)
    protected = np.array(protected_image) > 0
    distance = distance_transform_edt(~protected)
    t = np.clip(distance / 16.0, 0, 1)
    alpha = (t*t*(3-2*t)).astype(np.float32)
    assert not alpha[protected].any()
    assert np.array_equal(alpha, np.load(io.BytesIO(checked(prep['alpha_float'])), allow_pickle=False))
    native_radius = 6.0 * 523 / 256
    assert prep['blur']['nominal_radius_at256'] == 6.0
    assert prep['blur']['native_radius_pixels'] == native_radius
    blurred = np.array(source.filter(ImageFilter.GaussianBlur(radius=native_radius)))
    mixed = np.rint(source_rgb.astype(np.float64)*(1-alpha[...,None].astype(np.float64)) +
                    blurred.astype(np.float64)*alpha[...,None].astype(np.float64)).clip(0,255).astype(np.uint8)
    derived_data = checked(prep['derived_input'])
    assert np.array_equal(mixed, rgb(derived_data))
    assert np.array_equal(mixed[protected], source_rgb[protected])
    assert (mixed[~protected] != source_rgb[~protected]).any()
    assert np.array_equal(np.array(protected_image), np.array(Image.open(io.BytesIO(checked(prep['protected_mask'])))))
    canvas = Image.new('RGB', (523,523), (128,128,128)); canvas.paste(Image.fromarray(mixed), (34,0))
    original_canvas = Image.new('RGB', (523,523), (128,128,128)); original_canvas.paste(source, (34,0))
    prepared = np.array(canvas.resize((256,256), Image.Resampling.BILINEAR))
    original_prepared = np.array(original_canvas.resize((256,256), Image.Resampling.BILINEAR))
    support_image = Image.new('L', (523,523), 0); support_image.paste(255, (34,0,489,523))
    support = np.array(support_image.resize((256,256), Image.Resampling.NEAREST)) > 0
    mask_canvas = Image.new('L', (523,523), 0); mask_canvas.paste(protected_image, (34,0))
    interior = np.array(mask_canvas.resize((256,256), Image.Resampling.BILINEAR)) == 255
    assert np.array_equal(prepared, rgb(checked(prep['prepared_input'])))
    assert np.array_equal(interior, np.array(Image.open(io.BytesIO(checked(prep['prepared_protected_interior'])))) > 0)
    assert np.array_equal(prepared[interior], original_prepared[interior])
    geometry = prep['prepared_geometry']
    assert geometry['native_size'] == [455,523] and geometry['native_center_offset'] == [34,0]
    assert geometry['input_rgb_sha256'] == sha(mixed.tobytes())
    assert geometry['canvas_rgb_sha256'] == sha(prepared.tobytes())
    assert geometry['observed_sha256'] == sha(support.tobytes())
    assert geometry['removal_mask_sha256'] == sha(np.zeros((256,256), dtype=bool).tobytes())
    baseline_binding = experiment['baseline_and_prior_whole_blur_reused']['experiment']
    baseline = json.loads(checked(baseline_binding))
    assert baseline['complete'] and baseline['source'] == prep['source']
    assert baseline['models'] == list(ARMS) and len(baseline['variants']) == 2
    baseline_files = 0
    for variant in baseline['variants']:
        for binding in variant['files']:
            checked(binding); baseline_files += 1
    prior_root = ROOT / experiment['baseline_and_prior_whole_blur_reused']['original_folder']
    original_metadata = json.loads((prior_root / 'metadata.json').read_bytes())
    prior_audit_path = prior_root.parent / 'independent_saved_payload_audit_v1.json'
    prior_audit = json.loads(prior_audit_path.read_bytes())
    assert prior_audit['audit_pass'] and prior_audit['experiment'] == baseline_binding
    model_manifest_path = ROOT / 'outputs/codeformer_spatial_continuation_v1_models/model_exports_v1.json'
    assert sha(model_manifest_path.read_bytes()) == 'f0b2f5b25c43b7702867833e28ed6a28fe88e6e994b5597698820f3950ad190b'
    models = json.loads(model_manifest_path.read_bytes())
    for item in models['arms'].values(): checked(item['export'])
    for binding in experiment['files']: checked(binding)
    folder = OUT / 'surroundings'
    metadata = json.loads((folder / 'metadata.json').read_bytes())
    assert metadata == experiment['response_metadata']
    assert metadata['geometry'] == geometry
    assert metadata['uploaded_bytes_sha256'] == sha(derived_data)
    assert metadata['model_qualified'] is False and metadata['main_app_default_changed'] is False
    common = ('original-upload.png','resize-control.png','observed-support.png')
    names = list(common) + [name for arm in ARMS for name in (arm+'.png',arm+'-raw.npy')] + ['processing.json']
    merged = zipped((folder / 'comparison.zip').read_bytes(), names)
    assert merged['original-upload.png'] == derived_data
    assert np.array_equal(rgb(merged['resize-control.png']), prepared)
    assert np.array_equal(np.array(Image.open(io.BytesIO(merged['observed-support.png']))) > 0, support)
    assert json.loads(merged['processing.json']) == metadata
    assert len(metadata['transport']['source_responses']) == 4
    assert metadata['transport']['no_retry'] is True
    rows = []
    for entry, arm in zip(metadata['transport']['source_responses'], ARMS):
        assert entry['arm'] == arm
        body_bytes = checked(entry['response_body'])
        body = json.loads(body_bytes)
        headers = {k.lower(): v for k,v in json.loads(checked(entry['headers'])).items()}
        assert int(headers['content-length']) == len(body_bytes)
        original_zip_bytes = checked(entry['ZIP'])
        assert blob(body['bundle']) == original_zip_bytes
        arm_meta = json.loads(checked(entry['metadata']))
        assert body['metadata'] == arm_meta and arm_meta['selection'] == arm
        assert arm_meta['geometry'] == geometry and arm_meta['uploaded_bytes_sha256'] == sha(derived_data)
        assert set(arm_meta['outputs']) == set(body['outputs']) == {arm}
        contents = zipped(original_zip_bytes, list(common)+[arm+'.png',arm+'-raw.npy','processing.json'])
        assert json.loads(contents['processing.json']) == arm_meta
        for name in list(common) + [arm+'.png',arm+'-raw.npy']:
            assert contents[name] == merged[name]
        assert blob(body['outputs'][arm]) == contents[arm+'.png']
        provenance = arm_meta['outputs'][arm]['metadata']
        assert provenance == original_metadata['outputs'][arm]['metadata']
        assert arm_meta['outputs'][arm] == metadata['outputs'][arm]
        assert provenance['fidelity'] == 1 and provenance['adain'] is True
        assert provenance['internal_resolution'] == 512 and provenance['output_resolution'] == 256
        if arm == 'original': assert provenance['weights_sha256'] == '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
        elif arm == 'r2':
            assert provenance['weights_sha256'] == 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
            assert provenance['model_tensor_sha256'] == models['base_tensor_sha256']
        else:
            assert provenance['weights_sha256'] == models['arms'][arm]['export']['sha256']
            assert provenance['adapter_tensor_sha256'] == models['arms'][arm]['adapter_tensor_sha256']
            assert provenance['base_tensor_sha256'] == models['base_tensor_sha256']
        raw = np.load(io.BytesIO(contents[arm+'-raw.npy']), allow_pickle=False)
        pixels = rgb(contents[arm+'.png'])
        assert raw.dtype == np.float32 and raw.shape == (256,256,3) and np.isfinite(raw).all()
        assert 0 <= raw.min() <= raw.max() <= 1
        assert np.array_equal(raw, np.load(folder/(arm+'-raw.npy'), allow_pickle=False))
        assert (folder/(arm+'.png')).read_bytes() == contents[arm+'.png']
        assert np.array_equal(np.floor(raw*np.float32(255)).astype(np.uint8), pixels)
        assert np.array_equal(raw[~support], (prepared.astype(np.float32)/np.float32(255))[~support])
        assert np.array_equal(pixels[~support], prepared[~support])
        output_meta = metadata['outputs'][arm]
        assert sha(raw.tobytes()) == output_meta['raw_rgb_sha256']
        assert sha(pixels.tobytes()) == output_meta['PNG_RGB8_sha256']
        assert sha(contents[arm+'.png']) == output_meta['PNG_file_sha256']
        original_output = rgb((prior_root/(arm+'.png')).read_bytes())
        differences = pixels.astype(np.int16) - original_output.astype(np.int16)
        rows.append({'arm':arm,'PNG':pin(folder/(arm+'.png')),'RAW':pin(folder/(arm+'-raw.npy')),
                     'source_response':entry,'same_provenance_as_reused_original':True,
                     'protected_interior_output_RGB8_mean_absolute_change_vs_original_input_output':
                     float(np.abs(differences[interior]).astype(np.float64).mean()),
                     'output_change_is_not_accuracy_or_quality_gain':True,
                     'RAW_PNG_floor_and_exact_offsupport_preservation_pass':True})
    delta = {key:experiment['server_after'][key]-experiment['server_before'][key]
             for key in ('neural_forward_attempts','neural_forward_returns','neural_forward_successes','completed_comparisons')}
    assert all(value >= 4 for value in delta.values())
    assert delta['neural_forward_successes'] == experiment['server_success_delta']
    assert experiment['other_server_activity_observed'] is (delta['neural_forward_successes'] != 4)
    assert 'torch' not in sys.modules
    report = {'format':'wild5-surroundings-independent-saved-payload-audit-v1','complete':True,'audit_pass':True,
              'reviewer':'/root/wild5_surroundings_peer','auditor':pin(Path(__file__)),
              'experiment':pin(experiment_path),'input_preparation':experiment['input_preparation'],
              'preparation_review':pin(OUT/'independent_input_preparation_review_v1.json'),
              'source':prep['source'],'protected_native_RGB_pixels_verified_exact':int(protected.sum()),
              'protected_conservative_prepared256_RGB_pixels_verified_exact':int(interior.sum()),
              'Gaussian_and_outside_only_smoothstep_rederived_exact':True,'nominal_radius_at256':6,
              'native_radius_pixels':native_radius,'matching_to_face_blur_is_visual_approximation_only':True,
              'new_model_outputs':4,'new_original_server_single_arm_ZIPs':4,'client_merged_ZIP_members':12,
              'reused_original_and_whole_blur_model_outputs':8,'reused_physical_file_pins_verified':baseline_files,
              'reused_baseline':baseline_binding,'reused_baseline_independent_audit':pin(prior_audit_path),
              'all4_original_HTTP_ContentLength_sourceZIP_upload_provenance_and_RAW_PNG_bindings_pass':True,
              'server_counter_deltas':delta,'other_server_activity_observed':experiment['other_server_activity_observed'],
              'outputs':rows,'model_manifest':pin(model_manifest_path),'audit_neural_gradient_optimizer_HTTP_browser_VM_calls':0,
              'model_qualified':False,'no_aligned_clean_original_or_identity_accuracy_claim':True,
              'scope':'Saved-payload and deterministic-input integrity, not output usefulness or recovered facial truth.'}
    path = OUT/'independent_saved_payload_audit_v1.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'audit_pass':True,'receipt':pin(path),'new_outputs':4,
                      'reused_outputs':8,'server_counter_deltas':delta,'neural_HTTP_VM_calls':0}))


if __name__ == '__main__': main()

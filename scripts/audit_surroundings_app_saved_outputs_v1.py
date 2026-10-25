"""Independent root-browser saved-payload verification; no HTTP or model calls."""
from pathlib import Path
import base64
import hashlib
import io
import json
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT/'scratch/codeformer-context-v1'
OUT = ROOT/'outputs/codeformer_surroundings_app_v1'


def sha(data): return hashlib.sha256(data).hexdigest()


def pin(path):
    data = path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':sha(data)}


def blob(value):
    head,payload = value.split(',',1)
    assert head.startswith('data:') and head.endswith(';base64')
    return base64.b64decode(payload,validate=True)


def read_zip(data, names):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(names)
        assert set(z.namelist()) == set(names)
        return {name:z.read(name) for name in names}


def rgb(data):
    with Image.open(io.BytesIO(data)) as im:
        assert im.mode == 'RGB'
        return np.array(im)


def json_bytes(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def main():
    destination = OUT/'independent_saved_app_output_audit_v1.json'; assert not destination.exists()
    browser_path = SCRATCH/'browser-audit.json'; browser = json.loads(browser_path.read_bytes())
    response_path = SCRATCH/'comparison-response.json'; response = json.loads(response_path.read_bytes())
    preview_path = SCRATCH/'preview-response.json'; preview = json.loads(preview_path.read_bytes())
    refusals_path = SCRATCH/'refusals.json'; refusals = json.loads(refusals_path.read_bytes())
    assert browser['complete'] and browser['errors'] == [] and browser['overflowWidths'] == []
    assert {x['width'] for x in browser['checks']} == {375,768,1280}
    assert all(not x['overflow'] and x['unconfirmed_disabled'] for x in browser['checks'])
    assert len(browser['downloads']) == 5
    assert {x['case'] for x in refusals['refused']} == {'unconfirmed','stale','insufficient','out-of-bounds','empty'}
    assert all(x['status'] == 422 for x in refusals['refused'])
    assert refusals['status']['neural_forward_attempts'] == 0
    assert browser['model_quality_qualification'] is False and browser['main_app_default_changed'] is False
    counters = ('neural_forward_attempts','neural_forward_returns','neural_forward_successes','completed_comparisons')
    delta = {key:browser['after'][key]-browser['before'][key] for key in counters}
    assert all(v == 2 for v in delta.values()) and browser['model_forward_delta'] == 2
    for arm in ('original','spatial_u','spatial_s'):
        assert browser['after']['per_arm'][arm] == browser['before']['per_arm'][arm]
    assert all(browser['after']['per_arm']['r2'][key]-browser['before']['per_arm']['r2'][key] == 2
               for key in ('attempts','returns','successes'))
    metadata = response['metadata']; preprocessing = metadata['preprocessing']
    assert metadata['selection'] == 'r2' and metadata['confirmed_protection'] is True
    assert metadata['model_qualified'] is False and metadata['main_app_default_changed'] is False
    assert metadata['training_optimizer_VM_calls'] == 0 and metadata['no_automatic_retry'] is True
    assert metadata['server_before']['neural_forward_attempts'] == 0
    assert metadata['server_after']['neural_forward_successes'] == 2
    assert preview['preview_id'] == metadata['preview_id'] == preprocessing['preview_id']
    assert preprocessing == preview['metadata']
    assert preprocessing['protection'] == {'preset':'reviewed-wild5'}
    assert preprocessing['manual_assisted_protection_not_automatic_blur_detection'] is True
    assert preprocessing['preprocessing_neural_calls'] == 0 and preprocessing['server_preview_cache'] is False
    assert preprocessing['protected_native_RGB_values_exact'] and preprocessing['protected_prepared_interior_RGB_exact']
    for name,binding in preprocessing['source_fingerprints'].items():
        data = (ROOT/name).read_bytes()
        assert len(data) == binding['bytes'] and sha(data) == binding['sha256']
    identity = {'source_sha256':preprocessing['source_sha256'],'protection':preprocessing['protection'],
                'algorithm':preprocessing['algorithm'],'source_fingerprints':preprocessing['source_fingerprints']}
    combined_bytes = blob(response['bundle'])
    assert combined_bytes == (SCRATCH/'protected-surroundings-comparison.zip').read_bytes()
    combined = read_zip(combined_bytes,['original-comparison.zip','prepared-comparison.zip','preprocessing.zip','comparison.json'])
    assert json.loads(combined['comparison.json']) == metadata
    for name,binding in metadata['bundles'].items():
        assert len(combined[name]) == binding['bytes'] and sha(combined[name]) == binding['sha256']
    prep_names = ['uploaded-original.bin','original-decoded-rgb.png','protection-native.png','blur-weight-float32.npy',
                  'derived-native.png','prepared-original-256.png','prepared-derived-256.png','observed-support-256.png',
                  'protected-interior-256.png','preprocessing.json']
    prep = read_zip(combined['preprocessing.zip'],prep_names)
    preview_bundle = blob(preview['bundle']); assert preview_bundle == (SCRATCH/'surroundings-preparation.zip').read_bytes()
    preview_prep = read_zip(preview_bundle,prep_names)
    assert preview_prep == prep, 'ZIP timestamps may differ; all payload bytes must match.'
    server_preprocessing = json.loads(prep['preprocessing.json'])
    assert server_preprocessing == preprocessing
    # Browser JSON.stringify preserves values but normalizes 6.0 to 6. Exact
    # canonical-ID arithmetic uses the original server JSON retained in its ZIP.
    identity['algorithm'] = server_preprocessing['algorithm']
    assert sha(json_bytes(identity)) == preview['preview_id']
    for name,binding in preprocessing['bundle_payloads'].items():
        assert len(prep[name]) == binding['bytes'] and sha(prep[name]) == binding['sha256']
    source = (ROOT/'test_images/wild5.png').read_bytes()
    fixture = ROOT/'outputs/codeformer_wild5_precise_surroundings_blur_v1'
    assert prep['uploaded-original.bin'] == source and sha(source) == preprocessing['source_sha256']
    assert prep['derived-native.png'] == (fixture/'wild5_surroundings_blurred.png').read_bytes()
    assert prep['protection-native.png'] == (fixture/'protected_face_region.png').read_bytes()
    assert prep['blur-weight-float32.npy'] == (fixture/'surroundings_blur_weight-float32.npy').read_bytes()
    assert prep['derived-native.png'] == blob(preview['prepared_input']) == (SCRATCH/'protected-surroundings-input.png').read_bytes()
    assert prep['original-decoded-rgb.png'] == blob(preview['original'])
    source_rgb = rgb(prep['original-decoded-rgb.png']); derived_rgb = rgb(prep['derived-native.png'])
    mask = np.array(Image.open(io.BytesIO(prep['protection-native.png']))) > 0
    alpha = np.load(io.BytesIO(prep['blur-weight-float32.npy']),allow_pickle=False)
    assert int(mask.sum()) == 159548 and not alpha[mask].any()
    assert np.array_equal(source_rgb[mask],derived_rgb[mask])
    patch = np.zeros(mask.shape,dtype=bool); patch[62:479,75:384] = True
    assert int(patch.sum()) == 128853 and mask[patch].all() and np.array_equal(source_rgb[patch],derived_rgb[patch])
    interior = np.array(Image.open(io.BytesIO(prep['protected-interior-256.png']))) > 0
    original256 = rgb(prep['prepared-original-256.png']); derived256 = rgb(prep['prepared-derived-256.png'])
    assert int(interior.sum()) == 37650 and np.array_equal(original256[interior],derived256[interior])
    support = np.array(Image.open(io.BytesIO(prep['observed-support-256.png']))) > 0
    assert np.array_equal(rgb(blob(response['original_input'])),original256)
    assert np.array_equal(rgb(blob(response['prepared_input'])),derived256)
    outputs = []
    cases = [('original',source,original256,'original_result','original-context-restoration.png',
              ROOT/'outputs/codeformer_wild5_whole_blur_v1_r1/original_input'),
             ('prepared',prep['derived-native.png'],derived256,'prepared_result','protected-context-restoration.png',
              fixture/'surroundings')]
    for prefix,uploaded,canvas,key,download,previous in cases:
        contents = read_zip(combined[prefix+'-comparison.zip'],
            ['original-upload.png','resize-control.png','observed-support.png','r2.png','r2-raw.npy','processing.json'])
        upstream = json.loads(contents['processing.json']); assert upstream == metadata[prefix+'_result']
        assert upstream['uploaded_bytes_sha256'] == sha(uploaded) and upstream['selection'] == 'r2'
        assert np.array_equal(rgb(contents['resize-control.png']),canvas)
        assert np.array_equal(np.array(Image.open(io.BytesIO(contents['observed-support.png']))) > 0,support)
        assert upstream['geometry'] == preprocessing[prefix+'_geometry' if prefix == 'original' else 'derived_geometry']
        assert sha(canvas.tobytes()) == upstream['geometry']['canvas_rgb_sha256']
        assert contents['r2.png'] == blob(response[key]) == (SCRATCH/download).read_bytes()
        raw = np.load(io.BytesIO(contents['r2-raw.npy']),allow_pickle=False); pixels = rgb(contents['r2.png'])
        assert raw.dtype == np.float32 and raw.shape == (256,256,3) and np.isfinite(raw).all()
        assert 0 <= raw.min() <= raw.max() <= 1
        assert np.array_equal(np.floor(raw*np.float32(255)).astype(np.uint8),pixels)
        assert np.array_equal(raw[~support],(canvas.astype(np.float32)/np.float32(255))[~support])
        assert np.array_equal(pixels[~support],canvas[~support])
        output_meta = upstream['outputs']['r2']; provenance = output_meta['metadata']
        assert provenance['weights_sha256'] == 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
        assert provenance['model_tensor_sha256'] == 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
        assert provenance['fidelity'] == 1 and provenance['adain'] is True
        assert provenance['internal_resolution'] == 512 and provenance['output_resolution'] == 256
        assert provenance['model_qualified'] is False and provenance['main_app_default_changed'] is False
        assert output_meta['raw_rgb_sha256'] == sha(raw.tobytes())
        assert output_meta['PNG_RGB8_sha256'] == sha(pixels.tobytes())
        assert output_meta['PNG_file_sha256'] == sha(contents['r2.png'])
        assert contents['r2.png'] == (previous/'r2.png').read_bytes()
        assert np.array_equal(raw,np.load(previous/'r2-raw.npy',allow_pickle=False))
        outputs.append({'input':prefix,'PNG_sha256':sha(contents['r2.png']),'RAW_array_sha256':sha(raw.tobytes()),
                        'exact_vs_prior_accepted_stress_fixture':True,'RAW_PNG_and_offsupport_preservation_pass':True,
                        'provenance':provenance})
    backend_review = json.loads((OUT/'backend_review_v1.json').read_bytes())
    assert pin(ROOT/'research_surroundings_restoration_v1.py') == backend_review['source']
    assert 'torch' not in sys.modules
    report = {'format':'surroundings-app-independent-saved-output-audit-v1','complete':True,'audit_pass':True,
        'reviewer':'/root/wild5_surroundings_peer','auditor':pin(Path(__file__)),
        'browser_audit':pin(browser_path),'comparison_response':pin(response_path),'preview_response':pin(preview_path),
        'five_pre_neural422_refusals':pin(refusals_path),'backend_source_unchanged_since_pure_fixture_review':True,
        'source_review':'UI and backend contract reviewed read-only: isolated opt-in router, native decoded editor, preview/confirmation invalidated on image or mask edit, exact server stale-ID check, manual example preset, no hidden retry; startup launcher only mounts new route and records it.',
        'sources':{name:pin(ROOT/name) for name in ('research_surroundings_restoration_v1.py',
            'serve_codeformer_spatial_research_comparison_v1.py','templates/research_surroundings_restoration_v1.html',
            'static/research_surroundings_restoration_v1.js','static/research_surroundings_restoration_v1.css')},
        'preview10memberZIP_and_all_payload_hashes_pass':True,'combined4memberZIP_and_two_exact_upstream6memberZIPs_pass':True,
        'five_actual_downloads':[{**row,'physical':pin(ROOT/row['path'])} for row in browser['downloads']],
        'actual_uploaded_original_bytes_retained':True,'preview_and_comparison_preprocessing_payloads_identical':True,
        'preview_id_exactly_recomputed_from_original_server_ZIP_JSON':True,
        'browser_JSON_numeric_type_normalization':'JSON.stringify changes 6.0 to 6; server ZIP retains canonical float types. Values compare exactly; no quality tolerance or data change.',
        'protected_native_RGB_pixels_exact':159548,'full_manual_existing_patch_RGB_pixels_exact':128853,
        'prepared_conservative_RGB_pixels_exact':37650,'wild5_preset_matches_reviewed_input_and_prior_outputs_exact':True,
        'actual_R2_server_counter_deltas':delta,'other_arm_counter_delta_zero':True,
        'root_browser_reported_layouts':[375,768,1280],'root_browser_reported_overflow_or_console_errors':False,
        'outputs':outputs,'reviewer_neural_gradient_optimizer_HTTP_browser_VM_calls':0,
        'limitations':'One opt-in assisted wild5/R2 functional flow; not automatic boundary detection, general usefulness, identity accuracy, native CCTV or full thesis/main-app qualification. Browser execution and visual inspection belong to root.',
        'model_qualified':False,'main_app_default_changed':False,'overall_goal_complete':False}
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'audit_pass':True,'receipt':pin(destination),'actual_root_model_forwards':2,
                      'downloads':5,'reviewer_neural_HTTP_browser_VM_calls':0}))


if __name__ == '__main__': main()

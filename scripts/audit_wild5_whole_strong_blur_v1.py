"""Independent saved whole-image stress payload audit; no neural/HTTP/VM calls."""
from pathlib import Path
import io
import json
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from audit_wild5_surroundings_blur_v1 import ROOT, ARMS, sha, pin, checked, rgb, blob, zipped

OUT = ROOT/'outputs/codeformer_wild5_whole_strong_blur_v1'


def main():
    path = OUT/'experiment.json'; experiment = json.loads(path.read_bytes())
    assert experiment['complete'] and 'failure' not in experiment
    assert experiment['models'] == list(ARMS) and experiment['new_planned_requests_and_forwards'] == 4
    assert experiment['training_optimizer_VM_calls'] == 0
    assert experiment['model_qualified'] is False and experiment['main_app_default_changed'] is False
    source = checked(experiment['source'])
    with Image.open(io.BytesIO(source)) as im:
        assert im.format == 'PNG' and im.mode == 'RGBA' and im.size == (455,523)
        source_image = ImageOps.exif_transpose(im).convert('RGB')
    assert experiment['blur']['native_radius_pixels'] == 12.2578125
    assert experiment['blur']['nominal_radius_at256'] == 6
    assert experiment['blur']['parameter_search'] is False
    assert experiment['blur']['equal_final_face_and_background_blur_claimed'] is False
    whole = np.array(source_image.filter(ImageFilter.GaussianBlur(radius=12.2578125)))
    data = checked(experiment['derived_input']); assert np.array_equal(rgb(data), whole)
    prior_root = ROOT/'outputs/codeformer_wild5_surroundings_blur_v1_r1'
    prep = json.loads((prior_root/'input_preparation.json').read_bytes())
    assert prep['source'] == experiment['source']
    assert prep['blur']['native_radius_pixels'] == experiment['blur']['native_radius_pixels']
    mask = np.array(Image.open(io.BytesIO(checked(prep['protected_mask'])))) > 0
    face_difference = whole.astype(np.int16)-np.array(source_image).astype(np.int16)
    assert np.count_nonzero(np.any(face_difference[mask] != 0, axis=1)) > 0
    before = Image.new('RGB',(523,523),(128,128,128)); before.paste(Image.fromarray(whole),(34,0))
    canvas = np.array(before.resize((256,256), Image.Resampling.BILINEAR))
    sup = Image.new('L',(523,523),0); sup.paste(255,(34,0,489,523))
    support = np.array(sup.resize((256,256),Image.Resampling.NEAREST)) > 0
    reused = experiment['baseline_reuse']; assert reused['reused_outputs'] == 8 and reused['new_baseline_calls'] == 0
    original_exp = json.loads(checked(reused['original_experiment']))
    surroundings_exp = json.loads(checked(reused['surroundings_experiment']))
    assert original_exp['complete'] and surroundings_exp['complete']
    original_variant = next(x for x in original_exp['variants'] if x['id'] == 'original_input')
    reused_files = 0
    for binding in original_variant['files'] + surroundings_exp['files']:
        checked(binding); reused_files += 1
    surroundings_audit_path = prior_root/'independent_saved_payload_audit_v1.json'
    surroundings_audit = json.loads(surroundings_audit_path.read_bytes())
    assert surroundings_audit['audit_pass'] and surroundings_audit['experiment'] == reused['surroundings_experiment']
    original_root = ROOT/reused['original_folder']
    original_metadata = json.loads((original_root/'metadata.json').read_bytes())
    for binding in experiment['files']: checked(binding)
    for key in ('worker','transport_worker'): checked(experiment[key])
    folder = OUT/'whole'; metadata = json.loads((folder/'metadata.json').read_bytes())
    assert metadata == experiment['response_metadata'] and metadata['uploaded_bytes_sha256'] == sha(data)
    geometry = metadata['geometry']; assert geometry['native_size'] == [455,523]
    assert geometry['native_center_offset'] == [34,0] and geometry['input_rgb_sha256'] == sha(whole.tobytes())
    assert geometry['canvas_rgb_sha256'] == sha(canvas.tobytes()) and geometry['observed_sha256'] == sha(support.tobytes())
    assert geometry['removal_mask_sha256'] == sha(np.zeros((256,256),dtype=bool).tobytes())
    common = ['original-upload.png','resize-control.png','observed-support.png']
    merged = zipped((folder/'comparison.zip').read_bytes(),common+[n for arm in ARMS for n in (arm+'.png',arm+'-raw.npy')]+['processing.json'])
    assert merged['original-upload.png'] == data and np.array_equal(rgb(merged['resize-control.png']),canvas)
    assert np.array_equal(np.array(Image.open(io.BytesIO(merged['observed-support.png']))) > 0,support)
    assert json.loads(merged['processing.json']) == metadata
    assert metadata['transport']['no_retry'] and len(metadata['transport']['source_responses']) == 4
    model_path = ROOT/'outputs/codeformer_spatial_continuation_v1_models/model_exports_v1.json'
    assert sha(model_path.read_bytes()) == 'f0b2f5b25c43b7702867833e28ed6a28fe88e6e994b5597698820f3950ad190b'
    models = json.loads(model_path.read_bytes())
    for item in models['arms'].values(): checked(item['export'])
    rows = []
    for entry,arm in zip(metadata['transport']['source_responses'],ARMS):
        assert entry['arm'] == arm
        body_bytes = checked(entry['response_body']); body = json.loads(body_bytes)
        headers = {k.lower():v for k,v in json.loads(checked(entry['headers'])).items()}
        assert int(headers['content-length']) == len(body_bytes)
        original_zip = checked(entry['ZIP']); assert blob(body['bundle']) == original_zip
        arm_meta = json.loads(checked(entry['metadata']))
        assert body['metadata'] == arm_meta and arm_meta['selection'] == arm
        assert arm_meta['geometry'] == geometry and arm_meta['uploaded_bytes_sha256'] == sha(data)
        assert set(arm_meta['outputs']) == set(body['outputs']) == {arm}
        contents = zipped(original_zip, common+[arm+'.png',arm+'-raw.npy','processing.json'])
        assert json.loads(contents['processing.json']) == arm_meta
        for name in common+[arm+'.png',arm+'-raw.npy']: assert contents[name] == merged[name]
        assert blob(body['outputs'][arm]) == contents[arm+'.png']
        output_meta = arm_meta['outputs'][arm]; assert output_meta == metadata['outputs'][arm]
        provenance = output_meta['metadata']; assert provenance == original_metadata['outputs'][arm]['metadata']
        assert provenance == surroundings_exp['response_metadata']['outputs'][arm]['metadata']
        assert provenance['fidelity'] == 1 and provenance['adain'] is True
        assert provenance['internal_resolution'] == 512 and provenance['output_resolution'] == 256
        if arm == 'original': assert provenance['weights_sha256'] == '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
        elif arm == 'r2': assert provenance['weights_sha256'] == 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
        else:
            assert provenance['weights_sha256'] == models['arms'][arm]['export']['sha256']
            assert provenance['adapter_tensor_sha256'] == models['arms'][arm]['adapter_tensor_sha256']
            assert provenance['base_tensor_sha256'] == models['base_tensor_sha256']
        raw = np.load(io.BytesIO(contents[arm+'-raw.npy']),allow_pickle=False); pixels = rgb(contents[arm+'.png'])
        assert raw.dtype == np.float32 and raw.shape == (256,256,3) and np.isfinite(raw).all()
        assert 0 <= raw.min() <= raw.max() <= 1
        assert np.array_equal(raw,np.load(folder/(arm+'-raw.npy'),allow_pickle=False))
        assert (folder/(arm+'.png')).read_bytes() == contents[arm+'.png']
        assert np.array_equal(np.floor(raw*np.float32(255)).astype(np.uint8),pixels)
        assert np.array_equal(raw[~support],(canvas.astype(np.float32)/np.float32(255))[~support])
        assert np.array_equal(pixels[~support],canvas[~support])
        assert sha(raw.tobytes()) == output_meta['raw_rgb_sha256']
        assert sha(pixels.tobytes()) == output_meta['PNG_RGB8_sha256']
        assert sha(contents[arm+'.png']) == output_meta['PNG_file_sha256']
        rows.append({'arm':arm,'PNG':pin(folder/(arm+'.png')),'RAW':pin(folder/(arm+'-raw.npy')),
                     'original_response':entry,'same_provenance_as_both_reused_comparators':True,
                     'RAW_PNG_floor_bounds_and_exact_offsupport_pass':True})
    delta = {key:experiment['server_after'][key]-experiment['server_before'][key]
             for key in ('neural_forward_attempts','neural_forward_returns','neural_forward_successes','completed_comparisons')}
    assert all(value >= 4 for value in delta.values())
    assert delta['neural_forward_successes'] == experiment['server_success_delta']
    assert experiment['other_server_activity_observed'] is (delta['neural_forward_successes'] != 4)
    assert 'torch' not in sys.modules
    receipt = {'format':'wild5-whole-strong-independent-saved-payload-audit-v1','complete':True,'audit_pass':True,
               'reviewer':'/root/wild5_surroundings_peer','auditor':pin(Path(__file__)),
               'experiment':pin(path),'source':experiment['source'],'derived_input':experiment['derived_input'],
               'whole_image_RGB_Gaussian_independently_rederived_exact':True,'native_radius_pixels':12.2578125,
               'same_added_Gaussian_kernel_as_surroundings':True,'equal_final_unknown_face_background_blur_claimed':False,
               'face_input_modified_by_whole_blur':True,'previous_protected_native_pixels':int(mask.sum()),
               'previous_protected_native_pixels_changed':int(np.count_nonzero(np.any(face_difference[mask] != 0,axis=1))),
               'face_RGB8_mean_absolute_change_from_global_blur':float(np.abs(face_difference[mask]).astype(np.float64).mean()),
               'input_change_is_not_recovered_accuracy':True,'new_original_server_single_arm_ZIPs':4,
               'new_verified_model_outputs':4,'client_merged_ZIP_members':12,'reused_outputs':8,
               'reused_physical_files_verified':reused_files,'reused_baselines':reused,
               'surroundings_independent_audit':pin(surroundings_audit_path),'model_manifest':pin(model_path),
               'all4_ContentLength_original_response_ZIP_geometry_source_provenance_RAW_PNG_bindings_pass':True,
               'server_counter_deltas':delta,'other_server_activity_observed':experiment['other_server_activity_observed'],
               'outputs':rows,'model_qualified':False,'paired_clean_reference':None,
               'no_accuracy_identity_native_or_quality_acceptance_claim':True,
               'audit_neural_gradient_optimizer_HTTP_browser_VM_calls':0}
    target = OUT/'independent_saved_payload_audit_v1.json'; assert not target.exists()
    target.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'audit_pass':True,'receipt':pin(target),'new_outputs':4,
                      'reused_outputs':8,'server_counter_deltas':delta,'neural_HTTP_VM_calls':0}))


if __name__ == '__main__': main()

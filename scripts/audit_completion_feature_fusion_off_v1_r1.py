"""Independent complete saved-stage/PNG audit and one frozen w=0 network replay."""
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
from PIL import Image
import cv2
import torch
from torch.nn import functional as F
from completion_feature_fusion_off_v1_common import ROOT, BASE, OUT, sha, read, write, rgb, binary, overlay, verify_bindings


def audit_stage(path, source, union, embedding, keys):
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(keys); data = {key: archive[key] for key in keys}
    expected_shapes = {'neural_input512': (1, 3, 512, 512), 'internal512': (1, 3, 512, 512), 'completion': (1, 3, 256, 256),
                       'logits': (1, 256, 512), 'lq_feat': (1, 256, 16, 16), 'code_indices': (1, 256, 1), 'quantized': (1, 256, 16, 16)}
    for key, value in data.items():
        assert value.shape == expected_shapes[key] and np.isfinite(value).all()
        assert value.dtype == (np.int64 if key == 'code_indices' else np.float32)
    assert data['completion'].min() >= 0 and data['completion'].max() <= 1
    canonical = torch.from_numpy((source.astype(np.float32)/np.float32(255)).transpose(2, 0, 1).copy())[None]
    mask = torch.from_numpy(union.astype(np.float32))[None, None]
    with torch.inference_mode():
        visible = 1-mask; high_visible = F.interpolate(visible, size=(512, 512), mode='bilinear', align_corners=False)
        context = F.interpolate(canonical*visible, size=(512, 512), mode='bilinear', align_corners=False)/high_visible.clamp_min(1e-8)
        high_mask = F.interpolate(mask, size=(512, 512), mode='nearest'); actual_input = (context*(1-high_mask)+high_mask)*2-1
        resized = F.interpolate(((torch.from_numpy(data['internal512'])+1)/2).clamp(0, 1), size=(256, 256), mode='bilinear', align_corners=False)
        composed = canonical*(1-mask)+resized*mask
        indices = torch.topk(F.softmax(torch.from_numpy(data['logits']), dim=2), 1, dim=2).indices
        gathered = embedding.index_select(0, indices.reshape(-1)).reshape(1, 16, 16, 256).permute(0, 3, 1, 2).contiguous()
    np.testing.assert_array_equal(data['neural_input512'], actual_input.numpy())
    np.testing.assert_array_equal(data['completion'], composed.numpy())
    np.testing.assert_array_equal(data['code_indices'], indices.numpy()); np.testing.assert_array_equal(data['quantized'], gathered.numpy())
    return data


def main():
    started = time.monotonic(); torch.set_num_threads(4)
    p, r, outer = read(OUT/'protocol.json'), read(OUT/'results.json'), read(OUT/'external_receipt.json')
    assert r['complete'] and outer['complete'] and outer['worker_exit_code'] == 0 and not outer['timeout']
    assert r['protocol_sha256'] == outer['protocol_sha256'] == sha(OUT/'protocol.json')
    assert r['seconds'] <= 600 and outer['external_seconds'] <= 660 and outer['log_sha256'] == sha(OUT/'inference.log')
    assert r['forwards'] == p['max_forwards'] and r['total_fusion_calls'] == p['expected_fusion_calls']
    assert r['candidate_fusion_calls'] == p['expected_candidate_fusion_calls'] and r['state_before'] == r['state_after'] == p['parent_model_state']
    assert r['requests'] == 37 and r['gradient_calls'] == r['backward_calls'] == r['optimizer_updates'] == r['automatic_generated_outputs'] == 0
    assert not r['new_checkpoint'] and not r['app_changes'] and not r['app_adoption'] and not r['native_or_reserved_used']
    verify_bindings(p)
    for name, value in r['artifacts_sha256'].items(): assert sha(OUT/name) == value
    assert sum((OUT/n).stat().st_size for n in r['artifacts_sha256']) == r['artifact_bytes'] <= 268435456
    assert sha(OUT/'input_review.json') == p['input_review_sha256'] and sha(OUT/'user_architecture_decision.json') == p['user_decision_sha256']
    assert read(OUT/'independent_protocol_audit.json')['complete']
    checkpoint = torch.load(ROOT/'checkpoints/codeformer_inpainting.pth', map_location='cpu', weights_only=True)
    embedding = checkpoint['params_ema']['quantize.embedding.weight'].detach().clone(); del checkpoint
    assert embedding.dtype == torch.float32 and embedding.shape == (512, 256) and torch.isfinite(embedding).all()
    parity_case = next(c for c in p['cases'] if c['id'] == p['parity_case'])
    source = rgb(ROOT/parity_case['input']); union = binary(ROOT/parity_case['conditioning'])
    before = audit_stage(OUT/'parity/stages.npz', source, union, embedding, p['saved_stages'])
    with np.load(ROOT/parity_case['baseline_stages'], allow_pickle=False) as old:
        for key in ['neural_input512', 'completion', 'internal512']: np.testing.assert_array_equal(before[key], old[key])
    np.testing.assert_array_equal(rgb(OUT/'parity/estimate.png'), rgb(ROOT/parity_case['baseline_output']))
    assert read(OUT/'parity/receipt.json')['cached_input_raw_and_PNG_exact']
    matched = read(OUT/'parity/matched_w0_receipt.json'); assert matched['complete'] and matched['case'] == p['parity_case']
    mapping = {c['id']: c for c in p['cases']}; assert [c['id'] for c in p['cases']] == [row['id'] for row in r['rows']]
    visible_bytes = protected_bytes = 0; generated = bypasses = exclusions = 0; findings = []
    for row in r['rows']:
        c = mapping[row['id']]; assert row['rejected_before_neural'] == c['rejected']
        if c['rejected']:
            assert row['input_review'] in ['needs_clearer', 'out_of_scope'] and row['input_review'] == c['input_review']
            assert not (OUT/'images'/(c['id']+'.png')).exists(); exclusions += 1; continue
        source = rgb(ROOT/c['input']); output = rgb(OUT/row['output']); final = binary(ROOT/c['masks']['removal'])
        union = binary(ROOT/c['conditioning']); protected = binary(ROOT/c['masks']['protected']); meta = read(OUT/row['metadata'])
        expected = source.copy()
        if final.any():
            data = audit_stage(OUT/row['stages'], source, union, embedding, p['saved_stages'])
            raw = data['completion'][0].transpose(1, 2, 0); expected[final] = np.floor(raw[final]*np.float32(255)).astype(np.uint8)
            assert meta['completion']['network_w'] == 0 and not meta['completion']['adain'] and meta['completion']['weights_sha256'] == p['weights_sha256']
            assert meta['actual_fusion_calls'] == p['expected_candidate_fusion_calls']
            if c['id'] == p['parity_case']:
                for key in ['neural_input512', 'logits', 'lq_feat', 'code_indices', 'quantized']: np.testing.assert_array_equal(data[key], before[key])
            generated += 1
        else:
            with np.load(OUT/row['stages'], allow_pickle=False) as empty: assert not empty.files
            assert not union.any() and meta['completion'] == {'bypassed': 'empty mask'}; bypasses += 1
        colour = meta['display_processing']['colour_policy']
        assert colour == read(ROOT/c['baseline_metadata'])['display_processing']['colour_policy']
        support = cv2.erode((~final).astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
        blurred = cv2.GaussianBlur(source.astype(np.float32), (5, 5), 1)
        chroma = float((blurred.max(-1)-blurred.min(-1))[support].mean()) if support.sum() >= 512 else None
        assert colour['mean_visible_channel_range_255'] == chroma
        assert colour['applied'] == colour['grayscale_input'] == (chroma is not None and chroma <= 4)
        if colour['applied']: expected[final] = np.repeat(cv2.cvtColor(expected, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)[final]
        np.testing.assert_array_equal(output, expected); np.testing.assert_array_equal(output[~final], source[~final]); np.testing.assert_array_equal(output[protected], source[protected])
        assert meta['restoration_requested'] == 'off' and not meta['restoration_applied'] and meta['mask_source'] == 'assisted_reviewed'
        assert meta['optimizer_updates'] == meta['gradient_calls'] == meta['backward_calls'] == 0
        changed = int(np.any(output != rgb(ROOT/c['baseline_output']), -1)[final].sum())
        assert row['changed_pixels_vs_w1_within_final'] == changed and row['final_pixels'] == int(final.sum()) and row['conditioning_pixels'] == int(union.sum())
        visible_bytes += int(source[~final].size); protected_bytes += int(source[protected].size)
        findings.append({'id': c['id'], 'family': c['family'], 'condition': c['condition'], 'empty_bypass': not bool(final.any()),
                         'source_changed_pixels_outside_final': 0, 'protected_changed_pixels': 0, 'changed_estimated_pixels_vs_w1': changed})
        assert time.monotonic()-started < 180
    assert generated == 28 and len(findings) == 32 and bypasses == exclusions == 4
    exact_cells = 0; page_ids = []
    for page in r['pages']:
        with Image.open(OUT/page['path']) as im: assert im.size == (1280, 1214); array = np.array(im.convert('RGB'))
        for entry in page['entries']:
            c = mapping[entry['id']]; source = rgb(ROOT/c['input']); y = 30+296*entry['row']
            cells = [source, overlay(source, binary(ROOT/c['masks']['removal']), (16, 185, 129)), overlay(source, binary(ROOT/c['conditioning']), (240, 140, 32)), rgb(ROOT/c['baseline_output']), rgb(OUT/'images'/(c['id']+'.png'))]
            for k, cell in enumerate(cells): np.testing.assert_array_equal(array[y:y+256, k*256:(k+1)*256], cell); exact_cells += 1
            page_ids.append(c['id'])
    assert len(r['pages']) == 8 and len(page_ids) == len(set(page_ids)) == 32 and exact_cells == 160
    # One independently loaded frozen direct-network replay, using actual captured input.
    sys.path.insert(0, str(ROOT)); from pretrained_completion import load_codeformer
    from cctv_dgp_pilot import state_hash
    model, provenance = load_codeformer(ROOT/'checkpoints/codeformer_inpainting.pth', 'cpu'); initial_state = state_hash(model)
    assert initial_state == p['parent_model_state'] and all(not v.requires_grad and v.grad is None for v in model.parameters())
    with np.load(OUT/'stages'/(p['parity_case']+'.npz'), allow_pickle=False) as saved:
        replay_source = {key: saved[key] for key in ['neural_input512', 'internal512', 'logits', 'lq_feat']}
    with torch.inference_mode(): actual = model.net(torch.from_numpy(replay_source['neural_input512']).to(memory_format=torch.channels_last), w=0, adain=False)
    for key, value in zip(['internal512', 'logits', 'lq_feat'], actual): np.testing.assert_array_equal(value.numpy(), replay_source[key])
    assert state_hash(model) == initial_state and all(v.grad is None for v in model.parameters())
    verify_bindings(p); assert time.monotonic()-started < 180
    write(OUT/'independent_saved_output_audit_r1.json', {'complete': True, 'protocol_sha256': sha(OUT/'protocol.json'), 'results_sha256': sha(OUT/'results.json'),
          'checker_sha256': sha(Path(__file__)), 'sources_verified': len(p['sources_sha256']), 'artifacts_verified': len(r['artifacts_sha256']),
          'cases': 36, 'all28_actual_neural_inputs_and_codebook_lookups_exact': True, 'all28_raw512_to256_and_final_PNG_compositions_exact': True,
          'delivered_outputs': 32, 'empty_bypasses': 4, 'pre_neural_exclusions': 4, 'all160_page_cells_exact': True,
          'visible_source_bytes_exact': visible_bytes, 'protected_source_bytes_exact': protected_bytes, 'rows': findings,
          'matched_w1_w0_encoder_and_codebook_exact_cases': 1, 'independent_frozen_w0_network_replay': {'cases': 1, 'case': p['parity_case'], 'network_raw_logits_and_encoder_exact': True, 'state_unchanged': True},
          'model_forwards_in_audit': 1, 'DGP_forwards': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
          'automatic_quality_qualification': False, 'assisted_quality_qualification': False, 'app_adoption': False,
          'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-started, 'cap_seconds': 180})
    print({'complete': True, 'outputs': 32, 'exact_cells': exact_cells, 'seconds': time.monotonic()-started}, flush=True)


if __name__ == '__main__': main()

"""Finite frozen CPU feature-fusion ablation; no app edits or local learning."""
from pathlib import Path
import sys
import time
import traceback
from types import MethodType
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from completion_feature_fusion_off_v1_common import ROOT, BASE, OUT, sha, read, write, rgb, binary, overlay, verify_bindings
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from dgp_face_workflow_v3 import DGPFaceWorkflow, canonical_tensor, float_rgb, review_input
from dgp_face_restoration import prepare_crop
from face_color_policy import preserve_input_palette
from face_workflow import visibility_check


def main():
    assert Path(sys.executable).resolve() == (ROOT/'venv/Scripts/python.exe').resolve()
    assert not (OUT/'execution.json').exists() and not (OUT/'results.json').exists()
    p = read(OUT/'protocol.json'); checked = read(OUT/'independent_protocol_audit.json')
    assert checked['complete'] and checked['protocol_sha256'] == sha(OUT/'protocol.json')
    assert p['input_review_sha256'] == sha(OUT/'input_review.json') and p['user_decision_sha256'] == sha(OUT/'user_architecture_decision.json')
    verify_bindings(p); started = time.monotonic(); rows = []; handles = []; captured = {}; active_w = 1
    counts = dict.fromkeys(p['max_forwards'], 0); fusions = dict.fromkeys(p['expected_fusion_calls'], 0); call_fusions = {}; before = None; quantizer = None
    try:
        torch.manual_seed(p['seed']); np.random.seed(p['seed'])
        assert torch.__version__ == p['torch'] and not torch.cuda.is_available()
        engine = DGPFaceWorkflow(device='cpu'); engine._runtime(); model = engine._generator()
        assert engine.restorer is None and engine.detector is None
        assert not model.training and all(not v.requires_grad and v.grad is None for v in model.parameters())
        before = state_hash(model); assert before == p['parent_model_state']
        def network_input(module, args, kwargs):
            assert kwargs == {'w': 1, 'adain': False} and len(args) == 1
            assert not captured and args[0].shape == (1, 3, 512, 512) and args[0].dtype == torch.float32
            captured['neural_input512'] = args[0].detach().numpy().copy()
            return args, {'w': active_w, 'adain': False}
        def network_output(module, args, result):
            counts['internal512'] += 1
            assert isinstance(result, tuple) and len(result) == 3
            for name, value in zip(['internal512', 'logits', 'lq_feat'], result): captured[name] = value.detach().numpy().copy()
        def completion_output(module, args, result):
            counts['completion'] += 1; captured['completion'] = result.detach().numpy().copy()
        def fusion(key):
            def hook(*_):
                assert active_w == 1, 'Fusion should never execute for w=0'
                fusions[key] += 1; call_fusions[key] += 1
            return hook
        assert not model.net._forward_pre_hooks and not model.net._forward_hooks
        handles = [model.net.register_forward_pre_hook(network_input, with_kwargs=True), model.net.register_forward_hook(network_output), model.register_forward_hook(completion_output)]
        for key, block in model.net.fuse_convs_dict.items(): handles.append(block.register_forward_hook(fusion(key)))
        quantizer = model.net.quantize; assert 'get_codebook_feat' not in quantizer.__dict__
        original_codebook = quantizer.get_codebook_feat
        def capture_codebook(self, indices, shape):
            assert shape == [1, 16, 16, 256] and indices.shape == (1, 256, 1) and indices.dtype == torch.int64
            result = original_codebook(indices, shape)
            counts['codebook'] += 1; captured['code_indices'] = indices.detach().numpy().copy(); captured['quantized'] = result.detach().numpy().copy()
            return result
        quantizer.get_codebook_feat = MethodType(capture_codebook, quantizer)
        write(OUT/'execution.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'state_before': before, 'device': 'cpu', 'torch': torch.__version__,
              'threads': torch.get_num_threads(), 'frozen': True, 'optimizer_constructed': False, 'gradient_calls': 0, 'cap_seconds': 600})
        for folder in ['images', 'stages', 'metadata', 'pages', 'parity']: (OUT/folder).mkdir()
        def generate(c, weight):
            nonlocal active_w, call_fusions
            assert time.monotonic()-started < 600
            active_w = weight; captured.clear(); call_fusions = dict.fromkeys(fusions, 0)
            source = rgb(ROOT/c['input']); final = binary(ROOT/c['masks']['removal']); union = binary(ROOT/c['conditioning'])
            review_input(c['input_review']); canvas, observed, prepared_mask, geometry = prepare_crop(source, final.astype(np.uint8))
            np.testing.assert_array_equal(canvas, source); assert observed.all(); np.testing.assert_array_equal(prepared_mask, final)
            visibility = visibility_check((~observed | final).astype(np.uint8)); assert not visibility['rejected']
            assert float(union.mean()) < .85 and not (final & ~union).any()
            with engine.lock, torch.inference_mode():
                x = canonical_tensor(canvas, 'cpu'); u = torch.from_numpy(union.astype(np.float32))[None, None]
                value = float_rgb(model(x, u), x)
            assert set(captured) == set(p['saved_stages'])
            assert call_fusions == (p['expected_fusion_calls'] if weight == 1 else p['expected_candidate_fusion_calls'])
            assert all(counts[key] <= maximum for key, maximum in p['max_forwards'].items())
            output = source.copy(); output[final] = np.floor(value[final]*np.float32(255)).astype(np.uint8)
            output, colour = preserve_input_palette(source, final.astype(np.uint8), output, visible_restored=False)
            assert colour == read(ROOT/c['baseline_metadata'])['display_processing']['colour_policy']
            meta = {'restoration_requested': 'off', 'restoration_applied': False, 'mask_source': 'assisted_reviewed',
                    'completion': {**engine.generator_provenance, 'network_w': weight, 'adain': False, 'default_inpainting_w': 1},
                    'geometry': geometry, 'visibility': visibility, 'actual_fusion_calls': dict(call_fusions),
                    'display_processing': {'float_to_png': 'floor(float32*255)', 'colour_policy': colour},
                    'additional_final_expansion': 0, 'input_review': c['input_review'],
                    'original_rgb_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
                    'final_mask_sha256': hashlib.sha256(final.astype(np.uint8).tobytes()).hexdigest(),
                    'conditioning_mask_sha256': hashlib.sha256(union.astype(np.uint8).tobytes()).hexdigest(),
                    'output_rgb_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
                    'optimizer_updates': 0, 'gradient_calls': 0, 'backward_calls': 0,
                    'meaning': 'One plausible estimate; no clean hidden reference or exact hidden identity claim'}
            return output, meta
        c = next(c for c in p['cases'] if c['id'] == p['parity_case']); fresh, meta = generate(c, 1)
        np.testing.assert_array_equal(fresh, rgb(ROOT/c['baseline_output']))
        with np.load(ROOT/c['baseline_stages'], allow_pickle=False) as cache:
            for key in ['neural_input512', 'internal512', 'completion']: np.testing.assert_array_equal(captured[key], cache[key])
        parity = {key: value.copy() for key, value in captured.items()}
        Image.fromarray(fresh).save(OUT/'parity/estimate.png'); np.savez_compressed(OUT/'parity/stages.npz', **captured)
        write(OUT/'parity/receipt.json', {'complete': True, 'case': c['id'], 'cached_input_raw_and_PNG_exact': True,
              'new_tolerance': False, 'network_w': 1, 'fusion_calls': dict(call_fusions)})
        print({'parity_exact': c['id'], 'seconds': time.monotonic()-started}, flush=True)
        for c in p['cases']:
            assert time.monotonic()-started < 600
            captured.clear(); source = rgb(ROOT/c['input']); count_before = dict(counts)
            if c['rejected']:
                try: engine.generate(source, np.zeros((256, 256), np.uint8), 'off', c['input_review'], True)
                except ValueError as exc:
                    assert count_before == counts and not captured
                    rows.append({'id': c['id'], 'rejected_before_neural': True, 'input_review': c['input_review'], 'message': str(exc)}); continue
                raise AssertionError('Prior input-only rejection must be retained')
            final = binary(ROOT/c['masks']['removal']); protected = binary(ROOT/c['masks']['protected'])
            if final.any():
                output, meta = generate(c, 0)
                if c['id'] == p['parity_case']:
                    keys = ['neural_input512', 'logits', 'lq_feat', 'code_indices', 'quantized']
                    for key in keys: np.testing.assert_array_equal(captured[key], parity[key])
                    write(OUT/'parity/matched_w0_receipt.json', {'complete': True, 'case': c['id'], 'exact_encoder_and_codebook_stages': keys,
                          'w1_fusion_calls': p['expected_fusion_calls'], 'w0_fusion_calls': dict(call_fusions), 'quality_pass': False})
            else:
                result = engine.generate(source, final.astype(np.uint8), 'off', c['input_review'], True)
                output, meta = result['output'], result['metadata']; assert count_before == counts and not captured
                np.testing.assert_array_equal(output, source)
                meta.update({'completion': {'bypassed': 'empty mask'}, 'actual_fusion_calls': dict.fromkeys(fusions, 0),
                             'optimizer_updates': 0, 'gradient_calls': 0, 'backward_calls': 0})
            np.testing.assert_array_equal(output[~final], source[~final]); np.testing.assert_array_equal(output[protected], source[protected])
            assert engine.restorer is None and engine.detector is None
            path = 'images/'+c['id']+'.png'; stage = 'stages/'+c['id']+'.npz'; metadata = 'metadata/'+c['id']+'.json'
            Image.fromarray(output).save(OUT/path); np.savez_compressed(OUT/stage, **captured); write(OUT/metadata, meta)
            rows.append({'id': c['id'], 'rejected_before_neural': False, 'output': path, 'stages': stage, 'metadata': metadata,
                         'empty_bypass': not bool(final.any()), 'final_pixels': int(final.sum()), 'conditioning_pixels': int(binary(ROOT/c['conditioning']).sum()),
                         'changed_pixels_vs_w1_within_final': int(np.any(output != rgb(ROOT/c['baseline_output']), axis=-1)[final].sum()),
                         'source_changed_pixels_outside_final': 0})
            print({'case': c['id'], 'forwards': counts['completion'], 'seconds': round(time.monotonic()-started, 2)}, flush=True)
        after = state_hash(model); assert before == after and counts == p['max_forwards'] and fusions == p['expected_fusion_calls']
        assert len(rows) == 36 and sum(r['rejected_before_neural'] for r in rows) == sum(r.get('empty_bypass', False) for r in rows) == 4
        assert all(not v.requires_grad and v.grad is None for v in model.parameters())
        pages = []; font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
        for condition in ['original_photo', 'synthetic_degraded_photo']:
            selected = [c for c in p['cases'] if not c['rejected'] and c['condition'] == condition]
            for first in range(0, 16, 4):
                page = Image.new('RGB', (1280, 1214), 'white'); draw = ImageDraw.Draw(page); entries = []
                for k, label in enumerate(['Input256', 'Final removal unchanged', 'Same union conditioning', 'Cached w1 fused estimate', 'w0 fusion-disabled estimate']): draw.text((k*256+4, 7), label, font=font, fill='black')
                for row, c in enumerate(selected[first:first+4]):
                    source = rgb(ROOT/c['input']); cells = [source, overlay(source, binary(ROOT/c['masks']['removal']), (16, 185, 129)),
                          overlay(source, binary(ROOT/c['conditioning']), (240, 140, 32)), rgb(ROOT/c['baseline_output']), rgb(OUT/'images'/(c['id']+'.png'))]
                    y = 30+row*296
                    for k, cell in enumerate(cells): page.paste(Image.fromarray(cell), (k*256, y))
                    draw.text((4, y+260), c['id']+' | '+c['family'], font=font, fill='black'); entries.append({'id': c['id'], 'row': row})
                path = 'pages/'+condition+'_'+str(first//4+1)+'.png'; page.save(OUT/path)
                pages.append({'path': path, 'sha256': sha(OUT/path), 'entries': entries})
        artifacts = {q.relative_to(OUT).as_posix(): sha(q) for folder in ['images', 'stages', 'metadata', 'pages', 'parity'] for q in sorted((OUT/folder).rglob('*')) if q.is_file()}
        size = sum((OUT/n).stat().st_size for n in artifacts); assert size <= p['artifact_cap_bytes'] and time.monotonic()-started < 600
        verify_bindings(p)
        write(OUT/'results.json', {'complete': True, 'protocol_sha256': sha(OUT/'protocol.json'), 'seconds': time.monotonic()-started,
              'rows': rows, 'forwards': counts, 'total_fusion_calls': fusions, 'state_before': before, 'state_after': after,
              'artifacts_sha256': artifacts, 'artifact_bytes': size, 'pages': pages, 'requests': 37,
              'matched_case_encoder_logits_indices_quantized_exact': True, 'candidate_fusion_calls': p['expected_candidate_fusion_calls'],
              'gradient_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'new_checkpoint': False, 'automatic_generated_outputs': 0,
              'app_changes': False, 'app_adoption': False, 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
              'native_or_reserved_used': False, 'independent_final_review': False, 'hidden_metrics': None, 'goal_complete': False})
        print({'complete': True, 'seconds': time.monotonic()-started, 'forwards': counts, 'artifact_bytes': size}, flush=True)
    except BaseException as exc:
        write(OUT/'failure.json', {'complete': False, 'error': repr(exc), 'traceback': traceback.format_exc(), 'seconds': time.monotonic()-started,
              'protocol_sha256': sha(OUT/'protocol.json'), 'rows_completed': len(rows), 'forwards': counts, 'optimizer_updates': 0, 'automatic_repeat_permitted': False})
        raise
    finally:
        for handle in handles: handle.remove()
        if quantizer is not None and 'get_codebook_feat' in quantizer.__dict__: delattr(quantizer, 'get_codebook_feat')


if __name__ == '__main__': main()

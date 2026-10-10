"""Bounded detector-only production trace; no reconstruction, gradients or tuning."""
import argparse
import time
import traceback
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from dgp_face_workflow_v3 import DGPFaceWorkflow
from automatic_proposal_score_v1_common import (
    ROOT, OUT, BUDGETS, sha, read, write, rgb, binary, overlay, regions, stats, state_sha, bindings)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol-sha', required=True)
    pin = parser.parse_args().protocol_sha; assert sha(OUT/'protocol.json') == pin
    p = read(OUT/'protocol.json'); bindings(p); assert not (OUT/'stages').exists()
    started = time.monotonic(); counts = {'detector': 0, 'internal_completion': 0}; rows = []
    for name in ['stages', 'masks', 'pages']: (OUT/name).mkdir()
    model = None; before = None; captured = {}
    try:
        engine = DGPFaceWorkflow(device='cpu'); engine._runtime(); model = engine._detector()
        assert not model.training and all(not value.requires_grad for value in model.parameters())
        before = state_sha(model)
        def trace(_model, inputs, output):
            counts['detector'] += 1; assert counts['detector'] <= 36
            captured.clear(); captured.update({'input': inputs[0].detach().cpu().numpy().copy(),
                'input_strides': list(inputs[0].stride()), 'logits': output.detach().cpu().numpy().copy(),
                'logit_strides': list(output.stride())})
        def unexpected_completion(*_):
            counts['internal_completion'] += 1; raise AssertionError('Detector completion branch forbidden')
        hooks = [model.segmenter.register_forward_hook(trace), model.generator.register_forward_hook(unexpected_completion)]
        for case in p['cases']:
            assert time.monotonic()-started <= 180
            source = rgb(ROOT/case['input']); count = counts['detector']
            result = engine.review_mask(source); assert counts['detector'] == count+1
            assert engine.generator is None and engine.restorer is None
            assert result['proposal_margin_pixels'] == 3 and result['threshold'] == .5
            assert np.array_equal(result['mask'] != 0, binary(ROOT/case['automatic'])), 'Cached production proposal differs'
            with torch.inference_mode():
                network_probability = torch.from_numpy(captured['logits']).sigmoid()[0, 0].numpy().copy()
            canvas_probability = cv2.resize(network_probability, (256, 256), interpolation=cv2.INTER_LINEAR)
            raw = canvas_probability >= .5
            assert np.array_equal(raw, result['raw_mask'] != 0)
            proposed = cv2.dilate(raw.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
            assert np.array_equal(proposed, result['mask'] != 0)
            stage = OUT/'stages'/(case['id']+'.npz')
            np.savez_compressed(stage, neural_input=captured['input'], logits=captured['logits'],
                network_probability=network_probability, canvas_probability=canvas_probability)
            Image.fromarray(raw.astype(np.uint8)*255).save(OUT/'masks'/(case['id']+'_raw.png'))
            Image.fromarray(proposed.astype(np.uint8)*255).save(OUT/'masks'/(case['id']+'_proposal.png'))
            support = regions(case)
            row = {'id': case['id'], 'base_id': case['base_id'], 'family': case['family'], 'condition': case['condition'],
                'rejected_input_retained': case['rejected'], 'input_review': case['input_review'],
                'raw_pixels': int(raw.sum()), 'proposal_pixels': int(proposed.sum()),
                'whole_canvas_scores': stats(canvas_probability, np.ones((256, 256), bool)),
                'regions': {key: stats(canvas_probability, mask) for key, mask in support.items()},
                'proposal_region_pixels': {key: int((proposed & mask).sum()) for key, mask in support.items()},
                'input_strides': captured['input_strides'], 'logit_strides': captured['logit_strides'],
                'neural_input_shape': list(captured['input'].shape), 'logit_shape': list(captured['logits'].shape),
                'cached_proposal_exact': True, 'historical_footprints_not_expert_truth': True,
                'generation_requests': 0, 'quality_qualification': False}
            rows.append(row)
            write(OUT/'stages'/(case['id']+'.json'), row)
            assert sum(q.stat().st_size for q in OUT.rglob('*') if q.is_file()) <= BUDGETS['artifact_bytes']
            print({'case': case['id'], 'detector_forwards': counts['detector'], 'proposal_pixels': row['proposal_pixels']}, flush=True)
        for hook in hooks: hook.remove()
        pairs = []
        for base in dict.fromkeys(c['base_id'] for c in p['cases']):
            a = next(r for r in rows if r['base_id'] == base and r['condition'] == 'original_photo')
            b = next(r for r in rows if r['base_id'] == base and r['condition'] == 'synthetic_degraded_photo')
            pairs.append({'base_id': base, 'original_id': a['id'], 'degraded_id': b['id'],
                'proposal_pixels_original': a['proposal_pixels'], 'proposal_pixels_degraded': b['proposal_pixels'],
                'raw_pixels_original': a['raw_pixels'], 'raw_pixels_degraded': b['raw_pixels'],
                'maximum_score_difference': b['whole_canvas_scores']['maximum']-a['whole_canvas_scores']['maximum'],
                'region_mean_score_differences': {key: None if a['regions'][key]['mean'] is None else
                    b['regions'][key]['mean']-a['regions'][key]['mean'] for key in a['regions']}})
        for index in range(9):
            cases = p['cases'][index*4:index*4+4]; sheet = Image.new('RGB', (1280, 1200), '#16181c')
            draw = ImageDraw.Draw(sheet)
            for col, title in enumerate(['input', 'automatic area', 'score brightness [0,1]', 'assisted core / held footprint', 'protected appearance']):
                draw.text((col*256+4, 8), title, fill='white')
            for slot, case in enumerate(cases):
                y = 32+slot*292
                draw.text((4, y), case['id']+(' | HELD INPUT' if case['rejected'] else ''), fill='white')
                source = rgb(ROOT/case['input']); proposed = binary(OUT/'masks'/(case['id']+'_proposal.png'))
                with np.load(OUT/'stages'/(case['id']+'.npz'), allow_pickle=False) as saved: probability = saved['canvas_probability']
                gray = np.floor(probability*255+.5).astype(np.uint8); heatmap = np.repeat(gray[..., None], 3, axis=2)
                support = regions(case); core = support.get('assisted_core', binary(ROOT/case['reviewed']))
                protected = support.get('protected_appearance', np.zeros((256, 256), bool))
                cells = [source, overlay(source, proposed, [16, 185, 129]), heatmap,
                         overlay(source, core, [255, 140, 20]), overlay(source, protected, [190, 80, 255])]
                for col, cell in enumerate(cells): sheet.paste(Image.fromarray(cell), (col*256, y+20))
            sheet.save(OUT/'pages'/('page_'+format(index+1, '02d')+'.png'))
        assert counts == {'detector': 36, 'internal_completion': 0}
        assert state_sha(model) == before and engine.generator is None and engine.restorer is None
        bindings(p); assert time.monotonic()-started <= 180
        files = {q.relative_to(OUT).as_posix(): sha(q) for folder in ['stages', 'masks', 'pages']
                 for q in (OUT/folder).rglob('*') if q.is_file()}
        assert sum((OUT/name).stat().st_size for name in files) <= BUDGETS['artifact_bytes']
        write(OUT/'results.json', {'complete': True, 'protocol_sha256': pin, 'rows': rows, 'pairs': pairs,
            'state_before': before, 'state_after': state_sha(model), 'counts': counts,
            'DGP_forwards': 0, 'completion_forwards': 0, 'gradient_queries': 0, 'backward_calls': 0,
            'optimizer_updates': 0, 'generation_requests': 0, 'artifact_sha256': files,
            'all36_cached_proposals_exact': True, 'seconds': time.monotonic()-started,
            'detector_checkpoint_epoch': engine.detector_state['epoch'], 'detector_input_size': engine.detector_state['size'],
            'native_or_final_used': False, 'app_adoption': False, 'goal_complete': False,
            'visual_review_pending': True, 'automatic_completion_quality_qualified': False})
        print({'complete': True, 'detector_forwards': 36, 'seconds': time.monotonic()-started}, flush=True)
    except BaseException as exc:
        write(OUT/'failure.json', {'complete': False, 'error': repr(exc), 'traceback': traceback.format_exc(),
            'rows': rows, 'counts': counts, 'seconds': time.monotonic()-started,
            'state_before': before, 'state_after': state_sha(model) if model is not None else None,
            'optimizer_updates': 0, 'generation_requests': 0, 'automatic_resume': False})
        raise


if __name__ == '__main__': main()

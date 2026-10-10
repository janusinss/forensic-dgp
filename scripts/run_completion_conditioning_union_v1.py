"""Finite frozen CPU conditioning test; final reviewed support stays unchanged."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_conditioning_union_v1'
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from dgp_face_workflow_v3 import DGPFaceWorkflow, canonical_tensor, float_rgb, review_input
from dgp_face_restoration import prepare_crop
from face_color_policy import preserve_input_palette
from face_workflow import visibility_check


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def rgb(path):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.array(im.convert('RGB'))


def binary(path):
    with Image.open(path) as im:
        assert im.mode == 'L' and im.size == (256, 256)
        value = np.array(im)
    assert set(np.unique(value)) <= {0, 255}
    return value != 0


def overlay(source, mask, colour):
    values = source.astype(np.float64)
    values[mask] = .55 * values[mask] + .45 * np.array(colour)
    return np.floor(values + .5).astype(np.uint8)


def main():
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert not (OUT / 'execution.json').exists() and not (OUT / 'results.json').exists()
    p, audit = read(OUT / 'protocol.json'), read(OUT / 'independent_input_audit.json')
    assert audit['complete'] and audit['draft_sha256'] == p['draft_sha256'] == sha(OUT / 'draft.json')
    assert p['input_audit_sha256'] == sha(OUT / 'independent_input_audit.json')
    assert p['input_visual_review_sha256'] == sha(OUT / 'input_visual_review.json')
    assert p['max_forwards'] == {'completion': 29, 'internal512': 29, 'DGP': 0, 'detector': 0}
    assert p['cap_seconds'] == 600 and p['expected_trial_forwards'] == 28
    for name, digest in {**p['sources_sha256'], **p['app_preservation_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    started = time.monotonic()
    counts = {'completion': 0, 'internal512': 0, 'DGP': 0, 'detector': 0}
    captured, handles, rows, before = {}, [], [], None
    try:
        torch.manual_seed(p['seed'])
        np.random.seed(p['seed'])
        assert torch.__version__ == '2.13.0+cpu' and not torch.cuda.is_available()
        engine = DGPFaceWorkflow(device='cpu')
        engine._runtime()
        model = engine._generator()
        assert engine.restorer is None and engine.detector is None
        assert not model.training and all(not value.requires_grad for value in model.parameters())
        before = state_hash(model)
        assert before == p['parent_model_state']
        def capture_input(module, args, kwargs):
            assert kwargs == {'w': 1, 'adain': False} and len(args) == 1
            assert args[0].shape == (1, 3, 512, 512) and args[0].dtype == torch.float32
            captured['neural_input512'] = args[0].detach().cpu().numpy().copy()
        def capture(name):
            def hook(module, args, result):
                counts[name] += 1
                assert counts[name] <= p['max_forwards'][name]
                value = result[0] if name == 'internal512' else result
                captured[name] = value.detach().cpu().numpy().copy()
            return hook
        handles = [model.net.register_forward_pre_hook(capture_input, with_kwargs=True),
                   model.net.register_forward_hook(capture('internal512')),
                   model.register_forward_hook(capture('completion'))]
        write(OUT / 'execution.json', {'protocol_sha256': sha(OUT / 'protocol.json'),
              'input_audit_sha256': p['input_audit_sha256'], 'state_before': before,
              'device': 'cpu', 'torch': torch.__version__, 'threads': torch.get_num_threads(),
              'optimizer_constructed': False, 'backward_calls': 0, 'cap_seconds': 600})
        for folder in ['images', 'stages', 'metadata', 'pages', 'parity']:
            (OUT / folder).mkdir()
        parity = next(c for c in p['cases'] if c['id'] == p['parity_case'])
        source, mask = rgb(ROOT / parity['input']), binary(ROOT / parity['masks']['removal']).astype(np.uint8)
        captured.clear()
        fresh = engine.generate(source, mask, 'off', parity['input_review'], True)
        np.testing.assert_array_equal(fresh['output'], rgb(ROOT / parity['baseline_output']))
        with np.load(ROOT / parity['baseline_stages'], allow_pickle=False) as saved:
            for name in ['completion', 'internal512']:
                np.testing.assert_array_equal(captured[name], saved[name])
        Image.fromarray(fresh['output']).save(OUT / 'parity/estimate.png')
        np.savez_compressed(OUT / 'parity/stages.npz', **captured)
        write(OUT / 'parity/receipt.json', {'complete': True, 'case': parity['id'],
              'cached_PNG_byte_exact': True, 'cached_raw512_and256_exact': True,
              'counts': dict(counts), 'model_state': before, 'new_tolerance': False})
        print(json.dumps({'parity': parity['id'], 'exact': True, 'seconds': time.monotonic() - started}), flush=True)
        for c in p['cases']:
            assert time.monotonic() - started < 600, '600-second conditioning inference stop'
            source = rgb(ROOT / c['input'])
            captured.clear()
            if c['rejected']:
                count_before = dict(counts)
                try:
                    engine.generate(source, np.zeros((256, 256), np.uint8), 'off', c['input_review'], True)
                except ValueError as exc:
                    assert c['input_review'] in ['out_of_scope', 'needs_clearer'] and counts == count_before and not captured
                    rows.append({'id': c['id'], 'rejected_before_neural': True, 'message': str(exc), 'input_review': c['input_review']})
                    continue
                raise AssertionError('Expected preserved input-only rejection')
            final, union = binary(ROOT / c['masks']['removal']), binary(ROOT / c['conditioning'])
            assert np.array_equal(union, final | binary(ROOT / c['reviewed'])) and not (final & ~union).any()
            if final.any():
                review_input(c['input_review'])
                canvas, observed, prepared_mask, geometry = prepare_crop(source, final.astype(np.uint8))
                np.testing.assert_array_equal(canvas, source)
                np.testing.assert_array_equal(prepared_mask, final)
                assert observed.all() and (~final).any() and np.ptp(source[~final].astype(np.int16), axis=0).max() > 0
                visibility = visibility_check((~observed | final).astype(np.uint8))
                assert not visibility['rejected'] and float(union.mean()) < .85
                with engine.lock, torch.inference_mode():
                    x = canonical_tensor(canvas, 'cpu')
                    u = torch.from_numpy(union.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    result = model(x, u)
                    value = float_rgb(result, x)
                assert set(captured) == {'neural_input512', 'internal512', 'completion'}
                output = source.copy()
                output[final] = np.floor(value[final] * np.float32(255)).astype(np.uint8)
                output, colour = preserve_input_palette(source, final.astype(np.uint8), output, visible_restored=False)
                old_colour = read(ROOT / c['baseline_metadata'])['display_processing']['colour_policy']
                assert colour == old_colour, 'Display selection must use fixed FINAL support'
                meta = {'restoration_requested': 'off', 'restoration_applied': False,
                        'mask_source': 'assisted_reviewed', 'additional_final_expansion': 0,
                        'completion': engine.generator_provenance, 'geometry': geometry, 'visibility': visibility,
                        'display_processing': {'float_to_png': 'floor(float32*255)', 'colour_policy': colour,
                                               'composition': 'Only unchanged final reviewed support; union used for network conditioning only'}}
            else:
                assert not union.any()
                result = engine.generate(source, final.astype(np.uint8), 'off', c['input_review'], True)
                assert not captured
                output, meta = result['output'], result['metadata']
                np.testing.assert_array_equal(output, source)
            np.testing.assert_array_equal(output[~final], source[~final])
            np.testing.assert_array_equal(output[binary(ROOT / c['masks']['protected'])], source[binary(ROOT / c['masks']['protected'])])
            assert engine.restorer is None and engine.detector is None
            meta.update({'conditioning_policy': p['conditioning_rule'], 'input_review': c['input_review'],
                         'original_rgb_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
                         'final_mask_sha256': hashlib.sha256(final.astype(np.uint8).tobytes()).hexdigest(),
                         'conditioning_mask_sha256': hashlib.sha256(union.astype(np.uint8).tobytes()).hexdigest(),
                         'output_rgb_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
                         'optimizer_updates': 0, 'backward_calls': 0, 'gradient_calls': 0,
                         'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
                         'meaning': 'One plausible estimate; no aligned hidden reference or exact identity claim'})
            path = 'images/' + c['id'] + '.png'
            Image.fromarray(output).save(OUT / path)
            stage, metadata = 'stages/' + c['id'] + '.npz', 'metadata/' + c['id'] + '.json'
            np.savez_compressed(OUT / stage, **captured)
            write(OUT / metadata, meta)
            baseline = rgb(ROOT / c['baseline_output'])
            rows.append({'id': c['id'], 'rejected_before_neural': False, 'output': path,
                         'stages': stage, 'metadata': metadata, 'mask_pixels': int(final.sum()),
                         'conditioning_pixels': int(union.sum()), 'empty_bypass': not bool(final.any()),
                         'outside_final_changed_pixels': 0,
                         'changed_estimated_pixels_vs_same_final_baseline': int(np.any(output != baseline, axis=-1)[final].sum())})
            print(json.dumps({'case': c['id'], 'forwards': counts['completion'], 'seconds': round(time.monotonic() - started, 2)}), flush=True)
        after = state_hash(model)
        assert before == after and counts == p['max_forwards'] and len(rows) == 36
        assert sum(row['rejected_before_neural'] for row in rows) == 4 and sum(row.get('empty_bypass', False) for row in rows) == 4
        font, pages = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13), []
        for condition in ['original_photo', 'synthetic_degraded_photo']:
            eligible = [c for c in p['cases'] if not c['rejected'] and c['condition'] == condition]
            for first in range(0, 16, 4):
                page, entries = Image.new('RGB', (1280, 1214), 'white'), []
                draw = ImageDraw.Draw(page)
                for k, label in enumerate(['Input256', 'Final removal unchanged', 'Current-mask baseline', 'Union conditioning only', 'Union-conditioned estimate']):
                    draw.text((k * 256 + 4, 7), label, font=font, fill='black')
                for row, c in enumerate(eligible[first:first + 4]):
                    source = rgb(ROOT / c['input'])
                    cells = [source, overlay(source, binary(ROOT / c['masks']['removal']), (16, 185, 129)),
                             rgb(ROOT / c['baseline_output']), overlay(source, binary(ROOT / c['conditioning']), (240, 140, 32)),
                             rgb(OUT / 'images' / (c['id'] + '.png'))]
                    y = 30 + row * 296
                    for k, cell in enumerate(cells):
                        page.paste(Image.fromarray(cell), (k * 256, y))
                    draw.text((4, y + 260), c['id'] + ' | ' + c['family'], font=font, fill='black')
                    entries.append({'id': c['id'], 'row': row})
                path = 'pages/' + condition + '_' + str(first // 4 + 1) + '.png'
                page.save(OUT / path)
                pages.append({'path': path, 'sha256': sha(OUT / path), 'entries': entries})
        artifacts = {q.relative_to(OUT).as_posix(): sha(q) for folder in ['images', 'stages', 'metadata', 'pages', 'parity']
                     for q in sorted((OUT / folder).rglob('*')) if q.is_file()}
        artifact_bytes = sum((OUT / name).stat().st_size for name in artifacts)
        assert artifact_bytes <= p['artifact_cap_bytes'] and time.monotonic() - started < 600
        write(OUT / 'results.json', {'complete': True, 'protocol_sha256': sha(OUT / 'protocol.json'),
              'seconds': time.monotonic() - started, 'cap_seconds': 600, 'requests': 37, 'rows': rows,
              'forwards': counts, 'state_before': before, 'state_after': after, 'artifacts_sha256': artifacts,
              'artifact_bytes': artifact_bytes, 'pages': pages, 'parity_before_trials': True,
              'optimizer_updates': 0, 'gradient_calls': 0, 'backward_calls': 0, 'new_checkpoint': False,
              'automatic_forwards': 0, 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
              'app_changes': False, 'native_CCTV_or_reserved_final_used': False, 'hidden_metrics': None,
              'visual_review_pending': True, 'independent_final_review': False, 'goal_complete': False})
        print(json.dumps({'complete': True, 'seconds': time.monotonic() - started, 'forwards': counts,
                          'state_unchanged': True, 'artifact_bytes': artifact_bytes}), flush=True)
    except BaseException as exc:
        if not (OUT / 'failure.json').exists():
            write(OUT / 'failure.json', {'complete': False, 'error': repr(exc), 'traceback': traceback.format_exc(),
                  'protocol_sha256': sha(OUT / 'protocol.json'), 'seconds': time.monotonic() - started,
                  'rows_completed': len(rows), 'forwards': counts, 'state_before': before,
                  'optimizer_updates': 0, 'app_changes': False, 'automatic_repeat_permitted': False})
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == '__main__':
    main()

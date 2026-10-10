"""Freeze an input-only conditioning ablation; no model imports or generation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/completion_input_footprints_comparison_v1_r1'
OUT = ROOT / 'outputs/completion_conditioning_union_v1'


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
        result = np.array(im)
    assert set(np.unique(result)) <= {0, 255}
    return result != 0


def overlay(source, support, colour):
    result = source.astype(np.float64)
    result[support] = .55 * result[support] + .45 * np.array(colour)
    return np.floor(result + .5).astype(np.uint8)


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve prior preparation or execution; no automatic repeat'
    parent, results = read(BASE / 'protocol.json'), read(BASE / 'results.json')
    audit = read(BASE / 'independent_saved_output_audit.json')
    review = read(BASE / 'visual_review.json')
    assert results['complete'] and audit['complete'] and results['state_before'] == results['state_after']
    assert review['all32_eligible_outputs_actually_reviewed'] and not review['app_adoption']
    assert results['protocol_sha256'] == audit['protocol_sha256'] == sha(BASE / 'protocol.json')
    bindings = dict(parent['sources_sha256'])
    app = dict(parent['app_preservation_sha256'])
    for name, digest in {**bindings, **app}.items():
        assert sha(ROOT / name) == digest, name
    for name, digest in results['artifacts_sha256'].items():
        assert sha(BASE / name) == digest, name
        bindings[(BASE / name).relative_to(ROOT).as_posix()] = digest
    for name in ['protocol.json', 'results.json', 'external_receipt.json', 'independent_mask_audit.json',
                 'independent_saved_output_audit.json', 'visual_review.json', 'input_visual_review.json']:
        path = BASE / name
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ['pretrained_completion.py', 'completion.py', 'cctv_dgp_pilot.py',
                 'scripts/prepare_completion_conditioning_union_v1.py',
                 'scripts/run_completion_conditioning_union_v1.py',
                 'scripts/supervise_completion_conditioning_union_v1.py',
                 'scripts/audit_completion_conditioning_union_v1.py',
                 'scripts/audit_completion_conditioning_union_v1_inputs.py',
                 'checkpoints/codeformer_inpainting.pth']:
        bindings[name] = sha(ROOT / name)
    assert bindings['checkpoints/codeformer_inpainting.pth'] == 'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
    OUT.mkdir()
    (OUT / 'masks').mkdir()
    (OUT / 'input_pages').mkdir()
    cases, mask_rows = [], []
    for original in parent['cases']:
        c = dict(original)
        if c['rejected']:
            assert c['input_review'] in ('needs_clearer', 'out_of_scope') and 'masks' not in c
            cases.append(c)
            continue
        source = rgb(ROOT / c['input'])
        final = binary(ROOT / c['masks']['removal'])
        old = binary(ROOT / c['reviewed'])
        protected = binary(ROOT / c['masks']['protected'])
        union = old | final
        assert np.array_equal(union & final, final) and not (final & protected).any()
        assert float(union.mean()) < .85, 'Existing pretrained component requires less than85% conditioning mask'
        if not final.any():
            assert not union.any() and protected.all()
        path = OUT / 'masks' / (c['id'] + '_conditioning.png')
        Image.fromarray(union.astype(np.uint8) * 255).save(path)
        c['conditioning'] = path.relative_to(ROOT).as_posix()
        c['baseline_output'] = (BASE / 'images' / (c['id'] + '.png')).relative_to(ROOT).as_posix()
        c['baseline_stages'] = (BASE / 'stages' / (c['id'] + '.npz')).relative_to(ROOT).as_posix()
        c['baseline_metadata'] = (BASE / 'metadata' / (c['id'] + '.json')).relative_to(ROOT).as_posix()
        c['geometry_counts'] = {
            'final_pixels': int(final.sum()), 'conditioning_pixels': int(union.sum()),
            'extra_conditioning_pixels': int((union & ~final).sum()),
            'hidden_protected_context_pixels': int((union & protected).sum()),
            'hidden_outside_face_context_pixels': int((union & ~binary(ROOT / c['masks']['face'])).sum()),
            'delivered_protected_overlap_pixels': 0,
            'visible_context_rgb_sha256': hashlib.sha256(source[~union].tobytes()).hexdigest(),
        }
        cases.append(c)
        mask_rows.append({'id': c['id'], **c['geometry_counts']})
        bindings[c['conditioning']] = sha(path)
    assert len(cases) == 36 and len(mask_rows) == 32
    assert sum(c['rejected'] for c in cases) == 4
    assert sum(not row['final_pixels'] for row in mask_rows) == 4
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    pages = []
    for condition in ['original_photo', 'synthetic_degraded_photo']:
        selected = [c for c in cases if not c['rejected'] and c['condition'] == condition]
        assert len(selected) == 16
        for first in range(0, 16, 4):
            page = Image.new('RGB', (1024, 1214), 'white')
            draw = ImageDraw.Draw(page)
            for k, label in enumerate(['Input256', 'Final removal unchanged', 'Union conditioning only', 'Extra hidden context only']):
                draw.text((k * 256 + 4, 7), label, font=font, fill='black')
            entries = []
            for row, c in enumerate(selected[first:first + 4]):
                source = rgb(ROOT / c['input'])
                final, union = binary(ROOT / c['masks']['removal']), binary(ROOT / c['conditioning'])
                cells = [source, overlay(source, final, (16, 185, 129)),
                         overlay(source, union, (240, 140, 32)), overlay(source, union & ~final, (210, 64, 180))]
                y = 30 + row * 296
                for k, cell in enumerate(cells):
                    page.paste(Image.fromarray(cell), (k * 256, y))
                draw.text((4, y + 260), c['id'] + ' | ' + c['family'], font=font, fill='black')
                draw.text((4, y + 278), 'Final pixels ' + str(c['geometry_counts']['final_pixels']) +
                          '; extra context ' + str(c['geometry_counts']['extra_conditioning_pixels']) +
                          '; protected context hidden ' + str(c['geometry_counts']['hidden_protected_context_pixels']), font=font, fill='black')
                entries.append({'id': c['id'], 'row': row})
            path = OUT / 'input_pages' / (condition + '_' + str(first // 4 + 1) + '.png')
            page.save(path)
            pages.append({'path': path.relative_to(OUT).as_posix(), 'sha256': sha(path), 'entries': entries})
    rejected = [c for c in cases if c['rejected']]
    page = Image.new('RGB', (1024, 324), 'white')
    draw = ImageDraw.Draw(page)
    for k, c in enumerate(rejected):
        page.paste(Image.fromarray(rgb(ROOT / c['input'])), (k * 256, 24))
        draw.text((k * 256 + 4, 286), c['input_review'], font=font, fill='black')
        draw.text((k * 256 + 4, 306), c['base_id'], font=font, fill='black')
    path = OUT / 'input_pages/excluded.png'
    page.save(path)
    pages.append({'path': path.relative_to(OUT).as_posix(), 'sha256': sha(path), 'entries': [{'id': c['id'], 'column': k} for k, c in enumerate(rejected)]})
    draft = {
        'format': 'completion-conditioning-union-draft-v1', 'complete': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'cases': cases,
        'sources_sha256': bindings, 'app_preservation_sha256': app,
        'parent_protocol_sha256': sha(BASE / 'protocol.json'), 'parent_results_sha256': sha(BASE / 'results.json'),
        'parent_model_state': results['state_before'], 'input_pages': pages, 'mask_counts': mask_rows,
        'family_case_counts': dict(Counter(c['family'] for c in cases if not c['rejected'])),
        'hypothesis': 'Conditioning on covering pixels outside the tighter final removal area can contribute to generated remnants; hide old|new support from the frozen component while keeping delivered new support exactly fixed.',
        'causal_limit': 'Changing context cannot repair copied fragments outside final support. Conditioning may hide clear frames or ordinary hair; final copying must preserve these exactly. Union support is approximate historical input annotation, not covering truth.',
        'conditioning_rule': 'binary union of prior reviewed removal mask and newer input-only reviewed removal mask; no dilation, new annotation or output-driven geometry',
        'delivered_rule': 'same frozen newer reviewed mask and two-pixel clipped margin; no change to final pixels or app',
        'comparison': 'Cached same-final-mask baseline versus one union-conditioning estimate; four exact controls and four unchanged pre-neural input exclusions',
        'parity_case': '00_cloth_mask_native', 'parity_rule': 'one fresh current-mask Off call must exactly match its cached PNG and saved internal512/completion256 arrays before union trials',
        'max_forwards': {'completion': 29, 'internal512': 29, 'DGP': 0, 'detector': 0},
        'expected_trial_forwards': 28, 'max_requests': 37, 'expected_eligible': 32,
        'expected_rejections': 4, 'expected_empty_bypasses': 4,
        'cap_seconds': 600, 'external_timeout_seconds': 630, 'artifact_cap_bytes': 268435456,
        'seed': parent['seed'], 'search_trials_per_case': 1, 'device': 'cpu', 'restoration': 'off',
        'internal_resolution': 512, 'delivered_resolution': 256,
        'completion': 'same frozen official CodeFormer inpainting checkpoint; w=1, adain=False; visible-normalized-resize-v1',
        'display': 'same floor(float32*255) and uploaded-visible-input grayscale policy using FINAL support only',
        'saved_stages': 'actual neural input512, raw internal512 and composited-conditioning256 before PNG; parity arrays and exact32 delivered images',
        'prospective_audit': 'all union masks, actual neural input math, exact raw512-to256 composition, visible/protected bytes, all hashes, all case order/page cells, same state/count/runtime and preserved exclusions',
        'quality_criteria': parent['quality_criteria'],
        'source_scope': parent['source_scope'], 'exposed_development_photos': True,
        'original_photo_assistance_reused_for_synthetic_pair': True, 'unknown_pretraining_overlap': True,
        'automatic': 'saved automatic proposals remain separate; no new automatic generation or qualification',
        'hidden_ground_truth': None, 'hidden_metrics': None, 'native_CCTV_or_reserved_final_used': False,
        'new_checkpoint': False, 'gradient_calls': 0, 'optimizer_updates': 0, 'backward_calls': 0,
        'app_changes': False, 'quality_qualification': False, 'independent_final_review': False, 'goal_complete': False,
        'model_forwards_in_preparation': 0, 'seconds': time.monotonic() - started,
        'input_review_before_freezing_required': True,
    }
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    write(OUT / 'draft.json', draft)
    print(json.dumps({'complete': True, 'cases': 36, 'input_pages': len(pages),
                      'changed_conditioning_cases': sum(r['extra_conditioning_pixels'] > 0 for r in mask_rows),
                      'max_extra_pixels': max(r['extra_conditioning_pixels'] for r in mask_rows),
                      'seconds': draft['seconds'], 'neural_calls': 0}), flush=True)


if __name__ == '__main__':
    main()

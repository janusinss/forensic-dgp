"""Post-hoc saved-pixel attribution: copied support versus estimated support.

No models, gradients, mask edits, threshold fitting or new image acquisition.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_pixel_support_v1_r1'
BASE = ROOT / 'outputs/dgp_app_covering_review_v3'
CONTEXT = ROOT / 'outputs/completion_margin_full_v3'
ROIS = [
    {'base_id': 'val_18_hand_eyes', 'name': 'lower_right_hand_join', 'xyxy': [139, 137, 185, 173],
     'reason': 'Previously noted lower-finger/cheek join; rectangle includes mixed support'},
    {'base_id': 'val_336_scarf_gloves', 'name': 'left_scarf_join', 'xyxy': [26, 145, 64, 208],
     'reason': 'Previously noted peripheral scarf/cheek join; facial boundary is not annotated truth'},
    {'base_id': 'val_336_scarf_gloves', 'name': 'central_chin_estimate', 'xyxy': [70, 205, 191, 252],
     'reason': 'Previously noted red/beard-like texture in the estimated lower face; hidden appearance unknown'},
    {'base_id': 'val_362_hair_eye', 'name': 'estimated_eye_context', 'xyxy': [137, 94, 185, 136],
     'reason': 'Previously noted eye/gaze coherence issue; no aligned hidden eye reference'},
]


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def pixels(path, mask=False):
    with Image.open(path) as image: array = np.asarray(image.convert('L' if mask else 'RGB')).copy()
    assert array.shape == ((256, 256) if mask else (256, 256, 3))
    if mask:
        assert np.isin(array, (0, 255)).all()
        return array == 255
    return array


def prepare():
    assert not OUT.exists(), 'Preserve any previous diagnostic'
    plan, results = read(BASE / 'plan.json'), read(BASE / 'results.json')
    proof, visual = read(BASE / 'saved_output_audit.json'), read(BASE / 'visual_review.json')
    context, context_proof = read(CONTEXT / 'results.json'), read(CONTEXT / 'saved_output_audit.json')
    assert all(d['complete'] for d in [results, proof, visual, context, context_proof])
    assert results['plan_sha256'] == sha(BASE / 'plan.json') and proof['results_sha256'] == sha(BASE / 'results.json')
    assert results['state_before'] == results['state_after']
    assert context['state_before'] == context['state_after'] == results['state_before']['completion']
    assert len(plan['cases']) == len(results['rows']) == len(context['rows']) == 36
    assets = dict(plan['sources_sha256'])
    for name, digest in assets.items(): assert sha(ROOT / name) == digest, name
    files = [BASE / n for n in ['plan.json', 'results.json', 'saved_output_audit.json', 'visual_review.json']]
    files += [CONTEXT / n for n in ['plan.json', 'results.json', 'saved_output_audit.json', 'visual_review.json']]
    files += [Path(__file__), ROOT / 'scripts/audit_completion_pixel_support_v1_r1.py',
              ROOT / 'outputs/completion_pixel_support_v1_preparation_failure/failure.json',
              ROOT / 'outputs/completion_pixel_support_v1_preparation_failure/diagnostic_source_before_correction.py',
              ROOT / 'outputs/dgp_mask_review_race_fix_v1/milestone.json',
              ROOT / 'outputs/dgp_mask_review_race_fix_v1/independent_closure_audit.json',
              ROOT / 'static/face_workflow.js']
    files += [ROOT / name for name in [
        'outputs/completion_pixel_support_v1/plan.json',
        'outputs/completion_pixel_support_v1/execution.json',
        'outputs/completion_pixel_support_v1/analysis_failure.json',
        'outputs/completion_pixel_support_v1/diagnostic_source_at_analysis_failure.py',
        'scripts/diagnose_completion_pixel_support_v1.py',
        'scripts/audit_completion_pixel_support_v1.py']]
    cases = []
    for case in plan['cases']:
        row = next(r for r in results['rows'] if r['id'] == case['id'])
        old_context = next(r for r in context['rows'] if r['id'] == case['id'])
        rejected = row['assisted']['off']['rejected']
        assert rejected == old_context['rejected'] == (case['input_review'] != 'usable')
        item = {'id': case['id'], 'family': case['family'], 'condition': case['condition'],
                'input': case['input'], 'reviewed': case['reviewed'], 'rejected': rejected,
                'operator_input_review': case['input_review'], 'legacy_expected_rejection': case['expected_rejection']}
        if not rejected:
            off = BASE / row['assisted']['off']['output']
            stage = BASE / ('stages/' + case['id'] + '_off.npz')
            metadata = BASE / ('metadata/' + case['id'] + '_off.json')
            old_output = CONTEXT / old_context['outputs']['off']
            for path in [off, stage, metadata]:
                assert sha(path) == results['artifacts_sha256'][path.relative_to(BASE).as_posix()]
            assert sha(old_output) == context['artifacts_sha256'][old_output.relative_to(CONTEXT).as_posix()]
            files += [off, stage, metadata, old_output]
            item.update(output=off.relative_to(ROOT).as_posix(), stages=stage.relative_to(ROOT).as_posix(),
                        metadata=metadata.relative_to(ROOT).as_posix(), context6=old_output.relative_to(ROOT).as_posix())
        cases.append(item)
    assert sum(not c['rejected'] for c in cases) == 32
    assets.update({f.relative_to(ROOT).as_posix(): sha(f) for f in files})
    OUT.mkdir()
    write(OUT / 'plan.json', {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
          'frozen_before_saved_pixel_recount': True, 'ROIs_selected_after_previous_development_output_review': True,
          'ROIs_are_diagnostic_rectangles_not_segmentation_truth': True, 'cases': cases, 'ROIs': ROIS,
          'sources_sha256': assets, 'maximum_case_recounts': 32, 'maximum_ROI_recounts': 8, 'cap_seconds': 120,
          'criteria': ['Reconstruct every eligible Off PNG from saved floats and unchanged composition',
                       'Attribute selected pixels to reviewed estimate support or exactly retained uploaded support',
                       'Compare the earlier context6 output on the same support; no claim of hidden accuracy',
                       'Keep automatic detector quality separate and all earlier failures binding'],
          'official_inpainting_reference': {'url': 'https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/inference_inpainting.py',
              'revision': 'b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b', 'internal_resolution': 512,
              'requires_aligned_crop': True, 'w': 1, 'adain': False,
              'role': 'Primary source confirms adapter parameters; no settings change is justified by a mismatch'},
          'source_scope': 'Previously exposed development photographs; legacy native means original photo, not CCTV',
          'scope': 'Post-hoc processing provenance diagnostic, not independent quality evaluation',
          'model_forwards': 0, 'training': False, 'mask_edits': False, 'app_changes': False,
          'reserved_final_used': False, 'hidden_reference_metrics': None, 'goal_complete': False})
    print(json.dumps({'prepared': True, 'cases': 36, 'eligible_saved_cases': 32, 'ROIs': 8, 'model_calls': 0}))


def analyze():
    assert not (OUT / 'results.json').exists() and not (OUT / 'execution.json').exists()
    p = read(OUT / 'plan.json')
    for name, digest in p['sources_sha256'].items(): assert sha(ROOT / name) == digest, name
    assert p['ROIs'] == ROIS and p['model_forwards'] == 0 and not p['mask_edits']
    start = time.monotonic()
    write(OUT / 'execution.json', {'plan_sha256': sha(OUT / 'plan.json'), 'mode': 'saved arrays only',
          'torch_imported': 'torch' in sys.modules, 'model_forwards': 0, 'optimizer_updates': 0, 'cap_seconds': 120})
    rows, regions = [], []
    for case in p['cases']:
        assert time.monotonic() - start < p['cap_seconds']
        if case['rejected']:
            rows.append({'id': case['id'], 'rejected_before_previous_inference': True}); continue
        source, mask = pixels(ROOT / case['input']), pixels(ROOT / case['reviewed'], True)
        output, context = pixels(ROOT / case['output']), pixels(ROOT / case['context6'])
        metadata = read(ROOT / case['metadata'])
        expected = source.copy()
        with np.load(ROOT / case['stages'], allow_pickle=False) as arrays:
            if mask.any():
                assert arrays.files == ['completion']
                raw = arrays['completion'].copy()
                assert raw.shape == (1, 3, 256, 256) and raw.dtype == np.float32 and np.isfinite(raw).all()
                assert raw.min() >= 0 and raw.max() <= 1
                generated = np.floor(raw[0].transpose(1, 2, 0) * np.float32(255)).astype(np.uint8)
                if metadata['display_processing']['colour_policy']['applied']:
                    generated = np.repeat(cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1)
                expected[mask] = generated[mask]
            else:
                assert not arrays.files and metadata['completion'] == {'bypassed': 'empty mask'}
        np.testing.assert_array_equal(expected, output)
        np.testing.assert_array_equal(output[~mask], source[~mask])
        np.testing.assert_array_equal(context[~mask], source[~mask])
        changed = np.any(output != source, axis=-1)
        rows.append({'id': case['id'], 'family': case['family'], 'condition': case['condition'],
            'source_support_pixels': int((~mask).sum()), 'estimate_support_pixels': int(mask.sum()),
            'source_changed_pixels': int((changed & ~mask).sum()),
            'current_output_composition_or_bypass_exact': True, 'completion_bypassed': not bool(mask.any()), 'context6_source_support_exact': True})
        for region in p['ROIs']:
            if case['id'] not in [region['base_id'] + '_native', region['base_id'] + '_degraded']: continue
            x1, y1, x2, y2 = region['xyxy']; selection = mask[y1:y2, x1:x2]
            current = output[y1:y2, x1:x2]; original = source[y1:y2, x1:x2]
            old_context = context[y1:y2, x1:x2]
            regions.append({'id': case['id'], 'name': region['name'], 'xyxy': region['xyxy'],
                'rectangle_pixels': int(selection.size), 'estimate_support_pixels': int(selection.sum()),
                'source_support_pixels': int((~selection).sum()), 'estimate_fraction': float(selection.mean()),
                'source_changed_pixels': int(np.any(current[~selection] != original[~selection], axis=-1).sum()),
                'estimated_pixels_different_from_uploaded_covering': int(np.any(current[selection] != original[selection], axis=-1).sum()),
                'context6_changed_estimated_pixels': int(np.any(old_context[selection] != current[selection], axis=-1).sum()),
                'note': region['reason'], 'hidden_ground_truth': None})
    assert len(rows) == 36 and sum('family' in r for r in rows) == 32 and len(regions) == 8
    assert not any(r['source_changed_pixels'] for r in rows if 'family' in r)
    assert sum(r['completion_bypassed'] for r in rows if 'family' in r) == 4
    for condition, suffix in [('original_photo', '_native'), ('synthetic_degraded_photo', '_degraded')]:
        sheet = Image.new('RGB', (1320, 1192), '#171b20'); draw = ImageDraw.Draw(sheet)
        headings = ['Uploaded photo', 'Reviewed removal area', 'Current Off estimate', 'Orange=estimate / blue=source', 'Earlier context6 estimate']
        for col, title in enumerate(headings): draw.text((col * 264 + 4, 8), title, fill='white')
        for index, region in enumerate(p['ROIs']):
            case = next(c for c in p['cases'] if c['id'] == region['base_id'] + suffix)
            source, mask = pixels(ROOT / case['input']), pixels(ROOT / case['reviewed'], True)
            output, context = pixels(ROOT / case['output']), pixels(ROOT / case['context6'])
            overlay = source.copy(); overlay[mask] = (source[mask].astype(np.float32) * .45 + np.array([255, 70, 40]) * .55).astype(np.uint8)
            support = np.empty_like(source); support[mask] = [231, 133, 49]; support[~mask] = [40, 124, 193]
            yy = 28 + index * 291
            draw.text((4, yy + 3), case['id'] + ' / ' + region['name'], fill='white')
            for col, array in enumerate([source, overlay, output, support, context]):
                tile = Image.fromarray(array); mark = ImageDraw.Draw(tile)
                x1, y1, x2, y2 = region['xyxy']; mark.rectangle((x1, y1, x2 - 1, y2 - 1), outline='white', width=1)
                sheet.paste(tile, (col * 264 + 4, yy + 26))
        sheet.save(OUT / (condition + '_support.png'))
    for name, digest in p['sources_sha256'].items(): assert sha(ROOT / name) == digest, name
    assert 'torch' not in sys.modules
    artifacts = {f.relative_to(OUT).as_posix(): sha(f) for f in OUT.iterdir() if f.is_file()}
    write(OUT / 'results.json', {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'rows': rows,
          'regions': regions, 'artifacts_sha256': artifacts, 'seconds': time.monotonic() - start,
          'cap_seconds': p['cap_seconds'], 'eligible_cases_recounted': 32, 'ROI_recounts': 8,
          'raw_completion_compositions': 28, 'empty_mask_exact_bypasses': 4,
          'model_forwards': 0, 'local_gradients': 0, 'optimizer_updates': 0, 'mask_edits': False,
          'app_changes': False, 'quality_qualification': False, 'independent_final_review': False,
          'native_CCTV_or_reserved_final_used': False, 'hidden_reference_metrics': None, 'goal_complete': False})
    print(json.dumps({'complete': True, 'eligible_cases': 32, 'regions': regions, 'seconds': time.monotonic() - start}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'analyze'])
    args = parser.parse_args(); prepare() if args.mode == 'prepare' else analyze()

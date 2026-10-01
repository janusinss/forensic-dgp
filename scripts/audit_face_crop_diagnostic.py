"""Recount the fixed crop diagnostic from saved masks; no inference or fitting."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL_SHA = 'de64663b607f34f65f105029d4862d29cc78e799c8b993ff10ad36d9280c31a3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def crop_roi(case):
    shape = case['source_shape']; bounds = case['bounds']
    require(len(shape) == 2 and all(type(v) is int and v > 0 for v in shape)
            and len(bounds) == 4 and all(type(v) is int for v in bounds)
            and type(case['cropped']) is bool, 'Invalid saved crop geometry')
    height, width = shape
    left, top, right, bottom = bounds
    if not case['cropped']:
        require(bounds == [0, 0, width, height], 'Fallback geometry must be identity')
        return np.ones(shape, dtype=bool)
    require(right-left == bottom-top and 32 <= right-left < max(shape),
            'Crop must be a square with the declared size bounds')
    yy, xx = np.mgrid[:height, :width]
    roi = (left <= xx) & (xx < right) & (top <= yy) & (yy < bottom)
    require(roi.any(), 'Crop has no original image support')
    return roi


def check_composition(prediction, original, case):
    require(all(isinstance(v, np.ndarray) and v.dtype == np.bool_ and v.ndim == 2
                for v in (prediction, original))
            and prediction.shape == original.shape == tuple(case['source_shape']),
            'Require matching binary 2D masks and source geometry')
    roi = crop_roi(case)
    differences = prediction != original
    outside = int((differences & ~roi).sum())
    require(outside == 0, 'Mapped prediction changed pixels outside crop ROI')
    changed = int(differences.sum())
    require(case['cropped'] or changed == 0, 'Fallback prediction differs from original')
    return {'outside_roi_differences': outside, 'changed_pixels': changed,
            'roi_pixels': int(roi.sum())}


def count_scores(predictions, targets, valid):
    """Independent integer totals on the original target support."""
    require(set(predictions) == set(targets) == set(valid), 'Metric membership differs')
    total = Counter(); records = []
    for name in sorted(targets):
        p, t, v = predictions[name], targets[name], valid[name]
        require(p.shape == t.shape == v.shape and all(a.dtype == np.bool_ for a in (p, t, v))
                and v.any() and not (t & ~v).any(), 'Invalid target or valid support')
        row = {'id': name, 'tp': int((p & t & v).sum()), 'fp': int((p & ~t & v).sum()),
               'fn': int((~p & t & v).sum()), 'visible': int((~t & v).sum()),
               'ignored_positive_pixels': int((p & ~v).sum()),
               'covered_cases': int(t.any()), 'negative_cases': int(not t.any()),
               'empty_mask_cases': int(t.any() and not (p & v).any()),
               'negative_false_positive_cases': int(not t.any() and (p & v).any())}
        total.update({k: value for k, value in row.items() if k != 'id'})
        records.append(row)
    scores = {k: total[k] for k in ('covered_cases', 'negative_cases', 'empty_mask_cases',
                                    'negative_false_positive_cases')}
    scores.update(iou=total['tp']/max(1, total['tp']+total['fp']+total['fn']),
                  missed_fraction=total['fn']/max(1, total['tp']+total['fn']),
                  visible_false_positive=total['fp']/max(1, total['visible']))
    return scores, dict(total), records


def main():
    import cv2
    import torch
    from PIL import Image
    from detector_training import load_manifest, ReviewedMasks
    from scripts.package_face_occlusion_vm import sha
    from scripts.evaluate_coverage_results import binary
    from scripts.run_face_crop_diagnostic import (
        PROTOCOL, OUTPUT, TRAINING, DATA_PATH, PIXELS_SHA, AUDIT_OUTPUT,
        EXTRACTION, PREFIX, protocol_value, training_members, diagnostic_signal,
    )

    destination = OUTPUT/'verification.json'
    require(not destination.exists(), 'Preserve completed independent verification')
    expected = protocol_value()
    require(sha(PROTOCOL) == PROTOCOL_SHA and json.loads(PROTOCOL.read_text()) == expected,
            'Frozen protocol, executable code, inputs or model bindings changed')
    report_path = OUTPUT/'results.json'; cases_path = OUTPUT/'cases.json'
    report = json.loads(report_path.read_text()); cases = json.loads(cases_path.read_text())
    require(report['complete'] is True and report['protocol_sha256'] == PROTOCOL_SHA
            and report['code_sha256'] == expected['code']['scripts/run_face_crop_diagnostic.py']
            and report['device'] == 'cpu' and report['held_out_inference_cases'] == 0
            and report['optimizer_constructed'] is False and report['optimizer_updates_locally'] == 0
            and report['promoted'] is False and set(report['models']) == {'source30', 'reflective42'},
            'Run completion or scope differs')
    names = [f'{domain}/{i:04}' for domain, count in
             (('training_real', 73), ('training_reflection', 280)) for i in range(count)]
    require(isinstance(cases, list) and [r['id'] for r in cases] == names,
            'Saved case order, identity or membership differs')
    metadata = {r['id']: r for r in cases}
    counts = Counter(r['reason'] for r in cases)
    require(dict(counts) == report['crop_counts']
            and all(r['cropped'] == (r['reason'] == 'single_reliable_face') for r in cases),
            'Crop/fallback counts differ')
    rows = training_members(load_manifest(TRAINING)); real = ReviewedMasks(rows, 256)
    cache_path = ROOT/DATA_PATH/'pixels.pth'
    require(sha(cache_path) == PIXELS_SHA, 'Training fixture bytes changed')
    pixels = torch.load(cache_path, map_location='cpu', weights_only=True)['pixels']
    require(set(pixels) == set(range(280)), 'Fixture membership differs')
    old = {r['image_sha256']: r for r in
           load_manifest(ROOT/'dataset/detector_expanded_review_v2/manifest.json')}
    targets = {}; valid = {}; parents = {}
    for name in names:
        domain, position = name.split('/'); i = int(position); case = metadata[name]
        if domain == 'training_real':
            image, mask = real[i]
            rgb = np.rint(image.permute(1, 2, 0).numpy()*255).astype('uint8')
            target = mask[0].numpy().astype(bool); support = np.ones(target.shape, dtype=bool)
            if rows[i].get('glare_stratum') == 'strong_lens_reflection':
                parents[name] = ReviewedMasks([old[rows[i]['image_sha256']]], 256)[0][1][0].numpy().astype(bool)
        else:
            item = pixels[i]; rgb = item['input'].permute(1, 2, 0).numpy()
            target = item['mask'][0].numpy().astype(bool); support = item['valid'][0].numpy().astype(bool)
        require(case['source_shape'] == list(target.shape)
                and hashlib.sha256(rgb.tobytes()).hexdigest() == case['input_byte_sha256'],
                'Saved source image bytes or dimensions differ')
        roi = crop_roi(case)
        left, top, right, bottom = case['bounds']
        if case['cropped']:
            scale = 255/(right-left-1)
            affine = np.array([[scale, 0., -scale*left], [0., scale, -scale*top]])
            require(.6 <= case['face_confidence'] <= 1, 'Reliable-face confidence differs')
            mapped = cv2.warpAffine(target.astype('uint8'), affine, (256, 256),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            affine = np.array([[1., 0., 0.], [0., 1., 0.]])
            mapped = target
        require(np.allclose(case['affine'], affine, rtol=0, atol=1e-12)
                and np.allclose(case['inverse'], cv2.invertAffineTransform(affine), rtol=0, atol=1e-12)
                and case['roi_pixels'] == int(roi.sum()) and case['target_pixels'] == int(target.sum())
                and case['target_pixels_outside_roi'] == int((target & ~roi).sum())
                and case['cropped_target_pixels'] == int(mapped.sum()), 'Saved target/affine counts differ')
        targets[name], valid[name] = target, support
    require(set(parents) == {'training_real/0015', 'training_real/0046'}, 'Lens-only cohort changed')
    preview_ids = list(parents) + [f'training_real/{i:04}' for i, r in enumerate(rows)
                                 if r['kind'] == 'uncovered'][:4]
    require(report['preview_ids'] == preview_ids, 'Fixed preview membership differs')
    with Image.open(OUTPUT/'preview.png') as preview:
        require(preview.size == (1024, len(preview_ids)*148), 'Fixed preview dimensions differ')
        preview.verify()
    members = json.loads((AUDIT_OUTPUT/'members.json').read_text())
    result = {'complete': False, 'script_sha256': sha(__file__), 'protocol_sha256': sha(PROTOCOL),
              'run_report_sha256': sha(report_path), 'cases_sha256': sha(cases_path),
              'preview_sha256': sha(OUTPUT/'preview.png'),
              'test_sha256': sha(ROOT/'tests/test_face_crop_return_audit.py'),
              'models': {}, 'mask_sha256': {}, 'held_out_inference_cases': 0,
              'new_model_forward_passes': 0, 'optimizer_updates_locally': 0, 'promoted': False}
    independently_counted = {}
    for label, folder in (('source30', 'initial_masks'), ('reflective42', 'reflective/epoch_42_masks')):
        masks_root = OUTPUT/label
        require({p.relative_to(masks_root).as_posix() for p in masks_root.rglob('*') if p.is_file()}
                == {n+'.png' for n in names}, 'Missing/extra exported crop masks')
        ordinary = {}; cropped = {}; composition = []
        for name in names:
            original_path = EXTRACTION/PREFIX/folder/(name+'.png')
            require(sha(original_path) == members[original_path.relative_to(EXTRACTION).as_posix()],
                    'Previously audited original prediction changed')
            crop_path = masks_root/(name+'.png')
            ordinary[name], cropped[name] = binary(original_path), binary(crop_path)
            composition.append({'id': name, **check_composition(cropped[name], ordinary[name], metadata[name])})
            result['mask_sha256'][crop_path.relative_to(OUTPUT).as_posix()] = sha(crop_path)
        modes = {}
        for mode, prediction in (('ordinary', ordinary), ('face_crop', cropped)):
            scores = {}; integers = {}
            for domain in ('training_real', 'training_reflection'):
                select = lambda values: {n: v for n, v in values.items() if n.startswith(domain+'/')}
                recounted, totals, records = count_scores(select(prediction), select(targets), select(valid))
                logged = report['models'][label][mode][domain]
                if domain == 'training_reflection':
                    recounted.update(ignored_positive_pixels=totals['ignored_positive_pixels'],
                                     records=[{**{k: r[k] for k in ('tp', 'fp', 'fn', 'visible', 'ignored_positive_pixels')},
                                               'case_id': int(r['id'].split('/')[1])} for r in records],
                                     scope='Training-only fixture diagnostics; excluded from selection')
                require(recounted == logged, 'Independent saved-mask metrics differ: '+label+'/'+mode+'/'+domain)
                scores[domain], integers[domain] = recounted, totals
            lens = {}
            for name, parent in parents.items():
                p, t = prediction[name], targets[name]
                require(not (parent & ~t).any(), 'V3 removed V2 mask pixels')
                added = t & ~parent; count = int(added.sum()); recovered = int((p & added).sum())
                require(count > 0, 'Missing reviewed reflection addition')
                lens[name] = {'added_glare_pixels': count, 'recovered_added_glare_pixels': recovered,
                              'missed_added_glare_pixels': count-recovered, 'glare_recall': recovered/count,
                              'original_mask_pixels': int(parent.sum()),
                              'original_mask_recovered_pixels': int((p & parent).sum()),
                              'whole_mask_false_positive_pixels': int((p & ~t).sum())}
            scores.update(lens_only=lens,
                          lens_recovered_pixels=sum(r['recovered_added_glare_pixels'] for r in lens.values()),
                          lens_target_pixels=sum(r['added_glare_pixels'] for r in lens.values()))
            require(scores == report['models'][label][mode], 'Independent lens-only recount differs')
            modes[mode] = scores; modes[mode+'_integer_totals'] = integers
        state = report['models'][label]
        require(state['cases'] == 353 and state['forward_count'] == sum(r['cropped'] for r in cases)
                and state['model_state_unchanged'] is True and state['optimizer_constructed'] is False
                and state['optimizer_updates_locally'] == 0, 'Recorded inference/state policy differs')
        independently_counted[label] = modes
        result['models'][label] = {'composition': composition,
                                  'outside_roi_differences': sum(r['outside_roi_differences'] for r in composition),
                                  'fallback_masks_exact': True,
                                  'metrics_exact': True, 'integer_totals': {k: v for k, v in modes.items() if k.endswith('_integer_totals')},
                                  'model_state_invariance': 'Executed runner check; this audit does not load models'}
    signal = diagnostic_signal(independently_counted['reflective42']['ordinary'],
                               independently_counted['reflective42']['face_crop'])
    require(signal == report['training_signal'], 'Fixed advancement decision differs')
    result.update(complete=True, saved_crop_masks_recounted=len(result['mask_sha256']),
                  original_masks_hash_checked=706, cases_geometry_checked=353,
                  crop_counts=dict(counts), training_signal=signal,
                  scope='Independent saved-mask/geometry recount only; no model, face detector or optimizer execution')
    with destination.open('x', encoding='utf-8', newline='\n') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('complete', 'saved_crop_masks_recounted', 'cases_geometry_checked',
                                            'training_signal', 'new_model_forward_passes', 'optimizer_updates_locally', 'promoted')}), flush=True)


if __name__ == '__main__':
    main()

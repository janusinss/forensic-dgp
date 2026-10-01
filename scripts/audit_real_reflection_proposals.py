"""Independent native polygon, RGB, mask and supervision-support recount."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT/'outputs/real_reflection_proposals_v4_refined'
MANIFEST_SHA = '4d75620863314e7ff26e9c3a4abf3e28a99be46744f7f08730e9af697b9ffe91'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_support(mask, valid, source, unknown):
    require(all(isinstance(a, np.ndarray) and a.ndim == 2 and a.dtype == np.bool_
                for a in (mask, valid, source, unknown))
            and mask.shape == valid.shape == source.shape == unknown.shape,
            'Matching binary target and supervision support required')
    require(np.array_equal(valid, source & ~unknown) and valid.any()
            and not (mask & ~valid).any(), 'Padding or unknown pixels incorrectly supervised')


def check_pending(row):
    require(row['status'] == 'pending_native_review'
            and all(row.get(key) is None for key in ('mask', 'valid', 'image', 'files')),
            'Pending source incorrectly materialized as a supervised/clear target')


def polygon_mask(polygons, width, height):
    canvas = Image.new('L', (width, height))
    for points in polygons:
        a = np.asarray(points, np.float64)
        require(a.shape[1:] == (2,) and len(a) >= 3 and np.isfinite(a).all()
                and np.equal(a, a.astype('int64')).all() and np.all(a >= 0)
                and np.all(a[:, 0] < width) and np.all(a[:, 1] < height), 'Invalid native polygon')
        ImageDraw.Draw(canvas).polygon([tuple(int(v) for v in p) for p in a], fill=255)
    return np.array(canvas).astype(bool)


def binary(path):
    with Image.open(path) as image:
        a = np.array(image)
    require(a.ndim == 2 and np.isin(a, [0, 255]).all(), 'Saved label/support is not binary grayscale')
    return a.astype(bool)


def main():
    destination = OUTPUT/'verification.json'
    require(not destination.exists(), 'Preserve independent verification')
    require(sha(OUTPUT/'manifest.json') == MANIFEST_SHA, 'Refined proposal manifest changed')
    data = json.loads((OUTPUT/'manifest.json').read_text())
    parent_path = ROOT/'outputs/real_reflection_proposals_v4/manifest.json'
    require(sha(parent_path) == data['parent_proposal_sha256'], 'Initial proposal lineage changed')
    parent = json.loads(parent_path.read_text())
    require(data['complete'] is True and data['training_enabled'] is False
            and data['training_recipe_ready'] is False and data['new_approved_masks'] == 0
            and data['model_forward_passes'] == data['optimizer_updates_locally'] == 0
            and data['promoted'] is False and all(sha(ROOT/p) == h for p, h in data['code_sha256'].items())
            and sha(ROOT/'OCCLUSION_POLICY_V3.md') == data['policy_sha256'], 'Executed source/scope bindings differ')
    review_path = ROOT/'outputs/real_reflection_review_v4/manifest.json'
    visual_path = ROOT/'outputs/real_reflection_review_v4/visual_review.json'
    require(sha(review_path) == data['source_review_sha256'] and sha(visual_path) == data['source_visual_review_sha256'],
            'Qualified source review changed')
    review = json.loads(review_path.read_text())
    require(all(sha(ROOT/p) == h for p, h in review['input_sha256'].items())
            and review['overlap_flagged_sources'] == 0 and not review['within_candidate_pairs'], 'Source qualification inputs changed')
    rows = data['records']; previous = {r['index']: r for r in parent['records']}
    require(len(rows) == len({r['index'] for r in rows}) == 24
            and Counter(r['status'] for r in rows) == data['status_counts']
            == {'pending_native_review': 14, 'proposed_covering': 4, 'proposed_clear': 6}, 'Proposal cohort differs')
    expected_files = {'manifest.json', *data['preview_sha256']}
    counts = []; hashes = {}; unchanged = 0; pending = 0
    for row in rows:
        require(row['split'] == 'train' and row['training_enabled'] is False and row['reviewed'] is False
                and row['pixel_annotation_approved'] is False
                and sha(ROOT/row['source']) == row['source_sha256']
                and sha(ROOT/row['native_review_image']) == row['native_review_sha256'], 'Source/member or review state changed')
        if row['status'] == 'pending_native_review':
            check_pending(row); require(row == previous[row['index']], 'Unreviewed source changed during refinement')
            pending += 1; continue
        require(row['status'] in ('proposed_clear', 'proposed_covering')
                and set(row['files']) == {'images', 'masks', 'valid', 'source_valid', 'native_masks', 'native_valid'},
                'Proposal artifact membership differs')
        for artifact in row['files'].values():
            path = (OUTPUT/artifact['path']).resolve()
            require(path.is_relative_to(OUTPUT) and sha(path) == artifact['sha256'], 'Saved proposal bytes changed')
            expected_files.add(artifact['path']); hashes[artifact['path']] = sha(path)
        if row['index'] != 374:
            require(row == previous[row['index']], 'Refinement changed an undeclared source')
            unchanged += 1
        with Image.open(ROOT/row['native_review_image']) as image:
            native_rgb = np.array(image.convert('RGB'))
        height, width = native_rgb.shape[:2]
        require(row['native_size'] == [width, height], 'Oriented native size differs')
        target = polygon_mask(row['native_polygons'], width, height)
        unknown = polygon_mask(row['unknown_native_polygons'], width, height)
        require(not (target & unknown).any(), 'Native known target overlaps unknown region')
        require(np.array_equal(binary(OUTPUT/row['files']['native_masks']['path']), target)
                and np.array_equal(binary(OUTPUT/row['files']['native_valid']['path']), ~unknown), 'Native raster differs')
        scale = 255/(max(width, height)-1)
        affine = np.array([[scale, 0., (255-scale*(width-1))/2],
                           [0., scale, (255-scale*(height-1))/2]])
        require(np.allclose(row['affine'], affine, rtol=0, atol=1e-12) and row['scale'] == scale,
                'Uniform source geometry differs')
        kwargs = {'borderMode': cv2.BORDER_CONSTANT, 'borderValue': 0}
        source_coverage = cv2.warpAffine(np.ones((height, width), np.float32), affine, (256, 256),
                                         flags=cv2.INTER_LINEAR, **kwargs)
        annotated_coverage = cv2.warpAffine((~unknown).astype('float32'), affine, (256, 256),
                                            flags=cv2.INTER_LINEAR, **kwargs)
        source = source_coverage >= 1-1e-6
        unreviewed = annotated_coverage < 1-1e-6
        expected_valid = source & ~unreviewed
        raw_target = cv2.warpAffine(target.astype('uint8'), affine, (256, 256), flags=cv2.INTER_NEAREST, **kwargs).astype(bool)
        mask, valid, source_saved = (binary(OUTPUT/row['files'][key]['path']) for key in ('masks', 'valid', 'source_valid'))
        require(np.array_equal(source_saved, source) and np.array_equal(valid, expected_valid)
                and np.array_equal(mask, raw_target & expected_valid), 'Original-coordinate mapped target/support differs')
        check_support(mask, valid, source, unreviewed)
        expected_rgb = cv2.warpAffine(native_rgb, affine, (256, 256), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(96, 96, 96))
        expected_rgb[~source] = 96
        with Image.open(OUTPUT/row['image']) as image:
            saved_rgb = np.array(image.convert('RGB'))
        require(np.array_equal(expected_rgb, saved_rgb), 'Input RGB altered or ignored source context erased')
        numerical = {'native_positive_pixels': int(target.sum()), 'native_ignored_pixels': int(unknown.sum()),
                     'positive_pixels': int(mask.sum()), 'supervised_pixels': int(valid.sum()),
                     'source_support_pixels': int(source.sum()),
                     'mapped_positive_pixels_outside_source_support': int((raw_target & ~source).sum())}
        require(all(row[k] == v for k, v in numerical.items())
                and bool(target.any()) == (row['status'] == 'proposed_covering')
                and row['uncovered_face_reference'] is None and row['high_resolution_reference'] is False,
                'Counts, proposed kind or paired-target claim differs')
        counts.append({'index': row['index'], **numerical,
                       'unknown_source_context_preserved': True, 'binary_targets_verified': True})
    require(unchanged == 9 and pending == 14
            and {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob('*') if p.is_file()} == expected_files,
            'Extra/missing artifacts or undeclared refinement')
    for p, h in data['preview_sha256'].items():
        require(sha(OUTPUT/p) == h, 'Reviewed preview changed')
    require(data['boundary_refinement']['index'] == 374
            and data['boundary_refinement']['validation_or_model_scores_used'] is False, 'Refinement scope changed')
    result = {'complete': True, 'date': '2026-10-02', 'script_sha256': sha(__file__),
              'test_sha256': sha(ROOT/'tests/test_real_reflection_proposal_audit.py'),
              'proposal_manifest_sha256': MANIFEST_SHA, 'source_review_sha256': sha(review_path),
              'records': counts, 'verified_artifact_sha256': hashes, 'pending_sources_unlabelled': pending,
              'other_proposals_unchanged': unchanged, 'new_approved_masks': 0,
              'model_forward_passes': 0, 'optimizer_updates_locally': 0, 'promoted': False,
              'scope': 'Independent saved-RGB/native-polygon/affine/binary/support check; no label authority or model-quality claim.'}
    with destination.open('x', encoding='utf-8', newline='\n') as file:
        file.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete': True, 'verified_proposals': len(counts), 'pending_sources_unlabelled': pending,
                      'unknown_context_preserved': True, 'new_approved_masks': 0, 'optimizer_updates_locally': 0}), flush=True)


if __name__ == '__main__':
    main()

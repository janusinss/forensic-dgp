"""Native source-only mask proposals; uncertain pixels never become negatives."""
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
from completion_data_v2 import square_preserving_geometry

REVIEW = ROOT/'outputs/real_reflection_review_v4'
OUTPUT = ROOT/'outputs/real_reflection_proposals_v4'
REVIEW_SHA = '76bcfe8061b331f09eaa78ae8f2c2e6a372e9da1042d5b4723454526b81d846c'
VISUAL_SHA = 'c0de12206894d2c04f23e0090dd1dadd69e00bb2816fa812d901ee662139f1fb'
ALL_IDS = {14, 52, 84, 106, 119, 157, 171, 177, 207, 208, 212, 216,
           217, 218, 219, 224, 230, 232, 240, 241, 249, 310, 348, 374}
CLEAR = {207, 208, 217, 230, 240, 241}
POLYGONS = {
    171: [
        [(2, 79), (10, 74), (24, 73), (43, 75), (61, 79), (68, 84), (69, 94),
         (63, 103), (51, 109), (33, 109), (17, 105), (6, 99), (2, 91)],
        [(80, 84), (91, 79), (108, 75), (125, 75), (133, 80), (134, 89),
         (129, 100), (120, 108), (106, 110), (93, 108), (84, 102), (80, 92)],
    ],
    216: [
        [(30, 54), (41, 51), (53, 51), (60, 54), (64, 60), (62, 67),
         (57, 72), (47, 73), (37, 73), (31, 69), (28, 63)],
        [(72, 52), (85, 50), (93, 51), (98, 55), (99, 60), (98, 65),
         (95, 71), (87, 73), (77, 73), (69, 70), (67, 64), (67, 59)],
    ],
    348: [
        [(32, 58), (39, 56), (49, 57), (58, 58), (63, 63), (61, 71),
         (56, 77), (51, 80), (43, 81), (36, 79), (32, 75), (30, 67)],
        [(69, 60), (75, 55), (83, 55), (91, 56), (96, 59), (97, 65),
         (94, 71), (88, 76), (83, 79), (77, 78), (72, 74), (69, 67)],
    ],
    374: [
        [(32, 55), (39, 54), (49, 55), (58, 57), (62, 60), (61, 69),
         (57, 75), (50, 80), (42, 78), (36, 74), (32, 68), (30, 61)],
        [(68, 57), (76, 56), (87, 56), (94, 58), (93, 65), (90, 72),
         (85, 77), (78, 78), (71, 76), (66, 70), (65, 62)],
    ],
}
UNKNOWN = {171: [[(0, 156), (157, 156), (157, 201), (0, 201)]]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposal_status(index):
    require(index in ALL_IDS, 'Source is outside the fixed qualified review')
    if index in POLYGONS:
        return 'proposed_covering'
    if index in CLEAR:
        return 'proposed_clear'
    return 'pending_native_review'


def rasterize(polygons, width, height):
    require(isinstance(polygons, (list, tuple)), 'Native polygon list required')
    canvas = Image.new('L', (width, height))
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3
                and np.isfinite(points).all() and np.equal(points, np.rint(points)).all()
                and len(set(map(tuple, points))) == len(points)
                and np.all(points >= 0) and np.all(points[:, 0] < width)
                and np.all(points[:, 1] < height), 'Polygon coordinates must be distinct finite native pixels')
        x, y = points.T
        require(abs(float(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))) > 0,
                'Degenerate native polygon')
        ImageDraw.Draw(canvas).polygon([tuple(int(v) for v in p) for p in points], fill=255)
    return np.array(canvas).astype(bool)


def prepare_target(rgb, polygons, unknown, size=256):
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3
            and rgb.shape[2] == 3, 'Native byte RGB required')
    height, width = rgb.shape[:2]
    native_mask = rasterize(polygons, width, height)
    native_unknown = rasterize(unknown, width, height)
    require(not (native_mask & native_unknown).any(), 'Known positive overlaps unreviewed native pixels')
    native_valid = ~native_unknown
    geometry = square_preserving_geometry(rgb, size)
    raw_mask = cv2.warpAffine(native_mask.astype('uint8'), geometry['affine'], (size, size),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    # Loss support is conservative. Unknown source pixels remain in the RGB input
    # as context; only true padding is neutral. They are never target negatives.
    coverage = cv2.warpAffine(native_valid.astype('float32'), geometry['affine'], (size, size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    source_valid = geometry['valid'].astype(bool)
    valid = source_valid & (coverage >= 1-1e-6)
    require(valid.any(), 'Empty annotated support')
    # Targets may lose conservative interpolation-border pixels only; a known
    # positive cannot be discarded because an unknown region was introduced.
    require(not (raw_mask & source_valid & ~valid).any(), 'Mapped positive overlaps ignored annotation region')
    mask = raw_mask & valid
    return {'rgb': geometry['rgb'], 'affine': geometry['affine'], 'scale': geometry['scale'],
            'source_valid': source_valid, 'valid': valid, 'mask': mask,
            'native_mask': native_mask, 'native_valid': native_valid,
            'mapped_positive_pixels_outside_source_support': int((raw_mask & ~source_valid).sum())}


def main():
    require(not OUTPUT.exists(), 'Preserve completed/partial annotation proposals')
    require(sha(REVIEW/'manifest.json') == REVIEW_SHA and sha(REVIEW/'visual_review.json') == VISUAL_SHA,
            'Qualified native review binding changed')
    manifest = json.loads((REVIEW/'manifest.json').read_text())
    review = json.loads((REVIEW/'visual_review.json').read_text())
    require(manifest['complete'] is True and manifest['training_enabled'] is False
            and manifest['overlap_flagged_sources'] == 0 and not manifest['within_candidate_pairs']
            and review['manifest_sha256'] == REVIEW_SHA and review['new_approved_masks'] == 0,
            'Qualified source-only state differs')
    require(all(sha(ROOT/path) == digest for path, digest in manifest['input_sha256'].items())
            and sha(REVIEW/'reference_signatures.json') == manifest['reference_signatures_sha256'],
            'Frozen source qualification references changed')
    rows = manifest['records']; require({r['index'] for r in rows} == ALL_IDS and len(rows) == 24,
                                       'Fixed native source cohort differs')
    notes = {r['index']: r['finding'] for r in review['records']}
    prepared = {}; output_rows = []
    for row in rows:
        index = row['index']
        require(row['mask'] is None and row['training_enabled'] is False and row['reviewed'] is False
                and row['qualification'] == 'eligible_for_native_review'
                and sha(ROOT/row['path']) == row['sha256']
                and sha(ROOT/row['native_review_image']) == row['native_review_sha256'], 'Qualified source bytes/state changed')
        with Image.open(ROOT/row['native_review_image']) as image:
            rgb = np.array(image.convert('RGB'))
        require([rgb.shape[1], rgb.shape[0]] == row['native_review_size'], 'Native oriented geometry differs')
        status = proposal_status(index)
        record = {'index': index, 'source': row['path'], 'source_sha256': row['sha256'],
                  'native_review_image': row['native_review_image'], 'native_review_sha256': row['native_review_sha256'],
                  'native_size': row['native_review_size'], 'split': 'train', 'group': 'source-'+row['sha256'],
                  'status': status, 'mask': None, 'image': None, 'valid': None,
                  'training_enabled': False, 'reviewed': False, 'pixel_annotation_approved': False,
                  'rationale': notes[index], 'identity_disjointness': 'not verified'}
        if status != 'pending_native_review':
            polygons = POLYGONS.get(index, [])
            item = prepare_target(rgb, polygons, UNKNOWN.get(index, []))
            require(bool(item['native_mask'].any()) == (status == 'proposed_covering'), 'Proposal kind/target differs')
            prepared[index] = item
            record.update(native_polygons=polygons, unknown_native_polygons=UNKNOWN.get(index, []),
                          native_positive_pixels=int(item['native_mask'].sum()),
                          native_ignored_pixels=int((~item['native_valid']).sum()),
                          positive_pixels=int(item['mask'].sum()), supervised_pixels=int(item['valid'].sum()),
                          source_support_pixels=int(item['source_valid'].sum()),
                          affine=item['affine'].tolist(), scale=item['scale'],
                          mapped_positive_pixels_outside_source_support=item['mapped_positive_pixels_outside_source_support'],
                          geometry='uniform full-source pixel-center affine to 256; neutral padding',
                          annotation_scope='lens footprints only; lower helmet/strap support pending' if index == 171
                                           else 'complete inspected source under strong-lens covering policy',
                          proposed_stratum=('scene_reflective_lenses' if index in (171, 348) else 'opaque_lenses')
                                            if status == 'proposed_covering' else 'transparent_control',
                          proposal_quality='Approximate assistant native-source polygons; not expert ground truth',
                          uncovered_face_reference=None, high_resolution_reference=False)
        output_rows.append(record)
    require(len(prepared) == 10 and len(POLYGONS) == 4 and len(CLEAR) == 6,
            'Fixed proposal count differs')
    OUTPUT.mkdir()
    for folder in ('images', 'masks', 'valid', 'source_valid', 'native_masks', 'native_valid'):
        (OUTPUT/folder).mkdir()
    for row in output_rows:
        index = row['index']
        if index not in prepared:
            continue
        item = prepared[index]; name = f'{index:03}.png'; files = {}
        for folder, field in (('images', 'rgb'), ('masks', 'mask'), ('valid', 'valid'),
                              ('source_valid', 'source_valid'), ('native_masks', 'native_mask'), ('native_valid', 'native_valid')):
            array = item[field]
            if field != 'rgb':
                array = array.astype('uint8')*255
            path = OUTPUT/folder/name; Image.fromarray(array).save(path)
            files[folder] = {'path': path.relative_to(OUTPUT).as_posix(), 'sha256': sha(path)}
        row.update(image=files['images']['path'], mask=files['masks']['path'], valid=files['valid']['path'], files=files)
    for status in ('proposed_covering', 'proposed_clear'):
        selected = [r for r in output_rows if r['status'] == status]
        sheet = Image.new('RGB', (1280, len(selected)*280), 'white'); draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected):
            item = prepared[row['index']]; rgb = item['rgb']; overlay = rgb.copy()
            p, ignored = item['mask'], item['source_valid'] & ~item['valid']
            overlay[p] = np.rint(.55*overlay[p] + .45*np.array([255, 45, 45])).astype('uint8')
            # Gray is review-only loss exclusion, not a newly invented occluder.
            overlay[ignored] = np.rint(.6*overlay[ignored] + .4*np.array([160, 160, 190])).astype('uint8')
            fields = [rgb, overlay, item['mask'].astype('uint8')*255,
                      item['valid'].astype('uint8')*255, item['source_valid'].astype('uint8')*255]
            for j, array in enumerate(fields):
                sheet.paste(Image.fromarray(array).convert('RGB'), (j*256, i*280+24))
            draw.text((3, i*280+4), f"{row['index']} | input | red target / gray ignored | proposed mask | loss support | source support", fill='black')
        sheet.save(OUTPUT/(status+'.png'))
    result = {'format': 'dgp-unpaired-real-mask-proposals-v4', 'date': '2026-10-02', 'complete': True,
              'source_review_sha256': REVIEW_SHA, 'source_visual_review_sha256': VISUAL_SHA,
              'code_sha256': {name: sha(ROOT/name) for name in ('scripts/prepare_real_reflection_proposals.py',
                             'completion_data_v2.py', 'tests/test_real_reflection_proposals.py')},
              'policy_sha256': sha(ROOT/'OCCLUSION_POLICY_V3.md'), 'records': output_rows,
              'status_counts': dict(Counter(r['status'] for r in output_rows)),
              'preview_sha256': {p.name: sha(p) for p in OUTPUT.glob('*.png')},
              'training_enabled': False, 'training_recipe_ready': False, 'new_approved_masks': 0,
              'model_forward_passes': 0, 'optimizer_updates_locally': 0, 'promoted': False,
              'support_policy': 'Ignore padding and unreviewed native regions in every supervised reduction; preserve unknown RGB as input context.',
              'scope': 'Unpaired real occlusion-mask proposals only; no uncovered-face targets, masks accepted or original splits/gates changed.',
              'consumer_note': 'Legacy ReviewedMasks ignores valid support. Do not use this proposal registry as a training manifest.',
              'limitations': ['Approximate native assistant polygons require overlay review; no independent expert labels.',
                             'Fourteen sources remain unlabelled, not empty clear targets.',
                             'Source171 lower helmet/strap annotation is pending and excluded from loss support.',
                             'Opaque/scene-reflective lenses do not establish small/partial glare transfer or repaired synthetic retention.']}
    (OUTPUT/'manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status_counts', 'training_enabled', 'training_recipe_ready', 'new_approved_masks')}
                     | {'ignored_native_pixels_171': next(r['native_ignored_pixels'] for r in output_rows if r['index'] == 171)}), flush=True)


if __name__ == '__main__':
    main()

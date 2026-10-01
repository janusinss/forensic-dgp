"""Preserve draft one; refine a source-reviewed lens boundary in a new draft."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_real_reflection_proposals import prepare_target, require, sha

PARENT = ROOT/'outputs/real_reflection_proposals_v4'
OUTPUT = ROOT/'outputs/real_reflection_proposals_v4_refined'
PARENT_SHA = '368cdf82cfeef3088c225f2b70cd797697419875989fe1d7bfd0482d9d95d24a'
CORRECTED = [
    [(31, 54), (39, 53), (49, 53), (59, 54), (63, 58), (61, 64),
     (57, 69), (51, 72), (41, 73), (35, 69), (31, 64), (30, 59)],
    [(67, 55), (76, 54), (87, 54), (94, 55), (95, 61), (92, 68),
     (87, 72), (80, 73), (72, 71), (68, 66), (66, 60)],
]


def main():
    require(not OUTPUT.exists(), 'Preserve completed/partial refined proposals')
    require(sha(PARENT/'manifest.json') == PARENT_SHA, 'Preserved initial proposal changed')
    parent = json.loads((PARENT/'manifest.json').read_text())
    require(parent['complete'] is True and parent['new_approved_masks'] == 0
            and parent['training_enabled'] is False and parent['training_recipe_ready'] is False
            and all(sha(ROOT/p) == h for p, h in parent['code_sha256'].items()), 'Initial code/state bindings differ')
    require(all(sha(PARENT/p) == h for p, h in parent['preview_sha256'].items()), 'Initial reviewed preview changed')
    result = copy.deepcopy(parent); changed = next(r for r in result['records'] if r['index'] == 374)
    require(changed['status'] == 'proposed_covering' and not changed['unknown_native_polygons'], 'Fixed correction source differs')
    require(sha(ROOT/changed['source']) == changed['source_sha256']
            and sha(ROOT/changed['native_review_image']) == changed['native_review_sha256'], 'Native correction source changed')
    with Image.open(ROOT/changed['native_review_image']) as image:
        rgb = np.array(image.convert('RGB'))
    item = prepare_target(rgb, CORRECTED, [])
    require(item['mask'].any() and item['mapped_positive_pixels_outside_source_support'] == 0,
            'Refined target empty or outside source support')
    for row in result['records']:
        if row['mask'] is not None:
            require(all(sha(PARENT/f['path']) == f['sha256'] for f in row['files'].values()), 'Initial proposal bytes changed')
    OUTPUT.mkdir()
    for row in result['records']:
        if row['mask'] is None:
            continue
        for folder, artifact in row['files'].items():
            path = OUTPUT/artifact['path']; path.parent.mkdir(parents=True, exist_ok=True)
            if row['index'] == 374:
                field = {'images': 'rgb', 'masks': 'mask', 'valid': 'valid', 'source_valid': 'source_valid',
                         'native_masks': 'native_mask', 'native_valid': 'native_valid'}[folder]
                array = item[field]
                if field != 'rgb':
                    array = array.astype('uint8')*255
                Image.fromarray(array).save(path)
                artifact['sha256'] = sha(path)
            else:
                shutil.copyfile(PARENT/artifact['path'], path)
                require(sha(path) == artifact['sha256'], 'Unchanged proposal copy differs')
    changed.update(native_polygons=CORRECTED, native_positive_pixels=int(item['native_mask'].sum()),
                   positive_pixels=int(item['mask'].sum()),
                   correction='Native pixel-grid review reduced the lower lens footprint; first draft included visible cheek skin.')
    # Rebuild only the affected contact sheet from saved output; the six clear
    # controls preserve their original contact-sheet bytes.
    selected = [r for r in result['records'] if r['status'] == 'proposed_covering']
    sheet = Image.new('RGB', (1280, len(selected)*280), 'white'); draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        rgb = np.array(Image.open(OUTPUT/row['image']).convert('RGB'))
        arrays = {key: np.array(Image.open(OUTPUT/row['files'][key]['path'])).astype(bool)
                  for key in ('masks', 'valid', 'source_valid')}
        overlay = rgb.copy(); p = arrays['masks']; ignored = arrays['source_valid'] & ~arrays['valid']
        overlay[p] = np.rint(.55*overlay[p]+.45*np.array([255, 45, 45])).astype('uint8')
        overlay[ignored] = np.rint(.6*overlay[ignored]+.4*np.array([160, 160, 190])).astype('uint8')
        fields = [rgb, overlay, arrays['masks'].astype('uint8')*255,
                  arrays['valid'].astype('uint8')*255, arrays['source_valid'].astype('uint8')*255]
        for j, array in enumerate(fields):
            sheet.paste(Image.fromarray(array).convert('RGB'), (j*256, i*280+24))
        draw.text((3, i*280+4), f"{row['index']} | input | red target / gray ignored | proposed mask | loss support | source support", fill='black')
    sheet.save(OUTPUT/'proposed_covering.png')
    shutil.copyfile(PARENT/'proposed_clear.png', OUTPUT/'proposed_clear.png')
    result.update(parent_proposal_sha256=PARENT_SHA,
                  boundary_refinement={'index': 374, 'method': 'assistant source-only native pixel-grid/overlay review',
                                       'validation_or_model_scores_used': False,
                                       'previous_native_positive_pixels': next(r['native_positive_pixels'] for r in parent['records'] if r['index'] == 374)},
                  preview_sha256={p.name: sha(p) for p in OUTPUT.glob('*.png')})
    result['code_sha256']['scripts/refine_real_reflection_proposals.py'] = sha(__file__)
    (OUTPUT/'manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'changed_source': 374, 'native_positive_pixels': changed['native_positive_pixels'],
                      'positive_pixels': changed['positive_pixels'], 'other_proposals_unchanged': 9,
                      'new_approved_masks': 0, 'training_recipe_ready': False}), flush=True)


if __name__ == '__main__':
    main()

"""Source-only manual annotation proposals; never changes training manifests."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from audit_real_source_pool import signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/real_expansion_proposals_v3')
    out = Path(parser.parse_args().output)
    out.mkdir(exist_ok=False)
    for name in ('images', 'masks'):
        (out / name).mkdir()
    specs = [
        ('masked (1655).jpg', (135, 0, 585, 450), 'profile_respirator', [
            [(244,106),(307,124),(357,151),(381,204),(399,267),(391,313),(359,354),(328,356),(302,356),(258,349),(229,337),(216,318),(208,285),(194,258),(184,232),(175,205),(190,165),(222,132)],
            [(351,151),(451,99),(465,98),(368,158)],
            [(394,268),(457,215),(471,183),(474,174),(475,204),(462,224),(398,281)],
        ]),
        ('masked (1257).jpg', (200, 0, 680, 480), 'patterned_respirator', [
            [(282,162),(296,194),(338,207),(403,206),(456,218),(503,242),(559,259),(573,259),(564,293),(550,329),(523,375),(480,417),(440,441),(390,436),(339,416),(293,383),(260,349),(265,305),(274,268)],
            [(282,165),(281,112),(289,65),(302,39),(306,39),(296,72),(289,122),(291,175)],
        ]),
        ('masked (1439).jpg', (270, 45, 555, 330), 'hand_over_mask', [
            [(359,190),(380,190),(399,185),(431,189),(468,197),(481,191),(494,183),(491,226),(480,256),(459,284),(425,302),(395,279),(369,245)],
            [(371,277),(378,254),(385,236),(391,246),(393,265),(414,249),(436,233),(457,219),(465,221),(461,233),(451,245),(468,233),(478,231),(482,237),(475,250),(463,264),(479,254),(487,255),(486,265),(472,282),(478,278),(482,282),(472,297),(455,310),(443,324),(416,323),(394,301)],
            [(492,186),(508,175),(512,176),(496,195)],
        ]),
    ]
    audited = {r['path']: r for r in json.loads(Path('outputs/real_source_pool_audit/source_review.json').read_text())['records']}
    v3_path = Path('dataset/detector_glare_review_v3/manifest.json')
    v3 = json.loads(v3_path.read_text())
    refs = [(r, signature(v3_path.parent / r['image'])) for r in v3['records']]
    rows = []
    canvas = Image.new('RGB', (768, 3 * 280), 'white')
    for i, (name, box, kind, polygons) in enumerate(specs):
        source = Path('dataset/real_occlusion_review') / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == audited[source.as_posix()]['sha256']
        rgb = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
        target = Image.new('L', rgb.size)
        draw = ImageDraw.Draw(target)
        for polygon in polygons:
            draw.polygon(polygon, fill=255)
        crop = rgb.crop(box).resize((256, 256), Image.Resampling.LANCZOS)
        mask = target.crop(box).resize((256, 256), Image.Resampling.NEAREST)
        image_path = out / 'images' / f'{i:02}.png'
        mask_path = out / 'masks' / f'{i:02}.png'
        crop.save(image_path)
        mask.save(mask_path)
        sig = signature(image_path)
        nearest = sorted([{'image': r['image'], 'split': r['split'],
                           'distance': (sig['phash'] ^ s['phash']).bit_count(),
                           'exact': sig['decoded_sha256'] == s['decoded_sha256']}
                          for r, s in refs], key=lambda r: r['distance'])[:5]
        overlay = np.array(crop).copy()
        selected = np.array(mask) > 0
        overlay[selected] = (overlay[selected] * .55 + np.array([255, 40, 40]) * .45).astype('uint8')
        for col, tile in enumerate((crop, Image.fromarray(overlay), mask.convert('RGB'))):
            canvas.paste(tile, (col * 256, i * 280 + 24))
        ImageDraw.Draw(canvas).text((5, i * 280 + 4), f'{i}: {name} | {kind} | PROPOSAL', fill='black')
        rows.append({'source': source.as_posix(), 'source_sha256': digest,
                     'group': 'source-' + digest, 'crop_ltrb': box, 'native_polygons': polygons,
                     'image': str(image_path), 'mask': str(mask_path), 'kind': kind,
                     'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                     'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                     'nearest_v3_crops': nearest, 'split': None, 'training_enabled': False,
                     'reviewed': False, 'annotation': 'assistant approximate source polygons; pending overlay review'})
    canvas.save(out / 'preview.png')
    (out / 'manifest.json').write_text(json.dumps({'training_enabled': False,
        'v3_manifest_sha256': hashlib.sha256(v3_path.read_bytes()).hexdigest(),
        'scope': 'mask body and straps; hand pixels only where covering face, not entire arm',
        'limitations': 'Approximate polygons. Crop hash screening is not identity-disjoint proof. No uncovered reconstruction targets.',
        'records': rows}, indent=2) + '\n')
    print([(r['kind'], r['nearest_v3_crops'][0]) for r in rows])


if __name__ == '__main__':
    main()

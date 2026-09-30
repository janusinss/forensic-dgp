"""Create source-traced eyewear labels for review, without enabling training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/eyewear_annotation_proposals_v2')
    out = Path(parser.parse_args().output)
    out.mkdir(exist_ok=False)
    for folder in ('images', 'masks'):
        (out / folder).mkdir()
    queue = json.loads(Path('outputs/eyewear_review_v1/queue.json').read_text())
    records = {r['index']: r for r in queue['records']}
    # Coordinates manually traced on native sources, not model predictions.
    specifications = [
        (13, 'opaque_lenses', [
            [(21,62),(28,56),(38,55),(53,57),(64,60),(70,68),(68,82),(62,91),(52,96),(38,96),(27,92),(20,83),(18,73)],
            [(79,65),(89,59),(102,57),(117,57),(129,61),(137,68),(139,76),(135,84),(129,92),(116,97),(101,99),(90,96),(81,90),(77,78)],
        ]),
        (49, 'opaque_lenses', [
            [(0,91),(14,84),(32,80),(53,80),(73,84),(85,91),(89,101),(86,119),(79,136),(67,145),(48,149),(29,148),(13,143),(3,135),(0,123)],
            [(104,95),(113,85),(130,80),(151,80),(171,84),(185,92),(191,101),(191,115),(185,126),(174,138),(159,144),(141,147),(125,145),(113,140),(106,128),(101,111)],
        ]),
        (46, 'transparent_control', []),
    ]
    sheet = Image.new('RGB', (768, 840), 'white')
    rows = []
    split = json.loads(Path('outputs/downloaded_phase4/outputs/phase4_with_progress/split.json').read_text())
    train = {p.replace('\\', '/') for p in split['train']}
    held = {p.replace('\\', '/') for p in split['validation']}
    for i, (index, kind, polygons) in enumerate(specifications):
        r = records[index]
        source = Path(r['path'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == r['sha256']
        assert r['path'] in train and r['path'] not in held
        rgb = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
        target = Image.new('L', rgb.size)
        for polygon in polygons:
            ImageDraw.Draw(target).polygon(polygon, fill=255)
        image = rgb.resize((256, 256), Image.Resampling.LANCZOS)
        mask = target.resize((256, 256), Image.Resampling.NEAREST)
        image_path = out / 'images' / f'{index}.png'
        mask_path = out / 'masks' / f'{index}.png'
        image.save(image_path)
        mask.save(mask_path)
        overlay = np.array(image)
        positive = np.array(mask) > 0
        overlay[positive] = (overlay[positive] * .55 + np.array([255,40,40]) * .45).astype('uint8')
        for col, tile in enumerate((image, Image.fromarray(overlay), mask.convert('RGB'))):
            sheet.paste(tile, (col * 256, i * 280 + 24))
        ImageDraw.Draw(sheet).text((4, i * 280 + 4), f'{index} {kind} | proposal', fill='black')
        rows.append({**r, 'native_polygons': polygons, 'image': str(image_path),
                     'mask': str(mask_path), 'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                     'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                     'native_size': rgb.size, 'resize': 'full-source anisotropic to 256x256',
                     'group': 'source-' + r['sha256'], 'kind': kind, 'reviewed': False,
                     'training_enabled': False, 'positive_pixels': int(positive.sum())})
    sheet.save(out / 'preview.png')
    (out / 'manifest.json').write_text(json.dumps({
        'training_enabled': False,
        'policy': 'Opaque lens footprints conceal facial detail; preserve transparent lenses and visible face. No automatic tiny-glare label.',
        'annotation_quality': 'Approximate manually traced polygons; not expert ground truth',
        'held_native_source': {'index': 84, 'reason': 'Dark tinted glasses retain some eye detail; insufficient confidence for whole-lens target'},
        'records': rows}, indent=2) + '\n')
    print([(r['index'], r['positive_pixels']) for r in rows])


if __name__ == '__main__':
    main()

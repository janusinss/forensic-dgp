"""Recount reviewed V3 reflection additions; saved-mask analysis, no fitting."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector_training import load_manifest, ReviewedMasks
from scripts.audit_face_occlusion_results import require, read_json, verified_inputs
from scripts.audit_face_occlusion_continuation_results import ARCHIVE, AUDIT_OUTPUT, EXTRACTION, PREFIX
from scripts.package_face_occlusion_vm import sha
from scripts.evaluate_coverage_results import binary


def added_region_counts(prediction, target, parent):
    require(all(isinstance(a, np.ndarray) and a.dtype == np.bool_ and a.ndim == 2
                for a in (prediction, target, parent)) and prediction.shape == target.shape == parent.shape,
            'Require matching binary 2D arrays')
    require(not np.any(parent & ~target), 'V3 removed pre-existing parent mask pixels')
    added = target & ~parent; pixels = int(added.sum())
    require(pixels > 0, 'Require a nonempty reviewed addition')
    recovered = int((prediction & added).sum())
    return {'added_glare_pixels': pixels, 'recovered_added_glare_pixels': recovered,
            'missed_added_glare_pixels': pixels - recovered, 'glare_recall': recovered / pixels,
            'original_mask_pixels': int(parent.sum()),
            'original_mask_recovered_pixels': int((prediction & parent).sum()),
            'whole_mask_false_positive_pixels': int((prediction & ~target).sum())}


def main():
    from PIL import Image, ImageDraw
    out = ROOT / 'outputs/face_occlusion_continuation_glare'
    require(not out.exists(), 'Preserve existing glare evidence')
    audit = read_json(AUDIT_OUTPUT / 'results.json'); reproduction = read_json(AUDIT_OUTPUT / 'reproduction.json')
    require(reproduction['complete'] is True and sha(ARCHIVE) == audit['archive_sha256'] == reproduction['archive_sha256'],
            'Completed return/checkpoint audit changed')
    _, validation_rows, data = verified_inputs()
    v3_path = ROOT / 'dataset/detector_glare_review_v3/manifest.json'; v3 = read_json(v3_path)
    parent_path = ROOT / 'dataset/detector_expanded_review_v2/manifest.json'
    require(sha(parent_path) == v3['parent_manifest_sha256'], 'V2 review parent changed')
    old_rows = load_manifest(parent_path); old = {r['image_sha256']: r for r in old_rows}
    require(len(old) == len(old_rows) == 100, 'Parent image membership differs')
    v3_rows = {r['image_sha256']: r for r in v3['records']}
    returned = EXTRACTION / PREFIX; members = read_json(AUDIT_OUTPUT / 'members.json')
    fit_path = ROOT / 'outputs/face_occlusion_continuation_fit/results.json'; fit = read_json(fit_path)
    require(fit['complete'] is True and fit['archive_sha256'] == audit['archive_sha256'], 'Fit diagnostic differs')
    choices = [(10, 'initial_masks')] + [(e, f'epoch_{e}_masks') for e in (11, 15, 20, 30)]
    records = []; visuals = []
    for domain, rows in [('training_real', data['training_real'].rows), ('real', validation_rows)]:
        for i, row in enumerate(rows):
            if row.get('glare_stratum') != 'strong_lens_reflection': continue
            digest = row['image_sha256']; require(digest in old and digest in v3_rows, 'Glare parent source missing')
            original = old[digest]; reviewed = v3_rows[digest]
            require(row['mask_sha256'] == reviewed['mask_sha256'] and row['split'] == original['split'] == reviewed['split']
                    and reviewed['v2_mask_sha256'] == original['mask_sha256'], 'Glare mask/source/split lineage differs')
            image, m = data[domain][i]; target = m[0].numpy().astype(bool)
            parent = ReviewedMasks([original], 256)[0][1][0].numpy().astype(bool)
            added = target & ~parent; states = {}; predictions = {}
            for e, folder in choices:
                path = returned / folder / domain / f'{i:04}.png'
                require(sha(path) == members[path.relative_to(EXTRACTION).as_posix()], 'Verified prediction mask changed')
                p = binary(path); states[str(e)] = added_region_counts(p, target, parent); predictions[e] = p & ~parent
            exposure = {str(e): next(r['training_exposures'] for r in
                         fit['real_fit'][f'pretrained{e}'][domain]['glare']['records'] if r['case'] == i)
                        for e, _ in choices}
            records.append({'domain': domain, 'case': i, 'image': row['image'], 'image_sha256': digest,
                            'v2_mask_sha256': original['mask_sha256'], 'v3_mask_sha256': reviewed['mask_sha256'],
                            'states': states, 'training_exposures': exposure,
                            'review_rationale': reviewed['glare_review_rationale']})
            visuals.append((domain, i, image, added, predictions))
    require(sum(r['domain'] == 'training_real' for r in records) == 2 and
            sum(r['domain'] == 'real' for r in records) == 1, 'Expected two train/one validation glare case')
    summary = {}
    for e, _ in choices:
        summary[str(e)] = {}
        for domain in ('training_real', 'real'):
            subset = [r['states'][str(e)] for r in records if r['domain'] == domain]
            pixels = sum(r['added_glare_pixels'] for r in subset)
            recovered = sum(r['recovered_added_glare_pixels'] for r in subset)
            summary[str(e)][domain] = {'cases': len(subset), 'added_glare_pixels': pixels,
                                      'recovered_added_glare_pixels': recovered, 'glare_recall': recovered / pixels}
    out.mkdir(); sheet = Image.new('RGB', (1024, 148 * len(visuals)), 'white')
    for line, (domain, i, x, added, predictions) in enumerate(visuals):
        image = np.rint(x.permute(1, 2, 0).numpy() * 255).astype('uint8'); yy, xx = np.where(added)
        box = (max(0, int(xx.min()) - 12), max(0, int(yy.min()) - 12),
               min(256, int(xx.max()) + 13), min(256, int(yy.max()) + 13))
        tiles = [Image.fromarray(image), Image.fromarray(image).crop(box),
                 Image.fromarray(added.astype('uint8') * 255).convert('RGB').crop(box)]
        tiles.extend(Image.fromarray(predictions[e].astype('uint8') * 255).convert('RGB').crop(box) for e, _ in choices)
        ImageDraw.Draw(sheet).text((2, line * 148 + 2),
                                  f'{domain} {i}: face | glare zoom | V3 addition | source10 | epoch11 | epoch15 | epoch20 | epoch30', fill='black')
        for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128), Image.Resampling.NEAREST), (j * 128, line * 148 + 20))
    sheet.save(out / 'preview.png')
    result = {'complete': True, 'script_sha256': sha(__file__), 'archive_sha256': audit['archive_sha256'],
              'v2_manifest_sha256': sha(parent_path), 'v3_manifest_sha256': sha(v3_path),
              'fit_report_sha256': sha(fit_path), 'records': records, 'summary': summary,
              'optimizer_updates_locally': 0, 'promoted': False,
              'scope': 'Reviewed V3-minus-V2 reflection pixels only; training and validation, no test scoring',
              'precision_note': 'Whole-mask visible false pixels are reported separately; mouth-mask overlap cannot count as glare recovery',
              'preview_note': 'Exclude the original V2 mask from predictions; target-derived crop plus12px context',
              'annotation_limitation': 'User-accepted approximate assistant polygons; no independent expert adjudication'}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'records': records, 'optimizer_updates_locally': 0}, indent=2), flush=True)


if __name__ == '__main__': main()

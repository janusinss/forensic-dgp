"""Paste exact saved256 DEV pixels into source-separated review sheets; no NN."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_spatial_dev_review_v1'
ARMS = ('original', 'r2', 'spatial_u', 'spatial_s')


def pin(path):
    b = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def main():
    cohort = json.loads((REVIEW/'cohort_manifest_v1.json').read_bytes())
    run = json.loads((REVIEW/'run_v1_r2/comparison.json').read_bytes())
    assert run['complete'] and len(run['cases']) == len(cohort['cases']) == 68
    out = REVIEW/'visuals_v1'
    assert not out.exists()
    out.mkdir()
    by_id = {r['id']: r for r in run['cases']}
    groups = defaultdict(list)
    for row in cohort['cases']:
        group = row['cohort'] + ('_' + row['reference_id'] if row['target'] else '')
        groups[group].append(row)
    result = {'complete': True, 'pixels_pasted_without_resampling_or_enhancement': True,
              'candidate_images': 272, 'rows': 68, 'sheets': [],
              'cohort': pin(REVIEW/'cohort_manifest_v1.json'),
              'run': pin(REVIEW/'run_v1_r2/comparison.json'),
              'planned_root_personal_review_complete': False,
              'quality_claim': 'Review sheets, not model qualification. Native CCTV is unpaired.'}
    for group, rows in groups.items():
        for start in range(0, len(rows), 6):
            selected = rows[start:start+6]
            sheet = Image.new('RGB', (1580, 94+306*len(selected)), '#e6e4df')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 6), group + ' | DEV only | exact saved256 pixels', fill='black')
            draw.text((8, 25), 'Native CCTV: no aligned clean target. Paired filter stress: four held photographs, not unseen identities.', fill='black')
            draw.text((8, 44), 'Experimental U96/S96; original CodeFormer and R2 retained. No display brightening/sharpening.', fill='black')
            for col, title in enumerate(('Observed input', 'Paired target / native note', 'Untouched CodeFormer', 'R2', 'U96', 'S96')):
                draw.text((8+262*col, 68), title, fill='black')
            records = []
            for index, row in enumerate(selected):
                y = 94+306*index
                label = row['id'] + (' | FLAGGED appearance' if row.get('appearance_flag') else '')
                draw.text((8, y), label, fill='black')
                frozen_review = row['input_review']
                decision = frozen_review.get('decision', frozen_review.get('input_structure_review', frozen_review.get('input_review')))
                draw.text((8, y+17), 'Input-only review: ' + str(decision) + ' | source: ' + row['source'], fill='black')
                files = [ROOT/row['input']['path'], ROOT/row['target']['path'] if row['target'] else None]
                files += [REVIEW/'run_v1_r2'/row['id']/(a+'.png') for a in ARMS]
                for col, path in enumerate(files):
                    if path:
                        with Image.open(path) as image:
                            assert image.mode == 'RGB' and image.size == (256,256)
                            sheet.paste(image, (8+262*col, y+40))
                    else:
                        draw.text((8+262*col+8, y+120), 'Native CCTV: unpaired', fill='black')
                        draw.text((8+262*col+8, y+138), 'No clean reference', fill='black')
                records.append({'id': row['id'], 'input_review': frozen_review,
                                'appearance_flag': row.get('appearance_flag', False),
                                'evidence': row['evidence'], 'images': [pin(p) if p else None for p in files]})
            path = out/(group+'_'+str(start//6+1)+'.png')
            sheet.save(path)
            result['sheets'].append({'sheet': pin(path), 'group': group, 'rows': records})
    assert len(result['sheets']) == 14 and sum(len(s['rows']) for s in result['sheets']) == 68
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'complete':True,'sheets':14,'rows':68,'candidate_PNGs':272,'root_review_pending':True}))


if __name__ == '__main__':
    main()

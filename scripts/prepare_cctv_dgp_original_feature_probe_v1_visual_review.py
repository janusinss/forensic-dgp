"""Exact saved-pixel sheets for every diagnostic face; zero neural calls."""
import hashlib
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm'
RETURN = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm_return'
OUT = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1/gallery'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert not OUT.exists(); OUT.mkdir()
    start = time.monotonic(); p = read(BUNDLE/'protocol.json'); r = read(RETURN/'outputs/results.json')
    assert sha(BUNDLE/'protocol.json') == r['protocol_sha256'] == 'ccc30870959c8f4f4f448c1d1e06bb0ae0403d450b81143c6bcb9032d122e94b'
    by = {c['id']: c for c in p['cases']}; pages = []; bindings = {}; unique_models = set(); cells = 0; group_index = 0
    for co in p['cohorts']:
        for begin in range(0, 50, 5):
            group_index += 1
            cases = [by[cid] for cid in co['case_ids'][begin:begin+5]]
            assert len({c['source_person_or_reference'] for c in cases}) == 1
            for scope in p['scopes']:
                labels = [scope+'_'+format(f, '.0e').replace('-', 'm') for f in p['relative_displacement_fractions']]
                headings = ['input', 'paired target', 'baseline']+labels
                sheet = Image.new('RGB', (6*268, 24+5*292), 'white'); draw = ImageDraw.Draw(sheet)
                for col, label in enumerate(headings): draw.text((5+268*col, 4), label, fill='black')
                source_cells = []
                for row, c in enumerate(cases):
                    yy = 24+292*row; draw.text((5, yy), co['name']+' / '+c['id'], fill='black')
                    files = [BUNDLE/c['input'], BUNDLE/c['target']]+[RETURN/'outputs'/label/(c['id']+'.png') for label in ['baseline']+labels]
                    for col, file in enumerate(files):
                        relative = file.relative_to(ROOT).as_posix(); bindings[relative] = sha(file)
                        with Image.open(file) as image: image = image.convert('RGB'); assert image.size == (256, 256); sheet.paste(image, (5+268*col, yy+24))
                        source_cells.append({'file': relative, 'box': [5+268*col, yy+24, 5+268*col+256, yy+24+256]}); cells += 1
                        if col >= 2: unique_models.add(relative)
                name = f'{group_index:02d}-{scope}.png'; sheet.save(OUT/name)
                pages.append({'file': name, 'cohort': co['name'], 'reference': cases[0]['source_person_or_reference'],
                    'scope': scope, 'case_ids': [c['id'] for c in cases], 'headings': headings,
                    'size': list(sheet.size), 'sha256': sha(OUT/name), 'cells': source_cells})
                assert time.monotonic()-start <= 120
    assert len(pages) == 60 and len(unique_models) == 1000 and cells == 1800
    plan = {'complete': True, 'protocol_sha256': sha(BUNDLE/'protocol.json'), 'results_sha256': sha(RETURN/'outputs/results.json'),
        'generator_sha256': sha(Path(__file__)), 'pages': pages, 'source_bindings': bindings,
        'unique_model_outputs': len(unique_models), 'exact256_cells': cells, 'pages_count': len(pages),
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'visual_review_pending': True, 'model_qualification': False,
        'seconds': time.monotonic()-start}
    with (OUT/'gallery_manifest.json').open('x', encoding='utf-8') as f: json.dump(plan, f, indent=2); f.write('\n')
    print({k: plan[k] for k in ['complete', 'unique_model_outputs', 'exact256_cells', 'pages_count', 'seconds']})


if __name__ == '__main__': main()

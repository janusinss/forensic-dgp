"""Exact saved-pixel review sheets; no inference or learning."""
import hashlib
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_vm'
RETURN = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_vm_return'
OUT = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1/gallery'
AUDIT = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_independent_audit.json'
PIN = '8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Retain any prior review sheets'
    assert sha(BUNDLE / 'protocol.json') == PIN
    p = read(BUNDLE / 'protocol.json')
    audit = read(AUDIT)
    r = read(RETURN / 'outputs/results.json')
    assert audit['complete'] and audit['protocol_sha256'] == r['protocol_sha256'] == PIN
    assert audit['all36_stored_row_comparisons_and_failure_decisions_exact']
    assert audit['CPU_replay_cases'] == 200
    assert r['gradient_queries'] == r['optimizer_updates'] == r['epochs'] == 0
    OUT.mkdir(parents=True)
    cases_by_id = {case['id']: case for case in p['cases']}
    pages, bindings, models = [], {}, set()
    cells, group_number = 0, 0
    for cohort in p['cohorts']:
        for begin in range(0, 50, 5):
            group_number += 1
            cases = [cases_by_id[cid] for cid in cohort['case_ids'][begin:begin + 5]]
            assert len({case['source_person_or_reference'] for case in cases}) == 1
            for mode in p['scopes']:
                labels = [mode + '_' + format(fraction, '.0e').replace('-', 'm')
                    for fraction in p['relative_displacement_fractions']]
                headings = ['input', 'paired target', 'baseline'] + labels
                sheet = Image.new('RGB', (6 * 268, 24 + 5 * 292), 'white')
                draw = ImageDraw.Draw(sheet)
                for col, heading in enumerate(headings):
                    draw.text((5 + 268 * col, 4), heading, fill='black')
                source_cells = []
                for row, case in enumerate(cases):
                    yy = 24 + 292 * row
                    draw.text((5, yy), cohort['name'] + ' / ' + case['id'], fill='black')
                    files = [BUNDLE / case['input'], BUNDLE / case['target']] + [
                        RETURN / 'outputs' / label / (case['id'] + '.png')
                        for label in ['baseline'] + labels]
                    for col, path in enumerate(files):
                        relative = path.relative_to(ROOT).as_posix()
                        bindings[relative] = sha(path)
                        with Image.open(path) as source:
                            image = source.convert('RGB')
                            assert image.size == (256, 256)
                            sheet.paste(image, (5 + 268 * col, yy + 24))
                        source_cells.append({'file': relative, 'box': [5 + 268 * col,
                            yy + 24, 5 + 268 * col + 256, yy + 24 + 256]})
                        cells += 1
                        if col >= 2:
                            models.add(relative)
                filename = f'{group_number:02d}-{mode}.png'
                sheet.save(OUT / filename)
                pages.append({'file': filename, 'cohort': cohort['name'],
                    'reference': cases[0]['source_person_or_reference'], 'scope': mode,
                    'case_ids': [case['id'] for case in cases], 'headings': headings,
                    'size': list(sheet.size), 'sha256': sha(OUT / filename),
                    'cells': source_cells})
                assert time.monotonic() - start <= 120
    assert group_number == 20 and len(pages) == 60 and len(models) == 1000 and cells == 1800
    manifest = {'complete': True, 'protocol_sha256': PIN,
        'results_sha256': sha(RETURN / 'outputs/results.json'),
        'independent_audit_sha256': sha(AUDIT), 'generator_sha256': sha(Path(__file__)),
        'pages': pages, 'source_bindings': bindings, 'unique_model_outputs': 1000,
        'exact256_cells': cells, 'pages_count': len(pages), 'neural_calls': 0,
        'gradient_queries': 0, 'optimizer_updates': 0, 'visual_review_pending': True,
        'model_qualification': False, 'goal_complete': False,
        'seconds': time.monotonic() - start}
    with (OUT / 'gallery_manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({key: manifest[key] for key in ['complete', 'unique_model_outputs',
        'exact256_cells', 'pages_count', 'seconds']}, flush=True)


if __name__ == '__main__':
    main()

"""Create exact saved-pixel sheets after the frozen return audit; no inference."""
from pathlib import Path
import hashlib
import json
import time
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_group_conflicts_v1_vm'
RETURN = ROOT / 'outputs/cctv_dgp_group_conflicts_v1_vm_return'
OUT = ROOT / 'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/gallery'
AUDIT = ROOT / 'outputs/cctv_dgp_group_conflicts_v1_independent_audit.json'
PIN = 'bc0dcbd92b876c53ebe4f484cc5743fad7c8e7cdce6ca8222e04de3d485ec4b2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve prior review sheets'
    assert sha(BUNDLE / 'protocol.json') == PIN
    p, audit, result = read(BUNDLE / 'protocol.json'), read(AUDIT), read(RETURN / 'outputs/results.json')
    assert audit['complete'] and audit['protocol_sha256'] == result['protocol_sha256'] == PIN
    assert audit['all_saved_raw_and_PNG_metrics_recomputed']
    assert audit['all32_saved_group_vectors_and_certificate_arithmetic_verified']
    assert audit['CPU_replay_cases'] == 80 and audit['conditional_displacements_verified'] == 3
    assert result['gradient_queries'] == 160 and result['optimizer_updates'] == result['epochs'] == 0
    assert not result['native_or_DEV_or_final_used']
    labels = ['baseline'] + [trial['variant'] for trial in result['trial_summaries']]
    assert labels == ['baseline', 'common_1em05', 'common_1em04', 'common_1em03']
    cases_by_id = {case['id']: case for case in p['cases']}
    OUT.mkdir(parents=True)
    pages, bindings, model_files = [], {}, set()
    cells, number = 0, 0
    for cohort in p['cohorts']:
        for begin in range(0, 50, 5):
            number += 1
            cases = [cases_by_id[cid] for cid in cohort['case_ids'][begin:begin + 5]]
            assert len({case['source_person_or_reference'] for case in cases}) == 1
            sheet = Image.new('RGB', (1608, 1484), 'white')
            draw = ImageDraw.Draw(sheet)
            headings = ['input', 'paired target'] + labels
            for col, heading in enumerate(headings):
                draw.text((5 + 268 * col, 4), heading, fill='black')
            source_cells = []
            for row, case in enumerate(cases):
                yy = 24 + 292 * row
                draw.text((5, yy), cohort['name'] + ' / ' + case['id'], fill='black')
                paths = [BUNDLE / case['input'], BUNDLE / case['target']] + [
                    RETURN / 'outputs' / label / (case['id'] + '.png') for label in labels]
                for col, path in enumerate(paths):
                    name = path.relative_to(ROOT).as_posix()
                    bindings[name] = sha(path)
                    with Image.open(path) as image:
                        assert image.size == (256, 256)
                        sheet.paste(image.convert('RGB'), (5 + 268 * col, yy + 24))
                    box = [5 + 268 * col, yy + 24, 5 + 268 * col + 256, yy + 24 + 256]
                    source_cells.append({'file': name, 'box': box})
                    cells += 1
                    if col >= 2:
                        model_files.add(name)
            filename = f'{number:02d}-common.png'
            sheet.save(OUT / filename)
            pages.append({'file': filename, 'cohort': cohort['name'],
                'reference': cases[0]['source_person_or_reference'],
                'case_ids': [case['id'] for case in cases], 'headings': headings,
                'size': list(sheet.size), 'sha256': sha(OUT / filename), 'cells': source_cells})
            assert time.monotonic() - started < 120
    assert number == len(pages) == 20 and len(model_files) == 400 and cells == 600
    manifest = {'complete': True, 'protocol_sha256': PIN,
        'results_sha256': sha(RETURN / 'outputs/results.json'),
        'independent_audit_sha256': sha(AUDIT), 'generator_sha256': sha(Path(__file__)),
        'pages': pages, 'source_bindings': bindings, 'unique_model_outputs': 400,
        'exact256_cells': cells, 'pages_count': 20, 'neural_calls': 0,
        'gradient_queries': 0, 'optimizer_updates': 0, 'visual_review_pending': True,
        'model_qualification': False, 'goal_complete': False,
        'seconds': time.monotonic() - started}
    with (OUT / 'gallery_manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({key: manifest[key] for key in ['complete', 'unique_model_outputs',
        'exact256_cells', 'pages_count', 'seconds']}, flush=True)


if __name__ == '__main__':
    main()

"""Prepare and verify source-only Asian replay contact sheets; no model calls."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    parent = Path('outputs/cctv_dgp_vm_bundle_v1')
    out = Path('outputs/cctv_dgp_asian_source_review_v9')
    assert not out.exists(), 'Preserve the existing source review'
    assert sha(parent / 'protocol.json') == 'b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6'
    coverage = Path('outputs/cctv_dgp_training_coverage_v9_audit.json')
    assert sha(coverage) == 'd852d0e06cb92d56b741a42342bfee470b4413817912cf845bb27a50101dddea'
    rubric = Path('outputs/cctv_dgp_hq_cohort_plan_v2/source_review_rubric.json')
    assert sha(rubric) == '1e0aa441917280168a87f73509e1846b15ada15ee4d8b0e62ec35bf7cc6bad3e'
    protocol = json.loads((parent / 'protocol.json').read_text())
    refs = sorted([r for r in protocol['references'] if r['role'] == 'train' and r['source'] == 'dataset/asian_faces'], key=lambda r: r['id'])
    assert len(refs) == 451
    out.mkdir(parents=True)
    shutil.copyfile(rubric, out / 'source_review_rubric.json')
    rows = []; pages = []
    for offset in range(0, len(refs), 16):
        assert time.monotonic() - started < 120
        selected = refs[offset:offset + 16]
        sheet = Image.new('RGB', (1040, 1192), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 3), 'Source-only replay review: original training target pixels; no model outputs', fill='black')
        page_rows = []
        for i, ref in enumerate(selected):
            for name in [ref['target'], ref['native'], ref['observed']]:
                assert sha(parent / name) == protocol['assets_sha256'][name]
            with Image.open(parent / ref['target']) as image:
                assert image.size == (256, 256) and image.mode == 'RGB'
                x, y = (i % 4) * 260 + 2, (i // 4) * 292 + 24
                draw.text((x, y + 2), ref['id'], fill='black')
                draw.text((x, y + 16), 'captured: ' + str(ref['native_size']), fill='black')
                sheet.paste(image, (x, y + 34))
            row = {'reference_id': ref['id'], 'original_role': 'train',
                'native': str(parent / ref['native']), 'target': str(parent / ref['target']),
                'native_size': ref['native_size'], 'source_sha256': ref['source_sha256'],
                'target_sha256': protocol['assets_sha256'][ref['target']],
                'observed_sha256': protocol['assets_sha256'][ref['observed']]}
            page_rows.append(row)
        page = out / f'source_review_{offset // 16 + 1:02d}.png'
        sheet.save(page)
        actual = np.asarray(Image.open(page).convert('RGB'))
        for i, row in enumerate(page_rows):
            x, y = (i % 4) * 260 + 2, (i // 4) * 292 + 24 + 34
            np.testing.assert_array_equal(actual[y:y + 256, x:x + 256], np.asarray(Image.open(row['target']).convert('RGB')))
            row['review_page'] = page.name
            rows.append(row)
        pages.append({'path': page.name, 'sha256': sha(page), 'reference_ids': [r['reference_id'] for r in page_rows]})
    receipt = {'complete': True, 'version': 9, 'parent_protocol_sha256': sha(parent / 'protocol.json'),
        'coverage_audit_sha256': sha(coverage), 'rubric_sha256': sha(rubric), 'rows': rows, 'pages': pages,
        'source_candidates': 451, 'pages_prepared': 29, 'all_page_cells_exact_target_pixels': True,
        'source_only': True, 'review_complete': False, 'validation_used': False, 'native_reserved_used': False,
        'local_model_forwards': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'seconds': time.monotonic() - started, 'runtime_cap_seconds': 120,
        'training_ready': False, 'execution_source_sha256': sha(Path(__file__)),
        'interpretation': 'Original small-source targets are replay candidates, not newly captured HQ truth. Existing source-only rubric applies; source dimensions do not establish usable detail. Inspect ambiguous original native images before final decisions.'}
    write(out / 'catalog.json', receipt)
    print({k: v for k, v in receipt.items() if k not in ('rows', 'pages', 'interpretation')}, flush=True)


if __name__ == '__main__':
    main()

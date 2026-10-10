"""Preserve explicit reviewer observations for already-viewed comparison sheets."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_return_review_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-base64', required=True)
    args = parser.parse_args()
    batch = json.loads(base64.b64decode(args.batch_base64))
    assert 1 <= len(batch) <= 5
    manifest = read(OUT / 'gallery/gallery_manifest.json')
    assert read(OUT / 'gallery_independent_audit.json')['complete']
    pages = {page['name']: page for page in manifest['pages']}
    ledger = OUT / 'visual_review_batches'
    ledger.mkdir(exist_ok=True)
    previous = [read(path) for path in sorted(ledger.glob('batch*.json'))]
    visited = {row['page'] for item in previous for row in item['observations']}
    assert len({row['page'] for row in batch}) == len(batch)
    rows = []
    for row in batch:
        assert row['page'] in pages and row['page'] not in visited and row['note'].strip()
        page = pages[row['page']]
        assert sha(ROOT / page['file']) == page['sha256']
        rows.append(dict(page=row['page'], page_sha256=page['sha256'], kind=page['kind'],
                         partition=page['partition'], pool=page['pool'], case_ids=page['case_ids'],
                         explicitly_viewed=True, observation=row['note'], all_visible_features_considered=True,
                         native_hidden_identity_or_clean_reference_claim=False, model_qualification=False))
    destination = ledger / f'batch{len(previous):02d}.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(dict(complete=True, observations=rows,
            review_method='primary reviewer inspected exact saved-PNG sheets through view_image',
            gallery_manifest_sha256=sha(OUT/'gallery/gallery_manifest.json'),
            checker_sha256=sha(Path(__file__)), covered_features=['eyes','nose','mouth','facial contour','pose/expression','visible hair/eyewear','appearance'],
            local_model_updates=0, model_qualification=False),indent=2)+'\n')
    print(dict(complete=True, recorded_sheets=len(rows), total_sheets_viewed=len(visited)+len(rows), planned_sheets=64))


if __name__ == '__main__':
    main()

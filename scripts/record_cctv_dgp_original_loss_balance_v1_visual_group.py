"""Record one actually inspected group; never infer visual review from metrics."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1'
GALLERY = OUT / 'gallery'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--note', required=True)
    args = parser.parse_args()
    assert 1 <= args.group <= 20 and len(args.note.strip()) >= 30
    manifest_path = GALLERY / 'gallery_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    pages = [page for page in manifest['pages'] if page['file'].startswith(f'{args.group:02d}-')]
    assert len(pages) == 3 and len({tuple(page['case_ids']) for page in pages}) == 1
    assert len({page['reference'] for page in pages}) == 1
    for page in pages:
        assert sha(GALLERY / page['file']) == page['sha256']
    folder = OUT / 'observations'
    folder.mkdir(exist_ok=True)
    record = {'complete': True, 'group_index': args.group,
        'reviewer': 'Primary assistant development review',
        'independent_final_reviewer': False, 'viewed_at_original_resolution': True,
        'all_three_modes_and_three_scales_viewed': True,
        'cohort': pages[0]['cohort'], 'reference': pages[0]['reference'],
        'case_ids': pages[0]['case_ids'],
        'pages_sha256': {page['file']: page['sha256'] for page in pages},
        'gallery_manifest_sha256': sha(manifest_path), 'note': args.note,
        'recorded_UTC': datetime.now(timezone.utc).isoformat(),
        'record_script_sha256': sha(Path(__file__)),
        'neural_calls': 0, 'optimizer_updates': 0, 'model_qualification': False}
    with (folder / f'{args.group:02d}.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({'recorded_group': args.group, 'actually_viewed_pages': len(pages)}, flush=True)


if __name__ == '__main__':
    main()

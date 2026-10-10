"""Bind one actually inspected sheet to a human-readable development observation."""
import argparse
from datetime import datetime, timezone
from prepare_cctv_dgp_group_conflicts_v1_visual_review import ROOT, OUT, sha, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--note', required=True)
    args = parser.parse_args()
    assert 1 <= args.group <= 20 and len(args.note.strip()) >= 30
    manifest_path = OUT / 'gallery_manifest.json'
    manifest = read(manifest_path)
    page = manifest['pages'][args.group - 1]
    assert page['file'] == f'{args.group:02d}-common.png'
    assert sha(OUT / page['file']) == page['sha256']
    record = {'complete': True, 'group_index': args.group,
        'reviewer': 'Primary assistant development review', 'independent_final_reviewer': False,
        'viewed_at_original_resolution': True, 'all_three_reset_scales_viewed': True,
        'cohort': page['cohort'], 'reference': page['reference'], 'case_ids': page['case_ids'],
        'page_sha256': page['sha256'], 'gallery_manifest_sha256': sha(manifest_path),
        'note': args.note, 'recorded_UTC': datetime.now(timezone.utc).isoformat(),
        'record_script_sha256': sha(ROOT / 'scripts/record_cctv_dgp_group_conflicts_v1_visual_group.py'),
        'neural_calls': 0, 'optimizer_updates': 0, 'model_qualification': False}
    folder = OUT.parent / 'observations'
    folder.mkdir(exist_ok=True)
    import json
    with (folder / f'{args.group:02d}.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({'recorded_group': args.group, 'actually_viewed_pages': 1}, flush=True)


if __name__ == '__main__':
    main()

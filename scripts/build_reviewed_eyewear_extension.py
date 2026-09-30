"""Package accepted pilot annotations with immutable V3 records; no training."""
import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dataset/detector_training_extension_v1')
    args = parser.parse_args()
    base_path = Path('dataset/detector_glare_review_v3/manifest.json')
    proposal_path = Path('outputs/eyewear_annotation_proposals_v2/manifest.json')
    audit_path = proposal_path.parent / 'overlap_audit.json'
    base = json.loads(base_path.read_text())
    proposal = json.loads(proposal_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert sha(audit_path) == proposal['overlap_audit_sha256']
    assert audit['reference_hashes'][str(base_path)] == sha(base_path)
    for path, digest in audit['reference_hashes'].items():
        assert sha(path) == digest, f'Stale reference: {path}'
    assert len(proposal['records']) == 3
    checks = {r['index']: r for r in audit['records']}
    copies = []
    rows = copy.deepcopy(base['records'])
    original_hashes = {r['image_sha256'] for r in rows}
    original_groups = {r['group'] for r in rows}
    for r in rows:
        for key in ('image', 'mask'):
            source = base_path.parent / r[key]
            assert sha(source) == r[key + '_sha256']
            copies.append((source, r[key]))
    split_path = Path('outputs/downloaded_phase4/outputs/phase4_with_progress/split.json')
    split = json.loads(split_path.read_text())
    train = {p.replace('\\', '/') for p in split['train']}
    held = {p.replace('\\', '/') for p in split['validation']}
    for r in proposal['records']:
        assert r['reviewed'] is True and not checks[r['index']]['flags']
        assert r['path'] in train and r['path'] not in held
        assert sha(r['path']) == r['sha256']
        assert r['image_sha256'] not in original_hashes and r['group'] not in original_groups
        name = f"eyewear_added_{r['index']:03}.png"
        row = {'image': f'images/{name}', 'mask': f'masks/{name}',
               'image_sha256': r['image_sha256'], 'mask_sha256': r['mask_sha256'],
               'source': r['path'], 'source_sha256': r['sha256'], 'group': r['group'],
               'split': 'train', 'kind': 'covered' if r['positive_pixels'] else 'uncovered',
               'reviewed': True, 'reviewer': r['reviewer'],
               'annotation_quality': 'Approximate assistant pilot polygons, not expert ground truth',
               'occlusion_stratum': r['kind'], 'native_size': r['native_size'],
               'native_polygons': r['native_polygons'], 'resize': r['resize'],
               'identity_disjointness': 'not verified'}
        for key in ('image', 'mask'):
            source = Path(r[key])
            assert sha(source) == r[key + '_sha256']
            copies.append((source, row[key]))
        rows.append(row)
    assert rows[:len(base['records'])] == base['records']
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    for source, relative in copies:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
    manifest = {**base, 'records': rows,
                'extension': {'base_manifest_sha256': sha(base_path),
                              'proposal_manifest_sha256': sha(proposal_path),
                              'overlap_audit_sha256': sha(audit_path),
                              'added_training_records': 3,
                              'experiment_authorized_by_this_file': False,
                              'note': 'Annotation package only; existing runners do not enforce this metadata. No new training recipe yet.'}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'records': len(rows), 'split_counts': dict(Counter(r['split'] for r in rows)),
              'all_original_records_unchanged': rows[:100] == base['records'],
              'manifest_sha256': sha(out / 'manifest.json'),
              'all_copied_hashes_verified': True, 'training_performed': False}
    (out / 'build_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()

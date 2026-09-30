"""Add reviewed real-source mask examples to the frozen eyewear extension."""
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector_training import load_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base_path = Path('dataset/detector_training_extension_v1/manifest.json')
    assert sha(base_path) == '12adcc9ad159fd92985fbd458673551350f24d375b2f8300a8a245b12463ee57'
    load_manifest(base_path)
    base = json.loads(base_path.read_text())
    prop_path = Path('outputs/real_expansion_proposals_v3/manifest.json')
    proposal = json.loads(prop_path.read_text())
    audit_path = prop_path.parent / 'overlap_audit.json'
    assert sha(audit_path) == proposal['overlap_audit_sha256']
    audit = json.loads(audit_path.read_text())
    for path, digest in audit['reference_hashes'].items():
        assert sha(path) == digest
    flags = {r['index']: r['flags'] for r in audit['records']}
    pool = json.loads(Path('outputs/real_source_pool_audit/results.json').read_text())
    eligible = {r['path']: r for r in pool['records'] if r['candidate_for_visual_review']}
    selected = [r for r in proposal['records'] if r['reviewed']]
    assert [r['index'] for r in selected] == [0, 2]
    rows = copy.deepcopy(base['records'])
    copies = [(base_path.parent / r[k], r[k]) for r in rows for k in ('image', 'mask')]
    for r in selected:
        assert not flags[r['index']]
        assert sha(r['source']) == r['source_sha256'] == eligible[r['source']]['sha256']
        assert all(r['group'] != x['group'] and r['image_sha256'] != x['image_sha256'] for x in rows)
        name = f"mixed_added_{r['index']:03}.png"
        row = {key: r[key] for key in ('source', 'source_sha256', 'group', 'image_sha256',
                                      'mask_sha256', 'crop_ltrb', 'native_polygons', 'reviewer')}
        row.update(image=f'images/{name}', mask=f'masks/{name}', split='train',
                   kind='covered', reviewed=True, occlusion_stratum=r['kind'],
                   annotation_quality=r['annotation'], identity_disjointness='not verified')
        for key in ('image', 'mask'):
            assert sha(r[key]) == r[key + '_sha256']
            copies.append((Path(r[key]), row[key]))
        rows.append(row)
    out = Path('dataset/detector_training_extension_v2')
    out.mkdir(exist_ok=False)
    for source, rel in copies:
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
    result = {**base, 'records': rows, 'mixed_extension': {
        'base_manifest_sha256': sha(base_path), 'proposal_manifest_sha256': sha(prop_path),
        'overlap_audit_sha256': sha(audit_path), 'new_training_sources': 2,
        'training_recipe_ready': False, 'readiness_note': 'Documentary metadata, not enforced by legacy runners'}}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    loaded = load_manifest(out / 'manifest.json')
    original = json.loads(Path('dataset/detector_glare_review_v3/manifest.json').read_text())
    assert rows[:100] == original['records']
    report = {'manifest_sha256': sha(out / 'manifest.json'), 'record_count': len(loaded),
              'split_counts': dict(Counter(r['split'] for r in loaded)),
              'original_v3_records_unchanged': True, 'actual_loader_passed': True,
              'training_performed': False}
    (out / 'build_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()

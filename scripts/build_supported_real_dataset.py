"""Build a separate portable real-mask registry; no fitting or original edits."""
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector_training import load_manifest
from supported_real_data import FORMAT, load_supported_manifest, require, sha

PREVIOUS = ROOT/'dataset/detector_training_extension_v2/manifest.json'
PROPOSALS = ROOT/'outputs/real_reflection_proposals_v4_refined'
REVIEW = ROOT/'outputs/real_reflection_review_v4'
OUTPUT = ROOT/'dataset/detector_supported_review_v1'
DECISIONS_SHA = 'e6eef67417fa9a152cf7277969a5cae83b8949f5f1038cdadb8bd140c08186ea'
PROPOSAL_SHA = '4d75620863314e7ff26e9c3a4abf3e28a99be46744f7f08730e9af697b9ffe91'
VERIFICATION_SHA = 'e820314b23913fbb59d8aa837ff776b15d0736f9ad4e3ffdbf7cd760b449e0bc'


def attach_file(record, key, source, destination):
    require(destination.resolve().is_relative_to(OUTPUT) and not destination.exists(), 'New portable artifact required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    record[key] = destination.relative_to(OUTPUT).as_posix()
    record[key+'_sha256'] = sha(destination)
    require(record[key+'_sha256'] == sha(source), 'Copied artifact changed')


def main():
    require(not OUTPUT.exists(), 'Preserve completed/partial supported dataset')
    require(sha(PROPOSALS/'annotation_decisions.json') == DECISIONS_SHA
            and sha(PROPOSALS/'manifest.json') == PROPOSAL_SHA
            and sha(PROPOSALS/'verification.json') == VERIFICATION_SHA, 'Reviewed annotation lineage changed')
    decisions = json.loads((PROPOSALS/'annotation_decisions.json').read_text())
    proposals = json.loads((PROPOSALS/'manifest.json').read_text())
    verification = json.loads((PROPOSALS/'verification.json').read_text())
    qualification = json.loads((REVIEW/'manifest.json').read_text())
    require(verification['complete'] is True and decisions['accepted_annotation_count'] == 10
            and decisions['training_recipe_ready'] is False and decisions['training_enabled'] is False
            and decisions['source_proposal_manifest_sha256'] == PROPOSAL_SHA
            and decisions['independent_support_audit_sha256'] == VERIFICATION_SHA
            and proposals['source_review_sha256'] == sha(REVIEW/'manifest.json')
            and all(sha(ROOT/name) == digest for name, digest in qualification['input_sha256'].items()),
            'Annotation/source qualification scope differs')
    previous = json.loads(PREVIOUS.read_text())
    previous_rows = previous['records']; validated = load_manifest(PREVIOUS)
    require(len(previous_rows) == len(validated) == 105 and Counter(r['split'] for r in previous_rows)
            == {'train': 73, 'validation': 25, 'test': 7}, 'Previous fixed real membership differs')
    require(sha(PREVIOUS) == qualification['input_sha256'][PREVIOUS.relative_to(ROOT).as_posix()], 'Previous accepted manifest changed')
    accepted = {r['index']: r for r in decisions['accepted_annotations']}
    eligible = [r for r in proposals['records'] if r['index'] in accepted]
    require(len(accepted) == len(eligible) == 10 and Counter(r['status'] for r in eligible)
            == {'proposed_covering': 4, 'proposed_clear': 6}
            and {r['index'] for r in eligible}.isdisjoint(decisions['pending_native_sources']), 'Approved/pending membership differs')
    previous_sources = {r['source_sha256'] for r in previous_rows}
    require(len({r['source_sha256'] for r in eligible}) == 10
            and not (previous_sources & {r['source_sha256'] for r in eligible}), 'New native source is already used')
    for record in eligible:
        decision = accepted[record['index']]
        require(decision['status'] == 'accepted_for_supported_detector_dataset' and decision['reviewed'] is True
                and decision['requires_valid_support'] is True and decision['unknown_regions_are_targets'] is False
                and decision['source_sha256'] == record['source_sha256'] and decision['files'] == record['files']
                and sha(ROOT/record['source']) == record['source_sha256'], 'Native accepted scope/source changed')
        require(all(sha(PROPOSALS/f['path']) == f['sha256'] for f in record['files'].values()), 'Native approved artifact changed')
    require(len({Path(r['image']).name.casefold() for r in previous_rows}) == 105
            and len({Path(r['mask']).name.casefold() for r in previous_rows}) == 105, 'Portable parent basenames collide')
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in
              (PREVIOUS, PROPOSALS/'manifest.json', PROPOSALS/'verification.json',
               PROPOSALS/'annotation_decisions.json', REVIEW/'manifest.json', ROOT/'OCCLUSION_POLICY_V3.md')}
    OUTPUT.mkdir(); (OUTPUT/'support').mkdir()
    full = OUTPUT/'support/full.png'; Image.fromarray(np.full((256, 256), 255, np.uint8)).save(full)
    full_sha = sha(full); rows = []
    for i, original in enumerate(previous_rows):
        row = copy.deepcopy(original)
        row.update(support_required=True, data_origin='previous_reviewed', parent_index=i,
                   previous_record=copy.deepcopy(original), valid='support/full.png', valid_sha256=full_sha,
                   source_valid='support/full.png', source_valid_sha256=full_sha,
                   geometry='Previous accepted square PNG unchanged; full supervision support')
        for key in ('image', 'mask'):
            attach_file(row, key, Path(validated[i][key+'_path']),
                        OUTPUT/({'image': 'images', 'mask': 'masks'}[key])/'base'/Path(original[key]).name)
            require(row[key+'_sha256'] == original[key+'_sha256'], 'Original real pixel hash changed')
        rows.append(row)
    for original in eligible:
        index = original['index']; decision = accepted[index]
        row = {'reviewed': True, 'support_required': True, 'data_origin': 'native_supported_extension',
               'source_index': index, 'group': original['group'], 'source': original['source'],
               'source_sha256': original['source_sha256'], 'split': 'train',
               'kind': 'covered' if original['status'] == 'proposed_covering' else 'uncovered',
               'reviewer': decision['reviewer'], 'annotation_quality': decision['annotation_quality'],
               'review_rationale': decision['rationale'], 'occlusion_stratum': original['proposed_stratum'],
               'native_size': original['native_size'], 'native_polygons': original['native_polygons'],
               'unknown_native_polygons': original['unknown_native_polygons'],
               'affine': original['affine'], 'geometry': original['geometry'],
               'annotation_scope': original['annotation_scope'], 'uncovered_face_reference': None,
               'high_resolution_reference': False, 'identity_disjointness': 'not verified',
               'approved_record': copy.deepcopy(original)}
        for key, field, relative in (('image', 'images', f'images/native_{index:03}.png'),
                                    ('mask', 'masks', f'masks/native_{index:03}.png'),
                                    ('valid', 'valid', f'support/native_{index:03}_valid.png'),
                                    ('source_valid', 'source_valid', f'support/native_{index:03}_source.png')):
            attach_file(row, key, PROPOSALS/original['files'][field]['path'], OUTPUT/relative)
            require(row[key+'_sha256'] == original['files'][field]['sha256'], 'New approved data bytes changed')
        rows.append(row)
    require(all(sha(ROOT/name) == digest for name, digest in inputs.items()), 'Inputs changed while exporting dataset')
    counts = {split: dict(Counter(r['kind'] for r in rows if r['split'] == split))
              for split in ('train', 'validation', 'test')}
    require(counts == {'train': {'covered': 51, 'uncovered': 32},
                       'validation': {'covered': 15, 'uncovered': 10}, 'test': {'covered': 4, 'uncovered': 3}}, 'Supported membership differs')
    data = {'format': FORMAT, 'date': '2026-10-02', 'size': 256, 'dataset_reviewed': True,
            'support_semantics': 'valid_one_is_supervised', 'supported_records': rows, 'counts': counts,
            'input_sha256': inputs, 'previous_manifest_sha256': sha(PREVIOUS),
            'accepted_annotation_decisions_sha256': DECISIONS_SHA,
            'code_sha256': {p: sha(ROOT/p) for p in ('supported_real_data.py', 'scripts/build_supported_real_dataset.py',
                                                   'tests/test_supported_real_data.py')},
            'previous_records_preserved': 105, 'new_training_records': 10,
            'pending_sources_excluded': decisions['pending_native_sources'], 'training_recipe_ready': False,
            'model_forward_passes': 0, 'optimizer_updates_locally': 0, 'promoted': False,
            'consumer_contract': 'Explicit SupportedMasks image/mask/valid triples; every supervised reduction must use valid.',
            'limitations': ['Approximate assistant real masks; no expert adjudication or uncovered-face ground truth.',
                           'No identity-disjoint proof; held-out image/mask bytes and original gates remain fixed.',
                           'Source171 lower helmet/strap is excluded from supervision but retained as RGB context.',
                           'Data readiness does not establish repaired retention, generalization or model quality.']}
    path = OUTPUT/'manifest.json'; path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    _, checked = load_supported_manifest(path)
    require(len(checked) == 115, 'Portable supported dataset verification failed')
    print(json.dumps({'dataset_records': len(checked), 'counts': counts, 'previous_records_preserved': 105,
                      'new_training_records': 10, 'pending_sources_excluded': 14, 'training_recipe_ready': False,
                      'manifest_sha256': sha(path), 'optimizer_updates_locally': 0}), flush=True)


if __name__ == '__main__':
    main()

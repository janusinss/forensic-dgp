"""Independently check portable supported labels; no model execution or fitting."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = ROOT/'dataset/detector_supported_review_v1'
OUTPUT = ROOT/'outputs/supported_real_dataset_validation_v1'
MANIFEST_SHA = '860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace'
PREVIOUS = ROOT/'dataset/detector_training_extension_v2/manifest.json'
PROPOSALS = ROOT/'outputs/real_reflection_proposals_v4_refined'
REVIEW = ROOT/'outputs/real_reflection_review_v4'
FIELDS = ('image', 'mask', 'valid', 'source_valid')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_parent_record(original, row):
    """Lineage alone is insufficient: active labels and basename must match."""
    require(row.get('previous_record') == original, 'Original record lineage changed')
    for key, value in original.items():
        if key in ('image', 'mask'):
            basename = PurePosixPath(value).name
            directory = 'images' if key == 'image' else 'masks'
            require(row.get(key) == f'{directory}/base/{basename}', 'Original basename or portable path changed')
        else:
            require(row.get(key) == value, 'Active original metadata changed: '+key)


def check_original_support(valid, source):
    require(all(isinstance(a, np.ndarray) and a.dtype == np.bool_ and a.ndim == 2
                and a.size > 0 for a in (valid, source)) and valid.shape == source.shape
            and valid.all() and source.all(), 'Original difficult pixels or full support changed')


def binary(path):
    with Image.open(path) as image:
        array = np.array(image)
    require(array.ndim == 2 and array.dtype == np.uint8 and np.isin(array, [0, 255]).all(),
            'Binary grayscale byte artifact required')
    return array.astype(bool)


def polygons(points, width, height):
    canvas = Image.new('L', (width, height))
    for polygon in points:
        array = np.asarray(polygon, np.float64)
        require(array.ndim == 2 and array.shape[1] == 2 and len(array) >= 3
                and np.isfinite(array).all() and np.equal(array, array.astype('int64')).all()
                and np.all(array >= 0) and np.all(array[:, 0] < width)
                and np.all(array[:, 1] < height), 'Invalid native annotation coordinates')
        ImageDraw.Draw(canvas).polygon([tuple(int(v) for v in p) for p in array], fill=255)
    return np.array(canvas).astype(bool)


def main():
    require(not OUTPUT.exists(), 'Preserve completed/partial independent dataset verification')
    path = DATASET/'manifest.json'
    require(sha(path) == MANIFEST_SHA, 'Supported registry changed before independent audit')
    data = json.loads(path.read_text(encoding='utf-8'))
    require(data['format'] == 'dgp-supported-real-masks-v1' and 'records' not in data
            and data['dataset_reviewed'] is True and data['size'] == 256
            and data['support_semantics'] == 'valid_one_is_supervised'
            and data['training_recipe_ready'] is False and data['promoted'] is False
            and data['model_forward_passes'] == data['optimizer_updates_locally'] == 0,
            'Dataset support/scope declaration changed')
    require(all(sha(ROOT/p) == h for p, h in {**data['input_sha256'], **data['code_sha256']}.items())
            and sha(PREVIOUS) == data['previous_manifest_sha256'], 'Input or reader/builder binding changed')
    previous = json.loads(PREVIOUS.read_text(encoding='utf-8'))['records']
    proposals = json.loads((PROPOSALS/'manifest.json').read_text(encoding='utf-8'))
    decisions = json.loads((PROPOSALS/'annotation_decisions.json').read_text(encoding='utf-8'))
    qualification = json.loads((REVIEW/'manifest.json').read_text(encoding='utf-8'))
    require(all(sha(ROOT/p) == h for p, h in qualification['input_sha256'].items())
            and sha(REVIEW/'reference_signatures.json') == qualification['reference_signatures_sha256']
            and qualification['overlap_flagged_sources'] == 0 and not qualification['within_candidate_pairs']
            and sha(REVIEW/'manifest.json') == proposals['source_review_sha256']
            and sha(REVIEW/'visual_review.json') == proposals['source_visual_review_sha256']
            and sha(PROPOSALS/'annotation_decisions.json') == data['accepted_annotation_decisions_sha256']
            and decisions['source_proposal_manifest_sha256'] == sha(PROPOSALS/'manifest.json')
            and decisions['independent_support_audit_sha256'] == sha(PROPOSALS/'verification.json')
            and decisions['policy_sha256'] == sha(ROOT/'OCCLUSION_POLICY_V3.md')
            and all(sha(PROPOSALS/p) == h for p, h in decisions['reviewed_preview_sha256'].items()),
            'Qualification or native annotation acceptance differs')
    require(decisions['training_recipe_ready'] is False and decisions['training_enabled'] is False
            and decisions['accepted_annotation_count'] == 10
            and len(previous) == data['previous_records_preserved'] == 105,
            'Original/approved scope changed')
    rows = data['supported_records']
    expected_counts = {'train': {'covered': 51, 'uncovered': 32},
                       'validation': {'covered': 15, 'uncovered': 10}, 'test': {'covered': 4, 'uncovered': 3}}
    counts = {s: dict(Counter(r['kind'] for r in rows if r['split'] == s)) for s in expected_counts}
    require(len(rows) == 115 and counts == data['counts'] == expected_counts,
            'Supported real split membership differs')
    files = {'manifest.json'}; hashes = {}; owner = {'group': {}, 'source_sha256': {}}
    source_hashes = set(); image_hashes = set()
    for row in rows:
        require(row['reviewed'] is True and row['support_required'] is True, 'Unreviewed/unsupported record included')
        for key in FIELDS:
            relative = row[key]; file = (DATASET/relative).resolve()
            require(PurePosixPath(relative).as_posix() == relative and '\\' not in relative
                    and file.is_relative_to(DATASET) and sha(file) == row[key+'_sha256'],
                    'Portable copied artifact changed: '+key)
            files.add(relative); hashes[relative] = sha(file)
        require(row['image_sha256'] not in image_hashes, 'Duplicate exact image across dataset')
        image_hashes.add(row['image_sha256'])
        for key in owner:
            require(owner[key].get(row[key], row['split']) == row['split'], 'Cross-split source/group membership')
            owner[key][row[key]] = row['split']
        source_hashes.add(row['source_sha256'])
        require(sha(ROOT/row['source'].replace('\\', '/')) == row['source_sha256'], 'Original raw source changed')
    for index, (original, row) in enumerate(zip(previous, rows[:105])):
        check_parent_record(original, row)
        require(row['data_origin'] == 'previous_reviewed' and row['parent_index'] == index
                and row['valid'] == row['source_valid'] == 'support/full.png', 'Parent order/support mapping differs')
        for key in ('image', 'mask'):
            require((DATASET/row[key]).read_bytes() == (PREVIOUS.parent/original[key]).read_bytes(),
                    'Original image or mask bytes changed')
        check_original_support(binary(DATASET/row['valid']), binary(DATASET/row['source_valid']))
    annotations = {r['index']: r for r in proposals['records']}
    approved = {r['index']: r for r in decisions['accepted_annotations']}
    require(len(approved) == 10 and {r['source_index'] for r in rows[105:]} == set(approved)
            and data['new_training_records'] == 10, 'Approved annotation membership differs')
    pending = set(decisions['pending_native_sources'])
    require(len(pending) == 14 and pending == set(data['pending_sources_excluded'])
            and not (source_hashes & {annotations[i]['source_sha256'] for i in pending}),
            'Pending source became a clear or supervised target')
    native_counts = []
    for row in rows[105:]:
        index = row['source_index']; original = annotations[index]; decision = approved[index]
        require(row['approved_record'] == original and row['split'] == 'train'
                and row['data_origin'] == 'native_supported_extension'
                and decision['status'] == 'accepted_for_supported_detector_dataset'
                and decision['reviewed'] is True and decision['requires_valid_support'] is True
                and decision['unknown_regions_are_targets'] is False and decision['files'] == original['files']
                and decision['source_sha256'] == row['source_sha256'], 'Active approved annotation scope differs')
        fields = ('group', 'source', 'source_sha256', 'native_size', 'native_polygons',
                  'unknown_native_polygons', 'affine', 'geometry', 'annotation_scope',
                  'identity_disjointness', 'uncovered_face_reference', 'high_resolution_reference')
        require(all(row[k] == original[k] for k in fields)
                and row['kind'] == ('covered' if original['status'] == 'proposed_covering' else 'uncovered')
                and row['reviewer'] == decision['reviewer'] and row['annotation_quality'] == decision['annotation_quality']
                and row['review_rationale'] == decision['rationale']
                and row['occlusion_stratum'] == original['proposed_stratum'], 'Active native labels or review metadata differ')
        for key, field in zip(FIELDS, ('images', 'masks', 'valid', 'source_valid')):
            artifact = original['files'][field]
            require(row[key+'_sha256'] == artifact['sha256']
                    and (DATASET/row[key]).read_bytes() == (PROPOSALS/artifact['path']).read_bytes(),
                    'Reviewed artifact changed while copying')
        require(sha(ROOT/original['native_review_image']) == original['native_review_sha256'], 'Native RGB reference changed')
        with Image.open(ROOT/row['source']) as image:
            rgb = np.array(ImageOps.exif_transpose(image).convert('RGB'))
        with Image.open(ROOT/original['native_review_image']) as image:
            require(np.array_equal(rgb, np.array(image.convert('RGB'))), 'Native view differs from oriented raw input')
        height, width = rgb.shape[:2]
        require(row['native_size'] == [width, height], 'Native dimensions differ')
        target = polygons(row['native_polygons'], width, height)
        unknown = polygons(row['unknown_native_polygons'], width, height)
        require(not (target & unknown).any(), 'Known positive overlaps native uncertainty')
        scale = 255/(max(width, height)-1)
        affine = np.array([[scale, 0., (255-scale*(width-1))/2], [0., scale, (255-scale*(height-1))/2]])
        require(np.allclose(row['affine'], affine, rtol=0, atol=1e-12), 'Uniform pixel-center geometry differs')
        warp = {'borderMode': cv2.BORDER_CONSTANT, 'borderValue': 0}
        source = cv2.warpAffine(np.ones((height, width), np.float32), affine, (256, 256), flags=cv2.INTER_LINEAR, **warp) >= 1-1e-6
        valid = source & (cv2.warpAffine((~unknown).astype('float32'), affine, (256, 256), flags=cv2.INTER_LINEAR, **warp) >= 1-1e-6)
        mask = cv2.warpAffine(target.astype('uint8'), affine, (256, 256), flags=cv2.INTER_NEAREST, **warp).astype(bool)
        expected_rgb = cv2.warpAffine(rgb, affine, (256, 256), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(96, 96, 96))
        expected_rgb[~source] = 96
        with Image.open(DATASET/row['image']) as image:
            require(np.array_equal(expected_rgb, np.array(image.convert('RGB'))), 'Padding or unknown RGB was altered')
        require(np.array_equal(binary(DATASET/row['source_valid']), source)
                and np.array_equal(binary(DATASET/row['valid']), valid)
                and np.array_equal(binary(DATASET/row['mask']), mask & valid)
                and not (mask & ~valid).any(), 'Copied native target/source/loss support differs')
        numerical = {'native_positive_pixels': int(target.sum()), 'native_ignored_pixels': int(unknown.sum()),
                     'positive_pixels': int(mask.sum()), 'supervised_pixels': int(valid.sum()),
                     'source_support_pixels': int(source.sum())}
        require(all(original[k] == v for k, v in numerical.items()), 'Native/support pixel counts differ')
        native_counts.append({'source_index': index, **numerical, 'unknown_rgb_preserved': True})
    actual = {p.relative_to(DATASET).as_posix() for p in DATASET.rglob('*') if p.is_file()}
    require(actual == files and len(files) == 252, 'Missing/extra portable dataset artifacts')
    from detector_training import ReviewedMasks, load_manifest
    from supported_real_data import SupportedMasks, load_supported_manifest
    try:
        load_manifest(path)
    except KeyError as error:
        require(error.args == ('records',), 'Legacy consumer failed for an unrelated reason')
    else:
        raise ValueError('Legacy pair-only loader accepted partial labels')
    _, checked = load_supported_manifest(path)
    old_train = ReviewedMasks([r for r in load_manifest(PREVIOUS) if r['split'] == 'train'], 256)
    new_train = SupportedMasks(path, split='train')
    require(len(checked) == 115 and len(old_train) == 73 and len(new_train) == 83, 'Reader split differs')
    for i in range(73):
        old_x, old_m = old_train[i]; new_x, new_m, valid = new_train[i]
        require(torch.equal(old_x, new_x) and torch.equal(old_m, new_m) and bool(valid.eq(1).all()),
                'Original training tensors changed through new reader')
    mannequin = [r for r in rows[:105] if r['split'] == 'validation'][8]
    require(PurePosixPath(mannequin['image']).name == 'new_covered_40.png'
            and mannequin['kind'] == 'covered', 'Known mannequin lost its original separate-report identity')
    require(sha(path) == MANIFEST_SHA and all(sha(ROOT/p) == h for p, h in
            {**data['input_sha256'], **data['code_sha256'], **qualification['input_sha256']}.items()),
            'Bound inputs changed during independent audit')
    result = {'complete': True, 'date': '2026-10-02', 'script_sha256': sha(__file__),
              'test_sha256': sha(ROOT/'tests/test_supported_real_dataset_audit.py'),
              'manifest_sha256': MANIFEST_SHA, 'counts': counts, 'verified_file_count': len(files),
              'verified_file_sha256': hashes, 'original_records_and_bytes_preserved': 105,
              'original_full_support_preserved': 105, 'original_training_tensors_equal': 73,
              'new_training_records': native_counts, 'pending_sources_excluded': sorted(pending),
              'known_mannequin': {'validation_index': 8, 'image': mannequin['image'], 'parent_index': mannequin['parent_index']},
              'legacy_reader_rejected': True, 'model_forward_passes': 0, 'held_out_model_forward_passes': 0,
              'optimizer_updates_locally': 0, 'training_recipe_ready': False, 'promoted': False,
              'scope': 'Independent data/lineage/support check only; no model-quality, generalization or identity-disjointness claim.'}
    OUTPUT.mkdir(parents=True)
    with (OUTPUT/'verification.json').open('x', encoding='utf-8', newline='\n') as file:
        file.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('verified_file_sha256', 'new_training_records')}, indent=2), flush=True)


if __name__ == '__main__':
    main()

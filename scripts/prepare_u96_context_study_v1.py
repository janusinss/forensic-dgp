"""Prepare original-input-only paired DEV probes; no restoration or training.

The fixed source order comes from retained historical DEV references. Blur ROI
is a synthetic-corruption footprint, never an admitted protection mask.
"""
import ast
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_context_study_v1'
BUNDLE = ROOT / 'outputs/cctv_codeformer_adaptation_v2_vm'
REFERENCE_SHA = '55b892ee5d55a123ea6cb55f84f322824faaec956f59168e443ff94b93337cde'
IDS = tuple('va_ffhq_' + value for value in (
    '28544', '48646', '37228', '58579', '28565', '59814',
    '40375', '13682', '19791', '10436', '27263', '67465'))
ROLES = ('clear', 'global_gaussian_2p4', 'local_gaussian_3p0')
ROI_XYXY = (64, 64, 192, 192)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)}


def png(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format='PNG')
    return stream.getvalue()


def checked_source(relative, expected_sha):
    rel = Path(relative)
    assert not rel.is_absolute() and '..' not in rel.parts
    path = BUNDLE / rel
    assert path.is_file() and not path.is_symlink()
    if path.stat().st_file_attributes & 1024:
        raise ValueError('Reparse source path refused.')
    data = path.read_bytes()
    assert sha(data) == expected_sha, path
    return path, data


def blur(rgb, sigma):
    filtered = gaussian_filter(rgb.astype(np.float64), sigma=(sigma, sigma, 0),
                               mode='reflect', truncate=3.0)
    return np.rint(filtered).clip(0, 255).astype(np.uint8)


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve any previous preparation; output must be new.'
    ast.parse(Path(__file__).read_bytes(), filename=__file__)
    reference_path = BUNDLE / 'references.json'
    assert sha(reference_path.read_bytes()) == REFERENCE_SHA
    reference_document = json.loads(reference_path.read_bytes())
    refs = reference_document['references']
    lookup = {row['id']: row for row in refs}
    assert len(lookup) == len(refs)
    train = [row for row in refs if row['historical_role'] == 'train']
    train_ids = {row['id'] for row in train}
    train_hashes = {row['source_sha256'] for row in train}
    train_photos = {row['attribution']['photo_url'] for row in train}
    membership_path = ROOT / 'outputs/cctv_dgp_asian_source_review_v9/mixed_cohort.json'
    membership = json.loads(membership_path.read_bytes())
    old_train = set(membership['training_reference_ids'])
    old_dev = set(membership['validation_reference_ids'])
    prepared = []
    # Finish all input/source checks before creating outputs.
    for id_ in IDS:
        row = lookup[id_]
        assert row['historical_role'] == 'validation' and row['use_for_optimization'] is False
        assert row['decision'] == 'accept_clean_restoration'
        assert id_ in old_dev and id_ not in old_train and id_ not in train_ids
        assert row['source_sha256'] not in train_hashes
        assert row['attribution']['photo_url'] not in train_photos
        source, source_data = checked_source(row['source_path'], row['source_sha256'])
        target, target_data = checked_source(row['existing_target_path'], row['existing_target_sha256'])
        with Image.open(io.BytesIO(source_data)) as im:
            assert im.size == (1024, 1024)
            source_rgb = np.asarray(im.convert('RGB'))
            derived = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS))
        with Image.open(io.BytesIO(target_data)) as im:
            assert im.size == (256, 256) and im.mode == 'RGB'
            target_rgb = np.asarray(im.convert('RGB'))
        assert np.array_equal(derived, target_rgb), id_
        global_blur = blur(target_rgb, 2.4)
        local_blur = target_rgb.copy()
        local_blur[64:192, 64:192] = blur(target_rgb, 3.0)[64:192, 64:192]
        outside = np.ones((256, 256), dtype=bool)
        outside[64:192, 64:192] = False
        assert np.array_equal(local_blur[outside], target_rgb[outside])
        # The operation is deterministic without any RNG or clean-target mask selection.
        assert np.array_equal(global_blur, blur(target_rgb, 2.4))
        prepared.append((row, source, source_rgb, target, target_data, target_rgb,
                         {'clear': target_rgb, 'global_gaussian_2p4': global_blur,
                          'local_gaussian_3p0': local_blur}))
    OUT.mkdir()
    (OUT / 'inputs').mkdir()
    (OUT / 'targets').mkdir()
    (OUT / 'input_only_sheets').mkdir()
    cases = []
    sources = []
    for row, source, source_rgb, target, target_data, target_rgb, images in prepared:
        id_ = row['id']
        saved_target = OUT / 'targets' / (id_ + '.png')
        saved_target.write_bytes(target_data)
        assert saved_target.read_bytes() == target_data
        sources.append({'reference_id': id_, 'historical_role': 'validation', 'study_role': 'DEV',
            'source1024': pin(source), 'source_RGB8_sha256': sha(source_rgb.tobytes()),
            'target256_original': pin(target), 'target256_saved': pin(saved_target),
            'target_RGB8_sha256': sha(target_rgb.tobytes()), 'native_source_size': [1024, 1024],
            'attribution': row['attribution'], 'official_FFHQ_category': row['official_category'],
            'original_source_path': row['original_source_path'],
            'historical_source_decision': row['decision'],
            'historical_source_review_reason': row['source_review_reason'],
            'historical_source_reviewed_at_original_resolution': row['source_reviewed_at_original_resolution'],
            'source_to_target_RGB1024_LANCZOS256_exact': True,
            'new_source_appearance_review': 'PENDING', 'new_generated_input_review': 'PENDING',
            'optimization_permitted': False})
        for role, rgb in images.items():
            id_role = id_ + '__' + role
            path = OUT / 'inputs' / (id_role + '.png')
            path.write_bytes(target_data if role == 'clear' else png(rgb))
            if role == 'clear':
                assert path.read_bytes() == target_data
            with Image.open(path) as readback:
                assert readback.mode == 'RGB' and readback.size == (256, 256)
                assert np.array_equal(np.asarray(readback), rgb)
            cases.append({'id': id_role, 'reference_id': id_, 'split': 'DEV', 'role': role,
                'input': pin(path), 'input_RGB8_sha256': sha(rgb.tobytes()),
                'target': pin(saved_target), 'paired_clean_target': True,
                'input_only_readability': 'PENDING', 'pose_review': 'PENDING',
                'appearance_flags': 'PENDING', 'optimization_permitted': False,
                'local_outside_corruption_ROI_exact': True if role == 'local_gaussian_3p0' else None,
                'protection_mask_selected': False})
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    assert font_path.is_file()
    font = ImageFont.truetype(str(font_path), 16)
    small = ImageFont.truetype(str(font_path), 13)
    sheets = []
    for sheet_index in range(3):
        group = prepared[sheet_index * 4:(sheet_index + 1) * 4]
        canvas = Image.new('RGB', (812, 1220), (245, 243, 236))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), 'INPUT ONLY | U96 context study candidate review | DEV', fill='black', font=font)
        draw.text((12, 32), 'No model outputs or target column. Readability/pose/appearance decisions pending.', fill='black', font=small)
        draw.text((12, 52), 'Synthetic local blur ROI is a corruption footprint; no protection mask is selected.', fill='black', font=small)
        for col, role in enumerate(ROLES):
            draw.text((12 + col * 268, 78), role, fill='black', font=small)
        for row_index, (row, _, _, _, _, _, images) in enumerate(group):
            y = 110 + row_index * 278
            draw.text((12, y - 20), row['id'] + ' | historical DEV | review pending', fill='black', font=small)
            for col, role in enumerate(ROLES):
                x = 12 + col * 268
                canvas.paste(Image.fromarray(images[role]), (x, y))
                # Exact pixel pasting; no screenshot display resampling or enhancement.
                assert np.array_equal(np.asarray(canvas)[y:y + 256, x:x + 256], images[role])
        path = OUT / 'input_only_sheets' / f'{sheet_index + 1:02d}_DEV_candidates.png'
        canvas.save(path)
        sheets.append({'file': pin(path), 'reference_ids': [item[0]['id'] for item in group],
            'roles': list(ROLES), 'input_cells': 12, 'size': list(canvas.size),
            'exact256_pixel_pastes': True, 'review_status': 'PENDING'})
    record = {'format': 'U96-paired-context-study-input-preparation-v1', 'complete': True,
        'status': 'PREPARATION_ONLY_REQUIRES_ROOT_INPUT_REVIEW_AND_PROTECTION_SELECTION',
        'source_script': pin(Path(__file__)), 'reference_manifest': pin(reference_path),
        'historical_membership_manifest': pin(membership_path),
        'fixed_reference_order': list(IDS), 'reference_count': 12, 'case_count': 36,
        'sources': sources, 'cases': cases,
        'operator_specification': {
            'working_canvas': 'retained clean256 RGB8 target; no new alignment',
            'clear': 'exact retained target PNG bytes',
            'global_gaussian_2p4': {'function': 'scipy.ndimage.gaussian_filter',
                'sigma': [2.4, 2.4, 0], 'working_dtype': 'float64', 'mode': 'reflect',
                'truncate': 3.0, 'quantization': 'np.rint ties-to-even, clip[0,255], uint8'},
            'local_gaussian_3p0': {'function': 'scipy.ndimage.gaussian_filter',
                'sigma': [3.0, 3.0, 0], 'working_dtype': 'float64', 'mode': 'reflect',
                'truncate': 3.0, 'quantization': 'np.rint ties-to-even, clip[0,255], uint8',
                'replacement_ROI_XYXY_half_open': list(ROI_XYXY),
                'replacement': 'only blur RGB pixels at64<=x<192,64<=y<192; all outside pixels exact',
                'ROI_is_not_a_deployment_protection_mask': True}},
        'input_only_sheets': sheets,
        'sheet_font': {'absolute_path': str(font_path), 'bytes': font_path.stat().st_size,
                       'sha256': sha(font_path.read_bytes())},
        'overlap': {'first12_ids_in_historical_validation': True, 'first12_ids_in_historical_TRAIN': False,
            'source_SHA_overlap_with391TRAIN': False, 'photo_URL_overlap_with391TRAIN': False,
            'subject_identity_overlap': 'unknown; file/photo separation is not subject separation',
            'CodeFormer_FFHQ_pretraining_overlap': 'possible/unknown; not an unseen-identity evaluation',
            'repeated_DEV_exposure': True, 'reserved_FINAL_used': False},
        'invariants': {'clear_bytes_exact': True, 'source_to_target_pixel_derivation_exact': True,
            'local_outside_ROI_exact': True, 'gaussian_repeat_exact': True,
            'output_RGBA_or_resizing_enhancement': False},
        'root_review_status': 'PENDING', 'selected_for_inference': False,
        'readability_admitted': False, 'protection_masks_selected': False,
        'optimization_permitted': False, 'neural_calls': 0, 'gradient_calls': 0,
        'optimizer_updates': 0, 'training_calls': 0, 'VM_calls': 0,
        'model_improvement_claimed': False, 'study_completed': False,
        'versions': {'Python': platform.python_version(), 'NumPy': np.__version__,
                     'SciPy': scipy.__version__, 'Pillow': PILLOW_VERSION},
        'seconds': time.monotonic() - started}
    path = OUT / 'candidate_manifest_v1.json'
    path.write_text(json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')
    # Saved readback verifies every declared payload; the manifest excludes its own recursive pin.
    for row in sources:
        for key in ('source1024', 'target256_original', 'target256_saved'):
            assert pin(ROOT / row[key]['path']) == row[key]
    for row in cases:
        assert pin(ROOT / row['input']['path']) == row['input']
    for row in sheets:
        assert pin(ROOT / row['file']['path']) == row['file']
    print(json.dumps({'manifest': pin(path), 'prepared_references': 12, 'input_cases': 36,
                      'input_only_sheets': [row['file']['path'] for row in sheets],
                      'invariants_pass': True, 'readability_review': 'PENDING',
                      'protection_selection': 'PENDING', 'neural_training_VM_calls': 0}))


if __name__ == '__main__':
    main()

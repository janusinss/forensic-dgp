"""Prepare 96 fixed CPU synthetic stress pairs; never run a model or training."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_pairs_v1.filters import (
    PROFILES, apply_profile, profiles_document, require, seed_record)

SELECTION_SHA = 'f1e0fa95dd2140a70f5ac091ecf83d09f51ac80e6bc1eabd8acb9223594ced9e'
REFERENCE_IDS = {'TRAIN': ('tr_ffhq_00084', 'tr_ffhq_00178', 'tr_ffhq_00323', 'tr_ffhq_00616'),
    'DEV': ('va_ffhq_00383', 'va_ffhq_01093', 'va_ffhq_08025', 'va_ffhq_09056')}
MAX_FILES, MAX_BYTES = 256, 64 * 1024**2
SOURCE_FILES = ('scripts/prepare_codeformer_filter_pairs_v1.py',
    'research/codeformer_filter_pairs_v1/__init__.py',
    'research/codeformer_filter_pairs_v1/filters.py', 'research/codeformer_filter_pairs_v1/README.md')
INPUT_REVIEW_ORDER = ('global_mosaic_16', 'local_mosaic_16', 'local_gaussian_10p0', 'gaussian_6p0',
    'repeated_rows_4_blend0p75', 'gaussian_2p4', 'anisotropic_h5_v0p7', 'anisotropic_h0p7_v5',
    'global_mosaic_8', 'local_gaussian_3p0', 'local_mosaic_8', 'clear')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def read_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(data.decode('utf-8'), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def regular(path):
    path = Path(path).absolute()
    require(path.resolve(strict=True).is_relative_to(ROOT), 'Evidence must remain within workspace')
    for part in (path, *path.parents):
        if part == ROOT.parent:
            break
        info = part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
            'No symlink/reparse evidence')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Ordinary single-link file required')
    return path


def read_bound(path, fingerprint, size=None, cap=8 * 1024**2):
    path = regular(path)
    before = path.stat()
    require(before.st_size <= cap and (size is None or before.st_size == size), 'Exact bounded source size required')
    data = path.read_bytes(); after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
        (after.st_size, after.st_mtime_ns, after.st_ino), 'Source changed during read')
    require(digest(data) == fingerprint, 'Source byte fingerprint differs')
    return data


def from_pin(pin):
    path = ROOT / pin['path']
    require(path.absolute() == Path(pin['absolute_path']).absolute(), 'Physical/logical source path differs')
    return read_bound(path, pin['sha256'], pin['bytes'])


def rgb(data, size):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (size, size),
            'Exact square RGB PNG source/reference required')
        return np.asarray(image).copy()


def image_bytes(pixels):
    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
    data = stream.getvalue()
    with Image.open(io.BytesIO(data)) as image:
        require(np.array_equal(np.asarray(image), pixels), 'Encoded image pixel readback differs')
    return data


def input_only_sheet(profile_id, images):
    expected = [rid for split in ('TRAIN', 'DEV') for rid in REFERENCE_IDS[split]]
    require(profile_id in INPUT_REVIEW_ORDER and len(images) == 8 and [rid for rid, _, _ in images] == expected,
        'Exact eight ordered photographs in a single-profile input-only sheet required')
    sheet = Image.new('RGB', (1024, 612), 'white'); draw = ImageDraw.Draw(sheet)
    draw.text((4, 3), profile_id + ' | INPUT ONLY: pending usability review; clear profile reviewed last', fill='black')
    draw.text((4, 19), 'Exact256 panels. No clean reference, model output, enhancement or resampling.', fill='black')
    cells = []
    for index, (reference_id, split, pixels) in enumerate(images):
        x, y = (index % 4) * 256, 40 + (index // 4) * 286
        draw.text((x + 3, y + 3), reference_id + ' | ' + split, fill='black')
        sheet.paste(Image.fromarray(pixels), (x, y + 30))
        cells.append({'reference_id': reference_id, 'split': split, 'profile': profile_id, 'x': x, 'y': y + 30,
            'width': 256, 'height': 256, 'RGB8_sha256': digest(pixels.tobytes())})
    data = image_bytes(np.asarray(sheet))
    with Image.open(io.BytesIO(data)) as saved:
        readback = np.asarray(saved)
        for cell in cells:
            tile = readback[cell['y']:cell['y'] + 256, cell['x']:cell['x'] + 256]
            require(digest(tile.tobytes()) == cell['RGB8_sha256'], 'Input-only exact256 panel changed')
    return data, cells


def pixel_record(pixels):
    return {'shape': list(pixels.shape), 'dtype': str(pixels.dtype),
        'RGB8_sha256': digest(np.ascontiguousarray(pixels).tobytes()),
        'float32_sha256': digest((pixels.astype(np.float32) / np.float32(255)).tobytes())}


def check_pixels(pixels, metadata):
    require(digest(pixels.tobytes()) == metadata['rgb8']['pixel_sha256'] and
        digest((pixels.astype(np.float32) / np.float32(255)).tobytes()) == metadata['float32']['pixel_sha256'],
        'Recorded source/reference pixel fingerprint differs')


def validate_selection(selection, catalog):
    require(selection.get('complete') is True and selection['format'] ==
        'codeformer-filter-pairs-deterministic-source-cohort-v1', 'Completed prospective cohort required')
    refs = {row['id']: row for row in catalog['references']}
    require(len(refs) == 444 and Counter(r['historical_role'] for r in refs.values()) == {'train': 391, 'validation': 53},
        'Retained391 TRAIN/53 DEV reference catalog required')
    hashes = {role: {r['source_sha256'] for r in refs.values() if r['historical_role'] == role}
        for role in ('train', 'validation')}
    require(not hashes['train'] & hashes['validation'], 'Historical TRAIN/DEV source-byte overlap')
    cohort = selection['cohort']
    require(len(cohort) == 8 and Counter(r['split'] for r in cohort) == {'TRAIN': 4, 'DEV': 4},
        'Only the approved4 TRAIN+4 DEV diagnostic cohort allowed')
    require(len({r['reference_id'] for r in cohort}) == 8 and
        len({r['source1024']['sha256'] for r in cohort}) == 8, 'Duplicate source photograph')
    for split, ids in REFERENCE_IDS.items():
        require(tuple(r['reference_id'] for r in cohort if r['split'] == split) == ids,
            'Prospectively locked reference membership/order changed')
    for row in cohort:
        reference = refs[row['reference_id']]
        require(row['reference_manifest_row'] == reference and row['historical_role'] == reference['historical_role']
            and reference['historical_role'] == ('train' if row['split'] == 'TRAIN' else 'validation')
            and reference['decision'] == 'accept_clean_restoration'
            and row['source1024']['sha256'] == reference['source_sha256']
            and row['target256']['sha256'] == reference['existing_target_sha256']
            and row['source1024']['path'] == reference['original_source_path']
            and row['target256']['path'] == reference['original_existing_target_path']
            and row['attribution'] == reference['attribution'], 'Cohort role/source/terms binding differs')
    return cohort


def training_admission(split, strong_control):
    require(split in ('TRAIN', 'DEV') and type(strong_control) is bool, 'Known diagnostic split/control required')
    return {'optimization_allowed_by_split': split == 'TRAIN', 'optimization_permitted_now': False,
        'restoration_training_eligibility': 'EXCLUDED_DEV' if split == 'DEV' else
            ('EXCLUDED_INITIALLY_STRONG_CONTROL' if strong_control else 'PENDING_INPUT_ONLY_REVIEW')}


class Writer:
    def __init__(self, root):
        root = Path(root).absolute()
        require(root.parent.resolve(strict=True).is_relative_to(ROOT / 'outputs') and not root.exists(),
            'New output namespace under workspace outputs required; preserve partial/history')
        for part in (root.parent, *root.parent.parents):
            if part == ROOT.parent:
                break
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                'Linked output parent forbidden')
        root.mkdir(); self.root = root; self.files = {}; self.bytes = 0

    def put(self, name, data):
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and relative.as_posix() == name and ':' not in name and '\\' not in name
            and all(part not in ('', '.', '..') for part in relative.parts), 'Canonical relative output path required')
        require(name not in self.files and len(self.files) + 1 <= MAX_FILES and self.bytes + len(data) <= MAX_BYTES,
            'Finite unique output file/byte cap exceeded')
        path = self.root.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
        require(regular(path).read_bytes() == data, 'Exclusive output readback differs')
        pin = {'path': name, 'sha256': digest(data), 'bytes': len(data)}
        self.files[name] = pin; self.bytes += len(data)
        return dict(pin)


def prepare(selection_path, selection_sha, profiles_path, profiles_sha, out):
    require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')), 'No neural framework permitted')
    require(selection_sha == SELECTION_SHA, 'Exact independently selected eight-reference cohort required')
    started = time.monotonic()
    selected_bytes = read_bound(selection_path, selection_sha)
    configuration_bytes = read_bound(profiles_path, profiles_sha)
    selection, configuration = read_json(selected_bytes), read_json(configuration_bytes)
    require(configuration == profiles_document(), 'Prospectively approved12-profile configuration changed')
    catalog_bytes = from_pin(selection['catalog'])
    cohort = validate_selection(selection, read_json(catalog_bytes))
    for pin in selection['supporting_audit_and_approval_manifests']:
        from_pin(pin)
    version = {'Python': sys.version, 'NumPy': np.__version__, 'Pillow': pillow_version, 'SciPy': scipy.__version__}
    sources = {name: {'sha256': digest(regular(ROOT / name).read_bytes()),
        'bytes': (ROOT / name).stat().st_size} for name in SOURCE_FILES}
    writer = Writer(out)
    writer.put('cohort_selection.json', selected_bytes)
    writer.put('approved_profiles.json', configuration_bytes)
    writer.put('README.md', (ROOT / 'research/codeformer_filter_pairs_v1/README.md').read_bytes())
    observed_pin = writer.put('observed-support.png', image_bytes(np.full((256, 256), 255, np.uint8)))
    footprints = {}
    for profile in PROFILES:
        _, footprint = apply_profile(np.zeros((256, 256, 3), np.uint8), profile)
        footprints[profile['id']] = writer.put('corruption_footprints/' + profile['id'] + '.png', image_bytes(footprint))
    rows, reference_proofs, sheets, sheet_cells = [], [], [], []
    all_inputs = {profile_id: [] for profile_id in INPUT_REVIEW_ORDER}
    for selected in cohort:
        rid = selected['reference_id']; source_bytes = from_pin(selected['source1024'])
        target256_bytes = from_pin(selected['target256'])
        source = rgb(source_bytes, 1024); target256 = rgb(target256_bytes, 256)
        require(digest(source.tobytes()) == selected['source1024']['rgb8']['pixel_sha256'], 'Source1024 RGB differs')
        with Image.open(io.BytesIO(source_bytes)) as image:
            target512 = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS)).copy()
            direct256 = np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS)).copy()
            clear256 = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR)).copy()
        require(np.array_equal(target256, direct256), 'Retained clean256 differs from direct1024 LANCZOS256')
        for pixels, proof in ((target512, selected['clean512']), (target256, selected['target256']),
            (clear256, selected['clear256'])):
            check_pixels(pixels, proof)
        reference_pins = {'clean512': writer.put('references/' + rid + '/clean512.png', image_bytes(target512)),
            'clean256': writer.put('references/' + rid + '/clean256.png', target256_bytes)}
        reference_proofs.append({'reference_id': rid, 'split': selected['split'],
            'source1024': selected['source1024'], 'target256_retained_bytes_exact': True,
            'clean512_direct1024_proof': pixel_record(target512), 'clean256_direct1024_proof': pixel_record(target256),
            'clear256_direct1024_proof': pixel_record(clear256), 'attribution': selected['attribution'],
            'generated_reference_files': reference_pins})
        for profile in PROFILES:
            case_id = rid + '__' + profile['id']
            output, footprint = apply_profile(clear256, profile)
            all_inputs[profile['id']].append((rid, selected['split'], output))
            inp_pin = writer.put('inputs/' + case_id + '.png', image_bytes(output))
            record = {'format': 'codeformer-filter-synthetic-pair-v1', 'id': case_id,
                'reference_id': rid, 'split': selected['split'], 'historical_role': selected['historical_role'],
                'evidence': 'paired_synthetic_fixed_filter_stress', 'source1024': selected['source1024'],
                'attribution': selected['attribution'], 'observed_input': inp_pin,
                'input256_pixels': pixel_record(output), 'targets': reference_pins,
                'target512_pixels': pixel_record(target512), 'target256_pixels': pixel_record(target256),
                'observed_support': observed_pin, 'corruption_footprint': footprints[profile['id']],
                'corruption_footprint_pixels_sha256': digest(footprint.tobytes()),
                'corruption_footprint_is_not_completion_or_removal_mask': True,
                'profile': profile, 'seed': seed_record(rid, profile['id']), 'RNG_calls': 0,
                'operation_order': ['verify direct source1024', 'BILINEAR direct1024 to input256',
                    profile['operator'], 'round uint8 RGB255; save PNG'],
                'local_rectangle_xyxy_half_open': configuration['local_rectangle_xyxy_half_open']
                    if profile['operator'].startswith('local_') else None,
                'geometry': {'crop': None, 'warp': None, 'face_alignment': None, 'observed_support': 'all256pixels',
                    'targets': 'clean512 and clean256 direct1024 LANCZOS, never input256 upsampling'},
                'input_only_review': {'status': 'PENDING', 'decision': None, 'reviewer': None,
                    'input_only': True, 'model_output_used': False},
                **training_admission(selected['split'], profile['strong_control']),
                'automatic_usable_or_insufficient_label': None, 'actual_training_performed': False,
                'models_or_output_based_parameters': False, 'user_wild_images_used': False,
                'reserved_FINAL_used': False, 'versions': version}
            record_pin = writer.put('pairs/' + case_id + '.json', json_bytes(record))
            rows.append({'id': case_id, 'reference_id': rid, 'split': selected['split'],
                'profile': profile['id'], 'pair_record': record_pin, 'input': inp_pin,
                'input_review': 'PENDING', **training_admission(selected['split'], profile['strong_control'])})
        print(json.dumps({'prepared_references': len(reference_proofs), 'of': 8, 'paired_inputs': len(rows)}), flush=True)
    require(len(rows) == 96 and Counter(row['split'] for row in rows) == {'TRAIN': 48, 'DEV': 48}
        and Counter(row['profile'] for row in rows) == dict.fromkeys((p['id'] for p in PROFILES), 8),
        'Fixed96/12profile/eight-source census differs')
    require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')), 'Neural framework imported')
    for review_index, profile_id in enumerate(INPUT_REVIEW_ORDER, 1):
        sheet, cells = input_only_sheet(profile_id, all_inputs[profile_id])
        sheet_pin = writer.put('input_only_sheets/%02d_%s.png' % (review_index, profile_id), sheet)
        sheets.append(dict(sheet_pin, profile=profile_id, review_index=review_index, cases=8, width=1024, height=612))
        sheet_cells.extend(dict(cell, sheet=sheet_pin['path']) for cell in cells)
    require(len(sheets) == 12 and len(sheet_cells) == 96, 'All96 input-only cells/twelve sheets required')
    writer.put('input_only_visual_manifest.json', json_bytes({'complete': True, 'sheets': sheets,
        'cells': sheet_cells, 'input_cases': 96, 'exact256_cells_readback': 96,
        'clean_reference_or_model_output_panels': 0, 'root_input_only_review_status': 'PENDING',
        'prospective_review_order': list(INPUT_REVIEW_ORDER), 'clear_profile_withheld_until_last': True,
        'severe_input_judgments_must_be_locked_before_clear_or_clean_reference_view': True,
        'blinded_to_other_degraded_versions': False,
        'no_automatic_input_quality_labels': True}))
    manifest = {'format': 'codeformer-fixed-filter-pair-audit-v1', 'complete': True,
        'selection_sha256': selection_sha, 'approved_profiles_sha256': profiles_sha, 'source_pins': sources,
        'references': reference_proofs, 'cases': rows, 'profiles': list(PROFILES), 'versions': version,
        'input_only_sheets': sheets,
        'files': dict(writer.files), 'manifest_excludes_its_own_recursive_hash': True,
        'pairs': 96, 'reference_photographs': 8, 'TRAIN_pairs': 48, 'DEV_pairs': 48,
        'all_input_reviews_pending': True, 'strong_control_cases_excluded_initially_from_training': 40,
        'TRAIN_strong_control_cases_excluded_initially_from_training': 20,
        'DEV_cases_excluded_from_optimization_regardless_of_input_review': 48,
        'TRAIN_nonstrong_cases_pending_input_review_before_any_training_admission': 28,
        'small_diagnostic_cohort_not_complete_training_dataset': True,
        'new_model_or_inference_or_training_calls': 0, 'input_quality_threshold_calibrated': False,
        'user_images_native_CCTV_FINAL_used': False, 'historical_membership_or_quality_gates_changed': False,
        'source_terms_and_photo_attribution_retained': True,
        'no_identity_or_Zamboanga_performance_claim': True,
        'finite_caps': {'files': MAX_FILES, 'payload_bytes': MAX_BYTES}, 'seconds': time.monotonic() - started}
    final_pin = writer.put('manifest.json', json_bytes(manifest))
    require({p.relative_to(writer.root).as_posix() for p in writer.root.rglob('*') if p.is_file()} == set(writer.files),
        'Output namespace not closed')
    for name, pin in writer.files.items():
        data = regular(writer.root / name).read_bytes()
        require(len(data) == pin['bytes'] and digest(data) == pin['sha256'], 'Final output hash closure differs')
    return {'complete': True, 'pairs': 96, 'files': len(writer.files), 'bytes': writer.bytes,
        'manifest': final_pin, 'all_input_reviews_pending': True, 'new_neural_or_training_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--selection-sha', required=True)
    parser.add_argument('--profiles', required=True, type=Path)
    parser.add_argument('--profiles-sha', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.selection_sha, args.profiles, args.profiles_sha, args.out)), flush=True)

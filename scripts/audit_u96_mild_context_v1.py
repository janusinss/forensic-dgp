"""Independent saved-only U96 matched-operator study audit. No inference entrypoint.

All computations read saved artifacts. Main requires a completed comparison and a
prior sealed auditor/source receipt; output creation is exclusive. Scores and the
randomized panel key are written to disk, never printed before root visual review.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import secrets
import stat
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_mild_context_v1'
PRIOR = ROOT / 'outputs/codeformer_u96_context_study_v1'
PLAN_SHA = 'e6014a07f3dcfb76045201de25f45cd2e7c260d44a98a9d55c20f3fe508b9543'
WORKER_SHA = '00829d982895cc716083a7857a2ec1fae62101d77797ee51f4fefe4a5036b46b'
METRIC = ROOT / 'scripts/audit_codeformer_spatial_dev_local_v1_r2.py'
METRIC_SHA = '619a7863031ede9d1dcb78fa0bd116014e514cfc7ee1e5fb3d4c7155bc91fb2b'
ROLES = ('clear', 'global_gaussian_1p2', 'local_gaussian_1p2')
CONDITIONS = ('original', 'prepared')
REGIONS = ('whole_image', 'protected_region', 'surroundings', 'central128_patch')
SCORE_FIELDS = ('mae', 'mse', 'psnr', 'ssim', 'adjacent_RGB_residual_gradient_mae',
                'mean_centered_residual_mae', 'mean_centered_residual_mse',
                'signed_luminance_bias', 'absolute_luminance_bias')
U96 = {
    'arm': 'spatial_u',
    'weights_sha256': '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0',
    'R2_weights_sha256': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896',
    'adapter_tensor_sha256': 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a',
    'base_tensor_sha256': 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f',
    'source_checkpoint_sha256': '9a71153b1d3d127d00f193a93ea15ebb0b66c102033f6f16517ef42e7f41165b',
    'source_training_updates': 96, 'additional_continuation_updates': 72,
    'source_terminal_status': 'complete',
    'protocol_sha256': 'c355679d2475a65e19338a6c4fa7e8682b0c154442070e06b92a073037ea50ed',
    'architecture': 'CodeFormer by S-Lab; our four learned encoder-feature residual adapters',
    'adapter_parameters': 51328, 'all515_R2_tensors_frozen': True,
    'shared_frozen_R2_base_serial_inference': True,
    'internal_resolution': 512, 'output_resolution': 256,
    'fidelity': 1., 'adain': True, 'alignment': 'none', 'display_processing': 'none',
    'model_qualified': False, 'main_app_default_changed': False,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_json(data):
    def pairs(items):
        result = {}
        for k, v in items:
            assert k not in result, 'Duplicate JSON key: ' + k
            result[k] = v
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def checked(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve(strict=True).is_relative_to(ROOT)
    for item in (path, *path.parents):
        if item == ROOT.parent:
            break
        info = item.lstat()
        assert not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024
    before = path.stat()
    assert stat.S_ISREG(before.st_mode) and before.st_size <= 512 * 1024**2
    data = path.read_bytes()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return data


def pin(path):
    path = Path(path).resolve()
    data = checked(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)}


def bound(binding):
    assert set(binding) == {'path', 'bytes', 'sha256'}
    relative = PurePosixPath(binding['path'])
    assert not relative.is_absolute() and relative.as_posix() == binding['path']
    assert '\\' not in binding['path'] and ':' not in binding['path']
    assert all(p not in ('.', '..', '') for p in relative.parts)
    data = checked(ROOT / relative)
    assert len(data) == binding['bytes'] and sha(data) == binding['sha256'], binding['path']
    return data


def doc(path):
    return unique_json(checked(Path(path)))


def bound_doc(binding):
    return unique_json(bound(binding))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def decode(data, mode='RGB', size=(256, 256)):
    with Image.open(io.BytesIO(data)) as im:
        assert im.mode == mode and im.size == size
        return np.asarray(im).copy()


def mask(data):
    values = decode(data, 'L')
    assert set(np.unique(values)).issubset({0, 255})
    return values > 0


def blob(data):
    header, encoded = data.split(',', 1)
    assert header in ('data:image/png;base64', 'data:application/zip;base64')
    return base64.b64decode(encoded, validate=True)


def gaussian(values, sigma):
    return np.rint(gaussian_filter(values.astype(np.float64), (sigma, sigma, 0),
                                  mode='reflect', truncate=3.)).clip(0, 255).astype(np.uint8)


def geometry(values):
    digest = sha(values.tobytes())
    return {'native_size': [256, 256], 'native_center_offset': [0, 0],
            'observed_bounds_256': [0, 0, 256, 256], 'output_size': [256, 256],
            'policy': 'dgp256-observed-canvas-v1', 'input_rgb_sha256': digest,
            'canvas_rgb_sha256': digest, 'observed_sha256': sha(np.ones((256, 256), bool).tobytes()),
            'removal_mask_sha256': sha(np.zeros((256, 256), bool).tobytes())}


def seal():
    plan_path = OUT / 'comparison_plan_v1.json'
    assert pin(plan_path)['sha256'] == PLAN_SHA
    plan = doc(plan_path)
    assert plan['worker']['sha256'] == WORKER_SHA
    candidate = bound_doc(plan['candidate_manifest'])
    admission = bound_doc(plan['input_admission'])
    stable = [pin(Path(__file__).resolve()), pin(plan_path), plan['candidate_manifest'],
              plan['input_admission'], plan['selection_contract'], plan['prior_masks'],
              pin(OUT / 'root_protection_review_v1.json'), pin(OUT / 'independent_preparation_audit_v1.json'),
              pin(METRIC), *plan['source_pins'], *plan['model_files'].values(),
              *plan['model_manifest_and_reviews'], admission['reviewer_record'], admission['anonymous_map']]
    stable += [candidate[k] for k in ('source_script', 'prior_selection', 'prior_source_manifest',
                                     'reference_manifest', 'historical_membership_manifest', 'criteria', 'input_only_packet_rules')]
    unique = {r['path']: r for r in stable}
    for row in unique.values():
        bound(row)
    assert pin(METRIC)['sha256'] == METRIC_SHA
    write(OUT / 'auditor_source_bindings_v1.json', {
        'complete': True, 'sealed_UTC': datetime.now(timezone.utc).isoformat(),
        'audit_not_run': True, 'bindings': list(unique.values()),
        'draft_worker_pin_is_historical_not_current': True,
        'model_HTTP_VM_training_calls': 0,
    })
    print(json.dumps({'auditor_sealed': True, 'bindings': len(unique), 'audit_not_run': True}))


def verify_preparation(plan):
    candidate = bound_doc(plan['candidate_manifest'])
    receipt = doc(OUT / 'independent_preparation_audit_v1.json')
    assert receipt['independent_preparation_audit_pass'] and receipt['candidate_manifest'] == plan['candidate_manifest']
    assert candidate['complete'] and len(candidate['sources']) == 8 and len(candidate['cases']) == 40
    refs = bound_doc(candidate['reference_manifest'])['references']
    by_ref = {r['id']: r for r in refs}
    assert len(by_ref) == len(refs)
    membership = bound_doc(candidate['historical_membership_manifest'])
    train = [r for r in refs if r['historical_role'] == 'train']
    train_ids = {r['id'] for r in train}
    train_hashes = {r['source_sha256'] for r in train}
    train_photos = {r['attribution']['photo_url'] for r in train}
    clean = {}
    for row in candidate['sources']:
        ref = row['reference_id']; original = by_ref[ref]
        assert ref in membership['validation_reference_ids'] and ref not in membership['training_reference_ids']
        assert ref not in train_ids and original['source_sha256'] not in train_hashes
        assert original['attribution']['photo_url'] not in train_photos
        assert row['historical_role'] == original['historical_role'] == 'validation'
        assert row['study_role'] == 'DEV' and not row['optimization_permitted'] and not original['use_for_optimization']
        assert original['decision'] == 'accept_clean_restoration'
        assert row['attribution'] == original['attribution']
        assert all(row['attribution'][k] for k in ('photo_url', 'author', 'license', 'license_url'))
        source = bound(row['source1024']); target = bound(row['target256_original'])
        assert sha(source) == original['source_sha256'] and sha(target) == original['existing_target_sha256']
        assert bound(row['prior_saved_target']) == target == bound(row['target256_saved'])
        pixels = decode(source, size=(1024, 1024)); clean[ref] = decode(target)
        assert sha(pixels.tobytes()) == row['source_RGB8_sha256']
        assert sha(clean[ref].tobytes()) == row['target_RGB8_sha256']
        expected = np.asarray(Image.fromarray(pixels).resize((256, 256), Image.Resampling.LANCZOS))
        assert np.array_equal(expected, clean[ref])
    lookup = {(r['reference_id'], r['role']): r for r in candidate['cases']}
    assert len(lookup) == 40
    for row in candidate['cases']:
        inp = decode(bound(row['input'])); target = decode(bound(row['target']))
        assert np.array_equal(target, clean[row['reference_id']])
        assert row['split'] == 'DEV' and not row['optimization_permitted'] and row['paired_clean_target']
        assert sha(inp.tobytes()) == row['input_RGB8_sha256']
        expected = target.copy()
        role = row['role']
        if role != 'clear':
            sigma = 1.2 if role.endswith('1p2') else 1.8
            assert role in ('global_gaussian_1p2', 'global_gaussian_1p8', 'local_gaussian_1p2', 'local_gaussian_1p8')
            assert row['Gaussian_sigma'] == sigma
            filtered = gaussian(target, sigma)
            if role.startswith('global'):
                expected = filtered
            else:
                expected[64:192, 64:192] = filtered[64:192, 64:192]
                outside = np.ones((256, 256), bool); outside[64:192, 64:192] = False
                assert np.array_equal(inp[outside], target[outside])
        else:
            assert row['Gaussian_sigma'] is None and bound(row['input']) == bound(row['target'])
        assert np.array_equal(inp, expected), row['id']
    return candidate, lookup


def verify_admission(plan, candidate):
    admission = bound_doc(plan['input_admission'])
    contract = bound_doc(plan['selection_contract'])
    review = bound_doc(admission['reviewer_record'])
    mapping = bound_doc(admission['anonymous_map'])
    assert admission['complete'] and admission['frozen_before_study_model_outputs']
    assert contract['policy_before_reading_input_decisions'] and contract['primary_sigma'] == 1.2
    assert contract['minimum_distinct_photographs'] == 4 and contract['maximum_distinct_photographs'] == 8
    assert contract['roles_per_selected_photo'] == list(ROLES)
    assert admission['candidate_manifest'] == contract['candidate_manifest'] == plan['candidate_manifest']
    assert review['complete'] and review['clean_references_or_model_outputs_seen'] is False and review['sheets_viewed'] == 8
    decisions = {r['anonymous_id']: r for r in review['decisions']}
    assert len(decisions) == len(review['decisions']) == 32
    joined = {r['id']: r for r in admission['cases']}
    assert len(joined) == len(mapping['cases']) == 32
    for row in mapping['cases']:
        verdict = decisions[row['anonymous_id']]; actual = joined[row['id']]
        assert all(actual[k] == v for k, v in row.items())
        assert actual['decision'] == verdict['decision'] and actual['reason'] == verdict['reason']
        assert bound(row['original_input']) == bound(row['anonymous_input'])
    selected = [ref for ref in candidate['fixed_reference_order'] if all(
        joined[ref + '__' + role]['decision'] == 'readable' for role in ROLES[1:])]
    assert 4 <= len(selected) <= 8 and len(selected) == 7
    assert selected == admission['selected_reference_ids'] == plan['selected_reference_ids']
    assert dict(Counter(r['decision'] for r in admission['cases'])) == admission['counts']
    assert admission['selected_sigma'] == plan['primary_sigma'] == 1.2
    assert admission['secondary_sigma1p8_is_not_selected'] and plan['sigma1p8_not_selected_for_this_run']
    return admission


def verify_output(pins, submitted, expected_geometry):
    assert set(pins) == {'PNG', 'RAW', 'ZIP', 'metadata', 'body', 'headers'}
    png = bound(pins['PNG']); raw_bytes = bound(pins['RAW']); archive = bound(pins['ZIP'])
    body_bytes = bound(pins['body']); body = unique_json(body_bytes)
    headers = {k.lower(): v for k, v in bound_doc(pins['headers']).items()}
    assert int(headers['content-length']) == len(body_bytes) and headers['content-type'] == 'application/json'
    meta = bound_doc(pins['metadata']); assert body['metadata'] == meta
    assert meta['policy'] == 'research-CodeFormer-spatial-comparison-v1'
    assert meta['selection'] == 'spatial_u' and meta['input_review'] == 'usable'
    assert meta['uploaded_bytes_sha256'] == sha(submitted) and meta['geometry'] == expected_geometry
    assert meta['inference_only'] and meta['device'] == 'cpu'
    assert meta['model_qualified'] is False and meta['main_app_default_changed'] is False
    assert meta['RAW_PNG_separate'] and meta['no_completion_or_covering_removal']
    assert meta['operator_input_review_is_not_a_validated_automatic_threshold']
    assert meta['preferred_development_model'] == 'spatial_u'
    assert set(meta['outputs']) == set(body['outputs']) == {'spatial_u'}
    pixels = decode(png); raw = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    assert raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
    assert raw.min() >= 0 and raw.max() <= 1
    assert np.array_equal(pixels, np.floor(raw * np.float32(255)).astype(np.uint8))
    declaration = meta['outputs']['spatial_u']; provenance = declaration['metadata']
    assert declaration['raw_rgb_sha256'] == sha(raw.tobytes())
    assert declaration['PNG_RGB8_sha256'] == sha(pixels.tobytes()) and declaration['PNG_file_sha256'] == sha(png)
    for k, v in U96.items():
        assert provenance[k] == v, ('provenance', k)
    assert blob(body['outputs']['spatial_u']) == png and blob(body['bundle']) == archive
    assert np.array_equal(decode(blob(body['original'])), decode(submitted))
    expected_members = {'original-upload.png', 'resize-control.png', 'observed-support.png',
                        'spatial_u.png', 'spatial_u-raw.npy', 'processing.json'}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert z.testzip() is None and len(z.namelist()) == 6 and set(z.namelist()) == expected_members
        assert z.read('original-upload.png') == submitted
        assert z.read('spatial_u.png') == png and z.read('spatial_u-raw.npy') == raw_bytes
        assert unique_json(z.read('processing.json')) == meta
        assert np.array_equal(decode(z.read('resize-control.png')), decode(submitted))
        assert mask(z.read('observed-support.png')).all()
    return raw, pixels


def main():
    seal_receipt = doc(OUT / 'auditor_source_bindings_v1.json')
    assert seal_receipt['complete'] and seal_receipt['audit_not_run']
    for binding in seal_receipt['bindings']:
        bound(binding)
    assert pin(Path(__file__).resolve()) in seal_receipt['bindings']
    plan_path = OUT / 'comparison_plan_v1.json'; assert pin(plan_path)['sha256'] == PLAN_SHA
    plan = doc(plan_path); run_path = OUT / 'comparison.json'; run = doc(run_path)
    assert plan['complete'] and run['complete'] and 'failure' not in run and 'active_call' not in run
    assert run['plan'] == pin(plan_path) and run['worker'] == plan['worker']
    assert plan['worker']['sha256'] == WORKER_SHA
    assert run['training_optimizer_VM_calls'] == 0 and not run['model_qualified'] and not run['retry_or_resume']
    for binding in [plan['candidate_manifest'], plan['input_admission'], plan['selection_contract'], plan['prior_masks'],
                    *plan['source_pins'], *plan['model_files'].values(), *plan['model_manifest_and_reviews'],
                    run['confirmation'], run['server_receipt']]:
        bound(binding)
    confirmation = bound_doc(run['confirmation'])
    assert confirmation['confirmed_before_model_outputs'] and confirmation['plan'] == pin(plan_path)
    assert confirmation['sheets_personally_viewed'] == 2 and confirmation['photographs'] == 7
    assert confirmation['sheets'] == plan['protection_sheets']
    for binding in confirmation['sheets']:
        bound(binding)
    candidate, candidate_lookup = verify_preparation(plan)
    admission = verify_admission(plan, candidate)
    expected_case_order = [(ref, role) for ref in plan['selected_reference_ids'] for role in ROLES]
    assert [(r['reference_id'], r['role']) for r in plan['cases']] == expected_case_order
    assert [(r['reference_id'], r['role']) for r in run['cases']] == expected_case_order
    assert len(plan['cases']) == len(run['cases']) == 21 and len({r['id'] for r in run['cases']}) == 21
    assert plan['planned_total_outputs'] == 42 and plan['planned_reused_outputs'] == 7 and plan['planned_new_forwards'] == 35
    assert run['new_forwards'] == run['submitted_HTTP_calls'] == 35 and run['reused_outputs'] == 7
    for k in ('neural_forward_attempts', 'neural_forward_returns', 'neural_forward_successes', 'completed_comparisons'):
        assert run['server_before'][k] == 0 and run['server_after'][k] == 35
    for arm, count in (('original', 0), ('r2', 0), ('spatial_s', 0), ('spatial_u', 35)):
        for k in ('attempts', 'returns', 'successes'):
            assert run['server_before']['per_arm'][arm][k] == 0
            assert run['server_after']['per_arm'][arm][k] == count
    assert run['server_before']['operator_review_refusals'] == run['server_after']['operator_review_refusals'] == 0
    assert pin(METRIC)['sha256'] == METRIC_SHA
    spec = importlib.util.spec_from_file_location('retained_saved_only_metrics', METRIC)
    metric = importlib.util.module_from_spec(spec); spec.loader.exec_module(metric)
    prior_review = bound_doc(plan['prior_masks'])
    prior_rectangles = {r['reference_id']: r for r in prior_review['references']}
    prior_run = doc(PRIOR / 'recovery_r1/comparison.json')
    assert prior_run['complete']
    prior_cases = {r['id']: r for r in prior_run['cases']}
    rows = []; csvrows = []; new_outputs = []; reused_outputs = []
    for source, saved in zip(plan['cases'], run['cases'], strict=True):
        assert source['id'] == saved['id'] and saved['complete'] and set(saved['outputs']) == set(CONDITIONS)
        retained = candidate_lookup[(source['reference_id'], source['role'])]
        assert all(source[k] == v for k, v in retained.items())
        inp = decode(bound(source['input'])); target = decode(bound(source['target']))
        derived = decode(bound(source['derived']))
        decision = prior_rectangles[source['reference_id']]
        assert source['appearance_flag'] == decision['appearance_flag']
        assert source['protection_rectangle_inclusive'] == decision['protection_rectangle_inclusive']
        left, top, right, bottom = source['protection_rectangle_inclusive']
        base = np.zeros((256, 256), bool); base[top:bottom + 1, left:right + 1] = True
        protected = distance_transform_edt(~base) <= 6.
        assert np.array_equal(mask(bound(source['protection'])), protected) and protected[64:192, 64:192].all()
        assert source['protected_pixels'] == int(protected.sum()) and 0 < protected.sum() < 256 * 256
        t = np.clip(distance_transform_edt(~protected) / 8., 0, 1)
        alpha = (t * t * (3 - 2 * t)).astype(np.float32)
        saved_alpha = np.load(io.BytesIO(bound(source['alpha'])), allow_pickle=False)
        assert saved_alpha.dtype == np.float32 and np.array_equal(saved_alpha, alpha) and not alpha[protected].any()
        ys, xs = np.nonzero(alpha > 0)
        assert not ((xs + 4 >= 64) & (xs - 4 < 192) & (ys + 4 >= 64) & (ys - 4 < 192)).any()
        softened = gaussian(inp, 1.2)
        expected = np.rint(inp.astype(np.float64) * (1 - alpha[..., None].astype(np.float64)) +
                           softened.astype(np.float64) * alpha[..., None].astype(np.float64)).clip(0, 255).astype(np.uint8)
        assert np.array_equal(derived, expected) and np.array_equal(derived[protected], inp[protected])
        colors = np.zeros_like(inp); colors[:] = (200, 60, 60)
        colors[(alpha > 0) & (alpha < 1)] = (245, 180, 45); colors[protected] = (45, 185, 90)
        assert np.array_equal(decode(bound(source['overlay'])), np.rint(inp.astype(np.float64) * .65 + colors.astype(np.float64) * .35).astype(np.uint8))
        assert source['geometry'] == geometry(inp) and source['derived_geometry'] == geometry(derived)
        assert source['forced_harm_control'] == (source['role'] != 'local_gaussian_1p2')
        assert source['optimization_permitted'] is False
        if source['role'] == 'clear':
            reuse = source['reused_original']
            assert reuse['exact_input_bytes'] and reuse['same_frozen_settings']
            assert reuse['from_comparison'] == pin(PRIOR / 'recovery_r1/comparison.json')
            assert saved['outputs']['original'] == reuse['outputs'] == prior_cases[source['id']]['outputs']['original:spatial_u']
        else:
            assert source['reused_original'] is None and source['input_review'] == 'readable'
        outputs = {}; scores = {}
        for condition in CONDITIONS:
            pins = saved['outputs'][condition]
            if source['role'] == 'clear' and condition == 'original':
                reused_outputs.append(pins)
            else:
                new_outputs.append(pins)
                folder = OUT / 'results' / source['id'] / condition
                assert {str(Path(p['path']).parent).replace('\\', '/') for p in pins.values()} == {folder.relative_to(ROOT).as_posix()}
            submitted = bound(source['input'] if condition == 'original' else source['derived'])
            raw, pixels = verify_output(pins, submitted, source['geometry'] if condition == 'original' else source['derived_geometry'])
            outputs[condition] = pins; scores[condition] = {}
            for format_, values in (('RAW', raw), ('PNG', pixels.astype(np.float32) / np.float32(255))):
                clean = target.astype(np.float32) / np.float32(255)
                current = {name: metric.metrics(values, clean, region) for name, region in (
                    ('whole_image', np.ones((256, 256), bool)), ('protected_region', protected), ('surroundings', ~protected))}
                current['central128_patch'] = metric.metrics(values[64:192, 64:192], clean[64:192, 64:192], np.ones((128, 128), bool))
                scores[condition][format_] = current
                for region, score in current.items():
                    csvrows.append({'id': source['id'], 'reference_id': source['reference_id'], 'role': source['role'],
                                    'appearance_flag': source['appearance_flag'], 'forced_harm_control': source['forced_harm_control'],
                                    'condition': condition, 'format': format_, 'region': region,
                                    **{k: score[k] for k in SCORE_FIELDS}})
        rows.append({'id': source['id'], 'reference_id': source['reference_id'], 'role': source['role'],
                     'appearance_flag': source['appearance_flag'], 'forced_harm_control': source['forced_harm_control'],
                     'protected_pixels': int(protected.sum()), 'source': source, 'outputs': outputs, 'metrics': scores})
    assert len(new_outputs) == 35 and len(reused_outputs) == 7 and len(csvrows) == 336
    assert len({p['body']['path'] for p in new_outputs}) == 35
    assert len({p['body']['path'] for p in reused_outputs}) == 7
    # Output meanings are randomized per role and kept in a separate file until root records its visual review.
    visuals = OUT / 'visuals'; visuals.mkdir(exist_ok=False)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    key = {}; sheets = []
    for role in ROLES:
        subset = [r for r in rows if r['role'] == role]; assert len(subset) == 7
        coded = list(CONDITIONS); secrets.SystemRandom().shuffle(coded); key[role] = dict(zip('AB', coded))
        for start in (0, 4):
            batch = subset[start:start + 4]
            sheet = Image.new('RGB', (6 * 264 + 8, len(batch) * 288 + 110), (245, 244, 238))
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), 'Mild U96 DEV | ' + role + ' | exact256 cells | assisted masks; not FINAL', font=font, fill='black')
            for col, label in enumerate(('Input', 'Prepared input', 'Clean target', 'Protection', 'A', 'B')):
                draw.text((8 + 264 * col, 44), label, font=font, fill='black')
            for ri, row in enumerate(batch):
                y = 82 + ri * 288; draw.text((8, y), row['reference_id'] + (' | APPEARANCE FLAG' if row['appearance_flag'] else ''), font=font, fill='black')
                src = row['source']
                cells = [decode(bound(src['input'])), decode(bound(src['derived'])), decode(bound(src['target'])),
                         np.repeat((mask(bound(src['protection'])) * 255).astype(np.uint8)[..., None], 3, axis=2)]
                cells += [decode(bound(row['outputs'][condition]['PNG'])) for condition in coded]
                for col, cell in enumerate(cells):
                    sheet.paste(Image.fromarray(cell), (8 + 264 * col, y + 22))
            path = visuals / (f'{len(sheets) + 1:02d}_{role}.png'); sheet.save(path)
            check = np.asarray(Image.open(path))
            for ri, row in enumerate(batch):
                for index, condition in enumerate(coded):
                    assert np.array_equal(check[104 + ri * 288:360 + ri * 288, 8 + (index + 4) * 264:264 + (index + 4) * 264], decode(bound(row['outputs'][condition]['PNG'])))
            sheets.append({'file': pin(path), 'cases': [r['id'] for r in batch], 'generated_PNG_cells': len(batch) * 2})
    assert len(sheets) == 6 and sum(r['generated_PNG_cells'] for r in sheets) == 42
    write(visuals / 'panel_key.json', {'complete': True, 'role_panel_conditions': key,
                                    'do_not_open_until_root_visual_review_saved': True})
    write(visuals / 'manifest.json', {'complete': True, 'review_complete': False, 'sheets': sheets,
                                    'panel_key': pin(visuals / 'panel_key.json'), 'generated_PNG_cells': 42})
    with (OUT / 'region_metrics_v1.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csvrows[0])); writer.writeheader(); writer.writerows(csvrows)
    write(OUT / 'saved_output_audit_v1.json', {
        'complete': True, 'audit_pass': True, 'auditor': pin(Path(__file__).resolve()),
        'auditor_source_bindings': pin(OUT / 'auditor_source_bindings_v1.json'),
        'comparison': pin(run_path), 'plan': pin(plan_path), 'retained_metrics': pin(METRIC),
        'preparation_audit': pin(OUT / 'independent_preparation_audit_v1.json'),
        'input_admission': plan['input_admission'], 'cases': rows, 'CSV': pin(OUT / 'region_metrics_v1.csv'),
        'visual_manifest': pin(visuals / 'manifest.json'), 'paired_photographs': 7, 'cases_count': 21,
        'outputs_count': 42, 'verified_new_forwards': 35, 'exact_reused_outputs': 7,
        'new_neural_forwards_are_not42_or_any_prior_run_combined_count': True,
        'metric_rows': 336, 'all40_candidate_inputs_reconstructed': True,
        'all32_anonymous_reviews_exactly_joined': True, 'all21_protections_exact': True,
        'all_positive_alpha_Gaussian_footprints_avoid_known_corruption': True,
        'RAW_PNG_transport_and_model_provenance_pass': True,
        'all21_preparations_start_from_original_observed_input': True,
        'root_visual_review_not_yet_complete': True,
        'known_synthetic_operator_and_assisted_prior_rectangles': True,
        'clear_global_prepared_are_forced_harm_controls': True,
        'regions_are_not_whole_face_identity_measurements': True,
        'source_terms_are_retained_snapshots_not_current_web_verification': True,
        'subject_identity_and_CodeFormer_pretraining_overlap_unknown': True,
        'repeated_DEV_exposure': True, 'native_or_FINAL': False, 'automatic_detector_qualified': False,
        'model_qualified': False, 'new_training_justified_by_integrity_alone': False,
        'auditor_model_HTTP_training_VM_calls': 0,
    })
    assert 'torch' not in sys.modules
    print(json.dumps({'audit_pass': True, 'cases': 21, 'outputs': 42, 'new_forwards': 35,
                      'exact_reused_outputs': 7, 'metric_rows': 336, 'coded_sheets': 6,
                      'scores_and_key_hidden_pending_root_visual_review': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    seal() if args.seal else main()

"""Independent, bounded DATA audit; never imports the preparation implementation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import sys


class NoModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tensorflow', 'jax', 'basicsr', 'facelib'}:
            raise ImportError('Independent data audit prohibits model runtime imports')


sys.meta_path.insert(0, NoModelRuntime())
import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION
import scipy
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
CLEAN_PATH = ROOT / 'outputs/codeformer_original_spatial_correction_v1_preparation/data_audit_v1/root_clean_target_review_v1.json'
CLEAN_SHA = '2e06c37a59269a19582b4d3066af456ae6b330dcf8fd3f73b54f07366e9b804f'
RANK_PREFIX = '20261024-codeformer-matched-feature-learning-v1-reserve'
ROLES = ('clear', 'global_gaussian2p0', 'local_gaussian2p0_soft', 'compound_g1p2_lr64_jpeg70', 'lowlight0p35_noise0p006')
consumed, checks, blockers = {}, [], []


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(path, raw=None):
    path = Path(path).resolve(strict=True)
    if raw is None:
        raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': digest(raw)}


def read(path, expected=None):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError('Only exact bounded workspace data paths may be read')
    raw = path.read_bytes()
    actual = pin(path, raw)
    if expected is not None:
        for key in ('bytes', 'sha256'):
            if actual[key] != expected[key]:
                raise ValueError('Historical pin mismatch: ' + actual['path'])
    if actual['path'] in consumed and consumed[actual['path']] != actual:
        raise ValueError('File changed across repeated reads: ' + actual['path'])
    consumed[actual['path']] = actual
    return raw


def load(path, expected=None):
    return json.loads(read(path, expected))


def require(condition, name, detail=None):
    row = {'check': name, 'pass': bool(condition)}
    if detail is not None:
        row['detail'] = detail
    checks.append(row)
    if not condition:
        blockers.append(name)
    return bool(condition)


def rgb(raw, side=None):
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or (side and image.size != (side, side)):
            raise ValueError('Expected exact RGB PNG geometry')
        return np.asarray(image).copy()


def quantize_rint(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def encode_png(values):
    buf = io.BytesIO()
    Image.fromarray(values).save(buf, format='PNG')
    return buf.getvalue()


def differing_channels(a, b):
    return int(np.count_nonzero(a != b))


def run():
    result = {'format': 'matched-feature-learning-independent-data-audit-v1',
              'complete': False, 'data_audit_pass': False, 'unresolved_blockers': blockers,
              'created_UTC': datetime.now(timezone.utc).isoformat(),
              'scope': 'Independent source provenance, deterministic selection, actual operators, exact inputs and sheet crops; no new readability decisions',
              'new_model_neural_gradient_optimizer_training_VM_HTTP_browser_calls': 0,
              'new_visual_readability_decisions': 0, 'new_optimization_admission': False,
              'training_packet_or_new_worker_ready': False, 'model_qualified': False,
              'native_DEV_FINAL_or_wild_inputs_accessed': False}
    try:
        read(Path(__file__).resolve())
        candidate_path = BASE / 'candidate_data_v1.json'
        candidate_raw = read(candidate_path)
        data = json.loads(candidate_raw)
        result['candidate'] = pin(candidate_path, candidate_raw)
        selection = load(ROOT / data['selection']['path'], data['selection'])
        result['selection'] = data['selection']
        read(ROOT / data['source']['path'], data['source'])
        result['preparation_source'] = data['source']
        parent_raw = read(PARENT / 'protocol.json')
        require(digest(parent_raw) == PARENT_SHA, 'fixed_parent_protocol_pin')
        parent = json.loads(parent_raw)
        parent_data = {}
        for name, item in parent['data'].items():
            parent_data[name] = load(PARENT / item['path'], item)
        old = parent_data['provenance']
        reuse = parent_data['conditional_reuse']
        clean_raw = read(CLEAN_PATH)
        require(digest(clean_raw) == CLEAN_SHA, 'fixed_root_clean_admission_pin')
        clean = json.loads(clean_raw)
        for item in clean['source_manifests']:
            load(ROOT / item['path'], item)
        admitted = {row['reference_id']: row for row in clean['admitted_references']}
        require(clean['complete'] and clean['root_personal_input_only_review_complete'] and len(clean['admitted_reference_ids']) == 64, 'root64_clean_admission_top_level_complete')
        previous_inputs = []
        for case in reuse['admitted_cases']:
            input_raw = read(ROOT / case['input']['path'], case['input'])
            target_raw = read(ROOT / case['target']['path'], case['target'])
            observed, target = rgb(input_raw, 256), rgb(target_raw, 256)
            require(digest(observed.tobytes()) == case['input_RGB8_sha256'], 'old96_input_pixel_pin:' + case['id'])
            require(digest(target.tobytes()) == case['target_RGB8_sha256'], 'old96_target_pixel_pin:' + case['id'])
            previous_inputs.append(case['id'])
        require(len(previous_inputs) == 96 and len(set(previous_inputs)) == 96, 'all96_historical_input_pins_preserved')
        for item in data['consumed_before_after_exact']:
            read(ROOT / item['path'], item)
        require(len(data['consumed_before_after_exact']) == 68, 'all68_preparation_historical_consumed_pins_exact_now')

        old_opt = reuse['optimization_reference_order']
        monitors = reuse['monitor_reference_order']
        preflight = set(reuse['preflight4_readout_only_not_optimization_or_monitor']['photographs'])
        deferred = set(old['split_reference_ids']['DEFERRED_FROM_THIS_LEARNING_PILOT'])
        reserve = set(old['split_reference_ids']['UNUSED_TRAIN_RESERVE'])
        rank = lambda ref: digest((RANK_PREFIX + ':' + ref).encode('ascii'))
        ranked = sorted(reserve - preflight - deferred - set(old_opt) - set(monitors), key=rank)
        added = ranked[:8]
        opt = list(old_opt) + added
        refs = opt + list(monitors)
        require(selection['rank_prefix'] == RANK_PREFIX, 'fixed_reserve_ranking_prefix')
        require(selection['eligible_reserve_rank'] == [{'reference_id': r, 'rank_sha256': rank(r)} for r in ranked], 'complete_deterministic_reserve_ranking_exact')
        require(selection['additional_optimization_reference_order'] == added, 'first8_reserve_additions_exact')
        require(selection['retained_optimization_reference_order'] == old_opt and selection['optimization_reference_order'] == opt and selection['adaptive_monitor_reference_order'] == monitors, 'original16opt8adaptive_monitor_order_retained')
        require(len(preflight) == 4 and len(deferred) == 2 and len(set(refs)) == 32 and len(opt) == 24 and len(monitors) == 8, 'exact24opt8monitor32sources_exclude4preflight2deferred')
        require(not set(refs) & (preflight | deferred) and set(refs) <= set(clean['admitted_reference_ids']), 'selected32_are_admitted_TRAIN_and_exclusions_absent')
        require(selection['preflight_readout_only_exclusions'] == sorted(preflight) and selection['deferred_exclusions'] == sorted(deferred), 'exclusion_lists_exact')
        require(selection['roles'] == list(ROLES), 'exact5_role_order')
        require(data['source_count'] == len(data['sources']) == 32 and data['paired_inputs'] == len(data['cases']) == 160 and data['degraded_inputs'] == 128 and data['clear_controls'] == 32, 'declared32sources160inputs128degraded32clear')
        require([r['reference_id'] for r in data['sources']] == refs, 'source_table_order_exact')
        require([r['id'] for r in data['cases']] == [r + '__' + role for r in refs for role in ROLES], 'case_table_exact_cartesian32x5_order')
        require(data['versions'] == {'NumPy': np.__version__, 'Pillow': PILLOW_VERSION, 'SciPy': scipy.__version__}, 'current_independent_audit_library_versions_match')
        require(data['operators']['global_gaussian2p0']['sigma'] == [2.0, 2.0, 0] and data['operators']['global_gaussian2p0']['mode'] == 'reflect' and data['operators']['global_gaussian2p0']['truncate'] == 3.0 and data['operators']['global_gaussian2p0']['radius_pixels'] == 6, 'global_gaussian_declared_sigma2_reflect_radius6')
        require(not selection['new_data_optimization_admission'] and not data['new_optimization_admission'] and not data['input_only_review_complete'] and not data['training_packet_or_new_worker_ready'], 'candidate_retains_pending_review_false_admission')

        yy, xx = np.mgrid[0:256, 0:256]
        ellipse_radius = np.sqrt(((xx - 127.5) / 72.0)**2 + ((yy - 127.5) / 88.0)**2)
        fraction = np.clip((1 - ellipse_radius) / 0.20, 0, 1)
        alpha = fraction**2 * (3 - 2 * fraction)
        alpha3 = alpha[:, :, None]
        source_results, case_results, input_arrays = [], [], {}
        alternative_changes = []
        cases = {case['id']: case for case in data['cases']}
        for source_row in data['sources']:
            ref = source_row['reference_id']
            row = admitted[ref]
            original_raw = read(ROOT / row['physical_source']['path'], row['physical_source'])
            original = rgb(original_raw, 1024)
            target_raw = read(ROOT / row['clean256_existing']['path'], row['clean256_existing'])
            target = rgb(target_raw, 256)
            copied_target_raw = read(ROOT / source_row['target']['path'], source_row['target'])
            direct_target = np.asarray(Image.fromarray(original).resize((256, 256), resample=Image.Resampling.LANCZOS))
            require(source_row['source'] == {k: row['physical_source'][k] for k in ('path', 'bytes', 'sha256')}, 'source_binding_exact:' + ref)
            require(row['historical_role'] == 'TRAIN' and row['official_category'] == 'training' and ref in clean['admitted_reference_ids'], 'root_clean_TRAIN_admission:' + ref)
            require(digest(original.tobytes()) == row['physical_source']['RGB8_sha256'], 'source1024_RGB8_exact:' + ref)
            require(hashlib.md5(original_raw).hexdigest() == row['physical_source']['official_file_MD5'] and hashlib.md5(original.tobytes()).hexdigest() == row['physical_source']['official_pixel_MD5'], 'official_source_file_and_pixel_MD5:' + ref)
            require(copied_target_raw == target_raw and np.array_equal(target, direct_target) and digest(target.tobytes()) == row['clean256_existing']['RGB8_sha256'] == source_row['target_RGB8_sha256'], 'original1024_direct_LANCZOS256_copy_exact:' + ref)
            require(source_row['attribution'] == row['attribution'], 'source_retained_attribution_exact:' + ref)
            require(source_row['historical_clean_admission'] == pin(CLEAN_PATH), 'source_root_clean_admission_pin:' + ref)
            require(source_row['cohort'] == ('optimize' if ref in opt else 'adaptive_monitor') and source_row['person_identity_or_pretraining_nonoverlap_established'] is False, 'source_scope_metadata_exact:' + ref)
            base = target.astype(np.float64)
            gaussian2_float = gaussian_filter(base, sigma=(2.0, 2.0, 0), mode='reflect', truncate=3.0)
            gaussian2_rgb8 = quantize_rint(gaussian2_float)
            local = quantize_rint(base * (1 - alpha3) + gaussian2_rgb8.astype(np.float64) * alpha3)
            local[alpha == 0] = target[alpha == 0]
            alternative_local = quantize_rint(base * (1 - alpha3) + gaussian2_float * alpha3)
            alternative_local[alpha == 0] = target[alpha == 0]
            alternative_changes.append({'reference_id': ref,
                'differing_RGB_channels': differing_channels(local, alternative_local),
                'differing_pixels': int(np.count_nonzero(np.any(local != alternative_local, axis=2))),
                'maximum_absolute_RGB8_difference': int(np.max(np.abs(local.astype(np.int16) - alternative_local.astype(np.int16))))})
            compound_initial = quantize_rint(gaussian_filter(base, sigma=(1.2, 1.2, 0), mode='reflect', truncate=3.0))
            compound_image = Image.fromarray(compound_initial).resize((64, 64), Image.Resampling.BILINEAR).resize((256, 256), Image.Resampling.BILINEAR)
            jpeg_buffer = io.BytesIO()
            compound_image.save(jpeg_buffer, format='JPEG', quality=70, subsampling=0, optimize=False, progressive=False)
            with Image.open(io.BytesIO(jpeg_buffer.getvalue())) as jpeg_image:
                compound = np.asarray(jpeg_image.convert('RGB')).copy()
            s = base / 255.0
            linear = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055)**2.4)
            exposed = linear * 0.35
            encoded = np.where(exposed <= 0.0031308, exposed * 12.92, 1.055 * exposed**(1 / 2.4) - 0.055)
            seed = int(digest(('20261024-matched-feature-data-v1:' + ref + ':lowlight0p35_noise0p006').encode('ascii'))[:16], 16)
            generator = np.random.Generator(np.random.PCG64(seed))
            lowlight = np.floor(np.clip(encoded + generator.normal(loc=0.0, scale=0.006, size=target.shape), 0, 1) * 255 + 0.5).astype(np.uint8)
            expected_by_role = dict(zip(ROLES, (target, gaussian2_rgb8, local, compound, lowlight)))
            source_results.append({'reference_id': ref, 'source': source_row['source'], 'target': source_row['target'], 'direct1024to256_LANCZOS_pixel_differences': differing_channels(target, direct_target), 'target_copy_bytes_exact': copied_target_raw == target_raw, 'retained_attribution_exact': source_row['attribution'] == row['attribution']})
            for role in ROLES:
                case_id = ref + '__' + role
                case = cases[case_id]
                expected = expected_by_role[role]
                actual_raw = read(ROOT / case['input']['path'], case['input'])
                actual = rgb(actual_raw, 256)
                reconstructed_raw = target_raw if role == 'clear' else encode_png(expected)
                pixel_differences = differing_channels(actual, expected)
                byte_exact = actual_raw == reconstructed_raw
                require(pixel_differences == 0 and byte_exact, 'independent_operator_RGB_and_PNG_exact:' + case_id)
                require(digest(actual.tobytes()) == case['input_RGB8_sha256'] and digest(target.tobytes()) == case['target_RGB8_sha256'], 'case_RGB8_bindings:' + case_id)
                require(case['target'] == source_row['target'] and case['reference_id'] == ref and case['role'] == role and case['cohort'] == source_row['cohort'] and case['historical_split'] == 'TRAIN', 'case_target_cohort_role_binding:' + case_id)
                require(case['anonymous_review_id'] == 'D' + digest(('matched-feature-input-review-v1:' + case_id).encode('ascii'))[:10], 'anonymous_case_id_exact:' + case_id)
                require(case['noise_seed'] == (seed if role == 'lowlight0p35_noise0p006' else None) and case['input_only_decision'] == 'PENDING' and case['optimization_permitted'] is False and case['native_CCTV'] is False and case['paired_synthetic'] is True, 'case_pending_review_and_synthetic_scope:' + case_id)
                outside_exact = bool(np.array_equal(actual[alpha == 0], target[alpha == 0])) if role == 'local_gaussian2p0_soft' else None
                if outside_exact is not None:
                    require(outside_exact, 'local_alpha0_original_exact:' + ref)
                case_results.append({'case_id': case_id, 'independently_recomputed_from_original_clean256_once': True, 'RGB8_differing_channels': pixel_differences, 'PNG_bytes_exact': byte_exact, 'local_alpha0_original_exact': outside_exact})
                input_arrays[case_id] = actual

        sheet_results, covered = [], []
        for sheet in data['sheets']:
            sheet_raw = read(ROOT / sheet['path'], sheet)
            sheet_pixels = rgb(sheet_raw)
            require(sheet_pixels.shape == (1286, 1104, 3) and len(sheet['cells']) == 16, 'sheet_exact_geometry16cells:' + sheet['path'])
            for index, cell in enumerate(sheet['cells']):
                case = cases[cell['case_id']]
                x, y = 16 + (index % 4) * 272, 84 + (index // 4) * 304
                expected_bounds = [x, y, x + 256, y + 256]
                require(cell['bounds_xyxy'] == expected_bounds and cell['input'] == case['input'] and cell['anonymous_review_id'] == case['anonymous_review_id'], 'sheet_cell_metadata_exact:' + cell['case_id'])
                x0, y0, x1, y1 = cell['bounds_xyxy']
                crop = sheet_pixels[y0:y1, x0:x1, :]
                count = differing_channels(crop, input_arrays[cell['case_id']])
                require(count == 0 and (case['role'] == 'clear') == (sheet['kind'] == 'clear'), 'sheet_actual_RGB_crop_exact_and_clear_separate:' + cell['case_id'])
                covered.append(cell['case_id'])
            sheet_results.append({'sheet': {k: sheet[k] for k in ('path', 'bytes', 'sha256')}, 'kind': sheet['kind'], 'exact_RGB8_crops': len(sheet['cells']), 'different_RGB_channels': 0})
        require(len(data['sheets']) == 10 and Counter(covered) == Counter(cases.keys()) and len(covered) == len(set(covered)) == 160, 'all160_actual_inputs_exactly_once_in10sheets')
        require(Counter(r['kind'] for r in data['sheets']) == {'degraded': 8, 'clear': 2}, 'eight_degraded_two_clear_sheets')
        expected_sheet_order = []
        for kind, rows in (('degraded', [r for r in data['cases'] if r['role'] != 'clear']), ('clear', [r for r in data['cases'] if r['role'] == 'clear'])):
            expected_sheet_order.extend(r['id'] for r in sorted(rows, key=lambda r: digest(('anonymous-input-order-v1:' + r['id']).encode('ascii'))))
        require(covered == expected_sheet_order, 'sheet_anonymous_deterministic_order_exact')
        result['source_results'] = source_results
        result['case_results'] = case_results
        result['sheet_results'] = sheet_results
        result['selection_recomputation'] = {'old_optimization_count': len(old_opt), 'additional_optimization_reference_order': added, 'optimization_count': len(opt), 'adaptive_monitor_count': len(monitors), 'preflight_exclusions': sorted(preflight), 'deferred_exclusions': sorted(deferred), 'eligible_reserve_count': len(ranked), 'input_role_counts': dict(Counter(r['role'] for r in data['cases'])), 'input_cohort_counts': dict(Counter(r['cohort'] for r in data['cases']))}
        result['local_operator_semantics'] = {'actual': 'G2_float64 = Gaussian(original_clean256_float64, sigma=(2,2,0), reflect, truncate=3); G2_RGB8 = rint-clip-uint8(G2_float64); local = rint-clip-uint8(original_float64*(1-alpha) + float64(G2_RGB8)*alpha); alpha0 explicitly original', 'no_second_blur': True, 'alpha0_pixel_count_per_input': int(np.count_nonzero(alpha == 0)), 'alpha1_pixel_count_per_input': int(np.count_nonzero(alpha == 1)), 'unquantized_G2_float64_blend_would_be_a_different_operator': True, 'unquantized_alternative_comparison': alternative_changes, 'unquantized_alternative_total_differing_RGB_channels': sum(r['differing_RGB_channels'] for r in alternative_changes), 'unquantized_alternative_maximum_absolute_RGB8_difference': max(r['maximum_absolute_RGB8_difference'] for r in alternative_changes)}

        terms_path = BASE / 'independent_terms_audit_v1.json'
        terms = load(terms_path)
        result['independent_terms_audit'] = pin(terms_path)
        require(terms.get('complete') is True and terms.get('terms_metadata_audit_pass') is True and terms.get('source_count') == 32 and terms.get('all_source_attributions_exact_retained_root_clean_snapshots') is True, 'independent_terms_companion_complete_pass')
        supplement_path = BASE / 'local_operator_clarification_v1.json'
        if supplement_path.exists():
            supplement = load(supplement_path)
            result['local_operator_clarification'] = pin(supplement_path)
            supplement_serialized = json.dumps(supplement, sort_keys=True)
            require(data['source']['sha256'] in supplement_serialized and result['candidate']['sha256'] in supplement_serialized, 'operator_clarification_binds_immutable_candidate_and_source')
            require('RGB8' in supplement_serialized and 'float64' in supplement_serialized and ('rint' in supplement_serialized), 'operator_clarification_explicit_shared_RGB8_then_float64_blend')
        else:
            require(False, 'local_operator_clarification_record_required')
        result['next_requirements_outside_this_audit'] = ['Root personal input-only classification of all160 actual inputs; no clean-pair/model output use for degradation readability decisions', 'Separate explicit new data admission after root review; current candidate admission remains false', 'Separately sealed/manual execution evidence before any model, optimizer or VM use; no execution authorized by this data audit']
        result['complete'] = True
    except Exception as exc:
        blockers.append(type(exc).__name__ + ': ' + str(exc))
        result['exception'] = {'type': type(exc).__name__, 'message': str(exc)}
    after = []
    changed = []
    for name, original_pin in sorted(consumed.items()):
        current = pin(ROOT / name)
        after.append(current)
        if current != original_pin:
            changed.append(name)
    result['consumed_before'] = sorted(consumed.values(), key=lambda r: r['path'])
    result['consumed_after'] = after
    result['consumed_before_after_exact'] = not changed
    if changed:
        blockers.extend('consumed_file_changed:' + name for name in changed)
    result['consumed_file_count'] = len(consumed)
    result['checks'] = checks
    result['checks_passed'] = sum(c['pass'] for c in checks)
    result['checks_failed'] = sum(not c['pass'] for c in checks)
    result['unresolved_blockers'] = blockers
    result['data_audit_pass'] = result['complete'] and not blockers
    result['audit_source'] = pin(Path(__file__).resolve())
    destination = BASE / 'independent_data_audit_v1.json'
    with destination.open('x', encoding='utf8', newline='\n') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({k: result[k] for k in ('complete', 'data_audit_pass', 'checks_passed', 'checks_failed', 'consumed_file_count', 'consumed_before_after_exact', 'unresolved_blockers')}, indent=2))
    return 0 if result['data_audit_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(run())

"""Prepare deterministic paired data for the selected comparison; no model runtime."""
from datetime import datetime, timezone
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import sys


class NoModelFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tensorflow', 'jax', 'basicsr', 'facelib'}:
            raise ImportError('Data preparation cannot import model or training frameworks')


sys.meta_path.insert(0, NoModelFramework())
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION
from scipy.ndimage import gaussian_filter
import scipy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
CLEAN_REVIEW = ROOT / 'outputs/codeformer_original_spatial_correction_v1_preparation/data_audit_v1/root_clean_target_review_v1.json'
CLEAN_REVIEW_SHA = '2e06c37a59269a19582b4d3066af456ae6b330dcf8fd3f73b54f07366e9b804f'
RANK_PREFIX = '20261024-codeformer-matched-feature-learning-v1-reserve'
ROLE_ORDER = ('clear', 'global_gaussian2p0', 'local_gaussian2p0_soft', 'compound_g1p2_lr64_jpeg70', 'lowlight0p35_noise0p006')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': sha(raw)}


def png(array):
    output = io.BytesIO(); Image.fromarray(array).save(output, format='PNG')
    return output.getvalue()


def read_rgb(raw, size):
    with Image.open(io.BytesIO(raw)) as image:
        need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (size, size), 'Exact RGB PNG geometry required')
        image.load()
        return np.asarray(image).copy()


def main():
    need(not OUT.exists(), 'Retain any preceding or partial data preparation; destination must be new')
    consumed = {}

    def read(path, expected=None):
        path = path.resolve(strict=True)
        need(path.is_relative_to(ROOT) and path.is_file() and path.stat().st_size <= 16 * 1024**2, 'Bounded workspace file required')
        raw = path.read_bytes(); current = pin(path)
        need(len(raw) == current['bytes'] and sha(raw) == current['sha256'], 'File changed while read')
        if expected:
            need(all(current[k] == expected[k] for k in ('bytes', 'sha256')), 'Exact historical source pin mismatch')
        need(current['path'] not in consumed or current == consumed[current['path']], 'Concurrent source change')
        consumed[current['path']] = current
        return raw

    parent_raw = read(PARENT / 'protocol.json'); need(sha(parent_raw) == PARENT_SHA, 'Sealed parent protocol mismatch')
    parent = json.loads(parent_raw)
    manifest = json.loads(read(PARENT / parent['data']['provenance']['path'], parent['data']['provenance']))
    reuse = json.loads(read(PARENT / parent['data']['conditional_reuse']['path'], parent['data']['conditional_reuse']))
    clean_raw = read(CLEAN_REVIEW); need(sha(clean_raw) == CLEAN_REVIEW_SHA, 'Root clean admission mismatch')
    clean = json.loads(clean_raw)
    need(clean['root_personal_input_only_review_complete'] and len(clean['admitted_reference_ids']) == 64, 'Exact existing64 clean admissions required')
    records = {row['reference_id']: row for row in clean['admitted_references']}
    old_opt, monitors = reuse['optimization_reference_order'], reuse['monitor_reference_order']
    preflight = set(reuse['preflight4_readout_only_not_optimization_or_monitor']['photographs'])
    deferred = set(manifest['split_reference_ids']['DEFERRED_FROM_THIS_LEARNING_PILOT'])
    reserve = list(manifest['split_reference_ids']['UNUSED_TRAIN_RESERVE'])
    rank = lambda ref: sha((RANK_PREFIX + ':' + ref).encode('ascii'))
    candidates = sorted(set(reserve) - preflight - deferred - set(old_opt) - set(monitors), key=rank)
    additions = candidates[:8]
    opt = list(old_opt) + additions
    refs = opt + list(monitors)
    need(len(old_opt) == 16 and len(opt) == 24 and len(monitors) == 8 and len(set(refs)) == 32, 'Fixed24/8 photograph split required')
    need(set(refs) <= set(clean['admitted_reference_ids']) and not set(refs) & (preflight | deferred), 'Unadmitted/excluded source selected')
    # Freeze the photograph choice before rendering any new corruption or model output.
    OUT.mkdir()
    for sub in ('inputs', 'targets', 'input_only_sheets'):
        (OUT / sub).mkdir()
    selection = {'complete': True, 'status': 'USER_SELECTED_TWO_ARM_DESIGN_DATA_PREPARATION_ONLY',
        'user_selected_option': 'Matched two-arm comparison (Recommended)',
        'user_answer_evidence': 'Direct user reply to request_user_input_async call_6b10615660654f06b9324fa15354509f, question0',
        'decision_does_not_authorize_automatic_VM_training': True,
        'rank_prefix': RANK_PREFIX, 'eligible_reserve_rank': [{'reference_id': r, 'rank_sha256': rank(r)} for r in candidates],
        'retained_optimization_reference_order': old_opt, 'additional_optimization_reference_order': additions,
        'optimization_reference_order': opt, 'adaptive_monitor_reference_order': monitors,
        'preflight_readout_only_exclusions': sorted(preflight), 'deferred_exclusions': sorted(deferred),
        'roles': ROLE_ORDER, 'optimization_photographs': 24, 'adaptive_monitor_photographs': 8,
        'proposed_optimizer_updates_per_arm': 48, 'proposed_photograph_table_passes_per_arm': 2,
        'proposed_singleton_role_exposures_per_update': 5, 'maximum_training_exposures_both_arms': 480,
        'input_only_review_pending': True, 'new_data_optimization_admission': False,
        'old96_inputs_and_splits_not_modified': True, 'FINAL_DEV_native_wild_accessed': False}
    with (OUT / 'selection_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(selection, stream, indent=2, allow_nan=False); stream.write('\n')

    yy, xx = np.mgrid[:256, :256]
    radius = np.sqrt(((xx - 127.5) / 72)**2 + ((yy - 127.5) / 88)**2)
    fraction = np.clip((1.0 - radius) / 0.20, 0, 1)
    alpha = fraction * fraction * (3 - 2 * fraction)
    roles = {'clear': {'operator': 'Exact existing direct1024to256 LANCZOS photographic target bytes'},
        'global_gaussian2p0': {'operator': 'SciPy Gaussian RGB64 float; rint clipRGB8', 'sigma': [2.0, 2.0, 0], 'mode': 'reflect', 'truncate': 3.0, 'radius_pixels': 6},
        'local_gaussian2p0_soft': {'operator': 'Same original-target sigma2.0 Gaussian, original/blur float64 blend; rint clipRGB8; alpha0 pixels explicitly original',
            'ellipse_center_xy': [127.5, 127.5], 'ellipse_radii_xy': [72, 88], 'normalized_inner_radius': .8, 'normalized_outer_radius': 1.0,
            'alpha': 'smoothstep(clip((1-normalized_ellipse_radius)/0.20,0,1))',
            'scope': 'Synthetic corrupted-face input, not Protect/Apply preprocessing or removal-mask prediction'},
        'compound_g1p2_lr64_jpeg70': {'operator': 'Original clean256 -> sigma1.2 reflect/truncate3 float64 Gaussian -> rintRGB8 -> bilinear64 -> bilinear256 -> JPEG70 -> RGB decode',
            'JPEG': {'quality': 70, 'subsampling': 0, 'optimize': False, 'progressive': False}, 'corruption_order_fixed': True},
        'lowlight0p35_noise0p006': {'operator': 'Decode sRGB to linear, exposure0.35, encode sRGB; add independent zero-mean Gaussian encoded-RGB noise std0.006; half-up clipRGB8',
            'seed': 'first16 hex SHA256(20261024-matched-feature-data-v1:<reference>:lowlight0p35_noise0p006)',
            'generator': 'NumPy default_rng PCG64', 'camera_calibration_claim': False}}
    sources, cases = [], []
    for ref_index, ref in enumerate(refs):
        row = records[ref]
        source_raw = read(ROOT / row['physical_source']['path'], row['physical_source'])
        source = read_rgb(source_raw, 1024)
        need(sha(source.tobytes()) == row['physical_source']['RGB8_sha256'], 'Source pixel mismatch')
        target_path = ROOT / 'outputs/cctv_dgp_hq_cohort_v2/targets256' / (ref + '.png')
        target_raw = read(target_path)
        target = read_rgb(target_raw, 256)
        direct = np.asarray(Image.fromarray(source).resize((256, 256), Image.Resampling.LANCZOS))
        need(np.array_equal(direct, target), 'Target not direct photograph1024-to256 LANCZOS')
        need(row['historical_role'] == 'TRAIN' and row['official_category'] == 'training', 'TRAIN-only photographic source required')
        target_dest = OUT / 'targets' / (ref + '.png')
        with target_dest.open('xb') as stream:
            stream.write(target_raw)
        sources.append({'reference_id': ref, 'cohort': 'optimize' if ref in opt else 'adaptive_monitor',
            'source': pin(ROOT / row['physical_source']['path']), 'target': pin(target_dest),
            'target_RGB8_sha256': sha(target.tobytes()), 'direct1024to256_LANCZOS_exact': True,
            'historical_clean_admission': pin(CLEAN_REVIEW), 'attribution': row['attribution'],
            'person_identity_or_pretraining_nonoverlap_established': False})
        blur = lambda sigma: np.clip(np.rint(gaussian_filter(target.astype(np.float64), sigma=(sigma, sigma, 0), mode='reflect', truncate=3.0)), 0, 255).astype(np.uint8)
        global_blur = blur(2.0)
        local = np.clip(np.rint(target.astype(np.float64) * (1 - alpha[..., None]) + global_blur.astype(np.float64) * alpha[..., None]), 0, 255).astype(np.uint8)
        local[alpha == 0] = target[alpha == 0]
        compound = Image.fromarray(blur(1.2)).resize((64, 64), Image.Resampling.BILINEAR).resize((256, 256), Image.Resampling.BILINEAR)
        jpeg = io.BytesIO(); compound.save(jpeg, format='JPEG', quality=70, subsampling=0, optimize=False, progressive=False)
        with Image.open(io.BytesIO(jpeg.getvalue())) as image:
            compound_rgb = np.asarray(image.convert('RGB')).copy()
        values = target.astype(np.float64) / 255
        linear = np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055)**2.4) * .35
        encoded = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear**(1 / 2.4) - .055)
        seed = int(sha(('20261024-matched-feature-data-v1:' + ref + ':' + ROLE_ORDER[-1]).encode('ascii'))[:16], 16)
        noise = np.random.default_rng(seed).normal(0, .006, target.shape)
        lowlight = np.floor(np.clip(encoded + noise, 0, 1) * 255 + .5).astype(np.uint8)
        arrays = (target, global_blur, local, compound_rgb, lowlight)
        for role_index, (role, observed) in enumerate(zip(ROLE_ORDER, arrays)):
            case_id = ref + '__' + role
            filename = OUT / 'inputs' / (case_id + '.png')
            raw = target_raw if role == 'clear' else png(observed)
            with filename.open('xb') as stream:
                stream.write(raw)
            anonymous = 'D' + sha(('matched-feature-input-review-v1:' + case_id).encode('ascii'))[:10]
            cases.append({'id': case_id, 'reference_id': ref, 'role': role, 'cohort': 'optimize' if ref in opt else 'adaptive_monitor',
                'historical_split': 'TRAIN', 'input': pin(filename), 'target': pin(target_dest),
                'input_RGB8_sha256': sha(observed.tobytes()), 'target_RGB8_sha256': sha(target.tobytes()),
                'noise_seed': seed if role == ROLE_ORDER[-1] else None, 'anonymous_review_id': anonymous,
                'input_only_decision': 'PENDING', 'optimization_permitted': False,
                'native_CCTV': False, 'paired_synthetic': True})
    # Input-only sheets deliberately keep the clear references separate from degraded observations.
    sheets, cells = [], []
    for label, rows in (('degraded', [r for r in cases if r['role'] != 'clear']), ('clear', [r for r in cases if r['role'] == 'clear'])):
        order = sorted(rows, key=lambda r: sha(('anonymous-input-order-v1:' + r['id']).encode('ascii')))
        for start in range(0, len(order), 16):
            page_rows = order[start:start + 16]
            sheet = Image.new('RGB', (4 * 272 + 16, 4 * 304 + 70), '#e9e6dd')
            draw = ImageDraw.Draw(sheet)
            draw.text((16, 10), 'INPUT-ONLY ' + label + ' | exact RGB256 | no model output or paired clean reference', fill='black')
            draw.text((16, 30), 'Classify visible structure: readable / uncertain / insufficient. Blur simulation is not camera calibration.', fill='black')
            local_cells = []
            for index, row in enumerate(page_rows):
                x, y = 16 + (index % 4) * 272, 62 + (index // 4) * 304
                observed = read_rgb((ROOT / row['input']['path']).read_bytes(), 256)
                draw.text((x, y), row['anonymous_review_id'], fill='black')
                sheet.paste(Image.fromarray(observed), (x, y + 22))
                cell = {'anonymous_review_id': row['anonymous_review_id'], 'case_id': row['id'],
                        'input': row['input'], 'bounds_xyxy': [x, y + 22, x + 256, y + 278]}
                local_cells.append(cell)
            path = OUT / 'input_only_sheets' / (label + '_' + str(start // 16 + 1).zfill(2) + '.png')
            with path.open('xb') as stream:
                sheet.save(stream, format='PNG')
            sheets.append({**pin(path), 'kind': label, 'cells': local_cells})
            cells.extend(local_cells)
    need(len(cases) == 160 and len(sheets) == 10 and len(cells) == 160, 'Exact160 inputs/10sheets required')
    need(all(pin(ROOT / name) == value for name, value in consumed.items()), 'Historical inputs changed during preparation')
    manifest_out = {'complete': True, 'status': 'DATA_RENDERED_INPUT_ONLY_REVIEW_PENDING_NOT_OPTIMIZATION_ADMITTED',
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'selection': pin(OUT / 'selection_v1.json'),
        'sources': sources, 'cases': cases, 'operators': roles, 'sheets': sheets,
        'source_count': 32, 'optimization_photographs': 24, 'adaptive_monitor_photographs': 8,
        'paired_inputs': 160, 'degraded_inputs': 128, 'clear_controls': 32,
        'versions': {'NumPy': np.__version__, 'Pillow': PILLOW_VERSION, 'SciPy': scipy.__version__},
        'consumed_before_after_exact': sorted(consumed.values(), key=lambda p: p['path']),
        'source': pin(Path(__file__).resolve()), 'new_training_model_gradient_optimizer_VM_browser_calls': 0,
        'old_sources_inputs_targets_splits_modified': False, 'input_only_review_complete': False,
        'new_optimization_admission': False, 'training_packet_or_new_worker_ready': False,
        'native_DEV_FINAL_or_wild_inputs_accessed': False, 'terms_are_retained_snapshots': True,
        'no_source_ethnicity_or_Zamboanga_performance_claim': True, 'model_qualified': False}
    with (OUT / 'candidate_data_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(manifest_out, stream, indent=2, allow_nan=False); stream.write('\n')
    print({'complete': True, 'sources': 32, 'paired_inputs': 160, 'input_only_sheets': 10,
           'optimization_admitted': False, 'actual_training_or_model_calls': 0, 'additional_refs': additions})


if __name__ == '__main__':
    main()

"""Independent bounded-fusion saved-pixel audit. CPU only; no neural imports."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import time
import traceback

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

PROTOCOL_SHA = 'c3b0f5f0f3ff83b508625df0259ed0c9acbc2d071da2fdde066a2bda429a15eb'
ARCHIVE_SHA = '3e1a0a4bee698b397747ba39d3546cbca3cbd298e9cb450bc1abfc61aa880bb9'
MANIFEST_SHA = 'c9286d80ac6586aa281aa91bb00b6b430521b2d797f79308f450559c244548b3'
MODES = ('original', 'stopped', 'rollback', 'bounded_constant', 'bounded_fixed')
VIEWS = ('native512', 'raw256', 'native_png')
Q10, Q90 = .21007671428419458, .5546973290483728
WEIGHTS = np.asarray([.2126, .7152, .0722], dtype=np.float64)
METRICS = ('mae', 'mse', 'psnr', 'ssim', 'mean_luminance', 'target_mean_luminance',
           'signed_luminance_bias', 'luminance_bias', 'luminance_contrast',
           'target_luminance_contrast', 'contrast_absolute_error', 'coarse_rgb_mae',
           'gradient_vector_mae', 'gradient_magnitude_mae')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inside(root, relative):
    posix = PurePosixPath(relative)
    require(isinstance(relative, str) and not posix.is_absolute() and posix.as_posix() == relative
        and '..' not in posix.parts and '\\' not in relative and ':' not in relative, 'Noncanonical evidence path')
    path = Path(root).joinpath(*posix.parts)
    require(path.resolve(strict=True).is_relative_to(Path(root).resolve(strict=True)) and
        path.is_file() and not path.is_symlink(), 'Evidence path escapes or is not a regular file')
    return path


def receipt_path(root, receipt):
    require(set(receipt) == {'path', 'sha256', 'bytes'}, 'Artifact receipt schema differs')
    path = inside(root, receipt['path'])
    require(path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256'], 'Artifact receipt bytes/hash differ')
    return path


def rgb8(path, size):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (size, size), 'Expected exact RGB PNG geometry')
        return np.asarray(image).copy()


def array(path, shape, dtype):
    with Path(path).open('rb') as stream:
        value = np.load(stream, allow_pickle=False)
        require(stream.read(1) == b'', 'Trailing NPY bytes')
    require(value.shape == shape and value.dtype == np.dtype(dtype) and value.flags.c_contiguous
        and np.isfinite(value).all(), 'Saved array geometry/dtype/finite/contiguity differs')
    return value


def gate(pixels):
    require(pixels.dtype == np.uint8 and pixels.shape == (256, 256, 3), 'Gate uses observed RGB8 only')
    z = float(((pixels.astype(np.float64) / 255.) @ WEIGHTS)[32:224, 32:224].mean())
    t = min(1., max(0., (Q90 - z) / (Q90 - Q10)))
    return {'central192_sRGB_mean_luminance': z, 'gate': .25 + .75 * t * t * (3 - 2 * t)}


def gate_metadata(saved, rebuilt, tolerance):
    require(tolerance == 1e-12 and set(saved) == set(rebuilt) == {'central192_sRGB_mean_luminance', 'gate'},
        'Only prospectively frozen scalar metadata policy is allowed')
    errors = {}
    for key in rebuilt:
        require(type(saved[key]) is float and math.isfinite(saved[key]), 'Finite binary64 gate metadata required')
        errors[key] = abs(saved[key] - rebuilt[key])
        require(errors[key] <= tolerance, 'Actual observed gate exceeds frozen scalar-roundoff allowance')
    require(np.float32(saved['gate']) == np.float32(rebuilt['gate']), 'Applied FP32 gate scalar differs')
    return errors


def delivery_check(native, raw):
    """Source-derived positive FP32 arithmetic bound, never a quality tolerance.

    align_corners=False at exactly2x downsampling samples each2x2 center.
    All weights are exact binary powers; at most3 rounded additions occur.
    Compare to an ideal float64 average with gamma3 and subnormal allowance.
    """
    require(native.shape == (512, 512, 3) and native.dtype == np.float32 and
        raw.shape == (256, 256, 3) and raw.dtype == np.float32 and
        np.isfinite(native).all() and np.isfinite(raw).all(), 'Delivery geometry/dtype/finite differs')
    require(raw.min() >= 0 and raw.max() <= 1, 'Delivered RAW must be in [0,1]')
    clipped = np.clip(native, np.float32(0), np.float32(1)).astype(np.float64)
    ideal = clipped.reshape(256, 2, 256, 2, 3).mean(axis=(1, 3))
    unit = float(np.finfo(np.float32).eps) / 2
    gamma3 = 3 * unit / (1 - 3 * unit)
    tolerance = gamma3 * ideal + 4 * float(np.nextafter(np.float32(0), np.float32(1)))
    difference = np.abs(raw.astype(np.float64) - ideal)
    require((difference <= tolerance).all(), 'Saved RAW is inconsistent with clamp-before-bilinear delivery')
    return {'maximum_absolute_error_to_ideal_2x2': float(difference.max()),
        'maximum_pointwise_roundoff_allowance': float(tolerance.max()),
        'gamma3': gamma3, 'pointwise_pass': True,
        'policy': 'Ideal float64 mean of clamped saved native512; positive4term FP32 gamma3 bound, no quality threshold change'}


def paired_metrics(prediction, target):
    require(prediction.shape == target.shape and prediction.shape in ((256, 256, 3), (512, 512, 3))
        and np.isfinite(prediction).all() and np.isfinite(target).all(), 'Paired metric geometry/finite differs')
    p, t = prediction.astype(np.float64), target.astype(np.float64)
    difference = p - t
    mse = float(np.square(difference).mean(dtype=np.float64))
    yp, yt = p @ WEIGHTS, t @ WEIGHTS
    size = p.shape[0]
    step = size // 16
    pp = p.reshape(16, step, 16, step, 3).mean(axis=(1, 3))
    pt = t.reshape(16, step, 16, step, 3).mean(axis=(1, 3))
    xp, xt = (yp[1:-1, 2:] - yp[1:-1, :-2]) / 2, (yt[1:-1, 2:] - yt[1:-1, :-2]) / 2
    gp, gt = (yp[2:, 1:-1] - yp[:-2, 1:-1]) / 2, (yt[2:, 1:-1] - yt[:-2, 1:-1]) / 2
    signed = float((yp - yt).mean())
    return {'mae': float(np.abs(difference).mean()), 'mse': mse,
        'psnr': None if mse == 0 else float(-10 * np.log10(mse)),
        'ssim': float(structural_similarity(p, t, data_range=1., win_size=7, channel_axis=2,
            gaussian_weights=False, use_sample_covariance=True, K1=.01, K2=.03)),
        'mean_luminance': float(yp.mean()), 'target_mean_luminance': float(yt.mean()),
        'signed_luminance_bias': signed, 'luminance_bias': abs(signed),
        'luminance_contrast': float(yp.std()), 'target_luminance_contrast': float(yt.std()),
        'contrast_absolute_error': abs(float(yp.std()) - float(yt.std())),
        'coarse_rgb_mae': float(np.abs(pp - pt).mean()),
        'gradient_vector_mae': float(((np.abs(xp - xt) + np.abs(gp - gt)) / 2).mean()),
        'gradient_magnitude_mae': float(np.abs(np.hypot(xp, gp) - np.hypot(xt, gt)).mean())}


def summarize(rows, key_names):
    buckets = defaultdict(list)
    for row in rows:
        for scope in ('exposure', 'profile'):
            key = tuple(row[k] for k in key_names) + (scope, row[scope])
            buckets[key].append(row)
    result = []
    for key, items in sorted(buckets.items()):
        row = dict(zip((*key_names, 'grouping', 'group'), key), cases=len(items))
        for name in METRICS:
            values = [r[name] for r in items if r[name] is not None]
            row[name] = {'mean': float(np.mean(values)), 'minimum': min(values), 'maximum': max(values),
                         'cases_below_zero': sum(x < 0 for x in values), 'cases_above_zero': sum(x > 0 for x in values)} if values else None
        result.append(row)
    return result


def clear_subset(rows):
    """Descriptive4 normal-clear TRAIN controls, never the complete61 DEV gate."""
    lookup = {(r['case_id'], r['mode'], r['view']): r for r in rows if r['section'] == 'paths'}
    ids = sorted({r['case_id'] for r in rows if r['profile'] == 'clear_normal' and r['section'] == 'paths'})
    require(len(ids) == 4, 'This protocol has exactly4 normal-clear TRAIN controls')
    result = []
    for mode in MODES:
        for view in ('raw256', 'native_png'):
            values = {rid: lookup[rid, mode, view]['mae'] - lookup[rid, 'original', view]['mae'] for rid in ids}
            mean, worst = float(np.mean(list(values.values()))), max(values.values())
            result.append({'mode': mode, 'view': view, 'cases': 4, 'mean_mae_delta': mean,
                'worst_mae_delta': worst, 'mean_limit': .01, 'per_case_limit': .03,
                'descriptive_subset_within_thresholds': mean <= .01 and worst <= .03,
                'cases_exceeding_per_case_limit': {k: v for k, v in values.items() if v > .03}, 'all_deltas': values})
    return {'scope': 'Only4 clear_normal TRAIN cases; unchanged .01/.03 thresholds shown descriptively',
        'full61_clear_and53_blur_DEV_guards_evaluated': False, 'prior_FP32_guard_failure_overridden': False,
        'model_qualification': False, 'subsets': result}


def input_specs(packet, p):
    calibration = []
    inputs = {}
    for row in p['calibration_inputs']:
        path = inside(packet, row['input_path'])
        require(sha(path) == row['input_sha256'] == p['files_sha256'][row['input_path']], 'Retained input PNG binding differs')
        pixels = rgb8(path, 256)
        require(pixel_sha(pixels) == row['input256_rgb8_sha256'] and pixel_sha(
            pixels.astype(np.float32) / np.float32(255)) == row['input256_float32_sha256'], 'Retained input pixel binding differs')
        rebuilt = gate(pixels)
        require(rebuilt['central192_sRGB_mean_luminance'] == row['central192_sRGB_mean_luminance'] and
            rebuilt['gate'] == row['gate'], 'Input-only photometry/gate differs')
        inputs[row['id']] = (row, path, pixels)
        calibration.append(rebuilt['central192_sRGB_mean_luminance'])
    require(len(inputs) == 200 and list(np.quantile(calibration, [.1, .9], method='linear')) == [Q10, Q90], 'Exact200 input-only calibration differs')
    refs = read(inside(packet, 'references.json'))
    require(len(refs['references']) == 16, 'Exactly16 genuine source references required')
    targets = {}
    source_receipts = []
    for ref in refs['references']:
        require(ref['historical_role'] == 'train' and ref['decision'] == 'accept_clean_restoration', 'Only audited TRAIN sources permitted')
        source = inside(packet, ref['source_path'])
        require(sha(source) == ref['source_sha256'] == p['files_sha256'][ref['source_path']], 'Genuine source hash differs')
        with Image.open(source) as image:
            require(image.mode == 'RGB' and image.size == (1024, 1024), 'Genuine source geometry differs')
            for size in (512, 256):
                path = inside(packet, 'data/clean%d/%s.png' % (size, ref['id']))
                require(sha(path) == p['files_sha256'][path.relative_to(packet).as_posix()], 'Metric-only target bytes differ')
                derived = image.resize((size, size), Image.Resampling.LANCZOS)
                pixels = rgb8(path, size)
                require(np.array_equal(pixels, np.asarray(derived)), 'Direct source target pixels differ')
                if size == 512:
                    require(pixel_sha(pixels) == ref['proposed_512_rgb_sha256'], 'Historical clean512 derivation differs')
                targets[ref['id'], size] = (path, pixels.astype(np.float32) / np.float32(255))
        source_receipts.append({'reference_id': ref['id'], 'source_sha256': ref['source_sha256'],
            'direct512_sha256': sha(targets[ref['id'], 512][0]), 'direct256_sha256': sha(targets[ref['id'], 256][0]),
            'both_direct_target_pixels_exact': True})
    specs = {}
    for row in p['cases']:
        original, path, pixels = inputs[row['id']]
        require(all(row[k] == v for k, v in original.items()), 'Selected input/cohort differs')
        target256, t256 = targets[row['reference_id'], 256]
        target512, t512 = targets[row['reference_id'], 512]
        require(sha(target256) == row['target256_sha256'] and sha(target512) == row['target512_sha256'], 'Selected target binding differs')
        specs[row['id']] = {**row, 'input_absolute': str(path), 'input_pixels': pixels,
            'target256_absolute': str(target256), 'target512_absolute': str(target512),
            'target256_pixels': t256, 'target512_pixels': t512,
            'exposure': row['profile'].rsplit('_', 1)[1]}
    require(len(specs) == 32 and len({s['reference_id'] for s in specs.values()}) == 16, 'Exact32/16 selected cohort required')
    return specs, source_receipts


def expected_plan(ids, canaries):
    result = [(rid, mode, 'paths') for rid in ids for mode in MODES]
    result += [(rid, mode, 'repeats') for rid in canaries for mode in ('original', 'stopped')]
    result += [(rid, mode, 'zero_delta') for rid in canaries for mode in ('bounded_constant', 'bounded_fixed')]
    result += [(rid, 'native_identity', 'identity') for rid in canaries]
    return result


def audit(imported, packet, output, transport):
    imported, packet = Path(imported).resolve(strict=True), Path(packet).resolve(strict=True)
    output = Path(output).resolve()
    transport_receipt = read(transport)
    require(transport_receipt.get('complete') is True and
        transport_receipt.get('format') == 'bounded-fusion-feasibility-strict-return-transport-v1' and
        transport_receipt.get('archive_sha256') == ARCHIVE_SHA and transport_receipt.get('archive_bytes') == 668557079 and
        transport_receipt.get('protocol_sha256') == PROTOCOL_SHA and transport_receipt.get('manifest_sha256') == MANIFEST_SHA and
        Path(transport_receipt['import_directory']).resolve(strict=True) == imported and
        transport_receipt.get('exact_namespace_verified') is True and
        transport_receipt.get('whole_archive_and_all_member_hashes_verified_before_import') is True and
        transport_receipt.get('member_maps_include_manifest') is True and
        transport_receipt.get('returned_code_executed') is False,
        'Exact closed transport/archive/protocol/manifest/destination binding required')
    require(sha(imported / 'protocol.json') == sha(packet / 'protocol.json') == PROTOCOL_SHA, 'Frozen actual protocol differs')
    require(not output.exists(), 'Preserve prior audit evidence; use a new output directory')
    started = time.monotonic()
    output.mkdir(parents=True)
    p = read(packet / 'protocol.json')
    numerical_policy_path = inside(packet, p['numerical_return_policy']['path'])
    require(sha(numerical_policy_path) == p['numerical_return_policy']['sha256'], 'Frozen numerical policy differs')
    numerical_policy = read(numerical_policy_path)
    metadata_tolerance = numerical_policy['calibration_gate_abs_tolerance']
    require(metadata_tolerance == 1e-12 and numerical_policy['roundoff_policy_selected_before_new_measurement'] is True,
        'Prospective metadata-roundoff policy required')
    manifest_path = inside(imported, transport_receipt['manifest_member'])
    require(sha(manifest_path) == MANIFEST_SHA and
        transport_receipt['files_sha256'][transport_receipt['manifest_member']] == MANIFEST_SHA,
        'Actual imported manifest binding differs')
    def transport_bound(path):
        name = Path(path).relative_to(imported).as_posix()
        require(transport_receipt['files_sha256'].get(name) == sha(path) and
            transport_receipt['files_bytes'].get(name) == Path(path).stat().st_size,
            'Imported scientific artifact no longer matches closed transport')
    specs, source_receipts = input_specs(packet, p)
    run = imported / p['outputs']
    transport_bound(run / 'run_outputs.json')
    saved = read(run / 'run_outputs.json')
    require(saved['complete'] and saved['counts'] == p['call_caps'] and saved['optimizer_updates'] == saved['gradient_queries'] == 0, 'Complete finite zero-update measurement required')
    plan = expected_plan(list(specs), p['canary_ids'])
    require(len(plan) == 170 and [(r['case_id'], r['mode'], r['section']) for r in saved['results']] == plan, 'Actual170 output namespace/order differs')
    rows, visual, controls, delivery, signatures, gate_errors = [], [], [], [], {}, []
    input_controls = []
    for spec in specs.values():
        input_controls.append({'case_id': spec['id'], 'profile': spec['profile'], 'exposure': spec['exposure'],
            'reference_id': spec['reference_id'], **paired_metrics(spec['input_pixels'].astype(np.float32) / np.float32(255), spec['target256_pixels'])})
    for index, item in enumerate(saved['results'], 1):
        rid, mode, section = item['case_id'], item['mode'], item['section']
        spec = specs[rid]
        directory = section + '/' + rid + '/' + mode
        record_path = receipt_path(imported, item['record'])
        transport_bound(record_path)
        require(item['record']['path'] == p['outputs'] + '/' + directory + '/record.json', 'Output record path differs')
        record = read(record_path)
        require((record['case_id'], record['mode'], record['section']) == (rid, mode, section), 'Saved record identity differs')
        require(record['input_path'] == spec['input_path'] and record['input_sha256'] == spec['input_sha256'] and record['reference_id'] == spec['reference_id'], 'Saved input binding differs')
        gate_errors.append({'case_id': rid, 'mode': mode, 'section': section,
            **gate_metadata(record['gate_observed_input'], gate(spec['input_pixels']), metadata_tolerance)})
        require(record['targets_for_independent_metrics_only'] == {k: spec[k] for k in ('target512_path', 'target512_sha256', 'target256_path', 'target256_sha256')}, 'Metric-only target binding differs')
        artifacts = record['artifacts']
        require(set(artifacts) == {'native512', 'raw256', 'PNG', 'tokens'}, 'Exact saved output artifacts required')
        paths = {key: receipt_path(imported, value) for key, value in artifacts.items()}
        for path in paths.values():
            transport_bound(path)
        require({key: path.name for key, path in paths.items()} == {'native512': 'native512.npy', 'raw256': 'raw256.npy', 'PNG': 'native.png', 'tokens': 'actual_tokens.npy'}, 'Artifact filename policy differs')
        native = array(paths['native512'], (512, 512, 3), 'float32')
        raw = array(paths['raw256'], (256, 256, 3), 'float32')
        tokens = array(paths['tokens'], (256,), 'int64')
        require(((tokens >= 0) & (tokens < 1024)).all(), 'Actual code indices outside dictionary')
        pixels = rgb8(paths['PNG'], 256)
        require(np.array_equal(pixels, np.floor(raw * np.float32(255)).astype(np.uint8)), 'Delivered PNG is not exact floor(RAW*255)')
        require(record['pixel_hashes'] == {'native512': pixel_sha(native), 'raw256': pixel_sha(raw), 'tokens': pixel_sha(tokens)}, 'Actual pixel/token hash binding differs')
        delivery.append({'case_id': rid, 'mode': mode, 'section': section, **delivery_check(native, raw),
            'native_minimum': float(native.min()), 'native_maximum': float(native.max()),
            'native_fraction_below_zero': float((native < 0).mean()), 'native_fraction_above_one': float((native > 1).mean()),
            'PNG_floor_exact': True})
        signature = (pixel_sha(native), pixel_sha(raw), pixel_sha(tokens), pixel_sha(pixels))
        if section == 'paths' and mode in ('original', 'stopped'):
            signatures[rid, mode] = signature
        if section in ('repeats', 'identity', 'zero_delta'):
            expected = 'original' if section == 'zero_delta' else 'stopped' if section == 'identity' else mode
            require(signature == signatures[rid, expected], 'Repeat/zero/native-identity saved pixels/tokens differ')
            controls.append({'case_id': rid, 'mode': mode, 'section': section, 'expected_mode': expected,
                'all_native_RAW_PNG_tokens_exact': True})
        for view, prediction, target in [('native512', native, spec['target512_pixels']), ('raw256', raw, spec['target256_pixels']),
                                         ('native_png', pixels.astype(np.float32) / np.float32(255), spec['target256_pixels'])]:
            rows.append({'case_id': rid, 'reference_id': spec['reference_id'], 'profile': spec['profile'],
                'exposure': spec['exposure'], 'section': section, 'mode': mode, 'view': view,
                'evidence': 'paired_synthetic_TRAIN', **paired_metrics(prediction, target)})
        visual.append({'case_id': rid, 'reference_id': spec['reference_id'], 'profile': spec['profile'],
            'exposure': spec['exposure'], 'mode': mode, 'section': section, 'is_stress_canary': rid in p['canary_ids'],
            'input': spec['input_absolute'], 'input_sha256': spec['input_sha256'],
            'target': spec['target256_absolute'], 'target_sha256': spec['target256_sha256'],
            'native_png': str(paths['PNG']), 'native_png_sha256': artifacts['PNG']['sha256'],
            'native512': str(paths['native512']), 'native512_sha256': artifacts['native512']['sha256'],
            'raw256': str(paths['raw256']), 'raw256_sha256': artifacts['raw256']['sha256'],
            'gate': spec['gate'], 'central192_sRGB_mean_luminance': spec['central192_sRGB_mean_luminance']})
        if index % 20 == 0 or index == len(saved['results']):
            print(json.dumps({'CPU_saved_pixel_audit': index, 'of': 170, 'seconds': time.monotonic() - started}), flush=True)
    main = [r for r in rows if r['section'] == 'paths']
    require(len(rows) == 510 and len(main) == 480 and len(controls) == 10, 'Metric/control row census differs')
    lookup = {(r['case_id'], r['mode'], r['view']): r for r in main}
    delta_rows = []
    for baseline_index, baseline in enumerate(MODES):
        for candidate in MODES[baseline_index + 1:]:
            for rid in specs:
                for view in VIEWS:
                    a, b = lookup[rid, candidate, view], lookup[rid, baseline, view]
                    delta_rows.append({k: a[k] for k in ('case_id', 'reference_id', 'profile', 'exposure', 'view')} | {
                        'comparison': candidate + '_minus_' + baseline,
                        **{name: a[name] - b[name] if a[name] is not None and b[name] is not None else None for name in METRICS}})
    write(output / 'metrics.json', {'complete': True, 'rows': rows, 'input256_controls': input_controls,
        'groups': summarize(main, ('mode', 'view')), 'policy': {'scope': '32 saved PNG TRAIN observations /16 photographic references; no independent people/DEV/FINAL/native CCTV claim',
        'normal_dark_separate': True, 'native512': 'Unbounded RGB float32; genuine direct clean512 target',
        'raw256': 'Clamp before bilinear256; genuine direct clean256 target', 'native_png': 'Exact floor(raw256*255) RGB8 then float32/255',
        'target_normalization': 'RGB8 to float32/255 before float64 metric arithmetic',
        'SSIM': 'RGB float64, data_range1, win7, sample covariance, Gaussianfalse, K1=.01 K2=.03; full support with default3pixel valid-window border',
        'coarse_RGB': '16x16 grid means: native512 block32; delivered256 block16',
        'luma': 'sRGB weighted .2126/.7152/.0722, not linear-light or identity evidence',
        'gradients': 'Central luminance differences, border1 removed; texture/error descriptors, not perceptual identity metrics',
        'PSNR_perfect': 'NULL for MSE0; no nonfinite JSON'}})
    write(output / 'comparisons.json', {'complete': True, 'rows': delta_rows,
        'groups': summarize(delta_rows, ('comparison', 'view')), 'difference': 'candidate minus baseline',
        'interpretation': 'Lower MAE/MSE/bias/coarse/gradient/contrast errors and higher PSNR/SSIM are descriptor directions; no visual or identity verdict'})
    write(output / 'clear_TRAIN_subset.json', clear_subset(rows))
    write(output / 'delivery_and_controls.json', {'complete': True, 'delivery': delivery, 'exact_controls': controls,
        'output_PNGs': 170, 'native512_arrays': 170, 'raw256_arrays': 170, 'token_arrays': 170,
        'all_PNG_floor_exact': True, 'all_delivery_pointwise_roundoff_checks_pass': True})
    write(output / 'inputs_targets.json', {'complete': True, 'all200_inputs_and_quantiles_exact': True,
        'selected_inputs': 32, 'photographic_references': 16, 'source_target_bindings': source_receipts,
        'genuine_direct_source_target_pixel_derivations_exact': True, 'unique_people_not_established': True,
        'reserved_DEV_FINAL_native_images_read': 0,
        'VM_scalar_gate_metadata_policy': {'prospective_abs_tolerance': metadata_tolerance,
            'maximum_luminance_difference': max(r['central192_sRGB_mean_luminance'] for r in gate_errors),
            'maximum_gate_difference': max(r['gate'] for r in gate_errors),
            'all_applied_float32_gate_scalars_exact': True, 'rows': gate_errors}})
    write(output / 'visual_manifest.json', {'complete': True, 'cases': visual, 'main_cases': list(specs),
        'canary_ids': p['canary_ids'], 'unique_saved_outputs': 170, 'root_visual_review_pending': True,
        'model_qualified': False, 'policy': 'Exact saved delivered PNGs; display may not alter pixels'})
    with (output / 'metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    receipt = {'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'source_sha256': sha(__file__),
        'strict_transport_receipt': str(Path(transport).resolve()), 'strict_transport_sha256': sha(transport),
        'transport_binding': {key: transport_receipt[key] for key in ('archive_sha256', 'archive_bytes', 'protocol_sha256',
            'manifest_sha256', 'manifest_member', 'import_directory', 'members', 'payload_bytes',
            'exact_namespace_verified', 'whole_archive_and_all_member_hashes_verified_before_import')},
        'saved_output_count': 170, 'metric_rows': 510, 'main_metric_rows': 480, 'comparison_delta_rows': len(delta_rows),
        'TRAIN_observations': 32, 'photographic_references': 16, 'exact_controls': 10,
        'input_source_target_gate_hash_and_pixel_checks_pass': True, 'all_floor_delivery_and_control_checks_pass': True,
        'full_DEV_guards_or_model_qualification': False, 'root_visual_review_pending': True,
        'local_neural_gradient_optimizer_VM_calls': 0, 'seconds': time.monotonic() - started,
        'files_sha256': {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    write(output / 'metrics_audit.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imported', required=True, type=Path)
    parser.add_argument('--packet', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--transport', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.imported, args.packet, args.output, args.transport)
        print(json.dumps({key: value for key, value in result.items() if key != 'files_sha256'}), flush=True)
    except BaseException as error:
        if args.output.exists() and not (args.output / 'audit_failure.json').exists():
            write(args.output / 'audit_failure.json', {'complete': False, 'error_type': type(error).__name__,
                'error': str(error), 'traceback': traceback.format_exc(), 'source_sha256': sha(__file__),
                'model_qualified': False, 'local_neural_gradient_optimizer_VM_calls': 0})
        raise


if __name__ == '__main__':
    main()

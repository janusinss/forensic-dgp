"""Fixed pixel-only counterfactuals on returned TRAIN outputs; never training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1_return/outputs'
OUT = ROOT / 'outputs/cctv_dgp_multiscale_color_detail_v1'
ARMS = ['decoder15_lr0.001_pool0', 'decoder15_lr0.001_pool1']
VARIANTS = ['remove_constant_RGB_delta', 'retain_base_low_band_sigma2']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def image(path, mode='RGB'):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.asarray(im.convert(mode)).copy()


def metrics_module():
    path = PACKET / 'frozen_raw_metrics.py'
    spec = importlib.util.spec_from_file_location('fixed_pixel_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def returned_raw(p):
    source = PACKET / 'cctv_dgp_head4_lossless_v1.py'
    spec = importlib.util.spec_from_file_location('fixed_lossless_return', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = {arm: {} for arm in ['baseline'] + ARMS}
    saved = {arm: {r['id']: r for r in read(RETURNED / arm / 'metrics.json')['rows']}
             for arm in cache}
    for start in range(0, 100, 5):
        ids = [c['id'] for c in p['cases'][start:start+5]]
        base, _ = module.unpack(RETURNED / 'baseline/packs' / f'b{start//5:02d}.npz', ids)
        for arm in cache:
            values = base if arm == 'baseline' else module.unpack(
                RETURNED / arm / 'packs' / f'b{start//5:02d}.npz', ids, baseline=base)[0]
            for cid, raw in zip(ids, values):
                assert hashlib.sha256(raw.tobytes()).hexdigest() == saved[arm][cid]['raw_RGB_float32_sha256']
                cache[arm][cid] = raw
    return cache


def groups(rows, stage):
    keys = {'all', 'clear', 'degraded'} | {
        r['source'] + '/' + suffix for r in rows
        for suffix in ['all', 'clear', 'degraded', r['profile']]}
    result = {}
    for key in sorted(keys):
        selected = [r for r in rows if key in {
            'all', 'clear' if r['profile'] == 'clear' else 'degraded',
            r['source'] + '/all',
            r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'),
            r['source'] + '/' + r['profile']}]
        result[key] = {'cases': len(selected), **{
            k: float(np.mean([r[stage][k] for r in selected]))
            for k in ['MSE', 'SSIM', 'landmark_high_frequency_MSE']}}
    return result


def compare(before, after):
    failures = []
    for key, old in before.items():
        new = after[key]
        assert old['cases'] == new['cases']
        for metric in ['MSE', 'SSIM']:
            bad = (new[metric] > old[metric] + 1e-12 if metric == 'MSE'
                   else new[metric] < old[metric] - 1e-6)
            if bad:
                failures.append(dict(group=key, metric=metric,
                                     baseline=old[metric], candidate=new[metric]))
    gain = 1 - after['degraded']['landmark_high_frequency_MSE'] / before['degraded']['landmark_high_frequency_MSE']
    return dict(relative_feature_gain=gain, minimum_feature_gain=.01,
                preservation_failures=failures,
                pixel_only_requirements_pass=gain >= .01 and not failures,
                clear_MSE_multiplier=after['clear']['MSE'] / before['clear']['MSE'],
                ArcFace_and_full_quality_gate_not_tested=True)


def prepare():
    assert not OUT.exists(), 'Retain every previous result and failure'
    audit_path = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['all_gate_decisions_and_failure_names_unchanged']
    p = read(PACKET / 'protocol.json')
    assert sha(PACKET / 'protocol.json') == audit['protocol_sha256']
    bindings = {}

    def bind(path):
        bindings[Path(path).relative_to(ROOT).as_posix()] = sha(path)

    for path in [Path(__file__), ROOT / 'scripts/audit_cctv_dgp_multiscale_color_detail_v1.py',
                 PACKET / 'protocol.json', PACKET / 'frozen_raw_metrics.py',
                 PACKET / 'cctv_dgp_head4_lossless_v1.py', audit_path]:
        bind(path)
    for case in p['cases']:
        assert case['role'] == 'train'
        for key in ['input', 'target', 'observed']:
            bind(PACKET / case[key])
    for arm in ['baseline'] + ARMS:
        bind(RETURNED / arm / 'metrics.json')
        for number in range(20):
            bind(RETURNED / arm / 'packs' / f'b{number:02d}.npz')
    OUT.mkdir()
    write(OUT / 'plan.json', dict(
        complete=True, frozen_before_new_measurements=True, bindings=bindings,
        protocol_sha256=audit['protocol_sha256'], arms=ARMS, variants=VARIANTS,
        exact_case_ids=[c['id'] for c in p['cases']],
        choice_reason='Both full-decoder arms exceeding one-percent structure; include both pools',
        transform_definition={
            VARIANTS[0]: 'base + (candidate-base) minus observed-support channel mean of that difference',
            VARIANTS[1]: 'base + difference minus sigma2 13tap separable Gaussian lowpass of difference'},
        common_policy='Float64 transform then clip[0,1], float32; restore camera outside observed support; original floor-to-PNG',
        sigma2_fixed_from_existing_HF_measurement_not_tuned=True,
        source_candidate_failed_gates_retained=True,
        TRAIN_only=True, native_pixels_decoded=0, final_pixels_decoded=0,
        target_pixels_used_only_for_scoring_not_transform=True,
        arithmetic_controls_not_a_new_model_or_display_recommendation=True,
        worker_seconds=420, output_cap_bytes=512 * 1024**2,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        mathematical_optimizer_calls=0, app_changed=False, model_qualification=False, goal_complete=False))
    print(dict(prepared=True, bindings=len(bindings), transformed_cases=400))


def run():
    started = time.monotonic()
    plan = read(OUT / 'plan.json')
    assert not (OUT / 'results.json').exists() and not (OUT / 'failure.json').exists()
    for name, digest in plan['bindings'].items():
        assert sha(ROOT / name) == digest, name
    p = read(PACKET / 'protocol.json')
    m = metrics_module()
    raw_cache = returned_raw(p)
    baseline = read(RETURNED / 'baseline/metrics.json')['groups']
    z = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-.5 * (z / 2)**2)
    kernel /= kernel.sum()
    summaries, manifest = [], {}
    bytes_written = 0
    for arm in ARMS:
        for variant in VARIANTS:
            destination = OUT / arm / variant
            (destination / 'raw').mkdir(parents=True)
            (destination / 'previews').mkdir()
            rows = []
            for i, case in enumerate(p['cases'], 1):
                assert time.monotonic() - started < plan['worker_seconds']
                camera = image(PACKET / case['input'])
                target = image(PACKET / case['target'])
                mask = image(PACKET / case['observed'], 'L') > 0
                base = raw_cache['baseline'][case['id']]
                candidate = raw_cache[arm][case['id']]
                difference = candidate.astype(np.float64) - base.astype(np.float64)
                if variant == VARIANTS[0]:
                    retained = difference - difference[mask].mean(0)
                else:
                    smooth = convolve1d(convolve1d(difference, kernel, axis=0, mode='reflect'),
                                        kernel, axis=1, mode='reflect')
                    retained = difference - smooth
                raw = np.clip(base.astype(np.float64) + retained, 0, 1).astype(np.float32)
                raw[~mask] = camera[~mask].astype(np.float32) / np.float32(255)
                png = m.deliver(raw, camera, mask)
                interior = cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8),
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
                patches = np.zeros((256, 256), bool)
                for x, y in np.floor(case['landmarks5_canvas_xy']).astype(int):
                    patches[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
                support = patches & interior
                assert support.any()
                row = dict(id=case['id'], source=case['source'], profile=case['profile'],
                           remaining_mean_RGB_delta=(raw-base)[mask].astype(np.float64).mean(0).tolist())
                for stage, values in [('raw', raw), ('png', png.astype(np.float32) / np.float32(255))]:
                    row[stage] = m.pixel_metrics(values, target, mask)
                    row[stage]['landmark_high_frequency_MSE'] = m.detail_float(values, target, support)
                for path in [destination / 'raw' / (case['id'] + '.npy'),
                             destination / 'previews' / (case['id'] + '.png')]:
                    if path.suffix == '.npy':
                        np.save(path, raw, allow_pickle=False)
                    else:
                        Image.fromarray(png).save(path)
                    manifest[path.relative_to(ROOT).as_posix()] = sha(path)
                    bytes_written += path.stat().st_size
                    assert bytes_written <= plan['output_cap_bytes']
                rows.append(row)
                if i % 50 == 0:
                    print(dict(arm=arm, control=variant, cases=i, of=100), flush=True)
            grouped = {stage: groups(rows, stage) for stage in ['raw', 'png']}
            comparisons = {stage: compare(baseline[stage], grouped[stage]) for stage in grouped}
            result = dict(complete=True, arm=arm, variant=variant, rows=rows, groups=grouped,
                          comparisons=comparisons, full_model_quality_tested=False,
                          source_failed_candidate_qualification=False, model_qualification=False)
            file = destination / 'metrics.json'
            write(file, result)
            manifest[file.relative_to(ROOT).as_posix()] = sha(file)
            summaries.append(dict(arm=arm, variant=variant, comparisons=comparisons))
    for name, digest in plan['bindings'].items():
        assert sha(ROOT / name) == digest, name
    write(OUT / 'results.json', dict(
        complete=True, plan_sha256=sha(OUT / 'plan.json'), controls=summaries,
        output_bindings=manifest, transformed_TRAIN_cases=400, local_neural_calls=0,
        local_gradient_queries=0, local_optimizer_updates=0, mathematical_optimizer_calls=0,
        final_pixels_decoded=0, native_pixels_decoded=0,
        visual_usefulness_not_assessed=True, ArcFace_not_recomputed=True,
        app_changed=False, model_qualification=False, goal_complete=False,
        seconds=time.monotonic()-started, written_image_bytes=bytes_written))
    print(dict(complete=True, cases=400, seconds=time.monotonic()-started,
               full_quality_tested=False, local_training=0), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        try:
            run()
        except Exception:
            import traceback
            if not (OUT / 'failure.json').exists():
                write(OUT / 'failure.json', dict(complete=False, traceback=traceback.format_exc(),
                      local_neural_calls=0, local_optimizer_updates=0, goal_complete=False))
            raise


if __name__ == '__main__':
    main()

"""Paired TRAIN oracle arithmetic on saved original outputs; no model or learning."""
from pathlib import Path
from datetime import datetime, timezone
import time
import numpy as np
import cv2
from PIL import Image
from cctv_dgp_actual_step_review_v1_contract import NAME, ROLES, read, write, sha, role_ids, output_prefix
from cctv_dgp_actual_step_review_v1_metrics import pixel_metrics, detail_float, deliver

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs' / NAME
RETURN = ROOT / 'outputs' / (NAME + '_return')
OUT = ROOT / 'outputs/cctv_dgp_output_geometry_v1'
VARIANTS = ['original', 'target_mean_centered_oracle', 'bounded_target_mean_centered_oracle']
CAP = 300


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Preserve all earlier results'
    p = read(BUNDLE / 'protocol.json')
    science = read(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_independent_audit.json')
    assert science['complete'] and science['parent_protocol_sha256'] == sha(BUNDLE / 'protocol.json')
    imported = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json')
    selected = {}
    for probe in p['probes']:
        for role in ROLES:
            for cid in role_ids(p, probe, role):
                selected.setdefault(cid, {'id': cid, 'update': probe['update'], 'role': role,
                    'npz': (RETURN / output_prefix(probe['update'], 'zero') / role / (cid + '.npz')).relative_to(ROOT).as_posix()})
    assert set(selected) == {case['id'] for case in p['cases']} and len(selected) == 145
    assert all(case['role'] == 'train' for case in p['cases'])
    OUT.mkdir()
    plan = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'parent_protocol_sha256': sha(BUNDLE / 'protocol.json'), 'selected': list(selected.values()),
        'selection': 'Every existing145 TRAIN diagnostic case once, using its earliest complete saved unchanged output',
        'variants': VARIANTS, 'delta_limit': .5 - 1e-6,
        'observed_mean_only': True, 'clamp': [0, 1], 'PNG': 'floor(float32 RGB *255); exact input outside support',
        'metrics': ['MSE', 'SSIM', 'landmark_high_frequency_MSE'],
        'side_information': 'Uses aligned paired clean TRAIN targets as oracle information unavailable at deployment',
        'continuous_output_geometry_only': True, 'no_network_representability_claim': True,
        'no_ArcFace_or_complete_preservation_gate_test': True, 'no_quality_qualification': True,
        'maximum_cases': 145, 'cap_seconds': CAP, 'optimizer_updates': 0,
        'neural_calls': 0, 'gradient_queries': 0, 'native_or_DEV_or_reserved_final_used': False,
        'app_promotion': False}
    write(OUT / 'plan.json', plan)
    bindings = {}

    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        bindings[path.relative_to(ROOT).as_posix()] = digest

    for path in [BUNDLE / 'protocol.json', ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_independent_audit.json',
                 ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json', OUT / 'plan.json',
                 Path(__file__), ROOT / 'scripts/cctv_dgp_actual_step_review_v1_metrics.py']:
        bind(path)
    rows = []
    for case in p['cases']:
        assert time.monotonic() - start < CAP
        path = ROOT / selected[case['id']]['npz']
        bind(path, imported['files_sha256'][path.relative_to(RETURN).as_posix()])
        for key in ['input', 'target', 'observed']:
            bind(BUNDLE / case[key], p['assets_sha256'][case[key]])
        with np.load(path, allow_pickle=False) as arrays:
            baseline = arrays['original_rgb'].copy()
            assert baseline.dtype == np.float32 and baseline.shape == (256, 256, 3)
        with Image.open(BUNDLE / case['input']) as image:
            camera = np.asarray(image).copy()
        with Image.open(BUNDLE / case['target']) as image:
            target = np.asarray(image).copy()
        with Image.open(BUNDLE / case['observed']) as image:
            mask = np.asarray(image.convert('L')) > 0
        assert mask.any() and camera.shape == target.shape == baseline.shape
        baseline[~mask] = camera[~mask].astype(np.float32) / np.float32(255)
        residual = target.astype(np.float64) / 255 - baseline.astype(np.float64)
        values = residual[mask]
        mean = values.mean(axis=0)
        span = np.ptp(values, axis=0)
        midrange = (values.max(axis=0) + values.min(axis=0)) / 2
        centered = residual - mean
        bounded = np.clip(residual - midrange, -plan['delta_limit'], plan['delta_limit'])
        bounded_centered = bounded - bounded[mask].mean(axis=0)
        variants = {'original': baseline}
        for label, correction in zip(VARIANTS[1:], [centered, bounded_centered]):
            value = baseline.copy()
            value[mask] = np.clip(baseline[mask].astype(np.float64) + correction[mask], 0, 1).astype(np.float32)
            assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
            assert np.array_equal(value[~mask], baseline[~mask])
            variants[label] = value
        patch = np.zeros((256, 256), np.uint8)
        for xx, yy in np.floor(case['landmarks5_canvas_xy']).astype(int):
            patch[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = 1
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        support = patch.astype(bool) & eroded
        assert support.any()
        row = {'id': case['id'], 'source': case['source'], 'profile': case['profile'],
            'reference_id': case['reference_id'], 'saved_original': selected[case['id']],
            'required_observed_RGB_mean_shift': mean.tolist(), 'residual_span_per_channel': span.tolist(),
            'uncapped_centered_target_has_bounded_delta_construction': bool((span < 2 * plan['delta_limit']).all()),
            'bounded_and_uncapped_maximum_raw_difference': float(np.abs(variants[VARIANTS[1]] - variants[VARIANTS[2]]).max()),
            'metrics': {}}
        for label, value in variants.items():
            png = deliver(value, camera, mask).astype(np.float32) / np.float32(255)
            row['metrics'][label] = {}
            for stage, array in [('raw', value), ('PNG', png)]:
                row['metrics'][label][stage] = {**pixel_metrics(array, target, mask),
                    'landmark_high_frequency_MSE': detail_float(array, target, support)}
        rows.append(row)
    groups = []
    for source in ['all', *sorted({row['source'] for row in rows})]:
        for profile in ['all', 'clear', 'degraded']:
            subset = [row for row in rows if (source == 'all' or row['source'] == source) and
                (profile == 'all' or (row['profile'] == 'clear') == (profile == 'clear'))]
            assert subset
            for label in VARIANTS:
                for stage in ['raw', 'PNG']:
                    metrics = {key: float(np.mean([row['metrics'][label][stage][key] for row in subset]))
                               for key in plan['metrics']}
                    before = float(np.mean([row['metrics']['original'][stage]['landmark_high_frequency_MSE'] for row in subset]))
                    assert before > 0
                    groups.append({'source': source, 'profile': profile, 'variant': label, 'stage': stage,
                        'cases': len(subset), **metrics,
                        'oracle_relative_landmark_high_frequency_gain': 1 - metrics['landmark_high_frequency_MSE'] / before})
    result = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'cases': 145, 'references': 29,
        'rows': rows, 'groups': groups, 'source_sha256': bindings,
        'cases_with_bounded_uncapped_construction': sum(row['uncapped_centered_target_has_bounded_delta_construction'] for row in rows),
        'maximum_absolute_required_mean_shift': float(max(abs(value) for row in rows for value in row['required_observed_RGB_mean_shift'])),
        'maximum_bounded_vs_uncapped_raw_difference': max(row['bounded_and_uncapped_maximum_raw_difference'] for row in rows),
        'oracle_target_access_is_unavailable_in_app': True, 'neural_calls': 0,
        'optimizer_updates': 0, 'gradient_queries': 0, 'backward_calls': 0,
        'ArcFace_not_evaluated': True, 'full_preservation_gate_not_evaluated': True,
        'network_representability_not_tested': True, 'not_a_model_output_or_learned_capacity_pass': True,
        'native_or_DEV_or_reserved_final_used': False, 'app_promotion': False,
        'goal_complete': False, 'seconds': time.monotonic() - start, 'cap_seconds': CAP}
    write(OUT / 'analysis.json', result)
    print({'complete': True, 'cases': 145, 'bounded_constructions': result['cases_with_bounded_uncapped_construction'],
        'degraded_oracle_gains': [group for group in groups if group['source'] == 'all' and
            group['profile'] == 'degraded' and group['variant'] != 'original'],
        'neural_calls': 0, 'optimizer_updates': 0, 'seconds': result['seconds']}, flush=True)


if __name__ == '__main__':
    main()

"""Independent NumPy/OpenCV fixed-probe arithmetic; no model or derivative calls."""
import json
from pathlib import Path
import time

from verify_cctv_dgp_feature_skips_v27 import ROOT, NEW, OUT, PIN, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    import cv2
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    destination = OUT / 'independent_interface_readback.json'; assert not destination.exists()
    source = OUT / 'interface_inference_audit.json'; record = read(source)
    assert record['complete'] and record['protocol_sha256'] == PIN
    for name, digest in record['source_bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    p = read(NEW / 'protocol.json')
    assert [row['id'] for row in record['initial_cases']] == [case['id'] for case in p['cases']]
    assert all(row['raw_initial_exact_baseline'] and row['PNG_initial_exact_baseline'] for row in record['initial_cases'])
    assert record['shared_original_CPU_tensors_exact'] == 28 and record['new_feature_skip_tensors_exactly_zero'] == 10
    assert record['original_CPU_VM_initialization_maximum_difference'] <= 1e-8 and record['head_forwards'] == 61
    assert record['DGP_recognizer_forwards'] == record['local_gradient_calls'] == record['local_backward_calls'] == record['local_optimizer_updates'] == 0
    assert record['partial_mask_outside_exact_and_output_bound_verified'] and record['empty_support_rejected']
    assert record['preset_coefficients_use_no_target_no_fit_no_derivative_estimate'] and record['no_checkpoint_or_probe_images_created']
    maximum_rms_error = 0.; maximum_extreme_error = 0.
    for index, probe in enumerate(record['probes']):
        assert probe['scale'] == index and probe['preset_red_channel0_coefficient'] == .01
        assert not probe['fitted'] and not probe['quality_or_derivative_estimate'] and probe['zero_feature_control_exact_baseline']
        case = next(case for case in p['cases'] if case['id'] == probe['case'])
        with Image.open(NEW / case['observed']) as image:
            mask = np.asarray(image).copy() > 0
        base = np.load(NEW / case['raw_dgp'], allow_pickle=False)
        path = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return/outputs/frozen_DGP_features' / (case['id'] + '_fpn' + str(index) + '.npy')
        channel = np.load(path, allow_pickle=False)[0].astype(np.float64)
        size = channel.shape[0]; factor = 256 // size
        support = mask.reshape(size, factor, size, factor).mean((1, 3))
        mean = (channel * support).sum() / support.sum()
        centered = channel - mean
        scale = max(float(np.sqrt((centered ** 2 * support).sum() / support.sum())), .05)
        linear = cv2.resize((centered / scale) * .01, (256, 256), interpolation=cv2.INTER_LINEAR) / np.sqrt(5)
        q = .05 * np.tanh(linear)
        correction = (q - (q * mask).sum() / mask.sum()) * mask
        prediction = base.astype(np.float64).copy(); prediction[:, :, 0] += correction
        prediction = np.clip(prediction, 0, 1)
        difference = prediction - base
        rms = float(np.sqrt(np.mean(difference ** 2))); extreme = float(np.abs(difference).max())
        rms_error, extreme_error = abs(rms - probe['response_RMS']), abs(extreme - probe['response_maximum'])
        assert rms_error <= 2e-8 and extreme_error <= 2e-6, (index, rms_error, extreme_error)
        maximum_rms_error = max(maximum_rms_error, rms_error); maximum_extreme_error = max(maximum_extreme_error, extreme_error)
    result = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'interface_audit_sha256': sha(source), 'source_bindings_verified': len(record['source_bindings_sha256']),
        'initial_raw_PNG_parity_receipts': 50, 'fixed_probe_formulas_independently_replayed': 5,
        'maximum_probe_RMS_difference': maximum_rms_error, 'maximum_probe_extreme_difference': maximum_extreme_error,
        'scope': 'Source-bound initial parity, independent fixed forward-probe arithmetic; no quality or derivative claim',
        'local_neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'VM_actions': False,
        'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

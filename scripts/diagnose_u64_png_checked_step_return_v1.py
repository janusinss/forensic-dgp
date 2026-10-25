"""Explain one retained PNG endpoint from saved arrays; no model or new trial."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_audit'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_png_checked_step_v1_vm'
OUTPUT = 'outputs/u64_png_checked_step_v1/'


def main():
    inventory = json.loads((AUDIT / 'inventory_v1.json').read_bytes())
    evidence = Path(inventory['evidence_root'])
    protocol = json.loads((STAGE / 'protocol.json').read_bytes())
    consumed = {}
    def pin(path):
        data = path.read_bytes()
        return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    def checked(base, binding):
        path = base / binding['path']; actual = pin(path)
        assert {k: actual[k] for k in ('bytes', 'sha256')} == {k: binding[k] for k in ('bytes', 'sha256')}
        consumed[actual['path']] = actual
        return path
    def document(name):
        return json.loads(checked(evidence, inventory['files'][OUTPUT + name]).read_bytes())
    original, selected = document('banks/initial_quality.json'), document('banks/selected_quality.json')
    def png(base, binding):
        with Image.open(checked(base, binding)) as image:
            assert image.mode == 'RGB' and image.size == (256, 256)
            return np.asarray(image).copy()
    def metric(value, target):
        residual = value.astype(np.float64) - target.astype(np.float64)
        h, v = np.abs(np.diff(residual, axis=1)), np.abs(np.diff(residual, axis=0))
        return float((h.sum() + v.sum()) / (h.size + v.size))
    cases = []
    for row in protocol['selection']['monitors']:
        if row['role'] != 'local_gaussian_1p2':
            continue
        target = png(PARENT, row['target'])[64:192, 64:192]
        originals, codes, scalar = [], [], []
        for bank in (original, selected):
            record = bank['records'][row['id']]
            binding = record['outputs']['raw']
            raw = np.load(checked(evidence, binding), allow_pickle=False)
            q = png(evidence, record['outputs']['png'])
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
            assert np.array_equal(q, (np.clip(raw, 0, 1) * np.float32(255)).astype(np.uint8))
            roi = raw[64:192, 64:192]; code = q[64:192, 64:192]
            normalized_target = target.astype(np.float32) / np.float32(255)
            raw_scalar = metric(roi, normalized_target)
            png_scalar = metric(code.astype(np.float32) / np.float32(255), normalized_target)
            assert raw_scalar == record['regional_metrics']['raw']['central128']['adjacent_rgb_error']
            assert png_scalar == record['regional_metrics']['native_png']['central128']['adjacent_rgb_error']
            originals.append(roi); codes.append(code); scalar.append((raw_scalar, png_scalar))
        delta = codes[1].astype(np.int16) - codes[0].astype(np.int16)
        integer_change = []
        for axis in (0, 1):
            r0 = codes[0].astype(np.int32) - target.astype(np.int32)
            r1 = codes[1].astype(np.int32) - target.astype(np.int32)
            integer_change.append(np.abs(np.diff(r1, axis=axis)) - np.abs(np.diff(r0, axis=axis)))
        cases.append({'id': row['id'], 'original_RAW_adjacent': scalar[0][0], 'selected_RAW_adjacent': scalar[1][0],
            'original_PNG_adjacent': scalar[0][1], 'selected_PNG_adjacent': scalar[1][1],
            'RAW_delta': scalar[1][0] - scalar[0][0], 'PNG_delta': scalar[1][1] - scalar[0][1],
            'changed_RGB_components': int(np.count_nonzero(delta)), 'RGB_components': int(delta.size),
            'positive_code_crossings': int((delta > 0).sum()), 'negative_code_crossings': int((delta < 0).sum()),
            'max_absolute_code_delta': int(np.abs(delta).max()),
            'integer_residual_edge_increased_units': int(sum(x[x > 0].sum() for x in integer_change)),
            'integer_residual_edge_decreased_units': int(sum(x[x < 0].sum() for x in integer_change)),
            'integer_residual_edge_net_units': int(sum(x.sum() for x in integer_change)),
            'edge_components': int(sum(x.size for x in integer_change))})
    assert len(cases) == 8
    cases.sort(key=lambda row: row['id'])
    def ordered_mean(values):
        total = 0.
        for value in values:
            total += value
        return total / len(values)
    p0 = ordered_mean([r['original_PNG_adjacent'] for r in cases])
    p1 = ordered_mean([r['selected_PNG_adjacent'] for r in cases])
    assert p0 == selected['original_means']['native_png']['local_gaussian_1p2']['adjacent_rgb_error']
    assert p1 == selected['current_means']['native_png']['local_gaussian_1p2']['adjacent_rgb_error']
    net = sum(r['integer_residual_edge_net_units'] for r in cases)
    denominator = 255 * sum(r['edge_components'] for r in cases)
    summary = {'original_PNG_adjacent': p0, 'selected_PNG_adjacent': p1, 'PNG_delta': p1-p0,
        'PNG_relative_error_increase_percent': (p1/p0-1)*100,
        'RAW_delta': ordered_mean([r['selected_RAW_adjacent'] for r in cases]) - ordered_mean([r['original_RAW_adjacent'] for r in cases]),
        'changed_RGB_components': sum(r['changed_RGB_components'] for r in cases),
        'RGB_components': sum(r['RGB_components'] for r in cases),
        'integer_residual_edge_increased_units': sum(r['integer_residual_edge_increased_units'] for r in cases),
        'integer_residual_edge_decreased_units': sum(r['integer_residual_edge_decreased_units'] for r in cases),
        'integer_residual_edge_net_units': net, 'integer_code_predicted_delta': net/denominator,
        'FP32_normalization_minus_integer_prediction': p1-p0-net/denominator,
        'monitor_cases_RAW_improved': sum(r['RAW_delta'] < 0 for r in cases),
        'monitor_cases_PNG_worsened': sum(r['PNG_delta'] > 0 for r in cases)}
    assert all(pin(ROOT / row['path']) == row for row in consumed.values())
    result = {'complete': True, 'saved_endpoint_diagnosis_pass': True, 'summary': summary, 'cases': cases,
        'consumed_before_after_exact': list(consumed.values()), 'source': pin(Path(__file__).resolve()),
        'quality_threshold_or_quantizer_changed': False, 'monitor_fallback_authorized': False,
        'actual_new_model_gradient_parameter_VM_calls': 0, 'model_qualified': False}
    with (AUDIT / 'endpoint_diagnosis_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(summary)


if __name__ == '__main__':
    main()

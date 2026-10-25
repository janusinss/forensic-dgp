"""Append a descriptive interpretation; retain the original measured receipt unchanged."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_matched_feature_learning_v1_return_audit_v1'
EXPECTED = '5e460063eb7a80945e0c325f796c0ddb816957ec2bd73c80be28e41869f9a893'


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    source = OUT / 'photometric_diagnosis_v1.json'
    assert pin(source)['sha256'] == EXPECTED, 'Measured receipt changed'
    data = json.loads(source.read_text(encoding='utf-8'))
    assert data['complete'] and data['photographs'] == 8
    assert data['all_consumed_before_after_exact']
    rows = []
    for row in data['case_rows']:
        arms = {}
        for arm, value in row['arms'].items():
            raw = value['raw']
            integer = value['floor_PNG_integer_error_decomposition']
            arms[arm] = {
                'RAW_signed_RGB_residual_initial': raw['signed_RGB_mean_residual_initial'],
                'RAW_signed_RGB_residual_current': raw['signed_RGB_mean_residual_current'],
                'RAW_signed_RGB_mean_output_shift': raw['signed_RGB_mean_output_shift'],
                'constant_RGB_share_of_squared_output_delta_percent':
                    100 * raw['constant_mean_fraction_of_output_delta_energy'],
                'RAW_MAE_error_attribution': raw['raw_MAE_attribution'],
                'coarse_RGB_error_attribution': raw['coarse_RGB_error_attribution'],
                'coarse_blocks_worse': raw['coarse_error_blocks_worse'],
                'coarse_blocks_better': raw['coarse_error_blocks_better'],
                'PNG_MAE_delta': value['PNG_MAE_delta'],
                'integer_PNG_error_decomposition': integer,
                'quantization_effect_on_error_delta':
                    value['normalization_quantization_change_of_error_delta'],
            }
        rows.append({'reference_id': row['reference_id'], 'arms': arms})
    report = {
        'complete': True,
        'status': 'DESCRIPTIVE_INTERPRETATION_ONLY_GATES_AND_IMAGES_UNCHANGED',
        'source': pin(Path(__file__)), 'measured_receipt': pin(source),
        'visualization': data['visualization'],
        'interpretation_revision': 'Replaces only the original conditional spatial-dominance hypothesis; all measured values and original receipt remain unchanged.',
        'aggregate': data['aggregate'], 'photographs': rows,
        'grounded_findings': [
            'B11039 has 82.8334% constant-channel mean output-delta energy. Its RAW MAE error increase is +0.000272839794, partitioned +0.000249562111 global and +0.000023277683 spatial. All RGB means move further below the target. This case has broad color drift.',
            'B08064 has only 4.3348% constant-channel output-delta energy. Nevertheless, its chosen symmetric RAW MAE attribution is +0.000039686942 global versus -0.000018472346 spatial, yielding +0.000021214597 total error. Spatially varying output does not establish spatially caused error.',
            'B08094 has 5.7917% constant-channel output-delta energy. Its RAW MAE attribution is -0.000001101598 global versus +0.000066205884 spatial, yielding +0.000065104286 total error. This error increase is dominated by the remaining spatial term under this decomposition.',
            'Across eight B clear photographs, the mean RAW MAE change is -0.000000359514; positive global attribution +0.000008439965 nearly cancels negative spatial attribution -0.000008799479. Exact floor-PNG target error increases by 1126 RGB code units. The PNG failure remains authoritative.',
        ],
        'interpretive_limits': [
            'Squared output-delta energy and target-error attribution measure different quantities. Neither identifies an architectural cause or a proven remedy.',
            'The global-versus-spatial error attribution uses a declared symmetric two-order decomposition and unclipped analytical counterfactual arrays; it is descriptive and not uniquely causal.',
            'RGB residuals use encoded sRGB-domain values; no calibrated lighting or physical luminance inference.',
            'Eight reused TRAIN photographs at update8 cannot establish native CCTV, independent final identity, or Zamboanga performance.',
        ],
        'next_testable_design_hypothesis': 'Clear-image changes have mixed broad color and spatial error. A prospective design comparison should isolate these two effects with per-photo RAW and exact delivered-PNG clear controls and retain the degraded-face structure endpoints. Test whether condition-sensitive feature updates preserve both channel means and local structure; do not assume a uniform brightness/offset correction will solve the problem. No additional training or correction is authorized or performed by this report.',
        'gate_status': data['gate_status_immutable'],
        'model_qualified': False, 'new_training_model_gradient_VM_calls': 0,
        'original_receipt_unchanged': pin(source)['sha256'] == EXPECTED,
    }
    target = OUT / 'photometric_diagnosis_interpretation_r1.json'
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    assert pin(source)['sha256'] == EXPECTED
    print(json.dumps({'complete': True, 'report': pin(target),
                      'source': pin(Path(__file__)), 'original_unchanged': True}))


if __name__ == '__main__':
    main()

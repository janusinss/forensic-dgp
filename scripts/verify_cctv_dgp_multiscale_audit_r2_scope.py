"""Known failing ratio reproduction plus strict non-ratio/decision regressions."""
import hashlib
import json
from pathlib import Path
import audit_cctv_dgp_multiscale_calibration_return_v1 as old
import audit_cctv_dgp_multiscale_calibration_return_v1_r2 as new

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_audit_r2_preparation'


def fails(function, *args):
    try:
        function(*args)
    except AssertionError:
        return True
    return False


def main():
    a, b = 0.9264908100902142, 0.9264908100876813
    assert fails(old.close, {'brightness_gain_fraction': a}, {'brightness_gain_fraction': b}, 2e-12)
    new.close_gate({'brightness_gain_fraction': a}, {'brightness_gain_fraction': b})
    assert fails(new.close_gate, {'MSE': 0.1}, {'MSE': 0.1+3e-12})
    assert fails(new.close_gate, {'SSIM': a}, {'SSIM': b})
    assert fails(new.close_gate, {'ArcFace_observed_fixed': a}, {'ArcFace_observed_fixed': b})
    assert fails(new.close_gate, {'relative_feature_gain': a}, {'relative_feature_gain': b})
    assert fails(new.close_gate, {'minimum_gain': .01}, {'minimum_gain': .001})
    assert fails(new.close_gate, {'pass': False}, {'pass': True})
    assert fails(new.close_gate, [{'group': 'clear', 'metric': 'MSE'}], [{'group': 'clear', 'metric': 'SSIM'}])
    assert fails(new.close_gate, {'brightness_gain_fraction': a}, {'brightness_gain_fraction': a+1e-9})
    from cctv_dgp_multiscale_calibration_contract_v1 import BUDGETS
    assert BUDGETS['maximum_optimizer_updates'] == 12 and BUDGETS['worker_seconds'] == 1800
    result = dict(complete=True, known_R1_failure_reproduced=True, only_ratio_roundoff_accepted=True,
                  strict_MSE_SSIM_ArcFace_structure_threshold_decisions_and_failure_names_verified=True,
                  regression_controls=9, budgets_unchanged=True, local_neural_calls=0,
                  local_gradient_queries=0, local_optimizer_updates=0)
    with (OUT / 'repair_scope_regressions.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(result)


if __name__ == '__main__':
    main()

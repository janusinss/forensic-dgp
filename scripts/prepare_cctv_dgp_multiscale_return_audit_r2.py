"""Preserve frozen R1; repair only diagnosed derived brightness division tolerance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_audit_r2_preparation'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir()
    diagnostic = json.loads((ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_gate_roundoff/diagnostic.json').read_text())
    assert diagnostic['complete'] and diagnostic['sampled_decisions_and_failure_names_unchanged']
    assert all(not row['saved_gate_differences'] and not row['group_differences'] for row in diagnostic['entries'])
    source = ROOT / 'scripts/audit_cctv_dgp_multiscale_calibration_return_v1.py'
    assert sha(source) == diagnostic['original_checker_sha256'] == 'a2e92da64ae224b1ec2f5b0d3bfaec0582e16e300c85f22b0d1c8f3124bbe7c9'
    original = source.read_text(encoding='utf-8')
    added = '''DERIVED_RATIO_ATOL = 5e-11
DERIVED_RATIO_DIFFERENCES = []


def close_gate(a, b, brightness=False):
    """Only division of near-cancelling pixel gains receives a ratio allowance."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            is_ratio = key == 'brightness_gain_fraction' or (
                key == 'candidate' and a.get('metric') == b.get('metric') == 'brightness_gain_fraction')
            close_gate(a[key], b[key], is_ratio)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close_gate(x, y, brightness)
    elif brightness:
        difference = abs(a-b)
        DERIVED_RATIO_DIFFERENCES.append(difference)
        assert np.isclose(a, b, rtol=0, atol=DERIVED_RATIO_ATOL), (a, b, 'derived brightness ratio')
    else:
        close(a, b, 2e-12)


'''
    marker = 'def audit(a):\n'
    assert original.count(marker) == 1
    revised = original.replace(marker, added+marker, 1)
    old = "            close(recalculated, gate['comparisons'], 2e-12)\n"
    new = '''            stored_base = read(out / 'baseline/metrics.json')['rows']
            stored_gates = {s: capacity(review_groups(stored_base, s), review_groups(receipt['rows'], s), .01) for s in ['raw', 'png']}
            close(stored_gates, gate['comparisons'], 0.)
            close_gate(recalculated, gate['comparisons'])
            for s in ['raw', 'png']:
                assert recalculated[s]['pass'] == stored_gates[s]['pass']
                names = lambda g: [(r['group'], r['metric']) for r in g['preservation_failures']]
                assert names(recalculated[s]) == names(stored_gates[s])
'''
    assert revised.count(old) == 1
    revised = revised.replace(old, new, 1)
    marker = "        'visual_review_pending': True, 'model_qualification': False, 'goal_complete': False,\n"
    addition = "        'original_checker_failure_preserved': True, 'stored_row_gates_exact': True,\n        'derived_brightness_ratio_atol': DERIVED_RATIO_ATOL,\n        'maximum_derived_brightness_ratio_difference': max(DERIVED_RATIO_DIFFERENCES),\n        'all_gate_decisions_and_failure_names_unchanged': True,\n"
    assert revised.count(marker) == 1
    revised = revised.replace(marker, addition+marker, 1)
    target = ROOT / 'scripts/audit_cctv_dgp_multiscale_calibration_return_v1_r2.py'
    assert not target.exists()
    target.write_text(revised, encoding='utf-8', newline='\n')
    driver_source = ROOT / 'scripts/run_cctv_dgp_multiscale_return_audit_v1.py'
    driver = driver_source.read_text(encoding='utf-8')
    driver = driver.replace("OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_return_audit'", "OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_return_audit_r2'", 1)
    driver = driver.replace("checker = ROOT / 'scripts/audit_cctv_dgp_multiscale_calibration_return_v1.py'", "checker = ROOT / 'scripts/audit_cctv_dgp_multiscale_calibration_return_v1_r2.py'", 1)
    driver = driver.replace("outputs/cctv_dgp_multiscale_calibration_v1_independent_audit.json", "outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json", 1)
    old_assert = "    assert sha(checker) == protected[checker.relative_to(ROOT).as_posix()]"
    new_assert = "    assert sha(ROOT / 'scripts/audit_cctv_dgp_multiscale_calibration_return_v1.py') == protected['scripts/audit_cctv_dgp_multiscale_calibration_return_v1.py']\n    assert sha(checker) == read(ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_audit_r2_preparation/plan.json')['r2_checker_sha256']"
    assert driver.count(old_assert) == 1
    driver = driver.replace(old_assert, new_assert, 1).replace('checker_unchanged=True', 'original_checker_preserved=True, derived_ratio_repair_only=True', 1)
    driver_target = ROOT / 'scripts/run_cctv_dgp_multiscale_return_audit_v1_r2.py'
    assert not driver_target.exists()
    driver_target.write_text(driver, encoding='utf-8', newline='\n')
    plan = dict(complete=True, original_checker_sha256=sha(source), r2_checker_sha256=sha(target),
                driver_sha256=sha(driver_target), diagnostic_sha256=sha(ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_gate_roundoff/diagnostic.json'),
                original_failure_sha256=diagnostic['original_failure_sha256'], derived_brightness_ratio_atol=5e-11,
                repair_scope='only derived brightness ratio/failure-row ratio; all other tolerances retained',
                exact_stored_row_gates_required=True, exact_boolean_decisions_and_failure_names_required=True,
                source_bytes_diagnostics_returned_states_and_scientific_thresholds_unchanged=True,
                no_VM_work_required=True, local_optimizer_updates=0, local_gradient_queries=0,
                model_qualification=False, goal_complete=False)
    with (OUT / 'plan.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(plan, indent=2)+'\n')
    assert sha(source) == diagnostic['original_checker_sha256']
    print(dict(complete=True, original_R1_preserved=True, repair_scope=plan['repair_scope']))


if __name__ == '__main__':
    main()

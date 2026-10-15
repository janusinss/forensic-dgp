"""Preserve the stopped prospective checker; create a failure-only supplement."""
from datetime import datetime, timezone
from pathlib import Path

from import_cctv_dgp_raw_conditioning_learning_v1_r1_return import ROOT, ANALYSIS, RETURN, sha, read, write


def main():
    original = ROOT / 'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_r1_return.py'
    target = ROOT / 'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_r1_forensic_return.py'
    assert not target.exists() and not (ANALYSIS / 'original_checker_stop.json').exists()
    source = original.read_text(encoding='utf-8')
    backup = ANALYSIS / 'prospective_checker_original.py'; backup.write_bytes(original.read_bytes())
    assert sha(backup) == sha(original)
    stopped = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(original),
        tool_chunk='8cac3e', exit_code=1, location='audit_gradient_round:79',
        assertion='np.testing.assert_allclose(v,expected,rtol=1e-4,atol=1e-8)',
        index=2485488, actual=-0.00012899858411401406, expected=-0.00012897558044642089,
        maximum_reported_absolute_difference=abs(-0.00012899858411401406 + 0.00012897558044642089),
        original_bound_not_relaxed=True, cause_not_established_as_CUDA_nondeterminism=True,
        original_checker_passed=False, complete_neural_derivative_reproducibility_not_established=True)
    write(ANALYSIS / 'original_checker_stop.json', stopped)
    replacements = {
        'from datetime import datetime,timezone': 'from datetime import datetime,timezone\nfrom cctv_dgp_cross_run_gradient_report_v1 import report as cross_run_report',
        '    vectors=[];norms=[];losses=[]': '    vectors=[];norms=[];losses=[];cross_runs=[]',
        '            np.testing.assert_allclose(v,expected,rtol=1e-4,atol=1e-8)\n            assert np.isclose(np.linalg.norm(v),np.linalg.norm(expected),rtol=1e-4,atol=1e-10)':
        "            cross_runs.append(dict(key=row['key'],actual_file_sha256=row['sha256'],prior_file_sha256=old['sha256'],**cross_run_report(v,expected)))",
        '        initial_RAW_conditioning_derivatives_match_prior_audited_manual_measurement=initial_round,':
        '        initial_RAW_conditioning_derivatives_match_prior_audited_manual_measurement=initial_round and all(v[\'original_parity_check_pass\'] for v in cross_runs),\n        cross_run_original_bound_results=cross_runs,',
        "    assert manifest['failure_present']==bool(failure)":
        "    assert manifest['failure_present']==bool(failure)\n    assert failure and not manifest['complete_pilot'] and manifest['accepted_updates']==0, 'This supplement can only audit a rejected zero-update pilot'",
        "ANALYSIS/'failure_audit.json'": "ANALYSIS/'failure_forensic_audit.json'",
        "ANALYSIS/('independent_audit.json' if run else 'failure_audit.json')": "ANALYSIS/'failure_forensic_audit.json'",
        "        underlying_scientific_metric_replay_bound_unchanged=1e-13,":
        "        original_prospective_checker_passed=False, original_cross_run_bound_not_relaxed=True,\n        original_checker_sha256=sha(ROOT/'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_r1_return.py'),\n        cross_run_derivative_parity_pass=all(r['initial_RAW_conditioning_derivatives_match_prior_audited_manual_measurement'] for r in gradient_audits),\n        original_checker_stop_sha256=sha(ANALYSIS/'original_checker_stop.json'),\n        supplement_scope='Verify already rejected finite outputs and saved arithmetic while retaining failed cross-run parity',\n        underlying_scientific_metric_replay_bound_unchanged=1e-13,",
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after, 1)
    with target.open('x', encoding='utf-8', newline='\n') as stream: stream.write(source)
    assert sha(original) == stopped['checker_sha256']
    write(ANALYSIS / 'forensic_checker_preparation.json', dict(complete=True,
        preparer_sha256=sha(__file__), original_checker_sha256=sha(original), original_backup_sha256=sha(backup),
        supplemental_checker_sha256=sha(target), difference_reporting_helper_sha256=sha(ROOT/'scripts/cctv_dgp_cross_run_gradient_report_v1.py'),
        exact_source_replacements=[dict(before=b,after=a) for b,a in replacements.items()],
        actual_image_quality_gates_and_metric_replay_bounds_unchanged=True,
        prospective_checker_not_reclassified_as_passed=True, neural_derivative_reproducibility_pending=True,
        changed_VM_protocol_model_or_gradient_vectors=False, model_qualified=False))
    print(dict(complete=True, supplemental_checker=str(target), original_checker_unchanged=True))


if __name__ == '__main__': main()

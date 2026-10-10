"""Preserve the stopped auditor; narrowly verify an ill-conditioned quotient.

Saved-row arithmetic stays subject to the original exact decisions and1e-10
allowance. Only recomputation of a diagnostic ratio with a floored denominator
may use an explicit floating-point bound, on an already failed MSE comparison.
"""
from pathlib import Path
import ast
from cctv_dgp_actual_step_review_v1_contract import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit'
OUT = BASE / 'recovery_r1'
CHECKS = ROOT / 'scripts/cctv_dgp_actual_step_review_v1_partial_checks.py'
WORKER = ROOT / 'scripts/audit_cctv_dgp_actual_step_review_v1_partial_return.py'
NEW_CHECKS = ROOT / 'scripts/cctv_dgp_actual_step_review_v1_partial_checks_r1.py'
NEW_WORKER = ROOT / 'scripts/audit_cctv_dgp_actual_step_review_v1_partial_return_r1.py'

HELPER = '''
roundoff_receipts=[]

def check_comparison_roundoff(actual,reported,zero_condition,grouped,receipt,tolerance):
    # Retained saved-row comparisons must still reproduce exactly under the
    # original allowance. There is no exception for categories or gate values.
    canonical_groups={role:{stage:review_groups(receipt['rows'][role],stage) for stage in ['raw','PNG']} for role in ROLES}
    canonical=None if zero_condition is None else {role:{stage:finite_comparison(zero_condition['groups'][role][stage],canonical_groups[role][stage]) for stage in ['raw','PNG']} for role in ROLES}
    derived_equal(canonical,reported,tolerance)
    if actual is None:
        derived_equal(actual,reported,tolerance);return
    for role in ROLES:
        for stage in ['raw','PNG']:
            a,s=actual[role][stage],reported[role][stage]
            derived_equal({k:v for k,v in a.items() if k!='mean_only_fraction'},
                          {k:v for k,v in s.items() if k!='mean_only_fraction'},tolerance)
            error=abs(a['mean_only_fraction']-s['mean_only_fraction'])
            if error<=tolerance:continue
            baseline=zero_condition['groups'][role][stage]['degraded']['MSE']
            after=grouped[role][stage]['degraded']
            saved_after=receipt['groups'][role][stage]['degraded']
            assert after['MSE']>baseline+1e-12 and saved_after['MSE']>baseline+1e-12
            assert max(baseline-after['MSE'],1e-12)==max(baseline-saved_after['MSE'],1e-12)==1e-12
            assert not a['finite_preservation_pass'] and not s['finite_preservation_pass']
            assert any(r['group']=='degraded' and r['metric']=='MSE' for r in a['preservation_failures'])
            assert any(r['group']=='degraded' and r['metric']=='MSE' for r in s['preservation_failures'])
            scale=max(abs(baseline),abs(after['constant_mean_shift_only_MSE']),abs(saved_after['constant_mean_shift_only_MSE']))
            bound=8*np.finfo(np.float64).eps*scale/1e-12+8*max(abs(np.spacing(a['mean_only_fraction'])),abs(np.spacing(s['mean_only_fraction'])))
            assert np.isfinite(bound) and error<=bound
            roundoff_receipts.append({'update':receipt['update'],'proposal':receipt['proposal'],'role':role,'stage':stage,
                'computed_ratio':a['mean_only_fraction'],'reported_ratio':s['mean_only_fraction'],'absolute_ratio_error':error,
                'roundoff_bound':float(bound),'denominator':1e-12,'group_mean_control_MSE_error':abs(after['constant_mean_shift_only_MSE']-saved_after['constant_mean_shift_only_MSE']),
                'original_stored_row_arithmetic_pass':True,'both_degraded_MSE_gates_failed':True,'all_categorical_decisions_exact':True})
'''


def main():
    failure = read(BASE / 'failure.json')
    diagnostic = read(BASE / 'derived_comparison_float32_diagnostic_r1.json')
    assert not failure['complete'] and failure['type'] == 'AssertionError'
    assert diagnostic['frozen_float32_square_then_float64_mean_definition_used']
    assert any(row['field'] == 'mean_only_fraction' for row in diagnostic['differences'])
    assert not OUT.exists() and not NEW_CHECKS.exists() and not NEW_WORKER.exists()
    changes = {'checks': [], 'worker': []}
    def adapt(source, label, before, after):
        assert source.count(before) == 1, before
        changes[label].append({'before': before, 'after': after})
        return source.replace(before, after, 1)
    checks = CHECKS.read_text(encoding='utf-8')
    checks = adapt(checks, 'checks', 'def scientific_partial(p,failure,start,completed):', HELPER + '\ndef scientific_partial_r1(p,failure,start,completed):')
    checks = adapt(checks, 'checks', "            derived_equal(comparison,receipt['comparison_to_same_before_state'],p['prospective_audit']['derived_group_and_comparison_absolute_error'])",
        "            check_comparison_roundoff(comparison,receipt['comparison_to_same_before_state'],zero_condition,grouped,receipt,p['prospective_audit']['derived_group_and_comparison_absolute_error'])")
    checks = adapt(checks, 'checks', "'comparison_to_same_before_state':comparison})", "'comparison_to_same_before_state':receipt['comparison_to_same_before_state']})")
    worker = WORKER.read_text(encoding='utf-8')
    worker = adapt(worker, 'worker', "OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit'", "OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit/recovery_r1'")
    worker = adapt(worker, 'worker', "scope_path = OUT / 'prospective_partial_scope.json'", "scope_path = OUT.parent / 'prospective_partial_scope.json'")
    worker = adapt(worker, 'worker', "    start = time.monotonic()\n", "    start = time.monotonic()\n    verify_recovery_sources()\n")
    worker = adapt(worker, 'worker', 'from cctv_dgp_actual_step_review_v1_partial_checks import scientific_partial, CPU_partial_replay', 'from cctv_dgp_actual_step_review_v1_partial_checks_r1 import scientific_partial_r1 as scientific_partial, CPU_partial_replay, roundoff_receipts')
    worker = adapt(worker, 'worker', "'original_tolerances_unchanged': True", "'raw_pixel_term_replay_and_categorical_tolerances_unchanged': True, 'derived_floored_ratio_recomputation_exception': roundoff_receipts, 'original_stored_row_ratio_tolerance_unchanged': True")
    worker = adapt(worker, 'worker', "'generated_checks_sha256': sha(generated_path)", "'generated_checks_sha256': sha(ROOT / 'scripts/cctv_dgp_actual_step_review_v1_partial_checks_r1.py'), 'original_partial_checks_sha256': sha(generated_path)")
    verifier = '''
def verify_recovery_sources():
    preparation=read(OUT/'preparation.json')
    for name,digest in preparation['source_bindings'].items():assert sha(ROOT/name)==digest,name
    for label,original_name,new_name in [('checks','scripts/cctv_dgp_actual_step_review_v1_partial_checks.py','scripts/cctv_dgp_actual_step_review_v1_partial_checks_r1.py'),('worker','scripts/audit_cctv_dgp_actual_step_review_v1_partial_return.py','scripts/audit_cctv_dgp_actual_step_review_v1_partial_return_r1.py')]:
        value=(ROOT/new_name).read_text(encoding='utf-8')
        if label=='worker':value=value.replace(preparation['inserted_verifier'],'',1)
        for change in reversed(preparation['source_changes'][label]):
            assert value.count(change['after'])==1
            value=value.replace(change['after'],change['before'],1)
        assert value==(ROOT/original_name).read_text(encoding='utf-8')

'''
    point = worker.index('\ndef main():')
    worker = worker[:point] + verifier + worker[point:]
    for label, original, altered in [('checks', CHECKS, checks), ('worker', WORKER, worker.replace(verifier, '', 1))]:
        for change in reversed(changes[label]):
            assert altered.count(change['after']) == 1
            altered = altered.replace(change['after'], change['before'], 1)
        assert altered == original.read_text(encoding='utf-8')
    ast.parse(checks, feature_version=(3, 10)); ast.parse(worker, feature_version=(3, 10))
    OUT.mkdir()
    NEW_CHECKS.write_text(checks, encoding='utf-8', newline='\n')
    NEW_WORKER.write_text(worker, encoding='utf-8', newline='\n')
    write(OUT / 'preparation.json', {
        'complete': True, 'original_failure_preserved': True,
        'source_changes': changes, 'inserted_verifier': verifier, 'inverse_source_verified': True,
        'exception_scope': 'Only independently recomputed mean_only_fraction with a floored1e-12 denominator and an already failed degraded MSE gate; saved-row1e-10 check and every category/gate stay exact',
        'bound': '8*float64_epsilon*maximum_MSE_scale/1e-12 +8*maximum_ratio_spacing',
        'source_bindings': {path.relative_to(ROOT).as_posix(): sha(path) for path in [CHECKS, WORKER, NEW_CHECKS, NEW_WORKER, Path(__file__), BASE / 'failure.json', BASE / 'derived_comparison_diagnostic.json', BASE / 'derived_comparison_float32_diagnostic_r1.json', BASE / 'prospective_partial_scope.json']},
        'model_forwards': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
        'new_trained_checkpoint': False, 'app_promotion': False,
    })
    print({'recovery_prepared': True, 'old_failure_retained': True, 'raw_and_gate_tolerances_unchanged': True, 'model_forwards': 0}, flush=True)


if __name__ == '__main__':
    main()

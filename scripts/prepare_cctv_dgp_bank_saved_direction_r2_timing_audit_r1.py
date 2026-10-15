"""Repair one checker clock assumption, preserving VM data and scientific gates."""
import ast
from datetime import datetime, timezone
import numpy as np

from import_cctv_dgp_bank_saved_direction_forward_v1_r2_return import (
    ROOT, OUT, RETURN, ANALYSIS, read, sha, write,
)


def main():
    receipt=ANALYSIS/'timing_audit_repair.json'
    original=ROOT/'scripts/audit_cctv_dgp_bank_saved_direction_forward_v1_r2_return.py'
    revised=ROOT/'scripts/audit_cctv_dgp_bank_saved_direction_forward_v1_r2_return_timing_r1.py'
    assert not receipt.exists() and not revised.exists() and not (ANALYSIS/'independent_audit.json').exists()
    prep=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_preparation.json')
    assert sha(original)==prep['prospective_return_checker_files_sha256'][original.relative_to(ROOT).as_posix()]
    out=RETURN/'outputs/saved_direction_forward_v1'
    run,timing,p=read(out/'run_results.json'),read(out/'timing_projection.json'),read(RETURN/'protocol.json')
    assert run['complete'] and len(run['records'])==80
    assert timing['measured_B_batches']==22 and timing['candidate_reference_batches']==2 and timing['safety_factor']==1.5
    assert timing['cap_seconds']==p['measurement_seconds']==900 and timing['projection_seconds']<900
    former=run['records'][1]['seconds']/22*100*1.5
    assert not np.isclose(timing['projection_seconds'],former,rtol=0,atol=1e-7)
    inferred=timing['projection_seconds']*22/(p['model_forward_cap']*timing['safety_factor'])
    assert run['records'][1]['seconds']<inferred<run['records'][2]['seconds']
    worker=RETURN/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r2_vm.py'
    assert sha(worker)==p['assets_sha256']['scripts/'+worker.name]
    source=worker.read_text(encoding='utf-8')
    assert source.count("projection = (time.monotonic()-start)/counts['B']*p['model_forward_cap']*1.5")==1
    assert source.index('records.append(entry)')<source.index('if len(records) == 2:')
    assert source.index('if len(records) == 2:')<source.index("assert time.monotonic()-start < p['measurement_seconds']")
    before=original.read_text(encoding='utf-8')
    old="    assert np.isclose(projection['projection_seconds'],run['records'][1]['seconds']/22*100*1.5,rtol=0,atol=1e-7)"
    new="""    # Entry timestamp precedes printing and the separate projection clock read.
    # Its inferred clock must lie between the second and third successful outputs.
    assert projection['safety_factor'] == 1.5
    projection_clock = projection['projection_seconds']*22/(p['model_forward_cap']*projection['safety_factor'])
    assert run['records'][1]['seconds'] <= projection_clock <= run['records'][2]['seconds']"""
    assert before.count(old)==1
    after=before.replace(old,new)
    assert after.replace(new,old)==before
    anchor="        VM_seconds=run['seconds'],VM_peak_VRAM=run['peak_VRAM'],exact_control_case_replays=100,"
    assert after.count(anchor)==1
    after=after.replace(anchor,anchor+"\n        timing_audit_repair_sha256=sha(ANALYSIS/'timing_audit_repair.json'),\n        timing_projection_clock_within_second_and_third_output=True,\n        original_prepared_checker_sha256=sha(ROOT/'scripts/audit_cctv_dgp_bank_saved_direction_forward_v1_r2_return.py'),")
    tree_before,tree_after=ast.parse(before),ast.parse(after)
    for name in ['metrics','delivered','compare_groups','grouped','support','state_hash']:
        a=next(n for n in tree_before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in tree_after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    namespace={'projection':timing,'run':run,'p':p}
    exec(compile(new.strip().replace('\n    ','\n'),'<actual revised timing guard>','exec'),namespace)
    rejected=[]
    for label,clock in [('before_second_output',run['records'][1]['seconds']-1e-4),
                        ('after_third_output',run['records'][2]['seconds']+1e-4)]:
        changed=dict(timing,projection_seconds=clock*p['model_forward_cap']*1.5/22)
        try:
            exec(compile(new.strip().replace('\n    ','\n'),'<invalid timing fixture>','exec'),{'projection':changed,'run':run,'p':p})
        except AssertionError:
            rejected.append(dict(fixture=label,rejected=True,not_model_or_VM_results=True))
        else:
            raise AssertionError('Invalid clock ordering accepted')
    with revised.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(after)
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        original_checker_sha256=sha(original),revised_checker_sha256=sha(revised),
        revised_checker=revised.relative_to(ROOT).as_posix(),run_results_sha256=sha(out/'run_results.json'),
        projection_receipt_sha256=sha(out/'timing_projection.json'),sealed_VM_worker_sha256=sha(worker),
        initial_failure=dict(file=original.relative_to(ROOT).as_posix(),line=140,type='AssertionError',
            assertion=old.strip(),original_checker_preserved=True),
        earlier_output_seconds=run['records'][1]['seconds'],inferred_projection_clock_seconds=inferred,
        later_output_seconds=run['records'][2]['seconds'],separate_clock_delta_seconds=inferred-run['records'][1]['seconds'],
        actual_projection_seconds=timing['projection_seconds'],former_incorrectly_expected_projection_seconds=former,
        original_equality_failure_reproduced=True,actual_revised_guard_passed=True,invalid_clock_fixtures=rejected,
        metric_support_PNG_state_hash_and_scientific_gate_functions_AST_unchanged=True,
        no_timing_limits_or_quality_thresholds_changed=True,VM_return_and_worker_unchanged=True,
        no_local_neural_gradient_optimizer_or_parameter_displacement=True,no_VM_rerun_needed=True,
        full_quality_audit_still_pending=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,clock_read_difference_microseconds=(inferred-run['records'][1]['seconds'])*1e6,
        revised_checker=revised.name,VM_rerun_needed=False))


if __name__=='__main__':
    main()

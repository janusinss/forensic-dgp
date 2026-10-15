"""Retain the original checker; repair only exact derived-receipt comparison."""
import ast
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, ANALYSIS, read, sha, write


def main():
    original = ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_return.py'
    destination = ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_return_r1.py'
    stop = read(ANALYSIS / 'original_checker_stop.json')
    transfer = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_transfer_audit.json')
    assert sha(original) == stop['checker_sha256'] == transfer['future_auditor_sha256']
    red = read(ANALYSIS / 'receipt_check_red.json')
    green = read(ANALYSIS / 'receipt_check_green.json')
    assert not red['complete'] and green['complete']
    assert len(green['real_replayed_trials_verified']) == 3 and len(green['corruptions_rejected']) == 18
    helper = ROOT / 'scripts/conditioning_decision_receipt_check_v1.py'
    assert green['helper_sha256'] == sha(helper)
    text = original.read_text(encoding='utf-8')
    replacements = [
        ('from cctv_dgp_conditioning_learning_v1_r4_contract import decision,bounded_rate\n',
         'from cctv_dgp_conditioning_learning_v1_r4_contract import decision,bounded_rate\n'
         'from conditioning_decision_receipt_check_v1 import validate_receipt\n'),
        ("    accepted=initial.copy();current='initial';accepted_trials=[];trial_count=0\n",
         "    accepted=initial.copy();current='initial';accepted_trials=[];trial_count=0\n    receipt_checks=[]\n"),
        ("        assert checked==trial['decision']\n",
         "        verified=validate_receipt(checked,trial['decision'],(\n"
         "            snapshots['initial']['snapshot']['groups'],snapshots[current]['snapshot']['groups'],\n"
         "            snapshots[label]['snapshot']['groups']))\n"
         "        receipt_checks.append(dict(trial=label,**verified))\n"),
        ("        import_sha256=sha(ANALYSIS/'import.json'),full_pilot_returned=bool(run),failure=failure,\n",
         "        import_sha256=sha(ANALYSIS/'import.json'),full_pilot_returned=bool(run),failure=failure,\n"
         f"        original_checker_sha256='{sha(original)}',\n"
         "        original_checker_stop_sha256=sha(ANALYSIS/'original_checker_stop.json'),\n"
         "        receipt_helper_sha256=sha(ROOT/'scripts/conditioning_decision_receipt_check_v1.py'),\n"
         "        exact_saved_receipt_and_independent_gate_checks=receipt_checks,\n"
         "        underlying_scientific_metric_replay_bound_unchanged=1e-13,\n"),
    ]
    repaired = text
    for before, after in replacements:
        assert repaired.count(before) == 1
        repaired = repaired.replace(before, after)
    restored = repaired
    for before, after in reversed(replacements):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == text
    ast.parse(repaired, feature_version=(3, 10))
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(repaired)
    assert sha(original) == stop['checker_sha256']
    write(ANALYSIS / 'return_audit_r1_preparation.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__),
        original_checker_sha256=sha(original), new_checker_sha256=sha(destination),
        helper_sha256=sha(helper), red_test_sha256=sha(ANALYSIS / 'receipt_check_red.json'),
        green_test_sha256=sha(ANALYSIS / 'receipt_check_green.json'),
        inverse_source_diff_restores_original_exactly=True,
        only_derived_receipt_validation_and_provenance_changed=True,
        scientific_pixel_replay_bound_and_all_quality_gates_unchanged=True,
        original_VM_packet_code_models_outputs_and_checker_unchanged=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        model_qualified=False, goal_complete=False))
    print(dict(complete=True, original_checker_preserved=True,
               meaningful_corrupt_receipts_rejected=18, actual_model_quality_pass=False))


if __name__ == '__main__':
    main()

"""Future independent exposure/full-state audit, added loss history and fit time."""
import argparse
import json
from pathlib import Path

import audit_cctv_dgp_bank_split_execution_v1 as execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--import-receipt',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    a = parser.parse_args()
    assert not a.receipt.exists()
    common = a.receipt.with_name('execution_base_audit.json')
    execution.audit(a.root,a.import_receipt,common)
    result = execution.read(common)
    p = execution.read(a.root/'protocol.json')
    assert p['format'] == 'own256-conditioned-frozen-bank-photometric-training-guard-v1'
    assert p['start_from_unchanged_initial_B_not_failed_B50'] and not p['bank_feature_DC_removal']
    out = a.root/'outputs/bank_comparison_v1'
    timing = execution.read(out/'fit_seconds_B.json')
    assert timing['complete'] and timing['updates'] == 50 and timing['route'] == 'B'
    assert 0 < timing['seconds'] < timing['cap_seconds'] == p['budgets']['arm_fit_seconds']
    history = execution.read(out/'history_B.json')
    assert len(history['rows']) == 50 and all(row['photometric_guard'] >= 0 for row in history['rows'])
    result.update(checker_sha256=execution.sha(__file__),base_checker_sha256=execution.sha(execution.__file__),
        base_execution_audit_sha256=execution.sha(common),actual_fit_seconds_separately_recorded=True,
        actual_fit_seconds=timing['seconds'],fit_timing_sha256=execution.sha(out/'fit_seconds_B.json'),
        finite_nonnegative_guard_history_checked=True,guard_coefficient=p['photometric_guard']['coefficient'],
        single_new_TRAIN_loss_not_inference_processing=True)
    with a.receipt.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,updates=50,fit_seconds=timing['seconds'],local_optimizer_updates=0),flush=True)


if __name__ == '__main__': main()

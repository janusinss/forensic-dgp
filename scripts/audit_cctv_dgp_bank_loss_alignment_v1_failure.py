"""Audit the retained first-forward stop; never execute or import VM worker code."""
import ast
from datetime import datetime, timezone
import json

from import_cctv_dgp_bank_loss_alignment_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write


def main():
    receipt = ANALYSIS / 'independent_failure_audit.json'
    assert not receipt.exists()
    imported = read(ANALYSIS / 'import.json')
    assert imported['complete'] and imported['protocol_sha256'] == PIN
    assert sha(RETURN / 'export_manifest.json') == imported['export_manifest_sha256']
    manifest = read(RETURN / 'export_manifest.json')
    p = read(RETURN / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == PIN
    assert not manifest['complete_diagnostic'] and manifest['failure_present']
    assert {f.relative_to(RETURN).as_posix() for f in RETURN.rglob('*') if f.is_file()} == set(manifest['files']) | {'export_manifest.json'}
    for name, digest in manifest['files'].items():
        assert sha(RETURN / name) == digest
    for name, digest in p['parent_required_sha256'].items():
        assert sha(PARENT / name) == digest
    out = RETURN / 'outputs/loss_alignment_v1'
    assert {f.relative_to(out).as_posix() for f in out.rglob('*') if f.is_file()} == {'environment.json', 'failure.json'}
    failure = read(out / 'failure.json')
    environment = read(out / 'environment.json')
    assert failure['type'] == 'AssertionError' and failure['message'] == 'Same L4 frozen raw control differs'
    assert failure['progress']['stage'] == 'component_gradients'
    assert failure['progress']['state'] == 'initial' and failure['progress']['begin'] == 180
    counts = dict(B=1, recognizer=0, original=0, gradient_queries=0, backwards=0, optimizer_updates=0)
    assert failure['progress']['counts'] == counts and failure['optimizer_updates'] == 0
    assert 0 < failure['seconds'] < p['worker_seconds']
    assert environment['GPU'] == 'NVIDIA L4' and environment['host'] == 'forensic-dgp-thesis'
    assert environment['manual_tmux'] and not environment['optimizer_constructed']
    assert environment['failed_state_used_for_diagnosis_only']
    assert (RETURN / 'review_exit_code.txt').read_text().strip() == '1'
    log = (RETURN / 'review.log').read_text()
    assert failure['message'] in log and 'line 155' in log
    assert not list(RETURN.rglob('*.pth')) and not list(RETURN.rglob('*.npz'))
    source = (RETURN / 'scripts/cctv_dgp_bank_loss_alignment_v1_vm.py').read_text()
    tree = ast.parse(source)
    assert any(isinstance(n, ast.Assert) and ast.literal_eval(n.msg) == failure['message']
               for n in ast.walk(tree) if isinstance(n, ast.Assert) and isinstance(n.msg, ast.Constant))
    old = read(PARENT / 'protocol.json')
    case_ids = [c['id'] for c in old['cases'][180:185]]
    assert all(c in p['case_ids'] for c in case_ids)
    parent_parity = read(PARENT / 'outputs/bank_comparison_v1/preflight_complete.json')
    assert parent_parity['complete'] and parent_parity['initial_TRAIN_cases_each'] == 100
    assert parent_parity['initial_native_each'] == 24
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == value for name, value in protected.items())
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        import_sha256=sha(ANALYSIS / 'import.json'), protocol_sha256=PIN,
        failure_sha256=sha(out / 'failure.json'), environment_sha256=sha(out / 'environment.json'),
        namespace_files_checked=len(manifest['files']) + 1, parent_assets_checked=len(p['parent_required_sha256']),
        failure_type=failure['type'], failure_message=failure['message'], failure_location_line=155,
        state='initial', reference_batch_begin=180, case_ids=case_ids,
        source=old['cases'][180]['source'], VM_counts_recomputed=counts, VM_seconds=failure['seconds'],
        diagnostic_complete=False, packaging_complete=True, no_loss_measurements_or_gradient_samples=True,
        no_new_checkpoint=True, original_scientific_gates_unchanged=True,
        parent_initial_100_TRAIN_and_24_native_exact_parity_pass_retained=True,
        maximum_raw_error_not_recorded_by_failed_worker=True,
        exact_cause_not_proven_by_failure_receipt_alone=True,
        protected_bindings_unchanged=True, local_neural_gradient_or_optimizer_calls=0,
        agent_VM_connection_or_launch=False, model_qualified=False, goal_complete=False)
    write(receipt, result)
    print(json.dumps(dict(complete=True, diagnostic_complete=False, failure=failure['message'],
                         VM_counts=counts, seconds=failure['seconds'])))


if __name__ == '__main__':
    main()

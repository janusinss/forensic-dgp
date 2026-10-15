"""Retain original checkers and repair only the local return scope/failure audit."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    original = ROOT / 'scripts/import_cctv_dgp_bank_group_gradients_v1_return.py'
    original_audit = ROOT / 'scripts/audit_cctv_dgp_bank_group_gradients_v1_return.py'
    prepared = json.loads((OUT / 'cctv_dgp_bank_group_gradients_v1_preparation.json').read_text())
    transfer = json.loads((OUT / 'cctv_dgp_bank_group_gradients_v1_transfer_audit.json').read_text())
    assert sha(original) == prepared['importer_sha256']
    assert sha(original_audit) == transfer['future_auditor_sha256']
    dest = ROOT / 'scripts/import_cctv_dgp_bank_group_gradients_v1_return_r1.py'
    dest_audit = ROOT / 'scripts/audit_cctv_dgp_bank_group_gradients_v1_return_r1.py'
    receipt = OUT / 'cctv_dgp_bank_group_gradients_v1_local_audit_repair_r1.json'
    assert not any(p.exists() for p in [dest, dest_audit, receipt])
    text = original.read_text(encoding='utf-8')
    assert text.count('cctv_dgp_bank_saved_direction_group_gradients_v1') == 4
    assert text.count('cctv-dgp-bank-saved-direction-group-gradients-v1') == 1
    text = text.replace('cctv_dgp_bank_saved_direction_group_gradients_v1', 'cctv_dgp_bank_group_gradients_v1')
    text = text.replace('cctv-dgp-bank-saved-direction-group-gradients-v1', 'cctv-dgp-bank-group-gradients-v1')
    ast.parse(text)
    with dest.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    audit = original_audit.read_text(encoding='utf-8')
    assert audit.count('from import_cctv_dgp_bank_group_gradients_v1_return import ') == 1
    audit = audit.replace('from import_cctv_dgp_bank_group_gradients_v1_return import ',
                          'from import_cctv_dgp_bank_group_gradients_v1_return_r1 import ')
    begin = audit.index("    if not (out/'run_results.json').exists():")
    end = audit.index("    run = read(out/'run_results.json')", begin)
    failure_branch = '''    if not (out/'run_results.json').exists():
        assert not manifest['complete_diagnostic']
        if (out/'failure.json').exists():
            failure = read(out/'failure.json')
            assert manifest['failure_present']
            detail = dict(failure=failure, failure_sha256=sha(out/'failure.json'),
                          scope='structured retained diagnostic failure')
        else:
            # A storage assertion precedes output creation and the try/logger.
            # Bind the traceback to the exact sealed source, not a fabricated receipt.
            import ast
            assert not out.exists() and not manifest['failure_present']
            log = (RETURN/'review.log').read_text(encoding='utf-8')
            assert int((RETURN/'review_exit_code.txt').read_text().strip()) == 1
            assert log.endswith('AssertionError: Need4GiB free after installation\\n')
            assert 'line 89, in run' in log and 'line 305, in main' in log
            worker = RETURN/'scripts/cctv_dgp_bank_group_gradients_v1_vm.py'
            source = worker.read_text(encoding='utf-8')
            tree = ast.parse(source)
            fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
            guard = fn.body[2]
            assert isinstance(guard,ast.Assert) and guard.lineno == 89
            assert ast.unparse(guard.test) == "shutil.disk_usage(root).free >= p['minimum_free_bytes']"
            first_torch = next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
            assert guard.lineno < first_torch and p['minimum_free_bytes'] == 4294967296
            assert not any(n.startswith('outputs/') for n in manifest['files'])
            detail = dict(scope='pre-model storage preflight stop', traceback=log,
                review_log_sha256=sha(RETURN/'review.log'), worker_sha256=sha(worker),
                required_free_bytes=4294967296, actual_free_bytes_not_recorded_in_return=True,
                neural_gradient_optimizer_and_parameter_displacement_calls=0,
                no_new_checkpoint_or_image_outputs=True, no_failure_json_created=True,
                failure_flag_only_checks_failure_json_and_is_not_success=True)
        write(ANALYSIS/'failure_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
            checker_sha256=sha(__file__), import_sha256=sha(ANALYSIS/'import.json'), **detail,
            independently_checked_all_return_file_and_parent_hashes=True,
            partial_measurement_not_a_completed_gradient_audit=True,
            no_neural_reexecution_or_post_failure_RAM_state_hash_claimed=True,
            local_neural_or_optimizer_calls=0, no_training_qualification=True,
            model_qualified=False, goal_complete=False))
        print(dict(complete=True,retained_failure_audited=True,scope=detail['scope'],model_qualified=False))
        return
'''
    audit = audit[:begin] + failure_branch + audit[end:]
    ast.parse(audit)
    with dest_audit.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(audit)
    value = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        preparer_sha256=sha(__file__), original_importer_sha256=sha(original),
        original_auditor_sha256=sha(original_audit), repaired_importer_sha256=sha(dest),
        repaired_auditor_sha256=sha(dest_audit), original_sources_preserved=True,
        importer_change='four folder/receipt scope names and one archive stem',
        auditor_change='corrected importer plus hash-bound pre-initialization storage traceback branch',
        success_measurement_checks_preserved=True, protocol_and_VM_worker_unchanged=True,
        thresholds_unchanged=True, local_neural_gradient_optimizer_calls=0,
        model_qualified=False, goal_complete=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')
    print(dict(complete=True, original_checkers_retained=True, repaired_importer=dest.name,
               repaired_auditor=dest_audit.name))


if __name__ == '__main__':
    main()

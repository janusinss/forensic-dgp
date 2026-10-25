"""Publish the tested inert contract and finite call table; never train."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DESTINATION = ROOT / 'outputs/codeformer_matched_feature_learning_v1_source_preparation'
NEW_FILES = ('contract_v1.json', 'selection_v1.json', 'operation_plan_v1.json', 'contract_tests_v1.json')
SOURCES = (
    'scripts/prepare_codeformer_matched_contract_v1.py',
    'research/codeformer_matched_feature_learning_v1/contract.py',
    'research/codeformer_matched_feature_learning_v1/objective.py',
    'tests/test_matched_feature_contract_v1.py',
    'research/codeformer_u64_cleananchors_continuation_v1/contract.py',
    'research/codeformer_u64_cleananchors_continuation_v1/objective.py',
    'outputs/codeformer_matched_feature_learning_v1_preparation/data_admission_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_preparation/candidate_data_v1.json',
    'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm/evidence/u64_parent_protocol.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_contract_review_v1.json',
)


def pin(relative):
    path = ROOT / relative
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Consumed file changed during hashing')
    return {'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_new(name, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    path = DESTINATION / name
    with path.open('xb') as stream:
        stream.write(payload)
    if path.read_bytes() != payload:
        raise ValueError('Contract artifact readback differs')
    return pin(path.relative_to(ROOT).as_posix())


def main():
    if not DESTINATION.is_dir() or any((DESTINATION / name).exists() for name in NEW_FILES):
        raise ValueError('Existing output artifacts are immutable; fresh declared filenames required')
    before = {relative: pin(relative) for relative in SOURCES}
    # Importing this test module installs its explicit framework/worker blocker.
    from tests import test_matched_feature_contract_v1 as tests
    from research.codeformer_matched_feature_learning_v1 import contract
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful() or result.testsRun != 27 or tests.BLOCKER.attempted:
        raise ValueError('Expected27 framework-free contract tests must pass')
    preparation = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'
    selection = contract.select_cases((preparation / 'data_admission_v1.json').read_bytes(),
                                     (preparation / 'candidate_data_v1.json').read_bytes(),
                                     tests.PARENT.read_bytes())
    operations = contract.operation_plan(selection)
    for operation in operations:
        contract.resolve_case(selection, operation)
    review = json.loads((DESTINATION / 'independent_contract_review_v1.json').read_bytes())
    for relative, expected in review['reviewed_source_pins'].items():
        if relative in before and {'bytes': before[relative]['bytes'], 'sha256': before[relative]['sha256']} != expected:
            raise ValueError('Independent contract review does not bind current bytes')
    if {relative: pin(relative) for relative in SOURCES} != before:
        raise ValueError('Consumed source changed before publication')
    artifacts = {
        'contract': write_new('contract_v1.json', contract.prospective_contract()),
        'selection': write_new('selection_v1.json', selection),
        'operations': write_new('operation_plan_v1.json', {'maximum_plan': True, 'calls': operations,
            'actual_calls_executed': 0, 'scope_qualified_case_keys_required': True,
            'quality_stop_is_per_arm_prefix_and_global_ordered_subsequence': True}),
    }
    receipt = {
        'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'TESTED_SOURCE_CONTRACT_AND_MAXIMUM_SCHEDULE_ONLY_NOT_EXECUTION_READY',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'test_output': capture.getvalue(), 'blocked_import_attempts': tests.BLOCKER.attempted,
        'all1736_scope_qualified_call_rows_resolved': True,
        'consumed_before_after_exact': before == {relative: pin(relative) for relative in SOURCES},
        'consumed': before, 'artifacts': artifacts,
        'maximum_forward_calls': 1736, 'maximum_backward_calls': 480,
        'maximum_optimizer_constructions': 2, 'maximum_optimizer_updates': 96,
        'author_test_history': 'Initial25-test run had two fixture failures caused by strict binary-float expectations; corrected expected LR representation and isolated cumulative-bound fixture. No parameter bound or quality threshold was changed. Added2 bank-collision/resolution tests.',
        'new_neural_gradient_optimizer_training_VM_browser_calls': 0,
        'worker_archive_upload_guide_ready': False, 'model_qualified': False,
    }
    if not receipt['consumed_before_after_exact']:
        raise ValueError('Consumed source changed during publication')
    artifacts['receipt'] = write_new('contract_tests_v1.json', receipt)
    print(json.dumps({'complete': True, 'tests': result.testsRun, 'artifacts': artifacts,
                      'training_calls': 0, 'execution_ready': False}))


if __name__ == '__main__':
    main()

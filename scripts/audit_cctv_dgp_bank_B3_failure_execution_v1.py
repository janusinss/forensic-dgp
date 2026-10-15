"""Independently audit the Python-3.10 pre-baseline failure; no model calls."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_return'
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis'
PIN = 'f34b7b54cd322d26a1a2159cffa28aabb1258fcb484914fd781d70e3e7bc0c35'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    receipt = ANALYSIS / 'failure_execution_audit.json'
    assert not receipt.exists()
    imported = read(ANALYSIS / 'import.json')
    audited = read(ANALYSIS / 'independent_audit.json')
    assert imported['complete'] and audited['complete']
    assert imported['every_member_path_and_byte_hash_checked']
    assert imported['protocol_sha256'] == audited['protocol_sha256'] == PIN
    assert sha(RETURN / 'protocol.json') == PIN
    p = read(RETURN / 'protocol.json')
    manifest = read(RETURN / 'export_manifest.json')
    assert sha(RETURN / 'export_manifest.json') == imported['export_manifest_sha256']
    for name, expected in manifest['files'].items():
        assert sha(RETURN / name) == expected, name
    out = RETURN / 'outputs/bank_comparison_v1'
    assert {path.name for path in out.iterdir()} == {
        'environment.json', 'failure.json', 'initializer_load_B_0.json'}
    failure, env = read(out / 'failure.json'), read(out / 'environment.json')
    loading = read(out / 'initializer_load_B_0.json')
    message = "module 'hashlib' has no attribute 'file_digest'"
    assert failure['type'] == loading['failure_type'] == 'AttributeError'
    assert failure['message'] == loading['failure_message'] == message
    assert failure['completed_routes'] == [] and failure['original_checkpoints_and_failures_preserved']
    assert failure['progress']['stage'] == 'pinned_initializer_before_baseline'
    assert failure['progress']['route'] == 'B' and failure['progress']['route_updates'] == 0
    counts = failure['progress']['counts']
    assert set(counts) == {'original', 'A', 'B', 'recognizer', 'prior_fixture',
                           'gradient_queries', 'backwards', 'optimizer_updates'}
    assert all(value == 0 for value in counts.values())
    assert env['python'].startswith('3.10.12') and env['GPU'] == 'NVIDIA L4'
    assert env['packages']['torch'] == loading['torch'] == '2.9.1+cu129'
    assert loading['loaded_on'] == 'cuda' and not loading['learning_enabled']
    assert 'constructor_differences' not in loading and 'strict_load_complete' not in loading
    assert audited['saved_paired_raw_PNG_cases_checked'] == audited['native_compositions_checked'] == 0
    assert audited['optimizer_updates_in_VM'] == 0 and not audited['strict_full_state_checks']
    supervisor = read(RETURN / 'supervisor_receipt.json')
    assert supervisor['complete'] and supervisor['protocol_sha256'] == PIN
    assert supervisor['trainer_exit_code'] == int((RETURN / 'trainer_exit_code.txt').read_text()) == 1
    assert supervisor['optimizer_updates'] == 0
    assert 0 < failure['seconds'] <= supervisor['seconds'] < p['budgets']['external_seconds']
    loader_path = RETURN / 'cctv_dgp_bank_pinned_initializer_v1.py'
    tree = ast.parse(loader_path.read_text(encoding='utf-8'))
    function, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'sha']
    namespace = {'Path': Path, 'hashlib': types.SimpleNamespace(sha256=hashlib.sha256)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(loader_path), 'exec'), namespace)
    reproduced = False
    try:
        namespace['sha'](RETURN / 'initializers/initial_B.pth')
    except AttributeError as exc:
        assert 'file_digest' in str(exc)
        reproduced = True
    assert reproduced
    source = (RETURN / 'scripts/cctv_dgp_bank_comparison_v1_vm.py').read_text(encoding='utf-8')
    assert source.index("candidate = create('B', learning=False)") < source.index("baseline = metrics('baseline', original.net)")
    assert 'hashlib.file_digest' in failure['traceback']
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), protocol_sha256=PIN,
        import_receipt_sha256=sha(ANALYSIS / 'import.json'),
        independent_artifact_audit_sha256=sha(ANALYSIS / 'independent_audit.json'),
        failure_sha256=sha(out / 'failure.json'), counts=counts, actual_optimizer_updates=0,
        worker_seconds=failure['seconds'], supervisor_seconds=supervisor['seconds'],
        failure_type='AttributeError', failure_message=message,
        recorded_VM_python='3.10.12', recorded_VM_torch=env['packages']['torch'],
        missing_hash_API_reproduced=True, reproduction_is_API_absence_simulation=True,
        actual_local_Python310_interpreter_used=False, model_or_Torch_imported_for_reproduction=False,
        initializer_hash_or_state_load_completed=False,
        original_B2_constructor_difference_still_untested=True,
        saved_candidate_or_baseline_outputs=0, local_neural_gradient_or_training_calls=0,
        original_return_and_failure_unchanged=True, model_qualified=False,
        app_promotion=False, goal_complete=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(dict(complete=True, failure=message, optimizer_updates=0,
               saved_images=0, hash_API_failure_reproduced=True), flush=True)


if __name__ == '__main__':
    main()

"""Independent zero-update B failure/counter/source audit; no model calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_return'
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_analysis'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = ANALYSIS / 'failure_execution_audit.json'
    assert not receipt.exists()
    imported = read(ANALYSIS / 'import.json')
    assert imported['complete'] and imported['every_member_path_and_byte_hash_checked']
    protocol = read(RETURN / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == imported['protocol_sha256']
    manifest = read(RETURN / 'export_manifest.json')
    out = RETURN / 'outputs/bank_comparison_v1'
    failure = read(out / 'failure.json')
    env = read(out / 'environment.json')
    supervisor = read(RETURN / 'supervisor_receipt.json')
    assert failure['type'] == 'AssertionError' and failure['message'] == 'Fresh initializer differs'
    assert 'candidate = create(route, learning=True)' in failure['traceback']
    assert "state_hash(model) == state_hash(initial)" in failure['traceback']
    counts = failure['progress']['counts']
    assert counts == dict(original=805, A=0, B=0, recognizer=781, prior_fixture=0,
        gradient_queries=0, backwards=0, optimizer_updates=0)
    assert failure['progress']['stage'] == 'initial_GPU_parity_and_gradients'
    assert failure['progress']['route'] == 'B' and failure['progress']['route_updates'] == 0
    assert failure['completed_routes'] == []
    assert env['GPU'] == 'NVIDIA L4' and env['host'].split('.')[0] == 'forensic-dgp-thesis'
    assert env['packages']['torch'] == '2.9.1+cu129' and not env['AMP'] and not env['TF32']
    assert supervisor['complete'] and supervisor['protocol_sha256'] == imported['protocol_sha256']
    assert supervisor['trainer_exit_code'] == int((RETURN / 'trainer_exit_code.txt').read_text()) == 1
    assert supervisor['optimizer_updates'] == 0
    assert 0 < failure['seconds'] <= supervisor['seconds'] < protocol['budgets']['external_seconds']
    assert (out / 'baseline/metrics.json').is_file()
    assert not (out / 'results.json').exists() and not list(out.glob('state_*'))
    assert not any((out / name).exists() for name in ['B50', 'B50_bank_disabled',
        'preflight_complete.json', 'prior_CPU_CUDA_fixture.json', 'preflight_B.json', 'history_B.json'])
    assert not list(out.glob('quality_stop_*'))
    file = RETURN / 'scripts/cctv_dgp_bank_comparison_v1_vm.py'
    source = file.read_text(encoding='utf-8')
    assert source.index("assert state_hash(model) == state_hash(initial), 'Fresh initializer differs'") < source.index('model.load_state_dict(initial, strict=True)')
    for path in [out / 'failure.json', out / 'environment.json', RETURN / 'supervisor_receipt.json',
                 RETURN / 'trainer_exit_code.txt', file]:
        assert sha(path) == manifest['files'][path.relative_to(RETURN).as_posix()]
    baseline = read(out / 'baseline/metrics.json')
    assert baseline['complete'] and len(baseline['rows']) == 3905
    assert len(baseline['native_unpaired_rows']) == 24 and not baseline['native_quality_claim']
    stages = Counter(r['role'] for r in protocol['cases'])
    assert stages == {'train': 3905} and protocol['reserved_final_pixels'] == 0
    record = dict(complete=True, checker_sha256=sha(__file__), protocol_sha256=imported['protocol_sha256'],
        import_receipt_sha256=sha(ANALYSIS / 'import.json'), failure_sha256=sha(out / 'failure.json'),
        failure_type=failure['type'], failure_message=failure['message'], worker_seconds=failure['seconds'],
        counts=counts, actual_optimizer_updates=0, B_model_forwards=0, B_gradients=0,
        B_initializer_tensor_differences_not_saved=True, B_full_state_not_saved=True,
        B_quality_not_tested=True, A_B_training_exposures_not_matched=True,
        original_stop_and_protocol_unchanged=True, initial_load_occurs_after_failed_check=True,
        local_neural_gradient_or_optimizer_calls=0, baseline_output_audit_separate=True,
        model_qualified=False, app_promotion=False, goal_complete=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, optimizer_updates=0, cause=failure['message'], B_quality_tested=False))


if __name__ == '__main__':
    main()

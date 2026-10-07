"""Independent byte-encoding and complete receipt readback; no neural calls."""
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_decoder_app_normalization_v1'
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    target = OUT / 'independent_readback.json'
    assert not target.exists(), 'Preserve existing readback'
    plan_path, result_path = OUT / 'plan.json', OUT / 'results.json'
    plan, result = read(plan_path), read(result_path)
    assert plan['complete'] and plan['frozen_before_new_inference'] and result['complete']
    assert result['plan_sha256'] == sha(plan_path)
    assert plan['source_bindings_sha256'] == result['source_bindings_sha256']
    bindings = result['source_bindings_sha256']
    assert len(bindings) == 252
    for name, digest in bindings.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, name
    protocol = read(BUNDLE / 'protocol.json')
    for name, digest in protocol['assets_sha256'].items():
        assert bindings[(BUNDLE / name).relative_to(ROOT).as_posix()] == digest

    # Compile only the two fingerprinted conversion functions, never a model or
    # the runner. Compare all byte values with an independent rational oracle.
    import numpy as np
    import torch
    namespace = {'np': np, 'torch': torch}
    for path, name, key in [
        (ROOT / 'dgp_face_workflow_v3.py', 'canonical_tensor', 'app_input_function_AST_sha256'),
        (BUNDLE / 'dgp_face_restoration.py', 'as_tensor', 'legacy_input_function_AST_sha256')]:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert sha(path) == bindings[path.relative_to(ROOT).as_posix()]
        assert hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() == result[key]
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<input-only-readback>', 'exec'), namespace)
    probe = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (1, 256, 3)).copy()
    with torch.no_grad():
        app = namespace['canonical_tensor'](probe, 'cpu')
        legacy = namespace['as_tensor'](probe, 'cpu')
    oracle = np.array([np.float32(float(Fraction(i, 255))) for i in range(256)], dtype=np.float32)
    assert app.shape == legacy.shape == (1, 3, 1, 256)
    assert app.dtype == legacy.dtype == torch.float32
    for encoded in [app, legacy]:
        assert np.array_equal(encoded.numpy(), np.broadcast_to(oracle, (1, 3, 1, 256)))
    assert torch.equal(app, legacy)
    assert result['byte_values_with_CPU_encoding_difference'] == []
    assert result['maximum_CPU_byte_encoding_difference'] == 0.0

    previous_path = ROOT / 'outputs/cctv_dgp_original_decoder_review_v1/review.json'
    previous = read(previous_path)
    assert sha(previous_path) == bindings[previous_path.relative_to(ROOT).as_posix()]
    old_rows = previous['initial_cases']
    cases, rows = protocol['cases'], result['cases']
    assert len(cases) == len(rows) == len(old_rows) == 50
    assert plan['case_ids'] == [c['id'] for c in cases] == [r['id'] for r in rows] == [r['id'] for r in old_rows]
    for case, row, old in zip(cases, rows, old_rows):
        assert case['role'] == row['role'] == 'train'
        assert (case['source'], case['profile']) == (row['source'], row['profile'])
        for key in ['input_values_differ', 'maximum_input_difference', 'maximum_raw_output_difference',
                    'changed_delivered_RGB_components', 'changed_delivered_pixels', 'maximum_delivered_byte_difference']:
            assert row[key] == 0, (row['id'], key)
        assert row['actual_app_input_raw_candidate_parity_exact']
        assert row['actual_app_input_PNG_candidate_parity_exact'] and row['legacy_raw_and_PNG_receipt_replayed']
        assert row['actual_app_input_raw_sha256'] == old['initial_raw_rgb_sha256']
        assert row['actual_app_input_PNG_sha256'] == old['initial_PNG_rgb_sha256']
    assert result['neural_forward_counts'] == {'original': 100, 'candidate': 50}
    assert result['actual_app_input_exact_initial_candidate_parity_cases'] == result['historical_legacy_receipts_replayed'] == 50
    assert result['DGP_state_before_after'] == previous['DGP_state_before_after'] == 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
    assert 0 < result['seconds'] < plan['cap_seconds'] == 300
    assert plan['maximum_original_forwards'] == 100 and plan['maximum_candidate_forwards'] == 50
    runner = ast.parse((ROOT / 'scripts/review_cctv_dgp_decoder_app_normalization_v1.py').read_text(encoding='utf-8'))
    calls = [ast.unparse(node.func) for node in ast.walk(runner) if isinstance(node, ast.Call)]
    assert 'torch.no_grad' in calls
    assert not any(name.endswith(('backward', 'autograd.grad', 'step', 'save', 'enable_grad')) for name in calls)
    assert not any(isinstance(node, ast.Attribute) and node.attr in ['Adam', 'AdamW', 'SGD'] for node in ast.walk(runner))
    assert result['local_gradient_calls'] == result['local_backward_calls'] == result['local_optimizer_updates'] == 0
    for key in ['R2_packet_and_commands_changed', 'CUDA_difference_measured', 'canonical_full_app_flow_verified',
                'quality_gain_or_identity_claim', 'VM_actions', 'native_or_reserved_used', 'independent_final_review',
                'app_promotion', 'goal_complete']:
        assert result[key] is False, key
    assert not plan['training'] and not plan['VM_actions'] and not plan['native_or_reserved_used']
    assert 'NumPy float32 division before device transfer' in result['encoding_policy_clarification']
    assert 'torch division after transfer' in result['encoding_policy_clarification']
    assert 'distinct finite training protocol' in result['future_training_requirement']
    record = {'complete': True, 'checker_sha256': sha(Path(__file__)),
              'results_sha256': sha(result_path), 'plan_sha256': sha(plan_path),
              'source_bindings_verified': len(bindings), 'independent_rational_byte_values_verified': 256,
              'exact_CPU_byte_encoding_equivalence': True, 'initial_app_reference_receipts_verified': 50,
              'legacy_initial_receipts_replayed': 50, 'DGP_state_unchanged_receipt_verified': True,
              'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
              'released_R2_change': False, 'CUDA_equivalence_measured': False,
              'local_neural_or_gradient_calls': 0, 'VM_actions': False,
              'independent_final_review': False, 'app_promotion': False, 'goal_complete': False,
              'seconds': time.monotonic() - started}
    with target.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

"""Read-only saved-gradient proof for the inert NumPy gate-aligned prototype."""
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u64_gate_aligned_v1_design'


class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'tensorflow', 'jax'}:
            raise RuntimeError('Framework import forbidden in source-only proof')
        return None


def main():
    sys.meta_path.insert(0, NoFramework())
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from research.codeformer_u64_gate_aligned_v1 import solver, contract
    consumed = {}
    def need(ok, message):
        if not ok:
            raise ValueError(message)
    def read(path, pin=None):
        raw = path.read_bytes()
        actual = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        if pin:
            need(actual == {k: pin[k] for k in actual}, 'Source pin differs: ' + str(path))
        consumed[path.relative_to(ROOT).as_posix()] = actual
        return raw
    for relative in ('research/codeformer_u64_gate_aligned_v1/__init__.py',
                     'research/codeformer_u64_gate_aligned_v1/solver.py',
                     'research/codeformer_u64_gate_aligned_v1/contract.py',
                     'tests/test_u64_gate_aligned_solver_v1.py',
                     'scripts/prove_u64_gate_aligned_solver_v1.py'):
        read(ROOT / relative)
    geometry = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_geometry'
    inventory = json.loads(read(ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_audit/inventory_v1.json'))
    receipts = json.loads(read(geometry / 'vectors.json'))
    prior = json.loads(read(geometry / 'pooled_common_descent_review_v1.json'))
    protocol_raw = read(ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm/protocol.json')
    standard_raw = read(ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm/evidence/standard_protocol.json')
    design = contract.prospective_contract(protocol_raw, standard_raw)
    lookup = {(r['id'], r['query']): r for r in receipts}
    pooled = {r: {q: np.zeros(168960) for q in contract.GATES} for r in contract.ROLES}
    per_case = {}
    for row in design['selection']['optimization']:
        per_case[row['id']] = {}
        for query in contract.GATES:
            receipt = lookup[row['id'], query]
            raw = read(Path(inventory['evidence_root']) / receipt['file']['path'], receipt['file'])
            with np.load(io.BytesIO(raw), allow_pickle=False) as decoded:
                vector = np.concatenate([decoded[n].reshape(-1).astype(np.float64) for n in sorted(decoded.files)])
            need(vector.shape == (168960,) and hashlib.sha256(vector.tobytes()).hexdigest() == receipt['packed_FP64_sha256'], 'Audited complete gradient coordinates differ')
            pooled[row['role']][query] += vector / 16.
            per_case[row['id']][query] = vector
    actual = solver.solve_eight_pooled_gates(pooled)
    rows = np.stack([pooled[r][q] for r, q in solver.LABEL_ORDER])
    lengths = np.sqrt(np.sum(rows * rows, axis=1, dtype=np.float64))
    units = rows / lengths[:, None]
    need(np.allclose(units @ units.T, prior['unit_gram_8x8'], rtol=1e-12, atol=1e-14), 'Eight actual saved pooled gradients differ from SLSQP exploration')
    old_combination = np.sum(np.array(prior['simplex_coefficients'])[:, None] * units, axis=0)
    reference_direction = -old_combination / np.linalg.norm(old_combination)
    difference = actual['direction'] - reference_direction
    l2_difference = float(np.linalg.norm(difference))
    max_difference = float(np.max(np.abs(difference)))
    need(l2_difference < 1e-5 and max_difference < 1e-5, 'Exhaustive-face direction differs from independent SLSQP by >=1e-5')
    need(actual['minimum_normalized_descent_margin'] > 1e-6
         and np.all(actual['original_gradient_dot_direction'] < 0), 'Actual saved eight-gate strict certificate absent')
    full_dots = np.array([float(np.sum(v * actual['direction'], dtype=np.float64)) for v in rows])
    need(np.allclose(full_dots, actual['original_gradient_dot_direction'], rtol=1e-12, atol=1e-14), 'Independent full-coordinate dots differ')
    case_results = []
    for row in design['selection']['optimization']:
        case_results.append({k: row[k] for k in ('id', 'reference_id', 'role')} | {
            'gate_directional_derivatives': {q: float(np.sum(per_case[row['id']][q] * actual['direction'], dtype=np.float64)) for q in contract.GATES}})
    sign_counts = {r: {q: {'decrease': sum(c['gate_directional_derivatives'][q] < 0 for c in case_results if c['role'] == r),
        'zero': sum(c['gate_directional_derivatives'][q] == 0 for c in case_results if c['role'] == r),
        'increase': sum(c['gate_directional_derivatives'][q] > 0 for c in case_results if c['role'] == r)} for q in contract.GATES} for r in contract.ROLES}
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_u64_gate_aligned_solver_v1')
    test_stream = io.StringIO()
    tested = unittest.TextTestRunner(stream=test_stream, verbosity=2).run(suite)
    need(tested.wasSuccessful(), 'Analytical prototype tests failed')
    need(not any(n.split('.')[0] in {'torch', 'tensorflow', 'jax'} for n in sys.modules), 'Framework unexpectedly loaded')
    for relative, pin in consumed.items():
        raw = (ROOT / relative).read_bytes()
        need({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == pin, 'Consumed file changed: ' + relative)
    arithmetic = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in actual.items() if k != 'direction'}
    proof = {'complete': True, 'status': 'INERT_SOURCE_ONLY_PROTOTYPE_TESTED_ON_SAVED_ARRAYS',
        'analytical_tests': {'passed': True, 'run': tested.testsRun, 'failures': len(tested.failures), 'errors': len(tested.errors), 'output': test_stream.getvalue()},
        'actual_saved_gradient_files': 128, 'actual_pooled_gates': 8, 'head_coordinates': 168960,
        'exhaustive_face_solver': arithmetic,
        'independent_SLSQP_direction_comparison': {'pass': True, 'L2_difference': l2_difference,
            'maximum_absolute_coordinate_difference': max_difference, 'exclusive_tolerance': 1e-5},
        'independent_actual_full_coordinate_dot_crosscheck': True,
        'per_role_per_gate_photograph_sign_counts': sign_counts, 'per64_rolecase_results': case_results,
        'consumed_files': consumed, 'all_consumed_before_after_exact': True,
        'execution': {'framework_loaded': False, 'model_constructed': False, 'actual_gradient_computed': False,
            'optimizer_constructed': False, 'parameter_applied': False, 'direction_array_file_written': False,
            'worker_or_training_or_VM_called': False},
        'saved_local_feasibility_is_not_finite_step_or_PNG_success': True,
        'guard_monitor_gradients_not_measured': True,
        'root_review_required_before_any_pilot_selection': True,
        'manual_packet_ready': False, 'pilot_selected': False, 'model_qualified': False}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in (('prototype_contract_v1.json', design), ('prototype_tests_receipt_v1.json', proof)):
        (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'analytical_tests': tested.testsRun,
        'all8_actual_dot_negative': actual['all_actual_derivatives_strict_negative'],
        'minimum_normalized_descent_margin': actual['minimum_normalized_descent_margin'],
        'faces': actual['face_count'], 'accepted_faces': actual['accepted_faces'],
        'solver_seconds': actual['elapsed_seconds'], 'SLSQP_L2_direction_difference': l2_difference,
        'operation_limits': design['limits'], 'no_executable_worker_or_training': True}, indent=2))


if __name__ == '__main__':
    main()

"""Pure contract/NumPy/stub tests; no actual Torch, neural forward or gradient."""
import ast
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np

from research.codeformer_u64_png_surrogate_measurement_v1 import contract, analysis, surrogate


class Scalar:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class StubTensor:
    def __init__(self, array, *, requires_grad=False, device='stub'):
        self.array = np.asarray(array)
        self.requires_grad = requires_grad
        self.device = device
        self.dtype = str(self.array.dtype)
        self.shape = self.array.shape
    def min(self):
        return Scalar(float(self.array.min()))
    def max(self):
        return Scalar(float(self.array.max()))
    def detach(self):
        return StubTensor(self.array, requires_grad=False, device=self.device)
    def clone(self):
        return StubTensor(self.array.copy(), requires_grad=self.requires_grad, device=self.device)


class FiniteStub:
    def __init__(self, value):
        self.value = value
    def all(self):
        return Scalar(bool(np.isfinite(self.value.array).all()))


STUB = types.SimpleNamespace(Tensor=StubTensor, float32='float32',
    isfinite=lambda value: FiniteStub(value), autograd=types.SimpleNamespace(Function=type('Function', (), {})))


def synthetic_bank():
    return {r: {q: np.array([1., 2., 3.]) for q in contract.QUERIES} for r in contract.ROLES}


class MeasurementContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[1] / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
        cls.parent = (base / 'evidence/parent_protocol.json').read_bytes()
        cls.standard = (base / 'evidence/standard_protocol.json').read_bytes()
        cls.selection = contract.select_cases(cls.parent, cls.standard)

    def test_exact72_256_plan_and_opt_only_queries(self):
        plan = contract.operation_plan(self.selection)
        self.assertEqual(len(plan), 72)
        self.assertEqual(sum(len(c['gradient_queries']) for c in plan), 256)
        self.assertEqual([c['slot'] for c in plan], list(range(1, 73)))
        self.assertEqual([c['phase'] for c in plan[:8]], ['historical_u64_parity'] * 4 + ['deterministic_same_u64_repeat'] * 4)
        self.assertTrue(all(not c['gradient_queries'] for c in plan[:8]))
        self.assertEqual([c['id'] for c in plan[8:]], [r['id'] for r in self.selection['optimization']])
        self.assertTrue(all(c['gradient_queries'] == list(contract.QUERIES) for c in plan[8:]))
        self.assertTrue(all(c['metric_region'] == ('central128' if c['role'] == contract.ROLES[0] else 'whole') for c in plan))

    def test_constants_zero_updates575_16vectors_and_resources(self):
        result = contract.prospective_contract()
        self.assertEqual(result['unchanged_tensors'], 575)
        self.assertEqual(result['pooled_vectors'], 16)
        for field in ('parameter_applications', 'optimizer_constructions', 'optimizer_updates'):
            self.assertEqual(result['limits'][field], 0)
        self.assertEqual(result['limits']['worker_seconds'], 900)
        self.assertEqual(result['limits']['external_seconds'], 960)
        self.assertEqual(result['limits']['export_seconds'], 300)
        self.assertEqual(result['limits']['export_external_seconds'], 330)
        self.assertFalse(result['surrogate']['actual_PNG_derivative'])
        self.assertFalse(result['sealed_packet_or_launch_guide_ready'])

    def test_changed_parent_rejects(self):
        with self.assertRaises(ValueError):
            contract.select_cases(self.parent + b' ', self.standard)


class SurrogateStubTests(unittest.TestCase):
    def test_forward_exact_detached_reference_clone_backward_identity(self):
        klass = surrogate.build_png_surrogate(STUB)
        shape = (1, 3, 256, 256)
        raw = StubTensor(np.full(shape, .33333, dtype=np.float32), requires_grad=True)
        decoded = StubTensor(np.full(shape, np.float32(85) / np.float32(255), dtype=np.float32))
        before = decoded.array.tobytes()
        result = klass.forward(None, raw, decoded)
        self.assertEqual(result.array.tobytes(), before)
        self.assertFalse(result.requires_grad)
        self.assertFalse(np.shares_memory(result.array, decoded.array))
        gradient = StubTensor(np.full(shape, -2., dtype=np.float32))
        backward = klass.backward(None, gradient)
        self.assertIs(backward[0], gradient)
        self.assertIsNone(backward[1])
        self.assertEqual(decoded.array.tobytes(), before)

    def test_forward_rejects_wrong_shape_dtype_device_grad_and_nonfinite(self):
        klass = surrogate.build_png_surrogate(STUB)
        shape = (1, 3, 256, 256)
        raw = StubTensor(np.zeros(shape, np.float32), requires_grad=True)
        bad_refs = [StubTensor(np.zeros((1,), np.float32)), StubTensor(np.zeros(shape, np.float64)),
                    StubTensor(np.zeros(shape, np.float32), device='other'),
                    StubTensor(np.zeros(shape, np.float32), requires_grad=True),
                    StubTensor(np.full(shape, np.nan, np.float32)), StubTensor(np.full(shape, 2., np.float32))]
        for reference in bad_refs:
            with self.subTest(shape=reference.shape, dtype=reference.dtype):
                with self.assertRaises(ValueError):
                    klass.forward(None, raw, reference)
        with self.assertRaises(ValueError):
            klass.forward(None, StubTensor(raw.array), StubTensor(raw.array))

    def test_source_does_not_requantize_or_import_framework(self):
        tree = ast.parse(Path(surrogate.__file__).read_text())
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)))
        text = Path(surrogate.__file__).read_text()
        self.assertIn('return decoded_png_tensor.detach().clone()', text)
        self.assertNotIn('floor(', text)
        self.assertNotIn('round(', text)


class MeasurementGeometryTests(unittest.TestCase):
    def test_known_norm_dot_zero_cosine(self):
        result = analysis.comparison(np.array([3., 4.]), np.array([-4., 3.]), np=np)
        self.assertEqual(result['dot'], 0.)
        self.assertEqual(result['left_norm'], 5.)
        self.assertEqual(result['cosine'], 0.)
        self.assertIsNone(analysis.comparison(np.zeros(2), np.ones(2), np=np)['cosine'])

    def test_16vector_pairwise_opposed_RAW_PNG_and_zeros_retained(self):
        bank = synthetic_bank()
        bank[contract.ROLES[0]]['png_surrogate_mae'] *= -1
        bank[contract.ROLES[-1]]['png_surrogate_adjacent'][:] = 0
        result = analysis.summarize_vectors(bank, np=np)
        self.assertEqual(result['vector_count'], 16)
        self.assertEqual(len(result['pairwise16']), 16)
        self.assertAlmostEqual(result['RAW_vs_PNG_surrogate_by_role'][contract.ROLES[0]]['mae']['cosine'], -1.)
        self.assertIsNone(result['RAW_vs_PNG_surrogate_by_role'][contract.ROLES[-1]]['adjacent']['cosine'])
        self.assertEqual(result['measured_zero_vectors'], [{'role': contract.ROLES[-1], 'query': 'png_surrogate_adjacent'}])
        self.assertFalse(result['parameter_directions_computed'])
        self.assertFalse(result['common_direction_solver_used'])

    def test_exact36_synthetic_packing_and_missing_query_rejects(self):
        values = {name: np.zeros(shape, np.float32) for name, shape in contract.head_shapes().items()}
        packed = analysis.pack_gradients(values, np=np)
        self.assertEqual(packed.shape, (168960,))
        self.assertFalse(np.any(packed))
        del values[next(iter(values))]
        with self.assertRaises(ValueError):
            analysis.pack_gradients(values, np=np)
        bank = synthetic_bank()
        del bank[contract.ROLES[0]]['png_surrogate_adjacent']
        with self.assertRaises(ValueError):
            analysis.summarize_vectors(bank, np=np)

    def test_geometry_rejects_dimension_nonfinite_and_wrong_schema(self):
        with self.assertRaises(ValueError):
            analysis.comparison(np.ones(2), np.ones(3), np=np)
        with self.assertRaises(ValueError):
            analysis.comparison(np.array([np.nan]), np.ones(1), np=np)
        values = {name: np.zeros(shape, np.float32) for name, shape in contract.head_shapes().items()}
        values[next(iter(values))] = np.zeros(1, np.float32)
        with self.assertRaises(ValueError):
            analysis.pack_gradients(values, np=np)

    def test_complete_photo_order_mean16_cancellation_preserves_all256_queries(self):
        # Reduced analytical vectors isolate pooling/order from tensor packing;
        # packing itself is checked separately against the exact36-head schema.
        base = Path(__file__).resolve().parents[1] / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
        selection = contract.select_cases((base / 'evidence/parent_protocol.json').read_bytes(),
                                          (base / 'evidence/standard_protocol.json').read_bytes())
        rows = []
        for index, reference in enumerate(selection['optimization']):
            sign = 1. if (index // 4) % 2 == 0 else -1.
            rows.append({k: reference[k] for k in ('id', 'reference_id', 'role')} | {
                'metric_region': contract.metric_region(reference['role']),
                'gradients': {q: np.full(168960, sign * (j + 1.)) for j, q in enumerate(contract.QUERIES)}})
        with mock.patch.object(analysis, 'pack_gradients', side_effect=lambda value, **kwargs: value):
            result = analysis.analyze(rows, selection, np=np)
        self.assertEqual(result['mandatory_query_vectors'], 256)
        self.assertEqual(len(result['per_photograph']), 16)
        self.assertEqual(len(result['pooled_role_mean16']['measured_zero_vectors']), 16)
        self.assertIsNone(result['pooled_role_mean16']['pairwise16'][0][0]['cosine'])
        for role in contract.ROLES:
            for query in contract.QUERIES:
                self.assertEqual(result['pooled_cancellation_diagnostics'][role][query]['pooled_over_mean_individual_norm'], 0.)


if __name__ == '__main__':
    unittest.main()

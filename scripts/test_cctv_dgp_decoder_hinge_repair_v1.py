"""CPU-only boundary tests for the prospective independent repair auditor."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from audit_cctv_dgp_decoder_hinge_repair_v1_return import (
    continuation_allowed, verify_guard_boundary, verify_probe_vector,
    verify_projection_journal, verify_stop_sequence, verify_corrected_guard_binding,
    fixed_objective_arrays, _compare_fixed_arithmetic,
    saved_Adam_arithmetic,
)


def gates(gain=.001, failures=None, source_gain=None):
    return {mode: dict(global_structure_gain=gain,
        source_structure_gains={'asian/degraded': gain if source_gain is None else source_gain,
                                'ffhq/degraded': gain},
        preservation_failures=[] if failures is None else copy.deepcopy(failures),
        brightness_gain_fraction=0., **{'pass': gain >= .01 and (source_gain is None or source_gain >= .01) and not failures})
        for mode in ['raw', 'png']}


class RepairBoundaries(unittest.TestCase):
    def test_legacy_guard_in_training_objective_rejected_before_any_model(self):
        legacy = "from cctv_dgp_bank_photometric_guard_v1_loss import brightness_guard\ndef objective(pred):\n return brightness_guard(pred)\n"
        with self.assertRaises(AssertionError): verify_corrected_guard_binding(legacy)
        corrected = legacy.replace('cctv_dgp_bank_photometric_guard_v1_loss', 'cctv_dgp_bank_hinge_boundary_v1_loss')
        verify_corrected_guard_binding(corrected)
        old = Path(__file__).with_name('cctv_dgp_spatial_readout_v1_vm.py').read_text(encoding='utf-8')
        with self.assertRaises(AssertionError): verify_corrected_guard_binding(old)

    def test_positive_small_gain_allows_continuation_but_does_not_qualify(self):
        value = gates(.0001)
        self.assertTrue(continuation_allowed(value))
        self.assertFalse(all(v['pass'] for v in value.values()))

    def test_probe_pass_never_overrides_full_step_failure(self):
        probe = gates(.02)
        full = gates(.001, [{'group': 'clear', 'metric': 'SSIM'}])
        self.assertTrue(continuation_allowed(probe))
        self.assertFalse(continuation_allowed(full))
        verify_stop_sequence([dict(update=1, gates=full)], 1, 'quality_stop', maximum=50)

    def test_zero_negative_source_and_nonfinite_rejected(self):
        for value in [gates(0), gates(-1e-9), gates(.001, source_gain=0), gates(float('nan'))]:
            with self.subTest(value=value):
                self.assertFalse(continuation_allowed(value))

    def test_PNG_regression_alone_stops_continuation(self):
        value = gates(.002)
        value['png']['preservation_failures'] = [{'group': 'ffhq/clear', 'metric': 'MSE'}]
        self.assertFalse(continuation_allowed(value))

    def test_guard_boundary_requires_exact_scalar_metadata_and_zero_corrected_gradient(self):
        a = np.array([0., 1., -2.], np.float32)
        zero = np.zeros_like(a)
        row = dict(old_penalty=0., new_penalty=0., old_metadata={'guard': 0.}, new_metadata={'guard': 0.})
        verify_guard_boundary(row, a, zero)
        for change in ['gradient', 'scalar', 'metadata', 'nonfinite']:
            altered = copy.deepcopy(row); derivative = zero.copy()
            if change == 'gradient': derivative[0] = np.finfo(np.float32).tiny
            elif change == 'scalar': altered['new_penalty'] = 1e-20
            elif change == 'metadata': altered['new_metadata']['guard'] = 1e-20
            else: derivative[0] = np.nan
            with self.subTest(change=change), self.assertRaises(AssertionError):
                verify_guard_boundary(altered, a, derivative)

    def test_probe_interpolation_float32_bytes_and_radius_are_exact(self):
        origin = np.array([1., .001, -.5], np.float32)
        first = origin + np.array([.0001, -.00002, .00004], np.float32)
        for fraction in [1/16, 1/4]:
            expected = (origin.astype(np.float64) + fraction*(first.astype(np.float64)-origin.astype(np.float64))).astype(np.float32)
            verify_probe_vector(origin, first, expected, fraction, .1)
            with self.assertRaises(AssertionError):
                verify_probe_vector(origin, first, expected.astype(np.float64), fraction, .1)
            with self.assertRaises(AssertionError):
                verify_probe_vector(origin, first, expected, fraction, 1e-8)

    def test_projection_journal_bound_reference_is_original_not_last_update(self):
        origin = np.zeros(3, np.float32)
        proposal = np.array([.12, .16, 0.], np.float32)
        distance = float(np.linalg.norm(proposal.astype(np.float64)))
        reserve = float(2*np.linalg.norm(np.spacing(origin).astype(np.float64)))
        scale = (.1-reserve)/distance
        actual = (origin.astype(np.float64)+scale*(proposal-origin).astype(np.float64)).astype(np.float32)
        row = dict(unconstrained_distance=distance, actual_distance=float(np.linalg.norm(actual.astype(np.float64))),
                   radius=.1, scale=scale, float32_roundoff_reserve=reserve,
                   bound_reference='original selected vector, never previous step', scientific_bound_relaxed=False)
        verify_projection_journal(origin, proposal, actual, row, .1)
        bad = actual.copy(); bad[0] = np.nextafter(bad[0], np.float32(1))
        with self.assertRaises(AssertionError): verify_projection_journal(origin, proposal, bad, row, .1)

    def test_cannot_continue_after_first_scheduled_failed_snapshot(self):
        good, bad = gates(.001), gates(.001, [{'group': 'clear', 'metric': 'MSE'}])
        verify_stop_sequence([dict(update=1, gates=good), dict(update=10, gates=bad)], 10, 'quality_stop', maximum=50)
        with self.assertRaises(AssertionError):
            verify_stop_sequence([dict(update=1, gates=good), dict(update=10, gates=bad), dict(update=20, gates=good)], 20, 'quality_stop', maximum=50)
        with self.assertRaises(AssertionError):
            verify_stop_sequence([dict(update=1, gates=good)], 10, 'quality_stop', maximum=50)

    def test_nonqualified_endpoint_is_still_completed_finite_study(self):
        snapshots = [dict(update=u, gates=gates(.001)) for u in [1, 10, 20, 30, 40, 50]]
        verify_stop_sequence(snapshots, 50, 'maximum_updates', maximum=50)
        self.assertFalse(all(v['pass'] for v in snapshots[-1]['gates'].values()))

    def test_auditor_has_no_candidate_or_autograd_or_GPU_calls(self):
        path = Path(__file__).with_name('audit_cctv_dgp_decoder_hinge_repair_v1_return.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        forbidden = {'backward', 'grad', 'step', 'Adam', 'SGD', 'load_state_dict', 'cuda', 'build_model', 'load_dependencies'}
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else '' for node in calls]
        self.assertFalse(set(names) & forbidden)

    def test_actual_worker_binds_corrected_file_not_legacy_import(self):
        path = Path(__file__).with_name('cctv_dgp_decoder_hinge_repair_v1_vm.py')
        self.assertTrue(path.is_file())
        verify_corrected_guard_binding(path.read_text(encoding='utf-8'))

    def test_fixed_arithmetic_calibrates_against_retained_TWO_source_six_state_rows(self):
        import torch  # Fixed saved-pixel arithmetic, no autograd or model.
        from PIL import Image
        base = Path(__file__).resolve().parent.parent
        parent = base/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
        source = base/'outputs/cctv_dgp_spatial_readout_v1_return'
        sys.path.insert(1, str(parent))
        from cctv_dgp_head4_lossless_v1 import unpack
        from frozen_capacity_contract import feature_support
        read = lambda path: json.loads(path.read_text(encoding='utf-8'))
        old = read(source/'protocol.json'); data = read(parent/'protocol.json')
        saved = read(base/'outputs/cctv_dgp_spatial_readout_v1_analysis/saved_objective_audit.json')
        references, cases = {r['id']: r for r in data['references']}, {c['id']: c for c in data['cases']}
        saved_rows = {r['reference']: r for r in saved['rows']}
        def pixels(path, mode='RGB'):
            with Image.open(path) as image: return np.asarray(image.convert(mode)).copy()
        torch.set_num_threads(4)
        checked = 0
        for serial in [0, 16]:
            block = old['reference_blocks'][serial]; reference = references[block['reference']]
            target = pixels(parent/reference['target']); mask = pixels(parent/reference['observed'], 'L') > 0
            support = [feature_support(mask, cases[cid]['landmarks5_canvas_xy']) for cid in block['case_ids']]
            retained, embeddings = unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz", block['case_ids'])
            for label in ['baseline', 'A50', 'B50']:
                with np.load(source/'outputs/spatial_readout_v1'/label/'packs'/f'r{serial:02d}.npz', allow_pickle=False) as pack:
                    value = fixed_objective_arrays(pack['raw'], retained, target, mask, support, embeddings,
                                                   pack['embeddings_raw'], label == 'baseline', old)
                self.assertLessEqual(_compare_fixed_arithmetic(value, saved_rows[block['reference']]['snapshots'][label]), 1e-10)
                checked += 1
        self.assertEqual(checked, 6)

    def test_Adam_arithmetic_calibrates_all50_retained_actual_source_proposals_and_final_moments(self):
        import torch  # Saved tensor deserialization only.
        source = Path(__file__).resolve().parent.parent/'outputs/cctv_dgp_spatial_readout_v1_return/outputs/spatial_readout_v1'
        current = np.load(source/'A_initial_vector.npy', allow_pickle=False)
        moments = np.zeros_like(current); squares = np.zeros_like(current)
        for update in range(1, 51):
            with np.load(source/'trajectory_A'/f'update_{update:03d}.npz', allow_pickle=False) as archive:
                proposal, moments, squares = saved_Adam_arithmetic(current, archive['clipped_gradient'], moments, squares, update)
                np.testing.assert_allclose(proposal, archive['unconstrained'], rtol=2e-6, atol=3e-7)
                current = archive['actual'].copy()
        state = torch.load(source/'state_A50/full_state.pth', map_location='cpu', weights_only=True)
        layout = json.loads((source/'A_parameter_layout.json').read_text(encoding='utf-8'))
        offset = 0
        for identifier, row in zip(state['optimizer']['param_groups'][0]['params'], layout):
            count = int(np.prod(row['shape']))
            for key, values in [('exp_avg', moments), ('exp_avg_sq', squares)]:
                actual = state['optimizer']['state'][identifier][key].numpy()
                expected = values[offset:offset+count].reshape(row['shape'])
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-8)
            offset += count
        self.assertEqual(offset, current.size)


if __name__ == '__main__': unittest.main(verbosity=2)

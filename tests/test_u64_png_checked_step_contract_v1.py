"""Exact parent/finite-plan fixtures only; no real model/copy/query/VM call."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import unittest

from research.codeformer_u64_png_checked_step_v1 import contract as C

ROOT = Path(__file__).resolve().parents[1]


class CheckedStepContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
        cls.parent = (base / 'evidence/parent_protocol.json').read_bytes()
        cls.standard = (base / 'evidence/standard_protocol.json').read_bytes()
        cls.selection = C.select_cases(cls.parent, cls.standard)
        cls.plan = C.operation_plan(cls.selection)

    def test_exact445_rows436_forwards9_apps_zero_queries(self):
        self.assertEqual(len(self.plan), 445)
        self.assertEqual([c['slot'] for c in self.plan], list(range(1, 446)))
        self.assertEqual(sum(c['kind'] == 'forward' for c in self.plan), 436)
        self.assertEqual(sum(c['kind'] == 'parameter_application' for c in self.plan), 9)
        self.assertTrue(all(c['gradient_queries'] == [] for c in self.plan))
        self.assertEqual(C.LIMITS['gradient_query_attempts'], 0)
        self.assertEqual(C.LIMITS['optimizer_constructions'], 0)
        self.assertEqual(C.LIMITS['selected_training_updates'], 1)

    def test_exact_categories_initial124_trials256_selected56(self):
        expected = {'initial_quality_forwards': 56, 'initial_repeat_forwards': 4,
            'initial_optimization_forwards': 64, 'trial_optimization_forwards': 256,
            'selected_quality_forwards': 56}
        self.assertEqual(C.CATEGORIES, expected)
        for category, count in expected.items():
            self.assertEqual(sum(c.get('category') == category for c in self.plan), count)
        self.assertTrue(all(c['kind'] == 'forward' for c in self.plan[:124]))
        self.assertEqual(self.plan[124]['action'], 'trial_apply')
        self.assertEqual(self.plan[124]['trial_index'], 0)

    def test_four_independent_trials_each64_optimization_then_original_restore(self):
        expected_ids = [r['id'] for r in self.selection['optimization']]
        for trial in range(4):
            start = 124 + trial * 66
            apply, bank, restore = self.plan[start], self.plan[start + 1:start + 65], self.plan[start + 65]
            self.assertEqual((apply['action'], apply['trial_index']), ('trial_apply', trial))
            self.assertEqual(apply['source_state'], 'original_u64')
            self.assertEqual([c['id'] for c in bank], expected_ids)
            self.assertTrue(all(c['trial_index'] == trial and c['phase'] == 'trial_optimization_bank' for c in bank))
            self.assertEqual((restore['action'], restore['trial_index']), ('trial_restore', trial))
            self.assertEqual(restore['destination_state'], 'original_u64')
            self.assertEqual(restore['destination_head_canonical_sha256'], C.HEAD_CANONICAL_SHA256)
            self.assertTrue(restore['worker_must_record_exact_before_after_canonical'])

    def test_no_eligible_prefix_restores_fourth_before_any_selected_branch(self):
        prefix = self.plan[:C.NO_ELIGIBLE_PREFIX['plan_rows']]
        self.assertEqual(len(prefix), 388)
        self.assertEqual(sum(c['kind'] == 'forward' for c in prefix), 380)
        self.assertEqual(sum(c['kind'] == 'parameter_application' for c in prefix), 8)
        self.assertEqual(prefix[-1]['action'], 'trial_restore')
        self.assertEqual(prefix[-1]['trial_index'], 3)
        self.assertEqual(C.NO_ELIGIBLE_PREFIX['final_head_canonical_sha256'], C.HEAD_CANONICAL_SHA256)
        selected = self.plan[388]
        self.assertEqual(selected['action'], 'selected_commit')
        self.assertIsNone(selected['trial_index'])
        self.assertEqual(selected['source_state'], 'original_u64')
        self.assertEqual(selected['destination_state'], 'selected_eligible_candidate')
        self.assertEqual(len(self.plan[389:]), 56)
        self.assertTrue(all(c['phase'] == 'selected_quality_bank' for c in self.plan[389:]))

    def test_quality_rows_disjoint_from_selection_and_exact_ROI(self):
        optimization = {r['id'] for r in self.selection['optimization']}
        quality = {r['id'] for r in self.selection['monitors'] + self.selection['guards']}
        self.assertFalse(optimization & quality)
        for call in self.plan:
            if call['kind'] != 'forward':
                continue
            if call['category'] in ('initial_optimization_forwards', 'trial_optimization_forwards'):
                self.assertIn(call['id'], optimization)
            else:
                self.assertIn(call['id'], quality)
            if call['role'] in C.ROLES:
                self.assertEqual(call['metric_region'], 'central128' if call['role'] == C.ROLES[0] else 'whole')

    def test_exact_original_parent_provenance_and_schema(self):
        self.assertEqual(hashlib.sha256(self.parent).hexdigest(), C.PARENT_SHA256)
        self.assertEqual(hashlib.sha256(self.standard).hexdigest(), C.STANDARDIZED_SHA256)
        shapes = C.head_shapes()
        self.assertEqual(len(shapes), 36)
        import math
        self.assertEqual(sum(math.prod(s) for s in shapes.values()), 168960)
        with self.assertRaises(ValueError):
            C.select_cases(self.parent + b' ', self.standard)
        with self.assertRaises(ValueError):
            C.select_cases(self.parent, self.standard + b' ')

    def test_contract_caps_status_and_restores_not_descent(self):
        from research.codeformer_u64_png_checked_step_v1 import selection
        self.assertEqual(C.CANDIDATE_MAXABS, selection.CANDIDATE_MAXABS)
        self.assertEqual(C.NUMERICAL_HEADROOM, selection.NUMERICAL_HEADROOM)
        c = C.prospective_contract()
        self.assertEqual(c['candidate_maximum_absolute_coordinate_bounds'], [2e-5, 1.5e-5, 1e-5, 5e-6])
        self.assertEqual(c['fixed_numerical_headroom'], .99)
        self.assertEqual((c['total_tensors'], c['frozen_R2_U96_tensors'], c['trainable_head_tensors']), (575, 539, 36))
        for field, value in {'worker_seconds': 1200, 'external_seconds': 1260,
            'termination_grace_seconds': 20, 'per_phase_seconds': 180,
            'export_seconds': 300, 'export_external_seconds': 330,
            'maximum_RSS_bytes': 10 * 1024**3, 'maximum_CUDA_reserved_bytes': 18 * 1024**3,
            'initial_free_disk_bytes': 8 * 1024**3, 'continuing_free_disk_bytes': 4 * 1024**3,
            'maximum_output_bytes': 2 * 1024**3, 'maximum_archive_bytes': 2 * 1024**3}.items():
            self.assertEqual(c['limits'][field], value)
        self.assertIn('not descent steps', c['restore_semantics'])
        self.assertTrue(c['all_original_quality_gates_unchanged'])
        self.assertFalse(c['worker_implemented_by_this_contract'])
        self.assertFalse(c['sealed_packet_or_launch_guide_ready_by_this_contract'])
        self.assertFalse(c['model_qualified'])

    def test_inert_contract_import_neural_blocker(self):
        code = '''import importlib.abc,sys,importlib
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname.split('.')[0] in {'numpy','torch','tensorflow','jax'}:raise RuntimeError('Forbidden scientific import')
  return None
sys.meta_path.insert(0,Block())
c=importlib.import_module('research.codeformer_u64_png_checked_step_v1.contract')
assert c.LIMITS['gradient_query_attempts']==0
assert not any(n.split('.')[0] in {'numpy','torch','tensorflow','jax'} for n in sys.modules)
print('PURE_CONTRACT_ONLY')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'PURE_CONTRACT_ONLY')


if __name__ == '__main__':
    unittest.main()

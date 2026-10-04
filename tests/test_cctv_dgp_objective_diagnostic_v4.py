"""No backward calls: test cohort boundaries and gradient-summary arithmetic."""
import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import torch

from cctv_dgp_objective_diagnostic_v4 import training_groups, gradient_summary, summary_from_gram, TERM_WEIGHTS, TEACHERS
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_cctv_dgp_objective_diagnostic_vm as runner
import audit_cctv_dgp_objective_diagnostic_v4 as auditor
from cctv_dgp_pilot import read, sha
from cctv_dgp_perceptual_training_v3 import START_SHA, START_STATE


def write(path, value):
    """Only mutable synthetic test fixtures use replacement writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def cohort():
    sources = ['dataset/asian_faces', 'dataset/thumbnails128x128']
    profiles = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
    cases = []
    refs = []
    for source in sources:
        for profile in profiles:
            for index in range(6):
                name = source + '/' + profile + '/' + str(index)
                refs.append({'id': name, 'role': 'train', 'source': source})
                cases.append({'id': name, 'reference_id': name, 'source': source, 'profile': profile})
    return {'references': refs, 'training_epochs': {'1': cases}}


class ObjectiveDiagnosticContracts(unittest.TestCase):
    def test_first_four_per_source_profile_are_fixed_training_cases(self):
        protocol = cohort()
        original = copy.deepcopy(protocol)
        groups = training_groups(protocol)
        self.assertEqual(len(groups), 10)
        cases = [c for group in groups for c in group['cases']]
        self.assertEqual(len(cases), 40)
        self.assertEqual(len({c['reference_id'] for c in cases}), 40)
        self.assertTrue(all(c['id'].endswith(('/0', '/1', '/2', '/3')) for c in cases))
        self.assertEqual(protocol, original)

    def test_validation_reference_cannot_enter_the_probe(self):
        protocol = cohort()
        protocol['references'][0]['role'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'training'):
            training_groups(protocol)

    def test_missing_profile_or_duplicate_reference_is_rejected(self):
        protocol = cohort()
        protocol['training_epochs']['1'] = [c for c in protocol['training_epochs']['1'] if c['profile'] != 'clear']
        with self.assertRaises(ValueError):
            training_groups(protocol)
        protocol = cohort()
        protocol['training_epochs']['1'][1]['reference_id'] = protocol['training_epochs']['1'][0]['reference_id']
        with self.assertRaises(ValueError):
            training_groups(protocol)

    def test_parallel_opposing_and_zero_gradients_are_distinguished(self):
        value = gradient_summary({'pixel': torch.tensor([3., 4.]),
                                  'identity': torch.tensor([-6., -8.]),
                                  'zero': torch.zeros(2)})
        self.assertAlmostEqual(value['terms']['pixel']['l2_norm'], 5.)
        self.assertAlmostEqual(value['pairwise_cosines']['pixel/identity'], -1.)
        self.assertIsNone(value['pairwise_cosines']['pixel/zero'])
        self.assertAlmostEqual(value['total_l2_norm'], 5.)
        self.assertAlmostEqual(value['cancellation_ratio'], 1 / 3)

    def test_nonfinite_or_different_gradient_shapes_are_rejected(self):
        for bad in ({'pixel': torch.tensor([float('nan')])},
                    {'pixel': torch.ones(2), 'identity': torch.ones(3)}):
            with self.assertRaises(ValueError):
                gradient_summary(bad)

    def test_impossible_gram_and_zero_gradient_cross_product_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            summary_from_gram(['a', 'b', 'c'], 3, [[1., .9, .9], [.9, 1., -.9], [.9, -.9, 1.]])
        with self.assertRaisesRegex(ValueError, 'Zero gradient'):
            summary_from_gram(['a', 'zero'], 2, [[1., 1e-12], [1e-12, 0.]])

    def test_local_probe_stops_before_loading_output_or_any_gradient(self):
        with patch('torch.cuda.is_available', return_value=False), \
                patch.object(runner, 'verify_recipe') as verify, \
                patch.object(Path, 'mkdir') as mkdir, patch('torch.load') as load, \
                patch('torch.autograd.grad') as gradient, patch('torch.optim.Adam') as optimizer:
            with self.assertRaisesRegex(RuntimeError, 'CUDA required'):
                runner.run(ROOT, ROOT, ROOT / 'scratch/forbidden-objective-probe')
        for operation in (verify, mkdir, load, gradient, optimizer):
            operation.assert_not_called()


class ReturnedDiagnosticContracts(unittest.TestCase):
    """Synthetic report fixtures only; no network forward or backward replay."""

    def setUp(self):
        self.directory = (ROOT / 'scratch' / ('objective-report-test-' + uuid.uuid4().hex)).resolve()
        self.assertTrue(self.directory.is_relative_to((ROOT / 'scratch').resolve()))
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        self.bundle = self.directory / 'bundle'
        self.output = self.directory / 'output'
        self.bundle.mkdir()
        self.output.mkdir()
        write(self.bundle / 'objective_diagnostic_protocol_v4.json', {'fixture': True})
        source = 'v3_local_audit_for_diagnostic.json'
        write(self.output / 'runtime_sources' / source, {'fixture': True})
        self.protocol = {'groups': training_groups(cohort()),
                         'assets_sha256': {source: sha(self.output / 'runtime_sources' / source)}}
        write(self.output / 'execution.json', {
            'protocol_sha256': sha(self.bundle / 'objective_diagnostic_protocol_v4.json'),
            'host': 'forensic-dgp-thesis', 'device': 'cuda', 'gpu': 'NVIDIA L4',
            'optimizer_constructed': False, 'actual_training': False,
            'starting_checkpoint_sha256': START_SHA, 'starting_state_hash': START_STATE,
            'teachers_before': TEACHERS, 'student_parameter_dimension': 5,
            'runtime_cap_seconds': 600, 'normalization_stats_cloned': True,
        })
        gram = [[float(i == j) for i in range(5)] for j in range(5)]
        for index, group in enumerate(self.protocol['groups']):
            write(self.output / f'group_{index:02d}.json', {
                'group': group, 'optimizer_updates': 0, 'state_unchanged': True,
                'autograd_grad_calls': 5, 'cumulative_autograd_grad_calls': 5 * (index + 1),
                'elapsed_seconds': float(index + 1),
                'unweighted_losses': {k: 2. for k in TERM_WEIGHTS},
                'weighted_losses': {k: 2. * v for k, v in TERM_WEIGHTS.items()},
                'input_gradient_support': 'Observed pixels only; uncaptured padding gradient zeroed',
                'student_parameters': summary_from_gram(list(TERM_WEIGHTS), 5, gram),
                'restoration_input': summary_from_gram(list(TERM_WEIGHTS), 4 * 3 * 256 * 256, gram),
            })
        self.report = {
            'complete': True, 'protocol_sha256': sha(self.bundle / 'objective_diagnostic_protocol_v4.json'),
            'student_state_before': START_STATE, 'student_state_after': START_STATE,
            'teachers_before': TEACHERS, 'teachers_after': TEACHERS,
            'optimizer_updates': 0, 'optimizer_constructed': False, 'parameter_grads_accumulated': False,
            'actual_training': False, 'improved_model_claimed': False, 'production_checkpoint_promoted': False,
            'groups': 10, 'training_references': 40, 'student_forwards': 10, 'autograd_grad_calls': 50,
            'seconds': 12., 'validation_references_used': 0, 'native_cases_used': 0, 'native_reserved_used': False,
        }
        self.seal()

    def seal(self):
        self.report['artifacts_sha256'] = {p.relative_to(self.output).as_posix(): sha(p)
                                          for p in self.output.rglob('*') if p.is_file() and p.name != 'results.json'}
        write(self.output / 'results.json', self.report)

    def audit(self):
        with patch.object(auditor, 'verify_recipe', return_value=({}, self.protocol)), \
                patch('torch.autograd.grad') as gradient, patch('torch.optim.Adam') as optimizer:
            result = auditor.audit(ROOT, self.bundle, self.output)
        gradient.assert_not_called()
        optimizer.assert_not_called()
        return result

    def test_valid_fixture_rebuilds_twenty_gram_summaries_without_backward(self):
        result = self.audit()
        self.assertEqual(result['serialized_gram_summaries_rebuilt'], 20)
        self.assertFalse(result['cuda_gradients_recomputed_by_this_audit'])
        self.assertFalse(result['model_improvement_established'])

    def test_resealed_cohort_substitution_is_rejected(self):
        path = self.output / 'group_00.json'
        row = read(path)
        row['group']['cases'][0]['reference_id'] = 'validation-substitution'
        write(path, row)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Training-only group'):
            self.audit()

    def test_resealed_false_gradient_summary_is_rejected(self):
        path = self.output / 'group_00.json'
        row = read(path)
        row['student_parameters']['terms']['identity']['l2_norm'] = 2.
        write(path, row)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Gram-derived summary'):
            self.audit()

    def test_optimizer_update_or_changed_teacher_cannot_be_accepted(self):
        self.report['optimizer_updates'] = 1
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Diagnostic/training'):
            self.audit()
        self.report['optimizer_updates'] = 0
        self.report['teachers_after'] = {**TEACHERS, 'identity': 'changed'}
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Frozen teacher'):
            self.audit()

    def test_resealed_replacement_of_pinned_source_is_rejected(self):
        write(self.output / 'runtime_sources/v3_local_audit_for_diagnostic.json', {'fixture': False})
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Returned executable/capsule'):
            self.audit()

    def test_changed_execution_budget_or_norm_policy_is_rejected(self):
        path = self.output / 'execution.json'
        for key, value in [('runtime_cap_seconds', 1200), ('normalization_stats_cloned', False)]:
            original = read(path)
            write(path, {**original, key: value})
            self.seal()
            with self.assertRaisesRegex(ValueError, 'Runtime/normalization'):
                self.audit()
            write(path, original)


if __name__ == '__main__':
    unittest.main()

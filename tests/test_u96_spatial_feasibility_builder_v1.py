"""Current-source binding tests, using only small temporary regular files."""
import argparse
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/build_u96_spatial_feasibility_v1.py'


def load():
    spec = importlib.util.spec_from_file_location('inert_spatial_builder_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Builder(unittest.TestCase):
    def setUp(self):
        self.module = load()
        scratch = ROOT / 'scratch'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='spatial-builder-fixture-', dir=scratch)
        self.base = Path(self.temporary.name).resolve()
        self.assertTrue(self.base.is_relative_to(scratch.resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.module.ROOT = self.base
        self.module.PACKAGE = self.base
        original_pin = self.module.pin
        self.module.pin = lambda path, base=None: original_pin(path, self.base if base is None else base)
        self.source = self.base / 'worker.py'
        self.source.write_bytes(b'"""Fixture only; never a runtime worker."""\n')
        self.required = {'worker.py'}

    def records(self):
        bound = self.module.pin(self.source)
        return ({'complete': True, 'source_review_pass': True,
                 'unresolved_blockers': [], 'sources': [bound]},
                {'complete': True, 'all_tests_pass': True,
                 'neural_gradient_optimizer_training_VM_calls': 0, 'sources': [bound]})

    def test_exact_complete_review_and_tests_accept_static_bytes(self):
        review, tests = self.records()
        self.module.validate_review(review, tests, self.required)
        self.assertNotIn('torch', sys.modules)

    def test_reviewed_source_mutation_refused(self):
        review, tests = self.records()
        self.source.write_bytes(self.source.read_bytes() + b'# changed\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.module.validate_review(review, tests, self.required)

    def test_missing_new_executable_review_or_test_binding_refused(self):
        review, tests = self.records()
        for record in (review, tests):
            original = record['sources']
            record['sources'] = []
            with self.assertRaisesRegex(ValueError, 'Every executable'):
                self.module.validate_review(review, tests, self.required)
            record['sources'] = original
        with self.assertRaisesRegex(ValueError, 'Every executable'):
            self.module.validate_review(review, tests, {'worker.py', 'new_entry.py'})

    def test_duplicate_pins_and_unresolved_findings_refused(self):
        review, tests = self.records()
        review['sources'] *= 2
        with self.assertRaisesRegex(ValueError, 'Unique'):
            self.module.validate_review(review, tests, self.required)
        review, tests = self.records()
        review['unresolved_blockers'] = ['Actual RAW gate missing']
        with self.assertRaisesRegex(ValueError, 'Independent complete-runner'):
            self.module.validate_review(review, tests, self.required)

    def test_actual_model_calls_cannot_be_labeled_pure_source_tests(self):
        review, tests = self.records()
        tests['neural_gradient_optimizer_training_VM_calls'] = 1
        with self.assertRaisesRegex(ValueError, 'Pure current-source tests'):
            self.module.validate_review(review, tests, self.required)

    def test_help_exits_before_review_read_or_artifact_creation(self):
        with patch.object(sys, 'argv', ['builder', '--help']), \
                patch.object(self.module, 'doc', side_effect=AssertionError('No review reads on --help')), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            with self.assertRaises(SystemExit) as caught:
                self.module.main()
        self.assertEqual(caught.exception.code, 0)
        self.assertIn('--build', stream.getvalue())

    def test_build_flag_is_required_and_python310_source_is_valid(self):
        with patch.object(sys, 'argv', ['builder']):
            with self.assertRaisesRegex(ValueError, 'Use --build'):
                self.module.main()
        ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))

    def test_mutation_after_approval_is_refused_at_copy_binding(self):
        review, tests = self.records()
        approved = self.module.validate_review(review, tests, self.required)
        self.source.write_bytes(self.source.read_bytes() + b'# changed AFTER approval\n')
        with self.assertRaisesRegex(ValueError, 'after review'):
            self.module.approved_copy_binding(self.source, approved)

    def test_new_unreviewed_executable_is_refused_at_copy_binding(self):
        review, tests = self.records()
        approved = self.module.validate_review(review, tests, self.required)
        extra = self.base / 'extra.py'
        extra.write_bytes(b'# not reviewed\n')
        with self.assertRaisesRegex(ValueError, 'after review'):
            self.module.approved_copy_binding(extra, approved)

    def test_parent_bound_data_is_rechecked_at_copy_boundary(self):
        review, tests = self.records()
        approved = self.module.validate_review(review, tests, self.required)
        prior = self.module.pin(self.source)
        with self.assertRaisesRegex(ValueError, 'Parent-bound content'):
            self.module.approved_copy_binding(self.source, approved,
                expected_content={**prior, 'sha256': '0' * 64})

    def test_lexical_dotdot_and_outside_base_pin_are_refused(self):
        inner = self.base / 'inner'
        inner.mkdir()
        outside = self.base / 'outside.txt'
        outside.write_bytes(b'kept within our temporary fixture\n')
        for bad in (inner / '..' / 'outside.txt', outside):
            with self.assertRaisesRegex(ValueError, 'Canonical source'):
                self.module.pin(bad, inner)


if __name__ == '__main__':
    unittest.main()

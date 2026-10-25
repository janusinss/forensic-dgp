"""Current-source binding tests, using only small temporary regular files."""
import argparse
import ast
import builtins
import contextlib
import importlib.util
import io
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/build_u64_cleananchors_continuation_v1.py'


def load():
    spec = importlib.util.spec_from_file_location('inert_u64_builder_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Builder(unittest.TestCase):
    def setUp(self):
        self.module = load()
        scratch = ROOT / 'scratch'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='u64-builder-fixture-', dir=scratch)
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

    def test_archive_omits_only_large_R2_U96_files_and_carries_learned_heads(self):
        files = {name: {'bytes': 1, 'sha256': 'a' * 64} for name in (
            'weights/codeformer_r2.pth', 'weights/codeformer_u96_adapter.pth',
            'weights/terminal_u0064_heads.npz', 'scripts/run_continuation.sh')}
        omitted = {'weights/codeformer_r2.pth', 'weights/codeformer_u96_adapter.pth'}
        actual = self.module.included_archive_files(files, omitted)
        self.assertEqual(set(actual), set(files) - omitted)
        self.assertIn('weights/terminal_u0064_heads.npz', actual)
        with self.assertRaisesRegex(ValueError, 'terminal-u64 heads'):
            self.module.included_archive_files(files, {'weights/codeformer_r2.pth', 'weights/terminal_u0064_heads.npz'})
        with self.assertRaisesRegex(ValueError, 'two initializer'):
            self.module.included_archive_files(files, set())

    def cache_fixture(self):
        prep = self.base / 'prep'
        prep.mkdir()
        self.module.PREP = prep
        self.module.CACHE_RECEIPT = prep / 'materialized.json'
        monitors = [{'id': 'monitor' + str(i), 'role': 'role', 'reference_id': 'ref' + str(i), 'split': 'TRAIN'}
                    for i in range(32)]
        guards = [{'id': 'guard' + str(i), 'role': 'guard', 'reference_id': 'guardref' + str(i), 'split': 'TRAIN'}
                  for i in range(24)]
        scope = {'complete': True}
        materialized = {'complete': True, 'cache_payload_hashes_exact': True, 'physical_cache_files': 180,
                        'physical_cache_inventory': {}}
        for name, cases in (('cached_frozen_U96_controls', monitors + guards),
                            ('terminal_u64_parity_witnesses', monitors[:4])):
            scope[name], materialized[name] = [], []
            for case in cases:
                row = {**case, 'cache_materialized': False, 'outputs': {}}
                physical = {}
                for field in ('raw', 'png', 'record'):
                    old_path = name + '/' + case['id'] + '/' + field
                    path = prep / 'cached_controls' / old_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(('synthetic ' + field).encode())
                    binding = self.module.pin(path)
                    old = {'path': old_path, 'bytes': binding['bytes'], 'sha256': binding['sha256']}
                    physical[field] = binding
                    materialized['physical_cache_inventory'][old_path] = binding
                    if field == 'record':
                        row['record_member'] = old
                    else:
                        row['outputs'][field] = old
                scope[name].append(copy.deepcopy(row))
                materialized[name].append({**row, 'cache_materialized': True,
                    'cache_record': physical['record'], 'cache_outputs': {k: physical[k] for k in ('raw', 'png')}})
        scope_path = prep / 'scope.json'
        scope_path.write_text(json.dumps(scope))
        materialized['scope'] = self.module.pin(scope_path)
        materialized['physical_cache_bytes'] = sum(row['bytes'] for row in materialized['physical_cache_inventory'].values())
        self.module.CACHE_RECEIPT.write_text(json.dumps(materialized))
        peer = {'complete': True, 'independent_cached_controls_readback_pass': True,
                'materialization_receipt': self.module.pin(self.module.CACHE_RECEIPT),
                'all_actual_cache_bytes_equal_fixed_archive_index': True}
        return materialized, peer, {'monitors': monitors, 'guards': guards}

    def test_exact_180_file_cache_and_both_ordered_tables_validate(self):
        numpy_before = {name for name in sys.modules if name == 'numpy' or name.startswith('numpy.')}
        original_import = builtins.__import__
        prohibited_attempts = []
        def static_import(name, *args, **kwargs):
            if name in ('torch', 'numpy') or name.startswith(('torch.', 'numpy.')):
                prohibited_attempts.append(name)
                raise AssertionError('Builder must not import numerical frameworks: ' + name)
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, '__import__', static_import):
            materialized, peer, selection = self.cache_fixture()
            rows = self.module.validate_cache(materialized, peer, selection)
        self.assertEqual(len(rows['cached_frozen_U96_controls']), 56)
        self.assertEqual(len(rows['terminal_u64_parity_witnesses']), 4)
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(prohibited_attempts, [])
        self.assertEqual({name for name in sys.modules if name == 'numpy' or name.startswith('numpy.')}, numpy_before)

    def test_cache_mutation_or_unbound_extra_payload_cannot_be_sealed(self):
        materialized, peer, selection = self.cache_fixture()
        raw = self.base / materialized['cached_frozen_U96_controls'][0]['cache_outputs']['raw']['path']
        raw.write_bytes(b'changed after cache review')
        with self.assertRaisesRegex(ValueError, 'physical cached file'):
            self.module.validate_cache(materialized, peer, selection)
        # Restore exact source bytes before independently testing an extra file.
        raw.write_bytes(b'synthetic raw')
        extra = self.module.PREP / 'cached_controls/unreviewed.bin'
        extra.write_bytes(b'not selected')
        with self.assertRaisesRegex(ValueError, 'unexpected file'):
            self.module.validate_cache(materialized, peer, selection)

    def test_cache_review_cannot_be_stale_or_reorder_control_roles(self):
        materialized, peer, selection = self.cache_fixture()
        stale = {**peer, 'materialization_receipt': {**peer['materialization_receipt'], 'sha256': '0' * 64}}
        with self.assertRaisesRegex(ValueError, 'Independent current'):
            self.module.validate_cache(materialized, stale, selection)
        materialized['cached_frozen_U96_controls'][0], materialized['cached_frozen_U96_controls'][1] = \
            materialized['cached_frozen_U96_controls'][1], materialized['cached_frozen_U96_controls'][0]
        with self.assertRaisesRegex(ValueError, 'Exact ordered56'):
            self.module.validate_cache(materialized, peer, selection)

    def test_review_and_admission_checks_precede_any_stage_creation_in_source(self):
        source = SOURCE.read_text(encoding='utf-8')
        self.assertLess(source.index('approved = validate_review'), source.index('STAGE.mkdir()'))
        self.assertLess(source.index("review['continuation_reuse'] == pin(ADMISSION)"), source.index('STAGE.mkdir()'))
        self.assertLess(source.index('cache_rows = validate_cache'), source.index('STAGE.mkdir()'))
        self.assertIn("'historical_archive_index': historical_index", source)
        self.assertIn("'CCTV_CODEFORMER_U64_CLEANANCHORS_CONTINUATION_V1_PREPARATION.md'", source)


if __name__ == '__main__':
    unittest.main()

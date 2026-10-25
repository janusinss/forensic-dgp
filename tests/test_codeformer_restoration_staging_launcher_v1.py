"""Pure staging manifest validation; no server, model, workers or source approval."""
import copy
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('staging_launcher', ROOT / 'serve_codeformer_restoration_staging_v1.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def fixture():
    pin = lambda path: {'path': path, 'bytes': 1, 'sha256': '1' * 64}
    old = {key: pin(key + '.json') for key in ('parent', 'brush', 'lifecycle')}
    sources = [pin(name) for name in M.SOURCES]
    review_pin = pin('outputs/independent_review.json')
    value = {'complete': True, 'format': M.FORMAT, 'scope': copy.deepcopy(M.SCOPE),
             'sources': sources, 'old_manifests': old, 'source_review': review_pin}
    review = {'complete': True, 'source_review_pass': True, 'scope': copy.deepcopy(M.SCOPE),
              'sources': copy.deepcopy(sources), 'old_manifests': copy.deepcopy(old)}
    return value, review, old


class LauncherContracts(unittest.TestCase):
    def validate(self, value, review, old):
        seen = []
        return M.validate_staging_contract(value, old, lambda row, cap: seen.append(row), lambda row: review)

    def test_exact_seven_sources_and_review_accept(self):
        value, review, old = fixture(); result = self.validate(value, review, old)
        self.assertEqual(result['sources'], value['sources'])
        result['sources'][0]['sha256'] = '2' * 64
        self.assertNotEqual(result['sources'], value['sources'])

    def test_missing_duplicate_or_extra_source_refused(self):
        for mutate in ('missing', 'duplicate', 'extra'):
            value, review, old = fixture()
            if mutate == 'missing': value['sources'].pop()
            if mutate == 'duplicate': value['sources'][-1] = value['sources'][0]
            if mutate == 'extra': value['sources'].append({'path': 'extra.py'})
            with self.assertRaises(ValueError): self.validate(value, review, old)

    def test_independent_review_must_match_current_sources(self):
        value, review, old = fixture(); review['sources'][0]['sha256'] = '2' * 64
        with self.assertRaises(ValueError): self.validate(value, review, old)
        value, review, old = fixture(); review['source_review_pass'] = False
        with self.assertRaises(ValueError): self.validate(value, review, old)

    def test_promotion_or_training_or_preparation_auto_refused(self):
        for key, changed in [('main_app_default_changed', True), ('model_qualified', True),
                             ('training_optimizer_VM_calls', 1), ('automatic_preparation', True),
                             ('maximum_forwards_per_request', 2)]:
            value, review, old = fixture(); value['scope'][key] = changed
            with self.assertRaises(ValueError): self.validate(value, review, old)
        value, review, old = fixture(); value['scope']['training_optimizer_VM_calls'] = False
        with self.assertRaises(ValueError): self.validate(value, review, old)

    def test_changed_old_manifest_refused(self):
        for key in ('parent', 'brush', 'lifecycle'):
            value, review, old = fixture(); value['old_manifests'] = copy.deepcopy(old)
            value['old_manifests'][key]['sha256'] = '2' * 64
            with self.assertRaises(ValueError): self.validate(value, review, old)

    def test_exact_sha_binder_failure_propagates(self):
        value, review, old = fixture()
        def reject(row, cap): raise ValueError('Current source hash mismatch')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            M.validate_staging_contract(value, old, reject, lambda row: review)

    def test_cli_preserves_old_manifest_flags_and_verify_no_receipt(self):
        flags = []
        for name in ('manifest', 'source-manifest', 'lifecycle-manifest', 'staging-manifest'):
            flags += ['--' + name, name + '.json', '--' + name + '-sha', '1' * 64]
        args = M.parser().parse_args(flags + ['--verify-only'])
        self.assertTrue(args.verify_only); self.assertIsNone(args.receipt); self.assertEqual(args.port, 0)

    def test_verify_only_never_opens_socket_or_creates_worker(self):
        args = type('Args', (), {'port': 0, 'verify_only': True})()
        source = type('Sources', (), {'check': lambda self: {'models': {}, 'stable': True}})()
        fake_parser = type('Parser', (), {'parse_args': lambda self: args})()
        with patch.object(M, 'parser', return_value=fake_parser), patch.object(M, 'configure', return_value=source), \
             patch.object(M.socket, 'socket', side_effect=AssertionError('No socket in verify-only')), \
             patch.object(M.L, 'ProcessSelectedRestorer', side_effect=AssertionError('No worker in verify-only')), \
             patch('sys.stdout', new_callable=io.StringIO) as out:
            M.main()
        self.assertIn('"socket_server_worker_neural_training_VM_calls": 0', out.getvalue())

    def test_fresh_receipt_rejects_existing_or_outside_scratch(self):
        with self.assertRaises(ValueError): M.fresh_receipt(ROOT / 'app.py')
        with self.assertRaises(ValueError): M.fresh_receipt(ROOT / 'scratch/../app.py')

    def test_no_torch_import_during_source_only_suite(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__': unittest.main(verbosity=2)

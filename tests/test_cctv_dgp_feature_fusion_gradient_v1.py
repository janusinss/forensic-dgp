"""Adversarial metadata/return boundaries and diagnostic scope; no models."""
import ast
import importlib.util
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


worker = module('fusion_worker_test', 'scripts/cctv_dgp_feature_fusion_gradient_v1_vm.py')
boundary = module('fusion_return_test', 'scripts/cctv_dgp_feature_fusion_gradient_v1_return_audit_template.py')


class DiagnosticRegressions(unittest.TestCase):
    def case(self, ref, profile, source='s1'):
        return {'id': ref + '_' + profile, 'source_person_or_reference': ref, 'profile': profile, 'source': source}

    def member(self, name='outputs/results.json', kind=tarfile.REGTYPE, size=10):
        member = tarfile.TarInfo(boundary.PREFIX + name)
        member.type = kind; member.size = size
        return member

    def test_repeated_reference_profile_and_source_match_excludes_touched(self):
        exposed = [self.case('old1', 'blur'), self.case('old1', 'clear'), self.case('old2', 'blur', 's2')]
        pool = [self.case('used', 'blur'), self.case('used', 'clear'), self.case('new1', 'blur'),
                self.case('new1', 'clear'), self.case('wrong_source', 'blur'), self.case('new2', 'blur', 's2')]
        selected, pairs = worker.matched_unexposed_cases(exposed, pool, {'old1', 'old2', 'used'})
        self.assertEqual([c['id'] for c in selected], ['new1_blur', 'new1_clear', 'new2_blur'])
        self.assertEqual(pairs, {'old1': 'new1', 'old2': 'new2'})

    def test_insufficient_distinct_reference_pool_stops(self):
        with self.assertRaisesRegex(AssertionError, 'Insufficient unexposed TRAIN references'):
            worker.matched_unexposed_cases([self.case('old1', 'blur'), self.case('old2', 'clear')],
                                           [self.case('new1', 'blur'), self.case('new1', 'clear')], {'old1', 'old2'})

    def test_regular_archive_member_is_allowed(self):
        values, size = boundary.safe_members([self.member()], {'outputs/results.json'})
        self.assertEqual((len(values), size), (1, 10))

    def test_traversal_links_and_case_duplicate_are_rejected(self):
        for name in ['../escape', '/escape', 'C:/escape', 'x\\escape']:
            with self.assertRaises(AssertionError):
                boundary.safe_members([self.member(name)], {name})
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            with self.assertRaises(AssertionError):
                boundary.safe_members([self.member(kind=kind)], {'outputs/results.json'})
        with self.assertRaises(AssertionError):
            boundary.safe_members([self.member(), self.member('outputs/RESULTS.json')],
                                  {'outputs/results.json', 'outputs/RESULTS.json'})

    def test_new_checkpoint_and_oversize_array_are_rejected(self):
        for member in [self.member('outputs/new.pth'), self.member(size=64 * 1024**2 + 1)]:
            with self.assertRaises(AssertionError):
                boundary.safe_members([member], {'outputs/results.json'})

    def test_worker_contains_no_optimizer_backward_or_checkpoint_writer(self):
        tree = ast.parse((ROOT / 'scripts/cctv_dgp_feature_fusion_gradient_v1_vm.py').read_text(), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ['backward', 'step', 'AdamW', 'Adam', 'SGD'])
                if node.func.attr == 'save' and isinstance(node.func.value, ast.Name):
                    self.assertNotEqual(node.func.value.id, 'torch')

    def test_return_checker_contains_no_local_differentiation(self):
        tree = ast.parse((ROOT / 'scripts/cctv_dgp_feature_fusion_gradient_v1_return_audit_template.py').read_text(), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ['grad', 'backward', 'step', 'AdamW', 'Adam', 'SGD'])


if __name__ == '__main__':
    unittest.main()

"""Receipt-roundoff and zero-update return boundaries; no model work."""
import ast
import importlib.util
import math
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


repair = module('receipt_r1', 'scripts/audit_cctv_dgp_broader_mean_v30_return_r1.py')
boundary = module('sampling_boundary', 'scripts/cctv_dgp_v30_sampling_gradient_v1_return_audit_template.py')
worker = module('sampling_worker', 'scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py')


class MetadataMatching(unittest.TestCase):
    def case(self, ref, profile, source='source1'):
        return {'id': ref + '_' + profile, 'source_person_or_reference': ref, 'source': source, 'profile': profile}

    def test_reference_repetition_and_exclusion_preserve_cohort_comparability(self):
        exposed = [self.case('old1', 'blur'), self.case('old2', 'blur'), self.case('old1', 'clear'), self.case('old3', 'motion', 'source2')]
        pool = [self.case('touched', 'blur'), self.case('touched', 'clear'), self.case('new1', 'blur'),
                self.case('new1', 'clear'), self.case('new2', 'blur'), self.case('new3', 'motion', 'source2')]
        selected, pairs = worker.matched_unexposed_cases(exposed, pool, {'old1', 'old2', 'old3', 'touched'})
        self.assertEqual([c['id'] for c in selected], ['new1_blur', 'new2_blur', 'new1_clear', 'new3_motion'])
        self.assertEqual(pairs, {'old1': 'new1', 'old2': 'new2', 'old3': 'new3'})
        self.assertEqual([(c['source'], c['profile']) for c in selected], [(c['source'], c['profile']) for c in exposed])

    def test_insufficient_distinct_reference_pool_is_rejected(self):
        exposed = [self.case('old1', 'blur'), self.case('old2', 'clear')]
        pool = [self.case('new1', 'blur'), self.case('new1', 'clear')]
        with self.assertRaisesRegex(AssertionError, 'Insufficient unexposed TRAIN references'):
            worker.matched_unexposed_cases(exposed, pool, {'old1', 'old2'})


class ReceiptRoundoff(unittest.TestCase):
    def setUp(self):
        self.gain = .008057174231146158
        self.receipt = {'update': 50, 'minimum': .01, 'relative_feature_error_gain': self.gain, 'pass': False}

    def test_reported_failure_survives_one_ulp_roundoff(self):
        actual = math.nextafter(self.gain, 1)
        self.assertNotEqual(actual, self.gain)
        repair.verify_early_receipt(self.receipt, actual)
        self.assertFalse(self.receipt['pass'])

    def test_threshold_change_is_rejected(self):
        with self.assertRaises(AssertionError): repair.verify_early_receipt({**self.receipt, 'minimum': .009}, self.gain)

    def test_falsified_gate_pass_is_rejected(self):
        with self.assertRaises(AssertionError): repair.verify_early_receipt({**self.receipt, 'pass': True}, self.gain)

    def test_material_or_nonfinite_discrepancy_is_rejected(self):
        for value in [self.gain + 1e-6, float('nan'), float('inf')]:
            with self.assertRaises(AssertionError): repair.verify_early_receipt(self.receipt, value)


class DiagnosticBoundary(unittest.TestCase):
    def member(self, name='outputs/results.json', kind=tarfile.REGTYPE, size=10):
        member = tarfile.TarInfo(boundary.PREFIX + name); member.type = kind; member.size = size
        return member

    def test_regular_bound_member(self):
        rows, size = boundary.safe_members([self.member()], {'outputs/results.json'})
        self.assertEqual((len(rows), size), (1, 10))

    def test_escape_paths_links_and_duplicate_names_are_rejected(self):
        for name in ['../escape', '/escape', 'C:/escape', 'x\\escape']:
            with self.assertRaises(AssertionError): boundary.safe_members([self.member(name)], {name})
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            with self.assertRaises(AssertionError): boundary.safe_members([self.member(kind=kind)], {'outputs/results.json'})
        with self.assertRaises(AssertionError): boundary.safe_members([self.member(), self.member('outputs/RESULTS.json')], {'outputs/results.json', 'outputs/RESULTS.json'})

    def test_unexpected_checkpoint_and_oversize_array_are_rejected(self):
        with self.assertRaises(AssertionError): boundary.safe_members([self.member('outputs/new.pth')], {'outputs/results.json'})
        with self.assertRaises(AssertionError): boundary.safe_members([self.member(size=64 * 1024**2 + 1)], {'outputs/results.json'})

    def test_worker_has_no_optimizer_step_backward_or_checkpoint_writer(self):
        tree = ast.parse((ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py').read_text(), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ['backward', 'step', 'AdamW', 'Adam', 'SGD'])
                if node.func.attr == 'save' and isinstance(node.func.value, ast.Name): self.assertNotEqual(node.func.value.id, 'torch')

    def test_CPU_checker_has_no_local_derivative_or_optimizer_call(self):
        tree = ast.parse((ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_return_audit_template.py').read_text(), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute): self.assertNotIn(node.func.attr, ['grad', 'backward', 'step', 'AdamW', 'Adam', 'SGD'])


if __name__ == '__main__': unittest.main()

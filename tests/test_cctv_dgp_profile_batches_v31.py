"""Batch anchoring and return-boundary regressions; no model or gradient calls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_profile_batches_v31_schedule import PROFILES, make_profile_batches, validate_profile_batches
from prepare_cctv_dgp_profile_batches_v31 import return_auditor_template


def auditor():
    module = types.ModuleType('trusted_V31_return_template_test')
    module.__file__ = str(ROOT / 'scripts/cctv_dgp_profile_batches_v31_return_audit_template.py')
    exec(compile(return_auditor_template(), module.__file__, 'exec'), module.__dict__)
    return module


class ProfileBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30/protocol.json').read_text())
        cls.old = json.loads((ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30/schedule.json').read_text())['batches']
        cls.batches, cls.refs = make_profile_batches(cls.p['case_rows'], cls.old, 781, 800)
        cls.audit = auditor()

    def test_full_epoch_and_every_clear_control_paired(self):
        self.assertEqual(sorted(i for b in self.batches[:781] for i in b), list(range(3905)))
        self.assertEqual(len({i for b in self.batches[781:] for i in b}), 95)
        self.assertEqual(len(set(self.refs[:50])), 50)
        for batch in self.batches:
            rows = [self.p['case_rows'][i] for i in batch]
            self.assertEqual([r['profile'] for r in rows], PROFILES)
            self.assertEqual(len({r['source_person_or_reference'] for r in rows}), 1)

    def test_reference_order_uses_frozen_metadata_only(self):
        def order(batches):
            refs = [self.p['case_rows'][i]['source_person_or_reference'] for b in batches for i in b]
            return list(dict.fromkeys(refs))
        self.assertEqual(self.refs[:781], order(self.old[:781]))
        self.assertEqual(self.refs[781:], order(self.old[781:])[:19])
        self.assertEqual(make_profile_batches(self.p['case_rows'], self.old, 781, 800)[0], self.batches)

    def test_mixed_references_rejected(self):
        batch = self.batches[0].copy(); batch[1] = self.batches[1][1]
        with self.assertRaises(AssertionError): validate_profile_batches(self.p['case_rows'], [batch])

    def test_evaluation_and_incomplete_profiles_rejected(self):
        cases = copy.deepcopy(self.p['case_rows']); cases[self.batches[0][0]]['role'] = 'val'
        with self.assertRaises(AssertionError): validate_profile_batches(cases, [self.batches[0]])
        cases = copy.deepcopy(self.p['case_rows']); cases[0]['profile'] = cases[1]['profile']
        with self.assertRaises(AssertionError): make_profile_batches(cases, self.old, 781, 800)

    def test_case_index_not_bool_and_in_range(self):
        for index in [True, -1, 3905, 0.5]:
            batch = self.batches[0].copy(); batch[0] = index
            with self.assertRaises(AssertionError): validate_profile_batches(self.p['case_rows'], [batch])

    def member(self, name='protocol.json', size=10):
        value = tarfile.TarInfo(self.audit.PREFIX + name); value.size = size; return value

    def test_regular_allowlisted_archive_accepted(self):
        rows, total = self.audit.validate_members([self.member()], {'protocol.json'})
        self.assertEqual(total, 10); self.assertEqual(rows[0][1], 'protocol.json')

    def test_traversal_link_and_windows_collision_rejected(self):
        malicious = self.member(); malicious.type = tarfile.SYMTYPE; malicious.linkname = '../elsewhere'
        for members, allowed in [([self.member('../protocol.json')], {'../protocol.json'}),
                                 ([malicious], {'protocol.json'}),
                                 ([self.member(), self.member('PROTOCOL.JSON')], {'protocol.json', 'PROTOCOL.JSON'}),
                                 ([self.member('a\\b.json')], {'a\\b.json'})]:
            with self.assertRaises(AssertionError): self.audit.validate_members(members, allowed)

    def test_unknown_checkpoint_and_size_caps_rejected(self):
        for members, allowed, cap in [([self.member('outputs/unknown.pth')], {'protocol.json'}, 100),
                                     ([self.member(size=64 * 1024**2 + 1)], {'protocol.json'}, 3 * 1024**3),
                                     ([self.member(size=11)], {'protocol.json'}, 10)]:
            with self.assertRaises(AssertionError): self.audit.validate_members(members, allowed, cap)

    def receipt(self, gain, passed=None):
        return {'update': 50, 'minimum': .01, 'relative_feature_error_gain': gain,
                'pass': gain >= .01 if passed is None else passed}

    def test_roundoff_receipt_does_not_change_failed_gate(self):
        gain = .008057174230956
        self.audit.verify_early_receipt(self.receipt(gain + 1.11e-16), gain)
        with self.assertRaises(AssertionError): self.audit.verify_early_receipt(self.receipt(gain, True), gain)

    def test_threshold_crossing_material_and_nonfinite_receipts_rejected(self):
        for receipt, gain in [(self.receipt(.01 + 1e-15), .01 - 1e-15),
                              (self.receipt(.0081), .008),
                              (self.receipt(float('nan')), .008),
                              ({**self.receipt(.008), 'minimum': .005}, .008),
                              ({**self.receipt(.008), 'update': 49}, .008)]:
            with self.assertRaises(AssertionError): self.audit.verify_early_receipt(receipt, gain)


if __name__ == '__main__': unittest.main(verbosity=2)

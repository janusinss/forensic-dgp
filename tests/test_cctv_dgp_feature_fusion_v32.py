"""Partition, gradient schema and unchanged gate/return-boundary regressions."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_feature_fusion_v32_policy import validate_layout
from prepare_cctv_dgp_feature_fusion_v32 import return_auditor_template


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm/protocol.json').read_text())
        cls.audit = types.ModuleType('local_prospective_V32_return_test')
        cls.audit.__file__ = str(ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_return.py')
        exec(compile(return_auditor_template(), cls.audit.__file__, 'exec'), cls.audit.__dict__)
        cls.h = cls.audit.helpers()

    def test_measured_partition_accepts_exact23_names_and_sizes(self):
        self.assertTrue(validate_layout(self.p['parameter_layout']))

    def test_backbone_or_inactive_head_cannot_replace_measured_tensor(self):
        for forbidden in ['head4.block0.weight', 'fpn.features.0.0.weight', 'fpn.enc0.0.0.weight']:
            layout = copy.deepcopy(self.p['parameter_layout']); layout[0]['name'] = forbidden
            with self.assertRaises(AssertionError): validate_layout(layout)

    def test_missing_duplicate_or_reordered_tensors_rejected(self):
        layout = self.p['parameter_layout']
        duplicate = copy.deepcopy(layout); duplicate[1]['name'] = duplicate[0]['name']
        reordered = copy.deepcopy(layout); reordered[0], reordered[1] = reordered[1], reordered[0]
        for bad in [layout[:-1], duplicate, reordered]:
            with self.assertRaises(AssertionError): validate_layout(bad)

    def test_bad_shapes_offsets_and_boolean_dimensions_rejected(self):
        for key, value in [('shape', [True]), ('start', True), ('end', -1), ('shape', [0]), ('shape', [1])]:
            layout = copy.deepcopy(self.p['parameter_layout']); layout[0][key] = value
            with self.assertRaises(AssertionError): validate_layout(layout)

    def test_new_gradient_schema_accepts_measured23_and_rejects_old12(self):
        accepted = np.broadcast_to(np.float64(0), (7, 978243))
        self.assertIs(self.h.matrix(accepted), accepted)
        for bad in [np.broadcast_to(np.float64(0), (7, 498627)),
                    np.broadcast_to(np.float32(0), (7, 978243)),
                    np.broadcast_to(np.float64(np.nan), (7, 978243))]:
            with self.assertRaises(AssertionError): self.h.matrix(bad)

    def member(self, name='protocol.json', size=10):
        member = tarfile.TarInfo(self.audit.PREFIX + name); member.size = size
        return member

    def test_regular_allowlisted_archive_accepted(self):
        self.assertEqual(self.audit.validate_members([self.member()], {'protocol.json'})[1], 10)

    def test_traversal_symlink_collision_and_unknown_weight_rejected(self):
        link = self.member(); link.type = tarfile.SYMTYPE; link.linkname = '../elsewhere'
        for members, allowed in [([self.member('../protocol.json')], {'../protocol.json'}),
                                 ([link], {'protocol.json'}),
                                 ([self.member(), self.member('PROTOCOL.JSON')], {'protocol.json', 'PROTOCOL.JSON'}),
                                 ([self.member('outputs/unapproved.pth')], {'protocol.json'})]:
            with self.assertRaises(AssertionError): self.audit.validate_members(members, allowed)

    def test_file_and_aggregate_size_caps_rejected(self):
        with self.assertRaises(AssertionError): self.audit.validate_members([self.member(size=64 * 1024**2 + 1)], {'protocol.json'})
        with self.assertRaises(AssertionError): self.audit.validate_members([self.member(size=11)], {'protocol.json'}, cap=10)

    def test_roundoff_does_not_waive_failed_one_percent_gate(self):
        gain = .006945252687208803
        good = {'update': 50, 'minimum': .01, 'relative_feature_error_gain': gain + 1e-16, 'pass': False}
        self.audit.verify_early_receipt(good, gain)
        for bad in [{**good, 'pass': True}, {**good, 'minimum': .005},
                    {**good, 'relative_feature_error_gain': .01}, {**good, 'relative_feature_error_gain': float('nan')}]:
            with self.assertRaises(AssertionError): self.audit.verify_early_receipt(bad, gain)

    def test_threshold_crossing_rejected_even_inside_receipt_tolerance(self):
        bad = {'update': 50, 'minimum': .01, 'relative_feature_error_gain': .01 + 1e-15, 'pass': True}
        with self.assertRaises(AssertionError): self.audit.verify_early_receipt(bad, .01 - 1e-15)

    def test_actual_V31_stopped_groups_still_fail_final_capacity(self):
        folder = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return/outputs'
        groups = [json.loads((folder / f'update{s}/metrics.json').read_text())['groups'] for s in [0, 50]]
        failures, gain, sources, brightness, passed = self.h.capacity(*groups)
        self.assertFalse(passed); self.assertLess(gain, .01); self.assertEqual(len(groups[0]), 17)
        self.assertEqual(failures, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)

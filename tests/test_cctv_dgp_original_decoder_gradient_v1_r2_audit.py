"""Malformed scientific return regressions; synthetic arrays are not VM results."""
import importlib.util
from pathlib import Path
import tarfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('decoder_return_audit',ROOT/'scripts/audit_cctv_dgp_original_decoder_gradient_v1_r2_return.py')
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)
PREFIX='cctv_dgp_original_decoder_gradient_v1_r2_return/'


def member(name='protocol.json',kind=tarfile.REGTYPE,size=2):
    value=tarfile.TarInfo(PREFIX+name);value.type=kind;value.size=size
    return value


class MalformedReturnTests(unittest.TestCase):
    def test_link_is_rejected(self):
        with self.assertRaises(AssertionError):AUDIT.validate_members([member(kind=tarfile.SYMTYPE)],{'protocol.json'})

    def test_traversal_is_rejected(self):
        with self.assertRaises(AssertionError):AUDIT.validate_members([member('../protocol.json')],{'../protocol.json'})

    def test_duplicate_and_case_collisions_are_rejected(self):
        for name in ['protocol.json','Protocol.json']:
            with self.subTest(name=name),self.assertRaises(AssertionError):
                AUDIT.validate_members([member(),member(name)],{'protocol.json','Protocol.json'})

    def test_archive_and_member_bounds_are_rejected(self):
        with self.assertRaises(AssertionError):AUDIT.validate_members([member(size=5)],{'protocol.json'},total_cap=4)
        with self.assertRaises(AssertionError):AUDIT.validate_members([member(size=65*1024**2)],{'protocol.json'})

    def test_unexpected_checkpoint_is_rejected(self):
        with self.assertRaises(AssertionError):AUDIT.validate_members([member('model.pth')],{'protocol.json'})

    def test_nonfinite_gradient_is_rejected(self):
        value=np.zeros((7,4),dtype=np.float64);value[0,1]=np.nan
        with self.assertRaises(AssertionError):AUDIT.matrix(value,(7,4))

    def test_gradient_dtype_and_layout_are_checked(self):
        for value in [np.zeros((7,4),dtype=np.float32),np.zeros((7,3),dtype=np.float64)]:
            with self.assertRaises(AssertionError):AUDIT.matrix(value,(7,4))

    def test_full_parameter_partition_arithmetic_detects_a_changed_value(self):
        value=np.array([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=np.float64)
        layout=[{'name':'one','start':0,'end':2},{'name':'two','start':2,'end':4}]
        norms,gram,cosine,blocks=AUDIT.gradient_statistics(value,layout)
        np.testing.assert_array_equal(norms,[1,2,3,0,0,0,0]);np.testing.assert_array_equal(gram,np.diag([1,4,9,0,0,0,0]))
        self.assertAlmostEqual(blocks['one']['improvement_gradient_norm'],5**.5)
        self.assertEqual(blocks['two']['improvement_gradient_norm'],3)
        value[0,0]+=1
        changed=AUDIT.gradient_statistics(value,layout)
        self.assertFalse(np.array_equal(norms,changed[0]));self.assertNotEqual(blocks,changed[3])


if __name__=='__main__':unittest.main()

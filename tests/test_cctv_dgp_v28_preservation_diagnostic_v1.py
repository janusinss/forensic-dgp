"""Meaningful prospective malformed-return and no-training regressions."""
import ast
from pathlib import Path
import sys
import tarfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_cctv_dgp_v28_preservation_diagnostic_v1_return as a


class ReturnBoundary(unittest.TestCase):
    def member(self,name,size=3,type=tarfile.REGTYPE):
        m=tarfile.TarInfo(a.PREFIX+name);m.size=size;m.type=type;return m

    def test_regular_member_accepted(self):
        rows,total=a.members_checked([self.member('protocol.json')],{'protocol.json'})
        self.assertEqual((len(rows),total),(1,3))

    def test_traversal_is_rejected(self):
        with self.assertRaises(AssertionError):a.members_checked([self.member('../protocol.json')],{'../protocol.json'})

    def test_symlink_is_rejected(self):
        with self.assertRaises(AssertionError):a.members_checked([self.member('protocol.json',type=tarfile.SYMTYPE)],{'protocol.json'})

    def test_case_collision_is_rejected(self):
        with self.assertRaises(AssertionError):a.members_checked([self.member('protocol.json'),self.member('PROTOCOL.json')],{'protocol.json','PROTOCOL.json'})

    def test_wrong_or_unknown_checkpoint_is_rejected(self):
        with self.assertRaises(AssertionError):a.members_checked([self.member('new_checkpoint.pth')],{'protocol.json'})

    def test_oversized_member_is_rejected(self):
        with self.assertRaises(AssertionError):a.members_checked([self.member('protocol.json',size=64*1024**2+1)],{'protocol.json'})

    def test_old_seven_component_matrix_is_rejected(self):
        with self.assertRaises(AssertionError):a.matrix(np.zeros((7,498627),dtype=np.float64))

    def test_nonfinite_gradient_is_rejected(self):
        x=np.zeros((10,498627),dtype=np.float64);x[4,1]=np.nan
        with self.assertRaises(AssertionError):a.matrix(x)

    def test_worker_has_no_optimizer_or_checkpoint_creation(self):
        tree=ast.parse((ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_vm/scripts/cctv_dgp_v28_preservation_diagnostic_v1_vm.py').read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ['AdamW','Adam','SGD','backward','step','zero_grad'] for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='save' and isinstance(n.value,ast.Name) and n.value.id=='torch' for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n,ast.Name) and n.id in ['optimizer','train'] for n in ast.walk(tree)))


if __name__=='__main__':unittest.main()

"""Reject unsafe packets and invalid saved gradient evidence without gradients."""
import copy
import io
from pathlib import Path
import sys
import tarfile
import unittest
import uuid

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import verify_cctv_dgp_v25_gradient_diagnostic_v1 as check


class DiagnosticGuards(unittest.TestCase):
    def setUp(self):
        self.folder=ROOT/'scratch'/('v25_gradient_guard_'+uuid.uuid4().hex);self.folder.mkdir(parents=True)
    def archive(self,extra=None,omit=None):
        path=self.folder/'fixture.tar.gz'
        with tarfile.open(path,'x:gz') as tar:
            for name in ['protocol.json','protocol.sha256','scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py','scripts/run_gradient.sh']:
                if name==omit:continue
                info=tarfile.TarInfo(check.BUNDLE.name+'/'+name);info.size=1;tar.addfile(info,io.BytesIO(b'x'))
            if extra:
                tar.addfile(extra,io.BytesIO(b'x') if extra.isreg() else None)
        return path
    def test_parent_traversal_archive_rejected(self):
        extra=tarfile.TarInfo(check.BUNDLE.name+'/../outside');extra.size=1
        with self.assertRaisesRegex(AssertionError,'Unsafe'):check.inspect_archive(self.archive(extra))
    def test_link_archive_rejected(self):
        extra=tarfile.TarInfo(check.BUNDLE.name+'/linked');extra.type=tarfile.SYMTYPE;extra.linkname='/tmp/elsewhere'
        with self.assertRaisesRegex(AssertionError,'regular'):check.inspect_archive(self.archive(extra))
    def test_duplicate_archive_rejected(self):
        extra=tarfile.TarInfo(check.BUNDLE.name+'/protocol.json');extra.size=1
        with self.assertRaisesRegex(AssertionError,'duplicate'):check.inspect_archive(self.archive(extra))
    def test_missing_member_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Missing'):check.inspect_archive(self.archive(omit='protocol.sha256'))
    def test_optimizer_step_source_rejected(self):
        source=(check.BUNDLE/'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py').read_text()
        with self.assertRaisesRegex(AssertionError,'optimizer'):check.source_contract(source+'\ndef hidden_update():\n    optimizer.step()\n')
    def test_nonfinite_gradient_rejected(self):
        matrix=np.zeros((7,53781),np.float64);matrix[0,0]=float('nan')
        with self.assertRaisesRegex(AssertionError,'Finite'):check.gradient_arithmetic(matrix,check.expected_layout())
    def test_parameter_names_cannot_be_swapped(self):
        layout=check.expected_layout();layout[0]['name']='other.weight'
        with self.assertRaisesRegex(AssertionError,'names'):check.gradient_arithmetic(np.zeros((7,53781),np.float64),layout)
    def test_float32_gradient_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'float64'):check.gradient_arithmetic(np.zeros((7,53781),np.float32),check.expected_layout())
    def test_zero_norm_and_opposed_components_are_reported(self):
        matrix=np.zeros((7,53781),np.float64);matrix[0,0]=2;matrix[1,0]=-2
        result=check.gradient_arithmetic(matrix,check.expected_layout())
        self.assertEqual(result['cosines'][0][1],-1);self.assertEqual(result['norms'][2],0)
        self.assertEqual(result['total_norm'],0)


if __name__=='__main__':unittest.main()

"""Meaningful return-boundary regressions; no models, gradients or training."""
import importlib.util
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v30_return',ROOT/'scripts/audit_cctv_dgp_broader_mean_v30_return.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class ReturnBoundary(unittest.TestCase):
    def member(self,name='outputs/results.json',size=20,kind=tarfile.REGTYPE):
        item=tarfile.TarInfo(audit.PREFIX+name);item.size=size;item.type=kind
        return item

    def test_valid_regular_member(self):
        rows,total=audit.validate_members([self.member()],{'outputs/results.json'})
        self.assertEqual((len(rows),total),(1,20))

    def test_traversal_is_rejected_even_if_allowlisted(self):
        with self.assertRaises(AssertionError): audit.validate_members([self.member('../escape')],{'../escape'})

    def test_symlink_and_hardlink_are_rejected(self):
        for kind in [tarfile.SYMTYPE,tarfile.LNKTYPE]:
            with self.assertRaises(AssertionError):audit.validate_members([self.member(kind=kind)],{'outputs/results.json'})

    def test_case_collision_is_rejected_on_windows(self):
        with self.assertRaises(AssertionError):audit.validate_members([self.member(),self.member('outputs/RESULTS.json')],{'outputs/results.json','outputs/RESULTS.json'})

    def test_unexpected_file_is_rejected(self):
        with self.assertRaises(AssertionError):audit.validate_members([self.member('run_me.py')],{'outputs/results.json'})

    def test_member_and_total_size_limits_are_enforced(self):
        with self.assertRaises(AssertionError):audit.validate_members([self.member(size=64*1024**2+1)],{'outputs/results.json'})
        with self.assertRaises(AssertionError):audit.validate_members([self.member(size=21)],{'outputs/results.json'},cap=20)

    def test_preservation_failure_cannot_be_overridden_by_structure_gain(self):
        h=audit.helpers()
        b={k:{'cases':1,'MSE':1.,'SSIM':.8,'ArcFace_observed_fixed':.8,
              'landmark_high_frequency_MSE':1.,'constant_mean_shift_only_MSE':1.}
           for k in ['all','clear','degraded','source/degraded']}
        a={k:{**v,'MSE':.9,'landmark_high_frequency_MSE':.5} for k,v in b.items()}
        a['clear']['ArcFace_observed_fixed']=.7
        failures,gain,_,_,accepted=h.capacity(b,a)
        self.assertEqual(gain,.5);self.assertTrue(failures);self.assertFalse(accepted)


if __name__=='__main__':unittest.main()

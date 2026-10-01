import importlib.util
import math
import unittest


class PackageContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.package_reflection_coverage_vm'),
                             'Reflection experiment packager missing')

    def test_rejects_prior_member_overwrite_and_nonportable_paths(self):
        from scripts.package_reflection_coverage_vm import validate_members
        for members in (['old.py'],['../new.py'],['C:/new.py'],['new.py','new.py']):
            with self.assertRaises(ValueError):validate_members(members,[{'old.py':'x'}])
        validate_members(['new.py'],[{'old.py':'x'}])

    def test_requires_measured_unchanged_source_forward_and_zero_optimizer_updates(self):
        from scripts.package_reflection_coverage_vm import verify_source_check
        good={'model_sha256':'model','optimizer_sha256':'moments','data_sha256':'data',
              'model_state_unchanged':True,'optimizer_constructed':False,'optimizer_updates_locally':0,
              'terms':{'core':1.,'control':1.,'reflective':1.,'padded_probe':1.},'moment_step':420}
        verify_source_check(good,'model','moments','data')
        for key,value in [('model_state_unchanged',False),('optimizer_constructed',True),
                          ('optimizer_updates_locally',1),('moment_step',0),('data_sha256','changed')]:
            bad={**good,key:value}
            with self.assertRaises(ValueError):verify_source_check(bad,'model','moments','data')
        with self.assertRaises(ValueError):verify_source_check({**good,'terms':{'core':float('nan')}},'model','moments','data')


if __name__=='__main__':unittest.main()

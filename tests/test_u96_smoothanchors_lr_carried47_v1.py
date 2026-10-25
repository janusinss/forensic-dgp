"""Pure carry-order fixtures plus existing independent pixel fixtures."""
import ast
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_u96_smoothanchors_lr_carried47_v1 as C
sys.path.insert(0,str(ROOT/'tests'))
from test_u96_smoothanchors_lr_fixed_DEV_saved_v1 import DeliveryFixtures, MetricFixtures, ParserAndClassificationFixtures


class CarryPrefixFixtures(unittest.TestCase):
    def test_exact47is23pairs_plus_case24incumbent(self):
        plan={'cases':[{'id':f'case{i:02d}'}for i in range(68)]}
        paths=[Path('cases')/row['id']/arm/'record.json'for row in plan['cases']for arm in C.A.ARMS][:47]
        prefix=C.exact_prefix(plan,paths)
        self.assertEqual(prefix[-1],('case23','incumbent_U96'))
        self.assertEqual(len(prefix),47)
        for changed in [paths[:-1],paths+[paths[-1]],paths[:-1]+[Path('cases/case24/incumbent_U96/record.json')],paths[:-1]+[Path('cases/case23/postfusion_LR_u64/record.json')]]:
            with self.assertRaises(ValueError):C.exact_prefix(plan,changed)

    def test_held_pixelhelper_and_no_modelimports(self):
        self.assertEqual(C.A.pin(Path(C.A.__file__))['sha256'],C.HELPER_SHA)
        ast.parse((ROOT/'scripts/audit_u96_smoothanchors_lr_carried47_v1.py').read_text(),feature_version=(3,10))
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()

"""Independent evidence rejects changed state and deployable-oracle claims."""
import copy
import unittest
import numpy as np
from scripts.audit_frozen_complementarity import check_execution, independent_counts, independent_systems


class FrozenComplementarityAuditTests(unittest.TestCase):
    def fixture(self):
        protocol = {'counts': {'real':83,'replay':610,'reflection':280}, 'new_forward_images':1593,
                    'reused_predictions':353, 'threshold':.5}
        report = {'complete':True,'new_forward_images':1593,'reused_predictions':353,
                  'held_out_forward_images':0,'optimizer_constructed':False,'optimizer_updates_locally':0,
                  'oracles_deployable':False,'training_recipe_ready':False,'promoted':False,
                  'execution':{arm:{'forward_images':count,'state_before_sha256':'a'*64,
                      'state_after_sha256':'a'*64,'model_state_unchanged':True,'threshold':.5,
                      'optimizer_constructed':False,'optimizer_updates_locally':0}
                      for arm,count in [('parent',973),('candidate',620)]}}
        return report, protocol

    def test_readiness_or_changed_state_cannot_hide_behind_complete_flags(self):
        report, protocol = self.fixture(); check_execution(report,protocol)
        for field,value in [('promoted',True),('oracles_deployable',True),('training_recipe_ready',True),
                            ('held_out_forward_images',1),('reused_predictions',354)]:
            altered=copy.deepcopy(report); altered[field]=value
            with self.assertRaises(ValueError):
                check_execution(altered,protocol)
        for field,value in [('state_after_sha256','b'*64),('forward_images',972),('threshold',.4)]:
            altered=copy.deepcopy(report); altered['execution']['parent'][field]=value
            with self.assertRaises(ValueError):
                check_execution(altered,protocol)

    def test_pixel_recount_excludes_unknown_and_rejects_positive_unknown_target(self):
        target=np.zeros((4,4),bool); target[0,:2]=True; valid=np.ones_like(target); valid[3]=False
        p=target.copy(); p[2,2]=True; p[3]=True
        r=independent_counts(p,target,valid)
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['visible'],r['ignored_positive_pixels']),(2,1,0,10,4))
        target[3,0]=True
        with self.assertRaises(ValueError):
            independent_counts(p,target,valid)

    def test_bounds_preserve_shared_errors_and_do_not_trade_recall_for_fp(self):
        target=np.zeros((4,4),bool); target[0,:3]=True; valid=np.ones_like(target)
        parent=np.zeros_like(target); parent[0,0]=True; parent[2,2]=True
        candidate=parent.copy(); candidate[0,1]=True; candidate[2,3]=True
        systems,dominates=independent_systems(parent,candidate,target,valid,'real')
        self.assertFalse(dominates); self.assertTrue(np.array_equal(systems['dominance_oracle'],parent))
        r=independent_counts(systems['pixel_oracle'],target,valid)
        self.assertEqual((r['tp'],r['fp'],r['fn']),(2,1,1))


if __name__=='__main__':
    unittest.main()

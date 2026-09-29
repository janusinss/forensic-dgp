import unittest
from scripts.audit_feature_mixed_vm import summarize, parity_metrics
import torch

class AuditTests(unittest.TestCase):
    def test_presence_error_denominators(self):
        rows=[{'area':.1,'probability':.49},{'area':.2,'probability':.5},
              {'area':0.,'probability':.8},{'area':0.,'probability':.2}]
        self.assertEqual(summarize(rows),{'cases':4,'covered':2,'clear':2,'missed_covered':1,'false_positive_clear':1})

    def test_parity_uses_mask_disagreement_and_probability_delta(self):
        a=torch.tensor([[[[0.,1.]]]])
        b=torch.tensor([[[[0.,2.]]]])
        result=parity_metrics(a,b,{'probability':.49,'mask':torch.tensor([False,True])},
                              {'probability':.51,'mask':torch.tensor([True,True])})
        self.assertAlmostEqual(result['embedding_max_abs'],1.)
        self.assertAlmostEqual(result['mask_disagreement_fraction'],.5)
        self.assertTrue(result['presence_decision_changed'])

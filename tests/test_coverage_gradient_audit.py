import unittest
import torch
from coverage_gradient_audit import objective_terms, cosine


class GradientDecompositionTests(unittest.TestCase):
    def test_weighted_value_and_logit_gradient_match(self):
        torch.manual_seed(42)
        x=torch.randn(8,1,8,8,dtype=torch.float64,requires_grad=True)
        mask=torch.zeros_like(x);mask[:2,:,2:6,2:6]=1;mask[4:6,:,1:5,1:5]=1
        synthetic=torch.tensor([False]*4+[True]*4)
        added=torch.tensor([False,True,False,False]+[False]*4)
        terms,full=objective_terms(x,mask,synthetic,added,torch.randn_like(x[synthetic]))
        self.assertTrue(torch.allclose(sum(terms.values()),full,atol=1e-12,rtol=0))
        a=torch.autograd.grad(sum(terms.values()),x,retain_graph=True)[0]
        b=torch.autograd.grad(full,x)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=0))

    def test_cosine_handles_zero_and_opposition(self):
        self.assertIsNone(cosine(torch.zeros(2),torch.ones(2)))
        self.assertAlmostEqual(cosine(torch.tensor([1.,0]),torch.tensor([-1.,0])),-1.)

import unittest
import torch
from coverage_projection import project_real_against_replay, step_alignment


class CoverageProjectionTests(unittest.TestCase):
    def test_conflict_removed_and_reference_unchanged(self):
        real=torch.tensor([-2.,3.]);replay=torch.tensor([1.,0.])
        original=real.clone();reference=replay.clone()
        result,info=project_real_against_replay(real,replay)
        self.assertTrue(torch.equal(result,torch.tensor([1.,3.])))
        self.assertTrue(info['applied'])
        self.assertEqual(info['real_replay_dot_after'],0.)
        self.assertTrue(torch.equal(real,original));self.assertTrue(torch.equal(replay,reference))

    def test_aligned_orthogonal_zero_and_opposite_cases(self):
        for real,replay,expected,applied in [
            ([2.,3.],[1.,0.],[3.,3.],False),
            ([0.,3.],[1.,0.],[1.,3.],False),
            ([2.,3.],[0.,0.],[2.,3.],False),
            ([-2.,0.],[1.,0.],[1.,0.],True)]:
            result,info=project_real_against_replay(torch.tensor(real),torch.tensor(replay))
            self.assertTrue(torch.equal(result,torch.tensor(expected)))
            self.assertEqual(info['applied'],applied)

    def test_float32_projection_has_nonnegative_reference_alignment_with_tolerance(self):
        generator=torch.Generator().manual_seed(42)
        for _ in range(20):
            b=torch.randn(101,generator=generator)
            a=torch.randn(101,generator=generator)-3*b
            result,info=project_real_against_replay(a,b)
            self.assertGreaterEqual(float(result.double()@b.double()),-1e-6)
            self.assertLess(abs(info['real_replay_dot_after']),1e-10)

    def test_actual_step_can_oppose_projected_gradient(self):
        before=torch.tensor([0.,0.]);after=torch.tensor([.1,0.]);reference=torch.tensor([1.,0.])
        audit=step_alignment(before,after,reference)
        self.assertGreater(audit['replay_first_order_loss_change'],0)
        self.assertAlmostEqual(audit['replay_descent_cosine'],-1.)
        self.assertIsNone(step_alignment(before,before,reference)['replay_descent_cosine'])

    def test_nonfinite_mismatch_and_empty_rejected(self):
        for a,b in [(torch.tensor([float('nan')]),torch.ones(1)),
                    (torch.ones(2),torch.ones(3)),(torch.empty(0),torch.empty(0)),
                    (torch.ones(2,dtype=torch.int64),torch.ones(2,dtype=torch.int64)),
                    (torch.ones(1),torch.ones(1,dtype=torch.float64))]:
            with self.assertRaises(ValueError):project_real_against_replay(a,b)

"""Prospective objective invariants with constants only; no model or backward."""
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from cctv_dgp_degraded_objective_v24 import assemble_terms


class DegradedObjectiveContracts(unittest.TestCase):
    def terms(self, clear=False, **changes):
        t=lambda v:torch.tensor([v],dtype=torch.float64)
        values=dict(feature=t(.004),interior=t(.008),pixel=t(.02),base_pixel=t(.02),score=t(.8),base_score=t(.8),
                    cosine=t(.9),base_cosine=t(.9),baseline_anchor=t(0),degraded_weight=t(0 if clear else 1.25),
                    clear_weight=t(1 if clear else 0),normalizers=(t(.004),t(.008)))
        values.update({k:t(v) for k,v in changes.items()})
        return assemble_terms(**values)

    def test_clear_target_improvements_have_no_reward(self):
        base=sum(self.terms(clear=True).values()).item()
        improved=sum(self.terms(clear=True,feature=.001,interior=.002,pixel=.01,score=.9,cosine=.99).values()).item()
        self.assertEqual(base,0);self.assertEqual(improved,0)

    def test_clear_visible_change_has_baseline_preservation_cost(self):
        self.assertAlmostEqual(self.terms(clear=True,baseline_anchor=.002)['clear_baseline_anchor'].item(),.005)

    def test_clear_and_degraded_preservation_penalties_are_same(self):
        for clear in [False,True]:
            v=self.terms(clear=clear,pixel=.022,score=.79,cosine=.87)
            self.assertAlmostEqual(v['pixel_regression'].item(),.2)
            self.assertAlmostEqual(v['SSIM_regression'].item(),.05)
            self.assertAlmostEqual(v['ArcFace_regression'].item(),.15)

    def test_global_cohort_term_tracks_absolute_mean_not_per_case_ratios(self):
        # Unequal baselines .1/1.0: per-case ratios prefer A(.5,.95) to B(1,.8).
        # Absolute cohort error correctly prefers B(total.9) to A(total1.0).
        t=lambda v:torch.tensor(v,dtype=torch.float64)
        def cohort(feature):
            return assemble_terms(t(feature),t([.008,.008]),t([.02,.02]),t([.02,.02]),t([.8,.8]),t([.8,.8]),
                t([.9,.9]),t([.9,.9]),t([0,0]),t([1.25,1.25]),t([0,0]),(t(.55),t(.008)))['degraded_landmark_detail'].mean().item()
        self.assertLess(cohort([.10,.80]),cohort([.05,.95]))
        self.assertAlmostEqual(cohort([.10,.80])/cohort([.05,.95]),.9)

    def test_equal50_baseline_normalization_keeps_scale(self):
        # 10 clear controls have zero reward; 40 degraded have1.25*(1+.25+.05).
        clear=sum(self.terms(clear=True).values()).item();degraded=sum(self.terms().values()).item()
        self.assertAlmostEqual((10*clear+40*degraded)/50,1.3)

    def test_inconsistent_flags_rejected(self):
        with self.assertRaises(AssertionError):self.terms(clear=True,degraded_weight=1.25)

    def test_nonbinary_flags_rejected(self):
        with self.assertRaises(AssertionError):self.terms(clear=True,clear_weight=.5)

    def test_pixel_floor_is_retained(self):
        self.assertAlmostEqual(self.terms(base_pixel=0,pixel=1e-5)['pixel_regression'].item(),2)


if __name__=='__main__':unittest.main()

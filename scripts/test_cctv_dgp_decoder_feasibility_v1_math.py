"""Synthetic-only feasibility arithmetic fixtures; never load research gradients."""
from fractions import Fraction
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import cctv_dgp_decoder_feasibility_v1_math as m


def photo_fixture():
    baseline=np.zeros((5,3,2,2),np.float32)
    baseline[:,1]=1.;baseline[:,2]=.5
    target=np.ones_like(baseline);target[:,1]=0.;target[:,2]=.25
    return baseline,target,np.ones((5,1,2,2),np.float32)


def simple_fixture():
    gradients=np.zeros((2,35,3),np.float32)
    gradients[:,:19,0]=1.
    gradients[:,19:34,1]=1.
    gradients[:,34,0]=1.
    photo=[[[[Fraction(0),Fraction(0)] for _ in range(3)] for _ in range(5)] for _ in range(2)]
    return gradients,np.ones((2,4)),photo


class ExactArithmetic(unittest.TestCase):
    def test_exact_sum_cancellation_extremes_and_large_bins(self):
        values=np.r_[np.full(1003,.1),np.full(1003,-.1),np.nextafter(0.,1.),1e200,-1e200]
        self.assertEqual(m.exact_sum(values),sum(map(Fraction.from_float,map(float,values)),Fraction(0)))
        self.assertEqual(m.exact_sum([]),0)
        with self.assertRaises(ValueError):m.exact_sum([math.inf])

    def test_exact_endpoint_dot_includes_subnormal_difference(self):
        tiny=np.nextafter(np.float32(0),np.float32(1))
        gradient=np.asarray([1.,tiny,-.1],np.float32)
        initial=np.asarray([tiny,1.,-.25],np.float32)
        placed=np.asarray([1.,tiny,.125],np.float32)
        expected=sum((Fraction.from_float(float(g))*(Fraction.from_float(float(p))-Fraction.from_float(float(i)))
                      for g,i,p in zip(gradient,initial,placed)),Fraction(0))
        self.assertEqual(m.exact_endpoint_dot(gradient,initial,placed),expected)
        exact_first=m.exact_endpoint_dot(gradient[:1],initial[:1],placed[:1])
        rounded_first=Fraction.from_float(float(placed.astype(np.float64)[0]-initial.astype(np.float64)[0]))
        self.assertNotEqual(exact_first,rounded_first)

    def test_exact_endpoint_radius_does_not_trust_rounded_delta(self):
        tiny=np.nextafter(np.float32(0),np.float32(1))
        initial=np.asarray([tiny,1.],np.float32);placed=np.asarray([1.,tiny],np.float32)
        proof=m.exact_endpoint_radius(initial,placed,2.)
        value=proof['exact_squared_norm']
        self.assertEqual(Fraction(int(value['numerator']),int(value['denominator'])),2*(1-Fraction.from_float(float(tiny)))**2)
        self.assertTrue(proof['exact_radius_pass'])
        self.assertFalse(m.exact_endpoint_radius(initial,initial,2.)['exact_radius_pass'])

    def test_dyadic_radius_rejects_one_ULP_over_boundary(self):
        self.assertTrue(m.exact_radius_proof(np.asarray([.25],np.float64),.25)['exact_radius_pass'])
        with self.assertRaises(ValueError):m.exact_radius_proof(np.asarray([math.nextafter(.25,math.inf)],np.float64),.25)
        with self.assertRaises(ValueError):m.exact_radius_proof(np.zeros(3,np.float64),.25)


class ClipAndLP(unittest.TestCase):
    def test_exact_clip_coefficients_and_all8_branches(self):
        coeff=m.exact_photo_coefficients(*photo_fixture())
        self.assertEqual(coeff[0],[[Fraction(-2,3),Fraction(0)],[Fraction(0),Fraction(2,3)],[Fraction(1,6),Fraction(1,6)]])
        branches=m.branch_coefficients(coeff)
        for signs in ((-1,-1,-1),(-1,1,1),(1,-1,1),(1,1,-1),(0,0,0)):
            shifts=list(map(Fraction,signs))
            chosen=sum(coeff[0][c][0 if shifts[c]>=0 else 1]*shifts[c] for c in range(3))
            self.assertEqual(chosen,min(sum(row[c]*shifts[c] for c in range(3)) for row in branches[0]))

    def test_brightness_epigraph_requires_nonpositive_actual_slope(self):
        coeff=m.exact_photo_coefficients(*photo_fixture());branches=m.branch_coefficients(coeff)[0]
        for shifts in ((1,-1,1),(-1,1,-1),(0,0,0)):
            shifts=list(map(Fraction,shifts));mean=min(sum(row[c]*shifts[c] for c in range(3)) for row in branches)
            for actual in (Fraction(-10),Fraction(-1,10),Fraction(0)):
                nonlinear=max(Fraction(0),-mean)+Fraction.from_float(.2)*actual<=0
                linear=all(-sum(row[c]*shifts[c] for c in range(3))+Fraction.from_float(.2)*actual<=0 for row in branches)
                self.assertEqual(nonlinear,linear)

    def test_build_has_all_profile_and_branch_constraints(self):
        gradients,base,photo=simple_fixture();lp=m.build_lp(gradients,base,photo,1e-6)
        self.assertEqual(lp['A_ub'].shape,(102,71));self.assertEqual(len(lp['bounds']),71)
        self.assertEqual(len([s for s in lp['receipt']['constraint_names'] if '/brightness_' in s]),64)
        self.assertEqual(len([s for s in lp['receipt']['constraint_names'] if '/structure_' in s]),8)
        self.assertFalse(lp['receipt']['finite_scientific_slack_consumed'])
        self.assertTrue(lp['receipt']['coefficient_box_is_only_homogeneous_ray_normalizer'])
        self.assertFalse(lp['receipt']['exact_L2_radius_replaced_by_box'])

    def test_single_toy_LP_finds_ray_but_does_not_certify(self):
        gradients,base,photo=simple_fixture();lp=m.build_lp(gradients,base,photo,1e-6)
        options=dict(time_limit=5.,maxiter=1000,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,presolve=True)
        candidate,receipt=m.solve_lp(lp,options)
        self.assertIsNotNone(candidate);self.assertGreater(candidate[-1],0)
        self.assertEqual(receipt['solver_calls'],1);self.assertFalse(receipt['solver_success_is_certificate'])
        self.assertFalse(receipt['full_decoder_infeasibility_certified'])

    def test_nonfinite_rejected_and_zero_basis_rows_retained(self):
        gradients,base,photo=simple_fixture()
        changed=gradients.copy();changed[1,15]=math.nan
        with self.assertRaises(ValueError):m.build_lp(changed,base,photo,1e-6)
        changed=gradients.copy();changed[1,15]=0.
        lp=m.build_lp(changed,base,photo,1e-6)
        self.assertEqual(lp['receipt']['basis_zero_indices'],[50])
        self.assertFalse(lp['receipt']['effective_basis_rank_claimed'])

    def test_allzero_gradients_report_no_ray_without_infeasibility_claim(self):
        gradients,base,photo=simple_fixture();gradients[:]=0.
        lp=m.build_lp(gradients,base,photo,1e-6)
        options=dict(time_limit=5.,maxiter=1000,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,presolve=True)
        candidate,receipt=m.solve_lp(lp,options)
        self.assertIsNone(candidate);self.assertFalse(receipt['full_decoder_infeasibility_certified'])


class ActualStateCertificate(unittest.TestCase):
    def test_exact_all30_preservation_all8_structure_and_photo_pass(self):
        gradients,base,photo=simple_fixture();initial=np.zeros(3,np.float32);placed=np.asarray([-.02,0,0],np.float32)
        proof=m.certify_placement(initial,placed,gradients,base,photo,.04,.01)
        self.assertTrue(proof['admitted'])
        self.assertEqual(sum(len(r['preservation_pass']) for r in proof['reference_certificates']),30)
        self.assertEqual(sum(len(r['structure_pass']) for r in proof['reference_certificates']),8)
        self.assertFalse(proof['neural_branch_or_finite_loss_certified']);self.assertFalse(proof['PNG_quality_certified'])

    def test_tiny_positive_MSE_rejected_despite_solver_tolerance_or_finite_slack(self):
        gradients,base,photo=simple_fixture();gradients[:,:15]=[0,1,0]
        placed=np.asarray([-.02,1e-13,0],np.float32)
        self.assertLess(float(placed[1]),1e-12)
        proof=m.certify_placement(np.zeros(3,np.float32),placed,gradients,base,photo,.04,.01)
        self.assertFalse(proof['admitted'])
        self.assertFalse(proof['reference_certificates'][0]['preservation_pass'][0])

    def test_each_reference_and_each_structure_term_must_pass(self):
        gradients,base,photo=simple_fixture();initial=np.zeros(3,np.float32);placed=np.asarray([-.02,0,0],np.float32)
        for ref,scalar in ((0,15),(1,18),(1,9)):
            changed=gradients.copy();changed[ref,scalar]*=-1
            with self.subTest(ref=ref,scalar=scalar):
                self.assertFalse(m.certify_placement(initial,placed,changed,base,photo,.04,.01)['admitted'])

    def test_sign_correct_clip_photo_rejects_hidden_mean_only_gain(self):
        gradients,base,photo=simple_fixture();coeff=m.exact_photo_coefficients(*photo_fixture());photo=[coeff,coeff]
        gradients[:,19:34]=0
        # Clear and every degraded channel0 mean slope is positive.  The
        # baseline0/target1 channel gains from brightness beyond20%ofactual.
        gradients[:,19:34:3,0]=-1
        placed=np.asarray([-.02,0,0],np.float32)
        proof=m.certify_placement(np.zeros(3,np.float32),placed,gradients,base,photo,.04,.01)
        self.assertFalse(proof['admitted'])
        self.assertTrue(any(int(row['numerator'])>0 for row in proof['reference_certificates'][0]['brightness_residuals']))

    def test_full_one_percent_and_partial_strict_positive_are_separate(self):
        gradients,base,photo=simple_fixture();initial=np.zeros(3,np.float32);placed=np.asarray([-.005,0,0],np.float32)
        self.assertTrue(m.certify_placement(initial,placed,gradients,base,photo,.04,0.)['admitted'])
        self.assertFalse(m.certify_placement(initial,placed,gradients,base,photo,.04,.01)['admitted'])
        self.assertFalse(m.certify_placement(initial,initial,gradients,base,photo,.04,0.)['admitted'])

    def test_manual_guard_precedes_array_access_or_solver(self):
        fake=types.ModuleType('cctv_dgp_decoder_feasibility_v1_contract')
        def rejected(root):raise RuntimeError('manual guard rejected before arrays')
        fake.require_manual=rejected
        class Forbidden:
            def __array__(self,*args,**kwargs):raise AssertionError('Array touched before guard')
        with patch.dict(sys.modules,{'cctv_dgp_decoder_feasibility_v1_contract':fake}):
            with self.assertRaisesRegex(RuntimeError,'manual guard rejected'):
                m.select_direction(Forbidden(),Forbidden(),None,None,None,None,None)

    def test_source_has_no_neural_optimizer_or_data_loading(self):
        tree=ast.parse(Path(m.__file__).read_text(encoding='utf-8'),feature_version=(3,10))
        imports=[n.name for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom)) for n in node.names]
        self.assertFalse(any(name.startswith(('torch','tensorflow','cvxpy')) for name in imports))
        calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertFalse(any(s.endswith(('.load','.backward','.step','.train')) for s in calls))

    def test_failure_preservation_callbacks_precede_exact_certificate(self):
        tree=ast.parse(Path(m.__file__).read_text(encoding='utf-8'),feature_version=(3,10))
        select=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='select_direction')
        calls={}
        for node in ast.walk(select):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                calls.setdefault(node.func.id,[]).append(node.lineno)
        self.assertEqual(len(calls['solve_lp']),1)
        self.assertLess(calls['require_manual'][0],calls['solve_lp'][0])
        self.assertLess(calls['solve_lp'][0],calls['on_solver'][0])
        self.assertLess(calls['on_solver'][0],calls['on_proposal'][0])
        self.assertLess(calls['on_proposal'][0],calls['certify_placement'][0])
        solver_stop=next(node for node in ast.walk(select) if isinstance(node,ast.If)
                         and "solver['seconds'] > p['budgets']['solver_seconds']"==ast.unparse(node.test))
        self.assertLess(calls['on_solver'][0],solver_stop.lineno)
        self.assertLess(solver_stop.lineno,calls['on_proposal'][0])
        self.assertTrue(any(isinstance(node,ast.Raise) for node in ast.walk(solver_stop)))
        self.assertEqual(m.LABELS,('quarter','half','full'))
        self.assertEqual(m.AMPLITUDES,(.25,.5,1.))


def main():
    start=time.monotonic();log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    root=Path(__file__).resolve().parents[1];out=root/'outputs/cctv_dgp_decoder_feasibility_v1_preparation/math_tests'
    out.mkdir(parents=True,exist_ok=True);number=1
    while (out/f'run_{number:03d}.json').exists():number+=1
    content=log.getvalue();evidence=dict(complete=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-start,
        sources_sha256={str(Path(path).relative_to(root)).replace('\\','/'):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (__file__,m.__file__)},
        log_sha256=hashlib.sha256(content.encode('utf-8')).hexdigest(),actual_saved_gradients_read=False,actual_parameter_proposals=0,
        model_calls=0,gradient_queries=0,optimizer_updates=0,VM_connections=0,scientific_gates_changed=False)
    for suffix,value in (('txt',content),('json',json.dumps(evidence,indent=2)+'\n')):
        with (out/f'run_{number:03d}.{suffix}').open('x',encoding='utf-8',newline='\n') as stream:stream.write(value)
    print(content,end='');print(json.dumps(evidence));return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())

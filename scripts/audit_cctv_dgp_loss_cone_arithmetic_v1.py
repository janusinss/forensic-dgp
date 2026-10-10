"""Independent primal-SLSQP readback of saved-gradient projections; no model."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from cctv_dgp_loss_cone_v33 import project_vectors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v32_restoration_cone_v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def reference(gradient, proposal):
    norms = np.sqrt(np.square(gradient).sum(1))
    unit = gradient[norms > 0] / norms[norms > 0, None]
    if not len(unit): return proposal.copy()
    h = unit @ unit.T; b = unit @ proposal
    # Primal correction coefficients are unrestricted; unlike the active-set
    # module this solver enforces the halfspaces through a general SQP routine.
    result = minimize(lambda z: .5 * float(z @ h @ z), np.zeros(len(unit)),
                      jac=lambda z: h @ z,
                      constraints=[{'type': 'ineq', 'fun': lambda z: b + h @ z,
                                    'jac': lambda z: h}], method='SLSQP',
                      options={'maxiter': 500, 'ftol': 1e-14})
    assert result.success and np.min(b + h @ result.x) >= -1e-8
    return proposal + result.x @ unit


def main():
    started = time.monotonic()
    plan = read(OUT / 'plan.json'); result = read(OUT / 'analysis.json')
    assert result['complete'] and result['plan_sha256'] == sha(OUT / 'plan.json')
    for name, digest in plan['source_sha256'].items(): assert sha(ROOT / name) == digest, name
    previous = read(ROOT / 'outputs/cctv_dgp_v32_loss_directions_v1/analysis.json')
    maximum, checked = 0., 0
    for row, old in zip(result['rows'], previous['rows']):
        assert (row['state'],row['cohort'],row['matrix']) == (old['state'],old['cohort'],old['matrix'])
        file = ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state{row["state"]}_{row["cohort"]}/{row["matrix"]}'
        g = np.load(file, allow_pickle=False).astype(np.float64)
        for saved, proposal, key in [(row, g[:3].sum(0), 'projected_restoration_derivatives'),
                                      (old, g.sum(0), 'projected_directional_derivatives')]:
            d = reference(g, proposal)
            actual, proof = project_vectors(g, proposal)
            error = float(np.linalg.norm(d - actual)); maximum = max(maximum,error)
            assert error <= 2e-6 * max(np.linalg.norm(proposal),1)
            assert np.allclose(-(g @ actual), saved[key], atol=1e-10, rtol=1e-10)
            assert np.allclose(proof['multipliers'],saved['projection']['multipliers'],atol=1e-10,rtol=1e-10)
            checked += 1
        assert time.monotonic()-started < 180
    fixtures = [
        (np.array([[1.,0.],[0.,1.]]),np.array([-3.,4.]),np.array([0.,4.])),
        (np.array([[1.,0.],[-1.,0.]]),np.array([3.,4.]),np.array([0.,4.])),
        (np.array([[1.,0.],[1.,0.],[0.,0.]]),np.array([-3.,4.]),np.array([0.,4.])),
        (np.zeros((7,2)),np.array([3.,4.]),np.array([3.,4.])),
        (np.array([[1.,0.],[0.,1.]]),np.zeros(2),np.zeros(2))]
    for g,p,expected in fixtures:
        value,_=project_vectors(g,p);np.testing.assert_allclose(value,expected,rtol=0,atol=1e-10)
    rejected=0
    for g,p in [(np.ones((8,2)),np.ones(2)),(np.array([[np.nan,1.]]),np.ones(2)),
                (np.ones((2,2)),np.ones(3))]:
        try:project_vectors(g,p)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid projection input accepted')
    rng=np.random.RandomState(240)
    for _ in range(24):
        g=rng.normal(size=(7,12));p=rng.normal(size=12)
        actual,_=project_vectors(g,p);expected=reference(g,p)
        assert np.linalg.norm(actual-expected)<2e-6
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),
             'plan_sha256':sha(OUT/'plan.json'),'analysis_sha256':sha(OUT/'analysis.json'),
             'previous_analysis_sha256':sha(ROOT/'outputs/cctv_dgp_v32_loss_directions_v1/analysis.json'),
             'all44_saved_matrices_both_methods_verified':checked==88,
             'independent_primal_SLSQP_checks':checked,'maximum_direction_L2_difference':maximum,
             'analytic_boundary_fixtures':len(fixtures),'invalid_inputs_rejected':rejected,
             'random_independent_primal_checks':24,'neural_calls':0,'gradient_queries':0,
             'parameter_changes':0,'optimizer_updates':0,'finite_neural_improvement_proven':False,
             'seconds':time.monotonic()-started,'goal_complete':False}
    with (OUT/'independent_arithmetic_audit.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

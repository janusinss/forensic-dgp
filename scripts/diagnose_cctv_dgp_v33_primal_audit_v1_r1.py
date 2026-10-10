"""Compare independent primal coordinates using saved arrays only; no parameter assignment."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize
from cctv_dgp_loss_cone_v33 import project_vectors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation/independent_solver_diagnostic_v1_r1'


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False, default=lambda v: v.item()) + '\n')


def main():
    assert not OUT.exists(); OUT.mkdir(); start = time.monotonic()
    p = json.loads((ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_vm/protocol.json').read_text())
    paths = [Path(__file__), ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return.py',
             ROOT / 'scripts/verify_cctv_dgp_loss_cone_probe_v33_packet.py', ROOT / 'scripts/cctv_dgp_loss_cone_v33.py']
    write(OUT / 'plan.json', {'complete': True, 'source_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
          'previous_record_failure': 'First wrapper rejected a NumPy Boolean during JSON serialization after computing the first pair. Its original plan is retained.',
          'previous_tool_chunk': '38f89e', 'cases': 8, 'cap_seconds': 180, 'neural_calls': 0, 'optimizer_updates': 0,
          'methods': ['raw_Gram_coefficients_ftol1e-13', 'orthonormal_primal_ftol1e-15'], 'projection_error_bound_unchanged': True})
    weights = [torch.load(ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27/weights/dgp_v2.pth', map_location='cpu', weights_only=True),
               torch.load(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return/outputs/update50/dgp_candidate_v32.pth', map_location='cpu', weights_only=True)]
    rows = []
    for state, w in zip(p['states'], weights):
        theta = np.concatenate([w[r['name']].numpy().reshape(-1) for r in p['parameter_layout']]).astype(np.float64)
        for cohort in p['cohorts']:
            g = np.load(ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state{state}_{cohort["name"]}/gradient_components.npy', allow_pickle=False)
            for label, h in [('summed', g.sum(0)), ('restoration', g[:3].sum(0))]:
                assert time.monotonic() - start < 180
                c = h / max(1., np.linalg.norm(h) + 1e-6)
                proposal = .00003 * .01 * theta + .00003 * c / (np.abs(c) + 1e-8)
                direction, _ = project_vectors(g, proposal)
                norms = np.linalg.norm(g, axis=1); unit = g[norms > 0] / norms[norms > 0, None]
                gram = unit @ unit.T; scale = np.linalg.norm(proposal); b = (unit @ proposal) / scale
                eigenvalues, rotation = np.linalg.eigh(gram); nz = eigenvalues > 1e-12
                matrix = rotation[:, nz] * np.sqrt(eigenvalues[nz]); back = rotation[:, nz] / np.sqrt(eigenvalues[nz])
                results = []
                for method in ['raw_Gram_coefficients_ftol1e-13', 'orthonormal_primal_ftol1e-15']:
                    raw = method.startswith('raw')
                    fun = (lambda v: .5 * float(v @ gram @ v)) if raw else (lambda v: .5 * float(v @ v))
                    jac = (lambda v: gram @ v) if raw else (lambda v: v)
                    constraints = gram if raw else matrix
                    fit = minimize(fun, np.zeros(constraints.shape[1]), jac=jac,
                                   constraints=[{'type': 'ineq', 'fun': lambda v: b + constraints @ v, 'jac': lambda v: constraints}],
                                   method='SLSQP', options={'ftol': 1e-13 if raw else 1e-15, 'maxiter': 1000})
                    coefficient = fit.x if raw else back @ fit.x
                    fresh = proposal + (coefficient * scale) @ unit
                    error = float(np.linalg.norm(fresh - direction)); bound = float(max(2e-7 * scale, 2e-10))
                    results.append({'method': method, 'success': bool(fit.success), 'message': fit.message, 'iterations': int(fit.nit),
                                    'direction_L2_error': error, 'unchanged_bound': bound, 'passes_unchanged_bound': bool(error <= bound),
                                    'primal_minimum': float((b + constraints @ fit.x).min()), 'objective': float(fit.fun)})
                row = {'state': state, 'cohort': cohort['name'], 'proposal': label, 'Gram_eigenvalues': eigenvalues.tolist(), 'methods': results}
                write(OUT / f'state{state}_{cohort["name"]}_{label}.json', row); rows.append(row)
                print(json.dumps({'state': state, 'cohort': cohort['name'], 'proposal': label, 'errors': [r['direction_L2_error'] for r in results], 'passes': [r['passes_unchanged_bound'] for r in results]}), flush=True)
    write(OUT / 'analysis.json', {'complete': True, 'rows': rows, 'seconds': time.monotonic() - start, 'neural_calls': 0, 'optimizer_updates': 0, 'parameter_assignments': 0})


if __name__ == '__main__': main()

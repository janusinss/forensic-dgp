"""Compare a restoration-priority cone using existing saved gradients only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cctv_dgp_loss_cone_v33 import project_vectors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v32_restoration_cone_v1'
OLD = ROOT / 'outputs/cctv_dgp_v32_loss_directions_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(); OUT.mkdir(); started = time.monotonic()
    old, old_plan = read(OLD / 'analysis.json'), read(OLD / 'plan.json')
    assert old['complete'] and old['matrices_checked'] == 44
    sources = dict(old_plan['source_sha256'])
    for name, digest in sources.items(): assert sha(ROOT / name) == digest, name
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in
                    [Path(__file__), OLD / 'analysis.json', OLD / 'plan.json']})
    write(OUT / 'plan.json', {'frozen_UTC': datetime.now(timezone.utc).isoformat(),
           'source_sha256': sources, 'matrices': 44, 'cap_seconds': 180,
           'purpose': 'The full-sum projection can sit at a zero detail derivative. Test projection of the original three restoration gradients instead.',
           'all_seven_nonzero_gradients_constrain_the_proposal': True,
           'proposed_update_formation_changes': True, 'loss_values_or_quality_gates_changed': False,
           'no_actual_AdamW_or_finite_parameter_step': True, 'neural_calls': 0,
           'gradient_queries': 0, 'optimizer_updates': 0, 'goal_complete': False})
    rows = []
    for before in old['rows']:
        assert time.monotonic() - started < 180
        state, cohort, name = before['state'], before['cohort'], before['matrix']
        path = ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state{state}_{cohort}/{name}'
        g = np.load(path, allow_pickle=False).astype(np.float64)
        assert g.shape == (7, 978243)
        proposal = g[:3].sum(0)
        direction, proof = project_vectors(g, proposal)
        derivative = -(g @ direction)
        bound = proof['KKT_tolerance'] * np.maximum(np.linalg.norm(g, axis=1), 1)
        assert np.all(derivative <= bound)
        rows.append({'state': state, 'cohort': cohort, 'matrix': name, 'kind': before['kind'],
                     'restoration_proposal_derivatives': (-(g @ proposal)).tolist(),
                     'projected_restoration_derivatives': derivative.tolist(), 'projection': proof})
    aggregates = [r for r in rows if r['kind'] == 'aggregate50']
    result = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'rows': rows,
              'all44_matrices_projected': True, 'all_nonzero_component_halfspaces_verified': True,
              'aggregates': aggregates, 'seconds': time.monotonic() - started,
              'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
              'finite_model_output_improvement_proven': False, 'AdamW_history_reconstructed': False,
              'full_TRAIN_or_DEV_capacity_proven': False, 'goal_complete': False}
    write(OUT / 'analysis.json', result)
    print(json.dumps({'complete': True, 'seconds': result['seconds'], 'aggregates':
          [{k:r[k] for k in ['state','cohort','projected_restoration_derivatives']} for r in aggregates]}, indent=2))


if __name__ == '__main__': main()

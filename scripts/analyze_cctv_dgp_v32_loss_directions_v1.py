"""Saved-gradient arithmetic only; no model, differentiation or parameter update."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cctv_dgp_loss_cone_v33 import project_vectors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v32_loss_directions_v1'
RETURN = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(); OUT.mkdir()
    started = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['diagnostic_complete'] and audit['optimizer_updates'] == 0
    protocol = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm/protocol.json'
    p = read(protocol)
    sources = [Path(__file__), ROOT / 'scripts/cctv_dgp_loss_cone_v33.py', audit_path, protocol]
    matrices = []
    for state in [0, 50]:
        for cohort in ['exposed', 'unexposed']:
            folder = RETURN / f'outputs/state{state}_{cohort}'
            sources.append(folder / 'receipt.json')
            for name in ['gradient_components.npy'] + [f'batch{i}.npy' for i in range(10)]:
                sources.append(folder / name)
                matrices.append((state, cohort, name, folder / name))
    plan = {'frozen_UTC': datetime.now(timezone.utc).isoformat(), 'scope': 'All44 saved matrices, no neural calls',
            'source_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sources},
            'methods': ['original weighted sum', 'original weighted sum projected into all nonzero loss halfspaces'],
            'gradients_recomputed': False, 'optimizer_history_reconstructed': False,
            'cap_seconds': 180, 'matrices': 44, 'local_training': False, 'VM_writes': 0,
            'finite_parameter_changes': 0, 'goal_complete': False}
    write(OUT / 'plan.json', plan)
    rows = []
    for state, cohort, name, path in matrices:
        assert time.monotonic() - started < 180
        gradient = np.load(path, allow_pickle=False).astype(np.float64)
        assert gradient.shape == (7, 978243) and np.isfinite(gradient).all()
        proposed = gradient.sum(0)
        corrected, receipt = project_vectors(gradient, proposed)
        before = -(gradient @ proposed)
        after = -(gradient @ corrected)
        bound = receipt['KKT_tolerance'] * np.maximum(np.linalg.norm(gradient, axis=1), 1)
        assert np.all(after <= bound), 'No relaxed component constraint'
        rows.append({'state': state, 'cohort': cohort, 'matrix': name,
                     'kind': 'aggregate50' if name == 'gradient_components.npy' else 'batch5',
                     'original_directional_derivatives': before.tolist(),
                     'projected_directional_derivatives': after.tolist(),
                     'original_restoration_terms_increasing': int((before[:3] > 1e-12).sum()),
                     'projected_restoration_terms_increasing': int((after[:3] > bound[:3]).sum()),
                     'projection': receipt})
    summary = []
    for state in [0, 50]:
        for cohort in ['exposed', 'unexposed']:
            batches = [r for r in rows if r['state'] == state and r['cohort'] == cohort and r['kind'] == 'batch5']
            aggregate = next(r for r in rows if r['state'] == state and r['cohort'] == cohort and r['kind'] == 'aggregate50')
            summary.append({'state': state, 'cohort': cohort,
                            'original_batches_increasing_landmark_term': sum(r['original_directional_derivatives'][0] > 1e-12 for r in batches),
                            'original_batches_increasing_any_restoration_term': sum(r['original_restoration_terms_increasing'] > 0 for r in batches),
                            'projected_batches_increasing_any_term': sum(any(v > 1e-9 for v in r['projected_directional_derivatives']) for r in batches),
                            'original_aggregate_derivatives': aggregate['original_directional_derivatives'],
                            'projected_aggregate_derivatives': aggregate['projected_directional_derivatives'],
                            'aggregate_projected_norm': aggregate['projection']['projected_norm']})
    receipt = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'rows': rows, 'summary': summary,
               'seconds': time.monotonic() - started, 'matrices_checked': len(rows),
               'model_calls': 0, 'gradient_queries': 0, 'parameter_changes': 0, 'optimizer_updates': 0,
               'AdamW_direction_or_history_reconstructed': False,
               'curvature_or_finite_step_proven': False, 'quality_qualification': False,
               'VM_pilot_selected_from_arithmetic_alone': False, 'goal_complete': False}
    assert receipt['seconds'] < 180
    write(OUT / 'analysis.json', receipt)
    print(json.dumps({'complete': True, 'seconds': receipt['seconds'], 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()

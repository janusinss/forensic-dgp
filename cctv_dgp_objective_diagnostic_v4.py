"""Training-cohort and numeric helpers for a VM-only, zero-update diagnostic."""
import math
from pathlib import Path

import torch

SOURCES = ('dataset/asian_faces', 'dataset/thumbnails128x128')
PROFILES = ('clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24')
TERM_WEIGHTS = {'pixel': 1., 'color': .05, 'vgg': .1, 'sobel': .05, 'identity': .1}
ROOT = Path(__file__).resolve().parent
V3_RETURN_SHA = 'cdf603f35a62398c48c11f967da993792c8fad5cb79c42db7e61bf74363068d2'
V3_AUDIT_SHA = '2010cb66ea1571972367775bb1dbb1100e8d4c595e253ecb2fe9c1416ce461b1'
TEACHERS = {'identity': '9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963',
            'perceptual': '26ced9c88642e3972aeb71815cff4eb78291630473425432f52bb410f3f0ab12'}


def training_groups(protocol):
    refs = {r['id']: r for r in protocol['references']}
    groups = []
    for source in SOURCES:
        for profile in PROFILES:
            cases = [c for c in protocol['training_epochs']['1']
                     if c['source'] == source and c['profile'] == profile][:4]
            if len(cases) != 4 or any(refs[c['reference_id']]['role'] != 'train'
                                    or refs[c['reference_id']]['source'] != source for c in cases):
                raise ValueError('Four original training cases per source/profile are required')
            groups.append({'id': source + '/' + profile, 'source': source, 'profile': profile, 'cases': cases})
    if len({c['reference_id'] for g in groups for c in g['cases']}) != 40:
        raise ValueError('Training diagnostic references must be unique')
    return groups


def gradient_summary(vectors):
    if not vectors or any(not isinstance(v, torch.Tensor) or v.ndim != 1 or not v.numel()
                          or not v.is_floating_point() or not torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError('Finite nonempty flat gradient vectors are required')
    if len({v.shape for v in vectors.values()}) != 1:
        raise ValueError('Gradient vector shapes differ')
    with torch.no_grad():
        rows = torch.stack([v.detach().to('cpu', torch.float64) for v in vectors.values()])
        gram = rows @ rows.T
    return summary_from_gram(list(vectors), rows.shape[1], gram.tolist())


def summary_from_gram(names, dimension, matrix):
    if not names or len(set(names)) != len(names) or type(dimension) is not int or dimension < 1:
        raise ValueError('Invalid gradient Gram metadata')
    if len(matrix) != len(names) or any(len(row) != len(names) for row in matrix):
        raise ValueError('Gradient Gram dimensions differ')
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in matrix for v in row):
        raise ValueError('Nonfinite gradient Gram')
    norms = {}
    for i, name in enumerate(names):
        if matrix[i][i] < 0 or any(abs(matrix[i][j] - matrix[j][i]) > 1e-9 * max(1., abs(matrix[i][j])) for j in range(len(names))):
            raise ValueError('Invalid gradient Gram symmetry/diagonal')
        norms[name] = math.sqrt(matrix[i][i])
    cosines = {}
    for i, first in enumerate(names):
        for j in range(i + 1, len(names)):
            second = names[j]
            scale = norms[first] * norms[second]
            if not scale and matrix[i][j] != 0:
                raise ValueError('Zero gradient cannot have a nonzero cross product')
            value = matrix[i][j] / scale if scale else None
            if value is not None and abs(value) > 1 + 1e-9:
                raise ValueError('Invalid gradient cosine')
            cosines[first + '/' + second] = min(1., max(-1., value)) if value is not None else None
    total_square = sum(v for row in matrix for v in row)
    if total_square < -1e-9 * max(1., sum(matrix[i][i] for i in range(len(names)))):
        raise ValueError('Invalid total gradient norm')
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(torch.tensor(matrix, dtype=torch.float64))
    if float(eigenvalues.min()) < -1e-8 * max(1., max(matrix[i][i] for i in range(len(names)))):
        raise ValueError('Gradient Gram is not positive semidefinite')
    total = math.sqrt(max(0., total_square))
    denominator = sum(norms.values())
    return {'term_order': names, 'dimension': dimension, 'gram_matrix': matrix,
            'terms': {k: {'l2_norm': v} for k, v in norms.items()}, 'pairwise_cosines': cosines,
            'total_l2_norm': total, 'cancellation_ratio': total / denominator if denominator else None}


def verify_recipe(root, bundle_dir, perceptual_bundle_dir=None, parent_return=None, v3_return=None):
    from cctv_dgp_pilot import read, sha
    from cctv_dgp_perceptual_training_v3 import configure_paths, verify_protocol_v3
    root, bundle_dir = Path(root).resolve(), Path(bundle_dir).resolve()
    configure_paths(perceptual_bundle_dir or ROOT, parent_return)
    protocol = verify_protocol_v3(root)
    manifest_path = bundle_dir / 'objective_diagnostic_protocol_v4.json'
    if (bundle_dir / 'objective_diagnostic_protocol_v4.sha256').read_text(encoding='ascii').strip() != sha(manifest_path):
        raise ValueError('Diagnostic protocol fingerprint differs')
    manifest = read(manifest_path)
    if (manifest['format'] != 'cctv-objective-zero-update-diagnostic-v4'
            or manifest['v3_return_results_sha256'] != V3_RETURN_SHA
            or manifest['v3_local_audit_sha256'] != V3_AUDIT_SHA
            or manifest['runtime_cap_seconds'] != 600 or manifest['expected_autograd_grad_calls'] != 50
            or manifest['optimizer_updates'] != 0 or manifest['term_weights'] != TERM_WEIGHTS
            or manifest['native_cases_used'] or manifest['validation_references_used']
            or manifest['groups'] != training_groups(protocol)):
        raise ValueError('Fixed diagnostic cohort/objective/budget differs')
    required = {'cctv_dgp_objective_diagnostic_v4.py', 'scripts/run_cctv_dgp_objective_diagnostic_vm.py',
                'scripts/run_cctv_dgp_objective_diagnostic_vm.sh', 'scripts/audit_cctv_dgp_objective_diagnostic_v4.py',
                'CCTV_DGP_OBJECTIVE_DIAGNOSTIC_V4.md', 'v3_local_audit_for_diagnostic.json'}
    if not required <= set(manifest['assets_sha256']):
        raise ValueError('Required diagnostic executable/lineage assets missing')
    for name, pin in manifest['assets_sha256'].items():
        path = (bundle_dir / name).resolve()
        if not path.is_relative_to(bundle_dir) or sha(path) != pin:
            raise ValueError('Diagnostic asset differs/escapes: ' + name)
        if name.endswith('.py') and sha(ROOT / name) != pin:
            raise ValueError('Executed diagnostic source differs: ' + name)
    receipt = read(bundle_dir / 'v3_local_audit_for_diagnostic.json')
    if (sha(bundle_dir / 'v3_local_audit_for_diagnostic.json') != V3_AUDIT_SHA
            or not receipt['complete'] or receipt['update_trace_checked'] != 452
            or receipt['recognizer_preview_forwards'] != 52
            or receipt['returned_results_sha256'] != V3_RETURN_SHA):
        raise ValueError('Independent V3 lineage receipt differs')
    completed = Path(v3_return) if v3_return else root / 'outputs/cctv_dgp_perceptual_v3'
    if sha(completed / 'results.json') != V3_RETURN_SHA or (completed / 'failure.json').exists():
        raise ValueError('Completed V3 return differs')
    return protocol, manifest

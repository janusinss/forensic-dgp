"""Independent derivative-span arithmetic using different chunking."""
from datetime import datetime, timezone
import numpy as np
from import_cctv_dgp_live_conditioning_v1_r1_return import ROOT, RETURN, ANALYSIS, read, sha, write


def main():
    path = ANALYSIS / 'learning_subspace_review.json'
    r = read(path)
    assert r['complete'] and sha(ANALYSIS / 'independent_audit.json') == r['independent_return_audit_sha256']
    p = read(RETURN / 'protocol.json')
    edges = np.cumsum([0] + [int(np.prod(p['selected_parameter_shapes'][n])) for n in p['learning_names']])
    vectors = []
    for binding in r['aggregate_gradient_bindings']:
        assert sha(ROOT / binding['file']) == binding['sha256']
        vectors.append(np.load(ROOT / binding['file'], allow_pickle=False, mmap_mode='r'))
    losses = np.asarray(r['scientific_loss_values'])
    for label, record in r['subspaces'].items():
        assert record['supported'] and record['all32_dimensions_retained']
        indices = [i for i, n in enumerate(p['learning_names']) if n in record['names']]
        norms = np.asarray(record['gradient_norms'])
        gram = np.zeros((32, 32)); dots = np.zeros(32); norm2 = 0.
        dual = np.asarray(record['dual_multipliers'])
        for i in indices:
            for begin in range(edges[i], edges[i + 1], 32768):
                block = np.vstack([v[begin:min(begin + 32768, edges[i + 1])] / norm
                                   for v, norm in zip(vectors, norms)])
                gram += np.einsum('ij,kj->ik', block, block, optimize=True)
                d = np.einsum('i,ij->j', -dual, block, optimize=True).astype(np.float32).astype(np.float64)
                dots += np.einsum('ij,j->i', block * norms[:, None], d, optimize=True)
                norm2 += float(d @ d)
        np.testing.assert_allclose(gram, record['normalized_Gram'], rtol=0, atol=1e-10)
        target = np.asarray(record['target_fractions_per_unit_rate']) * losses / norms
        np.testing.assert_allclose(target, record['normalized_targets'], rtol=0, atol=1e-12)
        slack = gram @ dual - target
        assert dual.min() >= 0 and slack.min() > -1e-9 and np.max(abs(slack * dual)) < 1e-9
        assert np.linalg.eigvalsh(gram)[0] > record['rank_bound']
        np.testing.assert_allclose(dots, record['actual_float32_direction_dots'], rtol=1e-9, atol=1e-10)
        assert np.all(-dots / losses >= np.asarray(record['target_fractions_per_unit_rate']) - 1e-6)
        assert np.isclose(np.sqrt(norm2), record['float32_direction_norm'], rtol=1e-10)
    write(ANALYSIS / 'learning_subspace_audit.json', dict(complete=True,
        UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        review_sha256=sha(path), independently_recomputed_both_full_vector_Grams_and_KKT=True,
        all32_RAW_groups_retained_in_both_subspaces=True, conditioning_tensors=12,
        PNG_learning_or_finite_preservation_not_established=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        model_qualified=False, goal_complete=False))
    print(dict(complete=True,subspaces=2,RAW_groups=32,local_neural_calls=0))


if __name__ == '__main__': main()

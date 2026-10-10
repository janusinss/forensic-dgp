"""Read back all saved210 rows and empirical margins; no solver or neural work."""
def verify_geometry(root, p, sha, read):
    from pathlib import Path
    import numpy as np

    rawroot = root.parent / 'cctv_dgp_group_guard_grad_v34_vm'
    pngroot = root.parent / 'cctv_dgp_delivered_guard_grad_v37_vm'
    oldroot = root.parent / 'cctv_dgp_v32_loss_gradient_v1_vm'
    marginroot = root.parent / 'cctv_dgp_finite_clearance_probe_v36_vm'

    def saved(path, base, hashes):
        assert sha(path) == hashes[path.relative_to(base).as_posix()]
        return np.load(path, allow_pickle=False)

    def grouped(base, hashes, labels, metrics):
        rows = []
        for ci, cohort in enumerate(p['cohorts']):
            cases = cohort['cases']; weights = np.zeros((51, 150), np.float64)
            for index, label in enumerate(labels[51 * ci:51 * (ci + 1)]):
                group = label['group']
                if group == 'all': selected = list(range(50))
                elif group == 'clear': selected = [i for i, c in enumerate(cases) if c['profile'] == 'clear']
                elif group == 'degraded': selected = [i for i, c in enumerate(cases) if c['profile'] != 'clear']
                else:
                    source, kind = group.rsplit('/', 1)
                    selected = [i for i, c in enumerate(cases) if c['source'] == source and
                                (kind == 'all' or (kind == 'degraded' and c['profile'] != 'clear') or c['profile'] == kind)]
                assert selected and label['cohort'] == cohort['name']
                weights[index, 3 * np.asarray(selected) + metrics.index(label['metric'])] = 1. / len(selected)
            matrix = np.zeros((51, 978243), np.float64)
            for batch in range(10):
                path = base / f'outputs/{cohort["name"]}/batch{batch}_guard_gradients.npy'
                g = saved(path, base, hashes)
                assert g.dtype == np.float32 and g.shape == (15, 978243) and np.isfinite(g).all()
                matrix += weights[:, 15 * batch:15 * (batch + 1)] @ g.astype(np.float64)
            rows.append(matrix)
        return np.concatenate(rows, axis=0)

    raw = grouped(rawroot, p['guard_return_sha256'], p['constraint_labels'][:102], p['guard_metrics'])
    png = grouped(pngroot, p['PNG_gradient_return_sha256'], p['constraint_labels'][108:], p['PNG_guard_metrics'])
    components = []
    for cohort in p['cohorts']:
        path = oldroot / f'outputs/state0_{cohort["name"]}/gradient_components.npy'
        g = saved(path, oldroot, p['restoration_gradient_sha256'])
        assert g.dtype == np.float64 and g.shape == (7, 978243) and np.isfinite(g).all() and np.count_nonzero(g[3:]) == 0
        components.append(g[:3].copy())
    full = np.concatenate([raw, *components, png], axis=0)
    assert full.shape == (210, 978243) and np.isfinite(full).all()
    for name, digest in p['V36_margin_basis_sha256'].items():
        assert sha(marginroot / name) == digest, name
    v36p = read(marginroot / 'protocol.json')
    assert v36p['cohorts'] == p['cohorts'] and v36p['parameter_layout'] == p['parameter_layout']
    assert v36p['constraint_labels'] == p['constraint_labels'][:108] and v36p['terms'] == p['terms']
    assert v36p['retained_capacity_gates'] == p['retained_capacity_gates']
    targets = np.concatenate([np.asarray(read(marginroot / 'projection.json')['clearance_targets'], np.float64), np.zeros(102)])
    assert np.count_nonzero(targets[:108]) == 65 and np.count_nonzero(targets[102:108]) == 0
    theta = np.load(root / 'theta_before.npy', allow_pickle=False)
    proposal = np.load(root / 'mean_original_displacement.npy', allow_pickle=False)
    direction = np.load(root / 'projected_displacement.npy', allow_pickle=False)
    old_direction = np.load(marginroot / 'projected_displacement.npy', allow_pickle=False)
    assert theta.dtype == np.float32 and proposal.dtype == direction.dtype == old_direction.dtype == np.float64
    assert theta.shape == proposal.shape == direction.shape == old_direction.shape == (978243,)
    assert all(np.isfinite(v).all() for v in [theta, proposal, direction, old_direction])
    assert np.array_equal(theta, np.load(marginroot / 'theta_before.npy', allow_pickle=False))
    assert np.array_equal(proposal, np.load(marginroot / 'mean_original_displacement.npy', allow_pickle=False))
    observations = 0
    metric_keys = {'PNG_MSE': 'MSE', 'one_minus_PNG_SSIM': 'SSIM', 'one_minus_PNG_ArcFace': 'ArcFace_observed_fixed'}
    for ci, cohort in enumerate(p['cohorts']):
        base = marginroot / f'outputs/state0_{cohort["name"]}'
        before = read(base / 'before/receipt.json'); maxima = np.zeros(51)
        labels = p['constraint_labels'][108 + 51 * ci:108 + 51 * (ci + 1)]
        for variant, scale in [('clearance_1', 1.), ('clearance_half', .5), ('clearance_quarter', .25), ('clearance_eighth', .125)]:
            after = read(base / variant / 'receipt.json')
            effective = theta.astype(np.float64) - (theta.astype(np.float64) - scale * old_direction).astype(np.float32).astype(np.float64)
            linear = -(png[51 * ci:51 * (ci + 1)] @ effective)
            for i, label in enumerate(labels):
                key = metric_keys[label['metric']]
                a = before['groups'][label['group']][key]; b = after['groups'][label['group']][key]
                finite = b - a if key == 'MSE' else a - b
                maxima[i] = max(maxima[i], (finite - linear[i]) / scale); observations += 1
        targets[108 + 51 * ci:108 + 51 * (ci + 1)] = 2. * maxima
    proof = read(root / 'projection.json'); shipped = np.asarray(proof['clearance_targets'], np.float64)
    assert shipped.shape == targets.shape == (210,) and (shipped >= 0).all() and np.isfinite(shipped).all()
    assert np.array_equal(shipped[:108], targets[:108]) and proof['clearance_targets'] == p['clearance_targets']
    # Roundoff only in independently reassembled aggregates, never a gate change.
    assert np.allclose(shipped[108:], targets[108:], rtol=2e-10, atol=1e-11)
    assert observations == 408 and np.count_nonzero(shipped[108:]) == 97
    norms = np.linalg.norm(full, axis=1); assert (norms > 0).all()
    unit = full / norms[:, None]; multipliers = np.asarray(proof['multipliers'], np.float64)
    assert multipliers.shape == (210,) and (multipliers >= 0).all() and np.isfinite(multipliers).all()
    assert proof['KKT_tolerance'] == 2e-10 * float(np.linalg.norm(proposal))
    slack = unit @ direction - shipped / norms
    stationarity = float(np.linalg.norm(direction - proposal - multipliers @ unit))
    assert stationarity <= 1e-12 and slack.min() >= -proof['KKT_tolerance']
    assert np.max(np.abs(multipliers * slack)) <= 1e-12
    assert np.allclose(full @ direction, proof['raw_dots_after'], rtol=2e-10, atol=1e-11)
    ratio = float(np.linalg.norm(direction) / np.linalg.norm(proposal))
    assert 0 < ratio <= 2. and abs(ratio - proof['magnitude_ratio']) <= 1e-12
    certificate = {'complete': True, 'preservation_guard_rows': 102, 'existing_restoration_rows': 6,
                   'coarse_PNG_guard_rows': 102, 'full_rows': 210, 'original65_clearances_unchanged': True,
                   'new_PNG_margin_rows': 97, 'calibration_observations': observations,
                   'KKT_stationarity_error': stationarity, 'minimum_unit_clearance_residual': float(slack.min()),
                   'magnitude_ratio': ratio, 'projection_sha256': sha(root / 'projection.json'),
                   'empirical_clearance_not_a_validated_loss_bound': True, 'true_PNG_derivative_claimed': False,
                   'finite_outputs_still_required': True, 'gradient_calls': 0, 'optimizer_updates': 0, 'neural_calls': 0}
    return theta, direction, full, certificate

"""Frozen V16 scheduling, provenance and cache contracts; no neural imports."""
import collections
import random
from pathlib import Path

from cctv_dgp_generalization_v15 import (
    require, sha, read, write, safe, rgb, png, metrics, aggregate, guard_report,
    PROFILES, SOURCES, PARENT_PIN, MIXED_PIN,
)

from cctv_dgp_cache_timing_v16_r2 import cache_order

PLAN = 'broader_codes_protocol_v16_r2.json'
FORMAT = 'reset-broader-code-only-component-v16-r2'
BASELINE_PIN = '04456a809fb57c9c77dc2d5334938fa38813aa19cc6380f6f21c44b4b4eb5af4'
DESIGN = {
    'seed': 20261004, 'epochs': 8, 'batch_size': 10, 'updates_per_epoch': 391,
    'updates': 3128, 'exposures': 31280, 'train_references': 781,
    'train_cases': 3905, 'validation_references': 104, 'validation_cases': 520,
    'trainable_parameters': 2422432, 'optimizer': 'AdamW', 'learning_rate': .0003,
    'weight_decay': .01, 'betas': [.9, .999], 'gradient_clip_norm': 1.,
    'loss': 'observed_token_cross_entropy_only', 'amp': False,
    'statistics': 'none', 'fidelity': 0., 'snapshot_epochs': [0, 4, 8],
    'cache_training_batch_size': 5, 'cache_validation_batch_size': 1,
    'cache_cap_seconds': 900, 'fit_cap_seconds': 1200,
    'supervisor_cap_seconds': 2400, 'audit_cap_seconds': 240,
    'peak_vram_cap_bytes': 20 * 1024**3, 'free_disk_required_bytes': 12 * 1024**3,
    'cache_timing_at_reference': 30, 'timing_update': 25,
    'cache_projection_method': 'source-role-steady-reference-rate-startup-warmups-once-v16-r2',
    'timing_safety_factor': 1.25, 'minimum_fit_ce_improvement_epoch4': .01,
    'fresh_parity_epochs': [0, 8], 'float_parity_tolerance': 2e-6,
    'selection': 'None; component diagnostic, no best.pth or promotion.',
}
SOURCES_FILES = [
    'cctv_dgp_broader_codes_v16.py', 'dgp_broader_code_conditioner_v16.py',
    'cctv_dgp_generalization_v15.py', 'dgp_direct_face_code_v14.py',
    'scripts/train_cctv_dgp_broader_codes_v16.py',
    'scripts/audit_cctv_dgp_broader_codes_v16.py',
    'scripts/prepare_cctv_dgp_broader_codes_v16.py',
    'scripts/supervise_cctv_dgp_broader_codes_v16.py',
    'scripts/launch_cctv_dgp_broader_codes_v16.py',
    'scripts/import_cctv_dgp_broader_codes_v16.py',
    'tests/test_cctv_dgp_broader_codes_v16.py',
    'tests/test_dgp_broader_code_conditioner_v16.py',
    'tests/test_cctv_dgp_broader_codes_runtime_v16.py', 'face_prior_grid_v10.py',
    'cctv_dgp_cache_timing_v16_r2.py', 'tests/test_cctv_dgp_v16_cache_timing_r2.py',
]


def previews(refs):
    return [r['id'] for s in SOURCES for r in sorted(
        (r for r in refs if r['source'] == s), key=lambda r: r['id'])[:5]]


def validate_cohort(p):
    refs = p['references']; require(len(refs) == 885 and len({r['id'] for r in refs}) == 885,
                                  'Require885 unique references')
    training = [r for r in refs if r['role'] == 'train']
    validation = [r for r in refs if r['role'] == 'validation']
    require(len(training) == 781 and len(validation) == 104, 'Wrong split sizes/roles')
    for key in ['id', 'source_sha256', 'target_rgb_sha256']:
        require(not ({r[key] for r in training} & {r[key] for r in validation}),
                'Own exact train/validation overlap:' + key)
    for cohort, counts in [(training, [390, 391]), (validation, [51, 53])]:
        require([sum(r['source'] == s for r in cohort) for s in SOURCES] == counts,
                'Source counts changed')
    for role, key, count in [('train', 'training_cases', 3905), ('validation', 'validation_cases', 520)]:
        cases = p[key]; lookup = {r['id']: r for r in refs if r['role'] == role}
        require(len(cases) == count and len({c['id'] for c in cases}) == count, 'Case count differs')
        groups = collections.defaultdict(list)
        for c in cases:
            require(c['reference_id'] in lookup and c['source'] == lookup[c['reference_id']]['source'],
                    'Training/validation case crossed roles or source')
            groups[c['reference_id']].append(c['profile'])
        require(set(groups) == set(lookup) and all(sorted(x) == sorted(PROFILES) for x in groups.values()),
                'Missing/repeated camera profile')
        require(p[role + '_preview_reference_ids'] == previews(list(lookup.values())), 'Preview differs')
    require(all(p[k] is False for k in ['native_used', 'native_reserved_used', 'production_promoted', 'checkpoint_selected']),
            'Wrong diagnostic scope')


def schedule(p):
    validate_cohort(p)
    lookup = {(c['source'], c['profile'], c['reference_id']): c['id'] for c in p['training_cases']}
    rids = {s: sorted(r['id'] for r in p['references'] if r['role'] == 'train' and r['source'] == s) for s in SOURCES}
    rng = random.Random(DESIGN['seed']); steps = []
    for epoch in range(1, 9):
        columns = []
        for s in SOURCES:
            for profile in PROFILES:
                order = rids[s].copy(); rng.shuffle(order)
                columns.append([lookup[s, profile, order[i % len(order)]] for i in range(391)])
        for i in range(391):
            steps.append({'update': len(steps) + 1, 'epoch': epoch,
                          'case_ids': [col[i] for col in columns]})
    return steps


def validate_schedule(p, steps):
    require(steps == schedule(p), 'Seeded schedule differs')
    cases = {c['id']: c for c in p['training_cases']}
    for epoch in range(1, 9):
        counts = collections.Counter()
        for step in [s for s in steps if s['epoch'] == epoch]:
            require(len(step['case_ids']) == 10 and
                    {(cases[c]['source'], cases[c]['profile']) for c in step['case_ids']} ==
                    {(s, x) for s in SOURCES for x in PROFILES}, 'Batch balance differs')
            counts.update(step['case_ids'])
        require(set(counts) == set(cases), 'Epoch misses training cases')
        require(sum(counts.values()) == 3910 and sum(v - 1 for v in counts.values()) == 5,
                'Unexpected epoch replay')
        require(all(v == 1 or (v == 2 and cases[c]['source'] == SOURCES[0]) for c, v in counts.items()),
                'Only five shorter-source cases may replay')


def cache_arrays(dgp, features, logits):
    import numpy as np
    for x, shape in [(dgp, (3, 256, 256)), (features, (256, 16, 16)), (logits, (256, 1024))]:
        require(x.dtype == np.float32 and x.shape == shape and np.isfinite(x).all(), 'Invalid frozen cache')
    require(dgp.min() >= 0 and dgp.max() <= 1, 'DGP cache outside RGB range')


def cache_path(out, cid):
    return safe(out, 'cache/' + cid + '.npz')


def load_cache(out, cid, manifest):
    """Read one pinned case, releasing the archive before returning its arrays."""
    import numpy as np
    path = cache_path(out, cid)
    require(sha(path) == manifest[cid]['sha256'], 'Frozen cache fingerprint differs:' + cid)
    with np.load(path, allow_pickle=False) as stream:
        require(set(stream.files) == {'dgp', 'features', 'logits'}, 'Cache fields differ')
        values = [stream[k].copy() for k in ['dgp', 'features', 'logits']]
    cache_arrays(*values)
    return values


def code_metrics(logits, labels, support):
    """Independent float64 observed-token CE and accuracy; no model or gradient."""
    import numpy as np
    require(logits.shape == (256, 1024) and logits.dtype == np.float32 and
            np.isfinite(logits).all() and labels.shape == (256,) and
            labels.dtype == np.int64 and labels.min() >= 0 and labels.max() < 1024 and
            support.shape == (256,) and support.dtype == np.bool_ and support.any(),
            'Invalid code metric arrays')
    x = logits[support].astype(np.float64); y = labels[support]
    maxima = x.max(1)
    ce = maxima + np.log(np.exp(x - maxima[:, None]).sum(1)) - x[np.arange(len(x)), y]
    return {'code_ce': float(ce.mean()), 'code_accuracy': float((x.argmax(1) == y).mean())}


def near(actual, expected, tolerance=1e-6):
    import math
    require(set(actual) == set(expected), 'Metric keys differ')
    for key, value in expected.items():
        if isinstance(value, dict): near(actual[key], value, tolerance)
        elif value is None or isinstance(value, (bool, int, str, list)):
            require(actual[key] == value, 'Value differs:' + key)
        else:
            require(math.isfinite(actual[key]) and abs(actual[key] - value) <= tolerance,
                    'Arithmetic differs:' + key)


def archive_members(stream, destination, cap):
    """Reject links, special files, aliases, duplicates and escaping archive paths."""
    from pathlib import PurePosixPath
    members = stream.getmembers(); seen = set()
    require(sum(m.size for m in members) < cap, 'Archive exceeds frozen size cap')
    for member in members:
        name = member.name
        require(member.isfile() or member.isdir(), 'Unsafe archive member type')
        safe(destination, name)
        normalized = str(PurePosixPath(name))
        require(normalized == name and name not in seen and name != '.', 'Duplicate/aliased archive path')
        seen.add(name)
    return members


def verify(root, parent, mixed, baseline, pin):
    root, parent, mixed, baseline = map(Path, [root, parent, mixed, baseline])
    require(sha(root / PLAN) == pin == (root / 'protocol.sha256').read_text().strip(), 'V16 protocol differs')
    p = read(root / PLAN)
    require(p['format'] == FORMAT and p['design'] == DESIGN, 'Frozen design differs')
    require(p['parent_protocol_sha256'] == PARENT_PIN and p['mixed_protocol_sha256'] == MIXED_PIN
            and p['baseline_results_sha256'] == BASELINE_PIN, 'Lineage pins differ')
    validate_cohort(p); validate_schedule(p, read(root / 'schedule_v16.json')['steps'])
    require(read(root / 'cache_order_v16_r2.json')['reference_ids'] == [ref['id'] for ref in cache_order(p)], 'Cache order differs')
    for name, expected in p['assets_sha256'].items():
        require(sha(safe(root, name)) == expected, 'V16 asset differs:' + name)
    require(sha(parent / 'face_code_fit_protocol_v12.json') == PARENT_PIN, 'Parent protocol differs')
    require(read(parent / 'face_code_fit_protocol_v12.json')['assets_sha256'] == p['parent_assets_sha256'], 'Parent manifest differs')
    for name, expected in p['parent_assets_sha256'].items():
        require(sha(safe(parent, name)) == expected, 'Parent asset differs:' + name)
    require(sha(mixed / 'mixed_protocol_v9.json') == MIXED_PIN, 'Mixed protocol differs')
    mp = read(mixed / 'mixed_protocol_v9.json')
    require(p['references'] == mp['references'] and p['training_cases'] == mp['training_cases'] and
            p['validation_cases'] == mp['validation_cases'], 'Data transcription differs')
    for name, expected in p['data_assets_sha256'].items():
        require(mp['assets_sha256'][name] == expected and sha(safe(mixed, name)) == expected, 'Data asset differs:' + name)
    require(sha(baseline / 'results.json') == BASELINE_PIN, 'V15 baseline receipt differs')
    br = read(baseline / 'results.json')
    require(br['complete'] and br['optimizer_updates'] == br['backward_calls'] == 0, 'Baseline scope differs')
    for name, expected in p['baseline_assets_sha256'].items():
        require(br['artifacts_sha256'][name] == expected and sha(safe(baseline, name)) == expected,
                'Baseline artifact differs:' + name)
    require({r['id'] for r in br['rows']} == {c['id'] for c in p['validation_cases']}, 'Baseline cohort differs')
    return p

"""Finite spatial-residual capacity contract and arithmetic; no neural imports."""
import collections
import math
from pathlib import Path
import sys

PLAN = 'structure_protocol_v18.json'
FORMAT = 'cctv-dgp-spatial-residual-capacity-v18'
R2_PIN = '4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0'
R2_HEAD = '3ef704e70f68d633ac7624eb47f79cfada189341dfdc58bad08444595d7d6757'
PROTOTYPE_RESULTS = 'b589998457496f5cbecd1e8933b3a9e5a1bc9108c14e9c6381da5d721ab98a7c'
MODULE_PIN = 'e98af67f110b4ebc276ff2394345a81bc6ca7e082b36a493ad5c3fde94593e5d'
DESIGN = {'seed': 20261005, 'training_references': 10, 'training_cases': 50,
    'updates': 600, 'batch_size': 10, 'exposures': 6000, 'cohort_epochs': 120,
    'snapshot_updates': [0, 50, 200, 600], 'trainable_parameters': 684395,
    'optimizer': 'fresh AdamW', 'learning_rate': .0003, 'weight_decay': .01,
    'betas': [.9, .999], 'gradient_clip_norm': 1., 'amp': False, 'ema': False,
    'pixel_base': 'audited retained DGP V2, frozen',
    'prior_feature': 'frozen R2 epoch8 codes -> frozen CodeFormer generator block12:N256x64x64',
    'statistics': 'none', 'fidelity': 0., 'prior_RGB_replacement': False,
    'residual_amplitude': .5, 'objective': 'observed MSE +0.5 masked multi-scale MSE +calibrated fixed-affine identity',
    'structure_sizes': [32, 64, 128], 'coarse_weight': .5,
    'identity_calibration': 'initial first balanced training batch gradient norm ratio; frozen thereafter',
    'identity_weight_min': 1e-4, 'identity_weight_max': 10.,
    'fit_stop_measure': 'full cohort weighted raw reconstruction plus exported-PNG identity; same measure at updates0/50',
    'fit_stop_update': 50, 'minimum_full_cohort_loss_improvement': .01,
    'minimum_degraded_MSE_improvement': .10, 'minimum_degraded_PSNR_gain_dB': .1,
    'group_MSE_tolerance': 1e-12, 'group_SSIM_cosine_tolerance': 1e-6,
    'cache_cap_seconds': 300, 'fit_cap_seconds': 900, 'trainer_cap_seconds': 1230,
    'audit_cap_seconds': 300, 'supervisor_cap_seconds': 1800,
    'cache_timing_case': 10, 'fit_timing_update': 20, 'timing_safety_factor': 1.25,
    'peak_vram_cap_bytes': 20 * 1024**3, 'free_disk_required_bytes': 4 * 1024**3,
    'fresh_parity_case_ids_policy': 'first reference per source, clear and blur, snapshots0/600',
    'VM_float_parity_tolerance': 2e-6, 'local_head_replay_tolerance': 5e-5,
    'selection': 'None. Capacity evidence precedes separately frozen generalization/native verification.'}


def environment(parent, r2):
    sys.path.insert(0, str(parent))
    sys.path.insert(0, str(r2))
    import cctv_dgp_broader_codes_v16 as legacy
    return legacy


def selected_cohort(p):
    ids = p['train_preview_reference_ids']
    refs = [r for r in p['references'] if r['id'] in ids]
    lookup = {c['id']: c for c in p['training_cases']}
    cases = [c for c in lookup.values() if c['reference_id'] in ids]
    return refs, cases


def schedule(refs, cases, sources, profiles):
    ids = {source: sorted(r['id'] for r in refs if r['source'] == source) for source in sources}
    lookup = {(c['reference_id'], c['profile']): c['id'] for c in cases}
    return [[lookup[(ids[source][index % 5], profile)] for source in sources for profile in profiles]
            for index in range(DESIGN['updates'])]


def verify(root, parent, r2, mixed, baseline, pin):
    root, parent, r2, mixed, baseline = map(Path, [root, parent, r2, mixed, baseline])
    v = environment(parent, r2)
    v.require(v.sha(root / PLAN) == pin == (root / 'protocol.sha256').read_text().strip(), 'V18 protocol differs')
    p = v.read(root / PLAN)
    v.require(p['format'] == FORMAT and p['design'] == DESIGN, 'V18 finite design differs')
    legacy = v.verify(r2, parent, mixed, baseline, R2_PIN)
    refs, cases = selected_cohort(legacy)
    v.require(p['references'] == refs and p['training_cases'] == cases
              and p['reference_ids'] == legacy['train_preview_reference_ids']
              and len(refs) == 10 and len(cases) == 50
              and all(r['role'] == 'train' for r in refs), 'V18 training-only cohort differs')
    v.require([sum(r['source'] == source for r in refs) for source in v.SOURCES] == [5, 5], 'Source balance differs')
    v.require(v.read(root / 'schedule_v18.json')['batches'] == schedule(refs, cases, v.SOURCES, v.PROFILES),
              'V18 exposures/schedule differ')
    for name, digest in p['assets_sha256'].items():
        v.require(v.sha(v.safe(root, name)) == digest, 'V18 source/asset differs:' + name)
    v.require(p['assets_sha256']['dgp_structure_conditioner_v18.py'] == MODULE_PIN
              and p['assets_sha256']['weights/r2_conditioner_epoch8.pth'] == R2_HEAD
              and p['r2_protocol_sha256'] == R2_PIN, 'Declared decoder/code lineage differs')
    proof = v.read(root / 'lineage/prototype_results.json')
    audit = v.read(root / 'lineage/prototype_saved_output_audit.json')
    v.require(v.sha(root / 'lineage/prototype_results.json') == PROTOTYPE_RESULTS
              == audit['results_sha256'] and audit['complete'] and proof['complete']
              and proof['module_sha256'] == MODULE_PIN
              and proof['backward_calls'] == proof['optimizer_updates'] == 0, 'Initial local parity proof differs')
    full = v.read(root / 'lineage/r2_full_audit.json')
    review = v.read(root / 'lineage/r2_review.json')
    v.require(full['complete'] and full['protocol_sha256'] == R2_PIN
              and full['results_sha256'] == p['r2_results_sha256'] == review['results_sha256']
              and not review['production_promoted'], 'Audited negative R2 lineage differs')
    expected_data = {name: legacy['data_assets_sha256'][name] for name in
        {c['input'] for c in cases} | {r['target'] for r in refs} | {r['observed'] for r in refs}}
    v.require(p['data_assets_sha256'] == expected_data, 'Selected data binding differs')
    v.require(p['training_location'] == 'existing forensic-dgp-thesis Linux NVIDIA L4 VM'
              and not any(p[k] for k in ['validation_used', 'native_used', 'native_reserved_used',
                                         'checkpoint_selected', 'production_promoted']), 'V18 scope differs')
    return p, v


def normalized_group_weights(means):
    if len(means) != 10 or not all(math.isfinite(x) and x > 0 for x in means.values()):
        raise ValueError('Require ten finite positive training-only group errors')
    overall = math.fsum(means.values()) / 10
    raw = {key: min(4., max(.25, overall / value)) for key, value in means.items()}
    scale = math.fsum(raw.values()) / 10
    return {key: value / scale for key, value in raw.items()}


def calibrated_identity_weight(reconstruction_norm, identity_norm):
    if not all(math.isfinite(x) and x > 0 for x in [reconstruction_norm, identity_norm]):
        raise ValueError('Initial reconstruction/identity gradient norms must be finite and nonzero')
    return min(DESIGN['identity_weight_max'], max(DESIGN['identity_weight_min'], reconstruction_norm / identity_norm))


def strict_preservation(candidate, baseline):
    if set(candidate) != set(baseline):
        raise ValueError('Every clear/source/profile/degraded group is required')
    failures = []
    for name, reference in baseline.items():
        actual = candidate[name]
        if not all(math.isfinite(row[key]) for row in [actual, reference]
                   for key in ['MSE', 'SSIM', 'ArcFace_observed_fixed']):
            raise ValueError('Nonfinite preservation evidence')
        if actual['cases'] != reference['cases']:
            raise ValueError('Identical cases required for every preservation group')
        if actual['MSE'] > reference['MSE'] + DESIGN['group_MSE_tolerance']:
            failures.append(name + ':MSE')
        for key in ['SSIM', 'ArcFace_observed_fixed']:
            if actual[key] < reference[key] - DESIGN['group_SSIM_cosine_tolerance']:
                failures.append(name + ':' + key)
    gain = candidate['degraded']['PSNR'] - baseline['degraded']['PSNR']
    reduction = 1 - candidate['degraded']['MSE'] / baseline['degraded']['MSE']
    return {'degraded_PSNR_gain_dB': gain, 'degraded_MSE_improvement_fraction': reduction,
            'failed_groups_metrics': failures, 'strict_preservation_passed': not failures and gain >= .1,
            'capacity_gain_passed': reduction >= .10,
            'qualified_for_separate_generalization_protocol': not failures and gain >= .1 and reduction >= .10}


def verify_trace(p, records, complete):
    batches = schedule(p['references'], p['training_cases'],
                       ['dataset/asian_faces', 'dataset/thumbnails128x128'],
                       ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24'])
    if len(records) > DESIGN['updates'] or (complete and len(records) != DESIGN['updates']):
        raise ValueError('Finite update count differs')
    counts = collections.Counter()
    for index, row in enumerate(records):
        if row['update'] != index + 1 or row['case_ids'] != batches[index]:
            raise ValueError('Update or exposure order differs')
        if not all(math.isfinite(row[k]) and row[k] >= 0
                   for k in ['loss', 'reconstruction', 'identity', 'gradient_norm', 'seconds']):
            raise ValueError('Nonfinite training receipt')
        counts.update(row['case_ids'])
    return {'updates': len(records), 'exposures': sum(counts.values()), 'case_exposures': dict(counts)}


def expected_counts(updates=600):
    if updates != 600:
        raise ValueError('Full counts require complete finite pilot')
    return {'dgp': 58, 'prior_encoder': 58, 'prior_classifier': 58, 'r2_head': 58,
            'prior_generator': 58, 'prior_RGB_tail': 0, 'residual_head': 860,
            'recognizer': 912, 'unused_v11': 0}

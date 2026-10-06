"""Finite V19 inference contract; frozen training-calibrated selection, no fitting."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

PLAN = 'input_selection_protocol_v19.json'
FORMAT = 'dgp-input-selection-development-inference-v19'
V18_PIN = 'e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb'
V18_RESULTS = 'f0acefaadbf2890627339038371673aeed30cf9a066b9f74ca71d28766e9c206'
SELECTOR_PIN = 'cf72354d5092d0aab93dd7ecd646088999dfb009ea5cd4c0b0f19551f0c80180'
ARMS = ['basic_resizing', 'retained_dgp_v2', 'pretrained_codeformer_none',
        'structure_v18_update600', 'automatic_v19']
DESIGN = {'seed': 20261005, 'batch_size': 1, 'training_parity_cases': 50,
    'validation_references': 104, 'validation_cases': 520, 'training': False,
    'optimizer_updates': 0, 'backward_calls': 0, 'terminal_checkpoint': 'V18 fixed update600',
    'runner_cap_seconds': 1200, 'audit_cap_seconds': 300, 'supervisor_cap_seconds': 1800,
    'export_cap_seconds': 180, 'free_disk_required_bytes': 4 * 1024**3,
    'peak_vram_cap_bytes': 20 * 1024**3, 'timing_at_validation_case': 20,
    'timing_safety_factor': 1.25, 'fresh_parity_tolerance': 2e-6,
    'input_feature_absolute_tolerance': 1e-12,
    'CPU_cached_head_tolerance': 5e-5, 'CPU_cached_head_cases': 24,
    'MSE_preservation_tolerance': 1e-12, 'SSIM_cosine_preservation_tolerance': 1e-6,
    'minimum_degraded_PSNR_gain_dB': .1, 'minimum_degraded_MSE_improvement': .10,
    'arms': ARMS, 'pretrained_baseline': 'reuse audited V15 starting CodeFormer codes, no statistics/fidelity w0; no fresh baseline network call',
    'input_qualification': 'Separate required input-only face/pose/insufficient-information review; selector alone does not qualify a crop',
    'selection': 'Fixed V18 terminal snapshot plus already frozen input rule; no best checkpoint/threshold fitting during evaluation'}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def safe(root, name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts
            and ':' not in name and '\\' not in name and str(p) == name, 'Unsafe V19 member')
    target = (Path(root) / name).resolve()
    require(target.is_relative_to(Path(root).resolve()), 'V19 member escapes root')
    return target


def aggregate(rows):
    """Use canonical Python fsum for means; derived log10 is audited by ULPs."""
    sources = ['dataset/asian_faces', 'dataset/thumbnails128x128']
    profiles = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
    groups = {'clear': [r for r in rows if r['profile'] == 'clear'],
              'degraded': [r for r in rows if r['profile'] != 'clear']}
    for source in sources:
        for profile in profiles:
            groups[source + '/' + profile] = [r for r in rows if r['source'] == source and r['profile'] == profile]
        groups[source + '/degraded'] = [r for r in rows if r['source'] == source and r['profile'] != 'clear']
    result = {}
    for key, items in groups.items():
        require(bool(items), 'Empty V19 summary group')
        means = {k: math.fsum(r[k] for r in items) / len(items) for k in ['MSE', 'SSIM', 'MAE', 'ArcFace_observed_fixed']}
        require(all(math.isfinite(x) for x in means.values()), 'Nonfinite summary')
        result[key] = {'cases': len(items), **means, 'PSNR': -10 * math.log10(means['MSE']) if means['MSE'] else None}
    return result


def expected_counts():
    return {'dgp': 570, 'prior_encoder': 570, 'prior_classifier': 570, 'r2_head': 570,
            'prior_generator': 570, 'prior_RGB_tail': 0, 'residual_head': 570,
            'recognizer': 624, 'unused_v11': 0}


def validate_result_scope(result, pin):
    require(result['complete'] and result['protocol_sha256'] == pin
            and result['training'] is False and result['validation_used'] is True
            and result['optimizer_updates'] == result['backward_calls'] == 0,
            'Complete inference-only V19 result required')
    for key in ['teacher_used', 'native_used', 'native_reserved_used', 'best_checkpoint_selected',
                'thresholds_refitted', 'production_promoted']:
        require(result[key] is False, 'V19 diagnostic scope differs:' + key)
    require(math.isfinite(result['seconds']) and 0 < result['seconds'] <= 1200, 'V19 runner timing differs')


def validate_automatic_alias(row):
    require(set(row['arms']) == set(ARMS), 'Every declared arm required')
    branch = row['decision']['branch']
    require(branch in ['retained_dgp_v2', 'structure_v18_update600'], 'Undeclared selection branch')
    require(row['arms']['automatic_v19'] == row['arms'][branch],
            'Automatic output must alias the chosen original raw/PNG/embedding')


def verify(root, parent, r2, mixed, baseline, pin):
    root, parent, r2, mixed, baseline = map(Path, [root, parent, r2, mixed, baseline])
    require(sha(root / PLAN) == pin == (root / 'protocol.sha256').read_text().strip(), 'V19 protocol differs')
    p = read(root / PLAN)
    require(p['format'] == FORMAT and p['design'] == DESIGN, 'V19 finite design differs')
    for name, digest in p['assets_sha256'].items():
        require(sha(safe(root, name)) == digest, 'V19 asset differs:' + name)
    require(p['assets_sha256']['dgp_input_selector_v19.py'] == SELECTOR_PIN, 'Frozen training-calibrated selector differs')
    require(p['assets_sha256']['dgp_structure_conditioner_v18.py'] ==
            'e98af67f110b4ebc276ff2394345a81bc6ca7e082b36a493ad5c3fde94593e5d', 'Frozen spatial architecture differs')
    v18 = root / 'parent_v18'
    sys.path.insert(0, str(v18))
    import cctv_dgp_structure_v18 as old
    cp, legacy = old.verify(v18, parent, r2, mixed, baseline, V18_PIN)
    broad = legacy.read(r2 / legacy.PLAN)
    refs = [r for r in broad['references'] if r['role'] == 'validation']
    require(p['references'] == refs and len(refs) == 104 and p['cases'] == broad['validation_cases']
            and len(p['cases']) == 520, 'Fixed development cohort differs')
    require(p['training_parity_cases'] == cp['training_cases'] and p['training_parity_references'] == cp['references'],
            'Training-only parity cohort differs')
    require(p['preview_reference_ids'] == broad['validation_preview_reference_ids'], 'Preview identities differ')
    for key in ['id', 'source_sha256', 'target_rgb_sha256']:
        require(not ({r[key] for r in refs} & {r[key] for r in broad['references'] if r['role'] == 'train'}),
                'Own train/development exact overlap:' + key)
    require({s: sum(r['source'] == s for r in refs) for s in legacy.SOURCES} ==
            {legacy.SOURCES[0]: 51, legacy.SOURCES[1]: 53}, 'Development source balance differs')
    names = ({c['input'] for c in p['cases']} | {r['target'] for r in refs}
             | {r['observed'] for r in refs} | set(cp['data_assets_sha256']))
    require(p['data_assets_sha256'] == {name: broad['data_assets_sha256'][name] for name in names}, 'Input/data fingerprints differ')
    original = read(root / 'lineage/v18_results.json')
    full = read(root / 'lineage/v18_full_audit.json')
    require(sha(root / 'lineage/v18_results.json') == V18_RESULTS and original['complete']
            and full['complete'] and full['results_sha256'] == V18_RESULTS
            and original['snapshots'][-1]['update'] == 600
            and p['terminal_state_hash'] == original['snapshots'][-1]['state_hash'], 'V18 terminal lineage differs')
    require(p['assets_sha256']['weights/structure_update600.pth'] == original['artifacts_sha256']['update600/decoder.pth'],
            'Fixed terminal checkpoint differs')
    for c in cp['training_cases']:
        for new_prefix, old_prefix in [('parity/dgp_base', 'update0'), ('parity/structure_update600', 'update600')]:
            require(p['assets_sha256'][new_prefix + '/' + c['id'] + '.npy'] ==
                    original['artifacts_sha256'][old_prefix + '/' + c['id'] + '.npy'], 'Returned parity tensor differs')
    control = read(root / 'lineage/processing_results_v19.json')
    frozen_control = read(root / 'lineage/processing_protocol_v19.json')
    require(control['complete'] and control['preservation']['qualified_for_separate_generalization_protocol']
            and control['protocol_sha256'] == sha(root / 'lineage/processing_protocol_v19.json')
            and frozen_control['selector_sha256'] == SELECTOR_PIN, 'Positive training-only processing prerequisite differs')
    require(p['baseline_results_sha256'] == legacy.sha(baseline / 'results.json') == broad['baseline_results_sha256'], 'Audited baseline differs')
    require(p['training'] is False and p['validation_used'] is True and all(p[key] is False for key in
            ['native_used', 'native_reserved_used', 'teacher_used', 'production_promoted']), 'V19 diagnostic scope differs')
    sys.path.insert(0, str(root))
    return p, legacy, old

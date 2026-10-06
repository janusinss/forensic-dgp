"""Post-return luminance ablation; saved outputs only, no neural calls or fitting.

This is exploratory analysis of the repeatedly used development cohort. It does
not amend V19 r2, select a model, compute fresh recognizer embeddings, qualify
native CCTV, or provide independent final review. All original failures remain.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_input_selection_return_v19_r2'
OUT = RETURN / 'outputs/input_selection_v19_r2'
DEST = ROOT / 'outputs/cctv_dgp_input_selection_v19_r2_analysis'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
R2 = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
RESULT_PIN = '98c311465351b36c20d26eca2a1d2310194b1ac255ebfad82f6d49b5c039f9c8'
PROTOCOL_PIN = '5c128d6715785f84858f762035a168f396b46787d6a864b1d8d6437ffc69dc3f'
CAP_SECONDS = 180


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def analyze():
    start = time.monotonic()
    assert not DEST.exists(), 'Preserve previous/partial analysis; no overwrite'
    result = read(OUT / 'results.json')
    audit = read(RETURN / 'local_full_audit.json')
    imported = read(RETURN / 'local_import_and_audit.json')
    assert sha(OUT / 'results.json') == RESULT_PIN == audit['results_sha256']
    assert result['protocol_sha256'] == PROTOCOL_PIN == audit['protocol_sha256']
    assert audit['complete'] and imported['complete'] and imported['full_output_audit']
    assert audit['validation_cases'] == 520 and audit['head_forwards'] == 24
    assert result['training'] is False and result['optimizer_updates'] == result['backward_calls'] == 0
    protocol = read(RETURN / 'input_selection_protocol_v19_r2.json')
    assert sha(RETURN / 'input_selection_protocol_v19_r2.json') == PROTOCOL_PIN
    parent = read(R2 / 'broader_codes_protocol_v16_r2.json')
    helper = R2 / 'cctv_dgp_generalization_v15.py'
    assert sha(helper) == parent['assets_sha256']['cctv_dgp_generalization_v15.py']
    sys.path.insert(0, str(R2))
    import cctv_dgp_generalization_v15 as v
    import numpy as np
    from PIL import Image
    assert Path(v.__file__).resolve() == helper.resolve()
    refs = {row['id']: row for row in protocol['references']}
    cases = {row['id']: row for row in protocol['cases']}
    DEST.mkdir()
    (DEST / 'brightness_only_previews').mkdir()
    rows, artifacts = [], {}
    for index, row in enumerate(result['rows']):
        assert time.monotonic() - start <= CAP_SECONDS, 'Finite analysis cap exceeded; preserve partial evidence'
        cid = row['id']; ref = refs[row['reference_id']]; case = cases[cid]
        camera = v.rgb(MIXED / case['input'])
        target = v.rgb(MIXED / ref['target'])
        with Image.open(MIXED / ref['observed']) as image:
            support = np.asarray(image).copy() > 0
        base = np.load(OUT / row['arms']['retained_dgp_v2']['raw'], allow_pickle=False)
        candidate = np.load(OUT / row['arms']['structure_v18_update600']['raw'], allow_pickle=False)
        assert base.shape == candidate.shape == (256, 256, 3) and base.dtype == candidate.dtype == np.float32
        # One fixed decomposition. No targets, identities, labels or searched
        # coefficients enter the candidate computation. Targets score it only.
        delta = candidate.astype(np.float64) - base.astype(np.float64)
        luminance = delta @ np.array([.299, .587, .114], dtype=np.float64)
        offset = float(luminance[support].mean(dtype=np.float64))
        raw = np.clip(base.astype(np.float64) + offset, 0, 1).astype(np.float32)
        prediction = v.png(raw, camera, support)
        scores = v.metrics(prediction, target, support)
        before = row['arms']['retained_dgp_v2']; after = row['arms']['structure_v18_update600']
        rows.append({'id': cid, 'source': row['source'], 'profile': row['profile'],
            'uniform_luminance_offset': offset, **scores,
            'full_candidate_MSE': after['MSE'], 'base_MSE': before['MSE'],
            'base_SSIM': before['SSIM'], 'full_candidate_SSIM': after['SSIM']})
        if row['reference_id'] in protocol['preview_reference_ids']:
            name = 'brightness_only_previews/' + cid + '.png'
            Image.fromarray(prediction).save(DEST / name)
            artifacts[name] = sha(DEST / name)
        if (index + 1) % 100 == 0:
            print({'diagnostic_cases': index + 1, 'seconds': time.monotonic() - start}, flush=True)
    assert len(rows) == 520 and len(artifacts) == 50
    groups = {'clear': [r for r in rows if r['profile'] == 'clear'],
              'degraded': [r for r in rows if r['profile'] != 'clear']}
    for source in v.SOURCES:
        for profile in v.PROFILES:
            groups[source + '/' + profile] = [r for r in rows if r['source'] == source and r['profile'] == profile]
        groups[source + '/degraded'] = [r for r in rows if r['source'] == source and r['profile'] != 'clear']
    summaries = {}
    failures = []
    for name, items in groups.items():
        means = {key: math.fsum(row[key] for row in items) / len(items) for key in ['MSE', 'SSIM', 'MAE', 'uniform_luminance_offset']}
        summaries[name] = {'cases': len(items), **means, 'PSNR': -10 * math.log10(means['MSE']) if means['MSE'] else None,
                           'ArcFace_observed_fixed': None}
        baseline = result['summaries']['retained_dgp_v2'][name]
        if means['MSE'] > baseline['MSE'] + 1e-12:
            failures.append(name + ':MSE')
        if means['SSIM'] < baseline['SSIM'] - 1e-6:
            failures.append(name + ':SSIM')
    with (DEST / 'all_groups.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['arm', 'group', 'cases', 'MSE', 'PSNR', 'SSIM', 'MAE', 'ArcFace_observed_fixed']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for arm, grouped in list(result['summaries'].items()) + [('uniform_luminance_diagnostic', summaries)]:
            for name, scores in grouped.items():
                writer.writerow({'arm': arm, 'group': name, **{k: scores[k] for k in fields[2:]}})
    with (DEST / 'input_choices.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['id', 'source', 'profile', 'branch', 'laplacian_MSE', 'luminance_variance', 'laplacian_over_variance']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in result['rows']:
            writer.writerow({k: row[k] if k in row else row['decision'][k] for k in fields})
    write(DEST / 'uniform_luminance_diagnostic.json', {
        'complete': True, 'results_sha256': RESULT_PIN, 'protocol_sha256': PROTOCOL_PIN,
        'parent_full_audit_sha256': sha(RETURN / 'local_full_audit.json'), 'script_sha256': sha(Path(__file__)),
        'scope': 'Exploratory saved-output ablation on previously used photographic development cases; not independent final/native evidence.',
        'formula': 'clip(canonical_DGP_raw + mean_observed((fixed_V18_raw - canonical_DGP_raw) dot [0.299,0.587,0.114]), 0,1)',
        'target_used_for_candidate_computation': False, 'luminance_projection_weights': [.299, .587, .114],
        'per_case_offsets_measured': 520, 'searched_coefficients': 0,
        'coefficient_search': False, 'selector_refit': False, 'fresh_recognizer_computed': False,
        'full_preservation_qualification': None, 'native_used': False, 'reserved_used': False,
        'training': False, 'neural_forwards': 0, 'optimizer_updates': 0, 'backward_calls': 0,
        'production_promoted': False, 'summaries': summaries, 'MSE_SSIM_failed_checks': failures,
        'rows': rows, 'preview_artifacts_sha256': artifacts,
        'original_V19_r2_failed_guards': result['preservation'],
        'seconds': time.monotonic() - start, 'cap_seconds': CAP_SECONDS})
    branch_counts = {profile: dict(collections.Counter(row['decision']['branch'] for row in result['rows'] if row['profile'] == profile)) for profile in v.PROFILES}
    write(DEST / 'original_return_summary.json', {
        'complete': True, 'results_sha256': RESULT_PIN, 'protocol_sha256': PROTOCOL_PIN,
        'archive_sha256': imported['archive_sha256'], 'archive_bytes': imported['bytes'],
        'local_audit_seconds': audit['seconds'], 'artifact_checks': len(result['artifacts_sha256']),
        'validation_references': len(refs), 'validation_cases': len(rows), 'branch_counts': branch_counts,
        'false_clear_routes': [row['id'] for row in result['rows'] if row['profile'] == 'clear' and row['decision']['branch'] != 'retained_dgp_v2'],
        'preservation': result['preservation'], 'original_summaries': result['summaries'],
        'production_promoted': False, 'scientific_gates_changed': False})
    print({'complete': True, 'diagnostic_cases': 520, 'MSE_SSIM_failed_checks': failures,
           'degraded': summaries['degraded'], 'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    analyze()

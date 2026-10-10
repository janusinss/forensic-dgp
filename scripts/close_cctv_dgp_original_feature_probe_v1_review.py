"""Saved-pixel review closure and saved-gradient arithmetic, with no neural calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm'
RETURN = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm_return'
OUT = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1'
PIN = 'ccc30870959c8f4f4f448c1d1e06bb0ae0403d450b81143c6bcb9032d122e94b'
NOTES = {
    5: 'Decoder changes remain small, including mouth/nose contrast; blurred glasses are not convincingly restored. Strong feature/joint steps smear clear frames, face and hair with color wash and edge ghosts.',
    6: 'Decoder retains the coarse stubble and mouth without convincing eye/detail recovery. Strong feature/joint steps wash the clear face pink, introduce colored edge bands and soften facial outline and moustache.',
    7: 'Decoder retains coarse eyes and open mouth with modest changes. Strong feature/joint steps blur clear eyes, mouth, hair and outline; degraded outputs stay soft.',
    8: 'Decoder retains baseline coarse eyes, nose and mouth with minor changes. Strong feature/joint steps soften clear facial features and the pink collar boundary, changing appearance without useful detail.',
    9: 'Decoder stays close to the baby-face baseline with no convincing clarity gain. Strong feature/joint steps blur clear eyes and open mouth and add colored boundary bands; degraded landmarks remain diffuse.',
    10: 'Decoder retains coarse smile and the visible hand with small contrast changes. Strong feature/joint steps soften clear eyelids, smile and hair and add pink/green color wash; degraded facial detail is unresolved.',
    11: 'Decoder keeps baseline smile, forehead hand and hair, without convincing degraded clarity gain. Strong feature/joint steps soften the clear smile, teeth, eyelids and hair boundaries; image borders also change color.',
    12: 'Decoder retains baseline mouth and visible eye but does not recover blurred landmarks. Strong feature/joint steps soften clear eye, lips and hair boundaries, with colored edge wash. Existing skin/detail softness remains.',
    13: 'Decoder changes are small around lips and nose and retain the head wrap. Strong feature/joint steps blur clear face, eyelids and wrap boundary and shift color; degraded facial features remain soft.',
    14: 'Decoder preserves baseline coarse smile and eyes with no convincing additional clarity. Strong feature/joint steps soften clear eyes, teeth and face outline and introduce boundary color bands.',
    15: 'Decoder retains the coarse broad smile with small changes. Strong feature/joint steps soften clear eyelids, nose and smile and wash facial color; degraded eye and tooth detail remains diffuse.',
    16: 'Decoder retains baseline face and hair without convincing degraded detail. Strong feature/joint steps wash and blur clear eyes, lips, stubble and facial outline, with green/pink edge ghosts.',
    17: 'Decoder retains baseline lips, eye placement and ordinary hair with small changes. Strong feature/joint steps soften clear eyes, lips and hair detail and introduce colored boundary ghosts; degraded landmarks stay blurred.',
    18: 'Decoder keeps the coarse smile and eye placement. Strong feature/joint steps blur the clear eyes, smile and hair boundary and add color wash; degraded detail shows no convincing useful gain.',
    19: 'Decoder retains baseline smile and short hair with small changes. Strong feature/joint steps soften clear eyes, teeth, nose and hair boundaries and alter face/background color; degraded detail remains soft.',
    20: 'Decoder retains baseline eye, lip and hair placement with small changes. Strong feature/joint steps soften the clear face and hair, change color and add edge bands; degraded eye/mouth structure stays diffuse.',
}


def read(q): return json.loads(q.read_text(encoding='utf-8'))
def sha(q):
    with q.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(q, x):
    with q.open('x', encoding='utf-8', newline='\n') as f: json.dump(x, f, indent=2, allow_nan=False); f.write('\n')


def main():
    start = time.monotonic(); assert sha(BUNDLE/'protocol.json') == PIN
    p = read(BUNDLE/'protocol.json'); result = read(RETURN/'outputs/results.json')
    audit_path = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_independent_audit_r2.json'
    audit = read(audit_path); assert audit['complete'] and audit['protocol_sha256'] == PIN
    assert result['protocol_sha256'] == PIN and result['optimizer_updates'] == result['epochs'] == 0
    gallery = OUT/'gallery'; manifest = read(gallery/'gallery_manifest.json')
    assert manifest['results_sha256'] == sha(RETURN/'outputs/results.json') and manifest['protocol_sha256'] == PIN
    for n, d in manifest['source_bindings'].items(): assert sha(ROOT/n) == d, n
    models = set(); cells = 0
    for page in manifest['pages']:
        assert sha(gallery/page['file']) == page['sha256']
        with Image.open(gallery/page['file']) as image:
            assert list(image.size) == page['size']
            for cell in page['cells']:
                with Image.open(ROOT/cell['file']) as source:
                    assert source.size == (256, 256)
                    assert image.crop(cell['box']).tobytes() == source.convert('RGB').tobytes()
                cells += 1
                if '_vm_return/outputs/' in cell['file']: models.add(cell['file'])
    assert cells == 1800 and len(models) == 1000 and len(manifest['pages']) == 60
    progress = read(OUT/'visual_review_progress.json')
    assert [r['group_index'] for r in progress['viewed_groups']] == [1, 2, 3, 4]
    groups = progress['viewed_groups'][:]
    for number in range(5, 21):
        pages = [r for r in manifest['pages'] if r['file'].startswith(f'{number:02d}-')]
        assert len(pages) == 3 and len({tuple(r['case_ids']) for r in pages}) == 1
        groups.append({'group_index': number, 'pages': [r['file'] for r in pages],
            'reference': pages[0]['reference'], 'cohort': pages[0]['cohort'], 'case_ids': pages[0]['case_ids'],
            'all_three_scopes_and_scales_viewed': True, 'note': NOTES[number]})
    assert set(cid for g in groups for cid in g['case_ids']) == {c['id'] for c in p['cases']}
    visual = {'complete': True, 'reviewer': 'Primary assistant development review',
        'independent_final_reviewer': False, 'protocol_sha256': PIN, 'results_sha256': sha(RETURN/'outputs/results.json'),
        'independent_evidence_audit_sha256': sha(audit_path), 'gallery_manifest_sha256': sha(gallery/'gallery_manifest.json'),
        'all60_pages_viewed_at_original_resolution': True, 'unique_model_outputs_reviewed': 1000,
        'exact256_cells_independently_verified': cells, 'page_pixel_compositions_exact': True,
        'groups': groups, 'remaining_group_indices': [], 'decision': 'All nine directions remain unqualified; no app promotion or epoch continuation.',
        'native_or_DEV_or_final_used': False, 'model_qualification': False, 'goal_complete': False}
    write(OUT/'visual_review.json', visual)
    rows = []
    for trial in result['trial_summaries']:
        row = {'variant': trial['variant'], 'comparisons': {}}
        for cohort, stages in trial['comparisons'].items():
            row['comparisons'][cohort] = {}
            for stage, gate in stages.items():
                assert gate['pass'] is False
                row['comparisons'][cohort][stage] = {
                    'structure_gain_percent': 100*gate['relative_feature_gain'],
                    'preservation_failures': len(gate['preservation_failures']),
                    'failure_metric_counts': dict(Counter(r['metric'] for r in gate['preservation_failures'])),
                    'failure_membership': [{'group': r['group'], 'metric': r['metric']} for r in gate['preservation_failures']],
                    'source_feature_gains_percent': {k: 100*v for k, v in gate['source_feature_gains'].items()},
                    'brightness_gain_fraction': gate['brightness_gain_fraction'], 'pass': False}
        rows.append(row)
    with np.load(RETURN/'outputs/aggregate_gradients.npz', allow_pickle=False) as arrays:
        aggregate = {k: arrays[k].copy() for k in p['terms']}
    weights = np.load(RETURN/'outputs/initial_parameters.npy', allow_pickle=False).astype(np.float64)
    masks = {}
    for scope in ['decoder_control', 'feature_only', 'joint']:
        mask = np.zeros(len(weights), bool)
        for r in p['parameter_layout']:
            if scope == 'joint' or r['partition'] == scope: mask[r['start']:r['end']] = True
        masks[scope] = mask
    stats = {}
    for scope, mask in masks.items():
        gradients = {k: v[mask] for k, v in aggregate.items()}
        norms = {k: float(np.linalg.norm(v)) for k, v in gradients.items()}
        cosines = {a+'/'+b: float(gradients[a]@gradients[b]/(norms[a]*norms[b]))
            for i, a in enumerate(p['terms']) for b in p['terms'][i+1:]}
        stats[scope] = {'gradient_norms': norms, 'cosines': cosines,
            'selected_weight_L2': float(np.linalg.norm(weights[mask]))}
    d = -np.where(masks['decoder_control'], aggregate[p['terms'][0]], 0.)
    f = -np.where(masks['feature_only'], aggregate[p['terms'][2]], 0.)
    d /= np.linalg.norm(d); f /= np.linalg.norm(f)
    adverse = float(aggregate[p['terms'][2]]@d); protection = -float(aggregate[p['terms'][2]]@f)
    assert adverse > 0 and protection > 0
    ratio = adverse/protection
    directions = {'feature_identity_control': f, 'balanced_ratio1': d+ratio*f, 'balanced_ratio2': d+2*ratio*f}
    predictions = {}
    for name, direction in directions.items():
        derivatives = {k: float(v@direction) for k, v in aggregate.items()}
        assert derivatives[p['terms'][0]] < 0 and derivatives[p['terms'][1]] < 0
        assert derivatives[p['terms'][2]] <= 1e-15
        predictions[name] = {'directional_derivatives_per_unit_scale': derivatives,
            'finite_outputs_tested': False, 'group_preservation_proven': False}
    analysis = {'complete': True, 'protocol_sha256': PIN,
        'aggregate_gradients_sha256': sha(RETURN/'outputs/aggregate_gradients.npz'),
        'initial_parameters_sha256': sha(RETURN/'outputs/initial_parameters.npy'),
        'independent_evidence_audit_sha256': sha(audit_path), 'visual_review_sha256': sha(OUT/'visual_review.json'),
        'trial_comparisons': rows, 'partition_statistics': stats,
        'decoder_identity_adverse_derivative': adverse, 'feature_identity_protective_derivative': protection,
        'feature_to_decoder_displacement_ratio_for_mean_identity_neutrality': ratio,
        'prospective_direction_predictions': predictions,
        'interpretation': 'Loss magnitudes and directions differ across the original partitions. Mean first-order predictions do not establish finite or per-group preservation.',
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
        'model_qualification': False, 'goal_complete': False, 'seconds': time.monotonic()-start}
    assert time.monotonic()-start < 120
    write(OUT/'partition_analysis.json', analysis)
    print({'complete': True, 'visual_outputs': 1000, 'pixel_cells': cells,
        'all_nine_trials_unqualified': True, 'balanced_ratio': ratio, 'seconds': analysis['seconds']})


if __name__ == '__main__': main()

"""Compare hash-verified fixed V30 previews; no inference or qualification."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30'
RETURNED = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'
V29 = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_broader_mean_v30_failure_review_v1'
PIN = 'b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    start = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json'
    audit = read(audit_path) if audit_path.exists() else None
    if audit:
        assert audit['complete'] and audit['VM_failure_retained']
        assert not audit['training_completed800'] and not audit['necessary_capacity_pass']
        assert [r['update'] for r in audit['snapshots_audited']] == [0, 50]
    imported_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return_import.json'
    imported = read(imported_path)
    assert imported['complete'] and imported['archive_sha256'] == '19294b086020378df2c9157de6954c27ae3d3e2008d375cf91bb4cc15c43af69'
    assert imported['members'] == 27622
    assert not OUT.exists(), 'Preserve existing review'
    assert sha(BUNDLE / 'protocol.json') == PIN
    if audit: assert PIN == audit['protocol_sha256']
    p = read(BUNDLE / 'protocol.json')
    base, stopped = [read(RETURNED / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    early = read(RETURNED / 'outputs/early_structure_stop.json')
    failure = read(RETURNED / 'outputs/failure.json')
    assert failure['optimizer_updates'] == early['update'] == 50 and not early['pass']
    assert failure['cause'] == 'No one-percent early structural gain; retain stop'
    old_audit_path = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json'
    old_audit = read(old_audit_path)
    assert old_audit['complete'] and old_audit['training_completed800']
    schedule = read(BUNDLE / 'schedule.json')['batches']
    seen_indices = {i for batch in schedule[:50] for i in batch}
    assert len(seen_indices) == 250
    exposed = {p['case_rows'][i]['id'] for i in seen_indices}
    touched = {p['case_rows'][i]['source_person_or_reference'] for i in seen_indices}
    before = {r['id']: r for r in base['rows']}
    after = {r['id']: r for r in stopped['rows']}
    preview = p['initial_proof_case_rows']
    assert [c['id'] for c in preview] == p['preview_case_ids'] and len(preview) == 50
    OUT.mkdir()
    plan = {
        'format': 'V30-stopped50-fixed-TRAIN-preview-review-v1',
        'scope': 'All50 previously fixed photographic TRAIN previews; failed capacity candidate, not native or held-out evidence',
        'case_ids': p['preview_case_ids'],
        'selection': 'Same50 original proof cases, fixed before V30 optimization; both runs at update50',
        'columns': ['Input', 'Unchanged DGP', 'V29 update50', 'V30 stopped50', 'Paired TRAIN target'],
        'regions': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
        'criteria': [
            'Inspect every region together and accept softness when useful structure is preserved.',
            'Separate brightness changes from edge or contour clarity.',
            'Record changes to clear glasses, hair, gaze, expression and other visible appearance.',
            'These are paired synthetic degradations of photographs; targets are not native CCTV references.',
            'Visual impressions never waive the failed1% stop or qualify the stopped checkpoint.',
        ],
        'audit_sha256': sha(audit_path) if audit else None, 'protocol_sha256': PIN,
        'verified_import_sha256': sha(imported_path), 'R1_audit_pending_at_preparation': audit is None,
        'cell_size': [256, 256], 'image_processing_or_resize': False,
        'native_or_reserved_used': False, 'optimizer_updates': 0,
        'app_promotion': False, 'independent_final_review': False, 'goal_complete': False,
    }
    write(OUT / 'plan.json', plan)
    bindings = {}
    def bind(path):
        bindings[Path(path).relative_to(ROOT).as_posix()] = sha(path)
    for path in [imported_path, old_audit_path, BUNDLE / 'protocol.json', BUNDLE / 'schedule.json',
                 RETURNED / 'outputs/failure.json', RETURNED / 'outputs/early_structure_stop.json',
                 RETURNED / 'outputs/update0/metrics.json', RETURNED / 'outputs/update50/metrics.json',
                 RETURNED / 'outputs/execution_receipt.json', RETURNED / 'outputs/gradient_summary.json',
                 V29 / 'outputs/update0/metrics.json', V29 / 'outputs/update50/metrics.json']:
        bind(path)
    if audit: bind(audit_path)
    buckets = {}
    for label, ids in [('all_TRAIN', set(before)), ('exposed_by50_TRAIN', exposed),
                       ('not_yet_exposed_TRAIN', set(before) - exposed), ('fixed50_TRAIN', set(p['preview_case_ids']))]:
        ids = [cid for cid in before if cid in ids and before[cid]['profile'] != 'clear']
        m0 = float(np.mean([before[cid]['metrics']['landmark_high_frequency_MSE'] for cid in ids]))
        m50 = float(np.mean([after[cid]['metrics']['landmark_high_frequency_MSE'] for cid in ids]))
        buckets[label] = {'degraded_cases': len(ids), 'baseline_feature_MSE': m0,
                          'stopped50_feature_MSE': m50, 'relative_feature_gain': 1 - m50 / m0,
                          'not_held_out_evaluation': True}
    group_comparison, regressions = {}, []
    for label, old in base['groups'].items():
        current = stopped['groups'][label]
        bad = [key for key in ['MSE', 'SSIM', 'ArcFace_observed_fixed']
               if (current[key] > old[key] + 1e-12 if key == 'MSE' else current[key] < old[key] - 1e-6)]
        regressions.extend({'group': label, 'metric': key} for key in bad)
        group_comparison[label] = {'cases': old['cases'], 'baseline': old, 'stopped50': current,
                                  'feature_gain': 1 - current['landmark_high_frequency_MSE'] / old['landmark_high_frequency_MSE'],
                                  'preservation_regressions': bad}
    assert abs(buckets['all_TRAIN']['relative_feature_gain'] - early['relative_feature_error_gain']) < 1e-12
    old_early = read(V29 / 'outputs/early_structure_stop.json'); bind(V29 / 'outputs/early_structure_stop.json')
    old_metrics = read(V29 / 'outputs/update50/metrics.json')
    old_base = read(V29 / 'outputs/update0/metrics.json')
    # Saved-gradient/checkpoint algebra only: no model construction or autograd.
    arrays = RETURNED / 'outputs/gradient_components.npy'; bind(arrays)
    gradient = np.load(arrays, allow_pickle=False)
    assert gradient.dtype == np.float64 and gradient.shape == (7, 498627) and np.isfinite(gradient).all()
    checkpoints = [RETURNED / 'outputs/update0/dgp_candidate_v30.pth',
                   RETURNED / 'outputs/update50/dgp_candidate_v30.pth', V29 / 'outputs/update50/dgp_candidate_v29.pth']
    states = []
    for path in checkpoints:
        bind(path); states.append(torch.load(path, map_location='cpu', weights_only=True))
    displacement = []
    for state in states[1:]:
        displacement.append(np.concatenate([(state[row['name']] - states[0][row['name']]).numpy().reshape(-1).astype(np.float64)
                                             for row in p['parameter_layout']]))
    d30, d29 = displacement
    norm = lambda a: float(np.linalg.norm(a))
    cosine = lambda a, b: float(a @ b / max(norm(a) * norm(b), 1e-300))
    algebra = {'selected_parameter_values': 498627, 'V30_update50_displacement_L2': norm(d30),
               'V29_update50_displacement_L2': norm(d29), 'displacement_cosine_V30_V29': cosine(d30, d29),
               'V30_descent_cosine_to_initial50_objective': cosine(-d30, gradient.sum(0)),
               'V29_descent_cosine_to_initial50_objective': cosine(-d29, gradient.sum(0)),
               'initial50_is_not_broad_cohort_gradient': True,
               'checkpoint_algebra_does_not_identify_unique_learning_cause': True}
    rows, byref = [], defaultdict(list)
    def rgb(path):
        with Image.open(path) as im:
            assert im.size == (256, 256)
            return np.asarray(im.convert('RGB')).copy()
    for c in preview:
        byref[c['source_person_or_reference']].append(c)
        cid = c['id']
        input_paths = [PARENT / c['input'], RETURNED / f'outputs/update0/{cid}.png',
                       V29 / f'outputs/update50/{cid}.png', RETURNED / f'outputs/update50/{cid}.png', PARENT / c['target']]
        raw_paths = [RETURNED / f'outputs/update{n}/{cid}.npy' for n in [0, 50]]
        for path in input_paths + raw_paths + [PARENT / c['observed']]: bind(path)
        with Image.open(PARENT / c['observed']) as im: mask = np.asarray(im) > 0
        correction = (np.load(raw_paths[1], allow_pickle=False) - np.load(raw_paths[0], allow_pickle=False))[mask].astype(np.float64)
        byte = rgb(input_paths[3]).astype(np.int16) - rgb(input_paths[1]).astype(np.int16)
        rows.append({'id': cid, 'reference': c['source_person_or_reference'], 'source': c['source'], 'profile': c['profile'],
                     'exposed_by50': cid in exposed, 'raw_correction_RMS': float(np.sqrt(np.square(correction).mean())),
                     'postclip_mean_RGB_shift': correction.mean(0).tolist(),
                     'changed_PNG_pixels': int(np.any(byte != 0, axis=2).sum()), 'maximum_PNG_byte_change': int(np.abs(byte).max())})
    sheets, cells_checked = [], 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for index, (ref, cases) in enumerate(byref.items(), 1):
        assert len(cases) == 5
        sheet = Image.new('RGB', (1336, 1516), '#f1f2f4'); draw = ImageDraw.Draw(sheet)
        draw.text((12, 5), 'Paired TRAIN | stopped50 diagnostic | ' + ref + ' | ' + cases[0]['source'], font=font, fill='#111111')
        for column, label in enumerate(plan['columns']): draw.text((12 + column * 264, 30), label, font=font, fill='#111111')
        cells = []
        for row, c in enumerate(cases):
            cid = c['id']; y = 64 + row * 288
            draw.text((12, y), c['profile'] + ' | ' + cid + (' | trained by50' if cid in exposed else ' | unexposed TRAIN'), font=font, fill='#111111')
            paths = [PARENT / c['input'], RETURNED / f'outputs/update0/{cid}.png', V29 / f'outputs/update50/{cid}.png',
                     RETURNED / f'outputs/update50/{cid}.png', PARENT / c['target']]
            for column, path in enumerate(paths):
                pixels = rgb(path); xy = [12 + column * 264, y + 24]
                sheet.paste(Image.fromarray(pixels), tuple(xy))
                cells.append({'case': cid, 'column': column, 'xy': xy, 'source': path.relative_to(ROOT).as_posix(),
                              'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        name = f'sheet_{index:02d}_{ref}.png'; sheet.save(OUT / name)
        with Image.open(OUT / name) as im:
            canvas = np.asarray(im).copy()
        assert canvas.shape == (1516, 1336, 3)
        for cell in cells:
            x, y = cell['xy']; assert np.array_equal(canvas[y:y+256, x:x+256], rgb(ROOT / cell['source'])); cells_checked += 1
        sheets.append({'file': name, 'reference': ref, 'cases': [c['id'] for c in cases], 'cells': cells})
    assert len(rows) == 50 and len(sheets) == 10 and cells_checked == 250
    source_counts = {source: sum(r['id'] in touched and r['source'] == source for r in p['training_references'])
                     for source in sorted({r['source'] for r in p['training_references']})}
    report = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'generator_sha256': sha(Path(__file__)),
              'audit_sha256': sha(audit_path) if audit else None, 'R1_audit_pending_at_preparation': audit is None,
              'verified_import_sha256': sha(imported_path), 'rows': rows, 'sheets': sheets, 'exact_cells_checked': cells_checked,
              'source_bindings_sha256': bindings, 'sheet_sha256': {s['file']: sha(OUT / s['file']) for s in sheets},
              'early_stop': early, 'groups': group_comparison, 'preservation_regressions_at50': regressions,
              'exposure_buckets': buckets, 'unique_cases_optimized_by50': len(exposed),
              'unique_references_touched_by50': len(touched), 'touched_reference_source_counts': source_counts,
              'fixed50_cases_exposed_by50': len(exposed & set(p['preview_case_ids'])),
              'V29_early_stop': old_early, 'V29_fixed50_feature_gain': 1 - old_metrics['groups']['degraded']['landmark_high_frequency_MSE'] / old_base['groups']['degraded']['landmark_high_frequency_MSE'],
              'checkpoint_and_saved_gradient_algebra': algebra,
              'visual_review_pending': True, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
              'native_or_reserved_used': False, 'independent_final_review': False, 'app_promotion': False,
              'goal_complete': False, 'seconds': time.monotonic() - start}
    write(OUT / 'preparation.json', report)
    print(json.dumps({k: report[k] for k in ['complete', 'exact_cells_checked', 'early_stop', 'exposure_buckets', 'checkpoint_and_saved_gradient_algebra', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()

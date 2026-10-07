"""Compare all fixed TRAIN previews after the independent R2 return audit."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
RETURNED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'
PRIOR = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1'
PIN = '87582314eb1313a6914b4940496eeeb246c3023b7b6bf4f84da43e3f5e739d6c'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def rgb(path):
    with Image.open(path) as image:
        assert image.size == (256, 256)
        return np.asarray(image.convert('RGB')).copy()


def main():
    started = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json'
    prior_audit_path = ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json'
    imported_path = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_import.json'
    audit, prior_audit, imported = map(read, [audit_path, prior_audit_path, imported_path])
    assert audit['complete'] and audit['VM_failure_retained'] and prior_audit['complete']
    assert not audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_r2_return.py')
    assert [s['update'] for s in audit['snapshots_audited']] == [0, 50]
    assert audit['local_gradient_calls'] == audit['local_backward_calls'] == audit['local_optimizer_updates'] == 0
    assert imported['complete'] and imported['archive_sha256'] == audit['archive_sha256']
    assert audit['protocol_sha256'] == sha(BUNDLE / 'protocol.json') == PIN
    p = read(BUNDLE / 'protocol.json')
    old_bundle = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
    oldp = read(old_bundle / 'protocol.json')
    assert p['case_rows'] == oldp['case_rows'] and p['preview_case_ids'] == oldp['preview_case_ids']
    assert p['initial_proof_case_rows'] == oldp['initial_proof_case_rows']
    schedule = read(BUNDLE / 'schedule.json')['batches']
    assert schedule == read(old_bundle / 'schedule.json')['batches']
    initial, current = [read(RETURNED / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    old_initial, old_current = [read(PRIOR / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    assert initial['groups'] == old_initial['groups']
    before, after = [{r['id']: r for r in value['rows']} for value in [initial, current]]
    exposed = {p['case_rows'][i]['id'] for batch in schedule[:50] for i in batch}
    references = {p['case_rows'][i]['source_person_or_reference'] for batch in schedule[:50] for i in batch}
    assert len(exposed) == 250 and len(references) == 50
    early = read(RETURNED / 'outputs/early_structure_stop.json')
    failure = read(RETURNED / 'outputs/failure.json')
    assert failure['optimizer_updates'] == early['update'] == 50 and not early['pass']
    assert failure['cause'] == 'No one-percent early structural gain; retain stop'
    assert not OUT.exists(), 'Keep every prior review unchanged'
    OUT.mkdir()
    plan = {'format': 'V32-r2-stopped50-fixed-TRAIN-review-v1',
            'case_ids': p['preview_case_ids'], 'cell_size': [256, 256],
            'selection': 'All50 initial proof cases fixed before V30, V31 and V32; no output selection',
            'columns': ['Input', 'Retained DGP', 'V31 stopped50', 'V32 r2 stopped50', 'Paired TRAIN target'],
            'regions': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
            'criteria': ['Preserve all visible features, accepting some softness.',
                         'Distinguish extra facial definition from brightness and smoothing.',
                         'Preserve clear glasses, hair, gaze, expression and visible appearance.',
                         'Photographic TRAIN evidence does not establish native CCTV or hidden identity accuracy.',
                         'No visual impression waives the failed1% early structure requirement.'],
            'protocol_sha256': PIN, 'independent_return_audit_sha256': sha(audit_path),
            'image_resizing_or_display_processing': False, 'neural_or_gradient_calls': 0,
            'optimizer_updates': 0, 'native_or_reserved_used': False, 'app_promotion': False,
            'independent_final_review': False, 'goal_complete': False}
    write(OUT / 'plan.json', plan)
    bindings = {}

    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)

    for path in [audit_path, prior_audit_path, imported_path, BUNDLE / 'protocol.json',
                 BUNDLE / 'schedule.json', old_bundle / 'protocol.json', old_bundle / 'schedule.json',
                 RETURNED / 'outputs/failure.json', RETURNED / 'outputs/early_structure_stop.json',
                 RETURNED / 'outputs/execution_receipt.json', RETURNED / 'outputs/gradient_preflight.json',
                 RETURNED / 'outputs/update0/metrics.json', RETURNED / 'outputs/update50/metrics.json',
                 PRIOR / 'outputs/update0/metrics.json', PRIOR / 'outputs/update50/metrics.json']:
        bind(path)
    groups, regressions = {}, []
    for key, baseline in initial['groups'].items():
        stopped, prior = current['groups'][key], old_current['groups'][key]
        bad = [metric for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']
               if (stopped[metric] > baseline[metric] + 1e-12 if metric == 'MSE'
                   else stopped[metric] < baseline[metric] - 1e-6)]
        regressions.extend({'group': key, 'metric': metric} for metric in bad)
        groups[key] = {'cases': baseline['cases'], 'baseline': baseline, 'V31_stopped50': prior,
                       'V32_r2_stopped50': stopped,
                       'V31_feature_gain': 1 - prior['landmark_high_frequency_MSE'] / baseline['landmark_high_frequency_MSE'],
                       'V32_r2_feature_gain': 1 - stopped['landmark_high_frequency_MSE'] / baseline['landmark_high_frequency_MSE'],
                       'V32_r2_preservation_regressions': bad}
    assert len(groups) == 17 and abs(groups['degraded']['V32_r2_feature_gain'] - early['relative_feature_error_gain']) <= 1e-12
    buckets = {}
    for label, ids in [('all_TRAIN', set(before)), ('exposed_by50_TRAIN', exposed),
                       ('not_yet_exposed_TRAIN', set(before) - exposed), ('fixed50_TRAIN', set(p['preview_case_ids']))]:
        ids = [cid for cid in before if cid in ids and before[cid]['profile'] != 'clear']
        b, a = [float(np.mean([data[cid]['metrics']['landmark_high_frequency_MSE'] for cid in ids]))
                for data in [before, after]]
        buckets[label] = {'degraded_cases': len(ids), 'baseline_feature_MSE': b,
                          'stopped50_feature_MSE': a, 'relative_feature_gain': 1 - a / b,
                          'TRAIN_only_not_evaluation': True}
    rows, by_reference = [], defaultdict(list)
    for case in p['initial_proof_case_rows']:
        cid = case['id']
        by_reference[case['source_person_or_reference']].append(case)
        paths = [PARENT / case['input'], RETURNED / f'outputs/update0/{cid}.png',
                 PRIOR / f'outputs/update50/{cid}.png', RETURNED / f'outputs/update50/{cid}.png',
                 PARENT / case['target']]
        raw_paths = [RETURNED / f'outputs/update{n}/{cid}.npy' for n in [0, 50]]
        for path in paths + raw_paths + [PARENT / case['observed']]:
            bind(path)
        with Image.open(PARENT / case['observed']) as image:
            mask = np.asarray(image) > 0
        assert np.array_equal(rgb(paths[1]), rgb(PRIOR / f'outputs/update0/{cid}.png'))
        correction = (np.load(raw_paths[1], allow_pickle=False) - np.load(raw_paths[0], allow_pickle=False))[mask].astype(np.float64)
        byte_change = rgb(paths[3]).astype(np.int16) - rgb(paths[1]).astype(np.int16)
        rows.append({'id': cid, 'reference': case['source_person_or_reference'], 'source': case['source'],
                     'profile': case['profile'], 'exposed_by50': cid in exposed,
                     'raw_correction_RMS': float(np.sqrt(np.square(correction).mean())),
                     'postclip_mean_RGB_shift': correction.mean(0).tolist(),
                     'changed_PNG_pixels': int(np.any(byte_change != 0, axis=2).sum()),
                     'maximum_PNG_byte_change': int(np.abs(byte_change).max()),
                     'V32_r2_feature_MSE': after[cid]['metrics']['landmark_high_frequency_MSE']})
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    sheets, cells_checked = [], 0
    for index, (reference, cases) in enumerate(by_reference.items(), 1):
        assert len(cases) == 5
        sheet = Image.new('RGB', (1336, 1516), '#f1f2f4')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 5), 'Paired TRAIN | stopped50 | ' + reference + ' | ' + cases[0]['source'], font=font, fill='#111111')
        for column, label in enumerate(plan['columns']):
            draw.text((12 + column * 264, 30), label, font=font, fill='#111111')
        cells = []
        for row, case in enumerate(cases):
            cid, y = case['id'], 64 + row * 288
            draw.text((12, y), case['profile'] + ' | ' + cid + (' | trained by50' if cid in exposed else ' | unexposed TRAIN'), font=font, fill='#111111')
            paths = [PARENT / case['input'], RETURNED / f'outputs/update0/{cid}.png',
                     PRIOR / f'outputs/update50/{cid}.png', RETURNED / f'outputs/update50/{cid}.png', PARENT / case['target']]
            for column, path in enumerate(paths):
                pixels, xy = rgb(path), [12 + column * 264, y + 24]
                sheet.paste(Image.fromarray(pixels), tuple(xy))
                cells.append({'case': cid, 'column': column, 'xy': xy,
                              'source': path.relative_to(ROOT).as_posix(),
                              'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        name = f'sheet_{index:02d}_{reference}.png'
        sheet.save(OUT / name)
        with Image.open(OUT / name) as image:
            canvas = np.asarray(image).copy()
        for cell in cells:
            x, y = cell['xy']
            assert np.array_equal(canvas[y:y + 256, x:x + 256], rgb(ROOT / cell['source']))
            cells_checked += 1
        sheets.append({'file': name, 'reference': reference, 'cases': [c['id'] for c in cases], 'cells': cells})
    assert len(rows) == 50 and len(sheets) == 10 and cells_checked == 250
    write(OUT / 'preparation.json', {'complete': True, 'generator_sha256': sha(Path(__file__)),
          'plan_sha256': sha(OUT / 'plan.json'), 'audit_sha256': sha(audit_path), 'rows': rows, 'sheets': sheets,
          'source_bindings_sha256': bindings, 'sheet_sha256': {s['file']: sha(OUT / s['file']) for s in sheets},
          'groups': groups, 'preservation_regressions_at50': regressions, 'exposure_buckets': buckets,
          'early_stop': early, 'exact_cells_checked': cells_checked, 'unique_cases_optimized_by50': len(exposed),
          'unique_references_touched_by50': len(references), 'fixed50_cases_exposed_by50': len(exposed & set(p['preview_case_ids'])),
          'visual_review_pending': True, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
          'native_or_reserved_used': False, 'independent_final_review': False, 'app_promotion': False,
          'goal_complete': False, 'seconds': time.monotonic() - started})
    print(json.dumps({'complete': True, 'exact_cells_checked': cells_checked,
                      'relative_feature_gain': early['relative_feature_error_gain'], 'preservation_regressions': regressions}))


if __name__ == '__main__':
    main()

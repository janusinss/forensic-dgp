"""Fixed TRAIN preview and exposure comparison from independently audited pixels."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
RETURNED = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
V30 = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_profile_batches_v31_failure_review_v1'
PIN = 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def rgb(path):
    with Image.open(path) as image:
        assert image.size == (256, 256)
        return np.asarray(image.convert('RGB')).copy()


def main():
    start = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['VM_failure_retained']
    assert not audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert [r['update'] for r in audit['snapshots_audited']] == [0, 50]
    assert not audit['native_or_reserved_used'] and audit['local_gradient_calls'] == 0
    assert audit['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_profile_batches_v31_return.py')
    imported_path = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return_import.json'
    imported = read(imported_path)
    assert imported['complete'] and imported['archive_sha256'] == audit['archive_sha256']
    assert sha(BUNDLE / 'protocol.json') == audit['protocol_sha256'] == PIN
    p = read(BUNDLE / 'protocol.json')
    oldp = read(ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30/protocol.json')
    old_audit_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json'
    old_audit = read(old_audit_path)
    assert old_audit['complete'] and old_audit['VM_failure_retained']
    assert p['case_rows'] == oldp['case_rows'] and p['preview_case_ids'] == oldp['preview_case_ids']
    assert p['initial_proof_case_rows'] == oldp['initial_proof_case_rows']
    base, stopped = [read(RETURNED / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    oldbase, oldstop = [read(V30 / f'outputs/update{n}/metrics.json') for n in [0, 50]]
    assert base['groups'] == oldbase['groups'], 'Same original baseline required for batch-only comparison'
    before, after, oldafter = [{r['id']: r for r in value['rows']} for value in [base, stopped, oldstop]]
    schedule = read(BUNDLE / 'schedule.json')['batches']
    oldschedule = read(ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30/schedule.json')['batches']
    selected = {i for batch in schedule[:50] for i in batch}
    oldselected = {i for batch in oldschedule[:50] for i in batch}
    exposed = {p['case_rows'][i]['id'] for i in selected}
    touched = {p['case_rows'][i]['source_person_or_reference'] for i in selected}
    oldexposed = {p['case_rows'][i]['id'] for i in oldselected}
    assert len(selected) == len(oldselected) == 250 and len(touched) == 50
    preview = p['initial_proof_case_rows']
    assert [r['id'] for r in preview] == p['preview_case_ids'] and len(preview) == 50
    early = read(RETURNED / 'outputs/early_structure_stop.json')
    failure = read(RETURNED / 'outputs/failure.json')
    assert failure['optimizer_updates'] == early['update'] == 50 and not early['pass']
    assert failure['cause'] == 'No one-percent early structural gain; retain stop'
    assert not OUT.exists(), 'Preserve prior reviews'
    OUT.mkdir()
    plan = {'format': 'V31-stopped50-fixed-TRAIN-review-v1',
            'scope': 'All50 fixed photographic TRAIN previews; stopped capacity candidate, not independent evaluation',
            'case_ids': p['preview_case_ids'],
            'selection': 'Same50 original proof cases fixed before V30 and V31; both candidates stopped at50',
            'columns': ['Input', 'Retained DGP', 'V30 stopped50', 'V31 stopped50', 'Paired TRAIN target'],
            'regions': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
            'criteria': ['Preserve all visible features; accept softness when usable structure remains.',
                         'Separate contrast changes from eye, nose, lip and outline definition.',
                         'Preserve clear glasses, hair, gaze, expression and other visible appearance.',
                         'Paired synthetic TRAIN photos do not establish native CCTV or hidden identity accuracy.',
                         'Visual impressions never waive the retained numerical stop.'],
            'protocol_sha256': PIN, 'audit_sha256': sha(audit_path), 'cell_size': [256, 256],
            'image_processing_or_resize': False, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
            'native_or_reserved_used': False, 'app_promotion': False, 'independent_final_review': False,
            'goal_complete': False}
    write(OUT / 'plan.json', plan)
    bindings = {}

    def bind(path):
        bindings[Path(path).relative_to(ROOT).as_posix()] = sha(path)

    for path in [audit_path, old_audit_path, imported_path, BUNDLE / 'protocol.json', BUNDLE / 'schedule.json',
                 ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30/protocol.json',
                 ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30/schedule.json',
                 RETURNED / 'outputs/failure.json', RETURNED / 'outputs/early_structure_stop.json',
                 RETURNED / 'outputs/execution_receipt.json', RETURNED / 'outputs/gradient_preflight.json',
                 RETURNED / 'outputs/gradient_summary.json', RETURNED / 'outputs/update0/metrics.json',
                 RETURNED / 'outputs/update50/metrics.json', V30 / 'outputs/update0/metrics.json',
                 V30 / 'outputs/update50/metrics.json']:
        bind(path)
    buckets = {}
    for label, ids in [('all_TRAIN', set(before)), ('exposed_by50_TRAIN', exposed),
                       ('not_yet_exposed_TRAIN', set(before) - exposed), ('fixed50_TRAIN', set(p['preview_case_ids']))]:
        ids = [cid for cid in before if cid in ids and before[cid]['profile'] != 'clear']
        m0, m50 = [float(np.mean([values[cid]['metrics']['landmark_high_frequency_MSE'] for cid in ids]))
                    for values in [before, after]]
        buckets[label] = {'degraded_cases': len(ids), 'baseline_feature_MSE': m0,
                          'stopped50_feature_MSE': m50, 'relative_feature_gain': 1 - m50 / m0,
                          'not_held_out_evaluation': True}
    assert abs(buckets['all_TRAIN']['relative_feature_gain'] - early['relative_feature_error_gain']) <= 1e-12
    comparisons, regressions = {}, []
    for label, old in base['groups'].items():
        current, prior = stopped['groups'][label], oldstop['groups'][label]
        bad = [key for key in ['MSE', 'SSIM', 'ArcFace_observed_fixed']
               if (current[key] > old[key] + 1e-12 if key == 'MSE' else current[key] < old[key] - 1e-6)]
        regressions.extend({'group': label, 'metric': key} for key in bad)
        comparisons[label] = {'cases': old['cases'], 'baseline': old, 'V30_stopped50': prior,
                              'V31_stopped50': current,
                              'V30_feature_gain': 1 - prior['landmark_high_frequency_MSE'] / old['landmark_high_frequency_MSE'],
                              'V31_feature_gain': 1 - current['landmark_high_frequency_MSE'] / old['landmark_high_frequency_MSE'],
                              'V31_preservation_regressions': bad}
    rows, byref = [], defaultdict(list)
    for c in preview:
        byref[c['source_person_or_reference']].append(c)
        cid = c['id']
        paths = [PARENT / c['input'], RETURNED / f'outputs/update0/{cid}.png',
                 V30 / f'outputs/update50/{cid}.png', RETURNED / f'outputs/update50/{cid}.png', PARENT / c['target']]
        raw_paths = [RETURNED / f'outputs/update{n}/{cid}.npy' for n in [0, 50]]
        for path in paths + raw_paths + [PARENT / c['observed']]:
            bind(path)
        assert np.array_equal(rgb(paths[1]), rgb(V30 / f'outputs/update0/{cid}.png'))
        with Image.open(PARENT / c['observed']) as image:
            mask = np.asarray(image) > 0
        correction = (np.load(raw_paths[1], allow_pickle=False) - np.load(raw_paths[0], allow_pickle=False))[mask].astype(np.float64)
        byte = rgb(paths[3]).astype(np.int16) - rgb(paths[1]).astype(np.int16)
        rows.append({'id': cid, 'reference': c['source_person_or_reference'], 'source': c['source'],
                     'profile': c['profile'], 'V31_exposed_by50': cid in exposed, 'V30_exposed_by50': cid in oldexposed,
                     'raw_correction_RMS': float(np.sqrt(np.square(correction).mean())),
                     'postclip_mean_RGB_shift': correction.mean(0).tolist(),
                     'changed_PNG_pixels': int(np.any(byte != 0, axis=2).sum()),
                     'maximum_PNG_byte_change': int(np.abs(byte).max()),
                     'V30_feature_MSE': oldafter[cid]['metrics']['landmark_high_frequency_MSE'],
                     'V31_feature_MSE': after[cid]['metrics']['landmark_high_frequency_MSE']})
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    sheets, cells_checked = [], 0
    for index, (reference, cases) in enumerate(byref.items(), 1):
        assert len(cases) == 5
        sheet = Image.new('RGB', (1336, 1516), '#f1f2f4')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 5), 'Paired TRAIN | stopped50 comparison | ' + reference + ' | ' + cases[0]['source'], font=font, fill='#111111')
        for column, label in enumerate(plan['columns']):
            draw.text((12 + column * 264, 30), label, font=font, fill='#111111')
        cells = []
        for row, c in enumerate(cases):
            cid, y = c['id'], 64 + row * 288
            draw.text((12, y), c['profile'] + ' | ' + cid + (' | V31 trained by50' if cid in exposed else ' | unexposed TRAIN'), font=font, fill='#111111')
            paths = [PARENT / c['input'], RETURNED / f'outputs/update0/{cid}.png',
                     V30 / f'outputs/update50/{cid}.png', RETURNED / f'outputs/update50/{cid}.png', PARENT / c['target']]
            for column, path in enumerate(paths):
                pixels = rgb(path)
                xy = [12 + column * 264, y + 24]
                sheet.paste(Image.fromarray(pixels), tuple(xy))
                cells.append({'case': cid, 'column': column, 'xy': xy,
                              'source': path.relative_to(ROOT).as_posix(),
                              'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        name = f'sheet_{index:02d}_{reference}.png'
        sheet.save(OUT / name)
        with Image.open(OUT / name) as image:
            canvas = np.asarray(image).copy()
        assert canvas.shape == (1516, 1336, 3)
        for cell in cells:
            x, y = cell['xy']
            assert np.array_equal(canvas[y:y+256, x:x+256], rgb(ROOT / cell['source']))
            cells_checked += 1
        sheets.append({'file': name, 'reference': reference, 'cases': [c['id'] for c in cases], 'cells': cells})
    assert len(rows) == 50 and len(sheets) == 10 and cells_checked == 250
    report = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'generator_sha256': sha(Path(__file__)),
              'audit_sha256': sha(audit_path), 'verified_import_sha256': sha(imported_path),
              'rows': rows, 'sheets': sheets, 'exact_cells_checked': cells_checked,
              'source_bindings_sha256': bindings, 'sheet_sha256': {s['file']: sha(OUT / s['file']) for s in sheets},
              'early_stop': early, 'groups': comparisons, 'preservation_regressions_at50': regressions,
              'exposure_buckets': buckets, 'V31_unique_cases_optimized_by50': len(exposed),
              'V31_unique_references_touched_by50': len(touched),
              'V31_clear_cases_optimized_by50': sum(before[cid]['profile'] == 'clear' for cid in exposed),
              'V30_unique_references_touched_by50': len({p['case_rows'][i]['source_person_or_reference'] for i in oldselected}),
              'V30_clear_cases_optimized_by50': sum(before[cid]['profile'] == 'clear' for cid in oldexposed),
              'fixed50_cases_exposed_by50': len(exposed & set(p['preview_case_ids'])),
              'visual_review_pending': True, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
              'native_or_reserved_used': False, 'independent_final_review': False, 'app_promotion': False,
              'goal_complete': False, 'seconds': time.monotonic() - start}
    assert time.monotonic() - start < 120
    write(OUT / 'preparation.json', report)
    print(json.dumps({k: report[k] for k in ['complete', 'exact_cells_checked', 'early_stop',
                      'exposure_buckets', 'preservation_regressions_at50', 'fixed50_cases_exposed_by50', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()

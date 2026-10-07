"""Exact-pixel V27 stopped-output review sheets; no neural or training calls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import audit_cctv_dgp_feature_skips_v27 as a

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_skips_v27_saved_output_review'
RETURNED = ROOT / 'outputs/cctv_dgp_feature_skips_v27_return'


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    start = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_feature_skips_v27_independent_audit.json'
    audit = a.read(audit_path)
    a.require(audit['complete'] and audit['complete_snapshot_updates'] == [0, 50] and
              audit['early_structure_stop']['pass'] is False, 'Independently audited stopped V27 required')
    a.require(not OUT.exists(), 'Preserve prior review artifacts')
    p = a.verify_bundle(a.BUNDLE)
    OUT.mkdir()
    by_reference = defaultdict(list)
    rows, bindings = [], {str(audit_path.relative_to(ROOT)).replace('\\', '/'): a.sha(audit_path)}
    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = a.sha(path)
    for case in p['cases']:
        by_reference[case['source_person_or_reference']].append(case)
        original_path = RETURNED / 'outputs/update0' / (case['id'] + '.png')
        result_path = RETURNED / 'outputs/update50' / (case['id'] + '.png')
        raw_path = RETURNED / 'outputs/update50' / (case['id'] + '.npy')
        for path in (original_path, result_path, raw_path, a.BUNDLE / case['raw_dgp'],
                     a.BUNDLE / case['input'], a.BUNDLE / case['target'], a.BUNDLE / case['observed']):
            bind(path)
        baseline, prediction = a.rgb(original_path), a.rgb(result_path)
        support = a.observed(a.BUNDLE / case['observed'])
        delta = (a.raw_rgb(raw_path) - a.raw_rgb(a.BUNDLE / case['raw_dgp']))[support].astype(np.float64)
        byte = prediction.astype(np.int16) - baseline.astype(np.int16)
        rows.append({'id': case['id'], 'source': case['source'], 'reference': case['source_person_or_reference'],
                     'profile': case['profile'], 'raw_correction_RMS': float(np.sqrt(np.square(delta).mean())),
                     'raw_correction_maximum': float(np.abs(delta).max()),
                     'changed_PNG_pixels': int(np.any(byte != 0, axis=2).sum()),
                     'maximum_PNG_byte_change': int(np.abs(byte).max()),
                     'PNG_identical': bool(np.array_equal(prediction, baseline))})
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    sheets = []
    for index, (reference, cases) in enumerate(by_reference.items(), 1):
        a.require(len(cases) == 5, 'Exactly five frozen profiles per training reference')
        sheet = Image.new('RGB', (1072, 1516), '#f1f2f4')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 5), 'Paired TRAINING photographs only | ' + reference + ' | ' + cases[0]['source'], font=font, fill='#111111')
        for column, label in enumerate(('Input', 'Own DGP baseline', 'V27 stopped50', 'Paired TRAIN target')):
            draw.text((12 + column * 264, 30), label, font=font, fill='#111111')
        cells = []
        for row_index, case in enumerate(cases):
            y = 64 + row_index * 288
            draw.text((12, y), case['profile'] + ' | ' + case['id'], font=font, fill='#111111')
            sources = [a.BUNDLE / case['input'], RETURNED / 'outputs/update0' / (case['id'] + '.png'),
                       RETURNED / 'outputs/update50' / (case['id'] + '.png'), a.BUNDLE / case['target']]
            for column, path in enumerate(sources):
                value = a.rgb(path)
                xy = [12 + column * 264, y + 24]
                sheet.paste(Image.fromarray(value), tuple(xy))
                cells.append({'case': case['id'], 'column': column, 'xy': xy,
                              'source': path.relative_to(ROOT).as_posix(),
                              'pixel_sha256': hashlib.sha256(value.tobytes()).hexdigest()})
        filename = 'sheet_' + str(index).zfill(2) + '_' + reference + '.png'
        sheet.save(OUT / filename)
        sheets.append({'file': filename, 'reference': reference, 'cases': [case['id'] for case in cases], 'cells': cells})
    # Independent saved-cell readback requires all200 original source arrays,
    # with no interpolation/enhancement or a model-derived visual verdict.
    cells_verified = 0
    for entry in sheets:
        with Image.open(OUT / entry['file']) as image:
            canvas = np.asarray(image).copy()
        a.require(canvas.shape == (1516, 1072, 3), 'Review sheet dimensions differ')
        for cell in entry['cells']:
            x, y = cell['xy']
            value = canvas[y:y + 256, x:x + 256]
            original = a.rgb(ROOT / cell['source'])
            a.require(np.array_equal(value, original) and hashlib.sha256(value.tobytes()).hexdigest() == cell['pixel_sha256'],
                      'Review cell differs from saved original pixels')
            cells_verified += 1
    a.require(len(sheets) == 10 and cells_verified == 200 and len(rows) == 50, 'Whole-cohort review required')
    groups = {}
    for label in ('all', 'clear', 'degraded'):
        chosen = [row for row in rows if label == 'all' or (row['profile'] == 'clear') == (label == 'clear')]
        groups[label] = {'cases': len(chosen), 'identical_PNGs': sum(row['PNG_identical'] for row in chosen),
                         'median_raw_correction_RMS': float(np.median([row['raw_correction_RMS'] for row in chosen])),
                         'median_raw_correction_byte_units': 255 * float(np.median([row['raw_correction_RMS'] for row in chosen])),
                         'maximum_PNG_byte_change': max(row['maximum_PNG_byte_change'] for row in chosen)}
    report = {'complete': True, 'scope': 'Exact original-size paired photographic TRAINING review preparation only',
              'protocol_sha256': a.PIN, 'audit_sha256': a.sha(audit_path), 'generator_sha256': a.sha(Path(__file__)),
              'rows': rows, 'groups': groups, 'sheets': sheets, 'exact_source_cells_verified': cells_verified,
              'source_bindings_sha256': bindings, 'sheet_sha256': {entry['file']: a.sha(OUT / entry['file']) for entry in sheets},
              'seconds': time.monotonic() - start, 'neural_calls': 0, 'gradient_or_optimizer_calls': 0,
              'visual_review_pending': True, 'native_or_reserved_used': False, 'independent_final_review': False,
              'app_promotion': False, 'goal_complete': False}
    a.write(OUT / 'preparation.json', report)
    print(json.dumps({'complete': True, 'groups': groups, 'exact_cells': cells_verified, 'seconds': report['seconds']}, indent=2))


if __name__ == '__main__':
    main()

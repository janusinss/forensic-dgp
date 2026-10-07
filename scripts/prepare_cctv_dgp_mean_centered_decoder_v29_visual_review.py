"""Exact delivered V29 final800 photographic TRAIN review; no neural calls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
BUNDLE = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_vm_v29'
RETURNED = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_return'
OUT = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_visual_review'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    start = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['training_completed800'] and audit['necessary_capacity_pass']
    assert [row['update'] for row in audit['snapshots_audited']] == [0, 50, 400, 800]
    assert not OUT.exists(), 'Preserve previous review evidence'
    assert sha(BUNDLE / 'protocol.json') == audit['protocol_sha256']
    p = read(BUNDLE / 'protocol.json')
    parent_protocol = read(PARENT / 'protocol.json')
    for name, digest in parent_protocol['assets_sha256'].items():
        assert sha(PARENT / name) == digest, name
    OUT.mkdir()
    plan = {
        'format': 'V29-exact-final800-TRAIN-whole-face-visual-plan-v1',
        'scope': 'All50 paired photographic TRAINING cases; not native CCTV or held-out performance',
        'case_ids': [row['id'] for row in p['case_rows']],
        'checkpoint_selection': 'Final800 only; no earlier checkpoint selection',
        'columns': ['Input', 'Unchanged DGP', 'V29 final800', 'Paired TRAIN target'],
        'cell_size': [256, 256],
        'regions': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
        'criteria': [
            'Compare all visible facial regions together; softness is acceptable when useful structure survives.',
            'Record boundary clarity separately from overall lightness or contrast.',
            'Keep visible glasses, hair, expression and other appearance; record alterations.',
            'Training target is a paired photographic reference only; do not claim recovered identity.',
            'Visual impressions do not waive preservation, brightness or capacity gate failures.',
        ],
        'interpolation_or_enhancement': False,
        'audit_sha256': sha(audit_path),
        'protocol_sha256': audit['protocol_sha256'],
        'preparation_source_sha256': sha(Path(__file__)),
        'neural_or_gradient_calls': 0,
        'native_or_reserved_used': False,
        'independent_final_review': False,
        'app_promotion': False,
        'goal_complete': False,
    }
    write(OUT / 'plan.json', plan)
    bindings = {audit_path.relative_to(ROOT).as_posix(): sha(audit_path),
                (BUNDLE / 'protocol.json').relative_to(ROOT).as_posix(): sha(BUNDLE / 'protocol.json')}
    def rgb(path):
        with Image.open(path) as im:
            assert im.size == (256, 256)
            return np.asarray(im.convert('RGB')).copy()
    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    by_reference = defaultdict(list)
    rows = []
    for case in p['case_rows']:
        by_reference[case['source_person_or_reference']].append(case)
        cid = case['id']
        before = RETURNED / 'outputs/update0' / (cid + '.png')
        final = RETURNED / 'outputs/update800' / (cid + '.png')
        raw_before = RETURNED / 'outputs/update0' / (cid + '.npy')
        raw_final = RETURNED / 'outputs/update800' / (cid + '.npy')
        for path in [before, final, raw_before, raw_final, PARENT / case['input'], PARENT / case['target'], PARENT / case['observed']]:
            bind(path)
        baseline, prediction = rgb(before), rgb(final)
        with Image.open(PARENT / case['observed']) as im:
            support = np.asarray(im).copy() > 0
        delta = (np.load(raw_final, allow_pickle=False) - np.load(raw_before, allow_pickle=False))[support].astype(np.float64)
        byte = prediction.astype(np.int16) - baseline.astype(np.int16)
        rows.append({'id': cid, 'reference': case['source_person_or_reference'], 'source': case['source'], 'profile': case['profile'],
                     'raw_correction_RMS': float(np.sqrt(np.square(delta).mean())),
                     'postclip_mean_RGB_shift': delta.mean(0).tolist(),
                     'zero_mean_correction_RMS': float(np.sqrt(np.square(delta - delta.mean(0)).mean())),
                     'changed_PNG_pixels': int(np.any(byte != 0, axis=2).sum()),
                     'maximum_PNG_byte_change': int(np.abs(byte).max()),
                     'PNG_identical': bool(np.array_equal(baseline, prediction))})
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    sheets = []
    for index, (reference, cases) in enumerate(by_reference.items(), 1):
        assert len(cases) == 5
        sheet = Image.new('RGB', (1072, 1516), '#f1f2f4')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 5), 'Paired TRAIN photographs | ' + reference + ' | ' + cases[0]['source'], font=font, fill='#111111')
        for column, label in enumerate(plan['columns']):
            draw.text((12 + column * 264, 30), label, font=font, fill='#111111')
        cells = []
        for row_index, case in enumerate(cases):
            y = 64 + row_index * 288
            draw.text((12, y), case['profile'] + ' | ' + case['id'], font=font, fill='#111111')
            paths = [PARENT / case['input'], RETURNED / 'outputs/update0' / (case['id'] + '.png'),
                     RETURNED / 'outputs/update800' / (case['id'] + '.png'), PARENT / case['target']]
            for column, path in enumerate(paths):
                value = rgb(path)
                xy = [12 + column * 264, y + 24]
                sheet.paste(Image.fromarray(value), tuple(xy))
                cells.append({'case': case['id'], 'column': column, 'xy': xy,
                              'source': path.relative_to(ROOT).as_posix(),
                              'pixel_sha256': hashlib.sha256(value.tobytes()).hexdigest()})
        filename = 'sheet_' + str(index).zfill(2) + '_' + reference + '.png'
        sheet.save(OUT / filename)
        sheets.append({'file': filename, 'reference': reference, 'cases': [case['id'] for case in cases], 'cells': cells})
    cells_checked = 0
    for entry in sheets:
        with Image.open(OUT / entry['file']) as im:
            canvas = np.asarray(im).copy()
        assert canvas.shape == (1516, 1072, 3)
        for cell in entry['cells']:
            x, y = cell['xy']
            actual = canvas[y:y+256, x:x+256]
            assert np.array_equal(actual, rgb(ROOT / cell['source']))
            assert hashlib.sha256(actual.tobytes()).hexdigest() == cell['pixel_sha256']
            cells_checked += 1
    assert cells_checked == 200 and len(rows) == 50 and len(sheets) == 10
    grouped = {}
    for label in ['all', 'clear', 'degraded']:
        chosen = [r for r in rows if label == 'all' or (r['profile'] == 'clear') == (label == 'clear')]
        grouped[label] = {'cases': len(chosen), 'identical_PNGs': sum(r['PNG_identical'] for r in chosen),
                          'median_raw_correction_byte_units': 255 * float(np.median([r['raw_correction_RMS'] for r in chosen])),
                          'median_zero_mean_correction_byte_units': 255 * float(np.median([r['zero_mean_correction_RMS'] for r in chosen])),
                          'maximum_PNG_byte_change': max(r['maximum_PNG_byte_change'] for r in chosen)}
    report = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'audit_sha256': sha(audit_path),
              'generator_sha256': sha(Path(__file__)), 'rows': rows, 'groups': grouped, 'sheets': sheets,
              'exact_source_cells_verified': cells_checked, 'source_bindings_sha256': bindings,
              'sheet_sha256': {r['file']: sha(OUT / r['file']) for r in sheets},
              'neural_or_gradient_calls': 0, 'visual_review_pending': True, 'native_or_reserved_used': False,
              'independent_final_review': False, 'app_promotion': False, 'goal_complete': False,
              'seconds': time.monotonic() - start}
    write(OUT / 'preparation.json', report)
    print(json.dumps({'complete': True, 'cases': len(rows), 'exact_cells': cells_checked, 'groups': grouped}, indent=2))


if __name__ == '__main__':
    main()

"""Render the fixed 100 TRAIN previews and 24 native DEV comparisons.

Creating sheets is not a visual review. Every sheet needs an actual displayed
view and a separate observation before the independent closure can pass.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_analysis'
CURRENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
OLD = ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'
A = ROOT/'outputs/cctv_dgp_bank_comparison_v1_r3_A_return'
DEST = ANALYSIS/'visual_review'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def render():
    assert not DEST.exists()
    matched_path = ANALYSIS/'visual_input_match.json'
    matched = read(matched_path)
    assert matched['complete'] and matched['matching_baseline_preview_PNGs'] == 100
    assert matched['unchanged_native_baseline_cases'] == 24
    for path, expected in matched['compared_receipt_bindings'].items():
        assert sha(path) == expected
    old_AB = read(ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis/matched_AB_comparison.json')
    assert old_AB['complete'] and old_AB['unchanged_baseline_RGB_embedding_packs'] == 781
    assert sha(A/'protocol.json') == old_AB['protocol_A_sha256']
    assert sha(OLD/'protocol.json') == old_AB['protocol_B_sha256']
    p = read(CURRENT/'protocol.json')
    for name in ['cases', 'references', 'native_development', 'preview_case_ids']:
        assert p[name] == read(A/'protocol.json')[name]
    manifests = {root.resolve(): read(root/'export_manifest.json')['files'] for root in [A, OLD, CURRENT]}
    DEST.mkdir()
    tile_bindings = {}
    sheets = []
    refs = {r['id']: r for r in p['references']}
    c_out, b_out, a_out = [r/'outputs/bank_comparison_v1' for r in [CURRENT, OLD, A]]

    def tile(sheet, draw, root, path, label, row, column, cells):
        expected = manifests[root.resolve()][path.relative_to(root).as_posix()]
        assert path.is_file() and not path.is_symlink() and sha(path) == expected
        tile_bindings[str(path.resolve())] = expected
        x, y = column*256, row*278
        draw.text((x+3, y+3), label, fill='black')
        with Image.open(path) as image:
            assert image.size == (256, 256)
            sheet.paste(image.convert('RGB'), (x, y+22))
        cells.append(dict(path=str(path.resolve()), sha256=expected, row=row, column=column,
                          x=x, y=y+22, width=256, height=256, label=label))

    selected = [c for c in p['cases'] if c['id'] in p['preview_case_ids']]
    assert len(selected) == 100
    reference_ids = list(dict.fromkeys(c['source_person_or_reference'] for c in selected))
    assert len(reference_ids) == 20
    for ref_id in reference_ids:
        cases = [c for c in selected if c['source_person_or_reference'] == ref_id]
        assert len(cases) == 5
        sheet = Image.new('RGB', (1792, 1390), 'white')
        draw = ImageDraw.Draw(sheet)
        cells = []
        for row, case in enumerate(cases):
            paths = [(CURRENT, CURRENT/case['input'], 'input'),
                     (CURRENT, CURRENT/refs[ref_id]['target'], 'clean TRAIN target'),
                     (CURRENT, c_out/'baseline/previews'/(case['id']+'.png'), 'original DGP'),
                     (A, a_out/'A50/previews'/(case['id']+'.png'), 'A50 rejected'),
                     (OLD, b_out/'B50/previews'/(case['id']+'.png'), 'B r4 rejected'),
                     (CURRENT, c_out/'B50/previews'/(case['id']+'.png'), 'brightness guard'),
                     (CURRENT, c_out/'B50_bank_disabled/previews'/(case['id']+'.png'), 'guard bank disabled')]
            for column, (root, path, label) in enumerate(paths):
                tile(sheet, draw, root, path, case['profile']+': '+label, row, column, cells)
        name = 'paired_'+ref_id+'.png'
        sheet.save(DEST/name)
        sheets.append(dict(name=name, sha256=sha(DEST/name), evidence_role='paired photographic TRAIN',
                           case_ids=[c['id'] for c in cases], cells=cells))
    for page in range(6):
        sheet = Image.new('RGB', (1536, 1112), 'white')
        draw = ImageDraw.Draw(sheet)
        cells = []
        cases = p['native_development'][page*4:(page+1)*4]
        assert len(cases) == 4
        for row, case in enumerate(cases):
            paths = [(CURRENT, CURRENT/case['input'], 'input'),
                     (CURRENT, c_out/'baseline/native'/(case['id']+'.png'), 'original DGP'),
                     (A, a_out/'A50/native'/(case['id']+'.png'), 'A50 rejected'),
                     (OLD, b_out/'B50/native'/(case['id']+'.png'), 'B r4 rejected'),
                     (CURRENT, c_out/'B50/native'/(case['id']+'.png'), 'brightness guard'),
                     (CURRENT, c_out/'B50_bank_disabled/native'/(case['id']+'.png'), 'guard bank disabled')]
            for column, (root, path, label) in enumerate(paths):
                tile(sheet, draw, root, path, case['id']+': '+label, row, column, cells)
        name = f'native_{page:02d}.png'
        sheet.save(DEST/name)
        sheets.append(dict(name=name, sha256=sha(DEST/name), evidence_role='native CCTV DEV unpaired',
                           case_ids=[c['id'] for c in cases], cells=cells))
    write(DEST/'sheet_manifest.json', dict(complete=True, checker_sha256=sha(__file__),
        visual_input_match_sha256=sha(matched_path), source_tile_bindings=tile_bindings,
        sheets=sheets, paired_sheets=20, native_sheets=6, paired_cases=100, native_cases=24,
        cells=sum(len(row['cells']) for row in sheets), missing_tiles=[], viewed_sheets=0,
        creating_sheets_does_not_prove_visual_review=True, independent_final_review_pending=True,
        full_metric_and_state_audit_separate=True,
        final_identities_used=False, model_qualified=False, goal_complete=False))
    print(dict(sheets=26, cells=844, viewed_sheets=0), flush=True)


def record(sheet, notes):
    manifest_path = DEST/'sheet_manifest.json'
    manifest = read(manifest_path)
    sheets = {row['name']: row['sha256'] for row in manifest['sheets']}
    assert sheet in sheets and Path(sheet).name == sheet
    assert sha(DEST/sheet) == sheets[sheet]
    assert len(notes.strip()) >= 30
    observations = ANALYSIS/'visual_observations'
    observations.mkdir(exist_ok=True)
    write(observations/(sheet+'.json'), dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        sheet=sheet, sheet_sha256=sheets[sheet], sheet_manifest_sha256=sha(manifest_path),
        displayed_through='view_image at original resolution', reviewer='implementing assistant',
        independent_final_review=False, image_generation_alone_not_view_evidence=True,
        notes=notes, model_qualified=False, goal_complete=False))
    print(dict(recorded_sheet=sheet, independent_final_review=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--sheet')
    parser.add_argument('--notes')
    args = parser.parse_args()
    if args.render:
        assert args.sheet is None and args.notes is None
        render()
    else:
        assert args.sheet and args.notes
        record(args.sheet, args.notes)

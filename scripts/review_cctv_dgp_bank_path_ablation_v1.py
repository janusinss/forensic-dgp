"""Render/record all planned frozen path previews; rendering is not review."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from import_cctv_dgp_bank_path_ablation_v1_return import ROOT, RETURN, PARENT, ANALYSIS, PIN, read, sha, write

DEST = ANALYSIS/'visual_review'


def render():
    assert not DEST.exists()
    assert read(ANALYSIS/'import.json')['complete'] and sha(RETURN/'protocol.json') == PIN
    p, old = read(RETURN/'protocol.json'), read(PARENT/'protocol.json')
    out = RETURN/'outputs/path_ablation_v1'
    parent_out = PARENT/'outputs/bank_comparison_v1'
    manifests = {r: read(r/'export_manifest.json')['files'] for r in [RETURN, PARENT]}
    DEST.mkdir()
    sheets, bindings = [], {}

    def tile(sheet, draw, root, path, label, row, column, cells):
        expected = manifests[root][path.relative_to(root).as_posix()]
        assert sha(path) == expected and not path.is_symlink()
        bindings[path.relative_to(ROOT).as_posix()] = expected
        x, y = column*256, row*278
        draw.text((x+3, y+3), label, fill='black')
        with Image.open(path) as im:
            assert im.size == (256, 256)
            sheet.paste(im.convert('RGB'), (x, y+22))
        cells.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=expected,
                          x=x, y=y+22, width=256, height=256, row=row, column=column, label=label))

    selected = [c for c in old['cases'] if c['id'] in p['case_ids']]
    refs = {r['id']: r for r in old['references']}
    reference_ids = list(dict.fromkeys(c['source_person_or_reference'] for c in selected))
    assert len(reference_ids) == 20 and len(selected) == 100
    for ref_id in reference_ids:
        cases = [c for c in selected if c['source_person_or_reference'] == ref_id]
        assert len(cases) == 5
        sheet = Image.new('RGB', (2560, 1390), 'white'); draw = ImageDraw.Draw(sheet); cells = []
        for row, case in enumerate(cases):
            paths = [(PARENT, PARENT/case['input'], 'input'), (PARENT, PARENT/refs[ref_id]['target'], 'clean TRAIN target'),
                     (PARENT, parent_out/'baseline/previews'/(case['id']+'.png'), 'original DGP')]
            paths += [(RETURN, out/variant/'previews'/(case['id']+'.png'), variant) for variant in p['variants']]
            for column, (root, path, label) in enumerate(paths):
                tile(sheet, draw, root, path, case['profile']+': '+label, row, column, cells)
        name = 'paired_'+ref_id+'.png'; sheet.save(DEST/name)
        sheets.append(dict(name=name, sha256=sha(DEST/name), role='paired photographic TRAIN',
                           case_ids=[c['id'] for c in cases], cells=cells))
    for page in range(6):
        cases = old['native_development'][page*4:(page+1)*4]
        assert len(cases) == 4
        sheet = Image.new('RGB', (2304, 1112), 'white'); draw = ImageDraw.Draw(sheet); cells = []
        for row, case in enumerate(cases):
            paths = [(PARENT, PARENT/case['input'], 'input'), (PARENT, parent_out/'baseline/native'/(case['id']+'.png'), 'original DGP')]
            paths += [(RETURN, out/variant/'native'/(case['id']+'.png'), variant) for variant in p['variants']]
            for column, (root, path, label) in enumerate(paths):
                tile(sheet, draw, root, path, case['id']+': '+label, row, column, cells)
        name = f'native_{page:02d}.png'; sheet.save(DEST/name)
        sheets.append(dict(name=name, sha256=sha(DEST/name), role='native CCTV DEV unpaired',
                           case_ids=[c['id'] for c in cases], cells=cells))
    write(DEST/'sheet_manifest.json', dict(complete=True, renderer_sha256=sha(__file__),
        import_sha256=sha(ANALYSIS/'import.json'), source_tile_bindings=bindings, sheets=sheets,
        paired_sheets=20, native_sheets=6, paired_cases=100, native_cases=24,
        variant_paired_cells=700, variant_native_cells=168, cells=sum(len(s['cells']) for s in sheets),
        viewed_sheets=0, rendering_does_not_imply_review=True, final_identities_used=False,
        independent_final_review_pending=True, model_qualified=False, goal_complete=False))
    print(dict(sheets=len(sheets), cells=sum(len(s['cells']) for s in sheets), viewed_sheets=0), flush=True)


def record(sheet, notes):
    manifest = read(DEST/'sheet_manifest.json')
    plan = {s['name']: s['sha256'] for s in manifest['sheets']}
    assert sheet in plan and Path(sheet).name == sheet
    assert sha(DEST/sheet) == plan[sheet] and len(notes.strip()) >= 40
    folder = ANALYSIS/'visual_observations'; folder.mkdir(exist_ok=True)
    write(folder/(sheet+'.json'), dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        sheet=sheet, sheet_sha256=plan[sheet], sheet_manifest_sha256=sha(DEST/'sheet_manifest.json'),
        displayed_through='view_image at original resolution', reviewer='implementing assistant',
        independent_final_review=False, notes=notes, model_qualified=False, goal_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true'); parser.add_argument('--sheet'); parser.add_argument('--notes')
    a = parser.parse_args()
    if a.render:
        assert not a.sheet and not a.notes
        render()
    else:
        assert a.sheet and a.notes
        record(a.sheet, a.notes)

"""Create all 26 matched A/B visual sheets after both independent return audits."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def review(a_root, b_root, comparison_receipt, dest):
    assert not dest.exists()
    comparison = read(comparison_receipt)
    assert comparison['complete'] and comparison['unchanged_baseline_RGB_embedding_packs'] == 781
    assert comparison['unchanged_native_baseline_cases'] == 24
    assert sha(a_root/'protocol.json') == comparison['protocol_A_sha256']
    assert sha(b_root/'protocol.json') == comparison['protocol_B_sha256']
    for path, expected in comparison['compared_receipt_bindings'].items():
        assert sha(path) == expected
    p = read(a_root/'protocol.json')
    manifests = {root.resolve(): read(root/'export_manifest.json')['files'] for root in [a_root,b_root]}
    tile_bindings = {}
    dest.mkdir(parents=True)
    a_out = a_root/'outputs/bank_comparison_v1'
    b_out = b_root/'outputs/bank_comparison_v1'
    refs = {r['id']: r for r in p['references']}
    selected = [c for c in p['cases'] if c['id'] in p['preview_case_ids']]
    reference_ids = list(dict.fromkeys(c['source_person_or_reference'] for c in selected))
    assert len(reference_ids) == 20
    files = []

    def tile(sheet, draw, root, path, label, row, column):
        relative = path.relative_to(root).as_posix()
        expected = manifests[root.resolve()][relative]
        assert path.is_file() and not path.is_symlink() and sha(path) == expected
        tile_bindings[str(path.resolve())] = expected
        x, y = column*256, row*278
        draw.text((x+3,y+3),label,fill='black')
        with Image.open(path) as image:
            assert image.size == (256,256)
            sheet.paste(image.convert('RGB'),(x,y+22))

    for reference_id in reference_ids:
        cases = [c for c in selected if c['source_person_or_reference'] == reference_id]
        assert len(cases) == 5
        sheet = Image.new('RGB',(1536,1390),'white'); draw = ImageDraw.Draw(sheet)
        for row,c in enumerate(cases):
            paths = [(a_root,a_root/c['input']),(a_root,a_root/refs[reference_id]['target']),
                     (a_root,a_out/'baseline/previews'/(c['id']+'.png')),
                     (a_root,a_out/'A50/previews'/(c['id']+'.png')),
                     (b_root,b_out/'B50/previews'/(c['id']+'.png')),
                     (b_root,b_out/'B50_bank_disabled/previews'/(c['id']+'.png'))]
            for column,((root,path),label) in enumerate(zip(paths,p['visual_review_plan']['paired_COLUMNS'])):
                tile(sheet,draw,root,path,c['profile']+': '+label,row,column)
        name = 'paired_'+reference_id+'.png'; sheet.save(dest/name); files.append(name)
    for page in range(6):
        sheet = Image.new('RGB',(1280,1112),'white'); draw = ImageDraw.Draw(sheet)
        for row,c in enumerate(p['native_development'][page*4:(page+1)*4]):
            paths = [(a_root,a_root/c['input']),
                     (a_root,a_out/'baseline/native'/(c['id']+'.png')),
                     (a_root,a_out/'A50/native'/(c['id']+'.png')),
                     (b_root,b_out/'B50/native'/(c['id']+'.png')),
                     (b_root,b_out/'B50_bank_disabled/native'/(c['id']+'.png'))]
            for column,((root,path),label) in enumerate(zip(paths,p['visual_review_plan']['native_COLUMNS'])):
                tile(sheet,draw,root,path,c['id']+': '+label,row,column)
        name = f'native_{page:02d}.png'; sheet.save(dest/name); files.append(name)
    record = dict(complete=True,checker_sha256=sha(__file__),comparison_receipt_sha256=sha(comparison_receipt),
        protocol_A_sha256=comparison['protocol_A_sha256'],protocol_B_sha256=comparison['protocol_B_sha256'],
        source_tile_bindings=tile_bindings,sheets=[dict(name=n,sha256=sha(dest/n)) for n in files],
        paired_sheets=20,native_sheets=6,missing_tiles=[],viewed_sheets=0,
        all_planned_sheets_must_be_viewed=True,sheet_generation_not_visual_review=True,
        qualification=False,independent_final_review_pending=True,goal_complete=False)
    with (dest/'sheet_manifest.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(sheets=26,missing_tiles=0,viewed_sheets=0),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['a-root','b-root','comparison-receipt','dest']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    review(args.a_root,args.b_root,args.comparison_receipt,args.dest)

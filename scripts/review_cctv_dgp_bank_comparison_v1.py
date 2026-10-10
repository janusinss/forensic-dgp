"""Create the prospectively planned sheets from audited saved PNGs only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(root, audit_receipt, dest):
    root = root.resolve(); assert not dest.exists()
    sys.path.insert(0, str(root))
    from cctv_dgp_bank_comparison_v1_contract import verify
    p = verify(root, sha(root/'protocol.json'))
    a = json.loads(audit_receipt.read_text()); assert a['complete'] and a['protocol_sha256'] == sha(root/'protocol.json')
    assert a['manifest_sha256'] == sha(root/'export_manifest.json')
    dest.mkdir(parents=True)
    out = root/'outputs/bank_comparison_v1'; files = []; missing = set()
    refs = {r['id']: r for r in p['references']}
    selected = [c for c in p['cases'] if c['id'] in p['preview_case_ids']]
    ids = list(dict.fromkeys(c['source_person_or_reference'] for c in selected)); assert len(ids) == 20

    def tile(sheet, draw, path, label, row, col):
        x, y = col*256, row*278
        draw.text((x+3, y+3), label, fill='black')
        if path.exists():
            with Image.open(path) as im:
                assert im.size == (256, 256)
                sheet.paste(im.convert('RGB'), (x, y+22))
        else:
            draw.text((x+4, y+50), 'Not produced; see retained stop', fill='black')
            missing.add(path.relative_to(root).as_posix())

    for id_ in ids:
        cs = [c for c in selected if c['source_person_or_reference'] == id_]; assert len(cs) == 5
        sheet = Image.new('RGB', (1536, 1390), 'white'); draw = ImageDraw.Draw(sheet)
        for row, c in enumerate(cs):
            paths = [root/c['input'], root/refs[id_]['target']]
            paths += [out/label/'previews'/(c['id']+'.png') for label in ['baseline', 'A50', 'B50', 'B50_bank_disabled']]
            for col, (label, path) in enumerate(zip(p['visual_review_plan']['paired_COLUMNS'], paths)):
                tile(sheet, draw, path, c['profile']+': '+label, row, col)
        name = 'paired_'+id_+'.png'; sheet.save(dest/name); files.append(name)
    native = p['native_development']
    for page in range(6):
        sheet = Image.new('RGB', (1280, 1112), 'white'); draw = ImageDraw.Draw(sheet)
        for row, c in enumerate(native[page*4:(page+1)*4]):
            paths = [root/c['input']] + [out/label/'native'/(c['id']+'.png') for label in ['baseline', 'A50', 'B50', 'B50_bank_disabled']]
            for col, (label, path) in enumerate(zip(p['visual_review_plan']['native_COLUMNS'], paths)):
                tile(sheet, draw, path, c['id']+': '+label, row, col)
        name = f'native_{page:02d}.png'; sheet.save(dest/name); files.append(name)
    record = {'complete': True, 'protocol_sha256': a['protocol_sha256'], 'audit_receipt_sha256': sha(audit_receipt),
        'checker_sha256': sha(__file__), 'sheets': [{'name': n, 'sha256': sha(dest/n)} for n in files],
        'paired_sheets': 20, 'native_sheets': 6, 'missing_tiles': sorted(missing),
        'all_planned_sheets_must_be_viewed': True, 'viewed_sheets': 0,
        'sheet_generation_not_visual_review': True, 'quality_qualified': False, 'goal_complete': False}
    with (dest/'sheet_manifest.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'sheets': len(files), 'missing_tiles': len(missing), 'viewed_sheets': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit-receipt', type=Path, required=True); parser.add_argument('--dest', type=Path, required=True)
    a = parser.parse_args(); review(a.root, a.audit_receipt, a.dest)

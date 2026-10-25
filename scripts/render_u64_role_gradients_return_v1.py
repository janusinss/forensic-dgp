"""Arrange exact saved RGB cells for all planned slots; no image enhancement."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_audit'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
OUT = 'outputs/u64_role_gradients_v1'


def digest(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    inventory = json.loads((AUDIT / 'inventory_v1.json').read_bytes())
    protocol = json.loads((STAGE / 'protocol.json').read_bytes())
    parent = json.loads((PARENT / 'protocol.json').read_bytes())
    evidence = Path(inventory['evidence_root'])
    plan = protocol['plan']; assert len(plan) == 72
    rows = {r['id']: r for r in parent['cases'] + parent['guards']}
    directory = AUDIT / 'visuals_v1'; directory.mkdir(exist_ok=False)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
        bold = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 14)
    except OSError:
        font = bold = ImageFont.load_default()
    consumed = {}; bindings = []; pages = []
    def checked(base, binding):
        path = base / binding['path']; pin = digest(path)
        assert pin['bytes'] == binding['bytes'] and pin['sha256'] == binding['sha256']
        consumed[pin['path']] = pin
        return path
    def rgb(path):
        with Image.open(path) as image:
            assert image.mode == 'RGB' and image.size == (256, 256) and getattr(image, 'n_frames', 1) == 1
            return image.copy()
    for start in range(0, 72, 4):
        page = start // 4 + 1; calls = plan[start:start+4]
        canvas = Image.new('RGB', (820, 1272), '#f0eee8'); draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), 'Original trained u64 / zero-update diagnostic | saved RGB256 | '+str(page)+'/18', font=bold, fill='black')
        draw.text((10, 30), 'TRAIN / correlated roles; no accepted new model. Exact cells; no display enhancement.', font=font, fill='black')
        for col, title in enumerate(('Observed input', 'Photographic TRAIN target', 'Original trained u64 (unchanged)')):
            draw.text((10+270*col, 52), title, font=font, fill='black')
        for index, call in enumerate(calls):
            row = rows[call['id']]; y = 102+290*index
            draw.text((10, y-29), str(call['slot']).zfill(2)+' '+call['id'], font=font, fill='black')
            draw.text((10, y-15), call['phase']+' | '+call['metric_region'], font=font, fill='black')
            name = OUT + '/calls/' + str(call['slot']).zfill(4) + '/record.json'
            record_path = checked(evidence, inventory['files'][name]); record = json.loads(record_path.read_bytes())
            assert record['call'] == call
            sources = (checked(PARENT, row['input']), checked(PARENT, row['target']),
                       checked(evidence, record['outputs']['png']))
            for col, source in enumerate(sources):
                image = rgb(source); x = 10+270*col; canvas.paste(image, (x,y))
                assert canvas.crop((x,y,x+256,y+256)).tobytes() == image.tobytes()
                bindings.append({'page':page, 'slot':call['slot'], 'column':col, 'xy':[x,y],
                    'source':digest(source), 'RGB8_sha256':hashlib.sha256(image.tobytes()).hexdigest()})
        target = directory / ('page_'+str(page).zfill(2)+'.png'); canvas.save(target)
        with Image.open(target) as saved:
            for cell in [r for r in bindings if r['page']==page]:
                x,y=cell['xy']; assert hashlib.sha256(saved.crop((x,y,x+256,y+256)).tobytes()).hexdigest() == cell['RGB8_sha256']
        pages.append(digest(target))
    for binding in consumed.values():
        assert digest(ROOT / binding['path']) == binding
    result = {'complete':True, 'planned_pages':18, 'actual_pages':len(pages), 'output_slots':list(range(1,73)),
        'output_cells_once_each':72, 'total_cells':len(bindings), 'all_cells_exact_256_without_resampling_or_enhancement':True,
        'pages':pages, 'cells':bindings, 'consumed_files':list(consumed.values()),
        'all_consumed_bytes_rehashed_after':True,'source':digest(Path(__file__)),
        'root_visual_review_pending':True,'model_qualified':False}
    with (AUDIT/'visual_bindings_v1.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print({'pages':len(pages),'cells':len(bindings),'output_slots':72,'root_visual_review_pending':True})


if __name__ == '__main__':
    main()

"""Render all unchanged-state returned outputs at exact RGB256, without inference."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
OUT = 'outputs/u64_png_surrogate_measurement_v1'


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def main():
    inventory = json.loads((AUDIT / 'inventory_v1.json').read_bytes())
    evidence = Path(inventory['evidence_root'])
    protocol = json.loads((STAGE / 'protocol.json').read_bytes())
    consumed = {}
    def checked(base, binding):
        path = base / binding['path']; actual = pin(path)
        assert {k: actual[k] for k in ('bytes', 'sha256')} == {k: binding[k] for k in ('bytes', 'sha256')}
        consumed[actual['path']] = actual
        return path
    by_case, records, aliases = {}, [], []
    for call in protocol['plan']:
        name = OUT + '/calls/' + str(call['slot']).zfill(4) + '/record.json'
        record = json.loads(checked(evidence, inventory['files'][name]).read_bytes())
        assert record['call'] == call
        output = checked(evidence, record['outputs']['png'])
        if call['id'] in by_case:
            previous = by_case[call['id']]
            assert call['phase'] == 'deterministic_same_u64_repeat'
            assert pin(output)['sha256'] == previous['png']['sha256']
            aliases.append({'slot': call['slot'], 'same_RGB_cell_as_slot': previous['call']['slot'],
                            'exact_same_PNG_sha256': pin(output)['sha256']})
        else:
            by_case[call['id']] = {'call': call, 'record': record, 'png': pin(output)}
        records.append(record)
    assert len(records) == 72 and len(by_case) == 68 and len(aliases) == 4
    order = list(by_case)
    directory = AUDIT / 'visuals_v1'; directory.mkdir(exist_ok=False)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
    bold = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 14)
    cells, pages = [], []
    for start in range(0, len(order), 6):
        number = start // 6 + 1
        canvas = Image.new('RGB', (820, 1838), '#eeeae2'); draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f'Original u64 / PNG-surrogate diagnostic | {number}/12 | ZERO UPDATES', font=bold, fill='black')
        draw.text((10, 31), 'Exact RGB256; TRAIN only. Same-state repeats share hash-identical output cells.', font=font, fill='black')
        for col, title in enumerate(('Observed', 'Photographic target', 'Unchanged original trained u64')):
            draw.text((10 + col * 270, 54), title, font=font, fill='black')
        for row_index, key in enumerate(order[start:start+6]):
            entry = by_case[key]; record = entry['record']; y = 100 + 288 * row_index
            draw.text((10, y - 21), key + ' | ' + record['role'], font=font, fill='black')
            sources = [(0, checked(PARENT, record['input']), None),
                       (1, checked(PARENT, record['target']), None),
                       (2, ROOT / entry['png']['path'], entry['call']['slot'])]
            for column, path, slot in sources:
                with Image.open(path) as image:
                    assert image.mode == 'RGB' and image.size == (256, 256)
                    image = image.copy()
                x = 10 + column * 270; canvas.paste(image, (x, y))
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                assert hashlib.sha256(canvas.crop((x, y, x+256, y+256)).tobytes()).hexdigest() == digest
                cells.append({'page': number, 'id': key, 'column': column, 'slot': slot,
                              'xy': [x, y], 'source': pin(path), 'RGB8_sha256': digest})
        target = directory / ('page_' + str(number).zfill(2) + '.png'); canvas.save(target)
        with Image.open(target) as saved:
            for cell in [r for r in cells if r['page'] == number]:
                x, y = cell['xy']
                assert hashlib.sha256(saved.crop((x, y, x+256, y+256)).tobytes()).hexdigest() == cell['RGB8_sha256']
        pages.append(pin(target))
    covered = [r['slot'] for r in cells if r['slot'] is not None]
    assert len(covered) == len(set(covered)) == 68
    assert set(covered) | {r['slot'] for r in aliases} == set(range(1, 73))
    assert all(pin(ROOT / r['path']) == r for r in consumed.values())
    result = {'complete': True, 'planned_pages': 12, 'pages': pages, 'cells': cells,
        'all_cells_exact_256_without_resampling_or_enhancement': True,
        'actual_forward_outputs': 72, 'distinct_case_state_output_cells': 68,
        'exact_repeated_witness_aliases': aliases, 'all72_outputs_covered_once_or_exact_alias': True,
        'consumed_files': list(consumed.values()), 'all_consumed_bytes_rehashed_after': True,
        'source': pin(Path(__file__).resolve()), 'root_visual_review_pending': True,
        'zero_updates_not_an_improved_model': True, 'model_qualified': False}
    with (AUDIT / 'visual_bindings_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print({'pages': len(pages), 'cells': len(cells), 'all72_outputs_covered': True})


if __name__ == '__main__':
    main()

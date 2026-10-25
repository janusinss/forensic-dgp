"""Exact RGB contact sheets; repeated same-state witnesses share verified cells."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
OUT = 'outputs/u64_gate_aligned_pilot_v1'


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    inventory = json.loads((AUDIT / 'inventory_v1.json').read_bytes())
    evidence = Path(inventory['evidence_root'])
    parent = json.loads((PARENT / 'protocol.json').read_bytes())
    rows = {r['id']: r for r in parent['cases'] + parent['guards']}
    consumed = {}
    def checked(base, binding):
        path = base / binding['path']
        value = pin(path)
        assert {k: value[k] for k in ('bytes', 'sha256')} == {k: binding[k] for k in ('bytes', 'sha256')}
        consumed[value['path']] = value
        return path
    def returned(name):
        path = checked(evidence, inventory['files'][name])
        return json.loads(path.read_bytes())
    state_names = {}
    for index in (0, 1, 2):
        data = returned(OUT + '/checkpoints/u' + str(index).zfill(4) + '/checkpoint.json')
        state_names[data['head_description']['canonical_sha256']] = index
    by_case = {}
    records = []
    aliases = []
    for name in sorted(inventory['files']):
        if name.startswith(OUT + '/calls/') and name.endswith('/record.json'):
            record = returned(name)
            slot = record['call']['slot']
            state = state_names[record['head_before_sha256']]
            key = (record['id'], state)
            output = checked(evidence, record['outputs']['png'])
            if key in by_case:
                previous = by_case[key]
                assert record['call']['phase'] == 'deterministic_same_u64_repeat'
                assert pin(output)['sha256'] == previous['png']['sha256']
                aliases.append({'slot': slot, 'same_RGB_cell_as_slot': previous['slot'],
                                'exact_same_PNG_sha256': pin(output)['sha256']})
            else:
                by_case[key] = {'slot': slot, 'record': record, 'png': pin(output), 'state': state}
            records.append(record)
    assert len(records) == 300 and len(by_case) == 296 and len(aliases) == 4
    order = []
    # Put the two direct blur endpoints and clear/dark monitor faces first.
    for group in ('monitors', 'guards', 'optimization'):
        source = parent['guards'] if group == 'guards' else [r for r in parent['cases'] if r['cohort'] == ('monitor' if group == 'monitors' else 'optimize')]
        order.extend(row['id'] for row in source)
    assert len(order) == len(set(order)) == 120
    directory = AUDIT / 'visuals_v1'
    directory.mkdir(exist_ok=False)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
        bold = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 14)
    except OSError:
        font = bold = ImageFont.load_default()
    cells, pages = [], []
    for start in range(0, len(order), 6):
        number = start // 6 + 1
        canvas = Image.new('RGB', (1360, 1838), '#eeeae2')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f'Original u64 / gate-aligned return | {number}/20 | TRAIN only; quality stop at update2', font=bold, fill='black')
        draw.text((10, 31), 'Exact saved RGB256 cells; no resampling/enhancement. Repeated witnesses use hash-identical alias cells.', font=font, fill='black')
        for col, title in enumerate(('Observed', 'Photographic target', 'Original u64', 'After1 update (pilot rejected)', 'After2 updates (failed)')):
            draw.text((10 + col * 270, 54), title, font=font, fill='black')
        for row_index, key in enumerate(order[start:start+6]):
            reference = rows[key]
            y = 100 + 288 * row_index
            draw.text((10, y - 21), key + ' | ' + reference['role'], font=font, fill='black')
            sources = [(0, checked(PARENT, reference['input']), None)]
            if reference.get('target') is not None:
                sources.append((1, checked(PARENT, reference['target']), None))
            else:
                draw.text((280, y + 110), 'Legacy dark guard: no aligned target', font=font, fill='black')
            for state in range(3):
                entry = by_case.get((key, state))
                if entry:
                    sources.append((state + 2, ROOT / entry['png']['path'], entry['slot']))
                else:
                    draw.text((10 + 270 * (state + 2), y + 110), 'Not attempted; no generated substitute', font=font, fill='black')
            for column, path, slot in sources:
                with Image.open(path) as image:
                    assert image.mode == 'RGB' and image.size == (256, 256)
                    image = image.copy()
                x = 10 + column * 270
                canvas.paste(image, (x, y))
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                assert hashlib.sha256(canvas.crop((x, y, x + 256, y + 256)).tobytes()).hexdigest() == digest
                cells.append({'page': number, 'id': key, 'column': column, 'slot': slot,
                              'xy': [x, y], 'source': pin(path), 'RGB8_sha256': digest})
        target = directory / ('page_' + str(number).zfill(2) + '.png')
        canvas.save(target)
        with Image.open(target) as saved:
            for cell in [r for r in cells if r['page'] == number]:
                x, y = cell['xy']
                assert hashlib.sha256(saved.crop((x, y, x + 256, y + 256)).tobytes()).hexdigest() == cell['RGB8_sha256']
        pages.append(pin(target))
    covered = [r['slot'] for r in cells if r['slot'] is not None]
    assert len(covered) == len(set(covered)) == 296
    assert set(covered) | {r['slot'] for r in aliases} == {r['call']['slot'] for r in records}
    assert all(pin(ROOT / r['path']) == r for r in consumed.values())
    result = {'complete': True, 'planned_pages': 20, 'pages': pages, 'cells': cells,
        'all_cells_exact_256_without_resampling_or_enhancement': True,
        'actual_forward_outputs': 300, 'distinct_case_state_output_cells': 296,
        'exact_repeated_witness_aliases': aliases, 'all300_outputs_covered_once_or_exact_alias': True,
        'consumed_files': list(consumed.values()), 'all_consumed_bytes_rehashed_after': True,
        'source': pin(Path(__file__).resolve()), 'root_visual_review_pending': True, 'model_qualified': False}
    with (AUDIT / 'visual_bindings_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print({'pages': len(pages), 'cells': len(cells), 'all300_outputs_covered': True})


if __name__ == '__main__':
    main()

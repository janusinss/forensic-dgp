"""All returned finite trial outputs, exact RGB256 cells; no neural execution."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_audit'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_png_checked_step_v1_vm'
OUTPUT = 'outputs/u64_png_checked_step_v1'


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
    quality, optimize, aliases, records = {}, {}, [], []
    for call in protocol['plan']:
        if call['kind'] != 'forward':
            continue
        name = OUTPUT + '/calls/' + str(call['slot']).zfill(4) + '/record.json'
        if name not in inventory['files']:
            continue
        record = json.loads(checked(evidence, inventory['files'][name]).read_bytes())
        assert record['call'] == call
        path = checked(evidence, record['outputs']['png'])
        entry = {'record': record, 'png': pin(path), 'slot': call['slot']}
        phase = call['phase']
        if phase == 'initial_u64_quality_bank':
            quality.setdefault(call['id'], {})['original'] = entry
        elif phase == 'deterministic_same_u64_repeat':
            previous = quality[call['id']]['original']
            assert entry['png']['sha256'] == previous['png']['sha256']
            aliases.append({'slot': call['slot'], 'same_RGB_cell_as_slot': previous['slot'],
                            'exact_same_PNG_sha256': entry['png']['sha256']})
        elif phase == 'initial_optimization_bank':
            optimize.setdefault(call['id'], {})['original'] = entry
        elif phase == 'trial_optimization_bank':
            optimize[call['id']]['trial_' + str(call['trial_index'])] = entry
        else:
            assert phase == 'selected_quality_bank'
            quality[call['id']]['selected'] = entry
        records.append(record)
    assert len(records) == 436 and len(aliases) == 4 and len(quality) == 56 and len(optimize) == 64
    selection_file = OUTPUT + '/selection.json'
    selection = json.loads(checked(evidence, inventory['files'][selection_file]).read_bytes())
    assert selection['selected_index'] == 0
    groups = [
        ('monitor_guard', protocol['selection']['monitors'] + protocol['selection']['guards'], quality,
         ('original', 'selected'), ('Original u64', 'Selected trial0 / rejected')),
        ('optimization', protocol['selection']['optimization'], optimize,
         ('original', 'trial_0', 'trial_1', 'trial_2', 'trial_3'),
         ('Original u64', 'Trial0 cap2e-5', 'Trial1 cap1.5e-5', 'Trial2 cap1e-5', 'Trial3 cap5e-6')),
    ]
    directory = AUDIT / 'visuals_v1'; directory.mkdir(exist_ok=False)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
    bold = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 14)
    pages, cells = [], []
    for group, ordered, entries, states, titles in groups:
        for start in range(0, len(ordered), 6):
            page_number = len(pages) + 1
            canvas = Image.new('RGB', (10 + 270 * (len(states) + 2), 1838), '#eeeae2')
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), f'PNG-checked step / {group} | {page_number}/21 | TRAIN only; selected trial rejected', font=bold, fill='black')
            draw.text((10, 31), 'Exact saved RGB256; no resampling/enhancement. No retrospective monitor-based candidate selection.', font=font, fill='black')
            for col, title in enumerate(('Observed', 'Photographic target', *titles)):
                draw.text((10 + 270 * col, 54), title, font=font, fill='black')
            for row_number, reference in enumerate(ordered[start:start+6]):
                key = reference['id']; y = 100 + 288 * row_number
                draw.text((10, y - 21), key + ' | ' + reference['role'], font=font, fill='black')
                sources = [(0, checked(PARENT, reference['input']), None, 'input')]
                if reference.get('target') is not None:
                    sources.append((1, checked(PARENT, reference['target']), None, 'target'))
                else:
                    draw.text((280, y + 110), 'Legacy guard: no aligned target', font=font, fill='black')
                for col, state in enumerate(states, 2):
                    entry = entries[key][state]
                    sources.append((col, ROOT / entry['png']['path'], entry['slot'], state))
                for col, path, slot, state in sources:
                    with Image.open(path) as saved:
                        assert saved.mode == 'RGB' and saved.size == (256, 256)
                        image = saved.copy()
                    x = 10 + 270 * col; canvas.paste(image, (x, y))
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    assert hashlib.sha256(canvas.crop((x, y, x+256, y+256)).tobytes()).hexdigest() == digest
                    cells.append({'page': page_number, 'group': group, 'id': key, 'column': col,
                        'state': state, 'slot': slot, 'xy': [x, y], 'source': pin(path), 'RGB8_sha256': digest})
            target = directory / ('page_' + str(page_number).zfill(2) + '.png'); canvas.save(target)
            with Image.open(target) as saved:
                for cell in [r for r in cells if r['page'] == page_number]:
                    x, y = cell['xy']
                    assert hashlib.sha256(saved.crop((x, y, x+256, y+256)).tobytes()).hexdigest() == cell['RGB8_sha256']
            pages.append(pin(target))
    slots = [r['slot'] for r in cells if r['slot'] is not None]
    assert len(slots) == len(set(slots)) == 432
    assert set(slots) | {r['slot'] for r in aliases} == {r['call']['slot'] for r in records}
    assert all(pin(ROOT / r['path']) == r for r in consumed.values())
    result = {'complete': True, 'planned_pages': 21, 'pages': pages, 'cells': cells,
        'all_cells_exact_256_without_resampling_or_enhancement': True,
        'actual_forward_outputs': 436, 'distinct_case_state_output_cells': 432,
        'exact_repeated_witness_aliases': aliases, 'all436_outputs_covered_once_or_exact_alias': True,
        'selected_trial_index': 0, 'selected_trial_quality_failed': True,
        'monitor_based_candidate_reselection_permitted': False,
        'consumed_files': list(consumed.values()), 'all_consumed_bytes_rehashed_after': True,
        'source': pin(Path(__file__).resolve()), 'root_visual_review_pending': True, 'model_qualified': False}
    with (AUDIT / 'visual_bindings_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print({'pages': len(pages), 'exact_cells': len(cells), 'all436_outputs_covered': True})


if __name__ == '__main__':
    main()

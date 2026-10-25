"""Exact saved-PNG review sheets; missing future calls remain explicit blanks."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u64_cleananchors_continuation_v1_return_audit'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
RUN = 'outputs/u64_cleananchors_continuation_v1/'
PLAN = ROOT / 'outputs/codeformer_u64_cleananchors_continuation_v1_preparation/planned_visual_review_v1.json'
CELL, GAP, LEFT, TOP, ROW = 256, 12, 16, 116, 302


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': sha(raw)}


def main():
    began = time.monotonic()
    inventory_path = OUT / 'inventory_v1.json'
    inventory = json.loads(inventory_path.read_bytes())
    assert inventory['complete'] and inventory['format'] == 'u64-cleananchors-return-extracted-bindings-v1'
    extracted = ROOT / inventory['evidence_root']
    protocol = json.loads((STAGE / 'protocol.json').read_bytes())
    assert sha(PLAN.read_bytes()) == '5a2c69d93e5c724144fcff07e136d2d281e81234c274d2110841a1df0904ca22'
    planned = json.loads(PLAN.read_bytes())
    destination = OUT / 'visuals_v1_r1'
    assert not destination.exists()
    destination.mkdir()
    def payload(relative, runtime=True):
        expected = inventory['files'][relative]
        path = (extracted if runtime else STAGE) / relative
        raw = path.read_bytes()
        assert {'bytes': len(raw), 'sha256': sha(raw)} == expected
        if runtime:
            assert inventory['extracted_files'][relative] == expected
        else:
            assert protocol['files'][relative] == expected
        return raw, pin(path)
    records = {}
    for name in inventory['extracted_files']:
        if name.startswith(RUN + 'calls/') and name.endswith('/record.json'):
            raw, _ = payload(name)
            record = json.loads(raw)
            slot = record['call']['slot']
            assert name == RUN + f'calls/{slot:04d}/record.json' and slot not in records
            records[slot] = record
    opraw, _ = payload(RUN + 'operation_plan.json')
    operations = json.loads(opraw)['calls']
    assert len(operations) == 292 and set(records) == set(range(1, 205))
    by_slot = {row['slot']: row for row in operations}
    assert all(records[slot]['call'] == by_slot[slot] for slot in records)
    quality = {(row['call']['new_update'], row['id']): row for row in records.values()
               if row['call']['phase'] in ('initial_u64_quality_bank', 'new_quality_snapshot')}
    rows = {row['id']: row for row in protocol['cases'] + protocol['guards']}
    controls = {row['id']: row for row in protocol['controls']}
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    title_font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
    font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    labels = {
        'input': 'Observed input', 'photographic_target_or_explicit_legacy_dark_null': 'Photographic target / null',
        'photographic_clean_target': 'Photographic clean target', 'cached_frozen_U96': 'Frozen U96 control',
        'initial_trained_u64': 'Retained trained u64', 'new_update4': 'Continuation: new update 4',
        'new_update8': 'Continuation: new update 8', 'new_update16': 'Update 16: UNATTEMPTED',
        'actual_precommit_training_output': 'Actual precommit TRAIN output', 'same_candidate_repeat': 'Same-u64 repeat',
    }
    primary, cells, sheets = Counter(), [], []
    for sheet in planned['sheets']:
        assert time.monotonic() - began < 180
        columns = sheet['columns']
        canvas = Image.new('RGB', (LEFT * 2 + len(columns) * (CELL + GAP) - GAP,
                                  TOP + len(sheet['rows']) * ROW + 36), (246, 244, 239))
        draw = ImageDraw.Draw(canvas)
        draw.text((LEFT, 8), sheet['id'] + ' | retained quality stop at new update 8', font=title_font, fill='#161616')
        draw.text((LEFT, 38), 'Exact saved 256x256 pixels | no enhancement/resampling | TRAIN, not native/FINAL qualification', font=font, fill='#222222')
        state = sheet.get('state_label', 'Frozen controls and initial u64 are context; stopped states are not accepted successors.')
        if sheet['id'].startswith('optimization_') and not any(slot in records for slot in sheet['generated_call_slots']):
            state = 'UNATTEMPTED: the planned precommit state and model output were not observed.'
        draw.text((LEFT, 60), state, font=font, fill='#222222')
        for ci, column in enumerate(columns):
            draw.text((LEFT + ci * (CELL + GAP), 88), labels[column], font=font, fill='#111111')
        for ri, identifier in enumerate(sheet['rows']):
            row = rows[identifier]
            for ci, column in enumerate(columns):
                generated_slot, binding, runtime = None, None, False
                selected_record = None
                if column == 'input': binding = row['input']
                elif column in ('photographic_clean_target', 'photographic_target_or_explicit_legacy_dark_null'):
                    binding = row.get('target')
                elif column == 'cached_frozen_U96': binding = controls[identifier]['png']
                elif column == 'initial_trained_u64':
                    selected_record = quality[(0, identifier)]
                    if sheet['id'].startswith('trajectory_'):
                        generated_slot = selected_record['call']['slot']
                elif column.startswith('new_update'):
                    update = int(column.removeprefix('new_update'))
                    selected_record = quality.get((update, identifier))
                    if selected_record: generated_slot = selected_record['call']['slot']
                elif column in ('actual_precommit_training_output', 'same_candidate_repeat'):
                    expected = [s for s in sheet['generated_call_slots'] if by_slot[s]['id'] == identifier]
                    assert len(expected) == 1
                    generated_slot = expected[0]
                    selected_record = records.get(generated_slot)
                if selected_record:
                    binding, runtime = selected_record['outputs']['png'], True
                x, y = LEFT + ci * (CELL + GAP), TOP + ri * ROW
                draw.text((x, y), identifier, font=font, fill='#111111')
                box = [x, y + 23, x + CELL, y + 23 + CELL]
                cell = {'sheet': sheet['id'], 'id': identifier, 'column': column, 'box': box,
                        'slot': generated_slot, 'primary_generated': generated_slot is not None,
                        'available': binding is not None}
                if binding is None:
                    draw.rectangle((box[0], box[1], box[2]-1, box[3]-1), fill=(222, 219, 211))
                    message = 'No aligned clean reference' if 'target' in column else 'UNATTEMPTED after stop8'
                    draw.text((x + 8, y + 120), message, font=font, fill='#333333')
                    cell['missing_reason'] = message
                else:
                    raw, bound = payload(binding['path'], runtime=runtime)
                    assert bound['bytes'] == binding['bytes'] and bound['sha256'] == binding['sha256']
                    with Image.open(io.BytesIO(raw)) as decoded:
                        assert decoded.format == 'PNG' and decoded.mode == 'RGB' and decoded.size == (256, 256)
                        decoded.load(); image = decoded.copy()
                    canvas.paste(image, (box[0], box[1]))
                    cell.update({'source': bound, 'RGB8_sha256': sha(image.tobytes())})
                    if generated_slot is not None: primary[generated_slot] += 1
                cells.append(cell)
        path = destination / (sheet['id'] + '.png')
        canvas.save(path, format='PNG')
        with Image.open(path) as saved:
            assert saved.mode == 'RGB' and saved.size == canvas.size
            for cell in cells:
                if cell['sheet'] == sheet['id'] and cell['available']:
                    assert sha(saved.crop(cell['box']).tobytes()) == cell['RGB8_sha256']
        sheets.append({'id': sheet['id'], 'image': pin(path), 'size': list(canvas.size),
                       'available_generated': sum(c['available'] and c['primary_generated'] for c in cells if c['sheet'] == sheet['id'])})
    assert len(sheets) == 31 and set(primary) == set(records) and all(value == 1 for value in primary.values())
    result = {'complete': True, 'exact_saved_pixel_rendering_pass': True, 'inventory': pin(inventory_path),
              'plan': pin(PLAN), 'renderer': pin(Path(__file__).resolve()), 'sheets': sheets, 'cells': cells,
              'planned_sheets': 31, 'actual_generated_calls_covered_once': 204,
              'unattempted_call_slots': [slot for slot in by_slot if slot not in records],
              'all_source_and_rendered_cell_pixels_bound': True, 'root_visual_review_complete': False,
              'neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False}
    result['supersedes_visual_labels_only'] = pin(OUT / 'visual_bindings_v1.json')
    with (OUT / 'visual_bindings_v1_r1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'complete': True, 'sheets': len(sheets), 'generated_outputs': len(primary),
                      'unattempted': len(result['unattempted_call_slots']), 'seconds': time.monotonic()-began}))


if __name__ == '__main__':
    main()

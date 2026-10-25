"""Render fixed saved-output sheets only; no model or returned-code execution."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_matched_feature_learning_v1_return_audit_v1'
STAGE_NAME = 'cctv_codeformer_matched_feature_learning_v1_vm'
SEALED = ROOT / 'outputs' / STAGE_NAME
RETAINED = OUT / 'retained' / STAGE_NAME
RUN = RETAINED / 'outputs/matched_feature_learning_v1'
PLAN = ROOT / 'outputs/codeformer_matched_feature_learning_v1_source_preparation/visual_plan_v1.json'
VISUAL = OUT / 'visuals'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(path.read_bytes())


def main():
    need(not VISUAL.exists(), 'Existing rendered sheets are immutable')
    transport = read(OUT / 'transport.json')
    pixels = read(OUT / 'pixels.json')
    need(transport['complete'] is True and pixels['complete'] is True,
         'Complete independent transport and saved-pixel audits required first')
    need(not any(name.split('.')[0] in {'torch', 'tensorflow', 'jax'} for name in sys.modules),
         'Saved pixels only, no neural framework')
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    protocol, plan = read(SEALED / 'protocol.json'), read(PLAN)
    member_index = read(OUT / 'member_index.json')
    consumed = {str(path.relative_to(ROOT)).replace('\\', '/'): pin(path)
                for path in (Path(__file__), PLAN, SEALED / 'protocol.json', OUT / 'transport.json',
                             OUT / 'pixels.json', OUT / 'member_index.json')}
    need(pin(PLAN)['sha256'] == protocol['visual_plan']['sha256'] and
         pin(PLAN)['bytes'] == protocol['visual_plan']['bytes'], 'Exact sealed prospective visual plan required')
    need(pin(OUT / 'member_index.json') == transport['member_index'], 'Complete transport member index binding required')
    def record_json(path):
        binding = pin(path)
        name = path.relative_to(RETAINED).as_posix()
        need(member_index['members'].get(name) == {key: binding[key] for key in ('bytes', 'sha256')},
             'Call record metadata differs from authenticated archive member')
        consumed[binding['path']] = binding
        return read(path)
    index = {}
    for path in sorted((RUN / 'calls').glob('*/record.json')):
        record = record_json(path)
        call = record['call']
        key = (call['arm'], call['bank'], call['id'], call['phase'], call['new_update'])
        need(key not in index, 'Unique arm/bank/case/phase/update record required')
        index[key] = record
    selection = protocol['selection']
    by = {(bank, row['id']): row for bank in ('old_quality', 'new_monitors', 'new_optimization')
          for row in selection[bank]}
    controls = {row['id']: row for row in protocol['controls']}
    stop_updates = {}
    for arm in ('A', 'B'):
        available = [key[4] for key in index if key[0] == arm and key[3] == 'quality_snapshot']
        stop_updates[arm] = max(available, default=0)
    need(stop_updates == {'A': 8, 'B': 8}, 'This reviewed renderer is specific to the retained A8/B8 stop')
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 11) if font_path.exists() else ImageFont.load_default()
    title = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    def checked(base, binding):
        path = base / binding['path']
        need(path.resolve().is_relative_to(base.resolve()), 'Saved panel path escapes declared base')
        actual = pin(path)
        need((actual['bytes'], actual['sha256']) == (binding['bytes'], binding['sha256']),
             'Panel bytes differ from bound artifact')
        consumed[actual['path']] = actual
        return path
    def picture(base, binding, view='native_png'):
        path = checked(base, binding)
        if view == 'raw':
            with path.open('rb') as stream:
                array = np.load(stream, allow_pickle=False)
            need(array.dtype == np.float32 and array.shape == (256, 256, 3) and np.isfinite(array).all()
                 and array.min() >= 0 and array.max() <= 1, 'Exact valid RAW256 display required')
            # Display-only floor conversion; no scientific RAW values change.
            rgb = (array * np.float32(255)).astype(np.uint8)
        else:
            with Image.open(path) as image:
                need(image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 panel required')
                rgb = np.asarray(image).copy()
        return Image.fromarray(rgb), hashlib.sha256(rgb.tobytes()).hexdigest()
    def output(record, view):
        return picture(RETAINED, record['outputs']['raw' if view == 'raw' else 'png'], view)
    def find(arm, bank, case, phase, update):
        return index.get((arm, bank, case, phase, update))
    def panels(bank, row, view, *, training_update=None):
        result = []
        im, digest = picture(SEALED, row['input'])
        result.append(('Observed input', im, digest))
        if row.get('target') is None:
            result.append(('No aligned clean reference', None, None))
        else:
            im, digest = picture(SEALED, row['target'])
            result.append(('Photographic TRAIN target', im, digest))
        if bank == 'old_quality':
            control = controls[row['id']]
            im, digest = picture(SEALED, control['raw' if view == 'raw' else 'png'], view)
            result.append(('Frozen U96', im, digest))
        init_phase = 'initial_optimization_readout' if bank == 'new_optimization' else 'initial_quality'
        baseline = find('A', bank, row['id'], init_phase, 0)
        need(baseline is not None, 'Actual initial u64 panel required')
        im, digest = output(baseline, view)
        result.append(('Original u64 / initial', im, digest))
        for arm in ('A', 'B'):
            if training_update is not None:
                phase, update = 'photographic_optimization', training_update
                label = arm + ' / BEFORE commit ' + str(update)
            elif bank == 'new_optimization':
                phase, update = 'terminal_optimization_readout', 48
                label = arm + ' / terminal update 48'
            else:
                update = stop_updates[arm]
                phase = 'initial_quality' if update == 0 else 'quality_snapshot'
                label = arm + ' / quality snapshot ' + str(update)
            actual = find(arm, bank, row['id'], phase, update)
            if actual is None:
                result.append((label + ' MISSING', None, None))
            else:
                im, digest = output(actual, view)
                result.append((label, im, digest))
        return result
    VISUAL.mkdir()
    sheets = []
    def render(name, bank, view, keys, *, training_update=None):
        count = 6 if bank == 'old_quality' else 5
        width, row_h, top = count * 264 + 8, 296, 104
        sheet = Image.new('RGB', (width, top + len(keys) * row_h + 26), (245, 243, 238))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), name + ' | ' + bank + ' | ' + view, font=title, fill=(15, 18, 22))
        draw.text((8, 32), 'Saved paired/adaptive TRAIN evidence; original u64 versus matched A/B. Both arms stopped at update 8.', font=small, fill=(40, 40, 45))
        draw.text((8, 49), 'Exact 256px panels. RAW display = floor(RAW * 255) only; no enhancement or resampling. Missing means not executed.', font=small, fill=(40, 40, 45))
        draw.text((8, 66), 'No independent DEV/FINAL, native CCTV or identity-recovery claim. This sheet cannot qualify either model.', font=small, fill=(40, 40, 45))
        manifest = []
        for position, case_key in enumerate(keys):
            declared_bank, case = case_key.split(':', 1)
            need(declared_bank == bank, 'Visual plan scope-qualified bank required')
            row = by[(bank, case)]
            y = top + position * row_h
            draw.text((8, y), case + ' | ' + row['role'], font=font, fill=(10, 14, 22))
            cells = panels(bank, row, view, training_update=training_update)
            need(len(cells) == count, 'Exact prospective panel count required')
            record = {'case_key': case_key, 'cells': []}
            for column, (label, im, digest) in enumerate(cells):
                x = 8 + column * 264
                draw.text((x, y + 19), label, font=small, fill=(30, 30, 35))
                if im is None:
                    draw.rectangle((x, y + 35, x + 255, y + 290), fill=(218, 217, 214))
                    note = ('No aligned clean target' if column == 1 else 'MISSING - NOT EXECUTED')
                    draw.text((x + 7, y + 110), note, font=font, fill=(55, 52, 52))
                else:
                    sheet.paste(im, (x, y + 35))
                    need(np.array_equal(np.asarray(sheet.crop((x, y + 35, x + 256, y + 291))), np.asarray(im)),
                         'Rendered panel pixels differ')
                record['cells'].append({'label': label, 'RGB8_sha256': digest, 'missing': im is None,
                                        'rectangle': [x, y + 35, x + 256, y + 291]})
            manifest.append(record)
        target = VISUAL / (name + '.png')
        with target.open('xb') as stream:
            sheet.save(stream, format='PNG')
        with Image.open(target) as decoded:
            need(np.array_equal(np.asarray(decoded), np.asarray(sheet)), 'Complete saved sheet readback differs')
        sheets.append({'sheet_id': name, 'bank': bank, 'view': view, 'file': pin(target),
                       'case_rows': manifest, 'training_update': training_update,
                       'generated_from_actual_saved_outputs_only': True})
    for group in plan['groups']:
        render(group['sheet_id'], group['bank'], group['view'], group['case_keys'])
    primary_count = len(sheets)
    need(primary_count == 54, 'All54 prospective primary sheets required')
    train_updates = [key[4] for key in index if key[3] == 'photographic_optimization']
    if train_updates:
        latest = max(train_updates)
        keys = sorted({bank + ':' + case for arm, bank, case, phase, update in index
                       if phase == 'photographic_optimization' and update == latest})
        for view in ('raw', 'native_png'):
            for offset in range(0, len(keys), 8):
                render('partial_optimization_u' + str(latest).zfill(4) + '_' + view + '_' + str(offset // 8 + 1).zfill(2),
                       'new_optimization', view, keys[offset:offset + 8], training_update=latest)
    for binding in consumed.values():
        need(pin(ROOT / binding['path']) == binding, 'Consumed evidence changed during rendering')
    receipt = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'primary_sheets': primary_count, 'additional_partial_optimization_sheets': len(sheets) - primary_count,
        'sheets': sheets, 'consumed_files': list(consumed.values()), 'consumed_before_after_exact': True,
        'viewed_by_root': False, 'visual_review_complete': False,
        'RAW_display_conversion_only': 'floor(float32_RAW * float32_255), preserving separate actual RAW data/metrics',
        'missing_terminal_outputs_not_fabricated': True, 'new_neural_training_VM_calls': 0,
        'model_qualified': False}
    with (OUT / 'visual_sheets.json').open('xb') as stream:
        stream.write((json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({'complete': True, 'primary_sheets': primary_count,
                      'additional_sheets': len(sheets) - primary_count, 'root_review_pending': True}))


if __name__ == '__main__':
    main()

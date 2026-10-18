"""Render saved diagnostic PNGs for root visual review; no neural execution."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_path_diagnostic_v1_return_review'
MODES = ('E32', 'E32_grad', 'T32', 'TBF16')
ARMS = ('original_moments__current_SFT', 'current_moments__original_SFT',
        'original_moments__original_SFT')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest_path = REVIEW / 'pixels_v2/visual_manifest.json'
    audit = json.loads((REVIEW / 'pixels_v2/pixel_audit.json').read_text())
    assert audit['complete'] and audit['conditions_verified'] == 104
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['conditions']
    assert len(rows) == 104
    by = {row['condition']: row for row in rows}
    assert len(by) == 104
    cases = list(dict.fromkeys(row['case_id'] for row in rows if row['condition_kind'] == 'path'))
    repeat_cases = list(dict.fromkeys(row['case_id'] for row in rows if row['condition_kind'] == 'repeat'))
    assert len(cases) == 8 and len(repeat_cases) == 2
    out = REVIEW / 'visual_sheets'
    out.mkdir(exist_ok=False)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    title_font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 21)
    covered, sheets = set(), []

    def render(name, title, chosen, kind, labels, paths_for):
        width, tile, margin, stride, top = 1348, 192, 20, 218, 120
        canvas = Image.new('RGB', (width, top + len(chosen) * 248 + 34), '#f5f3ed')
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 14), title, fill='#162b38', font=title_font)
        draw.text((margin, 46), 'Eight fixed synthetic TRAIN inputs. Diagnostic images only; no new model or quality pass.',
                  fill='#273e4b', font=small)
        draw.text((margin, 68), 'Saved native PNGs, 256 to 192 px Lanczos display; no brightness, color or sharpening changes.',
                  fill='#273e4b', font=small)
        for col, label in enumerate(labels):
            draw.text((margin + col * stride, 94), label, fill='#162b38', font=small)
        bindings, sheet_conditions = [], set()
        for number, case in enumerate(chosen):
            cells, conditions = paths_for(case)
            assert len(cells) == 6
            row_y = top + number * 248
            for col, (path, expected) in enumerate(cells):
                path = Path(path)
                assert sha(path) == expected
                with Image.open(path) as image:
                    assert image.mode == 'RGB' and image.size == (256, 256)
                    display = image.resize((tile, tile), Image.Resampling.LANCZOS)
                    canvas.paste(display, (margin + col * stride, row_y))
                bindings.append({'case_id': case, 'column': labels[col], 'path': str(path), 'sha256': expected})
            draw.text((margin, row_y + tile + 7), case + ' | ' + kind, fill='#162b38', font=font)
            covered.update(conditions)
            sheet_conditions.update(conditions)
        path = out / name
        with path.open('xb') as stream:
            canvas.save(stream, format='PNG')
        sheets.append({'path': str(path), 'sha256': sha(path), 'title': title,
                       'conditions': [row['condition'] for row in rows
                                      if row['condition'] in sheet_conditions],
                       'bindings': bindings})

    def prefix_cells(row):
        return [(row['input'], row['input_sha256']), (row['target'], row['target_sha256'])]

    for state in ('original', 'stopped50'):
        for page in range(2):
            def paths_for(case, state=state):
                conditions = [f'paths/{state}/{case}/{mode}' for mode in MODES]
                data = [by[c] for c in conditions]
                return prefix_cells(data[0]) + [(r['native_png'], r['native_png_sha256']) for r in data], conditions
            render(f'paths_{state}_{page + 1}.png', f'{state}: FP32 inference / FP32 autograd / FP32 training / BF16 training',
                   cases[page * 4:(page + 1) * 4], 'path',
                   ('Input', 'Clean target', 'E32 inference', 'E32 autograd', 'T32 training', 'TBF16 training'), paths_for)
        def repeats_for(case, state=state):
            conditions = [f'repeats/{state}/{case}/{mode}' for mode in MODES]
            data = [by[c] for c in conditions]
            return prefix_cells(data[0]) + [(r['native_png'], r['native_png_sha256']) for r in data], conditions
        render(f'repeats_{state}.png', f'{state}: repeated passes (numerical repeat evidence)', repeat_cases, 'repeat',
               ('Input', 'Clean target', 'E32 repeat', 'E32 grad repeat', 'T32 repeat', 'TBF16 repeat'), repeats_for)

    for page in range(2):
        def interventions_for(case):
            baseline = by[f'paths/stopped50/{case}/E32']
            conditions = [f'interventions/{case}/{arm}' for arm in ARMS]
            data = [baseline] + [by[c] for c in conditions]
            return prefix_cells(baseline) + [(r['native_png'], r['native_png_sha256']) for r in data], conditions
        render(f'interventions_{page + 1}.png', 'Stopped50 conditional moments / fusion (same stopped content codes)',
               cases[page * 4:(page + 1) * 4], 'intervention',
               ('Input', 'Clean target', 'Current / current', 'Original moments', 'Original fusion', 'Both original'), interventions_for)

    assert covered == set(by) and len(sheets) == 8
    result = {'complete': True, 'sheets': sheets, 'conditions_covered': len(covered),
              'manifest_sha256': sha(manifest_path), 'source_sha256': sha(__file__),
              'display_only_no_model_execution': True, 'root_visual_review_complete': False}
    (out / 'render_receipt.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    links = '\n'.join(f"- [{Path(row['path']).name}]({Path(row['path']).name})" for row in sheets)
    (out / 'INDEX.md').write_text('# Saved diagnostic images\n\n' + links +
        '\n\nEight TRAIN cases; 104 conditions. No model qualification follows.\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'sheets': len(sheets), 'conditions': len(covered)}))


if __name__ == '__main__':
    main()

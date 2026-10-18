"""Render independently audited saved PNGs; no neural or enhancement calls."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    review = root / 'outputs/codeformer_fp32_comparison_v1_return_review'
    source = review / 'metrics_v1/visual_manifest.json'
    manifest = json.loads(source.read_text())
    require(manifest['complete'], 'Complete saved-pixel audit required')
    snapshots = {}
    for row in manifest['cases']:
        snapshots.setdefault(row['snapshot'], {})[row['id']] = row
    require(set(snapshots) == {0, 50} and len(snapshots[50]) == 406 and
            set(snapshots[0]) == set(snapshots[50]), 'Expected complete0/50 returned cohorts')
    out = review / 'visual_sheets'
    out.mkdir(exist_ok=False)
    small, font, title = [ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size) for size in (12, 14, 18)]
    tile, height = 192, 230
    columns = ('Input / resize', 'Target / native note', 'Original CodeFormer', 'BF16 V2: u50', 'FP32 comparison: u50')
    groups = {}
    for row in snapshots[50].values():
        group = ('native_' + row['source'] if row['evidence'] == 'native_unpaired' else
                 'old_paired_' + row['source'] if row['id'].startswith('comparison_') else 'HQ_validation')
        groups.setdefault(group, []).append(row)
    pages, shown = [], []
    for group, rows in sorted(groups.items()):
        rows.sort(key=lambda row: (row['profile'], row['id']))
        for start in range(0, len(rows), 8):
            selected = rows[start:start + 8]
            canvas = Image.new('RGB', (tile * 5, 86 + height * len(selected)), '#f4f1ea')
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 5), group + ' | delivered PNG | both trained comparisons quality-stopped', fill='#171717', font=title)
            draw.text((6, 31), 'Identical observations, first50 update budgets and losses. No exposure correction or sharpening.',
                      fill='#171717', font=small)
            for col, label in enumerate(columns):
                draw.text((col * tile + 4, 55), label, fill='#171717', font=font)
            bindings = []
            for line, row in enumerate(selected):
                old = snapshots[0][row['id']]
                y = 82 + line * height
                draw.text((4, y), row['id'].removeprefix('validation_').removeprefix('comparison_'),
                          fill='#171717', font=small)
                paths = (row['input'], row['target'], old['native_png'],
                         row['BF16_update50_native_png'], row['native_png'])
                hashes = (row['input_sha256'], row.get('target_sha256'), old['native_png_sha256'],
                          row['BF16_update50_native_png_sha256'], row['native_png_sha256'])
                for col, (path, expected) in enumerate(zip(paths, hashes)):
                    if path is None:
                        require(col == 1 and row['evidence'] == 'native_unpaired', 'Only native clean target may be absent')
                        draw.rectangle((col * tile, y + 18, (col + 1) * tile - 1, y + tile + 18), fill='#e1ded6')
                        draw.text((col * tile + 8, y + 74), 'Native CCTV: unpaired', fill='#171717', font=font)
                        draw.text((col * tile + 8, y + 96), 'No clean reference', fill='#171717', font=font)
                        continue
                    require(sha(path) == expected, 'Audited image changed before render')
                    with Image.open(path) as image:
                        require(image.size == (256, 256) and image.mode == 'RGB', 'Expected exact RGB256 source')
                        canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (col * tile, y + 18))
                    bindings.append({'case_id': row['id'], 'column': columns[col], 'path': str(path), 'sha256': expected})
                shown.append(row['id'])
            filename = f'comparison_{len(pages) + 1:02d}.png'
            with (out / filename).open('xb') as stream:
                canvas.save(stream, format='PNG')
            pages.append({'file': filename, 'sha256': sha(out / filename), 'group': group,
                          'case_ids': [r['id'] for r in selected], 'bindings': bindings, 'tile_pixels': tile})
    require(len(shown) == len(set(shown)) == 406, 'Visual cohort missing or duplicated')
    receipt = {'complete': True, 'pages': pages, 'case_pairs': 406,
        'returned_snapshot_outputs': 812, 'BF16_comparison_outputs': 406,
        'source_visual_manifest_sha256': sha(source), 'renderer_sha256': sha(__file__),
        'display_policy': 'Original saved native256PNG to192px Lanczos only; no enhancement',
        'root_visual_review_pending': True, 'user_acceptance_pending': True,
        'model_qualified': False, 'local_neural_gradient_optimizer_VM_calls': 0}
    with (out / 'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    lines = ['# FP32 comparison: all406 development cases', '',
        'Columns: input, paired target/native note, original CodeFormer, BF16 V2 u50, FP32 u50.', '',
        'Both adaptations stopped on quality checks. Same first50 observations/update budget; no performance claim.', '']
    lines.extend(f"- [{page['file']}: {page['group']}]({page['file']})" for page in pages)
    (out / 'INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'pages': len(pages), 'cases': len(shown), 'review_pending': True}))


if __name__ == '__main__':
    main()

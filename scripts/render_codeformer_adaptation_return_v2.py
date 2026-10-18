"""Render all audited V2 cases beside immutable original/V1 outputs."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from research.codeformer_adaptation_v1.common import require, sha256, write_json


def main():
    root = Path.cwd()
    review = root / 'outputs/codeformer_adaptation_v2_return_review'
    source = review / 'metrics_v1/visual_manifest.json'
    manifest = json.loads(source.read_text())
    require(manifest['complete'], 'Complete saved-pixel audit required')
    snapshots = {}
    for row in manifest['cases']:
        snapshots.setdefault(row['snapshot'], {})[row['id']] = row
    require(set(snapshots) == {0, 50} and len(snapshots[50]) == 406 and
            set(snapshots[0]) == set(snapshots[50]), 'Expected complete 0/50 returned cohorts')
    metrics = json.loads((review / 'metrics_v1/metrics.json').read_text())['rows']
    scores = {(r['snapshot'], r['id'], r['view']): r for r in metrics}
    out = review / 'visual_sheets'
    require(not out.exists(), 'Preserve prior visual evidence')
    out.mkdir()
    fonts = [ImageFont.truetype('C:/Windows/Fonts/arial.ttf', s) for s in [12, 14, 18]]
    small, font, title = fonts
    tile, height = 192, 234
    columns = ['Input / resize', 'Target / native note', 'Original CodeFormer', 'V1: stopped u200', 'V2: stopped u50']
    groups = {}
    for row in snapshots[50].values():
        group = ('native_' + row['source'] if row['evidence'] == 'native_unpaired' else
                 'old_paired_' + row['source'] if row['id'].startswith('comparison_') else 'HQ_validation')
        groups.setdefault(group, []).append(row)
    pages, shown = [], []
    for group, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (r['profile'], r['id']))
        for start in range(0, len(rows), 8):
            selected = rows[start:start + 8]
            canvas = Image.new('RGB', (tile * 5, 86 + height * len(selected)), '#f4f1ea')
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 5), group + ' | native PNG | both adaptations stopped', fill='#171717', font=title)
            draw.text((6, 31), 'Same inputs. No exposure correction/sharpening. Different training budgets: V1 200, V2 50 updates.',
                      fill='#171717', font=small)
            for col, label in enumerate(columns):
                draw.text((col * tile + 4, 55), label, fill='#171717', font=font)
            for line, row in enumerate(selected):
                old = snapshots[0][row['id']]
                require(row.get('V1_update200_native_png'), 'V1 comparison path missing')
                y = 82 + line * height
                draw.text((4, y), row['id'].removeprefix('validation_').removeprefix('comparison_'),
                          fill='#171717', font=small)
                paths = [row['input'], row['target'], old['native_png'],
                         row['V1_update200_native_png'], row['native_png']]
                hashes = [row['input_sha256'], row.get('target_sha256'), old['native_png_sha256'],
                          row['V1_update200_native_png_sha256'], row['native_png_sha256']]
                for col, (path, expected) in enumerate(zip(paths, hashes)):
                    if path is None:
                        draw.rectangle((col * tile, y + 18, (col + 1) * tile - 1, y + tile + 18), fill='#e1ded6')
                        draw.text((col * tile + 8, y + 74), 'Native CCTV: unpaired', fill='#171717', font=font)
                        draw.text((col * tile + 8, y + 96), 'No clean reference', fill='#171717', font=font)
                        continue
                    require(sha256(path) == expected, 'Audited image changed before render')
                    with Image.open(path) as image:
                        require(image.size == (256, 256), 'Unexpected 256 source geometry')
                        canvas.paste(image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS),
                                     (col * tile, y + 18))
                if row['evidence'] == 'paired_synthetic':
                    delta = (scores[50, row['id'], 'native_png']['mae'] - scores[0, row['id'], 'native_png']['mae']) * 255
                    draw.text((4 * tile + 4, y + 212), 'V2 MAE delta%+.2f /255' % delta,
                              fill='#8a2020' if delta > 0 else '#205c38', font=small)
                shown.append(row['id'])
            filename = f'comparison_{len(pages)+1:02d}.png'
            canvas.save(out / filename)
            pages.append({'file': filename, 'sha256': sha256(out / filename), 'group': group,
                          'case_ids': [r['id'] for r in selected], 'tile_pixels': tile})
    require(len(shown) == len(set(shown)) == 406, 'Visual cohort missing or duplicated')
    write_json(out / 'manifest.json', {'complete': True, 'pages': pages, 'case_pairs': 406,
        'distinct_snapshot_outputs_represented': 1218, 'V2_update0_equals_original_checked_by_metrics_audit': True,
        'source_visual_manifest_sha256': sha256(source),
        'review_thumbnail_policy': 'Pillow LANCZOS256-to192; all original256files retained',
        'visual_review_pending': True, 'user_acceptance_pending': True, 'model_qualified': False,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0})
    lines = ['# CodeFormer V2: all returned comparisons', '',
             'Both adaptations stopped. Same inputs; V1 200 updates versus V2 50, not an equal-budget causal comparison.', '',
             'Columns: input, paired target/native note, original CodeFormer, V1, V2.', '']
    lines += [f"- [{r['file']}: {r['group']}]({r['file']})" for r in pages]
    (out / 'INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'pages': len(pages), 'cases': len(shown), 'review_pending': True}))


if __name__ == '__main__':
    main()

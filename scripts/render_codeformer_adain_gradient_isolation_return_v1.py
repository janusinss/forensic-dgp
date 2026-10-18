"""Render every independently audited saved PNG at exactly256 pixels.

Pure CPU binding audit, no display processing and no visual acceptance claim.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1.return_pixels import require, sha, read, write, STATES


def render(review, metrics_directory):
    review = Path(review).resolve(strict=True)
    require(metrics_directory.startswith('metrics_v') and metrics_directory.removeprefix('metrics_v').isdigit(),
            'Explicit versioned metrics directory required')
    source = review / metrics_directory / 'visual_manifest.json'
    manifest = read(source); audit = read(review / metrics_directory / 'metrics_audit.json')
    require(manifest['complete'] and audit['complete'] and len(manifest['cases']) == 406,
            'Complete full-cohort pixel audit required')
    out = review / 'visual_sheets'; out.mkdir(exist_ok=False)
    font, small, title = [ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size) for size in (14, 12, 18)]
    column_step, tile, top, row_height = 270, 256, 110, 286
    pages, shown, all_cells = [], [], []
    columns = ('Observed input', 'Direct target / native note', 'Original CodeFormer: u0',
               'Retained FP32: u50', 'AdaIN gradient isolation: u50')

    def page(filename, label, rows, page_columns=columns, kind='full_DEV_cohort'):
        canvas = Image.new('RGB', (column_step * len(page_columns), top + row_height * len(rows) + 6), '#f4f1ea')
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), label, font=title, fill='#111111')
        draw.text((6, 31), 'Exact saved256 pixels | no enhancement/resampling | restoration candidates are not qualified', font=small, fill='#111111')
        draw.text((6, 51), 'Native CCTV has no aligned clean target; source groups remain separate. ROOT visual review pending.', font=small, fill='#111111')
        for col, name in enumerate(page_columns):
            draw.text((column_step * col + 5, 78), name, font=font, fill='#111111')
        cells = []
        for line, (rid, description, sources) in enumerate(rows):
            y = top + line * row_height
            draw.text((5, y), rid + ' | ' + description, font=small, fill='#111111')
            for col, source_cell in enumerate(sources):
                box = [col * column_step + 5, y + 22, col * column_step + 5 + tile, y + 22 + tile]
                if source_cell is None:
                    draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill='#e0dcd4')
                    draw.text((box[0] + 8, box[1] + 100), 'Native CCTV: unpaired', font=font, fill='#111111')
                    draw.text((box[0] + 8, box[1] + 124), 'No clean reference', font=font, fill='#111111')
                    continue
                path, digest, role = source_cell
                require(sha(path) == digest, 'Audited image changed before rendering')
                with Image.open(path) as image:
                    require(image.mode == 'RGB' and image.size == (tile, tile), 'Exact RGB256 source required')
                    pixels = np.asarray(image).copy()
                    canvas.paste(image, box[:2])
                cells.append({'case_id': rid, 'column': page_columns[col], 'column_index': col,
                    'role': role, 'source_path': str(Path(path).resolve()), 'source_sha256': digest,
                    'source_pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(), 'pixel_box': box})
                if role in STATES or role.startswith('witness_'):
                    shown.append((rid, role))
        target = out / filename
        with target.open('xb') as stream:
            canvas.save(stream, format='PNG')
        with Image.open(target) as saved:
            pixels = np.asarray(saved).copy()
        for cell in cells:
            x0, y0, x1, y1 = cell['pixel_box']
            with Image.open(cell['source_path']) as original:
                require(np.array_equal(pixels[y0:y1, x0:x1], np.asarray(original)),
                        'Saved page cell differs from its exact source pixels')
        all_cells.extend(cells)
        pages.append({'file': filename, 'sha256': sha(target), 'kind': kind, 'heading': label,
            'size': list(canvas.size), 'case_ids': [row[0] for row in rows],
            'tile_pixels': tile, 'resampling': 'none', 'bindings': cells})

    groups = defaultdict(list)
    for row in manifest['cases']:
        group = ('HQ_validation_' + row['profile'] if row['id'].startswith('validation_') else
            'native_' + row['source'] if row['evidence'] == 'native_unpaired' else 'old_paired_' + row['source'])
        groups[group].append(row)
    for group, cases in sorted(groups.items()):
        cases.sort(key=lambda row: (row['profile'], row['id']))
        for start in range(0, len(cases), 8):
            rows = []
            for row in cases[start:start + 8]:
                cells = [(row['input'], row['input_sha256'], 'input')]
                cells.append((row['target'], row['target_sha256'], 'target') if row['target'] else None)
                for state in ('original_update0', 'retained_FP32_update50', 'isolated_update50'):
                    value = row['states'][state]
                    cells.append((value['native_png'], value['native_png_sha256'], state))
                rows.append((row['id'], row['source'] + ' / ' + row['profile'], cells))
            page('comparison_%02d.png' % (len(pages) + 1), group + ' | %d–%d / %d' %
                (start + 1, min(start + 8, len(cases)), len(cases)), rows)
    cohort_page_count = len(pages)
    witness_rows = []
    for slot in (1, 2):
        values = {row['mode']: row for row in manifest['witness'] if row['slot'] == slot}
        require(set(values) == {'original', 'isolated'}, 'Full pre-Adam witness pair required')
        first, second = values['original'], values['isolated']
        require(first['input_sha256'] == second['input_sha256'], 'Witness pair does not share input')
        cells = [(first['input'], first['input_sha256'], 'input'),
            (first['native_png'], first['native_png_sha256'], 'witness_original'),
            (second['native_png'], second['native_png_sha256'], 'witness_isolated')]
        witness_rows.append(('update_0003_slot%d' % slot, first['reference_id'] + ' / TRAIN / pre-Adam', cells))
    page('witness_01.png', 'Pre-Adam witness | original and isolated values must be exact', witness_rows,
         ('Saved TRAIN input', 'Original route: before Adam', 'Isolated route: before Adam'), 'pre_Adam_witness')
    expected = {(row['id'], state) for row in manifest['cases'] for state in STATES} | {
        ('update_0003_slot%d' % slot, 'witness_' + mode) for slot in (1, 2) for mode in ('original', 'isolated')}
    require(len(shown) == len(set(shown)) == 1222 and set(shown) == expected,
            'Every812new+406prior+4witness output must appear exactly once')
    result = {'complete': True, 'source_visual_manifest_sha256': sha(source),
        'source_metrics_audit_sha256': sha(review / metrics_directory / 'metrics_audit.json'),
        'source_renderer_sha256': sha(__file__), 'pages': pages, 'cohort_pages': cohort_page_count,
        'total_pages': len(pages), 'cases': 406, 'new_snapshot_outputs': 812,
        'retained_FP32_u50_outputs': 406, 'witness_outputs': 4, 'total_output_cells': 1222,
        'all_source_cells': len(all_cells), 'source_cells_reopened_and_exact': True,
        'groups': dict(Counter({group: len(items) for group, items in groups.items()})),
        'tile_pixels': 256, 'resampling': 'none', 'display_processing': 'none',
        'root_personal_visual_review_pending': True, 'visual_acceptance_claim': False, 'model_qualified': False}
    write(out / 'manifest.json', result)
    with (out / 'INDEX.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('# AdaIN gradient isolation V1: complete DEV and pre-Adam witness sheets\n\n')
        stream.write('Exact saved256 pixels, no enhancement/resampling. All406 cases and all1222 output cells.\n\n')
        for entry in pages:
            stream.write('- [%s](%s): %s\n' % (entry['file'], entry['file'], entry['heading']))
    return {'complete': True, 'pages': len(pages), 'cohort_pages': cohort_page_count,
        'all_source_cells': len(all_cells), 'total_output_cells': len(shown), 'manifest_sha256': sha(out / 'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--review', required=True); parser.add_argument('--metrics', required=True)
    args = parser.parse_args(); print(json.dumps(render(args.review, args.metrics), indent=2))

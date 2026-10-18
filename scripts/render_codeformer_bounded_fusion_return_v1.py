"""Render exact saved256 PNGs; CPU binding audit only, no visual verdict."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_bounded_fusion_feasibility_v1.return_pixels import read, require, sha, write


def render(review, metrics_directory):
    review = Path(review).resolve(strict=True)
    require(metrics_directory.startswith('metrics_v') and metrics_directory.removeprefix('metrics_v').isdigit(),
        'Explicit versioned metrics directory required')
    metric_root = review / metrics_directory
    source = metric_root / 'visual_manifest.json'
    manifest = read(source)
    metrics = read(metric_root / 'metrics_audit.json')
    require(manifest['complete'] and metrics['complete'] and len(manifest['cases']) == 170,
        'Complete170 saved-pixel audit is required before render')
    out = review / 'visual_sheets'
    require(not out.exists(), 'Preserve existing visual sheets')
    out.mkdir()
    rows = {(r['case_id'], r['mode'], r['section']): r for r in manifest['cases']}
    require(len(rows) == 170, 'Duplicated visual source bindings')
    small, font, title = [ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size) for size in (13, 16, 21)]
    pages, shown, bindings_count = [], [], 0
    tile, row_height, top = 256, 308, 110

    def write_page(filename, heading, columns, planned, kind):
        nonlocal bindings_count
        canvas = Image.new('RGB', (tile * len(columns), top + row_height * len(planned) + 10), '#f4f1ea')
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 6), heading, font=title, fill='#151515')
        draw.text((6, 34), 'TRAIN only | zero-update stopped-state intervention | saved delivered PNGs pasted at exact256 pixels', font=small, fill='#151515')
        draw.text((6, 54), 'No display enhancement, resizing, clean-target correction or model qualification.', font=small, fill='#151515')
        for col, label in enumerate(columns):
            draw.text((col * tile + 5, 80), label, font=font, fill='#151515')
        bindings = []
        for line, (label, sublabel, cells) in enumerate(planned):
            y = top + line * row_height
            draw.text((6, y), label, font=font, fill='#151515')
            draw.text((6, y + 21), sublabel, font=small, fill='#151515')
            for col, cell in enumerate(cells):
                box = (col * tile, y + 44, (col + 1) * tile, y + 44 + tile)
                if cell is None:
                    draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill='#e3dfd7')
                    draw.text((box[0] + 10, box[1] + 110), 'No additional output', font=font, fill='#151515')
                    continue
                row, field = cell
                key = 'native_png' if field == 'output' else field
                path, digest = row[key], row[key + '_sha256']
                require(sha(path) == digest, 'Audited image changed before render')
                with Image.open(path) as image:
                    require(image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 display source required')
                    pixels = np.asarray(image).copy()
                    canvas.paste(image, box[:2])
                bindings.append({'case_id': row['case_id'], 'reference_id': row['reference_id'],
                    'profile': row['profile'], 'section': row['section'], 'mode': row['mode'],
                    'field': field, 'column': columns[col], 'row_index': line,
                    'source_path': str(Path(path).resolve()), 'source_sha256': digest,
                    'source_pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                    'pixel_box': list(box)})
                if field == 'output':
                    shown.append((row['case_id'], row['mode'], row['section']))
        target = out / filename
        with target.open('xb') as stream:
            canvas.save(stream, format='PNG')
        # Independently reopen the lossless page and verify every cell's exact pixels.
        with Image.open(target) as image:
            saved_page = np.asarray(image).copy()
        for binding in bindings:
            x0, y0, x1, y1 = binding['pixel_box']
            with Image.open(binding['source_path']) as source_image:
                wanted = np.asarray(source_image)
            require(np.array_equal(saved_page[y0:y1, x0:x1], wanted), 'Rendered cell pixels differ from source PNG')
        bindings_count += len(bindings)
        pages.append({'file': filename, 'sha256': sha(target), 'kind': kind,
            'size': list(canvas.size), 'tile_pixels': 256, 'resampling': 'none',
            'case_ids': [cells[0][0]['case_id'] for _, _, cells in planned], 'bindings': bindings})

    ids = manifest['main_cases']
    columns = ['Observed input', 'Direct clean256 target', 'Original CodeFormer', 'Stopped FP32: u50',
               'Original-fusion rollback', 'Bounded constant g=1', 'Bounded observed gate']
    require(len(ids) == 32, 'Exactly32 selected main observations required')
    for pair_index in range(16):
        planned = []
        pair = ids[pair_index * 2:pair_index * 2 + 2]
        first = rows[pair[0], 'original', 'paths']
        second = rows[pair[1], 'original', 'paths']
        require(first['reference_id'] == second['reference_id'] and first['exposure'] == 'normal' and second['exposure'] == 'dark',
            'Each page must show the matched normal/dark pair from one photograph')
        for rid in pair:
            original = rows[rid, 'original', 'paths']
            canary = ' | FIXED STRESS CANARY' if original['is_stress_canary'] else ''
            label = rid + ' | ' + original['reference_id'] + ' | ' + original['profile'] + canary
            sub = 'Observed central192 luminance %.6f | fixed input gate %.6f' % (
                original['central192_sRGB_mean_luminance'], original['gate'])
            planned.append((label, sub, [(original, 'input'), (original, 'target')] +
                [(rows[rid, mode, 'paths'], 'output') for mode in ('original', 'stopped', 'rollback', 'bounded_constant', 'bounded_fixed')]))
        write_page('pair_%02d.png' % (pair_index + 1), 'Matched TRAIN pair %02d /16 | %s' % (pair_index + 1, first['reference_id']),
            columns, planned, 'matched_normal_dark_pair')
    for canary_index, rid in enumerate(manifest['canary_ids'], 1):
        original, stopped = rows[rid, 'original', 'paths'], rows[rid, 'stopped', 'paths']
        columns = ['Observed input', 'Direct clean256 target', 'Native baseline', 'Exact native repeat', 'Zero g=1 / identity', 'Zero observed gate']
        planned = [
            (rid + ' | ORIGINAL parity controls', 'Original/original zero-delta wrappers; compare exact baseline and repeat.',
                [(original, 'input'), (original, 'target'), (original, 'output'),
                 (rows[rid, 'original', 'repeats'], 'output'), (rows[rid, 'bounded_constant', 'zero_delta'], 'output'),
                 (rows[rid, 'bounded_fixed', 'zero_delta'], 'output')]),
            (rid + ' | STOPPED identity controls', 'Identity evaluates both fusion blocks and returns FS directly, without subtract/add.',
                [(stopped, 'input'), (stopped, 'target'), (stopped, 'output'),
                 (rows[rid, 'stopped', 'repeats'], 'output'), (rows[rid, 'native_identity', 'identity'], 'output'), None])]
        write_page('canary_%02d.png' % canary_index, 'Fixed stress-canary controls %d /2 | %s' % (canary_index, original['profile']),
            columns, planned, 'exact_repeat_zero_identity_controls')
    require(set(shown) == set(rows) and len(set(shown)) == 170 and len(shown) == 174,
        'All170 output PNGs must appear; only4 native baselines repeat on control sheets')
    require(len(pages) == 18 and bindings_count == 246, 'Exact18-page/246-cell display census differs')
    index_lines = ['# Bounded-fusion TRAIN diagnostic: all32 observations', '',
        '16 matched normal/dark pair pages; 2 fixed stress-canary control pages.', '',
        'Every source PNG is pasted at exactly256 pixels without resampling or enhancement.',
        'Zero optimizer updates; no independent DEV/FINAL/native CCTV or visual usefulness claim.', '']
    index_lines.extend('- [%s: %s](%s)' % (page['file'], page['kind'], page['file']) for page in pages)
    index = out / 'INDEX.md'
    with index.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(index_lines) + '\n')
    result = {'complete': True, 'pages': pages, 'page_count': 18, 'main_observations': 32,
        'matched_photographic_pairs': 16, 'unique_saved_outputs': 170, 'displayed_output_tiles': 174,
        'all_nonblank_source_cells': 246, 'all_source_and_page_hashes_verified': True,
        'all_cell_pixels_exact_after_PNG_reopen': True, 'all32_case_pair_coverage_exact': True,
        'source_visual_manifest_sha256': sha(source), 'source_metrics_audit_sha256': sha(metric_root / 'metrics_audit.json'),
        'source_metrics_directory': metrics_directory,
        'renderer_sha256': sha(__file__), 'INDEX_sha256': sha(index),
        'display_policy': 'Exact RGB256 PNG paste, no rescaling or enhancement; native512 metrics remain separate',
        'root_visual_review_pending': True, 'user_acceptance_pending': True, 'model_qualified': False,
        'neural_gradient_optimizer_VM_calls': 0}
    write(out / 'manifest.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'pages'}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--metrics', required=True)
    args = parser.parse_args()
    render(args.review, args.metrics)

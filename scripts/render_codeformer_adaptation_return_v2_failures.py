"""Show all three independently failed clear controls at their original 256 pixels."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from research.codeformer_adaptation_v1.common import require, sha256, write_json


def main():
    root = Path.cwd()
    review = root / 'outputs/codeformer_adaptation_v2_return_review'
    manifest_path = review / 'metrics_v1/visual_manifest.json'
    rows = json.loads(manifest_path.read_text())['cases']
    cases = {r['id']: r for r in rows if r['snapshot'] == 50}
    original = {r['id']: r for r in rows if r['snapshot'] == 0}
    audit = json.loads((review / 'metrics_v1/metrics_audit.json').read_text())
    guard = next(r for r in audit['snapshot_summaries'] if r['snapshot'] == 50)['independent_clear_guard']
    failed = {view: {key for key, delta in data['deltas'].items() if delta > .03}
              for view, data in guard['views'].items()}
    require(failed['raw'] == failed['native_png'] and len(failed['raw']) == 3,
            'Expected all three audited failures in both views')
    output = review / 'visual_sheets/clear_failures_256.png'
    receipt = output.with_suffix('.json')
    require(not output.exists() and not receipt.exists(), 'Preserve prior comparison')
    canvas = Image.new('RGB', (1280, 60 + 3 * 294), '#f4f1ea')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    draw.text((4, 3), 'All failed clear controls | unmodified native PNGs | V1 u200 versus V2 u50: unequal budgets', fill='black', font=font)
    for col, title in enumerate(['Input', 'Target', 'Original CodeFormer', 'V1 stopped u200', 'V2 stopped u50']):
        draw.text((col * 256 + 4, 30), title, fill='black', font=font)
    bound = []
    for index, case_id in enumerate(sorted(failed['raw'])):
        row = cases[case_id]
        paths = [(row['input'], row['input_sha256']), (row['target'], row['target_sha256']),
                 (original[case_id]['native_png'], original[case_id]['native_png_sha256']),
                 (row['V1_update200_native_png'], row['V1_update200_native_png_sha256']),
                 (row['native_png'], row['native_png_sha256'])]
        y = 60 + index * 294
        draw.text((4, y), case_id + ' | V2 PNG MAE regression %.6f > 0.030000' % guard['views']['native_png']['deltas'][case_id], fill='#852222', font=font)
        for col, (path, digest) in enumerate(paths):
            require(sha256(path) == digest, 'Source hash changed')
            with Image.open(path) as image:
                require(image.size == (256, 256), 'Original 256 geometry required')
                canvas.paste(image.convert('RGB'), (col * 256, y + 25))
        bound.append({'id': case_id, 'source_paths_and_sha256': paths})
    canvas.save(output)
    write_json(receipt, {'complete': True, 'file': str(output), 'sha256': sha256(output),
                         'case_ids': sorted(failed['raw']), 'original_tile_pixels': 256,
                         'resampling': 'none', 'cases': bound, 'visual_review_pending': True,
                         'model_qualified': False, 'neural_calls': 0})
    print(json.dumps({'complete': True, 'cases': 3, 'file': str(output)}))


if __name__ == '__main__':
    main()

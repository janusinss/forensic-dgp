"""Bind root's actual saved-image review and render unaltered256px failures."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    review = root / 'outputs/codeformer_fp32_comparison_v1_return_review'
    notes_path = review / 'root_visual_notes_v1.json'
    notes = json.loads(notes_path.read_text())
    pages = json.loads((review / 'visual_sheets/manifest.json').read_text())
    binding = json.loads((review / 'visual_binding_audit_v1.json').read_text())
    assert binding['complete'] and notes['all52_seen'] and len(pages['pages']) == 52
    seen = [n for group in notes['batches'] for n in group['pages']]
    assert seen == list(range(1, 53))
    for page in pages['pages']:
        assert sha(review / 'visual_sheets' / page['file']) == page['sha256']
    visual = json.loads((review / 'metrics_v1/visual_manifest.json').read_text())
    cases = {(row['id'], row['snapshot']): row for row in visual['cases']}
    ids = ['comparison_va_asian_04382_clear', 'comparison_va_asian_09306_clear',
           'validation_va_ffhq_27440_clear']
    columns = ['Input', 'Paired target', 'Original CodeFormer', 'BF16 V2 u50', 'FP32 u50']
    canvas = Image.new('RGB', (1280, 940), '#f4f1ea')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    draw.text((6, 4), 'Same three clear-preservation failures | actual unaltered256px PNGs | both states quality-stopped', fill='black', font=font)
    for col, title in enumerate(columns):
        draw.text((256 * col + 4, 30), title, fill='black', font=font)
    native_bindings = []
    for index, rid in enumerate(ids):
        row = cases[rid, 50]; original = cases[rid, 0]
        paths = [row['input'], row['target'], original['native_png'],
                 row['BF16_update50_native_png'], row['native_png']]
        hashes = [row['input_sha256'], row['target_sha256'], original['native_png_sha256'],
                  row['BF16_update50_native_png_sha256'], row['native_png_sha256']]
        y = 60 + index * 292
        draw.text((6, y), rid, fill='black', font=font)
        for col, (path, digest) in enumerate(zip(paths, hashes)):
            assert sha(path) == digest
            with Image.open(path) as image:
                assert image.mode == 'RGB' and image.size == (256, 256)
                canvas.paste(image, (col * 256, y + 24))
            native_bindings.append({'id': rid, 'column': columns[col], 'path': path, 'sha256': digest})
    figure = review / 'visual_sheets/clear_failures_256.png'
    with figure.open('xb') as stream:
        canvas.save(stream, format='PNG')
    receipt = {'complete': True, 'reviewer': 'root', 'root_inspected_all_planned_pages': 52,
        'case_pairs': 406, 'returned_outputs_represented': 812, 'BF16_comparison_outputs': 406,
        'source_counts': {'HQ_validation': 318, 'native_ChokePoint': 24, 'native_QMUL': 24, 'old_paired': 40},
        'sheet_manifest_sha256': sha(review / 'visual_sheets/manifest.json'),
        'binding_audit_sha256': sha(review / 'visual_binding_audit_v1.json'),
        'root_notes_sha256': sha(notes_path), 'pages': pages['pages'], 'critical256_bindings': native_bindings,
        'critical256_direct_views': 12, 'critical_figure': {'path': str(figure), 'sha256': sha(figure)},
        'display_policy': '52 sheets use256-to192Lanczos; critical figure retains exact256 pixels; no enhancement',
        'visual_interpretation': notes, 'independent_final_review': False, 'user_acceptance_pending': True,
        'native_is_unpaired': True, 'model_qualified': False, 'app_adoption': False,
        'neural_gradient_optimizer_VM_calls': 0, 'reserved_final_pixel_reads': 0,
        'source_sha256': sha(__file__)}
    with (review / 'visual_review_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'complete': True, 'pages_reviewed': 52, 'cases': 406, 'model_qualified': False}))


if __name__ == '__main__':
    main()

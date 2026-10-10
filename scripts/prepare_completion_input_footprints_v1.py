"""Freeze an input-only assisted-mask review and exact256px source atlases; no models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_input_footprints_v1'
BASE = ROOT / 'outputs/dgp_app_covering_review_v3'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pixels(path, mask=False):
    with Image.open(path) as im: a = np.asarray(im.convert('L' if mask else 'RGB')).copy()
    assert a.shape == ((256, 256) if mask else (256, 256, 3))
    if mask: assert set(np.unique(a)) <= {0, 255}; return a > 0
    return a


def overlay(image, mask):
    out = image.astype(np.float64); out[mask] = .55 * out[mask] + .45 * np.array([16, 185, 129])
    return np.floor(out + .5).astype(np.uint8)


def main():
    assert not OUT.exists(); started = time.monotonic()
    plan = read(BASE / 'plan.json'); results = read(BASE / 'results.json'); assert results['complete'] and results['plan_sha256'] == sha(BASE / 'plan.json')
    pixel_plan = read(ROOT / 'outputs/completion_pixel_support_v1_r1/plan.json')
    assert len(plan['cases']) == len(pixel_plan['cases']) == 36
    old = {c['id']: c for c in pixel_plan['cases']}; cases, assets = [], {}
    for case in plan['cases']:
        cid = case['id']; prior = old[cid]
        automatic = BASE / ('images/' + cid + '_automatic.png')
        assert sha(automatic) == results['artifacts_sha256'][automatic.relative_to(BASE).as_posix()]
        for name in ['input', 'reviewed']: assert sha(ROOT / case[name]) == pixel_plan['sources_sha256'][case[name]]
        assert prior['operator_input_review'] == case['input_review']
        item = {key: case[key] for key in ['id', 'family', 'condition', 'input', 'reviewed', 'input_review', 'exposure', 'input_only_note']}
        item['automatic'] = automatic.relative_to(ROOT).as_posix(); item['existing_rejection_retained'] = case['input_review'] != 'usable'
        cases.append(item)
        for name in ['input', 'reviewed', 'automatic']: assets[item[name]] = sha(ROOT / item[name]); pixels(ROOT / item[name], name != 'input')
    app = read(ROOT / 'outputs/completion_mat_mirror_comparison_v1/protocol.json')['app_preservation_sha256']
    for name, digest in app.items(): assert sha(ROOT / name) == digest, name
    for source in [Path(__file__), BASE / 'plan.json', BASE / 'results.json', BASE / 'saved_output_audit.json',
                   ROOT / 'outputs/completion_pixel_support_v1_r1/plan.json', ROOT / 'outputs/completion_pixel_support_v1_r1/independent_audit.json',
                   ROOT / 'outputs/completion_mat_mirror_comparison_v1/visual_review.json', ROOT / 'PRACTICAL_OUTPUT_SCOPE.md', ROOT / 'SYSTEM_WORKFLOW_AND_GOAL.md']:
        assets[source.relative_to(ROOT).as_posix()] = sha(source)
    OUT.mkdir(); (OUT / 'input_atlases').mkdir()
    p = {'format': 'source-only-completion-footprint-review-v1', 'complete': True, 'frozen_UTC': datetime.now(timezone.utc).isoformat(),
         'purpose': 'Review complete input-observed covering extent inside the facial area before a new completion comparison; distinguish mask misses from generator failures.',
         'cases': cases, 'sources_sha256': assets, 'app_preservation_sha256': app,
         'case_selection': 'All36 previously exposed development photo cases; no output-based subset or new final identity.',
         'prior_outputs_already_reviewed': True, 'new_mask_annotations_use_inputs_only': True, 'no_claim_of_blinded_or_final_review': True,
         'source_scope': 'Original photographs and matched synthetic degradations; legacy native suffix does not mean native CCTV.',
         'hidden_ground_truth': None, 'unknown_pretraining_overlap': True, 'ethnicity_or_Zamboanga_inference': False,
         'mask_policy': {'core': 'Input-observed objects/obstructing strands inside approximate facial extent; not a diagnostic rectangle or uniform dilation of the prior mask.',
                         'context_margin_radius_pixels': 2, 'margin_shape': 'disk; clipped by per-input reviewed face eligibility and protected visible support',
                         'clear_glasses_and_uncovered_controls': 'Empty operator removal mask; no context expansion.',
                         'visible_frame_and_nonobstructing_hair': 'Protected, including where adjacent to a requested glare/hair margin.',
                         'matched_degraded_pair': 'Same source geometry and source-reviewed mask may be reused; record original-photo assistance, not isolated degraded-only labeling.',
                         'unsupported_or_insufficient': 'Keep four earlier input exclusions; request frontal/mild or less-covered crop.'},
         'criteria': ['Cover all visible obstructing material within the intended facial region, including mixed objects.',
                      'Preserve observed eyes/nose/mouth/outline, clear glasses and hair except where obstructed or inside the documented small margin.',
                      'Leave hands, hair, clothing and background outside facial obstruction intact.',
                      'Review source and degraded pairs at exact256px before fixing each annotation; record ambiguity instead of inventing segmentation truth.',
                      'Report saved automatic proposals and assisted annotations separately; no generator output is used to optimize the new mask.'],
         'prospective_generation': 'Must freeze corrected masks and comparisons before any generator call. No automatic seed, margin or outcome search.',
         'models_or_optimizers_constructed': False, 'model_forwards': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
         'app_changes': False, 'native_or_reserved_final_used': False, 'goal_complete': False}
    write(OUT / 'protocol.json', p)
    groups = {}
    for case in cases:
        suffix = '_native' if case['condition'] == 'original_photo' else '_degraded'
        assert case['id'].endswith(suffix); key = case['id'][:-len(suffix)]
        groups.setdefault(key, {})[case['condition']] = case
    assert len(groups) == 18 and all(len(g) == 2 for g in groups.values())
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13); row_height = 296; pages = []
    for begin in range(0, 18, 3):
        sheet = Image.new('RGB', (1280, 918), 'white'); draw = ImageDraw.Draw(sheet)
        for column, label in enumerate(['Source256', 'Existing assisted overlay', 'Saved automatic overlay', 'Degraded source256', 'Saved degraded automatic']): draw.text((column * 256 + 4, 7), label, font=font, fill='black')
        entries = []
        for row, (key, pair) in enumerate(list(groups.items())[begin:begin + 3]):
            a, b = pair['original_photo'], pair['synthetic_degraded_photo']
            original, degraded = pixels(ROOT / a['input']), pixels(ROOT / b['input'])
            old_mask = pixels(ROOT / a['reviewed'], True); assert np.array_equal(old_mask, pixels(ROOT / b['reviewed'], True))
            automatic, degraded_automatic = pixels(ROOT / a['automatic'], True), pixels(ROOT / b['automatic'], True)
            y = 30 + row * row_height
            for column, image in enumerate([original, overlay(original, old_mask), overlay(original, automatic), degraded, overlay(degraded, degraded_automatic)]): sheet.paste(Image.fromarray(image), (column * 256, y))
            draw.text((4, y + 260), key + ' | ' + a['family'] + ' | ' + a['input_review'], font=font, fill='black')
            entries.append({'base_id': key, 'case_ids': [a['id'], b['id']], 'row': row})
        path = OUT / 'input_atlases' / f'page_{begin // 3 + 1:02d}.png'; sheet.save(path)
        pages.append({'path': path.relative_to(OUT).as_posix(), 'sha256': sha(path), 'size': [1280, 918], 'entries': entries})
    write(OUT / 'atlas_receipt.json', {'complete': True, 'protocol_sha256': sha(OUT / 'protocol.json'), 'pages': pages,
          'source_cells_exact256_unresampled': True, 'overlay': 'Diagnostic45percent green only inside the indicated saved mask.',
          'model_outputs_displayed': False, 'new_masks_created': False, 'model_forwards': 0, 'optimizer_updates': 0,
          'seconds': time.monotonic() - started, 'cap_seconds': 120, 'actual_input_review_pending': True})
    print(json.dumps({'complete': True, 'protocol_sha256': sha(OUT / 'protocol.json'), 'cases': 36, 'input_pages': 6, 'new_masks': 0, 'model_calls': 0}))


if __name__ == '__main__': main()

"""Close root's completed visual review against exact saved PNG cells; CPU only."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_bounded_fusion_feasibility_v1_return_review'
NOTES = [
    'Clear-normal face outline and beard remain similar; stopped brightens and smooths. Both bounds restrain normal changes but reduce useful dark brightening.',
    'Clear glasses remain visible. Normal differences are modest; the dark row remains dimmer with the bounds than with stopped FP32.',
    'Glasses and headband remain visible. The bounds improve normal tone relative to stopped; the dark row loses some visibility. Teeth and texture still differ from the clean photograph.',
    'Darkest-normal stress case: bounded results reduce stopped smoothing/brightness, but the almost-unit fixed gate adds little beyond the constant bound. The severely dark counterpart loses much of stopped brightening.',
    'Stopped improves several gross severe-input facial details over original; bounds recover more crease/texture contrast. The dark outputs remain dim and neither intervention establishes exact clean facial geometry.',
    'Severe inputs: stopped removes some coarse false texture; bounds retain more contrast. Eye rendering and the visible forehead spot still differ from the paired target, especially in the dark row.',
    'Clear glasses are retained. The fixed gate mostly darkens the constant result; normal tone moves closer to target, while the dark row loses useful stopped brightness.',
    'Severe inputs still produce conspicuous eye/cheek lines and facial-shape deviations against the paired photograph. Stopped reduces some of the original false detail; bounds do not resolve the mismatch.',
    'Stopped reduces coarse forehead/face artifacts compared with original. Bounds retain some contrast but suppress dark visibility; jewelry, hair and mouth details remain uncertain.',
    'Strongly turned scope-challenge case: the clean photograph has closed eyelids, while restored outputs open the visible eye. Photometric interventions do not fix that facial-detail error; a clearer/more frontal crop is required for the intended app scope.',
    'Brightest-dark stress case: stopped avoids several original hair/face artifacts. The fixed gate suppresses useful adaptation more than the constant bound; nose/mouth/hair still do not reproduce the clean photograph exactly.',
    'Cap and visible face remain recognizable as image content, but skin, mouth and cap detail differ from the clean photograph. The fixed gate mainly changes tone; dark visibility remains limited.',
    'Stopped reduces several coarse original facial artifacts; bounds lie between rollback and stopped in tone/contrast. Eyes, cheek outline and hair remain different from the paired target.',
    'Original invents pronounced facial creases and an eyelid/expression change. Stopped improves several features, but neither bounded route resolves eyelid, expression, hairstyle or outline differences against the target.',
    'Light degradation: facial outline and clothing broadly remain; stopped smooths/brightens. Bounds preserve more contrast but diminish dark brightening, without a clear fixed-gate advantage in visible structure.',
    'Severe moustache/face example: stopped gives more readable dark output; bounds are dimmer. Eye rendering, smile and moustache shape remain imperfect against the paired target.',
    'All original repeat/zero-delta control images visibly match the native original. Stopped repeat and native-identity controls visibly match stopped; exact equality is additionally established by the independent pixel/state audits.',
    'All original repeat/zero-delta control images visibly match the native original. Stopped repeat and native-identity controls visibly match stopped; these are parity controls, not quality gains.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    sheet_root = REVIEW / 'visual_sheets'
    manifest_path = sheet_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['page_count'] == 18
    assert manifest['unique_saved_outputs'] == 170 and manifest['all_nonblank_source_cells'] == 246
    assert manifest['source_metrics_audit_sha256'] == sha(REVIEW / 'metrics_v2/metrics_audit.json')
    expected = ['pair_%02d.png' % i for i in range(1, 17)] + ['canary_01.png', 'canary_02.png']
    assert [row['file'] for row in manifest['pages']] == expected
    outputs, main_cases, pages, cells = set(), set(), [], 0
    for page, note in zip(manifest['pages'], NOTES):
        path = sheet_root / page['file']
        assert sha(path) == page['sha256'] and page['tile_pixels'] == 256 and page['resampling'] == 'none'
        with Image.open(path) as im:
            assert im.mode == 'RGB' and list(im.size) == page['size']
            rendered = np.asarray(im).copy()
        for row in page['bindings']:
            source = Path(row['source_path'])
            assert sha(source) == row['source_sha256']
            with Image.open(source) as im:
                assert im.mode == 'RGB' and im.size == (256, 256)
                pixels = np.asarray(im).copy()
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == row['source_pixel_sha256']
            x0, y0, x1, y1 = row['pixel_box']
            assert np.array_equal(rendered[y0:y1, x0:x1], pixels)
            if row['field'] == 'output':
                outputs.add((row['case_id'], row['mode'], row['section']))
                if row['section'] == 'paths':
                    main_cases.add(row['case_id'])
            cells += 1
        pages.append({'path': str(path.relative_to(ROOT)), 'sha256': page['sha256'],
                      'case_ids': page['case_ids'], 'root_personally_viewed': True, 'observation': note})
    assert len(outputs) == 170 and len(main_cases) == 32 and cells == 246
    value = {'complete': True, 'root_personally_viewed_all_planned_sheets': True,
        'sheets': pages, 'sheets_reviewed': 18, 'main_TRAIN_cases': 32,
        'photographic_references': 16, 'unique_saved_outputs_shown': 170,
        'all_nonblank_source_cells_independently_rechecked': cells,
        'manifest_sha256': sha(manifest_path), 'source_sha256': sha(__file__),
        'extra_actual256_canary_outputs_viewed': {'case_ids': ['train_u0011_slot1', 'train_u0032_slot2'],
            'modes_per_case': ['original', 'stopped', 'bounded_constant', 'bounded_fixed'], 'count': 8},
        'decision': 'Reject the fixed observed-brightness gate as an unchanged next training treatment: reduced normal brightness error costs dark visibility and does not resolve facial-detail failures.',
        'constant_bound_status': 'Retained as measured intervention evidence; its own dark costs preclude automatic training or model acceptance.',
        'full61_clear_53_blur_DEV_guards_evaluated': False,
        'native_CCTV_or_FINAL_images_evaluated': False, 'user_acceptance_established': False,
        'new_trained_model': False, 'model_qualified': False, 'training_authorized': False,
        'local_neural_gradient_optimizer_VM_calls': 0}
    with (REVIEW / 'visual_review_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete': True, 'sheets': 18, 'cases': 32, 'exact_cells': cells,
                      'fixed_gate_advanced': False, 'model_qualified': False}))


if __name__ == '__main__':
    main()

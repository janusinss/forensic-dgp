"""Independently inspect saved UI requests, downloaded pixels and raw bundle floats."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_mask_review_race_fix_v1'
FLOW = ROOT / 'scratch/mask-import-race-v1/model-flow'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def pixels(path):
    with Image.open(path) as image: return np.asarray(image.convert('RGB')).copy()


def main():
    plan = read(OUT / 'real-model-flow-plan.json'); results = read(FLOW / 'results.json')
    assert plan['frozen_before_real_model_flow'] and not plan['actual_training'] and not plan['VM_actions']
    assert results['complete'] and results['actual_models'] and results['plan_sha256'] == sha(OUT / 'real-model-flow-plan.json')
    assert results['generation_requests'] == results['mask_requests'] == 11 and len(results['records']) == 11
    assert results['seconds'] < 240 and not results['errors'] and len(results['rejections']) == 2
    assert [r['width'] for r in results['layout']] == [375, 768, 1280] and not any(r['overflow'] for r in results['layout'])
    for name, digest in plan['source_sha256'].items(): assert sha(ROOT / name) == digest, name
    original = ROOT / 'outputs/dgp_mask_review_race_fix_v1/before/static/face_workflow.js'
    assert sha(original) == '81edfc86fcd16d976f1ed98d146680b38ada9af7504f6bdc4d60a5471cb249b0'
    assert results['source_sha256'] == sha(ROOT / 'static/face_workflow.js')
    reproduced = read(ROOT / 'scratch/mask-import-race-v1/reproduction.json')
    regression = read(ROOT / 'scratch/mask-import-race-v1/regressions.json')
    assert reproduced['bug_reproduced'] and reproduced['submitted_requests_before_new_mask_review'] == 1
    assert reproduced['stale_result_visible_after_unreviewed_import'] and not reproduced['review_checked_after_import']
    assert reproduced['source_sha256'] == sha(original)
    assert regression['complete'] and regression['regressions_passed'] == 3 and not regression['errors']
    assert regression['source_sha256'] == sha(ROOT / 'static/face_workflow.js')
    assert [r['case'] for r in regression['records']] == ['slow_import', 'invalid_import', 'superseded_import']
    assert regression['records'][0]['pending_generation_blocked'] and regression['records'][0]['fresh_confirmation_required']
    assert regression['records'][1]['old_approval_invalidated'] and regression['records'][2]['new_detection_stays_locked']
    before_mask = pixels(ROOT / 'scratch/mask-import-race-v1/submitted-before-import.png')
    after_mask = pixels(ROOT / 'scratch/mask-import-race-v1/submitted-after-import.png')
    requested_mask = pixels(ROOT / 'outputs/broad_covering_gallery_v1/proposals/val_18_hand_eyes.png')
    assert not before_mask.any() and np.array_equal(after_mask, requested_mask)
    prior = read(ROOT / 'outputs/dgp_app_covering_review_v3/results.json')
    cases = {r['id']: r for r in plan['cases']}; protected_visible_bytes = 0; families = set(); raw_checks = 0
    for row in results['records']:
        case = cases[row['id']]; families.add(case['family'])
        png = ROOT / row['png_file']; meta = ROOT / row['metadata_file']
        assert sha(png) == row['png_sha256'] and sha(meta) == row['metadata_sha256']
        previous = next(r for r in prior['rows'] if r['id'] == row['id'])['assisted'][row['mode']]
        assert sha(png) == sha(ROOT / 'outputs/dgp_app_covering_review_v3' / previous['output'])
        image = pixels(png); source = pixels(ROOT / case['input']); mask = pixels(ROOT / case['reviewed'])[..., 0] > 127
        metadata = read(meta); assert image.shape == source.shape == (256, 256, 3)
        assert metadata['input_review']['decision'] == 'usable' and metadata['additional_mask_expansion'] == 0
        assert metadata['restoration_requested'] == row['mode'] and metadata['restoration_applied'] == row['restoration_applied']
        assert metadata['original_rgb_sha256'] == hashlib.sha256(source.tobytes()).hexdigest()
        assert metadata['native_mask_sha256'] == hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        assert metadata['output_rgb_sha256'] == hashlib.sha256(image.tobytes()).hexdigest()
        if not metadata['restoration_applied']:
            assert np.array_equal(image[~mask], source[~mask]); protected_visible_bytes += int((~mask).sum()) * 3
        else:
            assert metadata['visible_restoration']['weights_sha256'] == '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
        if row['bundle_file']:
            with zipfile.ZipFile(ROOT / row['bundle_file']) as archive:
                assert set(archive.namelist()) == {'original.png', 'removal-mask-original.png', 'input-256.png', 'removal-mask.png',
                                                  'estimate.png', 'dgp-raw-float32.npy', 'processing.json', 'README.txt'}
                for name, expected in [('original.png', source), ('input-256.png', source), ('estimate.png', image)]:
                    actual = np.asarray(Image.open(io.BytesIO(archive.read(name))).convert('RGB'))
                    assert np.array_equal(actual, expected)
                for name in ['removal-mask-original.png', 'removal-mask.png']:
                    actual = np.asarray(Image.open(io.BytesIO(archive.read(name))).convert('L'))
                    assert np.array_equal(actual, mask.astype(np.uint8) * 255)
                assert json.loads(archive.read('processing.json')) == metadata
                raw = np.load(io.BytesIO(archive.read('dgp-raw-float32.npy')), allow_pickle=False)
                assert raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
                saved = np.load(ROOT / 'outputs/dgp_app_covering_review_v3/stages' / (row['id'] + '_' + row['mode'] + '.npz'), allow_pickle=False)
                assert np.array_equal(raw, saved['dgp'][0].transpose(1, 2, 0))
                expected = np.floor(raw * np.float32(255)).astype(np.uint8)
                expected[mask] = np.floor(saved['completion'][0].transpose(1, 2, 0)[mask] * np.float32(255)).astype(np.uint8)
                assert not metadata['display_processing']['colour_policy']['applied'] and np.array_equal(expected, image)
                raw_checks += 1
    assert set(plan['families']).issubset(families) and raw_checks == 1
    assert all(r['generation_requests'] == 0 for r in results['rejections'])
    record = read(ROOT / 'outputs/dgp_app_v3_integration_record.json')
    for name, digest in {**record['sources_sha256'], **record['evidence_sha256']}.items():
        assert sha(original if name == 'static/face_workflow.js' else ROOT / name) == digest
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'flow_plan_sha256': sha(OUT / 'real-model-flow-plan.json'),
               'flow_results_sha256': sha(FLOW / 'results.json'), 'three_ordering_regressions_verified': True,
               'downloaded_PNGs_verified': 11, 'covering_families_functionally_verified': 7, 'controls_verified': 2,
               'rejected_inputs_submit_zero_generation': 2, 'visible_Off_bytes_verified': protected_visible_bytes,
               'raw_bundle_and_exact_composition_verified': True, 'original_app22_bindings_preserved_with_archived_JS': True,
               'original_JS_sha256': sha(original), 'current_JS_sha256': sha(ROOT / 'static/face_workflow.js'),
               'local_neural_calls_in_this_checker': 0, 'local_gradient_calls': 0, 'training': False, 'VM_actions': [],
               'native_CCTV_or_reserved_used': False, 'restoration_or_completion_quality_qualification': False, 'goal_complete': False}
    with (OUT / 'independent_saved_output_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()

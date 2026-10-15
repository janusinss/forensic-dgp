"""Freeze then record missing current-detector maps on exposed covering inputs.

Inference only, with unchanged app mask code/threshold. No completed faces,
restorer, gradients, optimizer, acquisition, mask editing or model selection.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/dgp_detector_confidence_v1_r1'
PARENT = ROOT / 'outputs/dgp_app_covering_review_v3'
CAP_SECONDS = 180


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def image(path, mode='RGB'):
    with Image.open(path) as source:
        assert source.size == (256, 256)
        return np.asarray(source.convert(mode)).copy()


def freeze():
    assert not OUT.exists(), 'Preserve all frozen diagnostics'
    readiness = read(ROOT / 'outputs/dgp_covering_readiness_20261012_v1/audit.json')
    plan, prior = read(PARENT / 'plan.json'), read(PARENT / 'results.json')
    assert readiness['complete'] and prior['complete']
    pins = readiness['source_and_output_sha256']
    assert sha(PARENT / 'plan.json') == pins['outputs/dgp_app_covering_review_v3/plan.json'] == prior['plan_sha256']
    assert sha(PARENT / 'results.json') == pins['outputs/dgp_app_covering_review_v3/results.json']
    assert len(plan['cases']) == 36 and sum(c['synthetically_degraded'] for c in plan['cases']) == 18
    assert sum(c['input_review'] == 'usable' for c in plan['cases']) == 32
    required = {'outputs/dgp_app_covering_review_v3/plan.json': sha(PARENT / 'plan.json'),
                'outputs/dgp_app_covering_review_v3/results.json': sha(PARENT / 'results.json'),
                'outputs/dgp_covering_readiness_20261012_v1/audit.json': sha(ROOT / 'outputs/dgp_covering_readiness_20261012_v1/audit.json')}
    for name in ['face_workflow.py', 'dgp_face_workflow_v3.py', 'completion.py', 'completion_data.py',
                 'completion_inference.py', 'dgp_face_restoration.py', 'dgp_frozen_inference_v2.py',
                 'cctv_dgp_frozen_norm.py', 'face_color_policy.py', 'cctv_input_quality.py',
                 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth',
                 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',
                 'outputs/downloaded_completion/outputs/completion_pilot/best_selection.json']:
        digest = sha(ROOT / name)
        if name in pins:
            assert digest == pins[name]
        required[name] = digest
    for case in plan['cases']:
        assert case['hidden_ground_truth'] is None
        for field in ['input', 'reviewed', 'protected']:
            if case[field]:
                digest = sha(ROOT / case[field])
                assert digest == plan['sources_sha256'][case[field]]
                required[case[field]] = digest
        name = 'outputs/dgp_app_covering_review_v3/images/' + case['id'] + '_automatic.png'
        assert sha(ROOT / name) == pins[name]
        required[name] = sha(ROOT / name)
        assert image(ROOT / case['input']).shape == (256, 256, 3)
    required['scripts/record_dgp_detector_confidence_v1_r1.py'] = sha(__file__)
    required['scripts/audit_dgp_detector_confidence_v1_r1.py'] = sha(ROOT / 'scripts/audit_dgp_detector_confidence_v1_r1.py')
    required['scripts/supervise_dgp_detector_confidence_v1_r1.py'] = sha(ROOT / 'scripts/supervise_dgp_detector_confidence_v1_r1.py')
    required['scripts/dgp_detector_process_memory_v1.py'] = sha(ROOT / 'scripts/dgp_detector_process_memory_v1.py')
    required['outputs/dgp_detector_process_memory_v1_check.json'] = sha(ROOT / 'outputs/dgp_detector_process_memory_v1_check.json')
    required['outputs/dgp_detector_confidence_v1_r1_checker.json'] = sha(ROOT / 'outputs/dgp_detector_confidence_v1_r1_checker.json')
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(
        format='current-dgp-detector-missing-confidence-diagnostic-v1', date='2026-10-14',
        frozen_before_new_inference=True, cases=plan['cases'], assets_sha256=required,
        budget=dict(detector_forward_images=36, completion_forward_images=0, restorer_forward_images=0,
                    neural_gradient_queries=0, optimizer_updates=0, seconds=180,
                    output_byte_cap=32 * 1024**2, RSS_byte_cap=4 * 1024**3, threads=4, device='cpu'),
        threshold=.5, native_margin_pixels=3, crop_geometry=[256, 256],
        questions=['Are empty proposals already low-score before threshold and margin?',
                   'How do scores change between each original photograph and its synthetic degradation?',
                   'Do clear controls and near-hidden exclusions reveal opposite detector errors?'],
        non_goals=['No threshold sweep, threshold fitting or new masks', 'No face generation or app change',
                   'No trained checkpoint selection or covering qualification', 'No native CCTV or reserved-final pixels'],
        operator_masks_are_approximate_removal_footprints_not_segmentation_truth=True,
        original_and_degraded_views_reuse_assistance_and_are_not_independent_people=True,
        exposed_sources_and_unknown_pretraining_overlap_retained=True,
        current_checkpoint_epoch_count_not_primary_DGP_epoch_count=True,
        automatic_generation_outputs=0, assisted_completion_outputs=0, family_qualified=False,
        actual_training_permitted=False, model_qualified=False, goal_complete=False))
    write(OUT / 'binding.json', dict(complete=True, protocol_sha256=sha(OUT / 'protocol.json'),
                                    preparation_only=True, detector_forwards=0, training_updates=0))
    print(dict(frozen=True, cases=36, confidence_maps_planned=36, completion_or_training=0), flush=True)


def model_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(array.dtype).encode() + b'\0')
        digest.update(str(tuple(array.shape)).encode() + b'\0' + array.numpy().tobytes())
    return digest.hexdigest()


def region_summary(probability, operator, protected):
    def values(mask):
        subset = probability[mask]
        return dict(pixels=int(subset.size), minimum=float(subset.min()) if subset.size else None,
                    median=float(np.median(subset)) if subset.size else None,
                    p95=float(np.quantile(subset, .95)) if subset.size else None,
                    maximum=float(subset.max()) if subset.size else None,
                    selected_fraction=float(np.mean(subset >= .5)) if subset.size else None)
    return dict(crop=values(np.ones_like(operator)), operator=values(operator),
                outside_operator=values(~operator), protected=values(protected))


def run():
    assert not (OUT / 'results.json').exists() and not (OUT / 'failure.json').exists()
    p = read(OUT / 'protocol.json')
    assert sha(OUT / 'protocol.json') == read(OUT / 'binding.json')['protocol_sha256']
    assert all(sha(ROOT / n) == h for n, h in p['assets_sha256'].items())
    started = time.monotonic()
    counts = dict(detector=0, completion=0, restorer=0, gradients=0, optimizer_updates=0)
    files = {}
    model = None
    try:
        from dgp_detector_process_memory_v1 import resident_bytes
        import torch
        from dgp_face_workflow_v3 import DGPFaceWorkflow
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        def check():
            assert time.monotonic() - started < p['budget']['seconds'], 'Local detector recording time stop'
            assert resident_bytes() <= p['budget']['RSS_byte_cap'], 'Local detector RSS stop'
            assert sum(v.stat().st_size for v in OUT.rglob('*') if v.is_file()) <= p['budget']['output_byte_cap'], 'Output cap stop'
            assert counts['detector'] <= 36 and counts['completion'] == counts['restorer'] == 0
        engine = DGPFaceWorkflow(device='cpu')
        assert engine.detector_path.resolve() == (ROOT / 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth').resolve()
        model = engine._detector()
        before = model_sha(model)
        state = engine.detector_state
        assert state['size'] == 256 and state['width'] == 24 and state['epoch'] == 2
        assert all(not m.training for m in model.modules()) and all(not v.requires_grad for v in model.parameters())
        captured = []
        def detector_hook(_module, _args, output):
            counts['detector'] += 1
            assert not output.requires_grad and output.shape == (1, 1, 256, 256)
            captured.append(output.detach().sigmoid()[0, 0].cpu().numpy().copy())
        def generator_hook(*_args):
            counts['completion'] += 1
            raise AssertionError('Completion generator must never run in detector recording')
        hooks = [model.segmenter.register_forward_hook(detector_hook), model.generator.register_forward_hook(generator_hook)]
        (OUT / 'probabilities').mkdir(); (OUT / 'masks').mkdir(); (OUT / 'sheets').mkdir()
        rows = []
        source_arrays = {}
        for case in p['cases']:
            check()
            rgb = image(ROOT / case['input'])
            supplied = image(ROOT / case['reviewed'], 'L') == 255
            protected = image(ROOT / case['protected'], 'L') == 255 if case['protected'] else np.zeros_like(supplied)
            captured.clear()
            with torch.inference_mode():
                proposed = engine.review_mask(rgb)
            assert len(captured) == 1 and proposed['threshold'] == .5 and proposed['proposal_margin_pixels'] == 3
            probability = captured[0]
            assert probability.dtype == np.float32 and probability.shape == (256, 256)
            assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
            assert np.array_equal(proposed['raw_mask'], probability >= .5)
            raw, mask = proposed['raw_mask'], proposed['mask']
            np.testing.assert_array_equal(mask, cv2.dilate(raw, np.ones((7, 7), np.uint8)))
            cached = image(PARENT / 'images' / (case['id'] + '_automatic.png'), 'L') == 255
            np.testing.assert_array_equal(mask.astype(bool), cached)
            npy = 'probabilities/' + case['id'] + '.npy'
            np.save(OUT / npy, probability, allow_pickle=False); files[npy] = sha(OUT / npy)
            for suffix, value in [('raw', raw), ('proposal', mask)]:
                name = 'masks/' + case['id'] + '_' + suffix + '.png'
                Image.fromarray(value.astype(np.uint8) * 255).save(OUT / name); files[name] = sha(OUT / name)
            union = mask.astype(bool) | supplied
            row = dict(id=case['id'], family=case['family'], condition=case['condition'], input_review=case['input_review'],
                probability=npy, raw='masks/' + case['id'] + '_raw.png', proposal='masks/' + case['id'] + '_proposal.png',
                raw_pixels=int(raw.sum()), proposal_pixels=int(mask.sum()), operator_pixels=int(supplied.sum()),
                missed_operator_pixels=int((supplied & ~mask.astype(bool)).sum()), extra_pixels=int((~supplied & mask.astype(bool)).sum()),
                iou_against_operator_footprint=float((supplied & mask.astype(bool)).sum() / union.sum()) if union.any() else 1.,
                scores=region_summary(probability, supplied, protected), exact_prior_proposal_reproduction=True,
                source_review_exclusion_retained=case['input_review'] != 'usable')
            rows.append(row); source_arrays[case['id']] = (rgb, probability, raw, mask, supplied)
            if len(rows) % 6 == 0:
                print(dict(recorded_confidence_maps=len(rows), of=36, completion_forwards=0, updates=0), flush=True)
        for hook in hooks:
            hook.remove()
        assert counts == dict(detector=36, completion=0, restorer=0, gradients=0, optimizer_updates=0)
        assert engine.generator is engine.restorer is None
        assert model_sha(model) == before and all(v.grad is None for v in model.parameters())
        for page in range(6):
            subset = rows[6 * page:6 * (page + 1)]
            sheet = Image.new('RGB', (1280, 6 * 292), '#171717'); draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(subset):
                rgb, probability, raw, mask, operator = source_arrays[row['id']]
                heat = cv2.cvtColor(cv2.applyColorMap(np.rint(probability * 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
                overlay = rgb.copy(); overlay[mask.astype(bool)] = (overlay[mask.astype(bool)].astype(np.float32) * .35 + np.array([255, 30, 30]) * .65).astype(np.uint8)
                columns = [('input', rgb), ('scores 0-1; not calibrated', heat), ('raw threshold .5', raw * 255),
                           ('app proposal margin3', overlay), ('operator footprint; approximate', operator.astype(np.uint8) * 255)]
                for col, (label, value) in enumerate(columns):
                    sheet.paste(Image.fromarray(value).convert('RGB'), (col * 256, index * 292 + 36))
                    draw.text((col * 256 + 3, index * 292 + 20), label, fill='white')
                draw.text((3, index * 292 + 3), row['id'] + ' / ' + row['input_review'], fill='white')
            name = f'sheets/page{page + 1:02d}.png'; sheet.save(OUT / name); files[name] = sha(OUT / name)
        check()
        assert all(sha(ROOT / n) == h for n, h in p['assets_sha256'].items())
        write(OUT / 'results.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(OUT / 'protocol.json'), counts=counts, rows=rows, artifacts_sha256=files,
            detector_checkpoint_sha256=p['assets_sha256'][str(engine.detector_path.relative_to(ROOT).as_posix())],
            detector_model_state_before=before, detector_model_state_after=model_sha(model),
            detector_checkpoint_metadata=dict(format=state['format'], size=state['size'], width=state['width'], epoch=state['epoch']),
            torch=torch.__version__, device='cpu', gradients_disabled=True, seconds=time.monotonic() - started,
            primary_DGP_and_separate_completion_backend_not_loaded=True,
            detector_container_includes_unused_generator_weights=True, generator_forward_calls=0,
            exact_all36_prior_app_proposals_reproduced=True,
            automatic_generation_outputs=0, assisted_completion_outputs=0, no_threshold_or_app_or_model_change=True,
            native_CCTV_or_reserved_final_pixels_used=False, automatic_or_assisted_family_qualified=False,
            visual_review_pending=True, independent_audit_pending=True, model_qualified=False, goal_complete=False))
        print(dict(complete=True, confidence_maps=36, sheets_to_review=6, counts=counts, seconds=time.monotonic() - started), flush=True)
    except BaseException as error:
        import traceback
        write(OUT / 'failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
              counts=counts, seconds=time.monotonic() - started, model_qualified=False, goal_complete=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else run()

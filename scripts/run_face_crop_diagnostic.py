"""Fixed training-only face-size diagnostic; frozen CPU inference, no optimizer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_crop_diagnostic import make_crop, warp_target, infer_crop_masks, require
from detector_training import load_manifest, ReviewedMasks
from scripts.audit_reflection_coverage_results import DATA_PATH, PIXELS_SHA, EXTRACTION, PREFIX, MODEL_SHA, AUDIT_OUTPUT, recount_supported, verify_package
from scripts.audit_face_occlusion_results import verified_inputs
from scripts.audit_face_occlusion_glare_pixels import added_region_counts
from scripts.evaluate_coverage_results import binary, recount
from scripts.package_face_occlusion_vm import sha

DETECTOR = Path('C:/Users/janus/.insightface/models/buffalo_l/det_10g.onnx')
DETECTOR_SHA = '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'
PROTOCOL = ROOT/'outputs/face_crop_protocol_v1/protocol.json'
OUTPUT = ROOT/'outputs/face_crop_diagnostic_v1'
TRAINING = ROOT/'dataset/detector_training_extension_v2/manifest.json'
FINAL = EXTRACTION/PREFIX/'reflective/epoch_42.pth'
SOURCE = EXTRACTION/PREFIX/'initial.pth'
CODE = ('face_crop_diagnostic.py', 'scripts/run_face_crop_diagnostic.py',
        'tests/test_face_crop_diagnostic.py', 'tests/test_face_crop_protocol.py', 'FACE_CROP_DIAGNOSTIC.md')


def training_members(rows):
    require(len(rows) == 105 and len({r['image_sha256'] for r in rows}) == 105 and
            Counter(r['split'] for r in rows) == {'train': 73, 'validation': 25, 'test': 7},
            'Fixed disjoint real membership/splits changed')
    selected = [r for r in rows if r['split'] == 'train']
    require(Counter(r['kind'] for r in selected) == {'covered': 47, 'uncovered': 26}, 'Real training strata changed')
    return selected


def diagnostic_signal(original, cropped):
    checks = {'lens_recovery_increases': cropped['lens_recovered_pixels'] > original['lens_recovered_pixels'],
              'real_training_iou_retained': cropped['training_real']['iou'] >= original['training_real']['iou'],
              'real_training_visible_fp_retained': cropped['training_real']['visible_false_positive'] <= original['training_real']['visible_false_positive'],
              'real_clear_errors_retained': cropped['training_real']['negative_false_positive_cases'] <= original['training_real']['negative_false_positive_cases'],
              'fixture_iou_retained': cropped['training_reflection']['iou'] >= original['training_reflection']['iou'],
              'fixture_clear_errors_retained': cropped['training_reflection']['negative_false_positive_cases'] <= original['training_reflection']['negative_false_positive_cases']}
    return {'checks': checks, 'training_signal_passes': all(checks.values()),
            'scope': 'Predeclared training diagnostic only; does not select or promote a model'}


def protocol_value():
    verify_package(); verified_inputs()
    audit_path = AUDIT_OUTPUT/'results.json'; repro_path = AUDIT_OUTPUT/'reproduction.json'
    review_path = ROOT/'outputs/reflection_coverage_analysis_v1/visual_review.json'
    audit, repro, review = (json.loads(p.read_text()) for p in (audit_path, repro_path, review_path))
    require(audit['checksum_verified'] is True and repro['complete'] is True and
            repro['audit_sha256'] == review['audit_sha256'] == sha(audit_path) and
            review['reproduction_sha256'] == sha(repro_path) and repro['selection_agrees_with_vm'] is True and
            repro['archive_sha256'] == audit['archive_sha256'] and
            audit['script_sha256'] == sha(ROOT/'scripts/audit_reflection_coverage_results.py') and
            repro['script_sha256'] == sha(ROOT/'scripts/reproduce_reflection_coverage_results.py'),
            'Completed return/reproduction/review bindings changed')
    require(sha(DETECTOR) == DETECTOR_SHA and sha(SOURCE) == MODEL_SHA and
            sha(FINAL) == audit['checkpoint_checks']['reflective/42']['sha256'], 'Frozen detector/model source changed')
    for key in ('source30', 'reflective/42'):
        require(all(repro['reproductions'][key]['mask_reproduction'][d]['all_exact'] is True for d in ('training_real',))
                and repro['fixture_reproductions'][key]['mask_reproduction']['all_exact'] is True,
                'Original training masks must exactly match completed CPU inference')
    rows = training_members(load_manifest(TRAINING))
    input_paths = (TRAINING, ROOT/DATA_PATH/'manifest.json', ROOT/DATA_PATH/'pixels.pth',
                   ROOT/'dataset/detector_glare_review_v3/manifest.json',
                   ROOT/'dataset/detector_expanded_review_v2/manifest.json', AUDIT_OUTPUT/'members.json')
    return {'format': 'dgp-face-crop-protocol-v1', 'date': '2026-10-02', 'device': 'cpu',
            'audit_sha256': sha(audit_path), 'reproduction_sha256': sha(repro_path), 'visual_review_sha256': sha(review_path),
            'source30_sha256': sha(SOURCE), 'reflective42_sha256': sha(FINAL), 'face_detector_sha256': DETECTOR_SHA,
            'inputs': {p.relative_to(ROOT).as_posix(): sha(p) for p in input_paths},
            'code': {p: sha(ROOT/p) for p in CODE},
            'training_image_sha256': [r['image_sha256'] for r in rows],
            'real_training_cases': 73, 'fixture_training_cases': 280, 'held_out_inference_cases': 0,
            'crop': {'face_confidence': .6, 'context_multiplier': 1.25, 'output_size': 256,
                     'minimum_face_dimension': 16, 'minimum_crop_side': 32, 'maximum_crop_side': 'strictly less than longest input dimension',
                     'selection': 'Exactly one finite reliable detected face; box center inside input; no target argument',
                     'rgb_interpolation': 'bilinear', 'mask_mapping': 'nearest; preserve original predictions outside crop ROI',
                     'padding_rgb': [96, 96, 96], 'fallback': 'unchanged original mask; no new forward'},
            'models': ['source30', 'reflective42'], 'advancement_model': 'reflective42 final budget; source30 is ancestry diagnosis',
            'training_signal': 'Strict lens recall gain, nondecreasing whole real and fixture IoU, nonincreasing real visible FP and real/fixture clear errors',
            'searches': 0, 'optimizer_updates_locally': 0, 'promoted': False}


def main(prepare=False):
    expected = protocol_value()
    if prepare:
        require(not PROTOCOL.parent.exists(), 'Preserve prepared protocol')
        PROTOCOL.parent.mkdir(); PROTOCOL.write_text(json.dumps(expected, indent=2)+'\n')
        print(json.dumps({'protocol_sha256': sha(PROTOCOL), 'training_cases': 353, 'held_out_inference_cases': 0,
                          'optimizer_updates_locally': 0, 'executed': False}), flush=True); return
    require(PROTOCOL.is_file() and json.loads(PROTOCOL.read_text()) == expected, 'Prepared protocol/code/data changed')
    require(not OUTPUT.exists(), 'Preserve existing diagnostic evidence')
    sys.path.insert(0, str(ROOT/'outputs/face_extraction_dependencies'))
    from face_occlusion_adapter import load_adapter
    from scripts.train_face_occlusion_vm import dependency_versions
    from insightface.model_zoo import get_model
    dependencies = dependency_versions(ROOT/'outputs/face_extraction_dependencies')
    torch.set_num_threads(4)
    # All dependency/model construction preflight precedes creation of output.
    source_model, source_payload = load_adapter(SOURCE, 'cpu')
    final_model, final_payload = load_adapter(FINAL, 'cpu')
    detector = get_model(str(DETECTOR), providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1, input_size=(256, 256), det_thresh=.6)
    require(detector.session.get_providers() == ['CPUExecutionProvider'], 'Face boxes must use CPU only')
    rows = training_members(load_manifest(TRAINING)); real = ReviewedMasks(rows, 256)
    cache_path = ROOT/DATA_PATH/'pixels.pth'; require(sha(cache_path) == PIXELS_SHA, 'Frozen fixture bytes changed')
    pixels = torch.load(cache_path, map_location='cpu', weights_only=True)['pixels']
    require(set(pixels) == set(range(280)), 'Fixed fixture membership differs')
    members = json.loads((AUDIT_OUTPUT/'members.json').read_text()); returned = EXTRACTION/PREFIX
    old_path = ROOT/'dataset/detector_expanded_review_v2/manifest.json'
    v3 = json.loads((ROOT/'dataset/detector_glare_review_v3/manifest.json').read_text())
    require(sha(old_path) == v3['parent_manifest_sha256'], 'Real reflection label lineage differs')
    old = {r['image_sha256']: r for r in load_manifest(old_path)}
    cases = []; targets = {}; valid = {}; lens_parents = {}; metadata = []
    for domain, count in (('training_real', 73), ('training_reflection', 280)):
        for i in range(count):
            name = f'{domain}/{i:04}'
            if domain == 'training_real':
                x, t = real[i]; rgb = np.rint(x.permute(1, 2, 0).numpy()*255).astype('uint8')
                target = t[0].numpy().astype(bool); support = np.ones(target.shape, bool)
                if rows[i].get('glare_stratum') == 'strong_lens_reflection':
                    lens_parents[name] = ReviewedMasks([old[rows[i]['image_sha256']]], 256)[0][1][0].numpy().astype(bool)
            else:
                case = pixels[i]; rgb = case['input'].permute(1, 2, 0).numpy()
                target = case['mask'][0].numpy().astype(bool); support = case['valid'][0].numpy().astype(bool)
            boxes, _ = detector.detect(rgb[:, :, ::-1].copy(), max_num=0)
            transform = make_crop(rgb, boxes); mapped_target = warp_target(target, transform)
            cases.append({'id': name, 'rgb': rgb, 'transform': transform})
            targets[name], valid[name] = target, support
            metadata.append({'id': name, 'input_byte_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                             **{k: transform[k] for k in ('cropped', 'reason', 'source_shape', 'bounds', 'face_confidence')},
                             'affine': transform['affine'].tolist(), 'inverse': transform['inverse'].tolist(),
                             'roi_pixels': int(transform['roi'].sum()), 'target_pixels': int(target.sum()),
                             'target_pixels_outside_roi': int((target & ~transform['roi']).sum()),
                             'cropped_target_pixels': int(mapped_target.sum())})
            if len(cases) % 50 == 0: print('Training input face boxes', len(cases), 353, flush=True)
    require(len(lens_parents) == 2 and len(cases) == 353, 'Fixed training reflection/cohort membership differs')
    OUTPUT.mkdir(); (OUTPUT/'cases.json').write_text(json.dumps(metadata, indent=2)+'\n')
    report = {'protocol_sha256': sha(PROTOCOL), 'code_sha256': sha(__file__), 'device': 'cpu', 'torch': str(torch.__version__),
              'dependencies': dependencies, 'models': {}, 'held_out_inference_cases': 0,
              'crop_counts': dict(Counter(r['reason'] for r in metadata)), 'complete': False,
              'optimizer_constructed': False, 'optimizer_updates_locally': 0, 'promoted': False}
    ordinary = {}; composed = {}
    def scores(predictions):
        main_scores = recount([predictions[f'training_real/{i:04}'] for i in range(73)],
                              [targets[f'training_real/{i:04}'] for i in range(73)])
        fixture = recount_supported({i: predictions[f'training_reflection/{i:04}'] for i in range(280)}, pixels)
        lens = {name: added_region_counts(predictions[name], targets[name], parent) for name, parent in lens_parents.items()}
        return {'training_real': main_scores, 'training_reflection': fixture, 'lens_only': lens,
                'lens_recovered_pixels': sum(r['recovered_added_glare_pixels'] for r in lens.values()),
                'lens_target_pixels': sum(r['added_glare_pixels'] for r in lens.values())}
    for label, model, folder in (('source30', source_model, returned/'initial_masks'),
                                 ('reflective42', final_model, returned/'reflective/epoch_42_masks')):
        baseline = {}; items = []
        for case in cases:
            path = folder/(case['id']+'.png')
            require(sha(path) == members[path.relative_to(EXTRACTION).as_posix()], 'Audited original training mask changed')
            baseline[case['id']] = binary(path)
            items.append({'id': case['id'], 'original': baseline[case['id']], 'transform': case['transform']})
        print(label, 'fixed crop inference', flush=True)
        prediction, invariant = infer_crop_masks(model, items)
        ordinary[label], composed[label] = baseline, prediction
        report['models'][label] = {'ordinary': scores(baseline), 'face_crop': scores(prediction), **invariant}
        for name, mask in prediction.items():
            path = OUTPUT/label/(name+'.png'); path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype('uint8')*255).save(path)
        (OUTPUT/'partial.json').write_text(json.dumps(report, indent=2)+'\n')
    report['training_signal'] = diagnostic_signal(report['models']['reflective42']['ordinary'], report['models']['reflective42']['face_crop'])
    fixed = [f'training_real/{i:04}' for i, row in enumerate(rows) if row.get('glare_stratum') == 'strong_lens_reflection']
    fixed += [f'training_real/{i:04}' for i, row in enumerate(rows) if row['kind'] == 'uncovered'][:4]
    by_id = {r['id']: r for r in cases}
    sheet = Image.new('RGB', (1024, len(fixed)*148), 'white')
    for line, name in enumerate(fixed):
        case = by_id[name]; transform = case['transform']
        tiles = [Image.fromarray(case['rgb']), Image.fromarray(transform['rgb']),
                 Image.fromarray(targets[name].astype('uint8')*255).convert('RGB'),
                 Image.fromarray(warp_target(targets[name], transform).astype('uint8')*255).convert('RGB')]
        tiles.extend(Image.fromarray(p[name].astype('uint8')*255).convert('RGB') for p in
                     (ordinary['source30'], composed['source30'], ordinary['reflective42'], composed['reflective42']))
        ImageDraw.Draw(sheet).text((2, line*148+2), name+' input | crop | target | crop target | source | crop source | final | crop final', fill='black')
        for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128)), (j*128, line*148+20))
    sheet.save(OUTPUT/'preview.png')
    report.update(complete=True, preview_ids=fixed, limitations=[
        'Training-only diagnostic; no independent validation or completion improvement claim.',
        'Input-only face detector can miss covered faces; fallback counts are reported.',
        'Crop interpolation adds no native detail; mask remapping can change pixel boundaries.'])
    (OUTPUT/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'complete': True, 'crop_counts': report['crop_counts'], 'training_signal': report['training_signal'],
                      'held_out_inference_cases': 0, 'optimizer_updates_locally': 0, 'promoted': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); main(args.prepare)

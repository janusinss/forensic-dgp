"""Review saved derivatives, parameter bytes and image arrays; no model calls."""
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
import torch

from import_cctv_dgp_joint_raw_png_forward_v1_return import (
    ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write)


OBSERVATIONS = {
    'b036': 'Clear glasses, lips and fine hair remain softened. All degraded proposals leave the eye area indistinct; the largest step changes nose/mouth contours without convincing whole-face clarity.',
    'b051': 'Clear eyelids, skin texture and white hair are smoother than the input. Degraded eyes and mouth remain blurred; the largest step adds slight central contours rather than useful facial detail.',
    'b076': 'Hair strands and eye/lip borders remain soft. The largest step adds darker nose/lip outlines in the compound row; the two smallest steps have no convincing visible advantage.',
    'b085': 'Clear eye borders and individual teeth are softer than the input. All degraded faces remain blurred; the largest step accentuates mouth/nose outlines without a convincing whole-face gain.',
    'b092': 'Smile, teeth, eyelids and bangs remain soft in degraded outputs. The largest step changes central nose/mouth contours; the smaller proposals remain visually close to the retained result.',
    'b146': 'Clear iris edges and teeth are smoothed. The degraded proposals do not resolve eyelids or teeth; the largest step introduces extra central lines without convincing useful detail.',
    'b169': 'The visible eyes, smile and graduation clothing remain recognizable but soft. The largest compound result adds a lip outline while leaving the face indistinct; no proposal shows convincing whole-face improvement.',
    'b357': 'Ordinary hair remains present, with softened eye/lip borders. The largest compound result has extra nose/lip outlines; all degraded proposals remain soft.',
    'b384': 'Clear beard texture and eyebrows are smoothed. Larger steps add central nose/lip outlines in degraded rows; small steps remain visually similar to the retained output without convincing clarity.',
    'b386': 'Clear glasses and teeth remain present but smooth. The degraded eye/eyewear area stays indistinct; the largest step adds central outlines without resolving the face.',
    'b391': 'Moustache and eye/mouth edges remain soft. The largest step introduces an extra curved nose contour in blur, low-light and compound rows; this is an artifact rather than convincing facial structure.',
    'b392': 'The open mouth and eyelid shapes remain blurred in degraded rows. The largest step changes dark central contours, while the two smallest proposals have no convincing visible benefit.',
    'b393': 'Cheek, nose and mouth remain smooth. The largest step introduces doubled curved nose/eye-area contours in degraded rows; the small steps do not resolve useful details.',
    'b394': 'Visible gaze and open mouth remain recognizable rough shapes. Extra nose/mouth curves appear at the largest step, especially in low-light and compound outputs; no convincing whole-face clarity gain.',
    'b395': 'The hand beside the face remains present; this is not a completion experiment. The largest step changes central nose contours while eyes and mouth remain soft.',
    'b480': 'The smile and teeth remain soft. The largest step adds curved central contours in low-light/compound rows; smaller steps leave the degraded eyes and mouth indistinct.',
    'b615': 'Side padding and the forehead hand remain present. Teeth, eyelids and ordinary hair remain soft; the largest step adds eye/nose/lip curves without convincing useful clarity. No covering completion is tested.',
    'b651': 'Side padding and ordinary hair remain present. The largest step adds eye/nose/lip outlines in degraded rows; the two small steps remain close to the blurred retained result.',
    'b701': 'Side padding and the visible head covering remain present. Degraded eyes and lips are soft; the largest step adds central outlines without a convincing whole-face improvement. No covering removal is tested.',
    'b745': 'Side padding remains fixed. Degraded smile/teeth and eye areas stay soft; the largest step adds central contours while the small proposals show no convincing clarity gain.'}


def pixels(path, mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256, 256)
        return np.asarray(image.convert(mode)).copy()


def high(a):
    z = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-.5 * np.square(z / 2)); kernel /= kernel.sum()
    luma = (a.astype(np.float64) * np.array([.299, .587, .114])).sum(2)
    return luma - convolve1d(convolve1d(luma, kernel, axis=0, mode='reflect'), kernel,
                            axis=1, mode='reflect')


def loss(row, finite):
    group = row['source'] + '/' + row['profile']
    metrics = finite[group]
    if row['metric'] == 'MSE': return metrics['MSE']
    if row['metric'] == 'SSIM_loss': return 1 - metrics['SSIM']
    if row['metric'] == 'ArcFace_loss': return 1 - metrics['ArcFace_observed_fixed']
    assert row['metric'] == 'feature_MSE'
    return metrics['landmark_high_frequency_MSE']


def main():
    started = time.monotonic()
    target = ANALYSIS / 'finite_learning_review.json'
    assert not target.exists() and not (ANALYSIS / 'visual_review.json').exists()
    audit = read(ANALYSIS / 'independent_audit.json')
    finite = read(ANALYSIS / 'finite_metrics.json')
    p = read(RETURN / 'protocol.json'); old = read(PARENT / 'protocol.json')
    joint = OUT / 'cctv_dgp_png_group_gradients_v1_r2_analysis/joint_raw_png_direction_v1_r2'
    predicted = read(joint / 'direction_review.json')
    assert all(r['complete'] for r in [audit, finite, predicted, read(joint / 'independent_audit.json')])
    assert sha(RETURN / 'protocol.json') == PIN == audit['protocol_sha256']
    assert sha(ANALYSIS / 'finite_metrics.json') == audit['finite_metrics_sha256']
    assert sha(RETURN / 'directions.npz') == predicted['direction_file_sha256']
    assert len(predicted['float32_group_predictions']) == 64
    torch.set_grad_enabled(False)
    initial_path = PARENT / 'initializers/initial_B.pth'
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    assert sha(initial_path) == predicted['initial_file_sha256']
    names = p['learning_names']
    assert len(names) == 37
    sizes = [initial[name].numel() for name in names]
    edges = np.cumsum([0] + sizes)
    partitions = {label: [i for i, name in enumerate(names) if name.startswith(prefix)]
                  for label, prefix in [('current_decoder', 'current.'),
                    ('style_conditioner', 'style_conditioner.'),
                    ('spatial_conditioners', 'spatial_conditioners.'), ('bank_readouts', 'fusions.')]}
    assert sum(map(len, partitions.values())) == 37
    with np.load(RETURN / 'directions.npz', allow_pickle=False) as d:
        direction = [d[f'balanced_p{i:02d}'] for i in range(37)]
    steps = {}
    changes = {}
    for proposal in p['proposals']:
        label, rate = proposal['label'], np.float32(proposal['rate'])
        receipt = read(RETURN / 'outputs/joint_raw_png_forward_v1' / label / 'parameter_receipt.json')
        # Reconstruct reported bytes in plain arrays, not a model or trainable state.
        steps[label] = np.concatenate([((initial[name].numpy() + rate * direction[i]).astype(np.float64)
                                       - initial[name].numpy().astype(np.float64)).ravel()
                                       for i, name in enumerate(names)])
        changes[label] = {part: sum(receipt['selected_parameter_change_statistics'][names[i]]['changed_values']
                                   for i in indices) for part, indices in partitions.items()}
        assert changes[label]['style_conditioner'] == changes[label]['spatial_conditioners'] == 0
    partition_rows = {}
    for part, indices in partitions.items():
        partition_rows[part] = dict(tensors=len(indices), values=sum(sizes[i] for i in indices),
            names=[names[i] for i in indices],
            initial_nonzero_values=sum(int(np.count_nonzero(initial[names[i]].numpy())) for i in indices),
            direction_nonzero_values=sum(int(np.count_nonzero(direction[i])) for i in indices),
            changed_values={label: values[part] for label, values in changes.items()})
    assert partition_rows['spatial_conditioners']['initial_nonzero_values'] == 0
    assert partition_rows['bank_readouts']['initial_nonzero_values'] == 0
    groups = []
    assert [r['key'] for r in predicted['aggregate_gradient_bindings']] == [r['key'] for r in predicted['float32_group_predictions']]
    for binding, prediction in zip(predicted['aggregate_gradient_bindings'], predicted['float32_group_predictions'], strict=True):
        path = ROOT / binding['file']
        assert sha(path) == binding['sha256']
        g = np.load(path, allow_pickle=False, mmap_mode='r')
        assert g.dtype == np.float64 and g.shape == (sum(sizes),)
        conditioning_nonzero = {part: sum(int(np.count_nonzero(g[edges[i]:edges[i+1]])) for i in partitions[part])
                                for part in ['style_conditioner', 'spatial_conditioners']}
        assert all(value == 0 for value in conditioning_nonzero.values())
        stage = prediction['metric_stage'].lower()
        before = loss(prediction, finite['baseline_groups'][stage])
        # Saved differentiable SSIM uses float64 moments; scientific skimage
        # evaluates float32 RGB. Recognizer replay reductions also differ.
        # Inherit the already-audited derivative alignment bound ONLY for
        # those representations. Exact array/PNG and acceptance gates stay.
        derivative_alignment_bound = 1e-6 if prediction['metric']=='ArcFace_loss' or (
            stage=='raw' and prediction['metric']=='SSIM_loss') else 1e-13
        assert math.isclose(before, prediction['loss'], rel_tol=0, abs_tol=derivative_alignment_bound)
        results = {}
        for proposal in p['proposals']:
            label = proposal['label']
            actual = loss(prediction, finite['proposals'][label]['groups'][stage])
            linear = math.fsum(float(np.dot(g[i:i+65536], steps[label][i:i+65536]))
                               for i in range(0, len(g), 65536))
            results[label] = dict(actual_loss=actual, measured_loss_change=actual-before,
                actual_parameter_bytes_first_order_change=linear,
                remainder_vs_saved_derivative=actual-before-linear,
                predicted_improvement_percent=-100*linear/before,
                measured_improvement_percent=100*(1-actual/before),
                locally_predicted_descent=linear<0, actual_loss_worsened=actual>before)
        groups.append(dict(key=prediction['key'], metric=prediction['metric'],
            metric_stage=prediction['metric_stage'], source=prediction['source'], profile=prediction['profile'],
            baseline_loss=before, saved_derivative_forward_loss=prediction['loss'],
            derivative_baseline_representation_difference=prediction['loss']-before,
            derivative_alignment_bound=derivative_alignment_bound,
            saved_gradient_sha256=binding['sha256'],
            conditioning_gradient_nonzero_values=conditioning_nonzero, steps=results))
    write(target, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        reviewer_sha256=sha(__file__), independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        finite_metrics_sha256=sha(ANALYSIS/'finite_metrics.json'), joint_direction_review_sha256=sha(joint/'direction_review.json'),
        partitions=partition_rows, groups=groups, group_actual_step_dots_checked=256,
        all64_saved_gradients_have_zero_style_and_spatial_conditioning_partitions=True,
        all_four_proposals_leave_all12_conditioning_tensors_unchanged=True,
        caveat='This limits these reset one-step experiments. Earlier audited B50 training changed all37 tensors; it does not prove conditioning can never learn or explain all earlier failures.',
        current_encoder_is_frozen_original_FPN=True, landmark_affines_used_for_metric_alignment_not_bank_geometry=True,
        alignment_error_not_proven_as_unique_cause=True, no_new_model_or_gradient_or_optimizer_calls=True,
        retained_strict_preparation_stop_sha256='2d3aba22e1811e5456486ea0c51d73fbbd97e13f5970cc6062630c9fb6c2e7c8',
        scientific_array_PNG_and_acceptance_checks_unchanged=True,
        more_epochs_qualified=False, model_qualified=False, seconds=time.monotonic()-started))
    refs = {r['id']: r for r in old['references']}
    region_names = ['eye_left_image', 'eye_right_image', 'nose', 'mouth_left_image', 'mouth_right_image']
    region_rows, quantization = [], []
    from importlib import import_module
    import sys
    sys.path.insert(0, str(PARENT))
    unpack = import_module('cctv_dgp_head4_lossless_v1').unpack
    run = read(RETURN/'outputs/joint_raw_png_forward_v1/run_results.json')
    for control in run['controls']:
        begin = control['begin']; cs = old['cases'][begin:begin+5]
        baseline, _ = unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz', [c['id'] for c in cs])
        raw = {prop['label']: np.load(RETURN/f"outputs/joint_raw_png_forward_v1/{prop['label']}/b{begin//5:03d}.npy", allow_pickle=False)
               for prop in p['proposals']}
        for i, case in enumerate(cs):
            ref = refs[case['source_person_or_reference']]
            target8, camera = pixels(PARENT/ref['target']), pixels(PARENT/case['input'])
            observed = pixels(PARENT/ref['observed'], 'L') > 0
            interior = cv2.erode(observed.astype(np.uint8), np.ones((13,13),np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)>0
            target_hp = high(target8.astype(np.float64)/255)
            base8 = pixels(PARENT/'outputs/bank_comparison_v1/baseline/previews'/(case['id']+'.png'))
            base_hp = dict(raw=high(baseline[i]), png=high(base8.astype(np.float64)/255))
            regions = []
            for name, (x,y) in zip(region_names, np.floor(case['landmarks5_canvas_xy']).astype(int), strict=True):
                region = np.zeros((256,256),bool)
                region[max(0,y-12):min(256,y+12),max(0,x-12):min(256,x+12)] = True
                region &= interior; assert region.any()
                regions.append((name,region))
            for prop in p['proposals']:
                label = prop['label']; arr = raw[label][i]
                png8 = pixels(RETURN/'outputs/joint_raw_png_forward_v1'/label/'png'/(case['id']+'.png'))
                assert np.array_equal(png8[~observed], camera[~observed])
                quantization.append(dict(proposal=label,id=case['id'],source=case['source'],profile=case['profile'],
                    observed_RGB_values=int(observed.sum()*3), raw_changed_RGB_values=int(np.count_nonzero(arr[observed]!=baseline[i][observed])),
                    PNG_changed_RGB_values=int(np.count_nonzero(png8[observed]!=base8[observed])),
                    PNG_maximum_absolute_byte_change=int(np.max(np.abs(png8[observed].astype(int)-base8[observed].astype(int))))))
                for stage, hp in [('raw',high(arr)), ('png',high(png8.astype(np.float64)/255))]:
                    for name, region in regions:
                        before = float(np.square(base_hp[stage]-target_hp)[region].mean())
                        after = float(np.square(hp-target_hp)[region].mean())
                        region_rows.append(dict(proposal=label,id=case['id'],source=case['source'],profile=case['profile'],
                            stage=stage,region=name,pixels=int(region.sum()),baseline_MSE=before,candidate_MSE=after))
    assert len(region_rows)==4000 and len(quantization)==400
    region_groups = {}
    for label in [prop['label'] for prop in p['proposals']]:
        region_groups[label] = {}
        for stage in ['raw','png']:
            region_groups[label][stage] = {}
            for source in ['all'] + sorted({c['source'] for c in cs} | {r['source'] for r in finite['baseline_rows']}):
                region_groups[label][stage][source] = {}
                for name in region_names:
                    rows = [r for r in region_rows if r['proposal']==label and r['stage']==stage and r['region']==name
                            and r['profile']!='clear' and (source=='all' or r['source']==source)]
                    before = float(np.mean([r['baseline_MSE'] for r in rows]))
                    after = float(np.mean([r['candidate_MSE'] for r in rows]))
                    region_groups[label][stage][source][name] = dict(cases=len(rows),baseline_MSE=before,candidate_MSE=after,
                        gain_percent=100*(1-after/before),individual_regressions=sum(r['candidate_MSE']>r['baseline_MSE'] for r in rows))
    write(ANALYSIS/'spatial_patch_review.json', dict(complete=True, reviewer_sha256=sha(__file__),
        independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'), rows=region_rows, degraded_region_groups=region_groups,
        additive_diagnostic_only=True, original_union_patch_structure_gate_unchanged=True,
        regions_are_existing_five_landmark_patches_not_all_visible_features=True,
        overlapping_patches_not_independent_regions=True, no_new_neural_calls=True, model_qualified=False))
    write(ANALYSIS/'quantization_review.json', dict(complete=True, independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        rows=quantization, cases=400, discontinuous_byte_delivery_and_saved_surrogate_not_equal_derivatives=True,
        no_display_conversion_or_quality_gate_change=True, model_qualified=False))
    sheets = audit['comparison_sheets']
    assert {Path(row['file']).stem for row in sheets} == set(OBSERVATIONS) and len(sheets)==20
    visual=[]
    for row in sheets:
        path=ROOT/row['file']; assert sha(path)==row['sha256']
        with Image.open(path) as image: assert image.size==(2048,1460)
        visual.append(dict(file=row['file'],sha256=row['sha256'],all40_cells_inspected=True,
            requested_display_detail='original',source_dimensions=[2048,1460],
            observations=OBSERVATIONS[path.stem],convincing_useful_gain_over_retained_DGP=False))
    write(ANALYSIS/'visual_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        reviewer='Primary assistant direct development visual review; not independent final human review',
        record_script_sha256=sha(__file__), raw_and_PNG_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        all20_sheets_and800_cells_inspected=True,sheets=visual,cells_inspected=800,
        inputs=100,unique_TRAIN_references=20,proposals=4,no_convincing_whole_face_detail_gain_observed=True,
        native_CCTV_pixels_reviewed=0,final_identity_pixels_reviewed=0,completion_families_tested=0,
        no_new_neural_calls_gradients_updates_or_VM_connections=True,model_adoption=False,independent_final_review=False,goal_complete=False))
    print(dict(complete=True,groups=64,actual_parameter_step_dots=256,spatial_region_rows=4000,
        quantization_cases=400,visual_sheets=20,neural_calls=0,optimizer_updates=0,seconds=time.monotonic()-started))


if __name__ == '__main__': main()

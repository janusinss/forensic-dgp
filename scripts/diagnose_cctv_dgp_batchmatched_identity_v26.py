"""Read-only saved V26 stop diagnosis and exact-pixel TRAINING review sheets."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_batchmatched_identity_v26 as a


def diagnose():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from torch.nn import functional as F
    started = time.monotonic(); torch.set_num_threads(4)
    bundle = a.BUNDLE
    returned = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
    out = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic'
    out.mkdir()
    plan = {'date': '2026-10-06', 'scope': 'The50 exposed TRAINING cases only; no new training or native/reserved output',
        'hypothesis': 'Measure whether the saved own-feature spatial path reaches structure and delivered pixels. Trace the exact stopped50 layers/250 input-conditioned features and paired TRAINING errors; no training, ablation or unchanged retry.',
        'protocol_sha256': a.PIN, 'independent_audit_sha256': a.sha(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json'),
        'fixed_operations': 'Exactly replay50 saved update50 class layers (no branch ablations), record spatial RMS, fixed per-tensor changes and raw/PNG paired TRAINING errors. Create ten four-column1:1 sheets.',
        'forward_cap': 50, 'seconds_cap': 300, 'CPU_raw_replay_tolerance': a.HEAD_RAW_TOLERANCE,
        'optimizer_updates': 0, 'backward_calls': 0, 'app_promotion': False, 'goal_complete': False}
    a.write(out / 'plan.json', plan)
    p = a.verify_bundle(bundle)
    independent = a.read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json')
    a.require(independent['complete'] and independent['complete_snapshot_updates'] == [0, 50] and
              independent['early_structure_stop']['pass'] is False, 'Exact audited early stop required')
    head = a.make_head(bundle)
    head.load_state_dict(torch.load(returned / 'outputs/update50/head.pth', map_location='cpu', weights_only=True), strict=True)
    before = a.state_hash(head.state_dict())
    initial = torch.load(returned / 'outputs/update0/head.pth', map_location='cpu', weights_only=True)
    parameter_changes = {}
    for name, value in head.state_dict().items():
        difference = value.double() - initial[name].double()
        parameter_changes[name] = {'changed_values': int((difference != 0).sum()),
            'maximum_absolute_change': float(difference.abs().max()),
            'RMS_change': float(difference.square().mean().sqrt())}
    fixed_kernel = np.exp(-.5 * (np.arange(-6, 7, dtype=np.float64) / 2)**2)
    fixed_kernel /= fixed_kernel.sum()
    def raw_feature_error(value, target, feature):
        import cv2
        delta = ((value.astype(np.float64) - target.astype(np.float64)/255) * np.array([.299,.587,.114])).sum(2)
        high = delta - cv2.sepFilter2D(delta, cv2.CV_64F, fixed_kernel, fixed_kernel, borderType=cv2.BORDER_REFLECT)
        return float(np.square(high)[feature].mean())

    source_bindings = {}
    rows = []
    def bind(path):
        source_bindings[path.relative_to(ROOT).as_posix()] = a.sha(path)
    for path in (bundle / 'protocol.json', bundle / 'cctv_dgp_spatial_features_v25.py',
                 returned / 'outputs/update0/head.pth', returned / 'outputs/update50/head.pth',
                 ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json'):
        bind(path)
    def spatial(value, support):
        field = value[0].detach().double().permute(1, 2, 0).numpy()
        resized = F.interpolate(support, size=value.shape[-2:], mode='nearest')
        observed = resized[0, 0].numpy() > 0
        selected = field[observed]
        return {'spatial_RMS': float(np.sqrt(np.square(selected - selected.mean(0)).mean())),
                'RMS': float(np.sqrt(np.square(selected).mean())), 'maximum_absolute': float(np.abs(selected).max()),
                'per_channel_mean': selected.mean(0).tolist()}
    for c in p['cases']:
        a.require(time.monotonic() - started < 300, 'Diagnostic CPU cap exceeded')
        paths = [bundle / c[name] for name in ('input', 'target', 'observed', 'raw_dgp', 'png_dgp')]
        paths += [returned / ('outputs/update' + str(update)) / (c['id'] + suffix)
                  for update in (0, 50) for suffix in ('.png', '.npy')]
        for path in paths:
            bind(path)
        camera, target = a.rgb(bundle / c['input']), a.rgb(bundle / c['target'])
        support, base = a.observed(bundle / c['observed']), a.raw_rgb(bundle / c['raw_dgp'])
        raw = a.raw_rgb(returned / 'outputs/update50' / (c['id'] + '.npy'))
        baseline = a.rgb(returned / 'outputs/update0' / (c['id'] + '.png'))
        prediction = a.rgb(returned / 'outputs/update50' / (c['id'] + '.png'))
        a.require(np.array_equal(prediction, a.png(raw, camera, support)), 'Delivered stopped PNG changed')
        x = torch.from_numpy(camera.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
        b = torch.from_numpy(base.copy()).permute(2, 0, 1)[None]
        mask = torch.from_numpy(support.astype(np.float32))[None, None]
        measurement = torch.from_numpy(a.interior(support, 6).astype(np.float32))[None, None]
        feature_arrays = []
        for index in range(5):
            path = returned / 'outputs/frozen_DGP_features' / (c['id'] + '_fpn' + str(index) + '.npy')
            bind(path)
            feature_arrays.append(torch.from_numpy(np.load(path, allow_pickle=False).copy())[None])
        stages = {}
        with torch.inference_mode():
            features = torch.cat((x, b, head.high(x), head.high(b), mask), dim=1)
            stages['conditioning'] = spatial(features, measurement)
            projected = [F.silu(layer(feature)) for layer,feature in zip(head.projections,feature_arrays)]
            for index,value in enumerate(projected): stages['projected'+str(index)] = spatial(value,measurement)
            up = lambda value,target: F.interpolate(value,size=target.shape[-2:],mode='bilinear',align_corners=False)
            value = F.silu(head.decode3(torch.cat((up(projected[4],projected[3]),projected[3]),1)))
            stages['decode3'] = spatial(value,measurement)
            value = F.silu(head.decode2(torch.cat((up(value,projected[2]),projected[2]),1))); stages['decode2'] = spatial(value,measurement)
            value = F.silu(head.decode1(torch.cat((up(value,projected[1]),projected[1]),1))); stages['decode1'] = spatial(value,measurement)
            value = F.silu(head.decode0(torch.cat((up(value,projected[0]),projected[0]),1))); stages['decode0'] = spatial(value,measurement)
            camera_feature = F.silu(head.camera(features)); stages['camera'] = spatial(camera_feature,measurement)
            full = F.silu(head.fuse(torch.cat((up(value,camera_feature),camera_feature),1))); stages['fused'] = spatial(full,measurement)
            direct = head.direct(features); stages['direct'] = spatial(direct,measurement)
            nonlinear = head.tail(full); stages['nonlinear'] = spatial(nonlinear,measurement)
            combined = direct + nonlinear; stages['combined_before_tanh'] = spatial(combined,measurement)
            q = .05 * torch.tanh(combined); stages['q_before_projection'] = spatial(q,measurement)
            mean = (q*mask).sum((2,3),keepdim=True)/mask.sum((2,3),keepdim=True)
            correction = (q-mean)*mask; stages['correction_after_mean'] = spatial(correction,measurement)
            replay = (b+correction).clamp(0,1)[0].permute(1,2,0).numpy().copy()
        error = float(np.max(np.abs(replay - raw)))
        a.require(error <= a.HEAD_RAW_TOLERANCE, 'Manual exact-layer trace differs from audited saved output')
        delta = (raw - base)[support].astype(np.float64)
        byte_delta = prediction.astype(np.int16) - baseline.astype(np.int16)
        row = {'id': c['id'], 'source': c['source'], 'reference': c['source_person_or_reference'], 'profile': c['profile'],
            'stages': stages, 'CPU_raw_replay_maximum': error,
            'saved_postclip_correction_RMS': float(np.sqrt(np.square(delta).mean())),
            'saved_postclip_correction_maximum': float(np.abs(delta).max()),
            'changed_PNG_pixels': int(np.any(byte_delta != 0, axis=2).sum()),
            'changed_PNG_values': int((byte_delta != 0).sum()), 'maximum_PNG_byte_change': int(np.abs(byte_delta).max()),
            'PNG_identical': bool(np.array_equal(prediction, baseline)),
            'raw_baseline_feature_MSE': raw_feature_error(base, target, a.feature_mask(c, support)),
            'raw_update50_feature_MSE': raw_feature_error(raw, target, a.feature_mask(c, support)),
            'baseline_PNG_feature_MSE': a.detail_metric(baseline, target, a.feature_mask(c, support)),
            'update50_PNG_feature_MSE': a.detail_metric(prediction, target, a.feature_mask(c, support))}
        rows.append(row)
    a.require(a.state_hash(head.state_dict()) == before and not any(v.grad is not None or v.requires_grad for v in head.parameters()), 'Diagnostic changed model/gradient state')
    font_path = Path('C:/Windows/Fonts/arial.ttf'); font = ImageFont.truetype(str(font_path), 15)
    by_ref = defaultdict(list)
    for c in p['cases']:
        by_ref[c['source_person_or_reference']].append(c)
    sheets = []
    for index, (reference, cases) in enumerate(by_ref.items(), 1):
        sheet = Image.new('RGB', (1072, 1516), '#f1f2f4'); draw = ImageDraw.Draw(sheet)
        draw.text((12, 5), 'Paired TRAINING photographs only | ' + reference + ' | ' + cases[0]['source'], font=font, fill='#111111')
        for col, label in enumerate(('Input', 'Own DGP baseline', 'V26 stopped50', 'Paired TRAIN target')):
            draw.text((12 + col * 264, 30), label, font=font, fill='#111111')
        cells = []
        for row_index, c in enumerate(cases):
            y = 64 + row_index * 288
            draw.text((12, y), c['profile'] + ' | ' + c['id'], font=font, fill='#111111')
            images = [a.rgb(bundle / c['input']), a.rgb(returned / 'outputs/update0' / (c['id'] + '.png')),
                      a.rgb(returned / 'outputs/update50' / (c['id'] + '.png')), a.rgb(bundle / c['target'])]
            for col, value in enumerate(images):
                xy = (12 + col * 264, y + 24)
                sheet.paste(Image.fromarray(value), xy)
                cells.append({'case': c['id'], 'column': col, 'xy': list(xy), 'pixel_sha256': __import__('hashlib').sha256(value.tobytes()).hexdigest()})
        name = 'sheet_' + str(index).zfill(2) + '_' + reference + '.png'
        sheet.save(out / name)
        sheets.append({'file': name, 'reference': reference, 'cases': [c['id'] for c in cases], 'cells': cells})
    groups = {}
    for label in ('all', 'clear', 'degraded'):
        chosen = [r for r in rows if label == 'all' or (r['profile'] == 'clear') == (label == 'clear')]
        groups[label] = {'cases': len(chosen), 'identical_PNGs': sum(r['PNG_identical'] for r in chosen),
            'changed_PNG_pixels': sum(r['changed_PNG_pixels'] for r in chosen),
            'maximum_PNG_byte_change': max(r['maximum_PNG_byte_change'] for r in chosen),
            'median_saved_correction_RMS': float(np.median([r['saved_postclip_correction_RMS'] for r in chosen])),
            'median_q_spatial_RMS': float(np.median([r['stages']['q_before_projection']['spatial_RMS'] for r in chosen])),
            'median_q_RMS': float(np.median([r['stages']['q_before_projection']['RMS'] for r in chosen])),
            'median_spatial_correction_RMS': float(np.median([r['stages']['correction_after_mean']['RMS'] for r in chosen])),
            'raw_baseline_feature_MSE': float(np.mean([r['raw_baseline_feature_MSE'] for r in chosen])),
            'raw_update50_feature_MSE': float(np.mean([r['raw_update50_feature_MSE'] for r in chosen])),
            'baseline_PNG_feature_MSE': float(np.mean([r['baseline_PNG_feature_MSE'] for r in chosen])),
            'update50_PNG_feature_MSE': float(np.mean([r['update50_PNG_feature_MSE'] for r in chosen]))}
    artifacts = {sheet['file']: a.sha(out / sheet['file']) for sheet in sheets}
    result = {'complete': True, 'plan_sha256': a.sha(out / 'plan.json'), 'runner_sha256': a.sha(Path(__file__)),
        'seconds': time.monotonic() - started, 'rows': rows, 'groups': groups, 'sheets': sheets,
        'source_bindings_sha256': source_bindings, 'artifacts_sha256': artifacts,
        'head_exact_layer_traces': 50, 'parameter_changes': parameter_changes,
        'diagnostic_update50_capacity_arithmetic': a.capacity(a.read(returned/'outputs/update0/metrics.json')['groups'], a.read(returned/'outputs/update50/metrics.json')['groups']),
        'final800_capacity_evaluated': False, 'DGP_recognizer_forwards': 0, 'model_state_unchanged': before,
        'backward_calls': 0, 'optimizer_updates': 0, 'visual_review_pending': True,
        'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False}
    a.write(out / 'results.json', result)
    print(json.dumps({'complete': True, 'groups': groups, 'sheets': len(sheets), 'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    diagnose()

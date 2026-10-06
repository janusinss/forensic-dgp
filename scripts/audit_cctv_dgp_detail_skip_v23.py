"""Independent V23 saved-output audit and CPU inference replay; no training."""
import argparse
import ast
from collections import Counter
import hashlib
import math
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_cctv_dgp_detail_skip_v23 import PIN, ROOT, read, relative_name, require, safe, sha, write

BUNDLE = ROOT / 'outputs/cctv_dgp_detail_skip_vm_v23'
GROUP_METRICS = ('MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE', 'constant_mean_shift_only_MSE')
HEAD_RAW_TOLERANCE = 2e-6
RECOGNIZER_VECTOR_TOLERANCE = 2e-5
RECOGNIZER_COSINE_TOLERANCE = 5e-5


def verify_bundle(bundle):
    require(sha(bundle / 'protocol.json') == PIN, 'Frozen local V23 protocol differs')
    p = read(bundle / 'protocol.json')
    require(p['format'] == 'dgp-learned-detail-skip-capacity-v23' and len(p['assets_sha256']) == 209, 'Wrong frozen bundle')
    for name, digest in p['assets_sha256'].items():
        require(sha(safe(bundle, name)) == digest, 'Local original asset changed: ' + name)
    require(len(p['cases']) == 50 and len(p['references']) == 10 and
            all(c['role'] == 'train' for c in p['cases']) and
            all(r['role'] == 'train' for r in p['references']), 'Only frozen exposed training cases permitted')
    require(len({c['id'] for c in p['cases']}) == 50 and
            p['native_or_reserved_used'] is False and p['goal_complete'] is False, 'Training scope differs')
    s = read(bundle / 'schedule.json')
    require(s['epochs'] == 80 and s['batch_size'] == 5 and s['updates'] == 800 and len(s['batches']) == 800,
            'Finite training schedule differs')
    counts = Counter()
    for epoch in range(80):
        samples = [i for batch in s['batches'][epoch * 10:(epoch + 1) * 10] for i in batch]
        require(all(len(batch) == 5 for batch in s['batches'][epoch * 10:(epoch + 1) * 10]) and
                sorted(samples) == list(range(50)), 'Epoch is not an exact50-case permutation')
        counts.update(samples)
    require(set(counts.values()) == {80}, 'Fitting exposures differ')
    return p


def rgb(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        result = np.asarray(image).copy()
    require(result.shape == (256, 256, 3) and result.dtype == np.uint8, 'Expected exact RGB uint8 256 image')
    return result


def raw_rgb(path):
    import numpy as np
    result = np.load(path, allow_pickle=False)
    require(result.shape == (256, 256, 3) and result.dtype == np.float32 and
            np.isfinite(result).all() and 0 <= result.min() <= result.max() <= 1, 'Raw RGB schema/range differs')
    return result


def vector(path):
    import numpy as np
    result = np.load(path, allow_pickle=False)
    require(result.shape == (512,) and result.dtype == np.float32 and np.isfinite(result).all() and
            abs(float(np.linalg.norm(result)) - 1) <= 2e-6, 'Recognizer vector schema/norm differs')
    return result


def observed(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        result = np.asarray(image).copy()
    require(result.shape == (256, 256) and result.dtype == np.uint8 and
            set(np.unique(result)).issubset({0, 255}) and (result > 0).any(), 'Observed mask schema differs')
    return result > 0


def interior(mask, radius):
    """Independent square erosion using summed-area counts with zero exterior."""
    import numpy as np
    size = 2 * radius + 1
    padded = np.pad(mask.astype(np.int64), radius)
    integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    counts = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return counts == size * size


def feature_mask(c, mask):
    import numpy as np
    feature = np.zeros(mask.shape, bool)
    for point in c['landmarks5_canvas_xy']:
        x, y = np.floor(point).astype(int)
        feature[max(0, y - 12):min(256, y + 12), max(0, x - 12):min(256, x + 12)] = True
    feature &= interior(mask, 6)
    require(feature.any(), 'Empty frozen landmark measurement support')
    return feature


def pixel_metrics(prediction, target, mask):
    """Historical win7/sample SSIM, independently assembled without skimage's metric."""
    import numpy as np
    from scipy.ndimage import uniform_filter
    a, b = prediction.astype(np.float32) / np.float32(255), target.astype(np.float32) / np.float32(255)
    error = a - b
    mse = float(np.square(error[mask]).astype(np.float64).mean())
    ux, uy = uniform_filter(b, size=(7, 7, 1)), uniform_filter(a, size=(7, 7, 1))
    vx = (uniform_filter(b * b, size=(7, 7, 1)) - ux * ux) * (49 / 48)
    vy = (uniform_filter(a * a, size=(7, 7, 1)) - uy * uy) * (49 / 48)
    vxy = (uniform_filter(a * b, size=(7, 7, 1)) - ux * uy) * (49 / 48)
    score = ((2 * ux * uy + .01**2) * (2 * vxy + .03**2)) / ((ux * ux + uy * uy + .01**2) * (vx + vy + .03**2))
    valid = interior(mask, 3)
    require(valid.any(), 'Empty historical SSIM support')
    return {'MSE': mse, 'PSNR': -10 * math.log10(mse) if mse else None,
            'perfect_match': mse == 0, 'SSIM': float(score[valid].astype(np.float64).mean()),
            'MAE': float(np.abs(error[mask]).astype(np.float64).mean())}


def detail_metric(prediction, target, feature):
    import cv2
    import numpy as np
    z = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-.5 * (z / 2)**2); kernel /= kernel.sum()
    delta = ((prediction.astype(np.float64) - target.astype(np.float64)) / 255 * np.array([.299, .587, .114])).sum(2)
    high = delta - cv2.sepFilter2D(delta, cv2.CV_64F, kernel, kernel, borderType=cv2.BORDER_REFLECT)
    return float(np.square(high)[feature].mean())


def png(raw, camera, mask):
    import numpy as np
    return np.where(mask[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8)


def aggregate(rows):
    import numpy as np
    labels = {'all', 'clear', 'degraded'}
    for row in rows:
        labels.update(row['source'] + '/' + label for label in ('all', 'clear', 'degraded', row['profile']))
    groups = {}
    for label in sorted(labels):
        chosen = []
        for row in rows:
            category = 'clear' if row['profile'] == 'clear' else 'degraded'
            membership = {'all', category, row['source'] + '/all', row['source'] + '/' + category,
                          row['source'] + '/' + row['profile']}
            if label in membership:
                chosen.append(row)
        require(chosen, 'Empty declared source/profile group')
        groups[label] = {'cases': len(chosen), **{k: float(np.mean([r['metrics'][k] for r in chosen])) for k in GROUP_METRICS}}
    return groups


def capacity(baseline, candidate):
    failures = []
    require(set(baseline) == set(candidate) and len(baseline) == 17, 'All17 frozen groups required')
    for label, b in baseline.items():
        a = candidate[label]
        require(a['cases'] == b['cases'], 'Source/profile count differs')
        for metric in ('MSE', 'SSIM', 'ArcFace_observed_fixed'):
            failed = a[metric] > b[metric] + 1e-12 if metric == 'MSE' else a[metric] < b[metric] - 1e-6
            if failed:
                failures.append({'group': label, 'metric': metric, 'baseline': b[metric], 'candidate': a[metric]})
    b, a = baseline['degraded'], candidate['degraded']
    gain = 1 - a['landmark_high_frequency_MSE'] / b['landmark_high_frequency_MSE']
    sources = sorted(label[:-9] for label in baseline if label.endswith('/degraded'))
    source_gains = {s: 1 - candidate[s + '/degraded']['landmark_high_frequency_MSE'] /
                   baseline[s + '/degraded']['landmark_high_frequency_MSE'] for s in sources}
    brightness = max(0, b['MSE'] - a['constant_mean_shift_only_MSE']) / max(b['MSE'] - a['MSE'], 1e-12)
    if brightness > .2:
        failures.append({'group': 'degraded', 'metric': 'brightness_gain_fraction', 'candidate': brightness, 'maximum': .2})
    return {'preservation_failures': failures, 'degraded_feature_MSE_relative_gain': gain,
            'source_feature_gains': source_gains, 'brightness_gain_fraction': brightness,
            'necessary_capacity_pass': not failures and gain >= .1 and all(v >= 0 for v in source_gains.values())}


def numeric_tree(actual, expected, tolerance=1e-9, label='receipt'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys differ')
        for key in expected:
            numeric_tree(actual[key], expected[key], tolerance, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list differs')
        for index, value in enumerate(expected):
            numeric_tree(actual[index], value, tolerance, label + '/' + str(index))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and abs(actual - expected) <= tolerance,
                label + ': numeric value differs')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact value differs')


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(value.dtype).encode() + b'\0')
        digest.update(str(tuple(value.shape)).encode() + b'\0' + value.numpy().tobytes())
    return digest.hexdigest()


def make_head(bundle):
    """Compile only the pinned local class AST, never the VM launcher/prepare/run."""
    import torch
    from torch.nn import functional as F
    tree = ast.parse((bundle / 'scripts/cctv_dgp_detail_skip_v23.py').read_text(encoding='utf-8'))
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
    head = next(node for node in prepare.body if isinstance(node, ast.ClassDef) and node.name == 'DetailHead')
    namespace = {'torch': torch, 'nn': torch.nn, 'F': F}
    exec(compile(ast.Module(body=[head], type_ignores=[]), '<pinned-local-V23-class-only>', 'exec'), namespace)
    torch.manual_seed(20261005)
    return namespace['DetailHead']().cpu().eval().requires_grad_(False)


def verify_initial_state(local, saved, reported_hash):
    """VM2.9/CPU2.13 random initialization differs by at most1e-8; tail/buffers exact."""
    import torch
    require(set(local) == set(saved) and state_hash(saved) == reported_hash, 'Initial VM head receipt/hash differs')
    maximum = 0.
    for name, value in saved.items():
        require(isinstance(value, torch.Tensor) and value.dtype == local[name].dtype and value.shape == local[name].shape and
                torch.isfinite(value).all(), 'Initial head tensor schema differs')
        delta = float((value.double() - local[name].double()).abs().max())
        maximum = max(maximum, delta)
        if name in ('kernel', 'reflect_indices', 'direct.weight', 'direct.bias', 'tail.weight', 'tail.bias'):
            require(torch.equal(value, local[name]), 'Fixed buffers/zero initial tail differ')
        else:
            require(delta <= 1e-8, 'Seeded CPU/VM initializer exceeds documented1e-8 compatibility difference')
    return maximum


def fixed_grid(matrix):
    import numpy as np
    affine = np.asarray(matrix, dtype=np.float64)
    require(affine.shape == (2, 3) and np.isfinite(affine).all(), 'Frozen affine schema differs')
    inverse = np.linalg.inv(np.vstack((affine, [0, 0, 1])))
    yy, xx = np.indices((112, 112))
    xy = inverse @ np.vstack((xx.ravel(), yy.ravel(), np.ones(112 * 112)))
    return np.stack((2 * (xy[0].reshape(112, 112) + .5) / 256 - 1,
                     2 * (xy[1].reshape(112, 112) + .5) / 256 - 1), axis=-1).astype(np.float32)


def audit_return(bundle, returned, output, cap_seconds=600):
    import numpy as np
    import torch
    started = time.monotonic()
    require(not output.exists(), 'Preserve previous independent audit')
    def clock():
        require(time.monotonic() - started < cap_seconds, 'Local inference/audit deadline exceeded')
    p = verify_bundle(bundle)
    manifest = read(returned / 'export_manifest.json')
    require(manifest['complete'] is True and manifest['protocol_sha256'] == PIN, 'Returned manifest scope differs')
    files = {f.relative_to(returned).as_posix() for f in returned.rglob('*') if f.is_file()}
    require(files == set(manifest['files_sha256']) | {'export_manifest.json'}, 'Unbound/omitted returned files')
    for name, digest in manifest['files_sha256'].items():
        clock(); require(sha(safe(returned, name)) == digest, 'Returned file changed: ' + name)
    for name in ('protocol.json', 'schedule.json', 'scripts/cctv_dgp_detail_skip_v23.py', 'scripts/run_v23.sh'):
        require(sha(returned / name) == sha(bundle / name), 'Returned frozen source differs: ' + name)
    head = make_head(bundle)
    torch.set_num_threads(4)
    fixed_buffers = {k: v.clone() for k, v in head.state_dict().items() if k in ('kernel', 'reflect_indices')}
    initial_state = state_hash(head.state_dict())
    local_initial_tensors = {k: v.clone() for k, v in head.state_dict().items()}
    VM_initial_state = None
    maximum_initialization_difference = 0.
    counts = {'head_forwards': 0, 'recognizer_forwards': 0, 'raw_PNG_pairs': 0, 'metric_rows': 0, 'complete_snapshots': 0}
    maxima = {'head_raw': 0., 'recognizer_vector': 0., 'recognizer_cosine': 0., 'saved_metric': 0.}
    preflights = []
    for path in sorted(returned.glob('preflight_*.json')):
        receipt = read(path)
        require(receipt['complete'] is True and receipt['protocol_sha256'] == PIN and 0 < receipt['seconds'] <= 300,
                'VM preflight receipt/protocol/time differs')
        require(re.fullmatch('[0-9a-f]{64}', receipt['initial_head_state']) and receipt['initial_exact_cached_DGP_cases'] == 50 and
                receipt['trainable_parameters'] == 4613 and 'L4' in receipt['gpu'], 'VM head/initial/cache/GPU preflight differs')
        require([c['id'] for c in receipt['fresh_DGP_parity']] == p['fresh_DGP_parity_cases'] and
                all(math.isfinite(c['maximum_raw_error']) and 0 <= c['maximum_raw_error'] <= 2e-6
                    for c in receipt['fresh_DGP_parity']), 'Saved fresh CUDA DGP parity differs')
        require(all(re.fullmatch('[0-9a-f]{64}', receipt[k]) for k in ('recognizer_state', 'DGP_state_unchanged')), 'VM frozen-state hashes missing')
        require(receipt.get('neural_forward_counts') == {'detail_head': 50, 'DGP': 4, 'fixed_recognizer': 100}, 'Preflight neural counts differ')
        preflights.append(receipt)
    snapshots = {}
    incomplete_folders = []
    reference_vectors = {}
    identity = None
    identity_state = None
    for folder in sorted((returned / 'outputs').glob('update*'), key=lambda f: int(f.name[6:]) if f.name[6:].isdigit() else -1):
        require(folder.is_dir() and folder.name in {'update0', 'update50', 'update400', 'update800'}, 'Unknown snapshot directory')
        update = int(folder.name[6:]); clock()
        if not (folder / 'metrics.json').exists():
            incomplete_folders.append(folder.name); continue
        require(preflights, 'Snapshots require a full successful VM preflight receipt')
        saved = read(folder / 'metrics.json')
        require(saved['update'] == update and 0 < saved['seconds'] <= 1800 and
                [r['id'] for r in saved['rows']] == [c['id'] for c in p['cases']], 'Snapshot update/time/50-case order differs')
        expected_names = {'head.pth', 'metrics.json'} | {c['id'] + suffix for c in p['cases'] for suffix in ('.npy', '.png', '_embedding.npy')}
        if update == 0:
            expected_names |= {c['id'] + '_target_embedding.npy' for c in p['cases']}
        require({f.name for f in folder.iterdir()} == expected_names, 'Complete snapshot file set differs')
        loaded = torch.load(folder / 'head.pth', map_location='cpu', weights_only=True)
        require(all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in loaded.values()), 'Nonfinite head state')
        head.load_state_dict(loaded, strict=True)
        require(state_hash(loaded) == saved['head_state'] and all(torch.equal(loaded[k], v) for k, v in fixed_buffers.items()),
                'Saved head hash/fixed buffers differ')
        if update == 0:
            require(all(r['initial_head_state'] == saved['head_state'] for r in preflights), 'Initial VM preflight/snapshot state changed')
            maximum_initialization_difference = verify_initial_state(local_initial_tensors, loaded, saved['head_state'])
            VM_initial_state = saved['head_state']
        state_before = state_hash(head.state_dict())
        if identity is None:
            sys.path.insert(0, str(bundle))
            from cctv_dgp_pilot import FixedObservedIdentity
            identity = FixedObservedIdentity(bundle / 'weights/w600k_r50.onnx', 'cpu')
            identity_state = state_hash(identity.state_dict())
            require(all(r['recognizer_state'] == identity_state for r in preflights), 'CPU/VM frozen recognizer state differs')
        refs = {r['id']: r for r in p['references']}
        rows = []
        for c, original in zip(p['cases'], saved['rows']):
            clock(); camera, target = rgb(bundle / c['input']), rgb(bundle / c['target'])
            mask, base = observed(bundle / c['observed']), raw_rgb(bundle / c['raw_dgp'])
            raw, delivered = raw_rgb(folder / (c['id'] + '.npy')), rgb(folder / (c['id'] + '.png'))
            require(np.array_equal(delivered, png(raw, camera, mask)), 'Exact raw/PNG composition differs')
            require(np.array_equal(raw[~mask], base[~mask]) and np.array_equal(delivered[~mask], camera[~mask]), 'Unobserved raw/delivered padding changed')
            if update == 0:
                require(np.array_equal(raw, base) and np.array_equal(delivered, rgb(bundle / c['png_dgp'])), 'Initial baseline is not exact cached DGP')
            shift = (raw - base)[mask].astype(np.float64).mean(0)
            mean_only = np.clip(base.astype(np.float64) + shift, 0, 1).astype(np.float32)
            metrics = pixel_metrics(delivered, target, mask)
            metrics['constant_mean_shift_only_MSE'] = pixel_metrics(png(mean_only, camera, mask), target, mask)['MSE']
            metrics['landmark_high_frequency_MSE'] = detail_metric(delivered, target, feature_mask(c, mask))
            predicted_vector = vector(folder / (c['id'] + '_embedding.npy'))
            truth = vector(returned / 'outputs/update0' / (c['id'] + '_target_embedding.npy'))
            metrics['ArcFace_observed_fixed'] = float(predicted_vector @ truth)
            numeric_tree(original['metrics'], metrics, tolerance=1e-9, label=c['id'] + '/metrics')
            numeric_tree(original['postclip_mean_RGB_shift'], shift.tolist(), tolerance=1e-12, label=c['id'] + '/mean')
            require(original['source'] == c['source'] and original['profile'] == c['profile'], 'Snapshot source/profile differs')
            def tensor(a):
                return torch.from_numpy(np.asarray(a).copy()).permute(2, 0, 1)[None]
            x = tensor(camera.astype(np.float32) / np.float32(255))
            b = tensor(base); m = torch.from_numpy(mask.astype(np.float32))[None, None]
            grid = torch.from_numpy(fixed_grid(refs[c['source_person_or_reference']]['matrix112']))[None]
            with torch.inference_mode():
                replay = head(x, b, m)[0].permute(1, 2, 0).numpy().copy(); counts['head_forwards'] += 1
                if c['source_person_or_reference'] not in reference_vectors:
                    reference_vectors[c['source_person_or_reference']] = identity.embedding(tensor(target.astype(np.float32) / np.float32(255)), m, grid)[0].numpy().copy()
                    counts['recognizer_forwards'] += 1
                replay_vector = identity.embedding(tensor(delivered.astype(np.float32) / np.float32(255)), m, grid)[0].numpy().copy()
                counts['recognizer_forwards'] += 1
            reference = reference_vectors[c['source_person_or_reference']]
            deltas = {'head_raw': float(np.max(np.abs(replay - raw))),
                      'recognizer_vector': max(float(np.max(np.abs(replay_vector - predicted_vector))), float(np.max(np.abs(reference - truth)))),
                      'recognizer_cosine': abs(float(replay_vector @ reference) - metrics['ArcFace_observed_fixed'])}
            for key, value in deltas.items():
                maxima[key] = max(maxima[key], value)
            require(deltas['head_raw'] <= HEAD_RAW_TOLERANCE and deltas['recognizer_vector'] <= RECOGNIZER_VECTOR_TOLERANCE and
                    deltas['recognizer_cosine'] <= RECOGNIZER_COSINE_TOLERANCE, 'Fixed CPU replay tolerance failed: ' + c['id'])
            require(np.max(np.abs(png(replay, camera, mask).astype(int) - delivered.astype(int))) <= 1, 'CPU replay PNG difference exceeds quantization bound')
            for key in metrics:
                if type(metrics[key]) is float:
                    maxima['saved_metric'] = max(maxima['saved_metric'], abs(metrics[key] - original['metrics'][key]))
            counts['raw_PNG_pairs'] += 1; counts['metric_rows'] += 1
            rows.append({'id': c['id'], 'source': c['source'], 'profile': c['profile'], 'metrics': metrics})
        require(state_hash(head.state_dict()) == state_before and state_hash(identity.state_dict()) == identity_state and
                not any(v.grad is not None or v.requires_grad for v in list(head.parameters()) + list(identity.parameters())), 'Inference mutated model state or enabled gradients')
        groups = aggregate(rows); numeric_tree(saved['groups'], groups, 1e-9, 'snapshot/groups')
        require(len(groups) == 17, 'All17 training source/profile groups required')
        snapshots[update] = {'groups': groups, 'head_state': saved['head_state'], 'seconds': saved['seconds']}
        counts['complete_snapshots'] += 1
    result_path, failure_path = returned / 'outputs/results.json', returned / 'outputs/failure.json'
    result, failure = read(result_path) if result_path.exists() else None, read(failure_path) if failure_path.exists() else None
    for receipt, is_failure in ((result, False), (failure, True)):
        if receipt is None:
            continue
        require(receipt['protocol_sha256'] == PIN and type(receipt['updates']) is int and 0 <= receipt['updates'] <= 800 and
                type(receipt['backwards']) is int and receipt['updates'] <= receipt['backwards'] <= receipt['updates'] + 2,
                'Training progress/backward budget differs')
        require(0 < receipt['seconds'] <= (2130 if is_failure else 1800) and receipt['app_promotion'] is False,
                'Worker/external time or promotion scope differs')
        if is_failure:
            require(receipt['complete'] is False and receipt['resume_permitted'] is False and
                    isinstance(receipt['cause'], str) and isinstance(receipt['traceback'], str), 'Failure evidence was waived')
    checked_capacity = None
    if result is not None:
        require(result['complete'] is True and result['updates'] == 800 and result['backwards'] == 801 and
                set(snapshots) == {0, 50, 400, 800} and not incomplete_folders and
                result['native_or_reserved_used'] is False and result['goal_complete'] is False and
                result['independent_audit_pending'] is True and result['visual_review_pending'] is True and
                result['recognizer_unchanged'] is True and result['DGP_checkpoint_unchanged'] is True and
                result['selection'] == 'Final800 only; no checkpoint selected for application', 'Full training/capacity receipt scope differs')
        checked_capacity = capacity(snapshots[0]['groups'], snapshots[800]['groups'])
        numeric_tree({k: result[k] for k in checked_capacity}, checked_capacity, 1e-9, 'final/capacity')
        require(snapshots[800]['head_state'] != snapshots[0]['head_state'], 'No demonstrated learned head change')
    timing_path = returned / 'outputs/timing_update20.json'
    timing = read(timing_path) if timing_path.exists() else None
    if timing is not None:
        require(timing['updates'] == 20 and timing['cap_seconds'] == 1500 and 0 < timing['seconds'] <= timing['projected_seconds'] and
                math.isfinite(timing['projected_seconds']) and timing['projected_seconds'] - timing['seconds'] >= 120,
                'Timing stop receipt differs')
        if result is not None:
            require(timing['projected_seconds'] <= 1500, 'Training continued after failed timing stop')
    early_path = returned / 'outputs/early_structure_stop.json'
    early = read(early_path) if early_path.exists() else None
    if early is not None:
        require(0 in snapshots and 50 in snapshots, 'Early stop requires its paired training snapshots')
        gain = 1 - snapshots[50]['groups']['degraded']['landmark_high_frequency_MSE'] / snapshots[0]['groups']['degraded']['landmark_high_frequency_MSE']
        numeric_tree(early, {'update': 50, 'relative_feature_error_gain': gain, 'minimum': .01, 'pass': gain >= .01}, 1e-9, 'early stop')
        require(early['pass'] or (result is None and not any(u > 50 for u in snapshots)), 'Training continued after failed early stop')
    if result is not None:
        require(timing is not None and early is not None and early['pass'], 'Full result lacks required timing/early receipts')
    stopped_state = None
    if failure is not None and (returned / 'outputs/stopped_head.pth').exists():
        stopped = torch.load(returned / 'outputs/stopped_head.pth', map_location='cpu', weights_only=True)
        require(set(stopped) == set(local_initial_tensors) and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all()
                and v.shape == local_initial_tensors[k].shape and v.dtype == local_initial_tensors[k].dtype
                for k, v in stopped.items()), 'Stopped head schema differs')
        stopped_state = state_hash(stopped)
        if failure['cause'] == 'No one-percent early structural gain; retain stop':
            require(failure['updates'] == 50 and failure['backwards'] == 51 and early is not None and early['pass'] is False and
                    stopped_state == snapshots[50]['head_state'], 'Early-stop progress or retained head differs')
    from audit_cctv_dgp_detail_skip_v23_execution import check_execution
    execution_audit = check_execution(returned, preflights, snapshots, result, failure, local_initial_tensors,
        stopped_state if stopped_state is not None else snapshots[max(snapshots)]['head_state'] if snapshots else None)
    report = {'complete': True, 'scope': 'Independent saved-source/arithmetic audit and CPU inference replay; photographic TRAINING capacity only',
        'protocol_sha256': PIN, 'checker_sha256': sha(Path(__file__)),
        'execution_checker_sha256': sha(Path(__file__).with_name('audit_cctv_dgp_detail_skip_v23_execution.py')), 'source_assets_verified': len(p['assets_sha256']),
        'returned_files_verified': len(files), 'export_manifest_sha256': sha(returned / 'export_manifest.json'),
        'seconds': time.monotonic() - started, 'local_cap_seconds': cap_seconds, 'counts': counts, 'maximum_differences': maxima,
        'VM_preflight_receipts': len(preflights), 'complete_snapshot_updates': sorted(snapshots), 'incomplete_snapshot_folders': incomplete_folders,
        'local_seeded_initial_state': initial_state, 'VM_initial_state': VM_initial_state,
        'maximum_initialization_compatibility_difference': maximum_initialization_difference,
        'initialization_limit': 'Exact saved VM preflight/snapshot hash, zero tail, fixed buffers and50 cached baseline arrays; seeded random weights across torch2.9/2.13 allow only documented1e-8 initialization rounding. CPU replay and quality tolerances unchanged.',
        'VM_training_result_present': result is not None, 'VM_failure_present': failure is not None,
        'training_progress_receipt': {k: (failure or result)[k] for k in ('updates', 'backwards', 'seconds')} if failure or result else None,
        'failure_cause': failure['cause'] if failure else None, 'early_structure_stop': early,
        'stopped_head_state': stopped_state,
        'training_receipt_verified': result is not None and failure is None, 'capacity_arithmetic': checked_capacity,
        'necessary_capacity_pass': checked_capacity is not None and checked_capacity['necessary_capacity_pass'] and failure is None,
        'timing_receipt_bounds_checked': timing is not None, 'timing_projection_sample_arithmetic_verified': execution_audit.get('update20_projection_arithmetic_verified', False),
        'execution_audit': execution_audit,
        'timing_limit': 'Source-bound execution receipts, exact step samples/projection and supervisor durations checked. Peak is torch allocated memory; no new GPU measurement by this local audit.',
        'DGP_replay': 'Saved4-case CUDA preflight checked; no original DGP forward is run by this local audit.',
        'replay_limit': 'Saved GPU vectors determine original gates. CPU replay has fixed numerical tolerances and is not canonical-app or native qualification.',
        'visual_review': 'Pending every returned case; this script issues no visual verdict',
        'backward_calls': 0, 'optimizer_updates': 0, 'native_or_reserved_used': False, 'independent_final_review': False,
        'app_promotion': False, 'goal_complete': False}
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--returned', type=Path, default=ROOT / 'outputs/cctv_dgp_detail_skip_v23_return')
    parser.add_argument('--receipt', type=Path, default=ROOT / 'outputs/cctv_dgp_detail_skip_v23_independent_audit.json')
    args = parser.parse_args()
    report = audit_return(args.bundle.resolve(), args.returned.resolve(), args.receipt.resolve())
    print(__import__('json').dumps({k: report[k] for k in ('complete', 'counts', 'seconds', 'necessary_capacity_pass', 'visual_review')}, indent=2))


if __name__ == '__main__':
    main()

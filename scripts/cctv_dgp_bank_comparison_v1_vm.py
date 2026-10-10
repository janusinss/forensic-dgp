"""Manual L4-only matched finite learning; no automatic continuation/promotion."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_bank_comparison_v1_contract import (NAME, STEM, STATE, RECOGNIZER, BUDGETS,
    NORMALIZERS, ANCHORS, read, write, sha, verify)


def vm(root):
    assert platform.system() == 'Linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root.resolve() == (Path.home() / 'forensic-dgp' / NAME).resolve()
    assert os.environ.get('TMUX'), 'Launch manually inside tmux'
    assert sys.prefix == str(Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv')


class ArmStop(RuntimeError):
    """Prospectively declared per-route learning/quality stop, preserved in export."""


def run(root, p, pin):
    vm(root)
    assert not (root / 'outputs').exists(), 'Immutable comparison root; never resume failed state'
    assert shutil.disk_usage(root).free >= BUDGETS['minimum_free_disk_bytes'], 'Need16GiB free after install'
    # A GPU task, rather than the user's own tmux shell, is a competing workload.
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=10)
    assert not result.stdout.strip(), 'Competing GPU process; no launch'
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_bank_comparison_v1_model import CorrectedCurrentDGP, ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior, bank_features
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
    from cctv_dgp_head4_lossless_v1 import pack, unpack
    from frozen_capacity_contract import feature_support, exported_pixel_metrics, detail_metric, capacity
    from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups

    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L4'
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(20261010); torch.cuda.manual_seed_all(20261010)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    out = root / 'outputs/bank_comparison_v1'; out.mkdir(parents=True)
    start = time.monotonic()
    counts = {n: 0 for n in ['original', 'A', 'B', 'recognizer', 'prior_fixture']}
    counts.update({'gradient_queries': 0, 'backwards': 0, 'optimizer_updates': 0})
    progress = {'stage': 'initializing', 'route': None, 'route_updates': 0, 'counts': counts}
    candidate = None; optimizer = None; scheduler = None; route = None
    completed = []

    def interrupt(_s, _frame):
        raise TimeoutError('External finite worker stop; retain completed state')

    signal.signal(signal.SIGTERM, interrupt)

    def retained():
        return sum(f.stat().st_size for f in out.rglob('*') if f.is_file())

    def clock():
        torch.cuda.synchronize()
        assert time.monotonic() - start < BUDGETS['worker_seconds'], 'Worker90-minute stop'
        assert shutil.disk_usage(root).free >= BUDGETS['disk_reserve_bytes'], 'Disk1GiB reserve stop'
        assert torch.cuda.max_memory_allocated() <= BUDGETS['peak_vram_bytes'], 'VRAM20GiB stop'
        assert retained() <= BUDGETS['return_uncompressed_bytes'], 'Output6GiB stop'
        for name in ['original', 'A', 'B', 'recognizer', 'prior_fixture']:
            assert counts[name] <= BUDGETS[name + '_forward_calls'], name + ' forward stop'
        for name in ['gradient_queries', 'backwards', 'optimizer_updates']:
            assert counts[name] <= BUDGETS['maximum_' + name], name + ' stop'

    def pixels(path, mode='RGB'):
        with Image.open(path) as im:
            assert im.size == (256, 256)
            return np.asarray(im.convert(mode)).copy()

    def tensor(a):
        return torch.from_numpy(a.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None].cuda()

    refs = {r['id']: r for r in p['references']}
    cases = p['cases']

    def batch(begin):
        cs = cases[begin:begin+5]
        cameras = [pixels(root / c['input']) for c in cs]
        targets = [pixels(root / refs[c['source_person_or_reference']]['target']) for c in cs]
        masks = [pixels(root / refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in cs]
        features = [feature_support(mask, c['landmarks5_canvas_xy']) for mask, c in zip(masks, cs)]
        grids = [grid112(refs[c['source_person_or_reference']]['matrix112']) for c in cs]
        return {'cases': cs, 'camera': cameras, 'target8': targets, 'mask8': masks, 'feature8': features,
            'x': torch.cat([tensor(a) for a in cameras]), 'target': torch.cat([tensor(a) for a in targets]),
            'mask': torch.from_numpy(np.stack(masks).astype(np.float32)).cuda()[:, None],
            'feature': torch.from_numpy(np.stack(features).astype(np.float32)).cuda()[:, None],
            'grid': torch.from_numpy(np.stack(grids)).cuda()}

    def base_raw(begin):
        ids = [c['id'] for c in cases[begin:begin+5]]
        a, _ = unpack(out / 'baseline/packs' / f'b{begin//5:03d}.npz', ids)
        return torch.from_numpy(a).cuda().permute(0, 3, 1, 2)

    def objective(b, prediction, baseline):
        terms = pixel_and_structure_losses(prediction, b['target'], b['mask'], b['feature'])
        with torch.no_grad():
            old_terms = pixel_and_structure_losses(baseline, b['target'], b['mask'], b['feature'])
        vec = identity.embedding(torch.cat([prediction, baseline.detach(), b['target']]),
            torch.cat([b['mask']] * 3), torch.cat([b['grid']] * 3))
        target_vec = vec[10:15].detach().double()
        arc = 1 - (vec[:5].double() * target_vec).sum(1)
        old_arc = 1 - (vec[5:10].detach().double() * target_vec).sum(1)
        raw_terms = [terms['MSE'].mean(), terms['SSIM_loss'].mean(), arc.mean(), terms['landmark_structure'][1:].mean()]
        normal = [value / scale for value, scale in zip(raw_terms, NORMALIZERS)]
        barrier = 5 * (torch.relu(terms['MSE'] - old_terms['MSE'] - 1e-12).mean() / NORMALIZERS[0]
            + torch.relu(terms['SSIM_loss'] - old_terms['SSIM_loss'] - 1e-6).mean() / NORMALIZERS[1]
            + torch.relu(arc - old_arc - 1e-6).mean() / NORMALIZERS[2])
        anchors = terms['MSE'][0] / ANCHORS[0] + terms['MSE'][1] / ANCHORS[1]
        value = sum(w * v for w, v in zip(p['reconstruction_weights'], normal)) + anchors + barrier
        assert torch.isfinite(value)
        return value, {'objective': float(value.detach()), 'raw_terms': [float(v.detach()) for v in raw_terms],
            'anchors': float(anchors.detach()), 'barrier': float(barrier.detach())}

    def save_state(label):
        if candidate is None:
            return
        folder = out / ('state_' + label)
        folder.mkdir()
        torch.save({'model': candidate.state_dict(), 'optimizer': None if optimizer is None else optimizer.state_dict(),
            'scheduler': None if scheduler is None else scheduler.state_dict(), 'CPU_rng': torch.get_rng_state(),
            'CUDA_rng': torch.cuda.get_rng_state_all(), 'protocol_sha256': pin, 'route': route,
            'updates': progress['route_updates'], 'counts': dict(counts), 'schedule': p['schedule'],
            'manual_failed_resume_allowed': False}, folder / 'full_state.pth')
        write(folder / 'receipt.json', {'model_state': state_hash(candidate), 'buffer_state': buffer_hash(candidate),
            'updates': progress['route_updates'], 'route': route, 'full_state_sha256': sha(folder / 'full_state.pth'),
            'optimizer_present': optimizer is not None, 'automatic_resume_allowed': False})

    try:
        original, provenance = load_frozen_dgp_restorer(root / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cuda')
        identity = FixedObservedIdentity(root / 'weights/w600k_r50.onnx', 'cuda')
        assert state_hash(original.net) == STATE and state_hash(identity) == RECOGNIZER
        original.net.register_forward_hook(lambda *_: counts.__setitem__('original', counts['original'] + 1))
        identity.encoder.register_forward_hook(lambda *_: counts.__setitem__('recognizer', counts['recognizer'] + 1))
        write(out / 'environment.json', {'python': sys.version, 'GPU': torch.cuda.get_device_name(0), 'host': platform.node(),
            'CUDA': torch.version.cuda, 'packages': {n: importlib.metadata.version(n) for n in
                ['torch', 'torchvision', 'numpy', 'Pillow', 'scipy', 'scikit-image', 'onnx', 'onnx2torch']},
            'AMP': False, 'TF32': False, 'stored_normalization_unchanged': True,
            'pretrained_prior_source_declared': True, 'exact_future_resume_guaranteed': False, 'DGP_provenance': provenance})

        def create(label, learning=False):
            if label == 'A':
                model = CorrectedCurrentDGP(original.net)
            else:
                prior = load_prior(root / 'prior_source', root / 'bank_implementation', p['standalone_prior_sha256']).cuda()
                mean = torch.from_numpy(np.load(root / 'initializers/mean_style.npy', allow_pickle=False)).cuda()
                model = ConditionedBankDGP(original.net, prior, mean)
            model.cuda().eval().requires_grad_(False)
            initial = torch.load(root / f'initializers/initial_{label}.pth', map_location='cuda', weights_only=True)
            assert state_hash(model) == state_hash(initial), 'Fresh initializer differs'
            model.load_state_dict(initial, strict=True)
            del initial
            if learning:
                model.enable_vm_learning(root)
            if label == 'B':
                assert Path(sys.modules['_cctv_dgp_generative_bank_reference_v1'].__file__).parent.resolve() == (root/'bank_implementation').resolve()
            model.register_forward_hook(lambda *_a, label=label: counts.__setitem__(label, counts[label] + 1))
            return model

        def frozen():
            assert state_hash(original.net) == STATE and state_hash(identity) == RECOGNIZER
            assert all(not m.training for m in [original, candidate, identity] for m in m.modules())
            assert all(not v.requires_grad and v.grad is None for m in [original, identity] for v in m.parameters())
            if candidate is not None:
                names = set(candidate.learning_names())
                assert state_hash({n: v for n, v in candidate.state_dict().items() if n not in names}) == frozen_state
                assert buffer_hash(candidate) == buffer_state
                assert all(not v.requires_grad and v.grad is None for n, v in candidate.named_parameters() if n not in names)

        def metrics(label, model, indices=None, bank_disabled=False):
            # Full source/profile metrics are always3905. Ablation is explicitly100.
            stamp = time.monotonic()
            folder = out / label; folder.mkdir(); (folder / 'packs').mkdir(); (folder / 'previews').mkdir(); (folder / 'native').mkdir()
            selected = list(range(0, 3905, 5)) if indices is None else indices
            rows = []; native = []
            baseline = label == 'baseline'
            with torch.no_grad():
                for serial, begin in enumerate(selected):
                    clock(); assert time.monotonic() - stamp < BUDGETS['snapshot_seconds'], 'Snapshot20-minute stop'
                    b = batch(begin); ids = [c['id'] for c in b['cases']]
                    generated = model(b['x'], bank_enabled=False) if bank_disabled else model(b['x'])
                    pred = torch.where(b['mask'].bool(), generated, b['x'])
                    a = pred.permute(0, 2, 3, 1).cpu().numpy().copy()
                    base = a if baseline else unpack(out / 'baseline/packs' / f'b{begin//5:03d}.npz', ids)[0]
                    pngs = [deliver(raw, camera, mask) for raw, camera, mask in zip(a, b['camera'], b['mask8'])]
                    vectors = identity.embedding(torch.cat([pred, torch.cat([tensor(q) for q in pngs]), b['target']]),
                        torch.cat([b['mask']] * 3), torch.cat([b['grid']] * 3)).cpu().numpy().copy()
                    emb = np.stack([vectors[:5], vectors[5:10], vectors[10:15]], axis=1)
                    pack(folder / 'packs' / f'b{begin//5:03d}.npz', a, ids, emb, baseline=None if baseline else base)
                    for i, c in enumerate(b['cases']):
                        raw, png, target, mask = a[i], pngs[i], b['target8'][i], b['mask8'][i]
                        mraw, mpng, shift = mean_only(raw, base[i], b['camera'][i], mask)
                        rm, pm = pixel_metrics(raw, target, mask), exported_pixel_metrics(png, target, mask)
                        rm.update({'ArcFace_observed_fixed': float(emb[i, 0] @ emb[i, 2]),
                            'landmark_high_frequency_MSE': detail_float(raw, target, b['feature8'][i]),
                            'constant_mean_shift_only_MSE': pixel_metrics(mraw, target, mask)['MSE']})
                        pm.update({'ArcFace_observed_fixed': float(emb[i, 1] @ emb[i, 2]),
                            'landmark_high_frequency_MSE': detail_metric(png, target, b['feature8'][i]),
                            'constant_mean_shift_only_MSE': exported_pixel_metrics(mpng, target, mask)['MSE']})
                        rows.append({'id': c['id'], 'source': c['source'], 'profile': c['profile'], 'raw': rm, 'png': pm,
                            'postclip_mean_RGB_shift': shift.tolist(), 'raw_RGB_float32_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                            'PNG_RGB_sha256': hashlib.sha256(png.tobytes()).hexdigest()})
                        if c['id'] in p['preview_case_ids']:
                            Image.fromarray(png).save(folder / 'previews' / (c['id'] + '.png'))
                    if serial % 50 == 0:
                        print(json.dumps({'snapshot': label, 'cases': (serial+1)*5, 'of': len(selected)*5}), flush=True)
                for c in p['native_development']:
                    clock(); assert time.monotonic() - stamp < BUDGETS['snapshot_seconds']
                    camera = pixels(root / c['input']); mask = pixels(root / c['observed'], 'L') > 0
                    x = tensor(camera)
                    generated = model(x, bank_enabled=False) if bank_disabled else model(x)
                    pred = torch.where(torch.from_numpy(mask).cuda()[None, None], generated, x)
                    raw = pred[0].permute(1, 2, 0).cpu().numpy().copy(); png = deliver(raw, camera, mask)
                    np.save(folder / 'native' / (c['id'] + '.npy'), raw, allow_pickle=False)
                    Image.fromarray(png).save(folder / 'native' / (c['id'] + '.png'))
                    native.append({'id': c['id'], 'evidence': 'unpaired native CCTV; no clean reference metrics',
                        'raw_RGB_float32_sha256': hashlib.sha256(raw.tobytes()).hexdigest(), 'PNG_RGB_sha256': hashlib.sha256(png.tobytes()).hexdigest()})
            result = {'complete': True, 'label': label, 'rows': rows, 'native_unpaired_rows': native,
                'groups': {stage: review_groups(rows, stage) for stage in ['raw', 'png']}, 'seconds': time.monotonic() - stamp,
                'model_state': state_hash(model), 'full_TRAIN': indices is None, 'bank_disabled': bank_disabled,
                'model_qualification': False, 'native_quality_claim': False}
            write(folder / 'metrics.json', result); clock(); return result

        # Record full retained baseline before either route's optimizer.
        progress['stage'] = 'baseline_snapshot'
        baseline = metrics('baseline', original.net)
        baseline_bytes = retained()
        projected = int(baseline_bytes * 3.5 * 1.35 + 512 * 1024**2)
        write(out / 'storage_projection.json', {'baseline_bytes': baseline_bytes, 'snapshot_factor': 3.5,
            'compression_uncertainty_factor': 1.35, 'checkpoint_reserve_bytes': 512 * 1024**2,
            'projected_return_bytes': projected, 'cap_bytes': BUDGETS['return_uncompressed_bytes'],
            'export_reserve_bytes': projected, 'optimizer_updates': 0})
        assert projected <= BUDGETS['return_uncompressed_bytes'], 'Measured output projection stop before optimizer'
        assert shutil.disk_usage(root).free >= 2*projected-baseline_bytes+BUDGETS['disk_reserve_bytes'], 'Output+export reserve stop before optimizer'

        # First100 paired/fixed24 native initializers must equal fresh CUDA baseline.
        preview_indices = [i for i in range(0, 3905, 5) if cases[i]['id'] in p['preview_case_ids']]
        preflight_indices = [p['schedule'][0][0], p['schedule'][0][1]]
        preflight_start = time.monotonic()
        preflight_rows = []
        for route in ['A', 'B']:
            progress.update({'route': route, 'stage': 'initial_GPU_parity_and_gradients', 'route_updates': 0})
            candidate = create(route, learning=True)
            frozen_state = state_hash({n: v for n, v in candidate.state_dict().items() if n not in candidate.learning_names()})
            buffer_state = buffer_hash(candidate)
            names = candidate.learning_names(); params = candidate.learning_parameters()
            assert len(params) == p['routes'][route]['tensors'] and sum(v.numel() for v in params) == p['routes'][route]['elements']
            with torch.no_grad():
                if route == 'B':
                    # This one generated-seed fixture deliberately tests the full
                    # standalone512 path; restoration itself still stops at256.
                    z = torch.from_numpy(np.load(root / 'prior_fixtures/call0_z.npy', allow_pickle=False)).cuda()
                    native_rgb, features = bank_features(candidate.prior, z)
                    fixture_rows = []
                    (out / 'prior_cuda_fixture').mkdir()
                    actuals = {'raw512': native_rgb[0].permute(1, 2, 0)}
                    actuals.update({'feature' + size: value for size, value in features.items()})
                    for name, value in actuals.items():
                        filename = 'call0_raw512.npy' if name == 'raw512' else 'seed0_' + name + '.npy'
                        expected = torch.from_numpy(np.load(root / 'prior_fixtures' / filename, allow_pickle=False)).cuda()
                        assert torch.allclose(value, expected, atol=1e-4, rtol=1e-4), 'CPU/CUDA prior fixture differs: ' + name
                        fixture_rows.append({'name': name, 'maximum_absolute_error': float((value-expected).abs().max())})
                        np.save(out / 'prior_cuda_fixture' / filename, value.cpu().numpy(), allow_pickle=False)
                    counts['prior_fixture'] += 1
                    write(out / 'prior_CPU_CUDA_fixture.json', {'rows': fixture_rows, 'seed': 0,
                        'atol': 1e-4, 'rtol': 1e-4, 'photographic_inputs': 0, 'optimizer_updates': 0,
                        'custom_CUDA_kernel_equivalence_asserted': False})
                    del native_rgb, features, actuals, value, expected
                for begin in preview_indices:
                    b = batch(begin); pred = torch.where(b['mask'].bool(), candidate(b['x']), b['x'])
                    old = base_raw(begin)
                    assert torch.equal(pred, old), 'Initial100 TRAIN CUDA parity failed'
                for c in p['native_development']:
                    camera = pixels(root / c['input']); mask = pixels(root / c['observed'], 'L') > 0; x = tensor(camera)
                    pred = torch.where(torch.from_numpy(mask).cuda()[None, None], candidate(x), x)[0].permute(1, 2, 0).cpu().numpy()
                    old = np.load(out / 'baseline/native' / (c['id'] + '.npy'), allow_pickle=False)
                    assert np.array_equal(pred, old), 'Initial24 native CUDA parity failed'
            timings = []; forward_timings = []
            # Stage-gating uses true derivatives. B's conditioning is expected to be
            # initially disconnected by zero fusion; its fusion/current gradients are required now.
            for begin in preflight_indices:
                clock(); assert time.monotonic()-preflight_start < BUDGETS['preflight_seconds'], 'Preflight5-minute stop'
                b = batch(begin)
                with torch.no_grad():
                    t = time.monotonic()
                    candidate(b['x']).cpu().numpy()
                    torch.cuda.synchronize(); forward_timings.append(time.monotonic()-t)
                t = time.monotonic()
                pred = torch.where(b['mask'].bool(), candidate(b['x']), b['x'])
                loss, terms = objective(b, pred, base_raw(begin))
                grads = torch.autograd.grad(loss, params, allow_unused=False)
                counts['gradient_queries'] += 1
                norms = [float(g.detach().double().norm()) for g in grads]
                assert all(torch.isfinite(g).all() for g in grads)
                required = [n.startswith('fusions.') or n.startswith('current.') for n in names] if route == 'B' else [True]*len(names)
                if not all(norm > 0 for norm, active in zip(norms, required) if active):
                    raise ArmStop('Nonzero initial current/fusion derivatives required; no optimizer')
                if route == 'B':
                    assert all(norm == 0 for n, norm in zip(names, norms) if n.startswith(('style_conditioner.', 'spatial_conditioners.')))
                torch.cuda.synchronize(); timings.append(time.monotonic()-t)
                preflight_rows.append({'route': route, 'reference': cases[begin]['reference_id'], 'names': names,
                    'gradient_norms': norms, 'seconds': timings[-1], 'terms': terms})
                del grads, pred, loss
            projection = max(timings) * 250 * 1.5
            snapshot_projection = baseline['seconds'] + 805 * max(forward_timings) * 1.5
            write(out / ('preflight_' + route + '.json'), {'rows': [r for r in preflight_rows if r['route'] == route],
                'fit_projection_seconds': projection, 'fit_cap_seconds': BUDGETS['arm_fit_seconds'],
                'inference_reference_batch_seconds': forward_timings,
                'snapshot_projection_seconds': snapshot_projection, 'snapshot_cap_seconds': BUDGETS['snapshot_seconds'],
                'conditioning_initial_zero_expected': route == 'B', 'optimizer_updates': 0})
            assert projection <= BUDGETS['arm_fit_seconds'], 'Measured fit timing stop before optimizer'
            assert snapshot_projection <= BUDGETS['snapshot_seconds'], 'Measured snapshot timing stop before optimizer'
            frozen(); del candidate; candidate = None; torch.cuda.empty_cache()
        timing = {label: read(out / ('preflight_' + label + '.json')) for label in ['A', 'B']}
        total_projection = time.monotonic()-start + sum(t['fit_projection_seconds'] + t['snapshot_projection_seconds'] for t in timing.values())
        total_projection += timing['B']['snapshot_projection_seconds'] * (44/805)
        write(out / 'timing_projection.json', {'projected_worker_seconds': total_projection,
            'worker_cap_seconds': BUDGETS['worker_seconds'], 'optimizer_updates': 0})
        assert total_projection <= BUDGETS['worker_seconds'], 'Total measured timing stop before optimizer'
        write(out / 'preflight_complete.json', {'complete': True, 'initial_TRAIN_cases_each': 100, 'initial_native_each': 24,
            'gradient_queries': counts['gradient_queries'], 'optimizer_updates': 0,
            'source_and_CPU_GPU_prior_fixtures_note': 'Fresh CUDA model output parity and saved pure-PyTorch CPU prior fixture agreement verified; upstream custom CUDA kernel equivalence is not asserted.'})

        # Independent route starts. A quality stop does not resume A; B starts at
        # its separately declared original initializer, not at any failed A state.
        for route in ['A', 'B']:
            progress.update({'route': route, 'stage': 'fitting', 'route_updates': 0})
            candidate = create(route, learning=True)
            names = candidate.learning_names(); params = candidate.learning_parameters()
            frozen_state = state_hash({n: v for n, v in candidate.state_dict().items() if n not in names})
            buffer_state = buffer_hash(candidate)
            optimizer = torch.optim.Adam(params, lr=p['optimizer']['lr'], betas=(.9, .999), eps=1e-8, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            fit_start = time.monotonic(); history = []; gradient_after_first = []
            try:
                for update, starts in enumerate(p['schedule'], 1):
                    clock(); assert time.monotonic()-fit_start < BUDGETS['arm_fit_seconds'], 'Per-route10-minute fitting stop'
                    optimizer.zero_grad(set_to_none=True); objective_mean = 0.
                    for begin in starts:
                        b = batch(begin)
                        pred = torch.where(b['mask'].bool(), candidate(b['x']), b['x'])
                        loss, terms = objective(b, pred, base_raw(begin))
                        (loss / 5).backward(); counts['backwards'] += 1; objective_mean += terms['objective']/5
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
                    assert np.isfinite(grad_norm) and grad_norm > 0
                    optimizer.step(); scheduler.step(); counts['optimizer_updates'] += 1
                    progress['route_updates'] = update
                    history.append({'update': update, 'objective': objective_mean, 'gradient_norm_before_clip': grad_norm, 'reference_starts': starts})
                    if update == 1 and route == 'B':
                        # Before update2 require propagation into every new conditioning tensor.
                        sums = np.zeros(len(params), dtype=np.float64)
                        for begin in preflight_indices:
                            b = batch(begin); pred = torch.where(b['mask'].bool(), candidate(b['x']), b['x'])
                            loss, _ = objective(b, pred, base_raw(begin))
                            grads = torch.autograd.grad(loss, params, allow_unused=False); counts['gradient_queries'] += 1
                            assert all(torch.isfinite(g).all() for g in grads)
                            norms = [float(g.detach().double().norm()) for g in grads]; sums += norms
                            gradient_after_first.append({'reference': cases[begin]['reference_id'], 'norms': norms})
                            del grads, pred, loss
                        write(out / 'B_conditioning_after_update1.json', {'names': names, 'rows': gradient_after_first,
                            'summed_norms': sums.tolist(), 'all_nonzero': bool((sums > 0).all()), 'optimizer_updates': 1})
                        if not (sums > 0).all():
                            raise ArmStop('New B conditioning does not receive nonzero finite gradients after first fusion update')
                    if update == 1 or update % 10 == 0:
                        print(json.dumps({'route': route, 'update': update, 'of': 50, 'objective': objective_mean}), flush=True)
                    frozen()
                progress['stage'] = 'snapshot50'
                save_state(route + '50')
                saved = metrics(route + '50', candidate)
                gates = {stage: capacity(baseline['groups'][stage], saved['groups'][stage], .01) for stage in ['raw', 'png']}
                outcome = {'route': route, 'updates': 50, 'gates': gates, 'capacity_pass': all(g['pass'] for g in gates.values()),
                    'epochs_qualified': False, 'model_qualified': False, 'native_visual_review_pending': True}
                write(out / ('outcome_' + route + '.json'), outcome)
                if route == 'B':
                    # Preserve saved bank-enabled outputs even on a negative gate.
                    metrics('B50_bank_disabled', candidate, preview_indices, bank_disabled=True)
                completed.append(outcome)
                if not outcome['capacity_pass']:
                    write(out / ('quality_stop_' + route + '.json'), {'type': 'quality_gate_failure',
                        'message': 'Early structure/preservation requirement failed; retain stop', 'outcome': outcome,
                        'automatic_extra_updates': 0, 'failed_recipe_resume_allowed': False})
            except ArmStop as exc:
                save_state(route + '_stopped')
                write(out / ('learning_stop_' + route + '.json'), {'type': type(exc).__name__, 'message': str(exc),
                    'updates': progress['route_updates'], 'history': history, 'failed_state_resume_allowed': False})
                completed.append({'route': route, 'updates': progress['route_updates'], 'capacity_pass': False,
                                  'learning_stop': str(exc), 'model_qualified': False})
            write(out / ('history_' + route + '.json'), {'rows': history, 'route': route, 'updates': progress['route_updates']})
            frozen(); optimizer = None; scheduler = None; del candidate; candidate = None; torch.cuda.empty_cache()
        progress['stage'] = 'finite_comparison_complete'
        verify(root, pin)
        all_pass = len(completed) == 2 and all(r['capacity_pass'] for r in completed)
        write(out / 'results.json', {'complete': True, 'execution_finished': True, 'routes': completed, 'counts': counts,
            'protocol_sha256': pin, 'seconds': time.monotonic()-start, 'goal_complete': False, 'app_promotion': False,
            'paired_TRAIN_capacity_not_DEV_or_final_qualification': True, 'full_epochs': 0,
            'external_frozen_prior_unchanged': True, 'useful_native_visual_review_pending': True,
            'all_routes_capacity_pass': all_pass, 'automatic_follow_on': False})
        return 0 if all_pass else 2
    except BaseException as exc:
        if candidate is not None and not (out / ('state_' + str(route) + '_failure')).exists():
            try:
                save_state(str(route) + '_failure')
            except BaseException as state_exc:
                write(out / 'failure_state_export_error.json', {'type': type(state_exc).__name__, 'message': str(state_exc)})
        write(out / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
            'progress': progress, 'seconds': time.monotonic()-start, 'completed_routes': completed,
            'original_checkpoints_and_failures_preserved': True, 'training_success_not_implied': True})
        raise


def export(root, p, pin):
    vm(root)
    assert (root / 'outputs/bank_comparison_v1').exists()
    stamp = time.monotonic()
    # Export a self-contained research/migration packet; generated prior/source,
    # inputs, scopes, full state and negative gates are retained together.
    files = sorted(f for f in root.rglob('*') if f.is_file())
    total = sum(f.stat().st_size for f in files)
    assert total <= BUDGETS['return_uncompressed_bytes'] + sum((root/n).stat().st_size for n in p['assets_sha256'])
    assert shutil.disk_usage(root).free >= total + BUDGETS['disk_reserve_bytes'], 'Export reserve stop'
    archive = Path.home() / (STEM + '-results.tar.gz')
    sidecar = Path(str(archive) + '.sha256'); receipt = Path.home() / (STEM + '-export.json')
    assert not any(f.exists() for f in [archive, sidecar, receipt]), 'Never overwrite historical exports'
    manifest = {f.relative_to(root).as_posix(): sha(f) for f in files}
    write(root / 'export_manifest.json', {'protocol_sha256': pin, 'files': manifest, 'uncompressed_bytes': total})
    temp = Path(str(archive) + '.part')
    assert not temp.exists()
    with tarfile.open(temp, 'w:gz', compresslevel=1) as tar:
        for f in files + [root / 'export_manifest.json']:
            assert time.monotonic()-stamp < BUDGETS['export_seconds'], 'Export10-minute stop'
            assert not f.is_symlink() and sha(f) == (manifest.get(f.relative_to(root).as_posix()) or sha(root/'export_manifest.json'))
            tar.add(f, arcname=NAME + '/' + f.relative_to(root).as_posix(), recursive=False)
    temp.rename(archive)
    digest = sha(archive)
    sidecar.write_text(digest + '  ' + archive.name + '\n', encoding='ascii')
    out = root / 'outputs/bank_comparison_v1'
    complete = (out / 'results.json').exists()
    write(receipt, {'complete': True, 'archive_sha256': digest, 'bytes': archive.stat().st_size,
        'seconds': time.monotonic()-stamp, 'protocol_sha256': pin, 'execution_results_present': complete,
        'failure_present': (out/'failure.json').exists(), 'training_success_not_implied': True,
        'quality_stops': sorted(f.name for f in out.glob('*stop*.json')),
        'optimizer_updates': read(out/'results.json')['counts']['optimizer_updates'] if complete else read(out/'failure.json')['progress']['counts']['optimizer_updates']})
    print(json.dumps(read(receipt)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--verify-transfer', action='store_true')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    p = verify(root, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps({'complete': True, 'assets': len(p['assets_sha256']), 'TRAIN_cases': 3905,
            'native_DEV_cases': 24, 'optimizer_updates': 0, 'neural_or_gradient_calls': 0}))
    elif args.export:
        export(root, p, args.protocol_sha)
    else:
        sys.exit(run(root, p, args.protocol_sha))

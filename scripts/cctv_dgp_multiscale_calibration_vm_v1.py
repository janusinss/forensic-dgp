"""Manual L4-only 12 independent first-step arms; exact retained states and outputs."""
import argparse
import ast
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cctv_dgp_multiscale_calibration_contract_v1 import NAME, STEM, STATE, INITIAL, RECOGNIZER, BUDGETS, NORMALIZERS, WEIGHTS, SIGNALS, read, write, sha, verify


def scope(root):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Manual existing Linux VM only'
    assert root == (Path.home() / 'forensic-dgp' / NAME).resolve()


def run(root, p, pin):
    scope(root); assert os.environ.get('TMUX'), 'Manual tmux required'
    assert not (root / 'outputs').exists(), 'Preserve previous or partial runs; no resume'
    out = root / 'outputs'; out.mkdir(); start = time.monotonic()
    candidate = optimizer = scheduler = save_state = None; arm_id = None
    progress = {'optimizer_updates': 0, 'backwards': 0, 'gradient_queries': 0, 'completed_epochs': 0,
                'fit_exposures': 0, 'completed_arms': [], 'optimizer_constructed': False}
    counts = {'original': 0, 'candidate': 0, 'recognizer': 0}
    try:
        assert shutil.disk_usage(root).free >= BUDGETS['minimum_free_disk_bytes'], 'Need7GiB after install; no neural calls'
        tree = ast.parse((root / 'frozen_definitions.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_vm')
        ns = {'sys': sys, 'platform': platform, 'Path': Path, 'os': os, 'subprocess': subprocess, 'urllib': __import__('urllib')}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-hardware-idle>', 'exec'), ns)
        ns['require_vm'](root, idle=True)
        import hashlib
        import importlib.metadata
        import numpy as np
        import torch
        from PIL import Image
        from cctv_dgp_multiscale_calibration_model_v1 import MultiscaleCalibrationDGP, FULL
        from cctv_dgp_head4_lossless_v1 import pack, unpack
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, buffer_hash
        from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
        from frozen_capacity_contract import feature_support, capacity, exported_pixel_metrics, detail_metric
        from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups
        random.seed(501050); np.random.seed(501050); torch.manual_seed(501050); torch.cuda.manual_seed_all(501050)
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        original, _ = load_frozen_dgp_restorer(root / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cuda')
        identity = FixedObservedIdentity(root / 'weights/w600k_r50.onnx', 'cuda')
        assert state_hash(original.net) == STATE and state_hash(identity) == RECOGNIZER
        for model, label in [(original.net, 'original'), (identity.encoder, 'recognizer')]:
            model.register_forward_hook(lambda *_a, label=label: counts.__setitem__(label, counts[label] + 1))
        refs = {r['id']: r for r in p['references']}; cases = p['cases']
        initial = MultiscaleCalibrationDGP(original.net).cuda().eval().requires_grad_(False)
        assert state_hash(initial) == INITIAL
        torch.save({n: v.detach().cpu().clone() for n, v in initial.state_dict().items()}, out / 'initial_candidate.pth')
        frozen_full = state_hash({n: v for n, v in initial.state_dict().items() if n not in FULL})
        frozen_partitions = {q: state_hash({n: v for n, v in initial.state_dict().items() if n not in initial.selected_names(q)})
                             for q in ['deep3', 'decoder15']}
        buffers = buffer_hash(initial)
        del initial

        def create(partition):
            m = MultiscaleCalibrationDGP(original.net).cuda().eval(); m.enable_vm_learning(root, partition)
            assert state_hash(m) == INITIAL
            m.register_forward_hook(lambda *_a: counts.__setitem__('candidate', counts['candidate'] + 1))
            return m

        def retained_bytes():
            return sum(f.stat().st_size for f in out.rglob('*') if f.is_file())

        def frozen():
            assert state_hash(original.net) == STATE and state_hash(identity) == RECOGNIZER
            assert state_hash({n: v for n, v in candidate.state_dict().items() if n not in candidate.selected_names(partition)}) == frozen_partitions[partition]
            assert buffer_hash(candidate) == buffers
            assert all(not m.training for model in [original, candidate, identity] for m in model.modules())
            assert all(v.grad is None and not v.requires_grad for m in [original, identity] for v in m.parameters())
            assert all(v.grad is None and not v.requires_grad for n, v in candidate.named_parameters() if n not in candidate.selected_names(partition))

        def clock():
            torch.cuda.synchronize()
            assert time.monotonic() - start < BUDGETS['worker_seconds'], 'Worker1800s stop'
            assert shutil.disk_usage(root).free >= BUDGETS['disk_reserve_bytes'], 'Disk1GiB reserve stop'
            assert torch.cuda.max_memory_allocated() <= BUDGETS['peak_vram_bytes'], 'VRAM20GiB stop'
            assert retained_bytes() <= BUDGETS['return_uncompressed_bytes'], 'Output2.5GiB stop'
            for label in counts:
                assert counts[label] <= BUDGETS[label + '_forward_calls'], label + ' forward cap'

        def pixels(path, mode='RGB'):
            with Image.open(path) as im:
                assert im.size == (256, 256); return np.asarray(im.convert(mode)).copy()

        def rgb(a):
            return torch.from_numpy(a.astype(np.float32) / np.float32(255)).cuda().permute(2, 0, 1)[None]

        def load_batch(begin):
            cs = cases[begin:begin+5]; cameras = []; targets = []; masks = []; features = []; grids = []
            for c in cs:
                r = refs[c['source_person_or_reference']]
                cameras.append(pixels(root / c['input'])); targets.append(pixels(root / r['target']))
                mask = pixels(root / r['observed'], 'L') > 0; masks.append(mask)
                features.append(feature_support(mask, c['landmarks5_canvas_xy'])); grids.append(grid112(r['matrix112']))
            return {'cases': cs, 'camera': cameras, 'target8': targets, 'mask8': masks, 'feature8': features,
                'x': torch.cat([rgb(a) for a in cameras]), 'target': torch.cat([rgb(a) for a in targets]),
                'mask': torch.from_numpy(np.stack(masks).astype(np.float32)).cuda()[:, None],
                'feature': torch.from_numpy(np.stack(features).astype(np.float32)).cuda()[:, None],
                'grid': torch.from_numpy(np.stack(grids)).cuda()}

        def get_base(begin):
            a, _ = unpack(out / 'baseline/packs' / f'b{begin//5:02d}.npz', [c['id'] for c in cases[begin:begin+5]])
            return torch.from_numpy(a).cuda().permute(0, 3, 1, 2)

        write(out / 'environment.json', {'python': sys.version, 'GPU': torch.cuda.get_device_name(0), 'host': platform.node(),
            'CUDA': torch.version.cuda, 'packages': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'numpy', 'Pillow', 'scipy', 'scikit-image', 'onnx', 'onnx2torch']},
            'AMP': False, 'TF32': False, 'deterministic_cudnn': True,
            'normalization': 'unchanged stored evaluation statistics', 'bitwise_future_resume_guaranteed': False})

        def snapshot(label, baseline=False):
            stamp = time.monotonic(); folder = out / label; folder.mkdir(exist_ok=True)
            (folder / 'packs').mkdir(); (folder / 'previews').mkdir(); (folder / 'native').mkdir(); rows = []; native = []
            frozen()
            with torch.no_grad():
                for begin in range(0, 100, 5):
                    clock(); assert time.monotonic() - stamp < BUDGETS['snapshot_seconds'], 'Snapshot120s stop'
                    b = load_batch(begin); ids = [c['id'] for c in b['cases']]
                    pred = torch.where(b['mask'].bool(), candidate(b['x']), b['x'])
                    a = pred.permute(0, 2, 3, 1).cpu().numpy().copy()
                    if baseline:
                        zero = torch.where(b['mask'].bool(), original.net(b['x']), b['x'])
                        assert torch.equal(zero, pred), 'Initial100 TRAIN parity failed'; base = a
                    else:
                        base, _ = unpack(out / 'baseline/packs' / f'b{begin//5:02d}.npz', ids)
                    pngs = [deliver(raw, camera, mask) for raw, camera, mask in zip(a, b['camera'], b['mask8'])]
                    vectors = identity.embedding(torch.cat([pred, torch.cat([rgb(q) for q in pngs]), b['target']]),
                        torch.cat([b['mask']] * 3), torch.cat([b['grid']] * 3)).cpu().numpy().copy()
                    em = np.stack([vectors[:5], vectors[5:10], vectors[10:15]], axis=1)
                    pack(folder / 'packs' / f'b{begin//5:02d}.npz', a, ids, em, baseline=None if baseline else base)
                    for i, c in enumerate(b['cases']):
                        raw = a[i]; png = pngs[i]; target = b['target8'][i]; mask = b['mask8'][i]
                        mraw, mpng, shift = mean_only(raw, base[i], b['camera'][i], mask)
                        rm = pixel_metrics(raw, target, mask); pm = exported_pixel_metrics(png, target, mask)
                        rm.update({'ArcFace_observed_fixed': float(em[i, 0] @ em[i, 2]),
                            'landmark_high_frequency_MSE': detail_float(raw, target, b['feature8'][i]),
                            'constant_mean_shift_only_MSE': pixel_metrics(mraw, target, mask)['MSE']})
                        pm.update({'ArcFace_observed_fixed': float(em[i, 1] @ em[i, 2]),
                            'landmark_high_frequency_MSE': detail_metric(png, target, b['feature8'][i]),
                            'constant_mean_shift_only_MSE': exported_pixel_metrics(mpng, target, mask)['MSE']})
                        rows.append({'id': c['id'], 'source': c['source'], 'profile': c['profile'], 'raw': rm, 'png': pm,
                            'postclip_mean_RGB_shift': shift.tolist(), 'raw_RGB_float32_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                            'PNG_RGB_sha256': hashlib.sha256(png.tobytes()).hexdigest()})
                        Image.fromarray(png).save(folder / 'previews' / (c['id'] + '.png'))
                for c in p['native_development']:
                    clock(); assert time.monotonic() - stamp < BUDGETS['snapshot_seconds']
                    camera = pixels(root / c['input']); mask = pixels(root / c['observed'], 'L') > 0
                    x = rgb(camera); mm = torch.from_numpy(mask).cuda()[None, None]
                    pred = torch.where(mm, candidate(x), x)
                    if baseline:
                        zero = torch.where(mm, original.net(x), x); assert torch.equal(pred, zero), 'Initial24 native parity failed'
                    raw = pred[0].permute(1, 2, 0).cpu().numpy().copy(); png = deliver(raw, camera, mask)
                    with (folder / 'native' / (c['id'] + '.npy')).open('xb') as f:
                        np.save(f, raw, allow_pickle=False)
                    Image.fromarray(png).save(folder / 'native' / (c['id'] + '.png'))
                    assert np.array_equal(png[~mask], camera[~mask])
                    native.append({'id': c['id'], 'role': 'development', 'evidence': 'unpaired native CCTV; no clean reference metrics',
                        'raw_RGB_float32_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                        'PNG_RGB_sha256': hashlib.sha256(png.tobytes()).hexdigest(), 'outside_mask_unchanged': True})
            result = {'complete': True, 'label': label, 'rows': rows, 'native_unpaired_rows': native,
                'groups': {stage: review_groups(rows, stage) for stage in ['raw', 'png']},
                'candidate_state': state_hash(candidate), 'seconds': time.monotonic() - stamp,
                'paired_TRAIN_only': True, 'native_quality_claim': False, 'model_qualification': False}
            write(folder / 'metrics.json', result); frozen(); clock(); return result

        partition = 'decoder15'; candidate = create(partition)
        baseline = snapshot('baseline', baseline=True)
        # Gradients are genuine float32 parameter derivatives, never quantized.
        parameters = candidate.selected_parameters(partition); layout = []; offset = 0
        for n, value in zip(candidate.selected_names(partition), parameters):
            layout.append({'name': n, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()}); offset += value.numel()
        assert offset == 609219 and layout[2]['end'] == 147456
        initial_bytes = retained_bytes(); gradient_bytes = 20 * 14 * 609219 * 4 + 256
        initial_state_bytes = (out / 'initial_candidate.pth').stat().st_size
        accumulated_gradient_bytes = (6 * 147456 + 6 * 609219) * 4
        projected = int(gradient_bytes + (initial_bytes - initial_state_bytes) * 13 * 1.35 +
                        initial_state_bytes + 12 * 32 * 1024**2 + accumulated_gradient_bytes + 1024**2)
        write(out / 'storage_projection.json', {'initial_bytes': initial_bytes, 'gradient_bytes': gradient_bytes,
            'snapshots': 13, 'factor': 1.35, 'state_reserve_each_bytes': 32 * 1024**2,
            'initial_state_bytes': initial_state_bytes, 'accumulated_gradient_bytes': accumulated_gradient_bytes,
            'projected_bytes': projected, 'cap_bytes': BUDGETS['return_uncompressed_bytes'], 'optimizer_updates': 0})
        assert projected <= BUDGETS['return_uncompressed_bytes'], 'Measured projection exceeds2.5GiB; no optimizer'
        assert shutil.disk_usage(root).free >= 2 * projected - initial_bytes + BUDGETS['disk_reserve_bytes'], 'Reserve output and export before optimizer'

        def training_terms(b, pred, base):
            t = pixel_and_structure_losses(pred, b['target'], b['mask'], b['feature'])
            with torch.no_grad():
                bt = pixel_and_structure_losses(base, b['target'], b['mask'], b['feature'])
            vec = identity.embedding(torch.cat([pred, base.detach(), b['target']]), torch.cat([b['mask']] * 3), torch.cat([b['grid']] * 3))
            truth = vec[10:15].detach().double(); arc = 1 - (vec[:5].double() * truth).sum(1)
            barc = 1 - (vec[5:10].detach().double() * truth).sum(1)
            raw = [t['MSE'].mean(), t['SSIM_loss'].mean(), arc.mean(), t['landmark_structure'][1:].mean()]
            components = [v / normalizer for v, normalizer in zip(raw, NORMALIZERS)]
            barrier = 5 * (torch.relu(t['MSE'] - bt['MSE'] - 1e-12).mean() / NORMALIZERS[0] +
                torch.relu(t['SSIM_loss'] - bt['SSIM_loss'] - 1e-6).mean() / NORMALIZERS[1] + torch.relu(arc - barc - 1e-6).mean() / NORMALIZERS[2])
            mean_rgb = (pred * b['mask']).sum((0, 2, 3)) / b['mask'].sum()
            signals = raw + [t['MSE'][0], t['SSIM_loss'][0], arc[0], t['MSE'][1], t['SSIM_loss'][1], arc[1], t['landmark_structure'][1]] + list(mean_rgb)
            return components, barrier, signals

        gradient_start = time.monotonic(); gradient_rows = []
        matrix = np.lib.format.open_memmap(out / 'initial_gradients.npy', mode='w+', dtype=np.float32, shape=(20, 14, 609219))
        part_sums = np.zeros(15, dtype=np.float64)
        for batch, begin in enumerate(range(0, 100, 5)):
            clock(); assert time.monotonic() - gradient_start < BUDGETS['gradient_seconds'], 'Gradient600s stop'
            b = load_batch(begin); base = get_base(begin); pred = torch.where(b['mask'].bool(), candidate(b['x']), b['x'])
            assert torch.equal(pred, base), 'Fresh preflight initial output changed'
            components, barrier, signals = training_terms(b, pred, base); norms = []
            assert len(signals) == 14 and float(barrier.detach()) == 0
            for index, term in enumerate(signals):
                gg = torch.autograd.grad(term, parameters, retain_graph=index < 13, allow_unused=False)
                progress['gradient_queries'] += 1
                assert all(torch.isfinite(g).all() for g in gg)
                row = np.concatenate([g.detach().cpu().numpy().reshape(-1) for g in gg]); matrix[batch, index] = row
                nn = [float(g.detach().double().norm()) for g in gg]; norms.append(nn)
                if index < 4:
                    part_sums += np.square(nn)
            gradient_rows.append({'batch': batch, 'case_ids': [c['id'] for c in b['cases']],
                'losses': [float(v.detach()) for v in signals], 'part_L2_by_signal': norms})
            matrix.flush(); print({'gradient_batch': batch + 1, 'of': 20, 'gradient_queries': progress['gradient_queries']}, flush=True)
            del pred, components, barrier, signals, gg
        matrix.flush(); del matrix
        write(out / 'gradient_preflight.json', {'complete': True, 'queries': 280, 'signals': SIGNALS, 'layout': layout,
            'rows': gradient_rows, 'all15_improvement_gradient_L2': np.sqrt(part_sums).tolist(),
            'gradient_sha256': sha(out / 'initial_gradients.npy'), 'optimizer_updates': 0})
        assert progress['gradient_queries'] == 280 and (part_sums > 0).all(), 'All15 decoder tensors need finite nonzero improvement gradients before optimizer'
        frozen(); del candidate; candidate = None
        arm_results = []
        for arm in p['arms']:
            clock(); arm_id = arm['id']; partition = arm['partition']; candidate = create(partition)
            parameters = candidate.selected_parameters(partition); folder = out / arm_id; folder.mkdir()
            optimizer = torch.optim.Adam(parameters, lr=arm['lr'], betas=(.9, .999), eps=1e-8, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)
            progress['optimizer_constructed'] = True; progress['current_arm_updates'] = 0

            def save(path, gate_status):
                rr = np.random.get_state()
                torch.save({'format': 'multiscale-calibration-portable-full-state-v1', 'protocol_sha256': pin,
                    'arm': arm, 'model': {n: v.detach().cpu().clone() for n, v in candidate.state_dict().items()},
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'python_rng': random.getstate(), 'numpy_rng': (rr[0], torch.from_numpy(rr[1].astype(np.int64)), *rr[2:]),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'completed_updates': progress['current_arm_updates'], 'completed_epochs': 0,
                    'fitting_reference_exposures': 10 if progress['current_arm_updates'] else len(step_rows),
                    'source_epoch_reference_count': 781, 'next_schedule_index': len(step_rows),
                    'optimizer_parameter_names': candidate.selected_names(partition), 'schedule_sha256': sha(root / 'protocol.json'),
                    'original_checkpoint_sha256': p['original_checkpoint_sha256'], 'gate_status': gate_status,
                    'automatic_resume': False, 'failed_gate_must_not_resume': True, 'bitwise_resume_guaranteed': False}, path)
            save_state = save; stamp = time.monotonic(); optimizer.zero_grad(set_to_none=True); step_rows = []
            for begin in p['pools'][arm['pool']]:
                clock(); assert time.monotonic() - stamp < BUDGETS['arm_fit_seconds'], 'Arm120s stop'
                b = load_batch(begin); base = get_base(begin)
                pred = torch.where(b['mask'].bool(), candidate(b['x']), b['x'])
                assert torch.equal(pred, base), 'One-step accumulation must start at original initializer'
                components, barrier, _ = training_terms(b, pred, base)
                objective = sum(w * term for w, term in zip(WEIGHTS, components)) + barrier
                assert torch.isfinite(objective)
                (objective / 10).backward(); progress['backwards'] += 1; progress['fit_exposures'] += 5
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters)
                step_rows.append({'begin': begin, 'case_ids': [c['id'] for c in b['cases']],
                    'objective': float(objective.detach()), 'normalized_components': [float(v.detach()) for v in components],
                    'barrier': float(barrier.detach())})
                del pred, components, barrier, objective
            frozen()
            with (folder / 'accumulated_gradient.npy').open('xb') as stream:
                np.save(stream, np.concatenate([v.grad.detach().cpu().numpy().reshape(-1) for v in parameters]), allow_pickle=False)
            gradnorm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            optimizer.step(); scheduler.step(); progress['optimizer_updates'] += 1; progress['current_arm_updates'] = 1
            optimizer.zero_grad(set_to_none=True)
            save(folder / 'training_state.pt', 'finite_calibration_evaluation_pending; no resume')
            write(folder / 'step_receipt.json', {'arm': arm, 'rows': step_rows, 'gradient_L2_before_clip': gradnorm,
                'actual_optimizer_updates': 1, 'backwards': 10, 'fit_exposures': 50, 'initial_candidate_state': INITIAL,
                'candidate_state': state_hash(candidate), 'seconds': time.monotonic() - stamp})
            current = snapshot(arm_id)
            gates = {stage: capacity(baseline['groups'][stage], current['groups'][stage], .01) for stage in ['raw', 'png']}
            summary = {'arm': arm, 'comparisons': gates, 'sampled_capacity_pass': all(g['pass'] for g in gates.values()),
                'full_3905_TRAIN_not_tested': True, 'one_step_does_not_qualify_model': True,
                'native_visual_review_pending': True, 'failed_state_resume_allowed': False}
            write(folder / 'quality_gate.json', summary); arm_results.append(summary)
            progress['completed_arms'].append(arm_id)
            print({'arm': arm_id, 'of': 12, 'sampled_capacity_pass': summary['sampled_capacity_pass'], 'model_qualification': False}, flush=True)
            frozen(); clock(); save_state = None; del candidate, optimizer, scheduler; candidate = optimizer = scheduler = None
        assert counts == {'original': 44, 'candidate': 712, 'recognizer': 400}
        assert progress['optimizer_updates'] == 12 and progress['backwards'] == 120 and progress['fit_exposures'] == 600
        assert state_hash(original.net) == STATE and state_hash(identity) == RECOGNIZER
        result = {'complete': True, 'protocol_sha256': pin, 'progress': progress, 'forward_counts': counts,
            'arms': arm_results, 'original_state': STATE, 'recognizer_state': RECOGNIZER, 'frozen_full_state_hash': frozen_full,
            'frozen_buffers_hash': buffers, 'frozen_partition_hashes': frozen_partitions,
            'optimizer_updates': 12, 'gradient_queries': 280, 'completed_epochs': 0,
            'initial_exact_TRAIN_cases': 100, 'initial_exact_native_cases': 24, 'native_clean_reference_metrics': False,
            'model_qualification': False, 'app_promotion': False, 'goal_complete': False,
            'automatic_follow_on': False, 'seconds': time.monotonic() - start}
        write(out / 'results.json', result); clock()
        print({'complete': True, 'independent_one_step_arms': 12, 'qualified_model': False}, flush=True)
    except BaseException:
        if save_state is not None:
            try:
                save_state(out / 'stopped_training_state.pt', 'failed_or_partial_stop; no resume')
            except Exception as e:
                write(out / 'state_export_failure.json', {'error': repr(e)})
        write(out / 'failure.json', {'complete': False, 'traceback': traceback.format_exc(), 'arm': arm_id,
            'progress': progress, 'forward_counts': counts, 'training_success_not_implied': True,
            'seconds': time.monotonic() - start}); raise


def export(root, p, pin):
    scope(root); stamp = time.monotonic(); destination = Path.home() / (STEM + '-results.tar.gz')
    partial = Path(str(destination) + '.partial')
    assert not destination.exists() and not partial.exists(), 'Preserve exports'
    files = [f for f in (root / 'outputs').rglob('*') if f.is_file()]
    files += [root / n for n in ['protocol.json', 'trainer.log', 'trainer_exit_code.txt', 'supervisor_receipt.json'] if (root / n).exists()]
    files += [root / n for n in p['assets_sha256'] if n.endswith('.py')]
    total = sum(f.stat().st_size for f in files)
    assert total <= BUDGETS['return_uncompressed_bytes'] and shutil.disk_usage(root).free >= total + BUDGETS['disk_reserve_bytes']
    write(root / 'export_manifest.json', {'complete': True, 'protocol_sha256': pin,
        'files_sha256': {f.relative_to(root).as_posix(): sha(f) for f in files}, 'uncompressed_bytes': total,
        'training_success_not_implied': True})
    with partial.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as tar:
            for f in files + [root / 'export_manifest.json']:
                assert time.monotonic() - stamp < BUDGETS['export_seconds'], 'Export600s stop'
                assert shutil.disk_usage(root).free >= BUDGETS['disk_reserve_bytes']
                tar.add(f, arcname=f.relative_to(root).as_posix(), recursive=False)
    partial.rename(destination); digest = sha(destination)
    with Path(str(destination) + '.sha256').open('x', encoding='ascii') as f:
        f.write(digest + '  ' + destination.name + '\n')
    result = {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size,
        'seconds': time.monotonic() - stamp, 'training_success_not_implied': True,
        'run_results_present': (root / 'outputs/results.json').exists(), 'failure_present': (root / 'outputs/failure.json').exists()}
    write(Path.home() / (STEM + '-export.json'), result); print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); parser.add_argument('--protocol-sha', required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--verify-transfer', action='store_true'); g.add_argument('--run', action='store_true'); g.add_argument('--export', action='store_true')
    a = parser.parse_args(); root = a.root.resolve(); p = verify(root, a.protocol_sha)
    if a.verify_transfer:
        print({'complete': True, 'assets': len(p['assets_sha256']), 'TRAIN_cases': 100, 'unpaired_native_DEV': 24,
               'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0}, flush=True)
    elif a.export:
        export(root, p, a.protocol_sha)
    else:
        run(root, p, a.protocol_sha)

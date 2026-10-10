"""Finite manual L4 learning of V39's proven spatial path, on full existing TRAIN."""
import argparse
import ast
from pathlib import Path
import platform
import shutil
import signal
import sys
import tarfile
import time
import traceback
from types import MethodType, SimpleNamespace

NAME = 'cctv_dgp_spatial_fit_vm_v40'
STEM = 'cctv-dgp-spatial-fit-v40'


def scope(root):
    assert sys.platform == 'linux', 'Actual learning requires existing Linux VM; stop before neural imports'
    assert platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing VM only'
    assert Path(root).resolve() == (Path.home()/'forensic-dgp'/NAME).resolve(), 'Distinct V40 root only'


def contract(root):
    sys.path.insert(0, str(root))
    from cctv_dgp_spatial_fit_v40_contract import sha, read, write, validate_schedule
    return sha, read, write, validate_schedule


def verify(root, pin):
    sha, read, write, validate_schedule = contract(root)
    assert sha(root/'protocol.json') == pin
    p = read(root/'protocol.json')
    assert p['format'] == 'own-DGP-proven-spatial-path-full-TRAIN-v40'
    assert p['updates'] == 800 and p['snapshots'] == [0, 50, 800]
    assert p['decoder_parameters'] == 17952 and p['decoder_tensors'] == 57
    assert len(p['references']) == 781 and len(p['cases']) == 3905
    validate_schedule(p['cases'], read(root/'schedule.json')['batches'])
    for name, digest in p['assets_sha256'].items():
        path = (root/name).resolve()
        assert path.is_relative_to(root) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return p


def run(root, p, pin):
    scope(root); assert not (root/'outputs').exists(), 'Retain every historical/partial run; no resume or repeat'
    sha, read, write, _ = contract(root)
    started = time.monotonic(); out = root/'outputs'; out.mkdir()
    progress = {'optimizer_updates': 0, 'backwards': 0, 'optimizer_constructed': False, 'new_trained_checkpoint': False}
    candidate = None; snapshots = []; times = []; fit_started = None
    try:
        assert shutil.disk_usage(root).free >= 6*1024**3, 'Need6GiB free; this pilot never deletes files'
        # Only the original metadata/idle guard and filter functions are compiled.
        import os, subprocess, urllib.request
        tree = ast.parse((root/'frozen_definitions.py').read_text())
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_vm')
        guard_ns = {'sys': sys, 'platform': platform, 'Path': Path, 'os': os, 'subprocess': subprocess, 'urllib': __import__('urllib')}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), '<original-L4-guard>', 'exec'), guard_ns)
        guard_ns['require_vm'](root, idle=True)
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        from cctv_dgp_spatial_decoder_v40 import SpatialDGPCandidateV40
        from cctv_dgp_pilot import FixedObservedIdentity, state_hash, grid112
        from cctv_dgp_spatial_fit_v40_contract import erode, feature_support, exported_pixel_metrics, detail_metric, groups, capacity
        from cctv_dgp_degraded_objective_v24 import cohort_normalizers
        from cctv_dgp_batchmatched_identity_v26 import objective_terms
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        original, provenance = load_frozen_dgp_restorer(root/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cuda')
        seed = torch.load(root/'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
        candidate = SpatialDGPCandidateV40(original, seed); candidate.enable_vm_learning(root)
        identity = FixedObservedIdentity(root/'weights/w600k_r50.onnx', 'cuda')
        def states():
            return {'original': state_hash(original.net), 'decoder': state_hash(candidate.decoder),
                    'reference_decoder': state_hash(candidate.reference_decoder), 'recognizer': state_hash(identity)}
        before = states(); assert before == p['initial_states']
        parameters = list(candidate.decoder.parameters())
        assert len(parameters) == 57 and sum(v.numel() for v in parameters) == 17952
        assert all(v.requires_grad for v in parameters)
        assert not ({v.data_ptr() for v in parameters} & {v.data_ptr() for v in candidate.reference_decoder.parameters()})
        counts = {'original_DGP': 0, 'decoder': 0, 'reference_decoder': 0, 'recognizer': 0}
        for model, key in [(original.net, 'original_DGP'), (candidate.decoder, 'decoder'), (candidate.reference_decoder, 'reference_decoder'), (identity.encoder, 'recognizer')]:
            model.register_forward_hook(lambda *_args, key=key: counts.__setitem__(key, counts[key]+1))
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic()-started <= 4500, 'V40 worker4500s stop'
            assert torch.cuda.max_memory_allocated() <= 20*1024**3, 'Allocated VRAM20GiB stop'
        def frozen():
            actual = states()
            for key in ['original', 'reference_decoder', 'recognizer']: assert actual[key] == before[key], key
            for model in [original, candidate.reference_decoder, identity]:
                assert all(not v.requires_grad and v.grad is None for v in model.parameters())
            assert sha(root/'weights/dgp_v2.pth') == p['original_checkpoint_sha256']
        def pixels(path, mode='RGB'):
            with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()
        def canonical(a):
            return torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None].cuda()
        t = lambda a: torch.from_numpy(np.asarray(a).copy()).cuda()
        shared = {}; cache_start = time.monotonic(); cache_times = []
        for ref in p['references']:
            clock(); assert time.monotonic()-cache_start < 900
            target = pixels(root/ref['target']); mask = pixels(root/ref['observed'], 'L') > 0
            assert target.shape == (256, 256, 3) and mask.shape == (256, 256) and mask.any()
            item = {'target8': target, 'mask8': mask, 'target': canonical(target), 'mask': t(mask.astype(np.float32))[None, None],
                    'feature': t(feature_support(mask, ref['landmarks5'][0]).astype(np.float32))[None, None],
                    'interior': t(erode(mask, 6).astype(np.float32))[None, None], 'valid7': t(erode(mask, 3).astype(np.float32))[None, None],
                    'grid': t(grid112(ref['matrix112']))[None]}
            with torch.no_grad(): item['truth'] = identity.embedding(item['target'], item['mask'], item['grid']).detach().clone()
            shared[ref['id']] = item
        items = []
        for c in p['cases']:
            camera = pixels(root/c['input']); assert camera.shape == (256, 256, 3)
            item = {**shared[c['source_person_or_reference']], 'case': c, 'camera': camera, 'x': canonical(camera)}
            item['degraded_weight'] = item['x'].new_tensor([0. if c['profile']=='clear' else 1.25])
            item['clear_weight'] = item['x'].new_tensor([1. if c['profile']=='clear' else 0.])
            items.append(item)
        for begin in range(0, 3905, 5):
            clock(); assert time.monotonic()-cache_start < 900; step = time.monotonic(); group = items[begin:begin+5]
            x = torch.cat([i['x'] for i in group]); mask = torch.cat([i['mask'] for i in group])
            with torch.no_grad(): base = torch.where(mask.bool(), original(x), x).detach().clone()
            assert not torch.is_inference(base)
            for slot, item in enumerate(group): item['base'] = base[slot:slot+1].clone()
            torch.cuda.synchronize(); cache_times.append(time.monotonic()-step)
            if begin == 95:
                elapsed = time.monotonic()-cache_start; projected = elapsed+761*float(np.mean(cache_times[1:]))*1.25
                write(out/'cache_timing.json', {'cases': 100, 'seconds': elapsed, 'steady_sample_seconds': cache_times[1:],
                      'remaining_batches': 761, 'safety_factor': 1.25, 'projected_seconds': projected, 'cap_seconds': 900})
                assert projected <= 900, 'Cache projection exceeds900s'
            if begin % 250 == 0: print({'V40_cached': begin+5, 'of': 3905}, flush=True)
        assert time.monotonic()-cache_start <= 900, 'Cache actual900s stop'
        write(out/'cache_receipt.json', {'complete': True, 'seconds': time.monotonic()-cache_start, 'references': 781,
              'cases': 3905, 'cap_seconds': 900, 'optimizer_updates': 0, 'shared_reference_assets': True})
        by_id = {i['case']['id']: i for i in items}
        ns = {'torch': torch, 'F': F}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != 'require_vm']
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<original-filters-and-loss-functions>', 'exec'), ns)
        z = torch.arange(-6, 7, dtype=torch.float32); kernel = torch.exp(-.5*(z/2).square()); kernel /= kernel.sum()
        head = SimpleNamespace(kernel=kernel.cuda(), reflect_indices=torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))).cuda())
        head.blur = MethodType(ns['blur'], head); head.high = MethodType(ns['high'], head); ns['head'] = head
        normalizers, cohort = cohort_normalizers([by_id[cid] for cid in p['preview_case_ids']], head)
        write(out/'cohort_loss_setup.json', cohort)
        keys = ['x', 'base', 'target', 'mask', 'feature', 'interior', 'valid7', 'grid', 'truth', 'degraded_weight', 'clear_weight']
        def batch(ids): return {key: torch.cat([items[i][key] for i in ids]) for key in keys}
        def snapshot(update):
            clock(); stamp = time.monotonic(); folder = out/('update'+str(update)); folder.mkdir(); rows = []
            torch.save(candidate.decoder.state_dict(), folder/'spatial_decoder.pth')
            progress['new_trained_checkpoint'] = update > 0 or progress['new_trained_checkpoint']; frozen()
            with torch.no_grad():
                for begin in range(0, 3905, 5):
                    clock(); group = items[begin:begin+5]; b = batch(list(range(begin, begin+5)))
                    pred = candidate(b['x'], b['mask'])
                    if update == 0: assert torch.equal(pred, b['base']), 'Exact initial parity across full TRAIN'
                    raw = pred.permute(0, 2, 3, 1).cpu().numpy().copy()
                    delivered = np.stack([np.where(i['mask8'][..., None], np.floor(a*np.float32(255)), i['camera']).astype(np.uint8) for i, a in zip(group, raw)])
                    vectors = identity.embedding(torch.cat([canonical(a) for a in delivered]), b['mask'], b['grid']).cpu().numpy().copy()
                    for item, a, png, vector in zip(group, raw, delivered, vectors):
                        c = item['case']; cid = c['id']; truth = item['truth'][0].cpu().numpy().copy()
                        Image.fromarray(png).save(folder/(cid+'.png')); np.save(folder/(cid+'_embedding.npy'), vector, allow_pickle=False)
                        if update == 0: np.save(folder/(cid+'_target_embedding.npy'), truth, allow_pickle=False)
                        if cid in p['preview_case_ids']: np.save(folder/(cid+'.npy'), a, allow_pickle=False)
                        base = item['base'][0].permute(1, 2, 0).cpu().numpy()
                        shift = (a-base)[item['mask8']].astype(np.float64).mean(0)
                        mean_raw = np.clip(base.astype(np.float64)+shift, 0, 1).astype(np.float32)
                        mean_png = np.where(item['mask8'][..., None], np.floor(mean_raw*np.float32(255)), item['camera']).astype(np.uint8)
                        Image.fromarray(mean_png).save(folder/(cid+'_mean_only.png'))
                        metric = exported_pixel_metrics(png, item['target8'], item['mask8'])
                        metric.update({'ArcFace_observed_fixed': float(vector@truth), 'landmark_high_frequency_MSE': detail_metric(png, item['target8'], feature_support(item['mask8'], c['landmarks5_canvas_xy'])),
                                       'constant_mean_shift_only_MSE': exported_pixel_metrics(mean_png, item['target8'], item['mask8'])['MSE']})
                        rows.append({'id': cid, 'source': c['source'], 'profile': c['profile'], 'metrics': metric,
                                     'postclip_mean_RGB_shift': shift.tolist(), 'raw_float32_sha256': __import__('hashlib').sha256(a.tobytes()).hexdigest()})
                    if begin % 250 == 0: print({'V40_snapshot': update, 'cases': begin+5, 'of': 3905}, flush=True)
            receipt = {'complete': True, 'update': update, 'rows': rows, 'groups': groups(rows), 'decoder_state': state_hash(candidate.decoder),
                       'checkpoint_sha256': sha(folder/'spatial_decoder.pth'), 'snapshot_duration_seconds': time.monotonic()-stamp}
            write(folder/'metrics.json', receipt); snapshots.append(receipt); frozen()
            print({'snapshot': update, 'feature_MSE': receipt['groups']['degraded']['landmark_high_frequency_MSE']}, flush=True)
            return receipt
        baseline = snapshot(0)
        optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=.01)
        progress['optimizer_constructed'] = True; fit_started = time.monotonic()
        for update, ids in enumerate(read(root/'schedule.json')['batches'], 1):
            clock(); assert time.monotonic()-fit_started < 3600, 'Fit3600s stop'; step = time.monotonic()
            optimizer.zero_grad(set_to_none=True); b = batch(ids); pred = candidate(b['x'], b['mask'])
            terms = objective_terms(b, pred, identity, ns['mean'], ns['feature_errors'], ns['ssim'], normalizers)
            assert list(terms) == p['terms']; objective = sum(terms.values()).mean(); assert torch.isfinite(objective)
            objective.backward(); progress['backwards'] += 1
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters)
            torch.nn.utils.clip_grad_norm_(parameters, 1); optimizer.step(); progress['optimizer_updates'] = update
            torch.cuda.synchronize(); times.append(time.monotonic()-step)
            if update == 20:
                elapsed = time.monotonic()-fit_started; allowance = 2*baseline['snapshot_duration_seconds']*1.25
                projected = elapsed+780*float(np.mean(times[1:]))*1.25+allowance
                write(out/'timing_update20.json', {'updates': 20, 'seconds': elapsed, 'steady_sample_seconds': times[1:],
                      'remaining_updates': 780, 'safety_factor': 1.25, 'overhead_seconds': allowance, 'projected_seconds': projected, 'cap_seconds': 3600})
                assert projected <= 3600, 'Training timing projection exceeds3600s'
            if update in [50, 800]:
                current = snapshot(update); gate = capacity(baseline['groups'], current['groups'], .01 if update == 50 else .1)
                write(out/('capacity_update'+str(update)+'.json'), {'update': update, **gate})
                if update == 50: assert gate['pass'], 'Early structure/preservation requirement failed; retain stop'
            if update == 1 or update % 50 == 0: print(f'V40 update {update}/800 objective={float(objective.detach()):.6f}', flush=True)
        clock(); assert time.monotonic()-fit_started <= 3600, 'Fit actual3600s stop'; frozen(); verify(root, pin)
        write(out/'results.json', {'complete': True, 'protocol_sha256': pin, **progress, 'seconds': time.monotonic()-started,
              'necessary_capacity_pass': gate['pass'], 'capacity': gate, 'states_before': before, 'states_after': states(),
              'training_cases': 3905, 'training_references': 781, 'epochs': 800/781, 'completed_epochs': 1,
              'step_times_seconds': times, 'fit_seconds': time.monotonic()-fit_started, 'forward_counts': counts,
              'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(), 'original_provenance': provenance,
              'all3905_initial_raw_parity_exact': True, 'original_and_fixed_reference_and_recognizer_unchanged': True,
              'no_resume_or_automatic_follow_on': True, 'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False})
    except BaseException as exc:
        if candidate is not None:
            candidate.decoder.requires_grad_(False)
            import torch
            torch.save(candidate.decoder.state_dict(), out/'stopped_spatial_decoder.pth')
            progress['new_trained_checkpoint'] = progress['optimizer_updates'] > 0
        write(out/'failure.json', {'complete': False, 'protocol_sha256': pin, **progress, 'error': repr(exc), 'traceback': traceback.format_exc(),
              'seconds': time.monotonic()-started, 'step_times_seconds': times, 'fit_seconds': None if fit_started is None else time.monotonic()-fit_started,
              'states_before': locals().get('before'), 'states_after': states() if 'states' in locals() else None,
              'forward_counts': locals().get('counts'), 'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated() if candidate is not None else None,
              'complete_snapshots': [r['update'] for r in snapshots], 'resume_permitted': False, 'automatic_repeat_permitted': False,
              'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False})
        raise


def export(root, p, pin):
    sha, read, write, _ = contract(root)
    assert (root/'outputs').is_dir(); start = time.monotonic()
    destination = Path.home()/(STEM+'-results.tar.gz'); assert not destination.exists() and not (root/'export_manifest.json').exists()
    paths = sorted(q for q in root.rglob('*') if q.is_file() and q.relative_to(root).as_posix() not in p['assets_sha256'])
    assert all(not q.is_symlink() for q in paths)
    assert sum(q.stat().st_size for q in paths) <= 3*1024**3
    mapping = {q.relative_to(root).as_posix(): sha(q) for q in paths}
    write(root/'export_manifest.json', {'complete': True, 'protocol_sha256': pin, 'files_sha256': mapping, 'input_assets_omitted_and_bound_to_protocol': True})
    paths.append(root/'export_manifest.json')
    with tarfile.open(destination, 'w:gz', compresslevel=1) as archive:
        for q in paths:
            assert time.monotonic()-start < 900
            archive.add(q, arcname='cctv_dgp_spatial_fit_v40_return/'+q.relative_to(root).as_posix(), recursive=False)
    digest = sha(destination)
    with Path(str(destination)+'.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest+'  '+destination.name+'\n')
    result = read(root/('outputs/results.json' if (root/'outputs/results.json').is_file() else 'outputs/failure.json'))
    write(Path.home()/(STEM+'-export.json'), {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size,
          'seconds': time.monotonic()-start, 'training_success_not_implied': True, 'optimizer_updates': result['optimizer_updates'],
          'run_results_present': (root/'outputs/results.json').is_file(), 'failure_present': (root/'outputs/failure.json').is_file()})
    print({'archive_sha256': digest, 'bytes': destination.stat().st_size, 'training_success_not_implied': True}, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); parser.add_argument('--protocol-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for key in ['verify-transfer', 'run', 'export', 'record-supervision']: mode.add_argument('--'+key, action='store_true')
    parser.add_argument('--elapsed', type=float); parser.add_argument('--exit-code', type=int)
    a = parser.parse_args(); root = a.root.resolve(); scope(root); p = verify(root, a.protocol_sha)
    sha, read, write, _ = contract(root)
    if a.verify_transfer: print({'complete': True, 'cases': 3905, 'references': 781, 'neural_calls': 0, 'optimizer_updates': 0}); return
    if a.record_supervision:
        assert a.elapsed is not None and a.exit_code is not None
        write(root/'supervisor_receipt.json', {'complete': True, 'protocol_sha256': a.protocol_sha, 'seconds': a.elapsed,
              'cap_seconds': 4800, 'kill_grace_seconds': 30, 'worker_exit_code': a.exit_code, 'quality_acceptance_not_implied': True}); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Finite V40 worker/export deadline')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('V40 external deadline')))
    signal.alarm(900 if a.export else 4500)
    export(root, p, a.protocol_sha) if a.export else run(root, p, a.protocol_sha)


if __name__ == '__main__': main()

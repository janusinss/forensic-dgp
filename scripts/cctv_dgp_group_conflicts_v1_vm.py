"""Manual L4 connectivity and finite trial outputs; no optimizer or trained save."""
import argparse
import ast
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import tarfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cctv_dgp_group_conflicts_v1_contract import (
    NAME, STEM, BUDGETS, TERMS, SCOPES, FRACTIONS, STATE, read, write, sha, verified_assets)


def scope(root):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root == (Path.home()/'forensic-dgp'/NAME).resolve(), 'Distinct diagnostic root only'


def run(root, p, pin):
    scope(root); assert os.environ.get('TMUX'), 'Manual tmux launch required'
    assert not (root/'outputs').exists(), 'Preserve every prior/partial diagnostic; no repeat or resume'
    start = time.monotonic(); out = root/'outputs'; out.mkdir()
    progress = {'gradient_queries': 0, 'optimizer_updates': 0, 'backwards': 0,
                'epochs': 0, 'committed_trajectory_updates': 0, 'trials_completed': [],
                'optimizer_constructed': False, 'new_trained_checkpoint': False}
    candidate = original = identity = None; before = None; counts = {}
    try:
        assert shutil.disk_usage(root).free >= BUDGETS['minimum_free_disk_bytes'], 'Need4GiB free after install'
        import subprocess, urllib.request
        tree = ast.parse((root/'frozen_definitions.py').read_text())
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_vm')
        ns = {'sys': sys, 'platform': platform, 'Path': Path, 'os': os,
              'subprocess': subprocess, 'urllib': __import__('urllib')}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), '<pinned-hardware-idle-guard>', 'exec'), ns)
        ns['require_vm'](root, idle=True)
        import importlib.metadata
        import numpy as np
        import torch
        from torch.nn import functional as F
        from PIL import Image
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_group_conflicts_v1_candidate import OriginalFeatureProbe
        from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, buffer_hash
        from frozen_capacity_contract import feature_support, exported_pixel_metrics, detail_metric, capacity
        from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups
        torch.set_num_threads(4); torch.manual_seed(430043); np.random.seed(430043)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
        from cctv_dgp_group_conflicts_v1_math import common_direction
        packages = {}
        for key in ['torch', 'torchvision', 'numpy', 'Pillow', 'scipy', 'scikit-image', 'onnx', 'onnx2torch']:
            packages[key] = importlib.metadata.version(key)
        write(out/'environment.json', {'python': sys.version, 'packages': packages,
            'host': platform.node(), 'GPU': torch.cuda.get_device_name(0), 'CUDA': torch.version.cuda,
            'machine_type_verified': 'g2-standard-4', 'TF32': False, 'AMP': False,
            'original_checkpoint_sha256': p['original_checkpoint_sha256']})
        original, provenance = load_frozen_dgp_restorer(root/'weights/dgp_v2.pth',
            expected_sha256=p['original_checkpoint_sha256'], device='cuda')
        candidate = OriginalFeatureProbe(original.net, p['parameter_layout'])
        identity = FixedObservedIdentity(root/'weights/w600k_r50.onnx', 'cuda')
        def states(): return {'original': state_hash(original.net), 'candidate': state_hash(candidate.net),
                             'candidate_buffers': buffer_hash(candidate.net), 'recognizer': state_hash(identity)}
        before = states(); assert before['original'] == before['candidate'] == STATE
        assert before['recognizer'] == p['recognizer_state']
        counts.update({'original': 0, 'candidate': 0, 'recognizer': 0})
        for model, key in [(original.net, 'original'), (candidate.net, 'candidate'), (identity.encoder, 'recognizer')]:
            model.register_forward_hook(lambda *_a, key=key: counts.__setitem__(key, counts[key]+1))
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic()-start <= BUDGETS['worker_seconds'], 'Worker deadline stop'
            assert torch.cuda.max_memory_allocated() <= BUDGETS['peak_vram_bytes'], 'VRAM20GiB stop'
            assert shutil.disk_usage(root).free >= BUDGETS['disk_reserve_bytes'], 'Disk reserve stop'
            for key in counts: assert counts[key] <= BUDGETS[key+'_forward_calls'], 'Forward budget '+key
        def frozen(require_initial=False):
            now = states()
            for key in ['original', 'candidate_buffers', 'recognizer']: assert now[key] == before[key]
            if require_initial: assert now['candidate'] == before['candidate']
            assert all(v.grad is None for v in candidate.net.parameters())
            assert all(not v.requires_grad and v.grad is None for m in [original, identity] for v in m.parameters())
            assert sha(root/'weights/dgp_v2.pth') == p['original_checkpoint_sha256']
        def pixels(q, mode='RGB'):
            with Image.open(q) as im: return np.asarray(im.convert(mode)).copy()
        def tensor(a): return torch.from_numpy(np.asarray(a).copy()).cuda()
        def rgb(a): return tensor(a.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
        refs = {r['id']: r for r in p['references']}; items = []; cache_start = time.monotonic()
        for c in p['cases']:
            r = refs[c['source_person_or_reference']]
            camera = pixels(root/c['input']); target = pixels(root/r['target']); mask = pixels(root/r['observed'], 'L') > 0
            assert camera.shape == target.shape == (256, 256, 3)
            feature = feature_support(mask, c['landmarks5_canvas_xy'])
            items.append({'case': c, 'camera': camera, 'target8': target, 'mask8': mask,
                'x': rgb(camera), 'target': rgb(target), 'mask': tensor(mask.astype(np.float32))[None, None],
                'feature': tensor(feature.astype(np.float32))[None, None], 'feature8': feature,
                'grid': tensor(grid112(r['matrix112']))[None]})
        def batch(ids):
            return {key: torch.cat([items[i][key] for i in ids]) for key in ['x', 'target', 'mask', 'feature', 'grid', 'base', 'truth']}
        with torch.no_grad():
            for begin in range(0, 100, 5):
                clock(); selected = items[begin:begin+5]
                x = torch.cat([i['x'] for i in selected]); mask = torch.cat([i['mask'] for i in selected])
                # Clone outside the adapter's inference-mode decorator.
                base = original(x).detach().clone()
                assert not torch.is_inference(base)
                for slot, item in enumerate(selected): item['base'] = base[slot:slot+1].clone()
                target = torch.cat([i['target'] for i in selected]); grid = torch.cat([i['grid'] for i in selected])
                truth = identity.embedding(target, mask, grid).detach().clone()
                for slot, item in enumerate(selected): item['truth'] = truth[slot:slot+1].clone()
        cache_seconds = time.monotonic()-cache_start
        assert cache_seconds <= BUDGETS['cache_seconds']
        write(out/'cache_receipt.json', {'complete': True, 'cases': 100, 'seconds': cache_seconds,
            'cap_seconds': BUDGETS['cache_seconds'], 'optimizer_updates': 0})
        by_id = {i['case']['id']: k for k, i in enumerate(items)}
        initial = candidate.vector(); assert np.array_equal(initial, np.load(root/'evidence/initial_parameters.npy', allow_pickle=False)); np.save(out/'initial_parameters.npy', initial, allow_pickle=False)
        def evaluate(label):
            folder = out/label; folder.mkdir(); rows = []; stamp = time.monotonic()
            with torch.no_grad():
                for begin in range(0, 100, 5):
                    clock(); ids = list(range(begin, begin+5)); b = batch(ids)
                    result = candidate(b['x'], b['mask'], b['base'])
                    if label == 'baseline': assert torch.equal(result, torch.where(b['mask'].bool(), b['base'], b['x']))
                    raw = result.permute(0, 2, 3, 1).cpu().numpy().copy()
                    pngs = [deliver(a, items[idx]['camera'], items[idx]['mask8']) for idx, a in zip(ids, raw)]
                    vectors = identity.embedding(torch.cat([result, torch.cat([rgb(a) for a in pngs])]),
                        torch.cat([b['mask']]*2), torch.cat([b['grid']]*2)).cpu().numpy().copy()
                    for slot, idx in enumerate(ids):
                        item = items[idx]; c = item['case']; cid = c['id']; a = raw[slot]; png = pngs[slot]
                        base = np.where(item['mask8'][..., None], item['base'][0].permute(1, 2, 0).cpu().numpy(), item['camera'].astype(np.float32)/np.float32(255))
                        mraw, mpng, shift = mean_only(a, base, item['camera'], item['mask8'])
                        truth = item['truth'][0].cpu().numpy().copy()
                        np.save(folder/(cid+'.npy'), a, allow_pickle=False)
                        np.save(folder/(cid+'_embeddings.npy'), np.stack([vectors[slot], vectors[slot+5], truth]), allow_pickle=False)
                        Image.fromarray(png).save(folder/(cid+'.png'))
                        Image.fromarray(mpng).save(folder/(cid+'_mean_only.png'))
                        rm = pixel_metrics(a, item['target8'], item['mask8'])
                        rm.update({'ArcFace_observed_fixed': float(vectors[slot]@truth),
                            'landmark_high_frequency_MSE': detail_float(a, item['target8'], item['feature8']),
                            'constant_mean_shift_only_MSE': pixel_metrics(mraw, item['target8'], item['mask8'])['MSE']})
                        pm = exported_pixel_metrics(png, item['target8'], item['mask8'])
                        pm.update({'ArcFace_observed_fixed': float(vectors[slot+5]@truth),
                            'landmark_high_frequency_MSE': detail_metric(png, item['target8'], item['feature8']),
                            'constant_mean_shift_only_MSE': exported_pixel_metrics(mpng, item['target8'], item['mask8'])['MSE']})
                        rows.append({'id': cid, 'source': c['source'], 'profile': c['profile'], 'raw': rm,
                            'png': pm, 'postclip_mean_RGB_shift': shift.tolist()})
            groups = {}
            for co in p['cohorts']:
                chosen = [r for r in rows if r['id'] in co['case_ids']]
                groups[co['name']] = {s: review_groups(chosen, s) for s in ['raw', 'png']}
            receipt = {'complete': True, 'variant': label, 'rows': rows, 'groups': groups,
                'candidate_state': state_hash(candidate.net), 'seconds': time.monotonic()-stamp,
                'full_TRAIN_capacity_not_tested': True, 'model_qualification': False}
            write(folder/'metrics.json', receipt); frozen(); return receipt
        baseline = evaluate('baseline'); frozen(True)
        remaining_estimate = 3*baseline['seconds']*1.25
        write(out/'trial_timing_projection.json', {'initial_100_case_seconds': baseline['seconds'],
            'remaining_variants': 3, 'safety_factor': 1.25, 'projected_trial_seconds': remaining_estimate,
            'cap_seconds': BUDGETS['trial_seconds']})
        assert remaining_estimate <= BUDGETS['trial_seconds'], 'Finite-output timing projection stop'
        baseline_bytes = sum(q.stat().st_size for q in (out/'baseline').rglob('*') if q.is_file())
        projected_bytes = 4*baseline_bytes*1.25 + 650*1024**2
        write(out/'storage_projection.json', {'baseline_bytes': baseline_bytes,
            'projected_uncompressed_bytes': projected_bytes, 'cap_bytes': BUDGETS['return_uncompressed_bytes']})
        assert projected_bytes <= BUDGETS['return_uncompressed_bytes']
        assert shutil.disk_usage(root).free >= 2*projected_bytes-baseline_bytes+BUDGETS['disk_reserve_bytes']
        # Forward parity first; no autograd or candidate forward is used here.
        parity_start = time.monotonic(); parity_rows = []
        limits = p['loss_surrogate_parity']; maximum = {'MSE': 0., 'SSIM': 0., 'structure': 0.}
        with torch.no_grad():
            for begin in range(0, 100, 5):
                clock(); ids = list(range(begin, begin+5)); b = batch(ids)
                raw = torch.where(b['mask'].bool(), b['base'], b['x'])
                values = pixel_and_structure_losses(raw, b['target'], b['mask'], b['feature'])
                for slot, idx in enumerate(ids):
                    stored = baseline['rows'][idx]['raw']
                    errors = {'MSE': abs(float(values['MSE'][slot])-stored['MSE']),
                        'SSIM': abs(1.-float(values['SSIM_loss'][slot])-stored['SSIM']),
                        'structure': abs(float(values['landmark_structure'][slot])-stored['landmark_high_frequency_MSE'])}
                    parity_rows.append({'id': items[idx]['case']['id'], 'errors': errors})
                    for key in maximum: maximum[key] = max(maximum[key], errors[key])
        passed = (maximum['MSE'] <= limits['MSE_abs'] and maximum['SSIM'] <= limits['SSIM_abs'] and
                  maximum['structure'] <= limits['landmark_structure_abs'])
        write(out/'loss_surrogate_parity.json', {'complete': True, 'passed': passed,
            'cases': 100, 'maximum_errors': maximum, 'rows': parity_rows,
            'seconds': time.monotonic()-parity_start, 'gradient_queries': 0})
        assert passed, 'Loss-surrogate forward parity failed; stop before derivatives'
        candidate.enable_diagnostic_gradients(root)
        parameters = [v for _, v in candidate.selected]
        aggregate = np.zeros((32, 1996035), np.float64); gradient_rows = []
        gradient_start = time.monotonic()
        lookup = {(r['source'], r['profile'], r['metric']): k for k, r in enumerate(p['group_losses'])}
        import hashlib
        for begin in range(0, 50, 5):
            clock(); assert time.monotonic()-gradient_start <= BUDGETS['gradient_seconds']
            ids = [by_id[cid] for cid in p['cohorts'][0]['case_ids'][begin:begin+5]]
            b = batch(ids); result = candidate(b['x'], b['mask'], b['base'])
            assert torch.equal(result, torch.where(b['mask'].bool(), b['base'], b['x']))
            values = pixel_and_structure_losses(result, b['target'], b['mask'], b['feature'])
            vector = identity.embedding(result, b['mask'], b['grid'])
            values['ArcFace_loss'] = 1.-(vector*b['truth']).sum(1)
            source_name = items[ids[0]]['case']['source']; queries = []
            for slot, idx in enumerate(ids):
                case = items[idx]['case']; assert case['source'] == source_name
                for metric in ['MSE', 'SSIM_loss', 'ArcFace_loss']:
                    queries.append((lookup[(source_name, case['profile'], metric)], values[metric][slot]))
            queries.append((lookup[(source_name, 'degraded', 'landmark_structure')], values['landmark_structure'][1:].mean()))
            assert len(queries) == 16
            for slot, (group_index, value) in enumerate(queries):
                clock(); assert time.monotonic()-gradient_start <= BUDGETS['gradient_seconds']
                gradients = torch.autograd.grad(value, parameters, retain_graph=slot < 15,
                                                allow_unused=True, materialize_grads=False)
                progress['gradient_queries'] += 1; parameter_rows = []; full = np.zeros(1996035, np.float32)
                for desc, grad in zip(p['parameter_layout'], gradients):
                    values_cpu = np.zeros(desc['elements'], np.float32) if grad is None else grad.detach().reshape(-1).cpu().numpy().copy()
                    assert values_cpu.dtype == np.float32 and np.isfinite(values_cpu).all()
                    full[desc['start']:desc['end']] = values_cpu
                    parameter_rows.append({'name': desc['name'], 'graph_connected': grad is not None,
                        'L2': float(np.linalg.norm(values_cpu.astype(np.float64))),
                        'nonzero_values': int(np.count_nonzero(values_cpu))})
                aggregate[group_index] += full.astype(np.float64)/5.
                gradient_rows.append({'batch': begin//5, 'group_index': group_index,
                    'group': p['group_losses'][group_index], 'value': float(value.detach()),
                    'case_ids': [items[i]['case']['id'] for i in ids],
                    'query_vector_sha256': hashlib.sha256(full.tobytes()).hexdigest(),
                    'query_L2': float(np.linalg.norm(full.astype(np.float64))), 'parameters': parameter_rows})
            del gradients, queries, values, result, vector
            frozen(True)
            print({'gradient_batch': begin//5+1, 'of': 10, 'gradient_queries': progress['gradient_queries']}, flush=True)
        candidate.net.requires_grad_(False); frozen(True)
        gradient_seconds = time.monotonic()-gradient_start
        assert gradient_seconds <= BUDGETS['gradient_seconds'] and progress['gradient_queries'] == 160
        assert np.isfinite(aggregate).all()
        np.savez_compressed(out/'group_gradients.npz', gradients=aggregate)
        write(out/'gradient_receipt.json', {'complete': True, 'rows': gradient_rows,
            'groups': p['group_losses'], 'seconds': gradient_seconds, 'gradient_queries': 160,
            'aggregate_vectors': 32, 'aggregation': 'five equal reference contributions per group',
            'individual_query_vectors_exported': False, 'local_autograd_replay_permitted': False,
            'optimizer_updates': 0, 'none_and_numeric_zero_separate': True})
        with np.load(root/'evidence/previous_directions.npz', allow_pickle=False) as saved:
            old_direction = saved['balanced_2'].copy()
        assert old_direction.dtype == np.float64 and old_direction.shape == (1996035,)
        norms = np.linalg.norm(aggregate, axis=1)
        old_unit = old_direction/np.linalg.norm(old_direction)
        previous_cosines = -(aggregate@old_unit)/np.maximum(norms, 1e-300)
        write(out/'previous_direction_group_predictions.json', {'complete': True,
            'direction': 'previous balanced_2', 'groups': p['group_losses'],
            'descent_cosines': previous_cosines.tolist(),
            'raw_directional_derivatives': (aggregate@old_direction).tolist(),
            'negative_cosine_group_indices': np.flatnonzero(previous_cosines < 0).tolist(),
            'sampled_first_order_only': True})
        solver_start = time.monotonic()
        direction, certificate = common_direction(aggregate, p['minimum_normalized_descent_cosine'])
        assert time.monotonic()-solver_start <= BUDGETS['solver_seconds']; clock()
        write(out/'direction_certificate.json', certificate)
        if direction is not None: np.save(out/'common_direction.npy', direction, allow_pickle=False)
        decoder = np.zeros(1996035, bool)
        for desc in p['parameter_layout']:
            if desc['partition'] == 'decoder_control': decoder[desc['start']:desc['end']] = True
        weight_norm = p['initial_decoder_weight_L2']
        assert abs(float(np.linalg.norm(initial.astype(np.float64)[decoder]))-weight_norm) <= 1e-10
        trials_start = time.monotonic(); trial_summaries = []
        if direction is not None:
            for fraction in FRACTIONS:
                clock(); assert time.monotonic()-trials_start <= BUDGETS['trial_seconds']
                label = 'common_'+format(fraction, '.0e').replace('-', 'm')
                trial = (initial.astype(np.float64)+fraction*weight_norm*direction).astype(np.float32)
                assert np.isfinite(trial).all(); actual = trial.astype(np.float64)-initial.astype(np.float64)
                candidate.assign_trial(trial); np.save(out/(label+'_parameters.npy'), trial, allow_pickle=False)
                receipt = evaluate(label)
                comparisons = {co['name']: {stage: capacity(baseline['groups'][co['name']][stage], receipt['groups'][co['name']][stage], .01)
                    for stage in ['raw', 'png']} for co in p['cohorts']}
                trial_summaries.append({'variant': label, 'scope': 'common', 'relative_fraction': fraction,
                    'actual_displacement_L2': float(np.linalg.norm(actual)), 'selected_weight_L2': weight_norm,
                    'scale_weight_partition': 'decoder_control',
                    'relative_displacement_actual': float(np.linalg.norm(actual))/weight_norm,
                    'gradient_dot_actual_displacement': (aggregate@actual).tolist(),
                    'all_actual_group_derivatives_negative': bool(np.all(aggregate@actual < 0)),
                    'candidate_state': receipt['candidate_state'], 'comparisons': comparisons,
                    'model_qualification': False})
                progress['trials_completed'].append(label)
                candidate.assign_trial(initial); frozen(True)
                print({'trial': label, 'of': 3, 'optimizer_updates': 0}, flush=True)
        else:
            print({'common_direction_found': False, 'finite_trials': 0, 'optimizer_updates': 0}, flush=True)
        trial_seconds = time.monotonic()-trials_start
        assert trial_seconds <= BUDGETS['trial_seconds']
        assert counts == {'original': 20, 'candidate': 30+20*len(trial_summaries), 'recognizer': 50+20*len(trial_summaries)}
        frozen(True); clock()
        write(out/'results.json', {'complete': True, 'protocol_sha256': pin, **progress,
            'trial_summaries': trial_summaries, 'seconds': time.monotonic()-start,
            'trial_seconds': trial_seconds, 'states_before': before, 'states_after': states(),
            'forward_counts': counts, 'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(),
            'original_provenance': provenance, 'raw_and_PNG_cases': 100*(1+len(trial_summaries)), 'common_direction_found': direction is not None,
            'full_TRAIN_capacity_not_tested': True, 'native_or_DEV_or_final_used': False,
            'model_qualification': False, 'app_promotion': False, 'goal_complete': False})
    except BaseException as exc:
        if candidate is not None and candidate.vm_root is not None and 'initial' in locals():
            candidate.assign_trial(initial); candidate.net.requires_grad_(False)
        write(out/'failure.json', {'complete': False, 'protocol_sha256': pin, **progress,
            'error': repr(exc), 'traceback': traceback.format_exc(), 'seconds': time.monotonic()-start,
            'states_before': before, 'states_after': states() if before else None,
            'forward_counts': counts, 'resume_permitted': False, 'automatic_repeat_permitted': False,
            'model_qualification': False, 'app_promotion': False})
        raise


def export(root, p, pin):
    start = time.monotonic(); destination = Path.home()/(STEM+'-results.tar.gz')
    assert not destination.exists() and not (root/'export_manifest.json').exists()
    assert (root/'outputs').is_dir()
    paths = sorted(q for q in root.rglob('*') if q.is_file() and q.relative_to(root).as_posix() not in p['assets_sha256'])
    assert all(not q.is_symlink() for q in paths)
    size = sum(q.stat().st_size for q in paths); assert size <= BUDGETS['return_uncompressed_bytes']
    assert shutil.disk_usage(root).free >= size+BUDGETS['disk_reserve_bytes']
    write(root/'export_manifest.json', {'complete': True, 'protocol_sha256': pin,
        'files_sha256': {q.relative_to(root).as_posix(): sha(q) for q in paths},
        'input_assets_omitted_and_bound_to_protocol': True})
    paths.append(root/'export_manifest.json')
    with tarfile.open(destination, 'w:gz', compresslevel=1) as tar:
        for q in paths:
            assert time.monotonic()-start <= BUDGETS['export_seconds']
            tar.add(q, arcname=NAME+'_return/'+q.relative_to(root).as_posix(), recursive=False)
    digest = sha(destination)
    with Path(str(destination)+'.sha256').open('x', encoding='ascii', newline='\n') as f:
        f.write(digest+'  '+destination.name+'\n')
    result = {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size,
        'seconds': time.monotonic()-start, 'training_success_not_implied': True,
        'optimizer_updates': 0, 'run_results_present': (root/'outputs/results.json').is_file(),
        'failure_present': (root/'outputs/failure.json').is_file()}
    write(Path.home()/(STEM+'-export.json'), result); print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for key in ['verify-transfer', 'run', 'export', 'record-supervision']: mode.add_argument('--'+key, action='store_true')
    parser.add_argument('--elapsed', type=float); parser.add_argument('--exit-code', type=int)
    a = parser.parse_args(); root = a.root.resolve(); scope(root)
    sys.path.insert(0, str(root)); p = verified_assets(root, a.protocol_sha)
    if a.verify_transfer:
        print({'complete': True, 'cases': 100, 'tensors': 158, 'optimizer_updates': 0, 'neural_or_gradient_calls': 0}); return
    if a.record_supervision:
        assert a.elapsed is not None and a.exit_code is not None
        write(root/'supervisor_receipt.json', {'complete': True, 'protocol_sha256': a.protocol_sha,
            'seconds': a.elapsed, 'cap_seconds': BUDGETS['external_seconds'], 'kill_grace_seconds': 30,
            'worker_exit_code': a.exit_code, 'quality_acceptance_not_implied': True}); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Finite diagnostic deadline')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('External diagnostic stop')))
    signal.alarm(BUDGETS['export_seconds'] if a.export else BUDGETS['worker_seconds'])
    export(root, p, a.protocol_sha) if a.export else run(root, p, a.protocol_sha)


if __name__ == '__main__': main()

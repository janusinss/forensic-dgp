"""Manual existing-L4 finite-displacement probe; no new gradients or trajectory."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import signal
import sys
import tarfile
import time
import traceback
from types import MethodType

NAME = 'cctv_dgp_finite_clearance_probe_v36_vm'
STEM = 'cctv-dgp-finite-clearance-probe-v36'
OLD = 'cctv_dgp_v32_loss_gradient_v1_vm'
OLD_PIN = '61c04aef4ee184adc9830482477bf37ad9c936a8c0247e44d26eefc1a6bf483d'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scope(root):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing Linux VM only; no local parameter updates'
    assert root == (Path.home() / 'forensic-dgp' / NAME).resolve(), 'Distinct V36 folder required'


def check_files(root, mapping):
    for name, digest in mapping.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file() and not path.is_symlink() and sha(path) == digest, name


def verify(root, pin):
    assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json')
    assert p['format'] == 'own-DGP-empirical-clearance-finite-image-probe-v36'
    assert p['optimizer_updates'] == p['committed_trajectory_updates'] == p['new_gradient_queries'] == p['backwards'] == p['epochs'] == 0
    assert p['states'] == [0] and p['candidate_displacement_trials'] == 4
    assert p['variants'] == [{'name':name,'proposal':'projected_restoration','scale':scale} for name,scale in
        [('clearance_1',1.),('clearance_half',.5),('clearance_quarter',.25),('clearance_eighth',.125)]]
    assert p['before_outputs'] == 100 and p['trial_outputs'] == 400
    assert p['budgets']['worker_seconds'] == 900 and p['budgets']['external_seconds'] == 930
    check_files(root,p['assets_sha256'])
    return p


def dependencies(root,p):
    guard_root=root.parent/'cctv_dgp_group_guard_grad_v34_vm'
    check_files(guard_root,p['guard_return_sha256'])
    guard_p=read(guard_root/'protocol.json')
    assert sha(guard_root/'protocol.json')==p['guard_protocol_sha256']
    spec=importlib.util.spec_from_file_location('pinned_V34_metadata_only',guard_root/'scripts/cctv_dgp_group_guard_grad_v34_vm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    basis,parent,active,closed,mixed,helper,probe=module.dependencies(root,guard_p)
    result=read(guard_root/'outputs/results.json')
    assert result['complete'] and result['gradient_queries']==300 and result['optimizer_updates']==result['parameter_updates']==0
    assert not (guard_root/'outputs/failure.json').exists()
    assert basis['cohorts']==p['cohorts'] and basis['parameter_layout']==p['parameter_layout'] and basis['terms']==p['terms']
    assert p['retained_capacity_gates']==guard_p['retained_capacity_gates']
    return root.parent/OLD,basis,parent,active,closed,mixed,helper


def run(root, p, pin):
    scope(root)
    assert not (root / 'outputs').exists(), 'Preserve every prior/partial run; no resume'
    started = time.monotonic(); out = root / 'outputs'; out.mkdir()
    progress = {'reference_DGP_forwards': 0, 'candidate_DGP_forwards': 0, 'recognizer_forwards': 0,
                'candidate_displacement_trials': 0, 'optimizer_updates': 0, 'committed_trajectory_updates': 0, 'new_gradient_queries': 0,
                'backwards': 0, 'epochs': 0, 'raw_outputs': 0, 'optimizer_constructed': False}
    try:
        assert shutil.disk_usage(root).free >= p['budgets']['minimum_free_disk_bytes'], 'Need6GiB free; no cleanup in this probe'
        old, basis, parent, active, closed, mixed, helper = dependencies(root, p)
        guard, definitions, filters = helper.original_functions(parent); guard(root, idle=True)
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        for path in [root, parent, active, closed]: sys.path.insert(0, str(path))
        from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_mean_centered_decoder_v29 import MeanCenteredOriginalDecoderV29
        from cctv_dgp_app_input_v28 import canonical_tensor
        from cctv_dgp_batchmatched_identity_v26 import objective_terms
        from cctv_dgp_finite_clearance_geometry_v36 import verify_geometry
        from cctv_dgp_loss_cone_probe_v33_metrics import case_metrics, groups, compare_groups
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic() - started <= p['budgets']['worker_seconds'], 'V36 worker900s cap'
            assert torch.cuda.max_memory_allocated() <= p['budgets']['peak_vram_bytes'], 'Allocated VRAM20GiB cap'
            for key, maximum in p['forward_call_limits'].items(): assert progress[key] <= maximum, key
        original, _ = load_frozen_dgp_restorer(parent / 'weights/dgp_v2.pth', expected_sha256=basis['original_checkpoint_sha256'], device='cuda')
        candidate = MeanCenteredOriginalDecoderV29(original.net).eval().requires_grad_(False)
        identity = FixedObservedIdentity(parent / 'weights/w600k_r50.onnx', 'cuda').eval().requires_grad_(False)
        assert state_hash(original.net) == basis['original_DGP_state'] and state_hash(identity) == basis['recognizer_state']
        for model, key in [(original.net, 'reference_DGP_forwards'), (candidate.net, 'candidate_DGP_forwards'), (identity.encoder, 'recognizer_forwards')]:
            model.register_forward_hook(lambda *_args, key=key: progress.__setitem__(key, progress[key] + 1))
        selected_names = {row['name'] for row in p['parameter_layout']}
        selected = [(name, value) for name, value in candidate.net.named_parameters() if name in selected_names]
        assert [name for name, _ in selected] == [row['name'] for row in p['parameter_layout']]
        parameters = [value for _, value in selected]
        assert len(parameters) == 23 and sum(value.numel() for value in parameters) == 978243
        assert all(not value.requires_grad and value.grad is None for value in candidate.parameters())
        initial = {name: value.detach().clone() for name, value in candidate.net.state_dict().items()}
        refs = {r['id']: r for r in read(closed / 'protocol.json')['training_references']}
        keys = ['x', 'base', 'target', 'mask', 'feature', 'interior', 'valid7', 'grid', 'truth', 'degraded_weight', 'clear_weight']
        cohorts = []
        for group in p['cohorts']:
            items = []
            for case in group['cases']:
                clock()
                with Image.open(mixed / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                with Image.open(mixed / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                with Image.open(mixed / case['observed']) as im: mask = np.asarray(im).copy() > 0
                assert camera.shape == target.shape == (256, 256, 3) and mask.shape == (256, 256) and mask.any()
                def erode(radius):
                    size = 2 * radius + 1; sums = np.pad(np.pad(mask.astype(np.int64), radius).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                    return sums[size:, size:] - sums[:-size, size:] - sums[size:, :-size] + sums[:-size, :-size] == size * size
                interior = erode(6); feature = np.zeros((256, 256), bool)
                for point in case['landmarks5_canvas_xy']:
                    xx, yy = np.floor(point).astype(int); feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
                feature &= interior; assert feature.any()
                tensor = lambda a: torch.from_numpy(np.asarray(a).copy()).cuda()
                items.append({'case': case, 'camera': camera, 'target8': target, 'mask8': mask, 'x': canonical_tensor(camera, 'cuda'),
                              'target': canonical_tensor(target, 'cuda'), 'mask': tensor(mask.astype(np.float32))[None, None],
                              'feature': tensor(feature.astype(np.float32))[None, None], 'interior': tensor(interior.astype(np.float32))[None, None],
                              'valid7': tensor(erode(3).astype(np.float32))[None, None], 'grid': tensor(grid112(refs[case['source_person_or_reference']]['matrix112']))[None],
                              'degraded_weight': tensor([0. if case['profile'] == 'clear' else 1.25]).float(),
                              'clear_weight': tensor([1. if case['profile'] == 'clear' else 0.]).float()})
            with torch.no_grad():
                for begin in range(0, 50, 5):
                    clock(); batch = items[begin:begin + 5]; mask = torch.cat([i['mask'] for i in batch]); x = torch.cat([i['x'] for i in batch])
                    baseline = torch.where(mask.bool(), original(x), x).detach().clone()
                    truth = identity.embedding(torch.cat([i['target'] for i in batch]), mask, torch.cat([i['grid'] for i in batch])).detach().clone()
                    for index, item in enumerate(batch): item['base'] = baseline[index:index + 1].clone(); item['truth'] = truth[index:index + 1].clone()
            cohorts.append((group['name'], items))
        namespace = {'torch': torch, 'F': F}
        exec(compile(ast.Module(body=filters, type_ignores=[]), '<pinned-original-fixed-filters>', 'exec'), namespace)
        class Fixed: pass
        fixed = Fixed(); z = torch.arange(-6, 7, dtype=torch.float32); kernel = torch.exp(-.5 * (z / 2).square()); fixed.kernel = (kernel / kernel.sum()).cuda()
        fixed.reflect_indices = torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))).cuda()
        fixed.blur = MethodType(namespace['blur'], fixed); fixed.high = MethodType(namespace['high'], fixed)
        namespace.update({'head': fixed, 'identity': identity})
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-unchanged-seven-losses>', 'exec'), namespace)
        normalizers = tuple(torch.tensor(value, dtype=torch.float32, device='cuda') for value in basis['normalizers'])
        def flat(): return torch.cat([value.detach().reshape(-1) for value in parameters]).cpu().numpy().copy()
        def assign(array):
            assert array.dtype == np.float32 and array.shape == (978243,) and np.isfinite(array).all()
            with torch.no_grad():
                for value, row in zip(parameters, p['parameter_layout']): value.copy_(torch.from_numpy(array[row['start']:row['end']].copy()).cuda().reshape(value.shape))
        def assert_frozen(state):
            assert state_hash(original.net) == basis['original_DGP_state'] and state_hash(identity) == basis['recognizer_state']
            for name, value in candidate.net.state_dict().items():
                if name not in selected_names: assert torch.equal(value, state[name]), name
            assert all(not value.requires_grad and value.grad is None for model in [original, candidate, identity] for value in model.parameters())
        def snapshot(folder, items, state, label, variant):
            folder.mkdir(); rows, raw_values, batches = [], [], []
            with torch.no_grad():
                for begin in range(0, 50, 5):
                    clock(); group = items[begin:begin + 5]; b = {key: torch.cat([item[key] for item in group]) for key in keys}
                    prediction = candidate(b['x'], b['mask'], b['base'])
                    assert not prediction.requires_grad and bool(torch.isfinite(prediction).all())
                    if state == 0 and variant == 'before': assert torch.equal(prediction, b['base'])
                    raw = prediction.permute(0, 2, 3, 1).cpu().numpy().copy()
                    pngs = np.stack([np.where(item['mask8'][..., None], np.floor(a * np.float32(255)), item['camera']).astype(np.uint8) for item, a in zip(group, raw)])
                    vectors = identity.embedding(torch.cat([canonical_tensor(a, 'cuda') for a in pngs]), b['mask'], b['grid']).cpu().numpy().copy()
                    terms = objective_terms(b, prediction, identity, namespace['mean'], namespace['feature_errors'], namespace['ssim'], normalizers)
                    assert list(terms) == p['terms']
                    values = torch.stack([terms[name] for name in p['terms']], dim=1).cpu().numpy().copy()
                    raw_vectors = identity.embedding(prediction, b['mask'], b['grid']).cpu().numpy().copy()
                    assert values.shape == (5, 7) and np.isfinite(values).all() and (values >= 0).all()
                    raw_values.extend(values.tolist()); batches.append({'ids': [i['case']['id'] for i in group], 'values': values.mean(0).astype(np.float64).tolist()})
                    for item, a, png, vector, raw_vector, value in zip(group, raw, pngs, vectors, raw_vectors, values):
                        cid = item['case']['id']; np.save(folder / (cid + '.npy'), a, allow_pickle=False); Image.fromarray(png).save(folder / (cid + '.png'))
                        np.save(folder / (cid + '_embedding.npy'), vector, allow_pickle=False); np.save(folder / (cid + '_raw_embedding.npy'), raw_vector, allow_pickle=False)
                        truth = item['truth'][0].cpu().numpy().copy()
                        if variant == 'before': np.save(folder / (cid + '_target_embedding.npy'), truth, allow_pickle=False)
                        baseline = item['base'][0].permute(1, 2, 0).cpu().numpy().copy()
                        metrics = case_metrics(a, png, item['target8'], item['camera'], item['mask8'], item['feature'][0, 0].cpu().numpy() > 0, baseline, vector, raw_vector, truth)
                        rows.append({'id': cid, 'source': item['case']['source'], 'profile': item['case']['profile'], 'raw_terms': value.tolist(), 'metrics': metrics})
                        if variant == 'before':
                            historical = old / f'outputs/state{state}_{label}'
                            assert np.max(np.abs(a - np.load(historical / (cid + '.npy'), allow_pickle=False))) <= p['same_VM_raw_tolerance']
                            with Image.open(historical / (cid + '.png')) as im: prior_png = np.asarray(im.convert('RGB')).copy()
                            assert np.abs(png.astype(int) - prior_png.astype(int)).max() <= 1
                        progress['raw_outputs'] += 1
            values = np.asarray(raw_values, np.float64).mean(0)
            receipt = {'complete': True, 'state': state, 'cohort': label, 'variant': variant, 'cases': 50, 'rows': rows,
                       'groups': groups(rows), 'raw_component_means': values.tolist(), 'raw_objective': float(values.sum()),
                       'batches': batches, 'candidate_state': state_hash(candidate.net), 'no_display_processing': True}
            write(folder / 'receipt.json', receipt); print(json.dumps({'V36_state': state, 'cohort': label, 'variant': variant, 'raw_outputs': progress['raw_outputs']}), flush=True)
            return receipt
        shipped_theta,direction,guards,geometry=verify_geometry(root,p,sha,read)
        assert np.array_equal(flat(),shipped_theta)
        write(out/'geometry_verification.json',geometry)
        before_receipts,receipts={},[]
        for label,items in cohorts:
            folder=out/f'state0_{label}';folder.mkdir()
            np.save(folder/'theta_before.npy',shipped_theta,allow_pickle=False)
            before_receipts[label]=snapshot(folder/'before',items,0,label,'before')
        for variant in p['variants']:
            clock();candidate.net.load_state_dict(initial,strict=True)
            actual_theta=(shipped_theta.astype(np.float64)-variant['scale']*direction).astype(np.float32)
            assign(actual_theta);assert_frozen(initial)
            progress['candidate_displacement_trials']+=1
            actual_delta=shipped_theta.astype(np.float64)-flat().astype(np.float64)
            for label,items in cohorts:
                folder=out/f'state0_{label}'/variant['name']
                current=snapshot(folder,items,0,label,variant['name'])
                before=before_receipts[label]
                decision=compare_groups(before['groups'],current['groups'])
                comparison={'variant':variant,'preservation_against_original':decision,
                    'incremental_degraded_PNG_structure_gain':1-current['groups']['degraded']['landmark_high_frequency_MSE']/before['groups']['degraded']['landmark_high_frequency_MSE'],
                    'finite_raw_component_change':(np.asarray(current['raw_component_means'])-np.asarray(before['raw_component_means'])).tolist(),
                    'all108_linear_function_changes':(-(guards@actual_delta)).tolist(),
                    'float32_roundoff_norm':float(np.linalg.norm(actual_delta-variant['scale']*direction)),
                    'capacity_qualification':False,'app_promotion':False,'independent_audit_and_visual_review_required':True}
                write(folder/'comparison.json',comparison)
                receipts.append({'state':0,'cohort':label,'variant':variant['name'],'receipt_sha256':sha(folder/'receipt.json')})
            candidate.net.load_state_dict(initial,strict=True)
            assert np.array_equal(flat(),shipped_theta) and state_hash(candidate.net)==basis['original_DGP_state']
        clock();dependencies(root,p);assert_frozen(initial)
        assert progress['candidate_displacement_trials']==4 and progress['raw_outputs']==500
        assert {k:progress[k] for k in p['forward_call_limits']}==p['forward_call_limits']
        assert progress['optimizer_updates']==progress['new_gradient_queries']==progress['backwards']==progress['committed_trajectory_updates']==progress['epochs']==0
        write(out/'results.json',{'complete':True,'protocol_sha256':pin,'seconds':time.monotonic()-started,**progress,
            'receipts':receipts,'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),
            'original_DGP_state':state_hash(original.net),'recognizer_state':state_hash(identity),
            'new_checkpoint_created':False,'all_trial_states_reset':True,'no_follow_on_training':True,
            'native_or_reserved_used':False,'app_promotion':False,'goal_complete':False})
        print(json.dumps({'complete':True,'disposable_trials':4,'optimizer_updates':0,'raw_outputs':500,'seconds':time.monotonic()-started}),flush=True)
    except BaseException as error:
        write(out/'failure.json',{'complete':False,'protocol_sha256':pin,'seconds':time.monotonic()-started,**progress,
            'cause':str(error),'traceback':traceback.format_exc(),'resume_permitted':False,
            'new_checkpoint_created':False,'app_promotion':False,'goal_complete':False})
        raise


def export(root, p, pin):
    scope(root); started = time.monotonic(); destination = Path.home() / (STEM + '-results.tar.gz')
    assert not destination.exists() and not Path(str(destination) + '.sha256').exists() and not (root / 'export_manifest.json').exists(), 'Preserve previous export'
    files = [root / 'protocol.json'] + [root / name for name in p['assets_sha256']]
    if (root / 'outputs').exists(): files.extend(f for f in (root / 'outputs').rglob('*') if f.is_file())
    files.extend(root / name for name in ['probe.log', 'probe_exit_code.txt', 'supervisor_receipt.json'] if (root / name).exists())
    assert all(not f.is_symlink() and f.resolve().is_relative_to(root) for f in files)
    size = sum(f.stat().st_size for f in files); assert size <= p['budgets']['export_uncompressed_bytes']
    assert shutil.disk_usage(root).free >= size + 16 * 1024**2, 'Need space to retain evidence and package it'
    write(root / 'export_manifest.json', {'complete': True, 'protocol_sha256': pin, 'files_sha256': {f.relative_to(root).as_posix(): sha(f) for f in files}, 'training_success_not_implied': True})
    files.append(root / 'export_manifest.json')
    with tarfile.open(destination, 'x:gz', compresslevel=3) as archive:
        for f in files:
            assert time.monotonic() - started < p['budgets']['export_seconds'], 'V36 export300s cap'
            archive.add(f, arcname='cctv_dgp_finite_clearance_probe_v36_return/' + f.relative_to(root).as_posix(), recursive=False)
    digest = sha(destination)
    with Path(str(destination) + '.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + destination.name + '\n')
    terminal = read(root / 'outputs/results.json') if (root / 'outputs/results.json').exists() else read(root / 'outputs/failure.json')
    receipt = {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size, 'seconds': time.monotonic() - started,
               'optimizer_updates': terminal['optimizer_updates'], 'candidate_displacement_trials':terminal['candidate_displacement_trials'], 'committed_trajectory_updates': 0, 'training_success_not_implied': True,
               'run_results_present': (root / 'outputs/results.json').exists(), 'failure_present': (root / 'outputs/failure.json').exists()}
    write(Path.home() / (STEM + '-export.json'), receipt); print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path); parser.add_argument('--protocol-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ['verify-transfer', 'run', 'export', 'record-supervision']: mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--elapsed', type=float); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); root = args.root.resolve(); scope(root); p = verify(root, args.protocol_sha)
    if args.verify_transfer:
        dependencies(root, p); print(json.dumps({'complete': True, 'assets': len(p['assets_sha256']), 'TRAIN_cases': 100, 'neural_calls': 0, 'optimizer_updates': 0})); return
    if args.record_supervision:
        assert args.elapsed is not None and args.elapsed >= 0 and args.exit_code is not None
        write(root / 'supervisor_receipt.json', {'complete': True, 'protocol_sha256': args.protocol_sha, 'seconds': args.elapsed,
                                               'cap_seconds': p['budgets']['external_seconds'], 'kill_grace_seconds': 30,
                                               'within_external_bound': args.elapsed <= p['budgets']['external_seconds'] + 30, 'probe_exit_code': args.exit_code}); return
    if args.export: export(root, p, args.protocol_sha); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('V36 worker900s deadline')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('V36 external deadline')))
    signal.alarm(p['budgets']['worker_seconds']); run(root, p, args.protocol_sha)


if __name__ == '__main__': main()

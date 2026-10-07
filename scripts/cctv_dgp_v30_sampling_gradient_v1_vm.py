"""Finite sampling-matched gradient measurements on the existing L4; no optimizer."""
import argparse
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

NAME = 'cctv_dgp_v30_sampling_gradient_v1_vm'
STEM = 'cctv-dgp-v30-sampling-gradient-v1'
V30_PIN = 'b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scope(root):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing Linux VM only; no local differentiation'
    assert root == (Path.home() / 'forensic-dgp' / NAME).resolve(), 'Use the distinct packet below ~/forensic-dgp'


def check_files(root, mapping):
    for name, digest in mapping.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file() and not path.is_symlink() and sha(path) == digest, name


def verify(root, pin):
    assert sha(root / 'protocol.json') == pin, 'Diagnostic protocol changed'
    p = read(root / 'protocol.json')
    assert p['format'] == 'V30-sampling-matched-zero-update-gradient-diagnostic-v1'
    assert p['optimizer_updates'] == p['backwards'] == p['epochs'] == 0
    assert p['gradient_queries'] == 280 and len(p['cohorts']) == 2
    assert p['selected_parameters'] == 498627 and p['selected_tensors'] == 12
    assert len(p['terms']) == 7
    check_files(root, p['assets_sha256'])
    return p


def matched_unexposed_cases(exposed, all_cases, touched):
    """Match source/profile and reference repetition using frozen metadata only."""
    pairs, matched, chosen = {}, [], set()
    for case in exposed:
        ref = case['source_person_or_reference']
        candidates = [c for c in all_cases if c['source'] == case['source'] and c['profile'] == case['profile']
                      and c['source_person_or_reference'] not in touched and c['id'] not in chosen]
        if ref in pairs:
            candidates = [c for c in candidates if c['source_person_or_reference'] == pairs[ref]]
        else:
            candidates = [c for c in candidates if c['source_person_or_reference'] not in pairs.values()]
        assert candidates, 'Insufficient unexposed TRAIN references for metadata matching'
        match = candidates[0]; pairs[ref] = match['source_person_or_reference']
        matched.append(match); chosen.add(match['id'])
    assert len(chosen) == len(exposed) and len(set(pairs.values())) == len(pairs)
    return matched, pairs


def dependencies(root, p):
    parent = root.parent / 'cctv_dgp_feature_skips_vm_v27'
    original_path = root.parent / 'cctv_dgp_active_original_decoder_vm_v28'
    closed = root.parent / 'cctv_dgp_broader_mean_vm_v30'
    mixed = root.parent / 'cctv_dgp_mixed_vm_v9_r2'
    assert sha(closed / 'protocol.json') == V30_PIN == p['V30_protocol_sha256']
    check_files(closed, p['V30_dependencies_sha256'])
    check_files(parent, p['parent_dependencies_sha256'])
    check_files(original_path, p['active_decoder_dependencies_sha256'])
    check_files(mixed, p['TRAIN_assets_sha256'])
    v30 = read(closed / 'protocol.json')
    assert v30['terms'] == p['terms'] and v30['parameter_layout'] == p['parameter_layout']
    assert v30['original_DGP_state'] == p['original_DGP_state']
    failure = read(closed / 'outputs/failure.json')
    assert failure['optimizer_updates'] == 50 and failure['cause'] == 'No one-percent early structural gain; retain stop'
    early = read(closed / 'outputs/early_structure_stop.json')
    assert not early['pass'] and early['minimum'] == .01 and not (closed / 'outputs/results.json').exists()
    assert failure['resume_permitted'] is False
    normalizers = read(closed / 'outputs/cohort_loss_setup.json')
    assert normalizers['feature_normalizer'] == p['normalizers'][0] and normalizers['interior_normalizer'] == p['normalizers'][1]
    assert normalizers['cases'] == 50 and normalizers['clear_controls'] == 10
    # Selection is derived from the frozen schedule and input metadata only.
    schedule = read(closed / 'schedule.json')['batches']
    actual_seen = [v30['case_rows'][i] for batch in schedule[:10] for i in batch]
    assert p['cohorts'][0]['cases'] == actual_seen and len(actual_seen) == 50
    touched = {v30['case_rows'][i]['source_person_or_reference'] for batch in schedule[:50] for i in batch}
    all_cases = {c['id']: c for c in v30['case_rows']}
    assert len(all_cases) == 3905 and len(touched) == 218
    matched, pairs = matched_unexposed_cases(actual_seen, v30['case_rows'], touched)
    assert p['cohorts'][1]['cases'] == matched and p['matched_reference_pairs'] == pairs and len(matched) == 50
    assert all(c['role'] == 'train' and all_cases[c['id']] == c for group in p['cohorts'] for c in group['cases'])
    spec = importlib.util.spec_from_file_location('pinned_V30_measurement_helpers', closed / 'scripts/cctv_dgp_broader_mean_v30_vm.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return parent, original_path, closed, mixed, helper


def run(root, p, pin):
    scope(root)
    assert not (root / 'outputs').exists(), 'Preserve prior or partial diagnostic; no resume'
    start = time.monotonic()
    progress = {'reference_DGP_forwards': 0, 'candidate_DGP_forwards': 0, 'recognizer_forwards': 0,
                'component_gradient_calls': 0, 'optimizer_updates': 0, 'backwards': 0, 'epochs': 0}
    out = root / 'outputs'; out.mkdir()
    try:
        assert shutil.disk_usage(root).free >= p['budgets']['minimum_free_disk_bytes'], 'Need4GiB free; no deletion by this diagnostic'
        parent, original_path, closed, mixed, helper = dependencies(root, p)
        guard, definitions, filters = helper.original_functions(parent)
        guard(root, idle=True)
        import ast
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        for path in [parent, original_path, closed]: sys.path.insert(0, str(path))
        from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_mean_centered_decoder_v29 import MeanCenteredOriginalDecoderV29
        from cctv_dgp_app_input_v28 import canonical_tensor
        from cctv_dgp_batchmatched_identity_v26 import objective_terms
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic() - start <= 600, 'Sampling diagnostic600s cap'
            assert torch.cuda.max_memory_allocated() <= 20 * 1024**3, 'Allocated VRAM20GiB cap'
        original, _ = load_frozen_dgp_restorer(parent / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cuda')
        assert state_hash(original.net) == p['original_DGP_state']
        candidate = MeanCenteredOriginalDecoderV29(original.net)
        identity = FixedObservedIdentity(parent / 'weights/w600k_r50.onnx', 'cuda')
        assert state_hash(identity) == p['recognizer_state']
        for model, key in [(original.net, 'reference_DGP_forwards'), (candidate.net, 'candidate_DGP_forwards'), (identity.encoder, 'recognizer_forwards')]:
            model.register_forward_hook(lambda *_args, key=key: progress.__setitem__(key, progress[key] + 1))
        selected = [(name, value) for name, value in candidate.net.named_parameters() if value.requires_grad]
        parameters = [v for _, v in selected]
        layout, offset = [], 0
        for name, value in selected:
            layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()}); offset += value.numel()
        assert layout == p['parameter_layout'] and len(parameters) == 12 and offset == 498627
        initial_state = {name: value.detach().clone() for name, value in candidate.net.state_dict().items()}
        stopped_state = torch.load(closed / 'outputs/update50/dgp_candidate_v30.pth', map_location='cuda', weights_only=True)
        assert set(stopped_state) == set(initial_state)
        for name, value in stopped_state.items():
            assert value.shape == initial_state[name].shape and value.dtype == initial_state[name].dtype and bool(torch.isfinite(value).all())
            if name not in {r['name'] for r in layout}: assert torch.equal(value, initial_state[name]), name
        assert read(closed / 'outputs/update50/metrics.json')['candidate_DGP_state'] == p['stopped50_DGP_state']
        refs = {r['id']: r for r in read(closed / 'protocol.json')['training_references']}
        keys = ['x', 'base', 'target', 'mask', 'feature', 'interior', 'valid7', 'grid', 'truth', 'degraded_weight', 'clear_weight']
        cohorts = []
        for group in p['cohorts']:
            items = []
            for case in group['cases']:
                clock()
                with Image.open(mixed / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                with Image.open(mixed / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                with Image.open(mixed / case['observed']) as im: support = np.asarray(im).copy() > 0
                assert camera.shape == target.shape == (256, 256, 3) and support.shape == (256, 256) and support.any()
                def erode(radius):
                    size = 2 * radius + 1; summed = np.pad(np.pad(support.astype(np.int64), radius).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                    return summed[size:, size:] - summed[:-size, size:] - summed[size:, :-size] + summed[:-size, :-size] == size * size
                interior = erode(6); feature = np.zeros((256, 256), bool)
                for point in case['landmarks5_canvas_xy']:
                    xx, yy = np.floor(point).astype(int); feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
                feature &= interior; assert feature.any()
                t = lambda a: torch.from_numpy(np.asarray(a).copy()).cuda()
                items.append({'case': case, 'camera': camera, 'mask8': support, 'x': canonical_tensor(camera, 'cuda'),
                              'target': canonical_tensor(target, 'cuda'), 'mask': t(support.astype(np.float32))[None, None],
                              'feature': t(feature.astype(np.float32))[None, None], 'interior': t(interior.astype(np.float32))[None, None],
                              'valid7': t(erode(3).astype(np.float32))[None, None], 'grid': t(grid112(refs[case['source_person_or_reference']]['matrix112']))[None],
                              'degraded_weight': t([0. if case['profile'] == 'clear' else 1.25]).float(),
                              'clear_weight': t([1. if case['profile'] == 'clear' else 0.]).float()})
            for begin in range(0, 50, 5):
                clock(); batch = items[begin:begin + 5]
                x = torch.cat([i['x'] for i in batch]); mask = torch.cat([i['mask'] for i in batch])
                with torch.no_grad():
                    raw = torch.where(mask.bool(), original(x), x).detach().clone()
                    truth = identity.embedding(torch.cat([i['target'] for i in batch]), mask, torch.cat([i['grid'] for i in batch])).detach().clone()
                for slot, item in enumerate(batch): item['base'] = raw[slot:slot + 1].clone(); item['truth'] = truth[slot:slot + 1].clone()
            cohorts.append((group['name'], items))
        ns = {'torch': torch, 'F': F}
        exec(compile(ast.Module(body=filters, type_ignores=[]), '<pinned-V27-fixed-filter>', 'exec'), ns)
        class Fixed: pass
        fixed = Fixed(); z = torch.arange(-6, 7, dtype=torch.float32)
        kernel = torch.exp(-.5 * (z / 2).square()); fixed.kernel = (kernel / kernel.sum()).cuda()
        fixed.reflect_indices = torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))).cuda()
        fixed.blur = MethodType(ns['blur'], fixed); fixed.high = MethodType(ns['high'], fixed)
        ns.update({'head': fixed, 'identity': identity})
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-V27-unchanged-loss>', 'exec'), ns)
        normalizers = tuple(torch.tensor(value, dtype=torch.float32, device='cuda') for value in p['normalizers'])
        receipts = []
        for state_number, state in [(0, initial_state), (50, stopped_state)]:
            candidate.net.load_state_dict(state, strict=True)
            expected_state = p['original_DGP_state'] if state_number == 0 else p['stopped50_DGP_state']
            assert state_hash(candidate.net) == expected_state
            for label, items in cohorts:
                clock(); folder = out / f'state{state_number}_{label}'; folder.mkdir()
                total = np.zeros((7, 498627), np.float64); values = np.zeros(7, np.float64); batch_rows = []; replay_rows = []
                for begin in range(0, 50, 5):
                    clock(); group = items[begin:begin + 5]; b = {key: torch.cat([item[key] for item in group]) for key in keys}
                    pred = candidate(b['x'], b['mask'], b['base'])
                    assert pred.requires_grad and not torch.is_inference(pred)
                    if state_number == 0: assert torch.equal(pred.detach(), b['base'])
                    raw = pred.detach().permute(0, 2, 3, 1).cpu().numpy().copy()
                    delivered = np.stack([np.where(item['mask8'][..., None], np.floor(a * np.float32(255)), item['camera']).astype(np.uint8) for item, a in zip(group, raw)])
                    with torch.no_grad(): vectors = identity.embedding(torch.cat([canonical_tensor(a, 'cuda') for a in delivered]), b['mask'], b['grid']).cpu().numpy().copy()
                    for item, a, png, vector in zip(group, raw, delivered, vectors):
                        cid = item['case']['id']; saved_png_path = closed / f'outputs/update{state_number}/{cid}.png'
                        with Image.open(saved_png_path) as im: saved_png = np.asarray(im.convert('RGB')).copy()
                        byte_error = int(np.abs(png.astype(int) - saved_png.astype(int)).max())
                        assert byte_error <= 1, ('Fresh same-VM batch-context parity', state_number, cid)
                        old_vector = np.load(closed / f'outputs/update{state_number}/{cid}_embedding.npy', allow_pickle=False)
                        truth = np.load(closed / f'outputs/update0/{cid}_target_embedding.npy', allow_pickle=False)
                        vector_error = float(np.abs(vector - old_vector).max())
                        truth_error = float(np.abs(item['truth'][0].cpu().numpy() - truth).max())
                        assert max(vector_error, truth_error) <= 5e-5
                        np.save(folder / (cid + '.npy'), a, allow_pickle=False); Image.fromarray(png).save(folder / (cid + '.png'))
                        np.save(folder / (cid + '_embedding.npy'), vector, allow_pickle=False)
                        if state_number == 0: np.save(folder / (cid + '_target_embedding.npy'), item['truth'][0].cpu().numpy().copy(), allow_pickle=False)
                        replay_rows.append({'id': cid, 'historical_PNG_maximum_byte_error': byte_error,
                                            'historical_vector_maximum_error': vector_error, 'historical_truth_maximum_error': truth_error})
                    terms = objective_terms(b, pred, identity, ns['mean'], ns['feature_errors'], ns['ssim'], normalizers)
                    assert list(terms) == p['terms']
                    matrix = np.empty((7, 498627), np.float32); scalar_rows = []
                    for term_index, term in enumerate(p['terms']):
                        clock(); scalar = terms[term].mean() / 10
                        assert bool(torch.isfinite(scalar))
                        pieces = torch.autograd.grad(scalar, parameters, retain_graph=term_index < 6, create_graph=False, allow_unused=False)
                        progress['component_gradient_calls'] += 1
                        assert len(pieces) == 12 and all(bool(torch.isfinite(g).all()) for g in pieces)
                        if state_number == 0 and term_index >= 3:
                            assert float(scalar.detach()) == 0 and all(torch.count_nonzero(g) == 0 for g in pieces)
                        matrix[term_index] = torch.cat([g.detach().reshape(-1) for g in pieces]).cpu().numpy()
                        value = float(scalar.detach()); values[term_index] += value; scalar_rows.append(value)
                    total += matrix.astype(np.float64)
                    gradient_path = folder / f'batch{begin // 5}.npy'; np.save(gradient_path, matrix, allow_pickle=False)
                    batch_rows.append({'batch': begin // 5, 'ids': [item['case']['id'] for item in group],
                                       'values': scalar_rows, 'norms': np.linalg.norm(matrix.astype(np.float64), axis=1).tolist(), 'sha256': sha(gradient_path)})
                    assert state_hash(candidate.net) == expected_state and all(v.grad is None for v in candidate.parameters())
                    print(json.dumps({'state': state_number, 'cohort': label, 'batch': begin // 5 + 1,
                                      'gradient_queries': progress['component_gradient_calls'], 'seconds': time.monotonic() - start}), flush=True)
                np.save(folder / 'gradient_components.npy', total, allow_pickle=False)
                norms = np.linalg.norm(total, axis=1); gram = total @ total.T; denominator = norms[:, None] * norms[None, :]
                cosines = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
                receipt = {'complete': True, 'state': state_number, 'cohort': label, 'candidate_state': expected_state,
                           'cases': 50, 'gradient_queries': 70, 'batches': batch_rows, 'replay': replay_rows,
                           'component_values': values.tolist(), 'objective': float(values.sum()), 'component_norms': norms.tolist(),
                           'component_gram': gram.tolist(), 'component_cosines': cosines.tolist(),
                           'aggregate_sha256': sha(folder / 'gradient_components.npy')}
                write(folder / 'receipt.json', receipt); receipts.append({'state': state_number, 'cohort': label, 'receipt_sha256': sha(folder / 'receipt.json')})
        clock(); dependencies(root, p)
        assert progress == {'reference_DGP_forwards': 20, 'candidate_DGP_forwards': 40, 'recognizer_forwards': 100,
                            'component_gradient_calls': 280, 'optimizer_updates': 0, 'backwards': 0, 'epochs': 0}
        assert state_hash(original.net) == p['original_DGP_state'] and state_hash(candidate.net) == p['stopped50_DGP_state'] and state_hash(identity) == p['recognizer_state']
        assert all(not v.requires_grad and v.grad is None for m in [original, identity] for v in m.parameters())
        assert all(v.grad is None for v in candidate.parameters())
        write(out / 'results.json', {'complete': True, 'protocol_sha256': pin, 'seconds': time.monotonic() - start, **progress,
                                     'receipts': receipts, 'normalizers': p['normalizers'], 'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(),
                                     'optimizer_constructed': False, 'new_checkpoint_created': False, 'DGP_and_recognizer_unmodified': True,
                                     'V30_failure_retained': True, 'no_new_training_recipe': True, 'native_or_reserved_used': False,
                                     'app_promotion': False, 'goal_complete': False})
        print(json.dumps({'complete': True, 'gradient_queries': 280, 'optimizer_updates': 0, 'seconds': time.monotonic() - start}), flush=True)
    except BaseException as error:
        write(out / 'failure.json', {'complete': False, 'protocol_sha256': pin, 'seconds': time.monotonic() - start, **progress,
                                     'cause': str(error), 'traceback': traceback.format_exc(), 'resume_permitted': False,
                                     'optimizer_constructed': False, 'new_checkpoint_created': False, 'app_promotion': False, 'goal_complete': False})
        raise


def export(root, p, pin):
    scope(root); start = time.monotonic(); destination = Path.home() / (STEM + '-results.tar.gz')
    assert not destination.exists() and not Path(str(destination) + '.sha256').exists(), 'Preserve previous export'
    files = [root / 'protocol.json'] + [root / name for name in p['assets_sha256']]
    if (root / 'outputs').exists(): files.extend(f for f in (root / 'outputs').rglob('*') if f.is_file())
    files.extend(root / name for name in ['diagnostic.log', 'diagnostic_exit_code.txt', 'supervisor_receipt.json'] if (root / name).exists())
    assert all(not f.is_symlink() and f.resolve().is_relative_to(root) for f in files)
    size = sum(f.stat().st_size for f in files); assert size <= p['budgets']['export_uncompressed_bytes']
    assert shutil.disk_usage(root).free >= size + 16 * 1024**2, 'Retain evidence if export space is insufficient'
    write(root / 'export_manifest.json', {'complete': True, 'protocol_sha256': pin, 'files_sha256': {f.relative_to(root).as_posix(): sha(f) for f in files}, 'training_success_not_implied': True})
    files.append(root / 'export_manifest.json')
    with tarfile.open(destination, 'x:gz', compresslevel=3) as archive:
        for f in files:
            assert time.monotonic() - start < 300, 'Diagnostic export300s cap'
            archive.add(f, arcname='cctv_dgp_v30_sampling_gradient_v1_return/' + f.relative_to(root).as_posix(), recursive=False)
    digest = sha(destination)
    with Path(str(destination) + '.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + destination.name + '\n')
    receipt = {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size, 'seconds': time.monotonic() - start,
               'optimizer_updates': 0, 'training_success_not_implied': True, 'run_results_present': (root / 'outputs/results.json').exists(), 'failure_present': (root / 'outputs/failure.json').exists()}
    write(Path.home() / (STEM + '-export.json'), receipt); print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path); parser.add_argument('--protocol-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ['verify-transfer', 'run', 'export', 'record-supervision']: mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--elapsed', type=float); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); root = args.root.resolve(); scope(root); p = verify(root, args.protocol_sha)
    if args.verify_transfer:
        dependencies(root, p); print(json.dumps({'complete': True, 'assets': len(p['assets_sha256']), 'TRAIN_cases': 100, 'gradient_queries': 0, 'optimizer_updates': 0})); return
    if args.record_supervision:
        assert args.elapsed is not None and args.elapsed >= 0 and args.exit_code is not None
        write(root / 'supervisor_receipt.json', {'complete': True, 'protocol_sha256': args.protocol_sha, 'seconds': args.elapsed,
                                               'cap_seconds': 900, 'kill_grace_seconds': 30, 'within_external_bound': args.elapsed <= 930, 'diagnostic_exit_code': args.exit_code}); return
    if args.export: export(root, p, args.protocol_sha); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic worker600s deadline')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic external deadline')))
    signal.alarm(600); run(root, p, args.protocol_sha)


if __name__ == '__main__':
    main()

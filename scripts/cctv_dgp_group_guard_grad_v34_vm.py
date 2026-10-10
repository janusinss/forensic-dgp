"""Original-state, non-hinged source/profile preservation gradients; VM only.

No optimizer is constructed. No parameters are updated. Historical workers
are loaded for hash/metadata validation and fixed definitions only.
"""
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

NAME = 'cctv_dgp_group_guard_grad_v34_vm'
STEM = 'cctv-dgp-group-guard-grad-v34'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''): value.update(block)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def scope(root):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing Linux VM only; no local gradients'
    assert root == (Path.home() / 'forensic-dgp' / NAME).resolve(), 'Distinct V34 folder required'


def check_files(root, mapping):
    for name, expected in mapping.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file() and not (root / name).is_symlink() and sha(path) == expected, name


def verify(root, pin):
    assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json')
    assert p['format'] == 'own-DGP-original-nonhinged-group-preservation-gradient-v34'
    assert p['optimizer_updates'] == p['backwards'] == p['epochs'] == p['parameter_updates'] == 0
    assert p['gradient_queries'] == 300 and p['cases'] == 100 and p['states'] == [0]
    assert p['guard_metrics'] == ['raw_MSE', 'one_minus_raw_SSIM', 'one_minus_raw_ArcFace']
    assert p['selected_parameters'] == 978243 and p['selected_tensors'] == 23
    assert p['budgets']['worker_seconds'] == 600 and p['budgets']['external_seconds'] == 630
    check_files(root, p['assets_sha256'])
    return p


def dependencies(root, p):
    basis_root = root.parent / 'cctv_dgp_v32_loss_gradient_v1_vm'
    check_files(basis_root, p['basis_VM_sha256'])
    basis = read(basis_root / 'protocol.json')
    assert sha(basis_root / 'protocol.json') == p['basis_protocol_sha256']
    assert basis['cohorts'] == p['cohorts'] and basis['parameter_layout'] == p['parameter_layout']
    helper = module('pinned_V32_readonly_dependencies', basis_root / 'scripts/cctv_dgp_v32_loss_gradient_v1_vm.py')
    parent, active, closed, mixed, source = helper.dependencies(root, basis)
    probe_root = root.parent / 'cctv_dgp_loss_cone_probe_v33_vm'
    check_files(probe_root, p['V33_original_readback_sha256'])
    result = read(probe_root / 'outputs/results.json')
    assert result['complete'] and result['optimizer_updates'] == 8 and result['committed_trajectory_updates'] == 0
    return basis, parent, active, closed, mixed, source, probe_root


def run(root, p, pin):
    scope(root); assert not (root / 'outputs').exists(), 'Retain any earlier or partial run; no resume'
    started = time.monotonic(); out = root / 'outputs'; out.mkdir()
    progress = {'reference_DGP_forwards': 0, 'candidate_DGP_forwards': 0, 'recognizer_forwards': 0,
                'gradient_queries': 0, 'optimizer_updates': 0, 'backwards': 0, 'epochs': 0, 'parameter_updates': 0}
    try:
        assert shutil.disk_usage(root).free >= p['budgets']['minimum_free_disk_bytes'], 'Need6GiB free; no cleanup by this diagnostic'
        basis, parent, active, closed, mixed, helper, probe = dependencies(root, p)
        guard, definitions, filters = helper.original_functions(parent); guard(root, idle=True)
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        for path in [parent, active, closed]: sys.path.insert(0, str(path))
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
        from cctv_dgp_mean_centered_decoder_v29 import MeanCenteredOriginalDecoderV29
        from cctv_dgp_app_input_v28 import canonical_tensor
        from cctv_dgp_batchmatched_identity_v26 import batchmatched_scores
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()

        def clock():
            torch.cuda.synchronize()
            assert time.monotonic() - started <= 600, 'V34 diagnostic600s cap'
            assert torch.cuda.max_memory_allocated() <= p['budgets']['peak_vram_bytes'], 'Allocated VRAM20GiB cap'
            for key, maximum in p['forward_call_limits'].items(): assert progress[key] <= maximum, key

        original, _ = load_frozen_dgp_restorer(parent / 'weights/dgp_v2.pth', expected_sha256=basis['original_checkpoint_sha256'], device='cuda')
        assert state_hash(original.net) == basis['original_DGP_state']
        candidate = MeanCenteredOriginalDecoderV29(original.net)
        for name, value in candidate.net.named_parameters():
            if name in basis['fusion_parameter_names']: value.requires_grad_(True)
        identity = FixedObservedIdentity(parent / 'weights/w600k_r50.onnx', 'cuda').eval().requires_grad_(False)
        assert state_hash(candidate.net) == basis['original_DGP_state'] and state_hash(identity) == basis['recognizer_state']
        selected = [(name, v) for name, v in candidate.net.named_parameters() if v.requires_grad]
        layout, offset = [], 0
        for name, v in selected:
            layout.append({'name': name, 'shape': list(v.shape), 'start': offset, 'end': offset + v.numel()}); offset += v.numel()
        assert layout == p['parameter_layout'] and offset == 978243 and len(selected) == 23
        parameters = [v for _, v in selected]
        for net, key in [(original.net, 'reference_DGP_forwards'), (candidate.net, 'candidate_DGP_forwards'), (identity.encoder, 'recognizer_forwards')]:
            net.register_forward_hook(lambda *_args, key=key: progress.__setitem__(key, progress[key] + 1))
        ns = {'torch': torch, 'F': F}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-raw-mean-SSIM-definitions>', 'exec'), ns)
        refs = {r['id']: r for r in read(closed / 'protocol.json')['training_references']}
        summaries = []
        for cohort in p['cohorts']:
            label = cohort['name']; folder = out / label; folder.mkdir(); rows, batches = [], []
            for begin in range(0, 50, 5):
                clock(); cases = cohort['cases'][begin:begin + 5]; items = []
                for case in cases:
                    with Image.open(mixed / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                    with Image.open(mixed / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                    with Image.open(mixed / case['observed']) as im: support = np.asarray(im).copy() > 0
                    assert camera.shape == target.shape == (256, 256, 3) and support.shape == (256, 256) and support.any()
                    summed = np.pad(np.pad(support.astype(np.int64), 3).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                    valid = summed[7:, 7:] - summed[:-7, 7:] - summed[7:, :-7] + summed[:-7, :-7] == 49
                    t = lambda a: torch.from_numpy(np.asarray(a).copy()).cuda()
                    items.append({'case': case, 'camera': camera, 'support': support, 'x': canonical_tensor(camera, 'cuda'),
                        'target': canonical_tensor(target, 'cuda'), 'mask': t(support.astype(np.float32))[None, None],
                        'valid7': t(valid.astype(np.float32))[None, None],
                        'grid': t(grid112(refs[case['source_person_or_reference']]['matrix112']))[None]})
                keys = ['x', 'target', 'mask', 'valid7', 'grid']; b = {key: torch.cat([i[key] for i in items]) for key in keys}
                with torch.no_grad():
                    b['base'] = torch.where(b['mask'].bool(), original(b['x']), b['x']).detach().clone()
                    b['truth'] = identity.embedding(b['target'], b['mask'], b['grid']).detach().clone()
                pred = candidate(b['x'], b['mask'], b['base']); assert pred.requires_grad and torch.equal(pred.detach(), b['base'])
                _, cosine = batchmatched_scores(b, pred, identity)
                values = torch.stack((ns['mean']((pred - b['target']).square(), b['mask']),
                    1 - ns['ssim'](pred, b['target'], b['valid7']), 1 - cosine), 1)
                assert values.shape == (5, 3) and bool(torch.isfinite(values).all())
                raw = pred.detach().permute(0, 2, 3, 1).cpu().numpy().copy()
                matrix = np.empty((15, 978243), np.float32); saved_rows = []
                for slot, item in enumerate(items):
                    cid = item['case']['id']; old = probe / f'outputs/state0_{label}/before/{cid}.npy'
                    prior_raw = np.load(old, allow_pickle=False); parity = float(np.abs(raw[slot] - prior_raw).max())
                    assert parity <= p['same_VM_raw_tolerance'], (cid, 'Original input context changed')
                    png = np.where(item['support'][..., None], np.floor(raw[slot] * np.float32(255)), item['camera']).astype(np.uint8)
                    np.save(folder / (cid + '.npy'), raw[slot], allow_pickle=False); Image.fromarray(png).save(folder / (cid + '.png'))
                    saved_rows.append({'id': cid, 'source': item['case']['source'], 'profile': item['case']['profile'],
                        'raw_guard_values': values[slot].detach().cpu().numpy().astype(np.float64).tolist(), 'raw_parity_maximum_error': parity})
                    for metric in range(3):
                        clock(); index = slot * 3 + metric
                        pieces = torch.autograd.grad(values[slot, metric], parameters, retain_graph=index < 14, create_graph=False, allow_unused=False)
                        progress['gradient_queries'] += 1
                        assert len(pieces) == 23 and all(bool(torch.isfinite(v).all()) for v in pieces)
                        matrix[index] = torch.cat([v.detach().reshape(-1) for v in pieces]).cpu().numpy().copy()
                        assert all(v.grad is None for v in parameters)
                destination = folder / f'batch{begin // 5}_guard_gradients.npy'; np.save(destination, matrix, allow_pickle=False)
                batch = {'batch': begin // 5, 'rows': saved_rows, 'shape': [15, 978243], 'gradient_dtype': 'float32',
                    'gradient_sha256': sha(destination), 'gradient_norms': np.linalg.norm(matrix.astype(np.float64), axis=1).tolist()}
                batches.append(batch); rows.extend(saved_rows)
                del pred, values, cosine, pieces, matrix, b, items
                assert state_hash(candidate.net) == basis['original_DGP_state'] and state_hash(original.net) == basis['original_DGP_state']
                print(json.dumps({'cohort': label, 'batch': begin // 5 + 1, 'of': 10, 'gradient_queries': progress['gradient_queries'],
                    'seconds': time.monotonic() - started}), flush=True)
            assert len(rows) == 50 and len(batches) == 10
            receipt = {'complete': True, 'cohort': label, 'state': 0, 'rows': rows, 'batches': batches,
                'guard_metrics': p['guard_metrics'], 'candidate_state': state_hash(candidate.net), 'optimizer_updates': 0,
                'original_batch_context_preserved': True, 'nonhinged_derivatives_not_new_training_losses': True}
            write(folder / 'receipt.json', receipt); summaries.append({'cohort': label, 'receipt_sha256': sha(folder / 'receipt.json')})
        clock(); dependencies(root, p)
        assert progress == {**p['forward_call_limits'], 'gradient_queries': 300, 'optimizer_updates': 0, 'backwards': 0, 'epochs': 0, 'parameter_updates': 0}
        assert state_hash(candidate.net) == state_hash(original.net) == basis['original_DGP_state'] and state_hash(identity) == basis['recognizer_state']
        assert all(v.grad is None for model in [original, candidate, identity] for v in model.parameters())
        write(out / 'results.json', {'complete': True, 'protocol_sha256': pin, **progress, 'seconds': time.monotonic() - started,
            'receipts': summaries, 'original_DGP_state': basis['original_DGP_state'], 'recognizer_state': basis['recognizer_state'],
            'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(), 'new_checkpoint_created': False,
            'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False})
        print(json.dumps({'complete': True, 'gradient_queries': 300, 'optimizer_updates': 0, 'seconds': time.monotonic() - started}), flush=True)
    except BaseException as error:
        write(out / 'failure.json', {'complete': False, 'protocol_sha256': pin, **progress, 'seconds': time.monotonic() - started,
            'cause': str(error), 'traceback': traceback.format_exc(), 'resume_permitted': False, 'new_checkpoint_created': False,
            'app_promotion': False, 'goal_complete': False})
        raise


def export(root, p, pin):
    scope(root); started = time.monotonic(); destination = Path.home() / (STEM + '-results.tar.gz')
    assert not destination.exists() and not Path(str(destination) + '.sha256').exists() and not (root / 'export_manifest.json').exists()
    files = [root / 'protocol.json'] + [root / name for name in p['assets_sha256']]
    if (root / 'outputs').exists(): files.extend(q for q in (root / 'outputs').rglob('*') if q.is_file())
    files.extend(root / name for name in ['diagnostic.log', 'diagnostic_exit_code.txt', 'supervisor_receipt.json'] if (root / name).exists())
    assert all(not q.is_symlink() and q.resolve().is_relative_to(root) for q in files)
    size = sum(q.stat().st_size for q in files); assert size <= p['budgets']['export_uncompressed_bytes']
    assert shutil.disk_usage(root).free >= size + 16 * 1024 ** 2
    write(root / 'export_manifest.json', {'complete': True, 'protocol_sha256': pin,
        'files_sha256': {q.relative_to(root).as_posix(): sha(q) for q in files}, 'training_success_not_implied': True})
    files.append(root / 'export_manifest.json')
    with tarfile.open(destination, 'x:gz', compresslevel=3) as archive:
        for q in files:
            assert time.monotonic() - started < 300, 'V34 export300s cap'
            archive.add(q, arcname='cctv_dgp_group_guard_grad_v34_return/' + q.relative_to(root).as_posix(), recursive=False)
    digest = sha(destination)
    with Path(str(destination) + '.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + destination.name + '\n')
    success = (root / 'outputs/results.json').exists()
    write(Path.home() / (STEM + '-export.json'), {'complete': True, 'archive_sha256': digest,
        'bytes': destination.stat().st_size, 'seconds': time.monotonic() - started, 'optimizer_updates': 0,
        'run_results_present': success, 'failure_present': (root / 'outputs/failure.json').exists(), 'training_success_not_implied': True})
    print(json.dumps(read(Path.home() / (STEM + '-export.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path); parser.add_argument('--protocol-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ['verify-transfer', 'run', 'export', 'record-supervision']: mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--elapsed', type=float); parser.add_argument('--exit-code', type=int)
    a = parser.parse_args(); root = a.root.resolve(); scope(root); p = verify(root, a.protocol_sha)
    if a.verify_transfer:
        dependencies(root, p); print(json.dumps({'complete': True, 'cases': 100, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0})); return
    if a.record_supervision:
        assert a.elapsed is not None and a.elapsed >= 0 and a.exit_code is not None
        write(root / 'supervisor_receipt.json', {'complete': True, 'protocol_sha256': a.protocol_sha, 'seconds': a.elapsed,
            'cap_seconds': 630, 'kill_grace_seconds': 30, 'within_external_bound': a.elapsed <= 660, 'diagnostic_exit_code': a.exit_code}); return
    if a.export: export(root, p, a.protocol_sha); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('V34 worker600s deadline')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('V34 external deadline')))
    signal.alarm(600); run(root, p, a.protocol_sha)


if __name__ == '__main__': main()

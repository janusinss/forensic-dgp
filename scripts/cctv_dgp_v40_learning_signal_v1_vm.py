"""Finite VM-only endpoint diagnostic. No optimizer, backwards or parameter update."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import signal
import shutil
import sys
import tarfile
import time
import traceback
from types import MethodType, SimpleNamespace

NAME = 'cctv_dgp_v40_learning_signal_v1_vm'
STEM = 'cctv-dgp-v40-learning-signal-v1'
RETURNED_ASSETS = {'scripts/cctv_dgp_v40_learning_signal_v1_vm.py', 'cctv_dgp_v40_learning_signal_v1_decoder.py',
                   'frozen_definitions.py', 'untrained_initial_decoder.pth', 'stopped_decoder_input.pth'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scope(root):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing Linux VM only; no local gradients'
    assert root == (Path.home() / 'forensic-dgp' / NAME).resolve(), 'Distinct diagnostic root required'


def verify(root, pin):
    assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json')
    assert p['format'] == 'own-DGP-V40-endpoint-learning-signal-v1'
    assert p['optimizer_updates'] == p['parameter_updates'] == p['epochs'] == p['backwards'] == 0
    assert p['component_gradient_calls'] == 280 and p['decoder_parameters'] == 17952 and p['decoder_tensors'] == 57
    assert len(p['cases']) == 100 and len(p['references']) == 20
    assert all(c['role'] == 'train' for c in p['cases']) and all(r['role'] == 'train' for r in p['references'])
    ids = [c['id'] for c in p['cases']]
    assert ids == p['cohorts']['not_yet_optimized'] + p['cohorts']['optimized'] and len(set(ids)) == 100
    offset = 0
    for row in p['parameter_layout']:
        assert row['start'] == offset and row['end'] > offset
        offset = row['end']
    assert offset == 17952 and len(p['parameter_layout']) == 57
    for name, digest in p['assets_sha256'].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return p


def definitions(root):
    import os
    import subprocess
    import urllib.request
    tree = ast.parse((root / 'frozen_definitions.py').read_text(encoding='utf-8'))
    guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_vm')
    ns = {'sys': sys, 'platform': platform, 'Path': Path, 'os': os, 'subprocess': subprocess, 'urllib': __import__('urllib')}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), '<pinned-original-L4-guard>', 'exec'), ns)
    return tree, ns['require_vm']


def run(root, p, pin):
    scope(root); assert not (root / 'outputs').exists(), 'Preserve every prior/partial diagnostic; no resume'
    start = time.monotonic(); out = root / 'outputs'; out.mkdir()
    progress = {'original_DGP_forwards': 0, 'decoder_forwards': 0, 'reference_decoder_forwards': 0,
                'recognizer_forwards': 0, 'component_gradient_calls': 0, 'optimizer_updates': 0,
                'parameter_updates': 0, 'backwards': 0, 'epochs': 0, 'optimizer_constructed': False}
    try:
        assert shutil.disk_usage(root).free >= p['budgets']['minimum_free_disk_bytes'], 'Need2GiB free; diagnostic performs no deletion'
        tree, require_vm = definitions(root); require_vm(root, idle=True)
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        sys.path.insert(0, str(root))
        from cctv_dgp_v40_learning_signal_v1_decoder import SpatialDGPCandidateV40LearningSignalV1
        from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_degraded_objective_v24 import cohort_normalizers
        from cctv_dgp_batchmatched_identity_v26 import objective_terms
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic() - start <= p['budgets']['worker_seconds'], 'Diagnostic600s stop'
            assert torch.cuda.max_memory_allocated() <= p['budgets']['peak_vram_bytes'], 'Allocated VRAM20GiB stop'
            for key, maximum in p['forward_call_limits'].items(): assert progress[key] <= maximum, key
        original, provenance = load_frozen_dgp_restorer(root / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cuda')
        seed = torch.load(root / 'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
        stopped = torch.load(root / 'stopped_decoder_input.pth', map_location='cpu', weights_only=True)
        candidate = SpatialDGPCandidateV40LearningSignalV1(original, seed); candidate.enable_vm_learning(root)
        identity = FixedObservedIdentity(root / 'weights/w600k_r50.onnx', 'cuda')
        def states():
            return {'original': state_hash(original.net), 'decoder': state_hash(candidate.decoder),
                    'reference_decoder': state_hash(candidate.reference_decoder), 'recognizer': state_hash(identity)}
        assert states() == p['expected_initial_states']
        assert not ({v.data_ptr() for v in candidate.decoder.parameters()} & {v.data_ptr() for v in candidate.reference_decoder.parameters()})
        for model, key in [(original.net, 'original_DGP_forwards'), (candidate.decoder, 'decoder_forwards'),
                           (candidate.reference_decoder, 'reference_decoder_forwards'), (identity.encoder, 'recognizer_forwards')]:
            model.register_forward_hook(lambda *_args, key=key: progress.__setitem__(key, progress[key] + 1))
        parameters = list(candidate.decoder.parameters()); layout = []; offset = 0
        for name, value in candidate.decoder.named_parameters():
            layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()}); offset += value.numel()
        assert layout == p['parameter_layout']
        displacement = np.concatenate([(stopped[row['name']].double() - seed[row['name']].double()).reshape(-1).numpy() for row in layout])
        assert np.array_equal(displacement, np.load(root / 'saved_parameter_delta.npy', allow_pickle=False))
        refs = {r['id']: r for r in p['references']}; items = []
        def pixels(path, mode='RGB'):
            with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()
        def canonical(rgb):
            return torch.from_numpy(rgb.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None].cuda()
        for c in p['cases']:
            image = pixels(root / c['input']); target = pixels(root / c['target']); mask = pixels(root / c['observed'], 'L') > 0
            assert image.shape == target.shape == (256, 256, 3) and mask.shape == (256, 256) and mask.any()
            def erode(radius):
                size = 2*radius + 1; integral = np.pad(np.pad(mask.astype(np.int64), radius).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size] == size**2
            feature = np.zeros((256, 256), bool)
            for point in c['landmarks5_canvas_xy']:
                x, y = np.floor(point).astype(int); feature[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
            feature &= erode(6); assert feature.any()
            t = lambda array: torch.from_numpy(np.asarray(array).copy()).cuda()
            clear = c['profile'] == 'clear'
            items.append({'case': c, 'camera': image, 'mask8': mask, 'x': canonical(image), 'target': canonical(target),
                          'mask': t(mask.astype(np.float32))[None, None], 'feature': t(feature.astype(np.float32))[None, None],
                          'interior': t(erode(6).astype(np.float32))[None, None], 'valid7': t(erode(3).astype(np.float32))[None, None],
                          'grid': t(grid112(refs[c['source_person_or_reference']]['matrix112']))[None],
                          'degraded_weight': torch.tensor([0. if clear else 1.25], device='cuda'),
                          'clear_weight': torch.tensor([1. if clear else 0.], device='cuda')})
        baseline_dir = out / 'baseline'; baseline_dir.mkdir(); parity = []
        for begin in range(0, 100, 5):
            clock(); group = items[begin:begin+5]; x = torch.cat([v['x'] for v in group]); mask = torch.cat([v['mask'] for v in group])
            with torch.no_grad():
                baseline = torch.where(mask.bool(), original(x), x).detach().clone()
                truth = identity.embedding(torch.cat([v['target'] for v in group]), mask, torch.cat([v['grid'] for v in group]))
            for slot, item in enumerate(group):
                item['base'] = baseline[slot:slot+1].clone(); item['truth'] = truth[slot:slot+1].detach().clone()
                cid = item['case']['id']; raw = item['base'][0].permute(1, 2, 0).cpu().numpy().copy()
                np.save(baseline_dir / (cid + '.npy'), raw, allow_pickle=False)
                np.save(baseline_dir / (cid + '_target_embedding.npy'), item['truth'][0].cpu().numpy().copy(), allow_pickle=False)
                png = np.floor(raw*np.float32(255)).astype(np.uint8); png[~item['mask8']] = item['camera'][~item['mask8']]
                Image.fromarray(png).save(baseline_dir / (cid + '.png'))
                if 'historical_initial_raw' in item['case']:
                    err = float(np.abs(raw - np.load(root / item['case']['historical_initial_raw'], allow_pickle=False)).max())
                    assert err <= p['historical_raw_tolerance']; parity.append({'id': cid, 'initial_error': err})
        ns = {'torch': torch, 'F': F}
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != 'require_vm']
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<pinned-original-filter-and-loss-definitions>', 'exec'), ns)
        z = torch.arange(-6, 7, dtype=torch.float32); kernel = torch.exp(-.5*(z/2).square()); kernel /= kernel.sum()
        fixed = SimpleNamespace(kernel=kernel.cuda(), reflect_indices=torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))).cuda())
        fixed.blur = MethodType(ns['blur'], fixed); fixed.high = MethodType(ns['high'], fixed); ns['head'] = fixed
        # Exact V40 normalizers stay fixed across both states and both cohorts.
        computed, setup = cohort_normalizers(items[:50], fixed)
        old = read(root / 'historical_cohort_loss_setup.json')
        assert setup == old, 'V40 normalization receipt changed; stop before derivatives'
        normalizers = tuple(torch.tensor(old[k], dtype=torch.float32, device='cuda') for k in ['feature_normalizer', 'interior_normalizer'])
        assert all(torch.equal(a, b) for a, b in zip(computed, normalizers))
        write(out / 'cohort_loss_setup.json', setup)
        keys = ['x', 'base', 'target', 'mask', 'feature', 'interior', 'valid7', 'grid', 'truth', 'degraded_weight', 'clear_weight']
        summaries = {}; endpoint_states = {}; values_by_case = {}
        for state_name, decoder_state in [('initial', seed), ('stopped50', stopped)]:
            candidate.decoder.load_state_dict(decoder_state, strict=True)
            expected = dict(p['expected_initial_states']); expected['decoder'] = p['decoder_states'][state_name]
            assert states() == expected; endpoint_states[state_name] = expected
            folder = out / state_name; folder.mkdir(); (folder / 'gradients').mkdir(); summaries[state_name] = {}
            for cohort_name, begin_cohort in [('not_yet_optimized', 0), ('optimized', 50)]:
                total = np.zeros((7, 17952), np.float64); values = np.zeros(7, np.float64); batches = []
                for index in range(10):
                    clock(); begin = begin_cohort + index*5; group = items[begin:begin+5]
                    b = {key: torch.cat([item[key] for item in group]) for key in keys}
                    parts = candidate.forward_components(b['x'], b['mask']); pred = parts['result']
                    assert pred.requires_grad and not torch.is_inference(pred)
                    assert torch.equal(parts['original_raw'], b['base']), 'Batch-matched frozen DGP changed'
                    if state_name == 'initial':
                        assert torch.equal(pred.detach(), b['base']) and torch.count_nonzero(parts['spatial_delta']) == 0
                    for slot, item in enumerate(group):
                        cid = item['case']['id']
                        for key, suffix in [('result', ''), ('spatial_delta', '_delta')]:
                            raw = parts[key][slot].detach().permute(1, 2, 0).cpu().numpy().copy()
                            np.save(folder / (cid + suffix + '.npy'), raw, allow_pickle=False)
                        image = parts['result'][slot].detach().permute(1, 2, 0).cpu().numpy().copy()
                        png = np.floor(image*np.float32(255)).astype(np.uint8); png[~item['mask8']] = item['camera'][~item['mask8']]
                        Image.fromarray(png).save(folder / (cid + '.png'))
                        if state_name == 'stopped50' and 'historical_stopped_raw' in item['case']:
                            err = float(np.abs(image - np.load(root / item['case']['historical_stopped_raw'], allow_pickle=False)).max())
                            assert err <= p['historical_raw_tolerance']; parity[next(i for i, r in enumerate(parity) if r['id'] == cid)]['stopped_error'] = err
                    terms = objective_terms(b, pred, identity, ns['mean'], ns['feature_errors'], ns['ssim'], normalizers)
                    assert list(terms) == p['terms']; matrix = np.empty((7, 17952), np.float64); scalar_values = []
                    per_case = np.stack([terms[k].detach().cpu().numpy() for k in p['terms']], 1)
                    assert per_case.shape == (5, 7) and np.isfinite(per_case).all()
                    for item, v in zip(group, per_case): values_by_case[state_name + '/' + item['case']['id']] = v.tolist()
                    for j, name in enumerate(p['terms']):
                        clock(); scalar = terms[name].mean() / 10
                        pieces = torch.autograd.grad(scalar, parameters, retain_graph=j < 6, create_graph=False, allow_unused=False)
                        progress['component_gradient_calls'] += 1
                        assert len(pieces) == 57 and all(torch.isfinite(g).all() for g in pieces)
                        if state_name == 'initial' and j >= 3:
                            assert float(scalar.detach()) == 0 and all(torch.count_nonzero(g) == 0 for g in pieces)
                        matrix[j] = torch.cat([g.detach().reshape(-1).double() for g in pieces]).cpu().numpy()
                        scalar_values.append(float(scalar.detach())); values[j] += scalar_values[-1]
                    total += matrix; path = folder / 'gradients' / (cohort_name + '_batch' + str(index) + '.npy')
                    np.save(path, matrix, allow_pickle=False)
                    batches.append({'batch': index, 'ids': [v['case']['id'] for v in group], 'values': scalar_values,
                                    'gradient_sha256': sha(path), 'norms': np.linalg.norm(matrix, axis=1).tolist()})
                    assert states() == expected and all(v.grad is None for v in candidate.parameters())
                    print(json.dumps({'state': state_name, 'cohort': cohort_name, 'batch': index+1, 'of': 10,
                          'gradient_queries': progress['component_gradient_calls'], 'seconds': time.monotonic()-start}), flush=True)
                path = folder / (cohort_name + '_gradient_components.npy'); np.save(path, total, allow_pickle=False)
                norms = np.linalg.norm(total, axis=1); dot = total @ displacement
                total_direction = -total.sum(0); direction_norm = np.linalg.norm(total_direction)
                summaries[state_name][cohort_name] = {'batches': batches, 'values': values.tolist(), 'component_norms': norms.tolist(),
                    'component_gram': (total @ total.T).tolist(), 'gradient_sha256': sha(path),
                    'gradient_dot_actual50_weight_change': dot.tolist(),
                    'negative_total_direction_component_derivatives': (total @ total_direction).tolist(),
                    'negative_total_direction_cosines': [None if n == 0 or direction_norm == 0 else float(d/(n*direction_norm)) for n, d in zip(norms, total @ total_direction)],
                    'partitions': {r['name']: np.linalg.norm(total[:, r['start']:r['end']], axis=1).tolist() for r in layout}}
        assert len(parity) == 50 and all('stopped_error' in row for row in parity)
        assert progress['component_gradient_calls'] == 280
        assert {k: progress[k] for k in p['forward_call_limits']} == p['forward_call_limits']
        assert all(not v.requires_grad and v.grad is None for v in original.parameters())
        assert all(not v.requires_grad and v.grad is None for v in candidate.reference_decoder.parameters())
        assert all(not v.requires_grad and v.grad is None for v in identity.parameters())
        write(out / 'gradient_summary.json', {'complete': True, 'terms': p['terms'], 'parameter_layout': layout,
              'states': summaries, 'case_term_values': values_by_case,
              'interpretation': 'Endpoint local derivatives, not AdamW trajectory, causal proof, or quality qualification.'})
        clock(); verify(root, pin)
        write(out / 'results.json', {'complete': True, 'protocol_sha256': pin, 'seconds': time.monotonic()-start, **progress,
              'endpoint_states_unchanged': endpoint_states, 'original_provenance': provenance, 'historical_parity': parity,
              'summary_sha256': sha(out / 'gradient_summary.json'), 'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(),
              'new_trained_checkpoint_created': False, 'native_or_reserved_used': False, 'app_promotion': False,
              'training_capacity_pass': False, 'goal_complete': False})
    except Exception as e:
        write(out / 'failure.json', {'protocol_sha256': pin, 'type': type(e).__name__, 'cause': str(e), 'traceback': traceback.format_exc(),
              'seconds': time.monotonic()-start, **progress, 'new_trained_checkpoint_created': False,
              'native_or_reserved_used': False, 'app_promotion': False, 'training_capacity_pass': False, 'goal_complete': False})
        raise


def export(root, p, pin):
    scope(root); assert (root / 'outputs').is_dir()
    start = time.monotonic(); destination = Path.home() / (STEM + '-results.tar.gz')
    assert not destination.exists() and not (root / 'export_manifest.json').exists(), 'Preserve prior exports'
    paths = sorted(path for path in root.rglob('*') if path.is_file())
    assert all(not path.is_symlink() for path in paths)
    paths = [path for path in paths if path.relative_to(root).as_posix() not in p['assets_sha256'] or path.relative_to(root).as_posix() in RETURNED_ASSETS]
    assert sum(path.stat().st_size for path in paths) <= p['budgets']['export_uncompressed_bytes']
    mapping = {path.relative_to(root).as_posix(): sha(path) for path in paths}
    write(root / 'export_manifest.json', {'complete': True, 'protocol_sha256': pin, 'files_sha256': mapping,
          'data_and_original_weights_omitted_and_bound_to_protocol': True, 'training_success_not_implied': True})
    paths.append(root / 'export_manifest.json')
    with tarfile.open(destination, 'w:gz') as archive:
        for path in paths:
            assert time.monotonic()-start <= 300
            archive.add(path, arcname='cctv_dgp_v40_learning_signal_v1_return/' + path.relative_to(root).as_posix(), recursive=False)
    digest = sha(destination)
    with Path(str(destination) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + destination.name + '\n')
    result = {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size, 'seconds': time.monotonic()-start,
              'training_success_not_implied': True, 'optimizer_updates': 0,
              'run_results_present': (root / 'outputs/results.json').exists(), 'failure_present': (root / 'outputs/failure.json').exists()}
    write(Path.home() / (STEM + '-export.json'), result); print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path); parser.add_argument('--protocol-sha', required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ['verify-transfer', 'run', 'export', 'record-supervision']: modes.add_argument('--' + mode, action='store_true')
    parser.add_argument('--elapsed', type=float); parser.add_argument('--exit-code', type=int)
    a = parser.parse_args(); root = a.root.resolve(); scope(root); p = verify(root, a.protocol_sha)
    if a.verify_transfer:
        print(json.dumps({'complete': True, 'packet_assets': len(p['assets_sha256']), 'cases': 100,
              'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0})); return
    if a.record_supervision:
        assert a.elapsed is not None and a.elapsed >= 0 and a.exit_code is not None
        write(root / 'supervisor_receipt.json', {'complete': True, 'protocol_sha256': a.protocol_sha, 'seconds': a.elapsed,
              'cap_seconds': 630, 'kill_grace_seconds': 30, 'within_external_bound': a.elapsed <= 660,
              'diagnostic_exit_code': a.exit_code, 'optimizer_updates': 0, 'quality_acceptance_not_implied': True}); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Finite diagnostic/export deadline')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic external deadline')))
    signal.alarm(300 if a.export else 600)
    export(root, p, a.protocol_sha) if a.export else run(root, p, a.protocol_sha)


if __name__ == '__main__': main()

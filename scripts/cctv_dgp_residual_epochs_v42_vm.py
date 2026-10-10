"""Finite manual L4 study; transfer verification performs no neural imports."""
import argparse
import ast
from pathlib import Path
import os
import platform
import random
import shutil
import signal
import sys
import tarfile
import time
import traceback

NAME = 'cctv_dgp_residual_epochs_vm_v42'
STEM = 'cctv-dgp-residual-epochs-v42'


def scope(root):
    assert sys.platform == 'linux', 'Actual learning requires the existing manual Linux VM'
    assert platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing VM only'
    assert Path(root).resolve() == (Path.home()/'forensic-dgp'/NAME).resolve(), 'Distinct V42 root only'


def verify(root, pin):
    sys.path.insert(0, str(root))
    from cctv_dgp_residual_epochs_v42_contract import read, verified_assets
    p = read(root/'protocol.json'); verified_assets(root, p, pin)
    return p


def run(root, p, pin):
    scope(root); assert not (root/'outputs').exists(), 'Preserve every prior/partial run; no repeat or resume'
    from cctv_dgp_residual_epochs_v42_contract import BUDGETS, sha, read, write
    started = time.monotonic(); out = root/'outputs'; out.mkdir()
    progress = {'optimizer_updates': 0, 'backwards': 0, 'gradient_queries': 0,
                'optimizer_constructed': False, 'new_trained_checkpoint': False}
    candidate = optimizer = scheduler = None; snapshots = []; step_times = []; fit_start = None
    try:
        assert shutil.disk_usage(root).free >= BUDGETS['minimum_free_disk_bytes'], 'Need8GiB free after installation'
        # Compile the hash-bound existing hardware/idle guard without executing historical scripts.
        import subprocess, urllib.request
        tree = ast.parse((root/'frozen_definitions.py').read_text())
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_vm')
        ns = {'sys': sys, 'platform': platform, 'Path': Path, 'os': os,
              'subprocess': subprocess, 'urllib': __import__('urllib')}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), '<L4-metadata-idle-guard>', 'exec'), ns)
        ns['require_vm'](root, idle=True)
        import numpy as np
        import torch
        from torch.nn import functional as F
        from cctv_dgp_spatial_decoder_v42 import SpatialDGPCandidateV42
        from cctv_dgp_pilot import FixedObservedIdentity, state_hash
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from cctv_dgp_residual_supervision_v1 import projected_teacher, correction_terms, TERMS
        from cctv_dgp_residual_epochs_v42_objective import objective_terms, reconstruction_terms
        from cctv_dgp_residual_epochs_v42_data import load_items, batch, snapshot, gates
        random.seed(420042); np.random.seed(420042); torch.manual_seed(420042); torch.cuda.manual_seed_all(420042)
        import importlib.metadata
        versions = {}
        for name in ['torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python',
                     'opencv-python-headless', 'onnx', 'onnx2torch', 'scipy', 'scikit-image']:
            try: versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name] = None
        write(out/'environment.json', {'python': sys.version, 'platform': platform.platform(),
            'host': platform.node(), 'packages': versions, 'CUDA_runtime': torch.version.cuda,
            'GPU': torch.cuda.get_device_name(0), 'machine_type_verified': 'g2-standard-4',
            'original_checkpoint_sha256': p['original_checkpoint_sha256'],
            'learning_tensors': 'Our spatial decoder only; original DGP remains frozen'})
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        original, provenance = load_frozen_dgp_restorer(root/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cuda')
        seed = torch.load(root/'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
        candidate = SpatialDGPCandidateV42(original, seed); candidate.enable_vm_learning(root)
        identity = FixedObservedIdentity(root/'weights/w600k_r50.onnx', 'cuda')
        parameters = list(candidate.decoder.parameters())
        assert len(parameters) == 57 and sum(v.numel() for v in parameters) == 17952
        assert not ({v.data_ptr() for v in parameters} & {v.data_ptr() for v in candidate.reference_decoder.parameters()})
        counts = {'original_DGP': 0, 'decoder': 0, 'reference_decoder': 0, 'recognizer': 0}
        for model, key in [(original.net, 'original_DGP'), (candidate.decoder, 'decoder'),
                           (candidate.reference_decoder, 'reference_decoder'), (identity.encoder, 'recognizer')]:
            model.register_forward_hook(lambda *_args, key=key: counts.__setitem__(key, counts[key]+1))
        def states():
            return {'original': state_hash(original.net), 'decoder': state_hash(candidate.decoder),
                    'reference_decoder': state_hash(candidate.reference_decoder), 'recognizer': state_hash(identity)}
        before = states(); assert before == p['initial_states']
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic()-started <= BUDGETS['worker_seconds'], 'Worker timing stop'
            assert torch.cuda.max_memory_allocated() <= BUDGETS['peak_vram_bytes'], 'VRAM20GiB stop'
            assert shutil.disk_usage(root).free >= BUDGETS['disk_reserve_bytes'], 'Protected disk reserve stop'
        def frozen():
            current = states()
            for key in ['original', 'reference_decoder', 'recognizer']: assert current[key] == before[key]
            for model in [original, candidate.reference_decoder, identity]:
                assert all(not v.requires_grad and v.grad is None for v in model.parameters())
            assert sha(root/'weights/dgp_v2.pth') == p['original_checkpoint_sha256']
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['mean', 'ssim']]
        filter_ns = {'torch': torch, 'F': F}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<fixed-SSIM>', 'exec'), filter_ns)
        items = load_items(root, p, original, identity, clock, out)
        by_id = {i['case']['id']: j for j, i in enumerate(items)}
        chosen = [by_id[cid] for cid in p['preview_case_ids']]
        normalizers = {}; records = []
        with torch.no_grad():
            for begin in range(0, 50, 5):
                clock(); b = batch(items, chosen[begin:begin+5])
                teacher, _ = projected_teacher(b['base'], b['target'], b['mask'].bool(), b['clear'])
                values = correction_terms(torch.zeros_like(teacher), teacher, b['mask'].bool(), b['feature'].bool())
                for slot, idx in enumerate(chosen[begin:begin+5]):
                    records.append({'id': items[idx]['case']['id'], 'clear': bool(b['clear'][slot]),
                                    **{key: float(values[key][slot]) for key in TERMS}})
            degraded = [r for r in records if not r['clear']]; assert len(degraded) == 40
            for key in TERMS:
                value = max(float(np.mean([r[key] for r in degraded])), p['normalizer_floor'])
                assert abs(value-p['audited_normalizers'][key]) <= 5e-7, 'Frozen calibration arithmetic differs'
                normalizers[key] = value
            assert all(all(r[key] == 0 for key in TERMS) for r in records if r['clear'])
        write(out/'cohort_loss_setup.json', {'complete': True, 'cases': 50, 'degraded_cases': 40,
              'normalizers': normalizers, 'rows': records, 'normalizer_floor': p['normalizer_floor'],
              'arithmetic_tolerance_only': 5e-7, 'optimizer_constructed': False})
        # Test every trainable tensor's new improvement path before constructing the optimizer.
        partitions = {n: {'improvement_gradient_norm': 0., 'shape': list(v.shape)}
                      for n, v in candidate.decoder.named_parameters()}; norms = []
        for begin in range(0, 50, 5):
            clock(); b = batch(items, chosen[begin:begin+5])
            components = candidate.forward_components(b['x'], b['mask'])
            assert torch.equal(components['result'], b['base'])
            terms = reconstruction_terms(b, components, normalizers); row = {}
            for key, value in terms.items():
                gradients = torch.autograd.grad(value.mean(), parameters, retain_graph=True, allow_unused=False)
                progress['gradient_queries'] += 1
                for (name, _), grad in zip(candidate.decoder.named_parameters(), gradients):
                    assert bool(torch.isfinite(grad).all())
                    partitions[name]['improvement_gradient_norm'] += float(grad.double().square().sum())
                row[key] = float(sum(grad.double().square().sum() for grad in gradients).sqrt())
            norms.append(row); del gradients, terms, components
            print({'V42_preflight_batch': begin//5+1, 'of': 10, 'component_norms': row}, flush=True)
        for row in partitions.values(): row['improvement_gradient_norm'] = row['improvement_gradient_norm']**.5
        write(out/'gradient_preflight.json', {'complete': True, 'partitions': partitions,
              'component_norms': norms, 'gradient_queries': progress['gradient_queries'], 'optimizer_updates': 0})
        assert all(v['improvement_gradient_norm'] > 0 for v in partitions.values()), 'New correction path has a disconnected tensor'
        assert all(v.grad is None for v in parameters); frozen()
        baseline = snapshot(out, 0, p, items, candidate, identity, clock, frozen); snapshots.append(baseline)
        # Reserve for every remaining snapshot and a complete portable export before any update.
        initial_bytes = sum(q.stat().st_size for q in (out/'update0').rglob('*') if q.is_file())
        projected_bytes = initial_bytes*len(p['snapshots'])*1.25 + 64*1024**2
        write(out/'storage_projection.json', {'initial_snapshot_bytes': initial_bytes,
              'snapshots': len(p['snapshots']), 'safety_factor': 1.25,
              'projected_uncompressed_bytes': projected_bytes,
              'cap_bytes': BUDGETS['export_uncompressed_bytes']})
        assert projected_bytes <= BUDGETS['export_uncompressed_bytes'], 'Finite output storage projection stop'
        assert shutil.disk_usage(root).free >= 2*projected_bytes-initial_bytes+BUDGETS['disk_reserve_bytes'], 'Cannot reserve outputs and migration export'
        optimizer = torch.optim.AdamW(parameters, lr=p['optimizer']['learning_rate'], weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1562], gamma=.3)
        progress['optimizer_constructed'] = True
        def save_state(update, filename):
            path = out/filename; assert not path.exists()
            np_state = np.random.get_state()
            torch.save({'format': 'v42-portable-full-training-state-v1', 'protocol_sha256': pin,
                'decoder': {k: v.detach().cpu().clone() for k, v in candidate.decoder.state_dict().items()},
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'python_rng': random.getstate(), 'numpy_rng': (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), *np_state[2:]),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'completed_updates': update, 'completed_epochs': update//781,
                'next_schedule_index': update, 'schedule_sha256': p['assets_sha256']['schedule.json'],
                'original_checkpoint_sha256': p['original_checkpoint_sha256'], 'initial_decoder_sha256': p['assets_sha256']['untrained_initial_decoder.pth'],
                'no_automatic_resume': True, 'failed_gate_must_not_be_resumed': True}, path)
        save_state(0, 'update0/training_state.pt'); fit_start = time.monotonic()
        with (out/'training_steps.jsonl').open('x', encoding='utf-8') as log:
            for update, ids in enumerate(read(root/'schedule.json')['batches'], 1):
                clock(); assert time.monotonic()-fit_start < BUDGETS['fit_seconds']; step = time.monotonic()
                optimizer.zero_grad(set_to_none=True); b = batch(items, ids)
                components = candidate.forward_components(b['x'], b['mask'])
                terms = objective_terms(b, components, identity, filter_ns['ssim'], normalizers)
                objective = sum(terms.values()).mean(); assert bool(torch.isfinite(objective))
                objective.backward(); progress['backwards'] += 1
                assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in parameters)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1); assert bool(torch.isfinite(norm))
                rate = optimizer.param_groups[0]['lr']; optimizer.step(); progress['optimizer_updates'] = update
                progress['new_trained_checkpoint'] = True
                scheduler.step(); torch.cuda.synchronize(); step_times.append(time.monotonic()-step)
                import json
                log.write(json.dumps({'update': update, 'epoch': (update-1)//781+1, 'case_ids': [items[i]['case']['id'] for i in ids],
                    'learning_rate': rate, 'objective': float(objective.detach()), 'gradient_norm_before_clip': float(norm),
                    'terms': {k: float(v.detach().mean()) for k, v in terms.items()}, 'seconds': step_times[-1]}, allow_nan=False)+'\n')
                log.flush(); del objective, terms, components
                if update == 20:
                    elapsed = time.monotonic()-fit_start
                    allowance = 4*baseline['snapshot_duration_seconds']*1.25
                    estimate = elapsed+3885*float(np.mean(step_times[1:]))*1.25+allowance
                    write(out/'timing_update20.json', {'updates': 20, 'seconds': elapsed,
                        'steady_sample_seconds': step_times[1:], 'remaining_updates': 3885, 'safety_factor': 1.25,
                        'overhead_seconds': allowance, 'projected_seconds': estimate, 'cap_seconds': BUDGETS['fit_seconds']})
                    assert estimate <= BUDGETS['fit_seconds'], 'Five-epoch timing projection stop'
                if update in p['snapshots']:
                    current = snapshot(out, update, p, items, candidate, identity, clock, frozen); snapshots.append(current)
                    save_state(update, 'update'+str(update)+'/training_state.pt')
                    gate = gates(baseline, current, .1 if update == 3905 else .01)
                    write(out/('capacity_update'+str(update)+'.json'), gate)
                    assert gate['pass'], 'Structure/preservation requirement failed; retain stop'
                    size = sum(q.stat().st_size for q in out.rglob('*') if q.is_file())
                    assert size <= BUDGETS['export_uncompressed_bytes'], 'Return storage stop'
                if update == 1 or update % 50 == 0:
                    print({'V42_update': update, 'of': 3905, 'epoch': (update-1)//781+1,
                           'learning_rate': rate}, flush=True)
        clock(); assert time.monotonic()-fit_start <= BUDGETS['fit_seconds']; frozen(); verify(root, pin)
        write(out/'results.json', {'complete': True, 'protocol_sha256': pin, **progress,
            'seconds': time.monotonic()-started, 'fit_seconds': time.monotonic()-fit_start,
            'necessary_capacity_pass': gate['pass'], 'completed_epochs': 5,
            'complete_snapshots': [r['update'] for r in snapshots], 'step_times_seconds': step_times,
            'states_before': before, 'states_after': states(), 'forward_counts': counts,
            'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(), 'original_provenance': provenance,
            'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False})
    except BaseException as exc:
        if candidate is not None:
            import torch
            candidate.decoder.requires_grad_(False)
            torch.save(candidate.decoder.state_dict(), out/'stopped_spatial_decoder.pth')
            progress['new_trained_checkpoint'] = progress['optimizer_updates'] > 0
            if optimizer is not None and 'save_state' in locals():
                save_state(progress['optimizer_updates'], 'stopped_training_state.pt')
        write(out/'failure.json', {'complete': False, 'protocol_sha256': pin, **progress,
            'error': repr(exc), 'traceback': traceback.format_exc(), 'seconds': time.monotonic()-started,
            'fit_seconds': None if fit_start is None else time.monotonic()-fit_start,
            'step_times_seconds': step_times, 'states_before': locals().get('before'),
            'states_after': states() if 'states' in locals() else None, 'forward_counts': locals().get('counts'),
            'complete_snapshots': [r['update'] for r in snapshots], 'resume_permitted': False,
            'automatic_repeat_permitted': False, 'native_or_reserved_used': False,
            'app_promotion': False, 'goal_complete': False})
        raise


def export(root, p, pin):
    from cctv_dgp_residual_epochs_v42_contract import BUDGETS, sha, read, write
    assert (root/'outputs').is_dir(); start = time.monotonic()
    destination = Path.home()/(STEM+'-results.tar.gz')
    assert not destination.exists() and not (root/'export_manifest.json').exists()
    paths = sorted(q for q in root.rglob('*') if q.is_file() and q.relative_to(root).as_posix() not in p['assets_sha256'])
    assert all(not q.is_symlink() for q in paths)
    assert sum(q.stat().st_size for q in paths) <= BUDGETS['export_uncompressed_bytes']
    assert shutil.disk_usage(root).free >= sum(q.stat().st_size for q in paths)+BUDGETS['disk_reserve_bytes']
    mapping = {q.relative_to(root).as_posix(): sha(q) for q in paths}
    write(root/'export_manifest.json', {'complete': True, 'protocol_sha256': pin,
          'files_sha256': mapping, 'input_assets_omitted_and_bound_to_protocol': True})
    paths.append(root/'export_manifest.json')
    with tarfile.open(destination, 'w:gz', compresslevel=1) as archive:
        for q in paths:
            assert time.monotonic()-start < BUDGETS['export_seconds']
            archive.add(q, arcname=NAME+'_return/'+q.relative_to(root).as_posix(), recursive=False)
    digest = sha(destination)
    with Path(str(destination)+'.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest+'  '+destination.name+'\n')
    result = read(root/('outputs/results.json' if (root/'outputs/results.json').is_file() else 'outputs/failure.json'))
    receipt = {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size,
        'seconds': time.monotonic()-start, 'training_success_not_implied': True,
        'optimizer_updates': result['optimizer_updates'], 'run_results_present': (root/'outputs/results.json').is_file(),
        'failure_present': (root/'outputs/failure.json').is_file()}
    write(Path.home()/(STEM+'-export.json'), receipt); print(receipt, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for key in ['verify-transfer', 'run', 'export', 'record-supervision']: mode.add_argument('--'+key, action='store_true')
    parser.add_argument('--elapsed', type=float); parser.add_argument('--exit-code', type=int)
    a = parser.parse_args(); root = a.root.resolve(); scope(root); p = verify(root, a.protocol_sha)
    from cctv_dgp_residual_epochs_v42_contract import BUDGETS, write
    if a.verify_transfer:
        print({'complete': True, 'cases': 3905, 'epochs_maximum': 5,
               'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0}); return
    if a.record_supervision:
        assert a.elapsed is not None and a.exit_code is not None
        write(root/'supervisor_receipt.json', {'complete': True, 'protocol_sha256': a.protocol_sha,
            'seconds': a.elapsed, 'cap_seconds': BUDGETS['external_seconds'],
            'kill_grace_seconds': 30, 'worker_exit_code': a.exit_code, 'quality_acceptance_not_implied': True}); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Finite V42 deadline')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('External V42 stop')))
    signal.alarm(BUDGETS['export_seconds'] if a.export else BUDGETS['worker_seconds'])
    export(root, p, a.protocol_sha) if a.export else run(root, p, a.protocol_sha)


if __name__ == '__main__': main()
